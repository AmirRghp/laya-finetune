"""Stage orchestration.

One function per CLI command. Each stage reads the previous stage's artifact,
writes its own, and records a manifest — so the pipeline can be re-run from any
point and every artifact is traceable to the inputs that produced it
(AGENTS.md 4.4).

Stage order, and why it is this order:

``scan`` -> ``process`` -> ``generate`` -> ``split`` -> ``negatives``
-> ``label`` -> ``tasks`` -> ``build`` -> ``validate`` -> ``stats``

``split`` comes **before** ``negatives`` so that every negative is mined from its
own split's chunk pool and no test document's text can reach a training row
(AGENTS.md 6.7).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .chunking import Chunker
from .config import Config
from .corpus import (
    Chunk,
    ParsedDocument,
    discover_documents,
    find_duplicates,
    parse_markdown,
)
from .dataset import (
    ChunkRef,
    GeneratedExample,
    LabeledExample,
    apply_splits,
    assign_splits,
    build_rows,
    chunk_stats,
    example_stats,
    render_markdown,
    split_coverage,
    sufficiency_balance,
    write_dataset,
)
from .dataset.split import SplitReport
from .errors import PipelineError
from .io import (
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_manifest,
    write_text,
)
from .labeling import (
    build_review_queue,
    detect_contradictions,
    export_review_csv,
    label_distribution,
    propose_labels,
)
from .llm.cache import make_cache
from .llm.ollama import OllamaClient
from .llm.prompts import PromptLibrary
from .logging_setup import get_logger
from .negatives import NegativeSampler
from .retrieval import build_index, make_embed_fn
from .tasks import GenerationContext, QuestionGenerator, RoutingLabeler, SufficiencyLabeler
from .tasks.relevance import assert_answerable
from .tokenizer import resolve_token_counter

__all__ = [
    "stage_scan",
    "stage_process",
    "stage_generate",
    "stage_split",
    "stage_negatives",
    "stage_label",
    "stage_tasks",
    "stage_build",
    "stage_validate",
    "stage_stats",
    "load_chunks",
    "load_examples",
]

LOGGER = get_logger("stages")

SCAN_DIR = "scan"
CHUNK_DIR = "chunks"
GENERATE_DIR = "generated"
SPLIT_DIR = "split"
NEGATIVE_DIR = "negatives"
LABEL_DIR = "labels"
DATASET_DIR = "dataset"
EVAL_DIR = "evaluation"
REVIEW_DIR = "review"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def make_client(config: Config, namespace: str, *, use_cache: bool = True) -> OllamaClient:
    """An Ollama client wired to the configured cache namespace."""
    cache = make_cache(
        config._resolve(config.ollama.cache_dir), namespace, enabled=use_cache
    )
    return OllamaClient(config.ollama, cache=cache)


def make_prompts(config: Config) -> PromptLibrary:
    return PromptLibrary(config.prompt_dir)


def _rng(config: Config, salt: str = "") -> random.Random:
    """A seeded generator. The salt keeps stages independent but reproducible."""
    seed = config.project.seed + (sum(ord(ch) for ch in salt) if salt else 0)
    return random.Random(seed)


def load_chunks(config: Config, stage: str = CHUNK_DIR) -> List[Chunk]:
    path = config.stage_dir(stage) / "chunks.jsonl"
    if not path.is_file():
        raise PipelineError(
            f"{path} does not exist. Run `laya-pipeline process` first — every later "
            "stage reads its chunks from there."
        )
    return [Chunk.from_dict(row) for row in read_jsonl(path)]


def load_examples(config: Config, stage: str = LABEL_DIR, filename: str = "labeled.jsonl") -> List[LabeledExample]:
    path = config.stage_dir(stage) / filename
    if not path.is_file():
        raise PipelineError(
            f"{path} does not exist. Run the earlier stages first."
        )
    return [LabeledExample.from_dict(row) for row in read_jsonl(path)]


def _chunks_by_source(chunks: Sequence[Chunk]) -> Dict[str, List[Chunk]]:
    grouped: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_id, []).append(chunk)
    for group in grouped.values():
        group.sort(key=lambda c: c.ordinal)
    return grouped


def _pool_for_split(chunks: Sequence[Chunk], report: SplitReport, split: str) -> Dict[str, Chunk]:
    """The chunk pool belonging to one split.

    This is the boundary that keeps negative mining from leaking: a candidate
    can only ever be drawn from the same split as the question it will label.
    """
    return {c.chunk_id: c for c in chunks if report.split_for(c.source_id) == split}


def chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def _write_examples(
    config: Config, stage: str, filename: str, examples: Sequence[LabeledExample]
) -> Path:
    directory = ensure_dir(config.stage_dir(stage))
    path = directory / filename
    write_jsonl(path, (example.to_dict() for example in examples), config.corpus_dir)
    return path


def _write_generated(
    config: Config, filename: str, examples: Sequence[GeneratedExample]
) -> Path:
    directory = ensure_dir(config.stage_dir(GENERATE_DIR))
    path = directory / filename
    write_jsonl(path, (example.to_dict() for example in examples), config.corpus_dir)
    return path


def _load_generated(config: Config, filename: str) -> List[GeneratedExample]:
    path = config.stage_dir(GENERATE_DIR) / filename
    if not path.is_file():
        raise PipelineError(
            f"{path} does not exist. Run `laya-pipeline generate` first."
        )
    return [GeneratedExample.from_dict(row) for row in read_jsonl(path)]


def _load_split_report_obj(config: Config) -> SplitReport:
    """Reload the split assignment written by the ``split`` stage.

    Both ``negatives`` and ``tasks`` need the same family→split map. Reloading
    the artifact rather than recomputing guarantees they scope their pools
    identically — and identically to what a reviewer can read in ``split.json``.
    """
    path = config.stage_dir(SPLIT_DIR) / "split.json"
    if not path.is_file():
        raise PipelineError(
            f"{path} does not exist. Run `laya-pipeline split` first: negatives and "
            "the downstream tasks are scoped by it, and running them without it "
            "would let a test document's text reach a training row."
        )
    raw = read_json(path)
    return SplitReport(
        family_of=dict(raw.get("source_to_family", {})),
        assignment=dict(raw.get("assignment", {})),
        stats=dict(raw.get("stats", {})),
        warnings=list(raw.get("warnings", [])),
        leakage=list(raw.get("leakage", [])),
    )


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_scan(config: Config) -> Dict[str, Any]:
    """List every Markdown document under the corpus, with hashes."""
    documents = discover_documents(config.corpus_dir)
    duplicates = find_duplicates(documents)

    directory = ensure_dir(config.stage_dir(SCAN_DIR))
    rows = [document.to_dict() for document in documents]
    write_jsonl(directory / "documents.jsonl", rows, config.corpus_dir)
    payload = {
        "corpus_dir": str(config.corpus_dir),
        "documents": len(documents),
        "total_bytes": sum(document.byte_size for document in documents),
        "unique_contents": len({document.content_id for document in documents}),
        "duplicates": duplicates,
    }
    write_json(directory / "scan.json", payload, config.corpus_dir)
    write_manifest(
        directory,
        stage="scan",
        config_hash=config.config_hash,
        seed=config.project.seed,
        counts=payload,
        corpus_dir=config.corpus_dir,
    )
    if duplicates:
        LOGGER.warning(
            "duplicate document contents found",
            extra={"fields": {"groups": len(duplicates)}},
        )
    return payload


def stage_process(config: Config) -> Dict[str, Any]:
    """Parse and chunk every discovered document."""
    documents = discover_documents(config.corpus_dir)
    counter = resolve_token_counter(config.chunking, allow_download=True, allow_estimates=True)
    chunker = Chunker(config.chunking, counter)

    parsed: List[ParsedDocument] = [parse_markdown(document) for document in documents]
    chunks: List[Chunk] = chunker.chunk_documents(parsed)

    directory = ensure_dir(config.stage_dir(CHUNK_DIR))
    document_rows = []
    for document in parsed:
        document_rows.append(
            {
                "source_id": document.source_id,
                "relpath": document.source.relpath,
                "content_id": document.content_id,
                "declared_id": document.declared_id,
                "declared_title": document.declared_title,
                "frontmatter": document.frontmatter,
                "blocks": len(document.blocks),
                "chunks": sum(1 for c in chunks if c.source_id == document.source_id),
            }
        )
    write_jsonl(directory / "documents.jsonl", document_rows, config.corpus_dir)
    write_jsonl(directory / "chunks.jsonl", (chunk.to_dict() for chunk in chunks), config.corpus_dir)

    stats = chunker.stats(parsed, chunks)
    payload = {"chunking": stats.to_dict(), "documents": len(parsed)}
    write_json(directory / "process.json", payload, config.corpus_dir)
    write_manifest(
        directory,
        stage="process",
        config_hash=config.config_hash,
        seed=config.project.seed,
        counts=payload,
        inputs={"token_source": stats.token_source},
        corpus_dir=config.corpus_dir,
    )
    return payload


def stage_generate(config: Config, *, resume: bool = True) -> Dict[str, Any]:
    """Generate Persian questions and key points for every eligible chunk."""
    chunks = load_chunks(config)
    client = make_client(config, "generate")
    prompts = make_prompts(config)
    ctx = GenerationContext(
        config=config,
        client=client,
        prompts=prompts,
        rng=_rng(config, "generate"),
        task="relevance",
    )

    existing: Dict[str, List[GeneratedExample]] = {}
    if resume:
        path = config.stage_dir(GENERATE_DIR) / "relevance.jsonl"
        if path.is_file():
            for row in read_jsonl(path):
                example = GeneratedExample.from_dict(row)
                existing.setdefault(example.source_chunk_id, []).append(example)
            LOGGER.info(
                "resuming generation",
                extra={"fields": {"examples": sum(len(v) for v in existing.values())}},
            )

    generator = QuestionGenerator(config, prompts, client)
    report = generator.generate(chunks, ctx, existing=existing or None)

    path = _write_generated(config, "relevance.jsonl", report.examples)
    write_json(
        config.stage_dir(GENERATE_DIR) / "relevance.report.json",
        report.to_dict(),
        config.corpus_dir,
    )
    return {
        "examples": len(report.examples),
        "stats": report.stats,
        "failures": len(report.failures),
        "path": str(path),
    }


def stage_split(config: Config) -> Dict[str, Any]:
    """Turn generated questions into positives and assign document-level splits."""
    generated = _load_generated(config, "relevance.jsonl")
    chunks = load_chunks(config)
    by_id = chunks_by_id(chunks)
    positive_label = config.laya.role_label("relevance", "positive")

    positives: List[LabeledExample] = []
    dropped: List[Dict[str, Any]] = []
    for example in generated:
        chunk = by_id.get(example.source_chunk_id)
        if chunk is None:
            dropped.append({"example_id": example.example_id, "reason": "source chunk missing"})
            continue
        try:
            assert_answerable(example)
        except PipelineError as exc:
            # A generated question that never produced key points cannot become a
            # positive or a verified negative. Record it and move on.
            dropped.append({"example_id": example.example_id, "reason": str(exc)})
            continue
        positives.append(_positive_from(example, chunk, positive_label))

    report = assign_splits(positives, config, _chunks_by_source(chunks), _rng(config, "split"))
    assigned = apply_splits(positives, report)

    directory = ensure_dir(config.stage_dir(SPLIT_DIR))
    _write_examples(config, SPLIT_DIR, "positives.jsonl", assigned)
    write_json(directory / "split.json", report.to_dict(), config.corpus_dir)
    write_json(
        directory / "dropped.json",
        {"dropped": dropped, "stats": example_stats(config, positives)},
        config.corpus_dir,
    )
    return {
        "positives": len(assigned),
        "dropped": len(dropped),
        **report.stats,
        "warnings": report.warnings,
        "leakage": len(report.leakage),
    }


def _positive_from(
    example: GeneratedExample, chunk: Chunk, positive_label: str
) -> LabeledExample:
    return LabeledExample(
        example_id=example.example_id,
        task="relevance",
        query=example.query,
        chunks=[
            ChunkRef(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_id=chunk.source_id,
                relpath=chunk.relpath,
                ordinal=chunk.ordinal,
            )
        ],
        label=positive_label,
        label_source="rule",
        role="positive",
        source_id=example.source_id,
        relpath=example.relpath,
        section_id=example.section_id,
        key_points=list(example.key_points),
        metadata={"query_type": example.query_type, "generation": example.generation},
    )


def stage_negatives(config: Config) -> Dict[str, Any]:
    """Mine negatives inside each split's own chunk pool."""
    positives = load_examples(config, SPLIT_DIR, "positives.jsonl")
    chunks = load_chunks(config)
    report = _load_split_report_obj(config)

    client = make_client(config, "negatives")
    prompts = make_prompts(config)
    sampler = NegativeSampler(config, prompts, client)
    ctx = GenerationContext(
        config=config,
        client=client,
        prompts=prompts,
        rng=_rng(config, "negatives"),
        task="relevance",
    )

    backend = config.evaluation.retrieval_backend
    embed_fn = (
        None if backend == "lexical" else make_embed_fn(backend, config.ollama, config.chunking, client)
    )

    collected: List[LabeledExample] = []
    totals: Dict[str, int] = {}
    rejected: List[Dict[str, Any]] = []

    for split in ("train", "validation", "test"):
        pool = _pool_for_split(chunks, report, split)
        if not pool:
            LOGGER.warning("split has no chunks; no negatives mined", extra={"fields": {"split": split}})
            continue
        split_positives = [p for p in positives if p.split == split]
        index = build_index(backend, list(pool.values()), embed_fn=embed_fn if backend != "lexical" else None)
        outcome = sampler.mine(split_positives, pool, ctx, retrieval_index=index, embed_fn=embed_fn)
        for example in outcome.examples:
            collected.append(_with_split(example, split))
        for key, value in outcome.stats.items():
            totals[key] = totals.get(key, 0) + value
        rejected.extend(outcome.rejected)

    directory = ensure_dir(config.stage_dir(NEGATIVE_DIR))
    _write_examples(config, NEGATIVE_DIR, "negatives.jsonl", collected)
    write_json(
        directory / "negatives.report.json",
        {"stats": totals, "rejected": rejected[:2000], "rejected_total": len(rejected)},
        config.corpus_dir,
    )
    return {"negatives": len(collected), "stats": totals, "rejected": len(rejected)}


def _with_split(example: LabeledExample, split: str) -> LabeledExample:
    return LabeledExample(
        example_id=example.example_id,
        task=example.task,
        query=example.query,
        chunks=list(example.chunks),
        label=example.label,
        label_source=example.label_source,
        confidence=example.confidence,
        strategy=example.strategy,
        margin=example.margin,
        verdict=example.verdict,
        role=example.role,
        split=split,
        source_id=example.source_id,
        relpath=example.relpath,
        section_id=example.section_id,
        key_points=list(example.key_points),
        metadata=dict(example.metadata),
    )


def stage_label(config: Config, *, review_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Teacher-label the mined negatives and produce a human review queue."""
    positives = load_examples(config, SPLIT_DIR, "positives.jsonl")
    negatives = load_examples(config, NEGATIVE_DIR, "negatives.jsonl")
    everything = positives + negatives

    client = make_client(config, "label")
    prompts = make_prompts(config)
    report = propose_labels(everything, config, client, prompts)

    contradictions = detect_contradictions(report.examples)
    queue = report.queue + build_review_queue(report.examples, config)

    directory = ensure_dir(config.stage_dir(LABEL_DIR))
    _write_examples(config, LABEL_DIR, "labeled.jsonl", report.examples)
    csv_path = Path(review_dir) if review_dir else config.stage_dir(REVIEW_DIR) / "review.csv"
    ensure_dir(csv_path.parent)
    export_review_csv(queue, csv_path, config.corpus_dir)

    options_by_task = {name: config.laya.options(name) for name in config.laya.question_ids()}
    payload = {
        "labeled": len(report.examples),
        "stats": report.stats,
        "distribution": label_distribution(report.examples, options_by_task),
        "contradictions": contradictions,
        "review_queue": len(queue),
        "review_csv": str(csv_path),
    }
    write_json(directory / "labels.json", payload, config.corpus_dir)
    return payload


def stage_tasks(config: Config, task: str, *, top_k: int = 3) -> Dict[str, Any]:
    """Run a downstream task (sufficiency or routing) on the split corpus."""
    if task == "relevance":
        raise PipelineError(
            "relevance is produced by `generate`; run `laya-pipeline generate --task relevance`"
        )

    chunks = load_chunks(config)
    generated = _load_generated(config, "relevance.jsonl")
    report = _load_split_report_obj(config)

    client = make_client(config, task)
    prompts = make_prompts(config)
    ctx = GenerationContext(
        config=config,
        client=client,
        prompts=prompts,
        rng=_rng(config, task),
        task=task,
    )

    backend = config.evaluation.retrieval_backend
    produced: List[LabeledExample] = []

    for split in ("train", "validation", "test"):
        pool = _pool_for_split(chunks, report, split)
        if not pool:
            continue
        split_examples = [
            example
            for example in generated
            if report.split_for(example.source_id) == split
        ]
        if not split_examples:
            continue

        if task == "routing":
            labeler = RoutingLabeler(config, prompts, client)
            outcome = labeler.label(split_examples, ctx)
        else:
            embed_fn = make_embed_fn(backend, config.ollama, config.chunking, client=client)
            index = build_index(
                backend,
                list(pool.values()),
                embed_fn=embed_fn if backend != "lexical" else None,
            )
            labeler = SufficiencyLabeler(config, prompts, client)
            outcome = labeler.label(
                split_examples,
                ctx,
                index,
                _pool_object(pool),
                embed_fn=embed_fn,
                top_k=top_k,
            )

        for example in outcome.examples:
            produced.append(_with_split(example, split))

    existing = []
    path = config.stage_dir(LABEL_DIR) / "labeled.jsonl"
    if path.is_file():
        existing = [e for e in load_examples(config) if e.task != task]

    combined = existing + produced
    _write_examples(config, LABEL_DIR, "labeled.jsonl", combined)
    task_dir = ensure_dir(config.stage_dir(task))
    write_json(
        task_dir / f"{task}.report.json",
        {"task": task, "examples": len(produced)},
        config.corpus_dir,
    )
    return {"task": task, "examples": len(produced), "total_labeled": len(combined)}


def _pool_object(pool: Mapping[str, Chunk]) -> "Any":
    """Adapt a plain chunk-id mapping to the sufficiency labeler's pool type."""
    from .tasks.sufficiency import ChunkPool

    return ChunkPool.build(list(pool.values()))


def stage_build(config: Config, tasks: Sequence[str], *, positive_control: bool = False) -> Dict[str, Any]:
    """Write Laya-format JSONL per task and split, plus questions.json.

    A positive-control build goes to ``dataset/<task>-control/`` so it can never
    be mistaken for, or trained in place of, the real dataset.
    """
    examples = load_examples(config)
    directory = ensure_dir(config.stage_dir(DATASET_DIR))
    suffix = "-control" if positive_control else ""
    results: Dict[str, Any] = {}

    for task in tasks:
        task_examples = [example for example in examples if example.task == task]
        if not task_examples:
            results[task] = {"rows": 0, "warning": f"no labeled examples for task {task!r}"}
            continue
        artifact = build_rows(
            config, task_examples, task, positive_control=positive_control
        )
        task_dir = ensure_dir(directory / f"{task}{suffix}")
        for split in ("train", "validation", "test"):
            subset = [row for row in artifact.rows if f"split:{split}" in row.get("tags", [])]
            write_jsonl(task_dir / f"{split}.jsonl", subset, config.corpus_dir)
        write_dataset(
            artifact,
            config,
            task_dir,
            filename="all.jsonl",
            extra_manifest={"positive_control": positive_control},
        )
        results[task] = artifact.counts
    return results


def stage_validate(
    config: Config, tasks: Sequence[str], *, with_laya: bool = False
) -> Dict[str, Any]:
    """Validate every generated dataset, structurally and (optionally) by token.

    Every split is checked, not only the train file: a validation or test split
    that does not satisfy the schema would fail later inside Laya's item
    builder, where the error names an item index rather than a row.
    """
    from .dataset import load_rows, validate_rows, validate_with_tokenizer

    directory = ensure_dir(config.stage_dir(DATASET_DIR))
    reports: Dict[str, Any] = {}
    problems: List[str] = []

    for task in tasks:
        task_reports: Dict[str, Any] = {}
        task_dir = directory / task
        for split in ("train", "validation", "test"):
            path = task_dir / f"{split}.jsonl"
            if not path.is_file():
                problems.append(
                    f"{task}/{split}: {path} does not exist — run `build` first"
                )
                continue
            rows = load_rows(path)
            if with_laya:
                report = validate_with_tokenizer(config, rows, task)
            else:
                report = validate_rows(config, rows, task, token_counter=None)
            task_reports[split] = report.to_dict()
            problems.extend(
                f"{task}/{split}: {problem}" for problem in report.problems
            )
        if task_reports:
            reports[task] = task_reports

    payload = {"tasks": reports, "problems": problems, "with_laya": with_laya}
    write_json(directory / "validation.json", payload, config.corpus_dir)
    if problems:
        raise PipelineError(
            f"{len(problems)} validation problem(s); see "
            f"{directory / 'validation.json'}"
        )
    return payload


def stage_stats(config: Config, tasks: Sequence[str]) -> Dict[str, Any]:
    """Counts, distributions and warnings across the built datasets."""
    examples = load_examples(config)
    chunks = load_chunks(config)
    payload = {
        "chunks": chunk_stats(chunks),
        "examples": example_stats(config, examples),
        "by_task": {task: example_stats(config, examples, task) for task in tasks},
        "sufficiency": sufficiency_balance(examples),
        "split_coverage": split_coverage(examples),
    }
    directory = ensure_dir(config.stage_dir(DATASET_DIR))
    write_json(directory / "stats.json", payload, config.corpus_dir)
    write_text(
        directory / "stats.md",
        render_markdown("Dataset statistics", payload),
        config.corpus_dir,
    )
    return payload