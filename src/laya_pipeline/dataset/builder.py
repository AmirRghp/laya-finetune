"""Conversion to Laya's typed-decisions JSONL.

Emits exactly ``{state, questions, expected}`` plus ``tags`` and ``language``,
which is what ``laya.train.read_jsonl`` and ``laya.evals.Dataset`` both read
(AGENTS.md 3.4).

Two invariants this module enforces:

*One row, one decision.* ``items_from_rows`` discards every question on a row
whose ``state`` is empty, so each ``(query, chunk)`` pair becomes its own row
(AGENTS.md 3.5).

*One question definition.* The question comes from ``config.laya.questions`` and
nowhere else, so the schema at training time is byte-identical to the one the
operator must use at inference (AGENTS.md 3.7).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import Config
from ..dataset.records import SPLITS, LabeledExample
from ..errors import ConfigError, DatasetValidationError
from ..ids import content_hash, short_hash
from ..io import ensure_dir, file_sha256, write_json, write_jsonl, write_manifest
from ..logging_setup import get_logger
from ..tokenizer import TokenCounter

__all__ = [
    "DatasetArtifact",
    "POSITIVE_CONTROL_TEMPLATE",
    "render_state",
    "question_definition",
    "build_row",
    "build_rows",
    "write_dataset",
]

LOGGER = get_logger("dataset.builder")

#: The label-encoding positive control the official Laya guidance recommends for
#: diagnosing a run that stays at chance: a tag the model can read the answer off.
#: Enabled per-run with ``train --positive-control``; it is a diagnostic, never a
#: default, and its rows are written to a separate file.
POSITIVE_CONTROL_TEMPLATE = "label={label}; "

_BLANK_RUN = re.compile(r"\n{3,}")
_CHUNKS_PLACEHOLDER = "{{chunks}}"
_QUERY_PLACEHOLDER = "{{query}}"


@dataclass
class DatasetArtifact:
    """What ``write_dataset`` produced."""

    task: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    questions: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None
    manifest_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "rows": len(self.rows),
            "questions": self.questions,
            "counts": dict(self.counts),
            "path": str(self.path) if self.path else None,
        }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_state(
    template: str, query: str, chunks: Sequence[str], *, numbered: bool = False
) -> str:
    """Render the Laya ``state`` from a task template.

    Chunk boundaries become paragraph breaks. ``numbered`` prefixes each chunk
    with an index for tasks where the model must reason over a *set* of chunks —
    sufficiency asks about the set as a whole, relevance about one chunk, and an
    index on a single chunk would only invite the model to key on it.
    """
    if numbered and len(chunks) > 1:
        body = "\n\n".join(
            f"[{index}] {text.strip()}" for index, text in enumerate(chunks, start=1)
        )
    else:
        body = "\n\n".join(text.strip() for text in chunks)

    rendered = template.replace(_QUERY_PLACEHOLDER, query.strip())
    rendered = rendered.replace(_CHUNKS_PLACEHOLDER, body)
    rendered = _BLANK_RUN.sub("\n\n", rendered)
    return rendered.strip()


def question_definition(config: Config, task: str) -> Dict[str, Any]:
    """The Laya question for ``task``, exactly as inference must present it."""
    definition = config.laya.question(task)
    question: Dict[str, Any] = {
        "type": definition["type"],
        "instructions": definition["instructions"],
    }
    criteria = definition.get("criteria")
    if criteria:
        question["criteria"] = dict(criteria)
    elif definition["type"] == "noul":
        pass  # a noul question needs no criteria
    else:
        raise ConfigError(
            f"laya.questions.{task} is a {definition['type']!r} question with no criteria"
        )
    return question


def template_needs_chunks(config: Config, task: str) -> bool:
    return _CHUNKS_PLACEHOLDER in config.laya.template(task)


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def build_row(
    config: Config,
    example: LabeledExample,
    task: str,
    *,
    positive_control: bool = False,
) -> Dict[str, Any]:
    """Convert one labeled example into a Laya training row."""
    question_id = task
    options = config.laya.options(task)
    if example.label not in options:
        raise DatasetValidationError(
            f"example {example.example_id!r} is labelled {example.label!r}, which is not "
            f"an option of the {task!r} question ({options})"
        )

    needs_chunks = template_needs_chunks(config, task)
    if needs_chunks and not example.chunks:
        raise DatasetValidationError(
            f"example {example.example_id!r} has no chunks but the {task!r} state "
            "template expects them; an empty state would make Laya discard the row"
        )
    if not needs_chunks and example.chunks:
        # Routing examples carry no chunks. Anything else is a wiring mistake
        # that would silently drop the evidence from the state.
        raise DatasetValidationError(
            f"example {example.example_id!r} carries {len(example.chunks)} chunk(s) but "
            f"the {task!r} state template has no chunks placeholder; the evidence would "
            "be dropped from the state"
        )

    numbered = len(example.chunks) > 1
    state = render_state(
        config.laya.template(task),
        example.query,
        [chunk.text for chunk in example.chunks],
        numbered=numbered,
    )
    if positive_control:
        state = POSITIVE_CONTROL_TEMPLATE.format(label=example.label) + state

    tags = [f"task:{task}", f"role:{example.role}"]
    if example.split:
        tags.append(f"split:{example.split}")
    if example.strategy:
        tags.append(f"strategy:{example.strategy}")
    if example.label_source:
        tags.append(f"source:{example.label_source}")

    return {
        "state": state,
        "questions": {question_id: question_definition(config, task)},
        "expected": {question_id: example.label},
        "tags": tags,
        "language": config.laya.language,
    }


def build_rows(
    config: Config,
    examples: Sequence[LabeledExample],
    task: str,
    *,
    positive_control: bool = False,
) -> DatasetArtifact:
    """Convert a task's examples, reporting what was skipped and why."""
    artifact = DatasetArtifact(task=task)
    artifact.questions = {task: question_definition(config, task)}

    skipped: Dict[str, int] = {}
    by_split: Dict[str, int] = {name: 0 for name in SPLITS}
    by_label: Dict[str, int] = {option: 0 for option in config.laya.options(task)}

    for example in examples:
        if example.task != task:
            skipped["other_task"] = skipped.get("other_task", 0) + 1
            continue
        if example.split is None:
            skipped["unassigned_split"] = skipped.get("unassigned_split", 0) + 1
            continue
        try:
            row = build_row(config, example, task, positive_control=positive_control)
        except DatasetValidationError:
            skipped["invalid"] = skipped.get("invalid", 0) + 1
            continue
        artifact.rows.append(row)
        by_split[example.split] = by_split.get(example.split, 0) + 1
        by_label[example.label] = by_label.get(example.label, 0) + 1

    artifact.counts = {
        "rows": len(artifact.rows),
        "by_split": by_split,
        "by_label": by_label,
        "skipped": skipped,
        "positive_control": positive_control,
    }
    return artifact


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_dataset(
    artifact: DatasetArtifact,
    config: Config,
    directory: Path,
    *,
    filename: Optional[str] = None,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> DatasetArtifact:
    """Write ``<task>.jsonl``, ``questions.json`` and ``manifest.json``.

    ``questions.json`` is written here as well as by ``laya.train.finetune`` so
    that the question definition travels with the dataset *before* training
    starts — a dataset whose questions cannot be reproduced is not reviewable.
    """
    target = ensure_dir(Path(directory), config.corpus_dir)
    corpus_dir = config.corpus_dir
    name = filename or f"{artifact.task}.jsonl"

    path = target / name
    write_jsonl(path, artifact.rows, corpus_dir)
    artifact.path = path

    write_json(target / "questions.json", artifact.questions, corpus_dir)
    fingerprint = questions_fingerprint(artifact.questions)

    counts = dict(artifact.counts)
    # Identity of the *content*, not the path: moving or renaming the dataset
    # file does not change what it contains, and a label change must move the
    # hash, so both the states and the expected answers are covered.
    counts["dataset_sha256"] = content_hash(
        "\n".join(str(row.get("state", "")) for row in artifact.rows),
        "\n".join(
            json.dumps(row.get("expected", {}), ensure_ascii=False, sort_keys=True)
            for row in artifact.rows
        ),
    )
    artifact.manifest_path = write_manifest(
        target,
        stage=f"build:{artifact.task}",
        config_hash=config.config_hash,
        seed=config.project.seed,
        counts=counts,
        inputs={
            "questions_sha256": fingerprint,
            "question_id_sha256": content_hash(artifact.task, fold=False),
            "base_model_sha256": content_hash(config.laya.base_model, fold=False),
            "max_len_sha256": content_hash(str(config.laya.max_len), fold=False),
            "head_max_len_sha256": content_hash(str(config.laya.head_max_len), fold=False),
            "examples_sha256": file_sha256(
                config.stage_dir("labels") / "labeled.jsonl"
            ),
        },
        extra=extra_manifest,
        corpus_dir=corpus_dir,
    )

    LOGGER.info(
        "dataset written",
        extra={
            "fields": {
                "task": artifact.task,
                "rows": len(artifact.rows),
                "path": str(path),
            }
        },
    )
    return artifact


def questions_fingerprint(questions: Mapping[str, Any]) -> str:
    """A stable hash of *what was asked*.

    Deliberately independent of how many rows exist: adding states to an
    unchanged question set is not a change to the decision being learned.
    """
    canonical = json.dumps(
        {str(k): _canonical_question(v) for k, v in sorted(questions.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return short_hash(canonical, fold=False)


def _canonical_question(question: Any) -> Any:
    if not isinstance(question, Mapping):
        return question
    canonical: Dict[str, Any] = {
        "type": question.get("type"),
        "instructions": question.get("instructions"),
    }
    criteria = question.get("criteria")
    if isinstance(criteria, Mapping):
        canonical["criteria"] = {str(k): v for k, v in criteria.items()}
    elif isinstance(criteria, (list, tuple)):
        canonical["criteria"] = list(criteria)
    if "labels" in question:
        canonical["labels"] = question["labels"]
    return canonical


def state_token_report(
    config: Config, rows: Sequence[Mapping[str, Any]], counter: TokenCounter
) -> Dict[str, Any]:
    """Token lengths of the rendered states, against the Laya state budget."""
    lengths = [counter.count(str(row.get("state", ""))) for row in rows]
    if not lengths:
        return {"count": 0, "source": counter.source}
    over = [length for length in lengths if length > config.laya.max_len]
    return {
        "count": len(lengths),
        "source": counter.source,
        "max_len_budget": config.laya.max_len,
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 1),
        "over_budget": len(over),
        "over_budget_fraction": round(len(over) / len(lengths), 4),
    }
