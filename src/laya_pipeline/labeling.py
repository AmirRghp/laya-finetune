"""Teacher labeling, human review queue, and label conflict detection.

Three rules this module exists to enforce (AGENTS.md 6.10, 6.11):

* a human-reviewed label always wins over an automatic one, and both are kept;
* the teacher's confidence is recorded as metadata and never used as a proxy for
  correctness — it only decides *who looks at it next*;
* ``uncertain`` is a discard, not a label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import Config
from .dataset.records import (
    LABEL_SOURCES,
    SPLITS,
    LabeledExample,
    strongest_label_source,
)
from .errors import LabelingError, OllamaError, SchemaError
from .ids import content_hash
from .io import read_csv_rows, write_csv
from .llm.ollama import OllamaClient
from .llm.prompts import PromptLibrary
from .logging_setup import get_logger
from .tasks.base import UNCERTAIN, Teacher
from .tasks.sufficiency import render_chunks_text

__all__ = [
    "ReviewItem",
    "LabelingReport",
    "REVIEW_FIELDS",
    "REVIEW_PROMPTS",
    "propose_labels",
    "build_review_queue",
    "export_review_csv",
    "read_review_csv",
    "apply_corrections",
    "detect_contradictions",
    "label_distribution",
    "REVIEW_PROMPTS",
]

LOGGER = get_logger("labeling")

#: Which prompt decides each task's label.
REVIEW_PROMPTS: Dict[str, str] = {
    "relevance": "label_relevance",
    "sufficiency": "label_sufficiency",
    "routing": "label_routing",
}

#: Columns of the reviewer-facing CSV. Deliberately flat and self-describing:
#: a reviewer should need nothing but this file.
REVIEW_FIELDS = (
    "example_id",
    "task",
    "split",
    "role",
    "query",
    "chunks_preview",
    "proposed_label",
    "final_label",
    "confidence",
    "status",
    "strategy",
    "relpath",
    "notes",
)


@dataclass
class ReviewItem:
    """One row in the human review queue."""

    example_id: str
    task: str
    query: str
    chunks_preview: str
    proposed_label: str
    final_label: str
    confidence: Optional[float]
    status: str
    split: Optional[str] = None
    role: str = ""
    strategy: Optional[str] = None
    relpath: str = ""
    notes: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in REVIEW_FIELDS}


@dataclass
class LabelingReport:
    examples: List[LabeledExample] = field(default_factory=list)
    queue: List[ReviewItem] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def bump(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount


# --------------------------------------------------------------------------- #
# Proposal
# --------------------------------------------------------------------------- #
def propose_labels(
    examples: Sequence[LabeledExample],
    config: Config,
    client: OllamaClient,
    prompts: PromptLibrary,
    *,
    teacher: Optional[Teacher] = None,
) -> LabelingReport:
    """Ask the teacher to label examples whose label is not already settled.

    Positives labeled by rule are left alone: they are the source chunk, and a
    teacher re-deciding that adds cost and variance without adding information.
    Human labels are never overwritten.
    """
    report = LabelingReport()
    resolved_teacher = teacher or Teacher(client, prompts, config)

    for example in examples:
        if example.label_source == "human":
            report.examples.append(example)
            report.bump("kept_human_labeled")
            continue
        if example.role == "positive" and example.label_source == "rule":
            report.examples.append(example)
            report.bump("kept_rule_labeled")
            continue

        prompt_name = REVIEW_PROMPTS.get(example.task)
        if prompt_name is None:
            report.examples.append(example)
            report.bump("kept_no_prompt_for_task", 1)
            continue

        try:
            verdict = _label_example(example, config, resolved_teacher, prompt_name)
        except (OllamaError, SchemaError) as exc:
            report.bump("labeling_failed")
            report.failures.append(
                {
                    "example_id": example.example_id,
                    "task": example.task,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            LOGGER.warning(
                "teacher labeling failed",
                extra={"fields": {"example_id": example.example_id, "error": str(exc)}},
            )
            continue

        label = verdict["label"]
        if label == UNCERTAIN:
            report.bump("discarded_uncertain")
            continue

        options = config.laya.options(example.task)
        if label not in options:
            report.bump("discarded_label_not_an_option")
            report.failures.append(
                {
                    "example_id": example.example_id,
                    "task": example.task,
                    "error": f"teacher returned {label!r}, not an option of {options}",
                }
            )
            continue

        confidence = float(verdict["confidence"])
        updated = _with_label(example, label, confidence, verdict, resolved_teacher.model)
        report.examples.append(updated)

        report.queue.append(
            _to_review_item(updated, status=_review_status(confidence, config))
        )
        report.bump("queued_for_review")

        report.bump(f"label_{label}")

    report.bump("teacher_calls", resolved_teacher.calls)
    LOGGER.info(
        "labeling finished",
        extra={"fields": {"examples": len(report.examples), **report.stats}},
    )
    return report


def _label_example(
    example: LabeledExample, config: Config, teacher: Teacher, prompt_name: str
) -> Dict[str, Any]:
    """Dispatch to the prompt that matches how this example is judged."""
    options = config.laya.options(example.task)

    if prompt_name == "label_sufficiency":
        return teacher.label_sufficiency(
            query=example.query,
            chunks_text=render_chunks_text(example.chunks),
            key_points=example.key_points,
            options=options,
        )

    if prompt_name == "label_relevance":
        text = example.chunks[0].text if example.chunks else ""
        return teacher.label_choice(
            prompt_name, options=options, query=example.query, chunk_text=text
        )

    # Routing: only the query crosses the boundary (AGENTS.md 6.6).
    return teacher.label_choice(prompt_name, options=options, query=example.query)


def _with_label(
    example: LabeledExample,
    label: str,
    confidence: float,
    verdict: Mapping[str, Any],
    teacher_model: str,
) -> LabeledExample:
    """Return a copy carrying the teacher's label, preserving the proposal."""
    metadata = dict(example.metadata)
    metadata["proposed_label"] = {
        "label": label,
        "confidence": confidence,
        "reason": verdict.get("reason", ""),
        "model": teacher_model,
        "prompt": verdict.get("prompt", ""),
    }
    if "covered" in verdict:
        metadata["covered"] = verdict.get("covered", [])
    return LabeledExample(
        example_id=example.example_id,
        task=example.task,
        query=example.query,
        chunks=list(example.chunks),
        label=label,
        label_source="teacher",
        confidence=confidence,
        strategy=example.strategy,
        margin=example.margin,
        verdict=example.verdict,
        role=example.role,
        split=example.split,
        source_id=example.source_id,
        relpath=example.relpath,
        section_id=example.section_id,
        key_points=list(example.key_points),
        metadata=metadata,
    )


def _review_status(confidence: float, config: Config) -> str:
    """Route every teacher proposal for review without treating confidence as truth."""
    settings = config.labeling
    if settings.uncertain_low <= confidence <= settings.uncertain_high:
        return "uncertain"
    if confidence < settings.review_threshold:
        return "needs_review"
    return "needs_second_opinion"


def _to_review_item(example: LabeledExample, *, status: str) -> ReviewItem:
    preview = "\n\n".join(chunk.text for chunk in example.chunks) or "(بدون متن)"
    if len(preview) > 600:
        preview = preview[:600] + " …"
    return ReviewItem(
        example_id=example.example_id,
        task=example.task,
        query=example.query,
        chunks_preview=preview,
        proposed_label=example.label,
        final_label=example.label,
        confidence=example.confidence,
        status=status,
        split=example.split,
        role=example.role,
        strategy=example.strategy,
        relpath=example.relpath,
        notes=str(example.metadata.get("review_notes", "")),
    )


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #
def build_review_queue(
    examples: Sequence[LabeledExample], config: Config
) -> List[ReviewItem]:
    """Everything a human should look at.

    Queues on two independent grounds: low teacher confidence, and a label the
    rules derived rather than the teacher (a mined negative, say), which has had
    no independent second opinion at all.
    """
    queue: List[ReviewItem] = []
    for example in examples:
        confidence = example.confidence
        pending_teacher = (
            example.label_source == "teacher" and not example.metadata.get("reviewed")
        )
        pending_review = not example.metadata.get("reviewed")
        low_confidence = (
            pending_review
            and confidence is not None
            and confidence < config.labeling.review_threshold
        )
        needs_second_opinion = pending_review and example.role == "negative"
        if pending_teacher or low_confidence or needs_second_opinion:
            queue.append(
                _to_review_item(
                    example,
                    status=(
                        _review_status(confidence, config)
                        if pending_teacher and confidence is not None
                        else "needs_review" if low_confidence else "needs_second_opinion"
                    ),
                )
            )
    queue.sort(key=lambda item: (item.task, item.confidence if item.confidence is not None else -1))
    return queue


def export_review_csv(items: Sequence[ReviewItem], path: Path, corpus_dir: Optional[Path] = None) -> Path:
    """Write the queue as CSV for manual correction."""
    return write_csv(Path(path), [item.to_row() for item in items], REVIEW_FIELDS, corpus_dir)


def read_review_csv(path: Path) -> List[Dict[str, str]]:
    """Read a corrected review file. Rows without an ``example_id`` are refused."""
    rows = read_csv_rows(Path(path))
    if not rows:
        raise LabelingError(f"{path} contains no rows")
    for index, row in enumerate(rows, start=2):
        if not (row.get("example_id") or "").strip():
            raise LabelingError(
                f"{path}:{index} has no example_id; corrections must name the example "
                "they correct, otherwise they cannot be matched back"
            )
    return rows


def apply_corrections(
    examples: Sequence[LabeledExample],
    rows: Iterable[Mapping[str, Any]],
    *,
    options_by_task: Mapping[str, Sequence[str]],
    reviewer: str = "human",
) -> tuple[List[LabeledExample], Dict[str, int]]:
    """Apply human corrections. A human label always outranks the proposal.

    The automatic label is preserved under ``metadata.proposed_label`` and under
    ``metadata.superseded_label`` so the original decision stays auditable.
    """
    by_id = {example.example_id: example for example in examples}
    corrections: Dict[str, str] = {}
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        final = str(row.get("final_label", "")).strip()
        if not final:
            continue
        if example_id in corrections and corrections[example_id] != final:
            raise LabelingError(
                f"conflicting review labels for example {example_id!r}"
            )
        corrections[example_id] = final

    stats = {"matched": 0, "unknown_id": 0, "changed": 0, "unchanged": 0, "invalid_label": 0}
    result: List[LabeledExample] = []

    for example in examples:
        proposed = corrections.get(example.example_id)
        if proposed is None:
            result.append(example)
            continue
        stats["matched"] += 1
        options = list(options_by_task.get(example.task, []))
        if options and proposed not in options:
            stats["invalid_label"] += 1
            LOGGER.warning(
                "ignoring a correction whose label is not an option",
                extra={
                    "fields": {
                        "example_id": example.example_id,
                        "proposed": proposed,
                        "options": options,
                    }
                },
            )
            result.append(example)
            continue
        changed = proposed != example.label
        stats["changed" if changed else "unchanged"] += 1
        metadata = dict(example.metadata)
        if changed:
            metadata["superseded_label"] = {
                "label": example.label,
                "label_source": example.label_source,
                "confidence": example.confidence,
            }
        else:
            metadata["confirmed_label"] = proposed
        metadata["reviewed"] = True
        metadata["reviewer"] = reviewer
        notes = metadata.get("review_notes")
        result.append(
            LabeledExample(
                example_id=example.example_id,
                task=example.task,
                query=example.query,
                chunks=list(example.chunks),
                label=proposed,
                label_source=strongest_label_source(example.label_source, "human"),
                confidence=example.confidence,
                strategy=example.strategy,
                margin=example.margin,
                verdict=example.verdict,
                role=example.role,
                split=example.split,
                source_id=example.source_id,
                relpath=example.relpath,
                section_id=example.section_id,
                key_points=list(example.key_points),
                metadata={**metadata, "review_notes": notes or ""},
            )
        )

    for example_id in corrections:
        if example_id not in by_id:
            stats["unknown_id"] += 1

    return result, stats


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def detect_contradictions(
    examples: Sequence[LabeledExample]
) -> List[Dict[str, Any]]:
    """Identical decision inputs carrying different labels.

    This is a data defect, not a modelling problem: the same state and question
    cannot have two correct answers. Human-reviewed labels are reported first so
    the unreviewed side is the one to fix.
    """
    buckets: Dict[str, List[LabeledExample]] = {}
    for example in examples:
        key = f"{example.task}|{example.content_signature()}"
        buckets.setdefault(key, []).append(example)

    contradictions: List[Dict[str, Any]] = []
    for key, group in buckets.items():
        labels = {example.label for example in group}
        if len(labels) <= 1:
            continue
        ordered = sorted(group, key=lambda e: LABEL_SOURCES.index(e.label_source))
        contradictions.append(
            {
                "key": key,
                "task": group[0].task,
                "query": group[0].query,
                "labels": sorted(labels),
                "example_ids": [example.example_id for example in group],
                "suggested_label": ordered[-1].label,
                "suggested_source": ordered[-1].label_source,
            }
        )
    return contradictions


def label_distribution(
    examples: Sequence[LabeledExample], options_by_task: Mapping[str, Sequence[str]] | None = None
) -> Dict[str, Dict[str, int]]:
    """Label counts per task, per split, and per role.

    Every configured option appears even at zero: an absent class is a finding,
    and a report that omits it hides it.
    """
    distribution: Dict[str, Dict[str, int]] = {}
    for example in examples:
        task_block = distribution.setdefault(
            example.task,
            {"all": 0, "by_split": {}, "by_role": {}, "by_label": {}},
        )
        task_block["all"] += 1
        split_key = example.split or "unassigned"
        task_block["by_split"][split_key] = task_block["by_split"].get(split_key, 0) + 1
        task_block["by_role"][example.role] = task_block["by_role"].get(example.role, 0) + 1
        task_block["by_label"][example.label] = task_block["by_label"].get(example.label, 0) + 1

    if options_by_task:
        for task, options in options_by_task.items():
            block = distribution.setdefault(
                task, {"all": 0, "by_split": {}, "by_role": {}, "by_label": {}}
            )
            for option in options:
                block["by_label"].setdefault(option, 0)

    return distribution


def example_fingerprint(example: LabeledExample) -> str:
    """Stable identity of a decision's inputs, for de-duplication."""
    return content_hash(example.task, example.query, *[c.text for c in example.chunks])


def split_summary(examples: Sequence[LabeledExample]) -> Dict[str, Dict[str, int]]:
    """Example counts per task per split."""
    summary: Dict[str, Dict[str, int]] = {}
    for example in examples:
        task_block = summary.setdefault(example.task, {name: 0 for name in SPLITS})
        if example.split in task_block:
            task_block[example.split] += 1  # type: ignore[index]
    return summary
