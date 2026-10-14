"""Dataset validation.

Runs before anything expensive happens. Every check names the offending row so a
failure is actionable without re-reading the whole file.

Split into two tiers:

``structural``
    Field presence, question type, label validity, option consistency, duplicate
    rows, split integrity. No dependencies — always runs.

``measured``
    State token length against ``laya.max_len`` using the real tokenizer. Skipped
    (and reported as skipped) when no tokenizer is available, never silently
    assumed to pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import Config
from ..dataset.builder import question_definition, template_needs_chunks
from ..dataset.records import SPLITS, LabeledExample
from ..errors import DatasetValidationError
from ..ids import content_hash
from ..io import read_jsonl
from ..logging_setup import get_logger
from ..tokenizer import TokenCounter, resolve_token_counter

__all__ = ["ValidationReport", "validate_rows", "validate_examples", "load_rows"]

#: Hard ceiling on a choice question's option count. Laya's structured layer
#: rejects more than ``MAX_OPTIONS``; this project stays well under it because
#: every option's description competes for a 256-token question head.
MAX_OPTIONS_ALLOWED = 8

LOGGER = get_logger("dataset.validate")

_REQUIRED_ROW_FIELDS = ("state", "questions", "expected")
_ALLOWED_ROW_FIELDS = frozenset((*_REQUIRED_ROW_FIELDS, "tags", "language"))


@dataclass
class ValidationReport:
    """The full picture, not just the first failure."""

    task: str = ""
    ok: bool = True
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    measured: Dict[str, Any] = field(default_factory=dict)

    def problem(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_failed(self) -> "ValidationReport":
        if not self.ok:
            preview = "\n  - ".join(self.problems[:20])
            suffix = "" if len(self.problems) <= 20 else f"\n  ... and {len(self.problems) - 20} more"
            raise DatasetValidationError(
                f"{self.task or 'dataset'} failed validation with "
                f"{len(self.problems)} problem(s):\n  - {preview}{suffix}",
                problems=list(self.problems),
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "ok": self.ok,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
            "counts": dict(self.counts),
            "measured": dict(self.measured),
        }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL dataset into row dicts."""
    return list(read_jsonl(Path(path)))


# --------------------------------------------------------------------------- #
# Structural validation
# --------------------------------------------------------------------------- #
def validate_rows(
    config: Config,
    rows: Sequence[Mapping[str, Any]],
    task: str,
    *,
    token_counter: Optional[TokenCounter] = None,
    check_tokens: bool = True,
    require_splits: bool = True,
) -> ValidationReport:
    """Validate a Laya dataset against the configured question definition."""
    report = ValidationReport(task=task)
    options = config.laya.options(task)
    expected_question = question_definition(config, task)
    question_id = task
    needs_chunks = template_needs_chunks(config, task)

    if not rows:
        report.problem("the dataset contains no rows")
        return report

    seen: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    by_label: Dict[str, int] = {option: 0 for option in options}
    state_lengths: List[int] = []
    counter = token_counter

    for index, row in enumerate(rows, start=1):
        where = f"row {index}"

        unexpected = set(row) - _ALLOWED_ROW_FIELDS
        if unexpected:
            report.problem(
                f"{where}: unexpected Laya row field(s) {sorted(unexpected)}; allowed "
                f"fields are {sorted(_ALLOWED_ROW_FIELDS)}"
            )

        for field_name in _REQUIRED_ROW_FIELDS:
            if field_name not in row:
                report.problem(f"{where}: missing required field {field_name!r}")

        state = row.get("state")
        if not isinstance(state, str) or not state.strip():
            report.problem(
                f"{where}: 'state' must be a non-empty string. Laya's items_from_rows "
                "discards every question on a row whose state is empty, so such a row "
                "is silently dropped from training."
            )
        else:
            signature = content_hash(state)
            if signature in seen:
                report.problem(
                    f"{where}: duplicate state, identical to row {seen[signature]}"
                )
            else:
                seen[signature] = index
            if counter is not None:
                state_lengths.append(counter.count(state))

        questions = row.get("questions")
        if not isinstance(questions, Mapping):
            report.problem(f"{where}: 'questions' must be an object")
        else:
            if set(questions) != {question_id}:
                report.problem(
                    f"{where}: questions {sorted(questions)} but this dataset answers "
                    f"only {question_id!r}"
                )
            definition = questions.get(question_id)
            if not isinstance(definition, Mapping):
                report.problem(f"{where}: questions.{question_id} must be an object")
            else:
                _validate_question(report, where, definition, expected_question, options, task)

        expected = row.get("expected")
        if not isinstance(expected, Mapping):
            report.problem(f"{where}: 'expected' must be an object")
        elif question_id not in expected:
            report.problem(f"{where}: expected is missing {question_id!r}")
        else:
            label = expected[question_id]
            if not isinstance(label, str):
                report.problem(f"{where}: expected.{question_id} must be a string label")
            elif label not in options:
                report.problem(
                    f"{where}: expected.{question_id} is {label!r}, not one of {options}"
                )
            else:
                by_label[label] = by_label.get(label, 0) + 1

        if needs_chunks:
            text = state if isinstance(state, str) else ""
            if "متن" not in text and "متن‌" not in text and text:
                report.warn(
                    f"{where}: the rendered state has no chunk section; if this row was "
                    "meant to carry retrieved evidence it was dropped"
                )

        if require_splits:
            split = _split_of(row)
            if split is None:
                report.problem(
                    f"{where}: no split tag; every row must declare train/validation/test "
                    "so the trainer and the evaluator agree on what is held out"
                )
            elif split not in SPLITS:
                report.problem(f"{where}: split {split!r} is not one of {list(SPLITS)}")
            else:
                by_split[split] = by_split.get(split, 0) + 1

    missing_labels = [option for option, count in by_label.items() if count == 0]
    if missing_labels:
        report.warn(
            f"option(s) {missing_labels} appear in the question definition but never in "
            "the data; the model can never learn them and ECE over that option is undefined"
        )
    for split in SPLITS:
        if split not in by_split:
            report.warn(f"split {split!r} has no rows")

    if state_lengths and counter is not None:
        over = [length for length in state_lengths if length > config.laya.max_len]
        report.measured = {
            "token_source": counter.source,
            "max_len_budget": config.laya.max_len,
            "min": min(state_lengths),
            "max": max(state_lengths),
            "mean": round(sum(state_lengths) / len(state_lengths), 1),
            "over_budget": len(over),
        }
        if over:
            report.problem(
                f"{len(over)} state(s) exceed laya.max_len ({config.laya.max_len}) and "
                "will be truncated during training and inference"
            )
    elif check_tokens and counter is None:
        report.measured = {"token_source": None, "skipped": "no token counter available"}
        report.warn(
            "state token lengths were not checked: no Laya tokenizer was available. "
            "Run `validate` again with transformers installed, or accept the risk."
        )

    report.counts = {
        "rows": len(rows),
        "by_split": by_split,
        "by_label": by_label,
        "unique_states": len(seen),
    }
    return report


def _validate_question(
    report: ValidationReport,
    where: str,
    definition: Mapping[str, Any],
    expected: Mapping[str, Any],
    options: Sequence[str],
    task: str,
) -> None:
    qtype = definition.get("type")
    if qtype != expected["type"]:
        report.problem(
            f"{where}: question type is {qtype!r} but the configured {task!r} question "
            f"is {expected['type']!r}"
        )
        return

    instructions = definition.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        report.problem(
            f"{where}: 'instructions' must be a non-empty string; Laya rejects a "
            "question without one"
        )
    elif instructions != expected["instructions"]:
        report.problem(
            f"{where}: 'instructions' differs from the configured definition. Training "
            "on wording inference never sees teaches the model a prompt that does not exist."
        )

    if qtype == "choice":
        criteria = definition.get("criteria")
        if not isinstance(criteria, Mapping) or not criteria:
            report.problem(
                f"{where}: a choice question needs a non-empty 'criteria' mapping"
            )
            return
        labels = list(criteria)
        if labels != list(options):
            report.problem(
                f"{where}: criteria {labels} but the configured options are {list(options)}; "
                "option order is part of the question identity"
            )
        expected_criteria = expected.get("criteria")
        if isinstance(expected_criteria, Mapping):
            # The description is the option text the model actually reads, so a
            # reworded description is a different question — not a cosmetic edit.
            for label in labels:
                if label not in expected_criteria:
                    report.problem(
                        f"{where}: criteria has {label!r}, which is not an option of the "
                        f"configured {task!r} question"
                    )
                elif criteria[label] != expected_criteria[label]:
                    report.problem(
                        f"{where}: criteria[{label!r}] differs from the configured "
                        f"description; the option text is part of the prompt"
                    )
        if len(labels) > MAX_OPTIONS_ALLOWED:
            report.problem(
                f"{where}: {len(labels)} options exceeds the {MAX_OPTIONS_ALLOWED} this "
                "project caps"
            )
        for label, description in criteria.items():
            if not isinstance(description, str) or not description.strip():
                report.problem(
                    f"{where}: criteria[{label!r}] must be a non-empty description; it is "
                    "the option text the model reads"
                )


def _split_of(row: Mapping[str, Any]) -> Optional[str]:
    tags = row.get("tags")
    if isinstance(tags, (list, tuple)):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("split:"):
                return tag.split(":", 1)[1]
    return None


# --------------------------------------------------------------------------- #
# Example-level validation
# --------------------------------------------------------------------------- #
def validate_examples(
    config: Config, examples: Sequence[LabeledExample], task: Optional[str] = None
) -> ValidationReport:
    """Validate internal examples before they become rows."""
    report = ValidationReport(task=task or "")
    options_by_task = {
        name: config.laya.options(name) for name in config.laya.question_ids()
    }
    seen: Dict[str, str] = {}

    for example in examples:
        where = f"example {example.example_id or '<no id>'}"
        if task and example.task != task:
            continue
        options = options_by_task.get(example.task, [])
        if options and example.label not in options:
            report.problem(
                f"{where}: label {example.label!r} is not an option of the "
                f"{example.task!r} question ({options})"
            )
        if not example.query.strip():
            report.problem(f"{where}: empty query")
        needs_chunks = template_needs_chunks(config, example.task) if example.task in options_by_task else False
        if needs_chunks and not example.chunks:
            report.problem(f"{where}: {example.task!r} needs chunks but has none")
        if not needs_chunks and example.chunks:
            report.problem(
                f"{where}: {example.task!r} takes no chunks but carries "
                f"{len(example.chunks)}"
            )
        if example.split is None:
            report.problem(f"{where}: no split assigned")
        signature = example.content_signature()
        if signature in seen:
            report.problem(f"{where}: duplicate decision, same as {seen[signature]}")
        else:
            seen[signature] = example.example_id

    report.counts = {"examples": sum(1 for e in examples if not task or e.task == task)}
    return report


def validate_with_tokenizer(
    config: Config,
    rows: Sequence[Mapping[str, Any]],
    task: str,
    *,
    allow_download: bool = True,
    allow_estimates: bool = False,
) -> ValidationReport:
    """Validate including real token counts, loading the tokenizer on demand.

    ``allow_estimates=False`` by default here: an estimate that "passes" is not
    evidence, and a measured failure is worth surfacing.
    """
    counter = resolve_token_counter(
        config.chunking, allow_download=allow_download, allow_estimates=allow_estimates
    )
    return validate_rows(config, rows, task, token_counter=counter)
