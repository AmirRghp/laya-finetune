"""Classification evaluation against a Laya checkpoint.

Wraps the official ``laya.evals`` harness rather than reimplementing it, and adds
only what it does not provide:

* macro-F1 and per-class F1 with a confusion matrix;
* for the relevance task, the recall of ``relevant`` and an explicit count of
  **useful evidence discarded** — the asymmetry that matters in a RAG pipeline,
  where throwing away a passage that would have answered the question costs the
  user an answer, while keeping an irrelevant passage costs a little context.

``laya.evals`` supplies accuracy, ECE, Brier, AURC, selective accuracy and
latency. Those are reported as measured; nothing here estimates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..errors import EvaluationError
from ..logging_setup import get_logger

__all__ = [
    "PairSet",
    "evaluate_checkpoint",
    "confusion_matrix",
    "per_class_scores",
    "macro_f1",
    "label_of",
    "answer_label",
    "answer_confidence_of",
    "CALIBRATION_NOTE",
]

LOGGER = get_logger("evaluation.classification")

#: Shown next to every confidence figure, because the base checkpoint ships unfit
#: and a reader must not mistake a number for a calibrated probability.
CALIBRATION_NOTE = (
    "Confidence figures are NOT calibrated unless the checkpoint's training report "
    "records a temperature fitted on a held-out slice. laya-multilingual ships with "
    "temperature [1, 1, 1] and no fitted temperature_by_options."
)


@dataclass
class PairSet:
    """Gold/predicted label pairs recovered from an evaluation report."""

    pairs: List[Tuple[str, str]] = field(default_factory=list)
    source: str = "unavailable"
    note: str = ""

    @property
    def available(self) -> bool:
        return bool(self.pairs)


# --------------------------------------------------------------------------- #
# Metric math (pure Python; no dependencies)
# --------------------------------------------------------------------------- #
def confusion_matrix(
    pairs: Sequence[Tuple[str, str]], labels: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """``matrix[gold][predicted]``. Every configured label appears, even at zero."""
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for gold, predicted in pairs:
        if gold in matrix and predicted in matrix[gold]:
            matrix[gold][predicted] += 1
    return matrix


def per_class_scores(
    pairs: Sequence[Tuple[str, str]], labels: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Precision, recall, F1 and support for each label."""
    matrix = confusion_matrix(pairs, labels)
    scores: Dict[str, Dict[str, float]] = {}
    for label in labels:
        true_positive = matrix[label][label]
        predicted_total = sum(matrix[gold][label] for gold in labels)
        actual_total = sum(matrix[label].values())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        scores[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": actual_total,
        }
    return scores


def macro_f1(scores: Mapping[str, Mapping[str, float]]) -> float | None:
    """Mean F1 over classes that appear in the data.

    Classes with zero support are excluded: averaging in an absent class as 0.0
    punishes a model for data the dataset does not contain.
    """
    supported = [entry["f1"] for entry in scores.values() if entry.get("support")]
    if not supported:
        return None
    return round(sum(supported) / len(supported), 6)


# --------------------------------------------------------------------------- #
# Report inspection
# --------------------------------------------------------------------------- #
def label_of(value: Any) -> Optional[str]:
    """Coerce an answer or expected value into a comparable label string."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(int(value))
    return None


def answer_label(answer: Any, question_id: str) -> Optional[str]:
    """The label an Laya answer reports, across the shapes it may take.

    ``predict`` returns ``{"answers": {qid: ...}}``. Depending on the question
    type and the call shape, the value is either the label itself or an object
    carrying it under one of a few conventional keys. All of them are accepted;
    an unrecognised shape yields ``None`` rather than a guess.
    """
    if answer is None:
        return None
    if isinstance(answer, Mapping):
        for key in ("choice", "value", "label", "answer"):
            if key in answer:
                return label_of(answer[key])
        return None
    return label_of(answer)


def answer_confidence_of(answer: Any) -> Optional[float]:
    """``answer_confidence`` when the answer carries it, else ``None``.

    Deliberately does not fall back to normalized-entropy ``confidence``: those
    are different quantities, and mixing them makes a confidence figure mean
    whichever field happened to be present.
    """
    if isinstance(answer, Mapping):
        raw = answer.get("answer_confidence")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return None


def extract_pairs(report: Any, task: str) -> PairSet:
    """Recover ``(gold, predicted)`` pairs from an ``laya.evals`` report.

    The harness's public metric list does not pin the exact shape of a per-case
    record, so several plausible shapes are probed and the one that worked is
    reported. When none does, the result is an empty :class:`PairSet` with a note,
    and F1/confusion are reported as unavailable rather than as zero
    (AGENTS.md 3.14).
    """
    cases = getattr(report, "cases", None)
    if cases is None and isinstance(report, Mapping):
        cases = report.get("cases")
    if not cases:
        return PairSet(source="none", note="the report carries no per-case records")

    pairs: List[Tuple[str, str]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        expected = case.get("expected")
        if isinstance(expected, Mapping):
            expected = expected.get(task)
        gold = label_of(expected)

        prediction: Optional[str] = None
        for key in ("answer", "answers", "prediction", "predicted", "choice"):
            container = case.get(key)
            if isinstance(container, Mapping):
                if task in container:
                    prediction = answer_label(container[task], task)
                    break
                prediction = answer_label(container, task)
                if prediction is not None:
                    break
            elif container is not None and key == "prediction":
                prediction = label_of(container)
                if prediction is not None:
                    break

        if gold is not None and prediction is not None:
            pairs.append((gold, prediction))

    if not pairs:
        return PairSet(
            source="none",
            note=(
                "per-case records were present but none carried both a gold label and a "
                "prediction in a recognised shape; macro-F1 and the confusion matrix are "
                "therefore unavailable"
            ),
        )
    return PairSet(pairs=pairs, source="expected/answer", note="")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_checkpoint(
    config: Config,
    data_path: Path,
    checkpoint: str,
    *,
    device: str = "auto",
    batch_size: Optional[int] = None,
    task: Optional[str] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Score a labelled Laya dataset against one checkpoint.

    Loads weights: this is a GPU/CPU-inference stage, not a data-preparation one.
    """
    try:
        import laya
        from laya.evals import Dataset, evaluate
    except ImportError as exc:
        raise EvaluationError(
            "evaluation needs the `laya` package. Install it with `pip install -e .[laya]`. "
            "Dataset validation and statistics do not."
        ) from exc

    dataset = Dataset.from_jsonl(str(data_path))
    agent = laya.load(checkpoint)

    LOGGER.info(
        "scoring dataset",
        extra={
            "fields": {
                "checkpoint": checkpoint,
                "dataset": str(data_path),
                "cases": len(dataset),
                "device": device,
            }
        },
    )
    report = evaluate(
        agent,
        dataset,
        batch_size=batch_size if batch_size is not None else config.evaluation.batch_size,
    )

    result: Dict[str, Any] = {
        "label": label or checkpoint,
        "checkpoint": checkpoint,
        "dataset": str(data_path),
        "cases": len(dataset),
        "device": device,
        "overall": dict(getattr(report, "overall", {}) or {}),
        "slices": _slim_slices(getattr(report, "slices", {})),
        "config": dict(getattr(report, "config", {}) or {}),
        "calibration_note": CALIBRATION_NOTE,
    }

    resolved_task = task or _single_task(dataset)
    if resolved_task:
        result["task"] = resolved_task
        labels = config.laya.options(resolved_task)
        pair_set = extract_pairs(report, resolved_task)
        if pair_set.available:
            scores = per_class_scores(pair_set.pairs, labels)
            result["classification"] = {
                "n": len(pair_set.pairs),
                "accuracy": round(
                    sum(1 for gold, predicted in pair_set.pairs if gold == predicted)
                    / len(pair_set.pairs),
                    6,
                ),
                "macro_f1": macro_f1(scores),
                "per_class": scores,
                "confusion": confusion_matrix(pair_set.pairs, labels),
            }
            result["classification"].update(
                _relevance_cost(pair_set.pairs, resolved_task, config)
            )
        else:
            result["classification"] = {
                "available": False,
                "note": pair_set.note,
            }
        result["labels"] = labels

    return result


def _single_task(dataset: Any) -> Optional[str]:
    """The single question id across a dataset, or ``None`` if it is mixed."""
    ids: set[str] = set()
    for example in getattr(dataset, "examples", [])[:50]:
        ids.update((example.questions or {}).keys())
        if len(ids) > 1:
            return None
    return next(iter(ids), None)


def _slim_slices(slices: Any) -> Dict[str, Any]:
    """Per-slice metrics, if the harness reported any."""
    if not isinstance(slices, Mapping):
        return {}
    trimmed: Dict[str, Any] = {}
    for name, by_value in slices.items():
        if not isinstance(by_value, Mapping):
            continue
        trimmed[name] = {
            str(value): dict(metrics)
            for value, metrics in sorted(by_value.items())
            if isinstance(metrics, Mapping)
        }
    return trimmed


def _relevance_cost(
    pairs: Sequence[Tuple[str, str]], task: str, config: Config
) -> Dict[str, Any]:
    """The relevance task's asymmetry: what fraction of useful evidence was dropped."""
    if task != "relevance":
        return {}
    try:
        positive = config.laya.role_label("relevance", "positive")
        negative = config.laya.role_label("relevance", "negative")
    except Exception:  # role labels are validated at config load; stay defensive
        return {}

    gold_positive = [(gold, predicted) for gold, predicted in pairs if gold == positive]
    if not gold_positive:
        return {
            "relevant_recall": None,
            "evidence_discarded": 0,
            "note": "no relevant examples in this split",
        }

    kept = sum(1 for gold, predicted in gold_positive if predicted == positive)
    discarded = [predicted for _gold, predicted in gold_positive if predicted != positive]
    return {
        "relevant_recall": round(kept / len(gold_positive), 6),
        "relevant_support": len(gold_positive),
        "evidence_discarded": len(discarded),
        "evidence_kept_as_irrelevant": sum(1 for label in discarded if label == negative),
    }