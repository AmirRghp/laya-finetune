"""Comparison reports: base checkpoint versus fine-tuned checkpoint.

A report here can only contain what was measured. Where an input did not carry a
metric, the output omits it rather than writing zero, and the delta between two
runs is computed only where both sides have the number (AGENTS.md 3.14).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..io import write_json
from ..logging_setup import get_logger

__all__ = ["compare_runs", "render_markdown", "write_report"]

LOGGER = get_logger("evaluation.report")

#: Metrics lifted to the top of a comparison, in reading order. A metric absent
#: from a run is simply absent from the table.
_HEADLINE = (
    "choice_accuracy",
    "accuracy",
    "macro_f1",
    "relevant_recall",
    "ece",
    "brier",
    "aurc",
    "mean_confidence",
    "latency_p50_ms",
    "latency_p95_ms",
)


def _flatten_metrics(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Pull the comparable numbers out of one run's payload."""
    flat: Dict[str, Any] = {}
    for key, value in (run.get("overall") or {}).items():
        flat[key] = value
    classification = run.get("classification")
    if isinstance(classification, Mapping):
        for key in ("accuracy", "macro_f1", "relevant_recall", "evidence_discarded", "n"):
            if key in classification:
                flat[key] = classification[key]
        per_class = classification.get("per_class")
        if isinstance(per_class, Mapping):
            for label, scores in per_class.items():
                if isinstance(scores, Mapping):
                    for metric in ("precision", "recall", "f1"):
                        if metric in scores:
                            flat[f"{label}.{metric}"] = scores[metric]
    return flat


def compare_runs(
    runs: Mapping[str, Mapping[str, Any]],
    *,
    title: str = "Laya checkpoint comparison",
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a machine-readable comparison of several evaluation runs."""
    if not runs:
        raise ValueError("compare_runs needs at least one run")

    flattened = {name: _flatten_metrics(run) for name, run in runs.items()}
    names = list(runs)

    table: Dict[str, Dict[str, Any]] = {}
    for metric in _ordered_metrics(flattened):
        row: Dict[str, Any] = {}
        for name in names:
            if metric in flattened[name]:
                row[name] = flattened[name][metric]
        if len(names) > 1 and len(row) == len(names):
            baseline, candidate = names[0], names[-1]
            base_value = flattened[baseline].get(metric)
            tuned_value = flattened[candidate].get(metric)
            if isinstance(base_value, (int, float)) and isinstance(tuned_value, (int, float)):
                row["delta"] = round(float(tuned_value) - float(base_value), 6)
        table[metric] = row

    caveats: List[str] = list(notes or [])
    for name, run in runs.items():
        note = run.get("calibration_note")
        if note:
            caveats.append(f"{name}: {note}")
        classification = run.get("classification")
        if isinstance(classification, Mapping) and classification.get("available") is False:
            caveats.append(
                f"{name}: macro-F1 and the confusion matrix were not computed — "
                + str(classification.get("note", "reason not recorded"))
            )
        if run.get("is_held_out") is False:
            caveats.append(
                f"{name}: these numbers are NOT held out; the report records "
                "eval_mode other than held_out"
            )

    return {
        "schema": "laya-pipeline-comparison/1",
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": {name: dict(run) for name, run in runs.items()},
        "metrics": table,
        "caveats": caveats,
        "baseline": names[0],
        "candidate": names[-1] if len(names) > 1 else None,
    }


def _ordered_metrics(flattened: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Headline metrics first, then anything else, alphabetically."""
    seen: List[str] = []
    for metric in _HEADLINE:
        if any(metric in block for block in flattened.values()):
            seen.append(metric)
    rest = sorted(
        {
            metric
            for block in flattened.values()
            for metric in block
            if metric not in _HEADLINE
        }
    )
    return seen + rest


def render_markdown(comparison: Mapping[str, Any]) -> str:
    """A readable table. Empty cells mean "not measured", never "zero"."""
    names = list(comparison.get("runs", {}))
    metrics = comparison.get("metrics", {})

    lines: List[str] = [f"# {comparison.get('title', 'Evaluation report')}", ""]
    lines.append(f"- generated: `{comparison.get('generated_at', 'unknown')}`")
    if comparison.get("baseline"):
        lines.append(f"- baseline: `{comparison['baseline']}`")
    if comparison.get("candidate"):
        lines.append(f"- candidate: `{comparison['candidate']}`")
    lines.append("")

    if not names:
        lines.append("_No runs were provided._")
        return "\n".join(lines)

    header = ["metric"] + [f"`{name}`" for name in names]
    if len(names) > 1:
        header.append("delta")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for metric, row in metrics.items():
        cells: List[str] = []
        for name in names:
            value = row.get(name)
            cells.append("—" if value is None else _format(value))
        if len(names) > 1:
            delta = row.get("delta")
            cells.append("—" if delta is None else _signed(delta))
        lines.append("| " + " | ".join([metric] + cells) + " |")

    caveats = comparison.get("caveats") or []
    if caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _signed(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.4f}"
    return f"+{value}" if value else "0"


def write_report(
    comparison: Mapping[str, Any], directory: Path, corpus_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Write ``evaluation.json`` and ``evaluation.md``."""
    target = Path(directory)
    json_path = write_json(target / "evaluation.json", comparison, corpus_dir)
    markdown_path = target / "evaluation.md"
    from ..io import _atomic_write_bytes  # local: keeps the atomic-write helper private

    _atomic_write_bytes(
        markdown_path, render_markdown(comparison).encode("utf-8"), corpus_dir
    )
    LOGGER.info(
        "evaluation report written",
        extra={"fields": {"json": str(json_path), "markdown": str(markdown_path)}},
    )
    return {"json": json_path, "markdown": markdown_path}