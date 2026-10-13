"""Dataset statistics.

Reports what is in the data, including the things that are easy to leave out:
label balance per split, chunk provenance, token length distribution and how much
of the data rests on a single document.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ..config import Config
from ..corpus.models import Chunk
from ..dataset.records import SPLITS, LabeledExample
from ..tokenizer import TokenCounter

__all__ = [
    "example_stats",
    "chunk_stats",
    "render_markdown",
    "class_balance_warning",
    "split_coverage",
    "sufficiency_balance",
]


def example_stats(
    config: Config, examples: Sequence[LabeledExample], task: Optional[str] = None
) -> Dict[str, Any]:
    """Counts, label balance and provenance for a set of examples."""
    selected = [e for e in examples if task is None or e.task == task]
    by_task: Dict[str, int] = {}
    by_split: Dict[str, Dict[str, int]] = {}
    by_label: Dict[str, Dict[str, int]] = {}
    by_role: Dict[str, int] = {}
    by_strategy: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_label_source: Dict[str, int] = {}

    for example in selected:
        by_task[example.task] = by_task.get(example.task, 0) + 1
        split = example.split or "unassigned"
        by_split.setdefault(example.task, {}).setdefault(split, 0)
        by_split[example.task][split] += 1  # type: ignore[index]
        by_label.setdefault(example.task, {}).setdefault(example.label, 0)
        by_label[example.task][example.label] += 1  # type: ignore[index]
        by_role[example.role] = by_role.get(example.role, 0) + 1
        if example.strategy:
            by_strategy[example.strategy] = by_strategy.get(example.strategy, 0) + 1
        if example.label_source:
            by_label_source[example.label_source] = by_label_source.get(example.label_source, 0) + 1
        if example.source_id:
            by_source[example.source_id] = by_source.get(example.source_id, 0) + 1

    stats: Dict[str, Any] = {
        "examples": len(selected),
        "by_task": dict(sorted(by_task.items())),
        "by_split": {k: dict(sorted(v.items())) for k, v in sorted(by_split.items())},
        "by_label": {k: dict(sorted(v.items())) for k, v in sorted(by_label.items())},
        "by_role": dict(sorted(by_role.items())),
        "by_negative_strategy": dict(sorted(by_strategy.items())),
        "by_label_source": dict(sorted(by_label_source.items())),
        "distinct_sources": len(by_source),
        "source_concentration": _concentration(by_source),
    }
    stats.update(class_balance_warning(config, selected, task))
    return stats


def _concentration(counts: Mapping[str, int]) -> Dict[str, Any]:
    """How much of the data a single document carries."""
    total = sum(counts.values())
    if not total:
        return {}
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_source, top_count = ordered[0]
    return {
        "largest_document": top_source,
        "largest_document_share": round(top_count / total, 4),
        "documents_above_10_percent": sum(1 for _, count in ordered if count / total > 0.10),
    }


def class_balance_warning(
    config: Config, examples: Sequence[LabeledExample], task: Optional[str] = None
) -> Dict[str, Any]:
    """Flag a split whose class mix is too skewed to trust.

    A relevance split that is 95% negatives will train a model that answers
    ``irrelevant`` and look excellent on accuracy while being useless.
    """
    selected = [e for e in examples if task is None or e.task == task]
    warnings: List[str] = []
    if not selected:
        return {"warnings": warnings}

    by_split: Dict[str, Dict[str, int]] = {}
    for example in selected:
        split = example.split or "unassigned"
        by_split.setdefault(split, {}).setdefault(example.label, 0)
        by_split[split][example.label] += 1  # type: ignore[index]

    for split in sorted(by_split):
        counts = by_split[split]
        total = sum(counts.values())
        if total < 2:
            continue
        majority = max(counts.values()) / total
        if majority > 0.9:
            label = max(counts, key=lambda k: (counts[k], k))
            warnings.append(
                f"split {split!r} is {majority:.0%} {label!r} ({total} rows); a majority "
                "baseline would score that well, so accuracy alone will not show whether "
                "the model learned anything"
            )
        if len(counts) == 1:
            warnings.append(
                f"split {split!r} has a single label ({next(iter(counts))!r}); recall and "
                "precision for the other options are undefined there"
            )

    return {"warnings": warnings}


def chunk_stats(chunks: Sequence[Chunk], counter: Optional[TokenCounter] = None) -> Dict[str, Any]:
    """Chunk counts, sizes and document coverage."""
    if not chunks:
        return {"chunks": 0}
    lengths = [chunk.token_count for chunk in chunks if chunk.token_count is not None]
    by_source: Dict[str, int] = {}
    oversized = 0
    small = 0
    estimated = 0
    for chunk in chunks:
        by_source[chunk.source_id] = by_source.get(chunk.source_id, 0) + 1
        if chunk.oversized:
            oversized += 1
        if chunk.metadata.get("small"):
            small += 1
        if chunk.token_count_source == "estimated":
            estimated += 1

    stats: Dict[str, Any] = {
        "chunks": len(chunks),
        "documents": len(by_source),
        "oversized": oversized,
        "small": small,
        "token_source": chunks[0].token_count_source,
        "token_count_estimated": estimated,
    }
    if lengths:
        ordered = sorted(lengths)
        stats["tokens"] = {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 1),
        }
    return stats


def render_markdown(title: str, payload: Mapping[str, Any]) -> str:
    """A readable summary. Never invents a value: absent fields stay absent."""
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        lines.append(f"## {key}")
        lines.append("")
        lines.extend(_render_block(value))
        lines.append("")
    return "\n".join(lines)


def _render_block(value: Any, depth: int = 0) -> List[str]:
    indent = "  " * depth
    if isinstance(value, Mapping):
        lines: List[str] = []
        for key in sorted(value):
            child = value[key]
            if isinstance(child, (Mapping, list)):
                lines.append(f"{indent}- **{key}**:")
                lines.extend(_render_block(child, depth + 1))
            else:
                lines.append(f"{indent}- **{key}**: {child}")
        return lines or [f"{indent}(empty)"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{indent}(none)"]
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.extend(_render_block(item, depth + 1))
            else:
                lines.append(f"{indent}- {item}")
        return lines
    return [f"{indent}{value}"]


def split_coverage(examples: Sequence[LabeledExample]) -> Dict[str, int]:
    """Rows per split, for a quick "did anything reach validation?" check."""
    counts: Dict[str, int] = {name: 0 for name in SPLITS}
    for example in examples:
        if example.split in counts:
            counts[example.split] += 1  # type: ignore[index]
    return counts


def sufficiency_balance(examples: Sequence[LabeledExample]) -> Dict[str, Any]:
    """Coverage of the sufficiency scale per split.

    A sufficiency set with only ``sufficient`` rows teaches a model one answer
    while presenting three options, so the coverage is worth stating outright.
    """
    per_split: Dict[str, Dict[str, int]] = {}
    for example in examples:
        if example.task != "sufficiency":
            continue
        split = example.split or "unassigned"
        per_split.setdefault(split, {}).setdefault(example.label, 0)
        per_split[split][example.label] += 1  # type: ignore[index]

    warnings: List[str] = []
    for split, counts in sorted(per_split.items()):
        if len(counts) == 1:
            only = next(iter(counts))
            warnings.append(
                f"split {split!r} contains only the {only!r} sufficiency label; the other "
                "levels are unlearnable and undefined in the metrics"
            )
        elif len(counts) < 3:
            warnings.append(
                f"split {split!r} covers only {len(counts)} of 3 sufficiency levels"
            )
    return {"by_split": {k: dict(sorted(v.items())) for k, v in sorted(per_split.items())},
            "warnings": warnings}