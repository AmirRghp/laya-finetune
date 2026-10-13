"""Group splitting with family merging and leakage detection.

Splitting happens **before** negative mining, and every group that shares
content stays together (AGENTS.md 6.7, 6.8):

* the group is the source document;
* documents whose content is at least ``family_similarity_threshold`` similar
  are merged into a family first, so a document and its near-copy cannot land in
  different splits;
* after assignment, three leakage checks run — identical decision inputs, source
  document ids, and paraphrased queries.

Validation is the calibration split. It is handed to Laya as ``eval_data`` and is
never fitted on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..config import Config
from ..corpus.models import Chunk
from ..dataset.records import SPLITS, LabeledExample
from ..errors import SplitError
from ..ids import char_ngrams, iter_tokens, jaccard, shingles
from ..logging_setup import get_logger

__all__ = [
    "SplitReport",
    "assign_splits",
    "apply_splits",
    "detect_leakage",
    "build_families",
]

LOGGER = get_logger("dataset.split")

_FAMILY_SHINGLE = 3


@dataclass
class SplitReport:
    """Which split each group landed in, and what the split cost."""

    family_of: Dict[str, str] = field(default_factory=dict)
    assignment: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    leakage: List[Dict[str, object]] = field(default_factory=list)

    def split_for(self, source_id: str) -> Optional[str]:
        family = self.family_of.get(source_id)
        return self.assignment.get(family) if family else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "families": len(self.assignment),
            "assignment": dict(sorted(self.assignment.items())),
            "source_to_family": dict(sorted(self.family_of.items())),
            "stats": dict(self.stats),
            "warnings": list(self.warnings),
            "leakage": list(self.leakage),
        }


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
def build_families(
    chunks_by_source: Mapping[str, Sequence[Chunk]], threshold: float
) -> Tuple[Dict[str, str], int]:
    """Merge near-duplicate documents into families. Returns (source->family, count).

    O(n^2) over documents, which is cheap at corpus scale and exact. A corpus of
    tens of thousands of documents would need an inverted index here; at that
    size the split strategy itself would need revisiting.
    """
    source_ids = sorted(chunks_by_source)
    if not source_ids:
        return {}, 0

    signatures: Dict[str, Set[Tuple[str, ...]]] = {}
    for source_id in source_ids:
        tokens: List[str] = []
        for chunk in sorted(chunks_by_source[source_id], key=lambda c: c.ordinal):
            tokens.extend(iter_tokens(chunk.body))
        signatures[source_id] = shingles(tokens, _FAMILY_SHINGLE)

    parent: Dict[str, str] = {source_id: source_id for source_id in source_ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            # Deterministic merge direction: the lexicographically smaller root
            # wins, so family ids do not depend on iteration order.
            if b < a:
                a, b = b, a
            parent[b] = a

    if threshold > 0:
        for i, left in enumerate(source_ids):
            for right in source_ids[i + 1 :]:
                if find(left) == find(right):
                    continue
                if jaccard(signatures[left], signatures[right]) >= threshold:
                    union(left, right)

    family_of = {source_id: find(source_id) for source_id in source_ids}
    families = sorted(set(family_of.values()))
    return family_of, len(families)


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def assign_splits(
    examples: Sequence[LabeledExample],
    config: Config,
    chunks_by_source: Mapping[str, Sequence[Chunk]],
    rng: Optional[random.Random] = None,
) -> SplitReport:
    """Assign whole document families to train / validation / test."""
    report = SplitReport()
    settings = config.splitting
    generator = rng or random.Random(settings.seed)

    families = sorted({example.source_id for example in examples if example.source_id})
    if not families:
        raise SplitError(
            "no examples carry a source document id; the split cannot keep documents "
            "together without one"
        )
    missing = sorted(f for f in families if f not in chunks_by_source)
    for source_id in missing:
        report.warnings.append(
            f"no chunks known for source {source_id}; it forms a family of one and "
            "cannot be similarity-merged with anything"
        )

    family_of, family_count = build_families(chunks_by_source, settings.family_similarity_threshold)
    for source_id in families:
        family_of.setdefault(source_id, source_id)
    report.family_of = family_of

    # Weight each family by how many examples it carries, so a handful of large
    # documents cannot all fall into one split and leave the others empty.
    weights: Dict[str, int] = {family: 0 for family in set(family_of.values())}
    for example in examples:
        family = family_of.get(example.source_id)
        if family is not None:
            weights[family] += 1

    total = sum(weights.values())
    if total == 0:
        raise SplitError("every family is empty; nothing to split")

    targets = {
        "train": total * settings.train,
        "validation": total * settings.validation,
        "test": total * settings.test,
    }

    order = sorted(weights)
    generator.shuffle(order)
    assignment: Dict[str, str] = {}
    filled = {name: 0.0 for name in SPLITS}
    # Largest documents first, so a single family cannot be stranded by a later
    # greedy choice; ties broken by family id for determinism.
    for family in sorted(order, key=lambda f: (-weights[f], f)):
        # Pick the split furthest below its target. The split order breaks ties.
        best = max(SPLITS, key=lambda name: (targets[name] - filled[name], -SPLITS.index(name)))
        assignment[family] = best
        filled[best] += weights[family]

    report.assignment = assignment
    report.stats = {
        "examples": total,
        "families": family_count,
        "sources": len(family_of),
        "targets": {name: round(value, 2) for name, value in targets.items()},
        "assigned_examples": {name: int(filled[name]) for name in SPLITS},
        "assigned_families": {
            name: sum(1 for f in assignment.values() if f == name) for name in SPLITS
        },
    }

    family_counts = report.stats["assigned_families"]  # type: ignore[index]
    for name, count in family_counts.items():
        if count == 0:
            report.warnings.append(
                f"split {name!r} received no document family; with "
                f"{family_count} families the ratios cannot be honoured and evaluation "
                f"on {name} will be meaningless"
            )
    if family_count < settings.warn_below_groups:
        report.warnings.append(
            f"only {family_count} document families for 3 splits "
            f"({settings.warn_below_groups} recommended); per-document metrics will "
            "have a very wide confidence interval"
        )

    report.leakage = detect_leakage(examples, config, assignment=assignment, family_of=family_of)
    for finding in report.leakage:
        report.warnings.append(
            f"leakage[{finding['kind']}]: {finding.get('detail', '')}"
        )

    LOGGER.info(
        "splits assigned",
        extra={
            "fields": {
                "families": family_count,
                **{name: family_counts[name] for name in SPLITS},
                "leakage": len(report.leakage),
            }
        },
    )
    return report


def apply_splits(
    examples: Sequence[LabeledExample], report: SplitReport
) -> List[LabeledExample]:
    """Return copies of ``examples`` carrying their assigned split."""
    result: List[LabeledExample] = []
    for example in examples:
        split = report.split_for(example.source_id)
        if split is None:
            result.append(example)
            continue
        clone = LabeledExample(
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
        result.append(clone)
    return result


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def detect_leakage(
    examples: Sequence[LabeledExample],
    config: Config,
    *,
    assignment: Optional[Mapping[str, str]] = None,
    family_of: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, object]]:
    """Find content or paraphrases that appear in more than one split.

    Independent of the assignment: given no assignment it derives splits from
    ``example.split``, so it can also audit an already-split dataset.
    """
    settings = config.splitting
    findings: List[Dict[str, object]] = []

    def split_of(example: LabeledExample) -> str:
        if assignment and family_of:
            family = family_of.get(example.source_id)
            if family and family in assignment:
                return assignment[family]
        return example.split or "unassigned"

    # 1. Identical decision inputs across splits.
    by_signature: Dict[str, List[LabeledExample]] = {}
    for example in examples:
        by_signature.setdefault(example.content_signature(), []).append(example)
    for signature, group in by_signature.items():
        splits = {split_of(example) for example in group}
        if len(splits) > 1:
            findings.append(
                {
                    "kind": "identical_decision",
                    "signature": signature[:16],
                    "splits": sorted(splits),
                    "example_ids": [example.example_id for example in group],
                    "detail": f"the same state+query appears in {sorted(splits)}",
                }
            )

    # 2. One source document contributing to several splits.
    by_source: Dict[str, Dict[str, List[str]]] = {}
    for example in examples:
        if not example.source_id:
            continue
        by_source.setdefault(example.source_id, {}).setdefault(
            split_of(example), []
        ).append(example.example_id)
    for source_id, per_split in sorted(by_source.items()):
        if len(per_split) > 1:
            findings.append(
                {
                    "kind": "source_document_split",
                    "source_id": source_id,
                    "splits": sorted(per_split),
                    "detail": f"document {source_id} contributes to {sorted(per_split)}",
                }
            )

    # 3. Paraphrased queries across splits.
    by_split_queries: Dict[str, List[LabeledExample]] = {}
    for example in examples:
        by_split_queries.setdefault(split_of(example), []).append(example)

    splits = sorted(by_split_queries)
    seen: Set[Tuple[str, str]] = set()
    # Precompute the trigram sets once. Recomputing them inside the pair loop
    # would make this quadratic in a much more expensive constant.
    prepared: Dict[str, List[Tuple[LabeledExample, frozenset, int]]] = {}
    for split_name, group in by_split_queries.items():
        prepared[split_name] = [
            (
                example,
                frozenset(char_ngrams(example.query, settings.paraphrase_ngram)),
                len(example.query),
            )
            for example in group
        ]

    for i, left_split in enumerate(splits):
        for right_split in splits[i + 1 :]:
            for left, left_grams, left_len in prepared[left_split]:
                for right, right_grams, right_len in prepared[right_split]:
                    key = tuple(sorted((left.example_id, right.example_id)))
                    if key in seen:
                        continue
                    seen.add(key)  # type: ignore[arg-type]
                    if left.query == right.query:
                        findings.append(
                            {
                                "kind": "duplicate_query",
                                "splits": [left_split, right_split],
                                "example_ids": list(key),
                                "detail": f"identical query in {left_split} and {right_split}",
                            }
                        )
                        continue
                    # Cheap prefilter: strings of very different length cannot
                    # reach a high Jaccard, so skip the comparison entirely.
                    longer, shorter = max(left_len, right_len), min(left_len, right_len)
                    if shorter and longer / shorter > 1.6:
                        continue
                    similarity = _set_jaccard(left_grams, right_grams)
                    if similarity >= settings.query_leakage_threshold:
                        findings.append(
                            {
                                "kind": "paraphrased_query",
                                "splits": [left_split, right_split],
                                "example_ids": list(key),
                                "similarity": round(similarity, 4),
                                "detail": (
                                    f"queries {similarity:.2f} similar across "
                                    f"{left_split}/{right_split}, above "
                                    f"{settings.query_leakage_threshold}"
                                ),
                            }
                        )
    return findings


def _set_jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)