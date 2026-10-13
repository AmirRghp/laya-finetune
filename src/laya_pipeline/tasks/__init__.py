"""The three decision tasks.

``relevance``
    Primary. Query + retrieved chunk -> relevant / irrelevant.
``sufficiency``
    Query + a real retrieval result -> sufficient / partial / insufficient.
``routing``
    Query alone -> semantic / keyword / hybrid.

Each is independently runnable. Nothing in the task modules knows about the
others, so a caller can train on one, or combine several whose labeled decisions
share a compatible question definition (AGENTS.md 3.4).
"""

from __future__ import annotations

from typing import Sequence

from .base import (
    UNCERTAIN,
    EvidenceVerdict,
    GenerationContext,
    Teacher,
    choice_label_schema,
    evidence_schema,
    key_points_json,
    query_response_schema,
)
from .relevance import GenerationReport, QuestionGenerator
from .routing import RoutingLabeler
from .sufficiency import SufficiencyLabeler

#: Canonical task order. Relevance first because it is the primary target.
TASKS = ("relevance", "sufficiency", "routing")


def resolve_tasks(requested: Sequence[str] | None) -> list[str]:
    """Normalise a ``--task`` selection, defaulting to every task."""
    if not requested:
        return list(TASKS)
    if "all" in requested:
        return list(TASKS)
    chosen: list[str] = []
    for name in requested:
        key = str(name).strip()
        if key not in TASKS:
            raise ValueError(
                f"unknown task {name!r}; choose from {', '.join(TASKS)} or 'all'"
            )
        if key not in chosen:
            chosen.append(key)
    return chosen


__all__ = [
    "TASKS",
    "UNCERTAIN",
    "EvidenceVerdict",
    "GenerationContext",
    "GenerationReport",
    "QuestionGenerator",
    "RoutingLabeler",
    "SufficiencyLabeler",
    "Teacher",
    "choice_label_schema",
    "evidence_schema",
    "key_points_json",
    "query_response_schema",
    "resolve_tasks",
]