"""Evaluation: classification against a checkpoint, and retrieval metrics."""

from __future__ import annotations

from .classification import (
    CALIBRATION_NOTE,
    PairSet,
    answer_confidence_of,
    answer_label,
    confusion_matrix,
    evaluate_checkpoint,
    label_of,
    macro_f1,
    per_class_scores,
)
from .report import compare_runs, render_markdown, write_report
from .retrieval import RetrievalQuery, build_queries, evaluate_retrieval

__all__ = [
    "CALIBRATION_NOTE",
    "PairSet",
    "RetrievalQuery",
    "answer_confidence_of",
    "answer_label",
    "build_queries",
    "compare_runs",
    "confusion_matrix",
    "evaluate_checkpoint",
    "evaluate_retrieval",
    "label_of",
    "macro_f1",
    "per_class_scores",
    "render_markdown",
    "write_report",
]