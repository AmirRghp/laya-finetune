"""Task C — query routing.

Predicts whether a query is best served by semantic, keyword or hybrid
retrieval. Two constraints shape this module:

**The label never comes from the source document.** Routing is a property of the
*query's form*, so the labeling prompt receives the query text and nothing else
— no document title, no heading path, no source id. A model that could see
"this came from the security document" would learn to route by domain, which is
exactly the failure mode this task must not have (AGENTS.md 6.6).

**Same-topic queries can route differently.** A query carrying a document
identifier needs keyword retrieval; the same topic asked in general terms needs
semantic. The prompt explains that distinction and shows one example of each.

This is a labeler, not a generator: it consumes questions produced for task A and
labels each one for routing. An ``uncertain`` answer is discarded.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from ..config import Config
from ..dataset.records import GeneratedExample, LabeledExample
from ..errors import OllamaError, PipelineError, SchemaError
from ..llm.ollama import OllamaClient
from ..llm.prompts import PromptLibrary
from ..logging_setup import get_logger
from .base import UNCERTAIN, GenerationContext, Teacher
from .relevance import GenerationReport

__all__ = ["RoutingLabeler"]

LOGGER = get_logger("tasks.routing")

PROMPT_NAME = "label_routing"


class RoutingLabeler:
    """Labels queries with a retrieval strategy, from query form alone."""

    task = "routing"

    def __init__(
        self,
        config: Config,
        prompts: PromptLibrary,
        client: OllamaClient,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.client = client
        self.teacher = Teacher(client, prompts, config)

    def label(
        self,
        examples: Sequence[GeneratedExample],
        ctx: GenerationContext,
    ) -> GenerationReport:
        report = GenerationReport()
        options = self.config.laya.options(self.task)
        if len(options) < 2:
            raise PipelineError(
                f"the {self.task!r} question needs at least two options, found {options}"
            )

        for example in examples:
            try:
                # Only the query crosses the boundary. No provenance fields.
                verdict = self.teacher.label_choice(
                    PROMPT_NAME, options=options, query=example.query
                )
            except (OllamaError, SchemaError) as exc:
                report.bump("labeling_failed")
                report.failures.append(
                    {
                        "example_id": example.example_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            label = verdict["label"]
            if label == UNCERTAIN or label not in options:
                report.bump("discarded_uncertain")
                continue
            confidence = float(verdict["confidence"])
            if confidence < self.config.labeling.uncertain_low:
                report.bump("discarded_low_confidence")
                continue

            report.examples.append(
                LabeledExample(
                    # Deliberately not derived from the source example id: a
                    # routing row's identity is its query and label, so the same
                    # query labeled the same way twice collapses to one row.
                    example_id="",
                    task=self.task,
                    query=example.query,
                    chunks=[],  # the routing template takes no {{chunks}}
                    label=label,
                    label_source="teacher",
                    confidence=confidence,
                    role="routing",
                    key_points=list(example.key_points),
                    metadata={
                        "reason": verdict.get("reason", ""),
                        "teacher_model": verdict.get("model"),
                        "source_example_id": example.example_id,
                    },
                )
            )
            report.bump("labeled")
            report.bump(f"label_{label}")

        report.bump("teacher_calls", self.teacher.calls)
        LOGGER.info(
            "routing labeling finished",
            extra={"fields": {"examples": len(report.examples), **report.stats}},
        )
        return report


def routing_from_features(features: Dict[str, Any], options: Sequence[str]) -> str:
    """Rule-based routing label from query surface features.

    Provided as a *diagnostic*, not a labeler: it is deterministic, so comparing
    it against the teacher's label on a sample shows whether the teacher is
    reasoning about query form or drifting toward the document's topic. It is
    never used to produce a training label.
    """
    has_literal_anchor = bool(features.get("literal_anchor"))
    needs_context = bool(features.get("needs_context"))
    if has_literal_anchor and needs_context:
        return "hybrid"
    if has_literal_anchor:
        return "keyword"
    if needs_context:
        return "semantic"
    return "semantic"


def label_distribution(examples: Sequence[LabeledExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for example in examples:
        counts[example.label] = counts.get(example.label, 0) + 1
    return dict(sorted(counts.items()))