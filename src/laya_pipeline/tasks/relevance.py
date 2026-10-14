"""Task A — passage relevance.

The primary target. For each source chunk, generate a diverse set of Persian
questions that the chunk answers, attach the key points each question requires,
and — unless disabled — confirm with the teacher that the chunk really does
answer the question before the pair becomes a positive example.

That confirmation is the point. A generated question is only *plausible*; without
the evidence check, "relevant" would be a label of convenience attached to a pair
that may not actually contain the answer, and the model would be trained on a
signal the deployment never sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import Config
from ..corpus.models import Chunk
from ..dataset.records import GeneratedExample
from ..errors import OllamaError, PipelineError, SchemaError
from ..llm.ollama import OllamaClient
from ..llm.prompts import PromptLibrary
from ..logging_setup import get_logger
from .base import (
    UNCERTAIN,
    GenerationContext,
    Teacher,
    clean_question,
    grounded_key_points,
    heading_path_text,
    make_example,
    query_response_schema,
)

__all__ = ["GenerationReport", "QuestionGenerator"]

LOGGER = get_logger("tasks.relevance")

PROMPT_NAME = "questions"


@dataclass
class GenerationReport:
    """What one generation run produced, and everything it dropped on the way.

    ``failures`` is kept rather than logged away: a chunk that never generated a
    question is a finding about the chunk or the model, not noise.
    """

    examples: List[GeneratedExample] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def bump(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    def to_dict(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "failures": list(self.failures)}


class QuestionGenerator:
    """Generates answerable Persian questions from chunks."""

    task = "relevance"

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

    @property
    def model(self) -> str:
        return self.config.ollama.generation_model

    def generate(
        self,
        chunks: Sequence[Chunk],
        ctx: GenerationContext,
        *,
        existing: Optional[Mapping[str, Sequence[GeneratedExample]]] = None,
    ) -> GenerationReport:
        """Generate questions for every eligible chunk.

        ``existing`` maps chunk id to already-generated examples, which lets an
        interrupted run resume without re-billing the model for work it already
        paid for (AGENTS.md 4.2).
        """
        report = GenerationReport()
        settings = self.config.generation
        already_done = existing or {}
        schema = query_response_schema(
            settings.question_types, settings.questions_per_chunk, settings.max_key_points
        )
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)

        for chunk in chunks:
            if len(chunk.body.strip()) < settings.min_chunk_chars:
                report.bump("chunks_too_short")
                continue

            cached = already_done.get(chunk.chunk_id)
            if cached:
                report.examples.extend(cached)
                report.bump("chunks_resumed")
                report.bump("examples_resumed", len(cached))
                continue

            try:
                produced = self._generate_for_chunk(chunk, ctx, schema, schema_text, report)
            except (OllamaError, SchemaError) as exc:
                report.bump("chunks_failed")
                report.failures.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "relpath": chunk.relpath,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                LOGGER.warning(
                    "question generation failed for chunk",
                    extra={
                        "fields": {
                            "chunk_id": chunk.chunk_id,
                            "relpath": chunk.relpath,
                            "error": str(exc),
                        }
                    },
                )
                continue

            report.examples.extend(produced)
            report.bump("chunks_processed")
            report.bump("examples_generated", len(produced))

        report.bump("teacher_calls", self.teacher.calls)
        LOGGER.info(
            "relevance generation finished",
            extra={"fields": {"examples": len(report.examples), **report.stats}},
        )
        return report

    # -- internals ---------------------------------------------------------- #
    def _generate_for_chunk(
        self,
        chunk: Chunk,
        ctx: GenerationContext,
        schema: Mapping[str, Any],
        schema_text: str,
        report: GenerationReport,
    ) -> List[GeneratedExample]:
        settings = self.config.generation
        prompt = self.prompts.render(
            PROMPT_NAME,
            document_title=str(chunk.metadata.get("document_title", "")) or "(بدون عنوان)",
            heading_path=heading_path_text(chunk) or "(بدون مسیر عنوان)",
            chunk_text=chunk.text,
            num_questions=settings.questions_per_chunk,
            question_types="، ".join(settings.question_types),
            max_key_points=settings.max_key_points,
            schema_json=schema_text,
        )
        result = self.client.generate_json(
            prompt,
            model=self.model,
            schema=schema,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=settings.seed,
        )
        data = result.data
        if not isinstance(data, dict):  # pragma: no cover - schema guarantees this
            raise SchemaError(
                f"expected an object, got {type(data).__name__}", raw=result.text
            )
        raw_questions = data.get("questions") or []
        if not isinstance(raw_questions, list):
            raise SchemaError(
                f"'questions' must be an array, got {type(raw_questions).__name__}",
                raw=result.text,
            )

        produced: List[GeneratedExample] = []
        seen_queries: set[str] = set()
        for entry in raw_questions:
            if not isinstance(entry, dict):
                report.bump("entries_malformed")
                continue
            query = clean_question(
                str(entry.get("question", "")),
                min_chars=settings.min_question_chars,
                max_chars=settings.max_question_chars,
            )
            if not query:
                report.bump("questions_rejected_length")
                continue
            if query in seen_queries:
                report.bump("questions_rejected_duplicate")
                continue
            seen_queries.add(query)

            raw_points = [str(p) for p in (entry.get("key_points") or []) if str(p).strip()]
            key_points = grounded_key_points(raw_points, chunk.text, settings.max_key_points)
            if not key_points:
                report.bump("questions_rejected_no_key_points")
                continue

            if settings.verify_positives:
                verdict = self.teacher.verify_answerability(query, chunk.text, key_points)
                if verdict.is_uncertain:
                    report.bump("questions_discarded_uncertain_evidence")
                    continue
                if not verdict.is_yes:
                    report.bump("questions_discarded_unanswerable")
                    continue
                if verdict.confidence < self.config.labeling.uncertain_low:
                    report.bump("questions_discarded_low_confidence")
                    continue
                generation = {
                    **result.metadata(),
                    "evidence_check": verdict.to_dict(),
                }
            else:
                report.bump("evidence_check_skipped")
                generation = {**result.metadata(), "evidence_check": None}

            produced.append(
                make_example(
                    task=self.task,
                    query=query,
                    query_type=str(entry.get("type", "direct")),
                    chunk=chunk,
                    key_points=key_points,
                    generation=generation,
                )
            )
        return produced


def assert_answerable(example: GeneratedExample) -> None:
    """Refuse to build a relevance positive with no specification of the answer.

    The rule that creates a positive — *the source chunk, labelled relevant by
    rule* — lives next to the code that decided the question is answerable, and
    it insists on the evidence that made that decision available. A positive with
    no key points is an unlabelable claim.
    """
    if not example.key_points:
        raise PipelineError(
            f"example {example.example_id!r} has no key points; a relevance positive "
            "without a specification of what a correct answer contains is unlabelable"
        )
