"""Shared machinery for the three task generators.

Holds the response schemas, the generation context every task receives, and the
teacher calls that verify or label.

One rule is enforced here rather than in each task: the string ``uncertain`` is
never a Laya option. The teacher is allowed to answer ``uncertain`` and it means
*discard*, because a coin-flipped ambiguous example is worse than one fewer
example (AGENTS.md 6.2).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import Config
from ..corpus.models import Chunk
from ..dataset.records import GeneratedExample, KeyPoint, make_example_id
from ..errors import PipelineError, SchemaError
from ..ids import normalize_whitespace
from ..llm.ollama import GenerationResult, OllamaClient
from ..llm.prompts import PromptLibrary
from ..llm.schema import array_schema, number_schema, object_schema, string_schema
from ..logging_setup import get_logger

__all__ = [
    "UNCERTAIN",
    "GenerationContext",
    "Teacher",
    "query_response_schema",
    "evidence_schema",
    "choice_label_schema",
    "heading_path_text",
    "key_points_json",
    "clean_question",
]

LOGGER = get_logger("tasks")

#: The teacher's escape hatch. Never becomes a Laya option; always a discard.
UNCERTAIN = "uncertain"

_QUESTION_TYPES = (
    "direct",
    "paraphrase",
    "conversational",
    "terminology",
    "alternative",
    "hard",
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def query_response_schema(
    question_types: Sequence[str],
    max_questions: int,
    max_key_points: int,
) -> Dict[str, Any]:
    """Schema for the question-and-key-points response."""
    allowed = [t for t in question_types if t in _QUESTION_TYPES] or list(_QUESTION_TYPES)
    return object_schema(
        {
            "questions": array_schema(
                object_schema(
                    {
                        "question": string_schema(minLength=8),
                        "type": string_schema(enum=allowed),
                        "key_points": array_schema(
                            string_schema(minLength=3), maxItems=max_key_points
                        ),
                    }
                ),
                minItems=1,
                maxItems=max(1, max_questions),
            )
        }
    )


def evidence_schema(max_key_points: int = 12) -> Dict[str, Any]:
    """Schema for the answerability check used on positives and negatives."""
    return object_schema(
        {
            "answers": string_schema(enum=["yes", "no", UNCERTAIN]),
            "covered_key_points": array_schema(number_schema(), maxItems=max_key_points),
            "confidence": number_schema(minimum=0, maximum=1),
        }
    )


def choice_label_schema(options: Sequence[str], with_coverage: bool = False) -> Dict[str, Any]:
    """Schema for a teacher labeling a ``choice`` question.

    ``UNCERTAIN`` is offered alongside the real options on purpose: a teacher
    that must choose between two labels invents one.
    """
    labels = list(options) + [UNCERTAIN]
    properties: Dict[str, Any] = {
        "label": string_schema(enum=labels),
        "confidence": number_schema(minimum=0, maximum=1),
        "reason": string_schema(minLength=3, maxLength=400),
    }
    if with_coverage:
        properties["covered"] = array_schema(number_schema(), maxItems=16)
    return object_schema(properties)


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class GenerationContext:
    """Everything a task generator needs, passed explicitly."""

    config: Config
    client: OllamaClient
    prompts: PromptLibrary
    rng: random.Random
    task: str = ""
    #: Filled in by each task so the caller can report a self-contained record.
    stats: Dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    @property
    def seed(self) -> int:
        return self.config.generation.seed


# --------------------------------------------------------------------------- #
# Teacher
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceVerdict:
    """The teacher's answer to "does this text answer this question?"."""

    answers: str
    covered: List[int]
    confidence: float

    @property
    def is_yes(self) -> bool:
        return self.answers == "yes"

    @property
    def is_no(self) -> bool:
        return self.answers == "no"

    @property
    def is_uncertain(self) -> bool:
        return self.answers == UNCERTAIN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answers": self.answers,
            "covered": list(self.covered),
            "confidence": self.confidence,
        }


class Teacher:
    """Every teacher call the pipeline makes, in one place."""

    EVIDENCE_PROMPT = "evidence_answerability"

    def __init__(self, client: OllamaClient, prompts: PromptLibrary, config: Config) -> None:
        self.client = client
        self.prompts = prompts
        self.config = config
        self.calls = 0

    @property
    def model(self) -> str:
        return self.config.ollama.teacher_model

    def _call(self, prompt_name: str, prompt: str, schema: Mapping[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        result = self.client.generate_json(
            prompt,
            model=self.model,
            schema=schema,
            temperature=self.config.labeling.temperature,
            seed=self.config.generation.seed,
        )
        if not isinstance(result.data, dict):  # pragma: no cover - schema guarantees object
            raise SchemaError(
                f"{prompt_name} returned {type(result.data).__name__}, expected an object",
                raw=result.text,
            )
        return result.data

    # -- answerability ------------------------------------------------------ #
    def verify_answerability(
        self,
        query: str,
        text: str,
        key_points: Sequence[KeyPoint] = (),
    ) -> EvidenceVerdict:
        """Ask whether ``text`` answers ``query``.

        The key points are shown as the specification of a correct answer, so a
        text that is on-topic but does not carry the answer is judged ``no``
        rather than accepted on topic similarity.
        """
        prompt = self.prompts.render(
            self.EVIDENCE_PROMPT,
            query=query,
            chunk_text=text,
            key_points_json=key_points_json(key_points),
            schema_json=_schema_text(evidence_schema()),
        )
        data = self._call(self.EVIDENCE_PROMPT, prompt, evidence_schema())
        return EvidenceVerdict(
            answers=str(data.get("answers", UNCERTAIN)),
            covered=[int(x) for x in data.get("covered_key_points", []) or []],
            confidence=_as_float(data.get("confidence"), default=0.0),
        )

    # -- labelling ---------------------------------------------------------- #
    def label_choice(
        self,
        prompt_name: str,
        *,
        options: Sequence[str],
        **values: str,
    ) -> Dict[str, Any]:
        """Ask the teacher to choose between ``options`` for a prepared prompt."""
        schema = choice_label_schema(options)
        prompt = self.prompts.render(
            prompt_name, schema_json=_schema_text(schema), **values
        )
        data = self._call(prompt_name, prompt, schema)
        return {
            "label": str(data.get("label", UNCERTAIN)),
            "confidence": _as_float(data.get("confidence"), default=0.0),
            "reason": str(data.get("reason", "")),
            "model": self.model,
            "prompt": prompt_name,
        }

    def label_sufficiency(
        self, query: str, chunks_text: str, key_points: Sequence[KeyPoint], options: Sequence[str]
    ) -> Dict[str, Any]:
        schema = choice_label_schema(options, with_coverage=True)
        prompt = self.prompts.render(
            "label_sufficiency",
            query=query,
            chunks_text=chunks_text,
            key_points_json=key_points_json(key_points),
            schema_json=_schema_text(schema),
        )
        data = self._call("label_sufficiency", prompt, schema)
        return {
            "label": str(data.get("label", UNCERTAIN)),
            "confidence": _as_float(data.get("confidence"), default=0.0),
            "reason": str(data.get("reason", "")),
            "covered": [int(x) for x in data.get("covered", []) or []],
            "model": self.model,
            "prompt": "label_sufficiency",
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _schema_text(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True)


def heading_path_text(chunk: Chunk) -> str:
    """The chunk's breadcrumb as a single readable line."""
    if not chunk.heading_path:
        return ""
    return " › ".join(heading.text for heading in chunk.heading_path)


def key_points_json(key_points: Sequence[KeyPoint]) -> str:
    """Key points rendered for a prompt, or an explicit emptiness marker.

    An empty list is stated in words rather than shown as ``[]``, because a model
    shown an empty list tends to invent points to fill it.
    """
    if not key_points:
        return "(نکتهٔ کلیدی‌ای برای این پرسش استخراج نشده است)"
    return json.dumps([point.text for point in key_points], ensure_ascii=False, indent=2)


def clean_question(text: str, *, min_chars: int, max_chars: int) -> Optional[str]:
    """Normalise and range-check a generated question, or reject it."""
    candidate = normalize_whitespace(text or "").strip().strip("«»\"'“”")
    if len(candidate) < min_chars:
        return None
    if len(candidate) > max_chars:
        return None
    return candidate


def grounded_key_points(
    texts: Sequence[str], source_text: str, limit: int
) -> List[KeyPoint]:
    """Keep only key points that occur verbatim in their source chunk.

    The stored span is the matched source text, never model output presented as
    evidence (AGENTS.md 6.4).
    """
    points: List[KeyPoint] = []
    for text in texts:
        candidate = text.strip()
        if not candidate or candidate not in source_text:
            continue
        points.append(KeyPoint(text=candidate, span=candidate, index=len(points)))
        if len(points) >= limit:
            break
    return points


def make_example(
    task: str,
    query: str,
    query_type: str,
    chunk: Chunk,
    key_points: Sequence[KeyPoint],
    generation: Mapping[str, Any],
) -> GeneratedExample:
    """Build a :class:`GeneratedExample` with its provenance filled in."""
    if not key_points:
        raise PipelineError(
            f"refusing to record a generated question for {chunk.chunk_id} with no key "
            "points: without a specification of what a correct answer contains, neither "
            "a positive nor a verified negative can be built from it"
        )
    return GeneratedExample(
        example_id=make_example_id(task, query, chunk.chunk_id),
        task=task,
        query=query,
        query_type=query_type,
        key_points=list(key_points),
        source_chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        relpath=chunk.relpath,
        section_id=chunk.section_id,
        heading_path=[h.text for h in chunk.heading_path],
        generation=dict(generation),
    )


def _as_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if number != number:  # NaN
        return default
    return number


def unwrap(data: GenerationResult) -> Dict[str, Any]:
    """The dict body of a validated generation result."""
    if not isinstance(data.data, dict):
        raise SchemaError(
            f"expected an object response, got {type(data.data).__name__}", raw=data.text
        )
    return data.data
