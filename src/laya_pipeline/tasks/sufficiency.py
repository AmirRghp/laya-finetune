"""Task B — context sufficiency.

Generated separately from relevance, on real retrieval output rather than on
assembled examples.

The label is decided against **coverage of the question's key points** by the
retrieved chunks, never against a similarity score. A retrieval that returns
three lexically similar chunks can easily fail to cover the answer, and a
similarity-derived label would call that ``sufficient`` and teach the model to
trust closeness instead of coverage (AGENTS.md 6.5).

Every example's label therefore comes from the teacher reading the retrieved
text against the key points. ``uncertain`` answers are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config
from ..corpus.models import Chunk
from ..dataset.records import ChunkRef, GeneratedExample, LabeledExample
from ..errors import OllamaError, PipelineError, SchemaError
from ..llm.ollama import OllamaClient
from ..llm.prompts import PromptLibrary
from ..logging_setup import get_logger
from ..retrieval import EmbedFn, RetrievalIndex
from .base import UNCERTAIN, GenerationContext, Teacher
from .relevance import GenerationReport

__all__ = ["SufficiencyLabeler", "ChunkPool"]

LOGGER = get_logger("tasks.sufficiency")

PROMPT_NAME = "label_sufficiency"
DEFAULT_TOP_K = 3


@dataclass
class ChunkPool:
    """The chunks a retrieval index was built over, addressable by id."""

    chunks: Dict[str, Chunk]

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> "ChunkPool":
        return cls({chunk.chunk_id: chunk for chunk in chunks})

    def ref(self, chunk_id: str, *, rank: Optional[int] = None, score: Optional[float] = None) -> Optional[ChunkRef]:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None
        return ChunkRef(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_id=chunk.source_id,
            relpath=chunk.relpath,
            ordinal=chunk.ordinal,
            rank=rank,
            score=score,
        )


class SufficiencyLabeler:
    """Turns real retrieval results into sufficiency examples."""

    task = "sufficiency"

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
        index: RetrievalIndex,
        pool: _ChunkPool,
        *,
        embed_fn: Optional[EmbedFn] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> GenerationReport:
        """Retrieve, then judge coverage, for every example."""
        report = GenerationReport()
        options = self.config.laya.options(self.task)
        if len(options) < 2:
            raise PipelineError(
                f"the {self.task!r} question needs at least two options, found {options}"
            )
        if top_k < 1:
            raise PipelineError(f"top_k must be at least 1, got {top_k}")

        for example in examples:
            hits = self._retrieve(index, example.query, top_k, embed_fn)
            refs: List[ChunkRef] = []
            for hit in hits:
                ref = pool.ref(hit.chunk_id, rank=hit.rank, score=hit.score)
                if ref is not None:
                    refs.append(ref)
            if not refs:
                report.bump("retrieval_empty")
                continue

            chunks_text = "\n\n".join(
                f"--- متن {position} ---\n{ref.text}"
                for position, ref in enumerate(refs, start=1)
            )
            try:
                verdict = self.teacher.label_sufficiency(
                    query=example.query,
                    chunks_text=chunks_text,
                    key_points=example.key_points,
                    options=options,
                )
            except (OllamaError, SchemaError) as exc:
                report.bump("labeling_failed")
                report.failures.append(
                    {
                        "example_id": example.example_id,
                        "relpath": example.relpath,
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
                    example_id=f"{example.example_id}-suf",
                    task=self.task,
                    query=example.query,
                    chunks=refs,
                    label=label,
                    label_source="teacher",
                    confidence=confidence,
                    role="retrieved",
                    source_id=example.source_id,
                    relpath=example.relpath,
                    section_id=example.section_id,
                    key_points=list(example.key_points),
                    metadata={
                        "retrieval": {
                            "top_k": top_k,
                            "backend": index.backend,
                            "chunk_ids": [ref.chunk_id for ref in refs],
                        },
                        "reason": verdict.get("reason", ""),
                        "covered": verdict.get("covered", []),
                        "source_example_id": example.example_id,
                        "teacher_model": verdict.get("model"),
                    },
                )
            )
            report.bump("labeled")
            report.bump(f"label_{label}")

        report.bump("teacher_calls", self.teacher.calls)
        LOGGER.info(
            "sufficiency labeling finished",
            extra={"fields": {"examples": len(report.examples), **report.stats}},
        )
        return report

    @staticmethod
    def _retrieve(
        index: RetrievalIndex,
        query: str,
        top_k: int,
        embed_fn: Optional[EmbedFn],
    ) -> List[Any]:
        """Rank the index, using the embed_fn only if the backend needs one."""
        if index.backend == "bm25":
            return index.rank(query, top_k)
        if embed_fn is None:
            raise PipelineError(
                f"the {index.backend!r} retrieval index needs an embed_fn to score a query"
            )
        return index.rank(query, top_k, embed_fn=embed_fn)


def sufficiency_prompt_schema(options: Sequence[str]) -> Dict[str, Any]:
    """The exact schema the sufficiency teacher is asked to fill.

    Exposed so a test can assert the prompt and the validator agree.
    """
    from .base import choice_label_schema

    return choice_label_schema(options, with_coverage=True)


def render_chunks_text(refs: Sequence[ChunkRef]) -> str:
    """Numbered chunk block for the sufficiency prompt.

    Shared with the labeling stage so the teacher sees exactly the same
    rendering there and here. Exposed for tests.
    """
    return "\n\n".join(
        f"--- متن {position} ---\n{ref.text}" for position, ref in enumerate(refs, start=1)
    )