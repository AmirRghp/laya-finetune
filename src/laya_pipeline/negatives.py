"""Hard and random negative sampling.

The ordering constraint is the whole point of this module
(AGENTS.md 6.1, 6.3, 6.7):

    split the corpus into train / validation / test first,
    then mine negatives **within each split's own chunk pool**.

Mining from the global pool first would let a test document's text appear in a
training row. That is label leakage, and no threshold fixes it afterwards — so
the caller passes an already split-scoped pool.

Every candidate then passes four gates before it becomes a negative:

1. it is not the positive chunk itself;
2. it is not a near-duplicate of the positive (word-shingle overlap);
3. it does not literally contain a key-point answer string;
4. the teacher says the candidate does **not** answer the query.

An ``uncertain`` teacher answer is a discard, not a label. Merely not being the
source document is not evidence of irrelevance, and without gate 4 nothing here
would catch that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Config
from .corpus.models import Chunk
from .dataset.records import ChunkRef, LabeledExample
from .errors import NegativeSamplingError, OllamaError, SchemaError
from .ids import fold_for_matching, iter_tokens, jaccard, shingles
from .llm.ollama import OllamaClient
from .llm.prompts import PromptLibrary
from .logging_setup import get_logger
from .retrieval import BM25Index, EmbedFn, RetrievalIndex
from .tasks.base import GenerationContext, Teacher

__all__ = ["MiningReport", "NegativeSampler", "Candidate", "STRATEGIES"]

LOGGER = get_logger("negatives")

#: The three supported strategies, in the order they are offered candidates.
STRATEGIES = ("random", "lexical", "retrieval")

_NEAR_DUP_SHINGLE = 3
#: Only a key point at least this long is used as a literal guard; shorter ones
#: match too much ordinary vocabulary to be evidence.
_MIN_LITERAL_GUARD_CHARS = 12


@dataclass(frozen=True)
class Candidate:
    """A mined candidate, before verification."""

    chunk_id: str
    strategy: str
    margin: float
    score: Optional[float] = None


@dataclass
class MiningReport:
    """Negatives produced, and every candidate rejected with its reason."""

    examples: List[LabeledExample] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    def bump(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    def reject(self, example_id: str, chunk_id: str, reason: str, **extra: Any) -> None:
        self.rejected.append(
            {"example_id": example_id, "chunk_id": chunk_id, "reason": reason, **extra}
        )

    def strategy_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in STRATEGIES}
        for example in self.examples:
            if example.strategy in counts:
                counts[str.example.strategy] += 1  # type: ignore[index]
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "rejected": list(self.rejected),
            "by_strategy": self.strategy_counts(),
        }


class NegativeSampler:
    """Draws and verifies negatives for a set of relevance positives."""

    def __init__(self, config: Config, prompts: PromptLibrary, client: OllamaClient) -> None:
        self.config = config
        self.prompts = prompts
        self.client = client
        self.teacher = Teacher(client, prompts, config)
        self.negative_label = config.laya.role_label("relevance", "negative")

    # -- entry point -------------------------------------------------------- #
    def mine(
        self,
        positives: Sequence[LabeledExample],
        pool: Mapping[str, Chunk],
        ctx: GenerationContext,
        *,
        retrieval_index: Optional[RetrievalIndex] = None,
        embed_fn: Optional[EmbedFn] = None,
    ) -> MiningReport:
        """Mine negatives for ``positives``, drawing candidates only from ``pool``."""
        report = MiningReport()
        settings = self.config.negatives
        if settings.ratio <= 0:
            report.bump("disabled_ratio_zero")
            return report
        if not pool:
            raise NegativeSamplingError(
                "the chunk pool is empty; negatives cannot be mined from nothing. "
                "This usually means mining ran before chunking."
            )
        if not settings.verify_with_teacher:
            LOGGER.warning(
                "teacher verification disabled; negative mining is skipped",
                extra={"fields": {"positives": len(positives)}},
            )
            report.bump("disabled_teacher_verification", len(positives))
            return report

        chunks = list(pool.values())
        bm25 = BM25Index([c.chunk_id for c in chunks], [c.body for c in chunks])

        for positive in positives:
            if positive.task != "relevance":
                report.bump("skipped_non_relevance_task")
                continue
            if positive.label_source == "human":
                report.bump("skipped_human_labeled")
                continue
            target = int(round(settings.ratio))
            if target <= 0:
                continue

            candidates = self._candidates(
                positive, pool, chunks, bm25, retrieval_index, embed_fn, report, ctx.rng
            )
            verified = self._verify(positive, pool, candidates, report)

            for candidate in verified[:target]:
                report.examples.append(
                    self._build_negative(positive, pool[candidate.chunk_id], candidate)
                )
                report.bump("negatives_accepted")
                report.bump(f"strategy_{candidate.strategy}")

            accepted = min(len(verified), target)
            if accepted < target:
                report.bump("positives_short_of_target")
                report.bump("negatives_unfilled", target - accepted)

        report.bump("teacher_calls", self.teacher.calls)
        LOGGER.info(
            "negative mining finished",
            extra={"fields": {"negatives": len(report.examples), **report.stats}},
        )
        return report

    # -- candidate selection ------------------------------------------------ #
    def _candidates(
        self,
        positive: LabeledExample,
        pool: Mapping[str, Chunk],
        chunks: Sequence[Chunk],
        bm25: BM25Index,
        retrieval_index: Optional[RetrievalIndex],
        embed_fn: Optional[EmbedFn],
        report: MiningReport,
        rng: random.Random,
    ) -> List[Candidate]:
        """Gated candidates, one per chunk, interleaved across strategies."""
        settings = self.config.negatives
        source_chunk_id = positive.chunk_ids[0] if positive.chunk_ids else ""

        budget = settings.candidates_per_positive
        drawn: Dict[str, List[Candidate]] = {}
        for strategy in settings.strategies:
            if strategy not in STRATEGIES:
                raise NegativeSamplingError(
                    f"unknown negative strategy {strategy!r}; expected one of {STRATEGIES}"
                )
            drawn[strategy] = self._draw(
                strategy, positive, pool, chunks, bm25, retrieval_index, embed_fn, report, rng
            )

        # Interleave the strategies round-robin so a prolific one cannot consume
        # the whole budget before a rarer one offers anything, then apply the
        # content gates, which can only remove candidates.
        ordered = _round_robin(
            [drawn[strategy] for strategy in settings.strategies],
            exclude=(source_chunk_id,),
        )
        report.bump("candidates_before_gates", len(ordered))

        gated: List[Candidate] = []
        for candidate in ordered:
            if len(gated) >= budget:
                break
            if not self._passes_content_gates(positive, pool[candidate.chunk_id], report):
                continue
            gated.append(candidate)

        report.bump("candidates_after_gates", len(gated))
        return gated

    def _draw(
        self,
        strategy: str,
        positive: LabeledExample,
        pool: Mapping[str, Chunk],
        chunks: Sequence[Chunk],
        bm25: BM25Index,
        retrieval_index: Optional[RetrievalIndex],
        embed_fn: Optional[EmbedFn],
        report: MiningReport,
        rng: random.Random,
    ) -> List[Candidate]:
        settings = self.config.negatives
        wanted = settings.candidates_per_positive
        source_chunk_id = positive.chunk_ids[0] if positive.chunk_ids else ""

        if strategy == "random":
            available = [c for c in chunks if c.chunk_id != source_chunk_id]
            # Fisher-Yates with the caller's seeded generator, so the random
            # strategy is as reproducible as the other two.
            for index in range(len(available) - 1, 0, -1):
                swap = rng.randint(0, index)
                available[index], available[swap] = available[swap], available[index]
            picked = available[:wanted]
            report.bump("candidates_random", len(picked))
            return [
                Candidate(chunk_id=c.chunk_id, strategy=strategy, margin=0.0, score=None)
                for c in picked
            ]

        if strategy == "lexical":
            hits = bm25.rank(positive.query, wanted)
            report.bump("candidates_lexical", len(hits))
            return [
                Candidate(hit.chunk_id, strategy, float(hit.score), float(hit.score))
                for hit in hits
            ]

        if retrieval_index is None:
            LOGGER.info(
                "no retrieval index supplied; skipping retrieval negatives",
                extra={"fields": {"example_id": positive.example_id}},
            )
            return []
        if retrieval_index.backend == "bm25":
            hits = retrieval_index.rank(positive.query, max(wanted * 4, wanted))
        else:
            if embed_fn is None:
                LOGGER.warning(
                    "retrieval backend needs an embed_fn; skipping retrieval negatives",
                    extra={"fields": {"backend": retrieval_index.backend}},
                )
                return []
            hits = retrieval_index.rank(
                positive.query, max(wanted * 4, wanted), embed_fn=embed_fn
            )

        # Skip the trivially-unrelated head and the near-duplicate tail: the
        # informative hard negatives live in between.
        kept = [
            Candidate(hit.chunk_id, strategy, float(hit.score), float(hit.score))
            for hit in hits
            if settings.min_similarity <= hit.score <= settings.max_similarity
        ]
        report.bump("candidates_retrieval", len(kept))
        return kept

    def _passes_content_gates(
        self, positive: LabeledExample, candidate: Chunk, report: MiningReport
    ) -> bool:
        """Gates 2 and 3: near-duplicate of the positive, or leaks an answer."""
        folded_candidate = fold_for_matching(candidate.text)
        for point in positive.key_points:
            span = fold_for_matching(point.span)
            if span and span in folded_candidate:
                report.reject(
                    positive.example_id, candidate.chunk_id, "contains_key_point_span"
                )
                return False
            text = fold_for_matching(point.text)
            if len(text) >= _MIN_LITERAL_GUARD_CHARS and text in folded_candidate:
                report.reject(
                    positive.example_id, candidate.chunk_id, "contains_key_point_text"
                )
                return False

        positive_body = positive.chunks[0].text if positive.chunks else positive.query
        positive_shingles = shingles(list(iter_tokens(positive_body)), _NEAR_DUP_SHINGLE)
        candidate_shingles = shingles(list(iter_tokens(candidate.body)), _NEAR_DUP_SHINGLE)
        overlap = jaccard(positive_shingles, candidate_shingles)
        if overlap >= self.config.negatives.near_duplicate_threshold:
            report.reject(
                positive.example_id,
                candidate.chunk_id,
                "near_duplicate_of_positive",
                overlap=round(overlap, 4),
            )
            return False
        return True

    # -- verification ------------------------------------------------------- #
    def _verify(
        self,
        positive: LabeledExample,
        pool: Mapping[str, Chunk],
        candidates: Sequence[Candidate],
        report: MiningReport,
    ) -> List[Candidate]:
        """Gate 4: the teacher must say the candidate does not answer the query."""
        accepted: List[Candidate] = []
        for candidate in candidates:
            text = pool[candidate.chunk_id].text
            try:
                verdict = self.teacher.verify_answerability(
                    positive.query, text, positive.key_points
                )
            except (OllamaError, SchemaError) as exc:
                report.reject(
                    positive.example_id, candidate.chunk_id, f"verification_error:{type(exc).__name__}"
                )
                continue

            if verdict.is_uncertain:
                report.reject(positive.example_id, candidate.chunk_id, "teacher_uncertain")
                report.bump("rejected_teacher_uncertain")
                continue
            if verdict.is_yes:
                report.reject(positive.example_id, candidate.chunk_id, "teacher_says_answers")
                report.bump("rejected_teacher_positive")
                continue
            accepted.append(candidate)
        return accepted

    # -- construction ------------------------------------------------------- #
    def _build_negative(
        self, positive: LabeledExample, chunk: Chunk, candidate: Candidate
    ) -> LabeledExample:
        return LabeledExample(
            example_id="",
            task=positive.task,
            query=positive.query,
            chunks=[
                ChunkRef(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_id=chunk.source_id,
                    relpath=chunk.relpath,
                    ordinal=chunk.ordinal,
                    score=candidate.score,
                )
            ],
            label=self.negative_label,
            label_source="teacher",
            strategy=candidate.strategy,
            margin=float(candidate.margin) if candidate.margin else None,
            verdict="no",
            role="negative",
            source_id=positive.source_id,
            relpath=positive.relpath,
            section_id=positive.section_id,
            key_points=list(positive.key_points),
            metadata={
                "positive_example_id": positive.example_id,
                "candidate_score": candidate.score,
            },
        )


def _round_robin(
    lists: Sequence[Sequence[Candidate]], *, exclude: Sequence[str] = ()
) -> List[Candidate]:
    """Interleave strategy outputs, de-duplicated by chunk id, order preserved.

    Terminates because each iteration either advances at least one cursor or
    exits: once every cursor is past its list, ``progressed`` stays False.
    """
    cursors = [0] * len(lists)
    blocked = set(exclude)
    seen: set[str] = set()
    ordered: List[Candidate] = []
    while True:
        progressed = False
        for position, items in enumerate(lists):
            while cursors[position] < len(items):
                candidate = items[cursors[position]]
                cursors[position] += 1
                if candidate.chunk_id in blocked or candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                ordered.append(candidate)
                progressed = True
                break
        if not progressed:
            return ordered


def candidate_pool_size(positive: LabeledExample, pool: Mapping[str, Chunk]) -> int:
    """How many chunks exist for a positive to draw from, excluding its own."""
    return sum(1 for chunk_id in pool if chunk_id not in positive.chunk_ids)


def supply_report(
    positives: Sequence[LabeledExample], pool: Mapping[str, Chunk], ratio: float
) -> Dict[str, Any]:
    """Whether the corpus can supply the requested ratio, per positive.

    A property of the corpus rather than of the sampler: if the answer is no,
    more retries will not help and the dataset is simply small.
    """
    target = int(round(ratio))
    sizes = [candidate_pool_size(positive, pool) for positive in positives]
    return {
        "positives": len(positives),
        "pool_chunks": len(pool),
        "target_per_positive": target,
        "min_candidates": min(sizes) if sizes else 0,
        "max_candidates": max(sizes) if sizes else 0,
        "short_positives": sum(1 for size in sizes if size < target),
    }


# Kept for callers that want the tuple form without the dataclass.
CandidateTuple = Tuple[str, str, float, Optional[float]]
