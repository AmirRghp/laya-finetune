"""Retrieval evaluation: Recall@K, MRR, nDCG@K.

A genuine retrieval measurement, not classification accuracy wearing a retrieval
hat (AGENTS.md 3.14, spec §12). The distinction is structural:

* a **real index** is built with the configured retrieval backend;
* a **real ranked list** comes out of it for every query;
* the gold set is the set of chunks that actually answer the query — the source
  chunk, plus any other chunk of the same document section that carries an
  answer.

A reranker sitting on top of the same index is measured by the same code, which
is how you tell whether it helps.

No query with an empty gold set may reach the mean: those would score 1.0 and
inflate it. They are counted and excluded, and the exclusion is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..corpus.models import Chunk
from ..dataset.builder import render_state
from ..dataset.records import LabeledExample
from ..errors import EvaluationError
from ..logging_setup import get_logger
from ..retrieval import (
    EmbedFn,
    RetrievalIndex,
    build_index,
    make_embed_fn,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

__all__ = ["RetrievalQuery", "build_queries", "evaluate_retrieval"]

LOGGER = get_logger("evaluation.retrieval")


@dataclass
class RetrievalQuery:
    """One query with its gold documents."""

    query: str
    relevant_ids: List[str] = field(default_factory=list)
    example_id: str = ""
    source_id: str = ""

    @property
    def has_gold(self) -> bool:
        return bool(self.relevant_ids)


def build_queries(
    examples: Sequence[LabeledExample],
    split: Optional[str] = None,
) -> List[RetrievalQuery]:
    """Turn labeled examples into retrieval queries.

    Uses positive relevance examples only: a negative's "chunk" is a distractor
    and would define the gold set backwards.
    """
    queries: List[RetrievalQuery] = []
    for example in examples:
        if example.role != "positive":
            continue
        if split and example.split != split:
            continue
        gold = [chunk.chunk_id for chunk in example.chunks]
        queries.append(
            RetrievalQuery(
                query=example.query,
                relevant_ids=gold,
                example_id=example.example_id,
                source_id=example.source_id,
            )
        )
    return queries


def evaluate_retrieval(
    config: Config,
    chunks: Sequence[Chunk],
    queries: Sequence[RetrievalQuery],
    *,
    index: Optional[RetrievalIndex] = None,
    embed_fn: Optional[EmbedFn] = None,
    ks: Optional[Sequence[int]] = None,
    rerank_with: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure the retrieval backend, optionally with a Laya reranker on top.

    Parameters
    ----------
    chunks:
        The pool the index is built over. Pass **one split's** chunks: indexing
        the whole corpus while querying with one split's questions is a valid
        retrieval measurement, but indexing another split's chunks changes what
        "retrieval" means and must be stated explicitly.
    queries:
        Evaluation queries with gold documents.
    rerank_with:
        A checkpoint to rerank the top candidates with, or ``None`` to measure
        the retriever alone.
    """
    if not queries:
        raise EvaluationError(
            "no retrieval queries to evaluate. Generate positives first, then pass "
            "the labeled examples."
        )
    if not chunks:
        raise EvaluationError("the retrieval index needs a chunk pool")

    cutoff = max(ks) if ks else max(config.evaluation.retrieval_ks)
    backend = config.evaluation.retrieval_backend

    resolved_index = index
    if resolved_index is None:
        resolved_embed = embed_fn
        if backend != "lexical":
            if resolved_embed is None:
                resolved_embed = make_embed_fn(
                    backend, config.ollama, config.chunking
                )
        resolved_index = build_index(backend, chunks, embed_fn=resolved_embed)
        resolved_embed = embed_fn

    id_to_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    per_query: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {}
    skipped_no_gold = 0

    for entry in queries:
        if not entry.has_gold:
            skipped_no_gold += 1
            continue
        missing = [cid for cid in entry.relevant_ids if cid not in id_to_chunk]
        if missing:
            LOGGER.warning(
                "query references chunks outside the indexed pool",
                extra={
                    "fields": {
                        "example_id": entry.example_id,
                        "missing": len(missing),
                    }
                },
            )

        ranked = _rank(resolved_index, entry.query, cutoff, resolved_embed)
        ranked_ids = [hit.chunk_id for hit in ranked]

        if rerank_with:
            ranked_ids = _rerank(config, rerank_with, entry.query, ranked, id_to_chunk)

        scores = {f"recall@{k}": recall_at_k(ranked_ids, entry.relevant_ids, k) for k in cutoff_list(config, ks)}
        scores["mrr"] = mrr(ranked_ids, entry.relevant_ids)
        for k in cutoff_list(config, ks):
            scores[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, entry.relevant_ids, k)

        for name, value in scores.items():
            totals[name] = totals.get(name, 0.0) + value
        per_query.append(
            {
                "example_id": entry.example_id,
                "source_id": entry.source_id,
                "query": entry.query,
                "gold": entry.relevant_ids,
                "ranked_top": ranked_ids[: max(cutoff_list(config, ks))],
                **{name: round(value, 6) for name, value in scores.items()},
            }
        )

    scored = len(per_query)
    result: Dict[str, Any] = {
        "backend": getattr(resolved_index, "backend", backend),
        "reranker": rerank_with,
        "indexed_chunks": len(chunks),
        "queries_total": len(queries),
        "queries_scored": scored,
        "queries_skipped_no_gold": skipped_no_gold,
        "ks": list(cutoff_list(config, ks)),
        "mean": {
            name: round(value / scored, 6) for name, value in sorted(totals.items())
        }
        if scored
        else {},
        "per_query": per_query,
    }
    if skipped_no_gold:
        result["note"] = (
            f"{skipped_no_gold} query/queries had no gold document and were excluded from "
            "the mean rather than counted as a perfect score"
        )
    LOGGER.info(
        "retrieval evaluation finished",
        extra={"fields": {"scored": scored, "backend": result["backend"]}},
    )
    return result


def cutoff_list(config: Config, ks: Optional[Sequence[int]]) -> List[int]:
    return sorted(set(ks)) if ks else list(config.evaluation.retrieval_ks)


def _rank(
    index: RetrievalIndex, query: str, top_k: int, embed_fn: Optional[EmbedFn]
) -> List[Any]:
    if index.backend == "bm25":
        return index.rank(query, top_k)
    if embed_fn is None:
        raise EvaluationError(
            f"the {index.backend!r} retrieval backend needs an embed_fn to score a query"
        )
    return index.rank(query, top_k, embed_fn=embed_fn)


def _rerank(
    config: Config,
    checkpoint: str,
    query: str,
    candidates: Sequence[Any],
    id_to_chunk: Mapping[str, Chunk],
) -> List[str]:
    """Rerank candidates with a Laya relevance decision.

    Reranking, not classification: the question is asked *per candidate*, so the
    measurement stays a retrieval measurement. The order of the returned ids is
    the order the model preferred.
    """
    try:
        import laya
    except ImportError as exc:
        raise EvaluationError(
            "reranking needs the `laya` package (`pip install -e .[laya]`)"
        ) from exc

    from ..evaluation.classification import answer_confidence_of, answer_label

    agent = laya.load(checkpoint)
    definition = config.laya.question("relevance")
    question = {
        "type": definition["type"],
        "instructions": definition["instructions"],
        "criteria": dict(definition["criteria"]),
    }
    positive = config.laya.role_label("relevance", "positive")

    scored: List[Tuple[float, float, str]] = []
    for hit in candidates:
        chunk = id_to_chunk.get(hit.chunk_id)
        if chunk is None:
            continue
        state = render_state(
            config.laya.template("relevance"), query, [chunk.text]
        )
        result = agent.predict(state, {"relevance": question})
        answers = result.get("answers") if isinstance(result, Mapping) else None
        answer = answers.get("relevance") if isinstance(answers, Mapping) else None
        predicted = answer_label(answer, "relevance")
        scored.append(
            (
                1.0 if predicted == positive else 0.0,
                answer_confidence_of(answer) or 0.0,
                hit.chunk_id,
            )
        )

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [chunk_id for _kept, _confidence, chunk_id in scored]