"""Retrieval backends for hard-negative mining and retrieval evaluation.

Deliberately not a vector database. The pipeline runs on one machine over a
corpus of a few hundred chunks; a numpy matrix is the right size of tool, is
fully reproducible, and adds no infrastructure (AGENTS.md 7.8).

Three backends behind one protocol:

``ollama``
    The configured Ollama embedding model. The default.
``laya_encoder``
    Mean-pooled encoder from a loaded Laya checkpoint, via the official
    ``laya.shortlist.embed_fn_from_agent``. No extra download.
``lexical``
    BM25 over folded tokens. No model at all; the fallback when no embedding
    backend is reachable.

Adding a backend means adding one class here and one name in
``evaluation.retrieval_backend``; nothing else changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from .config import ChunkingConfig, OllamaConfig
from .corpus.models import Chunk
from .errors import EvaluationError, PipelineError
from .ids import fold_for_matching, iter_tokens, token_set
from .logging_setup import get_logger

__all__ = [
    "EmbedFn",
    "RankedHit",
    "RetrievalIndex",
    "NumpyEmbeddingIndex",
    "BM25Index",
    "build_index",
    "cosine_matrix",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "lexical_overlap",
    "make_embed_fn",
    "laya_encoder_embed_fn",
]

LOGGER = get_logger("retrieval")

#: The contract an embedding backend must satisfy: texts in, one vector per text.
EmbedFn = Callable[[Sequence[str]], "object"]

_BM25_K1 = 1.5
_BM25_B = 0.75


@dataclass(frozen=True)
class RankedHit:
    """One ranked result."""

    chunk_id: str
    score: float
    rank: int


class RetrievalIndex(Protocol):
    """Anything that can rank chunks against a query."""

    backend: str

    def rank(self, query: str, top_k: Optional[int] = None) -> List[RankedHit]:
        ...  # pragma: no cover - protocol

    def score_of(self, query: str, chunk_id: str) -> Optional[float]:
        ...  # pragma: no cover - protocol

    @property
    def ids(self) -> Sequence[str]:
        ...  # pragma: no cover - protocol


# --------------------------------------------------------------------------- #
# Vector index
# --------------------------------------------------------------------------- #
def _l2_normalize(matrix: "object") -> "object":
    import numpy as np

    array = np.asarray(matrix, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def cosine_matrix(matrix: "object", vector: "object") -> "object":
    """Cosine similarity of one vector against every row of a matrix.

    Rows are normalised on build, so this is a single matrix product.
    """
    import numpy as np

    return np.asarray(matrix, dtype="float32") @ np.asarray(vector, dtype="float32")


class NumpyEmbeddingIndex:
    """Exhaustive inner-product search over normalised embeddings.

    Exhaustive is the correct algorithm here: a corpus of a few hundred chunks
    is smaller than any index's overhead, and it has no approximation error, so
    recall numbers are reproducible.
    """

    def __init__(
        self,
        chunk_ids: Sequence[str],
        vectors: "object",
        *,
        backend: str = "embeddings",
    ) -> None:
        import numpy as np

        if len(chunk_ids) == 0:
            raise PipelineError("cannot build an embedding index over zero chunks")
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise PipelineError(f"embeddings must be 2-D, got shape {matrix.shape}")
        if matrix.shape[0] != len(chunk_ids):
            raise PipelineError(
                f"embeddings has {matrix.shape[0]} rows for {len(chunk_ids)} chunks"
            )
        self._ids = list(chunk_ids)
        self._matrix = _l2_normalize(matrix)
        self.backend = backend
        self.dimension = int(matrix.shape[1])

    @property
    def ids(self) -> Sequence[str]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def embed_queries(self, queries: Sequence[str], embed_fn: EmbedFn) -> "object":
        vectors = embed_fn(list(queries))
        import numpy as np

        return _l2_normalize(np.asarray(vectors, dtype="float32"))

    def rank(self, query: str, top_k: Optional[int] = None, *, embed_fn: Optional[EmbedFn] = None) -> List[RankedHit]:
        return self.rank_vector(self._vector_for(query, embed_fn), top_k)

    def rank_vector(self, query_vector: "object", top_k: Optional[int] = None) -> List[RankedHit]:
        import numpy as np

        scores = np.asarray(cosine_matrix(self._matrix, query_vector)).ravel()
        limit = len(scores) if top_k is None else max(0, int(top_k))
        # argpartition then a stable sort of the shortlist: O(n) selection plus
        # O(k log k) ordering, with ties broken by chunk id for determinism.
        if limit and limit < len(scores):
            candidates = np.argpartition(-scores, limit - 1)[:limit]
        else:
            candidates = np.arange(len(scores))
        order = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), self._ids[i]))
        return [
            RankedHit(chunk_id=self._ids[index], score=float(scores[index]), rank=position)
            for position, index in enumerate(order[:limit])
        ]

    def score_of(self, query: str, chunk_id: str, *, embed_fn: Optional[EmbedFn] = None) -> Optional[float]:
        import numpy as np

        vector = self._vector_for(query, embed_fn)
        scores = np.asarray(cosine_matrix(self._matrix, vector)).ravel()
        try:
            index = self._ids.index(chunk_id)
        except ValueError:
            return None
        return float(scores[index])

    def _vector_for(self, query: str, embed_fn: Optional[EmbedFn]) -> "object":
        if embed_fn is None:
            raise PipelineError(
                "this index needs an embed_fn to score a raw query string; pass one "
                "or use rank_vector with a pre-computed vector"
            )
        import numpy as np

        vectors = np.asarray(embed_fn([query]), dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return _l2_normalize(vectors)[0]


# --------------------------------------------------------------------------- #
# Lexical index
# --------------------------------------------------------------------------- #
class BM25Index:
    """Okapi BM25 over folded Persian tokens.

    ZWNJ is a token separator in :func:`iter_tokens`, so ``دانش‌بنیان`` indexes
    as two terms. No stemming is attempted: Persian morphology is productive
    enough that a wrong light stemmer costs more recall than it buys
    (AGENTS.md 2.2).
    """

    backend = "bm25"

    def __init__(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise PipelineError(
                f"BM25 got {len(chunk_ids)} ids for {len(texts)} texts"
            )
        if not chunk_ids:
            raise PipelineError("cannot build a BM25 index over zero chunks")
        self._ids = list(chunk_ids)
        self._postings: List[Dict[str, int]] = []
        document_frequency: Dict[str, int] = {}
        total_length = 0
        for text in texts:
            counts: Dict[str, int] = {}
            for token in iter_tokens(fold_for_matching(text)):
                counts[token] = counts.get(token, 0) + 1
            self._postings.append(counts)
            total_length += sum(counts.values())
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self._document_frequency = document_frequency
        self._average_length = total_length / len(self._postings)

    @property
    def ids(self) -> Sequence[str]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def _idf(self, token: str) -> float:
        n = len(self._postings)
        df = self._document_frequency.get(token, 0)
        # Robertson/Sparck-Jones idf with the +1 that keeps it positive for a
        # term appearing in every document.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> "object":
        import numpy as np

        result = np.zeros(len(self._postings), dtype="float32")
        query_tokens = list(dict.fromkeys(iter_tokens(fold_for_matching(query))))
        if not query_tokens or self._average_length <= 0:
            return result
        for index, counts in enumerate(self._postings):
            length = sum(counts.values())
            if length == 0:
                continue
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token)
                if not frequency:
                    continue
                denominator = frequency + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * length / self._average_length
                )
                score += self._idf(token) * (frequency * (_BM25_K1 + 1.0)) / denominator
            result[index] = score
        return result

    def rank(self, query: str, top_k: Optional[int] = None) -> List[RankedHit]:
        import numpy as np

        scores = self.scores(query)
        limit = len(scores) if top_k is None else max(0, int(top_k))
        if limit and limit < len(scores):
            candidates = np.argpartition(-scores, limit - 1)[:limit].tolist()
        else:
            candidates = list(range(len(scores)))
        order = sorted(candidates, key=lambda i: (-float(scores[i]), self._ids[i]))
        return [
            RankedHit(chunk_id=self._ids[index], score=float(scores[index]), rank=position)
            for position, index in enumerate(order[:limit])
        ]

    def score_of(self, query: str, chunk_id: str) -> Optional[float]:
        import numpy as np

        try:
            index = self._ids.index(chunk_id)
        except ValueError:
            return None
        return float(self.scores(query)[index])


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_index(
    backend: str,
    chunks: Sequence[Chunk],
    *,
    embed_fn: Optional[EmbedFn] = None,
    batch_size: int = 32,
) -> RetrievalIndex:
    """Build an index of ``chunks`` using the named backend.

    ``embed_fn`` is required for the embedding backends and ignored by the
    lexical one. Building is eager and reported, so a corpus too large for this
    approach fails loudly at build time instead of slowly at query time.
    """
    if not chunks:
        raise PipelineError("cannot build a retrieval index over zero chunks")

    ids = [chunk.chunk_id for chunk in chunks]

    if backend == "lexical":
        LOGGER.info("building lexical index", extra={"fields": {"chunks": len(chunks)}})
        return BM25Index(ids, [chunk.body for chunk in chunks])

    if embed_fn is None:
        raise PipelineError(
            f"the {backend!r} retrieval backend needs an embed_fn; none was supplied"
        )

    LOGGER.info(
        "building embedding index",
        extra={"fields": {"backend": backend, "chunks": len(chunks), "batch_size": batch_size}},
    )
    vectors = _embed_batched(embed_fn, [chunk.text for chunk in chunks], batch_size)
    return NumpyEmbeddingIndex(ids, vectors, backend=backend)


def _embed_batched(embed_fn: EmbedFn, texts: Sequence[str], batch_size: int) -> "object":
    import numpy as np

    parts = []
    for start in range(0, len(texts), max(1, batch_size)):
        window = list(texts[start : start + max(1, batch_size)])
        vectors = np.asarray(embed_fn(window), dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(len(window), -1)
        if vectors.shape[0] != len(window):
            raise PipelineError(
                f"embed_fn returned {vectors.shape[0]} vectors for {len(window)} texts; "
                "a silent mismatch would corrupt the index"
            )
        parts.append(vectors)
    return np.concatenate(parts, axis=0)


def laya_encoder_embed_fn(checkpoint: str, max_length: int = 512, batch_size: int = 32) -> EmbedFn:
    """Build an embed_fn from a Laya checkpoint's own encoder.

    Uses the official ``laya.shortlist.embed_fn_from_agent``. Loads the
    checkpoint, so it belongs to the stages that already need weights — never
    to ingestion or dataset building.
    """
    try:
        import laya
        from laya.shortlist import embed_fn_from_agent
    except ImportError as exc:
        raise EvaluationError(
            "the laya_encoder retrieval backend needs the `laya` package; install it "
            "with `pip install -e .[laya]` or use retrieval_backend: ollama"
        ) from exc
    agent = laya.load(checkpoint)
    return embed_fn_from_agent(agent, max_length=max_length, batch_size=batch_size)


def make_embed_fn(
    backend: str,
    ollama_config: OllamaConfig,
    chunking_config: ChunkingConfig,
    client: Optional[object] = None,
) -> EmbedFn:
    """Return the embed_fn for a configured retrieval backend."""
    if backend == "lexical":
        return _unused_embed_fn
    if backend == "laya_encoder":
        return laya_encoder_embed_fn(chunking_config.tokenizer_model)
    if backend != "ollama":
        raise PipelineError(
            f"unknown retrieval backend {backend!r}; expected one of "
            "'ollama', 'laya_encoder', 'lexical'"
        )

    from .llm.ollama import Embedder, OllamaClient

    resolved = client if client is not None else OllamaClient(ollama_config)
    return Embedder(resolved, ollama_config.embedding_model)


def _unused_embed_fn(texts: Sequence[str]) -> "object":  # pragma: no cover
    raise PipelineError("the lexical backend does not use embeddings")


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of relevant documents found in the top ``k``.

    A query with no relevant document scores 1.0 by convention; callers that
    care must exclude such queries beforehand, because a silently generous 1.0
    inflates the mean.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def mrr(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant document, or 0.0."""
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    for position, identifier in enumerate(ranked_ids, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Binary-gain nDCG at ``k``.

    Uses the standard position discount ``1 / log2(rank + 1)``. Gain is binary:
    a chunk either answers the query or it does not, which is what a relevance
    judgment in this corpus means.
    """
    import math

    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 1.0 if not relevant else 0.0
    gain = 0.0
    for position, identifier in enumerate(ranked_ids[:k], start=1):
        if identifier in relevant:
            gain += 1.0 / math.log2(position + 1)
    best = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return gain / best if best else 0.0


def lexical_overlap(left: str, right: str) -> float:
    """Containment-style token overlap between two texts.

    ``|A ∩ B| / min(|A|, |B|)`` rather than Jaccard: the question is whether the
    shorter text's vocabulary is contained in the longer one, which is what
    makes two documents *about the same thing* rather than merely sharing words.
    """
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))