"""The pipeline's internal record types.

One record type flows from generation through labeling, splitting and dataset
building, so provenance is never lost:

``GeneratedExample``
    A question plus the evidence facts it was written against. Produced by a
    task generator.

``LabeledExample``
    A question, the retrieved chunks it is judged against, and a label. Produced
    by rule (positives), by the teacher, or by a human reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..ids import content_hash, stable_id

__all__ = [
    "SPLITS",
    "LABEL_SOURCES",
    "KeyPoint",
    "ChunkRef",
    "GeneratedExample",
    "LabeledExample",
    "split_of",
    "label_source_of",
]

#: The three canonical split names. Validation is the calibration split; test is
#: held out strictly (AGENTS.md 6.8).
SPLITS = ("train", "validation", "test")

#: Where a label came from. A human decision always outranks the others
#: (AGENTS.md 6.10).
LABEL_SOURCES = ("rule", "teacher", "human")

#: Ranking used when resolving which label wins.
_LABEL_PRECEDENCE = {"rule": 0, "teacher": 1, "human": 2}


@dataclass(frozen=True)
class KeyPoint:
    """One answer fact a correct response to the query must contain.

    ``span`` is the verbatim text from the source chunk that supports the point.
    It is stored so that an assertion of grounding can be checked later rather
    than taken on trust (AGENTS.md 6.4).
    """

    text: str
    span: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "span": self.span, "index": self.index}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KeyPoint":
        return cls(
            text=str(payload.get("text", "")),
            span=str(payload.get("span", "")),
            index=int(payload.get("index", 0)),
        )


@dataclass(frozen=True)
class ChunkRef:
    """A retrieved chunk as seen by a decision: its id and its text."""

    chunk_id: str
    text: str
    source_id: str = ""
    relpath: str = ""
    ordinal: int = 0
    score: Optional[float] = None
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_id": self.source_id,
            "relpath": self.relpath,
            "ordinal": self.ordinal,
        }
        if self.score is not None:
            payload["score"] = round(float(self.score), 6)
        if self.rank is not None:
            payload["rank"] = self.rank
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkRef":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            text=str(payload.get("text", "")),
            source_id=str(payload.get("source_id", "")),
            relpath=str(payload.get("relpath", "")),
            ordinal=int(payload.get("ordinal", 0)),
            score=payload.get("score"),
            rank=payload.get("rank"),
        )


@dataclass
class GeneratedExample:
    """A question written against one source chunk, with its evidence."""

    example_id: str
    task: str
    query: str
    query_type: str
    key_points: List[KeyPoint] = field(default_factory=list)
    source_chunk_id: str = ""
    source_id: str = ""
    relpath: str = ""
    section_id: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    generation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            self.example_id = make_example_id(self.task, self.query, self.source_chunk_id)

    @property
    def query_hash(self) -> str:
        return content_hash(self.query)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task": self.task,
            "query": self.query,
            "query_type": self.query_type,
            "key_points": [k.to_dict() for k in self.key_points],
            "source_chunk_id": self.source_chunk_id,
            "source_id": self.source_id,
            "relpath": self.relpath,
            "section_id": self.section_id,
            "heading_path": list(self.heading_path),
            "query_hash": self.query_hash,
            "generation": dict(self.generation),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeneratedExample":
        return cls(
            example_id=str(payload.get("example_id", "")),
            task=str(payload.get("task", "")),
            query=str(payload.get("query", "")),
            query_type=str(payload.get("query_type", "")),
            key_points=[KeyPoint.from_dict(k) for k in payload.get("key_points", [])],
            source_chunk_id=str(payload.get("source_chunk_id", "")),
            source_id=str(payload.get("source_id", "")),
            relpath=str(payload.get("relpath", "")),
            section_id=payload.get("section_id"),
            heading_path=list(payload.get("heading_path", [])),
            generation=dict(payload.get("generation", {})),
        )


@dataclass
class LabeledExample:
    """A question, the evidence it is judged against, and the judgment.

    ``chunks`` may be empty only for a task whose state template takes no
    ``{{chunks}}`` placeholder — query routing is the one such task. The dataset
    builder enforces that correspondence, so a relevance row can never reach
    Laya with an empty state, which ``items_from_rows`` would silently discard.
    """

    example_id: str
    task: str
    query: str
    chunks: List[ChunkRef]
    label: str
    label_source: str = "rule"
    confidence: Optional[float] = None
    #: For negatives: how the candidate was drawn (``random``/``lexical``/``retrieval``).
    strategy: Optional[str] = None
    #: For negatives: the similarity score that justified the candidate.
    margin: Optional[float] = None
    #: The teacher's verdict on the candidate, retained even when accepted.
    verdict: Optional[str] = None
    #: ``positive`` for a source chunk, ``negative`` for a mined candidate,
    #: ``retrieved`` for a sufficiency example judged against real retrieval.
    role: str = "positive"
    split: Optional[str] = None
    source_id: str = ""
    relpath: str = ""
    section_id: Optional[str] = None
    key_points: List[KeyPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(
                f"label_source must be one of {LABEL_SOURCES}, got {self.label_source!r}"
            )
        if self.split is not None and self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if not self.example_id:
            # Routing carries no chunks, so the query and label alone identify it.
            self.example_id = stable_id(
                "ex",
                self.task,
                self.query,
                *(chunk.chunk_id for chunk in self.chunks),
                self.label,
                self.role,
            )

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def source_ids(self) -> List[str]:
        seen: List[str] = []
        for chunk in self.chunks:
            if chunk.source_id and chunk.source_id not in seen:
                seen.append(chunk.source_id)
        return seen

    def content_signature(self) -> str:
        """Identity of the *inputs* to this decision: query plus chunk texts."""
        return content_hash(self.query, *[c.text for c in self.chunks])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task": self.task,
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "label": self.label,
            "label_source": self.label_source,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "margin": self.margin,
            "verdict": self.verdict,
            "role": self.role,
            "split": self.split,
            "source_id": self.source_id,
            "relpath": self.relpath,
            "section_id": self.section_id,
            "key_points": [k.to_dict() for k in self.key_points],
            "content_signature": self.content_signature(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LabeledExample":
        confidence = payload.get("confidence")
        margin = payload.get("margin")
        return cls(
            example_id=str(payload.get("example_id", "")),
            task=str(payload.get("task", "")),
            query=str(payload.get("query", "")),
            chunks=[ChunkRef.from_dict(c) for c in payload.get("chunks", [])],
            label=str(payload.get("label", "")),
            label_source=str(payload.get("label_source", "rule")),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            strategy=payload.get("strategy"),
            margin=float(margin) if isinstance(margin, (int, float)) else None,
            verdict=payload.get("verdict"),
            role=str(payload.get("role", "positive")),
            split=payload.get("split"),
            source_id=str(payload.get("source_id", "")),
            relpath=str(payload.get("relpath", "")),
            section_id=payload.get("section_id"),
            key_points=[KeyPoint.from_dict(k) for k in payload.get("key_points", [])],
            metadata=dict(payload.get("metadata", {})),
        )


def make_example_id(task: str, query: str, source_chunk_id: str) -> str:
    """Deterministic example id: task + query + the chunk it was written from."""
    return stable_id("ex", task, query, source_chunk_id)


def split_of(example: LabeledExample) -> Optional[str]:
    return example.split


def label_source_of(example: LabeledExample) -> str:
    return example.label_source


def strongest_label_source(current: Optional[str], candidate: str) -> str:
    """Which of two provenance labels wins. Human review outranks all."""
    if candidate not in LABEL_SOURCES:
        raise ValueError(f"unknown label source {candidate!r}")
    if current is None:
        return candidate
    return max(current, candidate, key=lambda name: _LABEL_PRECEDENCE[name])


def group_counts(examples: Sequence[LabeledExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {split: 0 for split in SPLITS}
    for example in examples:
        if example.split in counts:
            counts[example.split] += 1  # type: ignore[index]
    return counts