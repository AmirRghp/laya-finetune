"""In-memory document model.

These dataclasses are the pipeline's internal currency: discovery produces
:class:`DocumentSource`, parsing produces :class:`ParsedDocument`, and chunking
produces :class:`Chunk`. Each carries everything needed to trace a training row
back to the exact bytes it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ids import content_hash, short_hash, stable_id

__all__ = [
    "BLOCK_KINDS",
    "Block",
    "HeadingRef",
    "DocumentSource",
    "ParsedDocument",
    "Chunk",
]

#: Block kinds the Markdown parser emits. ``heading`` blocks are structural and
#: never carry body text into a chunk body; the rest do.
BLOCK_KINDS = (
    "paragraph",
    "list",
    "table",
    "code",
    "quote",
    "heading",
)

#: Heading levels 1..6, matching CommonMark.
MAX_HEADING_LEVEL = 6


@dataclass(frozen=True)
class Block:
    """One structural element of a document."""

    kind: str
    text: str
    level: Optional[int] = None
    line_start: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise ValueError(
                f"unknown block kind {self.kind!r}; expected one of {BLOCK_KINDS}"
            )
        if self.kind == "heading":
            if not isinstance(self.level, int) or not 1 <= self.level <= MAX_HEADING_LEVEL:
                raise ValueError(
                    f"heading block needs a level in 1..{MAX_HEADING_LEVEL}, got {self.level!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"kind": self.kind, "text": self.text}
        if self.level is not None:
            payload["level"] = self.level
        if self.line_start:
            payload["line_start"] = self.line_start
        return payload


@dataclass(frozen=True)
class HeadingRef:
    """A heading and its depth, for the breadcrumb carried on every chunk."""

    level: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "text": self.text}


@dataclass(frozen=True)
class DocumentSource:
    """A discovered file, before parsing."""

    path: Path
    relpath: str
    content_id: str
    byte_size: int
    text: str

    @property
    def source_id(self) -> str:
        """Stable identity of the document's *location*.

        Derived from the relative path so that renaming a file changes its id
        while leaving its ``content_id`` — and therefore its duplicate status —
        intact. Incremental processing keys off ``content_id``, not this.
        """
        return stable_id("src", self.relpath)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "relpath": self.relpath,
            "content_id": self.content_id,
            "byte_size": self.byte_size,
        }


@dataclass
class ParsedDocument:
    """A document after Markdown parsing."""

    source: DocumentSource
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    blocks: List[Block] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def content_id(self) -> str:
        return self.source.content_id

    @property
    def declared_title(self) -> Optional[str]:
        """A title the document declares about itself, if any.

        Read from frontmatter, falling back to the first level-1 heading.
        Metadata only — never used to classify, route or select documents.
        """
        for key in ("title", "name"):
            value = self.frontmatter.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for block in self.blocks:
            if block.kind == "heading" and block.level == 1:
                return block.text.strip()
        return None

    @property
    def declared_id(self) -> Optional[str]:
        """An identifier the document declares about itself, if any.

        Metadata only. The pipeline never branches on its shape (AGENTS.md 1.7).
        """
        value = self.frontmatter.get("document_id")
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
        return None


@dataclass
class Chunk:
    """A retrieval-sized unit of a document.

    ``text`` is the exact string handed to the LLM as context, including the
    heading breadcrumb. ``body`` is the same text without it, for similarity
    computation that should ignore structural context.
    """

    chunk_id: str
    source_id: str
    relpath: str
    content_id: str
    ordinal: int
    text: str
    body: str
    heading_path: List[HeadingRef] = field(default_factory=list)
    section_id: Optional[str] = None
    block_kinds: List[str] = field(default_factory=list)
    char_len: int = 0
    token_count: Optional[int] = None
    token_count_source: str = "estimated"
    oversized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.char_len:
            self.char_len = len(self.text)

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "relpath": self.relpath,
            "content_id": self.content_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "heading_path": [h.to_dict() for h in self.heading_path],
            "section_id": self.section_id,
            "block_kinds": list(self.block_kinds),
            "char_len": self.char_len,
            "token_count": self.token_count,
            "token_count_source": self.token_count_source,
            "oversized": self.oversized,
            "content_hash": self.content_hash,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Chunk":
        headings = [
            HeadingRef(level=int(h["level"]), text=str(h["text"]))
            for h in payload.get("heading_path", [])
        ]
        return cls(
            chunk_id=str(payload["chunk_id"]),
            source_id=str(payload["source_id"]),
            relpath=str(payload.get("relpath", "")),
            content_id=str(payload.get("content_id", "")),
            ordinal=int(payload.get("ordinal", 0)),
            text=str(payload["text"]),
            body=str(payload.get("body", payload["text"])),
            heading_path=headings,
            section_id=payload.get("section_id"),
            block_kinds=list(payload.get("block_kinds", [])),
            char_len=int(payload.get("char_len", 0)),
            token_count=payload.get("token_count"),
            token_count_source=str(payload.get("token_count_source", "estimated")),
            oversized=bool(payload.get("oversized", False)),
            metadata=dict(payload.get("metadata", {})),
        )

    def short_hash(self) -> str:
        return short_hash(self.text)


def make_chunk_id(
    source_id: str, heading_titles: List[str], ordinal: int, text: str
) -> str:
    """Deterministic chunk identifier.

    Includes the ordinal so two structurally identical sections in one document
    stay distinct, and the text so that editing a chunk changes its id — which
    is what lets downstream stages skip unchanged work.
    """
    return stable_id("chunk", source_id, *heading_titles, str(ordinal), text)