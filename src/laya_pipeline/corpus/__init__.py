"""Corpus ingestion: discovery, parsing and the document model."""

from __future__ import annotations

from .discovery import (
    MARKDOWN_SUFFIXES,
    discover_documents,
    find_duplicates,
    group_by_content,
)
from .markdown import block_texts, extract_section_id, parse_frontmatter, parse_markdown
from .models import (
    BLOCK_KINDS,
    Block,
    Chunk,
    DocumentSource,
    HeadingRef,
    ParsedDocument,
    make_chunk_id,
)

__all__ = [
    "BLOCK_KINDS",
    "Block",
    "Chunk",
    "DocumentSource",
    "HeadingRef",
    "MARKDOWN_SUFFIXES",
    "ParsedDocument",
    "block_texts",
    "discover_documents",
    "extract_section_id",
    "find_duplicates",
    "group_by_content",
    "make_chunk_id",
    "parse_frontmatter",
    "parse_markdown",
]