"""Heading-aware, structure-preserving chunking.

Boundaries fall between structural elements only (AGENTS.md 2.5, 2.6). A table,
a list or a fenced code block is never split: if one exceeds the budget it
becomes its own chunk and is flagged ``oversized`` so the validator can report
it rather than silently truncating a table's body from its header.

Sizing is in tokens of the configured tokenizer. When no tokenizer is
available the estimate is recorded as ``token_count_source: "estimated"`` and
carried through to every artifact, so an estimated run is distinguishable from
a measured one after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import ChunkingConfig
from .corpus.markdown import extract_section_id
from .corpus.models import Block, Chunk, HeadingRef, ParsedDocument, make_chunk_id
from .ids import content_hash
from .logging_setup import get_logger
from .tokenizer import EstimateTokenCounter, TokenCounter

__all__ = ["Chunker", "ChunkingStats", "chunk_document"]

LOGGER = get_logger("chunking")




@dataclass
class ChunkingStats:
    """What one document's chunking produced, for the processing report."""

    documents: int = 0
    blocks: int = 0
    chunks: int = 0
    oversized_chunks: int = 0
    small_chunks: int = 0
    token_source: str = "estimated"
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "blocks": self.blocks,
            "chunks": self.chunks,
            "oversized_chunks": self.oversized_chunks,
            "small_chunks": self.small_chunks,
            "token_source": self.token_source,
            "total_tokens": self.total_tokens,
        }


@dataclass
class _Draft:
    """A candidate chunk before merging and before its final id is assigned."""

    heading_path: List[HeadingRef]
    blocks: List[Block] = field(default_factory=list)
    oversized: bool = False


class Chunker:
    """Turns parsed documents into chunks.

    One instance per run. Holds the configuration and the token counter, and is
    therefore the only place chunk sizes are decided.
    """

    def __init__(self, config: ChunkingConfig, counter: Optional[TokenCounter] = None) -> None:
        self.config = config
        self.counter: TokenCounter = counter or EstimateTokenCounter(
            config.chars_per_token_estimate
        )

    # -- public ------------------------------------------------------------- #
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        drafts = self._draft(document)
        drafts = self._merge_small(drafts)
        chunks = self._materialize(document, drafts)
        LOGGER.debug(
            "chunked document",
            extra={
                "fields": {
                    "relpath": document.source.relpath,
                    "chunks": len(chunks),
                    "blocks": len(document.blocks),
                }
            },
        )
        return chunks

    def chunk_documents(self, documents: Sequence[ParsedDocument]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def stats(self, documents: Sequence[ParsedDocument], chunks: Sequence[Chunk]) -> ChunkingStats:
        return ChunkingStats(
            documents=len(documents),
            blocks=sum(len(d.blocks) for d in documents),
            chunks=len(chunks),
            oversized_chunks=sum(1 for c in chunks if c.oversized),
            small_chunks=sum(1 for c in chunks if c.metadata.get("small")),
            token_source=chunks[0].token_count_source if chunks else self.counter.source,
            total_tokens=sum(c.token_count or 0 for c in chunks),
        )

    def token_count(self, text: str) -> int:
        return self.counter.count(text)

    # -- drafting ----------------------------------------------------------- #
    def _draft(self, document: ParsedDocument) -> List[_Draft]:
        cfg = self.config
        drafts: List[_Draft] = []
        heading_path: List[HeadingRef] = []
        current = _Draft(heading_path=[], blocks=[])
        current_tokens = 0
        overlap_carried: List[Block] = []
        current_is_empty = True

        def flush(oversized: bool = False) -> None:
            nonlocal current, current_tokens, current_is_empty
            if not current.blocks:
                return
            draft = _Draft(
                heading_path=list(current.heading_path),
                blocks=list(current.blocks),
                oversized=oversized,
            )
            drafts.append(draft)
            carried: List[Block] = []
            if cfg.overlap_blocks > 0 and len(current.blocks) > 1:
                carried = [b for b in current.blocks[-cfg.overlap_blocks :] if b.kind != "heading"]
            overlap_carried[:] = carried
            current = _Draft(heading_path=list(current.heading_path), blocks=[])
            current_tokens = 0
            current_is_empty = True

        def seed(blocks: List[Block]) -> int:
            nonlocal current_tokens, current_is_empty
            current.blocks = list(blocks)
            current_tokens = sum(self.counter.count(b.text) for b in blocks)
            current_is_empty = False
            return current_tokens

        for block in document.blocks:
            if block.kind == "heading":
                # A heading starts a new section: close the current chunk, drop
                # any overlap (it belongs to the previous section) and re-root
                # the heading path.
                flush()
                overlap_carried.clear()
                while heading_path and heading_path[-1].level >= (block.level or 1):
                    heading_path.pop()
                heading_path.append(HeadingRef(level=block.level or 1, text=block.text))
                current.heading_path = list(heading_path)
                continue

            block_tokens = self.counter.count(block.text)

            if current_is_empty:
                if block_tokens > cfg.max_tokens:
                    current.blocks = [block]
                    current.heading_path = list(heading_path)
                    current_tokens = block_tokens
                    current_is_empty = False
                    flush(oversized=True)
                    continue
                seed(overlap_carried + [block])
                continue

            if current_tokens + block_tokens <= cfg.target_tokens:
                current.blocks.append(block)
                current_tokens += block_tokens
                continue

            if block_tokens <= cfg.max_tokens:
                # Close the chunk and let the overlap seed the next one.
                flush()
                seed(overlap_carried + [block])
                continue

            # The block alone exceeds the budget. Splitting a table or a fenced
            # block would separate a header from its body, so it is emitted whole
            # and flagged.
            flush()
            current.heading_path = list(heading_path)
            current.blocks = [block]
            current_tokens = block_tokens
            current_is_empty = False
            flush(oversized=True)

        flush()
        return drafts

    def _merge_small(self, drafts: List[_Draft]) -> List[_Draft]:
        """Fold a too-small chunk into its predecessor when they still fit.

        A section's last few lines routinely fall below ``min_tokens`` on their
        own. Attaching them to the preceding chunk keeps the section whole and
        avoids a retrieval unit that answers nothing.
        """
        cfg = self.config
        merged: List[_Draft] = []
        for draft in drafts:
            if merged and not draft.oversized and not merged[-1].oversized:
                previous = merged[-1]
                if previous.heading_path == draft.heading_path:
                    combined = sum(self.counter.count(b.text) for b in previous.blocks)
                    combined += sum(self.counter.count(b.text) for b in draft.blocks)
                    if combined <= cfg.max_tokens:
                        previous.blocks = previous.blocks + draft.blocks
                        continue
            merged.append(draft)
        return merged

    # -- materialisation ---------------------------------------------------- #
    def _materialize(self, document: ParsedDocument, drafts: List[_Draft]) -> List[Chunk]:
        cfg = self.config
        source = document.source
        frontmatter = dict(document.frontmatter)
        declared_title = document.declared_title
        declared_id = document.declared_id
        chunks: List[Chunk] = []

        for ordinal, draft in enumerate(drafts):
            body = "\n\n".join(b.text.strip() for b in draft.blocks if b.text.strip())
            if not body.strip():
                continue
            heading_path = draft.heading_path
            text = body
            if cfg.include_heading_path and heading_path:
                breadcrumb = "\n".join(
                    f"{'#' * h.level} {h.text}".rstrip() for h in heading_path
                )
                text = f"{breadcrumb}\n\n{body}"

            section_id = None
            for heading in reversed(heading_path):
                section_id = extract_section_id(heading.text)
                if section_id:
                    break

            metadata: Dict[str, Any] = {}
            if frontmatter:
                metadata["frontmatter"] = frontmatter
            if declared_title:
                metadata["document_title"] = declared_title
            if declared_id:
                metadata["declared_id"] = declared_id

            token_count = self.counter.count(text)
            body_len = len(body)
            if body_len < cfg.min_chunk_chars or token_count < cfg.min_tokens:
                metadata["small"] = True

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        source.source_id,
                        [h.text for h in heading_path],
                        ordinal,
                        text,
                    ),
                    source_id=source.source_id,
                    relpath=source.relpath,
                    content_id=source.content_id,
                    ordinal=ordinal,
                    text=text,
                    body=body,
                    heading_path=list(heading_path),
                    section_id=section_id,
                    block_kinds=[b.kind for b in draft.blocks],
                    char_len=len(text),
                    token_count=token_count,
                    token_count_source=self.counter.source,
                    oversized=draft.oversized,
                    metadata=metadata,
                )
            )

        self._detect_internal_duplicates(chunks)
        return chunks

    @staticmethod
    def _detect_internal_duplicates(chunks: Sequence[Chunk]) -> None:
        """Mark chunks whose body repeats an earlier chunk in the same document.

        Overlap between adjacent chunks is intentional; an *identical* body is
        an artefact worth surfacing, because duplicated content skews both
        negative sampling and retrieval evaluation.
        """
        seen: Dict[str, str] = {}
        for chunk in chunks:
            key = content_hash(chunk.body)
            if key in seen:
                chunk.metadata["duplicate_of"] = seen[key]
            else:
                seen[key] = chunk.chunk_id


def chunk_document(
    document: ParsedDocument,
    config: ChunkingConfig,
    counter: Optional[TokenCounter] = None,
) -> List[Chunk]:
    """Convenience wrapper around :class:`Chunker` for a single document."""
    return Chunker(config, counter).chunk_document(document)