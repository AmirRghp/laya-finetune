"""Markdown parsing: frontmatter plus structural blocks.

A dependency-free, line-oriented CommonMark subset. It recognises exactly the
structures the corpus actually uses and the chunker must not break: ATX and
setext headings, fenced code blocks, pipe tables, lists, blockquotes and
paragraphs.

Why hand-rolled rather than a Markdown library: the pipeline must run before
any heavy dependency is installed, and — more importantly — it must guarantee
that a table or a fenced block is handed to the chunker as one indivisible unit
(AGENTS.md 2.6). A library that re-flows paragraphs would quietly break that.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import CorpusError
from ..ids import normalize_text
from .models import Block, DocumentSource, ParsedDocument

__all__ = ["parse_markdown", "parse_frontmatter", "extract_section_id", "block_texts"]

_ATX_HEADING = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,9}[.)])[ \t]+\S")
_QUOTE = re.compile(r"^\s{0,3}>")
_TABLE_ROW = re.compile(r"^\s{0,3}\|")
_SETEXT = re.compile(r"^\s{0,3}(=+|-{2,})\s*$")
_THEMATIC_BREAK = re.compile(r"^\s{0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
# A self-declared section identifier, e.g. the "HR-001-S1" in "HR-001-S1 — title".
# Extracted as *metadata only*; nothing in the pipeline branches on its shape.
_SECTION_ID = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)\s*(?:[—–\-:|]\s*)?(.*)$")

_FRONT_MATTER_DELIMITERS = ("---", "...")


def parse_markdown(source: DocumentSource) -> ParsedDocument:
    """Parse a discovered document into frontmatter plus ordered blocks."""
    text = normalize_text(source.text)
    lines = text.split("\n")
    frontmatter, consumed = parse_frontmatter(lines)
    body_lines = lines[consumed:]
    blocks = _parse_blocks(body_lines, line_offset=consumed)
    return ParsedDocument(source=source, frontmatter=frontmatter, blocks=blocks)


def parse_frontmatter(lines: Sequence[str]) -> Tuple[Dict[str, Any], int]:
    """Split a leading YAML frontmatter block off ``lines``.

    Returns ``(mapping, lines_consumed)``. A document without frontmatter, or
    with an unterminated delimiter, consumes nothing — a stray ``---`` is a
    thematic break, not the start of metadata.

    Values are parsed with PyYAML when it is installed, because the corpus
    documents its own metadata in YAML. A small ``key: value`` parser is the
    fallback so that ingestion does not fail without PyYAML.
    """
    if not lines:
        return {}, 0
    if lines[0].strip() != "---":
        return {}, 0

    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() in _FRONT_MATTER_DELIMITERS:
            closing = index
            break
    if closing is None:
        return {}, 0

    raw = "\n".join(lines[1:closing])
    return _parse_yaml_block(raw), closing + 1


def _parse_yaml_block(raw: str) -> Dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(raw)
        if isinstance(loaded, dict):
            return {str(k): v for k, v in loaded.items()}
        if loaded is None:
            return {}
        return {"value": loaded}
    except ImportError:
        return _parse_simple_mapping(raw)
    except Exception:
        # Malformed frontmatter is metadata, not content. Keep the document.
        return _parse_simple_mapping(raw)


def _parse_simple_mapping(raw: str) -> Dict[str, Any]:
    """Minimal ``key: value`` / ``- item`` parser used as the fallback path."""
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for line in raw.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and current_key is not None:
            bucket = result.setdefault(current_key, [])
            if isinstance(bucket, list):
                bucket.append(_unquote(line.lstrip()[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if value:
            result[key] = _unquote(value)
            current_key = key
        else:
            result[key] = []
            current_key = key
    return result


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_blocks(lines: Sequence[str], line_offset: int = 0) -> List[Block]:
    blocks: List[Block] = []
    paragraph: List[str] = []
    paragraph_line = 0

    def flush_paragraph() -> None:
        nonlocal paragraph, paragraph_line
        if paragraph:
            body = "\n".join(paragraph).strip()
            if body:
                blocks.append(
                    Block(kind="paragraph", text=body, line_start=paragraph_line + line_offset + 1)
                )
        paragraph = []
        paragraph_line = 0

    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]

        if not line.strip():
            flush_paragraph()
            index += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            flush_paragraph()
            marker = fence.group(1)
            start = index
            block_lines = [line]
            index += 1
            while index < total:
                block_lines.append(lines[index])
                closing = _FENCE.match(lines[index])
                if closing and closing.group(1)[0] == marker[0] and len(closing.group(1)) >= len(marker):
                    index += 1
                    break
                index += 1
            blocks.append(
                Block(
                    kind="code",
                    text="\n".join(block_lines),
                    line_start=start + line_offset + 1,
                )
            )
            continue

        atx = _ATX_HEADING.match(line)
        if atx:
            flush_paragraph()
            blocks.append(
                Block(
                    kind="heading",
                    text=atx.group(2).strip(),
                    level=len(atx.group(1)),
                    line_start=index + line_offset + 1,
                )
            )
            index += 1
            continue

        # Setext is checked before the thematic break: with a paragraph open,
        # `-----` closes it as an H2 underline. CommonMark resolves it this way,
        # and checking the break first turns every underlined subheading into a
        # horizontal rule.
        setext = _SETEXT.match(line)
        if setext and paragraph:
            body = "\n".join(paragraph).strip()
            level = 1 if setext.group(1).startswith("=") else 2
            blocks.append(
                Block(
                    kind="heading",
                    text=body,
                    level=level,
                    line_start=paragraph_line + line_offset + 1,
                )
            )
            paragraph = []
            paragraph_line = 0
            index += 1
            continue

        # Only now, with no paragraph open, is `-----` unambiguously a rule.
        if _THEMATIC_BREAK.match(line):
            index += 1
            continue

        if _TABLE_ROW.match(line):
            flush_paragraph()
            start = index
            block_lines: List[str] = []
            while index < total and (_TABLE_ROW.match(lines[index]) or not lines[index].strip()):
                # A blank line inside a table body is tolerated; stop at two.
                if not lines[index].strip():
                    look = index + 1
                    while look < total and not lines[look].strip():
                        look += 1
                    if look >= total or not _TABLE_ROW.match(lines[look]):
                        break
                block_lines.append(lines[index])
                index += 1
            blocks.append(
                Block(kind="table", text="\n".join(block_lines).strip("\n"), line_start=start + line_offset + 1)
            )
            continue

        if _LIST_ITEM.match(line):
            flush_paragraph()
            start = index
            block_lines = []
            while index < total:
                current = lines[index]
                if not current.strip():
                    break
                if _LIST_ITEM.match(current) or current.startswith(("  ", "\t")):
                    block_lines.append(current)
                    index += 1
                    continue
                if (
                    _ATX_HEADING.match(current)
                    or _FENCE.match(current)
                    or _TABLE_ROW.match(current)
                    or _QUOTE.match(current)
                ):
                    break
                # Lazy continuation of the previous list item's paragraph.
                block_lines.append(current)
                index += 1
            blocks.append(
                Block(kind="list", text="\n".join(block_lines).rstrip(), line_start=start + line_offset + 1)
            )
            continue

        if _QUOTE.match(line):
            flush_paragraph()
            start = index
            block_lines = []
            while index < total and lines[index].strip():
                current = lines[index]
                if _QUOTE.match(current) or not _starts_new_block(current):
                    block_lines.append(current)
                    index += 1
                    continue
                break
            blocks.append(
                Block(kind="quote", text="\n".join(block_lines).rstrip(), line_start=start + line_offset + 1)
            )
            continue

        if not paragraph:
            paragraph_line = index
        paragraph.append(line)
        index += 1

    flush_paragraph()
    return blocks


def _starts_new_block(line: str) -> bool:
    return bool(
        _ATX_HEADING.match(line)
        or _FENCE.match(line)
        or _TABLE_ROW.match(line)
        or _LIST_ITEM.match(line)
        or _QUOTE.match(line)
    )


def extract_section_id(heading_text: str) -> Optional[str]:
    """Pull a self-declared identifier out of a heading, if the heading has one.

    Purely descriptive metadata: it records what the document says about
    itself. No stage selects, groups or classifies by its presence or shape
    (AGENTS.md 1.7).
    """
    match = _SECTION_ID.match((heading_text or "").strip())
    if not match:
        return None
    candidate = match.group(1)
    # A declared identifier carries at least one numeric segment (HR-001-S1,
    # SEC-12). Without that requirement an ordinary hyphenated word — "well-known",
    # "data-entry" — would be harvested as an id, which is exactly the kind of
    # accidental naming dependency AGENTS.md 1.7 forbids.
    segments = candidate.split("-")
    if len(segments) < 2 or not any(segment.isdigit() for segment in segments):
        return None
    return candidate


def block_texts(blocks: Sequence[Block], *, skip_headings: bool = True) -> List[str]:
    """The non-empty body texts of ``blocks``."""
    return [
        block.text
        for block in blocks
        if block.text.strip() and not (skip_headings and block.kind == "heading")
    ]


def require_parsable(source: DocumentSource) -> ParsedDocument:
    """Parse, converting an unexpected failure into a typed pipeline error."""
    try:
        return parse_markdown(source)
    except CorpusError:
        raise
    except Exception as exc:
        raise CorpusError(
            f"failed to parse {source.relpath}: {type(exc).__name__}: {exc}"
        ) from exc