"""Markdown parsing: frontmatter, structure, Persian, no mid-block splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from laya_pipeline.corpus.markdown import (
    block_texts,
    extract_section_id,
    parse_frontmatter,
    parse_markdown,
)
from laya_pipeline.corpus.models import Block, DocumentSource
from laya_pipeline.errors import CorpusError
from laya_pipeline.ids import content_hash, stable_id


def make_source(text: str, name: str = "doc.md") -> DocumentSource:
    from laya_pipeline.ids import normalize_text

    normalized = normalize_text(text)
    return DocumentSource(
        path=Path(name),
        relpath=name,
        content_id=stable_id("cnt", normalized),
        byte_size=len(normalized.encode("utf-8")),
        text=normalized,
    )


def blocks_of(text: str) -> list[Block]:
    return parse_markdown(make_source(text)).blocks


def kinds(text: str) -> list[str]:
    return [block.kind for block in blocks_of(text)]


class TestFrontmatter:
    def test_yaml_frontmatter_is_parsed(self) -> None:
        document = parse_markdown(
            make_source("---\ndocument_id: AA-1\ntitle: \"عنوان\"\n---\n\n# بدنه\n\nمتن بدنه\n")
        )
        assert document.frontmatter["document_id"] == "AA-1"
        assert document.frontmatter["title"] == "عنوان"
        # Frontmatter is consumed, not emitted as content.
        assert [block.kind for block in document.blocks] == ["heading", "paragraph"]
        assert document.blocks[1].text == "متن بدنه"

    def test_frontmatter_is_not_counted_as_content(self) -> None:
        document = parse_markdown(make_source("---\nkey: value\n---\n\n# فقط عنوان\n"))
        assert [block.kind for block in document.blocks] == ["heading"]

    def test_missing_frontmatter_consumes_nothing(self) -> None:
        lines = ["# عنوان", "", "متن"]
        assert parse_frontmatter(lines) == ({}, 0)

    def test_unterminated_frontmatter_is_not_frontmatter(self) -> None:
        assert parse_frontmatter(["---", "key: value"]) == ({}, 0)

    def test_empty_frontmatter_block(self) -> None:
        assert parse_frontmatter(["---", "---", "متن"])[0] == {}

    def test_declared_id_and_title_are_metadata(self) -> None:
        document = parse_markdown(
            make_source('---\ndocument_id: AA-1\n---\n\n# عنوان سند\n\nمتن\n')
        )
        assert document.declared_id == "AA-1"
        assert document.declared_title == "عنوان سند"

    def test_title_falls_back_to_first_h1(self) -> None:
        document = parse_markdown(make_source("## بند\n\nمتن\n"))
        assert document.declared_title is None


class TestHeadings:
    def test_atx_levels_are_captured(self) -> None:
        assert kinds("# یک\n## دو\n### سه\n") == ["heading", "heading", "heading"]

    def test_heading_text_is_stripped_of_hashes(self) -> None:
        assert blocks_of("## بند دوم ##\n")[0].text == "بند دوم"

    def test_setext_headings(self) -> None:
        parsed = blocks_of("عنوان یک\n=======\n\nعنوان دو\n-----\n")
        assert [(b.kind, b.level) for b in parsed] == [
            ("heading", 1),
            ("heading", 2),
        ]

    def test_thematic_break_is_not_a_heading(self) -> None:
        assert kinds("متن\n\n---\n\nمتن دو\n") == ["paragraph", "paragraph"]

    def test_section_id_is_extracted_when_declared(self) -> None:
        assert extract_section_id("AA-001-S1 — ثبت درخواست") == "AA-001-S1"

    def test_persian_heading_has_no_section_id(self) -> None:
        assert extract_section_id("برنامه‌ریزی مرخصی") is None

    def test_hyphenated_english_word_is_not_a_section_id(self) -> None:
        assert extract_section_id("well-known policy") is None


class TestStructures:
    def test_paragraphs_split_on_blank_lines(self) -> None:
        assert kinds("یک\n\nدو\n") == ["paragraph", "paragraph"]

    def test_list_is_one_block(self) -> None:
        parsed = blocks_of("- یک\n- دو\n- سه\n")
        assert len(parsed) == 1
        assert parsed[0].kind == "list"
        assert parsed[0].text.count("-") == 3

    def test_table_rows_stay_together(self) -> None:
        text = "| ستون | مقدار |\n| --- | --- |\n| الف | ۱ |\n| ب | ۲ |\n"
        parsed = blocks_of(text)
        assert len(parsed) == 1
        assert parsed[0].kind == "table"
        assert parsed[0].text.count("\n") == 3

    def test_fenced_code_block_is_never_split(self) -> None:
        text = "```python\nx = 1\ny = 2\n```\n"
        parsed = blocks_of(text)
        assert len(parsed) == 1
        assert parsed[0].kind == "code"
        assert parsed[0].text.startswith("```python")
        assert parsed[0].text.rstrip().endswith("```")

    def test_unterminated_fence_still_yields_one_block(self) -> None:
        parsed = blocks_of("```\nبدون بستن\n")
        assert len(parsed) == 1
        assert parsed[0].kind == "code"

    def test_blockquote_is_one_block(self) -> None:
        parsed = blocks_of("> خط یک\n> خط دو\n")
        assert len(parsed) == 1
        assert parsed[0].kind == "quote"

    def test_structure_boundary_starts_a_new_block(self) -> None:
        assert kinds("متن\n- فهرست\n") == ["paragraph", "list"]

    def test_ordered_list_is_detected(self) -> None:
        assert kinds("1. یک\n2. دو\n") == ["list"]

    def test_fence_inside_paragraph_interrupts_it(self) -> None:
        assert kinds("متن\n```\nکد\n```\n") == ["paragraph", "code"]


class TestPersian:
    def test_persian_text_round_trips(self) -> None:
        text = "هر کارمند باید درخواست مرخصی را ثبت کند.\n"
        assert blocks_of(text)[0].text == text.strip()

    def test_zwnj_is_preserved(self) -> None:
        assert "دانش‌بنیان" in blocks_of("دانش‌بنیان مهم است.\n")[0].text

    def test_persian_digits_are_preserved(self) -> None:
        assert "۱۳۹۹" in blocks_of("سال ۱۳۹۹ است.\n")[0].text

    def test_line_numbers_are_recorded(self) -> None:
        parsed = blocks_of("# عنوان\n\nمتن\n")
        assert parsed[1].line_start == 3


class TestRobustness:
    def test_empty_document_yields_no_blocks(self) -> None:
        assert blocks_of("") == []

    def test_blank_lines_only(self) -> None:
        assert blocks_of("\n\n\n") == []

    def test_block_kind_is_validated(self) -> None:
        with pytest.raises(ValueError):
            Block(kind="nonsense", text="x")

    def test_heading_block_requires_a_level(self) -> None:
        with pytest.raises(ValueError):
            Block(kind="heading", text="x", level=9)

    def test_block_texts_can_skip_headings(self) -> None:
        parsed = blocks_of("# عنوان\n\nمتن\n")
        assert block_texts(parsed) == ["متن"]
        assert len(block_texts(parsed, skip_headings=False)) == 2


class TestIdentity:
    def test_content_id_is_stable_across_parses(self) -> None:
        from tests.conftest import PERSIAN_DOC_A

        assert make_source(PERSIAN_DOC_A).content_id == make_source(PERSIAN_DOC_A).content_id

    def test_two_documents_get_distinct_content_ids(self) -> None:
        from tests.conftest import PERSIAN_DOC_A, PERSIAN_DOC_B

        assert make_source(PERSIAN_DOC_A).content_id != make_source(PERSIAN_DOC_B).content_id

    def test_source_id_follows_the_path(self) -> None:
        assert make_source("متن", "a.md").source_id != make_source("متن", "b.md").source_id


def test_corpus_error_is_typed() -> None:
    assert issubclass(CorpusError, Exception)
    assert content_hash("x")  # the helper used throughout is importable