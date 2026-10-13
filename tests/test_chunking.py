"""Chunking: heading hierarchy, structure preservation, size bounds, ids."""

from __future__ import annotations

import pytest

from laya_pipeline.chunking import Chunker
from laya_pipeline.config import ChunkingConfig
from laya_pipeline.corpus.markdown import parse_markdown
from laya_pipeline.corpus.models import Chunk, DocumentSource
from laya_pipeline.ids import normalize_text, stable_id
from laya_pipeline.tokenizer import EstimateTokenCounter, TokenCounter

from tests.conftest import PERSIAN_DOC_A


def source(text: str, name: str = "doc.md") -> DocumentSource:
    normalized = normalize_text(text)
    return DocumentSource(
        path=name,
        relpath=name,
        content_id=stable_id("cnt", normalized),
        byte_size=len(normalized.encode("utf-8")),
        text=normalized,
    )


def make_chunker(**overrides) -> Chunker:
    config = ChunkingConfig(min_chunk_chars=0, **overrides)
    return Chunker(config, EstimateTokenCounter(config.chars_per_token_estimate))


def chunk(text: str, **overrides) -> list[Chunk]:
    return make_chunker(**overrides).chunk_document(parse_markdown(source(text)))


class TestStructure:
    def test_every_chunk_has_a_body(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert chunks and all(c.body.strip() for c in chunks)

    def test_heading_path_is_preserved(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        paths = [" › ".join(h.text for h in c.heading_path) for c in chunks]
        assert any("ثبت درخواست" in path for path in paths)

    def test_breadcrumb_is_rendered_into_the_text(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert any(c.text.startswith("# سیاست مرخصی") for c in chunks)

    def test_breadcrumb_can_be_disabled(self) -> None:
        chunks = chunk(PERSIAN_DOC_A, include_heading_path=False)
        assert all(not c.text.startswith("#") for c in chunks)

    def test_a_new_heading_starts_a_new_chunk(self) -> None:
        chunks = chunk("# یک\n\nمتن یک\n\n# دو\n\nمتن دو\n")
        assert len(chunks) >= 2
        assert chunks[0].heading_path[0].text == "یک"
        assert chunks[-1].heading_path[0].text == "دو"

    def test_section_id_is_recorded(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert any(c.section_id == "AA-001-S1" for c in chunks)

    def test_frontmatter_reaches_chunk_metadata(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert all(c.metadata.get("document_title") == "سیاست مرخصی" for c in chunks)

    def test_block_kinds_are_recorded(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert all("paragraph" in c.block_kinds for c in chunks)


class TestIntegrity:
    def test_a_table_is_never_split(self) -> None:
        text = "# ت\n\n" + "\n\n".join(
            "متن " * 200 + "\n\n| الف | ب |\n| --- | --- |\n| ۱ | ۲ |" for _ in range(3)
        )
        chunks = chunk(text, max_tokens=200)
        tables = [c for c in chunks if "table" in c.block_kinds]
        assert tables, "expected at least one table chunk"
        for entry in tables:
            assert entry.body.count("| --- | --- |") == 1

    def test_a_code_fence_is_never_split(self) -> None:
        text = "# ت\n\n```\n" + "خط\n" * 300 + "```\n"
        chunks = chunk(text, max_tokens=150)
        code_chunks = [c for c in chunks if "code" in c.block_kinds]
        assert code_chunks
        for entry in code_chunks:
            assert entry.body.count("```") % 2 == 0

    def test_oversized_chunks_are_flagged(self) -> None:
        text = "# ت\n\n```\n" + "خط\n" * 400 + "```\n"
        chunks = chunk(text, max_tokens=120)
        assert any(c.oversized for c in chunks)


class TestSizing:
    def test_chunks_respect_the_budget_when_split_is_possible(self) -> None:
        """Blocks small enough to fit are packed up to ``target_tokens``.

        Each paragraph is sized below ``max_tokens`` on its own, so splitting is
        possible and the chunker must do it rather than emitting whole blocks.
        """
        unit = "این یک بند از متن فارسی است. "
        paragraphs = "\n\n".join(unit * 3 for _ in range(12))
        chunks = chunk("# ت\n\n" + paragraphs, target_tokens=40, max_tokens=120)

        assert chunks and not any(c.oversized for c in chunks)
        # Every chunk holds more than one block, so the budget is actually used.
        assert all(len(c.block_kinds) > 1 for c in chunks)
        assert max(c.token_count for c in chunks) <= 120

    def test_a_block_above_the_budget_becomes_its_own_chunk(self) -> None:
        """The counterpart: a single oversized block is not silently truncated."""
        big = "متن بسیار بلند. " * 200
        small = "متن کوتاه دیگر. " * 4
        chunks = chunk(f"# ت\n\n{big}\n\n{small}\n", target_tokens=60, max_tokens=120)
        assert any(c.oversized for c in chunks)
        assert all(len(c.body) > 0 for c in chunks)

    def test_token_count_source_is_recorded(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert all(c.token_count_source == "estimated" for c in chunks)

    def test_token_count_matches_the_counter(self) -> None:
        chunker = make_chunker()
        counter = EstimateTokenCounter(chunker.config.chars_per_token_estimate)
        for entry in chunk(PERSIAN_DOC_A):
            assert entry.token_count == counter.count(entry.text)

    def test_min_tokens_pulls_a_stray_line_into_its_neighbour(self) -> None:
        """A trailing stub is folded in rather than emitted on its own."""
        body = "متن بلند فارسی برای آزمون. " * 60
        text = f"# ت\n\n{body}\n\nیک جملهٔ کوتاه.\n"
        merged = chunk(text, target_tokens=400, max_tokens=900, min_tokens=40)
        assert len(merged) == 1, [c.body[:40] for c in merged]
        assert "یک جملهٔ کوتاه" in merged[0].body

    def test_a_stub_is_kept_when_it_cannot_be_merged(self) -> None:
        """Without room in the neighbour, the stub survives as its own chunk.

        Dropping it would lose content; the alternative is a chunk that answers
        nothing, which is why the merge is attempted but never forced.
        """
        body = "متن بلند فارسی برای آزمون. " * 60
        text = f"# ت\n\n{body}\n\nیک جملهٔ کوتاه.\n"
        chunks = chunk(text, target_tokens=40, max_tokens=60, min_tokens=20)
        assert any("یک جملهٔ کوتاه" in c.body for c in chunks)


class TestIdentity:
    def test_ids_are_stable_across_runs(self) -> None:
        first = [c.chunk_id for c in chunk(PERSIAN_DOC_A)]
        second = [c.chunk_id for c in chunk(PERSIAN_DOC_A)]
        assert first == second

    def test_editing_a_chunk_changes_its_id(self) -> None:
        before = chunk(PERSIAN_DOC_A)[0].chunk_id
        after = chunk(PERSIAN_DOC_A.replace("دو روز کاری", "پنج روز کاری"))[0].chunk_id
        assert before != after

    def test_identical_bodies_are_marked_as_duplicates(self) -> None:
        text = "# ت\n\nمتن یکسان\n\n## الف\n\nمتن یکسان\n"
        chunks = chunk(text)
        assert any("duplicate_of" in c.metadata for c in chunks)

    def test_ordinals_are_sequential(self) -> None:
        chunks = chunk(PERSIAN_DOC_A)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_round_trip_through_dict(self) -> None:
        original = chunk(PERSIAN_DOC_A)[0]
        assert Chunk.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_custom_counter_is_used() -> None:
    class FixedCounter:
        source = "fixed"

        def count(self, text: str) -> int:
            return 10

    config = ChunkingConfig(target_tokens=15, max_tokens=20, min_tokens=5, min_chunk_chars=0)
    chunks = Chunker(config, FixedCounter()).chunk_document(
        parse_markdown(source("# ت\n\n" + "\n\n".join(["متن " * 30] * 8)))
    )
    assert len(chunks) > 1
    assert all(c.token_count == 10 for c in chunks)


def test_counter_protocol_is_satisfied() -> None:
    assert isinstance(EstimateTokenCounter(2.5), TokenCounter)


def test_chunker_rejects_a_bad_estimate() -> None:
    with pytest.raises(ValueError):
        EstimateTokenCounter(0)