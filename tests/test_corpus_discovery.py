"""Recursive discovery, determinism, filename independence, duplicates."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from laya_pipeline.corpus.discovery import (
    MARKDOWN_SUFFIXES,
    discover_documents,
    find_duplicates,
    group_by_content,
)
from laya_pipeline.errors import DiscoveryError
from laya_pipeline.io import write_text
from tests.conftest import PERSIAN_DOC_A, PERSIAN_DOC_B


class TestDiscovery:
    def test_finds_nested_markdown(self, corpus_dir: Path) -> None:
        found = discover_documents(corpus_dir)
        relpaths = {document.relpath for document in found}
        assert relpaths == {"a.md", "c.md", "team/policies/b.markdown"}

    def test_extensions_are_case_insensitive(self, tmp_path: Path) -> None:
        write_text(tmp_path / "A.MD", PERSIAN_DOC_A)
        write_text(tmp_path / "B.Markdown", PERSIAN_DOC_B)
        assert len(discover_documents(tmp_path)) == 2

    def test_non_markdown_files_are_ignored(self, tmp_path: Path) -> None:
        write_text(tmp_path / "a.md", PERSIAN_DOC_A)
        write_text(tmp_path / "notes.txt", "متن")
        write_text(tmp_path / "data.json", "{}")
        assert len(discover_documents(tmp_path)) == 1

    def test_order_is_deterministic(self, tmp_path: Path) -> None:
        for name in ("z.md", "a.md", "m/b.md", "m/a.md"):
            write_text(tmp_path / name, PERSIAN_DOC_A + name)
        first = [d.relpath for d in discover_documents(tmp_path)]
        second = [d.relpath for d in discover_documents(tmp_path)]
        assert first == second == sorted(first)

    def test_missing_corpus_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DiscoveryError, match="does not exist"):
            discover_documents(tmp_path / "absent")

    def test_file_instead_of_directory_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "file.md"
        write_text(target, PERSIAN_DOC_A)
        with pytest.raises(DiscoveryError, match="not a directory"):
            discover_documents(target)

    def test_unsupported_suffix_is_rejected(self, corpus_dir: Path) -> None:
        with pytest.raises(DiscoveryError, match="unsupported document suffix"):
            discover_documents(corpus_dir, include_suffixes=(".txt",))

    def test_excluded_directory_names_are_skipped(self, tmp_path: Path) -> None:
        write_text(tmp_path / "keep.md", PERSIAN_DOC_A)
        write_text(tmp_path / "skipme.md", PERSIAN_DOC_B)
        found = discover_documents(tmp_path, exclude_dirs=["skipme.md"])
        assert [d.relpath for d in found] == ["keep.md"]

    def test_empty_corpus_returns_empty_list(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert discover_documents(tmp_path / "empty") == []


class TestFilenameIndependence:
    """AGENTS.md 1.7: nothing may branch on a filename."""

    def test_same_content_under_scrambled_names_yields_same_chunks(self, tmp_path: Path) -> None:
        from laya_pipeline.chunking import Chunker
        from laya_pipeline.config import ChunkingConfig
        from laya_pipeline.corpus.markdown import parse_markdown

        original_root = tmp_path / "one"
        original_root.mkdir()
        write_text(original_root / "aa.md", PERSIAN_DOC_A)
        write_text(original_root / "bb.markdown", PERSIAN_DOC_B)
        original = discover_documents(original_root)

        scrambled_root = tmp_path / "two"
        for index, document in enumerate(reversed(original)):
            write_text(scrambled_root / f"zz-{index * 7 + 3}.md", document.text)
        scrambled = discover_documents(scrambled_root)

        chunker = Chunker(ChunkingConfig(min_chunk_chars=0))
        first = [
            chunk.body
            for document in original
            for chunk in chunker.chunk_document(parse_markdown(document))
        ]
        second = [
            chunk.body
            for document in scrambled
            for chunk in chunker.chunk_document(parse_markdown(document))
        ]
        assert first == second
        assert len(first) > 0

    def test_numbering_does_not_change_content_ids(self, tmp_path: Path) -> None:
        write_text(tmp_path / "HR-001.md", PERSIAN_DOC_A)
        write_text(tmp_path / "qzx-999.md", PERSIAN_DOC_A)
        ids = {d.content_id for d in discover_documents(tmp_path)}
        assert len(ids) == 1

    def test_a_prefixed_name_is_not_privileged(self, tmp_path: Path) -> None:
        """No filename prefix changes grouping, ordering or content handling."""
        plain_root = tmp_path / "plain"
        plain_root.mkdir()
        write_text(plain_root / "b.md", PERSIAN_DOC_A)
        write_text(plain_root / "a.md", PERSIAN_DOC_B)

        prefixed_root = tmp_path / "prefixed"
        prefixed_root.mkdir()
        write_text(prefixed_root / "HR-001.md", PERSIAN_DOC_A)
        write_text(prefixed_root / "SEC-999.md", PERSIAN_DOC_B)

        plain = {d.relpath: d.content_id for d in discover_documents(plain_root)}
        prefixed = {d.relpath: d.content_id for d in discover_documents(prefixed_root)}
        # Same order (by path) and same content ids; only the names differ.
        assert [d.relpath for d in discover_documents(plain_root)] == ["a.md", "b.md"]
        assert [d.relpath for d in discover_documents(prefixed_root)] == [
            "HR-001.md", "SEC-999.md",
        ]
        assert set(plain.values()) == set(prefixed.values())

    def test_renaming_changes_source_id_but_not_content_id(self, tmp_path: Path) -> None:
        first_root = tmp_path / "before"
        write_text(first_root / "old.md", PERSIAN_DOC_A)
        before = discover_documents(first_root)[0]

        second_root = tmp_path / "after"
        write_text(second_root / "new.md", PERSIAN_DOC_A)
        after = discover_documents(second_root)[0]

        assert before.content_id == after.content_id
        assert before.source_id != after.source_id


class TestDuplicates:
    def test_identical_content_is_flagged(self, tmp_path: Path) -> None:
        write_text(tmp_path / "a.md", PERSIAN_DOC_A)
        write_text(tmp_path / "deep" / "b.md", PERSIAN_DOC_A)
        duplicates = find_duplicates(discover_documents(tmp_path))
        assert len(duplicates) == 1
        assert duplicates.popitem()[1] == ["a.md", "deep/b.md"]

    def test_distinct_content_is_not_flagged(self, corpus_dir: Path) -> None:
        assert find_duplicates(discover_documents(corpus_dir)) == {}

    def test_grouping_is_sorted(self, tmp_path: Path) -> None:
        for name in ("z.md", "a.md"):
            write_text(tmp_path / name, PERSIAN_DOC_A)
        groups = group_by_content(discover_documents(tmp_path))
        assert list(groups.values())[0] == ["a.md", "z.md"]


class TestNewDocuments:
    def test_a_newly_added_document_is_discovered(self, tmp_path: Path) -> None:
        write_text(tmp_path / "a.md", PERSIAN_DOC_A)
        assert len(discover_documents(tmp_path)) == 1
        write_text(tmp_path / "b.md", PERSIAN_DOC_B)
        assert len(discover_documents(tmp_path)) == 2

    def test_copied_corpus_produces_identical_results(self, tmp_path: Path) -> None:
        from tests.conftest import PERSIAN_DOC_C

        source = tmp_path / "src"
        write_text(source / "a.md", PERSIAN_DOC_A)
        write_text(source / "b.md", PERSIAN_DOC_B)
        write_text(source / "c.md", PERSIAN_DOC_C)
        replica = tmp_path / "replica"
        shutil.copytree(source, replica)
        assert [d.content_id for d in discover_documents(source)] == [
            d.content_id for d in discover_documents(replica)
        ]


def test_markdown_suffixes_are_the_documented_pair() -> None:
    assert set(MARKDOWN_SUFFIXES) == {".md", ".markdown"}