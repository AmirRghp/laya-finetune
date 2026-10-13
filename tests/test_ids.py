"""Identifiers, hashing and Persian text comparison."""

from __future__ import annotations

import pytest

from laya_pipeline.ids import (
    char_ngram_jaccard,
    content_hash,
    fold_for_matching,
    iter_tokens,
    jaccard,
    normalize_text,
    normalize_whitespace,
    shingles,
    stable_id,
    token_set,
)


class TestNormalisation:
    def test_bom_is_stripped(self) -> None:
        assert normalize_text("﻿سلام").startswith("سلام")

    def test_line_endings_are_folded(self) -> None:
        assert normalize_text("a\r\nb\rc") == "a\nb\nc"

    def test_persian_characters_survive(self) -> None:
        text = "دانش‌بنیان — ۱۳۹۹/۰۵/۱۲"
        assert normalize_text(text) == text

    def test_normalisation_is_idempotent(self) -> None:
        once = normalize_text("﻿خط اول\r\nخط دوم")
        assert normalize_text(once) == once

    def test_empty_input(self) -> None:
        assert normalize_text("") == ""


class TestFolding:
    def test_arabic_and_persian_yeh_agree(self) -> None:
        assert fold_for_matching("كتاب") == fold_for_matching("کتاب")

    def test_digits_fold_to_ascii(self) -> None:
        assert fold_for_matching("۱۲۳") == fold_for_matching("123")
        assert fold_for_matching("١٢٣") == fold_for_matching("123")

    def test_zwnj_is_removed_for_matching(self) -> None:
        assert fold_for_matching("دانش‌بنیان") == fold_for_matching("دانشبنیان")

    def test_folding_does_not_mutate_storage_text(self) -> None:
        original = "كتاب مخصوص"
        fold_for_matching(original)
        assert original == "كتاب مخصوص"


class TestTokenisation:
    def test_zwnj_separates_tokens(self) -> None:
        assert list(iter_tokens("دانش‌بنیان")) == ["دانش", "بنیان"]

    def test_punctuation_only_tokens_are_dropped(self) -> None:
        # Digits fold to ASCII for matching, so a Persian and an ASCII numeral
        # are the same token: that is what lets ١٢٣ and ۱۲۳ and 123 agree.
        assert list(iter_tokens("سلام --- ۱۲۳")) == ["سلام", "123"]

    def test_token_set_deduplicates(self) -> None:
        assert token_set("سیاست سیاست مرخصی") == frozenset({"سیاست", "مرخصی"})

    def test_folded_tokens_are_lowercased(self) -> None:
        assert all(token == token.lower() for token in token_set("HR-001-S1"))


class TestSimilarity:
    def test_identical_sets_score_one(self) -> None:
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets_score_zero(self) -> None:
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_two_empty_sets_are_identical(self) -> None:
        assert jaccard(set(), set()) == 1.0

    def test_empty_against_non_empty(self) -> None:
        assert jaccard(set(), {"a"}) == 0.0

    def test_char_ngrams_similar_for_paraphrase(self) -> None:
        left = "مدیر حداکثر دو روز کاری برای پاسخ فرصت دارد"
        right = "مدیر حداکثر دو روز کاری برای پاسخ‌دادن فرصت دارد"
        assert char_ngram_jaccard(left, right) > 0.7

    def test_char_ngrams_differ_for_unrelated(self) -> None:
        left = "سیاست مرخصی سالانه"
        right = "کنترل دسترسی به دادهٔ مشتری"
        assert char_ngram_jaccard(left, right) < 0.3

    def test_short_string_yields_itself_as_shingle(self) -> None:
        assert shingles(["a", "b"], 3) == {("a", "b")}

    def test_shingle_windows_are_complete(self) -> None:
        assert shingles(["a", "b", "c", "d"], 2) == {("a", "b"), ("b", "c"), ("c", "d")}


class TestHashing:
    def test_content_hash_is_stable(self) -> None:
        assert content_hash("سلام", "دنیا") == content_hash("سلام", "دنیا")

    def test_part_boundary_prevents_collisions(self) -> None:
        assert content_hash("ab", "c") != content_hash("a", "bc")

    def test_folded_hash_ignores_spelling_variation(self) -> None:
        assert content_hash("كتاب") == content_hash("کتاب")

    def test_unfolded_hash_is_exact(self) -> None:
        assert content_hash("كتاب", fold=False) != content_hash("کتاب", fold=False)

    def test_stable_id_has_prefix_and_is_deterministic(self) -> None:
        first = stable_id("chunk", "a", "b")
        assert first.startswith("chunk-")
        assert first == stable_id("chunk", "a", "b")

    def test_stable_id_requires_prefix(self) -> None:
        with pytest.raises(ValueError):
            stable_id("", "a")

    def test_different_inputs_give_different_ids(self) -> None:
        assert stable_id("chunk", "a") != stable_id("chunk", "b")


class TestWhitespace:
    def test_runs_collapse(self) -> None:
        assert normalize_whitespace("  a \n\n b  ") == "a b"