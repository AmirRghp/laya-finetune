"""Identifiers, content hashing and Unicode-aware text comparison.

Two things live here and the distinction matters (AGENTS.md 2.1-2.3):

* :func:`normalize_text` — the *only* transformation allowed on persisted text.
  BOM removal, CRLF folding, NFC. Nothing else.
* :func:`fold_for_matching` — a lossy, comparison-only projection used for
  hashing, dedup and similarity. It is never written back to a document.

Tokenisation is Persian-aware: ZWNJ (U+200C) separates words rather than being
whitespace, so ``دانش‌بنیان`` yields two tokens rather than one unsearchable one.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Sequence

__all__ = [
    "normalize_text",
    "fold_for_matching",
    "content_hash",
    "stable_id",
    "short_hash",
    "iter_tokens",
    "token_set",
    "shingles",
    "jaccard",
    "char_ngrams",
    "char_ngram_jaccard",
    "normalize_whitespace",
]

_BOM = "﻿"
_ZERO_WIDTH = "‌‍‎‏⁠﻿"
_TATWEEL = "ـ"
_HARAKAT = re.compile(r"[ً-ْٰٓ-ٕ]")
_ARABIC_DIGITS = re.compile(r"[٠-٩]")
_EXTENDED_ARABIC_DIGITS = re.compile(r"[۰-۹]")

# Characters treated as token separators when extracting words. ZWNJ is
# included because in Persian it joins morphemes, not words: "دانش‌بنیان" is
# two content units and treating it as one makes lexical overlap useless.
_SEPARATORS = re.compile(
    r"[\s‌­‐-―!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~«»؛،؟٪‐-‧‰-⁞]+"
)
_PUNCT_CLASS = re.compile(r"[^\w؀-ۿ]+", re.UNICODE)

_YEH_FOLD = str.maketrans({"ي": "ی", "ى": "ی"})
_KAF_FOLD = str.maketrans({"ك": "ک"})
_DIGIT_FOLD = {chr(0x0660 + i): str(i) for i in range(10)}
_DIGIT_FOLD.update({chr(0x06F0 + i): str(i) for i in range(10)})


def normalize_text(text: str) -> str:
    """Apply the only normalisation permitted on persisted text.

    BOM removal, CRLF/CR -> LF, and Unicode NFC. Persian characters, ZWNJ,
    punctuation, digits and diacritics are all preserved (AGENTS.md 2.1).
    """
    if not text:
        return ""
    if text.startswith(_BOM):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def fold_for_matching(text: str) -> str:
    """Project text onto a comparison-only alphabet.

    Arabic ``ي``/``ى`` become Persian ``ی``, ``ك`` becomes ``ک``, Arabic-Indic
    digits become ASCII, and diacritics, tatweel and zero-width marks are
    dropped. The result is for hashing and similarity only — never stored
    (AGENTS.md 2.3).
    """
    if not text:
        return ""
    text = normalize_text(text)
    text = text.translate(_YEH_FOLD).translate(_KAF_FOLD)
    text = _ARABIC_DIGITS.sub(lambda m: _DIGIT_FOLD.get(m.group(0), m.group(0)), text)
    text = _EXTENDED_ARABIC_DIGITS.sub(
        lambda m: _DIGIT_FOLD.get(m.group(0), m.group(0)), text
    )
    text = _HARAKAT.sub("", text)
    text = text.translate(str.maketrans({c: "" for c in _ZERO_WIDTH + _TATWEEL}))
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def content_hash(*parts: str, fold: bool = True) -> str:
    """SHA-256 over NUL-separated parts.

    Parts are separated by NUL so that ``("ab", "c")`` and ``("a", "bc")`` cannot
    collide. With ``fold=True`` each part is folded for matching first, so two
    spellings of the same Persian text hash identically.
    """
    digest = hashlib.sha256()
    for part in parts:
        piece = fold_for_matching(part) if fold else part
        digest.update(piece.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def short_hash(*parts: str, fold: bool = True, length: int = 16) -> str:
    """Truncated :func:`content_hash`, for use inside identifiers."""
    return content_hash(*parts, fold=fold)[:length]


def stable_id(prefix: str, *parts: str, fold: bool = True, length: int = 16) -> str:
    """A prefixed, deterministic identifier.

    Deterministic across runs and across machines, because it depends only on
    the content passed in — never on iteration order, wall-clock time or a
    process id.
    """
    if not prefix:
        raise ValueError("stable_id requires a non-empty prefix")
    return f"{prefix}-{short_hash(*parts, fold=fold, length=length)}"


def iter_tokens(text: str) -> Iterable[str]:
    """Yield lowercase comparison tokens from folded text.

    ZWNJ is turned into a space *before* folding. Folding deletes it outright,
    so tokenising the folded string would silently glue ``دانش‌بنیان`` into one
    unsearchable token — the exact failure AGENTS.md 2.4 rules out. It separates
    morphemes, not words, but a keyword retriever that cannot see either part is
    worse than one that splits too eagerly.

    A token of length 1 is still yielded: in Persian a one-letter token carries
    real lexical weight that keyword retrieval depends on.
    """
    # Replace first, then fold: the fold removes ZWNJ rather than separating on it.
    spaced = text.replace("‌", " ")
    folded = fold_for_matching(spaced).lower()
    for raw in _SEPARATORS.split(folded):
        token = raw.strip()
        if not token:
            continue
        # A token made only of punctuation is not a word.
        if not _PUNCT_CLASS.sub("", token):
            continue
        yield token


def token_set(text: str) -> frozenset[str]:
    """The distinct comparison tokens of ``text``."""
    return frozenset(iter_tokens(text))


def shingles(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    """Word n-gram shingles. A short sequence yields itself rather than nothing."""
    if size <= 0:
        raise ValueError("shingle size must be positive")
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(left: set, right: set) -> float:
    """Jaccard similarity of two sets. Two empty sets are defined as identical."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams of folded text.

    Used for paraphrase detection: two questions worded differently still share
    most of their character trigrams, which word shingles miss on short strings
    and on a language with rich agglutination.
    """
    folded = normalize_whitespace(fold_for_matching(text).lower())
    if not folded:
        return set()
    if len(folded) < n:
        return {folded}
    return {folded[i : i + n] for i in range(len(folded) - n + 1)}


def char_ngram_jaccard(left: str, right: str, n: int = 3) -> float:
    """Character-trigram Jaccard between two strings."""
    return jaccard(char_ngrams(left, n), char_ngrams(right, n))