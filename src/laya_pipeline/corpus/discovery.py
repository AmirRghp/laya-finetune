"""Recursive, deterministic Markdown discovery.

Rules this module exists to enforce (AGENTS.md 1.5-1.8):

* recursive over ``*.md`` and ``*.markdown``, case-insensitive;
* a stable order, so two runs over an unchanged corpus agree exactly;
* **no filename assumptions at all** — nothing here reads a name, a prefix, a
  numbering scheme or a path depth to decide anything. Filenames are used only
  as an opaque sort key and as provenance.
* the corpus is opened read-only and is never a write target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from ..errors import DiscoveryError
from ..ids import normalize_text, stable_id
from ..io import read_text, relative_posix
from ..logging_setup import get_logger
from .models import DocumentSource

__all__ = ["MARKDOWN_SUFFIXES", "discover_documents", "group_by_content"]

LOGGER = get_logger("corpus.discovery")

#: Recognised Markdown extensions, compared case-insensitively.
MARKDOWN_SUFFIXES = (".md", ".markdown")


def discover_documents(
    corpus_dir: Path,
    *,
    include_suffixes: Sequence[str] = MARKDOWN_SUFFIXES,
    exclude_dirs: Optional[Iterable[str]] = None,
) -> List[DocumentSource]:
    """Return every Markdown document under ``corpus_dir``, in a stable order.

    Parameters
    ----------
    corpus_dir:
        Root of the corpus. Must exist and be a directory.
    include_suffixes:
        Extensions to accept, lower-cased for comparison.
    exclude_dirs:
        Directory *names* to skip anywhere in the tree. Operates on names, never
        on path patterns, so it cannot accidentally encode a naming convention.

    Raises
    ------
    DiscoveryError
        If the corpus root is missing or is not a directory.
    """
    root = Path(corpus_dir).expanduser()
    if not root.exists():
        raise DiscoveryError(
            f"corpus directory does not exist: {root}. Create it and add Markdown "
            "documents, or point paths.corpus_dir at the right location."
        )
    if not root.is_dir():
        raise DiscoveryError(f"corpus_dir is not a directory: {root}")

    suffixes = {s.lower() for s in include_suffixes}
    unknown = suffixes - set(MARKDOWN_SUFFIXES)
    if unknown:
        raise DiscoveryError(
            f"unsupported document suffix(es): {', '.join(sorted(unknown))}; "
            f"supported: {', '.join(MARKDOWN_SUFFIXES)}"
        )

    skip: Set[str] = set(exclude_dirs or ())
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if skip and any(part in skip for part in path.relative_to(root).parts):
            continue
        candidates.append(path)

    # Deterministic order: POSIX relative path, compared by code point. Sorting
    # Path objects would order on the OS's collation, which varies.
    candidates.sort(key=lambda p: relative_posix(p, root))

    documents: List[DocumentSource] = []
    for path in candidates:
        relpath = relative_posix(path, root)
        try:
            text = normalize_text(read_text(path))
        except Exception as exc:
            raise DiscoveryError(f"cannot read corpus document {relpath}: {exc}") from exc
        content_id = stable_id("cnt", text)
        documents.append(
            DocumentSource(
                path=path,
                relpath=relpath,
                content_id=content_id,
                byte_size=path.stat().st_size,
                text=text,
            )
        )

    LOGGER.info(
        "discovered documents",
        extra={
            "fields": {
                "root": str(root),
                "documents": len(documents),
                "unique_content": len({d.content_id for d in documents}),
            }
        },
    )
    return documents


def group_by_content(documents: Sequence[DocumentSource]) -> dict[str, List[str]]:
    """Map content id to the relative paths carrying that content.

    A non-empty bucket with more than one path is a duplicate document under two
    names (AGENTS.md 1.8 — a renamed copy must not silently double the corpus).
    """
    grouped: dict[str, List[str]] = {}
    for document in documents:
        grouped.setdefault(document.content_id, []).append(document.relpath)
    return {key: sorted(paths) for key, paths in grouped.items()}


def find_duplicates(documents: Sequence[DocumentSource]) -> dict[str, List[str]]:
    """Only the content ids that appear under more than one path."""
    grouped = group_by_content(documents)
    return {key: paths for key, paths in grouped.items() if len(paths) > 1}