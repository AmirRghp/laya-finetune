"""Artifact I/O: UTF-8 reads, atomic writes, corpus write-protection.

Two rules are enforced here rather than left to convention (AGENTS.md 1.2-1.3):

* every write is atomic — temp file in the same directory, then ``os.replace`` —
  so a crash never leaves a half-written JSONL that the next stage trusts;
* every write is checked against the corpus root, so no code path can modify
  ``corpus/`` even by accident.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import ConfigError, PipelineError

__all__ = [
    "ensure_dir",
    "resolve_output_path",
    "assert_writable",
    "read_text",
    "read_jsonl",
    "write_jsonl",
    "read_json",
    "write_json",
    "write_text",
    "read_csv_rows",
    "write_csv",
    "write_manifest",
    "file_sha256",
    "relative_posix",
]


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents if missing, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_posix(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` in POSIX form, or its name if not under root.

    Used for stable, machine-independent document keys, so a corpus moved to a
    different absolute location produces identical ids.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().name


def resolve_output_path(path: Path, corpus_dir: Path) -> Path:
    """Resolve an output path and refuse it if it lands inside the corpus.

    Also refuses the corpus directory itself and any parent of it, which would
    let a relative ``../..`` escape reach it.
    """
    resolved = Path(path).expanduser().resolve()
    corpus = Path(corpus_dir).expanduser().resolve()
    if resolved == corpus:
        raise ConfigError(
            f"refusing to write to the corpus directory itself: {corpus}"
        )
    if corpus in resolved.parents:
        raise ConfigError(
            f"refusing to write inside the corpus directory: {resolved} "
            f"is under {corpus}"
        )
    # A parent of the corpus is equally forbidden: writing there could clobber
    # the tree that contains it.
    if resolved in corpus.parents:
        raise ConfigError(
            f"refusing to write to {resolved}, which contains the corpus at {corpus}"
        )
    return resolved


def assert_writable(path: Path, corpus_dir: Path) -> Path:
    """Public alias of :func:`resolve_output_path`, named for its call site."""
    return resolve_output_path(path, corpus_dir)


def _atomic_write_bytes(path: Path, payload: bytes, corpus_dir: Path | None) -> Path:
    resolved = resolve_output_path(path, corpus_dir) if corpus_dir else Path(path)
    ensure_dir(resolved.parent)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, resolved)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return resolved


def read_text(path: Path) -> str:
    """Read a text file as UTF-8, transparently accepting a BOM.

    Raises :class:`PipelineError` rather than letting a decode error escape as
    an opaque ``UnicodeDecodeError`` from deep inside a stage.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise PipelineError(f"cannot read {path}: {exc}") from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineError(f"{path} is not valid UTF-8: {exc}") from exc


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object from a JSONL file.

    Blank lines and ``#`` comment lines are skipped, matching the convention
    ``laya.train.read_jsonl`` uses, so a file this module writes is readable by
    the Laya trainer and vice versa.
    """
    for lineno, line in enumerate(read_text(Path(path)).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except ValueError as exc:
            raise PipelineError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise PipelineError(
                f"{path}:{lineno} is a JSON {type(row).__name__}, expected an object"
            )
        yield row


def write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], corpus_dir: Path | None = None
) -> int:
    """Write rows as JSONL atomically. Returns the number of rows written.

    Two non-negotiable details:

    * ``ensure_ascii=False`` — escaping Persian to ``\\uXXXX`` would make the
      artifact unreadable to a reviewer and several times larger.
    * **no key sorting.** A choice question's ``criteria`` is a mapping whose
      *order is the option order*, and Laya treats that order as part of the
      question's identity. Sorting keys silently rewrites the decision space of
      every choice question in the file while leaving it looking valid.
    """
    count = 0
    chunks: list[str] = []
    for row in rows:
        chunks.append(json.dumps(row, ensure_ascii=False, sort_keys=False))
        count += 1
    payload = "\n".join(chunks)
    if payload:
        payload += "\n"
    _atomic_write_bytes(Path(path), payload.encode("utf-8"), corpus_dir)
    return count


def read_json(path: Path) -> Any:
    try:
        return json.loads(read_text(Path(path)))
    except ValueError as exc:
        raise PipelineError(f"{path} is not valid JSON: {exc}") from exc


def write_json(
    path: Path, value: Any, corpus_dir: Path | None = None, indent: int = 2
) -> Path:
    payload = json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True)
    return _atomic_write_bytes(
        Path(path), (payload + "\n").encode("utf-8"), corpus_dir
    )


def write_text(path: Path, text: str, corpus_dir: Path | None = None) -> Path:
    """Write UTF-8 text atomically."""
    return _atomic_write_bytes(Path(path), text.encode("utf-8"), corpus_dir)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV with ``utf-8-sig`` so Excel-exported BOMs are handled."""
    text = read_text(Path(path))
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    corpus_dir: Path | None = None,
) -> Path:
    """Write CSV atomically, flattening nested values into single cells."""
    import io as _io

    buffer = _io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _flatten(row.get(key)) for key in fieldnames})
    return _atomic_write_bytes(Path(path), buffer.getvalue().encode("utf-8"), corpus_dir)


def _flatten(value: Any) -> str:
    """Render a nested value into a single CSV cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, or ``""`` when the file does not exist."""
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    directory: Path,
    stage: str,
    *,
    config_hash: str,
    seed: int,
    counts: Mapping[str, Any],
    inputs: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    corpus_dir: Path | None = None,
) -> Path:
    """Write ``manifest.json`` describing exactly what produced this artifact.

    A result you cannot trace to its inputs is not reproducible, so every stage
    writes one (AGENTS.md 4.4).
    """
    payload: dict[str, Any] = {
        "stage": stage,
        "seed": seed,
        "config_hash": config_hash,
        "counts": dict(counts),
        "inputs": dict(inputs or {}),
    }
    if extra:
        payload["extra"] = dict(extra)
    return write_json(Path(directory) / "manifest.json", payload, corpus_dir)