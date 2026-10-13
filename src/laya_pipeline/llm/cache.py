"""Content-addressed cache for model calls.

Keyed by everything that can change the answer — endpoint, model, full prompt,
system prompt, temperature, seed, top_p and the response schema — so a cache
hit is indistinguishable from a fresh call (AGENTS.md 4.5). A cache entry for a
different model or a different temperature is a different entry, never a
substitute.

Entries are written atomically. A partially written cache file is discarded on
read rather than trusted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..errors import PipelineError
from ..ids import content_hash, stable_id
from ..io import read_text
from ..logging_setup import get_logger

__all__ = ["Cache", "NullCache", "cache_key"]

LOGGER = get_logger("llm.cache")


def cache_key(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    system: Optional[str],
    temperature: float,
    top_p: float,
    seed: Optional[int],
    schema: Optional[Mapping[str, Any]],
    kind: str = "generate",
) -> str:
    """A stable key covering every input that can change the output."""
    payload = json.dumps(
        {
            "kind": kind,
            "endpoint": endpoint.rstrip("/"),
            "model": model,
            "prompt": prompt,
            "system": system,
            "temperature": round(float(temperature), 6),
            "top_p": round(float(top_p), 6),
            "seed": seed,
            "schema": schema,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return content_hash(payload, fold=False)


class NullCache:
    """A cache that stores nothing. Used when caching is disabled."""

    def get(self, key: str) -> Optional[Any]:  # noqa: D102
        return None

    def set(self, key: str, value: Any) -> None:  # noqa: D102
        return None

    def path_for(self, key: str) -> Optional[Path]:  # noqa: D102
        return None


class Cache:
    """A directory of one-JSON-file-per-entry.

    Sharded two levels deep by the first four hex characters, so a corpus with
    tens of thousands of generation calls does not put a hundred thousand
    entries in one directory.
    """

    def __init__(self, root: Path, namespace: str = "default") -> None:
        self.root = Path(root).expanduser()
        self.namespace = namespace
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        shard = self.root / self.namespace / key[:4]
        return shard / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(read_text(path))
        except (PipelineError, ValueError):
            # Corrupt entry: drop it and treat the call as a miss. Serving a
            # half-written body would be worse than re-asking the model.
            LOGGER.warning(
                "discarding unreadable cache entry",
                extra={"fields": {"path": str(path), "key": key}},
            )
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        if not isinstance(payload, dict) or payload.get("key") != key:
            self.misses += 1
            return None
        self.hits += 1
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
            )
            tmp = Path(tmp_name)
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    {"key": key, "namespace": self.namespace, "value": value},
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            # A cache is an optimisation. Failing to write one must never fail
            # the run, so this is logged and swallowed deliberately.
            LOGGER.warning(
                "could not write cache entry",
                extra={"fields": {"path": str(path), "error": str(exc)}},
            )

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def entry_id(self, key: str) -> str:
        """Stable short id for a cache entry, for provenance in artifacts."""
        return stable_id("cache", self.namespace, key)


def make_cache(root: Optional[Path], namespace: str, enabled: bool = True) -> Cache | NullCache:
    """Build a cache, or a no-op one when disabled."""
    if not enabled or root is None:
        return NullCache()
    return Cache(Path(root), namespace)