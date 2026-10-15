"""Token counting.

Chunk sizing must be expressed in the tokens the model will actually see, not
in characters. The Laya Multilingual checkpoint wraps mmBERT, whose vocabulary
is large and which tokenises Persian densely — a character estimate is only a
fallback for when the tokenizer cannot be loaded.

The Hugging Face tokenizer is imported lazily and never at module import time,
so ``--help`` and every stage that does not need token counts work without
``transformers`` or ``torch`` installed (AGENTS.md 5.7).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .config import ChunkingConfig
from .errors import TokenizerError
from .ids import fold_for_matching, normalize_whitespace
from .logging_setup import get_logger

__all__ = [
    "TokenCounter",
    "EstimateTokenCounter",
    "HuggingFaceTokenCounter",
    "resolve_token_counter",
    "reset_token_counter_cache",
    "estimate_tokens",
]

LOGGER = get_logger("tokenizer")


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can size a string in model tokens."""

    source: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Character-based fallback estimate.

    Persian script packs fewer characters per token than English, so the
    divisor is deliberately smaller than the English figure of ~4. The result
    is recorded as ``token_count_source: "estimated"`` and never presented as a
    measurement.
    """
    if not text:
        return 0
    normalized = normalize_whitespace(text)
    if not normalized:
        return 0
    return max(1, int(len(normalized) / chars_per_token) + 1)


class EstimateTokenCounter:
    """Character-based counter used when the real tokenizer is unavailable."""

    def __init__(self, chars_per_token: float = 2.6) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = float(chars_per_token)
        self.source = "estimated"

    def count(self, text: str) -> int:
        return estimate_tokens(text, self.chars_per_token)

    def count_many(self, texts: Sequence[str]) -> List[int]:
        return [self.count(text) for text in texts]


class HuggingFaceTokenCounter:
    """Counts with the checkpoint's own tokenizer.

    Prefers a local checkpoint's ``tokenizer/`` directory so that an already
    downloaded checkpoint is never re-fetched.
    """

    def __init__(self, tokenizer: object, source: str) -> None:
        self._tokenizer = tokenizer
        self.source = source

    @property
    def tokenizer(self) -> object:
        return self._tokenizer

    def encode(self, text: str) -> List[int]:
        encoded = self._tokenizer(  # type: ignore[operator]
            text, add_special_tokens=False, truncation=False
        )
        return list(encoded["input_ids"])

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))

    def count_many(self, texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        encoded = self._tokenizer(  # type: ignore[operator]
            list(texts), add_special_tokens=False, truncation=False
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def fits(self, text: str, budget: int) -> bool:
        """Whether ``text`` is at most ``budget`` tokens. Never over-allocates."""
        return self.count(text) <= budget


_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}


def _load_huggingface_counter(
    model: str, max_length: Optional[int], allow_download: bool
) -> Optional[HuggingFaceTokenCounter]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerError(
            "transformers is not installed, so the real Laya tokenizer cannot be "
            "loaded. Install the optional extra with `pip install -e .[laya]`, or "
            "run this stage with an estimated counter."
        ) from exc

    candidates: List[tuple[str, Optional[str]]] = []
    local = Path(model).expanduser()
    if local.is_dir():
        nested = local / "tokenizer"
        candidates.append((str(nested if nested.is_dir() else local), None))
    else:
        # Laya's Hub repositories keep tokenizer.json and tokenizer_config.json
        # in a `tokenizer/` subfolder while model config/weights stay at the root.
        candidates.append((model, "tokenizer"))
        candidates.append((model, None))

    last_error: Optional[Exception] = None
    for candidate, subfolder in candidates:
        try:
            kwargs: Dict[str, Any] = {"local_files_only": not allow_download}
            if subfolder is not None:
                kwargs["subfolder"] = subfolder
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                **kwargs,
            )
        except Exception as exc:  # transformers raises many concrete types
            last_error = exc
            continue
        if max_length is not None:
            # Keep truncation off: we want to know the true length, and Laya
            # truncates for us at the state budget.
            try:
                tokenizer.model_max_length = int(max_length)
            except (AttributeError, TypeError, ValueError):
                pass
        source = f"{candidate}/{subfolder}" if subfolder is not None else candidate
        return HuggingFaceTokenCounter(tokenizer, source=f"huggingface:{source}")

    raise TokenizerError(
        f"could not load a tokenizer for {model!r}: {last_error}. Set "
        "chunking.tokenizer_model to a resolvable id or local checkpoint, or pass "
        "allow_estimates=True to proceed with character estimates."
    )


def resolve_token_counter(
    config: ChunkingConfig,
    *,
    allow_download: bool = True,
    allow_estimates: bool = True,
) -> TokenCounter:
    """Return the best available counter, falling back to an estimate.

    The fallback is logged as a warning, not silently applied: a run whose chunk
    sizes were estimated must not be mistaken for one that measured tokens
    (AGENTS.md 3.14).
    """
    key = f"{config.tokenizer_model}|{config.tokenizer_max_length}|{allow_download}"
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

    counter: TokenCounter
    try:
        counter = _load_huggingface_counter(
            config.tokenizer_model, config.tokenizer_max_length, allow_download
        )
    except TokenizerError as exc:
        if not allow_estimates:
            raise
        LOGGER.warning(
            "falling back to character-estimated token counts",
            extra={
                "fields": {
                    "model": config.tokenizer_model,
                    "reason": str(exc),
                    "chars_per_token": config.chars_per_token_estimate,
                }
            },
        )
        counter = EstimateTokenCounter(config.chars_per_token_estimate)

    with _LOCK:
        _CACHE[key] = counter
    return counter


def reset_token_counter_cache() -> None:
    """Drop the cached counter. Used by tests and after a config change."""
    with _LOCK:
        _CACHE.clear()


def fingerprint_text(text: str) -> str:
    """A folded, whitespace-collapsed form for template/state fingerprints.

    Used to compare a state rendered at build time against one rendered at
    evaluation time without being confused by incidental whitespace.
    """
    return normalize_whitespace(fold_for_matching(text)).lower()
