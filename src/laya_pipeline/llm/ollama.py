"""Ollama HTTP client.

One seam for every external model call in the pipeline (AGENTS.md 7.3): tests
substitute a transport and exercise the retry, validation and caching logic
without a server, a network or a GPU.

Endpoints and model names come from configuration. Nothing is hardcoded
(AGENTS.md 5.4).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from ..config import OllamaConfig
from ..errors import (
    OllamaError,
    OllamaResponseError,
    OllamaTimeout,
    OllamaUnavailable,
    SchemaError,
)
from ..ids import content_hash
from ..logging_setup import get_logger
from .cache import Cache, NullCache, cache_key, make_cache
from .schema import ensure_json

__all__ = [
    "OllamaTransport",
    "HttpxTransport",
    "OllamaClient",
    "GenerationResult",
    "Embedder",
    "extract_json",
]

LOGGER = get_logger("llm.ollama")

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    """One structured generation, with everything needed for an audit trail."""

    data: Any
    text: str
    model: str
    prompt_hash: str
    cache_key: str
    attempts: int = 1
    from_cache: bool = False
    elapsed_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None

    def metadata(self) -> Dict[str, Any]:
        """Provenance to store alongside the generated example."""
        return {
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "cache_key": self.cache_key,
            "attempts": self.attempts,
            "from_cache": self.from_cache,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
        }


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class OllamaTransport(Protocol):
    """The single seam every Ollama call goes through.

    Both verbs are part of the seam, and the distinction matters: ``/api/tags``
    and ``/api/version`` only answer ``GET`` and return ``405`` to a ``POST``,
    while ``/api/chat`` and ``/api/embed`` only answer ``POST``. A fake that
    treats them alike will hide a wrong verb exactly the way the original one
    did, so :class:`~tests.conftest.FakeTransport` enforces it too.
    """

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...  # pragma: no cover - protocol

    def get(self, path: str) -> Mapping[str, Any]:
        ...  # pragma: no cover - protocol


#: Ollama endpoints that are ``GET``-only. Posting to one returns ``405``.
GET_ONLY_PATHS: Sequence[str] = ("/api/tags", "/api/version")


class HttpxTransport:
    """Real HTTP transport built on ``httpx``.

    Translates transport-level failures into the pipeline's typed errors so the
    retry policy in :class:`OllamaClient` can distinguish "try again" from
    "this request is wrong".
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float,
        connect_timeout: float,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise OllamaError(
                "httpx is required to talk to Ollama; install it with `pip install -e .`"
            ) from exc
        self._httpx = httpx
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.Client(
            base_url=self.endpoint,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers={"Content-Type": "application/json"},
        )

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        httpx = self._httpx
        try:
            response = self._client.post(path, json=dict(payload))
        except httpx.TimeoutException as exc:
            raise OllamaTimeout(
                f"request to {self.endpoint}{path} timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(
                f"cannot reach Ollama at {self.endpoint}: {exc}"
            ) from exc
        return self._body_of(response, path)

    def get(self, path: str) -> Mapping[str, Any]:
        """``/api/tags`` and ``/api/version`` answer GET only.

        A POST to either returns ``405``, which is why ``is_available`` used to
        report every reachable server as unreachable.
        """
        httpx = self._httpx
        try:
            response = self._client.get(path)
        except httpx.TimeoutException as exc:
            raise OllamaTimeout(
                f"request to {self.endpoint}{path} timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(
                f"cannot reach Ollama at {self.endpoint}: {exc}"
            ) from exc
        return self._body_of(response, path)

    def _body_of(self, response: Any, path: str) -> Mapping[str, Any]:
        if response.status_code >= 500:
            raise OllamaResponseError(
                f"{self.endpoint}{path} returned {response.status_code}: "
                f"{response.text[:300]}",
                retryable=True,
            )
        if response.status_code >= 400:
            raise OllamaResponseError(
                f"{self.endpoint}{path} returned {response.status_code}: "
                f"{response.text[:300]}",
                retryable=False,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                f"{self.endpoint}{path} returned a non-JSON body: {response.text[:300]}",
                retryable=True,
            ) from exc
        if not isinstance(body, dict):
            raise OllamaResponseError(
                f"{self.endpoint}{path} returned a JSON "
                f"{type(body).__name__}, expected an object",
                retryable=True,
            )
        return body

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxTransport":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class OllamaClient:
    """Structured generation and embedding over Ollama."""

    def __init__(
        self,
        config: OllamaConfig,
        *,
        transport: Optional[OllamaTransport] = None,
        cache: Optional[Cache | NullCache] = None,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.config = config
        self._transport = transport or HttpxTransport(
            config.endpoint,
            timeout=config.timeout_seconds,
            connect_timeout=config.connect_timeout_seconds,
        )
        self.cache = cache if cache is not None else make_cache(None, "ollama", False)
        self._sleep = sleep
        self._monotonic = monotonic
        self.request_count = 0
        self.retry_count = 0

    # -- diagnostics -------------------------------------------------------- #
    def is_available(self) -> bool:
        """Reachability probe. ``/api/tags`` is a GET endpoint."""
        try:
            self._transport.get("/api/tags")
            return True
        except OllamaError:
            return False

    def list_models(self) -> List[str]:
        body = self._transport.get("/api/tags")
        models = body.get("models") or []
        names = []
        for entry in models:
            if isinstance(entry, Mapping):
                name = entry.get("name") or entry.get("model")
                if isinstance(name, str):
                    names.append(name)
        return sorted(names)

    # -- generation --------------------------------------------------------- #
    def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        max_retries: Optional[int] = None,
        use_cache: bool = True,
        keep_alive: Optional[str] = None,
    ) -> GenerationResult:
        """Ask for a JSON object, validate it, retry on a malformed answer.

        The cache is consulted first and is keyed on everything that affects the
        answer, so a resumed run never re-generates an example it already has.
        """
        chosen_model = model or self.config.generation_model
        effective_temperature = 0.7 if temperature is None else float(temperature)
        effective_top_p = 0.9 if top_p is None else float(top_p)
        effective_seed = seed
        attempts_limit = self.config.max_retries if max_retries is None else max_retries

        key = cache_key(
            endpoint=self.config.endpoint,
            model=chosen_model,
            prompt=prompt,
            system=system,
            temperature=effective_temperature,
            top_p=effective_top_p,
            seed=effective_seed,
            schema=schema,
        )

        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return GenerationResult(
                    data=cached.get("data"),
                    text=str(cached.get("text", "")),
                    model=chosen_model,
                    prompt_hash=content_hash(prompt, fold=False),
                    cache_key=key,
                    attempts=int(cached.get("attempts", 1)),
                    from_cache=True,
                    prompt_tokens=cached.get("prompt_tokens"),
                    eval_tokens=cached.get("eval_tokens"),
                )

        payload: Dict[str, Any] = {
            "model": chosen_model,
            "stream": False,
            "messages": self._messages(prompt, system),
            "format": dict(schema) if schema is not None else "json",
            "keep_alive": keep_alive or self.config.keep_alive,
            "options": {
                "temperature": effective_temperature,
                "top_p": effective_top_p,
                "num_ctx": self.config.num_ctx,
            },
        }
        if effective_seed is not None:
            payload["options"]["seed"] = int(effective_seed)
        if self.config.think is not None:
            payload["think"] = self.config.think

        last_error: Optional[Exception] = None
        last_text = ""
        started = self._monotonic()

        for attempt in range(1, max(1, attempts_limit) + 1):
            try:
                self.request_count += 1
                body = self._transport.post("/api/chat", payload)
                text = self._content_of(body)
                last_text = text
                data = ensure_json(
                    extract_json(text),
                    schema or {"type": "object"},
                    raw=text,
                    context=f"{chosen_model} response",
                )
                elapsed_ms = (self._monotonic() - started) * 1000.0
                result = GenerationResult(
                    data=data,
                    text=text,
                    model=chosen_model,
                    prompt_hash=content_hash(prompt, fold=False),
                    cache_key=key,
                    attempts=attempt,
                    from_cache=False,
                    elapsed_ms=elapsed_ms,
                    prompt_tokens=_int_or_none(body.get("prompt_eval_count")),
                    eval_tokens=_int_or_none(body.get("eval_count")),
                )
                if use_cache:
                    self.cache.set(
                        key,
                        {
                            "data": result.data,
                            "text": result.text,
                            "attempts": attempt,
                            "prompt_tokens": result.prompt_tokens,
                            "eval_tokens": result.eval_tokens,
                        },
                    )
                return result
            except (OllamaUnavailable, OllamaTimeout) as exc:
                last_error = exc
            except OllamaResponseError as exc:
                last_error = exc
                if not getattr(exc, "retryable", True):
                    break
            except SchemaError as exc:
                # A well-formed HTTP response that failed the schema: worth one
                # more attempt, since a retry at a different temperature often
                # produces a conforming object.
                last_error = exc
                last_text = exc.raw or last_text

            if attempt < max(1, attempts_limit):
                self.retry_count += 1
                backoff = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "ollama call failed, retrying",
                    extra={
                        "fields": {
                            "model": chosen_model,
                            "attempt": attempt,
                            "backoff_s": round(backoff, 2),
                            "error": str(last_error),
                        }
                    },
                )
                self._sleep(backoff)

        detail = f" (last response: {last_text[:200]!r})" if last_text else ""
        raise OllamaError(
            f"{chosen_model} did not return valid JSON after "
            f"{max(1, attempts_limit)} attempt(s): {last_error}{detail}"
        ) from last_error

    def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ) -> str:
        """Free-text generation, used only where the answer is not structured."""
        chosen_model = model or self.config.generation_model
        effective_temperature = 0.7 if temperature is None else float(temperature)
        payload: Dict[str, Any] = {
            "model": chosen_model,
            "stream": False,
            "messages": self._messages(prompt, system),
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": effective_temperature,
                "num_ctx": self.config.num_ctx,
                **({"seed": int(seed)} if seed is not None else {}),
            },
        }
        if self.config.think is not None:
            payload["think"] = self.config.think
        body = self._transport.post("/api/generate", payload)
        return self._content_of(body)

    # -- embeddings --------------------------------------------------------- #
    def embed(
        self, texts: Sequence[str], *, model: Optional[str] = None
    ) -> "Any":
        """Embed texts, returning a ``numpy`` array of shape ``(n, dim)``.

        Tries the batch ``/api/embed`` endpoint first and falls back to the
        single-text ``/api/embeddings`` endpoint, so the pipeline works against
        both current and older Ollama builds.
        """
        import numpy as np

        chosen_model = model or self.config.embedding_model
        rows = ["" if text is None else str(text) for text in texts]
        if not rows:
            return np.zeros((0, 0), dtype="float32")

        vectors: List[List[float]]
        try:
            vectors = self._embed_batch(rows, chosen_model)
        except OllamaResponseError as exc:
            if " 404" not in str(exc) and "404" not in str(exc):
                raise
            LOGGER.info(
                "/api/embed unavailable, falling back to /api/embeddings",
                extra={"fields": {"model": chosen_model}},
            )
            vectors = [self._embed_one(row, chosen_model) for row in rows]

        width = len(vectors[0])
        for index, vector in enumerate(vectors):
            if len(vector) != width:
                raise OllamaResponseError(
                    f"{chosen_model} returned embedding {index} with "
                    f"{len(vector)} dimensions but the first had {width}; "
                    "a model change mid-batch corrupts the index"
                )
        return np.asarray(vectors, dtype="float32")

    def _embed_batch(self, rows: Sequence[str], model: str) -> List[List[float]]:
        body = self._transport.post(
            "/api/embed", {"model": model, "input": list(rows), "keep_alive": self.config.keep_alive}
        )
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise OllamaResponseError(
                f"/api/embed returned no embeddings for model {model}", retryable=False
            )
        return [_as_vector(item) for item in embeddings]

    def _embed_one(self, row: str, model: str) -> List[float]:
        body = self._transport.post(
            "/api/embeddings", {"model": model, "prompt": row, "keep_alive": self.config.keep_alive}
        )
        return _as_vector(body.get("embedding"))

    # -- internals ---------------------------------------------------------- #
    def _messages(self, prompt: str, system: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _content_of(body: Mapping[str, Any]) -> str:
        """Pull the assistant text out of an Ollama reply.

        An empty string is *not* treated as success. Measured against a Qwen3.5
        model on this project, ``think: true`` with a small ``num_ctx`` made the
        model spend its entire budget on a chain of thought and return
        ``content: ""`` with ``done_reason: "length"``. Reporting that as an
        empty JSON object costs two retries and ~200 seconds before anyone sees
        the real cause, so it is named here instead.
        """
        message = body.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            content = message["content"]
            if content.strip():
                return content
            raise OllamaResponseError(
                _empty_content_hint(body), retryable=_is_retryable_empty(body)
            )
        if isinstance(body.get("response"), str) and body["response"].strip():
            return body["response"]
        raise OllamaResponseError(
            "Ollama response has neither 'message.content' nor 'response': "
            f"{sorted(body)}",
            retryable=True,
        )

    def stats(self) -> Dict[str, Any]:
        cache_stats = self.cache.stats() if isinstance(self.cache, Cache) else {}
        return {
            "requests": self.request_count,
            "retries": self.retry_count,
            **cache_stats,
        }


class Embedder:
    """The ``embed_fn(texts) -> ndarray`` contract the retrieval layer consumes.

    Wraps a client so that ``retrieval.py`` never depends on Ollama directly and
    a test can pass a plain callable instead.
    """

    def __init__(self, client: OllamaClient, model: Optional[str] = None) -> None:
        self.client = client
        self.model = model or client.config.embedding_model
        self.calls = 0
        self.texts = 0

    def __call__(self, texts: Sequence[str]) -> Any:
        self.calls += 1
        self.texts += len(texts)
        return self.client.embed(texts, model=self.model)

    @property
    def id(self) -> str:
        """Identity of the embedding model, recorded in retrieval reports."""
        return f"ollama:{self.model}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _empty_content_hint(body: Mapping[str, Any]) -> str:
    """Explain an empty ``message.content`` instead of just reporting it.

    Two distinct faults produce an empty answer and they have opposite fixes:

    * the model reasoned its whole budget away -- ``think: false`` or a larger
      ``num_ctx``;
    * the context window was exhausted before generation began -- raise
      ``num_ctx`` or shorten the chunk.

    Only what the response actually shows is asserted; nothing is guessed.
    """
    message = body.get("message")
    thinking = ""
    if isinstance(message, Mapping) and isinstance(message.get("thinking"), str):
        thinking = message["thinking"]
    reason = body.get("done_reason")
    produced = body.get("eval_count")

    if thinking and reason == "length":
        return (
            "model returned an empty answer: it spent the whole context window "
            f"reasoning ({produced} tokens, done_reason='length') and never "
            "produced content. Send think: false for structured extraction, or "
            "raise ollama.num_ctx."
        )
    if thinking:
        return (
            "model returned an empty answer while its reasoning was still in "
            f"message.thinking (done_reason={reason!r}). Send think: false so "
            "the answer is not consumed before it is produced."
        )
    if reason == "length":
        return (
            "model returned an empty answer with done_reason='length' and no "
            f"thinking output ({produced} tokens): the context window was "
            "exhausted before generation. Raise ollama.num_ctx or shorten the "
            "chunk."
        )
    return (
        f"model returned an empty answer (done_reason={reason!r}). A reasoning "
        "model with thinking enabled is the usual cause; send think: false."
    )


def _is_retryable_empty(body: Mapping[str, Any]) -> bool:
    """An empty answer caused by thinking on will be empty again: do not retry.

    Each retry costs about 100 seconds and cannot succeed, because the same
    prompt against the same reasoning budget produces the same exhaustion. A
    genuinely transient stall is the exception, and Ollama signals that by
    leaving ``done_reason`` unset.
    """
    message = body.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("thinking"), str):
        return False
    return True


def extract_json(text: str) -> Any:
    """Parse JSON out of a model response, tolerating fences and prose.

    Raises :class:`OllamaResponseError` when no JSON object can be recovered,
    naming the text so the failure is debuggable without re-running the request.
    """
    if text is None:
        raise OllamaResponseError("model returned an empty response", retryable=True)
    candidate = text.strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass

    # Fall back to the outermost brace-balanced span.
    start = candidate.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : index + 1])
                    except ValueError:
                        break
    raise OllamaResponseError(
        f"model response contains no JSON object: {text[:300]!r}", retryable=True
    )


def _as_vector(value: Any) -> List[float]:
    if isinstance(value, Mapping):
        value = value.get("values")
    if not isinstance(value, (list, tuple)) or not value:
        raise OllamaResponseError(
            f"embedding is not a non-empty array, got {type(value).__name__}", retryable=False
        )
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise OllamaResponseError(f"embedding contains a non-numeric value: {exc}") from exc


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None