"""LLM access: prompt templates, response schemas, caching, Ollama client."""

from __future__ import annotations

from .cache import Cache, NullCache, cache_key, make_cache
from .ollama import (
    Embedder,
    GenerationResult,
    HttpxTransport,
    OllamaClient,
    OllamaTransport,
    extract_json,
)
from .prompts import PromptLibrary, PromptTemplate, placeholders_in
from .schema import array_schema, ensure_json, number_schema, object_schema, string_schema, validate_json

__all__ = [
    "Cache",
    "Embedder",
    "GenerationResult",
    "HttpxTransport",
    "NullCache",
    "OllamaClient",
    "OllamaTransport",
    "PromptLibrary",
    "PromptTemplate",
    "array_schema",
    "cache_key",
    "ensure_json",
    "extract_json",
    "make_cache",
    "number_schema",
    "object_schema",
    "placeholders_in",
    "string_schema",
    "validate_json",
]