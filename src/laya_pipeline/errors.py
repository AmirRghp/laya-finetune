"""Typed exceptions for the pipeline.

Every failure the pipeline raises deliberately is defined here, so a caller can
distinguish "the corpus is malformed" from "Ollama returned nonsense" without
matching on message text. Messages always name the offending path, id or value
(AGENTS.md 5.3).
"""

from __future__ import annotations

__all__ = [
    "PipelineError",
    "RetryableError",
    "ConfigError",
    "CorpusError",
    "DiscoveryError",
    "ChunkingError",
    "TokenizerError",
    "OllamaError",
    "OllamaUnavailable",
    "OllamaTimeout",
    "OllamaResponseError",
    "SchemaError",
    "GenerationError",
    "NegativeSamplingError",
    "LabelingError",
    "SplitError",
    "LeakageError",
    "DatasetValidationError",
    "TrainingError",
    "EvaluationError",
]


class PipelineError(Exception):
    """Base class for every deliberate pipeline failure."""


class RetryableError(PipelineError):
    """A failure that is worth retrying: transport, timeout, or a 5xx."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ConfigError(PipelineError):
    """Configuration is missing, malformed, or internally inconsistent."""


class CorpusError(PipelineError):
    """A document under the corpus root could not be read or parsed."""


class DiscoveryError(CorpusError):
    """The corpus root is missing or is not a directory."""


class ChunkingError(PipelineError):
    """A document could not be turned into chunks."""


class TokenizerError(PipelineError):
    """The Laya tokenizer could not be loaded and no fallback was acceptable."""


class OllamaError(RetryableError):
    """Base class for Ollama transport and protocol failures."""


class OllamaUnavailable(OllamaError):
    """The Ollama server could not be reached at all."""


class OllamaTimeout(OllamaError):
    """A request exceeded the configured timeout."""


class OllamaResponseError(OllamaError):
    """Ollama answered, but the body was not the JSON the caller asked for."""


class SchemaError(PipelineError):
    """A model response did not satisfy the schema it was asked to produce.

    Carries the raw response so the caller can log or inspect it without
    re-issuing the request.
    """

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class GenerationError(PipelineError):
    """Question or key-point generation could not complete."""


class NegativeSamplingError(PipelineError):
    """Negative mining produced nothing usable, or was asked for an impossible ratio."""


class LabelingError(PipelineError):
    """Teacher labeling failed in a way that is not a per-item schema error."""


class SplitError(PipelineError):
    """A split could not be produced (too few groups, impossible ratios, ...)."""


class LeakageError(SplitError):
    """Content or a paraphrase appears in more than one split."""


class DatasetValidationError(PipelineError):
    """A generated dataset failed validation.

    ``problems`` holds every issue found rather than only the first, so one run
    reports the full picture instead of one defect per invocation.
    """

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []


class TrainingError(PipelineError):
    """A training run could not be prepared or started."""


class EvaluationError(PipelineError):
    """Evaluation could not be prepared or completed."""