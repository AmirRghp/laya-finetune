"""Structured logging.

Library code logs through :func:`get_logger` and never prints (AGENTS.md 5.2).
The CLI configures the root handler once, at startup.

Event ids are stable, machine-greppable strings rather than prose, so a log
file can be filtered without regex gymnastics.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, MutableMapping

__all__ = ["get_logger", "configure_logging", "LOGGER_NAME"]

LOGGER_NAME = "laya_pipeline"

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_CONFIGURED = False


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return the pipeline logger, or a named child of it."""
    if suffix:
        return logging.getLogger(f"{LOGGER_NAME}.{suffix}")
    return logging.getLogger(LOGGER_NAME)


def _render(message: str, fields: MutableMapping[str, Any]) -> str:
    if not fields:
        return message
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            rendered = f"{value:.4g}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return f"{message} | " + " ".join(parts)


class _FieldFormatter(logging.Formatter):
    """Appends structured fields to the message instead of burying them."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, "fields", None)
        if not fields:
            return base
        return f"{base} | " + _render("", dict(fields))


def configure_logging(level: str | os.PathLike[str] | int | None = None) -> None:
    """Install a stderr handler on the pipeline logger. Idempotent."""
    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if _CONFIGURED:
        if level is not None:
            logger.setLevel(_coerce_level(level))
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        _FieldFormatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(_coerce_level(level if level is not None else os.environ.get("LAYA_LOG_LEVEL", "info")))
    _CONFIGURED = True


def _coerce_level(level: str | os.PathLike[str] | int) -> int:
    if isinstance(level, int):
        return level
    name = str(level).strip().lower()
    if name in _LEVELS:
        return _LEVELS[name]
    raise ValueError(
        f"unknown log level {level!r}; use one of {', '.join(sorted(_LEVELS))}"
    )