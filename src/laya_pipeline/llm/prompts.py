"""Editable prompt templates.

Templates live as files under ``paths.prompt_dir`` so that wording can be changed
without touching code. Placeholders use ``{{name}}`` rather than ``{name}`` so
that JSON examples inside a template need no escaping — the format syntax
Laya and JSON Schema both rely on.

Every render is checked in both directions: a missing value raises, and a value
that the template does not use raises too. A silently ignored argument usually
means the prompt was not updated the way the caller thinks it was.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from ..errors import ConfigError
from ..ids import content_hash
from ..io import read_text
from ..logging_setup import get_logger

__all__ = ["PromptTemplate", "PromptLibrary", "PLACEHOLDER_PATTERN"]

LOGGER = get_logger("llm.prompts")

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class PromptTemplate:
    """One prompt file, with its placeholders known up front."""

    __slots__ = ("name", "text", "path", "placeholders")

    def __init__(self, name: str, text: str, path: Optional[Path] = None) -> None:
        self.name = name
        self.text = text
        self.path = path
        self.placeholders: FrozenSet[str] = frozenset(PLACEHOLDER_PATTERN.findall(text))

    def render(self, **values: Any) -> str:
        missing = sorted(self.placeholders - set(values))
        if missing:
            raise ConfigError(
                f"prompt {self.name!r} needs values for: {', '.join(missing)}"
            )
        unused = sorted(set(values) - self.placeholders)
        if unused:
            raise ConfigError(
                f"prompt {self.name!r} was given values it does not use: "
                f"{', '.join(unused)}; placeholders are: "
                f"{', '.join(sorted(self.placeholders)) or '(none)'}"
            )
        return PLACEHOLDER_PATTERN.sub(
            lambda match: str(values[match.group(1)]), self.text
        )

    @property
    def fingerprint(self) -> str:
        """Hash of the template text, recorded in artifacts for provenance."""
        return content_hash(self.text, fold=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PromptTemplate(name={self.name!r}, placeholders={sorted(self.placeholders)})"


class PromptLibrary:
    """Loads every ``*.md`` and ``*.txt`` file in a directory as a template.

    Templates are loaded lazily on first use so that a typo in one filename is
    reported when that template is needed, with the caller visible, rather than
    at CLI start-up where the traceback names nothing useful.
    """

    SUFFIXES = (".md", ".txt", ".prompt")

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser()
        self._templates: Dict[str, PromptTemplate] = {}

    def exists(self) -> bool:
        return self.directory.is_dir()

    def available(self) -> List[str]:
        """Names loadable right now, sorted. Empty when the directory is absent."""
        if not self.exists():
            return []
        return sorted(
            path.stem
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUFFIXES
        )

    def names(self) -> List[str]:
        return self.available()

    def _locales(self) -> List[str]:
        """Locale tags to try, most specific first. Fixed order, no config."""
        configured = os.environ.get("LAYA_PROMPT_LOCALE", "").strip()
        locales = ["fa"] + [part for part in ("en",) if part != "fa"]
        if configured:
            locales.insert(0, configured)
        return locales

    def template(self, name: str) -> PromptTemplate:
        cached = self._templates.get(name)
        if cached is not None:
            return cached
        if not self.exists():
            raise ConfigError(
                f"prompt directory {self.directory} does not exist; prompts.dir must "
                "point at the repository's prompts/ directory"
            )
        suffixes = " or ".join(f"*{s}" for s in self.SUFFIXES)
        # Try the plain name first, then locale-suffixed ones: `questions` renders
        # questions.md, and questions.fa renders questions.fa.md. The locale is a
        # naming convenience, never a behavioural dependency — nothing reads it.
        candidates = [f"{name}{suffix}" for suffix in self.SUFFIXES]
        for locale in self._locales():
            candidates.extend(f"{name}.{locale}{suffix}" for suffix in self.SUFFIXES)

        for candidate_name in candidates:
            candidate = self.directory / candidate_name
            if candidate.is_file():
                return self._load(name, candidate)

        raise ConfigError(
            f"no prompt template named {name!r} in {self.directory} "
            f"(looked for {suffixes}, plain or locale-suffixed); "
            f"available: {', '.join(self.available()) or '(none)'}"
        )

    def _load(self, name: str, path: Path) -> PromptTemplate:
        template = PromptTemplate(name=name, text=read_text(path), path=path)
        self._templates[name] = template
        return template

    def render(self, name: str, **values: Any) -> str:
        return self.template(name).render(**values)

    def fingerprint(self, name: str) -> str:
        return self.template(name).fingerprint

    def fingerprints(self) -> Mapping[str, str]:
        return {name: self.fingerprint(name) for name in self.available()}


def placeholders_in(text: str) -> List[str]:
    """Distinct ``{{name}}`` placeholders in ``text``, in first-appearance order."""
    seen: List[str] = []
    for match in PLACEHOLDER_PATTERN.finditer(text):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen