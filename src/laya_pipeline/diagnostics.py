"""Preflight checks: will this configuration actually work before we spend time?

Every finding here corresponds to a failure that otherwise shows up *after* the
expensive part has started. A model that is not pulled does not fail on
download; it fails on the first generation call, inside a retry loop, roughly a
hundred seconds per attempt. On a corpus of a few hundred chunks that is a very
long time to wait to be told a name was misspelled.

The checks are pure functions over the resolved configuration plus a list of
model names, so the whole module is testable without a network (AGENTS.md 7.2).
Only :func:`fetch_pulled_models` talks to Ollama, and it takes an injectable
transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence

from .errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config


#: Model families known to emit a chain of thought before answering.
#:
#: Not a hard rule -- Ollama adds new reasoning models constantly -- but a
#: heuristic that only ever costs a warning is worth having.
REASONING_FAMILY_HINTS: Sequence[str] = (
    "qwen3",
    "deepseek-r1",
    "deepseek",
    "magistral",
    "phi4-reasoning",
    "marco-o1",
    "exaone-deep",
    "smallthinker",
)

#: Below this, a Persian chunk plus a JSON answer plus an instruction will not
#: fit, and the model will report a length-exhausted reply.
MIN_REASONABLE_NUM_CTX: int = 4096

#: The largest embedding model this pipeline has any use for. Larger ones cost
#: time and buy nothing here: embeddings only rank candidate negatives.
SANE_EMBEDDING_UPPER_BOUND = 3_000_000_000


@dataclass(frozen=True)
class Finding:
    """One preflight result.

    ``level`` is ``error`` (the run cannot work), ``warning`` (it can work, but
    expect worse output or a slower run), or ``info`` (something worth knowing
    that changes nothing).
    """

    level: str
    role: str
    message: str
    fix: str = ""

    def __str__(self) -> str:
        """One line, so a structured log renderer does not fold the fix away."""
        text = f"[{self.level}] {self.role}: {self.message}"
        return f"{text} | fix: {self.fix}" if self.fix else text


def _model_is_pulled(wanted: str, pulled: Sequence[str]) -> bool:
    """Match the way Ollama resolves a tag, in both directions.

    ``bge-m3`` and ``bge-m3:latest`` name the same model, and which one a user
    writes is a matter of habit. Being strict here would tell the operator to
    pull a model they already have, which trains them to ignore this command --
    worse than not having it.
    """

    def canonical(name: str) -> str:
        name = name.strip()
        return name[: -len(":latest")] if name.endswith(":latest") else name

    available = {canonical(name) for name in pulled if name.strip()}
    return canonical(wanted) in available


def check_models(config: "Config", pulled: Sequence[str]) -> List[Finding]:
    """Compare the configured models against what Ollama actually has."""
    findings: List[Finding] = []
    ollama = config.ollama

    if not pulled:
        findings.append(
            Finding(
                "error",
                "ollama",
                "the server reports no pulled models",
                f"ollama pull {ollama.generation_model}",
            )
        )
        return findings

    for role, model in (
        ("generation", ollama.generation_model),
        ("teacher", ollama.teacher_model),
        ("embedding", ollama.embedding_model),
    ):
        if _model_is_pulled(model, pulled):
            findings.append(Finding("info", role, f"{model} is pulled"))
        else:
            findings.append(
                Finding(
                    "error",
                    role,
                    f"{model} is not pulled, so every call will fail",
                    f"ollama pull {model}",
                )
            )

    if ollama.think is not False:
        for role, model in (
            ("generation", ollama.generation_model),
            ("teacher", ollama.teacher_model),
        ):
            lowered = model.lower()
            if any(hint in lowered for hint in REASONING_FAMILY_HINTS):
                findings.append(
                    Finding(
                        "warning",
                        role,
                        f"{model} is a reasoning model and ollama.think is "
                        f"{ollama.think!r}; it can spend its whole context "
                        "budget thinking and return an empty answer, which is "
                        "measured behaviour, not a theoretical risk",
                        "set ollama.think: false, or LAYA_OLLAMA_THINK=false",
                    )
                )

    if ollama.num_ctx < MIN_REASONABLE_NUM_CTX:
        findings.append(
            Finding(
                "warning",
                "ollama",
                f"num_ctx is {ollama.num_ctx}; a Persian chunk plus an "
                "instruction plus a JSON answer will not fit",
                "raise ollama.num_ctx, or lower chunking.max_tokens",
            )
        )

    return findings


def check_embedding_choice(config: "Config", pulled: Sequence[str]) -> List[Finding]:
    """Note when the embedding model is one whose best use is not ours.

    Not an error: embeddings here only rank candidate negatives, so a
    general-purpose model is perfectly serviceable. It is worth saying so
    because these models are usually chosen for retrieval, where the gap
    matters.
    """
    findings: List[Finding] = []
    name = config.ollama.embedding_model.lower()
    for family, note in (
        (
            "qwen3-embedding",
            "wants an 'Instruct:' prefix on the query for its best numbers; "
            "this pipeline sends the bare query",
        ),
        (
            "bge-m3",
            "is a strong multilingual retriever, which is more than this "
            "pipeline's hard-negative mining needs",
        ),
    ):
        if family in name:
            findings.append(Finding("info", "embedding", note))
    del pulled
    return findings


def summarize(findings: Sequence[Finding]) -> str:
    """One line: how many of each level, worst first."""
    if not findings:
        return "no findings"
    counts: dict = {}
    for finding in findings:
        counts[finding.level] = counts.get(finding.level, 0) + 1
    order = {"error": 0, "warning": 1, "info": 2}
    parts = [
        f"{count} {level}"
        for level, count in sorted(counts.items(), key=lambda item: order.get(item[0], 9))
    ]
    return ", ".join(parts)


def has_errors(findings: Sequence[Finding]) -> bool:
    """Whether any finding blocks the run."""
    return any(finding.level == "error" for finding in findings)


def run_preflight(config: "Config", client: object) -> List[Finding]:
    """Fetch the pulled model list and check the configuration against it.

    ``client`` is anything with ``list_models() -> list[str]``, which is the
    only seam this touches -- ``OllamaClient`` in production, a stub in tests.
    """
    lister = getattr(client, "list_models", None)
    if not callable(lister):
        raise ConfigError(
            f"preflight needs a client with list_models(); got {type(client).__name__}"
        )
    pulled = lister()
    return check_models(config, pulled) + check_embedding_choice(config, pulled)


__all__ = [
    "Finding",
    "check_embedding_choice",
    "check_models",
    "has_errors",
    "run_preflight",
    "summarize",
]