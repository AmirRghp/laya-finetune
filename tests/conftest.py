"""Shared fixtures.

Every fixture is synthetic. No production corpus document, no network, no GPU
and no Laya weights are involved in any test (AGENTS.md 7.1, 7.2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from laya_pipeline.config import Config, load_config  # noqa: E402
from laya_pipeline.errors import OllamaResponseError, OllamaTimeout  # noqa: E402
from laya_pipeline.io import write_text  # noqa: E402
from laya_pipeline.llm.ollama import GET_ONLY_PATHS  # noqa: E402

# --------------------------------------------------------------------------- #
# Persian sample content
# --------------------------------------------------------------------------- #
PERSIAN_DOC_A = """---
document_id: AA-001
title: "سیاست مرخصی"
access_role: employee
---

# سیاست مرخصی

## AA-001-S1 — ثبت درخواست

هر کارمند باید درخواست مرخصی را سه روز کاری پیش از شروع در سامانه ثبت کند.
مدیر حداکثر دو روز کاری برای پاسخ فرصت دارد.

## AA-001-S2 — انتقال مانده

حداکثر پنج روز کاری از مانده به دورهٔ بعد منتقل می‌شود.
"""

PERSIAN_DOC_B = """# رویهٔ امنیت داده

## کنترل دسترسی

دسترسی به دادهٔ مشتری تنها با درخواست ثبت‌شده و تأیید مالک سامانه ممکن است.
هر درخواست باید در سامانهٔ تیکت ثبت و پس از استفاده بسته شود.
"""

PERSIAN_DOC_C = """---
document_id: CC-003
title: "ارتباطات تیمی"
---

# ارتباطات تیمی

## انتخاب کانال

برای هماهنگی جاری از کانال رسمی تیم استفاده کنید و پیوند سند را ثبت کنید.
"""


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def hermetic_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test outside the repository root.

    ``load_config`` reads ``.env`` from the current directory (or its parent),
    which is correct for a real run and wrong for a test: a developer's
    ``.env`` names the models on their own machine, so a test that reads it
    passes there and fails on CI, or passes for the wrong reason.

    Running from an empty temporary directory means the only ``.env`` in reach
    is one the test itself created. ``test_dotenv.py`` calls ``_load_dotenv``
    with an explicit root, so it still exercises the real loader.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A tiny synthetic corpus: three documents, one nested a directory deep."""
    root = tmp_path / "corpus"
    nested = root / "team" / "policies"
    nested.mkdir(parents=True)
    write_text(root / "a.md", PERSIAN_DOC_A)
    write_text(nested / "b.markdown", PERSIAN_DOC_B)
    write_text(root / "c.md", PERSIAN_DOC_C)
    return root


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """The project's real configuration, redirected at a temporary directory.

    Loading the shipped YAML rather than a synthetic config means a change that
    breaks the real file fails the suite.
    """
    loaded = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
    loaded.paths.corpus_dir = str(tmp_path / "corpus")
    loaded.paths.output_dir = str(tmp_path / "outputs")
    loaded.paths.prompt_dir = str(REPO_ROOT / "prompts")
    loaded.generation.questions_per_chunk = 2
    loaded.generation.min_chunk_chars = 0
    # The shipped minimum is 15 characters. Test questions are short, and the
    # length gate is covered on its own in test_ids/test_dataset, so it is
    # lowered here rather than padding every fixture question.
    loaded.generation.min_question_chars = 5
    loaded.negatives.ratio = 1.0
    loaded.negatives.candidates_per_positive = 6
    return loaded


# --------------------------------------------------------------------------- #
# Fake Ollama transport
# --------------------------------------------------------------------------- #
class FakeTransport:
    """A scripted Ollama server.

    Every external model call in the pipeline goes through
    ``OllamaTransport``, so substituting this covers generation, embedding,
    verification and labeling without a network (AGENTS.md 7.3).
    """

    def __init__(
        self,
        responses: Optional[Sequence[Any]] = None,
        *,
        embedding_dim: int = 8,
        fail_times: int = 0,
    ) -> None:
        #: Consumed in order by ``/api/chat``. The last entry repeats.
        self.responses: List[Any] = list(responses or [])
        self.embedding_dim = embedding_dim
        self.fail_times = fail_times
        self.calls: List[Dict[str, Any]] = []
        self.embed_calls: List[List[str]] = []
        #: ``GET`` paths requested. Kept separate so a test can assert the verb.
        self.get_calls: List[str] = []

    # -- scripted behaviour ------------------------------------------------ #
    def next_response(self) -> Any:
        if not self.responses:
            return {"questions": [{"question": "پرسش پیش‌فرض چیست؟", "type": "direct", "key_points": ["نکته"]}]}
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        # Ollama answers 405 to the wrong verb. The fake enforces the same rule
        # so a wrong verb fails here rather than against a live server.
        if path in GET_ONLY_PATHS:
            raise OllamaResponseError(f"405 method not allowed for {path}", retryable=False)
        if path == "/api/embed":
            texts = list(payload.get("input", []))
            self.embed_calls.append(texts)
            return {"embeddings": [self._vector(text) for text in texts]}
        if path == "/api/embeddings":
            self.embed_calls.append([str(payload.get("prompt", ""))])
            return {"embedding": self._vector(str(payload.get("prompt", "")))}

        if self.fail_times > 0:
            self.fail_times -= 1
            raise OllamaTimeout("scripted failure")

        self.calls.append({"path": path, "payload": dict(payload)})
        return {"message": {"role": "assistant", "content": json.dumps(
            self.next_response(), ensure_ascii=False
        )}}

    def get(self, path: str) -> Mapping[str, Any]:
        if path not in GET_ONLY_PATHS:
            raise OllamaResponseError(f"405 method not allowed for {path}", retryable=False)
        self.get_calls.append(path)
        if path == "/api/version":
            return {"version": "0.0.0-fake"}
        return {"models": [{"name": "fake-model"}]}

    def _vector(self, text: str) -> List[float]:
        """A deterministic pseudo-embedding from character counts.

        Similar texts get similar vectors, which is enough for the ranking
        assertions without introducing a real model.
        """
        vector = [0.0] * self.embedding_dim
        for index, char in enumerate(str(text)):
            vector[index % self.embedding_dim] += (ord(char) % 17) / 17.0
        if all(value == 0.0 for value in vector):
            vector[0] = 1.0
        return vector

    # -- assertions helpers ------------------------------------------------ #
    @property
    def prompts(self) -> List[str]:
        return [
            str(call["payload"].get("messages", [{}])[-1].get("content", ""))
            for call in self.calls
        ]


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()


def chat_response(data: Any) -> Dict[str, Any]:
    """Build one scripted ``/api/chat`` body."""
    return {"message": {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)}}


def questions_response(*pairs: Any) -> Dict[str, Any]:
    """``{questions: [{question, type, key_points}, ...]}`` from tuples."""
    return {
        "questions": [
            {"question": question, "type": kind, "key_points": list(points)}
            for question, kind, *points in pairs
        ]
    }


def evidence_response(answer: str, confidence: float = 0.9) -> Dict[str, Any]:
    return {"answers": answer, "covered_key_points": [0], "confidence": confidence}


def label_response(label: str, confidence: float = 0.9) -> Dict[str, Any]:
    return {"label": label, "confidence": confidence, "reason": "دلیل نمونه"}


__all__ = [
    "FakeTransport",
    "PERSIAN_DOC_A",
    "PERSIAN_DOC_B",
    "PERSIAN_DOC_C",
    "chat_response",
    "evidence_response",
    "label_response",
    "questions_response",
]