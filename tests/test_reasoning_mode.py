"""Reasoning models must be told not to reason, or they never answer.

Measured on the operator's machine against ``qwen3.5:2b-q4_K_M``: with
``think`` left at the server default, a one-line Persian question consumed
4060 eval tokens and returned ``content: ""`` with ``done_reason: "length"``.
The pipeline then reported ``model response contains no JSON object: ''``
after three attempts and roughly five minutes -- accurate, but it names a
parsing problem rather than the configuration change that fixes it.

These tests pin both halves of the fix: the flag that is sent, and the
diagnostic that replaces the empty-JSON message when a model still answers
with nothing.
"""

from __future__ import annotations

import pytest

from laya_pipeline.config import OllamaConfig, load_config
from laya_pipeline.errors import ConfigError, OllamaError
from laya_pipeline.llm.cache import NullCache
from laya_pipeline.llm.ollama import OllamaClient
from laya_pipeline.llm.schema import object_schema

from tests.conftest import REPO_ROOT, FakeTransport


def _client(transport: FakeTransport) -> OllamaClient:
    config = OllamaConfig(
        endpoint="http://fake",
        generation_model="gen",
        max_retries=2,
        retry_backoff_seconds=0,
    )
    return OllamaClient(
        config, transport=transport, cache=NullCache(), sleep=lambda _s: None
    )


def _schema() -> dict[str, object]:
    return object_schema({"a": {"type": "integer"}})


class TestTheThinkFlag:
    def test_think_false_is_sent_by_default(self) -> None:
        transport = FakeTransport([{"a": 1}])
        _client(transport).generate_json("پرسش", schema=_schema())
        assert transport.calls[0]["payload"]["think"] is False

    def test_think_can_be_turned_on(self) -> None:
        transport = FakeTransport([{"a": 1}])
        client = _client(transport)
        client.config.think = True
        client.generate_json("پرسش", schema=_schema())
        assert transport.calls[0]["payload"]["think"] is True

    def test_think_can_be_left_to_the_model(self) -> None:
        """``null`` omits the key, leaving the server default in place."""
        transport = FakeTransport([{"a": 1}])
        client = _client(transport)
        client.config.think = None
        client.generate_json("پرسش", schema=_schema())
        assert "think" not in transport.calls[0]["payload"]

    def test_free_text_generation_also_sends_it(self) -> None:
        transport = FakeTransport([{"a": 1}])
        _client(transport).generate_text("پرسش")
        assert transport.calls[-1]["payload"]["think"] is False

    def test_embeddings_do_not_carry_it(self) -> None:
        """``think`` is a generation flag; sending it to /api/embed is noise."""
        transport = FakeTransport()
        _client(transport).embed(["یک"])
        assert transport.embed_calls == [["یک"]]
        assert all("think" not in call["payload"] for call in transport.calls)


class TestConfiguration:
    def test_the_shipped_config_disables_thinking(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
        assert config.ollama.think is False

    def test_the_env_var_can_override_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path = REPO_ROOT / "configs" / "default.yaml"
        monkeypatch.setenv("LAYA_OLLAMA_THINK", "true")
        assert load_config(path).ollama.think is True
        monkeypatch.setenv("LAYA_OLLAMA_THINK", "false")
        assert load_config(path).ollama.think is False

    def test_null_clears_it_back_to_the_server_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``null`` is the third state, and it has to be expressible."""
        monkeypatch.setenv("LAYA_OLLAMA_THINK", "null")
        config = load_config(REPO_ROOT / "configs" / "default.yaml")
        assert config.ollama.think is None

    def test_a_non_boolean_think_is_rejected(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
        config.ollama.think = "maybe"  # type: ignore[assignment]
        with pytest.raises(ConfigError, match="ollama.think"):
            config.validate()

    def test_an_unparseable_env_value_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo must not become ``False`` (AGENTS.md 5.5: raise, don't repair)."""
        monkeypatch.setenv("LAYA_OLLAMA_THINK", "flase")
        with pytest.raises(ConfigError, match="is not a boolean"):
            load_config(REPO_ROOT / "configs" / "default.yaml")


def _exhausted() -> dict[str, object]:
    """The exact shape a thought-exhausted Qwen3.5 reply had."""
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "Thinking Process:\n\n1. Analyse the request...",
        },
        "done_reason": "length",
        "eval_count": 4060,
    }


class TestEmptyAnswerDiagnostics:
    def test_the_reasoning_cause_is_named(self) -> None:
        class ThinkingTransport(FakeTransport):
            def post(self, path, payload):
                return _exhausted()

        with pytest.raises(OllamaError) as info:
            _client(ThinkingTransport()).generate_json("پرسش", schema=_schema())
        message = str(info.value)
        assert "think: false" in message
        assert "4060" in message

    def test_a_thought_exhausted_answer_is_not_retried(self) -> None:
        """Retrying cannot help: the same budget gets spent the same way.

        At roughly 100 seconds a call, three attempts is five minutes spent
        before saying anything useful.
        """

        class ThinkingTransport(FakeTransport):
            def __init__(self) -> None:
                super().__init__()
                self.hits = 0

            def post(self, path, payload):
                self.hits += 1
                return _exhausted()

        transport = ThinkingTransport()
        client = _client(transport)
        with pytest.raises(OllamaError, match="think: false"):
            client.generate_json("پرسش", schema=_schema())
        assert client.retry_count == 0
        assert transport.hits == 1

    def test_thinking_without_exhaustion_still_names_the_flag(self) -> None:
        """``done_reason: "stop"`` with empty content is the same root cause."""

        class ThinkingTransport(FakeTransport):
            def post(self, path, payload):
                return {
                    "message": {"role": "assistant", "content": "", "thinking": "hmm"},
                    "done_reason": "stop",
                }

        with pytest.raises(OllamaError, match="think: false"):
            _client(ThinkingTransport()).generate_json("پرسش", schema=_schema())

    def test_a_plain_empty_answer_is_still_retried(self) -> None:
        """No ``thinking`` field means no reasoning diagnosis, so retry as before."""

        class EmptyTransport(FakeTransport):
            def post(self, path, payload):
                return {"message": {"role": "assistant", "content": ""}}

        transport = EmptyTransport()
        client = _client(transport)
        with pytest.raises(OllamaError):
            client.generate_json("پرسش", schema=_schema())
        assert client.retry_count >= 1

    def test_an_exhausted_context_is_reported_as_a_context_problem(self) -> None:
        """``done_reason: "length"`` with nothing generated is a num_ctx fault."""

        class OverflowTransport(FakeTransport):
            def post(self, path, payload):
                return {
                    "message": {"role": "assistant", "content": ""},
                    "done_reason": "length",
                    "eval_count": 0,
                }

        with pytest.raises(OllamaError, match="num_ctx"):
            _client(OverflowTransport()).generate_json("پرسش", schema=_schema())

    def test_an_empty_response_field_is_caught_too(self) -> None:
        """The ``/api/generate`` shape has ``response``, not ``message.content``."""

        class EmptyTransport(FakeTransport):
            def post(self, path, payload):
                return {"response": ""}

        with pytest.raises(OllamaError):
            _client(EmptyTransport()).generate_json("پرسش", schema=_schema())

    def test_whitespace_only_content_is_not_success(self) -> None:
        """A model that answers with a newline is not answering."""

        class BlankTransport(FakeTransport):
            def post(self, path, payload):
                return {"message": {"role": "assistant", "content": "  \n "}}

        with pytest.raises(OllamaError):
            _client(BlankTransport()).generate_json("پرسش", schema=_schema())

    def test_a_normal_answer_is_returned_unchanged(self) -> None:
        """The guard must not swallow a real answer."""

        class OkTransport(FakeTransport):
            def post(self, path, payload):
                return {
                    "message": {"role": "assistant", "content": '{"a": 1}'},
                    "done_reason": "stop",
                }

        result = _client(OkTransport()).generate_json("پرسش", schema=_schema())
        assert result.data == {"a": 1}


def test_the_shipped_prompts_stay_within_a_sane_context() -> None:
    """A prompt that would overflow num_ctx should fail loudly, not silently.

    Cheap guard: the shipped generation prompt must stay far below the
    configured window, so ``done_reason: "length"`` keeps meaning "the model
    was thinking", never "the prompt was enormous".
    """
    config = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
    prompts = REPO_ROOT / "prompts"
    assert prompts.is_dir()
    largest = max(p.stat().st_size for p in prompts.glob("*.md"))
    # A rough 2 bytes/char worst case for UTF-8 Persian is still far inside 8192.
    assert largest / 2 < config.ollama.num_ctx


def test_the_config_window_is_positive() -> None:
    """Guard against a num_ctx that guarantees a length-exhausted reply."""
    config = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
    assert config.ollama.num_ctx >= 4096