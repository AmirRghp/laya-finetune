"""Preflight checks, and the verb bug they were written to catch.

The checks exist because the expensive failure modes announce themselves only
after the expensive part has started. A model that is not pulled does not fail
on load; it fails on the first call, inside a retry loop, about a hundred
seconds later.

They are pure functions over a config plus a list of names, so nothing here
touches a network (AGENTS.md 7.2).
"""

from __future__ import annotations

from typing import List

import pytest

from laya_pipeline import diagnostics
from laya_pipeline.config import Config, load_config
from laya_pipeline.diagnostics import Finding
from laya_pipeline.errors import OllamaResponseError
from laya_pipeline.llm.cache import NullCache
from laya_pipeline.llm.ollama import GET_ONLY_PATHS, OllamaClient

from tests.conftest import REPO_ROOT, FakeTransport

SHIPPED_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _config(**ollama: object) -> Config:
    config = load_config(SHIPPED_CONFIG, apply_env=False)
    for key, value in ollama.items():
        setattr(config.ollama, key, value)
    return config


def _all_pulled(config: Config) -> List[str]:
    return [
        config.ollama.generation_model,
        config.ollama.teacher_model,
        config.ollama.embedding_model,
    ]


def _levels(findings: List[Finding], role: str | None = None) -> List[str]:
    return [f.level for f in findings if role is None or f.role == role]


class TestModelPresence:
    def test_every_configured_model_being_pulled_is_clean(self) -> None:
        config = _config()
        findings = diagnostics.check_models(config, _all_pulled(config))
        assert not diagnostics.has_errors(findings)
        assert {f.role for f in findings} == {"generation", "teacher", "embedding"}

    def test_a_missing_generation_model_is_an_error(self) -> None:
        config = _config()
        pulled = [m for m in _all_pulled(config) if m != config.ollama.generation_model]
        findings = diagnostics.check_models(config, pulled)
        assert diagnostics.has_errors(findings)
        missing = [f for f in findings if f.role == "generation"]
        assert missing and missing[0].level == "error"
        assert f"ollama pull {config.ollama.generation_model}" in missing[0].fix

    def test_a_missing_embedding_model_is_an_error(self) -> None:
        config = _config()
        pulled = [m for m in _all_pulled(config) if m != config.ollama.embedding_model]
        findings = diagnostics.check_models(config, pulled)
        assert "error" in _levels(findings, "embedding")

    def test_nothing_pulled_is_one_clear_error(self) -> None:
        findings = diagnostics.check_models(_config(), [])
        assert diagnostics.has_errors(findings)
        # One finding, not three identical ones.
        assert len(findings) == 1
        assert findings[0].fix.startswith("ollama pull")

    def test_a_bare_name_matches_a_latest_tag(self) -> None:
        """Ollama resolves ``bge-m3`` to ``bge-m3:latest``; be lenient here.

        Being strict would tell the operator to pull a model they already have,
        which trains them to ignore this command.
        """
        config = _config(
            generation_model="bge-m3",
            teacher_model="bge-m3",
            embedding_model="bge-m3",
        )
        assert not diagnostics.has_errors(
            diagnostics.check_models(config, ["bge-m3:latest"])
        )

    def test_an_explicit_latest_tag_matches_a_bare_name(self) -> None:
        config = _config(
            generation_model="bge-m3:latest",
            teacher_model="bge-m3:latest",
            embedding_model="bge-m3:latest",
        )
        assert not diagnostics.has_errors(diagnostics.check_models(config, ["bge-m3"]))

    def test_a_different_tag_is_still_an_error(self) -> None:
        config = _config()
        assert diagnostics.has_errors(diagnostics.check_models(config, ["llama3.1:8b"]))


class TestReasoningModels:
    def test_thinking_enabled_on_a_reasoning_model_warns(self) -> None:
        config = _config(generation_model="qwen3.5:4b", teacher_model="qwen3.5:4b", think=None)
        findings = diagnostics.check_models(config, _all_pulled(config))
        warnings = [f for f in findings if f.level == "warning"]
        assert warnings
        assert "think: false" in warnings[0].fix

    def test_thinking_disabled_does_not_warn(self) -> None:
        config = _config(generation_model="qwen3.5:4b", teacher_model="qwen3.5:4b", think=False)
        findings = diagnostics.check_models(config, _all_pulled(config))
        assert "warning" not in _levels(findings)

    def test_a_non_reasoning_model_does_not_warn(self) -> None:
        """The heuristic must not nag about models it knows nothing about."""
        config = _config(generation_model="llama3.1:8b", teacher_model="llama3.1:8b", think=None)
        findings = diagnostics.check_models(config, _all_pulled(config))
        assert "warning" not in _levels(findings)

    def test_the_warning_covers_the_teacher_too(self) -> None:
        config = _config(
            generation_model="llama3.1:8b", teacher_model="qwen3.5:4b", think=True
        )
        findings = diagnostics.check_models(config, _all_pulled(config))
        assert any(f.level == "warning" and f.role == "teacher" for f in findings)


class TestContextWindow:
    def test_a_small_window_warns(self) -> None:
        config = _config(num_ctx=2048)
        findings = diagnostics.check_models(config, _all_pulled(config))
        assert "warning" in _levels(findings, "ollama")

    def test_a_healthy_window_is_quiet(self) -> None:
        config = _config(num_ctx=8192)
        findings = diagnostics.check_models(config, _all_pulled(config))
        assert "warning" not in _levels(findings, "ollama")


class TestEmbeddingNote:
    def test_qwen3_embedding_notes_the_missing_instruction_prefix(self) -> None:
        findings = diagnostics.check_embedding_choice(
            _config(embedding_model="qwen3-embedding:4b"), []
        )
        assert any("Instruct" in f.message for f in findings)

    def test_an_unknown_embedding_model_says_nothing(self) -> None:
        findings = diagnostics.check_embedding_choice(
            _config(embedding_model="some-other-embedder"), []
        )
        assert findings == []


class TestSummaryAndExit:
    def test_the_worst_level_comes_first(self) -> None:
        findings = [Finding("info", "a", "x"), Finding("error", "b", "y"), Finding("warning", "c", "z")]
        assert diagnostics.summarize(findings).startswith("1 error")

    def test_no_findings_summarise_cleanly(self) -> None:
        assert diagnostics.summarize([]) == "no findings"

    def test_has_errors_ignores_warnings(self) -> None:
        assert not diagnostics.has_errors([Finding("warning", "a", "x")])


class TestRunPreflight:
    def test_it_uses_the_client_seam(self) -> None:
        config = _config()
        client = OllamaClient(
            config.ollama, transport=FakeTransport(), cache=NullCache()
        )
        findings = diagnostics.run_preflight(config, client)
        # The fake reports only "fake-model", so every configured role is absent.
        assert diagnostics.has_errors(findings)
        assert {f.role for f in findings if f.level == "error"} == {
            "generation",
            "teacher",
            "embedding",
        }

    def test_it_passes_when_everything_is_present(self) -> None:
        config = _config(
            generation_model="fake-model",
            teacher_model="fake-model",
            embedding_model="fake-model",
        )
        client = OllamaClient(
            config.ollama, transport=FakeTransport(), cache=NullCache()
        )
        assert not diagnostics.has_errors(diagnostics.run_preflight(config, client))

    def test_a_client_without_the_seam_is_refused(self) -> None:
        from laya_pipeline.errors import ConfigError

        with pytest.raises(ConfigError, match="list_models"):
            diagnostics.run_preflight(_config(), object())


class TestHttpVerbs:
    """The bug the fake transport was hiding.

    ``/api/tags`` is a ``GET`` endpoint. Posting to it returns ``405``, so
    ``is_available()`` reported every reachable server as unreachable and
    ``list_models()`` always raised. A fake that answered ``/api/tags`` for any
    verb made both look fine for the entire life of the test suite.
    """

    def test_tags_is_in_the_get_only_set(self) -> None:
        assert "/api/tags" in GET_ONLY_PATHS
        assert "/api/chat" not in GET_ONLY_PATHS

    def test_the_fake_refuses_a_post_to_tags(self) -> None:
        with pytest.raises(OllamaResponseError, match="405"):
            FakeTransport().post("/api/tags", {})

    def test_the_fake_refuses_a_get_to_chat(self) -> None:
        with pytest.raises(OllamaResponseError, match="405"):
            FakeTransport().get("/api/chat")

    def test_availability_uses_get(self) -> None:
        transport = FakeTransport()
        client = OllamaClient(
            _config().ollama, transport=transport, cache=NullCache()
        )
        assert client.is_available() is True
        assert transport.get_calls == ["/api/tags"]

    def test_listing_uses_get(self) -> None:
        transport = FakeTransport()
        client = OllamaClient(
            _config().ollama, transport=transport, cache=NullCache()
        )
        assert client.list_models() == ["fake-model"]
        assert transport.get_calls == ["/api/tags"]

    def test_an_unreachable_server_is_not_available(self) -> None:
        class DeadTransport(FakeTransport):
            def get(self, path):
                from laya_pipeline.errors import OllamaUnavailable

                raise OllamaUnavailable("connection refused")

        client = OllamaClient(
            _config().ollama, transport=DeadTransport(), cache=NullCache()
        )
        assert client.is_available() is False


def test_the_preflight_command_is_registered_and_tagged() -> None:
    """It is a script gate, so it must carry a cost class like every stage."""
    from laya_pipeline.cli import build_parser

    parser = build_parser()
    action = next(
        a
        for a in parser._actions
        if hasattr(a, "choices") and isinstance(a.choices, dict) and "scan" in a.choices
    )
    assert "preflight" in action.choices
    assert "[ollama]" in (action.choices["preflight"].description or "")


def test_offline_preflight_succeeds_without_a_server() -> None:
    """``--offline`` must work with Ollama stopped, or it is no use as a probe."""
    from laya_pipeline.cli import main

    assert main(["--config", str(SHIPPED_CONFIG), "preflight", "--offline"]) == 0


def test_the_readme_documents_the_command() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "laya-pipeline preflight" in readme


def test_check_env_script_calls_preflight() -> None:
    """One implementation, not two: the shell script delegates."""
    script = (REPO_ROOT / "scripts" / "check_env.sh").read_text(encoding="utf-8")
    assert "laya-pipeline preflight" in script
    # It must not reimplement the model-name comparison.
    assert "/api/tags" in script  # the reachability ping is fine
    assert "ollama pull" not in script  # the fix text now lives in one place