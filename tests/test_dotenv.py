"""``.env`` has to actually be read.

``.env.example`` documents ``LAYA_*`` variables. If no code loads ``.env``, an
operator edits it, sees no change in behaviour, and has no way to tell whether
the edit took effect -- the worst possible failure for a configuration file.

On this project that failure was live: the shipped ``.env`` named
``LAYA_OLLAMA_GENERATION_MODEL``, and nothing read it, so the model actually
used was whatever the YAML said. These tests pin the loading, the precedence
rules, and the fact that every documented variable has somewhere to land.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from laya_pipeline.config import _load_dotenv, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "configs" / "default.yaml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@pytest.fixture(autouse=True)
def clean_laya_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test owns the whole ``LAYA_*`` namespace and starts it empty."""
    for name in [n for n in os.environ if n.startswith("LAYA_")]:
        monkeypatch.delenv(name, raising=False)


def write_env(root: Path, text: str) -> Path:
    path = root / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def assignable_variables() -> list[str]:
    """Every environment variable the shipped config actually consumes."""
    config = load_config(SHIPPED_CONFIG, apply_env=False)
    return list(getattr(config, "env_overrides", {}).values())


class TestDotenvLoading:
    def test_a_variable_reaches_the_configuration(self, tmp_path: Path) -> None:
        write_env(
            tmp_path,
            "LAYA_OLLAMA_GENERATION_MODEL=from-dotenv\n"
            "LAYA_OLLAMA_TEACHER_MODEL=teacher-from-dotenv\n",
        )
        _load_dotenv(tmp_path)
        config = load_config(SHIPPED_CONFIG)
        assert config.ollama.generation_model == "from-dotenv"
        assert config.ollama.teacher_model == "teacher-from-dotenv"

    def test_the_real_environment_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A one-off override must not require editing the file."""
        write_env(tmp_path, "LAYA_OLLAMA_GENERATION_MODEL=from-dotenv\n")
        monkeypatch.setenv("LAYA_OLLAMA_GENERATION_MODEL", "from-shell")
        _load_dotenv(tmp_path)
        assert load_config(SHIPPED_CONFIG).ollama.generation_model == "from-shell"

    def test_export_prefix_is_accepted(self, tmp_path: Path) -> None:
        write_env(tmp_path, "export LAYA_OLLAMA_GENERATION_MODEL=exported\n")
        _load_dotenv(tmp_path)
        assert load_config(SHIPPED_CONFIG).ollama.generation_model == "exported"

    def test_quotes_are_stripped(self, tmp_path: Path) -> None:
        write_env(
            tmp_path,
            'LAYA_OLLAMA_GENERATION_MODEL="double"\n'
            "LAYA_OLLAMA_TEACHER_MODEL='single'\n",
        )
        _load_dotenv(tmp_path)
        config = load_config(SHIPPED_CONFIG)
        assert config.ollama.generation_model == "double"
        assert config.ollama.teacher_model == "single"

    def test_comments_and_blanks_are_skipped(self, tmp_path: Path) -> None:
        write_env(
            tmp_path,
            "# a comment\n"
            "\n"
            "   \n"
            "LAYA_OLLAMA_GENERATION_MODEL=value # trailing note\n",
        )
        _load_dotenv(tmp_path)
        assert load_config(SHIPPED_CONFIG).ollama.generation_model == "value"

    def test_an_empty_value_is_kept_as_empty(self, tmp_path: Path) -> None:
        """An empty value means "no override"; it must not read as a set name."""
        write_env(tmp_path, "LAYA_OLLAMA_GENERATION_MODEL=\n")
        _load_dotenv(tmp_path)
        assert os.environ["LAYA_OLLAMA_GENERATION_MODEL"] == ""
        assert load_config(SHIPPED_CONFIG).ollama.generation_model != ""

    def test_a_line_without_an_equals_sign_is_skipped(self, tmp_path: Path) -> None:
        write_env(tmp_path, "LAYA_OLLAMA_GENERATION_MODEL\n")
        _load_dotenv(tmp_path)
        assert "LAYA_OLLAMA_GENERATION_MODEL" not in os.environ

    def test_a_malformed_key_is_skipped_rather_than_set(self, tmp_path: Path) -> None:
        write_env(tmp_path, "not a key=value\nLAYA_OLLAMA_GENERATION_MODEL=ok\n")
        _load_dotenv(tmp_path)
        assert "not a key" not in os.environ
        assert os.environ["LAYA_OLLAMA_GENERATION_MODEL"] == "ok"

    def test_a_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert _load_dotenv(tmp_path) is None

    def test_the_parent_directory_is_checked(self, tmp_path: Path) -> None:
        """Works when the command is invoked from a subdirectory."""
        root = tmp_path / "project"
        nested = root / "configs"
        nested.mkdir(parents=True)
        write_env(root, "LAYA_OLLAMA_GENERATION_MODEL=parent\n")
        assert _load_dotenv(nested) == root / ".env"
        assert load_config(SHIPPED_CONFIG).ollama.generation_model == "parent"

    def test_the_nearer_file_wins(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        nested = root / "configs"
        nested.mkdir(parents=True)
        write_env(root, "LAYA_OLLAMA_GENERATION_MODEL=parent\n")
        write_env(nested, "LAYA_OLLAMA_GENERATION_MODEL=nearer\n")
        assert _load_dotenv(nested) == nested / ".env"

    def test_loading_twice_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running must not append to or mangle an already-set value."""
        write_env(tmp_path, "LAYA_OLLAMA_GENERATION_MODEL=stable\n")
        _load_dotenv(tmp_path)
        first = load_config(SHIPPED_CONFIG).ollama.generation_model
        _load_dotenv(tmp_path)
        assert load_config(SHIPPED_CONFIG).ollama.generation_model == first == "stable"

    def test_apply_env_false_ignores_the_file(self, tmp_path: Path) -> None:
        write_env(tmp_path, "LAYA_OLLAMA_GENERATION_MODEL=from-dotenv\n")
        config = load_config(SHIPPED_CONFIG, apply_env=False)
        assert config.ollama.generation_model != "from-dotenv"


class TestPathOverrides:
    def test_the_corpus_and_output_directories_can_be_redirected(
        self, tmp_path: Path
    ) -> None:
        write_env(
            tmp_path,
            f"LAYA_CORPUS_DIR={tmp_path / 'other-corpus'}\n"
            f"LAYA_OUTPUT_DIR={tmp_path / 'other-outputs'}\n",
        )
        _load_dotenv(tmp_path)
        config = load_config(SHIPPED_CONFIG)
        assert config.corpus_dir == tmp_path / "other-corpus"
        assert config.output_dir == tmp_path / "other-outputs"

    def test_an_unknown_variable_name_is_rejected(self, tmp_path: Path) -> None:
        """A typo in ``env_overrides`` must not be a silent no-op."""
        write_env(tmp_path, "LAYA_NOT_A_SETTING=x\n")
        _load_dotenv(tmp_path)
        os.environ["LAYA_NOT_A_SETTING"] = "x"
        from laya_pipeline.config import _apply_env_overrides
        from laya_pipeline.errors import ConfigError

        config = load_config(SHIPPED_CONFIG, apply_env=False)
        config.env_overrides = {"not_a_setting": "LAYA_NOT_A_SETTING"}
        with pytest.raises(ConfigError, match="not_a_setting"):
            _apply_env_overrides(config)


class TestEnvExampleIsHonest:
    def test_it_is_a_valid_dotenv_file(self) -> None:
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("export "):
                stripped = stripped[len("export ") :].strip()
            key, separator, _value = stripped.partition("=")
            assert separator, f".env.example line is not an assignment: {line!r}"
            assert _KEY.fullmatch(key), f"bad key in .env.example: {key!r}"

    def test_every_active_example_variable_is_consumed(self) -> None:
        """A documented variable with nowhere to land is a lie in the docs.

        Every non-comment line in ``.env.example`` names a variable that the
        shipped ``env_overrides`` map must actually consume. This is the test
        that would have caught ``LAYA_CORPUS_DIR`` being ignored.
        """
        consumed = set(assignable_variables())
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name = stripped.split("=", 1)[0].strip()
            if name.lower().startswith("export "):
                name = name[len("export ") :].strip()
            assert name in consumed, (
                f".env.example documents {name} but env_overrides does not "
                f"consume it; it would be silently ignored"
            )

    def test_every_consumed_variable_is_documented(self) -> None:
        """The reverse direction: nothing configurable is undocumented."""
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for variable in assignable_variables():
            assert variable in text, f"{variable} is consumed but not in .env.example"

    def test_think_is_documented_because_it_is_not_obvious(self) -> None:
        """This one bites: a Qwen3 model that thinks returns nothing at all."""
        assert "LAYA_OLLAMA_THINK" in ENV_EXAMPLE.read_text(encoding="utf-8")