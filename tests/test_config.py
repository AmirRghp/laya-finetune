"""Configuration loading, validation and the TrainConfig bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from laya_pipeline.config import (
    Config,
    EvaluationConfig,
    NegativesConfig,
    TrainingConfig,
    config_hash,
    load_config,
)
from laya_pipeline.errors import ConfigError
from laya_pipeline.io import write_text

from tests.conftest import REPO_ROOT


@pytest.fixture
def defaults() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)


class TestLoading:
    def test_the_shipped_config_loads(self, defaults: Config) -> None:
        assert defaults.laya.base_model
        assert defaults.laya.questions

    def test_config_hash_is_stable(self, defaults: Config) -> None:
        assert defaults.config_hash == defaults.config_hash

    def test_a_different_config_hashes_differently(self, defaults: Config) -> None:
        other = defaults.with_overrides({"generation.questions_per_chunk": 99})
        assert other.config_hash != defaults.config_hash

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        write_text(path, "project: [unclosed\n")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(path)

    def test_non_mapping_document_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        write_text(path, "- a\n- b\n")
        with pytest.raises(ConfigError, match="YAML mapping"):
            load_config(path)

    def test_partial_config_inherits_defaults(self, tmp_path: Path) -> None:
        """A partial file is filled from the dataclass defaults.

        A partial config alone cannot pass validation, because the Laya questions
        have no defaults — so the file layers over ``configs/default.yaml`` in
        practice. Here the point is only that unspecified keys keep their
        defaults rather than becoming errors.
        """
        path = tmp_path / "partial.yaml"
        write_text(path, "project:\n  name: custom\n")
        with pytest.raises(ConfigError, match="questions is empty"):
            load_config(path)

    def test_source_path_is_recorded(self, defaults: Config) -> None:
        assert defaults.source_path is not None


class TestUnknownKeys:
    def test_unknown_top_level_key_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "typo.yaml"
        write_text(path, "projekt:\n  name: x\n")
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(path)

    def test_unknown_section_key_is_an_error(self, tmp_path: Path) -> None:
        """A typo'd key must fail loudly. ``question_per_chunk`` for
        ``questions_per_chunk`` is exactly the mistake a silent default hides."""
        path = tmp_path / "typo.yaml"
        write_text(path, "generation:\n  question_per_chunk: 3\n")
        with pytest.raises(ConfigError, match="question_per_chunk"):
            load_config(path)

    def test_override_of_a_missing_key_is_an_error(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="does not exist"):
            defaults.with_overrides({"generation.nonexistent": 1})

    def test_override_of_an_unknown_segment_is_an_error(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="unknown segment"):
            defaults.with_overrides({"nope.leaf": 1})


class TestValidation:
    def test_bad_endpoint_is_rejected(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="http"):
            defaults.with_overrides({"ollama.endpoint": "not-a-url"})

    def test_non_positive_timeout_is_rejected(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="timeout_seconds"):
            defaults.with_overrides({"ollama.timeout_seconds": 0})

    def test_split_ratios_must_sum_to_one(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="sum to 1.0"):
            defaults.with_overrides({"splitting.test": 0.5})

    def test_strategy_weights_must_match_strategies(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="strategy_weights"):
            defaults.with_overrides({"negatives.strategy_weights": [0.5, 0.5]})

    def test_strategy_weights_must_sum_to_one(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="sum to 1.0"):
            defaults.with_overrides({"negatives.strategy_weights": [0.9, 0.9, 0.9]})

    def test_unknown_strategy_is_rejected(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="unknown entries"):
            defaults.with_overrides({"negatives.strategies": ["telepathy"]})

    def test_similarity_bounds_must_be_ordered(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="min_similarity"):
            defaults.with_overrides({"negatives.max_similarity": 0.1})

    def test_label_roles_band_must_be_ordered(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="uncertain_low"):
            defaults.with_overrides({"labeling.uncertain_low": 0.9})

    def test_review_threshold_must_exceed_the_uncertain_band(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="review_threshold"):
            defaults.with_overrides({"labeling.review_threshold": 0.4})

    def test_head_max_len_cannot_exceed_max_len(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="head_max_len"):
            defaults.with_overrides({"laya.head_max_len": 4096})

    def test_chunk_max_cannot_exceed_the_state_budget(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="max_tokens"):
            defaults.with_overrides({"chunking.max_tokens": 4096})

    def test_chunk_sizes_must_be_ordered(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="min_tokens <="):
            defaults.with_overrides({"chunking.min_tokens": 900})

    def test_a_question_needs_instructions(self, defaults: Config) -> None:
        defaults.laya.questions["relevance"]["instructions"] = ""
        with pytest.raises(ConfigError, match="instructions"):
            defaults.validate()

    def test_a_choice_question_needs_criteria(self, defaults: Config) -> None:
        defaults.laya.questions["relevance"]["criteria"] = {}
        with pytest.raises(ConfigError, match="criteria"):
            defaults.validate()

    def test_criteria_descriptions_must_be_non_empty(self, defaults: Config) -> None:
        defaults.laya.questions["relevance"]["criteria"]["relevant"] = "  "
        with pytest.raises(ConfigError, match="criteria"):
            defaults.validate()

    def test_too_many_options_is_rejected(self, defaults: Config) -> None:
        defaults.laya.max_options = 1
        with pytest.raises(ConfigError, match="at least 2"):
            defaults.validate()

    def test_orphan_state_template_is_rejected(self, defaults: Config) -> None:
        defaults.laya.state_templates["ghost"] = "{{query}}"
        with pytest.raises(ConfigError, match="no question"):
            defaults.validate()

    def test_unknown_question_type_is_rejected(self, defaults: Config) -> None:
        defaults.laya.questions["relevance"]["type"] = "telepathy"
        with pytest.raises(ConfigError, match="type must be one of"):
            defaults.validate()

    def test_unknown_loss_is_rejected(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="training.loss"):
            defaults.with_overrides({"training.loss": "magic"})

    def test_retrieval_ks_must_be_sorted(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="sorted"):
            defaults.with_overrides({"evaluation.retrieval_ks": [5, 1]})

    def test_unknown_retrieval_backend_is_rejected(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="retrieval_backend"):
            defaults.with_overrides({"evaluation.retrieval_backend": "carrier-pigeon"})

    def test_question_length_bounds_are_checked(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="min_question_chars"):
            defaults.with_overrides({"generation.min_question_chars": 400})


class TestQuestionAccessors:
    def test_options_come_from_the_criteria_order(self, defaults: Config) -> None:
        assert defaults.laya.options("relevance") == ["relevant", "irrelevant"]

    def test_role_labels_resolve(self, defaults: Config) -> None:
        assert defaults.laya.role_label("relevance", "positive") == "relevant"
        assert defaults.laya.role_label("relevance", "negative") == "irrelevant"

    def test_unconfigured_role_raises(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="not configured"):
            defaults.laya.role_label("sufficiency", "positive")

    def test_role_label_must_be_an_option(self, defaults: Config) -> None:
        defaults.laya.label_roles["relevance"]["positive"] = "maybe"
        with pytest.raises(ConfigError, match="not an option"):
            defaults.laya.role_label("relevance", "positive")

    def test_unknown_task_raises_with_the_available_list(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="relevance"):
            defaults.laya.question("telepathy")

    def test_template_is_retrieved(self, defaults: Config) -> None:
        assert "{{query}}" in defaults.laya.template("relevance")

    def test_missing_template_raises(self, defaults: Config) -> None:
        with pytest.raises(ConfigError, match="state_templates"):
            defaults.laya.template("telepathy")


class TestTrainingBridge:
    def test_kwargs_match_the_laya_trainconfig_fields(self, defaults: Config) -> None:
        kwargs = defaults.training.to_train_config(
            defaults.laya.max_len, defaults.laya.head_max_len
        )
        expected = {
            "epochs", "micro_batch", "grad_accum", "encoder_lr", "head_lr", "min_lr",
            "weight_decay", "grad_clip", "loss", "rl_samples", "sigma_start", "sigma_end",
            "w_sph", "w_rps", "shuffle_options", "option_layout", "freeze_encoder",
            "calib_max", "calib_frac", "calib_seed", "seed", "max_len", "head_max_len",
            "amp", "gradient_checkpointing", "log_every", "label_smoothing",
            "target_error", "min_abstain_n", "eval_data",
        }
        assert set(kwargs) == expected

    def test_shuffle_options_becomes_a_tuple(self, defaults: Config) -> None:
        defaults.training.shuffle_options = ["choice"]
        assert defaults.training.to_train_config(1024, 256)["shuffle_options"] == ("choice",)

    def test_config_values_override_checkpoint_defaults(self) -> None:
        settings = TrainingConfig(max_len=512, head_max_len=128)
        kwargs = settings.to_train_config(1024, 256)
        assert kwargs["max_len"] == 512
        assert kwargs["head_max_len"] == 128

    def test_none_falls_back_to_the_checkpoint_budget(self) -> None:
        kwargs = TrainingConfig().to_train_config(1024, 256)
        assert kwargs["max_len"] == 1024
        assert kwargs["head_max_len"] == 256

    def test_effective_batch(self) -> None:
        assert TrainingConfig(micro_batch=4, grad_accum=16).effective_batch() == 64

    def test_optimizer_update_formula(self) -> None:
        settings = TrainingConfig(epochs=8, micro_batch=4, grad_accum=16)
        # 1000 items -> 250 micro-batches -> ceil(250/16)=16 updates/epoch
        assert settings.optimizer_updates(1000) == 128

    def test_optimizer_updates_of_zero_items(self) -> None:
        assert TrainingConfig().optimizer_updates(0) == 0

    def test_optimizer_updates_rounds_a_partial_window_up(self) -> None:
        settings = TrainingConfig(epochs=1, micro_batch=8, grad_accum=8)
        assert settings.optimizer_updates(100) == 2


class TestPaths:
    def test_relative_paths_resolve_against_the_root(self, defaults: Config) -> None:
        defaults.project.root = "/tmp/project"
        assert defaults.corpus_dir == Path("/tmp/project/corpus")

    def test_absolute_paths_are_kept(self, defaults: Config) -> None:
        defaults.project.root = "/tmp/project"
        defaults.paths.output_dir = "/var/tmp/out"
        assert defaults.output_dir == Path("/var/tmp/out")

    def test_stage_dir_is_under_the_output_dir(self, defaults: Config) -> None:
        defaults.project.root = "/tmp/project"
        assert defaults.stage_dir("chunks") == Path("/tmp/project/outputs/chunks")


class TestEnvironmentOverrides:
    def test_env_var_replaces_the_configured_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LAYA_OLLAMA_ENDPOINT", "http://ollama.test:1234")
        monkeypatch.setenv("LAYA_OLLAMA_GENERATION_MODEL", "some-model")
        loaded = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=True)
        assert loaded.ollama.endpoint == "http://ollama.test:1234"
        assert loaded.ollama.generation_model == "some-model"

    def test_env_can_be_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYA_OLLAMA_ENDPOINT", "http://ollama.test:1234")
        loaded = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
        assert loaded.ollama.endpoint == "http://127.0.0.1:11434"

    def test_non_numeric_int_override_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LAYA_OLLAMA_NUM_CTX", "soon")
        with pytest.raises(ConfigError, match="not an integer"):
            load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=True)

    def test_a_numeric_override_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYA_OLLAMA_NUM_CTX", "4096")
        loaded = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=True)
        assert loaded.ollama.num_ctx == 4096


def test_config_hash_changes_with_content() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_sections_have_usable_defaults() -> None:
    negatives = NegativesConfig()
    assert negatives.strategies and negatives.ratio > 0
    assert EvaluationConfig().retrieval_ks == [1, 3, 5, 10]