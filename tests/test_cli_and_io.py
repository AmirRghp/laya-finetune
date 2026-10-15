"""CLI parsing, configuration plumbing and safe artifact writes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from laya_pipeline.cli import _run_train, build_parser, main
from laya_pipeline.errors import ConfigError, LeakageError, PipelineError
from laya_pipeline.io import (
    read_json,
    read_jsonl,
    read_text,
    resolve_output_path,
    write_csv,
    write_json,
    write_jsonl,
    write_manifest,
    write_text,
)

from tests.conftest import REPO_ROOT


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class TestParser:
    def test_no_command_exits_with_a_usage_code(self, capsys) -> None:
        assert main([]) == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_every_stage_is_reachable(self) -> None:
        parser = build_parser()
        actions = [a for a in parser._actions if a.dest == "command"]
        choices = set(actions[0].choices)
        assert {
            "scan", "process", "generate", "split", "negatives", "label",
            "tasks", "build", "validate", "stats", "train", "evaluate",
        } <= choices

    def test_config_flag_defaults_to_the_shipped_file(self) -> None:
        assert build_parser().parse_args(["scan"]).config == "configs/default.yaml"

    def test_train_accepts_a_profile(self) -> None:
        args = build_parser().parse_args(
            ["train", "--profile", "configs/smoke.yaml", "--dry-run"]
        )
        assert args.profile == "configs/smoke.yaml"
        assert args.dry_run is True

    def test_train_profile_is_applied_and_set_wins(self, monkeypatch, repo_root) -> None:
        from laya_pipeline import cli

        captured = []
        monkeypatch.setattr(cli, "_run", lambda args: captured.append(args.config_obj) or {})
        common = ["--config", str(repo_root / "configs" / "default.yaml")]
        train_args = [
            "train", "--profile", str(repo_root / "configs" / "dry_run.yaml"),
            "--dry-run",
        ]

        assert main([*common, *train_args]) == 0
        assert captured[-1].training.epochs == 1

        assert main([*common, "--set", "training.epochs=2", *train_args]) == 0
        assert captured[-1].training.epochs == 2

    def test_dry_run_prints_plan_after_counting_items(self, config, monkeypatch, capsys) -> None:
        from laya_pipeline import training

        directory = config.stage_dir("dataset") / "relevance"
        directory.mkdir(parents=True)
        write_text(directory / "train.jsonl", "{}\n")
        write_text(directory / "validation.jsonl", "{}\n")

        def fake_dry_run(_config, plan):
            plan.estimated_train_items = 17
            return {"train_items": 17}

        monkeypatch.setattr(training, "run_dry_run", fake_dry_run)
        args = build_parser().parse_args(["train", "--task", "relevance", "--dry-run"])
        args.config_obj = config

        result = _run_train(args)

        assert result["mode"] == "dry-run"
        output = capsys.readouterr().out
        assert "estimated train items: 17" in output
        assert "optimizer updates    : 8" in output

    def test_real_run_counts_items_before_printing_budget(
        self, config, monkeypatch, capsys
    ) -> None:
        from laya_pipeline import stages, training

        directory = config.stage_dir("dataset") / "relevance"
        directory.mkdir(parents=True)
        write_text(directory / "train.jsonl", '{"state":"train"}\n')
        write_text(directory / "validation.jsonl", '{"state":"validation"}\n')

        monkeypatch.setattr(stages, "_load_split_report_obj", lambda _config: {})
        monkeypatch.setattr(training, "count_training_items", lambda _config, _path: 65)

        def fake_finetune(_config, plan):
            assert plan.estimated_train_items == 65
            return {"train_report": {}}

        monkeypatch.setattr(training, "run_finetune", fake_finetune)
        args = build_parser().parse_args(
            ["train", "--task", "relevance", "--yes", "--out", "synthetic-output"]
        )
        args.config_obj = config

        result = _run_train(args)

        assert result["mode"] == "finetune"
        output = capsys.readouterr().out
        assert "estimated train items: 65" in output
        assert "optimizer updates    : 16" in output

    def test_train_refuses_without_yes_by_default(self) -> None:
        assert build_parser().parse_args(["train"]).yes is False

    def test_task_argument_accepts_all(self) -> None:
        assert build_parser().parse_args(["build", "--task", "all"]).task == "all"

    def test_unknown_task_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["build", "--task", "telepathy"])

    def test_evaluate_has_subcommands(self) -> None:
        args = build_parser().parse_args(["evaluate", "classify", "--checkpoint", "x"])
        assert args.evaluate_command == "classify"

    def test_help_documents_the_execution_order(self, capsys) -> None:
        """The epilog is the operator's map of cost classes; it must survive.

        argparse hard-wraps help text to the terminal width, so the check is on
        words rather than on whole commands.
        """
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--help"])
        text = capsys.readouterr().out
        for expected in ("preflight", "process", "generate", "negatives", "train"):
            assert expected in text
        assert "[ollama]" in text
        assert "[gpu-train]" in text
        assert "Nothing loads Laya weights" in text

    def test_preflight_is_step_zero_in_the_execution_order(self, capsys) -> None:
        """It is listed first because it is the cheapest step that saves the
        most time: one second now, versus a retry loop per chunk later."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--help"])
        text = capsys.readouterr().out
        assert "0. preflight" in text.replace("  0.", "0.")

    def test_each_stage_declares_its_cost_class(self) -> None:
        """Every stage's description says what it needs, per AGENTS.md 9.2."""
        cost_classes = ("[local]", "[ollama]", "[weights]", "[gpu-train]")
        for name, summary in _stage_help().items():
            assert any(tag in summary for tag in cost_classes), (
                f"command {name!r} declares no cost class: {summary!r}"
            )

    def test_the_expensive_stages_are_tagged_ollama(self) -> None:
        help_text = _stage_help()
        for name in ("generate", "negatives", "label", "tasks"):
            assert "[ollama]" in help_text[name], f"{name} should be tagged [ollama]"

    def test_training_is_tagged_as_costly(self) -> None:
        assert "[gpu-train]" in _stage_help()["train"]
        assert "[weights]" in _stage_help()["evaluate"]

    def test_evaluation_subcommands_are_tagged(self) -> None:
        nested = _nested_help()
        assert "[weights]" in nested["classify"]
        assert "Recall@K" in nested["retrieval"]


def _stage_help() -> dict[str, str]:
    """Each command's own description, as shown by ``<command> --help``."""
    parser = build_parser()
    action = next(
        a
        for a in parser._actions
        if hasattr(a, "choices") and isinstance(a.choices, dict) and "scan" in a.choices
    )
    return {name: (sub.description or "") for name, sub in action.choices.items()}


def _nested_help() -> dict[str, str]:
    """The same, for ``laya-pipeline evaluate <what>``."""
    parser = build_parser()
    action = next(
        a
        for a in parser._actions
        if hasattr(a, "choices") and isinstance(a.choices, dict) and "scan" in a.choices
    )
    evaluate = action.choices["evaluate"]
    sub = next(
        a
        for a in evaluate._actions
        if hasattr(a, "choices") and isinstance(a.choices, dict) and "classify" in a.choices
    )
    return {name: (p.description or "") for name, p in sub.choices.items()}


class TestOverrides:
    def test_json_values_are_parsed(self) -> None:
        from laya_pipeline.cli import _parse_overrides

        assert _parse_overrides(["a.b=3"]) == {"a.b": 3}
        assert _parse_overrides(["a.b=true"]) == {"a.b": True}
        assert _parse_overrides(["a.b=مرخصی"]) == {"a.b": "مرخصی"}

    def test_a_missing_equals_is_rejected(self) -> None:
        from laya_pipeline.cli import _parse_overrides

        with pytest.raises(ConfigError, match="KEY=VALUE"):
            _parse_overrides(["oops"])


class TestEndToEndWithoutOllama:
    """scan and process must work with no network, no GPU and no model."""

    def test_scan_then_process(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        write_text(corpus / "a.md", "# سند\n\n## بخش\n\nمتن فارسی برای آزمون.\n")

        assert main(["--config", str(config_path), "scan"]) == 0
        assert (tmp_path / "outputs" / "scan" / "documents.jsonl").is_file()

        assert main(["--config", str(config_path), "process"]) == 0
        chunks_path = tmp_path / "outputs" / "chunks" / "chunks.jsonl"
        assert chunks_path.is_file()
        rows = list(read_jsonl(chunks_path))
        assert rows and all("chunk_id" in row for row in rows)

    def test_process_records_that_counts_were_estimated(self, tmp_path: Path) -> None:
        """No network in this suite, so the estimate must be labelled as such."""
        config_path = _write_config(tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        write_text(corpus / "a.md", "# سند\n\n## بخش\n\nمتن فارسی برای آزمون.\n")

        assert main(["--config", str(config_path), "process"]) == 0
        payload = read_json(tmp_path / "outputs" / "chunks" / "process.json")
        assert payload["chunking"]["token_source"] == "estimated"

        rows = list(read_jsonl(tmp_path / "outputs" / "chunks" / "chunks.jsonl"))
        assert all(row["token_count_source"] == "estimated" for row in rows)

    def test_reprocessing_is_idempotent(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        write_text(corpus / "a.md", "# سند\n\n## بخش\n\nمتن فارسی برای آزمون.\n")

        assert main(["--config", str(config_path), "process"]) == 0
        first = read_text(tmp_path / "outputs" / "chunks" / "chunks.jsonl")
        assert main(["--config", str(config_path), "process"]) == 0
        second = read_text(tmp_path / "outputs" / "chunks" / "chunks.jsonl")
        assert first == second

    def test_a_missing_config_exits_with_code_two(self, tmp_path: Path) -> None:
        assert main(["--config", str(tmp_path / "absent.yaml"), "scan"]) == 2

    def test_an_invalid_config_exits_with_code_two(self, tmp_path: Path) -> None:
        import yaml as yaml_module

        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml_module.safe_dump({"ollama": {"endpoint": "not-a-url"}}), encoding="utf-8"
        )
        assert main(["--config", str(bad), "scan"]) == 2

    def test_an_unknown_override_key_exits_with_code_two(self, tmp_path: Path) -> None:
        assert main(
            [
                "--config",
                str(REPO_ROOT / "configs" / "default.yaml"),
                "--set",
                "generation.nonexistent=1",
                "scan",
            ]
        ) == 2

    def test_a_malformed_override_exits_with_code_two(self) -> None:
        assert main(["--set", "nonsense", "scan"]) == 2

    def test_a_stage_failure_exits_with_code_one(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, corpus_dir="absent-corpus")
        assert main(["--config", str(config_path), "scan"]) == 1

    def test_generate_without_chunks_explains_what_to_run(
        self, tmp_path: Path, capsys
    ) -> None:
        config_path = _write_config(tmp_path)
        assert main(["--config", str(config_path), "generate"]) == 1
        assert "process" in capsys.readouterr().err

    def test_evaluate_without_a_dataset_explains_what_to_run(
        self, tmp_path: Path, capsys
    ) -> None:
        config_path = _write_config(tmp_path)
        code = main(
            [
                "--config",
                str(config_path),
                "evaluate",
                "classify",
                "--checkpoint",
                "some/model",
            ]
        )
        assert code == 1
        assert "build" in capsys.readouterr().err

    def test_train_without_a_dataset_explains_what_to_run(
        self, tmp_path: Path, capsys
    ) -> None:
        config_path = _write_config(tmp_path)
        assert main(["--config", str(config_path), "train", "--dry-run"]) == 1
        assert "build" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Artifact safety
# --------------------------------------------------------------------------- #
class TestCorpusProtection:
    def test_writing_into_the_corpus_is_refused(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        with pytest.raises(ConfigError, match="inside the corpus"):
            write_json(corpus / "out.json", {"a": 1}, corpus)

    def test_writing_to_the_corpus_itself_is_refused(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        with pytest.raises(ConfigError, match="corpus directory itself"):
            resolve_output_path(corpus, corpus)

    def test_writing_beside_the_corpus_is_allowed(self, tmp_path: Path) -> None:
        """A sibling of the corpus is a legitimate output location.

        The guard refuses paths *inside* the corpus, not its parent — refusing
        the parent would block every ordinary ``outputs/`` directory next to it.
        """
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        assert write_text(tmp_path / "sibling.md", "متن", corpus).is_file()

    def test_writing_over_the_corpus_parent_directory_is_refused(self, tmp_path: Path) -> None:
        """Targeting the directory that *holds* the corpus is refused.

        Writers address files, so this fires only when the path resolves to that
        directory itself — which would replace the whole tree rather than add to
        it. Files placed alongside the corpus are fine; see the sibling test.
        """
        root = tmp_path / "project"
        corpus = root / "corpus"
        corpus.mkdir(parents=True)
        with pytest.raises(ConfigError, match="contains the corpus"):
            write_json(root, {"a": 1}, corpus)

    def test_a_sibling_directory_is_allowed(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        assert write_json(tmp_path / "out" / "x.json", {"a": 1}, corpus).is_file()

    def test_every_writer_is_guarded(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        with pytest.raises(ConfigError):
            write_jsonl(corpus / "x.jsonl", [{"a": 1}], corpus)
        with pytest.raises(ConfigError):
            write_csv(corpus / "x.csv", [{"a": 1}], ["a"], corpus)
        with pytest.raises(ConfigError):
            write_manifest(corpus, "stage", config_hash="h", seed=0, counts={}, corpus_dir=corpus)
        with pytest.raises(ConfigError):
            write_text(corpus / "x.md", "متن", corpus)


def _write_config(tmp_path: Path, *, corpus_dir: str = "corpus") -> Path:
    """A valid config rooted at ``tmp_path``, reusing the shipped questions.

    ``process`` needs a tokenizer, which this suite must not download, so the
    counter is forced to the character estimate for these tests.
    """
    import yaml as yaml_module

    from laya_pipeline.config import load_config

    base = load_config(REPO_ROOT / "configs" / "default.yaml", apply_env=False)
    payload = base.to_dict()
    payload["project"]["root"] = str(tmp_path)
    payload["paths"] = {
        "corpus_dir": corpus_dir,
        "output_dir": "outputs",
        "prompt_dir": str(REPO_ROOT / "prompts"),
    }
    payload["chunking"]["min_chunk_chars"] = 0

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml_module.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


class TestAtomicWrites:
    def test_json_round_trips_with_persian(self, tmp_path: Path) -> None:
        path = write_json(tmp_path / "a.json", {"متن": "مرخصی"})
        assert read_json(path)["متن"] == "مرخصی"

    def test_persian_is_not_escaped(self, tmp_path: Path) -> None:
        path = write_json(tmp_path / "a.json", {"متن": "مرخصی"})
        assert "مرخصی" in read_text(path)

    def test_jsonl_skips_blank_and_comment_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        write_text(path, '{"a": 1}\n\n# comment\n{"a": 2}\n')
        assert [row["a"] for row in read_jsonl(path)] == [1, 2]

    def test_invalid_jsonl_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        write_text(path, '{"a": 1}\nnot json\n')
        with pytest.raises(PipelineError, match=":2"):
            list(read_jsonl(path))

    def test_a_non_object_row_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        write_text(path, "[1, 2]\n")
        with pytest.raises(PipelineError, match="expected an object"):
            list(read_jsonl(path))

    def test_no_temporary_files_remain(self, tmp_path: Path) -> None:
        write_json(tmp_path / "a.json", {"a": 1})
        assert not list(tmp_path.glob("*.tmp"))

    def test_a_failed_write_leaves_no_partial_file(self, tmp_path: Path) -> None:
        target = tmp_path / "a.json"

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            write_json(target, {"bad": Unserialisable()})
        assert not target.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_an_existing_file_survives_a_failed_rewrite(self, tmp_path: Path) -> None:
        target = write_json(tmp_path / "a.json", {"good": 1})

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            write_json(target, {"bad": Unserialisable()})
        assert read_json(target) == {"good": 1}

    def test_manifest_records_its_inputs(self, tmp_path: Path) -> None:
        write_manifest(
            tmp_path, "build", config_hash="h", seed=1,
            counts={"rows": 3}, inputs={"base": "x"},
        )
        payload = read_json(tmp_path / "manifest.json")
        assert payload["stage"] == "build"
        assert payload["inputs"]["base"] == "x"
        assert payload["stage_version"]


class TestReads:
    def test_bom_is_handled(self, tmp_path: Path) -> None:
        path = tmp_path / "a.md"
        path.write_bytes("﻿متن".encode("utf-8"))
        assert read_text(path) == "متن"

    def test_invalid_utf8_is_typed(self, tmp_path: Path) -> None:
        path = tmp_path / "a.md"
        path.write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(PipelineError, match="not valid UTF-8"):
            read_text(path)

    def test_a_missing_file_is_typed(self, tmp_path: Path) -> None:
        with pytest.raises(PipelineError, match="cannot read"):
            read_text(tmp_path / "absent.md")


def test_shipped_prompts_exist() -> None:
    assert (REPO_ROOT / "prompts" / "questions.fa.md").is_file()


def test_shipped_configs_exist() -> None:
    for name in ("default", "dry_run", "smoke", "full"):
        assert (REPO_ROOT / "configs" / f"{name}.yaml").is_file()


def test_real_training_rejects_missing_or_empty_validation(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from laya_pipeline import training

    task_dir = config.stage_dir("dataset") / "relevance"
    task_dir.mkdir(parents=True)
    write_jsonl(task_dir / "train.jsonl", [{"state": "نمونه"}])
    override_eval = tmp_path / "non_validation_eval.jsonl"
    write_jsonl(override_eval, [{"state": "نمونه"}])
    config.training.eval_data = str(override_eval)
    split_dir = config.stage_dir("split")
    split_dir.mkdir(parents=True)
    write_json(split_dir / "split.json", {"leakage": []})
    monkeypatch.setattr(
        training,
        "run_finetune",
        lambda *_args: pytest.fail("fine-tuning must not run without validation rows"),
    )
    args = argparse.Namespace(
        config_obj=config,
        task="relevance",
        positive_control=False,
        out=str(tmp_path / "checkpoint"),
        profile="",
        dry_run=False,
        yes=True,
    )
    with pytest.raises(PipelineError, match="requires a validation JSONL"):
        _run_train(args)

    write_jsonl(task_dir / "validation.jsonl", [])
    with pytest.raises(PipelineError, match="is empty"):
        _run_train(args)

    write_jsonl(task_dir / "validation.jsonl", [{"state": ""}])
    with pytest.raises(PipelineError, match="no usable rows"):
        _run_train(args)


def test_leaky_split_blocks_dataset_build(config) -> None:
    from laya_pipeline import stages

    split_dir = config.stage_dir("split")
    split_dir.mkdir(parents=True)
    write_json(split_dir / "split.json", {"leakage": [{"kind": "duplicate_query"}]})
    with pytest.raises(LeakageError, match="leakage finding"):
        stages.stage_build(config, ["relevance"])


def test_validate_and_stats_write_stage_manifests(config) -> None:
    from laya_pipeline import stages

    split_dir = config.stage_dir("split")
    split_dir.mkdir(parents=True)
    write_json(split_dir / "split.json", {"leakage": []})
    stages.stage_validate(config, [])
    dataset_dir = config.stage_dir("dataset")
    validation_manifest = read_json(dataset_dir / "validation.manifest.json")
    assert validation_manifest["stage"] == "validate"
    assert validation_manifest["stage_version"]

    labels_dir = config.stage_dir("labels")
    labels_dir.mkdir(parents=True)
    write_jsonl(labels_dir / "labeled.jsonl", [])
    chunks_dir = config.stage_dir("chunks")
    chunks_dir.mkdir(parents=True)
    write_jsonl(chunks_dir / "chunks.jsonl", [])
    stages.stage_stats(config, [])
    stats_manifest = read_json(dataset_dir / "stats.manifest.json")
    assert stats_manifest["stage"] == "stats"
    assert stats_manifest["stage_version"]
