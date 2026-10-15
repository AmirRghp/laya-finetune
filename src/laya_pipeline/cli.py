"""Command-line interface.

One subcommand per pipeline stage. Every stage is independently invokable and
writes its own artifacts, so the pipeline can be resumed from any point.

Four cost classes are marked in each command's help text, because they are the
thing an operator most needs to know before pressing enter (AGENTS.md 9.2):

``[local]``
    Reads and writes files only. No network, no GPU. Seconds.
``[ollama]``
    Calls a local Ollama model. Slow; billed in minutes, not compute.
``[weights]``
    Loads Laya weights. Downloads the checkpoint once, then runs inference.
``[gpu-train]``
    Consumes GPU time for hours. Never starts without ``--yes``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import load_config, load_training_profile_overrides
from .errors import ConfigError, PipelineError
from .ids import content_hash
from .io import file_sha256, read_jsonl, write_manifest
from .logging_setup import configure_logging, get_logger
from .tasks import TASKS, resolve_tasks

__all__ = ["build_parser", "main"]

LOGGER = get_logger("cli")

DEFAULT_CONFIG = "configs/default.yaml"

_EPILOG = """\
execution order from a clean environment:
  0. preflight            [ollama] one second; refuses to continue on a bad model name
  1. scan                 [local]
  2. process              [local]  (downloads the tokenizer once)
  3. generate             [ollama] slow
  4. split                [local]
  5. negatives            [ollama] slow
  6. label                [ollama] slow
  7. tasks --task sufficiency / routing   [ollama] slow
  8. build                [local]
  9. validate             [local]
 10. stats                [local]
 11. train --profile configs/dry_run.yaml   [weights] no training
 12. train --profile configs/smoke.yaml --yes   [gpu-train] minutes
 13. train --profile configs/full.yaml --yes    [gpu-train] hours
 14. evaluate classify / evaluate retrieval    [weights]

Step 0 costs one second and can save step 3 entirely: a mistyped model name
otherwise fails inside the retry loop, about a hundred seconds per chunk.

Nothing loads Laya weights or touches a GPU before step 11, and nothing trains
before step 12.
"""


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-pipeline",
        description=(
            "Persian RAG decision-model fine-tuning data pipeline for Laya Multilingual."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"path to the configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="logging verbosity (default: info)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help=(
            "override one configuration value, e.g. --set generation.questions_per_chunk=5. "
            "Repeatable. The value is parsed as JSON when possible, else as a string."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    def add_stage(name: str, help_text: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Register a stage with its cost class in both help and description.

        ``add_parser(help=...)`` only populates the parent's command list; the
        stage's own ``--help`` would otherwise show no description at all. The
        cost tag is repeated as the description so an operator who runs
        ``laya-pipeline <stage> --help`` sees it too.
        """
        return subparsers.add_parser(name, help=help_text, description=help_text, **kwargs)

    add_stage(
        "scan", "[local] list every Markdown document under the corpus, with hashes"
    )

    add_stage("process", "[local] parse and chunk every discovered document")

    generate = add_stage(
        "generate", "[ollama] generate Persian questions and key points per chunk"
    )
    generate.add_argument(
        "--no-resume",
        action="store_true",
        help="regenerate questions that already exist instead of resuming",
    )

    add_stage(
        "split", "[local] build positives and assign document-level train/val/test splits"
    )

    add_stage(
        "negatives",
        "[ollama] mine hard negatives inside each split's own chunk pool",
    )

    label = add_stage(
        "label", "[ollama] teacher-label mined negatives and write the review queue"
    )
    label.add_argument("--review-csv", help="where to write the review CSV")

    tasks_parser = add_stage(
        "tasks", "[ollama] run a downstream task: sufficiency or routing"
    )
    tasks_parser.add_argument(
        "--task", required=True, choices=["sufficiency", "routing"], help="which task to run"
    )
    tasks_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="chunks retrieved per query for the sufficiency task (default: 3)",
    )

    build = add_stage("build", "[local] write Laya-format JSONL per task and split")
    _add_task_argument(build)
    build.add_argument(
        "--positive-control",
        action="store_true",
        help=(
            "prefix each state with label=<answer>; — the official Laya diagnostic for a "
            "run stuck at the class prior. Diagnostic only; never use for a real checkpoint."
        ),
    )

    validate = add_stage("validate", "[local] validate the generated datasets")
    _add_task_argument(validate)
    validate.add_argument(
        "--with-laya",
        action="store_true",
        help=(
            "also check state token lengths with the real Laya tokenizer "
            "(downloads the tokenizer, not the weights)"
        ),
    )

    stats = add_stage("stats", "[local] dataset statistics and warnings")
    _add_task_argument(stats)

    train = add_stage(
        "train", "[weights]/[gpu-train] dry run or fine-tune via laya.train"
    )
    train.add_argument(
        "--task", default="relevance", choices=list(TASKS), help="which task's dataset to train"
    )
    train.add_argument(
        "--profile",
        help=(
            "a training config overriding the main one, e.g. configs/smoke.yaml. "
            "Omit for the defaults in the main config."
        ),
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "[weights] build items and print the optimizer-update budget without loading "
            "weights or training. Safe and fast; run this first."
        ),
    )
    train.add_argument(
        "--out", help="checkpoint output directory (required for a real run)"
    )
    train.add_argument(
        "--yes",
        action="store_true",
        help="confirm a real training run. Without this the command prints the plan and stops.",
    )
    train.add_argument(
        "--positive-control",
        action="store_true",
        help="train on the label-encoded control dataset under dataset/<task>-control/",
    )

    preflight = add_stage(
        "preflight",
        "[ollama] check the resolved config against what Ollama actually has",
    )
    preflight.add_argument(
        "--offline",
        action="store_true",
        help="skip the Ollama round trip and report only what the config says",
    )

    evaluate = add_stage("evaluate", "[weights] evaluate a checkpoint")
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_command", metavar="<what>")

    def add_evaluation(name: str, help_text: str) -> argparse.ArgumentParser:
        return evaluate_sub.add_parser(
            name, help=help_text, description=help_text
        )

    classify = add_evaluation(
        "classify", "[weights] score a labelled dataset against one checkpoint"
    )
    classify.add_argument("--checkpoint", required=True, help="checkpoint path or Hub id")
    classify.add_argument("--task", default=None, choices=list(TASKS))
    classify.add_argument("--label", help="name for this run in the report")
    classify.add_argument("--device", default="auto")
    classify.add_argument("--batch-size", type=int, default=None)
    classify.add_argument(
        "--out", default=None, help="write comparison.{json,md} into this directory"
    )

    retrieval = add_evaluation(
        "retrieval", "[weights]/[ollama] Recall@K, MRR and nDCG@K"
    )
    retrieval.add_argument("--split", default="test", choices=["train", "validation", "test"])
    retrieval.add_argument(
        "--rerank-with", default=None, help="a Laya checkpoint to rerank with, or omit"
    )
    retrieval.add_argument("--out", default=None, help="write retrieval.{json,md} here")

    return parser


def _add_task_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        default=None,
        choices=list(TASKS) + ["all"],
        help="which task to operate on (default: every configured task)",
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def _parse_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    import json

    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(
                f"--set expects KEY=VALUE, got {pair!r} (for example "
                "--set generation.questions_per_chunk=5)"
            )
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"--set got an empty key: {pair!r}")
        try:
            overrides[key] = json.loads(raw)
        except ValueError:
            overrides[key] = raw
    return overrides


def _tasks_from(args: argparse.Namespace) -> List[str]:
    try:
        return resolve_tasks([args.task] if getattr(args, "task", None) else None)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _print(payload: Any) -> None:
    """Human-readable stdout summary. Library code logs; the CLI prints."""
    import json

    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(payload)


def _run(args: argparse.Namespace) -> Any:
    from . import stages

    command = args.command
    tasks = _tasks_from(args) if hasattr(args, "task") else list(TASKS)

    if command == "scan":
        return stages.stage_scan(args.config_obj)
    if command == "process":
        return stages.stage_process(args.config_obj)
    if command == "generate":
        return stages.stage_generate(args.config_obj, resume=not args.no_resume)
    if command == "split":
        return stages.stage_split(args.config_obj)
    if command == "negatives":
        return stages.stage_negatives(args.config_obj)
    if command == "label":
        return stages.stage_label(args.config_obj, review_dir=args.review_csv)
    if command == "tasks":
        return stages.stage_tasks(args.config_obj, args.task, top_k=args.top_k)
    if command == "build":
        return stages.stage_build(args.config_obj, tasks, positive_control=args.positive_control)
    if command == "validate":
        return stages.stage_validate(args.config_obj, tasks, with_laya=args.with_laya)
    if command == "stats":
        return stages.stage_stats(args.config_obj, tasks)
    if command == "preflight":
        return _run_preflight(args)
    if command == "train":
        return _run_train(args)
    if command == "evaluate":
        return _run_evaluate(args)
    raise ConfigError(f"unknown command {command!r}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _run_preflight(args: argparse.Namespace) -> int:
    """Report whether this configuration can work, before anything expensive.

    Deliberately exits non-zero on an error finding, so it can gate a script:
    ``laya-pipeline preflight && laya-pipeline generate``.
    """
    from . import diagnostics
    from .llm.cache import NullCache
    from .llm.ollama import OllamaClient

    config = args.config_obj
    logger = get_logger(__name__)

    if args.offline:
        # Report what the config asserts without claiming the models exist.
        ollama = config.ollama
        _print(
            {
                "checked": "configuration only (--offline: models not verified)",
                "endpoint": ollama.endpoint,
                "generation_model": ollama.generation_model,
                "teacher_model": ollama.teacher_model,
                "embedding_model": ollama.embedding_model,
                "think": ollama.think,
                "num_ctx": ollama.num_ctx,
            }
        )
        return 0

    client = OllamaClient(config.ollama, cache=NullCache())
    findings = diagnostics.run_preflight(config, client)

    for finding in findings:
        level = {"error": "error", "warning": "warning", "info": "info"}.get(
            finding.level, "info"
        )
        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            "preflight",
            extra={"fields": {"finding": str(finding)}},
        )
    _print({"summary": diagnostics.summarize(findings)})

    if diagnostics.has_errors(findings):
        logger.error(
            "preflight found blocking problems; fix them before running generate",
            extra={"summary": diagnostics.summarize(findings)},
        )
        return 1
    logger.info("preflight passed", extra={"summary": diagnostics.summarize(findings)})
    return 0


def _run_train(args: argparse.Namespace) -> Any:
    from . import stages, training

    config = args.config_obj
    task = args.task
    suffix = "-control" if args.positive_control else ""
    directory = config.stage_dir("dataset") / f"{task}{suffix}"
    data_path = directory / "train.jsonl"
    eval_path = directory / "validation.jsonl"
    if not data_path.is_file():
        raise PipelineError(
            f"{data_path} does not exist. Run `build --task {task}` first."
        )
    if not eval_path.is_file():
        LOGGER.warning(
            "no validation split; Laya will report in-sample calibration numbers, "
            "which are not held-out results",
            extra={"fields": {"expected": str(eval_path)}},
        )

    plan = training.build_plan(
        config,
        data_path,
        output_dir=Path(args.out) if args.out else None,
        eval_path=eval_path if eval_path.is_file() else None,
        profile=args.profile or "",
    )

    if args.dry_run:
        summary = training.run_dry_run(config, plan)
    else:
        # Count usable decision items before reporting the budget. This reads
        # the data and tokenizer only; it does not start model training.
        plan.estimated_train_items = training.count_training_items(config, data_path)

    print("=" * 72)
    print(plan.describe())
    print("=" * 72)

    if args.dry_run:
        return {"mode": "dry-run", "summary": summary}

    if not args.yes:
        print(
            "\nThis is a REAL training run and it has NOT been started.\n"
            "Re-run with --yes to begin. Use --dry-run first if you have not yet "
            "checked the data."
        )
        return {"mode": "planned", "started": False}

    if not args.out:
        raise PipelineError("a real training run needs --out <checkpoint directory>")

    stages._load_split_report_obj(config)
    if not eval_path.is_file():
        raise PipelineError(
            "a real training run requires a validation JSONL file for held-out eval_data"
        )
    eval_rows = list(read_jsonl(eval_path))
    if not any(
        isinstance(row.get("state"), str) and row["state"].strip()
        for row in eval_rows
    ):
        raise PipelineError(
            f"validation file {eval_path} is empty or has no usable rows; real training "
            "requires non-empty held-out eval_data"
        )

    summary = training.run_finetune(config, plan)
    report = summary.get("train_report") or {}
    return {
        "mode": "finetune",
        "started": True,
        "output_dir": summary.get("output_dir"),
        "train_items": summary.get("train_items"),
        "calibration_items": summary.get("calibration_items"),
        "eval_mode": report.get("eval_mode"),
        "is_held_out": report.get("is_held_out"),
        "note": report.get("note"),
        "calibration_warnings": training.calibration_warnings(report),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _run_evaluate(args: argparse.Namespace) -> Any:
    from .evaluation import compare_runs, evaluate_checkpoint, render_markdown, write_report
    from . import stages

    config = args.config_obj
    what = getattr(args, "evaluate_command", None)
    if what == "classify":
        tasks = [args.task] if args.task else None
        target = tasks[0] if tasks else "relevance"
        data_path = config.stage_dir("dataset") / target / "test.jsonl"
        if not data_path.is_file():
            raise PipelineError(
                f"{data_path} does not exist. Run `build --task {target}` first."
            )
        stages._load_split_report_obj(config)
        runs: Dict[str, Any] = {}
        label = args.label or args.checkpoint
        runs[label] = evaluate_checkpoint(
            config,
            data_path,
            args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            task=args.task,
            label=label,
        )
        if len(runs) == 1 and config.evaluation.base_model != args.checkpoint:
            # Score the untouched base checkpoint too, so the comparison exists.
            runs["base"] = evaluate_checkpoint(
                config,
                data_path,
                config.evaluation.base_model,
                device=args.device,
                batch_size=args.batch_size,
                task=args.task,
                label="base",
            )
            runs = {"base": runs["base"], label: runs[label]}
        comparison = compare_runs(runs, title=f"Classification — {target} / test split")
        if args.out:
            directory = Path(args.out)
            write_report(comparison, directory, config.corpus_dir)
            write_manifest(
                directory,
                "evaluate:classify",
                config_hash=config.config_hash,
                seed=config.project.seed,
                counts={"runs": len(runs)},
                extra={"task": target},
                inputs={
                    "test_dataset_sha256": file_sha256(data_path),
                    "checkpoint_id_sha256": content_hash(args.checkpoint, fold=False),
                    "base_checkpoint_id_sha256": content_hash(
                        config.evaluation.base_model, fold=False
                    ),
                },
                corpus_dir=config.corpus_dir,
            )
        else:
            print(render_markdown(comparison))
        return comparison

    if what == "retrieval":
        from .evaluation import evaluate_retrieval

        stages._load_split_report_obj(config)
        examples = stages.load_examples(config)
        chunks = stages.load_chunks(config)
        from .evaluation import build_queries

        queries = build_queries(examples, args.split)
        report = evaluate_retrieval(config, chunks, queries, rerank_with=args.rerank_with)
        if args.out:
            directory = Path(args.out)
            from .io import ensure_dir, write_json

            ensure_dir(directory)
            write_json(directory / "retrieval.json", report, config.corpus_dir)
            write_manifest(
                directory,
                "evaluate:retrieval",
                config_hash=config.config_hash,
                seed=config.project.seed,
                counts={
                    "queries": report["queries_scored"],
                    "indexed_chunks": report["indexed_chunks"],
                },
                inputs={
                    "labeled_examples_sha256": file_sha256(
                        config.stage_dir("labels") / "labeled.jsonl"
                    ),
                    "chunks_sha256": file_sha256(
                        config.stage_dir("chunks") / "chunks.jsonl"
                    ),
                    "split_sha256": file_sha256(
                        config.stage_dir("split") / "split.json"
                    ),
                },
                corpus_dir=config.corpus_dir,
            )
        return report

    raise ConfigError("evaluate needs a subcommand: `classify` or `retrieval`")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if not args.command:
        parser.print_help()
        return 2

    try:
        overrides: Dict[str, Any] = {}
        profile_path = getattr(args, "profile", None)
        if args.command == "train" and profile_path:
            overrides.update(load_training_profile_overrides(profile_path))
        overrides.update(_parse_overrides(args.overrides))
        config = load_config(args.config, overrides=overrides)
    except PipelineError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    args.config_obj = config
    LOGGER.info(
        "starting",
        extra={
            "fields": {
                "command": args.command,
                "config": args.config,
                "corpus_dir": str(config.corpus_dir),
                "output_dir": str(config.output_dir),
            }
        },
    )

    try:
        result = _run(args)
    except PipelineError as exc:
        LOGGER.error("stage failed", extra={"fields": {"error": str(exc)}})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        problems = getattr(exc, "problems", None)
        if problems:
            for problem in problems[:20]:
                print(f"  - {problem}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if args.command == "preflight":
        # Its exit code is the answer, not a stage result to print again.
        return int(result)

    _print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
