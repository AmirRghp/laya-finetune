"""Fine-tuning entry point.

Talks to ``laya.train`` directly rather than shelling out to ``laya-train``,
because the CLI cannot express ``amp``, ``gradient_checkpointing``, ``calib_max``
or ``calib_frac``. Going through the CLI would silently apply defaults the
operator did not choose (AGENTS.md 3.3).

Nothing here starts a run on its own. :func:`run_finetune` is only reached after
the CLI has confirmed ``--yes``, and the optimizer-update budget is printed
beforehand so a multi-hour run is a decision rather than a surprise
(AGENTS.md 3.13).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .errors import TrainingError
from .logging_setup import get_logger

__all__ = [
    "TrainingPlan",
    "require_laya",
    "build_plan",
    "run_dry_run",
    "run_finetune",
    "describe_plan",
]

LOGGER = get_logger("training")

_LAYA_IMPORT_HINT = (
    "The `laya` package (and torch) are required to train or evaluate. Install them "
    "with:\n\n    pip install -e .[laya]\n\n"
    "The data pipeline itself — discovery, chunking, generation, splitting, dataset "
    "building and validation — runs without them."
)


def require_laya() -> Any:
    """Import ``laya.train`` lazily, with an actionable message when it is absent.

    Imported inside the function, never at module scope, so that every stage
    before training works without torch installed (AGENTS.md 5.7).
    """
    try:
        from laya.train import TrainConfig, dry_run, finetune
    except ImportError as exc:
        raise TrainingError(_LAYA_IMPORT_HINT) from exc
    return TrainConfig, dry_run, finetune


def require_laya_root() -> Any:
    """Import the top-level ``laya`` package."""
    try:
        import laya
    except ImportError as exc:
        raise TrainingError(_LAYA_IMPORT_HINT) from exc
    return laya


@dataclass
class TrainingPlan:
    """Everything a run will do, resolved before anything runs."""

    base_model: str
    data_path: Path
    eval_path: Optional[Path]
    output_dir: Optional[Path]
    train_config_kwargs: Dict[str, Any]
    max_len: int
    head_max_len: int
    estimated_train_items: int = 0
    profile: str = ""

    def optimizer_updates(self, epochs: int, micro_batch: int, grad_accum: int) -> int:
        items = max(0, self.estimated_train_items)
        if items == 0:
            return 0
        micro_batches = -(-items // micro_batch)
        updates_per_epoch = -(-micro_batches // grad_accum)
        return epochs * updates_per_epoch

    def describe(self) -> str:
        settings = self.train_config_kwargs
        effective_batch = settings["micro_batch"] * settings["grad_accum"]
        updates = self.optimizer_updates(
            settings["epochs"], settings["micro_batch"], settings["grad_accum"]
        )
        eval_note = (
            str(self.eval_path)
            if self.eval_path
            else "(none — Laya will report in-sample calibration numbers, "
            "which are NOT held-out results)"
        )
        lines = [
            f"base checkpoint      : {self.base_model}",
            f"profile              : {self.profile or '(config default)'}",
            f"training data        : {self.data_path}",
            f"eval data            : {eval_note}",
            f"output               : {self.output_dir or '(dry run)'}",
            f"epochs               : {settings['epochs']}",
            f"micro batch          : {settings['micro_batch']}",
            f"grad accum           : {settings['grad_accum']}",
            f"effective batch      : {effective_batch}",
            f"loss                 : {settings['loss']}",
            f"encoder lr           : {settings['encoder_lr']}",
            f"head lr              : {settings['head_lr']}",
            f"freeze encoder       : {settings['freeze_encoder']}",
            f"amp                  : {settings['amp']}",
            f"grad checkpointing   : {settings['gradient_checkpointing']}",
            f"max_len / head_max   : {self.max_len} / {self.head_max_len}",
            f"shuffle options      : {list(settings['shuffle_options']) or '(off)'}",
            f"calibration slice    : calib_max={settings['calib_max']}, "
            f"calib_frac={settings['calib_frac']}",
            f"seed                 : {settings['seed']}",
            "",
            f"estimated train items: {self.estimated_train_items}",
            f"optimizer updates    : {updates}",
        ]
        return "\n".join(lines)


def build_plan(
    config: Config,
    data_path: Path,
    *,
    output_dir: Optional[Path] = None,
    eval_path: Optional[Path] = None,
    profile: str = "",
    positive_control: bool = False,
) -> TrainingPlan:
    """Resolve a run without executing it."""
    settings = config.training
    eval_path = eval_path or (
        Path(settings.eval_data) if settings.eval_data else None
    )
    kwargs = settings.to_train_config(config.laya.max_len, config.laya.head_max_len)
    kwargs["eval_data"] = str(eval_path) if eval_path else None

    plan = TrainingPlan(
        base_model=config.laya.base_model,
        data_path=Path(data_path),
        eval_path=Path(eval_path) if eval_path else None,
        output_dir=Path(output_dir) if output_dir else None,
        train_config_kwargs=kwargs,
        max_len=int(kwargs["max_len"]),
        head_max_len=int(kwargs["head_max_len"]),
        profile=profile,
    )
    del positive_control  # the control lives in the dataset, not the run config
    return plan


def count_training_items(config: Config, data_path: Path) -> int:
    """Build training items with the real tokenizer to size the run.

    This is what makes the printed budget trustworthy: Laya counts *items*, not
    rows, and a row whose options do not fit ``head_max_len`` is dropped.
    """
    TrainConfig, _dry_run, _finetune = require_laya()
    import os

    from laya.train import items_from_rows, read_data, split_calibration

    settings = config.training
    train_config = TrainConfig(
        **settings.to_train_config(config.laya.max_len, config.laya.head_max_len)
    )
    train_config.validate()

    rows = read_data(
        str(data_path),
        text_column=train_config.text_column,
        label_column=train_config.label_column,
        question_id=train_config.question_id,
        instructions=train_config.instructions,
    )
    from transformers import AutoTokenizer

    from laya.train import resolve_checkpoint_dir

    resolved = resolve_checkpoint_dir(config.laya.base_model)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(resolved, "tokenizer"))

    items, _skipped = items_from_rows(
        tokenizer,
        rows,
        int(train_config.max_len),
        int(train_config.head_max_len),
        label_smoothing=train_config.label_smoothing,
    )
    train_items, calibration = split_calibration(
        items, train_config.calib_max, train_config.calib_frac, train_config.calib_seed
    )
    LOGGER.info(
        "counted training items",
        extra={
            "fields": {
                "items": len(items),
                "train": len(train_items),
                "calibration": len(calibration),
            }
        },
    )
    return len(train_items)


def describe_plan(plan: TrainingPlan) -> str:
    return plan.describe()


def run_dry_run(config: Config, plan: TrainingPlan) -> Dict[str, Any]:
    """Build items and print the budget without loading weights.

    Uses the official ``laya.train.dry_run``, so the numbers here are the same
    ones a real run would compute.
    """
    TrainConfig, dry_run, _finetune = require_laya()
    train_config = TrainConfig(**plan.train_config_kwargs)
    train_config.validate()

    summary = dry_run(
        data=str(plan.data_path),
        model_dir=plan.base_model,
        config=train_config,
    )
    plan.estimated_train_items = int(summary.get("train_items", 0))
    LOGGER.info(
        "dry run complete",
        extra={
            "fields": {
                "rows_read": summary.get("rows_read"),
                "valid_items": summary.get("valid_items"),
                "train_items": summary.get("train_items"),
                "calibration_items": summary.get("calibration_items"),
                "skipped": summary.get("skipped"),
            }
        },
    )
    return dict(summary)


def run_finetune(config: Config, plan: TrainingPlan) -> Dict[str, Any]:
    """Start a real fine-tune. Only reachable once ``--yes`` has been given."""
    TrainConfig, _dry_run, finetune = require_laya()
    if plan.output_dir is None:
        raise TrainingError("a real training run needs an output directory")
    corpus_dir = config.corpus_dir
    from .io import resolve_output_path

    output_dir = resolve_output_path(plan.output_dir, corpus_dir)

    train_config = TrainConfig(**plan.train_config_kwargs)
    train_config.validate()

    summary = finetune(
        data=str(plan.data_path),
        model_dir=plan.base_model,
        output_dir=str(output_dir),
        config=train_config,
        device=config.training.device,
    )
    report = summary.get("train_report") or {}
    LOGGER.info(
        "fine-tune finished",
        extra={
            "fields": {
                "train_items": summary.get("train_items"),
                "calibration_items": summary.get("calibration_items"),
                "eval_items": summary.get("eval_items"),
                "eval_mode": report.get("eval_mode"),
                "is_held_out": report.get("is_held_out"),
                "output_dir": summary.get("output_dir"),
            }
        },
    )
    return dict(summary)


def calibration_warnings(report: Dict[str, Any]) -> List[str]:
    """Surface the calibration caveats a run reported, in substance verbatim.

    ``laya-multilingual`` ships with no fitted temperatures, so a checkpoint whose
    calibration was not fitted on a held-out slice must not gate on confidence
    (AGENTS.md 3.12).
    """
    warnings: List[str] = []
    if not isinstance(report, dict):
        return warnings
    calibration = (
        report.get("training", {}).get("laya_train_calibration")
        if isinstance(report.get("training"), dict)
        else None
    )
    calibration = calibration or report.get("calibration") or {}
    if isinstance(calibration, dict):
        for question_type, entry in sorted(calibration.items()):
            if not isinstance(entry, dict):
                continue
            issues = entry.get("issues") or []
            if issues:
                warnings.append(
                    f"{question_type} calibration is not trustworthy: "
                    + "; ".join(str(issue) for issue in issues)
                )
    return warnings
