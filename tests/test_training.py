"""Training bridge tests use only fake Laya interfaces and synthetic plans."""

from __future__ import annotations

from pathlib import Path

from laya_pipeline import training
from laya_pipeline.training import TrainingPlan


def test_run_dry_run_calls_dry_run_and_updates_plan(config, monkeypatch) -> None:
    calls = []

    class FakeTrainConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def validate(self) -> None:
            pass

    def fake_dry_run(**kwargs):
        calls.append(("dry_run", kwargs))
        return {"train_items": 17, "rows_read": 20}

    def unexpected_finetune(**kwargs):
        raise AssertionError("dry-run must not call finetune")

    monkeypatch.setattr(
        training,
        "require_laya",
        lambda: (FakeTrainConfig, fake_dry_run, unexpected_finetune),
    )
    plan = TrainingPlan(
        base_model="synthetic-checkpoint",
        data_path=Path("synthetic-train.jsonl"),
        eval_path=None,
        output_dir=None,
        train_config_kwargs={"epochs": 1},
        max_len=128,
        head_max_len=32,
    )

    summary = training.run_dry_run(config, plan)

    assert summary["train_items"] == 17
    assert plan.estimated_train_items == 17
    assert calls[0][0] == "dry_run"
    assert calls[0][1]["data"] == "synthetic-train.jsonl"
