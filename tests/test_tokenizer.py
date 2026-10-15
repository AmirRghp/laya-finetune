"""Tokenizer loading, including Hub repositories with nested tokenizer files."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from laya_pipeline.tokenizer import _load_huggingface_counter


def _stub_transformers(monkeypatch: Any, loader: Any) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader)),
    )


def test_hub_tokenizer_loads_from_tokenizer_subfolder(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def load(model: str, **kwargs: Any) -> object:
        calls.append((model, kwargs))
        if kwargs.get("subfolder") == "tokenizer":
            return object()
        raise AssertionError("the root fallback should not be needed")

    _stub_transformers(monkeypatch, load)

    counter = _load_huggingface_counter(
        "convaiinnovations/laya-multilingual", 8192, allow_download=True
    )

    assert counter is not None
    assert calls == [
        (
            "convaiinnovations/laya-multilingual",
            {"local_files_only": False, "subfolder": "tokenizer"},
        )
    ]
    assert counter.source == "huggingface:convaiinnovations/laya-multilingual/tokenizer"


def test_hub_tokenizer_falls_back_to_repository_root(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def load(model: str, **kwargs: Any) -> object:
        calls.append((model, kwargs))
        if "subfolder" in kwargs:
            raise OSError("no tokenizer subfolder")
        return object()

    _stub_transformers(monkeypatch, load)

    counter = _load_huggingface_counter("example/model", None, allow_download=False)

    assert counter is not None
    assert calls == [
        ("example/model", {"local_files_only": True, "subfolder": "tokenizer"}),
        ("example/model", {"local_files_only": True}),
    ]
    assert counter.source == "huggingface:example/model"
