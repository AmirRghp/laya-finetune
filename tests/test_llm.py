"""Schema validation, prompt templates, caching and the Ollama client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from laya_pipeline.config import OllamaConfig, load_config
from laya_pipeline.errors import ConfigError, OllamaError
from laya_pipeline.io import write_json, write_text
from laya_pipeline.llm.cache import Cache, NullCache, cache_key
from laya_pipeline.llm.ollama import OllamaClient, extract_json
from laya_pipeline.llm.prompts import PromptLibrary
from laya_pipeline.llm.schema import (
    array_schema,
    object_schema,
    string_schema,
    unknown_keywords,
    validate_json,
)

from tests.conftest import FakeTransport


class TestSchemaValidation:
    def test_a_valid_object_passes(self) -> None:
        schema = object_schema({"name": string_schema(minLength=2)})
        assert validate_json({"name": "سیاست"}, schema) == []

    def test_missing_required_property_is_reported(self) -> None:
        schema = object_schema({"name": string_schema()})
        problems = validate_json({}, schema)
        assert any("missing required" in p for p in problems)

    def test_wrong_type_is_reported_with_the_path(self) -> None:
        schema = object_schema({"name": string_schema()})
        problems = validate_json({"name": 5}, schema)
        assert any("$.name" in p for p in problems)

    def test_unexpected_property_is_rejected_when_closed(self) -> None:
        schema = object_schema({"a": string_schema()})
        problems = validate_json({"a": "x", "b": "y"}, schema)
        assert any("unexpected property" in p for p in problems)

    def test_enum_violation_lists_the_options(self) -> None:
        schema = object_schema({"k": string_schema(enum=["x", "y"])})
        assert any("not one of" in p for p in validate_json({"k": "z"}, schema))

    def test_numeric_bounds(self) -> None:
        schema = object_schema({"n": {"type": "number", "minimum": 0, "maximum": 1}})
        assert validate_json({"n": 0.5}, schema) == []
        assert validate_json({"n": 2}, schema)

    def test_integer_is_not_a_boolean(self) -> None:
        assert validate_json(True, {"type": "integer"})

    def test_array_items_are_validated(self) -> None:
        schema = object_schema({"xs": array_schema(string_schema(minLength=3))})
        assert validate_json({"xs": ["abcd"]}, schema) == []
        assert validate_json({"xs": ["ab"]}, schema)

    def test_min_and_max_items(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}
        assert validate_json(["a", "b"], schema) == []
        assert validate_json(["a"], schema)
        assert validate_json(["a", "b", "c", "d"], schema)

    def test_every_problem_is_reported_not_just_the_first(self) -> None:
        schema = object_schema({"a": string_schema(), "b": string_schema()})
        assert len(validate_json({"a": 1, "b": 2}, schema)) == 2

    def test_any_of_accepts_a_matching_branch(self) -> None:
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert validate_json("x", schema) == []
        assert validate_json([], schema)

    def test_unknown_keywords_are_reported(self) -> None:
        assert unknown_keywords({"type": "object", "pattern": "x"}) == ["pattern"]

    def test_non_object_schema_is_reported(self) -> None:
        assert validate_json({}, ["not", "a", "schema"])


class TestPromptTemplates:
    @pytest.fixture
    def library(self, tmp_path: Path) -> PromptLibrary:
        write_text(
            tmp_path / "demo.md",
            "پرسش: {{query}}\nمتن: {{chunk}}\nساختار: {{schema}}",
        )
        return PromptLibrary(tmp_path)

    def test_placeholders_are_replaced(self, library: PromptLibrary) -> None:
        rendered = library.render("demo", query="چرا؟", chunk="متن", schema="{}")
        assert rendered == "پرسش: چرا؟\nمتن: متن\nساختار: {}"

    def test_json_braces_are_untouched(self, tmp_path: Path) -> None:
        write_text(tmp_path / "j.md", '{{schema}}')
        library = PromptLibrary(tmp_path)
        payload = json.dumps({"a": 1}, ensure_ascii=False)
        assert library.render("j", schema=payload) == payload

    def test_missing_value_raises(self, library: PromptLibrary) -> None:
        with pytest.raises(ConfigError, match="needs values"):
            library.render("demo", query="x")

    def test_unused_value_raises(self, library: PromptLibrary) -> None:
        with pytest.raises(ConfigError, match="does not use"):
            library.render("demo", query="x", chunk="y", schema="{}", extra="z")

    def test_unknown_template_names_the_alternatives(self, library: PromptLibrary) -> None:
        with pytest.raises(ConfigError, match="available: demo"):
            library.render("absent")

    def test_missing_directory_is_reported(self, tmp_path: Path) -> None:
        library = PromptLibrary(tmp_path / "absent")
        with pytest.raises(ConfigError, match="does not exist"):
            library.render("demo", query="x")

    def test_fingerprint_changes_with_content(self, tmp_path: Path) -> None:
        write_text(tmp_path / "d.md", "یک")
        library = PromptLibrary(tmp_path)
        before = library.fingerprint("d")
        write_text(tmp_path / "d.md", "دو")
        library._templates.pop("d")
        assert library.fingerprint("d") != before

    def test_the_shipped_prompts_all_parse(self, tmp_path: Path) -> None:
        from tests.conftest import REPO_ROOT

        library = PromptLibrary(REPO_ROOT / "prompts")
        assert set(library.available()) >= {
            "questions.fa",
            "evidence_answerability.fa",
            "label_relevance.fa",
            "label_sufficiency.fa",
            "label_routing.fa",
        }
        for name in library.available():
            assert library.template(name).placeholders


class TestJsonExtraction:
    def test_plain_object(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self) -> None:
        assert extract_json('این است: {"a": 1} پایان') == {"a": 1}

    def test_nested_braces(self) -> None:
        assert extract_json('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}

    def test_braces_inside_strings(self) -> None:
        assert extract_json('{"a": "}"}') == {"a": "}"}

    def test_unparseable_raises_with_the_text(self) -> None:
        with pytest.raises(OllamaError, match="no JSON object"):
            extract_json("اصلاً JSON نیست")

    def test_empty_response_raises(self) -> None:
        with pytest.raises(OllamaError):
            extract_json("")


class TestCache:
    def test_a_miss_then_a_hit(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path, "test")
        assert cache.get("k") is None
        cache.set("k", {"v": 1})
        assert cache.get("k") == {"v": 1}

    def test_null_cache_stores_nothing(self) -> None:
        cache = NullCache()
        cache.set("k", 1)
        assert cache.get("k") is None

    def test_a_corrupt_entry_is_discarded(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path, "test")
        cache.set("k", {"v": 1})
        write_text(cache.path_for("k"), "{ not json")
        assert cache.get("k") is None
        assert not cache.path_for("k").exists()

    def test_a_foreign_entry_is_ignored(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path, "test")
        path = cache.path_for("k")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"key": "other", "value": 1})
        assert cache.get("k") is None

    def test_key_depends_on_every_input(self) -> None:
        base = dict(
            endpoint="http://x", model="m", prompt="p", system=None,
            temperature=0.7, top_p=0.9, seed=1, schema=None,
        )
        reference = cache_key(**base)
        assert cache_key(**{**base, "model": "other"}) != reference
        assert cache_key(**{**base, "temperature": 0.8}) != reference
        assert cache_key(**{**base, "seed": 2}) != reference
        assert cache_key(**{**base, "prompt": "q"}) != reference
        assert cache_key(**{**base, "endpoint": "http://y/"}) != reference


class TestOllamaClient:
    def _client(self, transport: FakeTransport, cache_dir: Path) -> OllamaClient:
        config = OllamaConfig(
            endpoint="http://fake", generation_model="gen", embedding_model="emb",
            teacher_model="tea", max_retries=2, retry_backoff_seconds=0,
        )
        return OllamaClient(
            config, transport=transport, cache=Cache(cache_dir, "test"), sleep=lambda _s: None
        )

    def test_structured_generation_returns_the_object(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": 1}])
        client = self._client(transport, tmp_path)
        result = client.generate_json("پرسش", schema=object_schema({"a": {"type": "integer"}}))
        assert result.data == {"a": 1}
        assert result.attempts == 1

    def test_a_cache_hit_avoids_the_request(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": 1}])
        client = self._client(transport, tmp_path)
        schema = object_schema({"a": {"type": "integer"}})
        first = client.generate_json("پرسش", schema=schema)
        second = client.generate_json("پرسش", schema=schema)
        assert not first.from_cache and second.from_cache
        assert len(transport.calls) == 1

    def test_a_different_prompt_bypasses_the_cache(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": 1}])
        client = self._client(transport, tmp_path)
        schema = object_schema({"a": {"type": "integer"}})
        client.generate_json("یک", schema=schema)
        client.generate_json("دو", schema=schema)
        assert len(transport.calls) == 2

    def test_a_timeout_is_retried(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": 1}], fail_times=1)
        client = self._client(transport, tmp_path)
        result = client.generate_json("پرسش", schema=object_schema({"a": {"type": "integer"}}))
        assert result.attempts == 2
        assert client.retry_count == 1

    def test_persistent_failure_raises_ollama_error(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": 1}], fail_times=9)
        client = self._client(transport, tmp_path)
        with pytest.raises(OllamaError, match="did not return valid JSON"):
            client.generate_json("پرسش", schema=object_schema({"a": {"type": "integer"}}))

    def test_a_schema_violation_is_retried_then_raises(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": "not-an-integer"}])
        client = self._client(transport, tmp_path)
        with pytest.raises(OllamaError):
            client.generate_json("پرسش", schema=object_schema({"a": {"type": "integer"}}))
        assert client.retry_count >= 1

    def test_embedding_shape(self, tmp_path: Path) -> None:
        import numpy as np

        client = self._client(FakeTransport(embedding_dim=8), tmp_path)
        vectors = client.embed(["یک", "دو", "سه"])
        assert vectors.shape == (3, 8)
        assert np.isfinite(vectors).all()

    def test_embedding_of_nothing(self, tmp_path: Path) -> None:
        client = self._client(FakeTransport(), tmp_path)
        assert client.embed([]).shape[0] == 0

    def test_embeddings_fall_back_to_the_single_endpoint(self, tmp_path: Path) -> None:
        class LegacyTransport(FakeTransport):
            def post(self, path, payload):
                if path == "/api/embed":
                    from laya_pipeline.errors import OllamaResponseError

                    raise OllamaResponseError("404 not found", retryable=False)
                return super().post(path, payload)

        client = self._client(LegacyTransport(), tmp_path)
        vectors = client.embed(["یک", "دو"])
        assert vectors.shape[0] == 2

    def test_inconsistent_embedding_widths_are_rejected(self, tmp_path: Path) -> None:
        class RaggedTransport(FakeTransport):
            def post(self, path, payload):
                texts = list(payload.get("input", []))
                return {"embeddings": [[1.0] * (4 if i else 5) for i, _t in enumerate(texts)]}

        client = self._client(RaggedTransport(), tmp_path)
        with pytest.raises(OllamaError, match="dimensions"):
            client.embed(["یک", "دو"])

    def test_availability_check(self, tmp_path: Path) -> None:
        assert self._client(FakeTransport(), tmp_path).is_available() is True

    def test_model_listing(self, tmp_path: Path) -> None:
        assert self._client(FakeTransport(), tmp_path).list_models() == ["fake-model"]

    def test_seed_and_temperature_reach_the_payload(self, tmp_path: Path) -> None:
        transport = FakeTransport([{"a": 1}])
        client = self._client(transport, tmp_path)
        client.generate_json(
            "پرسش", schema=object_schema({"a": {"type": "integer"}}),
            temperature=0.1, seed=99,
        )
        options = transport.calls[0]["payload"]["options"]
        assert options["temperature"] == 0.1
        assert options["seed"] == 99

    def test_a_missing_content_field_is_an_error(self, tmp_path: Path) -> None:
        class EmptyTransport(FakeTransport):
            def post(self, path, payload):
                return {"unexpected": "shape"}

        client = self._client(EmptyTransport(), tmp_path)
        with pytest.raises(OllamaError, match="did not return valid JSON"):
            client.generate_json("پرسش", schema=object_schema({"a": {"type": "integer"}}))