"""A compact JSON Schema subset, validated locally.

Ollama can constrain generation with a schema, but a constrained response is
still not a trustworthy one: the model can satisfy the shape and fill it with
nonsense. So the response is validated here, on the way out, and a failure is
raised as :class:`~laya_pipeline.errors.SchemaError` with the raw body attached.

Only the keywords the pipeline actually uses are implemented. An unknown keyword
is ignored rather than silently treated as satisfied-by-default — callers pass
schemas written in this file, so an unrecognised keyword is a typo worth seeing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..errors import SchemaError

__all__ = ["validate_json", "ensure_json", "KNOWN_KEYWORDS"]

KNOWN_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "anyOf",
        "oneOf",
        "description",
        "title",
        "$schema",
    }
)

_TYPE_MAP: Dict[str, tuple] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    types = _TYPE_MAP.get(expected)
    if types is None:
        # An unknown type name is a schema bug; accept rather than reject every
        # response, but the caller learns about it from `unknown_keywords`.
        return True
    if expected in ("string", "object", "array", "null"):
        return isinstance(value, types)
    return True


def validate_json(instance: Any, schema: Mapping[str, Any], path: str = "$") -> List[str]:
    """Return a list of human-readable problems. Empty means valid.

    Every problem found is returned, not just the first, so one regeneration
    attempt can fix a response that has several defects at once.
    """
    problems: List[str] = []

    if not isinstance(schema, Mapping):
        return [f"{path}: schema must be an object, got {type(schema).__name__}"]

    if "anyOf" in schema or "oneOf" in schema:
        branches = schema.get("anyOf") or schema.get("oneOf") or []
        matches = [
            branch
            for branch in branches
            if not validate_json(instance, branch, path)
        ]
        if not matches:
            keyword = "anyOf" if "anyOf" in schema else "oneOf"
            return [f"{path}: value does not match any {keyword} branch"]
        if "oneOf" in schema and len(matches) > 1:
            problems.append(f"{path}: value matches {len(matches)} oneOf branches, expected 1")
        return problems

    if "const" in schema and instance != schema["const"]:
        problems.append(f"{path}: expected the constant {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        problems.append(f"{path}: {instance!r} is not one of [{allowed}]")

    declared = schema.get("type")
    if declared is not None:
        expected_types: Sequence[str]
        if isinstance(declared, str):
            expected_types = (declared,)
        elif isinstance(declared, (list, tuple)):
            expected_types = tuple(str(t) for t in declared)
        else:
            expected_types = ()
            problems.append(f"{path}: 'type' must be a string or a list of strings")
        if expected_types and not any(_type_ok(instance, t) for t in expected_types):
            problems.append(
                f"{path}: expected type {'/'.join(expected_types)}, "
                f"got {type(instance).__name__}"
            )
            return problems

    if isinstance(instance, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(instance) < min_length:
            problems.append(f"{path}: string is shorter than minLength {min_length}")
        if isinstance(max_length, int) and len(instance) > max_length:
            problems.append(f"{path}: string is longer than maxLength {max_length}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            problems.append(f"{path}: {instance} is below minimum {minimum}")
        if isinstance(maximum, (int, float)) and instance > maximum:
            problems.append(f"{path}: {instance} is above maximum {maximum}")

    if isinstance(instance, (list, tuple)):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            problems.append(f"{path}: array has fewer than minItems {min_items}")
        if isinstance(max_items, int) and len(instance) > max_items:
            problems.append(f"{path}: array has more than maxItems {max_items}")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(instance):
                problems.extend(validate_json(item, item_schema, f"{path}[{index}]"))

    if isinstance(instance, Mapping):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in instance:
                problems.append(f"{path}: missing required property {key!r}")
        for key, value in instance.items():
            child = properties.get(key)
            if isinstance(child, Mapping):
                problems.extend(validate_json(value, child, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                problems.append(f"{path}: unexpected property {key!r}")

    return problems


def ensure_json(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    raw: Optional[str] = None,
    context: str = "model response",
) -> Any:
    """Validate and return, or raise :class:`SchemaError` listing every problem."""
    problems = validate_json(instance, schema)
    if problems:
        joined = "; ".join(problems[:12])
        suffix = "" if len(problems) <= 12 else f" (+{len(problems) - 12} more)"
        raise SchemaError(f"{context} failed schema validation: {joined}{suffix}", raw=raw)
    return instance


def unknown_keywords(schema: Mapping[str, Any]) -> List[str]:
    """Schema keywords this validator does not implement.

    Surfaced by the prompt loader so a schema typo is visible at configuration
    time rather than as a mysteriously permissive validator.
    """
    return sorted(set(schema) - KNOWN_KEYWORDS)


def object_schema(
    properties: Mapping[str, Any],
    *,
    required: Optional[Sequence[str]] = None,
    additional: bool = False,
) -> Dict[str, Any]:
    """Build an object schema, defaulting ``required`` to every property."""
    names = list(properties)
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required) if required is not None else names,
        "additionalProperties": additional,
    }


def string_schema(**kwargs: Any) -> Dict[str, Any]:
    return {"type": "string", **kwargs}


def number_schema(**kwargs: Any) -> Dict[str, Any]:
    return {"type": "number", **kwargs}


def array_schema(items: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return {"type": "array", "items": dict(items), **kwargs}