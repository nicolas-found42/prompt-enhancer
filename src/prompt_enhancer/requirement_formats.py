"""Conservative whole-answer structured-format recognition and evidence."""

from __future__ import annotations

import csv
import io
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

JSON_DIRECTIVE = re.compile(
    r"[ \t]*(?:return|output|reply|respond)(?:[ \t]+with)?[ \t]+"
    r"(?:only[ \t]+(?:valid[ \t]+)?|valid[ \t]+)JSON"
    r"(?:[ \t]+(?:and[ \t]+nothing[ \t]+else|without[ \t]+any[ \t]+surrounding[ \t]+text))?"
    r"[.!]?[ \t]*",
    re.IGNORECASE,
)
JSON_SCHEMA_DIRECTIVE = re.compile(
    r"[ \t]*(?:return|output|reply|respond)(?:[ \t]+with)?[ \t]+a[ \t]+JSON[ \t]+object[ \t]+"
    r"with[ \t]+exactly[ \t]+these[ \t]+keys[ \t]+and[ \t]+types[ \t]*:[ \t]*"
    r"(?P<schema>\{[^\r\n]{1,4096}\})[.!]?[ \t]*",
    re.IGNORECASE,
)
JSON_TYPES = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null"}
)
CSV_DIRECTIVE = re.compile(
    r"[ \t]*(?:return|output|reply|respond)(?:[ \t]+with)?[ \t]+comma-delimited[ \t]+CSV[ \t]+"
    r"with[ \t]+exactly[ \t]+this[ \t]+header[ \t]+and[ \t]+matching[ \t]+record[ \t]+width[ \t]*:[ \t]*"
    r"(?P<columns>[A-Za-z_][A-Za-z0-9_]{0,63}(?:[ \t]*,[ \t]*[A-Za-z_][A-Za-z0-9_]{0,63}){0,31})"
    r"[.!]?[ \t]*",
    re.IGNORECASE,
)


class _NonJSONConstant(ValueError):
    pass


class _JSONObject(dict[str, Any]):
    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__(pairs)
        self.duplicate_names = len(pairs) != len(self)


def _reject_constant(_value: str) -> Any:
    raise _NonJSONConstant("non-JSON numeric constant")


def read_json(output: Any, *, unique_names: bool = False) -> tuple[str, Any, str]:
    """Keep parser capacity separate from known syntax failures."""
    if not isinstance(output, str):
        return "failed", None, "The complete answer must be JSON text."
    if output.startswith("\ufeff"):
        return (
            "untestable",
            None,
            "A byte-order mark needs an explicit JSON input convention.",
        )
    try:
        value = json.loads(
            output,
            parse_constant=_reject_constant,
            parse_float=Decimal,
            **({"object_pairs_hook": _JSONObject} if unique_names else {}),
        )
    except (json.JSONDecodeError, _NonJSONConstant):
        return (
            "failed",
            None,
            "The complete answer must be valid JSON, with no surrounding Markdown or prose.",
        )
    except (RecursionError, ValueError, InvalidOperation):
        return (
            "untestable",
            None,
            "This JSON answer exceeds the parser's supported nesting or numeric range.",
        )
    if isinstance(value, _JSONObject) and value.duplicate_names:
        return (
            "untestable",
            None,
            "Duplicate JSON keys make the applicable top-level key/type value ambiguous.",
        )
    return (
        "tested",
        value,
        "The complete answer is valid JSON; no unstated shape or value was required.",
    )


def schema_from_source(source: str) -> str | None:
    status, schema, _reason = read_json(source, unique_names=True)
    if status != "tested" or not isinstance(schema, dict) or not 1 <= len(schema) <= 32:
        return None
    # Escaped or complex names need a source/literal-scope audit beyond this grammar.
    if "\\" in source or any(
        not isinstance(value, str)
        or value not in JSON_TYPES
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key) is None
        for key, value in schema.items()
    ):
        return None
    return json.dumps(schema, ensure_ascii=False, sort_keys=True)


def _has_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "integer":
        return not isinstance(value, bool) and (
            isinstance(value, int)
            or isinstance(value, Decimal)
            and value == value.to_integral_value()
        )
    if expected == "number":
        return isinstance(value, (int, Decimal)) and not isinstance(value, bool)
    return isinstance(
        value, {"string": str, "boolean": bool, "array": list, "object": dict}[expected]
    )


def check_format(kind: str, output: Any, expected: str) -> tuple[str, str]:
    if kind == "json_format":
        status, _value, reason = read_json(output)
        return status, reason
    if kind == "json_schema":
        status, value, reason = read_json(output, unique_names=True)
        if status != "tested":
            return status, reason
        schema = json.loads(expected)
        passed = (
            isinstance(value, dict)
            and set(value) == set(schema)
            and all(
                _has_type(value[key], type_name) for key, type_name in schema.items()
            )
        )
        return (
            "tested" if passed else "failed",
            "The whole JSON object must have exactly the declared keys and their declared types; values remain unconstrained.",
        )
    if kind == "csv_shape":
        if not isinstance(output, str):
            return "failed", "The complete answer must be CSV text."
        try:
            rows = list(
                csv.reader(io.StringIO(output, newline=""), delimiter=",", strict=True)
            )
        except csv.Error as exc:
            if str(exc).startswith("field larger than field limit"):
                return "untestable", "A CSV field exceeds the parser's supported size."
            return "failed", "The answer has invalid comma-delimited CSV syntax."
        columns = json.loads(expected)
        if rows and any(not row for row in rows[1:]):
            return "untestable", "Blank CSV records need an explicit row convention."
        passed = (
            bool(rows)
            and rows[0] == columns
            and all(len(row) == len(columns) for row in rows[1:])
        )
        return (
            "tested" if passed else "failed",
            "The CSV header must match the declared columns in order, and every data record must have that width; no row count or cell values were required.",
        )
    return "untestable", "This format has no supported deterministic oracle."
