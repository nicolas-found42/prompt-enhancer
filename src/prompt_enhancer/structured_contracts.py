"""Bounded source grammars for structured outputs within compound requests."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterator
from typing import Any, Literal

from .protected_blocks import source_data_spans
from .requirement_formats import JSON_TYPES, _has_type, read_json

_START = re.compile(
    r"\b(?:return|output|reply|respond)(?:\s+with)?\s+(?:(?:a|only|valid|comma-delimited|semicolon-delimited|tab-delimited)\s+)*"
    r"(?P<format>JSON|CSV)\b",
    re.I,
)
_KEY = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
_DECLARATION = re.compile(
    rf"(?P<key>{_KEY}(?:\.{_KEY})*)\s+as\s+(?:an?\s+)?"
    r"(?P<type>array of (?:strings|integers|numbers|booleans|objects)|string|integer|number|boolean|array|object|null)",
    re.I,
)


def _valid_schema(value: Any, depth: int = 0) -> bool:
    if depth > 12:
        return False
    if isinstance(value, str):
        return value in JSON_TYPES
    if isinstance(value, list):
        return len(value) == 1 and _valid_schema(value[0], depth + 1)
    return (
        isinstance(value, dict)
        and 0 < len(value) <= 32
        and all(
            re.fullmatch(_KEY, key) and _valid_schema(item, depth + 1)
            for key, item in value.items()
        )
    )


def _json_contract(tail: str) -> dict[str, Any]:
    contract: dict[str, Any] = {"schema": None, "exact_keys": False}
    # Explicit key/type maps support nested objects and one-element array type
    # declarations. They never stand for example values or a full JSON Schema.
    match = re.fullmatch(
        r"\s*(?:object\s+)?with\s+(?P<exact>exactly\s+)?(?:these\s+)?keys\s+and\s+types\s*:\s*(?P<schema>\{.*\})\s*[.!]?\s*",
        tail,
        re.I | re.S,
    )
    if match:
        status, schema, _ = read_json(match["schema"], unique_names=True)
        if status == "tested" and _valid_schema(schema) and isinstance(schema, dict):
            contract.update(schema=schema, exact_keys=bool(match["exact"]))
            return contract
    declaration = re.fullmatch(
        r"\s*(?:object\s+)?(?:containing|with)\s+(.+?)[.!]?\s*", tail, re.I | re.S
    )
    if declaration:
        parts = re.split(r"\s*(?:,\s*(?:and\s+)?|\s+and\s+)\s*", declaration[1])
        schema = {}
        for part in parts:
            item = _DECLARATION.fullmatch(part.strip())
            if item is None:
                break
            node = schema
            keys = item["key"].split(".")
            for key in keys[:-1]:
                if key not in node:
                    node[key] = {}
                if not isinstance(node[key], dict):
                    break
                node = node[key]
            else:
                type_name = item["type"].lower()
                expected = (
                    [type_name.removeprefix("array of ").removesuffix("s")]
                    if type_name.startswith("array of ")
                    else type_name
                )
                if keys[-1] in node:
                    break
                node[keys[-1]] = expected
                continue
            break
        else:
            if _valid_schema(schema):
                contract["schema"] = schema
                return contract
    if not tail.strip().strip(".!"):
        return contract
    contract["uncertainty"] = (
        "The JSON declaration uses an unsupported schema or scope convention."
    )
    return contract


def _csv_contract(directive: str, tail: str) -> dict[str, Any]:
    contract: dict[str, Any] = {"columns": None, "delimiter": ",", "data_rows": None}
    match = re.fullmatch(
        rf"\s+with\s+(?:columns|(?:exactly\s+)?(?:this\s+)?header(?:\s+and\s+matching\s+record\s+width)?)\s*:?\s*"
        rf"(?P<columns>{_KEY}(?:\s*[,;]\s*(?!(?:with|using|quoted|without)\b){_KEY}){{0,31}})(?P<rest>.*)",
        tail,
        re.I | re.S,
    )
    if "semicolon-delimited" in directive.lower():
        contract["delimiter"] = ";"
    elif "tab-delimited" in directive.lower():
        contract["delimiter"] = "\t"
    if not match:
        contract["uncertainty"] = (
            "The CSV declaration needs a supported explicit header and dialect."
        )
        return contract
    contract["columns"] = re.split(r"\s*[,;]\s*", match["columns"])
    rest = match["rest"].strip().strip(".!").strip(" ,;")
    row = re.fullmatch(
        r"with\s+exactly\s+(\d{1,6}|one|two|three|four|five|zero)\s+data\s+rows?(?:\s*;\s*quoted commas inside "
        + _KEY
        + r" are valid)?",
        rest,
        re.I,
    )
    if row:
        number = row[1].lower()
        contract["data_rows"] = (
            int(number)
            if number.isdigit()
            else {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}[
                number
            ]
        )
    elif rest and not re.fullmatch(
        r"quoted commas inside " + _KEY + r" are valid", rest, re.I
    ):
        contract["uncertainty"] = (
            "The CSV row count, delimiter or quoting convention is unsupported or ambiguous."
        )
    return contract


def structured_contracts(
    prompt: str,
) -> Iterator[tuple[int, int, Literal["json_contract", "csv_contract"], str, str]]:
    """Yield source spans, kind, explicit scope and serialized contract."""
    spans = source_data_spans(prompt)
    for match in _START.finditer(prompt):
        if any(start <= match.start() < end for start, end in spans):
            continue
        # A sentence/line is the supported declaration boundary. Semicolons
        # remain in CSV declarations because they can introduce row constraints.
        end_match = re.search(r"[.!](?=\s|$)|\n", prompt[match.end() :])
        end = match.end() + end_match.end() if end_match else len(prompt)
        # Dot-path keys and decimal notation must not create sentence boundaries.
        tail = prompt[match.end() : end]
        scope_match = re.search(
            r"(?:in|for)\s+(?:the\s+)?section\s+([A-Za-z][A-Za-z0-9 _-]{0,63})\s*[:,]\s*$",
            prompt[: match.start()],
            re.I,
        )
        scope = "section:" + scope_match[1].strip() if scope_match else "whole_output"
        contract = (
            _json_contract(tail)
            if match["format"].upper() == "JSON"
            else _csv_contract(match.group(), tail)
        )
        yield (
            match.start(),
            end,
            "json_contract" if match["format"].upper() == "JSON" else "csv_contract",
            scope,
            json.dumps(contract, sort_keys=True),
        )


def _duplicates(value: Any) -> bool:
    return (
        bool(getattr(value, "duplicate_names", False))
        or (
            isinstance(value, dict)
            and any(_duplicates(item) for item in value.values())
        )
        or (isinstance(value, list) and any(_duplicates(item) for item in value))
    )


def _applicable_duplicates(value: Any, schema: Any) -> bool:
    if isinstance(schema, list) and isinstance(value, list):
        return any(_applicable_duplicates(item, schema[0]) for item in value)
    if isinstance(schema, dict) and isinstance(value, dict):
        return bool(set(schema) & getattr(value, "duplicate_keys", set())) or any(
            key in value and _applicable_duplicates(value[key], item)
            for key, item in schema.items()
        )
    return False


def _matches(value: Any, schema: Any, *, exact: bool = False) -> bool:
    if isinstance(schema, str):
        return _has_type(value, schema)
    if isinstance(schema, list):
        return isinstance(value, list) and all(
            _matches(item, schema[0]) for item in value
        )
    return (
        isinstance(value, dict)
        and (not exact or set(value) == set(schema))
        and all(
            key in value and _matches(value[key], item) for key, item in schema.items()
        )
    )


def check_contract(kind: str, output: Any, expected: str) -> tuple[str, str]:
    contract = json.loads(expected)
    if kind == "json_contract":
        status, value, reason = read_json(output, unique_names=True)
        schema = contract.get("schema")
        if _applicable_duplicates(value, schema):
            return (
                "failed",
                "Duplicate keys violate the declared JSON key/type bindings.",
            )
        if status != "tested":
            return status, reason
        if _duplicates(value):
            return (
                "untestable",
                "Duplicate undeclared JSON keys need an explicit parser convention.",
            )
        schema = contract.get("schema")
        if schema is not None and not _matches(
            value, schema, exact=contract["exact_keys"]
        ):
            return (
                "failed",
                "The JSON answer does not satisfy the explicitly declared keys, structure and types.",
            )
        if contract.get("uncertainty"):
            return "untestable", contract["uncertainty"]
        return (
            "tested",
            "The answer satisfies the declared JSON contract; undeclared values, keys and array lengths are unconstrained.",
        )
    if contract.get("uncertainty"):
        return "untestable", contract["uncertainty"]
    if not isinstance(output, str):
        return "failed", "The answer must be CSV text."
    try:
        rows = list(
            csv.reader(
                io.StringIO(output, newline=""),
                delimiter=contract["delimiter"],
                strict=True,
            )
        )
    except csv.Error as error:
        if "field larger" in str(error):
            return "untestable", "A CSV field exceeds the supported parser size."
        return "failed", "The CSV answer has malformed quoting or syntax."
    columns = contract.get("columns")
    if columns is not None:
        if (
            not rows
            or rows[0] != columns
            or any(row and len(row) != len(columns) for row in rows[1:])
        ):
            return (
                "failed",
                "The CSV header or data width differs from the explicit declaration.",
            )
        if any(not row for row in rows[1:]):
            return (
                "untestable",
                "Blank CSV records need an explicit data-row convention.",
            )
        if (
            contract.get("data_rows") is not None
            and len(rows) - 1 != contract["data_rows"]
        ):
            return (
                "failed",
                "The CSV data-row count differs from the explicit declaration; the header is not a data row.",
            )
    if contract.get("uncertainty"):
        return "untestable", contract["uncertainty"]
    return (
        "tested",
        "The CSV header, width and declared data-row count pass; cell contents remain unconstrained.",
    )


def contract_values(kind: str, expected: str) -> tuple[str, ...]:
    contract = json.loads(expected)
    if kind == "csv_contract":
        return tuple(contract.get("columns") or ())

    def values(schema: Any) -> list[str]:
        if isinstance(schema, str):
            return []
        if isinstance(schema, list):
            return values(schema[0])
        return [
            value
            for key, item in (schema or {}).items()
            for value in [key, *values(item)]
        ]

    return tuple(dict.fromkeys(values(contract.get("schema"))))
