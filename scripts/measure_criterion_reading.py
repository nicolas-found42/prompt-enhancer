"""Measure reading success criteria with a model instead of regexes (issue #116).

A model selects the constraint a criterion sets (which count, which comparison,
which number) and code owns everything else: the candidate numbers are found by
regex, the picked span is converted to an integer in code, and a criterion whose
yes/no judgments are uncertain stays unresolved. ``run`` records raw answers;
``report`` scores recordings offline against the labelled fixture, so a band or
policy change never needs another request.

    uv run --env-file .env python scripts/measure_criterion_reading.py run --reader jev \
        --split development --output .local/criterion-reading/jev-dev.json
    uv run --env-file .env python scripts/measure_criterion_reading.py report \
        --recording .local/criterion-reading/jev-dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evaluation_review_common import save_json

from prompt_enhancer.criterion_checks import check_criterion
from prompt_enhancer.gateway import GatewayConfig, HttpGateway, completion_text
from prompt_enhancer.jev import ChoiceDecision, NoulDecision, parse_decision

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/evaluation/criterion_reading_cases.json"
)
CHEAP_MODEL = "mistralai/mistral-nemo"
MAX_COMPLETION_TOKENS = 4096
NONE = "none"
BANDS = ((0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.05, 0.95))
NOUL_KEYS = ("partial", "conditional", "negated", "approximate")
OPERATORS = ("at most", "under", "at least", "more than", "exactly", "between")

JUDGMENTS = ("partial", "conditional", "approximate")


def parse_cutoffs(value: str) -> dict[str, float]:
    """Parse comma-separated cutoffs for partial, conditional and approximate."""
    cutoffs: dict[str, float] = {}
    for item in value.split(","):
        name, separator, raw_value = item.partition("=")
        name = name.strip()
        if not separator or name not in JUDGMENTS:
            raise ValueError(
                "cutoffs must be comma-separated partial=value, conditional=value "
                "or approximate=value"
            )
        if name in cutoffs:
            raise ValueError(f"duplicate cutoff for {name}")
        try:
            cutoff = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"cutoff for {name} must be numeric") from exc
        if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError(f"cutoff for {name} must be between 0 and 1")
        cutoffs[name] = cutoff
    return cutoffs


_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORD_VALUES: dict[str, int] = {
    **{word: value for value, word in enumerate(_UNITS)},
    **{word: 20 + 10 * value for value, word in enumerate(_TENS)},
}
_SMALL = rf"(?:(?:{'|'.join(_TENS)})(?:[- ](?:{'|'.join(_UNITS[1:10])}))?|{'|'.join(_UNITS)})"
_NUMBER_CANDIDATE = re.compile(
    r"\d{1,3}(?:,\d{3})+|\d+"
    rf"|\b{_SMALL}(?:\s+(?:hundred|thousand)"
    rf"(?:\s+(?:and\s+)?{_SMALL}(?!\s+(?:hundred|thousand)))?)?\b"
)


def number_candidates(criterion: str) -> dict[str, int]:
    """Numbers in the criterion, by the span as written, valued in code."""
    found: dict[str, int] = {}
    for match in _NUMBER_CANDIDATE.finditer(criterion.casefold()):
        span = match.group()
        found.setdefault(span, _value(span))
    return found


def _value(span: str) -> int:
    if span[0].isdigit():
        return int(span.replace(",", ""))
    total = current = 0
    for word in re.split(r"[-\s]+", span):
        if word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total, current = total + max(current, 1) * 1000, 0
        elif word != "and":
            current += _WORD_VALUES[word]
    return total + current


MAX_MODEL_LENGTH = 200


def parse_model(value: str) -> str:
    model = value.strip()
    if not model:
        raise argparse.ArgumentTypeError("--model must not be empty")
    if len(model) > MAX_MODEL_LENGTH:
        raise argparse.ArgumentTypeError(
            f"--model must be at most {MAX_MODEL_LENGTH} characters"
        )
    return model


def reading_questions(candidates: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The questions asked about one criterion, keyed by what code reads back."""
    numbers: dict[str, Any] = {span: None for span in candidates} | {
        NONE: "The criterion gives no such number."
    }
    return {
        "kind": {
            "type": "choice",
            "instructions": (
                "This is a success criterion that a piece of writing will be "
                "graded against. Which countable property of the whole output "
                "does it set a limit or target on?"
            ),
            "criteria": {
                "word_count": "A limit or target on the number of words, such as "
                "'under 100 words', '50 words or fewer', 'at least 500 words', "
                "'between 100 and 150 words'.",
                "sentence_count": "A limit or target on the number of sentences, "
                "such as 'at most 3 sentences' or 'exactly two sentences'.",
                "valid_json": "The output has to be JSON that parses, such as "
                "'valid JSON' or 'parseable as JSON'.",
                "other": "Anything else: tone, content, format, paragraphs, "
                "bullets, characters, sources, punctuation, a field inside JSON, "
                "or no countable property.",
            },
        },
        "op": {
            "type": "choice",
            "instructions": (
                "If the criterion sets a limit or target on a count of words or "
                "sentences, which comparison must the count satisfy? Read "
                "negations into the answer: 'not under 100' means at least 100 "
                "and 'never more than 60' means at most 60."
            ),
            "criteria": {
                "at most": "No more than the number, e.g. 'at most 50', '50 or "
                "fewer', 'no longer than 50', 'within 50', 'a 50-word limit'.",
                "under": "Strictly fewer than the number, e.g. 'under 50', "
                "'fewer than 50', 'shorter than 50'.",
                "at least": "No fewer than the number, e.g. 'at least 50', "
                "'50 or more', 'a minimum of 50', or a plus sign after the "
                "number as in '50+'.",
                "more than": "Strictly more than the number, e.g. 'over 50', "
                "'longer than 50', 'more than 50'.",
                "exactly": "Equal to the number.",
                "between": "Within a range of two numbers, e.g. 'between 3 and "
                "5', '100 to 150', '100-150'.",
                NONE: "No comparison on a count is set.",
            },
        },
        "bound": {
            "type": "choice",
            "instructions": (
                "Which of these numbers is the limit or target that the count of "
                "words or sentences is compared against? Ignore numbers that "
                "count something else, such as paragraphs, questions or ages."
            ),
            "criteria": numbers,
        },
        "low": {
            "type": "choice",
            "instructions": (
                "If the criterion gives a range for the count of words or "
                "sentences, such as 'between 3 and 5' or '100 to 150', which "
                "number is the lower end of the range? Otherwise choose none."
            ),
            "criteria": numbers,
        },
        "high": {
            "type": "choice",
            "instructions": (
                "If the criterion gives a range for the count of words or "
                "sentences, such as 'between 3 and 5' or '100 to 150', which "
                "number is the upper end of the range? Otherwise choose none."
            ),
            "criteria": numbers,
        },
        "partial": {
            "type": "noul",
            "instructions": (
                "Is the requirement in the criterion about only part of the "
                "output, such as a title, an intro, each section, each bullet, "
                "every sentence, or the JSON in a reply that also contains other "
                "text, rather than about the output as a whole?"
            ),
        },
        "conditional": {
            "type": "noul",
            "instructions": (
                "Does the criterion apply only under a condition, or offer an "
                "alternative that would also satisfy it, such as 'unless', 'if', "
                "'when' or 'or a table if longer'?"
            ),
        },
        "negated": {
            "type": "noul",
            "instructions": (
                "Does the criterion require the property it names to be absent "
                "or to fail, such as 'is not valid JSON' or 'must not be JSON'?"
            ),
        },
        "approximate": {
            "type": "noul",
            "instructions": (
                "Does the criterion itself say its target is only approximate, "
                "with a word such as 'roughly', 'about', 'around', "
                "'approximately' or 'give or take'?"
            ),
        },
    }


def questions_digest() -> str:
    blob = json.dumps(reading_questions(["100"]), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Reading:
    """The model's answers about one criterion, before any policy is applied."""

    kind: str
    op: str
    bound: str
    low: str
    high: str
    noul: Mapping[str, float]


def read_jev(answers: Mapping[str, Any]) -> Reading:
    choices: dict[str, str] = {}
    for key in ("kind", "op", "bound", "low", "high"):
        decision = parse_decision(answers[key])
        if not isinstance(decision, ChoiceDecision):
            raise TypeError(f"expected choice decision for {key}")
        choices[key] = decision.selected
    nouls: dict[str, float] = {}
    for key in NOUL_KEYS:
        decision = parse_decision(answers[key])
        if not isinstance(decision, NoulDecision):
            raise TypeError(f"expected Noul decision for {key}")
        nouls[key] = decision.probability
    return Reading(noul=nouls, **choices)


def read_chat(reply: str, candidates: Sequence[str]) -> Reading:
    """Read a chat reply whose choices must match its requested schema."""
    data = json.loads(reply)
    if not isinstance(data, Mapping):
        raise TypeError("chat reply must be a JSON object")
    schema = chat_schema(candidates)["json_schema"]["schema"]["properties"]
    choice_keys = ("kind", "op", "bound", "low", "high")
    if set(data) != {*choice_keys, *NOUL_KEYS}:
        raise ValueError("chat reply fields do not match the reading schema")
    choices: dict[str, str] = {}
    for key in choice_keys:
        value = data[key]
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string")
        if value not in schema[key]["enum"]:
            raise ValueError(f"{key} is not an allowed choice")
        choices[key] = value
    nouls: dict[str, float] = {}
    for key in NOUL_KEYS:
        value = data[key]
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be a boolean")
        nouls[key] = 1.0 if value else 0.0
    return Reading(noul=nouls, **choices)


def resolve(
    reading: Reading,
    candidates: Mapping[str, int],
    band: tuple[float, float],
    *,
    cutoffs: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """The check a reading supports, or None when the criterion stays unresolved.

    A yes/no judgment at or above its cutoff that the criterion is partial,
    conditional or approximate leaves it unresolved. Without per-judgment
    cutoffs, all three use the band's lower edge. For JSON, a negation at or
    above ``high`` becomes a negated check and one between the band edges is
    uncertain.
    """
    low, high = band
    thresholds = {
        key: cutoffs.get(key, low) if cutoffs is not None else low for key in JUDGMENTS
    }
    if reading.kind not in {"word_count", "sentence_count", "valid_json"}:
        return None
    if any(reading.noul[key] >= thresholds[key] for key in JUDGMENTS):
        return None
    if reading.kind == "valid_json":
        negation = reading.noul["negated"]
        if low < negation < high:
            return None
        return {"kind": "valid_json", "negated": negation >= high}
    if reading.op == "between":
        ends = {
            candidates[end] for end in (reading.low, reading.high) if end in candidates
        }
        if len(ends) != 2:
            return None
        bound, upper = sorted(ends)
        return {
            "kind": reading.kind,
            "operator": "between",
            "bound": bound,
            "upper_bound": upper,
        }
    if reading.op not in OPERATORS or reading.bound not in candidates:
        return None
    return {
        "kind": reading.kind,
        "operator": reading.op,
        "bound": candidates[reading.bound],
        "upper_bound": None,
    }


def regex_check(criterion: str) -> dict[str, Any] | None:
    """The current ``check_criterion`` result, in the fixture's label shape."""
    exact = check_criterion(criterion, "").exact
    if exact is None:
        return None
    if exact["kind"] == "valid_json":
        return {"kind": "valid_json", "negated": False}
    if exact["kind"] == "word_count":
        return {
            "kind": "word_count",
            "operator": exact["operator"],
            "bound": exact["bound"],
            "upper_bound": exact.get("upper_bound"),
        }
    return {"kind": exact["kind"]}


def outcome(check: dict[str, Any] | None, expected: dict[str, Any] | None) -> str:
    if check is None:
        return "missed" if expected is not None else "correct_abstain"
    return "correct_check" if check == expected else "wrong_confident"


def score(
    cases: Sequence[Mapping[str, Any]], checks: Mapping[str, dict[str, Any] | None]
) -> Counter[str]:
    return Counter(outcome(checks[c["id"]], c["expected"]) for c in cases)


def _fixture_cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def fixture_splits() -> tuple[str, ...]:
    """Split names present in the fixture, in fixture order."""
    return tuple(dict.fromkeys(case["split"] for case in _fixture_cases()))


def load_cases(split: str) -> list[dict[str, Any]]:
    cases = _fixture_cases()
    splits = tuple(dict.fromkeys(case["split"] for case in cases))
    if split != "all" and split not in splits:
        raise ValueError(
            f"unknown split {split!r}; available splits: {', '.join(splits)}"
        )
    return [case for case in cases if split == "all" or case["split"] == split]


def chat_messages(criterion: str, candidates: Sequence[str]) -> list[dict[str, str]]:
    questions = reading_questions(candidates)
    lines = [
        "Read the success criterion in the user message and answer every question "
        "below. Reply with one JSON object using each question's name as a key. "
        "Choice questions take one of the listed options; yes/no questions take "
        "true or false.",
        "",
    ]
    for key, question in questions.items():
        lines.append(f"{key}: {question['instructions']}")
        if question["type"] == "choice":
            for option, meaning in question["criteria"].items():
                lines.append(f"  - {option}" + (f": {meaning}" if meaning else ""))
    return [
        {"role": "system", "content": "\n".join(lines)},
        {"role": "user", "content": criterion},
    ]


def chat_schema(candidates: Sequence[str]) -> dict[str, Any]:
    questions = reading_questions(candidates)
    properties: dict[str, Any] = {
        key: {"type": "string", "enum": list(question["criteria"])}
        if question["type"] == "choice"
        else {"type": "boolean"}
        for key, question in questions.items()
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "criterion_reading",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def record(
    reader: str,
    cases: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    max_tokens: int = 300,
    model: str | None = None,
) -> None:
    config = GatewayConfig.from_env()
    if reader == "jev" and model is not None:
        config = replace(config, jev_model=model)
    gateway = HttpGateway(config=config)
    model = model or (CHEAP_MODEL if reader == "cheap" else gateway.jev_model)
    new_recording: dict[str, Any] = {
        "schema_version": 1,
        "reader": reader,
        "model": model,
        "questions_digest": questions_digest(),
        "rows": [],
    }
    if reader == "cheap":
        new_recording["max_tokens"] = max_tokens
    saved: dict[str, Any] = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else new_recording
    )
    if saved["reader"] != reader or saved["questions_digest"] != questions_digest():
        raise ValueError("recording belongs to a different reader or question wording")
    if saved.get("model") != model:
        raise ValueError("recording belongs to a different model")
    if reader == "cheap" and saved.get("max_tokens", 300) != max_tokens:
        raise ValueError("recording belongs to a different max_tokens setting")
    done = {row["case_id"] for row in saved["rows"]}
    for case in cases:
        if case["id"] in done:
            continue
        candidates = number_candidates(case["criterion"])
        row: dict[str, Any] = {"case_id": case["id"]}
        if reader == "jev":
            questions = reading_questions(list(candidates))
            answers = gateway.decide_batch(
                [
                    {**question, "key": key, "state": case["criterion"]}
                    for key, question in questions.items()
                ]
            )
            entry = gateway.decision_log[-1]
            row |= {
                "answers": dict(zip(questions, answers, strict=True)),
                "answered_by": entry["answered_by"],
                "usage": entry["usage"],
            }
        else:
            response = gateway.chat(
                model,
                chat_messages(case["criterion"], list(candidates)),
                role="writer",
                temperature=0,
                max_tokens=max_tokens,
                response_format=chat_schema(list(candidates)),
            )
            row |= {
                "reply": completion_text(response),
                "answered_by": response.get("model"),
                "usage": response.get("usage", {}),
            }
            finish_reason = response.get("finish_reason")
            choices = response.get("choices")
            if finish_reason is None and isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                row["finish_reason"] = finish_reason
        saved["rows"].append(row)
        save_json(output, saved)
        print(f"{case['id']} {len(saved['rows'])}/{len(cases)}", flush=True)


def _reading(
    recording: Mapping[str, Any],
    row: Mapping[str, Any],
    candidates: Sequence[str] = (),
) -> Reading | None:
    try:
        if recording["reader"] == "jev":
            return read_jev(row["answers"])
        return read_chat(row["reply"], candidates)
    except (ValueError, KeyError, TypeError):
        return None


def report(
    recording_path: Path,
    *,
    show_errors: bool,
    cutoffs: Mapping[str, float] | None = None,
) -> None:
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    if recording.get("questions_digest") != questions_digest():
        print(
            "WARNING: recording question wording digest does not match the current "
            "fixture questions; scores may be unusable.\n"
        )
    rows = {row["case_id"]: row for row in recording["rows"]}
    cases = [case for case in load_cases("all") if case["id"] in rows]
    usage = [row.get("usage") or {} for row in rows.values()]
    tokens = sum(
        item.get("input_tokens", item.get("prompt_tokens", 0)) for item in usage
    )
    tokens += sum(
        item.get("output_tokens", item.get("completion_tokens", 0)) for item in usage
    )
    # A bring-your-own-key route bills the provider directly and reports cost 0.
    cost = sum(
        float(
            item.get("cost")
            or item.get("cost_details", {}).get("upstream_inference_cost")
            or 0
        )
        for item in usage
    )
    print(
        f"# {recording['reader']} ({recording['model']}): {len(cases)} criteria, "
        f"{tokens / max(len(cases), 1):.0f} tokens and ${cost / max(len(cases), 1):.6f} "
        f"each, ${cost:.4f} total\n"
    )
    readings = {
        case["id"]: _reading(
            recording,
            rows[case["id"]],
            list(number_candidates(case["criterion"])),
        )
        for case in cases
    }
    unread = [case["id"] for case in cases if readings[case["id"]] is None]
    if unread:
        print(f"unusable answers: {', '.join(unread)}\n")
    max_tokens = recording.get("max_tokens")
    if max_tokens is None and recording.get("reader") == "cheap":
        # Legacy cheap recordings were made with the hard-coded 300-token cap.
        max_tokens = 300
    if isinstance(max_tokens, int):
        truncation_suspects: list[str] = []
        for case in cases:
            row = rows[case["id"]]
            if "finish_reason" in row:
                truncated = row["finish_reason"] == "length"
            else:
                row_usage = row.get("usage") or {}
                completion_tokens = row_usage.get(
                    "completion_tokens", row_usage.get("output_tokens")
                )
                truncated = completion_tokens == max_tokens
            if truncated:
                truncation_suspects.append(case["id"])
        if truncation_suspects:
            print(
                f"truncation-suspect: {', '.join(truncation_suspects)} "
                f"(finish_reason == 'length' or completion/output tokens == "
                f"max_tokens={max_tokens})\n"
            )
    _print_distribution(cases, readings)
    print("| split | policy | correct check | correct abstain | missed | wrong |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    regexes = {case["id"]: regex_check(case["criterion"]) for case in cases}
    for split in fixture_splits():
        subset = [case for case in cases if case["split"] == split]
        if not subset:
            continue
        tally = score(subset, regexes)
        print(_row(split, "regex `check_criterion`", tally))
        for band in BANDS:
            checks = {
                case["id"]: (
                    resolve(
                        reading,
                        number_candidates(case["criterion"]),
                        band,
                    )
                    if (reading := readings[case["id"]]) is not None
                    else None
                )
                for case in subset
            }
            tally = score(subset, checks)
            label = f"unresolved if p >= {band[0]}" + (
                f", JSON negation p >= {band[1]}" if band[0] != band[1] else ""
            )
            print(_row(split, label, tally))
            if show_errors and band == BANDS[0]:
                _print_errors(subset, checks, rows, recording)
        if cutoffs is not None:
            custom_checks = {
                case["id"]: (
                    resolve(
                        reading,
                        number_candidates(case["criterion"]),
                        BANDS[0],
                        cutoffs=cutoffs,
                    )
                    if (reading := readings[case["id"]]) is not None
                    else None
                )
                for case in subset
            }
            thresholds = {key: cutoffs.get(key, BANDS[0][0]) for key in JUDGMENTS}
            label = "per-judgment cutoffs " + ", ".join(
                f"{key} >= {thresholds[key]:g}" for key in JUDGMENTS
            )
            label += f"; JSON negation p >= {BANDS[0][1]:g}"
            print(_row(split, label, score(subset, custom_checks)))
            if show_errors:
                _print_errors(subset, custom_checks, rows, recording)


def _print_distribution(
    cases: Sequence[Mapping[str, Any]],
    readings: Mapping[str, Reading | None],
) -> None:
    print(
        "Observed judgment probabilities on checkable criteria "
        "(fixture expected != null; nearest-rank p90):"
    )
    for judgment in JUDGMENTS:
        probabilities: list[float] = []
        for case in cases:
            reading = readings[case["id"]]
            if case["expected"] is not None and reading is not None:
                probabilities.append(reading.noul[judgment])
        probabilities.sort()
        if not probabilities:
            print(f"  {judgment}: n=0 (no usable checkable readings)")
            continue
        p90 = probabilities[math.ceil(0.9 * len(probabilities)) - 1]
        print(
            f"  {judgment}: n={len(probabilities)} "
            f"median={statistics.median(probabilities):.3f} "
            f"p90={p90:.3f} max={max(probabilities):.3f}"
        )
    negations = [
        reading.noul["negated"]
        for case in cases
        if case["expected"] is not None
        and (reading := readings[case["id"]]) is not None
        and reading.kind == "valid_json"
    ]
    negations.sort()
    if not negations:
        print("  negated: n=0 (no usable valid_json readings)")
    else:
        p90 = negations[math.ceil(0.9 * len(negations)) - 1]
        print(
            f"  negated: n={len(negations)} "
            f"median={statistics.median(negations):.3f} "
            f"p90={p90:.3f} max={max(negations):.3f}"
        )

    print()


def _row(split: str, label: str, tally: Mapping[str, int]) -> str:
    return (
        f"| {split} | {label} | {tally['correct_check']} | "
        f"{tally['correct_abstain']} | {tally['missed']} | {tally['wrong_confident']} |"
    )


def _print_errors(
    cases: Sequence[Mapping[str, Any]],
    checks: Mapping[str, dict[str, Any] | None],
    rows: Mapping[str, Any],
    recording: Mapping[str, Any],
) -> None:
    print()
    for case in cases:
        kind = outcome(checks[case["id"]], case["expected"])
        if kind in {"missed", "wrong_confident"}:
            reading = _reading(
                recording,
                rows[case["id"]],
                list(number_candidates(case["criterion"])),
            )
            print(
                f"  {kind}: {case['criterion']!r}\n    expected {case['expected']}"
                f"\n    got      {checks[case['id']]}\n    read     {reading}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure reading success criteria with a model (issue #116)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="record a reader's raw answers (live calls)")
    run.add_argument("--reader", choices=("jev", "cheap"), required=True)
    run.add_argument("--split", choices=(*fixture_splits(), "all"), default="all")
    run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="completion token limit for the cheap reader (default: 300)",
    )
    run.add_argument(
        "--model", type=parse_model, help="model override (max 200 characters)"
    )
    run.add_argument("--output", type=Path, required=True)
    rep = commands.add_parser("report", help="score a recording offline")
    rep.add_argument("--recording", type=Path, required=True)
    rep.add_argument("--errors", action="store_true", help="list first-band errors")
    rep.add_argument(
        "--cutoff",
        type=parse_cutoffs,
        help="per-judgment cutoffs, e.g. partial=0.5,conditional=0.5,approximate=0.5",
    )
    args = parser.parse_args()
    if args.command == "run":
        if args.max_tokens is not None and args.max_tokens <= 0:
            parser.error("--max-tokens must be a positive integer")
        if args.max_tokens is not None and args.max_tokens > MAX_COMPLETION_TOKENS:
            parser.error(f"--max-tokens must be at most {MAX_COMPLETION_TOKENS}")
        if args.reader == "jev" and args.max_tokens is not None:
            parser.error(
                "--max-tokens applies only to the cheap chat reader (--reader cheap)"
            )
        record(
            args.reader,
            load_cases(args.split),
            args.output,
            max_tokens=300 if args.max_tokens is None else args.max_tokens,
            model=args.model,
        )
    else:
        report(args.recording, show_errors=args.errors, cutoffs=args.cutoff)


if __name__ == "__main__":
    main()
