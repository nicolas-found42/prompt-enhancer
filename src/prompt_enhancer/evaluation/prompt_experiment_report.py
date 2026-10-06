"""Build a fail-closed, status-first report for paired prompt samples.

This report keeps requested success criteria separate from stricter experiment
checks, treats Jev review decisions as unresolved, and uses relevance only as an
ordering signal. It does not infer prompt improvement from a high-ranked answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class PromptExperimentReportError(ValueError):
    """The supplied sample evidence cannot support a comparable report."""


_TERMINAL_PUNCTUATION = re.compile(r"[.!?](?:[\"’”'])?(?=\s|$)")
_INTRO_PREFIX = re.compile(
    r"^(?:here(?:'s| is)\b|below is\b|let(?:'s| us)\b|sure\b|okay\b)",
    re.IGNORECASE,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptExperimentReportError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PromptExperimentReportError(f"{label} must be an array")
    return value


def _has_intro(answer: str) -> bool:
    """Find common prefatory text before a blank-line separated answer body."""
    paragraphs = re.split(r"\n\s*\n", answer.strip(), maxsplit=1)
    if len(paragraphs) < 2:
        return False
    lead = paragraphs[0].strip()
    return bool(
        lead
        and (
            _INTRO_PREFIX.search(lead)
            or lead.endswith(":")
            or not _TERMINAL_PUNCTUATION.search(lead)
        )
    )


def _classification_results(
    judgments: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    found: dict[str, Mapping[str, Any]] = {}
    entries = _sequence(judgments.get("judgments", ()), "judgments")
    for item in entries:
        if not isinstance(item, Mapping) or item.get("tool") != "jev_classify":
            continue
        for result in _sequence(item.get("results", ()), "jev_classify results"):
            if not isinstance(result, Mapping):
                continue
            sample_id = result.get("id")
            if isinstance(sample_id, str):
                previous = found.get(sample_id)
                if previous is not None and (
                    previous.get("classification") != result.get("classification")
                    or previous.get("decision") != result.get("decision")
                ):
                    raise PromptExperimentReportError(
                        f"conflicting classification for sample {sample_id!r}"
                    )
                found[sample_id] = result
    return found


def _relevance_ranks(judgments: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    found: dict[str, Mapping[str, Any]] = {}
    entries = _sequence(judgments.get("judgments", ()), "judgments")
    for item in entries:
        if not isinstance(item, Mapping) or item.get("tool") != "jev_rerank":
            continue
        for result in _sequence(item.get("ranked", ()), "jev_rerank ranked"):
            if isinstance(result, Mapping) and isinstance(result.get("id"), str):
                found[result["id"]] = result
    return found


def _answer_quality(judgment: Mapping[str, Any] | None) -> dict[str, Any]:
    if judgment is None or judgment.get("decision") != "auto":
        return {
            "status": "unresolved",
            "classification": judgment.get("classification") if judgment else None,
            "decision": judgment.get("decision") if judgment else None,
        }
    classification = judgment.get("classification")
    if classification == "acceptable":
        status = "passed"
    elif classification == "scientific_error":
        status = "failed"
    else:
        status = "unresolved"
    return {
        "status": status,
        "classification": classification,
        "decision": "auto",
    }


def _optimizer_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    final = raw.get("final_optimizer_result")
    final_result = final if isinstance(final, Mapping) else None
    status = str(raw.get("optimizer_status") or "unknown")
    return {
        "status": status,
        "last_observed_stage": raw.get("last_observed_stage"),
        "cancel_requested": raw.get("cancel_requested") is True,
        "final_result_returned": final_result is not None,
        "final_optimizer_result": final_result,
        "final_prompt_returned": bool(
            final_result and isinstance(final_result.get("final_prompt"), str)
        ),
        "convergence_result": (
            final_result.get("report", {}).get("convergence")
            if final_result and isinstance(final_result.get("report"), Mapping)
            else None
        ),
        "local_api_stopped": raw.get("local_api_stopped") is True,
        "reason": raw.get("reason"),
    }


def _criteria_coverage(
    manifest: Mapping[str, Any] | None,
    *,
    original_prompt: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate an exhaustive, prompt-bound declaration of measured criteria."""
    if manifest is None:
        return {
            "manifest_status": "missing",
            "complete": False,
            "original_prompt_sha256": None,
            "all_original_criteria_listed": False,
            "measured_criteria": [],
            "unmeasured_criteria": [],
        }

    manifest_prompt = manifest.get("prompt")
    if not isinstance(manifest_prompt, str) or manifest_prompt != original_prompt:
        raise PromptExperimentReportError(
            "criteria manifest prompt must match the original prompt"
        )
    exhaustive = manifest.get("all_original_criteria_listed")
    if not isinstance(exhaustive, bool):
        raise PromptExperimentReportError(
            "criteria manifest must attest whether it lists all original criteria"
        )
    criteria = _sequence(manifest.get("criteria"), "criteria_manifest.criteria")
    if not criteria:
        raise PromptExperimentReportError(
            "criteria manifest must list original criteria"
        )

    supported_methods = {
        "three_explanatory_sentences": (
            "terminal_punctuation_count",
            lambda item: (
                "passed"
                if item["required_criteria"]["three_explanatory_sentences"]["passed"]
                else "failed"
            ),
        ),
        "scientific_accuracy": (
            "jev_classify",
            lambda item: item["quality"]["status"],
        ),
    }
    measured: list[dict[str, Any]] = []
    unmeasured: list[str] = []
    seen_ids: set[str] = set()
    for index, raw_criterion in enumerate(criteria):
        criterion = _mapping(raw_criterion, f"criteria_manifest.criteria[{index}]")
        criterion_id = criterion.get("id")
        status = criterion.get("status")
        method = criterion.get("method")
        if not isinstance(criterion_id, str) or not criterion_id.strip():
            raise PromptExperimentReportError(
                "criteria manifest ids must be nonempty strings"
            )
        if criterion_id in seen_ids:
            raise PromptExperimentReportError(
                f"criteria manifest contains duplicate id {criterion_id!r}"
            )
        seen_ids.add(criterion_id)
        if status not in {"measured", "unmeasured"}:
            raise PromptExperimentReportError(
                f"criteria manifest status for {criterion_id!r} must be measured or unmeasured"
            )
        if not isinstance(method, str) or not method.strip():
            raise PromptExperimentReportError(
                f"criteria manifest method for {criterion_id!r} must be nonempty"
            )
        if status == "unmeasured":
            unmeasured.append(criterion_id)
            continue
        supported = supported_methods.get(criterion_id)
        if supported is None or method != supported[0]:
            raise PromptExperimentReportError(
                f"measured criterion {criterion_id!r} has no matching report evidence method"
            )
        measured.append(
            {
                "id": criterion_id,
                "status": "measured",
                "method": method,
                "per_sample_outcomes": {
                    str(item["id"]): supported[1](item) for item in records
                },
            }
        )

    complete = exhaustive and bool(measured) and not unmeasured
    return {
        "manifest_status": "complete" if complete else "incomplete",
        "complete": complete,
        "original_prompt_sha256": hashlib.sha256(
            original_prompt.encode("utf-8")
        ).hexdigest(),
        "all_original_criteria_listed": exhaustive,
        "measured_criteria": measured,
        "unmeasured_criteria": unmeasured,
    }


def build_prompt_experiment_report(
    answers: Sequence[Mapping[str, Any]],
    sample_metrics: Mapping[str, Any],
    jev_experiments: Mapping[str, Any],
    optimizer_status: Mapping[str, Any],
    *,
    expected_sentence_count: int = 3,
    original_variant: str = "original",
    criteria_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report one completed paired sample experiment and its main run state.

    Samples must be ordered with an id-aligned metrics row, as in the captured
    Gateway experiment. Pairs are matched by (model, sample_index). The
    improvement rule is deliberately conservative: all answer-quality
    judgments must be auto-resolved, no matched pair may regress on the
    requested sentence-count criterion, and the rewrite must create more
    successful pairs. The experimental no-introduction check never counts as
    an original prompt requirement.
    """
    if expected_sentence_count < 1:
        raise PromptExperimentReportError("expected_sentence_count must be positive")
    metric_rows = _sequence(sample_metrics.get("rows", ()), "sample_metrics.rows")
    if len(answers) != len(metric_rows) or not answers:
        raise PromptExperimentReportError(
            "answers and metric rows must be nonempty and aligned"
        )

    classifications = _classification_results(jev_experiments)
    relevance = _relevance_ranks(jev_experiments)
    judgment_entries = _sequence(jev_experiments.get("judgments", ()), "judgments")
    recorded_tools = sorted(
        {
            item["tool"]
            for item in judgment_entries
            if isinstance(item, Mapping) and isinstance(item.get("tool"), str)
        }
    )
    used_tools = sorted(
        tool
        for tool, predicate in (
            ("jev_classify", bool(classifications)),
            ("jev_rerank", bool(relevance)),
        )
        if predicate
    )
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, int]] = set()
    seen_sample_ids: set[str] = set()
    variant_prompts: dict[str, set[str]] = {}
    for index, (answer_raw, metric_raw) in enumerate(
        zip(answers, metric_rows, strict=True)
    ):
        answer = _mapping(answer_raw, f"answers[{index}]")
        metric = _mapping(metric_raw, f"sample_metrics.rows[{index}]")
        variant = answer.get("variant")
        model = answer.get("model")
        sample_index = answer.get("sample_index")
        text = answer.get("answer")
        variant_prompt = answer.get("prompt")
        sample_id = metric.get("id")
        if (
            not isinstance(variant, str)
            or not variant
            or not isinstance(model, str)
            or not model
            or not isinstance(sample_index, int)
            or isinstance(sample_index, bool)
            or not isinstance(text, str)
            or not isinstance(variant_prompt, str)
            or not isinstance(sample_id, str)
        ):
            raise PromptExperimentReportError(
                f"answers[{index}] has invalid sample fields"
            )
        if metric.get("variant") != variant or metric.get("model") != model:
            raise PromptExperimentReportError(
                f"sample metric row {index} does not match its answer"
            )
        if sample_id in seen_sample_ids:
            raise PromptExperimentReportError(f"duplicate sample id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        variant_prompts.setdefault(variant, set()).add(variant_prompt)
        pair_key = (variant, model, sample_index)
        if pair_key in seen_pairs:
            raise PromptExperimentReportError(
                f"duplicate sample for matched pair {pair_key}"
            )
        seen_pairs.add(pair_key)

        terminals = len(_TERMINAL_PUNCTUATION.findall(text))
        quality = _answer_quality(classifications.get(sample_id))
        rank = relevance.get(sample_id)
        records.append(
            {
                "id": sample_id,
                "variant": variant,
                "model": model,
                "sample_index": sample_index,
                "required_criteria": {
                    "three_explanatory_sentences": {
                        "origin": "original_prompt",
                        "expected": expected_sentence_count,
                        "observed": terminals,
                        "passed": terminals == expected_sentence_count,
                        "method": "terminal_punctuation_count",
                    }
                },
                "experimental_criteria": {
                    "experimental_no_intro": {
                        "origin": "experiment_only",
                        "observed_intro": _has_intro(text),
                        "passed": not _has_intro(text),
                        "method": "prefatory_text_before_blank_line_heuristic",
                    }
                },
                "quality": quality,
                "relevance_rank": rank.get("rank") if rank else None,
                "relevance_score": rank.get("relevance") if rank else None,
            }
        )

    variants = sorted({item["variant"] for item in records})
    if len(variants) != 2 or original_variant not in variants:
        raise PromptExperimentReportError(
            "exactly two variants, including original, are required"
        )
    for variant, prompts in variant_prompts.items():
        if len(prompts) != 1:
            raise PromptExperimentReportError(
                f"all {variant} variant samples must use the same {variant} prompt"
            )
    original_prompt = next(iter(variant_prompts[original_variant]))
    rewrite_variant = next(item for item in variants if item != original_variant)
    if original_prompt.strip() == next(iter(variant_prompts[rewrite_variant])).strip():
        raise PromptExperimentReportError(
            "original and rewrite prompts must be distinct"
        )
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for item in records:
        key = (item["model"], item["sample_index"])
        variant_samples = by_key.setdefault(key, {})
        if item["variant"] in variant_samples:
            raise PromptExperimentReportError(f"duplicate matched pair {key}")
        variant_samples[item["variant"]] = item
    if not by_key or any(set(pair) != set(variants) for pair in by_key.values()):
        raise PromptExperimentReportError(
            "every model/sample index needs one matched pair"
        )

    coverage = _criteria_coverage(
        criteria_manifest, original_prompt=original_prompt, records=records
    )

    criterion_summary: dict[str, Any] = {}
    for criterion in ("three_explanatory_sentences", "experimental_no_intro"):
        summaries: dict[str, Any] = {}
        field = (
            "required_criteria"
            if criterion == "three_explanatory_sentences"
            else "experimental_criteria"
        )
        for variant in variants:
            selected = [item for item in records if item["variant"] == variant]
            passed = sum(item[field][criterion]["passed"] is True for item in selected)
            summaries[variant] = {"passed": passed, "total": len(selected)}
        criterion_summary[criterion] = summaries

    paired_regressions = 0
    original_successes = 0
    rewrite_successes = 0
    quality_unresolved = 0
    unresolved_pairs = 0
    for pair in by_key.values():
        original = pair[original_variant]
        rewrite = pair[rewrite_variant]
        old_pass = original["required_criteria"]["three_explanatory_sentences"][
            "passed"
        ]
        new_pass = rewrite["required_criteria"]["three_explanatory_sentences"]["passed"]
        old_quality = original["quality"]["status"]
        new_quality = rewrite["quality"]["status"]
        quality_unresolved += sum(
            value == "unresolved" for value in (old_quality, new_quality)
        )
        if old_quality == "unresolved" or new_quality == "unresolved":
            unresolved_pairs += 1
        old_success = old_pass and old_quality == "passed"
        new_success = new_pass and new_quality == "passed"
        original_successes += old_success
        rewrite_successes += new_success
        paired_regressions += old_success and (not new_pass or new_quality == "failed")

    improvement_established = (
        quality_unresolved == 0
        and coverage["complete"]
        and paired_regressions == 0
        and rewrite_successes > original_successes
    )
    optimizer = _optimizer_state(_mapping(optimizer_status, "optimizer_status"))
    optimizer_status_label = optimizer["status"]
    return {
        "schema_version": 1,
        "status": (
            "prompt_improvement_interrupted"
            if optimizer_status_label in {"interrupted", "cancelled"}
            else f"prompt_improvement_{optimizer_status_label}"
        ),
        "prompt_improvement_run": optimizer,
        "sample_experiment": {
            "status": "complete",
            "samples": len(records),
            "variants": variants,
            "matched_pairs": len(by_key),
            "criteria_coverage": coverage,
            "criteria": criterion_summary,
            "comparison": {
                "baseline_variant": original_variant,
                "rewrite_variant": rewrite_variant,
                "improvement_status": "improvement_established"
                if improvement_established
                else "not_established",
                "rule": {
                    "matched_by": ["model", "sample_index"],
                    "same_required_criteria_for_both_variants": True,
                    "requires_prompt_bound_exhaustive_criteria_manifest": True,
                    "requires_all_declared_criteria_measured": True,
                    "requires_all_answer_quality_judgments_resolved": True,
                    "requires_no_matched_regressions": True,
                    "requires_more_successful_matched_pairs": True,
                },
                "unresolved_quality_judgments": quality_unresolved,
                "unresolved_pairs": unresolved_pairs,
                "paired_regressions": paired_regressions,
                "successful_pairs": {
                    original_variant: original_successes,
                    rewrite_variant: rewrite_successes,
                },
                "limits": [
                    "This small sample comparison does not establish general model behavior.",
                    "The experimental no-introduction check is reported separately and is not an original prompt requirement.",
                ],
            },
            "answers": records,
            "jev_scope": {
                "recorded_tools": recorded_tools,
                "used_for_sample_decisions": used_tools,
                "not_used_for_sample_decisions": [
                    tool for tool in recorded_tools if tool not in used_tools
                ],
                "explanation": "Jev classification supplies answer-quality judgments and reranking supplies answer order. Relevance ranks do not resolve quality. Other recorded capabilities do not determine per-answer quality or the paired format counts.",
            },
        },
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--optimizer-status", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--criteria-manifest", type=Path)
    args = parser.parse_args()
    report = build_prompt_experiment_report(
        _sequence(_load(args.answers), "answers"),
        _mapping(_load(args.metrics), "sample_metrics"),
        _mapping(_load(args.jev), "jev_experiments"),
        _mapping(_load(args.optimizer_status), "optimizer_status"),
        criteria_manifest=(
            _mapping(_load(args.criteria_manifest), "criteria_manifest")
            if args.criteria_manifest
            else None
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
