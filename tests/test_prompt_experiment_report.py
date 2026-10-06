from __future__ import annotations

import json

import pytest

from prompt_enhancer.evaluation.prompt_experiment_report import (
    PromptExperimentReportError,
    build_prompt_experiment_report,
)


def _inputs():
    answers = [
        {
            "variant": "original",
            "model": "model-a",
            "sample_index": 0,
            "prompt": "Answer accurately in three sentences.",
            "answer": "Plants use light. They make food. They release oxygen.",
        },
        {
            "variant": "rewrite",
            "model": "model-a",
            "sample_index": 0,
            "prompt": "Please give an accurate answer in three sentences.",
            "answer": "Here is an answer:\n\nPlants use light. They make food. They release oxygen.",
        },
    ]
    metrics = {
        "rows": [
            {"id": "sample0", "variant": "original", "model": "model-a"},
            {"id": "sample1", "variant": "rewrite", "model": "model-a"},
        ]
    }
    judgments = {
        "judgments": [
            {
                "tool": "jev_classify",
                "results": [
                    {
                        "id": "sample0",
                        "classification": "acceptable",
                        "decision": "auto",
                    },
                    {
                        "id": "sample1",
                        "classification": "scientific_error",
                        "decision": "review",
                    },
                ],
            },
            {
                "tool": "jev_rerank",
                "ranked": [
                    {"id": "sample1", "rank": 1, "relevance": 0.99},
                    {"id": "sample0", "rank": 2, "relevance": 0.8},
                ],
            },
            {"tool": "jev_review", "action": "escalate"},
            {"tool": "jev_gate", "action": "escalate"},
        ]
    }
    optimizer = {
        "optimizer_status": "interrupted",
        "last_observed_stage": "writing_tests",
        "cancel_requested": True,
        "final_optimizer_result": None,
        "local_api_stopped": True,
    }
    return answers, metrics, judgments, optimizer


def _complete_criteria_manifest(answers):
    return {
        "prompt": answers[0]["prompt"],
        "all_original_criteria_listed": True,
        "criteria": [
            {
                "id": "three_explanatory_sentences",
                "status": "measured",
                "method": "terminal_punctuation_count",
            },
            {
                "id": "scientific_accuracy",
                "status": "measured",
                "method": "jev_classify",
            },
        ],
    }


def test_report_leads_with_optimizer_state_then_separate_matched_samples():
    answers, metrics, judgments, optimizer = _inputs()

    report = build_prompt_experiment_report(
        answers,
        metrics,
        judgments,
        optimizer,
        criteria_manifest=_complete_criteria_manifest(answers),
    )

    assert list(report)[:3] == ["schema_version", "status", "prompt_improvement_run"]
    assert report["status"] == "prompt_improvement_interrupted"
    run = report["prompt_improvement_run"]
    assert run["status"] == "interrupted"
    assert run["last_observed_stage"] == "writing_tests"
    assert run["final_result_returned"] is False
    assert run["convergence_result"] is None

    samples = report["sample_experiment"]
    assert samples["matched_pairs"] == 1
    assert samples["criteria"]["three_explanatory_sentences"]["original"]["passed"] == 1
    assert samples["criteria"]["three_explanatory_sentences"]["rewrite"]["passed"] == 1
    assert samples["criteria"]["experimental_no_intro"]["original"]["passed"] == 1
    assert samples["criteria"]["experimental_no_intro"]["rewrite"]["passed"] == 0
    assert samples["comparison"]["improvement_status"] == "not_established"
    assert samples["comparison"]["rule"]["matched_by"] == ["model", "sample_index"]
    assert samples["comparison"]["paired_regressions"] == 0
    assert samples["comparison"]["unresolved_pairs"] == 1
    science = next(
        item
        for item in samples["criteria_coverage"]["measured_criteria"]
        if item["id"] == "scientific_accuracy"
    )
    assert science["per_sample_outcomes"]["sample1"] == "unresolved"


def test_unresolved_quality_cannot_be_overridden_by_high_relevance():
    answers, metrics, judgments, optimizer = _inputs()

    samples = build_prompt_experiment_report(answers, metrics, judgments, optimizer)[
        "sample_experiment"
    ]

    rewrite = samples["answers"][1]
    assert rewrite["quality"]["status"] == "unresolved"
    assert rewrite["relevance_rank"] == 1
    assert rewrite["quality"]["status"] != "passed"
    assert samples["comparison"]["improvement_status"] == "not_established"


def test_prompt_edit_review_and_gate_are_not_counted_as_answer_quality_evidence():
    answers, metrics, judgments, optimizer = _inputs()

    samples = build_prompt_experiment_report(answers, metrics, judgments, optimizer)[
        "sample_experiment"
    ]

    scope = samples["jev_scope"]
    assert scope["used_for_sample_decisions"] == ["jev_classify", "jev_rerank"]
    assert scope["not_used_for_sample_decisions"] == ["jev_gate", "jev_review"]


def test_report_can_only_establish_improvement_when_matched_quality_evidence_passes():
    answers, metrics, judgments, optimizer = _inputs()
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "acceptable", "decision": "auto"},
    ]

    report = build_prompt_experiment_report(
        answers,
        metrics,
        judgments,
        optimizer,
        criteria_manifest=_complete_criteria_manifest(answers),
    )

    comparison = report["sample_experiment"]["comparison"]
    assert comparison["improvement_status"] == "improvement_established"
    assert comparison["successful_pairs"] == {"original": 0, "rewrite": 1}
    assert comparison["unresolved_quality_judgments"] == 0
    assert report["sample_experiment"]["criteria_coverage"]["complete"] is True

    incomplete = build_prompt_experiment_report(
        answers,
        metrics,
        judgments,
        optimizer,
        criteria_manifest={
            **_complete_criteria_manifest(answers),
            "criteria": [
                *_complete_criteria_manifest(answers)["criteria"],
                {
                    "id": "audience_age",
                    "status": "unmeasured",
                    "method": "not_evaluated",
                },
            ],
        },
    )
    assert (
        incomplete["sample_experiment"]["comparison"]["improvement_status"]
        == "not_established"
    )
    assert incomplete["sample_experiment"]["criteria_coverage"][
        "unmeasured_criteria"
    ] == ["audience_age"]


def test_omitted_original_criteria_manifest_never_establishes_improvement():
    answers, metrics, judgments, optimizer = _inputs()
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "acceptable", "decision": "auto"},
    ]

    report = build_prompt_experiment_report(answers, metrics, judgments, optimizer)

    assert (
        report["sample_experiment"]["comparison"]["improvement_status"]
        == "not_established"
    )
    assert report["sample_experiment"]["criteria_coverage"]["complete"] is False
    assert (
        report["sample_experiment"]["criteria_coverage"]["manifest_status"] == "missing"
    )


def test_criteria_manifest_must_be_bound_to_original_prompt():
    answers, metrics, judgments, optimizer = _inputs()
    manifest = _complete_criteria_manifest(answers)
    manifest["prompt"] = "A different original request."

    with pytest.raises(
        PromptExperimentReportError, match="must match the original prompt"
    ):
        build_prompt_experiment_report(
            answers, metrics, judgments, optimizer, criteria_manifest=manifest
        )


def test_criteria_manifest_rejects_duplicate_ids_and_unsupported_measurements():
    answers, metrics, judgments, optimizer = _inputs()
    manifest = _complete_criteria_manifest(answers)
    manifest["criteria"].append(dict(manifest["criteria"][0]))
    with pytest.raises(PromptExperimentReportError, match="duplicate id"):
        build_prompt_experiment_report(
            answers, metrics, judgments, optimizer, criteria_manifest=manifest
        )

    manifest = _complete_criteria_manifest(answers)
    manifest["criteria"][0]["method"] = "unrelated_claim"
    with pytest.raises(
        PromptExperimentReportError, match="no matching report evidence"
    ):
        build_prompt_experiment_report(
            answers, metrics, judgments, optimizer, criteria_manifest=manifest
        )


def test_complete_manifest_requires_exhaustive_criteria_attestation():
    answers, metrics, judgments, optimizer = _inputs()
    manifest = _complete_criteria_manifest(answers)
    manifest["all_original_criteria_listed"] = False
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "acceptable", "decision": "auto"},
    ]

    report = build_prompt_experiment_report(
        answers, metrics, judgments, optimizer, criteria_manifest=manifest
    )

    assert report["sample_experiment"]["criteria_coverage"]["complete"] is False
    assert (
        report["sample_experiment"]["comparison"]["improvement_status"]
        == "not_established"
    )


def test_cli_without_criteria_manifest_cannot_establish_improvement(
    tmp_path, monkeypatch
):
    answers, metrics, judgments, optimizer = _inputs()
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "acceptable", "decision": "auto"},
    ]
    paths = {}
    for name, value in {
        "answers": answers,
        "metrics": metrics,
        "jev": judgments,
        "optimizer": optimizer,
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-experiment-report",
            "--answers",
            str(paths["answers"]),
            "--metrics",
            str(paths["metrics"]),
            "--jev",
            str(paths["jev"]),
            "--optimizer-status",
            str(paths["optimizer"]),
            "--output",
            str(output),
        ],
    )

    from prompt_enhancer.evaluation.prompt_experiment_report import main

    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert (
        report["sample_experiment"]["comparison"]["improvement_status"]
        == "not_established"
    )
    assert (
        report["sample_experiment"]["criteria_coverage"]["manifest_status"] == "missing"
    )

    manifest_path = tmp_path / "criteria.json"
    manifest_path.write_text(
        json.dumps(_complete_criteria_manifest(answers)), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-experiment-report",
            "--answers",
            str(paths["answers"]),
            "--metrics",
            str(paths["metrics"]),
            "--jev",
            str(paths["jev"]),
            "--optimizer-status",
            str(paths["optimizer"]),
            "--criteria-manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    complete_report = json.loads(output.read_text(encoding="utf-8"))
    assert (
        complete_report["sample_experiment"]["comparison"]["improvement_status"]
        == "improvement_established"
    )


def test_auto_scientific_error_is_a_failure_but_review_is_unresolved():
    answers, metrics, judgments, optimizer = _inputs()
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "scientific_error", "decision": "review"},
    ]

    results = build_prompt_experiment_report(answers, metrics, judgments, optimizer)[
        "sample_experiment"
    ]["answers"]

    assert results[0]["quality"]["status"] == "failed"
    assert results[1]["quality"]["status"] == "unresolved"

    judgments["judgments"][0]["results"] = []
    missing = build_prompt_experiment_report(answers, metrics, judgments, optimizer)[
        "sample_experiment"
    ]["answers"]
    assert all(item["quality"]["status"] == "unresolved" for item in missing)


def test_sample_metrics_must_stay_aligned_with_answer_identity():
    answers, metrics, judgments, optimizer = _inputs()
    metrics["rows"][0]["model"] = "different-model"

    with pytest.raises(PromptExperimentReportError, match="does not match"):
        build_prompt_experiment_report(answers, metrics, judgments, optimizer)


def test_missing_or_duplicate_sample_pair_fails_closed():
    answers, metrics, judgments, optimizer = _inputs()
    answers[1]["sample_index"] = 1
    with pytest.raises(PromptExperimentReportError, match="matched pair"):
        build_prompt_experiment_report(answers, metrics, judgments, optimizer)

    answers, metrics, judgments, optimizer = _inputs()
    answers.append(dict(answers[0]))
    metrics["rows"].append({"id": "sample2", "variant": "original", "model": "model-a"})
    with pytest.raises(PromptExperimentReportError, match="duplicate"):
        build_prompt_experiment_report(answers, metrics, judgments, optimizer)


@pytest.mark.parametrize("variant", ["original", "rewrite"])
def test_each_variant_requires_one_consistent_prompt(variant):
    answers, metrics, judgments, optimizer = _inputs()
    offset = 0 if variant == "original" else 1
    extra = dict(answers[offset], sample_index=1, prompt="A different request.")
    other = dict(answers[1 - offset], sample_index=1)
    answers.extend([extra, other])
    metrics["rows"].extend(
        [
            {"id": "extra0", "variant": extra["variant"], "model": extra["model"]},
            {"id": "extra1", "variant": other["variant"], "model": other["model"]},
        ]
    )
    with pytest.raises(PromptExperimentReportError, match="same .* prompt"):
        build_prompt_experiment_report(answers, metrics, judgments, optimizer)


def test_identical_variant_prompts_cannot_establish_rewrite_improvement():
    answers, metrics, judgments, optimizer = _inputs()
    answers[1]["prompt"] = answers[0]["prompt"]
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "acceptable", "decision": "auto"},
    ]
    with pytest.raises(PromptExperimentReportError, match="distinct"):
        build_prompt_experiment_report(
            answers,
            metrics,
            judgments,
            optimizer,
            criteria_manifest=_complete_criteria_manifest(answers),
        )


def test_quality_judgments_match_arbitrary_string_sample_ids():
    answers, metrics, judgments, optimizer = _inputs()
    for index, metric in enumerate(metrics["rows"]):
        metric["id"] = f"case-{index}"
    judgments["judgments"][0]["results"] = [
        {"id": "case-0", "classification": "scientific_error", "decision": "auto"},
        {"id": "case-1", "classification": "acceptable", "decision": "auto"},
    ]
    samples = build_prompt_experiment_report(
        answers,
        metrics,
        judgments,
        optimizer,
        criteria_manifest=_complete_criteria_manifest(answers),
    )["sample_experiment"]
    assert samples["comparison"]["improvement_status"] == "improvement_established"
    assert [x["quality"]["status"] for x in samples["answers"]] == ["failed", "passed"]


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_classifications_fail_closed_in_either_order(reverse):
    answers, metrics, judgments, optimizer = _inputs()
    entries = [
        {"id": "sample1", "classification": "scientific_error", "decision": "auto"},
        {"id": "sample1", "classification": "acceptable", "decision": "auto"},
    ]
    if reverse:
        entries.reverse()
    judgments["judgments"][0]["results"] = [
        {"id": "sample0", "classification": "scientific_error", "decision": "auto"},
        entries[0],
    ]
    judgments["judgments"].append({"tool": "jev_classify", "results": [entries[1]]})
    with pytest.raises(PromptExperimentReportError, match="conflicting classification"):
        build_prompt_experiment_report(
            answers,
            metrics,
            judgments,
            optimizer,
            criteria_manifest=_complete_criteria_manifest(answers),
        )
