from prompt_enhancer.evaluation.diagnosis_fanout import compare_diagnosis_reports


def test_matched_diagnosis_comparison_separates_real_latency_from_replay() -> None:
    baseline = {
        "cases": [
            {
                "case_id": "a",
                "predicted_task_type": "writing",
                "predicted_gaps": ["context"],
                "predicted_problem_sentences": [],
                "diagnosis_request_evidence": {
                    "provider_requests": 4,
                    "complete": True,
                    "mode": "sequential",
                    "request_latencies_ms": [10.0, 20.0],
                    "latency_source": "measured_provider",
                },
            }
        ],
        "diagnosis": {"micro": {"precision": 1.0, "recall": 0.5}},
    }
    current = {
        "cases": [
            {
                **baseline["cases"][0],
                "diagnosis_request_evidence": {
                    "provider_requests": 1,
                    "complete": True,
                    "mode": "speculative_fanout",
                    "request_latencies_ms": [0.01],
                    "latency_source": "deterministic_or_replay",
                },
            }
        ],
        "diagnosis": {"micro": {"precision": 1.0, "recall": 0.5}},
    }

    result = compare_diagnosis_reports(baseline, current)

    assert result["deterministic_parity_count"] == 1
    assert result["sequential"]["provider_request_count"] == 4
    assert result["speculative"]["provider_request_count"] == 1
    assert result["sequential"]["latency_p50_ms"] == 30.0
    assert result["speculative"]["latency_p50_ms"] is None
    assert result["speculative"]["gap_precision"] == 1.0
