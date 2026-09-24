import json
from typing import Any

from prompt_enhancer.success_tests import SuccessTestCompiler


def test_calibrated_faithfulness_gate_accepts_boundary_probability() -> None:
    class Gateway:
        def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"choices": [{"message": {"content": json.dumps({"tests": [
                {"id": "accepted", "question": "Does the response give the requested summary?", "kind": "noul", "expected": "yes"},
                {"id": "rejected", "question": "Does the response add a chart?", "kind": "noul", "expected": "yes"},
            ]})}}]}

        def jev_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
            assert [request["key"] for request in requests] == ["faithful:accepted", "faithful:rejected"]
            return [{"type": "noul", "noul": 0.8}, {"type": "noul", "noul": 0.79}]

    compiled = SuccessTestCompiler(Gateway()).compile("Summarize this article.")

    assert [test.id for test in compiled.tests] == ["accepted"]
    assert [item.test.id for item in compiled.rejected] == ["rejected"]
    assert all(check.threshold == 0.8 for check in compiled.faithfulness_checks)
