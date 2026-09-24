import json

from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.success_tests import SuccessTestCompiler


def test_calibrated_faithfulness_gate_accepts_boundary_probability() -> None:
    tests = {"tests": [
        {"id": "accepted", "question": "Does the response give the requested summary?", "kind": "noul", "expected": "yes"},
        {"id": "rejected", "question": "Does the response add a chart?", "kind": "noul", "expected": "yes"},
    ]}
    faithfulness = {"faithful:accepted": 0.8, "faithful:rejected": 0.79}
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: {"choices": [{"message": {"content": json.dumps(tests)}}]},
        decision=lambda request, **_kwargs: {"type": "noul", "noul": faithfulness[request["key"]]},
    )

    compiled = SuccessTestCompiler(gateway).compile("Summarize this article.")

    assert [test.id for test in compiled.tests] == ["accepted"]
    assert [item.test.id for item in compiled.rejected] == ["rejected"]
    assert all(check.threshold == 0.8 for check in compiled.faithfulness_checks)
