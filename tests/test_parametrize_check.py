"""The lint CLI catches repeated cases without treating fixtures as values."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_parametrize.py"


def test_unused_direct_parameter_fails_but_used_and_indirect_values_pass(tmp_path):
    source = tmp_path / "test_sample.py"
    source.write_text("""import pytest
@pytest.mark.parametrize("bad_reply", ["{}", "null"])
def test_reply(bad_reply):
    replies = ["{}"]
    assert replies
""")
    failed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True
    )
    assert failed.returncode == 1
    assert "bad_reply" in failed.stdout
    source.write_text("""import pytest
@pytest.mark.parametrize("bad_reply, setup", [("{}", 1)], indirect=["setup"])
def test_reply(bad_reply, setup):
    assert bad_reply
""")
    passed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True
    )
    assert passed.returncode == 0
