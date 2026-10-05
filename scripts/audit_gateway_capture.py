"""Audit a recording against an independently retained ordered request-key list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prompt_enhancer.evaluation.capture_audit import validate_capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--expected-keys", type=Path, required=True)
    args = parser.parse_args()
    try:
        bundle = json.loads(args.capture.read_text())
        expected = json.loads(args.expected_keys.read_text())
        if not isinstance(expected, list) or not all(
            isinstance(key, str) for key in expected
        ):
            raise ValueError("expected keys must be an ordered JSON string array")
        receipt = validate_capture(
            bundle["request_captures"], expected, bundle["responses"]
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
