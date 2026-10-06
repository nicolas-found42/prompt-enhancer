"""Inspect validation liveness without modifying preserved evidence."""

import argparse
import json
from pathlib import Path

from validation.receipts import receipt_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    value = json.loads(args.receipt.read_text())
    status = receipt_status(value)
    print(
        json.dumps(
            {
                "status": status,
                "recorded_status": value.get("status"),
                "receipt": str(args.receipt),
            }
        )
    )
    return 0 if status in {"active", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
