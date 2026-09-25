"""Require GitHub's closing link for a PR built on an issue branch."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from typing import Any


def issue_number(branch: str) -> int | None:
    match = re.search(r"(?:^|/)issue-([1-9][0-9]*)(?:-|$)", branch)
    return int(match.group(1)) if match else None


def main(branch: str) -> int:
    number = issue_number(branch)
    if number is None:
        return 0

    try:
        payload: Any = json.load(sys.stdin)
        if not isinstance(payload, Mapping):
            raise ValueError("expected an object")
        references = payload["closingIssuesReferences"]
        if not isinstance(references, list):
            raise ValueError("closingIssuesReferences must be an array")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"Could not inspect GitHub closing issue links: {exc}", file=sys.stderr)
        return 1

    if any(
        isinstance(reference, Mapping) and reference.get("number") == number
        for reference in references
    ):
        print(f"Issue #{number} is linked for closure.")
        return 0

    print(
        f"Branch {branch} targets issue #{number}, but GitHub reports no closing "
        f"link. Add Closes #{number} to the PR body as plain text.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_pr_issue_link.py BRANCH", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
