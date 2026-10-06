"""Capture latest-head advisory review readiness with bounded waiting."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from validation.receipts import write_json


class ReadinessTimeout(TimeoutError):
    pass


def fingerprint(item: dict) -> str:
    fields = {
        key: item.get(key)
        for key in ("id", "body", "updated_at", "state", "path", "line", "side")
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def gh_json(endpoint: str, deadline: float) -> list:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ReadinessTimeout("review wait budget exhausted")
    result = subprocess.run(
        ["gh", "api", endpoint, "--paginate", "--slurp"],
        capture_output=True,
        text=True,
        check=True,
        timeout=min(30, remaining),
    )
    return json.loads(result.stdout)


def assess(
    head: str,
    reviews: list[dict],
    comments: list[dict],
    dispositions: dict,
    *,
    reviewer: str,
) -> dict:
    relevant = [
        r
        for r in reviews
        if r.get("user", {}).get("login") == reviewer
        and r.get("commit_id") == head
        and r.get("state") in {"COMMENTED", "APPROVED", "CHANGES_REQUESTED"}
    ]
    findings = [
        c
        for c in comments
        if c.get("user", {}).get("login") == reviewer and c.get("commit_id") == head
    ]
    ids = [str(c["id"]) for c in findings]
    fingerprints = {str(c["id"]): fingerprint(c) for c in findings}
    unresolved = [
        cid
        for cid in ids
        if not valid_disposition(dispositions.get(cid), fingerprints[cid])
    ]
    return {
        "head": head,
        "reviewer": reviewer,
        "review_ids": [r["id"] for r in relevant],
        "review_fingerprints": {str(r["id"]): fingerprint(r) for r in relevant},
        "comment_ids": ids,
        "comment_fingerprints": fingerprints,
        "unresolved": unresolved,
        "status": "triaged"
        if relevant and not unresolved
        else "needs_triage"
        if relevant
        else "pending",
        "dispositions": {cid: dispositions[cid] for cid in ids if cid in dispositions},
    }


def valid_disposition(value, current_fingerprint: str) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status") in {"fixed", "dismissed", "deferred"}
        and bool(str(value.get("evidence", "")).strip())
        and value.get("fingerprint") == current_fingerprint
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int)
    parser.add_argument("--repo", default="nicolas-found42/prompt-enhancer")
    parser.add_argument("--head", required=True)
    parser.add_argument("--reviewer", default="qodo-code-review[bot]")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--dispositions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not math.isfinite(args.timeout)
        or args.timeout < 0
        or args.timeout > 1800
        or args.output.exists()
    ):
        parser.error("Use a fresh output and a timeout between 0 and 1800 seconds")
    dispositions = (
        json.loads(args.dispositions.read_text()) if args.dispositions else {}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout
    last_signature = None
    stable_since = time.monotonic()
    prefix = f"repos/{args.repo}/pulls/{args.pr}"
    try:
        while True:
            before = gh_json(prefix, deadline)[0]["head"]["sha"]
            reviews = [
                r for page in gh_json(prefix + "/reviews", deadline) for r in page
            ]
            comments = [
                c for page in gh_json(prefix + "/comments", deadline) for c in page
            ]
            after = gh_json(prefix, deadline)[0]["head"]["sha"]
            result = assess(
                args.head, reviews, comments, dispositions, reviewer=args.reviewer
            )
            signature = (result["review_fingerprints"], result["comment_fingerprints"])
            if signature != last_signature:
                last_signature = signature
                stable_since = time.monotonic()
            settling = (
                result["status"] in {"triaged", "needs_triage"}
                and time.monotonic() - stable_since < 10
            )
            if before != args.head or after != args.head:
                result["status"] = "stale_head"
            elif (
                result["status"] == "pending" or settling
            ) and time.monotonic() >= deadline:
                result["status"] = "timed_out"
            elif settling:
                result["status"] = "pending"
            result.update(
                pr=args.pr,
                repository=args.repo,
                captured_at=datetime.now(UTC).isoformat(),
            )
            if result["status"] != "pending":
                write_json(args.output, result)
                print(json.dumps(result))
                return 0 if result["status"] == "triaged" else 1
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    except (ReadinessTimeout, subprocess.TimeoutExpired):
        write_json(
            args.output,
            {
                "status": "timed_out",
                "head": args.head,
                "pr": args.pr,
                "repository": args.repo,
            },
        )
        print("Review readiness timed out; see receipt")
        return 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        write_json(
            args.output,
            {
                "status": "unavailable",
                "head": args.head,
                "error_type": type(exc).__name__,
            },
        )
        print("Review readiness unavailable; see receipt")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
