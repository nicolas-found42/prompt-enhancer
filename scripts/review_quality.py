"""Optional, bounded Jev review. No command calls inference without --live."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from quality_review.evaluation import evaluate, pilot_plan
from quality_review.evidence import GitEvidence, build_plan, gateway_findings
from quality_review.lint import lint_snapshot
from quality_review.review import MODEL, markdown, run_review

from prompt_enhancer.gateway import GatewayConfig, HttpGateway

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plan", "run", "replay", "pilot", "evaluate", "lint"]
    )
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--findings", type=Path)
    parser.add_argument(
        "--path",
        action="append",
        help="Limit lint to this changed repository path; repeatable",
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "tests/fixtures/quality_review/pilot.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".local/quality-review/report.json"
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=50_000)
    args = parser.parse_args()
    if not 1 <= args.max_cases <= 100:
        parser.error("--max-cases must be between 1 and 100")
    if args.live and args.command not in {"run", "pilot", "lint"}:
        parser.error("--live only applies to run, pilot, or lint")
    if args.command in {"replay", "evaluate"} and args.recording is None:
        parser.error("this command requires --recording")
    gateway = None
    if args.live and args.command != "lint" and os.environ.get("OPENROUTER_API_KEY"):
        gateway = HttpGateway(
            config=GatewayConfig(
                openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
                jev_model=MODEL,
                timeout=20,
                max_retries=0,
            )
        )
    if args.command == "evaluate":
        report = evaluate(read(args.recording), read(args.corpus))
    elif args.command == "pilot" or (
        args.command == "replay" and read(args.recording).get("head_sha") == "synthetic"
    ):
        plan = pilot_plan(read(args.corpus))
        if len(plan["cases"]) > args.max_cases:
            parser.error(
                "pilot exceeds --max-cases; raise explicitly or select a smaller corpus"
            )
        report = run_review(
            plan, gateway, replay=read(args.recording) if args.recording else None
        )
        write(args.output.with_suffix(".plan.json"), plan)
    else:
        git = GitEvidence(args.repo, args.base, args.head)
        if args.command == "lint":
            report = lint_snapshot(
                git,
                ROOT,
                live=args.live,
                max_files=args.max_cases,
                max_tokens=args.max_tokens,
                selected_paths=args.path,
            )
        else:
            findings = read(args.findings) if args.findings else gateway_findings(git)
            plan = build_plan(
                git, read(ROOT / "quality/review-rules.json"), findings, args.max_cases
            )
            write(args.output.with_suffix(".plan.json"), plan)
            if args.command == "plan":
                report = {**plan, "status": "planned", "mode": "offline"}
            else:
                report = run_review(
                    plan,
                    gateway,
                    replay=read(args.recording) if args.recording else None,
                )
    write(args.output, report)
    if "results" in report:
        args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    # Warnings never fail. Service/evidence failures remain machine-visible.
    return 2 if report["status"] in {"failed", "partial"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
