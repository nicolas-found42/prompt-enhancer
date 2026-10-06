"""Run configured hooks in fast/test stages and retain every command's output."""

import argparse
import os
import shlex
from pathlib import Path

import yaml
from bootstrap import require_quality_tools
from validation.receipts import Receipt, commit, worktree_digest

ROOT = Path(__file__).resolve().parents[1]
EXPENSIVE = {"web-typecheck-build", "web-unit-tests", "python-tests", "web-tests"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--pre-commit", default="pre-commit")
    parser.add_argument("--phase", choices=["reproduction", "final"], default="final")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for this run"
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = Receipt(
        args.output.resolve(),
        kind="validation",
        phase=args.phase,
        sha=commit(repo, "HEAD"),
    )
    try:
        require_quality_tools(repo)
        config = yaml.safe_load((repo / ".pre-commit-config.yaml").read_text())
        hooks = [hook["id"] for source in config["repos"] for hook in source["hooks"]]
        if not hooks or len(set(hooks)) != len(hooks):
            raise ValueError("expected unique configured hook IDs")
        skipped = [name for name in os.environ.get("SKIP", "").split(",") if name]
        if set(skipped) - {"no-commit-to-branch"}:
            raise ValueError("validation receipts cannot skip deterministic checks")
        receipt.value.update(
            source_digest=worktree_digest(repo),
            configured_hooks=hooks,
            skipped_hooks=skipped,
        )
        receipt.save()
        for stage, selected in [
            ("fast", [name for name in hooks if name not in EXPENSIVE]),
            ("tests", [name for name in hooks if name in EXPENSIVE]),
        ]:
            for name in selected:
                print(f"{stage}: {name}", flush=True)
                env = dict(os.environ)
                if name == "python-tests":
                    report = receipt.output / "pytest.xml"
                    env["PYTEST_ADDOPTS"] = (
                        f"{env.get('PYTEST_ADDOPTS', '')} "
                        + shlex.join(["--durations=10", f"--junitxml={report}"])
                    )
                receipt.run(
                    name,
                    [args.pre_commit, "run", name, "--all-files", "--verbose"],
                    cwd=repo,
                    env=env,
                )
                receipt.value["checks"][-1]["stage"] = stage
                receipt.save()
        if worktree_digest(repo) != receipt.value["source_digest"]:
            raise ValueError(
                "source changed during validation; inspect changes and rerun"
            )
        receipt.finish()
        print(f"Validation completed. Receipt: {receipt.path}")
        return 0
    except KeyboardInterrupt:
        print(f"Validation interrupted. Receipt: {receipt.path}")
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        receipt.fail(exc)
        print(f"Validation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
