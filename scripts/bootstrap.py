"""Install locked local check dependencies, or verify their availability."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_quality_tools(repo: Path) -> None:
    package = repo / "tools/quality/package.json"
    if (
        package.exists()
        and not (repo / "tools/quality/node_modules/jev-lint/dist/cli.js").is_file()
    ):
        raise ValueError("Pinned quality CLI is missing; run scripts/bootstrap.py")


def check(repo: Path) -> None:
    missing = [
        name
        for name in ("uv", "node", "npm", "actionlint", "gitleaks")
        if not shutil.which(name)
    ]
    if missing:
        raise ValueError("Missing check tools: " + ", ".join(missing))
    for path in ("web/node_modules/.bin/eslint", "web/node_modules/.bin/playwright"):
        if not (repo / path).exists():
            raise ValueError(f"Missing {path}; run scripts/bootstrap.py")
    require_quality_tools(repo)


def install(repo: Path) -> None:
    if shutil.which("brew"):
        missing = [
            name for name in ("actionlint", "gitleaks") if not shutil.which(name)
        ]
        if missing:
            subprocess.run(["brew", "install", *missing], check=True, cwd=repo)
    commands = [
        ["uv", "sync", "--locked"],
        ["npm", "ci", "--prefix", "web"],
        ["npm", "ci", "--prefix", "tools/quality", "--ignore-scripts"],
        ["npm", "--prefix", "web", "exec", "--", "playwright", "install", "chromium"],
        ["uv", "run", "--locked", "pre-commit", "install"],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=repo)
    check(repo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        (check if args.check else install)(args.repo.resolve())
        print("Required local check dependencies are available.")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Bootstrap failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
