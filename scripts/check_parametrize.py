"""Reject unused direct pytest parameters; indirect parameters are setup fixtures."""

import argparse
import ast
from pathlib import Path


def check(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        used = {
            item.id
            for statement in node.body
            for item in ast.walk(statement)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
            ):
                continue
            arg = (
                decorator.args[0]
                if decorator.args
                else next(
                    (kw.value for kw in decorator.keywords if kw.arg == "argnames"),
                    None,
                )
            )
            try:
                names = ast.literal_eval(arg) if arg is not None else None
            except (ValueError, TypeError):
                continue
            if isinstance(names, str):
                names = [name.strip() for name in names.split(",")]
            if not isinstance(names, (list, tuple)):
                continue
            indirect = next(
                (kw.value for kw in decorator.keywords if kw.arg == "indirect"), None
            )
            try:
                fixtures = ast.literal_eval(indirect) if indirect is not None else []
            except (ValueError, TypeError):
                # Dynamic fixture selection cannot be classified statically.
                continue
            if fixtures is True:
                continue
            if fixtures is False:
                fixtures = []
            if not isinstance(fixtures, (list, tuple)):
                continue
            for name in names:
                if isinstance(name, str) and name not in fixtures and name not in used:
                    errors.append(
                        f"{path}:{node.lineno}: unused parametrized value {name!r}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("tests")])
    args = parser.parse_args()
    errors = []
    for path in args.paths:
        files = sorted(path.rglob("test_*.py")) if path.is_dir() else [path]
        for file in files:
            try:
                errors.extend(check(file))
            except (SyntaxError, OSError) as exc:
                errors.append(f"{file}: cannot check: {exc}")
    print("\n".join(errors) if errors else "All direct pytest parameters are used.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
