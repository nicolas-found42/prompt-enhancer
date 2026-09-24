"""Run ``python -m prompt_enhancer.evaluation`` safely.

Live execution is always explicit. Replay is the default maintainer workflow
and is backed by the product's strict ReplayGateway, which has no live fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import TextIO

from .datasets import DatasetError, load_datasets
from .harness import (
    EngineFactory,
    EvaluationError,
    EvaluationHarness,
    HarnessOptions,
)
from .recording import RecordingGateway


def _json_assignment(raw: str) -> tuple[str, object]:
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("expected KEY=JSON_VALUE")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON value for {key}: {exc}") from exc
    return key, parsed


def _assignments(raw_values: Sequence[str], *, prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for raw in raw_values:
        key, value = _json_assignment(raw)
        if prefix and key in result:
            raise EvaluationError(f"duplicate option {key!r}")
        result[key] = value
    return result


def _factory(reference: str) -> EngineFactory:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise EvaluationError("engine factory must use package.module:callable syntax")
    try:
        factory = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise EvaluationError(f"could not load engine factory {reference!r}: {exc}") from exc
    if not callable(factory):
        raise EvaluationError(f"engine factory {reference!r} is not callable")
    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prompt_enhancer.evaluation",
        description="Evaluate prompt optimization over maintainer datasets.",
    )
    parser.add_argument("datasets", nargs="+", help="one or more JSON dataset files")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--replay",
        type=Path,
        help="strict recorded gateway responses; supports optional case_latency_ms metadata",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="explicitly allow the product's configured live provider gateway",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--record", type=Path, help="capture live responses for strict replay; use with --live")
    parser.add_argument("--allow-snapshot-mismatch", action="store_true", help="replay decisions recorded with a different Jev snapshot")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    parser.add_argument(
        "--engine-factory",
        help="package.module:callable receiving replay path or None",
    )
    parser.add_argument(
        "--tier", choices=("fast", "standard", "deep"), default="standard"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clarification-allowed", action="store_true")
    parser.add_argument("--writer-model")
    parser.add_argument("--strong-check-model")
    parser.add_argument(
        "--weak-model",
        action="append",
        default=[],
        help="repeat to build the per-run weak panel override",
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="repeatable engine setting included in run identity",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="repeatable additional optimize option",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop on the first engine error"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        overrides: dict[str, object] = {}
        if args.writer_model:
            overrides["writer"] = args.writer_model
        if args.strong_check_model:
            overrides["strong"] = args.strong_check_model
        if args.weak_model:
            overrides["weak"] = list(args.weak_model)
        options = HarnessOptions(
            tier=args.tier,
            seed=args.seed,
            clarification_allowed=args.clarification_allowed,
            model_overrides=overrides,
            settings=_assignments(args.setting),
            extra=_assignments(args.option),
            fail_fast=args.fail_fast,
        )
        dataset = load_datasets(args.datasets)
        if args.record and not args.live:
            raise EvaluationError("--record requires --live")
        if args.record and args.engine_factory:
            raise EvaluationError("--record cannot be combined with --engine-factory")
        if args.allow_snapshot_mismatch and not args.replay:
            raise EvaluationError("--allow-snapshot-mismatch requires --replay")
        recording = None
        if args.engine_factory:
            harness = EvaluationHarness(
                engine_factory=_factory(args.engine_factory)
            )
        elif args.live:
            from ..diagnosis import checklist_impacts, checklist_keys
            from ..optimizer import PromptOptimizer

            engine = PromptOptimizer()
            if args.record:
                recording = RecordingGateway(engine.gateway, args.record)
                recording.rubric_thresholds = dict(engine.diagnosis_rubric.gap_thresholds)
                recording.writer_instruction_version = engine.writer_instruction_version
                recording.faithfulness_threshold = engine.faithfulness_threshold
                recording.checklist_keys = list(checklist_keys(engine.diagnosis_rubric))
                recording.checklist_impacts = checklist_impacts(engine.diagnosis_rubric)
                engine.gateway = recording
            harness = EvaluationHarness(engine)
        else:
            # With no custom factory, the harness lazily creates ReplayGateway.
            harness = EvaluationHarness(allow_snapshot_mismatch=args.allow_snapshot_mismatch)
        report = harness.run(dataset, options=options, replay_path=args.replay)
        if args.live and recording is not None:
            recording.attach_case_metrics(report)
        rendered = report.to_json(pretty=args.pretty)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered, file=out)
        failed_cases = [case.case_id for case in report.cases if case.status in {"failed", "error"}]
        if failed_cases:
            print(f"evaluation error: {len(failed_cases)} case(s) failed: {', '.join(failed_cases)}", file=err)
            return 2
        return 0
    except (DatasetError, EvaluationError, OSError) as exc:
        print(f"evaluation error: {exc}", file=err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
