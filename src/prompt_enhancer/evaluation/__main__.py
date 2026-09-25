"""Run ``python -m prompt_enhancer.evaluation`` safely.

Live execution is always explicit. Replay is the default maintainer workflow
and is backed by the product's strict ReplayGateway, which has no live fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO, cast

from ..gateway import GatewayConfig, HttpGateway
from .calibration import (
    CalibrationBudget,
    CalibrationError,
    CalibrationManifest,
    VerdictPolicy,
    calibrate_manifest,
    capture_live_manifest,
    load_calibration_manifest,
)
from .datasets import DatasetError, load_datasets
from .harness import (
    EngineFactory,
    EvaluationError,
    EvaluationHarness,
    HarnessOptions,
)
from .recording import RecordingGateway


def _json_object(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EvaluationError(f"invalid calibration policy JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError("calibration policy must be a JSON object")
    return value


def _verdict_policy(raw: str | None) -> VerdictPolicy:
    if raw is None:
        return VerdictPolicy()
    values = _json_object(raw)
    try:
        return VerdictPolicy(**cast(dict[str, Any], values))
    except TypeError as exc:
        raise EvaluationError(f"invalid calibration policy: {exc}") from exc


def _calibration_manifests(paths: Sequence[str]) -> CalibrationManifest:
    manifests = [load_calibration_manifest(path) for path in paths]
    if len(manifests) == 1:
        return manifests[0]
    combined = CalibrationManifest(
        name=" + ".join(manifest.name for manifest in manifests),
        events=tuple(event for manifest in manifests for event in manifest.events),
        metadata={"inputs": [manifest.to_dict() for manifest in manifests]},
    )
    return CalibrationManifest(
        name=combined.name,
        events=combined.events,
        metadata=combined.metadata,
        input_digest=combined.to_dict()["input_digest"],
    )


def _run_calibration(
    args: argparse.Namespace,
    *,
    out: TextIO,
) -> int:
    if args.record and not args.live:
        raise EvaluationError("--record requires --live")
    if args.allow_snapshot_mismatch:
        raise EvaluationError("--allow-snapshot-mismatch is not a calibration override")
    if args.live and args.budget is None:
        raise EvaluationError("--calibrate --live requires an explicit finite --budget")
    if args.live and args.record is None:
        raise EvaluationError("--calibrate --live requires --record for raw evidence")
    policy = _verdict_policy(args.calibration_policy)
    budget = CalibrationBudget(
        max_source_examples=args.max_source_examples,
        max_repeats=args.runs,
        max_question_evaluations=args.max_question_evaluations,
        budget_usd=args.budget,
    )
    manifest = _calibration_manifests(args.datasets)
    live_report = None
    if args.live:
        gateway = HttpGateway(config=replace(GatewayConfig.from_env(), max_retries=0))
        manifest, live_report = capture_live_manifest(
            manifest,
            gateway,
            budget=budget,
            runs=args.runs,
            request_cost_ceiling=args.request_cost_ceiling,
        )
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    artifact, report = calibrate_manifest(
        manifest,
        verdict_policy=policy,
        fit_mode=args.fit,
        seed=args.seed or 1729,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        budget=budget,
    )
    if live_report is not None:
        report["live_capture"] = live_report
        report["budget"]["live_capture"] = live_report
        artifact = replace(
            artifact,
            metadata={**artifact.metadata, "live_capture": live_report},
        )
        if live_report["status"] == "partial":
            report["status"] = "partial"
            artifact = replace(artifact, status="partial")
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered, file=out)
    artifact_path = args.artifact
    if artifact_path is None and args.output is not None:
        artifact_path = args.output.with_suffix(".artifact.json")
    if artifact_path is not None:
        artifact.save(artifact_path)
    return 0


def _json_assignment(raw: str) -> tuple[str, object]:
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("expected KEY=JSON_VALUE")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid JSON value for {key}: {exc}"
        ) from exc
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
        raise EvaluationError(
            f"could not load engine factory {reference!r}: {exc}"
        ) from exc
    if not callable(factory):
        raise EvaluationError(f"engine factory {reference!r} is not callable")
    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prompt_enhancer.evaluation",
        description="Evaluate prompt optimization or calibrate Jev questions.",
    )
    parser.add_argument(
        "datasets", nargs="+", help="JSON dataset or calibration event files"
    )
    mode = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate per-question Jev events and write a versioned artifact",
    )
    parser.add_argument(
        "--artifact", type=Path, help="write the calibration artifact here"
    )
    parser.add_argument(
        "--calibration-policy",
        help="JSON object or path overriding the versioned verdict policy",
    )
    parser.add_argument(
        "--fit", choices=("none", "temperature"), default="none", help="fit mode"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument(
        "--runs", type=int, default=3, help="maximum independent repeats"
    )
    parser.add_argument("--max-source-examples", type=int, default=100)
    parser.add_argument("--max-question-evaluations", type=int, default=5000)
    parser.add_argument("--budget", type=float, help="finite live budget in USD")
    parser.add_argument(
        "--request-cost-ceiling",
        type=float,
        default=0.01,
        help="reserve this many USD before each live request (default: 0.01)",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument(
        "--record",
        type=Path,
        help="capture live responses for strict replay; use with --live",
    )
    parser.add_argument(
        "--allow-snapshot-mismatch",
        action="store_true",
        help="replay decisions recorded with a different Jev snapshot",
    )
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "calibrate":
        raw_argv = ["--calibrate", *raw_argv[1:]]
    args = parser.parse_args(raw_argv)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        if args.calibrate:
            return _run_calibration(args, out=out)
        if not (args.replay or args.live):
            raise EvaluationError("one of --replay, --live, or --calibrate is required")
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
            harness = EvaluationHarness(engine_factory=_factory(args.engine_factory))
        elif args.live:
            from ..diagnosis import checklist_impacts, checklist_keys
            from ..optimizer import PromptOptimizer

            engine = PromptOptimizer()
            if args.record:
                recording = RecordingGateway(engine.gateway, args.record)
                recording.rubric_thresholds = dict(
                    engine.diagnosis_rubric.gap_thresholds
                )
                recording.writer_instruction_version = engine.writer_instruction_version
                recording.faithfulness_threshold = engine.faithfulness_threshold
                recording.checklist_keys = list(checklist_keys(engine.diagnosis_rubric))
                recording.checklist_impacts = checklist_impacts(engine.diagnosis_rubric)
                engine.gateway = recording
            harness = EvaluationHarness(engine)
        else:
            # With no custom factory, the harness lazily creates ReplayGateway.
            harness = EvaluationHarness(
                allow_snapshot_mismatch=args.allow_snapshot_mismatch
            )
        report = harness.run(dataset, options=options, replay_path=args.replay)
        if args.live and recording is not None:
            recording.attach_case_metrics(report)
        rendered = report.to_json(pretty=args.pretty)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered, file=out)
        failed_cases = [
            case.case_id for case in report.cases if case.status in {"failed", "error"}
        ]
        if failed_cases:
            print(
                f"evaluation error: {len(failed_cases)} case(s) failed: {', '.join(failed_cases)}",
                file=err,
            )
            return 2
        return 0
    except (CalibrationError, DatasetError, EvaluationError, OSError) as exc:
        print(f"evaluation error: {exc}", file=err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
