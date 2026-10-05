import type { Failure, OptimizeResult } from "./api";
import { STYLE_LABELS, type ImprovementStyle } from "./styles";

export function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function items(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record) : [];
}

/** "add_missing_context" -> "Add missing context". */
export function humanize(value: string): string {
  const spaced = value.replace(/[_-]+/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

const reasonText: Record<string, string> = {
  "candidate failed fidelity checks":
    "it changed your meaning or added things you didn't ask for",
};

export function plainReason(reason: string): string {
  return reasonText[reason] ?? reason;
}

export const stageLabels: Record<string, string> = {
  diagnosing: "Reading your prompt",
  clarifying: "Checking what's missing",
  writing_tests: "Deciding what a good answer looks like",
  choosing_strategy: "Choosing how to improve it",
  writing_candidates: "Writing improved versions",
  running_weak_models: "Trying them on test models",
  grading: "Scoring the answers",
  checking_fidelity: "Checking your meaning is kept",
  strong_check: "Checking on a stronger model",
};

export const stageOrder = Object.keys(stageLabels);

export type Gap = { key: string; label: string };

export function confirmedGaps(result: OptimizeResult): Gap[] {
  return items(record(result.report.diagnosis).confirmed_gaps).map((gap) => ({
    key: String(gap.key ?? ""),
    label: String(gap.label ?? gap.key ?? ""),
  }));
}

const possibleGapText: Record<string, string> = {
  outside_reference: "It may depend on details only you know",
};

/** Near-miss gaps: not strong enough to act on, but worth telling the user about. */
export function possibleGapHints(result: OptimizeResult): string[] {
  return items(record(result.report.diagnosis).possible_gaps).map((gap) => {
    const key = String(gap.key ?? "");
    const lead =
      possibleGapText[key] ?? `It may be missing ${String(gap.label ?? key)}`;
    const sentence =
      typeof gap.sentence === "string" && gap.sentence ? gap.sentence : null;
    return sentence
      ? `${lead}, such as what “${sentence}” refers to. If so, add those details before you use it.`
      : `${lead}. If so, add those details before you use it.`;
  });
}

/** Money for a sub-cent world: "under $0.01" rather than "$0.00". */
export function roughCost(value: number): string {
  return value < 0.01 ? "under $0.01" : `roughly $${value.toFixed(2)}`;
}

export function failureOf(result: OptimizeResult): Failure {
  const failure = record(result.report.failure);
  if (typeof failure.headline === "string")
    return failure as unknown as Failure;
  return {
    kind: "internal",
    headline: "The run stopped before finishing",
    hint: "Try again. The technical details below say what went wrong.",
    message: typeof result.report.error === "string" ? result.report.error : "",
  };
}

export type Pause = {
  reason: string;
  limit: number | null;
  spent_usd: number;
  elapsed_ms: number;
  completed_rounds: number;
};

/** A budget pause awaiting approval, or null when the run is not paused. */
export function pauseOf(result: OptimizeResult): Pause | null {
  if (result.status !== "needs_input") return null;
  if (String(result.report.status ?? "") !== "awaiting_approval") return null;
  const pause = record(result.report.pause);
  const spent = typeof pause.spent_usd === "number" ? pause.spent_usd : NaN;
  if (!Number.isFinite(spent)) return null;
  const limit = typeof pause.limit === "number" ? pause.limit : null;
  const elapsed =
    typeof pause.elapsed_ms === "number" ? Math.max(0, pause.elapsed_ms) : 0;
  const rounds =
    typeof pause.completed_rounds === "number"
      ? Math.max(0, Math.floor(pause.completed_rounds))
      : 0;
  return {
    reason: typeof pause.reason === "string" ? pause.reason : "limit",
    limit,
    spent_usd: spent,
    elapsed_ms: elapsed,
    completed_rounds: rounds,
  };
}

export function pauseText(pause: Pause): string {
  const limit =
    pause.reason === "spend_limit" && pause.limit !== null
      ? `spend limit of $${pause.limit.toFixed(2)}`
      : pause.reason === "time_limit" && pause.limit !== null
        ? `time limit of ${pause.limit}s`
        : "limit";
  const rounds = `${pause.completed_rounds} completed round${pause.completed_rounds === 1 ? "" : "s"}`;
  return `Paused after ${rounds} at your ${limit} ($${pause.spent_usd.toFixed(4)} spent).`;
}

export type CanonicalOutcome =
  | "converged"
  | "improved_tested"
  | "improved_unverified"
  | "impossible"
  | "failed_operational";

export type ControlState = "awaiting_approval" | "stopped" | "cancelled";

export const OUTCOME_LABELS: Record<CanonicalOutcome, string> = {
  converged: "Converged",
  improved_tested: "Improved (tested)",
  improved_unverified: "Improved (unverified)",
  impossible: "Impossible",
  failed_operational: "Failed (operational)",
};

const CONTROL_LABELS: Record<ControlState, string> = {
  awaiting_approval: "Paused for approval",
  stopped: "Stopped",
  cancelled: "Cancelled",
};

export function isCanonicalOutcome(value: unknown): value is CanonicalOutcome {
  return typeof value === "string" && Object.hasOwn(OUTCOME_LABELS, value);
}

export function controlLabel(value: unknown): string | null {
  return typeof value === "string" && Object.hasOwn(CONTROL_LABELS, value)
    ? CONTROL_LABELS[value as ControlState]
    : null;
}

export type Outcome = {
  headline: string;
  reason: string | null;
  appliedStyle: string | null;
  controlState: string | null;
};

/** Render only the outcome and style recorded by the engine. */
export function outcomeOf(result: OptimizeResult): Outcome {
  const value = result.report.outcome;
  if (!isCanonicalOutcome(value)) {
    const editedEvidenceSummary =
      result.report.status === "edited" &&
      !result.report.legacy_metadata &&
      typeof result.report.summary === "string" &&
      result.report.summary.trim()
        ? result.report.summary
        : null;
    return {
      headline: result.report.legacy_metadata
        ? "Legacy run"
        : "Outcome not established",
      reason: editedEvidenceSummary,
      appliedStyle: null,
      controlState: controlLabel(result.report.control_state),
    };
  }
  const style = result.report.applied_style;
  const appliedStyle =
    typeof style === "string" && Object.hasOwn(STYLE_LABELS, style)
      ? STYLE_LABELS[style as ImprovementStyle]
      : null;
  const reason = result.report.outcome_reason;
  return {
    headline: OUTCOME_LABELS[value],
    reason: typeof reason === "string" && reason.trim() ? reason : null,
    appliedStyle,
    controlState: controlLabel(result.report.control_state),
  };
}

export function elapsedText(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
