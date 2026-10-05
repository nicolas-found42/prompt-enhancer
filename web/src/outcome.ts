import type { Failure, OptimizeResult, TierEstimate } from "./api";

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

function originalPassRates(result: OptimizeResult): number[] {
  const selection = record(result.report.selection_evidence);
  const perModel = record(record(selection.original_score).per_model);
  return Object.values(perModel).filter(
    (value): value is number => typeof value === "number"
  );
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

export type Outcome = { headline: string; reason: string | null };

/** Choose the result headline from the evidence rather than from `original_kept` alone. */
export function outcomeOf(result: OptimizeResult): Outcome {
  const status = String(result.report.status ?? "");
  if (status === "edited") return { headline: "Updated prompt", reason: null };
  if (status === "clarified") {
    return {
      headline: "Updated with your details",
      reason:
        "Your confirmed answers were added to the prompt. No rewrite improved on it further.",
    };
  }
  if (status === "improved_unverified") {
    return {
      headline: "Improved (unverified)",
      reason:
        "No reliable way to check the answers was found, so this rewrite is unproven: it passed meaning and safety checks, but what the model answers to it was not tested. The original prompt is kept alongside it.",
    };
  }
  if (status === "no_qualified_candidate") {
    return {
      headline: "No rewrite passed its checks",
      reason:
        "Every rewrite was checked, but none kept your meaning while changing the prompt, so your original prompt is unchanged — retrying gives the models another chance.",
    };
  }
  // Without a verified test the run cannot claim an improvement, even when
  // the user's confirmed clarification answers were appended to the prompt.
  // The `unverified` status is no longer produced; it is kept so saved runs
  // from before the always-attempt loop still render.
  if (status === "unverified") {
    return {
      headline: "We couldn't test this prompt",
      reason: result.original_kept
        ? "No reliable way to check the answers was found, so your prompt is returned as it was. Retrying may find a verified improvement."
        : "No reliable way to check the answers was found, so nothing was tested. The only change is the details you confirmed.",
    };
  }
  if (status === "improvement_not_verified") {
    return {
      headline: "No verified improvement this time",
      reason:
        "Every rewrite was tested, but none passed verification while changing your prompt. Your original prompt is unchanged — retrying gives the models another chance.",
    };
  }
  if (!result.original_kept)
    return { headline: "Optimized prompt", reason: null };
  const gaps = confirmedGaps(result);
  const rates = originalPassRates(result);
  const weakest = rates.length > 0 ? Math.min(...rates) : null;
  const reasons: string[] = [];
  if (gaps.length > 0) {
    reasons.push(
      `It is missing ${gaps.map((gap) => gap.label).join(", ")}. Adding that yourself will likely help more than any rewrite.`
    );
  }
  if (weakest !== null && weakest < 0.8) {
    reasons.push(
      `Your prompt passed only ${Math.round(weakest * 100)}% of checks on the weakest test model, but no rewrite did better without changing your meaning.`
    );
  }
  return {
    headline: "We couldn't safely improve this",
    reason: reasons.join(" "),
  };
}

// Every run uses the Deep workload in a loop until the prompt converges:
// rewrites are written, tried, and retried with varied strategies while any
// quality dimension is below its floor or the gains stay above epsilon.
export const loopDescription =
  "Deep work on every run: rewrites are written, tried, and retried until the prompt converges.";

const fallbackEstimateTime =
  "Usually under a minute for simple prompts, up to 30 min while the loop keeps improving.";

// Every run is billed to the user's own provider keys (README: the private
// .env holds OPENCODE_GO_KEY and OPENROUTER_API_KEY), so the cost line names
// that account rather than leaving the dollars unattributed.
const BILLED_ACCOUNT = "billed to your own OpenCode Go and OpenRouter accounts";

const fallbackEstimateCost = `About $0.03–$0.15, ${BILLED_ACCOUNT}. (Rough estimate.)`;

function durationRange(low: number, high: number): string {
  const top = Math.max(high, low);
  if (top < 1) return "Usually under a minute.";
  if (low < 1) return `Usually under a minute, up to ${Math.round(top)} min.`;
  const [lowText, highText] = [Math.round(low), Math.round(top)];
  return lowText === highText
    ? `About ${lowText} min.`
    : `Usually ${lowText}–${highText} min.`;
}

/**
 * The two cost/time statements under the Improvement style control. Time and
 * cost are always separate sentences, and `cost` always names whose account
 * is billed. Cost accrues while the loop runs, until convergence, cancel, or
 * a budget pause.
 */
export type LoopEstimate = { time: string; cost: string };

export function estimateText(estimate?: TierEstimate): LoopEstimate {
  if (!estimate || estimate.runs < 3)
    return {
      time: fallbackEstimateTime,
      cost: fallbackEstimateCost,
    };
  const [low, high] = estimate.minutes;
  const [lowCost, highCost] = estimate.cost;
  return {
    time: durationRange(low, high),
    cost: `About $${lowCost.toFixed(3)}–$${Math.max(highCost, lowCost).toFixed(3)}, ${BILLED_ACCOUNT} — from your last ${estimate.runs} runs.`,
  };
}

export function elapsedText(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
