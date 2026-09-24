import type { Failure, OptimizeResult, Tier, TierEstimate } from "./api";

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

export type Outcome = { headline: string; reason: string | null };

/** Choose the result headline from the evidence rather than from `original_kept` alone. */
export function outcomeOf(result: OptimizeResult): Outcome {
  const status = String(result.report.status ?? "");
  if (status === "edited") return { headline: "Updated prompt", reason: null };
  if (!result.original_kept)
    return { headline: "Optimized prompt", reason: null };
  if (status === "unverified") {
    return {
      headline: "We couldn't test this prompt",
      reason:
        "No reliable way to check the answers was found, so your prompt is returned as it was.",
    };
  }
  const gaps = confirmedGaps(result);
  const rates = originalPassRates(result);
  const weakest = rates.length > 0 ? Math.min(...rates) : null;
  // No pass rates means the prompt was never run on a test model, so there is
  // no evidence that it works, only that nothing was clearly missing.
  if (gaps.length === 0 && weakest === null) {
    return {
      headline: "We didn't find anything to fix",
      reason:
        "Nothing was clearly missing, so no rewrite was tried. Your prompt wasn't tested on other models.",
    };
  }
  if (gaps.length === 0 && weakest !== null && weakest >= 0.8) {
    return {
      headline: "Your prompt already works well",
      reason: `It passed at least ${Math.round(weakest * 100)}% of checks on every test model, so it is returned unchanged.`,
    };
  }
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

// Runs are bimodal: under a minute when nothing needs fixing, several minutes
// when rewrites are written and tested.
export const fallbackEstimates: Record<Tier, string> = {
  fast: "Under a minute if nothing needs fixing, up to 4 min with rewrites · ~$0.001–$0.01",
  standard:
    "Under a minute if nothing needs fixing, up to 10 min with rewrites · ~$0.005–$0.03",
  deep: "Under a minute if nothing needs fixing, up to 30 min with rewrites · ~$0.03–$0.15",
};

export const tierDescriptions: Record<Tier, string> = {
  fast: "Fast: one round of rewrites, tried on 2 test models.",
  standard: "Standard: up to 2 rounds of rewrites, tried on 3 test models.",
  deep: "Deep: up to 3 rounds with more rewrites, tried on 5 test models.",
};

function durationRange(low: number, high: number): string {
  const top = Math.max(high, low);
  if (top < 1) return "Usually under a minute";
  if (low < 1) return `Usually under a minute, up to ${Math.round(top)} min`;
  const [lowText, highText] = [Math.round(low), Math.round(top)];
  return lowText === highText
    ? `About ${lowText} min`
    : `Usually ${lowText}–${highText} min`;
}

export function estimateText(tier: Tier, estimate?: TierEstimate): string {
  if (!estimate || estimate.runs < 3)
    return `${fallbackEstimates[tier]} (rough estimate)`;
  const [low, high] = estimate.minutes;
  const [lowCost, highCost] = estimate.cost;
  return `${durationRange(low, high)} · ~$${lowCost.toFixed(3)}–$${Math.max(highCost, lowCost).toFixed(3)} (from your last ${estimate.runs} ${tier} runs)`;
}

export function elapsedText(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
