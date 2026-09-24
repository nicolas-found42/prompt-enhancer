import { describe, expect, it } from "vitest";
import type { OptimizeResult } from "./api";
import { outcomeOf, roughCost } from "./outcome";

function result(report: Record<string, unknown>): OptimizeResult {
  return {
    status: "completed",
    run_id: "test-run",
    original_kept: true,
    report,
    cost: { total: 0 },
    timing: { total_ms: 0 },
  };
}

describe("outcomeOf", () => {
  it("does not call an untested prompt good", () => {
    const outcome = outcomeOf(result({ diagnosis: { confirmed_gaps: [] } }));
    expect(outcome.headline).toBe("We didn't find anything to fix");
    expect(outcome.reason).toContain("wasn't tested");
  });

  it("uses the weakest model when describing evidence", () => {
    const outcome = outcomeOf(
      result({
        diagnosis: { confirmed_gaps: [] },
        selection_evidence: {
          original_score: { per_model: { first: 0.95, second: 0.82 } },
        },
      })
    );
    expect(outcome.headline).toBe("Your prompt already works well");
    expect(outcome.reason).toContain("82% of checks on every test model");
  });
});

it("keeps sub-cent costs legible", () => {
  expect(roughCost(0.005)).toBe("under $0.01");
  expect(roughCost(0.026)).toBe("roughly $0.03");
});
