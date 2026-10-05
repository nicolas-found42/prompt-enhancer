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
  it.each([
    ["converged", "Converged"],
    ["improved_tested", "Improved (tested)"],
    ["improved_unverified", "Improved (unverified)"],
    ["impossible", "Impossible"],
    ["failed_operational", "Failed (operational)"],
  ])(
    "renders the canonical %s outcome and evidence reason",
    (status, label) => {
      const outcome = outcomeOf(
        result({
          outcome: status,
          outcome_reason:
            "The accepted prompt met the recorded evidence checks.",
          applied_style: "clearer",
        })
      );
      expect(outcome.headline).toBe(label);
      expect(outcome.reason).toBe(
        "The accepted prompt met the recorded evidence checks."
      );
      expect(outcome.appliedStyle).toBe("Clearer");
    }
  );

  it("keeps a run-control state separate from its canonical outcome", () => {
    const outcome = outcomeOf(
      result({
        outcome: "improved_tested",
        outcome_reason: "Accepted prompt passed the tests.",
        applied_style: "shorter",
        control_state: "stopped",
      })
    );
    expect(outcome.headline).toBe("Improved (tested)");
    expect(outcome.controlState).toBe("Stopped");
  });

  it("does not infer a canonical outcome or style for a legacy row", () => {
    const outcome = outcomeOf(
      result({
        status: "legacy_status",
        legacy_metadata: { source: "old_schema" },
      })
    );
    expect(outcome.headline).toBe("Legacy run");
    expect(outcome.appliedStyle).toBeNull();
  });

  it("shows when an assumption edit invalidates the prior quality outcome", () => {
    const outcome = outcomeOf(
      result({
        status: "edited",
        summary:
          "Assumption corrected; performance evidence is from the original optimization.",
      })
    );
    expect(outcome.headline).toBe("Outcome not established");
    expect(outcome.reason).toBe(
      "Assumption corrected; performance evidence is from the original optimization."
    );
  });

  it("does not treat inherited object keys as outcomes, controls, or styles", () => {
    const outcome = outcomeOf(
      result({
        outcome: "constructor",
        control_state: "toString",
        applied_style: "__proto__",
      })
    );
    expect(outcome.headline).toBe("Outcome not established");
    expect(outcome.controlState).toBeNull();
    expect(outcome.appliedStyle).toBeNull();
  });
});

it("keeps sub-cent costs legible", () => {
  expect(roughCost(0.005)).toBe("under $0.01");
  expect(roughCost(0.026)).toBe("roughly $0.03");
});
