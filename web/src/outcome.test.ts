import { describe, expect, it } from "vitest";
import type { OptimizeResult } from "./api";
import { estimateText, loopDescription, outcomeOf, roughCost } from "./outcome";

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
  it("reports an unverified improvement when nothing changed", () => {
    const outcome = outcomeOf(
      result({
        status: "improvement_not_verified",
        diagnosis: { confirmed_gaps: [] },
      })
    );
    expect(outcome.headline).toBe("No verified improvement this time");
    expect(outcome.reason).toContain("retrying");
  });

  it("reports an unproven improvement without claiming tested quality", () => {
    const outcome = outcomeOf(
      result({
        status: "improved_unverified",
        diagnosis: { confirmed_gaps: [] },
      })
    );
    expect(outcome.headline).toBe("Improved (unverified)");
    expect(outcome.reason).toContain("not tested");
    expect(outcome.reason).not.toMatch(/better|improvement.*measured/i);
  });

  it("names confirmed-details-only results without claiming optimization", () => {
    const outcome = outcomeOf(
      result({
        status: "clarified",
        diagnosis: { confirmed_gaps: [] },
      })
    );
    expect(outcome.headline).toBe("Updated with your details");
    expect(outcome.reason).toContain("confirmed answers");
  });

  it("names the no-qualified-candidate outcome with its attempts", () => {
    const outcome = outcomeOf(
      result({
        status: "no_qualified_candidate",
        diagnosis: { confirmed_gaps: [] },
      })
    );
    expect(outcome.headline).toBe("No rewrite passed its checks");
    expect(outcome.reason).toContain("unchanged");
  });

  it("names the impossible outcome without claiming a violation", () => {
    const outcome = outcomeOf(
      result({
        status: "impossible",
        diagnosis: { confirmed_gaps: [] },
      })
    );
    expect(outcome.headline).toBe(
      "Style and requirements cannot both be satisfied"
    );
    expect(outcome.reason).toContain("no violation was emitted");
  });

  it("still renders saved runs from before the always-attempt loop", () => {
    const outcome = outcomeOf({
      ...result({ status: "unverified" }),
      original_kept: false,
    });
    expect(outcome.headline).toBe("We couldn't test this prompt");
  });

  it("still calls a verified rewrite optimized", () => {
    expect(
      outcomeOf({ ...result({ status: "selected" }), original_kept: false })
        .headline
    ).toBe("Optimized prompt");
    expect(
      outcomeOf({ ...result({ status: "edited" }), original_kept: false })
        .headline
    ).toBe("Updated prompt");
  });

  it("explains why a high-scoring original still lost its tie", () => {
    const outcome = outcomeOf(
      result({
        status: "improvement_not_verified",
        diagnosis: { confirmed_gaps: [] },
        selection_evidence: {
          original_score: { per_model: { first: 0.95, second: 0.82 } },
        },
      })
    );
    expect(outcome.headline).toBe("No verified improvement this time");
  });
});

it("keeps sub-cent costs legible", () => {
  expect(roughCost(0.005)).toBe("under $0.01");
  expect(roughCost(0.026)).toBe("roughly $0.03");
});

describe("estimateText", () => {
  const billedAccount =
    "billed to your own OpenCode Go and OpenRouter accounts";

  it("describes the Deep-only loop with no tier phrasing", () => {
    expect(loopDescription).toContain("Deep work on every run");
    expect(loopDescription).toContain("until the prompt converges");
    expect(loopDescription).not.toMatch(/Fast|Standard|Deep:/);
    expect(loopDescription).not.toMatch(/1\/2\/3 rounds|\d rounds/);
  });

  it("splits the fallback estimate into a time statement and a billed cost statement", () => {
    const { time, cost } = estimateText();
    expect(time).toContain("under a minute");
    expect(time).toContain("30 min");
    // Time and cost are separate strings: neither contains a dollar amount.
    expect(time).not.toMatch(/\$/);
    expect(cost).toContain("$0.03–$0.15");
    expect(cost).toContain(billedAccount);
    expect(cost).toContain("Rough estimate.");
  });

  it("uses the fallback when fewer than 3 runs are recorded", () => {
    const { cost } = estimateText({
      runs: 2,
      minutes: [0.5, 8],
      cost: [0.001, 0.02],
    });
    expect(cost).toContain("Rough estimate");
  });

  it("splits the recorded-range estimate and names the billed account", () => {
    const { time, cost } = estimateText({
      runs: 4,
      minutes: [0.3, 6.6],
      cost: [0.001, 0.012],
    });
    expect(time).toBe("Usually under a minute, up to 7 min.");
    expect(time).not.toMatch(/\$/);
    expect(cost).toBe(
      `About $0.001–$0.012, ${billedAccount} — from your last 4 runs.`
    );
  });

  it("keeps the recorded-range provenance and both bounds", () => {
    const { cost } = estimateText({
      runs: 12,
      minutes: [4, 12],
      cost: [0.002, 0.05],
    });
    expect(cost).toContain("from your last 12 runs.");
    expect(cost).toContain("$0.002");
    expect(cost).toContain("$0.050");
  });
});
