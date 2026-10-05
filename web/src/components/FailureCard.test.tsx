import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { OptimizeResult } from "../api";
import FailureCard from "./FailureCard";

it("does not claim the original was kept when a stopped run retains an improvement", () => {
  const result: OptimizeResult = {
    status: "failed",
    run_id: "stopped-improvement",
    final_prompt: "Write a concise note to my neighbour.",
    original_kept: false,
    report: {
      failure: {
        kind: "stopped",
        headline: "Run stopped",
        hint: "What was completed so far is kept below.",
        message: "stopped by the user",
      },
    },
    cost: { total: 0 },
    timing: { total_ms: 0 },
  };

  render(<FailureCard result={result} />);

  expect(screen.getByRole("heading", { name: "Run stopped" })).toBeVisible();
  expect(
    screen.queryByText("Your prompt was not changed.")
  ).not.toBeInTheDocument();
});

it.each([
  { kind: "stopped", originalKept: true, unchanged: true },
  { kind: "internal", originalKept: false, unchanged: false },
  { kind: "internal", originalKept: undefined, unchanged: false },
  { kind: "cancelled", originalKept: true, unchanged: false },
])(
  "reports unchanged evidence honestly for $kind / originalKept=$originalKept",
  ({ kind, originalKept, unchanged }) => {
    render(
      <FailureCard
        result={{
          status: "failed",
          run_id: "terminal-run",
          original_kept: originalKept,
          report: {
            failure: {
              kind,
              headline: "Run ended",
              hint: "Review the result.",
              message: "",
            },
          },
          cost: { total: 0 },
          timing: { total_ms: 0 },
        }}
      />
    );

    expect(screen.getByRole("heading", { name: "Run ended" })).toBeVisible();
    if (unchanged) {
      expect(screen.getByText("Your prompt was not changed.")).toBeVisible();
    } else {
      expect(
        screen.queryByText("Your prompt was not changed.")
      ).not.toBeInTheDocument();
    }
  }
);
