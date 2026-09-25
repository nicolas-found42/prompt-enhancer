import { render, screen, within } from "@testing-library/react";
import { expect, it } from "vitest";
import type { OptimizeResult } from "./api";
import RunReport from "./RunReport";

const baseResult: OptimizeResult = {
  status: "completed",
  run_id: "test-run",
  report: { summary: "Your prompt already works well." },
  cost: { total: 0 },
  timing: { total_ms: 0 },
};

it("shows a plain-language summary of calibration abstentions", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          diagnosis: {
            calibration: {
              inventory: {
                disposition: "abstain",
                verdict: "too-few-examples",
                reason: "calibration_artifact",
                threshold: null,
                predicate: {},
              },
            },
          },
        },
      }}
    />
  );

  expect(
    screen.getByRole("heading", { name: "Calibration decisions" })
  ).toBeVisible();
  expect(
    within(screen.getByRole("list")).getByRole("listitem")
  ).toHaveTextContent(
    /^Question inventory: Abstained from applying calibration\. Verdict: Too few examples\. Reason: The calibration did not support applying a decision\.$/
  );
});

it("keeps the default report when calibration evidence is absent", () => {
  render(<RunReport result={baseResult} />);

  expect(screen.getByText("Your prompt already works well.")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Diagnosis" })).toBeVisible();
  expect(screen.getByRole("heading", { name: "Success tests" })).toBeVisible();
  expect(
    screen.queryByRole("heading", { name: "Calibration decisions" })
  ).not.toBeInTheDocument();
});

it("does not show calibration decisions for an empty mapping", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: { ...baseResult.report, diagnosis: { calibration: {} } },
      }}
    />
  );

  expect(
    screen.queryByRole("heading", { name: "Calibration decisions" })
  ).not.toBeInTheDocument();
});
