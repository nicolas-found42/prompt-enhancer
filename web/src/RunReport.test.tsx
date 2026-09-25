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

it("explains a calibrated gate and its confidence threshold", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          diagnosis: {
            calibration: {
              "gap:goal": {
                disposition: "gate-above-confidence",
                verdict: "gate-above-confidence",
                reason: "calibration_artifact",
                threshold: 0.8,
                predicate: { confidence_gte: 0.7 },
              },
            },
          },
        },
      }}
    />
  );

  const calibration = screen.getByRole("region", {
    name: "Calibration decisions",
  });
  expect(within(calibration).getByRole("listitem")).toHaveTextContent(
    /Calibration permits gating after an additional confidence check\. Verdict: Supports a calibrated gate with a confidence check\. Minimum event probability: 80%/
  );
});

it("explains why a runtime answer abstained below the calibrated threshold", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          diagnosis: {
            calibration: {
              "gap:goal": {
                disposition: "abstain",
                verdict: "gate",
                reason: "probability_below_calibrated_threshold",
                threshold: 0.8,
                predicate: {},
              },
            },
          },
        },
      }}
    />
  );

  const calibration = screen.getByRole("region", {
    name: "Calibration decisions",
  });
  expect(within(calibration).getByRole("listitem")).toHaveTextContent(
    /Calibration abstained; no calibrated decision was applied\. Verdict: Supports a calibrated gate\. Minimum event probability: 80%\. Reason: The answer's event probability was below the calibrated threshold\./
  );
});

it("explains when the existing policy is used because no artifact matched", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          diagnosis: {
            calibration: {
              inventory: {
                disposition: "legacy",
                reason: "no_calibration_artifact",
              },
            },
          },
        },
      }}
    />
  );

  const calibration = screen.getByRole("region", {
    name: "Calibration decisions",
  });
  expect(within(calibration).getByRole("listitem")).toHaveTextContent(
    /Existing policy remains in use\. Reason: No calibration artifact was available\./
  );
});

it("explains a too-few-examples abstention", () => {
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
              },
            },
          },
        },
      }}
    />
  );

  const calibration = screen.getByRole("region", {
    name: "Calibration decisions",
  });
  expect(within(calibration).getByRole("listitem")).toHaveTextContent(
    /Calibration abstained; no calibrated decision was applied\. Verdict: Too few examples to establish a policy\. Reason: The calibration verdict did not clear the requirements for applying a decision\./
  );
});

it("keeps older reports readable when they contain no calibration evidence", () => {
  render(<RunReport result={baseResult} />);

  expect(screen.getByText("Your prompt already works well.")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Diagnosis" })).toBeVisible();
  expect(screen.getByRole("heading", { name: "Success tests" })).toBeVisible();
  expect(
    screen.queryByRole("heading", { name: "Calibration decisions" })
  ).not.toBeInTheDocument();
});

it("omits calibration decisions for an empty mapping", () => {
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
