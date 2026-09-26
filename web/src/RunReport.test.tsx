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

it("shows a detected output and an unresolved screen without treating both as malicious", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          output_screen: [
            {
              candidate_id: "candidate-a",
              model: "weak-a",
              sample: 0,
              status: "steering_detected",
              reason: "evaluator_steering_detected",
            },
            {
              candidate_id: "candidate-b",
              model: "weak-b",
              sample: 0,
              status: "screen_unresolved",
              reason: "hazard_answer_incomplete_or_uncertain",
            },
          ],
        },
      }}
    />
  );

  const section = screen.getByRole("region", { name: "Output screen" });
  expect(
    within(section).getByText(/candidate-a.*evaluator steering detected/i)
  ).toBeInTheDocument();
  expect(
    within(section).getByText(/candidate-b.*screen unresolved/i)
  ).toBeInTheDocument();
});

it("identifies grading records from before the output-screen protocol", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          grading_observation: {
            protocol: "single_output_shared_state_v1",
            gateway_batch_calls: 2,
          },
        },
      }}
    />
  );

  expect(
    screen.getByText(/Output screen unavailable for this historical run/i)
  ).toBeInTheDocument();
});

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

it("shows which option-order grading policy was used and why", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          grading_policy: [
            {
              primitive: "choice",
              test_id: "format",
              question: "Does the answer use the requested format?",
              policy: "single",
              reason: "compatible_order_bias_evidence",
              snapshot: "typesafe/jev-test-snapshot",
            },
          ],
        },
      }}
    />
  );

  const grading = screen.getByRole("region", { name: "Grading policy" });
  expect(within(grading).getByRole("listitem")).toHaveTextContent(
    /Does the answer use the requested format\?: One option order\. A compatible matched experiment supports this policy\. Jev snapshot: typesafe\/jev-test-snapshot\./
  );
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

it("names an unsupported sentence in rejected rewrite details", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          selection_evidence: {
            rejected_candidates: [
              {
                candidate_id: "candidate-1",
                strategy: "specify_output_format",
                rejection_reasons: [
                  "candidate failed fidelity checks",
                  "fidelity rejected 'Respond in French.': new requirement " +
                    "(source mapping: gap 1; probability=0.99)",
                ],
              },
            ],
          },
        },
      }}
    />
  );

  const heading = screen.getByRole("heading", {
    name: "Rewrites that were not used",
  });
  const section = heading.closest("section");
  expect(section).not.toBeNull();
  expect(within(section!).getByRole("listitem")).toHaveTextContent(
    /Respond in French\..*new requirement.*source mapping: gap 1; probability=0\.99/
  );
});

it("shows preservation, uncertain roles, and cost for a structural candidate", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          lossless_restructuring: {
            outcome: "candidate_built",
            selection_outcome: "selected",
            source_preservation: { status: "passed", unit_count: 2 },
            roles: [
              { unit_id: "u0001", role: "task", confidence: 0.95 },
              { unit_id: "u0002", role: "other", confidence: 0.6 },
            ],
            unknowns: ["u0002"],
            role_assignment_requests: 2,
            cost: { cost_by_role: { judge: 0.002 } },
          },
        },
      }}
    />
  );

  const section = screen.getByRole("region", {
    name: "Content preserving structure",
  });
  expect(section).toHaveTextContent(
    "Source preservation: Passed. Outcome: Selected."
  );
  expect(section).toHaveTextContent("u0002: Other (60% confidence)");
  expect(section).toHaveTextContent(
    "Uncertain source units retained in Other: u0002."
  );
  expect(section).toHaveTextContent("Role assignment requests: 2.");
  expect(section).toHaveTextContent("Reported role assignment cost: $0.0020.");
});

it("shows discarded criteria and measured grading request counts", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          test_screening: {
            screening_checks: [
              { test_id: "t0", accepted: true },
              {
                test_id: "t1",
                accepted: false,
                reason: "evaluator_instructions",
              },
            ],
            screening_observation: {
              approved_count: 1,
              discarded_count: 1,
              gateway_batch_calls: 1,
              judge_cost_usd_measured: null,
            },
          },
          grading_observation: {
            gateway_batch_calls: 4,
            graded_output_count: 4,
            serialized_input_bytes_estimate: 5410,
            ungradable_output_count: 0,
            judge_cost_usd_measured: null,
          },
        },
      }}
    />
  );

  const screening = screen.getByRole("region", {
    name: "Success test screening",
  });
  expect(screening).toHaveTextContent("Approved: 1. Discarded: 1.");
  expect(screening).toHaveTextContent("t1: Evaluator instructions");
  expect(screening).toHaveTextContent("Measured judge cost: unavailable");
  const grading = screen.getByRole("region", { name: "Grading requests" });
  expect(grading).toHaveTextContent("4 requests for 4 outputs");
  expect(grading).toHaveTextContent("Estimated serialized input: 5410 bytes");
});
