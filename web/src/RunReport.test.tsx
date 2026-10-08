import { render, screen, within } from "@testing-library/react";
import { expect, it } from "vitest";
import type { CapabilitySummary, OptimizeResult } from "./api";
import RunReport from "./RunReport";

const baseResult: OptimizeResult = {
  status: "completed",
  run_id: "test-run",
  report: { summary: "Your prompt already works well." },
  cost: { total: 0 },
  timing: { total_ms: 0 },
};

it("explains requirement sources, partial coverage and the selected draft's own checks", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          requirements: {
            coverage: "partial",
            reason: "Other obligations still need coverage.",
            requirements: [
              {
                id: "r1",
                source: "Write exactly two words.",
                scope: "whole_output",
              },
            ],
            contradictions: [],
          },
          selection_evidence: {
            selected_candidate: {
              metadata: {
                requirement_findings: [
                  {
                    requirement_id: "r1",
                    status: "tested",
                    reason: "Exactly two words are required.",
                  },
                  {
                    requirement_id: "r1",
                    status: "untestable",
                    reason: "Word boundaries are ambiguous.",
                  },
                ],
              },
            },
          },
        },
      }}
    />
  );

  const section = screen.getByRole("region", { name: "Requirement coverage" });
  expect(within(section).getByText("Write exactly two words.")).toBeVisible();
  expect(within(section).getByText(/Coverage is partial/)).toBeVisible();
  expect(
    within(section).getByText(
      /1 passed check, 0 failed checks, 1 untestable check/
    )
  ).toBeVisible();
  expect(
    within(section).getByText("Word boundaries are ambiguous.")
  ).toBeVisible();
});

it("shows per-round source-backed failure hypotheses and uncertain pairs", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          history: [
            {
              evidence: {
                failure_attribution: {
                  attributed_count: 2,
                  unresolved_count: 1,
                  skipped_count: 0,
                  pairs: [
                    {
                      pair_id: "0:0",
                      candidate_id: "candidate-a",
                      model: "weak-a",
                      sample: 0,
                      test_id: "t0",
                      status: "supported",
                      sentence_id: "s0002",
                      sentence_text: "Keep it concise.",
                      kind: "ignored_constraint",
                    },
                    {
                      pair_id: "1:0",
                      candidate_id: "candidate-b",
                      model: "weak-b",
                      sample: 1,
                      test_id: "t0",
                      status: "supported",
                      sentence_id: "s0001",
                      sentence_text: "Summarize the report.",
                      kind: "misread_instruction",
                    },
                    {
                      pair_id: "2:0",
                      candidate_id: "candidate-b",
                      model: "weak-c",
                      sample: 0,
                      test_id: "t1",
                      status: "unresolved",
                      reason: "low_confidence_or_unsupported_source",
                    },
                  ],
                },
              },
            },
          ],
        },
      }}
    />
  );

  const section = screen.getByRole("region", { name: "Failure attribution" });
  expect(
    within(section).getByText(/2 supported hypotheses, 1 unresolved pair/)
  ).toBeInTheDocument();
  expect(
    within(section).getByText(
      /candidate-a.*s0002.*Keep it concise.*Ignored constraint/
    )
  ).toBeInTheDocument();
  expect(
    within(section).getByText(
      /candidate-b.*s0001.*Summarize the report.*Misread instruction/
    )
  ).toBeInTheDocument();
  expect(
    within(section).getByText(/weak-c.*Uncertain or unsupported source/)
  ).toBeInTheDocument();
});

it("renders historical reports without attribution", () => {
  render(<RunReport result={baseResult} />);
  expect(
    screen.queryByRole("region", { name: "Failure attribution" })
  ).not.toBeInTheDocument();
});

it("shows candidate evaluation, acceptance reasons, and usable versus raw judgments", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          judgment_provenance: [
            {
              capability: "compare",
              stage: "evaluate",
              candidate_id: "candidate-a",
              round_number: 3,
              source_round: 2,
              question_key: "compare:structure_added",
              model: "typesafe/jev-test",
              raw_answer: { choice: "added" },
              usable: true,
            },
            {
              capability: "compare",
              stage: "evaluate",
              candidate_id: "candidate-a",
              round_number: 3,
              source_round: 2,
              question_key: "compare:verbosity_direction",
              model: "typesafe/jev-test",
              raw_answer: { choice: "shorter" },
              usable: true,
            },
          ],
          evaluation_evidence: {
            candidates: {
              "candidate-a": {
                candidate_id: "candidate-a",
                round_number: 3,
                comparison: {
                  task_preserved: {
                    raw_answer: { probability_true: 0.94 },
                    usable: true,
                    probability: 0.94,
                  },
                  no_invented_detail: {
                    raw_answer: { error: "missing probability" },
                    usable: false,
                  },
                  structure_added: {
                    raw_answer: { choice: "added" },
                    usable: true,
                    selected: "added",
                  },
                  verbosity_direction: {
                    raw_answer: { choice: "shorter" },
                    usable: true,
                    selected: "shorter",
                  },
                },
                verification: {
                  success_test: {
                    raw_answer: { probability_true: 0.88 },
                    usable: true,
                    probability: 0.88,
                  },
                },
                audit: {
                  safety: {
                    raw_answer: { probability_true: 0.99 },
                    usable: true,
                    probability: 0.99,
                  },
                },
                rerank: {
                  raw_answer: { probability: 0.7 },
                  usable: true,
                  probability: 0.7,
                },
                review: {
                  raw_answer: { decision: "retain" },
                  usable: true,
                },
                score_vector: { scores: { clarity: 0.9 }, passed: true },
                fidelity: { passed: true },
                strong_check: { passed: true },
                downstream_verification: "verified",
                success_tests: [{ id: "summary" }],
                success_test_outputs: [{ test_id: "summary", output: "..." }],
                success_test_grade: { worst: 0.8 },
                accept: {
                  raw_answer: { probability_true: 0.92 },
                  usable: true,
                  probability: 0.92,
                  threshold: 0.8,
                  accepted: true,
                },
                eligible: true,
                rejection_reasons: [],
              },
              "candidate-b": {
                candidate_id: "candidate-b",
                round_number: 4,
                comparison: {},
                verification: {},
                audit: {},
                rerank: { raw_answer: null, usable: false },
                review: { raw_answer: null, usable: false },
                score_vector: null,
                fidelity: null,
                strong_check: null,
                downstream_verification: "unverified",
                success_tests: [],
                success_test_outputs: [],
                success_test_grade: null,
                accept: {
                  raw_answer: { probability_true: 0.2 },
                  usable: true,
                  probability: 0.2,
                  threshold: 0.8,
                  accepted: false,
                },
                eligible: false,
                rejection_reasons: ["safety floor breached"],
              },
            },
          },
        },
      }}
    />
  );

  const section = screen.getByRole("region", {
    name: "Evaluation and acceptance",
  });
  const acceptedCandidate = within(section)
    .getByText("candidate-a")
    .closest("li");
  expect(acceptedCandidate).toHaveTextContent(/round 3.*Eligible/i);
  expect(
    within(section).getByText(/Accepted by the acceptance check/i)
  ).toBeInTheDocument();
  const comparisonItems = within(section)
    .getAllByRole("listitem")
    .filter((item) =>
      [
        "Task preserved",
        "No invented detail",
        "Structure added",
        "Verbosity direction",
      ].some((label) => item.textContent?.trim().startsWith(`${label}:`))
    );
  expect(
    comparisonItems.find((item) => item.textContent?.includes("Task preserved"))
  ).toHaveTextContent(/usable.*94%/i);
  expect(
    comparisonItems.find((item) =>
      item.textContent?.includes("No invented detail")
    )
  ).toHaveTextContent(/not usable/i);
  expect(
    comparisonItems.find((item) =>
      item.textContent?.startsWith("Structure added:")
    )
  ).toHaveTextContent(/^Structure added: Usable.*added$/i);
  expect(
    comparisonItems.find((item) =>
      item.textContent?.startsWith("Verbosity direction:")
    )
  ).toHaveTextContent(/^Verbosity direction: Usable.*shorter$/i);
  expect(within(section).getByText(/success test.*88%/i)).toBeInTheDocument();
  expect(within(section).getByText(/safety.*99%/i)).toBeInTheDocument();
  expect(
    within(section).getByText(/safety floor breached/i)
  ).toBeInTheDocument();
  expect(
    within(section).getByText(/Rejected by the acceptance check/i)
  ).toBeInTheDocument();
  expect(within(section).getAllByText(/Not usable/i).length).toBeGreaterThan(0);
  expect(
    within(section).getByText(/Judgment provenance \(2\)/i)
  ).toBeInTheDocument();
});

it("shows all twelve capability counts and provenance across current and source rounds", () => {
  const capabilityNames = [
    "verify",
    "screen",
    "noul",
    "find",
    "rerank",
    "classify",
    "decide",
    "compare",
    "extract",
    "audit",
    "review",
    "gate",
  ];
  const summaryEntry = (
    count: number,
    ran: boolean,
    stages: Record<string, number>
  ) => ({
    count,
    ran,
    stages,
  });
  const emptySummary = summaryEntry(0, false, {});
  const capabilities: CapabilitySummary = {
    verify: emptySummary,
    screen: emptySummary,
    noul: emptySummary,
    find: emptySummary,
    rerank: emptySummary,
    classify: emptySummary,
    decide: emptySummary,
    compare: emptySummary,
    extract: emptySummary,
    audit: emptySummary,
    review: emptySummary,
    gate: emptySummary,
  };
  capabilities.verify = summaryEntry(2, true, { evaluate: 1, verification: 1 });
  capabilities.gate = summaryEntry(1, true, { accept: 1 });
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          capabilities_fired: capabilities,
          judgment_provenance: [
            {
              capability: "compare",
              stage: "evaluate",
              candidate_id: "candidate-a",
              round_number: 4,
              source_round: 2,
              question_key: "compare:task_preserved",
              model: "typesafe/jev-test",
              raw_answer: { probability_true: 0.94 },
              usable: true,
              probability: 0.94,
            },
            {
              capability: "audit",
              stage: "verification",
              candidate_id: "candidate-b",
              round_number: 4,
              source_round: null,
              question_key: "audit:safety",
              model: null,
              raw_answer: { malformed: true },
              usable: false,
            },
            {
              capability: "gate",
              stage: "accept",
              candidate_id: "candidate-b",
              round_number: 4,
              source_round: null,
              question_key: "accept:candidate",
              model: "typesafe/jev-test",
              raw_answer: { choice: "reject" },
              usable: true,
            },
          ],
        },
      }}
    />
  );

  const section = screen.getByRole("region", {
    name: "Evaluation and acceptance",
  });
  for (const capability of capabilityNames) {
    expect(
      within(section).getByText(
        new RegExp(
          `${capability}.*${capability === "verify" ? "2" : capability === "gate" ? "1" : "0"} times`,
          "i"
        )
      )
    ).toBeInTheDocument();
  }
  expect(
    within(section).getByText(/stages: Evaluate 1, Verification 1/i)
  ).toBeInTheDocument();
  expect(
    within(section).getByText(/gate: 1 times.*ran.*stages: Accept 1/i)
  ).toBeInTheDocument();
  expect(
    within(section).getByText(/candidate-a.*current round 4.*source round 2/i)
  ).toBeInTheDocument();
  expect(
    within(section).getAllByText(
      /candidate-b.*current round 4.*source round not recorded/i
    )
  ).toHaveLength(2);
  expect(within(section).getByText(/model not recorded/i)).toBeInTheDocument();
  expect(within(section).getByText(/not usable/i)).toBeInTheDocument();
  expect(within(section).getAllByText("Raw answer")).toHaveLength(3);
});

it("keeps the evaluation section absent for historical reports without provenance", () => {
  render(<RunReport result={baseResult} />);
  expect(
    screen.queryByRole("region", { name: "Evaluation and acceptance" })
  ).not.toBeInTheDocument();
});

it("shows incomplete bounded diagnosis without claiming the prompt has no gaps", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          diagnosis: {
            task_type: "general",
            request_evidence: {
              complete: false,
              mode: "bounded_sequential_fallback",
              provider_requests: 8,
            },
          },
        },
      }}
    />
  );

  expect(screen.getByText(/Diagnosis requests: 8/)).toBeInTheDocument();
  expect(
    screen.getByText(/Diagnosis evidence is incomplete/)
  ).toBeInTheDocument();
  expect(
    screen.queryByText("No confirmed missing pieces.")
  ).not.toBeInTheDocument();
});

it("explains unresolved weak-grade confirmation and cascade spending", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          grading_cascade: {
            confirmation_count: 1,
            escalation_count: 0,
            verification_count: 0,
            unresolved_count: 1,
            reserved_cost_usd: 0.001,
            dollar_cap: 0.02,
            pairs: [
              {
                pair_id: "0001:0000",
                candidate_id: "candidate-a",
                test_id: "t0",
                reason: "confirmation_not_decisive",
                status: "unresolved",
              },
            ],
          },
        },
      }}
    />
  );

  const section = screen.getByRole("region", { name: "Grade confirmation" });
  expect(within(section).getByText(/1 confirmation/i)).toBeInTheDocument();
  expect(
    within(section).getByText(/candidate-a.*confirmation not decisive/i)
  ).toBeInTheDocument();
  expect(within(section).getByText(/\$0\.0010.*\$0\.0200/)).toBeInTheDocument();
});

it("distinguishes missing cascade cost and cap from measured zero", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          grading_cascade: { confirmation_count: 0 },
        },
      }}
    />
  );

  const section = screen.getByRole("region", { name: "Grade confirmation" });
  expect(section).toHaveTextContent(
    "Reserved cascade cost: unavailable of unavailable."
  );
});

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
            cost: { status: "reported", cost_by_role: { judge: 0.002 } },
          },
        },
      }}
    />
  );

  const section = screen.getByRole("region", {
    name: "Lossless restructuring",
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

it("does not present an unavailable role assignment cost as zero", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          lossless_restructuring: {
            outcome: "candidate_built",
            role_assignment_requests: 1,
            cost: { status: "unavailable", cost_by_role: {} },
          },
        },
      }}
    />
  );

  const section = screen.getByRole("region", {
    name: "Lossless restructuring",
  });
  expect(section).toHaveTextContent(
    "Reported role assignment cost: unavailable."
  );
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

it("shows the applied style, marking Auto inference", () => {
  render(
    <RunReport
      result={{
        ...baseResult,
        report: {
          ...baseResult.report,
          improvement_style: "auto",
          applied_style: "shorter",
        },
      }}
    />
  );

  expect(document.body.textContent).toContain("Applied style: Shorter");
  expect(document.body.textContent).toContain("your style was Auto");
});
