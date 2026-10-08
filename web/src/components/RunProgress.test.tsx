import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import RunProgress from "./RunProgress";
import type { Job } from "../api";

describe("actual run activity", () => {
  it("retains repeated rounds and displays facts without inventing stage completion", () => {
    const job: Job = {
      run_id: "run",
      kind: "optimize",
      state: "running",
      stage: "writing_candidates",
      round: { round: 2 },
      stages_seen: ["diagnosing", "writing_candidates"],
      elapsed_ms: 45000,
      remaining_active_ms: 105000,
      cancel_requested: false,
      result: null,
      events: [
        {
          cursor: 1,
          kind: "started",
          summary: "Drafting",
          stage: "writing_candidates",
          elapsed_ms: 10000,
          round: 1,
        },
        {
          cursor: 2,
          kind: "retry",
          summary: "The model service is busy; waiting before another attempt.",
          elapsed_ms: 30000,
          round: 1,
        },
        {
          cursor: 3,
          kind: "started",
          summary: "Drafting",
          stage: "writing_candidates",
          elapsed_ms: 45000,
          round: 2,
        },
      ],
    };
    render(<RunProgress job={job} onCancel={vi.fn()} />);
    expect(
      screen.getAllByText("Writing improved versions — started")
    ).toHaveLength(2);
    expect(screen.getByText(/1:45 active time remaining/)).toBeInTheDocument();
    expect(screen.getByText(/model service is busy/)).toBeInTheDocument();
    expect(
      screen.queryByText("Checking on a stronger model")
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
  });
});

it("reconciles updates by actual draft identity and associates qualified findings", () => {
  const pending = {
    requirement_id: "r1",
    source: "Keep facts.",
    status: "unresolved",
    reason: "Awaiting judgment.",
  };
  const passed = { ...pending, status: "tested", reason: "Facts retained." };
  const checks = (finding: typeof passed) => [
    {
      requirement_id: "r1",
      source: "Keep facts.",
      tested: finding.status === "tested" ? 1 : 0,
      failed: 0,
      untestable: 0,
      unresolved: finding.status === "unresolved" ? 1 : 0,
      reasons: [],
      evidence: [finding],
    },
  ];
  const job: Job = {
    run_id: "run",
    kind: "optimize",
    state: "running",
    stage: "writing_candidates",
    round: { round: 2 },
    stages_seen: [],
    elapsed_ms: 100,
    cancel_requested: false,
    result: null,
    events: [
      {
        cursor: 1,
        kind: "requirement_coverage",
        summary: "Coverage",
        round: 1,
        elapsed_ms: 0,
        requirements: { requirements: [{ id: "r1", source: "Keep facts." }] },
      },
      {
        cursor: 2,
        kind: "check_update",
        summary: "Pending",
        round: 1,
        candidate_id: "c1",
        elapsed_ms: 1,
        checks: checks(pending),
      },
      {
        cursor: 3,
        kind: "checks",
        summary: "Returned",
        round: 1,
        candidate_id: "c1",
        elapsed_ms: 2,
        checks: checks(passed),
      },
      {
        cursor: 4,
        kind: "qualified",
        summary: "Qualified",
        round: 1,
        candidate_id: "c1",
        elapsed_ms: 3,
      },
      {
        cursor: 5,
        kind: "check_update",
        summary: "New pending draft",
        round: 2,
        candidate_id: "c1",
        elapsed_ms: 4,
        checks: checks(pending),
      },
    ],
  };
  render(<RunProgress job={job} onCancel={vi.fn()} />);
  const coverage = screen.getByRole("region", { name: "Requirement coverage" });
  expect(
    within(coverage).getByText(
      /Selected draft: 1 passed check, 0 failed checks, 0 untestable checks, 0 unresolved checks/
    )
  ).toBeInTheDocument();
  expect(within(coverage).getAllByText(/Draft c1 — qualified/)).toHaveLength(1);
  expect(
    within(coverage).getAllByText(/Draft c1 — checks pending/)
  ).toHaveLength(1);
  expect(
    within(coverage).queryByText(/did not qualify/)
  ).not.toBeInTheDocument();
});
