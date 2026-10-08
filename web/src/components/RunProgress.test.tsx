import { render, screen } from "@testing-library/react";
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
