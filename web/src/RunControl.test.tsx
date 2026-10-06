import userEvent from "@testing-library/user-event";
import { render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  cancelJob,
  getActiveJobs,
  getCatalog,
  getJob,
  getProviders,
  getRunResult,
  getSettings,
  startContinue,
  startOptimize,
  stopRun,
  type Job,
  type ModelCatalog,
  type ModelSettings,
  type OptimizeResult,
  type ProviderReport,
} from "./api";
import App from "./App";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    cancelJob: vi.fn(),
    getActiveJobs: vi.fn(),
    getCatalog: vi.fn(),
    getJob: vi.fn(),
    getProviders: vi.fn(),
    getRunResult: vi.fn(),
    getSettings: vi.fn(),
    startContinue: vi.fn(),
    startOptimize: vi.fn(),
    stopRun: vi.fn(),
  };
});

vi.mock("./ModelPicker", () => ({
  default: () => null,
}));

vi.mock("./History", () => ({ default: () => null }));

const catalog: ModelCatalog = {
  judge: { id: "typesafe/jev-1.13", provider: "openrouter" },
  providers: { go: [], openrouter: [] },
};

const settings: ModelSettings = {
  judge_model: "typesafe/jev-1.13",
  writer_model: "router-writer",
  strong_check_model: "router-strong",
  weak_models: ["weak-a", "weak-b", "weak-c"],
};

const providers: ProviderReport = {
  providers: {},
  fallback: { writer: "router-writer", strong: "router-strong" },
};

const runningJob: Job = {
  run_id: "run-1",
  kind: "optimize",
  prompt: "Write a note to my neighbour.",
  state: "running",
  stage: "writing_candidates",
  round: { round: 2 },
  stages_seen: ["diagnosing"],
  elapsed_ms: 65000,
  cost_total: 0.0621,
  cancel_requested: false,
  result: null,
};

function pausedResult(): OptimizeResult {
  return {
    status: "needs_input",
    run_id: "run-1",
    final_prompt: "Write a note to my neighbour.",
    original_kept: true,
    report: {
      status: "awaiting_approval",
      pause: {
        reason: "spend_limit",
        limit: 0.05,
        spent_usd: 0.0621,
        elapsed_ms: 65000,
        completed_rounds: 1,
      },
      history: [{ round_number: 1, status: "completed" }],
    },
    cost: { total: 0.0621 },
    timing: { total_ms: 65000 },
  };
}

function stoppedResult(): OptimizeResult {
  return {
    status: "failed",
    run_id: "run-1",
    final_prompt: "Write a note to my neighbour.",
    original_kept: true,
    report: {
      status: "stopped",
      failure: {
        kind: "stopped",
        headline: "Run stopped",
        hint: "You stopped this run before it converged.",
        message: "stopped by the user",
      },
      history: [{ round_number: 1, status: "completed" }],
    },
    cost: { total: 0.0621 },
    timing: { total_ms: 65000 },
  };
}

beforeEach(() => {
  window.localStorage.clear();
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vi.fn(),
  });
  vi.mocked(getActiveJobs).mockResolvedValue([]);
  vi.mocked(getCatalog).mockResolvedValue(catalog);
  vi.mocked(getProviders).mockResolvedValue(providers);
  vi.mocked(getSettings).mockResolvedValue(settings);
  vi.mocked(getRunResult).mockResolvedValue({ result: null });
  vi.mocked(getJob).mockResolvedValue(runningJob);
});

describe("run control", () => {
  it("uses cancellation_pending to show whether cancellation is still active", async () => {
    vi.mocked(getActiveJobs).mockResolvedValue([
      { ...runningJob, cancel_requested: true, cancellation_pending: false },
    ]);

    render(<App />);

    expect(await screen.findByRole("button", { name: "Cancel" })).toBeEnabled();
    expect(
      screen.queryByRole("button", { name: "Cancelling…" })
    ).not.toBeInTheDocument();
  });

  it("sends the optional spend limit with the run", async () => {
    vi.mocked(startOptimize).mockResolvedValue(runningJob);
    const user = userEvent.setup();
    render(<App />);

    await user.type(
      await screen.findByLabelText("Your prompt"),
      "Write a note to my neighbour."
    );
    await user.type(
      await screen.findByLabelText("Spend limit (USD, optional)"),
      "0.05"
    );
    await user.click(screen.getByRole("button", { name: "Optimize prompt" }));

    await waitFor(() =>
      expect(startOptimize).toHaveBeenCalledWith(
        "Write a note to my neighbour.",
        "auto",
        {
          writer: "router-writer",
          strong: "router-strong",
          weak: ["weak-a", "weak-b", "weak-c"],
        },
        { spend_limit_usd: 0.05 }
      )
    );
  });

  it("shows cancel, elapsed time, and spent cost while a run is in progress", async () => {
    vi.mocked(startOptimize).mockResolvedValue(runningJob);
    vi.mocked(cancelJob).mockResolvedValue({
      ...runningJob,
      cancel_requested: true,
    });
    const user = userEvent.setup();
    render(<App />);

    await user.type(
      await screen.findByLabelText("Your prompt"),
      "Write a note to my neighbour."
    );
    await user.click(screen.getByRole("button", { name: "Optimize prompt" }));

    const progress = await screen.findByText("Improving your prompt");
    const section = progress.closest("section")!;
    expect(
      within(section).getByRole("button", { name: "Cancel" })
    ).toBeInTheDocument();
    expect(section).toHaveTextContent("1:05 elapsed");
    expect(section).toHaveTextContent("$0.0621 spent");

    await user.click(within(section).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(cancelJob).toHaveBeenCalledWith("run-1"));
  });

  it("offers Continue and Stop with the spent amount on a paused run", async () => {
    vi.mocked(startOptimize).mockResolvedValue(runningJob);
    vi.mocked(getJob).mockResolvedValue({
      ...runningJob,
      state: "done",
      stage: null,
      result: pausedResult(),
    });
    vi.mocked(startContinue).mockResolvedValue({
      ...runningJob,
      kind: "continue",
    });
    const user = userEvent.setup();
    render(<App />);

    await user.type(
      await screen.findByLabelText("Your prompt"),
      "Write a note to my neighbour."
    );
    await user.click(screen.getByRole("button", { name: "Optimize prompt" }));

    const panel = await screen.findByText(
      "Paused at your limit",
      {},
      { timeout: 3000 }
    );
    const section = panel.closest("section")!;
    expect(section).toHaveTextContent("$0.0621 spent");
    expect(
      within(section).getByRole("button", { name: "Continue" })
    ).toBeInTheDocument();
    expect(
      within(section).getByRole("button", { name: "Stop" })
    ).toBeInTheDocument();

    await user.click(within(section).getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(startContinue).toHaveBeenCalledWith("run-1"));
  });

  it("stops a paused run permanently", async () => {
    vi.mocked(startOptimize).mockResolvedValue(runningJob);
    vi.mocked(getJob).mockResolvedValue({
      ...runningJob,
      state: "done",
      stage: null,
      result: pausedResult(),
    });
    vi.mocked(stopRun).mockResolvedValue(stoppedResult());
    const user = userEvent.setup();
    render(<App />);

    await user.type(
      await screen.findByLabelText("Your prompt"),
      "Write a note to my neighbour."
    );
    await user.click(screen.getByRole("button", { name: "Optimize prompt" }));

    const panel = await screen.findByText(
      "Paused at your limit",
      {},
      { timeout: 3000 }
    );
    const section = panel.closest("section")!;
    await user.click(within(section).getByRole("button", { name: "Stop" }));

    await waitFor(() => expect(stopRun).toHaveBeenCalledWith("run-1"));
    expect(await screen.findByText("Run stopped")).toBeVisible();
  });
});
