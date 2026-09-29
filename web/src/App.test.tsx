import userEvent from "@testing-library/user-event";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import {
  getActiveJobs,
  getCatalog,
  getEstimates,
  getJob,
  getProviders,
  getSettings,
  startOptimize,
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
    getActiveJobs: vi.fn(),
    getCatalog: vi.fn(),
    getEstimates: vi.fn(),
    getJob: vi.fn(),
    getProviders: vi.fn(),
    getSettings: vi.fn(),
    startOptimize: vi.fn(),
  };
});

vi.mock("./ModelPicker", () => ({
  default: ({
    selection,
  }: {
    selection: { writer: string; strong: string; weak: string[] };
  }) => (
    <details>
      <summary>Model choices</summary>
      <output aria-label="selected model choices">
        {JSON.stringify(selection)}
      </output>
    </details>
  ),
}));

vi.mock("./History", () => ({ default: () => null }));

const catalog: ModelCatalog = {
  judge: {
    id: "typesafe/jev-1.13",
    provider: "openrouter",
  },
  providers: {
    go: [
      { id: "space-bunny-free", provider: "go" },
      { id: "glm-5.3-flash", provider: "go" },
      { id: "go-weak-model", provider: "go" },
    ],
    openrouter: [
      { id: "router-writer", provider: "openrouter" },
      { id: "router-strong", provider: "openrouter" },
      { id: "meta-llama/llama-3.1-8b-instruct", provider: "openrouter" },
    ],
  },
};

const settings: ModelSettings = {
  judge_model: "typesafe/jev-1.13",
  writer_model: "space-bunny-free",
  strong_check_model: "glm-5.3-flash",
  weak_models: ["meta-llama/llama-3.1-8b-instruct", "go-weak-model"],
};

const providers: ProviderReport = {
  providers: {
    go: { status: "unavailable", http_status: 403, model: "space-bunny-free" },
  },
  fallback: { writer: "router-writer", strong: "router-strong" },
};

beforeEach(() => {
  window.localStorage.clear();
  vi.mocked(getActiveJobs).mockResolvedValue([]);
  vi.mocked(getCatalog).mockResolvedValue(catalog);
  vi.mocked(getEstimates).mockResolvedValue({});
  vi.mocked(getProviders).mockResolvedValue(providers);
  vi.mocked(getSettings).mockResolvedValue(settings);
});

async function warningBanner() {
  const message = await screen.findByText(/Some models selected for this app/);
  const banner = message.closest<HTMLElement>(".banner");
  if (!banner) throw new Error("The provider warning banner was not rendered.");
  return banner;
}

it("explains the unavailable models plainly and keeps diagnostics available", async () => {
  render(<App />);

  const banner = await warningBanner();
  const mainMessage = within(banner).getByText(
    /Some models selected for this app can't be reached/
  );
  expect(mainMessage).toHaveTextContent(
    "Some models selected for this app can't be reached, so your prompt can't run. Use the button to try a different set of models."
  );
  expect(mainMessage).not.toHaveTextContent(
    /OpenCode Go|HTTP|writer|strong check|space-bunny/i
  );

  fireEvent.click(within(banner).getByText("Show troubleshooting details"));
  expect(within(banner).getByText("Provider: OpenCode Go")).toBeVisible();
  expect(within(banner).getByText("Provider response: HTTP 403")).toBeVisible();
  expect(within(banner).getByText(/writer \(space-bunny-free\)/)).toBeVisible();
  expect(
    within(banner).getByText(/strong check \(glm-5.3-flash\)/)
  ).toBeVisible();

  fireEvent.click(
    within(banner).getByRole("button", { name: "Try different models" })
  );
  await waitFor(() =>
    expect(
      screen.queryByText(/Some models selected for this app/)
    ).not.toBeInTheDocument()
  );
  fireEvent.click(screen.getByText("Model choices"));
  expect(screen.getByLabelText("selected model choices")).toHaveTextContent(
    JSON.stringify({
      writer: "router-writer",
      strong: "router-strong",
      weak: ["meta-llama/llama-3.1-8b-instruct"],
    })
  );
});

it("does not offer a fallback action when OpenRouter is known to be unavailable", async () => {
  vi.mocked(getProviders).mockResolvedValue({
    ...providers,
    providers: {
      ...providers.providers,
      openrouter: { status: "unavailable", http_status: 401 },
    },
  });

  render(<App />);

  const banner = await warningBanner();
  expect(
    within(banner).getByText(/Open Model choices below/)
  ).toBeInTheDocument();
  expect(
    within(banner).queryByRole("button", { name: "Try different models" })
  ).not.toBeInTheDocument();
  expect(banner).not.toHaveTextContent(/working models|available models/);
});

it("does not offer a fallback action when no fallback models are configured", async () => {
  vi.mocked(getProviders).mockResolvedValue({
    ...providers,
    fallback: { writer: "", strong: "" },
  });

  render(<App />);

  const banner = await warningBanner();
  expect(
    within(banner).getByText(/Open Model choices below/)
  ).toBeInTheDocument();
  expect(
    within(banner).queryByRole("button", { name: "Try different models" })
  ).not.toBeInTheDocument();
  expect(banner).not.toHaveTextContent(/working models|available models/);
});

const runningJob: Job = {
  run_id: "run-1",
  kind: "optimize",
  prompt: "Write a note to my neighbour.",
  state: "running",
  stage: "diagnosing",
  round: {},
  stages_seen: [],
  elapsed_ms: 0,
  cancel_requested: false,
  result: null,
};

function finishedJob(result: OptimizeResult): Job {
  return { ...runningJob, state: "done", stage: null, result };
}

function trackScrolling() {
  const scrolled: string[] = [];
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vi.fn(function (this: Element) {
      scrolled.push(this.id || this.tagName.toLowerCase());
    }),
  });
  return scrolled;
}

async function startRun() {
  vi.mocked(getProviders).mockResolvedValue({
    providers: { go: { status: "ok" } },
    fallback: providers.fallback,
  });
  const user = userEvent.setup();
  render(<App />);
  await user.type(
    await screen.findByLabelText("Your prompt"),
    "Write a note to my neighbour."
  );
  await user.click(screen.getByRole("button", { name: "Optimize prompt" }));
}

it("brings a new run's progress into view and then its finished result", async () => {
  const scrolled = trackScrolling();
  vi.mocked(startOptimize).mockResolvedValue(runningJob);
  vi.mocked(getJob).mockResolvedValue(
    finishedJob({
      status: "completed",
      run_id: "run-1",
      original_prompt: "Write a note to my neighbour.",
      final_prompt: "Write a note to my neighbour.",
      original_kept: true,
      report: {},
      cost: { total: 0 },
      timing: { total_ms: 1 },
    })
  );

  await startRun();

  await waitFor(() => expect(scrolled).toContain("run-progress"));
  await waitFor(() => expect(scrolled).toContain("run-outcome"), {
    timeout: 3000,
  });
  expect(scrolled.indexOf("run-progress")).toBeLessThan(
    scrolled.indexOf("run-outcome")
  );
  expect(document.activeElement).toHaveAttribute("id", "run-outcome");
});

it("brings a failed run's error card into view", async () => {
  const scrolled = trackScrolling();
  vi.mocked(startOptimize).mockResolvedValue(runningJob);
  vi.mocked(getJob).mockResolvedValue(
    finishedJob({
      status: "failed",
      run_id: "run-1",
      original_prompt: "Write a note to my neighbour.",
      report: {
        failure: {
          kind: "provider_error",
          headline: "The provider refused the request",
          hint: "Try again later.",
          message: "",
        },
      },
      cost: { total: 0 },
      timing: { total_ms: 1 },
    })
  );

  await startRun();

  await waitFor(() => expect(scrolled).toContain("run-outcome"), {
    timeout: 3000,
  });
  expect(
    within(document.getElementById("run-outcome")!).getByRole("heading", {
      name: "The provider refused the request",
    })
  ).toBeVisible();
});

it("does not scroll when the page reattaches to a run after a reload", async () => {
  const scrolled = trackScrolling();
  vi.mocked(getActiveJobs).mockResolvedValue([
    finishedJob({
      status: "completed",
      run_id: "run-1",
      original_prompt: "Write a note to my neighbour.",
      final_prompt: "Write a note to my neighbour.",
      original_kept: true,
      report: {},
      cost: { total: 0 },
      timing: { total_ms: 1 },
    }),
  ]);

  render(<App />);

  await screen.findByText("RESULT");
  expect(scrolled).not.toContain("run-progress");
  expect(scrolled).not.toContain("run-outcome");
});
