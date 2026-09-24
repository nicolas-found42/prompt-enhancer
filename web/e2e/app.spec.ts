import { expect, test, type Page } from "@playwright/test";

type Json = Record<string, unknown>;

function job(runId: string, state: "running" | "done", result: Json | null = null, extra: Json = {}): Json {
  return {
    run_id: runId, kind: "optimize", state, stage: state === "running" ? "grading" : null,
    round: { round: 1, max_rounds: 2 }, stages_seen: [], elapsed_ms: 65000,
    cancel_requested: false, result, ...extra,
  };
}

/** Serve one mocked run through the job endpoints, finishing on the first poll. */
async function mockRun(page: Page, result: Json) {
  const runId = String(result.run_id);
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) => route.fulfill({ status: 202, json: job(runId, "running") }));
  await page.route(`**/api/jobs/${runId}`, (route) => route.fulfill({ json: job(runId, "done", result) }));
}

const completedBase = {
  status: "completed", original_kept: true,
  cost: { total: 0.002, cost_by_role: {} }, timing: { total_ms: 5 },
};

test("clarification, assumption editing, history, and feedback use the local API", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async () => undefined },
    });
  });

  await page.goto("/");
  await expect(page.getByLabel("Your prompt")).toHaveValue("");
  await page.getByLabel("Your prompt").fill("Help me with this.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(page.getByRole("heading", { name: "A few details will improve the result" })).toBeVisible();
  await expect(page.getByText("What should the assistant do?")).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Optimized prompt" })).toBeVisible();
  await expect(page.locator(".final-prompt")).toContainText("goal: Summarize");

  await page.getByText("View report").click();
  const assumption = page.getByLabel("Goal");
  await expect(assumption).toHaveValue("Summarize");
  await assumption.fill("Analyze");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Updated prompt" })).toBeVisible();
  await expect(page.locator(".final-prompt")).toContainText("goal: Analyze");

  await page.getByRole("button", { name: "Copy prompt" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
  const runs = page.getByRole("list", { name: "Saved optimization runs" }).getByRole("button");
  await expect(runs).toContainText("Help me with this.");
  await expect(runs).toContainText("Improved");
  await page.reload();
  await page.getByRole("list", { name: "Saved optimization runs" }).getByRole("button").click();
  await expect(page.getByRole("heading", { name: "Run details" })).toBeVisible();
  await expect(page.getByText(/goal: Analyze/)).toBeVisible();
  await page.getByRole("button", { name: "Accept result" }).click();
  await expect(page.getByText("Saved feedback: accept")).toBeVisible();
  await page.reload();
  await page.getByRole("list", { name: "Saved optimization runs" }).getByRole("button").click();
  await expect(page.getByText("Saved feedback: accept")).toBeVisible();
  await page.getByRole("button", { name: "Open in result view" }).click();
  await expect(page.locator(".final-prompt")).toContainText("goal: Analyze");
});

test("a retained original with weak evidence says it could not be improved", async ({ page }) => {
  await mockRun(page, {
    ...completedBase, run_id: "no-change-1", original_prompt: "Write a report.", final_prompt: "Write a report.",
    report: {
      status: "no_change",
      summary: "No candidate beat the original.",
      diagnosis: { task_type: "writing", confirmed_gaps: [{ key: "context", label: "relevant context" }], problem_sentences: [] },
      tests: [{ id: "test-1", question: "Does the answer address the report?" }],
      selection_evidence: {
        original_score: { per_model: { "weak-one": 0.75, "weak-two": 0 }, spread: 0.25 },
        winner_score: null,
        rejected_candidates: [{ candidate_id: "c1", strategy: "add_missing_context", rejection_reasons: ["candidate failed fidelity checks"] }],
      },
      strong_check: { original_score: 0.8, candidates: [] },
      offer_deep: { expected_evaluation_multiplier: 5.625 },
    },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a report.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(page.getByRole("heading", { name: "We couldn't safely improve this" })).toBeVisible();
  await expect(page.getByText(/It is missing relevant context/)).toBeVisible();
  await expect(page.getByText(/passed only 0% of checks/)).toBeVisible();
  await expect(page.getByText(/about 5\.6× the work of this run, roughly \$0\.01/)).toBeVisible();
  await page.getByText("View report").click();
  const row = page.getByRole("row", { name: /weak-one/ });
  await expect(row.getByRole("cell")).toHaveText(["weak-one", "75%", "75%"]);
  await expect(page.getByText("Variation between samples: original 25%, selected 25%")).toBeVisible();
  await expect(page.getByText("Add missing context: it changed your meaning or added things you didn't ask for")).toBeVisible();
});

test("a prompt with no gaps and strong results says it already works", async ({ page }) => {
  await mockRun(page, {
    ...completedBase, run_id: "clear-1", original_prompt: "Write two sentences.", final_prompt: "Write two sentences.",
    report: { status: "no_change", diagnosis: { confirmed_gaps: [], problem_sentences: [] }, tests: [] },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write two sentences.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(page.getByRole("heading", { name: "Your prompt already works well" })).toBeVisible();
});

test("a failed run explains the provider problem and offers a fix", async ({ page }) => {
  await mockRun(page, {
    status: "failed", run_id: "failed-1", final_prompt: "Write a reply.", original_kept: true,
    report: {
      status: "failed",
      error: "go request for space-bunny-free failed (HTTP 403): provider request failed",
      failure: {
        kind: "http", provider: "go", model: "space-bunny-free", role: "writer", http_status: 403,
        headline: "OpenCode Go refused the request",
        hint: "Go models such as space-bunny-free need an active OpenCode Go subscription.",
        message: "go request for space-bunny-free failed (HTTP 403): provider request failed",
      },
    },
    cost: { total: 0 }, timing: { total_ms: 1300 },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a reply.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(page.getByRole("heading", { name: "OpenCode Go refused the request" })).toBeVisible();
  await expect(page.getByText("Your prompt was not changed.")).toBeVisible();
  await page.getByText("Technical details").click();
  await expect(page.getByText("failed (HTTP 403)")).toBeVisible();
  await page.getByRole("button", { name: "Change models" }).click();
  await expect(page.getByLabel("Writer", { exact: true })).toBeVisible();
});

test("a running job shows its stage, elapsed time, and can be cancelled", async ({ page }) => {
  let cancelled = false;
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) => route.fulfill({ status: 202, json: job("slow-1", "running") }));
  await page.route("**/api/jobs/slow-1", (route) => route.fulfill({
    json: cancelled
      ? job("slow-1", "done", {
        status: "failed", run_id: "slow-1", final_prompt: "Plan a trip.", original_kept: true,
        report: { status: "cancelled", failure: { kind: "cancelled", headline: "Run cancelled", hint: "You cancelled this run. Your prompt was not changed.", message: "cancelled by the user" } },
        cost: { total: 0 }, timing: { total_ms: 1 },
      })
      : job("slow-1", "running"),
  }));
  await page.route("**/api/jobs/slow-1/cancel", (route) => {
    cancelled = true;
    return route.fulfill({ json: job("slow-1", "running", null, { cancel_requested: true }) });
  });

  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Plan a trip.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(page.getByRole("heading", { name: "Improving your prompt" })).toBeVisible();
  await expect(page.getByText("1:05 elapsed")).toBeVisible();
  await expect(page.getByText(/round 1 of 2/)).toBeVisible();
  await expect(page.locator('[aria-current="step"]')).toHaveText("Scoring the answers");
  await page.reload();
  await expect(page.getByRole("heading", { name: "Improving your prompt" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Your prompt" })).toHaveValue("Plan a trip.");
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("heading", { name: "Run cancelled" })).toBeVisible();
});

test("a refused provider shows a banner that switches to fallback models", async ({ page }) => {
  await page.route("**/api/catalog", (route) => route.fulfill({
    json: {
      judge: { id: "typesafe/jev-1.13", provider: "openrouter" },
      providers: {
        go: [{ id: "go-writer", provider: "go" }, { id: "go-strong", provider: "go" }],
        openrouter: [{ id: "or-writer", provider: "openrouter" }, { id: "or-strong", provider: "openrouter" }, { id: "weak-a", provider: "openrouter" }],
      },
    },
  }));
  await page.route("**/api/settings", (route) => route.fulfill({
    json: { judge_model: "typesafe/jev-1.13", writer_model: "go-writer", strong_check_model: "go-strong", weak_models: ["weak-a"] },
  }));
  await page.route("**/api/providers*", (route) => route.fulfill({
    json: { providers: { go: { status: "unavailable", http_status: 403 } }, fallback: { writer: "or-writer", strong: "or-strong" } },
  }));

  await page.goto("/");
  await expect(page.getByText("OpenCode Go isn't active")).toBeVisible();
  await page.getByRole("button", { name: "Switch to OpenRouter models" }).click();
  await expect(page.getByText("OpenCode Go isn't active")).toBeHidden();
  await page.getByText("Model choices").click();
  await expect(page.getByLabel("Writer", { exact: true })).toHaveValue("or-writer");
  await expect(page.getByLabel("Strong check", { exact: true })).toHaveValue("or-strong");
});
