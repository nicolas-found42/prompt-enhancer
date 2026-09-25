import { expect, test, type Page } from "@playwright/test";

type Json = Record<string, unknown>;

const runId = "history-search-refresh";
const prompt = "Write a reply to this supplier.";
const savedRun = {
  run_id: "saved-supplier-run",
  created_at: "2026-09-24T04:00:00Z",
  status: "completed",
  prompt,
  final_prompt: prompt,
  original_kept: true,
  tier: "standard",
};

function job(
  state: "running" | "done",
  result: Json | null = null,
  kind: "optimize" | "resume" = "optimize"
): Json {
  return {
    run_id: runId,
    kind,
    state,
    stage: state === "running" ? "grading" : null,
    round: { round: 1, max_rounds: 2 },
    stages_seen: [],
    elapsed_ms: 100,
    cancel_requested: false,
    result,
  };
}

async function mockHistoryAndJobs(page: Page) {
  const historySearches: (string | null)[] = [];
  let resumed = false;
  const needsInput = {
    status: "needs_input",
    run_id: runId,
    original_prompt: prompt,
    report: {},
    questions: [
      {
        id: "goal",
        prompt: "What should the assistant do?",
        options: [{ value: "summarize", label: "Summarize" }],
      },
    ],
    cost: { total: 0 },
    timing: { total_ms: 1 },
  };
  const completed = {
    status: "completed",
    run_id: runId,
    original_prompt: prompt,
    final_prompt: `${prompt} by Friday.`,
    original_kept: false,
    report: { status: "optimized", summary: "The prompt now has a deadline." },
    cost: { total: 0 },
    timing: { total_ms: 1 },
  };

  await page.route("**/api/runs?*", (route) => {
    const url = new URL(route.request().url());
    const search = url.searchParams.get("search");
    historySearches.push(search);
    return route.fulfill({
      json: search?.startsWith("banana") ? [] : [savedRun],
    });
  });
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) =>
    route.fulfill({ status: 202, json: job("running") })
  );
  await page.route(`**/api/jobs/${runId}/resume`, (route) => {
    resumed = true;
    return route.fulfill({
      status: 202,
      json: job("running", null, "resume"),
    });
  });
  await page.route(`**/api/jobs/${runId}`, (route) =>
    route.fulfill({ json: job("done", resumed ? completed : needsInput) })
  );

  return historySearches;
}

test("History distinguishes no matches and keeps its applied search through run refreshes", async ({
  page,
}) => {
  const historySearches = await mockHistoryAndJobs(page);
  await page.goto("/");

  await expect(
    page
      .getByRole("list", { name: "Saved optimization runs" })
      .getByText(prompt)
  ).toBeVisible();

  const search = page.getByRole("searchbox", {
    name: "Search prompts and metadata",
  });
  await search.fill("banana-not-a-supplier");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByText("No runs match this search.")).toBeVisible();
  await page.getByRole("button", { name: "Clear search" }).click();
  await expect(
    page
      .getByRole("list", { name: "Saved optimization runs" })
      .getByText(prompt)
  ).toBeVisible();

  await search.fill("supplier");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(
    page
      .getByRole("list", { name: "Saved optimization runs" })
      .getByText(prompt)
  ).toBeVisible();

  const searchesBeforeRun = historySearches.filter(
    (value) => value === "supplier"
  ).length;
  await page.getByLabel("Your prompt").fill(prompt);
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(
    page.getByRole("heading", { name: "A few details will improve the result" })
  ).toBeVisible();
  await expect
    .poll(() => historySearches.filter((value) => value === "supplier").length)
    .toBeGreaterThan(searchesBeforeRun);

  const searchesBeforeResume = historySearches.filter(
    (value) => value === "supplier"
  ).length;
  await page.getByRole("radio", { name: "Summarize" }).check();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Optimized prompt" })
  ).toBeVisible();
  await expect
    .poll(() => historySearches.filter((value) => value === "supplier").length)
    .toBeGreaterThan(searchesBeforeResume);
  await expect(search).toHaveValue("supplier");
});
