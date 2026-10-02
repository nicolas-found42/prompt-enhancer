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
  await page.getByRole("button", { name: "Use my answers" }).click();
  await expect(
    page.getByRole("heading", { name: "Optimized prompt" })
  ).toBeVisible();
  await expect
    .poll(() => historySearches.filter((value) => value === "supplier").length)
    .toBeGreaterThan(searchesBeforeResume);
  await expect(search).toHaveValue("supplier");
});

test("History explains every status pill and names a Failed row's reason", async ({
  page,
}) => {
  const failure = {
    kind: "provider",
    headline: "The model refused the request",
    hint: "Try a different model.",
    message: "provider error",
  };
  const runs = [
    {
      run_id: "failed-run",
      created_at: "2026-09-24T04:00:00Z",
      status: "failed",
      prompt: "Write a useful reply.",
      tier: "standard",
      report: { status: "failed", failure },
      result: {
        status: "failed",
        run_id: "failed-run",
        original_prompt: "Write a useful reply.",
        report: { status: "failed", failure },
        cost: { total: 0 },
        timing: { total_ms: 10 },
      },
    },
    { ...savedRun, run_id: "improved-run" },
  ];
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/runs?*", (route) => route.fulfill({ json: runs }));
  await page.route("**/api/runs/failed-run", (route) =>
    route.fulfill({ json: runs[0] })
  );

  await page.goto("/");

  const history = page.getByRole("list", { name: "Saved optimization runs" });
  const failedRow = history.getByRole("button", {
    name: /Write a useful reply/,
  });
  const unchangedRow = history.getByRole("button", { name: /supplier/ });

  // The Failed row carries its short reason without being opened.
  await expect(failedRow.getByText("Why it failed:")).toBeVisible();
  await expect(failedRow).toContainText("The model refused the request");
  await expect(page.getByRole("heading", { name: "Run details" })).toHaveCount(
    0
  );

  // Pointer: the pill exposes a native tooltip with the worded explanation.
  await expect(failedRow.locator(".badge")).toHaveAttribute(
    "title",
    "The run stopped before finishing. The reason is shown on this row; open it for the full failure card."
  );
  await expect(unchangedRow.locator(".badge")).toHaveAttribute(
    "title",
    "Your prompt was kept as it was; no rewrite changed it."
  );
  // Keyboard: focusing the row describes the pill in words.
  await failedRow.focus();
  await expect(failedRow).toHaveAttribute(
    "aria-describedby",
    "history-status-explanation-failed-run"
  );
  await expect(unchangedRow.locator(".badge")).toHaveText("Unchanged");

  // The explanation surfaces visibly on focus, and never relies on colour alone.
  const shownExplanation = failedRow.locator(
    "xpath=following-sibling::span[@class='status-explanation']"
  );
  await expect(shownExplanation).toHaveText(
    "The run stopped before finishing. The reason is shown on this row; open it for the full failure card."
  );
  // `toBeVisible` ignores clip-path, so assert the clipping is really lifted:
  // a clipped element is 1px wide, the revealed tooltip is much wider.
  const revealed = await shownExplanation.boundingBox();
  expect(revealed?.width ?? 0).toBeGreaterThan(200);
  const revealedStyle = await shownExplanation.evaluate((node) => {
    const style = getComputedStyle(node);
    return { clipPath: style.clipPath, color: style.color };
  });
  expect(revealedStyle.clipPath).toBe("none");

  // Both pills still say which status they are in words.
  await expect(failedRow.locator(".badge")).toHaveText("Failed");

  // Screenshot of the list with the explanations visible, for the visual issue.
  await page
    .getByRole("list", { name: "Saved optimization runs" })
    .screenshot({ path: "test-results/history-status-explanations.png" });

  // Opening the row still works and is unchanged.
  await failedRow.click();
  const details = page.getByRole("article", { name: "Run details" });
  await expect(details).toBeVisible();
  await expect(details.locator(".badge")).toHaveText("Failed");
  await expect(details.locator(".badge")).toHaveAttribute(
    "title",
    /stopped before finishing/
  );
  await expect(details.locator(".badge")).toHaveAttribute(
    "aria-describedby",
    "selected-run-status-explanation"
  );
  await expect(
    details.getByRole("heading", { name: "The model refused the request" })
  ).toBeVisible();
});
