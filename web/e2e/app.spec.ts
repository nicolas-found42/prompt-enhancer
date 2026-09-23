import { expect, test } from "@playwright/test";

test("clarification, assumption editing, history, and feedback use the local API", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async () => undefined },
    });
  });

  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Help me with this.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(page.getByRole("heading", { name: "A few details will improve the result" })).toBeVisible();
  await expect(page.getByText("What should the assistant do?")).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Optimized prompt" })).toBeVisible();
  await expect(page.locator(".final-prompt")).toContainText("goal: Summarize");

  await page.getByText("View report").click();
  const assumption = page.getByLabel("goal");
  await expect(assumption).toHaveValue("Summarize");
  await assumption.fill("Analyze");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Optimized prompt" })).toBeVisible();
  await expect(page.locator(".final-prompt")).toContainText("goal: Analyze");

  await page.getByRole("button", { name: "Copy prompt" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Saved optimization runs" }).getByRole("button")).toContainText("Help me with this.");
  await page.reload();
  await page.getByRole("list", { name: "Saved optimization runs" }).getByRole("button").click();
  await expect(page.getByRole("heading", { name: "Run details" })).toBeVisible();
  await expect(page.getByText(/goal: Analyze/)).toBeVisible();
  await page.getByRole("button", { name: "Accept result" }).click();
  await expect(page.getByText("Saved feedback: accept")).toBeVisible();
  await page.reload();
  await page.getByRole("list", { name: "Saved optimization runs" }).getByRole("button").click();
  await expect(page.getByText("Saved feedback: accept")).toBeVisible();
});

test("a retained original shows its available pass rates as the selected result", async ({ page }) => {
  await page.route("**/api/optimize", async (route) => {
    await route.fulfill({
      json: {
        status: "completed", run_id: "no-change-1", original_prompt: "Write a report.",
        final_prompt: "Write a report.", original_kept: true,
        report: {
          summary: "No candidate beat the original.",
          diagnosis: { task_type: "writing", confirmed_gaps: [], problem_sentences: [] },
          tests: [{ id: "test-1", question: "Does the answer address the report?" }],
          selection_evidence: {
            original_score: { per_model: { "weak-one": 0.75 }, spread: 0.25 },
            winner_score: null, rejected_candidates: [],
          },
          strong_check: { original_score: 0.8, candidates: [] },
        },
        cost: { total: 0, cost_by_role: {} }, timing: { total_ms: 5 },
      },
    });
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a report.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await page.getByText("View report").click();
  const row = page.getByRole("row", { name: /weak-one/ });
  await expect(row.getByRole("cell")).toHaveText(["weak-one", "75%", "75%"]);
  await expect(page.getByText("Sample spread: original 25%, selected 25%")).toBeVisible();
});
