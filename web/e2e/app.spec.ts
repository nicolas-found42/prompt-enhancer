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
