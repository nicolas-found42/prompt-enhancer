import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("a scoped hard conflict survives reload and preserves its source ledger in history", async ({
  page,
}, testInfo) => {
  test.setTimeout(90_000);
  const prompt =
    "Give Section A exactly two bullets and exactly three bullets. Preserve {topic}.";
  await page.goto("/");
  await page.getByLabel("Your prompt").fill(prompt);
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(
    page.getByText("Which requirement should apply to Section A?")
  ).toBeVisible({ timeout: 60_000 });
  await expect(
    page.getByRole("button", { name: "Skip", exact: true })
  ).toBeDisabled();
  await expect(
    page.getByRole("radio", { name: "two bullets", exact: true })
  ).not.toBeChecked();
  await page.reload();
  await expect(
    page.getByText("Which requirement should apply to Section A?")
  ).toBeVisible();
  await page.getByRole("radio", { name: "two bullets", exact: true }).check();
  await page.getByRole("button", { name: "Use my answers" }).click();
  await page.getByText("View report", { exact: true }).click();
  const coverage = page.getByRole("region", { name: "Requirement coverage" });
  await expect(
    coverage.getByText(/Your answer resolved these conflicting requirements/)
  ).toBeVisible();
  await expect(
    coverage.getByText(/Superseded by your explicit choice/)
  ).toBeVisible();
  await expect(
    coverage.getByText(/Applies to the Section A section/).first()
  ).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.screenshot({
    path: testInfo.outputPath("compound-requirements.png"),
    fullPage: true,
  });
  await page.reload();
  await expect(
    page
      .getByRole("list", { name: "Saved optimization runs" })
      .getByText(prompt)
  ).toBeVisible();
});
