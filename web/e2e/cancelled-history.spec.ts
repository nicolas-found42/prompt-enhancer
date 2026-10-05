import { expect, test } from "@playwright/test";

const cancelledRun = {
  run_id: "cancelled-run",
  status: "failed",
  control_state: "cancelled",
  prompt: "Keep the original prompt.",
};

const failedRun = {
  run_id: "failed-run",
  status: "failed",
  outcome: "failed_operational",
  prompt: "Write a useful reply.",
};

const cancellationFailure = {
  kind: "cancelled",
  headline: "Run cancelled",
  hint: "You cancelled this run. Your prompt was not changed.",
  message: "cancelled by the user",
};

test("History shows cancelled runs neutrally in the list and opened details", async ({
  page,
}) => {
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/runs?*", (route) =>
    route.fulfill({ json: [cancelledRun, failedRun] })
  );
  await page.route("**/api/runs/cancelled-run", (route) =>
    route.fulfill({
      json: {
        ...cancelledRun,
        report: {
          status: "cancelled",
          control_state: "cancelled",
          failure: cancellationFailure,
        },
        result: {
          status: "failed",
          run_id: "cancelled-run",
          report: {
            status: "cancelled",
            control_state: "cancelled",
            failure: cancellationFailure,
          },
          cost: { total: 0 },
          timing: { total_ms: 10 },
        },
      },
    })
  );

  await page.goto("/");

  const history = page.getByRole("list", { name: "Saved optimization runs" });
  const cancelledRow = history.getByRole("button", {
    name: /Keep the original prompt/,
  });
  await expect(
    cancelledRow.locator(".badge:not(.badge-control-state)")
  ).toHaveText("Outcome not established");
  await expect(cancelledRow.locator(".badge-control-state")).toHaveText(
    "Cancelled"
  );

  const failedRow = history.getByRole("button", {
    name: /Write a useful reply/,
  });
  await expect(failedRow.locator(".badge")).toHaveText("Failed (operational)");
  await expect(failedRow.locator(".badge")).toHaveClass(/badge-bad/);

  await cancelledRow.click();

  const details = page.getByRole("article", { name: "Run details" });
  await expect(details.locator(".badge:not(.badge-control-state)")).toHaveText(
    "Outcome not established"
  );
  await expect(details.locator(".badge-control-state")).toHaveText("Cancelled");
  await expect(
    details.getByRole("heading", { name: "Run cancelled" })
  ).toBeVisible();
});
