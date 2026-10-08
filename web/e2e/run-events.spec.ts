import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Provider text stays text, and reconnect restores chronological facts.
test("event cursors, draft diffs, repeated rounds and cancellation survive reload", async ({
  page,
}, testInfo) => {
  const runId = "activity-run";
  const base = {
    run_id: runId,
    kind: "optimize",
    state: "running",
    stage: "writing_candidates",
    round: { round: 2 },
    stages_seen: ["writing_candidates"],
    elapsed_ms: 45000,
    remaining_active_ms: 105000,
    cancel_requested: false,
    result: null,
  };
  const events = [
    {
      cursor: 1,
      kind: "started",
      stage: "writing_candidates",
      summary: "Writing",
      elapsed_ms: 1000,
      round: 1,
    },
    {
      cursor: 2,
      kind: "draft",
      summary: "A draft is ready for checks; it has not qualified yet.",
      draft: "<script>window.unsafeDraft = true</script>",
      diff: "- Original\n+ <script>window.unsafeDraft = true</script>",
      elapsed_ms: 5000,
      round: 1,
    },
    {
      cursor: 3,
      kind: "completed",
      stage: "writing_candidates",
      summary: "Writing completed.",
      elapsed_ms: 10000,
      round: 1,
    },
    {
      cursor: 4,
      kind: "retry",
      summary: "The model service is busy; waiting before another attempt.",
      elapsed_ms: 30000,
      round: 1,
    },
    {
      cursor: 5,
      kind: "started",
      stage: "writing_candidates",
      summary: "Writing",
      elapsed_ms: 45000,
      round: 2,
    },
  ];
  const requested: number[] = [];
  let cancelled = false;
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) =>
    route.fulfill({
      status: 202,
      json: { ...base, events: events.slice(0, 1), event_cursor: 1 },
    })
  );
  await page.route(`**/api/jobs/${runId}**`, (route) => {
    if (route.request().method() === "POST") {
      cancelled = true;
      return route.fulfill({
        json: {
          ...base,
          cancel_requested: true,
          cancellation_pending: true,
          events,
          event_cursor: 5,
        },
      });
    }
    const cursor = Number(
      new URL(route.request().url()).searchParams.get("after_cursor") ?? 0
    );
    requested.push(cursor);
    return route.fulfill({
      json: {
        ...base,
        events: events.filter((event) => event.cursor > cursor),
        event_cursor: 5,
      },
    });
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Improve my prompt.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  const activity = page.getByRole("list", { name: "Run activity" });
  await expect(
    activity.getByText(/Writing improved versions — started/)
  ).toHaveCount(2);
  await expect(
    activity.getByText(/Writing improved versions — completed/)
  ).toBeVisible();
  await expect.poll(() => requested).toContain(1);
  await expect.poll(() => requested).toContain(5);
  await activity.getByText("Draft preview — awaiting checks").click();
  await activity.getByText("Changes from your prompt").click();
  await expect(activity.locator("pre").first()).toHaveText(
    "<script>window.unsafeDraft = true</script>"
  );
  expect(await page.evaluate(() => "unsafeDraft" in window)).toBe(false);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.screenshot({
    path: testInfo.outputPath("run-events.png"),
    fullPage: true,
  });
  await page.reload();
  await expect(
    activity.getByText(/Writing improved versions — started/)
  ).toHaveCount(2);
  await expect(activity.locator("li.stage")).toHaveCount(5);
  expect(requested).toContain(0);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  expect(cancelled).toBe(true);
  await expect(
    page.getByRole("button", { name: "Cancelling…" })
  ).toBeDisabled();
});
