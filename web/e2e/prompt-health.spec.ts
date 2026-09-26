import { expect, test, type Page } from "@playwright/test";

const settings = {
  available: true,
  default_enabled: true,
  debounce_ms: 600,
  hourly_allowance_usd: 0.05,
  usage: { rolling_hour_usd: 0, refreshes_last_minute: 0 },
};

function assessment(prompt: string, revision: number) {
  return {
    draft: { revision, hash: `hash-${prompt}` },
    status: "complete",
    composite: 0.83,
    provisional: true,
    dimensions: [
      {
        id: "task",
        label: "Task or goal",
        applicable: true,
        applicability_probability: 0.99,
        score: 2.5,
        normalized: 0.83,
      },
    ],
    coverage: { assessed: 1, applicable: 1, unknown: 0 },
    flags: [
      {
        sentence_id: "s0001",
        kind: "vagueness",
        start: 0,
        end: prompt.length,
        text: prompt,
        probability: 0.91,
        threshold: 0.8,
      },
    ],
    cache: { hits: 0, misses: 1 },
    usage: {
      rolling_hour_usd: 0.001,
      request_usd: 0.001,
      provider_requests: 1,
    },
  };
}

async function routeSettings(page: Page) {
  await page.route("**/api/prompt-health/settings", (route) =>
    route.fulfill({ json: settings })
  );
}

test("debounces draft checks, shows current dimensions and flags, and persists toggle", async ({
  page,
}) => {
  await page.clock.install();
  await routeSettings(page);
  const prompts: string[] = [];
  await page.route("**/api/prompt-health", async (route) => {
    const body = route.request().postDataJSON() as {
      prompt: string;
      revision: number;
    };
    prompts.push(body.prompt);
    await route.fulfill({ json: assessment(body.prompt, body.revision) });
  });
  await page.goto("/");
  const draft = page.getByRole("textbox", { name: "Your prompt" });
  await draft.fill("Write a note.");
  await page.clock.runFor(500);
  expect(prompts).toHaveLength(0);
  await page.clock.runFor(100);
  await expect(page.getByText("Prompt clarity: 83%")).toBeVisible();
  await expect(page.getByText("Task or goal")).toBeVisible();
  await page.getByRole("button", { name: /Vague wording/ }).click();
  expect(
    await draft.evaluate(
      (element: HTMLTextAreaElement) => element.selectionStart
    )
  ).toBe(0);
  expect(
    await draft.evaluate((element: HTMLTextAreaElement) => element.selectionEnd)
  ).toBe(13);

  await page.getByRole("checkbox", { name: "Live prompt checks" }).uncheck();
  await draft.fill("Write a letter.");
  await page.clock.runFor(650);
  expect(prompts).toEqual(["Write a note."]);
  await page.reload();
  await expect(
    page.getByRole("checkbox", { name: "Live prompt checks" })
  ).not.toBeChecked();
  await page.getByRole("checkbox", { name: "Live prompt checks" }).check();
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await draft.fill("A hidden draft.");
  await page.clock.runFor(650);
  expect(prompts).toEqual(["Write a note."]);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => false,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.clock.runFor(600);
  await expect(
    page.getByRole("button", { name: /A hidden draft/ })
  ).toBeVisible();
});

test("late and failed health responses never replace the current draft", async ({
  page,
}) => {
  await page.clock.install();
  await routeSettings(page);
  let releaseFirst: (() => void) | undefined;
  const firstPending = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });
  await page.route("**/api/prompt-health", async (route) => {
    const body = route.request().postDataJSON() as {
      prompt: string;
      revision: number;
    };
    if (body.prompt === "Old draft.") await firstPending;
    if (body.prompt === "Failed draft.") {
      await route.fulfill({ status: 503, body: "unavailable" });
    } else {
      await route
        .fulfill({ json: assessment(body.prompt, body.revision) })
        .catch(() => undefined);
    }
  });
  await page.goto("/");
  const draft = page.getByRole("textbox", { name: "Your prompt" });
  await draft.fill("Old draft.");
  await page.clock.runFor(600);
  await draft.fill("Current draft.");
  await page.clock.runFor(600);
  await expect(
    page.getByRole("button", { name: /Current draft/ })
  ).toBeVisible();
  releaseFirst?.();
  await expect(page.getByRole("button", { name: /Old draft/ })).toHaveCount(0);
  await draft.fill("Failed draft.");
  await page.clock.runFor(600);
  await expect(page.getByRole("button", { name: /Current draft/ })).toHaveCount(
    0
  );
  await expect(page.getByRole("alert")).toHaveCount(0);
});
