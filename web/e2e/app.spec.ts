import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

type Json = Record<string, unknown>;

test("the prompt workbench has no automated accessibility violations", async ({
  page,
}) => {
  await mockRun(page, {
    ...completedBase,
    status: "needs_input",
    run_id: "accessibility-run",
    report: {},
    questions: [
      {
        id: "goal",
        prompt: "What should the assistant do?",
        options: [{ value: "summarize", label: "Summarize" }],
      },
    ],
  });
  await page.goto("/");
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);

  await page.getByLabel("Your prompt").fill("Help me with this.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(
    page.getByRole("heading", { name: "A few details will improve the result" })
  ).toBeVisible();
  const clarification = await new AxeBuilder({ page }).analyze();
  expect(clarification.violations).toEqual([]);
});

function job(
  runId: string,
  state: "running" | "done",
  result: Json | null = null,
  extra: Json = {}
): Json {
  return {
    run_id: runId,
    kind: "optimize",
    state,
    stage: state === "running" ? "grading" : null,
    round: { round: 1, max_rounds: 2 },
    stages_seen: [],
    elapsed_ms: 65000,
    cancel_requested: false,
    result,
    ...extra,
  };
}

/** Serve one mocked run through the job endpoints, finishing on the first poll. */
async function mockRun(page: Page, result: Json) {
  const runId = String(result.run_id);
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) =>
    route.fulfill({ status: 202, json: job(runId, "running") })
  );
  await page.route(`**/api/jobs/${runId}`, (route) =>
    route.fulfill({ json: job(runId, "done", result) })
  );
}

const completedBase = {
  status: "completed",
  original_kept: true,
  cost: { total: 0.002, cost_by_role: {} },
  timing: { total_ms: 5 },
};

test("an unsent draft keeps its exact whitespace after reload", async ({
  page,
}) => {
  await page.goto("/");
  const draft = "  Draft: Ask for a written date.\n\tKeep it friendly.  \n";
  await page.getByLabel("Your prompt").fill(draft);

  await page.reload();

  await expect(page.getByLabel("Your prompt")).toHaveValue(draft);
});

test("an active result identifies when it belongs to the earlier prompt", async ({
  page,
}) => {
  const runId = "earlier-prompt-result";
  const runPrompt = "Write a concise update about the delivery date.";
  const currentDraft = "Draft: Ask for the confirmed arrival date by Friday.";
  const completedResult = {
    ...completedBase,
    run_id: runId,
    original_prompt: runPrompt,
    final_prompt: `${runPrompt} Include the next steps.`,
    original_kept: false,
    report: {
      status: "optimized",
      summary: "The update has a clear deadline.",
    },
  };
  let finished = false;

  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) =>
    route.fulfill({
      status: 202,
      json: job(runId, "running", null, { prompt: runPrompt }),
    })
  );
  await page.route(`**/api/jobs/${runId}`, (route) =>
    route.fulfill({
      json: finished
        ? job(runId, "done", completedResult, { prompt: runPrompt })
        : job(runId, "running", null, { prompt: runPrompt }),
    })
  );

  await page.goto("/");
  await page.getByLabel("Your prompt").fill(runPrompt);
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(
    page.getByRole("heading", { name: "Improving your prompt" })
  ).toBeVisible();

  await page.getByRole("textbox", { name: "Your prompt" }).fill(currentDraft);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Improving your prompt" })
  ).toBeVisible();
  finished = true;

  await expect(page.locator(".final-prompt")).toContainText(
    "Include the next steps."
  );
  await expect(page.getByRole("textbox", { name: "Your prompt" })).toHaveValue(
    currentDraft
  );
  await expect(
    page.getByRole("status").filter({
      hasText:
        "A result for an earlier prompt is open below. Your current draft remains in Your prompt.",
    })
  ).toBeVisible();
});

test("clarification can recover from local and server-side Other answer errors", async ({
  page,
}) => {
  const runId = "clarification-recovery-run";
  const pendingResult = {
    status: "needs_input",
    run_id: runId,
    report: {},
    questions: [
      {
        id: "goal",
        prompt: "What should the assistant do?",
        options: [
          { value: "summarize", label: "Summarize", preselected: true },
          { value: "other", label: "Other", other: true },
        ],
        default_answer: "summarize",
        allow_other: true,
        other_value: "other",
      },
      {
        id: "audience",
        prompt: "Who is the report for?",
        options: [
          { value: "team", label: "The team", preselected: true },
          { value: "other", label: "Other", other: true },
        ],
        default_answer: "team",
        allow_other: true,
        other_value: "other",
      },
    ],
  };
  const completedResult = {
    ...completedBase,
    status: "completed",
    run_id: runId,
    original_prompt: "Write a concise report.",
    final_prompt: "Write a concise report about the supplier delivery date.",
    report: { assumptions: [] },
  };
  const resumeRequests: Json[] = [];
  let optimizationRequests = 0;
  let rejectedOnce = false;
  let resumed = false;

  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", async (route) => {
    optimizationRequests += 1;
    await route.fulfill({
      status: 202,
      json: job(runId, "running"),
    });
  });
  await page.route(`**/api/jobs/${runId}/resume`, async (route) => {
    resumeRequests.push(route.request().postDataJSON() as Json);
    if (!rejectedOnce) {
      rejectedOnce = true;
      await route.fulfill({
        status: 422,
        json: {
          detail: {
            code: "invalid_answer",
            question_id: "goal",
            message: "Answer for 'goal' requires other text",
          },
        },
      });
      return;
    }
    resumed = true;
    await route.fulfill({ status: 202, json: job(runId, "running") });
  });
  await page.route(`**/api/jobs/${runId}`, (route) =>
    route.fulfill({
      json: resumed
        ? job(runId, "done", completedResult, { kind: "resume" })
        : job(runId, "done", pendingResult),
    })
  );

  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a concise report.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  const heading = page.getByRole("heading", {
    name: "A few details will improve the result",
  });
  await expect(heading).toBeVisible();
  await page.getByRole("radio", { name: "Other" }).first().check();
  await page.getByRole("radio", { name: "Other" }).nth(1).check();
  const other = page.getByRole("textbox", {
    name: "Other answer for What should the assistant do?",
  });
  const audience = page.getByRole("textbox", {
    name: "Other answer for Who is the report for?",
  });
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  await expect(other).toHaveAttribute("aria-invalid", "true");
  await expect(other).toBeFocused();
  await expect(audience).toHaveAttribute("aria-invalid", "true");
  await expect(page.locator("#clarification-error-goal")).toHaveText(
    "Enter an answer for this question."
  );
  expect(resumeRequests).toHaveLength(0);

  await other.fill("A short summary");
  await audience.fill("Team leads");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(other).toHaveAttribute("aria-invalid", "true");
  await expect(other).toHaveValue("A short summary");
  await expect(audience).toHaveValue("Team leads");
  await expect(other).toBeFocused();
  await expect(page.locator("#clarification-error-goal")).toHaveText(
    "Answer for 'goal' requires other text"
  );
  expect(resumeRequests).toHaveLength(1);

  await other.fill("Summarize the supplier delivery date");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator(".final-prompt")).toContainText(
    "Write a concise report about the supplier delivery date."
  );
  expect(optimizationRequests).toBe(1);
  expect(resumeRequests).toHaveLength(2);
  expect(resumeRequests[0]).toEqual({
    answers: {
      goal: { value: "other", text: "A short summary" },
      audience: { value: "other", text: "Team leads" },
    },
  });
  expect(resumeRequests[1]).toEqual({
    answers: {
      goal: { value: "other", text: "Summarize the supplier delivery date" },
      audience: { value: "other", text: "Team leads" },
    },
  });
});

test("opening another paused run resets its answers and error", async ({
  page,
}) => {
  const runs = [
    { run_id: "paused-a", prompt: "First paused prompt" },
    { run_id: "paused-b", prompt: "Second paused prompt" },
  ];
  const question = {
    id: "goal",
    prompt: "What should the assistant do?",
    options: [
      { value: "summarize", label: "Summarize" },
      { value: "review", label: "Review" },
    ],
    allow_other: true,
  };
  const results: Record<string, Json> = {
    "paused-a": {
      run_id: "paused-a",
      status: "needs_input",
      report: {},
      questions: [{ ...question, default: "summarize" }],
    },
    "paused-b": {
      run_id: "paused-b",
      status: "needs_input",
      report: {},
      questions: [{ ...question, default: "review" }],
    },
  };
  let submittedAnswers: Json | null = null;
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/runs?*", (route) =>
    route.fulfill({
      json: runs.map((run) => ({
        ...run,
        created_at: "2026-09-24T04:00:00Z",
        status: "needs_input",
        tier: "standard",
      })),
    })
  );
  await page.route("**/api/runs/paused-*", (route) => {
    const runId = route.request().url().split("/").at(-1) ?? "";
    const run = runs.find((item) => item.run_id === runId);
    return route.fulfill({
      json: { ...run, status: "needs_input", result: results[runId] },
    });
  });
  await page.route("**/api/jobs/paused-a/resume", (route) =>
    route.fulfill({
      status: 422,
      json: {
        detail: {
          code: "invalid_answer",
          question_id: "goal",
          message: "First run needs a different goal.",
        },
      },
    })
  );
  await page.route("**/api/jobs/paused-b/resume", (route) => {
    submittedAnswers = route.request().postDataJSON() as Json;
    return route.fulfill({ status: 202, json: job("paused-b", "running") });
  });

  await page.goto("/");
  const history = page.getByRole("list", { name: "Saved optimization runs" });
  await history.getByRole("button", { name: /First paused prompt/ }).click();
  await page.getByRole("button", { name: "Open this result" }).click();
  await page.getByRole("radio", { name: "Other" }).check();
  await page
    .getByRole("textbox", {
      name: "Other answer for What should the assistant do?",
    })
    .fill("An answer for the first run");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByText("First run needs a different goal.")
  ).toBeVisible();

  await history.getByRole("button", { name: /Second paused prompt/ }).click();
  await page.getByRole("button", { name: "Open this result" }).click();
  await expect(page.getByRole("radio", { name: "Review" })).toBeChecked();
  await expect(page.getByText("First run needs a different goal.")).toHaveCount(
    0
  );

  await page.getByRole("radio", { name: "Other" }).check();
  await page
    .getByRole("textbox", {
      name: "Other answer for What should the assistant do?",
    })
    .fill("An answer for the second run");
  results["paused-b"] = {
    ...results["paused-b"],
    questions: [{ ...question, default: "summarize" }],
  };
  await history.getByRole("button", { name: /Second paused prompt/ }).click();
  await history.getByRole("button", { name: /Second paused prompt/ }).click();
  await page.getByRole("button", { name: "Open this result" }).click();
  await expect(page.getByRole("radio", { name: "Summarize" })).toBeChecked();
  await expect(
    page.getByRole("textbox", {
      name: "Other answer for What should the assistant do?",
    })
  ).toHaveCount(0);

  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect
    .poll(() => submittedAnswers)
    .toEqual({
      answers: { goal: "summarize" },
    });
});

test("skip and continue still resumes the same paused run", async ({
  page,
}) => {
  const runId = "clarification-skip-run";
  const pendingResult = {
    status: "needs_input",
    run_id: runId,
    report: {},
    questions: [
      {
        id: "goal",
        prompt: "What should the assistant do?",
        options: [{ value: "summarize", label: "Summarize" }],
        default_answer: "summarize",
      },
    ],
  };
  const completedResult = {
    ...completedBase,
    status: "completed",
    run_id: runId,
    original_prompt: "Write a concise report.",
    final_prompt: "Write a concise report.",
    report: { assumptions: [] },
  };
  let optimizationRequests = 0;
  let skipRequests = 0;
  let skipped = false;

  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", async (route) => {
    optimizationRequests += 1;
    await route.fulfill({ status: 202, json: job(runId, "running") });
  });
  await page.route(`**/api/jobs/${runId}/skip`, async (route) => {
    skipRequests += 1;
    skipped = true;
    await route.fulfill({
      status: 202,
      json: job(runId, "running", null, { kind: "skip" }),
    });
  });
  await page.route(`**/api/jobs/${runId}`, (route) =>
    route.fulfill({
      json: skipped
        ? job(runId, "done", completedResult, { kind: "skip" })
        : job(runId, "done", pendingResult),
    })
  );

  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a concise report.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();
  await expect(
    page.getByRole("heading", {
      name: "A few details will improve the result",
    })
  ).toBeVisible();
  await page.getByRole("button", { name: "Skip and continue" }).click();

  await expect(page.locator(".final-prompt")).toContainText(
    "Write a concise report."
  );
  expect(optimizationRequests).toBe(1);
  expect(skipRequests).toBe(1);
});

test("an intentionally empty draft stays empty when an older result restores", async ({
  page,
}) => {
  await page.addInitScript(() => {
    localStorage.setItem("prompt-enhancer.draft", "");
    localStorage.setItem(
      "prompt-enhancer.last-result",
      JSON.stringify({ runId: "older-result", at: Date.now() })
    );
  });
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/runs/older-result", (route) =>
    route.fulfill({
      json: {
        result: {
          run_id: "older-result",
          status: "failed",
          original_prompt: "An older prompt from History.",
          final_prompt: "An older prompt from History.",
          original_kept: true,
          report: {
            failure: {
              kind: "internal",
              headline: "Restored older result",
              hint: "This result was restored.",
            },
          },
          cost: { total: 0 },
          timing: { total_ms: 1 },
        },
      },
    })
  );

  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: "Restored older result" })
  ).toBeVisible();
  await expect(page.getByLabel("Your prompt")).toHaveValue("");
});

test("a late result restore does not repopulate a cleared draft", async ({
  page,
}) => {
  await page.addInitScript(() => {
    localStorage.setItem("prompt-enhancer.draft", "Draft to clear");
    localStorage.setItem(
      "prompt-enhancer.last-result",
      JSON.stringify({ runId: "late-result", at: Date.now() })
    );
  });
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  let releaseResult!: () => void;
  const resultGate = new Promise<void>((resolve) => {
    releaseResult = resolve;
  });
  await page.route("**/api/runs/late-result", async (route) => {
    await resultGate;
    await route.fulfill({
      json: {
        result: {
          run_id: "late-result",
          status: "failed",
          original_prompt: "An older prompt from a slow restore.",
          original_kept: true,
          report: {
            failure: {
              kind: "internal",
              headline: "Restored late result",
              hint: "This result was restored.",
            },
          },
          cost: { total: 0 },
          timing: { total_ms: 1 },
        },
      },
    });
  });

  await page.goto("/");
  const prompt = page.getByLabel("Your prompt");
  await expect(prompt).toHaveValue("Draft to clear");
  await prompt.fill("");
  releaseResult();

  await expect(
    page.getByRole("heading", { name: "Restored late result" })
  ).toBeVisible();
  await expect(prompt).toHaveValue("");
});

test("opening a saved result preserves the current draft", async ({ page }) => {
  const olderPrompt = "Write a friendly note about the delivery delay.";
  const result = {
    run_id: "history-open-result",
    status: "completed",
    original_prompt: olderPrompt,
    final_prompt: "Write a friendly note about the delivery delay by Friday.",
    original_kept: false,
    report: { status: "edited", summary: "Added a delivery date." },
    cost: { total: 0.001 },
    timing: { total_ms: 1 },
  };
  await page.route("**/api/runs?*", (route) =>
    route.fulfill({
      json: [
        {
          run_id: result.run_id,
          created_at: "2026-09-24T04:00:00Z",
          status: "completed",
          prompt: olderPrompt,
          final_prompt: result.final_prompt,
          original_kept: false,
          tier: "standard",
        },
      ],
    })
  );
  await page.route(`**/api/runs/${result.run_id}`, (route) =>
    route.fulfill({
      json: {
        run_id: result.run_id,
        created_at: "2026-09-24T04:00:00Z",
        status: "completed",
        prompt: olderPrompt,
        original_prompt: olderPrompt,
        final_prompt: result.final_prompt,
        original_kept: false,
        tier: "standard",
        cost: result.cost,
        timings: { total_ms: 1 },
        result,
      },
    })
  );

  await page.goto("/");
  const draft = "Draft: Confirm the date before replying.\nKeep it warm.";
  await page.getByLabel("Your prompt").fill(draft);
  await page
    .getByRole("list", { name: "Saved optimization runs" })
    .getByRole("button", { name: new RegExp(olderPrompt) })
    .click();
  await page.getByRole("button", { name: "Open this result" }).click();

  await expect(page.getByLabel("Your prompt")).toHaveValue(draft);
  await expect(
    page.getByRole("status").filter({
      hasText:
        "A saved result is open below. Your current draft remains in Your prompt.",
    })
  ).toBeVisible();
});

test("clarification, assumption editing, history, and feedback use the local API", async ({
  page,
}) => {
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
  await expect(
    page.getByRole("heading", { name: "A few details will improve the result" })
  ).toBeVisible();
  await expect(page.getByText("What should the assistant do?")).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Optimized prompt" })
  ).toBeVisible();
  await expect(page.locator(".final-prompt")).toContainText("goal: Summarize");

  await page.getByText("View report").click();
  const assumption = page.getByLabel("Goal");
  await expect(assumption).toHaveValue("Summarize");
  await assumption.fill("Analyze");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Updated prompt" })
  ).toBeVisible();
  await expect(page.locator(".final-prompt")).toContainText("goal: Analyze");

  await page.getByRole("button", { name: "Copy prompt" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
  const runs = page
    .getByRole("list", { name: "Saved optimization runs" })
    .getByRole("button");
  await expect(runs).toContainText("Help me with this.");
  await expect(runs).toContainText("Improved");
  await page.reload();
  await page
    .getByRole("list", { name: "Saved optimization runs" })
    .getByRole("button")
    .click();
  await expect(
    page.getByRole("heading", { name: "Run details" })
  ).toBeVisible();
  await expect(
    page.locator("#selected-run").getByText(/goal: Analyze/)
  ).toBeVisible();
  await page.getByRole("button", { name: "Yes", exact: true }).click();
  await expect(page.getByText("Thanks, saved as helpful.")).toBeVisible();
  await page.reload();
  // The result shown before the reload comes back instead of vanishing.
  await expect(
    page.getByRole("heading", { name: "Updated prompt" })
  ).toBeVisible();
  await page
    .getByRole("list", { name: "Saved optimization runs" })
    .getByRole("button")
    .click();
  await expect(page.getByText("Thanks, saved as helpful.")).toBeVisible();
  await page.getByRole("button", { name: "Open this result" }).click();
  await expect(page.locator(".final-prompt")).toContainText("goal: Analyze");
});

test("a retained original with weak evidence says it could not be improved", async ({
  page,
}) => {
  await mockRun(page, {
    ...completedBase,
    run_id: "no-change-1",
    original_prompt: "Write a report.",
    final_prompt: "Write a report.",
    report: {
      status: "no_change",
      summary: "No candidate beat the original.",
      diagnosis: {
        task_type: "writing",
        confirmed_gaps: [{ key: "context", label: "relevant context" }],
        problem_sentences: [],
      },
      tests: [
        { id: "test-1", question: "Does the answer address the report?" },
      ],
      selection_evidence: {
        original_score: {
          per_model: { "weak-one": 0.75, "weak-two": 0 },
          spread: 0.25,
        },
        winner_score: null,
        rejected_candidates: [
          {
            candidate_id: "c1",
            strategy: "add_missing_context",
            rejection_reasons: ["candidate failed fidelity checks"],
          },
        ],
      },
      strong_check: { original_score: 0.8, candidates: [] },
      offer_deep: { expected_evaluation_multiplier: 5.625 },
    },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a report.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(
    page.getByRole("heading", { name: "We couldn't safely improve this" })
  ).toBeVisible();
  await expect(page.getByText(/It is missing relevant context/)).toBeVisible();
  await expect(page.getByText(/passed only 0% of checks/)).toBeVisible();
  await expect(
    page.getByText(/about 5\.6× the work of this run, roughly \$0\.01/)
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Try a Deep pass" })
  ).toBeVisible();
  await page.getByText("View report").click();
  const row = page.getByRole("row", { name: /weak-one/ });
  await expect(row.getByRole("cell")).toHaveText(["weak-one", "75%", "75%"]);
  await expect(
    page.getByText("Variation between samples: original 25%, selected 25%")
  ).toBeVisible();
  await expect(
    page.getByText(
      "Add missing context: it changed your meaning or added things you didn't ask for"
    )
  ).toBeVisible();
});

test("a prompt with no gaps and strong results says it already works", async ({
  page,
}) => {
  await mockRun(page, {
    ...completedBase,
    run_id: "clear-1",
    original_prompt: "Write two sentences.",
    final_prompt: "Write two sentences.",
    report: {
      status: "no_change",
      diagnosis: { confirmed_gaps: [], problem_sentences: [] },
      tests: [],
      selection_evidence: {
        original_score: { per_model: { "weak-one": 0.9, "weak-two": 1 } },
      },
    },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write two sentences.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(
    page.getByRole("heading", { name: "Your prompt already works well" })
  ).toBeVisible();
  await expect(
    page.getByText(/passed at least 90% of checks on every test model/)
  ).toBeVisible();
});

test("an untested prompt is not called good, near misses are hinted, and Deep is not offered", async ({
  page,
}) => {
  const prompt =
    "Tell the lads about the new rules for the vans. Keep it short.";
  await mockRun(page, {
    ...completedBase,
    run_id: "untested-1",
    original_prompt: prompt,
    final_prompt: prompt,
    report: {
      status: "no_change",
      diagnosis: {
        confirmed_gaps: [],
        problem_sentences: [],
        possible_gaps: [
          {
            key: "outside_reference",
            label: "details only you know",
            missing_probability: 0.86,
            threshold: 0.9,
            sentence: "Tell the lads about the new rules for the vans.",
          },
        ],
      },
      tests: [{ id: "test-1", question: "Is it short?" }],
      offer_deep: { expected_evaluation_multiplier: 5.625 },
    },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill(prompt);
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(
    page.getByRole("heading", { name: "We didn't find anything to fix" })
  ).toBeVisible();
  await expect(
    page.getByText("Your prompt wasn't tested on other models.", {
      exact: false,
    })
  ).toBeVisible();
  await expect(page.getByText("Your prompt already works well")).toHaveCount(0);
  await expect(
    page.getByText(
      /It may depend on details only you know, such as what “Tell the lads about the new rules for the vans\.” refers to/
    )
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Try a Deep pass" })
  ).toHaveCount(0);
});

test("opening a history run shows its details next to the row", async ({
  page,
}) => {
  const runs = Array.from({ length: 12 }, (_, index) => ({
    run_id: `hist-${index}`,
    created_at: "2026-09-24T04:00:00Z",
    status: "completed",
    prompt: `Saved prompt number ${index}`,
    final_prompt: `Saved prompt number ${index}`,
    original_kept: true,
    tier: index === 0 ? "deep" : "standard",
    escalated_from: index === 0 ? "standard" : null,
  }));
  await page.route("**/api/runs?*", (route) => route.fulfill({ json: runs }));
  await page.route("**/api/runs/hist-*", (route) => {
    const run = runs.find((item) =>
      route.request().url().endsWith(item.run_id)
    )!;
    return route.fulfill({
      json: {
        ...run,
        original_prompt: run.prompt,
        timings: { total_ms: 28000 },
        cost: { total: 0.0001 },
      },
    });
  });
  await page.route("**/api/estimates", (route) =>
    route.fulfill({
      json: {
        standard: { runs: 4, minutes: [0.3, 6.6], cost: [0.001, 0.012] },
      },
    })
  );
  await page.setViewportSize({ width: 1366, height: 900 });
  await page.goto("/");
  await expect(
    page.getByText(
      /Standard: up to 2 rounds of rewrites, tried on 3 test models\. Usually under a minute, up to 7 min/
    )
  ).toBeVisible();

  const list = page.getByRole("list", { name: "Saved optimization runs" });
  const first = list.getByRole("button", { name: /Saved prompt number 0/ });
  await expect(first).toContainText("Unchanged");
  await expect(first).toContainText("standard, then deep");
  await first.click();
  await expect(first).toHaveAttribute("aria-expanded", "true");
  await expect(
    page.getByRole("heading", { name: "Run details" })
  ).toBeInViewport();
  await expect(
    page.getByText(/started on standard and then had a Deep pass/)
  ).toBeVisible();
  await first.click();
  await expect(page.getByRole("heading", { name: "Run details" })).toHaveCount(
    0
  );

  const later = list.getByRole("button", { name: /Saved prompt number 6/ });
  await later.click();
  await expect(
    page.getByRole("heading", { name: "Run details" })
  ).toBeInViewport();
  await expect(page.locator("#selected-run")).toContainText(
    "Saved prompt number 6"
  );
});

test("a failed run explains the provider problem and offers a fix", async ({
  page,
}) => {
  await mockRun(page, {
    status: "failed",
    run_id: "failed-1",
    final_prompt: "Write a reply.",
    original_kept: true,
    report: {
      status: "failed",
      error:
        "go request for space-bunny-free failed (HTTP 403): provider request failed",
      failure: {
        kind: "http",
        provider: "go",
        model: "space-bunny-free",
        role: "writer",
        http_status: 403,
        headline: "OpenCode Go refused the request",
        hint: "Go models such as space-bunny-free need an active OpenCode Go subscription.",
        message:
          "go request for space-bunny-free failed (HTTP 403): provider request failed",
      },
    },
    cost: { total: 0 },
    timing: { total_ms: 1300 },
  });
  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Write a reply.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(
    page.getByRole("heading", { name: "OpenCode Go refused the request" })
  ).toBeVisible();
  await expect(page.getByText("Your prompt was not changed.")).toBeVisible();
  await page.getByText("Technical details").click();
  await expect(page.getByText("failed (HTTP 403)")).toBeVisible();
  await page.getByRole("button", { name: "Change models" }).click();
  await expect(page.getByLabel("Writer", { exact: true })).toBeVisible();
});

test("a running job shows its stage, elapsed time, and can be cancelled", async ({
  page,
}) => {
  let cancelled = false;
  await page.route("**/api/jobs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/jobs/optimize", (route) =>
    route.fulfill({ status: 202, json: job("slow-1", "running") })
  );
  await page.route("**/api/jobs/slow-1", (route) =>
    route.fulfill({
      json: cancelled
        ? job("slow-1", "done", {
            status: "failed",
            run_id: "slow-1",
            final_prompt: "Plan a trip.",
            original_kept: true,
            report: {
              status: "cancelled",
              failure: {
                kind: "cancelled",
                headline: "Run cancelled",
                hint: "You cancelled this run. Your prompt was not changed.",
                message: "cancelled by the user",
              },
            },
            cost: { total: 0 },
            timing: { total_ms: 1 },
          })
        : job("slow-1", "running"),
    })
  );
  await page.route("**/api/jobs/slow-1/cancel", (route) => {
    cancelled = true;
    return route.fulfill({
      json: job("slow-1", "running", null, { cancel_requested: true }),
    });
  });

  await page.goto("/");
  await page.getByLabel("Your prompt").fill("Plan a trip.");
  await page.getByRole("button", { name: "Optimize prompt" }).click();

  await expect(
    page.getByRole("heading", { name: "Improving your prompt" })
  ).toBeVisible();
  await expect(page.getByText("1:05 elapsed")).toBeVisible();
  await expect(page.getByText(/round 1 of 2/)).toBeVisible();
  await expect(page.locator('[aria-current="step"]')).toHaveText(
    "Scoring the answers"
  );
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Improving your prompt" })
  ).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Your prompt" })).toHaveValue(
    "Plan a trip."
  );
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(
    page.getByRole("heading", { name: "Run cancelled" })
  ).toBeVisible();
});

test("another tab joining a running job shows which prompt it is working on", async ({
  page,
}) => {
  const running = job("other-tab-1", "running", null, {
    prompt: "Write the van rules notice.",
  });
  await page.route("**/api/jobs", (route) =>
    route.fulfill({ json: [running] })
  );
  await page.route("**/api/jobs/other-tab-1", (route) =>
    route.fulfill({ json: running })
  );

  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: "Improving your prompt" })
  ).toBeVisible();
  await expect(page.locator(".progress-prompt")).toHaveText(
    "Write the van rules notice."
  );
  await expect(page.getByRole("textbox", { name: "Your prompt" })).toHaveValue(
    "Write the van rules notice."
  );
});

test("a refused provider shows a banner that switches to fallback models", async ({
  page,
}) => {
  await page.route("**/api/catalog", (route) =>
    route.fulfill({
      json: {
        judge: { id: "typesafe/jev-1.13", provider: "openrouter" },
        providers: {
          go: [
            { id: "go-writer", provider: "go" },
            { id: "go-strong", provider: "go" },
          ],
          openrouter: [
            { id: "or-writer", provider: "openrouter" },
            { id: "or-strong", provider: "openrouter" },
            { id: "weak-a", provider: "openrouter" },
          ],
        },
      },
    })
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({
      json: {
        judge_model: "typesafe/jev-1.13",
        writer_model: "go-writer",
        strong_check_model: "go-strong",
        weak_models: ["weak-a"],
      },
    })
  );
  await page.route("**/api/providers*", (route) =>
    route.fulfill({
      json: {
        providers: { go: { status: "unavailable", http_status: 403 } },
        fallback: { writer: "or-writer", strong: "or-strong" },
      },
    })
  );

  await page.goto("/");
  await expect(page.getByText("OpenCode Go isn't active")).toBeVisible();
  await page
    .getByRole("button", { name: "Switch to OpenRouter models" })
    .click();
  await expect(page.getByText("OpenCode Go isn't active")).toBeHidden();
  await page.getByText("Model choices").click();
  await expect(page.getByLabel("Writer", { exact: true })).toHaveValue(
    "or-writer"
  );
  await expect(page.getByLabel("Strong check", { exact: true })).toHaveValue(
    "or-strong"
  );
});
