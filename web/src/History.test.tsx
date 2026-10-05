import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import type { OptimizeResult } from "./api";
import History, {
  BADGE_EXPLANATIONS,
  badgeFor,
  controlLabelFor,
  type BadgeLabel,
  type RunDetail,
  type RunSummary,
} from "./History";

const savedRun: RunSummary = {
  run_id: "saved-run",
  prompt: "Write a supplier reply.",
};

/** `getByRole` takes a string or regex; escape prompts used as patterns. */
function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

const originalScrollIntoView = Object.getOwnPropertyDescriptor(
  Element.prototype,
  "scrollIntoView"
);

const originalClipboard = Object.getOwnPropertyDescriptor(
  navigator,
  "clipboard"
);

afterEach(() => {
  vi.unstubAllGlobals();
  if (originalClipboard)
    Object.defineProperty(navigator, "clipboard", originalClipboard);
  else Reflect.deleteProperty(navigator, "clipboard");
  if (originalScrollIntoView)
    Object.defineProperty(
      Element.prototype,
      "scrollIntoView",
      originalScrollIntoView
    );
  else Reflect.deleteProperty(Element.prototype, "scrollIntoView");
});

it("copies the final prompt from completed run details", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  const completedRun = {
    ...savedRun,
    status: "completed",
    original_prompt: "Write the original prompt.",
    final_prompt: "Write the improved prompt.",
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return jsonResponse(
        url.endsWith(`/api/runs/${completedRun.run_id}`)
          ? completedRun
          : [completedRun]
      );
    })
  );
  const user = userEvent.setup();
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  render(<History />);

  await user.click(
    await screen.findByRole("button", { name: /Write a supplier reply/ })
  );
  expect(
    await screen.findByRole("heading", { name: "Final prompt" })
  ).toBeVisible();
  expect(document.querySelectorAll("pre.history-text")[1]).toHaveTextContent(
    "Write the improved prompt."
  );
  expect(navigator.clipboard.writeText).toBe(writeText);
  await user.click(screen.getByRole("button", { name: "Copy prompt" }));

  expect(writeText).toHaveBeenCalledWith("Write the improved prompt.");
  expect(screen.getByRole("button", { name: "Copied" })).toBeVisible();
});

it("explains how to answer a paused run from its History details", async () => {
  const result = {
    status: "needs_input" as const,
    run_id: "paused-run",
    original_prompt: "Write a supplier reply.",
    questions: [
      {
        id: "goal",
        prompt: "What should the assistant do?",
        options: [{ value: "summarize", label: "Summarize" }],
      },
    ],
    report: {},
    cost: { total: 0 },
    timing: { total_ms: 1 },
  };
  const pausedRun = {
    ...savedRun,
    run_id: "paused-run",
    status: "needs_input",
    result,
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return jsonResponse(
        url.endsWith("/api/runs/paused-run") ? pausedRun : [pausedRun]
      );
    })
  );
  const onOpen = vi.fn();
  const user = userEvent.setup();
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  render(<History onOpen={onOpen} />);

  await user.click(
    await screen.findByRole("button", { name: /Write a supplier reply/ })
  );
  await screen.findByRole("heading", { name: "Run details" });
  expect(
    screen.getByText(
      "1 question is waiting in the clarification panel above History."
    )
  ).toBeVisible();
  const answerButton = screen.getByRole("button", {
    name: "Answer the questions",
  });
  expect(answerButton).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Open this result" })
  ).not.toBeInTheDocument();

  await user.click(answerButton);

  expect(onOpen).toHaveBeenCalledWith(result);
});

async function openRunDetails(run: RunDetail) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return jsonResponse(
        url.endsWith(`/api/runs/${run.run_id}`) ? run : [run]
      );
    })
  );
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });
  const user = userEvent.setup();
  render(<History />);
  await user.click(
    await screen.findByRole("button", { name: /Write a supplier reply/ })
  );
  await screen.findByRole("heading", { name: "Run details" });
}

it("shows one final prompt with additions highlighted when the original was kept", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    original_prompt: savedRun.prompt,
    final_prompt: `${savedRun.prompt}\n\nClarifications:\nKeep it under 120 words.`,
    original_kept: true,
  });

  expect(
    screen.getByText(
      "Your original prompt was kept. Highlighted text was added to it."
    )
  ).toBeVisible();
  expect(screen.getAllByRole("heading", { name: "Final prompt" })).toHaveLength(
    1
  );
  expect(
    screen.queryByRole("heading", { name: "Your prompt" })
  ).not.toBeInTheDocument();
  expect(screen.queryByText("Optimized prompt")).not.toBeInTheDocument();
  expect(document.querySelector("pre.history-text")).toHaveTextContent(
    `${savedRun.prompt} Clarifications: Keep it under 120 words.`
  );
  expect(
    document.querySelector("mark.history-prompt-change")
  ).toHaveTextContent("Clarifications: Keep it under 120 words.");
});

it("shows weak dimensions linked to a rejected winner in history", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    original_prompt: savedRun.prompt,
    final_prompt: "Write a concise reply.",
    feedback: "reject",
    feedback_labels: {
      status: "linked",
      weak_dimensions: ["clarity", "specificity"],
    },
  });

  expect(
    screen.getByText("Weak dimensions: clarity, specificity.")
  ).toBeVisible();
});

it("shows canonical unverified outcome, style, and reason in feedback context", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    outcome: "improved_unverified",
    outcome_reason: "Answer quality could not be tested.",
    applied_style: "clearer",
    original_prompt: savedRun.prompt,
    final_prompt: `${savedRun.prompt}\n\nClarifications:\nContext: my notes`,
    original_kept: false,
  });

  expect(
    screen.getAllByText("Improved (unverified)").length
  ).toBeGreaterThanOrEqual(2);
  expect(screen.getByText("Applied style: Clearer · — · —")).toBeVisible();
  expect(
    screen.getAllByText("Answer quality could not be tested.")
  ).toHaveLength(2);
  expect(
    screen.getByText(/Improved \(unverified\) · Clearer — Answer quality/)
  ).toBeVisible();
});

it("shows a converged unchanged original with its applied style and reason", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    outcome: "converged",
    applied_style: "clearer",
    outcome_reason: "The original already met every quality floor.",
    original_prompt: savedRun.prompt,
    final_prompt: savedRun.prompt,
    original_kept: true,
  });

  expect(screen.getAllByText("Converged")).toHaveLength(2);
  expect(screen.getByText("Applied style: Clearer · — · —")).toBeVisible();
  expect(
    screen.getAllByText("The original already met every quality floor.")
  ).toHaveLength(2);
  expect(screen.getAllByRole("heading", { name: "Your prompt" })).toHaveLength(
    1
  );
  expect(
    screen.queryByRole("heading", { name: "Final prompt" })
  ).not.toBeInTheDocument();
  expect(screen.getAllByText(savedRun.prompt)).toHaveLength(2);
});

it("highlights changed wording in a rewritten final prompt", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    original_prompt: savedRun.prompt,
    final_prompt: "Write a concise reply to the supplier.",
    original_kept: false,
  });

  expect(screen.getByRole("heading", { name: "Your prompt" })).toBeVisible();
  expect(screen.getByRole("heading", { name: "Final prompt" })).toBeVisible();
  expect(
    screen.getByText(
      "Highlighted text was added or changed in the final prompt."
    )
  ).toBeVisible();
  expect(
    document.querySelector("mark.history-prompt-change")
  ).toHaveTextContent("concise reply to the supplier");
});

it("distinguishes no matches from empty history and clears the applied search", async () => {
  const requestedUrls: string[] = [];
  let unfilteredRequestCount = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.includes("search=")) return jsonResponse([]);
      unfilteredRequestCount += 1;
      return jsonResponse(unfilteredRequestCount === 1 ? [] : [savedRun]);
    })
  );
  const user = userEvent.setup();

  render(<History />);

  await screen.findByText("No saved runs yet.");
  await user.type(screen.getByRole("searchbox"), "banana-not-a-supplier");
  await user.click(screen.getByRole("button", { name: "Search" }));

  expect(await screen.findByText("No runs match this search.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Clear search" })).toBeVisible();

  await user.click(screen.getByRole("button", { name: "Clear search" }));

  expect(await screen.findByText("Write a supplier reply.")).toBeVisible();
  expect(screen.getByRole("searchbox")).toHaveValue("");
  expect(requestedUrls).toContain("/api/runs?");
});

it("refreshes with the applied query while keeping unsubmitted edits separate", async () => {
  const requestedUrls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requestedUrls.push(url);
      return jsonResponse([savedRun]);
    })
  );
  const user = userEvent.setup();
  const { rerender } = render(<History refreshKey="initial" />);

  await screen.findByText("Write a supplier reply.");
  await user.type(screen.getByRole("searchbox"), "supplier");
  await user.click(screen.getByRole("button", { name: "Search" }));
  await screen.findByText("Write a supplier reply.");

  await user.clear(screen.getByRole("searchbox"));
  await user.type(screen.getByRole("searchbox"), "draft only");
  rerender(<History refreshKey="updated" />);

  await waitFor(() =>
    expect(requestedUrls.at(-1)).toBe("/api/runs?search=supplier")
  );
  expect(screen.getByRole("searchbox")).toHaveValue("draft only");
});

it("does not let an older search response replace a newer applied query", async () => {
  let resolveOldSearch!: (response: Response) => void;
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("search=older"))
        return new Promise<Response>((resolve) => {
          resolveOldSearch = resolve;
        });
      if (url.endsWith("search=newer"))
        return Promise.resolve(
          jsonResponse([
            { ...savedRun, run_id: "newer-run", prompt: "Newer result" },
          ])
        );
      return Promise.resolve(jsonResponse([]));
    })
  );
  const user = userEvent.setup();
  render(<History />);

  await screen.findByText("No saved runs yet.");
  const search = screen.getByRole("searchbox");
  await user.type(search, "older");
  await user.click(screen.getByRole("button", { name: "Search" }));
  await user.clear(search);
  await user.type(search, "newer");
  await user.click(screen.getByRole("button", { name: "Search" }));

  expect(await screen.findByText("Newer result")).toBeVisible();
  await act(async () => {
    resolveOldSearch(
      jsonResponse([
        { ...savedRun, run_id: "older-run", prompt: "Older result" },
      ])
    );
  });

  expect(screen.getByText("Newer result")).toBeVisible();
  expect(screen.queryByText("Older result")).not.toBeInTheDocument();
});

it("shows cancellation as a control state separate from the legacy outcome", async () => {
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
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/runs/cancelled-run"))
        return jsonResponse({
          ...cancelledRun,
          report: {
            status: "cancelled",
            failure: {
              kind: "cancelled",
              headline: "Run cancelled",
              hint: "You cancelled this run. Your prompt was not changed.",
              message: "cancelled by the user",
            },
          },
          result: {
            status: "failed",
            run_id: "cancelled-run",
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
            timing: { total_ms: 10 },
          },
        });
      return jsonResponse([cancelledRun, failedRun]);
    })
  );
  const user = userEvent.setup();
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  render(<History />);

  expect(await screen.findByText("Outcome not established")).toHaveClass(
    "badge-neutral"
  );
  expect(screen.getByText("Cancelled")).toHaveClass("badge-warn");
  expect(screen.getByText("Failed (operational)")).toHaveClass("badge-bad");

  await user.click(
    screen.getByRole("button", { name: /Keep the original prompt/ })
  );

  expect(
    await screen.findByRole("heading", { name: "Run details" })
  ).toBeVisible();
  expect(screen.getAllByText("Cancelled")).toHaveLength(2);
  expect(screen.getByRole("heading", { name: "Run cancelled" })).toBeVisible();
});

/** The five canonical outcome labels and a deliberately legacy row. */
const labelCases: { label: BadgeLabel; run: RunSummary }[] = [
  {
    label: "Converged",
    run: { run_id: "cv", prompt: "Case converged.", outcome: "converged" },
  },
  {
    label: "Improved (tested)",
    run: {
      run_id: "it",
      prompt: "Case tested.",
      outcome: "improved_tested",
    },
  },
  {
    label: "Improved (unverified)",
    run: {
      run_id: "iu",
      prompt: "Case unverified.",
      outcome: "improved_unverified",
    },
  },
  {
    label: "Impossible",
    run: { run_id: "im", prompt: "Case impossible.", outcome: "impossible" },
  },
  {
    label: "Failed (operational)",
    run: {
      run_id: "fo",
      prompt: "Case failed.",
      outcome: "failed_operational",
    },
  },
  {
    label: "Outcome not established",
    run: {
      run_id: "paused",
      prompt: "Case paused before an outcome.",
      control_state: "awaiting_approval",
    },
  },
  {
    label: "Legacy run",
    run: {
      run_id: "legacy",
      prompt: "Case legacy.",
      status: "completed",
      legacy_metadata: { source_schema: "old" },
    },
  },
];

it("explains all badgeFor labels in words, not by colour", () => {
  expect(Object.keys(BADGE_EXPLANATIONS).sort()).toEqual(
    labelCases.map((entry) => entry.label).sort()
  );

  for (const { label, run } of labelCases) {
    // The table is keyed by the label `badgeFor` actually returns.
    expect(badgeFor(run).label).toBe(label);

    const explanation = BADGE_EXPLANATIONS[label];
    // A sentence, so the meaning is readable without seeing the pill colour.
    expect(explanation).toMatch(/^[A-Z].*[.]$/);
    expect(explanation.split(" ").length).toBeGreaterThan(6);
  }
});

it("shows stop control state separately from the outcome pill", () => {
  const run = {
    run_id: "stopped",
    prompt: "Case stopped after an accepted prompt.",
    outcome: "improved_tested",
    control_state: "stopped",
  } as RunSummary;
  expect(badgeFor(run).label).toBe("Improved (tested)");
  expect(controlLabelFor(run)).toBe("Stopped");
});

it("gives every list pill a title for the pointer and a description for the keyboard", async () => {
  const cancelled: RunSummary = {
    run_id: "c",
    prompt: "Case cancelled.",
    outcome: "cancelled",
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return url.includes("/api/runs/")
        ? jsonResponse(cancelled)
        : jsonResponse(labelCases.map((entry) => entry.run));
    })
  );
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  render(<History />);

  for (const { label, run } of labelCases) {
    const badge = await screen.findByText(label, { selector: ".badge" });
    // Pointer: the native tooltip carries the explanation.
    expect(badge).toHaveAttribute("title", BADGE_EXPLANATIONS[label]);

    // Keyboard / assistive tech: the focused row button describes itself.
    const row = screen.getByRole("button", {
      name: new RegExp(escapeRegExp(run.prompt)),
    });
    expect(row).toHaveAccessibleDescription(BADGE_EXPLANATIONS[label]);

    const describedBy = row.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      BADGE_EXPLANATIONS[label]
    );
  }
});

it("explains the pill in the opened run details to keyboard and pointer", async () => {
  const failure = {
    kind: "provider",
    headline: "The model refused the request",
    hint: "Try a different model.",
    message: "provider error",
  };
  const failedRun: RunDetail = {
    ...savedRun,
    run_id: "failed-details",
    status: "failed",
    outcome: "failed_operational",
    report: { status: "failed_operational", failure },
    result: {
      status: "failed",
      run_id: "failed-details",
      original_prompt: savedRun.prompt,
      report: { status: "failed_operational", failure },
      cost: { total: 0 },
      timing: { total_ms: 10 },
    } as OptimizeResult,
  };
  await openRunDetails(failedRun);

  const details = screen.getByRole("heading", { name: "Run details" })
    .parentElement as HTMLElement;
  const pill = details.querySelector(".badge") as HTMLElement;

  expect(pill).toHaveTextContent("Failed (operational)");
  expect(pill).toHaveAttribute(
    "title",
    BADGE_EXPLANATIONS["Failed (operational)"]
  );
  expect(pill).toHaveAccessibleDescription(
    BADGE_EXPLANATIONS["Failed (operational)"]
  );
  // Reachable from the keyboard (it is in the tab order), not only by pointer.
  expect(pill).toHaveAttribute("tabindex", "0");
  pill.focus();
  expect(document.activeElement).toBe(pill);
  // The details body still renders the failure card it rendered before.
  expect(screen.getByRole("heading", { name: "Run details" })).toBeVisible();
  expect(pill.closest("article")).not.toBeNull();
});

it("shows a Failed row's stored reason without opening the row", async () => {
  const failedRun: RunSummary = {
    run_id: "failed-with-result",
    prompt: "Write a useful reply.",
    status: "failed",
    result: {
      status: "failed",
      run_id: "failed-with-result",
      report: {
        status: "failed",
        failure: {
          kind: "provider",
          headline: "The model refused the request",
          hint: "Try a different model.",
          message: "provider error",
        },
      },
      cost: { total: 0 },
      timing: { total_ms: 10 },
    },
  } as RunSummary;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse([failedRun]))
  );
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  render(<History />);

  const row = await screen.findByRole("button", {
    name: /Write a useful reply/,
  });
  // Visible in the row, with no click and no opened details.
  expect(row).toHaveTextContent("Why it failed:");
  expect(row).toHaveTextContent("The model refused the request");
  expect(
    screen.queryByRole("heading", { name: "Run details" })
  ).not.toBeInTheDocument();
});

it("points a Failed row with no stored reason at where the reason appears", async () => {
  const failedRun: RunSummary = {
    run_id: "failed-without-result",
    prompt: "Write a useful reply.",
    status: "failed",
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse([failedRun]))
  );
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  render(<History />);

  const row = await screen.findByRole("button", {
    name: /Write a useful reply/,
  });
  expect(row).toHaveTextContent("Why it failed:");
  expect(row).toHaveTextContent("Open this row to see why it failed.");
  expect(
    screen.queryByRole("heading", { name: "Run details" })
  ).not.toBeInTheDocument();
});

it("keeps row content, ordering, and the Open action unchanged", async () => {
  const firstRun: RunSummary = {
    run_id: "first",
    prompt: "First prompt.",
    status: "failed",
  };
  const secondRun: RunSummary = {
    run_id: "second",
    prompt: "Second prompt.",
    original_kept: false,
  };
  const thirdRun: RunSummary = {
    run_id: "third",
    prompt: "Third prompt.",
    outcome: "cancelled",
  };
  const runs: RunSummary[] = [firstRun, secondRun, thirdRun];
  const resumed: RunSummary & { result: OptimizeResult } = {
    ...thirdRun,
    status: "needs_input",
    result: {
      status: "needs_input",
      run_id: "third",
      original_prompt: "Third prompt.",
      report: {},
      questions: [],
      cost: { total: 0 },
      timing: { total_ms: 1 },
    },
  } as RunSummary & { result: OptimizeResult };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return url.endsWith("/api/runs/third")
        ? jsonResponse(resumed)
        : jsonResponse(runs);
    })
  );
  const user = userEvent.setup();
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    value: vi.fn(),
    configurable: true,
  });

  const onOpen = vi.fn();
  render(<History onOpen={onOpen} />);

  await screen.findByText("First prompt.");
  const listRows = Array.from(
    document.querySelectorAll("button.history-row")
  ) as HTMLElement[];
  expect(listRows).toHaveLength(3);
  // Server order is preserved, and each row still leads with its pill.
  expect(listRows.map((row) => row.textContent)).toEqual([
    expect.stringContaining("First prompt."),
    expect.stringContaining("Second prompt."),
    expect.stringContaining("Third prompt."),
  ]);
  for (const row of listRows)
    expect(row.querySelector(".badge")).not.toBeNull();

  await user.click(screen.getByRole("button", { name: /Third prompt/ }));
  const answer = await screen.findByRole("button", {
    name: "Answer the questions",
  });
  expect(answer).toBeVisible();
  await user.click(answer);
  expect(onOpen).toHaveBeenCalledWith(resumed.result);
});
