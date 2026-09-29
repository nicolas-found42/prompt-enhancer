import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import History, { type RunDetail, type RunSummary } from "./History";

const savedRun: RunSummary = {
  run_id: "saved-run",
  prompt: "Write a supplier reply.",
};

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

it("does not label an untested run with confirmed details as improved", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    outcome: "unverified",
    original_prompt: savedRun.prompt,
    final_prompt: `${savedRun.prompt}\n\nClarifications:\nContext: my notes`,
    original_kept: false,
    report: { status: "unverified" },
  });

  expect(screen.queryByText("Improved")).not.toBeInTheDocument();
  expect(screen.getAllByText("Not tested").length).toBeGreaterThanOrEqual(2);
});

it("states that an unchanged original was kept and shows it once", async () => {
  await openRunDetails({
    ...savedRun,
    status: "completed",
    original_prompt: savedRun.prompt,
    final_prompt: savedRun.prompt,
    original_kept: true,
  });

  expect(screen.getByText("Your prompt was kept as-is.")).toBeVisible();
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

it("shows cancelled runs with neutral status in the list and opened details", async () => {
  const cancelledRun = {
    run_id: "cancelled-run",
    status: "failed",
    outcome: "cancelled",
    prompt: "Keep the original prompt.",
  };
  const failedRun = {
    run_id: "failed-run",
    status: "failed",
    outcome: "failed",
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

  const cancelledBadge = await screen.findByText("Cancelled");
  expect(cancelledBadge).toHaveClass("badge-neutral");
  expect(screen.getByText("Failed")).toHaveClass("badge-bad");

  await user.click(
    screen.getByRole("button", { name: /Keep the original prompt/ })
  );

  expect(
    await screen.findByRole("heading", { name: "Run details" })
  ).toBeVisible();
  const cancelledBadges = screen.getAllByText("Cancelled");
  expect(cancelledBadges).toHaveLength(2);
  for (const badge of cancelledBadges)
    expect(badge).toHaveClass("badge-neutral");
  expect(screen.getByRole("heading", { name: "Run cancelled" })).toBeVisible();
});
