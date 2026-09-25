import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import History, { type RunSummary } from "./History";

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

afterEach(() => vi.unstubAllGlobals());

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
