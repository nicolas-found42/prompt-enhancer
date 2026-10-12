import { afterEach, expect, it, vi } from "vitest";
import { ApiError, requestJson } from "./api";

afterEach(() => vi.unstubAllGlobals());

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

it("shares concurrent run reads, keeps caller values independent, and reads fresh afterward", async () => {
  const response = deferred<Response>();
  const fetcher = vi
    .fn()
    .mockReturnValueOnce(response.promise)
    .mockResolvedValue(
      new Response(JSON.stringify({ result: { prompt: "updated" } }))
    );
  vi.stubGlobal("fetch", fetcher);
  const first = requestJson<{ result: { prompt: string } }>(
    "/api/runs/shared-read"
  );
  const second = requestJson<{ result: { prompt: string } }>(
    "/api/runs/shared-read"
  );
  expect(fetcher).toHaveBeenCalledTimes(1);
  response.resolve(
    new Response(JSON.stringify({ result: { prompt: "original" } }))
  );
  const [a, b] = await Promise.all([first, second]);
  a.result.prompt = "local edit";
  expect(b.result.prompt).toBe("original");
  expect(await requestJson("/api/runs/shared-read")).toEqual({
    result: { prompt: "updated" },
  });
  expect(fetcher).toHaveBeenCalledTimes(2);
});

it("clears a failed shared read so a later attempt can recover", async () => {
  const response = deferred<Response>();
  const fetcher = vi
    .fn()
    .mockReturnValueOnce(response.promise)
    .mockResolvedValue(new Response(JSON.stringify({ result: "recovered" })));
  vi.stubGlobal("fetch", fetcher);
  const reads = Promise.allSettled([
    requestJson("/api/runs/failed-read"),
    requestJson("/api/runs/failed-read"),
  ]);
  response.resolve(
    new Response(JSON.stringify({ detail: "not ready" }), { status: 503 })
  );
  const settled = await reads;
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(
    settled.every(
      (r) =>
        r.status === "rejected" &&
        r.reason instanceof ApiError &&
        r.reason.status === 503
    )
  ).toBe(true);
  expect(await requestJson("/api/runs/failed-read")).toEqual({
    result: "recovered",
  });
});

it("invalidates pending reads on mutation without an older completion deleting a newer read", async () => {
  const older = deferred<Response>();
  const newer = deferred<Response>();
  const fetcher = vi
    .fn()
    .mockReturnValueOnce(older.promise)
    .mockResolvedValueOnce(new Response("{}"))
    .mockReturnValueOnce(newer.promise);
  vi.stubGlobal("fetch", fetcher);
  const old = requestJson("/api/runs/mutated-read");
  await requestJson("/api/runs/mutated-read/feedback", { method: "POST" });
  const current = requestJson("/api/runs/mutated-read");
  older.resolve(new Response(JSON.stringify({ version: "old" })));
  await old;
  const peer = requestJson("/api/runs/mutated-read");
  expect(fetcher).toHaveBeenCalledTimes(3);
  newer.resolve(new Response(JSON.stringify({ version: "new" })));
  expect(await current).toEqual({ version: "new" });
  expect(await peer).toEqual({ version: "new" });
});

it("keeps distinct URLs and caller cancellation requests separate", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(async () => new Response("{}"))
  );
  const signal = new AbortController().signal;
  await Promise.all([
    requestJson("/api/runs/different-a"),
    requestJson("/api/runs/different-b"),
    requestJson("/api/runs/different-a", { signal }),
    requestJson("/api/runs/different-a?view=summary"),
  ]);
  expect(fetch).toHaveBeenCalledTimes(4);
  expect(fetch).toHaveBeenCalledWith("/api/runs/different-a", { signal });
});

it("keeps a pending run read shared while an unrelated health assessment posts", async () => {
  const response = deferred<Response>();
  const fetcher = vi
    .fn()
    .mockReturnValueOnce(response.promise)
    .mockResolvedValueOnce(new Response("{}"));
  vi.stubGlobal("fetch", fetcher);
  const first = requestJson("/api/runs/health-independent");
  await requestJson("/api/prompt-health", { method: "POST" });
  const second = requestJson("/api/runs/health-independent");
  expect(fetcher).toHaveBeenCalledTimes(2);
  response.resolve(new Response(JSON.stringify({ result: "same run" })));
  expect(await first).toEqual({ result: "same run" });
  expect(await second).toEqual({ result: "same run" });
});
