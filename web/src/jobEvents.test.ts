import { describe, expect, it } from "vitest";
import { mergeJobEvents } from "./jobEvents";
import type { Job } from "./api";

describe("incremental activity", () => {
  it("retains earlier events, deduplicates overlap and sorts reconnect chunks", () => {
    const event = (cursor: number) => ({
      cursor,
      kind: "started",
      summary: "Writing",
      elapsed_ms: cursor * 1000,
      round: 1,
    });
    const previous: Job = {
      run_id: "run",
      kind: "optimize",
      state: "running",
      stage: "writing_candidates",
      round: { round: 1 },
      stages_seen: [],
      elapsed_ms: 3000,
      cancel_requested: false,
      result: null,
      event_cursor: 3,
      events: [event(1), event(2), event(3)],
    };
    const latest = {
      ...previous,
      events: [event(5), event(3), event(4)],
      event_cursor: 5,
    };
    const merged = mergeJobEvents(previous, latest);
    expect(merged.events?.map((item) => item.cursor)).toEqual([1, 2, 3, 4, 5]);
    expect(merged.event_cursor).toBe(5);
    expect(mergeJobEvents(merged, { ...latest, events: [] }).events).toEqual(
      merged.events
    );
    expect(previous.events).toHaveLength(3);
  });
});
