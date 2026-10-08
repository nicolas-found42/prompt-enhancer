import type { Job } from "./api";

export function mergeJobEvents(previous: Job, latest: Job): Job {
  if (previous.run_id !== latest.run_id) return latest;
  const events = new Map(
    [...(previous.events ?? []), ...(latest.events ?? [])].map((event) => [
      event.cursor,
      event,
    ])
  );
  return {
    ...latest,
    events: [...events.values()].sort((a, b) => a.cursor - b.cursor),
  };
}
