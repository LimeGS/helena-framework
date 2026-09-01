import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Coverage from "./Coverage";

/**
 * The live narration of a control run, on the page an operator already has
 * open -- not a terminal they need to have kept a window on for the hour a
 * boundary can take.
 */

const READINESS_BASE = {
  schema: "campaignx.first_letters_readiness.v1",
  mission_id: "first-letters", controlled: true, reason: null,
  deployed_revision: "a".repeat(40), mission_deployed_revision: "a".repeat(40),
  control: {
    available: true, evidence_status: "CURRENT", evidence_status_reason: "",
    control_state: "CONTROL_PASS", control_id: "PHerc0139-w025-public-positive-v1",
    content_sha256: "b".repeat(64), first_nonpassing_boundary: null,
    bound_deployed_revision: "a".repeat(40), stages: [],
  },
  scrolls: [], pause: null, queue: null, small_surfaces: { available: false },
  blockers: [], advisories: [],
  allowed_actions: [], queue_admitted: false, non_claims: [],
  readiness_sha256: "c".repeat(64),
};

type FixtureEvent = {
  schema: string; run_id: string; mission_id: string;
  event: string; boundary?: string; state?: string; reason?: string;
  control_state?: string; message: string; at_utc: string; received_at_utc: string;
};

function event(overrides: Partial<FixtureEvent> & { event: string; message: string }):
    FixtureEvent {
  return {
    schema: "campaignx.first_letters_control_progress_event.v1",
    run_id: "run-a", mission_id: "first-letters",
    at_utc: "2026-08-19T12:00:00+00:00", received_at_utc: "2026-08-19T12:00:00+00:00",
    ...overrides,
  };
}

const IN_FLIGHT_EVENTS = [
  event({ event: "run_started", message: "control run starting: mission=first-letters revision=aaaaaaaaaaaa run_id=run-a" }),
  event({ event: "boundary_started", boundary: "P0", message: "P0 starting" }),
  event({ event: "boundary_finished", boundary: "P0", state: "PASS", reason: "EXACT_RUNTIME_BINDING",
          message: "P0 -> PASS (EXACT_RUNTIME_BINDING) after 0s" }),
  event({ event: "boundary_started", boundary: "P1", message: "P1 starting" }),
  event({ event: "heartbeat", boundary: "P1", message: "P1 still waiting for the grow to finish (930s)" }),
];

const FINISHED_EVENTS = [
  ...IN_FLIGHT_EVENTS,
  event({ event: "boundary_finished", boundary: "P1", state: "PASS",
          reason: "DISCOVERY_AND_MANUAL_GROW_SURVIVED",
          message: "P1 -> PASS (DISCOVERY_AND_MANUAL_GROW_SURVIVED) after 3982s" }),
  event({ event: "run_finished", control_state: "CONTROL_INCOMPLETE",
          message: "control run finished: CONTROL_INCOMPLETE (first non-passing boundary: QC)" }),
];

let progressEvents = IN_FLIGHT_EVENTS;
let calls: { url: string }[] = [];

beforeEach(() => {
  progressEvents = IN_FLIGHT_EVENTS;
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (input: any) => {
    const url = String(input);
    calls.push({ url });
    return {
      ok: true, status: 200,
      json: async () => {
        if (url.startsWith("/api/coverage")) {
          return { available: true, grids: [], volumes: [], non_claims: [] };
        }
        if (url.includes("first-letters-readiness")) return READINESS_BASE;
        if (url.includes("first-letters-control/progress")) {
          return {
            run_id: "run-a", events: progressEvents,
            runs: [{ run_id: "run-a", started_at_utc: progressEvents[0].at_utc,
                     last_event_at_utc: progressEvents.at(-1)!.at_utc,
                     last_event: progressEvents.at(-1)!.event,
                     current_boundary: "P1",
                     finished: progressEvents.some((e: any) => e.event === "run_finished"),
                     control_state: progressEvents.find((e: any) => e.event === "run_finished")
                       ?.control_state ?? null,
                     event_count: progressEvents.length }],
          };
        }
        return { available: true, by_cause: {}, attempts: 0, note: "" };
      },
    };
  }));
});
afterEach(() => vi.unstubAllGlobals());

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <Coverage sample="PHerc0139" mission="first-letters" />
  </QueryClientProvider>);
}

describe("control progress", () => {
  it("shows the boundary that is currently running", async () => {
    draw();
    expect(await screen.findByText(/running · P1/)).toBeTruthy();
    expect(screen.getByText(/run run-a/)).toBeTruthy();
  });

  it("marks a finished boundary and leaves the rest pending", async () => {
    draw();
    await screen.findByText(/running · P1/);
    const p0 = screen.getByText("P0 · PASS");
    expect(p0).toBeTruthy();
    // P7 has not started: no status suffix.
    expect(screen.getByText("P7")).toBeTruthy();
  });

  it("switches to a finished-run display once run_finished has posted", async () => {
    progressEvents = FINISHED_EVENTS;
    draw();
    expect(await screen.findByText(/finished · CONTROL_INCOMPLETE/)).toBeTruthy();
    expect(screen.getByText("took", { exact: false })).toBeTruthy();
  });

  it("reveals the full log only after the toggle is clicked", async () => {
    draw();
    await screen.findByText(/running · P1/);
    expect(screen.queryByText(/still waiting for the grow to finish/)).toBeNull();
    fireEvent.click(screen.getByText(/show the full log/));
    expect(await screen.findByText(
      /still waiting for the grow to finish/)).toBeTruthy();
    expect(screen.getByText(/EXACT_RUNTIME_BINDING/)).toBeTruthy();
  });

  it("stops polling once the run has finished", async () => {
    progressEvents = FINISHED_EVENTS;
    draw();
    await screen.findByText(/finished · CONTROL_INCOMPLETE/);
    const before = calls.filter((c) => c.url.includes("control/progress")).length;
    await new Promise((resolve) => setTimeout(resolve, 50));
    const after = calls.filter((c) => c.url.includes("control/progress")).length;
    expect(after).toBe(before);
  });

  it("shows which boundary an unfinished earlier run stopped at", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: any) => {
      const url = String(input);
      return {
        ok: true, status: 200,
        json: async () => {
          if (url.startsWith("/api/coverage")) {
            return { available: true, grids: [], volumes: [], non_claims: [] };
          }
          if (url.includes("first-letters-readiness")) return READINESS_BASE;
          if (url.includes("first-letters-control/progress")) {
            return {
              run_id: "run-b", events: FINISHED_EVENTS,
              runs: [
                { run_id: "run-b", started_at_utc: "2026-08-19T13:00:00+00:00",
                  last_event_at_utc: "2026-08-19T13:10:00+00:00",
                  last_event: "run_finished", current_boundary: "QC",
                  finished: true, control_state: "CONTROL_INCOMPLETE", event_count: 7 },
                { run_id: "run-a", started_at_utc: "2026-08-19T12:00:00+00:00",
                  last_event_at_utc: "2026-08-19T12:05:00+00:00",
                  last_event: "heartbeat", current_boundary: "P4",
                  finished: false, control_state: null, event_count: 5 },
              ],
            };
          }
          return { available: true, by_cause: {}, attempts: 0, note: "" };
        },
      };
    }));
    draw();
    expect(await screen.findByText(/finished · CONTROL_INCOMPLETE/)).toBeTruthy();
    expect(screen.getByText(/run-a \(in progress · P4\)/)).toBeTruthy();
  });

  it("draws no progress card at all when the mission has never posted one", async () => {
    progressEvents = [];
    vi.stubGlobal("fetch", vi.fn(async (input: any) => {
      const url = String(input);
      return {
        ok: true, status: 200,
        json: async () => {
          if (url.startsWith("/api/coverage")) {
            return { available: true, grids: [], volumes: [], non_claims: [] };
          }
          if (url.includes("first-letters-readiness")) return READINESS_BASE;
          if (url.includes("first-letters-control/progress")) {
            return { run_id: null, events: [], runs: [] };
          }
          return { available: true, by_cause: {}, attempts: 0, note: "" };
        },
      };
    }));
    draw();
    await screen.findByText("First Letters campaign gates");
    expect(screen.queryByText(/show the full log/)).toBeNull();
  });
});
