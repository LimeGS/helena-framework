/**
 * The page people actually open, and whether it says a GPU worker is blind.
 *
 * Fleet.tsx's own Workers table is not wrong, but nobody sees it: it is not
 * mounted on any route (docs/handbook/30-panel/01-fleet.md says so plainly,
 * and App.tsx's own comment -- "queueing belongs inside the phase" -- is why).
 * helena-ink-0 lost its GPU passthrough and polled normally for five hours;
 * the one place a human would have noticed is Mission, so this is where the
 * distinction has to actually render.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Mission from "./Mission";

const STATE = {
  generated_at: "2026-09-03T00:00:00Z",
  fleet: { available: true, tasks: 10, surfaces: 3, stale_leases: 0 },
  integrity: [],
  targets: [],
  run_count: 4,
  lane_count: 2,
};

const HOSTS = { hosts: [] };

let fleetReply: Record<string, unknown> = { available: true, workers: [] };

function reply(path: string) {
  if (path.startsWith("/api/state")) return STATE;
  if (path.startsWith("/api/hosts")) return HOSTS;
  if (path.startsWith("/api/fleet")) return fleetReply;
  return {};
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (input: any) => ({
    ok: true,
    status: 200,
    json: async () => reply(String(input)),
  })));
});

afterEach(() => {
  vi.unstubAllGlobals();
  fleetReply = { available: true, workers: [] };
});

vi.mock("../mission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../mission")>()),
  useMission: () => ({ missionId: "test" }),
  useSubject: () => ({ subject: null }),
}));

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Mission />
    </QueryClientProvider>,
  );
}

describe("the Workers tile", () => {
  it("stays steady when every worker is polling and can see its GPU", async () => {
    fleetReply = {
      available: true,
      workers: [
        { worker_id: "helena-ink-9um", host_id: "gpu-1", runtime: "helena-ink-9um",
          phases: ["P5"], last_poll_at: new Date().toISOString(),
          last_claim_at: new Date().toISOString(), seconds_since_poll: 2,
          state: "POLLING", gpu_visible: true },
      ],
    };
    draw();

    // Not just the tile title -- "Workers" is also what the tile reads while
    // its own query is still loading, so waiting on the title alone would
    // pass before the fleet data (and the tone it decides) ever arrived.
    await screen.findByText(/polling/);
    expect(screen.queryByText(/blind to their GPU/)).toBeNull();
    expect(screen.queryByText(/silent/)).toBeNull();
  });

  it("goes to alert and names the worker when a GPU worker cannot see its card", async () => {
    fleetReply = {
      available: true,
      workers: [
        { worker_id: "helena-ink-0", host_id: "gpu-1", runtime: "helena-worker-gpu",
          phases: ["P4", "P5"], last_poll_at: new Date().toISOString(),
          last_claim_at: new Date().toISOString(), seconds_since_poll: 3,
          // POLLING, not SILENT -- the whole point: liveness alone hid this.
          state: "POLLING", gpu_visible: false },
        { worker_id: "helena-ink-9um", host_id: "gpu-1", runtime: "helena-ink-9um",
          phases: ["P5"], last_poll_at: new Date().toISOString(),
          last_claim_at: new Date().toISOString(), seconds_since_poll: 2,
          state: "POLLING", gpu_visible: true },
      ],
    };
    draw();

    await screen.findByText(/blind to their GPU/);
    expect(screen.getByText("helena-ink-0")).toBeDefined();
    const tile = screen.getByText("Workers").closest(".tile");
    expect(tile?.className).toContain("alert");
  });

  it("does not blame a worker that has never claimed a GPU at all", async () => {
    fleetReply = {
      available: true,
      workers: [
        { worker_id: "cpu-runner", host_id: "cpu-1", runtime: "helena-fleet-runner",
          phases: ["P8"], last_poll_at: new Date().toISOString(),
          last_claim_at: null, seconds_since_poll: 5,
          state: "POLLING", gpu_visible: null },
      ],
    };
    draw();

    await screen.findByText(/polling/);
    expect(screen.queryByText(/blind to their GPU/)).toBeNull();
    const tile = screen.getByText("Workers").closest(".tile");
    expect(tile?.className).toContain("steady");
  });
});
