/**
 * A worker that lost its GPU used to read exactly like a healthy one.
 *
 * helena-ink-0's container lost its GPU passthrough silently -- nvidia-smi
 * inside it started answering "No devices were found", not a crash -- while
 * the worker process kept polling and the fleet row kept saying POLLING for
 * five hours. `state` alone cannot carry that distinction: a worker with
 * nothing to claim and a worker that cannot claim GPU work both poll on
 * schedule. What is worth testing here is that the Workers table draws the
 * two apart, in the component's own visual language (Pill), rather than
 * leaving a lost card silent the way it used to.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Fleet from "./Fleet";
import type { Fleet as FleetState } from "../api";

const BASE: FleetState = {
  available: true,
  tasks: 10,
  attempts: 12,
  surfaces: 3,
  events: 40,
  leased: 1,
  stale_leases: 0,
  task_states: [],
  events_by_type: [],
  workers: [],
};

let reply: FleetState = BASE;

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => reply,
  })));
});

afterEach(() => {
  vi.unstubAllGlobals();
  reply = BASE;
});

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Fleet />
    </QueryClientProvider>,
  );
}

describe("the Workers table", () => {
  it("marks a GPU worker whose card is gone distinctly from a healthy one", async () => {
    reply = {
      ...BASE,
      workers: [
        {
          worker_id: "helena-ink-0", host_id: "gpu-1", runtime: "helena-worker-gpu",
          phases: ["P4", "P5"], last_poll_at: new Date().toISOString(),
          last_claim_at: new Date().toISOString(), seconds_since_poll: 3,
          state: "POLLING", gpu_visible: false,
        },
        {
          worker_id: "helena-ink-9um", host_id: "gpu-1", runtime: "helena-ink-9um",
          phases: ["P5"], last_poll_at: new Date().toISOString(),
          last_claim_at: new Date().toISOString(), seconds_since_poll: 2,
          state: "POLLING", gpu_visible: true,
        },
      ],
    };
    draw();

    await screen.findByText("helena-ink-0");
    // Both rows read POLLING -- that is the whole point of the bug: liveness
    // alone cannot say a card is gone.
    expect(screen.getAllByText("POLLING")).toHaveLength(2);

    const noGpu = screen.getByText("no GPU");
    expect(noGpu.className).toContain("crit");
    const hasGpu = screen.getByText("GPU");
    expect(hasGpu.className).toContain("ok");
  });

  it("draws no GPU opinion for a worker that has never claimed one", async () => {
    reply = {
      ...BASE,
      workers: [{
        worker_id: "cpu-runner", host_id: "cpu-1", runtime: "helena-fleet-runner",
        phases: ["P8"], last_poll_at: new Date().toISOString(),
        last_claim_at: null, seconds_since_poll: 5,
        state: "POLLING", gpu_visible: null,
      }],
    };
    draw();

    await screen.findByText("cpu-runner");
    expect(screen.queryByText("no GPU")).toBeNull();
    expect(screen.queryByText("GPU")).toBeNull();
  });

  it("still marks a worker SILENT when it has stopped polling", async () => {
    reply = {
      ...BASE,
      workers: [{
        worker_id: "helena-ink-1", host_id: "gpu-1", runtime: "helena-worker-gpu",
        phases: ["P5"], last_poll_at: new Date(0).toISOString(),
        last_claim_at: new Date(0).toISOString(), seconds_since_poll: 64800,
        state: "SILENT", gpu_visible: true,
      }],
    };
    draw();

    const silent = await screen.findByText("SILENT");
    expect(silent.className).toContain("warn");
  });
});
