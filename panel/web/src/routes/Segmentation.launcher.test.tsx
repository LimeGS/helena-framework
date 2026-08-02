/**
 * What the launcher actually sends.
 *
 * Three separate audits asked for this file, and the reason is in the defects
 * they found: a grid step typed into the form went into a dict the API skipped, a
 * backend was echoed back and never dispatched, a reason said it was kept and was
 * kept nowhere, and manual seeds sent three fields out of a form with a dozen.
 * Every one of those is invisible from the server side and invisible from the
 * page -- the control is there, it accepts input, and the request body is missing
 * a key. Only a test that reads the body catches them.
 *
 * So these render the real component and assert on the fetch it makes. No
 * screenshots and no snapshots: what matters is the payload.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// A mission and a subject, mocked rather than provided: the page's whole job here
// is the request body, and MissionProvider would pull in its own fetches and a
// gate that renders a chooser instead of the form.
vi.mock("../mission", async () => {
  const actual = await vi.importActual<typeof import("../mission")>("../mission");
  return {
    ...actual,
    useMission: () => ({ missionId: "test", setMissionId: () => {} }),
    useSubject: () => ({ subject: "PHerc0826", setSubject: () => {} }),
  };
});

import Segmentation from "./Segmentation";

const STATE = {
  public: { total: 0, by_sample: {}, origin: "test", for_sample: 0 },
  private: {
    total: 2, area_cm2: 12.5, imported: 1, certified: 1,
    certified_area_cm2: 6.25, ct_supported: 0, ct_supported_area_cm2: 0,
    by_sample: [], for_sample: null,
  },
  queue: { tasks: 0, attempts: 0, leased: 0, stale_leases: 0, by_state: {}, scope: "PHerc0826" },
  backends: [
    { id: "vc3d", name: "VC3D seeded grow", adoptable: true, note: "the production one" },
  ],
  // The default is deliberately NOT first. With it first, `planners[0]` and "the
  // one marked default" give the same answer and the test cannot tell them apart --
  // which is how the real list was shaped when the form started disagreeing with
  // the stage contract.
  planners: [
    { id: "deterministic", name: "Deterministic (history blind)", kind: "deterministic",
      repeatable: true, note: "v1", configures: [] },
    { id: "cost-aware-v2", name: "Cost-aware router", kind: "router",
      repeatable: false, note: "the fleet's default", configures: [], default: true },
  ],
  reads: [
    { artifact_id: "P0/PHerc0826/aaaa", phase: "P0", sample_id: "PHerc0826",
      selected: true, kind: "volume", registered_at_utc: "2026-07-01T00:00:00Z" },
  ],
  runs: { available: true, runs: [] },
  segments: { available: true, segments: [] },
};

const OPTIONS = {
  options: [
    { flag: "--grid-step", field: "grid_step", kind: "int", label: "Grid step",
      group: "where to look", note: "cell spacing in voxels" },
  ],
  probe: {
    modes: [
      { id: "off", name: "Off", note: "direct" },
      { id: "shadow", name: "Shadow", note: "record, do not steer" },
      { id: "select", name: "Select", note: "select or abstain" },
    ],
    default_mode: "off",
    top_k: { minimum: 1, maximum: 3, default: 2 },
    generations: { minimum: 10, maximum: 20, default: 12 },
    select_readiness: {
      available: true,
      rollout_enabled: true,
      benchmark_approved: true,
      benchmark_scope_allows: true,
      benchmark_id: "seed-probe-test-v1",
      decision_receipt_sha256: "a".repeat(64),
      source_locked: true,
      review_owner_declared: true,
      reason: null,
    },
    note: "beneath Cost-aware", caveat: "not proof of the correct lamina",
  },
  growth: { note: "", parameters: [] },
};

/** Every request the component made, newest last. */
let sent: { url: string; body: unknown }[] = [];

function reply(url: string): unknown {
  if (url.startsWith("/api/segmentation/options")) return OPTIONS;
  if (url.startsWith("/api/segmentation")) return STATE;
  return {};
}

beforeEach(() => {
  sent = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === "POST") {
      sent.push({ url, body: JSON.parse(String(init.body)) });
      return new Response(JSON.stringify({ queued_for: "PHerc0826" }),
                          { status: 201, headers: { "Content-Type": "application/json" } });
    }
    return new Response(JSON.stringify(reply(url)),
                        { status: 200, headers: { "Content-Type": "application/json" } });
  }));
});

afterEach(() => vi.unstubAllGlobals());

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <Segmentation job="new" onSwitch={() => {}} />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe("the launcher's request body", () => {
  it("carries the mission, so the run can record which P0 selection it read", async () => {
    mount();
    const start = await screen.findByRole("button", { name: /new run|queue|start/i });
    fireEvent.click(start);

    await waitFor(() => expect(sent.length).toBeGreaterThan(0));
    const run = sent.find((r) => r.url.includes("/segmentation/runs"));
    expect(run, "no request reached /api/segmentation/runs").toBeTruthy();
    const body = run!.body as Record<string, unknown>;

    // The mission and nothing more about the selection: the panel resolves the
    // artifact itself, and a browser naming its own version would be a per-run
    // override nothing else in the control plane knows about.
    expect(body).toHaveProperty("mission_id");
    expect(body).not.toHaveProperty("p0_selection_version");
    expect(body).not.toHaveProperty("p0_artifact_id");
  });

  it("sends the planner the form is showing, not the host's default", async () => {
    mount();
    const start = await screen.findByRole("button", { name: /new run|queue|start/i });
    fireEvent.click(start);
    await waitFor(() => expect(sent.length).toBeGreaterThan(0));

    const body = sent.find((r) => r.url.includes("/segmentation/runs"))!
      .body as Record<string, unknown>;
    // The fleet's declared default, which is what the form must open on -- taking
    // the first row of a list made the browser disagree with both the API and the
    // stage contract.
    expect(body.planner).toBe("cost-aware-v2");
  });

  it("puts the grid step where the API reads it", async () => {
    mount();
    const start = await screen.findByRole("button", { name: /new run|queue|start/i });
    fireEvent.click(start);
    await waitFor(() => expect(sent.length).toBeGreaterThan(0));

    const body = sent.find((r) => r.url.includes("/segmentation/runs"))!
      .body as { options?: Record<string, unknown> };
    // Inside `options`, because that is the one place the handler looks. It used
    // to be declared twice -- a top-level field and an option with the same flag
    // -- and the number typed here landed in the dict the handler skipped, so
    // every run used the 2048 default whatever the page said.
    expect(body.options).toBeDefined();
    expect(Object.keys(body)).not.toContain("grid_step");
  });

  it("sends the bounded probe controls the form is showing", async () => {
    mount();
    fireEvent.change(await screen.findByLabelText("seed probe mode"),
                     { target: { value: "select" } });
    expect(await screen.findByText("seed-probe-test-v1")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("seed probe candidates"),
                     { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText("seed probe generations"),
                     { target: { value: "18" } });

    fireEvent.click(screen.getByRole("button", { name: /^queue$/i }));
    await waitFor(() => expect(sent.length).toBeGreaterThan(0));

    const body = sent.find((r) => r.url.includes("/segmentation/runs"))!
      .body as Record<string, unknown>;
    expect(body.seed_probe_mode).toBe("select");
    expect(body.seed_probe_top_k).toBe(3);
    expect(body.seed_probe_generations).toBe(18);
  });
});
