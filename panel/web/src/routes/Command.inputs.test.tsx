/**
 * What the ink launcher lets you name as its input.
 *
 * This form keeps its own field list, and the list drifted from the queue's.
 * P5 has accepted three ways of naming an input for some time -- a layer stack
 * on disk, the P4 render that published one, or a surface volume already at the
 * model's scale -- and the form offered only the first. So from a browser there
 * was exactly one way to run ink, and the other two were reachable only by
 * calling the API by hand.
 *
 * It also demanded `upstream_dir` before it would enable its own button. That
 * is the 2 um lane's field; the 9 um lane carries its own architecture and has
 * no such flag, so the form refused to queue a lane whose parameters were
 * complete.
 *
 * The same defect the segmentation launcher's own test file was written about:
 * a flag exists in the queue, no field exists for it, and nothing notices.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import Command from "./Command";

const LANES = {
  profiles: [
    { profile_id: "ink-9um-hybrid-3d2d-screening@1.0.0", disqualified: false,
      input_contract: { model_type: "hybrid_3d2d", training_pixel_um: 9.362 } },
  ],
};

const posted: any[] = [];

beforeEach(() => {
  posted.length = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: any, init?: any) => {
    const path = String(input);
    if (init?.method === "POST") {
      posted.push(JSON.parse(init.body));
      return { ok: true, status: 201, json: async () => ({ job_id: "p5-x" }) };
    }
    const body = path.startsWith("/api/lanes") ? LANES
      : path.startsWith("/api/hosts") ? { hosts: [] }
      : path.startsWith("/api/jobs") ? { jobs: [] }
      : path.startsWith("/api/scrolls") ? { scrolls: [{ sample_id: "PHerc0139" }] }
      : {};
    return { ok: true, status: 200, json: async () => body };
  }));
});

afterEach(() => vi.unstubAllGlobals());

vi.mock("../mission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../mission")>()),
  useMission: () => ({ missionId: "public-ink-control-0139" }),
  useSubject: () => ({ subject: "PHerc0139" }),
}));

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}><Command /></QueryClientProvider>,
  );
}

async function pickProfile(id: string) {
  // By its options rather than by a label: the select's label text also matches
  // other controls. Awaited, because the lane list is a query and the form
  // renders before it lands.
  const select = await waitFor(() => {
    const found = Array.from(document.querySelectorAll("select"))
      .find((s) => Array.from(s.options).some((o) => o.value === id));
    if (!found) throw new Error("no profile select offers " + id);
    return found;
  });
  fireEvent.change(select, { target: { value: id } });
}

function type(label: RegExp, value: string) {
  const field = screen.getByLabelText(label);
  fireEvent.change(field, { target: { value } });
  return field;
}

it("offers all three ways of naming an input", async () => {
  draw();

  expect(await screen.findByLabelText(/TIFF directory/i)).toBeDefined();
  expect(screen.getByLabelText(/render that produced|layer stack id/i)).toBeDefined();
  expect(screen.getByLabelText(/surface volume/i)).toBeDefined();
});

it("queues a surface volume without a layer stack or an upstream directory", async () => {
  draw();
  await screen.findByLabelText(/surface volume/i);

  await pickProfile("ink-9um-hybrid-3d2d-screening@1.0.0");
  type(/surface volume/i, "https://open-data/vol.zarr");
  type(/Checkpoint/i, "/models/step-075000.pth");

  const button = await screen.findByRole("button", { name: /queue run/i });
  await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(button);

  await waitFor(() => expect(posted.length).toBe(1));
  expect(posted[0].parameters.surface_volume).toBe("https://open-data/vol.zarr");
  expect(posted[0].parameters.tiff_dir).toBeUndefined();
  expect(posted[0].parameters.upstream_dir).toBeUndefined();
  expect(posted[0].mission_id).toBe("public-ink-control-0139");
});

it("will not queue two inputs at once", async () => {
  draw();
  await screen.findByLabelText(/surface volume/i);

  await pickProfile("ink-9um-hybrid-3d2d-screening@1.0.0");
  type(/surface volume/i, "https://open-data/vol.zarr");
  type(/TIFF directory/i, "/layers");
  type(/Checkpoint/i, "/models/step.pth");

  const button = await screen.findByRole("button", { name: /queue run/i });
  expect((button as HTMLButtonElement).disabled).toBe(true);
});

it("will not queue with no input at all", async () => {
  draw();
  await screen.findByLabelText(/surface volume/i);

  await pickProfile("ink-9um-hybrid-3d2d-screening@1.0.0");
  type(/Checkpoint/i, "/models/step.pth");

  const button = await screen.findByRole("button", { name: /queue run/i });
  expect((button as HTMLButtonElement).disabled).toBe(true);
});


it("shows what a running job is doing", async () => {
  /**
   * The live column was added to the phase page's generic queue table, and P5
   * does not use that table -- it has this one. So on the single screen where
   * an ink run is actually watched, a job stayed `running` and said nothing,
   * which is the blindness the progress column exists to end.
   */
  vi.stubGlobal("fetch", vi.fn(async (input: any) => {
    const path = String(input);
    const body = path.startsWith("/api/lanes") ? LANES
      : path.startsWith("/api/hosts") ? { hosts: [] }
      : path.startsWith("/api/jobs") ? { jobs: [{
          job_id: "p5-live", sample_id: "PHerc0139", phase: "P5",
          profile_id: "ink-9um-hybrid-3d2d-screening@1.0.0",
          state: "running", attempts: 1, max_attempts: 1, result: null,
          requested_host: null, worker_id: "gpu-1-ink9um",
          progress: { line: "Infer:  63%|###   | 6857/10885 [07:41<04:31]",
                      source: "stderr", at: new Date().toISOString() },
        }] }
      : path.startsWith("/api/scrolls") ? { scrolls: [{ sample_id: "PHerc0139" }] }
      : {};
    return { ok: true, status: 200, json: async () => body };
  }));

  draw();

  expect(await screen.findByText(/6857\/10885/)).toBeDefined();
});
