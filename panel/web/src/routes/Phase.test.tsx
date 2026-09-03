/**
 * The form that draws itself from the queue's schema.
 *
 * What is worth testing here is not the markup but the decisions: which fields
 * exist at all, which pair must be exactly one, what the deployment fills in and
 * therefore must not be asked of a person, and what is actually POSTed. Those
 * are the things that were wrong when this file kept its own copy of the
 * parameter list -- a flag existed in the queue and no field existed for it, and
 * nothing anywhere noticed.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Phase from "./Phase";

// The P4 schema as the panel serves it, trimmed to what these assertions need.
const P4_SCHEMA = {
  available: true,
  phase: "P4",
  lanes: [
    { id: "vc-render-tifxyz", name: "volume-cartographer renderer",
      note: "reads a tifxyz directly", validated: null, required: [] },
    { id: "chunk-gather", name: "chunk-gather renderer (Scroll 3 only)",
      note: "needs a PPM", validated: "r = 0.89 against an official ink map",
      required: [] },
  ],
  exactly_one_of: [
    { lane: "vc-render-tifxyz", names: ["segmentation", "flattened_surface"] },
  ],
  fields: [
    { name: "volume", type: "text", required: true, lane: "vc-render-tifxyz",
      label: "Volume", note: null, placeholder: "/vol/scroll.zarr",
      filled_by_deployment: false },
    { name: "segmentation", type: "text", required: true, lane: "vc-render-tifxyz",
      label: "Surface path", note: null, placeholder: null,
      filled_by_deployment: false },
    { name: "flattened_surface", type: "text", required: false, lane: null,
      label: "…or a flattened surface id", note: null, placeholder: null,
      filled_by_deployment: false },
    { name: "flip_normals", type: "boolean", required: false, lane: null,
      label: "Reverse the direction along the normal",
      note: "r = 0.885 with this on", placeholder: null,
      filled_by_deployment: false },
    { name: "num_slices", type: "integer", required: false, lane: null,
      label: "Slices", note: null, placeholder: "63", filled_by_deployment: false },
    { name: "artifact_store", type: "text", required: false, lane: null,
      label: "Where it publishes", note: null, placeholder: null,
      filled_by_deployment: true },
    { name: "ppm", type: "text", required: true, lane: "chunk-gather",
      label: "PPM file", note: null, placeholder: null, filled_by_deployment: false },
    // A field that names another job, with the jobs this mission has.
    { name: "screening_of", type: "text", required: false, lane: null,
      label: "…or the screening that produced one", note: null, placeholder: null,
      filled_by_deployment: false, names_a_job_from: "P5",
      choices: [{ value: "p5-0c1c5934eaf442", note: "PHerc0139 · ink-9um" }] },
  ],
};

const PHASE = {
  contract: { id: "P4", name: "Surface volume rendering", one_line: "Sample the CT",
              maturity: "WORKING", distributed: true },
  state: {},
  artefacts: [], jobs: [], profiles: [], components: [],
  queueable: true, queueable_reason: null,
};

const posted: { path: string; body: any }[] = [];

// What this phase reports about itself, per test: a phase with state rows draws
// a tab bar, and the launcher stops being the default view.
let state: Record<string, unknown> = {};

function reply(path: string) {
  if (path.startsWith("/api/phases/P4/parameters")) return P4_SCHEMA;
  if (path.startsWith("/api/phase/P4")) return { ...PHASE, state };
  if (path.startsWith("/api/phase/P5")) return {
    ...PHASE, state,
    contract: { id: "P5", name: "Ink detection", one_line: "Run a detector",
                maturity: "WORKING", distributed: true },
  };
  // A fresh install: no receipt on disk, nothing screened yet.
  if (path.startsWith("/api/runs")) return { runs: [] };
  if (path.startsWith("/api/ink/maps")) return { available: true, runs: [] };
  return {};
}

beforeEach(() => {
  posted.length = 0;
  state = {};
  vi.stubGlobal("fetch", vi.fn(async (input: any, init?: any) => {
    const path = String(input);
    if (init?.method === "POST") {
      posted.push({ path, body: JSON.parse(init.body) });
      return { ok: true, status: 201, json: async () => ({ job_id: "p4-test" }) };
    }
    return { ok: true, status: 200, json: async () => reply(path) };
  }));
});

afterEach(() => vi.unstubAllGlobals());

// Read at render time, so one test can open another phase.
let phaseId = "P4";
vi.mock("react-router", () => ({ useParams: () => ({ phaseId }) }));
// Partial: this module also exports `scoped`, which Phase.tsx calls to build
// its own query. Replacing the module wholesale silently removed it.
vi.mock("../mission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../mission")>()),
  useMission: () => ({ missionId: "test" }),
  useSubject: () => ({ subject: "PHerc826" }),
}));

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Phase />
    </QueryClientProvider>,
  );
}

async function openTheForm() {
  // P4 has one tab, so the bar is not drawn and the form is already the view.
  draw();
  await screen.findByText("Queue work");
  const scroll = await screen.findByLabelText("Scroll");
  expect((scroll as HTMLInputElement).value).toBe("PHerc826");
  expect((scroll as HTMLInputElement).disabled).toBe(true);
}

describe("the queue form", () => {
  it("offers every field the schema serves, including the ones added last", async () => {
    await openTheForm();
    // The flag that took the community control from r = 0.09 to r = 0.885, and
    // could not be set from a browser at all while this file kept its own list.
    expect(await screen.findByLabelText(/Reverse the direction along the normal/))
      .toBeDefined();
    expect(screen.getByText(/Slices/)).toBeDefined();
  });

  it("does not ask a person for what the deployment fills in", async () => {
    await openTheForm();
    await screen.findByLabelText(/Reverse the direction along the normal/);
    expect(screen.queryByText("Where it publishes")).toBeNull();
  });

  it("shows only the chosen lane's fields", async () => {
    await openTheForm();
    await screen.findByPlaceholderText("/vol/scroll.zarr");
    expect(screen.queryByLabelText(/PPM file/)).toBeNull();
    fireEvent.change(document.querySelector("select")!,
                     { target: { value: "chunk-gather" } });
    // "PPM file" now names both the field's own label and, once it is
    // required and empty, the "needs …" reason beside the queue button --
    // getByLabelText matches the field itself and stays unambiguous either
    // way.
    await waitFor(() => expect(screen.getByLabelText(/PPM file/)).toBeDefined());
    expect(screen.queryByPlaceholderText("/vol/scroll.zarr")).toBeNull();
  });

  it("will not queue until exactly one of the surface fields is named", async () => {
    await openTheForm();
    fireEvent.change(screen.getByPlaceholderText("/vol/scroll.zarr"),
                     { target: { value: "/vol/x.zarr" } });
    const queue = screen.getByRole("button", { name: /Queue P4 job/ });
    expect((queue as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/name exactly one of/)).toBeDefined();
  });

  it("queues with the other half of an exactly-one-of pair", async () => {
    // The surface path is `required` and pairs with the flattened surface id.
    // Naming the id satisfied the pair and re-armed the path, so the button
    // stayed disabled however the form was filled in, and a flattened sheet
    // could not be rendered from a browser at all.
    await openTheForm();
    fireEvent.change(screen.getByPlaceholderText("/vol/scroll.zarr"),
                     { target: { value: "/vol/x.zarr" } });
    fireEvent.change(screen.getByLabelText(/or a flattened surface id/),
                     { target: { value: "151ea1c0-0308" } });
    await waitFor(() => expect(
      (screen.getByRole("button", { name: /Queue P4 job/ }) as HTMLButtonElement).disabled,
    ).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: /Queue P4 job/ }));
    await waitFor(() => expect(posted.length).toBe(1));
    expect(posted[0].body.parameters.flattened_surface).toBe("151ea1c0-0308");
    expect(posted[0].body.parameters.segmentation).toBeUndefined();
  });

  it("posts the types the queue declared, not strings", async () => {
    await openTheForm();
    fireEvent.change(screen.getByPlaceholderText("/vol/scroll.zarr"),
                     { target: { value: "/vol/x.zarr" } });
    const fields = screen.getAllByRole("textbox") as HTMLInputElement[];
    fireEvent.change(fields.find((f) => f.placeholder === "63")!,
                     { target: { value: "63" } });
    // Exactly one of the pair.
    fireEvent.change(fields.find((f) => f.placeholder === null || f.placeholder === "")!,
                     { target: { value: "/surfaces/s-1" } });
    fireEvent.click(screen.getByLabelText(/Reverse the direction along the normal/));
    await waitFor(() => expect(
      (screen.getByRole("button", { name: /Queue P4 job/ }) as HTMLButtonElement).disabled,
    ).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: /Queue P4 job/ }));
    await waitFor(() => expect(posted.length).toBe(1));

    const { body } = posted[0];
    expect(body.phase).toBe("P4");
    expect(body.mission_id).toBe("test");
    expect(body.sample_id).toBe("PHerc826");
    expect(body.parameters.lane).toBe("vc-render-tifxyz");
    expect(body.parameters.num_slices).toBe(63);          // integer, not "63"
    expect(body.parameters.flip_normals).toBe(true);      // boolean, not "on"
    expect(body.parameters.artifact_store).toBeUndefined();
  });
});

describe("what the phase reports", () => {
  it("reads a tally as words rather than as JSON", async () => {
    // {"ALIVE":2} on screen, braces and quotes included, is what P6's verdict
    // tile said for what is two words.
    state = { runs_with_liveness: 2, verdicts: { ALIVE: 2, DEAD: 1 } };
    draw();
    expect(await screen.findByText("ALIVE 2 · DEAD 1")).toBeDefined();
    expect(screen.queryByText(/\{"ALIVE"/)).toBeNull();
  });

  it("offers the jobs a job-naming field can name", async () => {
    await openTheForm();
    const field = await screen.findByLabelText(/or the screening that produced one/);
    // A datalist, not a select: an id from outside this mission is still an
    // answer somebody may have.
    expect((field as HTMLInputElement).getAttribute("list")).toBe("P4-screening_of");
    const option = document.querySelector("#P4-screening_of option");
    expect(option?.getAttribute("value")).toBe("p5-0c1c5934eaf442");
  });

  it("asks the queue for candidates in this mission and scroll", async () => {
    await openTheForm();
    const asked = (globalThis.fetch as any).mock.calls
      .map((call: any[]) => String(call[0]))
      .find((path: string) => path.startsWith("/api/phases/P4/parameters"));
    expect(asked).toContain("mission=test");
    expect(asked).toContain("sample=PHerc826");
  });
});

describe("P5", () => {
  afterEach(() => { phaseId = "P4"; });

  it("opens on Maps, where a screening queued through the fleet is", async () => {
    // Runs indexes receipts on disk under CX_RUNS, which a fresh install has
    // none of; the phase opened on that empty tab while its one map sat under
    // Maps, and the empty state asked about an environment variable.
    phaseId = "P5";
    draw();
    const maps = await screen.findByRole("button", { name: "Maps" });
    expect(maps.getAttribute("aria-current")).toBe("page");
    const runs = screen.getByRole("button", { name: "Runs" });
    expect(runs.getAttribute("aria-current")).toBeNull();
    fireEvent.click(runs);
    await screen.findByText(/Screenings queued through the fleet are under Maps/);
  });
});
