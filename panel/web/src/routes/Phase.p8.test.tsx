import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import Phase from "./Phase";

const schema = {
  available: true,
  phase: "P8",
  lanes: [
    { id: "column-atlas", name: "column atlas", note: "legacy", required: ["scroll", "out_path"], profiles: [] },
    { id: "vc3d-tifxyz-merge", name: "Volume Cartographer TIFXYZ merge", note: "fixed upstream lane",
      required: ["artifact_ids", "rows", "reference_artifact_id", "ransac_seed", "anchor_cap", "strip_cols", "artifact_store"],
      profiles: ["vc3d-tifxyz-merge@1.0.0"] },
  ],
  exactly_one_of: [],
  fields: [
    { name: "lane", type: "text", required: false, lane: null, label: "Lane", note: null, placeholder: null, filled_by_deployment: false },
    { name: "artifact_ids", type: "json", required: true, lane: "vc3d-tifxyz-merge", label: "Certified TIFXYZ artifacts", note: null, placeholder: null, filled_by_deployment: false },
    { name: "rows", type: "json", required: true, lane: "vc3d-tifxyz-merge", label: "Surface layout", note: null, placeholder: null, filled_by_deployment: false },
    { name: "reference_artifact_id", type: "text", required: true, lane: "vc3d-tifxyz-merge", label: "Reference surface", note: null, placeholder: null, filled_by_deployment: false },
    { name: "ransac_seed", type: "integer", required: true, lane: "vc3d-tifxyz-merge", label: "RANSAC seed", note: null, placeholder: null, filled_by_deployment: false },
    { name: "anchor_cap", type: "integer", required: true, lane: "vc3d-tifxyz-merge", label: "Anchor cap", note: null, placeholder: null, filled_by_deployment: false },
    { name: "strip_cols", type: "integer", required: true, lane: "vc3d-tifxyz-merge", label: "Strip columns", note: null, placeholder: null, filled_by_deployment: false },
    { name: "artifact_store", type: "text", required: true, lane: "vc3d-tifxyz-merge", label: "Where it publishes", note: null, placeholder: null, filled_by_deployment: true },
  ],
};

const phase = {
  contract: { id: "P8", name: "Reconstruction", one_line: "assemble surfaces", maturity: "WORKING", distributed: true },
  state: {}, artefacts: [], jobs: [], profiles: [], components: [],
  queueable: true, queueable_reason: null,
};

let posted: any = null;

vi.mock("react-router", () => ({ useParams: () => ({ phaseId: "P8" }) }));
vi.mock("../mission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../mission")>()),
  useMission: () => ({ missionId: "golden-run" }),
  useSubject: () => ({ subject: "PHerc826" }),
}));

beforeEach(() => {
  posted = null;
  vi.stubGlobal("fetch", vi.fn(async (input: any, init?: any) => {
    const path = String(input);
    if (init?.method === "POST") {
      posted = JSON.parse(init.body);
      return { ok: true, status: 201, json: async () => ({ job_id: "p8-merge" }) };
    }
    return { ok: true, status: 200,
      json: async () => path.includes("/parameters") ? schema : phase };
  }));
});

afterEach(() => vi.unstubAllGlobals());

it("posts typed fan-in data and the lane's frozen profile, never a command", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}><Phase /></QueryClientProvider>,
  );

  const renderer = await screen.findByRole("combobox");
  fireEvent.change(renderer,
                   { target: { value: "vc3d-tifxyz-merge" } });
  fireEvent.change(screen.getByLabelText(/Certified TIFXYZ artifacts/),
                   { target: { value: '["surf-a","surf-b"]' } });
  fireEvent.change(screen.getByLabelText(/Surface layout/),
                   { target: { value: '[["surf-a","surf-b"]]' } });
  fireEvent.change(screen.getByLabelText(/Reference surface/),
                   { target: { value: "surf-a" } });
  fireEvent.change(screen.getByLabelText(/RANSAC seed/), { target: { value: "1729" } });
  fireEvent.change(screen.getByLabelText(/Anchor cap/), { target: { value: "0" } });
  fireEvent.change(screen.getByLabelText(/Strip columns/), { target: { value: "0" } });

  const queue = screen.getByRole("button", { name: "Queue P8 job" });
  await waitFor(() => expect((queue as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(queue);
  await waitFor(() => expect(posted).not.toBeNull());

  expect(posted.profile_id).toBe("vc3d-tifxyz-merge@1.0.0");
  expect(posted.parameters.artifact_ids).toEqual(["surf-a", "surf-b"]);
  expect(posted.parameters.rows).toEqual([["surf-a", "surf-b"]]);
  expect(posted.parameters.ransac_seed).toBe(1729);
  expect(posted.parameters.command).toBeUndefined();
});
