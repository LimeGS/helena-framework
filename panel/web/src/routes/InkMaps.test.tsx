import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import InkMaps from "./InkMaps";

/**
 * What P5 produced, on a page.
 *
 * The three things worth holding here are the three ways this page could lie:
 * a fabricated statistic where a lane writes none, a picture with no statement
 * of the stretch that made it, and a run whose bytes are on another host
 * rendered as though they were here.
 */

const ALIVE = {
  verdict: "ALIVE",
  reason: "",
  metrics: { p50: 0.3121, p99: 0.884, spread_p99_p50: 0.5719,
             std: 0.1904, fraction_near_half: 0.221, valid_pixels: 262144 },
};
const DEGENERATE = {
  verdict: "DEGENERATE",
  reason: "p99-p50 0.0004 < 0.05; std 0.0002 < 0.02",
  metrics: { p50: 0.5001, p99: 0.5005, spread_p99_p50: 0.0004,
             std: 0.0002, fraction_near_half: 1.0, valid_pixels: 262144 },
  interpretation: "the map carries no decision. Do not screen this map.",
};

const CANONICAL = {
  job_id: "p5-canonical", sample_id: "PHerc0172",
  surface_id: "surface:PHerc0172:0041",
  input: { kind: "layer_stack", value: "/ssd/vc3d/artifacts/layers/p4-9931" },
  profile_id: "ink-2um-canonical", state: "succeeded",
  mission_id: "first-letters", attempts: 1, max_attempts: 3,
  created_at: "2026-08-30T11:00:00+00:00",
  updated_at: "2026-08-30T11:04:00+00:00",
  runtime_seconds: 214.6, liveness: ALIVE,
  statistics: { valid_pixels: 262144, p50: 0.3121, p90: 0.7, p99: 0.884,
                max: 0.99, fraction_above_0_5: 0.0145 },
  checkpoint_sha256: "c".repeat(64), map_shape_yx: [512, 512],
  output_dir: "/srv/helena/runs/first-letters/pherc0172-p5-canonical",
  maps: ["probability.npy"],
  published: { artifact_uri: "s3://helena/ink-maps/p5-canonical",
               artifact_sha256: "f".repeat(64),
               manifest_sha256: "0".repeat(64), files: 2 },
  error: null, refused: null,
};

// The TimeSformer lane writes no `statistics` block at all, and its map is not
// mounted on the panel host.
const TIMESFORMER = {
  ...CANONICAL,
  job_id: "p5-timesformer", sample_id: "PHerc0826",
  surface_id: null, profile_id: "timesformer-gp-scroll1-screening",
  created_at: "2026-08-29T09:00:00+00:00",
  updated_at: "2026-08-29T09:20:00+00:00",
  liveness: DEGENERATE, statistics: null,
  maps: [], map_shape_yx: [1610, 1610],
  published: { artifact_uri: "s3://helena/ink-maps/p5-timesformer",
               artifact_sha256: "a".repeat(64),
               manifest_sha256: "b".repeat(64), files: 4 },
  state: "failed",
  refused: "DEGENERATE map: the lane produced no decision",
};

const DISPLAY = {
  height: 512, width: 512, valid_pixels: 262144, invalid_pixels: 1024,
  normalisation: "percentile", low_percentile: 1, high_percentile: 99,
  low_value: 0.0592, high_value: 0.9411, flat: false, min: 0.05, max: 0.99,
  note: "displayed on a percentile stretch: p1=0.0592 is black and p99=0.9411 "
      + "is white. Brightness here is relative to this map alone and is not "
      + "comparable between runs.",
};

const DETAIL = {
  ...CANONICAL,
  selected_map: "probability.npy", display: DISPLAY, display_error: null,
  receipt: { schema: "campaignx.ink_profile_screening_receipt.v1" },
  receipt_path: "/srv/helena/runs/first-letters/pherc0172-p5-canonical"
                + "/INK_PROFILE_RECEIPT.json",
  receipt_unavailable: null,
  lineage: { schema: "campaignx.first_letters_p5_normalization.v1",
             p4_job_id: "p4-9931",
             p4_layer_artifact_sha256: "a".repeat(64),
             p4_layer_manifest_sha256: "b".repeat(64) },
  rendered_from: null,
  profile: { profile_id: "ink-2um-canonical", method_id: "youssef-2023",
             adapter: "run_ink.py", checkpoint_sha256: "c".repeat(64),
             input_contract: {} },
  probability_map: null,
};

let list: any = { available: true, runs: [CANONICAL, TIMESFORMER] };
let detail: any = DETAIL;

beforeEach(() => {
  list = { available: true, runs: [CANONICAL, TIMESFORMER] };
  detail = DETAIL;
  vi.stubGlobal("fetch", vi.fn(async (input: any) => ({
    ok: true, status: 200, statusText: "OK",
    json: async () => String(input).includes("/api/ink/maps/") ? detail : list,
  })));
});
afterEach(() => vi.unstubAllGlobals());

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <InkMaps sample="PHerc0172" mission="first-letters" />
  </QueryClientProvider>);
}

describe("the P5 run table", () => {
  it("shows the identity, the lane, the state and the liveness metrics", async () => {
    draw();
    expect(await screen.findByText("p5-canonical")).toBeTruthy();
    const row = screen.getByText("p5-canonical").closest("tr")!;
    expect(within(row).getByText("PHerc0172")).toBeTruthy();
    // The surface is shown by its last segment, with the whole id on the cell.
    expect(within(row).getByText("0041")).toBeTruthy();
    expect(within(row).getByText("ink-2um-canonical")).toBeTruthy();
    expect(within(row).getByText("succeeded")).toBeTruthy();
    expect(within(row).getByText("ALIVE")).toBeTruthy();
    expect(within(row).getByText("0.312")).toBeTruthy();
    expect(within(row).getByText("0.884")).toBeTruthy();
    expect(within(row).getByText("0.572")).toBeTruthy();
    expect(within(row).getByText("2026-08-30 11:04")).toBeTruthy();
  });

  it("says a map is published elsewhere rather than pretending it is here", async () => {
    draw();
    expect(await screen.findByText("p5-timesformer")).toBeTruthy();
    const row = screen.getByText("p5-timesformer").closest("tr")!;
    expect(within(row).getByText(/published, not on this host/)).toBeTruthy();
    expect(within(row).getByText("DEGENERATE")).toBeTruthy();
    expect(within(row).getByText(/none named/)).toBeTruthy();
  });

  it("filters by job, scroll, surface or lane", async () => {
    draw();
    expect(await screen.findByText("p5-canonical")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Filter screenings"),
                     { target: { value: "timesformer" } });
    expect(screen.queryByText("p5-canonical")).toBeNull();
    expect(screen.getByText("p5-timesformer")).toBeTruthy();
    expect(screen.getByText("1 of 2 shown")).toBeTruthy();
  });

  it("filters by liveness verdict, which is the column that decides", async () => {
    draw();
    expect(await screen.findByText("p5-canonical")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Filter by liveness verdict"),
                     { target: { value: "DEGENERATE" } });
    expect(screen.queryByText("p5-canonical")).toBeNull();
    expect(screen.getByText("p5-timesformer")).toBeTruthy();
  });

  it("sorts on a column and reverses on a second click", async () => {
    draw();
    expect(await screen.findByText("p5-canonical")).toBeTruthy();
    const jobIds = () => screen.getAllByRole("row").slice(1)
      .map((row) => row.querySelector("td button")!.textContent);
    // Newest first without touching anything.
    expect(jobIds()).toEqual(["p5-canonical", "p5-timesformer"]);

    fireEvent.click(screen.getByRole("button", { name: /^p50/ }));
    expect(jobIds()).toEqual(["p5-canonical", "p5-timesformer"]);
    fireEvent.click(screen.getByRole("button", { name: /^p50/ }));
    expect(jobIds()).toEqual(["p5-timesformer", "p5-canonical"]);
    expect(screen.getByRole("columnheader", { name: /^p50/ })
      .getAttribute("aria-sort")).toBe("descending");
  });

  it("says why the table is empty instead of showing an empty table", async () => {
    list = { available: false, reason: "CX_DB is not set: P5 runs live in the fleet queue" };
    draw();
    expect(await screen.findByText(/CX_DB is not set/)).toBeTruthy();
  });

  it("distinguishes no runs from no matches", async () => {
    list = { available: true, runs: [] };
    draw();
    expect(await screen.findByText(/no P5 job has run in this scope yet/)).toBeTruthy();
  });
});

describe("inspecting one run", () => {
  async function open() {
    draw();
    fireEvent.click(await screen.findByText("p5-canonical"));
    // Not "Liveness": that is also a column header on the table above, so it
    // resolves before the detail query has answered anything.
    return screen.findByText("Provenance");
  }

  it("renders the map through the server, never the array", async () => {
    await open();
    const image = screen.getByRole("img",
      { name: /Probability map probability.npy/ }) as HTMLImageElement;
    expect(image.getAttribute("src")).toContain(
      "/api/ink/maps/p5-canonical/render.png");
    expect(image.getAttribute("src")).toContain("map=probability.npy");
    // The array is named in the query string and is never itself the source:
    // a float32 .npy in the browser is a second implementation of the stretch.
    expect(image.getAttribute("src")!.split("?")[0].endsWith(".npy")).toBe(false);
    expect(document.querySelectorAll('a[href$=".npy"]').length).toBe(0);
  });

  it("states the stretch the picture was drawn on", async () => {
    await open();
    expect(screen.getByText(/percentile stretch/)).toBeTruthy();
    expect(screen.getByText(/p1=0.0592 is black/)).toBeTruthy();
    expect(screen.getByText(/not comparable between runs/)).toBeTruthy();
    expect(screen.getByText(/512×512 px/)).toBeTruthy();
    // Uncovered pixels are transparent, not the darkest probability.
    expect(screen.getByText(/drawn transparent rather than as a low probability/))
      .toBeTruthy();
  });

  it("shows the receipt's own numbers, the digest and the lineage", async () => {
    await open();
    expect(screen.getByText("spread p99 p50")).toBeTruthy();
    expect(screen.getByText("0.5719")).toBeTruthy();
    expect(screen.getByText("fraction above 0 5")).toBeTruthy();
    expect(screen.getByText("c".repeat(64))).toBeTruthy();
    expect(screen.getByText("matches the profile")).toBeTruthy();
    expect(screen.getByText("p4-9931")).toBeTruthy();
    expect(screen.getByText("a".repeat(64))).toBeTruthy();
    expect(screen.getByText(/INK_PROFILE_RECEIPT.json/)).toBeTruthy();
  });

  it("says an ALIVE verdict has no failing check rather than leaving it blank",
     async () => {
    await open();
    expect(screen.getByText(/no failing check/)).toBeTruthy();
  });

  it("refuses to invent a statistics block the lane never wrote", async () => {
    detail = { ...DETAIL, statistics: null };
    await open();
    expect(screen.getByText("this lane's receipt carries no statistics block"))
      .toBeTruthy();
    // The liveness metrics are still there: that block every lane writes.
    expect(screen.getByText("spread p99 p50")).toBeTruthy();
  });

  it("says a run was never chained to a P4 render rather than blanking it", async () => {
    detail = { ...DETAIL, lineage: null, rendered_from: null };
    await open();
    expect(screen.getByText(/not chained to a P4 render of this control plane/))
      .toBeTruthy();
    expect(screen.queryByText("p4-9931")).toBeNull();
  });

  it("draws no picture for a run whose bytes are on another host", async () => {
    detail = { ...DETAIL, maps: [], selected_map: null, display: null,
               display_error: null };
    await open();
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText(/published but not mounted on this host/)).toBeTruthy();
  });

  it("carries the refusal beside the run that earned it", async () => {
    detail = { ...DETAIL, liveness: DEGENERATE, state: "failed",
               refused: "DEGENERATE map: the lane produced no decision" };
    await open();
    // Scoped to the card: the verdict is on the row above it too, which is the
    // point -- the table says which runs are dead and the card says why.
    const card = screen.getByRole("heading", { name: "Liveness" })
      .closest("section")!;
    expect(within(card).getByText("DEGENERATE")).toBeTruthy();
    expect(within(card).getByText(/Do not screen this map/)).toBeTruthy();
    expect(screen.getByText(/the lane produced no decision/)).toBeTruthy();
  });
});
