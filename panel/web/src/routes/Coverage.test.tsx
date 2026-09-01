import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Coverage from "./Coverage";

const COVERAGE = {
  available: true,
  grids: [{ sample_id: "PHerc358", grid_version: "g1", cells_attempted: 2,
    cells_no_seed: 1, cells_with_surface: 1, tasks: 2, cells_in_volume: 10,
    fraction_attempted: 0.2, grid_step_xyz: [16, 16, 16] }],
  volumes: [], non_claims: [],
  campaign_decisions: [],
  candidate_preflight: {
    schema: "campaignx.segment_candidate_coverage_preflight.sanitized.v1",
    evidence_status: "CURRENT", evidence_status_reason: "all frozen bindings match",
    measurement_kind: "INCOMPLETE_ESTIMATE", planned_sampling_percentage: 25,
    achieved_successful_sampling_percentage: 22.5,
    funnel: { total_grid_cells: 80, grid_cells_in_design_sample: 20,
      geometrically_eligible_cells: null, geometrically_eligible_cells_estimate: 40,
      geometrically_eligible_sampled_cells: 11, cells_attempted: 10,
      cells_surveyed: 9, cells_surveyed_successfully: 9, cells_failed_source: 1,
      cells_with_raw_m7_candidates: 4, raw_m7_candidates: 8,
      post_ct_candidates: 6, post_cell_clearance_candidates: 4,
      post_volume_clearance_candidates: 3,
      packet_retained_candidates: 2, source_errors: 1 },
    spatial_bins: [{ bin_xyz: [0, 1, 0], total_cells: 12, surveyed_cells: 4,
      candidate_bearing_cells: 2, usable_candidate_cells: 1 }],
    no_candidate_causes: { NO_M7_CANDIDATES: 6 },
    non_claim: "Candidate scarcity is not evidence of surface, ink, text, or letter absence.",
  },
};

const DECISION = {
    schema: "campaignx.first_letters_campaign_decision.v1",
    decision: "PAUSE_CANDIDATE_STARVATION",
    evidence_status: "COMPLETE",
    mission_id: "first-letters", policy_version: "search-v1",
    evaluation_kind: "SCIENTIFIC_TERMINAL_BLOCK", evaluation_index: 1,
    no_m7_numerator: 7, scientific_terminal_denominator: 8,
    excluded_attempt_count: 1,
    excluded_attempts: [{ attempt_id: "attempt-source", task_id: "task-source",
      reason: "SOURCE_FAILURE" }],
    trigger_attempt_ids: ["attempt-0", "attempt-1", "attempt-2", "attempt-3",
      "attempt-4", "attempt-5", "attempt-6"],
    receipt_sha256: "d".repeat(64),
    allowed_next_actions: ["CREATE_MATERIALLY_CHANGED_VERSIONED_STRATEGY",
      "CLOSE_CAMPAIGN"],
    non_claim: "Historical starvation is not evidence of letter absence.",
};

const ACTIVE_DECISION = {
    schema: "campaignx.first_letters_campaign_active_decision.v1",
    decision: "CONTINUE",
    evidence_status: "IN_PROGRESS",
    mission_id: "first-letters", policy_version: "search-v2",
    policy_chain: ["search-v1", "search-v2"],
    evaluation_kind: "ACTIVE_SCIENTIFIC_TERMINAL_BLOCK", evaluation_index: 1,
    scientific_terminal_attempt_count: 0,
    no_m7_numerator: 0, scientific_terminal_denominator: 8,
    excluded_attempt_count: 0, excluded_attempts: [], trigger_attempt_ids: [],
    state_sha256: "e".repeat(64),
    allowed_next_actions: ["QUEUE_NEXT_BOUND_WAVE", "CLOSE_CAMPAIGN"],
    non_claim: "Live campaign progress is not evidence of letter presence.",
};

let coverageReply = COVERAGE;

beforeEach(() => {
  coverageReply = COVERAGE;
  vi.stubGlobal("fetch", vi.fn(async (input: any) => ({
    ok: true, status: 200,
    json: async () => String(input).startsWith("/api/coverage") ? coverageReply : {
      available: true, by_cause: {}, attempts: 0, note: "",
    },
  })));
});
afterEach(() => vi.unstubAllGlobals());

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <Coverage sample="PHerc358" mission="first-letters" />
  </QueryClientProvider>);
}

describe("candidate availability", () => {
  it("keeps candidate coverage separate from attempted-cell coverage", async () => {
    draw();
    expect(await screen.findByText("Candidate availability preflight")).toBeTruthy();
    expect(screen.getByText(/incomplete estimate.*25\.0% planned.*22\.5% successful/i)).toBeTruthy();
    expect(screen.getByText(/CURRENT evidence/i)).toBeTruthy();
    expect(screen.getByText(/raw M7/i)).toBeTruthy();
    expect(screen.getByText(/post CT/i)).toBeTruthy();
    expect(screen.getByText(/post cell clearance/i)).toBeTruthy();
    expect(screen.getByText(/post volume clearance/i)).toBeTruthy();
    expect(screen.getByText(/packet retained/i)).toBeTruthy();
  });

  it("states that scarcity is not absence and shows source errors", async () => {
    draw();
    expect(await screen.findByText(/Candidate scarcity is not evidence/)).toBeTruthy();
    expect(screen.getByText(/9 successful of 10 attempted.*1 source failure/i)).toBeTruthy();
    expect(screen.getByText(/estimated eligible population.*40/i)).toBeTruthy();
    expect(screen.getByText(/bin 0,1,0/i)).toBeTruthy();
  });

  it("shows the preflight even before any fleet cell was attempted", async () => {
    coverageReply = { ...COVERAGE, grids: [] };
    draw();
    expect(await screen.findByText("Candidate availability preflight")).toBeTruthy();
    expect(screen.queryByText("no cells have been attempted here yet")).toBeNull();
  });

  it("refuses to render measurements from invalid evidence", async () => {
    coverageReply = { ...COVERAGE, candidate_preflight: {
      schema: "campaignx.segment_candidate_coverage_preflight.sanitized.v1",
      evidence_status: "INVALID",
      evidence_status_reason: "latest sanitized receipt hash does not match its content",
    } as any };
    draw();
    expect(await screen.findByText(/INVALID evidence/i)).toBeTruthy();
    expect(screen.getByText(/hash does not match/i)).toBeTruthy();
    expect(screen.queryByText(/raw M7/i)).toBeNull();
  });

  it("shows the immutable starvation gate and its exact audit evidence", async () => {
    coverageReply = { ...COVERAGE, grids: [], candidate_preflight: null,
      active_campaign_decision: DECISION, campaign_decisions: [DECISION] } as any;
    draw();
    expect(await screen.findByText("Campaign decision")).toBeTruthy();
    expect(screen.getAllByText(/PAUSE CANDIDATE STARVATION/i).length).toBe(2);
    expect(screen.getByText(/7 of 8 scientific terminal attempts/i)).toBeTruthy();
    expect(screen.getByText(/SOURCE FAILURE.*attempt-source/i)).toBeTruthy();
    expect(screen.getByText(/attempt-0.*attempt-6/i)).toBeTruthy();
    expect(screen.getByText("d".repeat(64))).toBeTruthy();
    expect(screen.getByText(/CREATE MATERIALLY CHANGED VERSIONED STRATEGY/i)).toBeTruthy();
    expect(screen.getAllByText(DECISION.non_claim).length).toBe(2);
    expect(screen.queryByText("no cells have been attempted here yet")).toBeNull();
  });

  it("renders the server-derived successor policy and keeps the old pause as history", async () => {
    coverageReply = { ...COVERAGE, grids: [], candidate_preflight: null,
      active_campaign_decision: ACTIVE_DECISION,
      campaign_decisions: [DECISION] } as any;
    draw();
    expect(await screen.findByText("Campaign decision")).toBeTruthy();
    expect(screen.getByText(/search-v2.*active scientific terminal block/i)).toBeTruthy();
    expect(screen.getByText(/0 of 8 scientific terminal attempts/i)).toBeTruthy();
    expect(screen.queryByText(/7 of 8 scientific terminal attempts/i)).toBeNull();
    expect(screen.getByText(/search-v1: PAUSE CANDIDATE STARVATION/i)).toBeTruthy();
    expect(screen.getByText(ACTIVE_DECISION.non_claim)).toBeTruthy();
    expect(screen.getByText(DECISION.non_claim)).toBeTruthy();
    expect(screen.getByText("e".repeat(64))).toBeTruthy();
  });
});
