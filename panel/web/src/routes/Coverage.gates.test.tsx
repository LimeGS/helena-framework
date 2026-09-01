import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Coverage from "./Coverage";

/**
 * The campaign gates card offers only what the server said its evidence allows.
 *
 * The thing worth testing is what the page will NOT draw: there is no control
 * that accepts a blocked campaign, forces a queue, or turns a stale control
 * into a current one.
 */

const COVERAGE = {
  available: true, grids: [], volumes: [], non_claims: [],
  campaign_decisions: [],
};

const BLOCKED = {
  schema: "campaignx.first_letters_readiness.v1",
  mission_id: "first-letters", controlled: true, reason: null,
  deployed_revision: "a".repeat(40),
  mission_deployed_revision: "a".repeat(40),
  control: {
    available: true, evidence_status: "STALE",
    evidence_status_reason: "the control was run against a different deployed revision",
    control_state: "CONTROL_PASS", control_id: "PHerc0139-w025-public-positive-v1",
    content_sha256: "b".repeat(64), first_nonpassing_boundary: null,
    bound_deployed_revision: "9".repeat(40), stages: [],
  },
  scrolls: [{
    sample_id: "PHerc358", requested_sample_id: "PHerc0358",
    preflight: { evidence_status: "MISSING",
      evidence_status_reason: "no candidate preflight has been run against the current source",
      measurement_kind: null, status: null, private_receipt_sha256: null },
    budget: { evidence_status: "MISSING", decision: null, receipt_sha256: null,
      planned_task_count: null, requested_task_count: null,
      clipped_by_compute_cap: false, binds_current_preflight: false },
    blockers: [{ code: "PREFLIGHT_MISSING", scope: "PHerc0358",
      detail: "no candidate preflight has been run against the current source" }],
    advisories: [], allowed_actions: ["RUN_CANDIDATE_PREFLIGHT"],
    queue_admitted: false,
  }],
  pause: { available: true, active: false, decision: "CONTINUE",
    no_m7_numerator: 1, scientific_terminal_denominator: 4,
    excluded_attempt_count: 0, excluded_attempts: [], trigger_attempt_ids: [] },
  queue: { available: true, task_count: 0, attempt_count: 0, active_task_ids: [] },
  small_surfaces: { available: true, minimum_area_cm2: 0.1, policy_version: "1.0.0",
    promotion_in_place: "PROHIBITED", surfaces_available: true,
    diagnostic_count: 1, standard_count: 1,
    explicit_non_claims: ["SMALL_SURFACE_DIAGNOSTIC is not a finding of no ink."],
    surfaces: [{ surface_id: "surface-tiny", sample_id: "PHerc358",
      measured_area_cm2: 0.004, route: "SMALL_SURFACE_DIAGNOSTIC",
      why: "0.004 cm2 is below the 0.1 cm2 floor: too small for the standard "
        + "acceptance path, and no claim either way about what is written on it" }] },
  blockers: [{ code: "CONTROL_EVIDENCE_STALE", scope: "first-letters",
    detail: "the control was run against a different deployed revision" }],
  advisories: [],
  allowed_actions: ["CLOSE_CAMPAIGN", "INSPECT_SMALL_SURFACE_DIAGNOSTICS",
    "REFRESH_POSITIVE_CONTROL"],
  queue_admitted: false,
  non_claims: ["Candidate scarcity is not evidence that a scroll holds no "
    + "surface, no ink, and no text."],
  readiness_sha256: "c".repeat(64),
};

const READY = {
  ...BLOCKED,
  control: { ...BLOCKED.control, evidence_status: "CURRENT",
    evidence_status_reason: "the control is bound to the exact deployed revision",
    bound_deployed_revision: "a".repeat(40) },
  scrolls: [{
    ...BLOCKED.scrolls[0],
    preflight: { evidence_status: "CURRENT",
      evidence_status_reason: "all frozen bindings match",
      measurement_kind: "CENSUS", status: "COMPLETE",
      private_receipt_sha256: "d".repeat(64) },
    blockers: [], advisories: [],
    allowed_actions: ["ACCEPT_COMPUTED_TASK_BUDGET"],
  }],
  blockers: [],
  allowed_actions: ["CLOSE_CAMPAIGN", "INSPECT_PAUSE_CAUSES",
    "INSPECT_SMALL_SURFACE_DIAGNOSTICS"],
  queue_admitted: false,
};

let readiness: any = BLOCKED;
let calls: { url: string; init?: any }[] = [];

beforeEach(() => {
  readiness = BLOCKED;
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (input: any, init?: any) => {
    calls.push({ url: String(input), init });
    return {
      ok: true, status: 200,
      json: async () => {
        const url = String(input);
        if (url.startsWith("/api/coverage")) return COVERAGE;
        if (url.includes("first-letters-readiness")) return readiness;
        if (url.startsWith("/api/segmentation/task-budget")) {
          return { planned_task_count: 8, decision: "CONTINUE" };
        }
        return { available: true, by_cause: {}, attempts: 0, note: "" };
      },
    };
  }));
});
afterEach(() => vi.unstubAllGlobals());

function draw(sample = "PHerc358", mission: string | undefined = "first-letters") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <Coverage sample={sample} mission={mission} />
  </QueryClientProvider>);
}

describe("First Letters campaign gates", () => {
  it("names every blocker and the evidence that clears it", async () => {
    draw();
    expect(await screen.findByText("First Letters campaign gates")).toBeTruthy();
    expect(screen.getByText(/CONTROL EVIDENCE STALE/)).toBeTruthy();
    expect(screen.getByText(/PREFLIGHT MISSING/)).toBeTruthy();
    expect(screen.getByText("Refresh the positive control")).toBeTruthy();
    expect(screen.getByText("Run the candidate preflight")).toBeTruthy();
    expect(screen.getByText(/1 blocking/)).toBeTruthy();
  });

  it("offers no way to accept, force or override a blocked gate", async () => {
    draw();
    await screen.findByText("First Letters campaign gates");
    const labels = screen.getAllByRole("button").map(
      (node) => (node.textContent ?? "").toLowerCase());
    for (const banned of ["accept anyway", "force", "override", "bypass",
                          "skip", "queue anyway", "allow unvalidated"]) {
      expect(labels.some((label) => label.includes(banned))).toBe(false);
    }
    // The budget cannot be accepted while the preflight it would be derived
    // from does not exist.
    expect(screen.queryByText("Accept the computed budget")).toBeNull();
  });

  it("draws no gate card at all for a mission with no campaign binding", async () => {
    readiness = { ...BLOCKED, controlled: false };
    draw();
    await waitFor(() => expect(
      calls.some((call) => call.url.includes("first-letters-readiness"))).toBe(true));
    expect(screen.queryByText("First Letters campaign gates")).toBeNull();
  });

  it("asks for no readiness at all when no mission is open", async () => {
    draw("PHerc358", undefined);
    await waitFor(() => expect(calls.length).toBeGreaterThan(0));
    expect(calls.some((call) => call.url.includes("first-letters-readiness")))
      .toBe(false);
  });

  it("accepts the budget the server computed, naming only the compute cap",
     async () => {
    readiness = READY;
    draw();
    await screen.findByText("Accept the computed task budget");
    const button = screen.getByRole("button", { name: "Accept the computed budget" });
    expect((button as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(screen.getByPlaceholderText("first-letters-cap-1"),
                     { target: { value: "first-letters-cap-1" } });
    fireEvent.change(screen.getByPlaceholderText("64"), { target: { value: "64" } });
    expect((button as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(button);

    await waitFor(() => expect(calls.some(
      (call) => call.url === "/api/segmentation/task-budget")).toBe(true));
    const posted = JSON.parse(calls.find(
      (call) => call.url === "/api/segmentation/task-budget")!.init.body);
    expect(posted).toEqual({
      mission_id: "first-letters", sample_id: "PHerc0358",
      preflight_receipt_sha256: "d".repeat(64),
      compute_cap_id: "first-letters-cap-1", compute_cap_tasks: 64,
    });
    // No task count is sent: the operator authorizes compute, not the number.
    expect(Object.keys(posted)).not.toContain("manual_task_count");
    expect(await screen.findByText(/derived a CONTINUE budget of/)).toBeTruthy();
  });

  it("shows the pause causes and the small surfaces only when asked", async () => {
    readiness = READY;
    draw();
    await screen.findByText("First Letters campaign gates");
    expect(screen.queryByText(/scientific-terminal/)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Inspect pause causes" }));
    expect(screen.getByText(/scientific-terminal/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Inspect small surfaces" }));
    expect(screen.getByText(/Area floor 0\.1 cm/)).toBeTruthy();
    expect(screen.getByText(/no claim either way about what is written on it/))
      .toBeTruthy();
    expect(screen.getByText(
      "SMALL_SURFACE_DIAGNOSTIC is not a finding of no ink.")).toBeTruthy();
  });

  it("confirms before closing a campaign and never deletes evidence", async () => {
    readiness = READY;
    const confirm = vi.fn(() => false);
    vi.stubGlobal("confirm", confirm);
    draw();
    await screen.findByText("First Letters campaign gates");
    fireEvent.click(screen.getByRole("button", { name: "Close campaign" }));
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(String(confirm.mock.calls[0])).toContain("receipts stay readable");
    expect(calls.some((call) => call.url.includes("state=archived"))).toBe(false);

    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "Close campaign" }));
    await waitFor(() => expect(calls.some(
      (call) => call.url.includes("state=archived"))).toBe(true));
  });

  it("shows an unrecognised server action as evidence, never as a button",
     async () => {
    readiness = { ...READY, allowed_actions: ["SOMETHING_THIS_PANEL_HAS_NOT_LEARNED"] };
    draw();
    await screen.findByText("First Letters campaign gates");
    expect(screen.getByText(/SOMETHING THIS PANEL HAS NOT LEARNED/)).toBeTruthy();
    const labels = screen.getAllByRole("button").map((node) => node.textContent);
    expect(labels).not.toContain("SOMETHING THIS PANEL HAS NOT LEARNED");
  });

  it("carries the readiness receipt hash and its non-claims", async () => {
    readiness = READY;
    draw();
    await screen.findByText("First Letters campaign gates");
    expect(screen.getByText("c".repeat(64))).toBeTruthy();
    expect(screen.getByText(/is not evidence that a scroll holds no surface/))
      .toBeTruthy();
  });
});
