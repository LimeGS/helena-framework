import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Card, Empty, Info, Pill } from "../components/Bits";

/**
 * How much of a scroll has been looked at, and with what result.
 *
 * The framework is named for exploration and could not answer this. Coverage
 * existed only as a ranking input inside the bootstrap -- how far a candidate
 * cell is from the surfaces already grown -- and was never reported, so progress
 * was read off a surface count, which rises whether the fleet is finding new
 * ground or re-treading old.
 *
 * The column worth reading is the hit rate. On this control plane one grid found
 * a lamina in 30 of 30 cells and another in 1 of 128, and nothing had ever put
 * those two numbers beside each other.
 */

type Grid = {
  sample_id: string; grid_version: string;
  cells_attempted: number; cells_no_seed: number; cells_with_surface: number;
  tasks: number; cells_in_volume: number | null;
  fraction_attempted: number | null; grid_step_xyz: number[] | null;
};
type Coverage = {
  available?: boolean; reason?: string;
  grids: Grid[];
  volumes: { sample_id: string; shape_xyz: number[]; voxel_size_um: number | null;
             surface_area_cm2: number; surfaces: number }[];
  non_claims: string[];
  candidate_preflight?: CandidatePreflight | null;
  campaign_decisions?: CampaignDecision[];
  active_campaign_decision?: CampaignDecision | null;
};
type CampaignDecision = {
  schema: string;
  decision: "CONTINUE" | "PAUSE_CANDIDATE_STARVATION" | "CONTROL_INCOMPLETE";
  evidence_status: "COMPLETE" | "INCOMPLETE" | "IN_PROGRESS";
  mission_id: string; policy_version: string;
  evaluation_kind: string; evaluation_index: number;
  no_m7_numerator: number; scientific_terminal_denominator: number;
  excluded_attempt_count: number;
  excluded_attempts: { attempt_id: string; task_id: string; reason: string }[];
  trigger_attempt_ids: string[];
  receipt_sha256?: string;
  state_sha256?: string;
  policy_chain?: string[];
  allowed_next_actions: string[];
  non_claim: string;
};
type CandidatePreflight = {
  schema: string; evidence_status: "CURRENT" | "STALE" | "INVALID";
  evidence_status_reason?: string;
  measurement_kind?: "CENSUS" | "ESTIMATE" | "INCOMPLETE_CENSUS" | "INCOMPLETE_ESTIMATE";
  planned_sampling_percentage?: number;
  achieved_successful_sampling_percentage?: number;
  funnel?: { total_grid_cells: number; grid_cells_in_design_sample: number;
    geometrically_eligible_cells: number | null;
    geometrically_eligible_cells_estimate: number;
    geometrically_eligible_sampled_cells: number;
    cells_attempted: number; cells_surveyed: number;
    cells_surveyed_successfully: number; cells_failed_source: number;
    cells_with_raw_m7_candidates: number; raw_m7_candidates: number;
    post_ct_candidates: number; post_cell_clearance_candidates: number;
    post_volume_clearance_candidates: number;
    packet_retained_candidates: number; source_errors: number };
  spatial_bins: { bin_xyz: number[]; total_cells: number; surveyed_cells: number;
    candidate_bearing_cells: number; usable_candidate_cells: number }[];
  no_candidate_causes: Record<string, number>;
  non_claim: string;
};

function CandidateAvailability({ preflight }: { preflight: CandidatePreflight }) {
  const evidenceKind = preflight.evidence_status === "CURRENT" ? "ok"
    : preflight.evidence_status === "STALE" ? "warn" : "crit";
  if (preflight.evidence_status === "INVALID" || !preflight.funnel
      || preflight.measurement_kind === undefined
      || preflight.planned_sampling_percentage === undefined) {
    return <Card title="Candidate availability preflight"
      note={<Pill kind="crit">INVALID evidence</Pill>}>
      <div className="body-pad dash">
        {preflight.evidence_status_reason ?? "The latest receipt pair could not be verified."}
        {" "}No measurements from it are shown.
      </div>
    </Card>;
  }
  const funnel = preflight.funnel;
  const incomplete = preflight.measurement_kind.startsWith("INCOMPLETE_");
  const sampled = preflight.measurement_kind.endsWith("ESTIMATE");
  const measurement = incomplete
    ? `incomplete ${sampled ? "estimate" : "census"}`
    : sampled ? "estimated sample" : "exact census";
  const achieved = preflight.achieved_successful_sampling_percentage;
  return <Card title="Candidate availability preflight"
    note={`${measurement} · ${preflight.planned_sampling_percentage.toFixed(1)}% planned${achieved === undefined ? "" : ` · ${achieved.toFixed(1)}% successful`}`}>
    <div className="body-pad"><Pill kind={evidenceKind}>
      {preflight.evidence_status} evidence
    </Pill>{preflight.evidence_status_reason && <span className="dash">
      {" "}{preflight.evidence_status_reason}
    </span>}</div>
    <div className="knobgrid">
      <div><div className="big">{funnel.raw_m7_candidates}</div><div className="dash">raw M7</div></div>
      <div><div className="big">{funnel.post_ct_candidates}</div><div className="dash">post CT</div></div>
      <div><div className="big">{funnel.post_cell_clearance_candidates}</div><div className="dash">post cell clearance</div></div>
      <div><div className="big">{funnel.post_volume_clearance_candidates}</div><div className="dash">post volume clearance</div></div>
      <div><div className="big">{funnel.packet_retained_candidates}</div><div className="dash">packet retained</div></div>
    </div>
    <div className="body-pad dash">
      {funnel.cells_surveyed_successfully} successful of {funnel.cells_attempted} attempted eligible cells
      {` · ${funnel.cells_failed_source} source ${funnel.cells_failed_source === 1 ? "failure" : "failures"}`}
      {funnel.geometrically_eligible_cells === null
        ? ` · estimated eligible population ${funnel.geometrically_eligible_cells_estimate}`
        : ` · exact eligible population ${funnel.geometrically_eligible_cells}`}
    </div>
    {preflight.spatial_bins?.length > 0 && <div className="scroller">
      <table><thead><tr><th>spatial bin</th><th>cells</th><th>surveyed</th>
        <th>candidate-bearing</th><th>usable</th></tr></thead><tbody>
        {preflight.spatial_bins.map((bin) => <tr key={bin.bin_xyz.join(",")}>
          <td className="mono">bin {bin.bin_xyz.join(",")}</td>
          <td>{bin.total_cells}</td><td>{bin.surveyed_cells}</td>
          <td>{bin.candidate_bearing_cells}</td><td>{bin.usable_candidate_cells}</td>
        </tr>)}
      </tbody></table>
    </div>}
    <div className="body-pad dash">{preflight.non_claim}</div>
  </Card>;
}

const words = (value: string) => value.replaceAll("_", " ");

const CONTROL_BOUNDARIES = [
  "P0", "P1", "P2", "QC", "P3", "P4", "P5", "P7", "HUMAN_REVIEW",
] as const;

type ControlProgressEvent = {
  schema: string; run_id: string; mission_id: string;
  event: "run_started" | "boundary_started" | "heartbeat"
       | "boundary_finished" | "run_finished" | "note";
  boundary?: string | null;
  state?: string | null; reason?: string | null;
  control_state?: string | null; first_nonpassing_boundary?: string | null;
  message: string; at_utc: string; received_at_utc: string;
};
type ControlProgressRun = {
  run_id: string; started_at_utc: string | null; last_event_at_utc: string | null;
  last_event: string | null; current_boundary: string | null;
  finished: boolean; control_state: string | null; event_count: number;
};
type ControlProgress = {
  runs: ControlProgressRun[]; run_id: string | null; events: ControlProgressEvent[];
};

type BoundaryProgress = "pending" | "running" | "PASS" | "FAILED" | "INCOMPLETE";

/** What each boundary's own events say has happened to it so far, walked in
 * the order the run posted them -- a later event always overrides an earlier
 * one, so a boundary that started and is still waiting reads "running" until
 * its own `boundary_finished` line, if one ever arrives, replaces it. */
function boundaryProgress(events: ControlProgressEvent[]) {
  const state = new Map<string, { status: BoundaryProgress; reason?: string | null }>(
    CONTROL_BOUNDARIES.map((boundary) => [boundary, { status: "pending" }]));
  for (const event of events) {
    if (!event.boundary || !state.has(event.boundary)) continue;
    if (event.event === "boundary_started" || event.event === "heartbeat") {
      if (state.get(event.boundary)!.status === "pending") {
        state.set(event.boundary, { status: "running" });
      }
    } else if (event.event === "boundary_finished") {
      state.set(event.boundary, {
        status: (event.state as BoundaryProgress) ?? "running", reason: event.reason,
      });
    }
  }
  return state;
}

function progressPill(status: BoundaryProgress): "none" | "run" | "ok" | "crit" | "warn" {
  if (status === "PASS") return "ok";
  if (status === "FAILED") return "crit";
  if (status === "INCOMPLETE") return "warn";
  if (status === "running") return "run";
  return "none";
}

const relativeSeconds = (isoStamp: string | null | undefined, now: number) =>
  isoStamp ? Math.max(0, Math.round((now - Date.parse(isoStamp)) / 1000)) : null;

const humanDuration = (seconds: number) => {
  if (seconds < 90) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  return minutes < 90 ? `${minutes}m` : `${(minutes / 60).toFixed(1)}h`;
};

/**
 * A control run's own narration, live: which of the nine boundaries is
 * running now and for how long, plus the full event log a boundary that ran
 * an hour silent -- P1's grow measured at 66 minutes, P2's finalization at
 * 60 -- would otherwise give no way to tell from a hung process.
 *
 * Polls the panel rather than the runner: the runner may be a process on a
 * machine nobody watching this page has a terminal open to, and the whole
 * point of a persistent channel is that this page does not need one.
 */
function ControlProgress({ mission }: { mission?: string }) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [showLog, setShowLog] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  const query = useQuery<ControlProgress>({
    enabled: Boolean(mission),
    queryKey: ["first-letters-control-progress", mission ?? null, selectedRunId],
    queryFn: async () => {
      const q = selectedRunId ? `?run_id=${encodeURIComponent(selectedRunId)}` : "";
      const response = await fetch(
        `/api/missions/${encodeURIComponent(mission!)}/first-letters-control/progress${q}`);
      if (!response.ok) throw new Error(`progress could not be read: HTTP ${response.status}`);
      return response.json();
    },
    refetchInterval: (query) => (query.state.data?.events?.at(-1)?.event === "run_finished"
      ? false : 4000),
  });

  const data = query.data;
  const events = data?.events ?? [];
  const runId = selectedRunId ?? data?.run_id ?? null;
  const activeRun = data?.runs?.find((row) => row.run_id === runId);
  const finished = events.some((row) => row.event === "run_finished")
    || activeRun?.finished === true;

  // Ticks once a second while a run is in flight, so "still waiting" reads as
  // a live number rather than a value frozen until the next 4-second poll.
  useEffect(() => {
    if (finished) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [finished]);

  // Both derived from `events` alone, not from the 1-second ticker -- without
  // this, every tick rebuilt the boundary map and re-joined the whole log
  // string even though neither had changed since the last poll.
  const boundaries = useMemo(() => boundaryProgress(events), [events]);
  const logText = useMemo(() => events.map((row) =>
      `[${row.at_utc}] ${row.boundary ? `${row.boundary} ` : ""}${row.event}`
      + (row.state ? ` -> ${row.state}` : "")
      + (row.reason ? ` (${row.reason})` : "")
      + `  ${row.message}`
    ).join("\n"), [events]);

  if (!mission || query.isLoading || query.isError || !data || !runId
      || !Array.isArray(data.runs) || !Array.isArray(data.events)) return null;

  const startedAt = events[0]?.at_utc ?? activeRun?.started_at_utc ?? null;
  const lastEvent = events.at(-1);
  const running = CONTROL_BOUNDARIES.find(
    (boundary) => boundaries.get(boundary)?.status === "running");
  const elapsed = relativeSeconds(startedAt, finished
    ? Date.parse(lastEvent?.at_utc ?? startedAt ?? "") : now);
  const sinceLastEvent = finished ? null
    : relativeSeconds(lastEvent?.received_at_utc ?? lastEvent?.at_utc, now);

  return <div className="body-pad">
    <Pill kind={finished
      ? (lastEvent?.control_state === "CONTROL_PASS" ? "ok" : "crit") : "run"}>
      {finished ? `finished · ${lastEvent?.control_state ?? "unknown"}`
        : running ? `running · ${running}` : "in progress"}
    </Pill>{" "}
    {elapsed !== null && <span className="dash">
      {finished ? "took" : "elapsed"} {humanDuration(elapsed)}
    </span>}
    {!finished && sinceLastEvent !== null && <span className="dash">
      {" "}· last update {humanDuration(sinceLastEvent)} ago
    </span>}
    <div className="dash mono">run {runId}</div>

    <div className="body-pad" style={{ display: "flex", gap: "0.375rem", flexWrap: "wrap" }}>
      {CONTROL_BOUNDARIES.map((boundary) => {
        const row = boundaries.get(boundary)!;
        return <Pill key={boundary} kind={progressPill(row.status)}>
          {boundary}{row.status !== "pending" && row.status !== "running"
            ? ` · ${words(row.status)}` : ""}
        </Pill>;
      })}
    </div>

    {data.runs.length > 1 && <div className="dash">
      Earlier runs:{" "}
      {data.runs.map((row) => <button key={row.run_id} className="linklike"
          onClick={() => setSelectedRunId(row.run_id)}
          disabled={row.run_id === runId}>
        {row.run_id}{row.finished
          ? ` (${row.control_state ?? "?"})`
          : ` (in progress${row.current_boundary ? ` · ${row.current_boundary}` : ""})`}
      </button>)}
    </div>}

    <button className="linklike" onClick={() => setShowLog((v) => !v)}>
      {showLog ? "hide" : "show"} the full log ({events.length} events)
    </button>
    {showLog && <pre className="progresslog">{logText}</pre>}
  </div>;
}

type Blocker = { code: string; scope: string; detail: string };
type ReadinessControl = {
  available: boolean; evidence_status: string; evidence_status_reason: string;
  control_state: string | null; control_id: string | null;
  content_sha256: string | null; first_nonpassing_boundary: string | null;
  bound_deployed_revision: string | null;
  stages: { boundary: string; terminal_state: string; reason_code: string }[];
};
type ReadinessScroll = {
  sample_id: string; requested_sample_id: string;
  preflight: { evidence_status: string; evidence_status_reason: string;
               measurement_kind: string | null; status: string | null;
               private_receipt_sha256: string | null };
  budget: { evidence_status: string; decision: string | null;
            receipt_sha256: string | null; planned_task_count: number | null;
            requested_task_count: number | null;
            clipped_by_compute_cap: boolean;
            achieved_detection_probability?: number | null;
            target_detection_probability?: number | null;
            binds_current_preflight: boolean };
  blockers: Blocker[]; advisories: Blocker[];
  allowed_actions: string[]; queue_admitted: boolean;
};
type Readiness = {
  schema?: string; controlled?: boolean; reason?: string | null;
  mission_id: string;
  deployed_revision: string | null;
  mission_deployed_revision?: string | null;
  control: ReadinessControl | null;
  scrolls: ReadinessScroll[];
  pause: (CampaignDecision & { active: boolean; available: boolean;
                               reason?: string | null }) | null;
  queue: { available: boolean; reason?: string | null;
           task_count: number; attempt_count: number;
           active_task_ids: string[] } | null;
  small_surfaces: {
    available: boolean; reason?: string | null;
    minimum_area_cm2?: number; policy_version?: string;
    promotion_in_place?: string;
    surfaces_available?: boolean; surfaces_reason?: string | null;
    diagnostic_count?: number; standard_count?: number;
    explicit_non_claims?: string[];
    surfaces?: { surface_id: string; sample_id: string;
                 measured_area_cm2: number | null; route: string | null;
                 why: string }[];
  } | null;
  blockers: Blocker[]; advisories: Blocker[];
  allowed_actions: string[]; queue_admitted: boolean;
  non_claims: string[]; readiness_sha256: string;
};

/**
 * The seven things an operator may do, and what each one is.
 *
 * Nothing outside this table is drawable, and the table is only consulted for
 * codes the server put in `allowed_actions`. That is the whole design: the
 * control plane decides which action its evidence supports, and the page can
 * only render what it was offered. A future action code that nobody has taught
 * this page about appears as text, never as a button, so no new server string
 * can accidentally become a clickable way around a gate.
 *
 * There is deliberately no entry for accepting a blocked campaign, forcing a
 * queue, or overriding a gate. The way past a blocker is to produce the
 * evidence it names.
 */
const ACTIONS: Record<string, { label: string; note: string }> = {
  REFRESH_POSITIVE_CONTROL: {
    label: "Refresh the positive control",
    note: "Run scripts/harness/run_first_letters_positive_control.py against this "
      + "deployment. It publishes its own stage-survival matrix, and the server "
      + "re-derives that matrix before believing it.",
  },
  RUN_CANDIDATE_PREFLIGHT: {
    label: "Run the candidate preflight",
    note: "A source-locked full-grid survey of what the current source exposes. "
      + "Start it from P1 → New run, which is where its grid and clearance "
      + "choices are made.",
  },
  ACCEPT_COMPUTED_TASK_BUDGET: {
    label: "Accept the computed task budget",
    note: "The server derives the budget from the current preflight. Accepting "
      + "authorizes the compute; it does not choose the number.",
  },
  INCREASE_FROZEN_COMPUTE_CAP: {
    label: "Raise the frozen compute cap",
    note: "The budget was clipped, so the wave reaches a lower detection "
      + "probability than the target. Re-derive it against a larger cap.",
  },
  INSPECT_PAUSE_CAUSES: {
    label: "Inspect the pause causes",
    note: "The attempts that triggered the starvation rule, below.",
  },
  CREATE_NEW_VERSIONED_STRATEGY: {
    label: "Create a new versioned strategy",
    note: "A paused campaign resumes only through a successor policy whose "
      + "material change is hash-bound and attested to a signed-in principal. "
      + "Create it through the fleet bootstrap; it cannot be done from here.",
  },
  CHANGE_CANDIDATE_SOURCE_OR_POLICY: {
    label: "Change the candidate source or policy",
    note: "The census found no usable candidate-bearing cell in this source. "
      + "That is a statement about this source, not about the scroll.",
  },
  RUN_LARGER_PREFLIGHT: {
    label: "Run a larger preflight",
    note: "The sampled estimate saw no usable cell. A larger sample, or a "
      + "different source, is what settles it.",
  },
  INSPECT_SMALL_SURFACE_DIAGNOSTICS: {
    label: "Inspect the small surfaces",
    note: "Surfaces below the area floor, and what that does and does not mean.",
  },
  QUEUE_NEXT_BOUND_WAVE: {
    label: "Queue the next bounded wave",
    note: "Every gate holds. Run scripts/harness/run_first_letters_campaign.py, "
      + "which enqueues exactly the budget, waits for the wave to finish, and "
      + "re-reads this answer before considering another one.",
  },
  CLOSE_CAMPAIGN: {
    label: "Close the campaign",
    note: "Archive the mission. Its receipts stay readable.",
  },
};

function Offered({ codes }: { codes: string[] }) {
  if (!codes.length) return null;
  return <div className="body-pad">
    <div className="dash">What may be done next</div>
    <ul className="dash">
      {codes.map((code) => <li key={code}>
        <strong>{ACTIONS[code]?.label ?? words(code)}</strong>
        {ACTIONS[code] ? <> — {ACTIONS[code].note}</> : <> — this deployment has
          no description for this action, so it is shown as evidence only.</>}
      </li>)}
    </ul>
  </div>;
}

function pill(status: string): "ok" | "warn" | "crit" {
  if (status === "CURRENT" || status === "CONTROL_PASS") return "ok";
  if (status === "STALE" || status === "MISSING") return "warn";
  return "crit";
}

/**
 * Whether this campaign may queue, and the evidence for the answer.
 *
 * Four gates were enforced in four places and reported in none, so "why did the
 * queue refuse?" was answered by reading server logs. Every blocker here names
 * the evidence that is missing and the action that produces it.
 *
 * Nothing on this card accepts, overrides or bypasses a gate.
 */
function FirstLettersGates({ mission, sample }:
                           { mission?: string; sample?: string }) {
  const client = useQueryClient();
  const [capId, setCapId] = useState("");
  const [capTasks, setCapTasks] = useState("");
  const [showPause, setShowPause] = useState(false);
  const [showSurfaces, setShowSurfaces] = useState(false);

  const query = useQuery<Readiness>({
    enabled: Boolean(mission),
    queryKey: ["first-letters-readiness", mission ?? null],
    queryFn: async () => {
      const response = await fetch(
        `/api/missions/${encodeURIComponent(mission!)}/first-letters-readiness`);
      if (!response.ok && response.status !== 409) {
        throw new Error(`readiness could not be read: HTTP ${response.status}`);
      }
      return response.json();
    },
  });

  const scroll = query.data?.scrolls?.find(
    (row) => row.sample_id === sample || row.requested_sample_id === sample)
    ?? query.data?.scrolls?.[0];

  const acceptBudget = useMutation({
    mutationFn: async () => {
      const response = await fetch("/api/segmentation/task-budget", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mission_id: mission,
          sample_id: scroll?.requested_sample_id,
          preflight_receipt_sha256: scroll?.preflight.private_receipt_sha256,
          compute_cap_id: capId.trim(),
          compute_cap_tasks: Number(capTasks),
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail
          : JSON.stringify(body.detail ?? `HTTP ${response.status}`));
      }
      return body as { planned_task_count: number; decision: string };
    },
    onSuccess: () => client.invalidateQueries(
      { queryKey: ["first-letters-readiness"] }),
  });

  const close = useMutation({
    mutationFn: async () => {
      const response = await fetch(
        `/api/missions/${encodeURIComponent(mission!)}/state?state=archived`,
        { method: "POST" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
    onSuccess: () => client.invalidateQueries(
      { queryKey: ["first-letters-readiness"] }),
  });

  if (!mission || query.isLoading || query.isError) return null;
  const data = query.data;
  // Not a controlled campaign, or an answer this page does not understand.
  // Silence is right here: an ordinary mission has no campaign gate.
  if (!data || data.controlled !== true) return null;

  const offered = new Set(data.allowed_actions ?? []);
  const scrollOffered = new Set(scroll?.allowed_actions ?? []);
  const canAcceptBudget = scrollOffered.has("ACCEPT_COMPUTED_TASK_BUDGET");
  const control = data.control;
  const admitted = data.queue_admitted;

  return <Card title="First Letters campaign gates"
    note={<Pill kind={admitted ? "ok" : "warn"}>
      {admitted ? "every gate holds" : `${data.blockers.length} blocking`}
    </Pill>}>
    <div className="body-pad">
      <div className="dash">Deployed revision</div>
      <div className="mono">{data.deployed_revision ?? "unresolved"}</div>
      {data.mission_deployed_revision !== data.deployed_revision
        && <div className="dash">
          The mission was bound to {data.mission_deployed_revision}.
        </div>}
    </div>

    {control && <div className="body-pad">
      <Pill kind={pill(control.evidence_status)}>
        {words(control.evidence_status)} control
      </Pill>{" "}
      <Pill kind={pill(control.control_state ?? "MISSING")}>
        {words(control.control_state ?? "no control published")}
      </Pill>
      <div className="dash">{control.evidence_status_reason}</div>
      {control.first_nonpassing_boundary && <div className="dash">
        First boundary that did not pass: {control.first_nonpassing_boundary}
      </div>}
      {control.content_sha256 && <div className="mono">{control.content_sha256}</div>}
    </div>}

    <ControlProgress mission={mission} />

    {scroll && <div className="body-pad">
      <div className="dash">{scroll.sample_id}</div>
      <Pill kind={pill(scroll.preflight.evidence_status)}>
        {words(scroll.preflight.evidence_status)} preflight
        {scroll.preflight.measurement_kind
          ? ` · ${words(scroll.preflight.measurement_kind).toLowerCase()}` : ""}
      </Pill>{" "}
      <Pill kind={pill(scroll.budget.evidence_status)}>
        {words(scroll.budget.evidence_status)} budget
        {scroll.budget.planned_task_count === null ? ""
          : ` · ${scroll.budget.planned_task_count} tasks`}
      </Pill>
      <div className="dash">{scroll.preflight.evidence_status_reason}</div>
      {scroll.budget.clipped_by_compute_cap && <div className="dash">
        The frozen compute cap allows {scroll.budget.planned_task_count} of the
        {" "}{scroll.budget.requested_task_count} tasks the target detection
        probability needs, so this wave reaches a lower probability than the target.
      </div>}
    </div>}

    {(data.blockers.length > 0 || (scroll?.blockers.length ?? 0) > 0)
      && <div className="body-pad">
      <div className="dash">Blocking</div>
      <ul className="dash">
        {[...data.blockers, ...(scroll?.blockers ?? [])].map((row) =>
          <li key={`${row.code}:${row.scope}`}>
            <strong>{words(row.code)}</strong> — {row.detail}
          </li>)}
      </ul>
    </div>}

    {(data.advisories?.length ?? 0) > 0 && <div className="body-pad">
      <div className="dash">Worth knowing</div>
      <ul className="dash">
        {data.advisories.map((row) => <li key={`${row.code}:${row.scope}`}>
          <strong>{words(row.code)}</strong> — {row.detail}
        </li>)}
      </ul>
    </div>}

    <Offered codes={[...new Set([...(data.allowed_actions ?? []),
                                 ...(scroll?.allowed_actions ?? [])])]} />

    <div className="controls">
      {canAcceptBudget && <>
        <label>compute cap id
          <input value={capId} onChange={(event) => setCapId(event.target.value)}
                 placeholder="first-letters-cap-1" />
        </label>
        <label>cap tasks
          <input value={capTasks} inputMode="numeric"
                 onChange={(event) => setCapTasks(event.target.value)}
                 placeholder="64" />
        </label>
        <button onClick={() => acceptBudget.mutate()}
                disabled={acceptBudget.isPending || !capId.trim()
                          || !Number.isInteger(Number(capTasks))
                          || Number(capTasks) < 0}>
          {acceptBudget.isPending ? "deriving…" : "Accept the computed budget"}
        </button>
      </>}
      {offered.has("INSPECT_PAUSE_CAUSES") && <button
        onClick={() => setShowPause((shown) => !shown)}>
        {showPause ? "Hide pause causes" : "Inspect pause causes"}
      </button>}
      {offered.has("INSPECT_SMALL_SURFACE_DIAGNOSTICS") && <button
        onClick={() => setShowSurfaces((shown) => !shown)}>
        {showSurfaces ? "Hide small surfaces" : "Inspect small surfaces"}
      </button>}
      {offered.has("CLOSE_CAMPAIGN") && <button
        onClick={() => {
          if (window.confirm(
            `Archive mission ${mission}?\n\nIts receipts stay readable. No `
            + "attempt, artifact or decision is deleted.")) close.mutate();
        }} disabled={close.isPending}>
        {close.isPending ? "closing…" : "Close campaign"}
      </button>}
    </div>
    {acceptBudget.isError && <div className="body-pad">
      <Pill kind="crit">{String(acceptBudget.error)}</Pill>
    </div>}
    {acceptBudget.isSuccess && <div className="body-pad dash">
      The server derived a {acceptBudget.data.decision} budget of
      {" "}{acceptBudget.data.planned_task_count} tasks.
    </div>}
    {close.isError && <div className="body-pad">
      <Pill kind="crit">{String(close.error)}</Pill>
    </div>}

    {showPause && data.pause && <div className="body-pad dash">
      <div>Campaign gate: {words(data.pause.decision ?? "none")}
        {data.pause.active ? " · blocking" : " · not blocking"}</div>
      <div>{data.pause.no_m7_numerator} of
        {" "}{data.pause.scientific_terminal_denominator} scientific-terminal
        attempts in the block being evaluated recorded zero raw M7.</div>
      <div>Triggering attempts: {data.pause.trigger_attempt_ids?.length
        ? data.pause.trigger_attempt_ids.join(", ") : "none"}</div>
      <div>Excluded platform or control outcomes:
        {" "}{data.pause.excluded_attempt_count ?? 0}</div>
    </div>}

    {showSurfaces && data.small_surfaces && <div className="body-pad dash">
      {data.small_surfaces.available === false
        ? <div>{data.small_surfaces.reason}</div>
        : <>
          <div>Area floor {data.small_surfaces.minimum_area_cm2} cm² ·
            {" "}{data.small_surfaces.diagnostic_count} diagnostic ·
            {" "}{data.small_surfaces.standard_count} standard ·
            {" "}in-place promotion {String(
              data.small_surfaces.promotion_in_place).toLowerCase()}</div>
          {data.small_surfaces.surfaces_available === false && <div>
            {data.small_surfaces.surfaces_reason}
          </div>}
          {(data.small_surfaces.surfaces ?? []).map((row) => <div key={row.surface_id}>
            <span className="mono">{row.surface_id}</span> ·
            {" "}{row.measured_area_cm2} cm² · {words(row.route ?? "unknown")} ·
            {" "}{row.why}
          </div>)}
          <ul>
            {(data.small_surfaces.explicit_non_claims ?? []).map((claim) =>
              <li key={claim}>{claim}</li>)}
          </ul>
        </>}
    </div>}

    <div className="body-pad">
      <div className="dash">Readiness receipt</div>
      <div className="mono">{data.readiness_sha256}</div>
    </div>
    <ul className="body-pad dash">
      {(data.non_claims ?? []).map((claim) => <li key={claim}>{claim}</li>)}
    </ul>
  </Card>;
}

function CampaignGate({ decision }: { decision: CampaignDecision }) {
  const kind = decision.decision === "CONTINUE" ? "ok"
    : decision.decision === "CONTROL_INCOMPLETE" ? "warn" : "crit";
  return <Card title="Campaign decision"
    note={`${decision.policy_version} · ${words(decision.evaluation_kind).toLowerCase()} ${decision.evaluation_index}`}>
    <div className="body-pad">
      <Pill kind={kind}>{words(decision.decision)}</Pill>
      <span className="dash"> {decision.evidence_status.toLowerCase()} evidence</span>
    </div>
    <div className="knobgrid">
      <div>
        <div className="big">{decision.no_m7_numerator} of {decision.scientific_terminal_denominator} scientific terminal attempts</div>
        <div className="dash">recorded zero raw M7</div>
      </div>
      <div>
        <div className="big">{decision.excluded_attempt_count}</div>
        <div className="dash">excluded platform or control outcomes</div>
      </div>
    </div>
    {decision.excluded_attempts.length > 0 && <div className="body-pad dash">
      Excluded: {decision.excluded_attempts.map((row) =>
        `${words(row.reason)} (${row.attempt_id})`).join(", ")}
    </div>}
    <div className="body-pad dash">
      Triggering attempts: {decision.trigger_attempt_ids.length > 0
        ? decision.trigger_attempt_ids.join(", ") : "none"}
    </div>
    <div className="body-pad">
      <div className="dash">{decision.receipt_sha256
        ? "Decision receipt" : "Server-derived live state"}</div>
      <div className="mono">{decision.receipt_sha256 ?? decision.state_sha256}</div>
    </div>
    <div className="body-pad dash">
      Allowed next actions: {decision.allowed_next_actions.map(words).join(", ")}
    </div>
    <div className="body-pad dash">{decision.non_claim}</div>
  </Card>;
}

// A hit rate is not a quality: it says the planner found a seed worth growing
// in that cell, and nothing about what grew.
const hitKind = (rate: number): "ok" | "warn" | "crit" =>
  rate >= 0.5 ? "ok" : rate >= 0.1 ? "warn" : "crit";

const today = () => new Date().toISOString().slice(0, 10).replaceAll("-", "");

/**
 * Ask the cells that gave nothing again, under a different policy.
 *
 * The worker records for every NO_SEED how many candidates the provider offered
 * and which screen removed them, and until this existed nothing read it back:
 * the cell was terminal, the next bootstrap chose cells by distance from known
 * surfaces, and the fleet explored without learning.
 *
 * Task identity is (snapshot, grid, cell, policy) behind an ON CONFLICT DO
 * NOTHING, so re-asking under the same policy inserts nothing and looks like it
 * worked. The versions are prefilled with today's date to make that hard to do
 * by accident, and the fleet refuses it outright either way.
 */
function Replan({ sample, mission }: { sample?: string; mission?: string }) {
  const client = useQueryClient();
  const [grid, setGrid] = useState(`replan-${today()}`);
  const [policy, setPolicy] = useState(`replan-${today()}`);
  const [planner, setPlanner] = useState("");
  const [limit, setLimit] = useState("50");
  const [causes, setCauses] = useState<string[]>([]);
  const [dry, setDry] = useState(true);

  const diagnosis = useQuery<{ available: boolean; by_cause: Record<string, number>;
                               attempts: number; note: string }>({
    queryKey: ["no-seed-causes", sample ?? null, mission ?? null],
    queryFn: async () => {
      const q = new URLSearchParams();
      if (sample) q.set("sample", sample);
      if (mission) q.set("mission", mission);
      const response = await fetch(
        "/api/segmentation/no-seed" + (q.size ? `?${q}` : ""));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
  });
  const planners = useQuery<{ planners?: { id: string; name: string }[] }>({
    queryKey: ["segmentation-options"],
    queryFn: async () => (await fetch("/api/segmentation/options")).json(),
    staleTime: 300_000,
  });

  const run = useMutation({
    mutationFn: async () => {
      const response = await fetch("/api/segmentation/replan", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grid_version: grid, policy_version: policy,
          ...(planner ? { planner } : {}),
          ...(sample ? { sample_id: sample } : {}),
          ...(mission ? { mission_id: mission } : {}),
          causes, limit: Number(limit) || 50, dry_run: dry,
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail
          : body.detail?.stderr_tail?.trim().split("\n").slice(-1)[0]
            ?? `HTTP ${response.status}`);
      }
      return body as { considered: number; queued?: number; would_queue?: number };
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["coverage"] }),
  });

  const byCause = diagnosis.data?.by_cause ?? {};
  const outcome = run.data;

  return (
    <Card title={<>Re-ask the cells that gave no seed <Info
            label="What re-asking does and does not establish" title="Re-asking">
            A new policy version over the same ground, usually with a different
            planner — the planner is what decides the seed, so re-asking with the
            same one mostly gets the same answer. Re-asking is not evidence that a
            lamina is there, and nothing here changes the prediction volume or its
            threshold: a cell that failed for NO_M7_CANDIDATES fails the same way
            unless what the provider offers changes.
            {" "}The causes below select only cells whose diagnosis names one of
            them: a cell where the provider offered nothing and a cell where every
            candidate was rejected on clearance are different problems.
          </Info></>}
          note={diagnosis.data?.attempts
            ? `${diagnosis.data.attempts} attempts ended NO_SEED` : undefined}>
      <div className="formgrid">
        <label>
          Grid version *
          <input value={grid} onChange={(e) => setGrid(e.target.value)} />
          <span className="dash">a new coverage universe; the same one is a no-op</span>
        </label>
        <label>
          Policy version *
          <input value={policy} onChange={(e) => setPolicy(e.target.value)} />
        </label>
        <label>
          Planner
          <select value={planner} onChange={(e) => setPlanner(e.target.value)}>
            <option value="">whatever the task carried</option>
            {(planners.data?.planners ?? []).map((p) => (
              <option key={p.id} value={p.id}>{p.name ?? p.id}</option>
            ))}
          </select>
        </label>
        <label>
          How many cells
          <input value={limit} onChange={(e) => setLimit(e.target.value)} size={6} />
        </label>
      </div>
      {Object.keys(byCause).length > 0 && (
        <div className="body-pad">
          {Object.entries(byCause).map(([cause, count]) => (
            <label className="inline" key={cause}>
              <input type="checkbox" checked={causes.includes(cause)}
                     onChange={(e) => setCauses(e.target.checked
                       ? [...causes, cause] : causes.filter((c) => c !== cause))} />
              {cause.replaceAll("_", " ").toLowerCase()} ({count})
            </label>
          ))}
        </div>
      )}
      <div className="controls">
        <label className="inline">
          <input type="checkbox" checked={dry}
                 onChange={(e) => setDry(e.target.checked)} />
          list only, do not queue
        </label>
        <button onClick={() => run.mutate()}
                disabled={run.isPending || !grid.trim() || !policy.trim()
                          || Boolean(mission && !sample)}>
          {run.isPending ? "asking…" : "Replan"}
        </button>
        {run.isError && <Pill kind="crit">{String(run.error)}</Pill>}
        {outcome && (
          <span className="dash">
            considered {outcome.considered}
            {outcome.would_queue !== undefined
              ? ` · ${outcome.would_queue} would be queued (listed only)`
              : ` · ${outcome.queued} queued`}
          </span>
        )}
      </div>
    </Card>
  );
}

export default function Coverage({ sample, mission }:
                                 { sample?: string; mission?: string }) {
  const query = useQuery<Coverage>({
    queryKey: ["coverage", sample ?? null, mission ?? null],
    queryFn: async () => {
      const q = new URLSearchParams();
      if (sample) q.set("sample", sample);
      if (mission) q.set("mission", mission);
      const response = await fetch("/api/coverage" + (q.size ? `?${q}` : ""));
      // No throw on a refusal: the body carries the reason and the page shows
      // it, which is the difference between "unexplored" and "not readable".
      if (!response.ok && response.status !== 409) {
        throw new Error(`coverage could not be read: HTTP ${response.status}`);
      }
      return response.json();
    },
  });

  if (query.isLoading) return <Empty>loading…</Empty>;
  if (query.isError) return <Empty>{String(query.error)}</Empty>;
  const data = query.data!;
  if (data.available === false) return <Empty>{data.reason ?? "no control plane"}</Empty>;
  if (!data.grids?.length && !data.candidate_preflight
      && !data.active_campaign_decision && !data.campaign_decisions?.length)
    // The campaign gates still draw above the empty state: "nothing attempted
    // yet" and "the control is stale, so nothing may be attempted" are
    // different sentences, and the second is the one an operator opening an
    // idle campaign came here for.
    return <>
      <FirstLettersGates mission={mission} sample={sample} />
      <Empty>no cells have been attempted here yet</Empty>
    </>;

  const attempted = data.grids.reduce((total, g) => total + g.cells_attempted, 0);
  const withSurface = data.grids.reduce((total, g) => total + g.cells_with_surface, 0);

  return (
    <>
      <FirstLettersGates mission={mission} sample={sample} />
      {data.active_campaign_decision
        ? <CampaignGate decision={data.active_campaign_decision} /> : null}
      {data.campaign_decisions?.length ? <Card title="Campaign decision history">
        {data.campaign_decisions.map((decision) => <div className="body-pad dash"
          key={decision.receipt_sha256 ?? `${decision.policy_version}-${decision.evaluation_index}`}>
          <div>{decision.policy_version}: {words(decision.decision)} ({decision.receipt_sha256})</div>
          <div>{decision.non_claim}</div>
        </div>)}
      </Card> : null}
      {data.candidate_preflight && <CandidateAvailability preflight={data.candidate_preflight} />}
      <Card title="What has been looked at">
        <div className="knobgrid">
          <div>
            <div className="big">{attempted}</div>
            <div className="dash">cells attempted, across {data.grids.length} grids</div>
          </div>
          <div>
            <div className="big">{withSurface}</div>
            <div className="dash">cells that produced a surface</div>
          </div>
          <div>
            <div className="big">
              {data.volumes?.[0]?.surface_area_cm2 ?? 0} cm²
            </div>
            <div className="dash">
              surface area, an upper bound: overlap below the deduplication
              threshold is counted twice
            </div>
          </div>
        </div>
      </Card>

      <Card title="Per grid" note="a grid version is a coverage universe; cells under two of them are not the same cells">
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th>grid</th><th>step</th><th>attempted</th><th>of the volume</th>
                <th>with a surface</th><th>no seed</th><th>hit rate</th>
              </tr>
            </thead>
            <tbody>
              {data.grids.map((grid) => {
                const rate = grid.cells_attempted
                  ? grid.cells_with_surface / grid.cells_attempted : 0;
                return (
                  <tr key={`${grid.sample_id}:${grid.grid_version}`}>
                    <td className="mono">{grid.grid_version}</td>
                    <td className="mono">
                      {grid.grid_step_xyz ? grid.grid_step_xyz[0] : "—"}
                    </td>
                    <td>{grid.cells_attempted}</td>
                    <td className="dash">
                      {grid.cells_in_volume
                        ? `${grid.cells_in_volume} · ${((grid.fraction_attempted ?? 0) * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td>{grid.cells_with_surface}</td>
                    <td>{grid.cells_no_seed}</td>
                    <td><Pill kind={hitKind(rate)}>{(rate * 100).toFixed(0)}%</Pill></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Replan sample={sample} mission={mission} />

      <Card title="What this does not say">
        <ul className="dash">
          {(data.non_claims ?? []).map((claim) => <li key={claim}>{claim}</li>)}
        </ul>
      </Card>
    </>
  );
}
