import { useQuery } from "@tanstack/react-query";
import { scoped, useMission } from "./mission";

/**
 * Cache policy is split by how the data actually behaves.
 *
 * A receipt is written once and never edited, so its query is `staleTime:
 * Infinity` -- revisiting a run costs nothing. Host and fleet state change on
 * their own, so only `/api/state` polls. Polling everything on one interval is
 * the usual way a dashboard like this ends up hammering a database that had
 * nothing new to say.
 */

const FOREVER = Number.POSITIVE_INFINITY;

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} at ${url}`);
  return response.json() as Promise<T>;
}

/**
 * What went wrong, from a response that failed.
 *
 * The pattern this replaces was `throw new Error((await r.json()).detail ?? …)`,
 * copied into six places. It reads as defensive and is not: when the body is not
 * JSON the parse throws first, so the fallback after `??` never runs and the
 * parse error is what reaches the screen. FastAPI's unhandled-exception
 * response is the plain text "Internal Server Error", which is why a 500 in the
 * credentials endpoint surfaced as
 * `SyntaxError: Unexpected token 'I', "Internal S"... is not valid JSON`
 * -- a message about the shape of the error instead of the error.
 *
 * Text first, then try to find a detail in it. A 500 says 500.
 */
export async function failure(response: Response): Promise<Error> {
  const body = await response.text().catch(() => "");
  let detail = "";
  try {
    detail = String(JSON.parse(body)?.detail ?? "");
  } catch {
    // Not JSON, which is the case this exists for.
  }
  return new Error(detail || `HTTP ${response.status} ${response.statusText}`.trim());
}

export type Target = {
  sample_id: string;
  pixel_um: string;
  energy_kev: number | null;
  runs: number;
  lane: string | null;
  p90: number | null;
  run_id: string | null;
  higher_res: boolean;
  verdict: "SCREENED" | "NOT_SCREENED";
};
export type Finding = {
  run_id: string;
  sample_id: string;
  lane: string;
  kind: string;
  detail: string;
  severity: "critical" | "warning";
};
// One row per ink worker that has ever polled, from InkJobStore.workers().
// `gpu_visible` is the fresh, per-poll nvidia-smi answer -- true/false for a
// worker that claims a GPU, null for one that has never claimed one at all.
// `state` alone cannot say a GPU is gone: helena-ink-0 kept POLLING for five
// hours after its container's device passthrough broke, because the claim
// loop that writes last_poll_at never stopped running.
export type InkWorker = {
  worker_id: string;
  host_id: string;
  runtime: string | null;
  phases: string[];
  last_poll_at: string | null;
  last_claim_at: string | null;
  seconds_since_poll: number;
  state: "POLLING" | "SILENT";
  gpu_visible: boolean | null;
};
export type Fleet = {
  available: boolean;
  reason?: string;
  tasks?: number;
  attempts?: number;
  surfaces?: number;
  events?: number;
  leased?: number;
  stale_leases?: number;
  task_states?: { state: string; count: number }[];
  events_by_type?: { type: string; count: number }[];
  workers?: InkWorker[];
  workers_silent?: InkWorker[];
  surfaces_by_sample?: { sample_id: string; count: number; area_cm2: number }[];
};
export type State = {
  generated_at: string;
  fleet: Fleet;
  integrity: Finding[];
  targets: Target[];
  run_count: number;
  lane_count: number;
};
export type Liveness = { verdict: string; reason?: string; interpretation?: string; metrics?: Record<string, number> };
/** One threshold's forward/reverse pixel-count ratio, or the reason it is
 *  absent. `ratio` is null -- never a fabricated 0 or 1 -- below 300
 *  over-threshold pixels on either side; `reason` is set only then. */
export type AsymmetryThreshold = {
  forward_over_px: number;
  reverse_over_px: number;
  ratio: number | null;
  reason?: string;
};
/** How the forward/reverse pixel-count ratio moves as the threshold rises.
 *  Only present for a `direction: both` run whose forward and reverse maps
 *  share a shape; growing across 0.5 -> 0.6 -> 0.7 is what real ink looked
 *  like on the control this was measured against, and flat or falling is
 *  what a shuffled or out-of-domain stack looked like. `sustained_above_1_5`
 *  is their own reading of the numbers at 0.6 and 0.7, reported rather than
 *  enforced -- nothing here gates a job on it. */
export type Asymmetry = {
  thresholds: Record<"0.5" | "0.6" | "0.7", AsymmetryThreshold>;
  sustained_above_1_5: boolean;
};
export type Run = {
  run_id: string;
  schema: string;
  sample_id: string;
  lane_id: string;
  checkpoint_sha: string;
  generated_at: string;
  stats: Record<string, number>;
  clip_value: number | null;
  divisor: number | null;
  normalization: string;
  maps: string[];
  contract_ok: boolean;
  liveness: Liveness | null;
  receipt_path: string;
};
export type RunDetail = Run & { receipt: Record<string, unknown>; profile: Profile | null };
export type Profile = {
  profile_id: string;
  method_id: string;
  adapter: string;
  checkpoint_sha256: string;
  input_contract: Record<string, unknown>;
  registry_status?: string | null;
  registry_policy?: string | null;
  disqualified?: boolean;
};
export type MapMeta = {
  width: number;
  height: number;
  valid_pixels: number;
  p50?: number;
  p90?: number;
  p99?: number;
  min?: number;
  max?: number;
  fraction_above_0_5?: number;
};

/**
 * The dashboard's numbers, for the mission that is open.
 *
 * This asked for /api/state with no mission at all, while every neighbouring
 * hook used `scoped`. So the one field the endpoint already knew how to filter
 * received nothing, and the mission page showed the whole host: another
 * scroll's surfaces, another scroll's tasks, and that scroll named underneath.
 * The mission id is in the query key too, or switching missions would serve the
 * previous one's cached numbers.
 */
export const useAppState = () => {
  const { missionId } = useMission();
  return useQuery({
    queryKey: ["state", missionId],
    queryFn: () => get<State>(scoped("/api/state", missionId)),
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
    staleTime: 2500,
  });
};

export const useRuns = () => {
  const { missionId } = useMission();
  return useQuery({
    queryKey: ["runs", missionId],
    queryFn: () => get<{ runs: Run[] }>(scoped("/api/runs", missionId)),
    select: (d) => d.runs,
    staleTime: 30_000,
  });
};

export const useRun = (runId: string | undefined) =>
  useQuery({
    queryKey: ["run", runId],
    queryFn: () => get<RunDetail>(`/api/run/${encodeURIComponent(runId!)}`),
    enabled: Boolean(runId),
    staleTime: FOREVER,
  });

export const useMapMeta = (runId: string | undefined, name: string | undefined) =>
  useQuery({
    queryKey: ["map", runId, name],
    queryFn: () => get<MapMeta>(`/api/run/${encodeURIComponent(runId!)}/map/${encodeURIComponent(name!)}`),
    enabled: Boolean(runId && name),
    staleTime: FOREVER,
  });

/**
 * P5's own output, from the queue rather than from the receipt index.
 *
 * `useRuns` above reads /api/runs, which indexes the legacy CX_RUNS receipt
 * tree. A screening queued through the fleet keeps its verdict in the job row
 * and writes its map into the directory the worker named, so that index cannot
 * see it -- and until these existed the only way to look at one was to open a
 * shell on the GPU host and render the array by hand.
 */
export type ShuffleControl = {
  seeds: number;
  enough_seeds: boolean;
  min_seeds_for_a_percentile: number;
  percentile: number;
  real: Record<"0.5" | "0.6" | "0.7", number | null>;
  p95: Record<"0.5" | "0.6" | "0.7", number | null>;
  exceeds_p95: Record<"0.5" | "0.6" | "0.7", boolean | null>;
  sustained_exceeds_p95: boolean;
};

export type InkRun = {
  job_id: string;
  sample_id: string;
  /** P5 takes the surface it screens as a parameter; a run entered from a
   *  public layer stack names none, and that absence is real. */
  surface_id: string | null;
  /** What the job was pointed at, as it was pointed at it. Not the lineage. */
  input: { kind: string; value: string } | null;
  profile_id: string | null;
  state: string;
  mission_id: string | null;
  attempts: number | null;
  max_attempts: number | null;
  created_at: string | null;
  updated_at: string | null;
  runtime_seconds: number | null;
  /** The one block every ink lane produces identically. p99, spread and std
   *  live in `metrics` and separate a real detection from a dead one; `p50`
   *  is beside them in the same object but is not a peer -- measured
   *  directly, shuffling a confirmed control's layer order left p50
   *  unchanged while p99 and spread separated cleanly. p50 stays useful as a
   *  floor (a high one is the signature of an input outside the lane's
   *  training domain) and nothing more. The TimeSformer lane writes no
   *  `statistics` at all, so this is the only place p50/p99 are always
   *  found. */
  liveness: Liveness | null;
  statistics: Record<string, number> | null;
  /** Only present for a `direction: both` run on the 9 um lane; see
   *  `Asymmetry` above. */
  asymmetry: Asymmetry | null;
  /** The real asymmetry against the p95 of N shuffled-layer runs -- the only
   *  control where there are no labels. Null unless the job asked for
   *  `shuffle_seeds`. `enough_seeds` is whether N reached the floor (8) for
   *  the percentile to mean anything; it is stated, never enforced. */
  shuffle_control: ShuffleControl | null;
  checkpoint_sha256: string | null;
  map_shape_yx: [number, number] | null;
  output_dir: string | null;
  /** The arrays this host can read. Empty is the ordinary state of a panel
   *  that is not the machine that ran the job. */
  maps: string[];
  published: {
    artifact_uri: string | null;
    artifact_sha256: string | null;
    manifest_sha256: string | null;
    files: number | null;
  } | null;
  error: string | null;
  refused: string | null;
};

/** The stretch the server applied, so the page can say so rather than imply it. */
export type InkMapDisplay = {
  height: number;
  width: number;
  valid_pixels: number;
  invalid_pixels: number;
  normalisation: string;
  low_percentile: number;
  high_percentile: number;
  low_value: number | null;
  high_value: number | null;
  flat: boolean;
  min?: number;
  max?: number;
  note: string;
};

export type InkRunDetail = InkRun & {
  selected_map: string | null;
  display: InkMapDisplay | null;
  display_error: string | null;
  receipt: Record<string, any> | null;
  receipt_path: string | null;
  receipt_unavailable: string | null;
  /** `physical_normalization` as the worker wrote it. It names `p4_job_id`
   *  only when the render it read was a P4 job of this control plane. */
  lineage: Record<string, any> | null;
  /** What the worker resolved the named stack to, noted before the render
   *  started. The only lineage a run without a control binding has. */
  rendered_from: Record<string, any> | null;
  profile: Profile | null;
  probability_map: Record<string, any> | null;
};

const scopeQuery = (sample?: string | null, mission?: string | null) => {
  const q = new URLSearchParams();
  if (sample) q.set("sample", sample);
  if (mission) q.set("mission", mission);
  return q.size ? `?${q}` : "";
};

export const useInkRuns = (sample?: string | null, mission?: string | null) =>
  useQuery({
    queryKey: ["ink-maps", sample ?? "", mission ?? ""],
    queryFn: () => get<{ available: boolean; reason?: string; runs: InkRun[];
                         non_claims?: string[] }>(
      "/api/ink/maps" + scopeQuery(sample, mission)),
    // A queue moves while you watch it, and a screening takes minutes.
    refetchInterval: 15_000,
    staleTime: 8000,
  });

export const useInkRun = (jobId: string | null | undefined, name?: string | null) =>
  useQuery({
    queryKey: ["ink-map", jobId ?? "", name ?? ""],
    queryFn: () => get<InkRunDetail>(
      `/api/ink/maps/${encodeURIComponent(jobId!)}`
      + (name ? `?map=${encodeURIComponent(name)}` : "")),
    enabled: Boolean(jobId),
    // A receipt is written once. Only the row above it can still change.
    staleTime: FOREVER,
  });

export const useLanes = () =>
  useQuery({
    queryKey: ["lanes"],
    queryFn: () => get<{ profiles: Profile[]; upstream_clip: number }>("/api/lanes"),
    staleTime: 60_000,
  });

export const useFleet = () =>
  useQuery({
    queryKey: ["fleet"],
    queryFn: () => get<Fleet>("/api/fleet"),
    refetchInterval: 10_000,
    staleTime: 5000,
  });

export type Scroll = {
  sample_id: string;
  pixel_um: string;
  scans: number;
  /** Earliest scan date, which is when this scroll first became available. */
  scanned_on: string | null;
  scale_from: "catalog" | "scan name" | "";
  energy_kev: number | null;
  higher_res: boolean;
  runs: number;
  lane: string | null;
  p90: number | null;
  run_id: string | null;
  verdict: "SCREENED" | "NOT_SCREENED";
};
export type ScrollIndex = {
  scrolls: Scroll[];
  total: number;
  with_scale: number;
  screened_count: number;
  inventory_origin: string;
  skipped: string[];
  fetched_at: number;
};
export type EnvSetting = {
  /** Present and true when a `secret` setting has a value on the server.
   *  The value itself never arrives: it is redacted in `/api/config`, because
   *  masking it with an input of type "password" left the credential one GET
   *  away from anybody who could log in. */
  value_present?: boolean;
  name: string;
  value: string;
  default: string;
  source: "override" | "environment" | "default";
  doc: string;
  kind: string;
  allowed: string[] | null;
  example: string;
  requires_restart: boolean;
  secret: boolean;
};
export type Constant = {
  group: string;
  module: string;
  path: string;
  name: string;
  value: unknown;
  line: number;
};
export type Config = {
  environment: EnvSetting[];
  constants: Constant[];
  paths_exist: Record<string, boolean>;
  overrides_path: string;
  version: { version_id: string; content_sha256: string } | null;
};

/** `source` browses another bucket without changing the configured one. */
export const useScrolls = (source?: string) =>
  useQuery({
    queryKey: ["scrolls", source ?? ""],
    queryFn: () =>
      get<ScrollIndex>("/api/scrolls" + (source ? `?source=${encodeURIComponent(source)}` : "")),
    // The bucket listing changes on the scale of months and is disk-cached
    // server-side; refetching it on navigation would be pure waste.
    staleTime: 10 * 60 * 1000,
  });

export const useConfig = () =>
  useQuery({
    queryKey: ["config"],
    queryFn: () => get<Config>("/api/config"),
    staleTime: 5 * 60 * 1000,
  });

export type Job = {
  job_id: string;
  sample_id: string;
  profile_id: string;
  parameters: Record<string, unknown>;
  state: string;
  priority: number;
  requested_host: string | null;
  worker_id: string | null;
  attempts: number;
  max_attempts: number;
  output_dir: string | null;
  result: Record<string, any> | null;
  // The newest line the job wrote, carried by the heartbeat that renews its
  // lease. Absent until a worker has claimed it and it has said something.
  progress?: { line: string; source: string; at: string } | null;
  created_by: string;
  created_at: string;
  updated_at: string;
  lease_expires_at: string | null;
};
export type Host = {
  host_id: string;
  ssh_target: string;
  roles: string[];
  enabled: boolean;
  notes: string | null;
  last_seen_at: string | null;
  last_state: {
    gpus?: { index: number; name?: string; uuid?: string; util_pct: number;
             used_mb: number; total_mb: number }[];
    disk_free_gb?: number;
    /** Which filesystem disk_free_gb describes: the runs volume, not "/". */
    disk_path?: string;
    /** Cores the worker may actually use; cores_total only when it is confined. */
    cores?: number;
    cores_total?: number;
    ram_total_gb?: number;
    /** MemAvailable, not MemFree: reclaimable cache counts as free. */
    ram_free_gb?: number;
  } | null;
  /** Set when the panel measured this row itself instead of reading a heartbeat. */
  state_source?: string;
};

export const useJobs = () => {
  const { missionId } = useMission();
  return useQuery({
    queryKey: ["jobs", missionId],
    queryFn: () => get<{ jobs: Job[] }>(scoped("/api/jobs", missionId)),
    select: (d) => d.jobs,
    // The queue is the one thing that moves while you watch it.
    refetchInterval: 4000,
    staleTime: 2000,
  });
};

export const useHosts = () =>
  useQuery({
    queryKey: ["hosts"],
    queryFn: () => get<{ hosts: Host[] }>("/api/hosts"),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
