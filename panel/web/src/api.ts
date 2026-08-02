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
  workers?: { worker_id: string; attempts: number }[];
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
