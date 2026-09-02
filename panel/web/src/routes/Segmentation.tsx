import { Fragment, useState } from "react";
import { Link } from "react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMission, useSubject } from "../mission";
import { Card, Empty, Info, Pill, queryGate } from "../components/Bits";
import { useResizableColumns } from "../components/Table";
import { ImportSurface } from "../components/ImportSurface";

/**
 * P1, organised around the three things somebody actually does here:
 *
 *   Runs       what was attempted, and what each attempt produced
 *   Segments   what exists — every surface, however it got here
 *   New run    start one, choosing which upstream output it reads
 *   Import     put a surface grown elsewhere into the catalogue
 *
 * Import is the fourth because the inbound half of "however it got here" had
 * no door a person could use: registering a URI needs bytes already published
 * and hashed somewhere, and the artifact endpoint wants a tar and a machine
 * token. The Segments tab has always counted imported surfaces; until now
 * nothing on this panel could produce one.
 *
 * The tab bar is not here. These sit on the phase page's one bar beside
 * Profiles, and the active one arrives as a prop: two levels of tabs made the
 * reader choose twice to reach a single table, and the outer label had to name
 * what the inner ones already said.
 *
 * The counters belong to the tab. One fixed row meant the segment totals stayed
 * on screen while you read the queue and the queue stayed up while you read the
 * segments, so half of it was always answering a question nobody had asked.
 *
 * Segments and runs are separate views, and are not two vocabularies for one
 * set of rows. Inside Runs, a run and its output stay on one line -- listing
 * an attempt in one table and its "artifact" in another was the first version
 * of this page and it made the reader work out they were the same thing.
 *
 * What a runs table structurally cannot show is a surface with no run. Ten of
 * the surfaces on this control plane were imported from a catalogue and have no
 * attempt anywhere, so a page assembled only from runs under-reports what the
 * fleet holds -- and it under-reports it silently, which is the part that
 * matters. Segments answers "what do we have"; Runs answers "what did we do".
 *
 * Choosing the input lives inside the launcher, because that is when the choice
 * is made. It was a separate panel with version ids in it, which is
 * bookkeeping wearing the costume of a task.
 */

type Backend = {
  id: string; name: string; adoptable: boolean; note: string;
  /** Where a backend that is not planned as a grow is run instead. Present
   *  means "runnable, elsewhere" -- which is a different answer from the two
   *  this list used to have. */
  runs_from?: { phase: string; lane: string; profile_id: string };
};
type Configures = {
  field: string; label: string; type: "select" | "text" | "list";
  options?: string[]; suggestions?: string[]; note: string;
};
type Planner = {
  id: string; name: string; kind: "deterministic" | "agent" | "panel" | "router";
  repeatable: boolean; note: string; configures: Configures[];
  /** The fleet's default, declared by the queue rather than inferred here. */
  default?: boolean;
};
type Readable = {
  artifact_id: string; phase: string; sample_id: string; kind: string;
  note: string; registered_at_utc: string; selected: boolean;
};
type SegState = {
  public: { total: number; for_sample: number | null };
  private: { total: number; area_cm2: number | null;
             imported: number | null;
             certified: number | null; certified_area_cm2: number | null;
             ct_supported: number | null; ct_supported_area_cm2: number | null;
             for_sample: { count: number; area_cm2: number;
                           imported: number; imported_area_cm2: number } | null };
  queue: { tasks: number; attempts: number; leased: number; stale_leases: number;
           by_state: Record<string, number>; scope: string };
  backends: Backend[]; planners: Planner[];
  reads: Readable[];
  available: boolean; reason?: string;
};
type Run = {
  attempt_id: string; state: string; worker_id: string | null;
  created_at: string | null; updated_at: string | null;
  cell_id: string | null; sample_id: string | null;
  surface_id: string | null; output_path: string | null;
  artifact_sha256: string | null;
  area_cm2: number | null; qc_state: string | null; executor: string | null;
  exit_code: number | null; error: string | null;
};
type ProbeStatus = {
  available: boolean; reason?: string; missing_tables?: string[];
  counts: {
    runs: number; trials: number; decisions: number; promotions: number;
  };
  by_state: {
    runs: Record<string, number>;
    trials: Record<string, number>;
    decisions: Record<string, number>;
    promotions: Record<string, number>;
  };
  by_action: Record<string, number>;
  by_mode: Record<string, number>;
  by_mode_action: Record<string, Record<string, number>>;
  note?: string;
};
type ProbeMode = "off" | "shadow" | "select";
type ProbeOptions = {
  modes: { id: ProbeMode; name: string; note: string }[];
  default_mode: ProbeMode;
  top_k: { minimum: number; maximum: number; default: number };
  generations: { minimum: number; maximum: number; default: number };
  select_readiness: {
    available: boolean; rollout_enabled: boolean; benchmark_approved: boolean;
    benchmark_scope_allows: boolean; source_locked: boolean; benchmark_id?: string;
    review_owner_declared: boolean;
    decision_receipt_sha256?: string;
    paired_cell_count?: number; scroll_count?: number; authorized_sample_count?: number;
    reason?: string | null;
  };
  note: string; caveat: string;
};
type Job = "runs" | "segments" | "new" | "import";
type Segment = {
  surface_id: string; sample_id: string | null; area_cm2: number | null;
  state: string; physical_qc_state: string | null; geometry_qc_state: string;
  lamina_qc_state: string;
  seed_agreement_state: string;
  /** The normal component's median, in microns. The one number a cell holds:
   *  it answers at what depth the ink is sampled. Everything else — the
   *  lateral, the percentiles, and what the normalisation divided by — is in
   *  the surface's detail, where there is room to name the divisor. */
  seed_agreement_normal_um: number | null;
  lamina_assessment: {
    reason?: string; median_thickness_um?: number | null;
    clean_fraction?: number; bimodality?: number | null;
  } | null;
  artifact_uri: string | null; artifact_sha256: string | null;
  created_at: string | null; attempt_id: string | null;
  origin: "GROWN_HERE" | "IMPORTED"; source_catalog: string | null;
  human_review: {
    verdict: string; note: string | null; by: string; at: string;
    // Which route the fleet had already put the surface on when the verdict was
    // recorded. A verdict filed without it is one nobody can read back.
    surface_routing?: { route: string | null; advisory: string } | null;
  } | null;
};

const KIND: Record<string, "ok" | "crit" | "run" | "warn" | "neg"> = {
  SUCCEEDED: "ok", GROW_SUCCEEDED: "ok", CERTIFIED: "ok", QC_PENDING: "ok",
  FAILED: "crit", GROW_FAILED: "crit", CANCELLED: "neg", POLICY_REJECTED: "neg",
  RUNNING: "run", LEASED: "run", PENDING: "warn", NO_SEED: "warn",
  PROBE_REVIEW_PENDING: "warn", PROBE_REJECTED_ALL: "neg",
  BLOCKED_PROBE_ARTIFACT_UNAVAILABLE: "crit", PROBE_TECHNICAL_FAILURE: "crit",
};

/**
 * The order these are read in, rather than the order their counts fall in.
 *
 * Sorting by count put whatever was largest first, so the one good outcome on
 * the row -- QC_PENDING, a surface that grew and is waiting for P2 to certify
 * it -- landed wherever the failures left room. It leads now, and the rest run
 * from what can still be retried to what is finished with.
 *
 * A state nobody listed here still appears, at the end: this decides the
 * reading order, it does not decide what exists.
 */
const STATE_ORDER = [
  "QC_PENDING", "PROBE_REVIEW_PENDING", "NO_SEED",
  "RETRYABLE_SOURCE_UNAVAILABLE",
  "BLOCKED_SOURCE_UNAVAILABLE", "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE",
  "PROBE_TECHNICAL_FAILURE", "LEASE_EXPIRED", "POLICY_REJECTED", "GROW_FAILED",
  "PROBE_REJECTED_ALL",
];

/** The note for one field, behind the button this app already uses for "explain
 *  this". Fifteen sentences shown at once competed with the fifteen labels they
 *  belonged to, and every one of them was rendered in uppercase because it sat
 *  inside a label. */
function Explain({ text }: { text: string }) {
  if (!text) return null;
  // Wrapped so a click inside a <label> does not focus the field it explains.
  return (
    <span onClick={(e) => e.preventDefault()}>
      <Info label="what this does">{text}</Info>
    </span>
  );
}

// A scroll narrows to one, a mission to its own scrolls, neither to the fleet.
// The two tabs below sent only the scroll, so with a mission selected and no
// scroll chosen they showed every scroll the fleet has ever touched.
/**
 * A refusal, as a sentence.
 *
 * FastAPI details here are sometimes a string and sometimes an object carrying
 * `detail`, `why` and a list of what would have been accepted. Stringifying the
 * object put raw JSON on screen -- braces, quotes and all -- for what is
 * genuinely two readable sentences and a list.
 */
function readable(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    const parts = [d.detail, d.why, d.how].filter((x) => typeof x === "string") as string[];
    if (parts.length) {
      const known = Array.isArray(d.known) ? d.known : null;
      return parts.join(" — ")
        + (known ? ` Registered: ${known.join(", ")}.` : "");
    }
    // What a failed subprocess actually said. The bootstrap reports its refusals
    // as a Python traceback in `stderr_tail`, and reading only `detail` put "the
    // fleet bootstrap refused this request" on screen while the sentence that
    // named the cause -- `unknown samples requested: ['PHerc0139']` -- sat one
    // field away. The last non-empty line is the exception itself.
    const stderr = typeof d.stderr_tail === "string" ? d.stderr_tail : "";
    const last = stderr.trimEnd().split("\n").filter((line) => line.trim()).pop();
    if (last) return last.trim();
  }
  return JSON.stringify(detail);
}

const scopeQuery = (sample: string | null, mission: string | null): string => {
  const q = new URLSearchParams();
  if (sample) q.set("sample", sample);
  if (mission) q.set("mission", mission);
  return q.size ? `?${q}` : "";
};

export default function Segmentation(
  { job, onSwitch }: { job: Job; onSwitch: (next: Job) => void },
) {
  const client = useQueryClient();
  const { missionId } = useMission();
  const { subject } = useSubject();

  const state = useQuery({
    queryKey: ["segmentation", subject ?? "", missionId ?? ""],
    queryFn: async () => {
      const q = new URLSearchParams();
      if (subject) q.set("sample", subject);
      if (missionId) q.set("mission", missionId);
      const r = await fetch(`/api/segmentation?${q}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as SegState;
    },
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  const runs = useQuery({
    queryKey: ["segmentation-runs", subject ?? "", missionId ?? ""],
    queryFn: async () => {
      const r = await fetch(
        `/api/segmentation/runs${scopeQuery(subject, missionId)}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { available: boolean; reason?: string; runs: Run[] };
    },
    staleTime: 15_000,
    refetchInterval: 15_000,
  });

  const probes = useQuery({
    queryKey: ["segmentation-probes", subject ?? "", missionId ?? ""],
    queryFn: async () => {
      const r = await fetch(
        `/api/segmentation/probes${scopeQuery(subject, missionId)}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as ProbeStatus;
    },
    // A new-run form and the Segments catalogue do not read this optional
    // ledger. Keeping the query off there also means an older panel database is
    // asked about the additive migration only where its answer is useful.
    enabled: job === "runs",
    staleTime: 15_000,
    refetchInterval: job === "runs" ? 15_000 : false,
  });

  const segments = useQuery({
    queryKey: ["segmentation-segments", subject ?? "", missionId ?? ""],
    queryFn: async () => {
      const r = await fetch(
        `/api/segmentation/segments${scopeQuery(subject, missionId)}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as {
        available: boolean; reason?: string; segments: Segment[];
        count: number; grown_here: number; imported: number; total_area_cm2: number;
      };
    },
    staleTime: 15_000,
    refetchInterval: 30_000,
  });

  const d = state.data;
  const gate = queryGate(state, "reading the segmentation fleet…");
  if (gate) return gate;
  // The gate already returned for every case where this is unset; the compiler
  // cannot see that through a helper, and asserting it away would defeat the
  // point of the helper existing.
  if (!d) return null;

  const attempted = runs.data?.runs ?? [];
  const surfaces = subject ? (d.private.for_sample?.count ?? 0) : d.private.total;
  const area = subject ? (d.private.for_sample?.area_cm2 ?? null) : d.private.area_cm2;
  // Imported surfaces live in the same table and had been counted as ours: ten
  // on this control plane arrived from a catalogue with no attempt anywhere.
  const imported = subject
    ? (d.private.for_sample?.imported ?? 0)
    : (d.private.imported ?? 0);

  return (
    <>
      {job === "runs" && (
        <div className="strip">
          {/* Three numbers, in the order the questions get asked: did this
              phase produce anything, how much work stands behind that, and is
              any of it moving right now. The seven-tile row above these was the
              phase view printing one tile per key the API happened to return,
              which is how `attempts` appeared twice in adjacent rows and how a
              grid-version count sat beside the surfaces it knows nothing about. */}
          <div className={`tile ${surfaces ? "steady" : "warn"}`}>
            <h2>Surfaces we grew</h2>
            <div className="readout">
              {surfaces}
              {area !== null && <span className="readout-unit"> · {area.toFixed(1)} cm²</span>}
            </div>
            <p>
              {surfaces ? "gross area, an upper bound — P2 certifies" : "none grown yet"}
              {imported > 0 && ` · ${imported} imported, not counted here`}
            </p>
          </div>
          {/* Certified and CT-supported, which the API computed and never sent.
              Gross area is an upper bound; this is the pair that decides what
              P3 and P4 may consume. */}
          <div className={`tile ${d.private.certified ? "steady" : "warn"}`}>
            <h2>Certified</h2>
            <div className="readout">
              {d.private.certified ?? 0}
              {d.private.certified_area_cm2 != null && (
                <span className="readout-unit"> · {d.private.certified_area_cm2.toFixed(1)} cm²</span>
              )}
            </div>
            <p>
              {d.private.ct_supported
                ? `${d.private.ct_supported} also CT-supported`
                : "geometry certified — CT support is a separate gate"}
            </p>
          </div>
          <div className="tile steady">
            <h2>Attempts</h2>
            <div className="readout">{d.queue.attempts}</div>
            <p>{attempted.length} shown · every state below</p>
          </div>
          {/* Stalled was its own tile reading 0 almost always. It is the same
              question as "what is in flight" -- a lease that expired mid-run is
              work that is neither waiting nor moving -- so it rides here and
              turns the tile red when it is not zero. */}
          <div className={`tile ${d.queue.stale_leases ? "alert"
                                  : d.queue.leased ? "busy" : "steady"}`}>
            <h2>In flight</h2>
            <div className="readout">{d.queue.tasks}</div>
            <p>
              {d.queue.leased} being worked on · {d.queue.scope}
              {d.queue.stale_leases > 0 &&
                ` · ${d.queue.stale_leases} stalled, leases expired mid-run`}
            </p>
          </div>
        </div>
      )}

      {job === "segments" && (
        <div className="strip">
          <div className={`tile ${surfaces ? "steady" : "warn"}`}>
            <h2>Surfaces we grew</h2>
            <div className="readout">
              {surfaces}
              {area !== null && <span className="readout-unit"> · {area.toFixed(1)} cm²</span>}
            </div>
            <p>
              {surfaces ? "gross area, an upper bound — P2 certifies" : "none grown yet"}
              {imported > 0 && ` · ${imported} imported, not counted here`}
            </p>
          </div>
          <div className="tile steady">
            <h2>Imported</h2>
            <div className="readout">{segments.data?.imported ?? 0}</div>
            <p>no attempt produced these</p>
          </div>
          <div className="tile steady">
            <h2>Public segments</h2>
            <div className="readout">{subject ? (d.public.for_sample ?? 0) : d.public.total}</div>
            <p>traced by the community</p>
          </div>
        </div>
      )}

      {/* No counters on New run. They counted the form's own controls -- six
          seeders, one adoptable backend, three inputs -- which is the length of
          lists the form shows a few lines below. A tile restating the widget
          under it is furniture. */}

      {job === "runs" && Object.keys(d.queue.by_state).length > 0 && (
        <Card title="Where the runs are" note={`${d.queue.attempts} attempts`}>
          <div className="body-pad">
            {/* Every state, including the ones that produced nothing -- those
                are most of them, and a page that only counts successes makes a
                fleet look like it is working. */}
            <div className="statecounts">
              {Object.entries(d.queue.by_state)
                .sort((a, b) => {
                  const ia = STATE_ORDER.indexOf(a[0]);
                  const ib = STATE_ORDER.indexOf(b[0]);
                  // Unlisted states keep their old behaviour, after the listed
                  // ones: largest first.
                  if (ia === -1 && ib === -1) return b[1] - a[1];
                  if (ia === -1) return 1;
                  if (ib === -1) return -1;
                  return ia - ib;
                })
                .map(([state, count]) => (
                  <span key={state} className={`statecount is-${(KIND[state] ?? "warn")}`}>
                    <b>{count}</b>
                    <span>{state.toLowerCase().replaceAll("_", " ")}</span>
                  </span>
                ))}
            </div>
            {/* The NO_SEED breakdown, as a footnote to the chip it explains
                rather than the card it used to be. */}
            {(d.queue.by_state["NO_SEED"] ?? 0) > 0 && (
              <NoSeed sample={subject} mission={missionId} />
            )}
          </div>
        </Card>
      )}

      {job === "runs" && (
        <ProbeSummary status={probes.data} loading={probes.isLoading} />
      )}

      {!d.available && (
        <Card title="The fleet database is not reachable">
          <div className="body-pad">
            <p>
              The published count is read from the bucket and is real. Runs, queue and the
              surfaces we grew all live in the control plane: <code>{d.reason}</code>
            </p>
          </div>
        </Card>
      )}

      {job === "segments" && (
        <Card
          title="Segments"
          note={segments.data?.available
            ? `${segments.data.grown_here} grown here · ${segments.data.imported} imported`
            + ` · ${segments.data.total_area_cm2} cm²`
            : undefined}
        >
          {segments.data?.segments?.length
            ? <SegmentTable segments={segments.data.segments} />
            : (
              <Empty>
                {segments.data?.available === false
                  ? segments.data.reason
                  : "no surfaces on this control plane yet"}
              </Empty>
            )}
        </Card>
      )}

      {job === "runs" && (
        <Card title="Runs" note={`${attempted.length} attempts`}>
          <Maintenance sample={subject} mission={missionId} />
          {attempted.length ? <RunTable runs={attempted} /> : (
            <Empty>
              {runs.data?.available === false
                ? runs.data.reason
                : "nothing has run here yet — New run starts one"}
            </Empty>
          )}
        </Card>
      )}

      {job === "import" && (
        <Card title="Import surface"
              note="grown elsewhere, recorded here with its authorship">
          <ImportSurface
            sample={subject}
            missionId={missionId}
            onDone={() => {
              // Straight to Segments. The question after importing is whether
              // it is in the catalogue, and this is the table that answers it.
              onSwitch("segments");
            }}
          />
        </Card>
      )}

      {job === "new" && (
        <Card title="New run">
          <NewRun
            key={`${missionId ?? "global"}:${subject ?? "none"}`}
            backends={d.backends}
            planners={d.planners}
            reads={d.reads}
            sample={subject}
            missionId={missionId}
            onDone={() => {
              // Back to Runs rather than staying on an emptied form: the
              // question after starting a run is what it is doing.
              onSwitch("runs");
              client.invalidateQueries({ queryKey: ["segmentation"] });
              client.invalidateQueries({ queryKey: ["segmentation-runs"] });
              client.invalidateQueries({ queryKey: ["segmentation-segments"] });
            }}
          />
        </Card>
      )}
    </>
  );
}

/**
 * What exists, and where each one came from.
 *
 * `origin` is the column a runs table cannot have. A surface with no attempt is
 * not a broken row -- it arrived from a catalogue -- and showing it beside the
 * grown ones is the only way the count on this page matches the count in the
 * control plane.
 */
const VERDICTS = ["APPROVED", "DEFECTIVE", "REVIEWED", "INSPECT"];

/**
 * Three orthogonal CT slices through one surface.
 *
 * Read-only, and deliberately not VC3D's editor: enough to answer "does this
 * follow one sheet" without rebuilding a 3D tool in React. Until this existed the
 * review verdicts above were being asked of people with nothing to look at.
 *
 * ponytail: three <img> tags and native range inputs. The server renders the PNG,
 * the browser caches it, and there is no canvas, no tile grid and no WebGL.
 */
function Slices({ surfaceId }: { surfaceId: string }) {
  const [at, setAt] = useState(0.5);
  const [threshold, setThreshold] = useState(0);
  const [window_, setWindow] = useState(512);
  const [overlay, setOverlay] = useState(true);
  const [slab, setSlab] = useState(32);
  const src = (axis: string) =>
    `/api/segmentation/surface/${encodeURIComponent(surfaceId)}/slice.png`
    + `?axis=${axis}&at=${at}&window=${window_}&threshold=${threshold}`
    + `&overlay=${overlay}&slab=${slab}`;

  return (
    <div className="slices">
      <div className="slices-controls">
        <Info label="How to read these slices" title="Reading the slices">
          Slices are bounded by this surface's own box, not the whole volume, and
          each is normalised on its own range. The dots are the surface itself —
          every pixel of its TIFXYZ whose CT coordinate falls within ±{slab} voxels
          of the plane, so the curve is where the sheet crosses this slice. A
          surface whose artifact cannot be read falls back to its 256 stored
          sample points, which locate the patch rather than trace it, and the response
          says which you are looking at.
        </Info>
        <label>through
          <input type="range" min={0} max={1} step={0.02} value={at}
                 onChange={(e) => setAt(Number(e.target.value))} />
          <span className="dim">{(at * 100).toFixed(0)}%</span>
        </label>
        <label>threshold
          <input type="range" min={0} max={255} step={5} value={threshold}
                 onChange={(e) => setThreshold(Number(e.target.value))} />
          <span className="dim">{threshold}</span>
        </label>
        <label>window
          <input type="range" min={128} max={2048} step={128} value={window_}
                 onChange={(e) => setWindow(Number(e.target.value))} />
          <span className="dim">{window_}px</span>
        </label>
        <label>
          <input type="checkbox" checked={overlay}
                 onChange={(e) => setOverlay(e.target.checked)} />
          surface
        </label>
        {overlay && (
          <label>within
            <input type="range" min={4} max={128} step={4} value={slab}
                   onChange={(e) => setSlab(Number(e.target.value))} />
            <span className="dim">±{slab} vx</span>
          </label>
        )}
      </div>
      <div className="slices-row">
        {["z", "y", "x"].map((axis) => (
          <figure key={axis}>
            <img src={src(axis)} alt={`CT slice along ${axis}`} loading="lazy" />
            <figcaption>{axis}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

function SegmentTable({ segments }: { segments: Segment[] }) {
  const tableRef = useResizableColumns<HTMLTableElement>();
  const client = useQueryClient();
  const [saving, setSaving] = useState<string | null>(null);
  const [looking, setLooking] = useState<string | null>(null);

  // A person's opinion, kept apart from P2's verdict on purpose: the two
  // columns beside it are the fleet's and this one is not.
  // ponytail: a native <select>, four options, no modal and no form state.
  const review = async (surfaceId: string, verdict: string) => {
    if (!verdict) return;
    setSaving(surfaceId);
    try {
      const r = await fetch(`/api/segmentation/surface/${encodeURIComponent(surfaceId)}/review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict }),
      });
      const saved = await r.json();
      if (!r.ok) throw new Error(readable(saved.detail));
      // Said at the moment the verdict is recorded, because afterwards it is
      // just another APPROVED in a table: a surface the router sent to
      // SMALL_SURFACE_DIAGNOSTIC was never offered the standard path, and
      // approving 0.0198 cm2 of papyrus means nothing like approving five
      // square centimetres of it.
      if (saved.surface_routing?.route === "SMALL_SURFACE_DIAGNOSTIC")
        window.alert(saved.surface_routing.advisory);
      client.invalidateQueries({ queryKey: ["segmentation"] });
    } catch (e) {
      window.alert(String(e));
    } finally {
      setSaving(null);
    }
  };

  return (
    <div className="scroller">
      <table ref={tableRef}>
        <thead>
          <tr>
            <th className="l">Made</th>
            <th className="l grow">Surface</th>
            <th className="l">Scroll</th>
            <th className="l">Origin</th>
            <th>Area cm²</th>
            <th className="l">CT support</th>
            <th className="l">Geometry</th>
            <th className="l">Lamina</th>
            <th className="l">Seeds</th>
            <th className="l">Review</th>
            <th className="l">VC3D</th>
          </tr>
        </thead>
        <tbody>
          {segments.map((s) => (
            <Fragment key={s.surface_id}>
            <tr>
              <td className="l">
                {s.created_at?.slice(0, 16).replace("T", " ")
                  ?? <span className="dash">—</span>}
              </td>
              <td className="l grow" title={s.artifact_uri ?? undefined}>
                <code>{s.surface_id.split(":").pop()?.slice(0, 12)}</code>
              </td>
              <td className="l">{s.sample_id ?? <span className="dash">—</span>}</td>
              <td className="l" title={s.source_catalog ?? undefined}>
                {s.origin === "GROWN_HERE"
                  ? <Pill kind="ok">grown here</Pill>
                  : <Pill kind="neg">imported</Pill>}
              </td>
              <td>{s.area_cm2?.toFixed(2) ?? <span className="dash">—</span>}</td>
              <td className="l">
                {s.physical_qc_state
                  ? <Pill kind={s.physical_qc_state.includes("SUPPORTED") ? "ok" : "warn"}>
                      {s.physical_qc_state.toLowerCase().replaceAll("_", " ")}
                    </Pill>
                  : <span className="dash">—</span>}
              </td>
              <td className="l">
                {/* Unmeasured is not a pass. P2 is fail-soft, so a surface with
                    no verdict looks identical to a clean one unless it says so. */}
                <Pill kind={s.geometry_qc_state === "GEOMETRY_CERTIFIED" ? "ok"
                          : s.geometry_qc_state.startsWith("GEOMETRY_REJECTED") ? "crit"
                          : "warn"}>
                  {s.geometry_qc_state.replace("GEOMETRY_", "").replaceAll("_", " ").toLowerCase()}
                </Pill>
              </td>
              <td className="l">
                {/* The third axis: whether the CT resolves one sheet under this
                    surface or two welded together. Geometry can certify a mesh
                    that is a plausible sheet over a slab -- it says so in its
                    own non-claims -- and this is the question that decides
                    whether rendering is worth its cost. */}
                <span title={s.lamina_assessment?.reason
                        ?? "the lamina gate has not run on this surface"}>
                  <Pill kind={s.lamina_qc_state === "LAMINA_SINGLE_SHEET" ? "ok"
                            : s.lamina_qc_state === "LAMINA_FUSED" ? "crit"
                            : "warn"}>
                    {s.lamina_qc_state.replace("LAMINA_", "").replaceAll("_", " ").toLowerCase()}
                  </Pill>
                </span>
                {typeof s.lamina_assessment?.median_thickness_um === "number" && (
                  <span className="dash">
                    {" "}{s.lamina_assessment.median_thickness_um.toFixed(0)} µm
                  </span>
                )}
              </td>
              <td className="l">
                {/* The fifth judgement, and the only one that is not about this
                    surface: how far a second run of the same fit landed from
                    it. It sits here rather than folded into geometry because it
                    can contradict it — on w015 the most tangled band of the
                    patch had the *best* agreement between seeds — and because
                    two runs converge on the same wrong surface when the failure
                    is in the data rather than the optimization.

                    The state and one number, nothing else. The decomposition,
                    the percentiles and what the normalisation divided by are in
                    the surface's own detail: four scannable columns beside a
                    cell of four numbers is a table you cannot sweep. */}
                <span title={
                  s.seed_agreement_state === "SEED_UNPAIRED"
                    ? "one seed, so no error bar — which is not the same as a "
                      + "small one, and cannot be defended"
                    : s.seed_agreement_state === "SEED_OVERRIDE_DID_NOT_TAKE"
                    ? "the two runs came out identical, so the seed never "
                      + "reached the optimizer — this metric is the one that "
                      + "fails upward"
                    : "how far a second run of the same fit landed: "
                      + "reproducibility, not correctness"}>
                  {/* Neutral, not green: a measured pair is a measurement and
                      not a pass. The only red one is the seed that never took,
                      because that reads as perfect reproducibility. */}
                  <Pill kind={s.seed_agreement_state === "SEED_AGREEMENT_MEASURED"
                              ? "none"
                            : s.seed_agreement_state === "SEED_OVERRIDE_DID_NOT_TAKE"
                              ? "crit" : "neg"}>
                    {s.seed_agreement_state === "SEED_UNPAIRED" ? "unpaired"
                     : s.seed_agreement_state === "SEED_AGREEMENT_MEASURED"
                       ? "paired" : s.seed_agreement_state
                           .replace("SEED_", "").replaceAll("_", " ").toLowerCase()}
                  </Pill>
                </span>
                {typeof s.seed_agreement_normal_um === "number" && (
                  <span className="dash">
                    {" "}{s.seed_agreement_normal_um.toFixed(0)} µm normal
                  </span>
                )}
              </td>
              <td className="l">
                <select value={s.human_review?.verdict ?? ""}
                        disabled={saving === s.surface_id}
                        title={s.human_review
                          ? `${s.human_review.verdict} by ${s.human_review.by} at ${s.human_review.at}`
                            + (s.human_review.surface_routing?.advisory
                               ? ` — ${s.human_review.surface_routing.advisory}` : "")
                          : "a person's opinion, which is not P2's verdict"}
                        onChange={(e) => review(s.surface_id, e.target.value)}>
                  <option value="">—</option>
                  {VERDICTS.map((v) => (
                    <option key={v} value={v}>{v.toLowerCase()}</option>
                  ))}
                </select>
              </td>
              <td className="l">
                {/* Pointers, not copies: the bundle names the volume and the
                    surface and prints the command. See the endpoint. */}
                <a href={`/api/segmentation/surface/${encodeURIComponent(s.surface_id)}/vc3d`}
                   target="_blank" rel="noreferrer noopener"
                   title="the volume, the surface, the frame and the vc_grow_seg_from_seed command">
                  open
                </a>
                {" "}
                <button type="button" className="linky"
                        title="three orthogonal CT slices through this surface"
                        onClick={() => setLooking(looking === s.surface_id ? null : s.surface_id)}>
                  {looking === s.surface_id ? "hide CT" : "CT"}
                </button>
              </td>
            </tr>
            {looking === s.surface_id && (
              <tr>
                <td colSpan={10}><Slices surfaceId={s.surface_id} /></td>
              </tr>
            )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Why the fleet found nothing to grow from.
 *
 * NO_SEED is a label, not a cause -- it means none of the proposals survived.
 * Which screen removed them matters, because the three are unrelated problems:
 * the prediction saying there is no sheet, the raw CT disagreeing with the
 * prediction, or the candidates being fine and too close to ground already
 * covered. The last one is the fleet re-treading, which is good news wearing a
 * failure's label.
 */
const SCREEN_MEANING: Record<string, string> = {
  NO_M7_CANDIDATES:
    "the surface prediction has nothing above threshold in that box — either "
    + "the box is empty papyrus-wise, or 0.2 is too high for this scan",
  CT_MATERIAL_SUPPORT_REJECTED:
    "the prediction proposed a point and the raw CT has no material there — "
    + "the model saw a sheet that is not in the scan",
};

function NoSeed({ sample, mission }: { sample: string | null; mission: string | null }) {
  const q = useQuery({
    queryKey: ["no-seed", sample ?? "", mission ?? ""],
    queryFn: async () => {
      const r = await fetch(`/api/segmentation/no-seed${scopeQuery(sample, mission)}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as {
        available: boolean; reason?: string; attempts: number;
        by_cause: Record<string, number>;
        candidates_surviving_each_screen: Record<string, number>;
        undiagnosed: number; note: string;
        no_candidate_ever_proposed?: number; seed_service_suspected?: boolean;
      };
    },
    staleTime: 60_000,
  });
  const d = q.data;
  if (!d?.available) return null;

  // Three numbers, not a funnel. The question is "why did these find nothing",
  // and the answer splits in two: the search proposed nothing at all, or it
  // proposed something and a screen removed it. Those lead opposite ways -- one
  // is a search to fix, the other is ground to stop probing -- so they are the
  // two boxes, and the causes name which screen did the removing.
  const nothingProposed = d.no_candidate_ever_proposed ?? 0;
  const screenedOut = Math.max(d.attempts - nothingProposed, 0);
  const causes = Object.entries(d.by_cause).sort((a, b) => b[1] - a[1]);

  return (
    <p className="hint noseed">
      Of the {d.attempts} with no seed, <b>{nothingProposed}</b> had nothing
      proposed at all — the seed search returned no candidates, so no screen
      rejected anything — and <b>{screenedOut}</b> were proposed and then taken
      by a screen{causes.length > 0 && ": "}
      {/* The meaning stays reachable on the code rather than getting a line of
          its own: these are cryptic enough that dropping it would be a loss,
          and long enough that printing all of them was the section's bulk. */}
      {causes.map(([cause, count], i) => (
        <span key={cause}>
          {i > 0 && ", "}{count}{" "}
          <code title={SCREEN_MEANING[cause]
            ?? "the candidates were usable and too close to ground already segmented"}>
            {cause}
          </code>
        </span>
      ))}.
      {d.undiagnosed > 0 && (
        <> {d.undiagnosed} of them are older attempts that never recorded which
        screen it was.</>
      )}{" "}
      NO_SEED names the screen that removed the proposals; it does not establish
      that no physical surface is there.
    </p>
  );
}

const took = (from: string | null, to: string | null): string | null => {
  if (!from || !to) return null;
  const ms = Date.parse(to) - Date.parse(from);
  if (!Number.isFinite(ms) || ms < 0) return null;
  const s = Math.round(ms / 1000);
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s`
                                     : `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m`;
};

/**
 * The optional seed-probe ledger, compactly.
 *
 * These are experiments inside attempts, not extra canonical surfaces. Keeping
 * the three populations separate is what lets an operator see the compute
 * multiplier (trials per probe run) without mistaking rejected micro-patches
 * for segmentation yield.
 */
function ProbeSummary({ status, loading }: {
  status: ProbeStatus | undefined; loading: boolean;
}) {
  if (loading) {
    return (
      <Card title="Seed probes" note="read-only ledger">
        <Empty>reading deterministic micro-growth decisions…</Empty>
      </Card>
    );
  }
  if (!status?.available) {
    return (
      <Card title="Seed probes" note="not active on this control plane">
        <div className="body-pad">
          <p className="hint">
            {status?.reason ?? "seed-probe status is unavailable"}
            {status?.missing_tables?.length
              ? ` · missing ${status.missing_tables.join(", ")}`
              : ""}
          </p>
        </div>
      </Card>
    );
  }
  const actions = Object.entries(status.by_action ?? {})
    .sort((a, b) => b[1] - a[1]);
  const modeActions = Object.entries(status.by_mode_action ?? {})
    .flatMap(([mode, grouped]) => Object.entries(grouped)
      .map(([action, count]) => ({ mode, action, count })))
    .sort((a, b) => b.count - a.count
      || a.mode.localeCompare(b.mode)
      || a.action.localeCompare(b.action));
  return (
    <Card title={<>Seed probes <Info label="What a probe decision does and does not establish"
            title="Seed probes">
            {status.note ?? "Probe decisions compare bounded micro-growth."} They do not
            establish that a patch follows the correct lamina; geometry and human review
            remain separate evidence. Rejected trials are non-canonical and are not
            segmentation yield.
          </Info></>} note="seed-probe-v1 · read-only ledger">
      <div className="body-pad">
        <div className="statecounts">
          <span className="statecount is-run">
            <b>{status.counts.runs}</b><span>probe runs</span>
          </span>
          <span className="statecount is-warn">
            <b>{status.counts.trials}</b><span>micro-growth trials</span>
          </span>
          <span className="statecount is-ok">
            <b>{status.counts.decisions}</b><span>decisions</span>
          </span>
          <span className="statecount is-run">
            <b>{status.by_state.promotions.PROMOTED ?? 0}</b>
            <span>canonical continuations</span>
          </span>
          {(modeActions.length > 0
            ? modeActions
            : actions.map(([action, count]) => ({
                mode: "unknown", action, count,
              }))).map(({ mode, action, count }) => (
              <span key={`${mode}-${action}`} className={`statecount is-${
                action.includes("REVIEW") || action.includes("ABSTAIN") ? "warn"
                  : action.includes("REJECT") ? "neg" : "ok"}`}>
                <b>{count}</b>
                <span>
                  {mode} · {action.toLowerCase().replaceAll("_", " ")}
                </span>
              </span>
            ))}
        </div>
        <p className="hint">
          {status.counts.promotions} promotion record
          {status.counts.promotions === 1 ? "" : "s"} in all states.
        </p>
      </div>
    </Card>
  );
}

/**
 * The fleet's maintenance commands, which existed only over ssh.
 *
 * ponytail: window.confirm rather than a modal component -- these are operator
 * actions with one question each, and the browser already has the dialog. The
 * receipt goes to the console because the commands return one and swallowing it
 * would be worse than not showing it prettily.
 */
function Maintenance({ sample, mission }:
                     { sample: string | null; mission: string | null }) {
  const client = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const [said, setSaid] = useState<string | null>(null);

  const ACTIONS: [string, string][] = [
    ["republish", "Move surfaces sitting on a worker's local disk into object storage. Safe to repeat."],
    ["coverage", "Report how much of the scroll has been explored. Changes nothing."],
    ["certify", "Give a geometry verdict to surfaces that have none — which is what an imported surface needs before model QC can claim it. Safe to repeat."],
  ];

  const run = async (action: string, note: string) => {
    if (mission && !sample) return;
    if (!window.confirm(`${action}${sample ? ` for ${sample}` : " for every scroll"}?\n\n${note}`)) return;
    setBusy(action); setSaid(null);
    try {
      const r = await fetch("/api/segmentation/maintenance", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, sample_id: sample, mission_id: mission }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(readable(body.detail));
      setSaid(`${action}: ${JSON.stringify(body).slice(0, 200)}`);
      client.invalidateQueries({ queryKey: ["segmentation"] });
    } catch (e) {
      setSaid(String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="maint">
      {ACTIONS.map(([action, note]) => (
        <button key={action} type="button" disabled={busy !== null || Boolean(mission && !sample)}
                title={note} onClick={() => run(action, note)}>
          {busy === action ? `${action}…` : action}
        </button>
      ))}
      {said && <p className="formerror">{said}</p>}
    </div>
  );
}

/** A run and what it produced, on one line. */
function RunTable({ runs }: { runs: Run[] }) {
  const tableRef = useResizableColumns<HTMLTableElement>();
  return (
    <div className="scroller">
      <table ref={tableRef}>
        <thead>
          <tr>
            <th className="l">Started</th>
            <th className="l grow">Scroll · cell</th>
            <th className="l">State</th>
            <th className="l">Produced</th>
            <th>Area cm²</th>
            <th className="l">QC</th>
            <th className="l">Took</th>
            <th className="l">Backend</th>
            <th className="l">Worker</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <Fragment key={r.attempt_id}>
            <tr className={r.surface_id ? "" : "muted"}>
              <td className="l">
                {r.created_at?.slice(0, 16).replace("T", " ") ?? <span className="dash">—</span>}
              </td>
              <td className="l grow">
                {r.sample_id ?? <span className="dash">—</span>}
                {r.cell_id && <span className="dash"> · {r.cell_id}</span>}
              </td>
              <td className="l">
                {/* The state links to everything recorded about the attempt:
                    manifest, locked-plan hash, envelope, result. Raw JSON on
                    purpose -- see the endpoint's note. */}
                <a href={`/api/segmentation/attempt/${r.attempt_id}`}
                   target="_blank" rel="noreferrer noopener"
                   title="the manifest, envelope and result behind this attempt">
                  <Pill kind={KIND[r.state] ?? "warn"}>{r.state.toLowerCase().replaceAll("_", " ")}</Pill>
                </a>
              </td>
              <td className="l" title={[r.surface_id, r.artifact_sha256 && `sha256 ${r.artifact_sha256}`,
                                       r.output_path].filter(Boolean).join("\n") || undefined}>
                {r.surface_id
                  ? <><code>{r.surface_id.slice(0, 10)}</code>
                      {r.artifact_sha256 && <div className="dim">
                        <code>{r.artifact_sha256.slice(0, 12)}</code></div>}</>
                  : <span className="dash">nothing</span>}
              </td>
              <td>{r.area_cm2?.toFixed(2) ?? <span className="dash">—</span>}</td>
              <td className="l">
                {r.qc_state
                  ? <Pill kind={r.qc_state.includes("SUPPORTED") ? "ok" : "warn"}>{r.qc_state}</Pill>
                  : <span className="dash">—</span>}
              </td>
              <td className="l">{took(r.created_at, r.updated_at) ?? <span className="dash">—</span>}</td>
              <td className="l">{r.executor ?? <span className="dash">—</span>}</td>
              <td className="l">{r.worker_id ?? <span className="dash">—</span>}</td>
            </tr>
            {(r.error || r.exit_code) && (
              <tr className="why">
                <td className="l" colSpan={9}>
                  {r.exit_code ? <Pill kind="crit">exit {r.exit_code}</Pill> : null}
                  {r.error && <span className="whytext">{r.error}</span>}
                </td>
              </tr>
            )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NewRun({ backends, planners, reads, sample, missionId, onDone }: {
  backends: Backend[]; planners: Planner[]; reads: Readable[];
  sample: string | null; missionId: string | null; onDone: () => void;
}) {
  const client = useQueryClient();
  const [backend, setBackend] = useState(backends.find((b) => b.adoptable)?.id ?? backends[0].id);
  // The first offered seeder, so the order of that list is the default and the
  // two cannot drift. It used to name "deterministic" literally, which is now the
  // history-blind one -- a default nobody would have chosen on purpose.
  // The one the queue marks as the fleet's default, not whichever is listed
  // first. Taking [0] meant the form's default drifted with the order of a
  // Python list, and it disagreed with both the stage contract and the API.
  const [planner, setPlanner] = useState(
    planners.find((p) => p.default)?.id ?? planners[0]?.id ?? "cost-aware-v2");
  const [seedConfig, setSeedConfig] = useState<Record<string, string>>({});
  const [seedProbeMode, setSeedProbeMode] = useState<ProbeMode>("off");
  const [seedProbeTopK, setSeedProbeTopK] = useState(2);
  const [seedProbeGenerations, setSeedProbeGenerations] = useState(12);
  const [maxTasks, setMaxTasks] = useState(4);
  const [reason, setReason] = useState("");
  const selected = reads.find((r) => r.selected) ?? reads[0];
  const [input, setInput] = useState(selected?.artifact_id ?? "");
  // After `input`, not before it. Reading a const above its own declaration is a
  // temporal dead zone error, and minified it says "cannot access 'E' before
  // initialization" -- a blank page whose message names a variable that does not
  // exist in the source.
  //
  // The scroll a run covers is the scroll its input froze. Falling back to the
  // page's subject only when there is no input to read.
  const covers = reads.find((r) => r.artifact_id === input)?.sample_id ?? sample;
  const inputValid = Boolean(covers)
    && (!input || reads.some((r) => r.artifact_id === input));

  const [options, setOptions] = useState<Record<string, string>>({});
  // Where the seed comes from at all. Discovered means the fleet probes the
  // prediction inside a cell it chose; supplied means somebody names the point
  // and the prediction is skipped -- which is why the CT gate matters more there,
  // not less.
  const [seedFrom, setSeedFrom] = useState<"discovered" | "supplied">("discovered");
  const [points, setPoints] = useState("");
  // One parse for the count, the complaints and the request. Two parsers would
  // eventually disagree about which lines are points.
  const parsed = points.split("\n").map((line, index) => {
    const raw = line.trim();
    if (!raw || raw.startsWith("#")) return null;
    const parts = raw.split(/[,\s]+/).filter(Boolean);
    const [x, y, z] = parts.slice(0, 3).map(Number);
    if (parts.length < 3 || [x, y, z].some((n) => !Number.isFinite(n))) {
      return { line: index + 1, raw, bad: true as const };
    }
    return { line: index + 1, x, y, z, note: parts.slice(3).join(" "), bad: false as const };
  }).filter((row): row is NonNullable<typeof row> => row !== null);
  const good = parsed.filter((r) => !r.bad);
  const bad = parsed.filter((r) => r.bad);
  const [showOptions, setShowOptions] = useState(false);

  // The knob list comes from the server, which is the same list it validates
  // against and builds the command from. Typing it here would be a second copy
  // that drifts, and the fields it describes are the ones that decide where the
  // fleet looks and which candidate wins.
  const knobs = useQuery({
    queryKey: ["segmentation-options", covers ?? ""],
    queryFn: async () => {
      const q = covers ? `?sample=${encodeURIComponent(covers)}` : "";
      const r = await fetch(`/api/segmentation/options${q}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as {
        options: { flag: string; field: string; kind: string; label: string;
                   group: string; note: string; choices?: string[] }[];
        probe?: ProbeOptions;
        source?: { sample: string | null; growable: boolean;
                   growable_scrolls: string[]; note: string };
        growth: { note: string;
                  parameters: { name: string; range: string; default: unknown }[] };
      };
    },
    staleTime: 5 * 60_000,
  });

  const chosen = backends.find((b) => b.id === backend);
  // A scroll the frozen catalog does not name has no m7 prediction to seed on,
  // so no configuration of this form can produce a run for it. Said here, before
  // the button, because the queue used to be the one that found out.
  const source = knobs.data?.source;
  const notGrowable = source ? source.growable === false : false;
  const probeConfig = knobs.data?.probe;
  const probeCanSelect = (
    planner === "cost-aware-v2" || planner === "deterministic-v2"
  ) && probeConfig?.select_readiness?.available === true;
  const probeModes: ProbeOptions["modes"] = probeConfig?.modes ?? [
    { id: "off", name: "Off", note: "Grow the canonical seed directly." },
    { id: "shadow", name: "Shadow",
      note: "Record a deterministic micro-growth comparison without steering." },
    { id: "select", name: "Select",
      note: "Use a separated winner, or stop for human review." },
  ];
  // An unusable backend cannot be selected any more, so this only fires if the
  // server starts offering one the page thinks is adoptable and the page is
  // wrong -- in which case a written reason is still the right gate.
  const needsReason = chosen ? !chosen.adoptable : false;

  const queue = useMutation({
    mutationFn: async () => {
      // If the run is to read something other than what the mission currently
      // has selected, that is a change to the selection and is recorded as one
      // -- rather than a per-run override nothing else would know about.
      if (input && input !== selected?.artifact_id) {
        const picked = reads.find((r) => r.artifact_id === input);
        if (!picked) throw new Error("the selected P0 input is no longer in this mission");
        const r = await fetch(`/api/missions/${missionId}/selection`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            choices: { [`${picked.phase}/${picked.sample_id}`]: picked.artifact_id },
            reason: reason || `chosen when queueing ${backend} for ${sample}`,
          }),
        });
        if (!r.ok) {
          const body = await r.json();
          throw new Error(typeof body.detail === "string" ? body.detail : "input not accepted");
        }
      }
      if (seedFrom === "supplied") {
        const r = await fetch("/api/segmentation/manual-seeds", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_id: covers,
            points: good.map((g) => ({ x: g.x, y: g.y, z: g.z, note: g.note })),
            note: reason || null,
            // The rest of the form, which this used to leave behind: the planner
            // and its model, the options bootstrap-manual accepts, and the
            // mission so the seed records which P0 selection it read. A manual
            // seed is still a task and gets the same provenance as any other.
            planner,
            mission_id: missionId,
            seed_config: Object.fromEntries(
              Object.entries(seedConfig).filter(([field, v]) =>
                v !== "" && (planners.find((p) => p.id === planner)
                  ?.configures ?? []).some((c) => c.field === field))),
            options: Object.fromEntries(
              Object.entries(options).filter(([, v]) => v !== "")),
          }),
        });
        const body = await r.json();
        if (!r.ok) {
          throw new Error(typeof body.detail === "string" ? body.detail
            : body.detail?.stderr_tail?.trim().split("\n").slice(-1)[0]
              ?? "these points were refused");
        }
        return body as { queued_for: string };
      }
      const r = await fetch("/api/segmentation/runs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: covers, backend, planner,
                               // So the queue can record which P0 selection the
                               // run read. It resolves the selection itself; this
                               // only says which mission to ask.
                               mission_id: missionId,
                               // Only what the chosen seeder declares. Settings
                               // are kept while you switch between seeders, so
                               // picking a model and then going back to
                               // Deterministic still sent the provider along and
                               // the queue refused the whole run: "deterministic
                               // takes no ['provider']".
                               seed_config: Object.fromEntries(
                                 Object.entries(seedConfig).filter(([field, v]) =>
                                   v !== "" && (planners.find((p) => p.id === planner)
                                     ?.configures ?? []).some((c) => c.field === field))),
                               seed_probe_mode: seedProbeMode,
                               seed_probe_top_k: seedProbeTopK,
                               seed_probe_generations: seedProbeGenerations,
                               max_tasks: maxTasks, reason,
                               // Only what was touched. An empty field means
                               // "whatever the queue's own default is", which is
                               // not the same as sending the number the form
                               // happened to display.
                               options: Object.fromEntries(
                                 Object.entries(options).filter(([, v]) => v !== "")),
                             }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(readable(body.detail));
      return body;
    },
    onSuccess: () => { client.invalidateQueries({ queryKey: ["artifacts"] }); onDone(); },
  });

  if (!sample) {
    return (
      <div className="body-pad">
        <p>
          Pick a scroll first, in the selector below the phase list.
          <Info label="Why a run covers only one scroll">
            A run covers one scroll, because the grid it searches is that scroll's.
          </Info>
        </p>
      </div>
    );
  }

  return (
    <div className="body-pad launcher">
      {/* The scroll a run covers is derived from the input, so the header shows
          that -- not the page's subject. It used to show both: "scroll PHerc0826"
          beside a dropdown holding PHerc0841, and then a paragraph explaining the
          contradiction it had just created. One number, from the thing that
          decides it, and the paragraph is unnecessary. */}
      <div className="runhead">
        <span className="runhead-cover">
          <span className="runhead-label">covers</span>
          <b>{covers ?? "no scroll"}</b>
        </span>
        <label className="runhead-reads">
          <span className="runhead-label">reading</span>
          {reads.length ? (
            <select value={input} onChange={(e) => setInput(e.target.value)}>
              {reads.map((r) => (
                <option key={r.artifact_id} value={r.artifact_id}>
                  {r.sample_id} · {r.phase} {r.artifact_id.split(":").pop()}
                  {r.note ? ` — ${r.note}` : ""}{r.selected ? " (in use)" : ""}
                </option>
              ))}
            </select>
          ) : (
            <span className="dash">
              the current P0 selection — nothing versioned is registered for this
              scroll yet, so there is nothing to pick between
            </span>
          )}
        </label>
        {covers && sample && covers !== sample && (
          <span className="dash runhead-aside">
            the rest of this page is filtered to {sample}
          </span>
        )}
      </div>

      <div className="runsizing">
        <label className="inlinecheck">
          grown by
          <select value={backend} onChange={(e) => setBackend(e.target.value)}>
            {/* The unusable ones stay listed and are disabled. Removing them
                would answer "can we run ScrollFiesta yet" by saying nothing,
                and the answer is no rather than nothing -- one is in
                FAILED_REFERENCE_CONTROL and the other has no local code. */}
            {backends.map((b) => (
              <option key={b.id} value={b.id} disabled={!b.adoptable}
                      title={b.note}>
                {b.name}
                {/* Three answers, not two. One is not adoptable because it
                    failed its control or has no local code; the spiral fitter
                    is neither -- it runs, and it is not planned as a grow, so
                    saying "not yet working" of it would be false. */}
                {b.adoptable ? ""
                  : b.runs_from ? ` — runs from ${b.runs_from.phase}`
                  : " — not yet working"}
              </option>
            ))}
          </select>
        </label>
        <label className="inlinecheck">
          how many cells
          <input type="number" min={1} max={48} value={maxTasks}
                 onChange={(e) => setMaxTasks(Number(e.target.value))} />
        </label>
      </div>
      {chosen && !chosen.adoptable && (
        <p className="hint">
          {chosen.note}
          {chosen.runs_from && (
            <> Queue it under <Link to={`/phase/${chosen.runs_from.phase}`}>
              {chosen.runs_from.phase}
            </Link>, on the {chosen.runs_from.lane} lane, against{" "}
            {chosen.runs_from.profile_id}.</>
          )}
        </p>
      )}

      {/* The seed is what P1 decides -- only the volume comes from P0 -- so it
          gets the room a decision needs rather than a dropdown among dropdowns.
          Two planners searching the same cell can pick different seeds and grow
          different sheets, both correct, so which one ran is provenance. */}
      <fieldset className="seedpick">
        <legend>
          Where the seed comes from
          {/* The distinction between choosing cells and choosing a point inside
              one is worth stating once, not at the top of the decision every
              time. Four lines of prose above the first control is a wall. */}
          <Explain text={
            "Not where to look. That is settled when the run is queued, by cell "
            + "size, clearance and ranking under Options — by then the fleet has "
            + "already chosen the cells. This chooses which point inside one of "
            + "them, and with what parameters. Two seeders given the same cell can "
            + "pick different points and grow different sheets, both correct, so "
            + "which one ran is part of what the surface means."} />
        </legend>
        <div className="segswitch">
          {([["discovered", "Found in a cell"],
             ["supplied", "Points I supply"]] as const).map(([mode, label]) => (
            <button key={mode} aria-current={seedFrom === mode ? "true" : undefined}
                    onClick={(e) => { e.preventDefault(); setSeedFrom(mode); }}>
              {label}
            </button>
          ))}
        </div>

        {seedFrom === "supplied" ? (
          <div className="body-pad">
            <p>
              <Explain text={
                "One point per line: x y z in CT-L0 voxels, comma or space "
                + "separated. Anything after the third number becomes that seed's "
                + "note; lines starting with # are ignored. "
                + 
                "A supplied point skips the prediction, so the CT-material gate -- "
                + "does the raw scan have anything at that coordinate -- is the "
                + "only screen left between the point and hours of growing. It "
                + "still runs. A point outside the volume, or inside the edge "
                + "margin where a surface would grow into the crop, is refused "
                + "with the range it violated."} />
            </p>
            <textarea className="seedpoints" rows={6} value={points}
                      placeholder={"4000 4000 8000 partial lamina, west side\n4112, 3980, 8064"}
                      onChange={(e) => setPoints(e.target.value)} />
            <div className="controls">
              {good.length > 0 && (
                <Pill kind="ok">
                  {good.length} point{good.length === 1 ? "" : "s"} → {covers}
                </Pill>
              )}
              {bad.length > 0 && (
                <Pill kind="crit">
                  line{bad.length === 1 ? "" : "s"} {bad.map((b) => b.line).join(", ")}{" "}
                  need three numbers
                </Pill>
              )}
              {/* Named on the record, from the session. A human seed whose author
                  is self-reported is not auditable. */}
              <span className="dash">recorded as yours, with seed origin human</span>
            </div>
          </div>
        ) : planners.map((p) => (
          <label key={p.id} className={`seedopt ${planner === p.id ? "on" : ""}`}>
            <input type="radio" name="planner" value={p.id}
                   checked={planner === p.id}
                   onChange={() => setPlanner(p.id)} />
            <span className="seedname">
              {p.name}
              {p.kind !== "deterministic" && <Pill kind="run">{p.kind}</Pill>}
              {p.repeatable && <Pill kind="ok">repeatable</Pill>}
            </span>
            {/* The note for the one you picked, the way the settings already
                work. Six sentences at once is a page you skim instead of read,
                and the badges carry what separates them well enough to choose. */}
            {planner === p.id && <span className="seednote">{p.note}</span>}
            {/* Only the chosen seeder shows its settings. All six expanded at
                once is a wall, and five of them are about a lane not running. */}
            {planner === p.id && p.configures.length > 0 && (
              <span className="seedconfig">
                {p.configures.map((c) => (
                  <label key={c.field}>
                    <span>{c.label}</span>
                    {c.type === "select" ? (
                      <select value={seedConfig[c.field] ?? c.options?.[0] ?? ""}
                              onChange={(e) => setSeedConfig(
                                { ...seedConfig, [c.field]: e.target.value })}>
                        {c.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    ) : (
                      <input
                        list={`sug-${p.id}-${c.field}`}
                        value={seedConfig[c.field] ?? ""}
                        placeholder={c.type === "list"
                          ? "comma-separated, 1 to 8"
                          : c.suggestions?.[0] ?? ""}
                        onChange={(e) => setSeedConfig(
                          { ...seedConfig, [c.field]: e.target.value })} />
                    )}
                    {c.suggestions && (
                      <datalist id={`sug-${p.id}-${c.field}`}>
                        {c.suggestions.map((v) => <option key={v} value={v} />)}
                      </datalist>
                    )}
                    {c.note && <em>{c.note}</em>}
                  </label>
                ))}
              </span>
            )}
          </label>
        ))}
      </fieldset>

      {seedFrom === "discovered" && (
        <fieldset className="seedpick">
          <legend>
            Closed-loop seed probe
            <Explain text={
              "seed-probe-v1 runs short, bounded VC3D growths for up to three "
              + "candidate seeds and compares their deterministic geometry "
              + "evidence. It sits beneath the Cost-aware router; it does not "
              + "replace that router or invoke an LLM panel."} />
          </legend>
          <div className="runsizing">
            <label className="inlinecheck">
              mode
              <select aria-label="seed probe mode" value={seedProbeMode}
                      onChange={(e) => setSeedProbeMode(e.target.value as ProbeMode)}>
                {probeModes.map((mode) => (
                  <option key={mode.id} value={mode.id}
                          disabled={mode.id === "select" && !probeCanSelect}>
                    {mode.name}{mode.id === "select" && !probeCanSelect
                      ? " — unavailable for this source/rollout" : ""}
                  </option>
                ))}
              </select>
            </label>
            {seedProbeMode !== "off" && (
              <>
                <label className="inlinecheck">
                  candidates (1–3)
                  <input aria-label="seed probe candidates" type="number"
                         min={seedProbeMode === "select"
                           ? Math.max(2, probeConfig?.top_k.minimum ?? 1)
                           : probeConfig?.top_k.minimum ?? 1}
                         max={probeConfig?.top_k.maximum ?? 3}
                         value={seedProbeTopK}
                         onChange={(e) => setSeedProbeTopK(Number(e.target.value))} />
                </label>
                <label className="inlinecheck">
                  generations each (10–20)
                  <input aria-label="seed probe generations" type="number"
                         min={probeConfig?.generations.minimum ?? 10}
                         max={probeConfig?.generations.maximum ?? 20}
                         value={seedProbeGenerations}
                         onChange={(e) => setSeedProbeGenerations(Number(e.target.value))} />
                </label>
              </>
            )}
          </div>
          <p className="seednote">
            {probeModes.find((mode) => mode.id === seedProbeMode)?.note}
            {" "}
            {seedProbeMode === "shadow"
              ? "The canonical grow is unchanged."
              : seedProbeMode === "select"
                ? "If the evidence cannot separate a winner, the attempt enters human review rather than guessing."
                : ""}
          </p>
          {seedProbeMode === "select" && !probeCanSelect && (
            <p className="formerror">
              {probeConfig?.select_readiness?.reason
                ?? "Select needs Cost-aware v2 or Deterministic v2, an approved rollout, and content-locked CT/m7 inputs."}
              {" "}Shadow remains observational.
            </p>
          )}
          {seedProbeMode === "select" && probeCanSelect && (
            <p className="hint">
              Approved causal benchmark{" "}
              <code>{probeConfig?.select_readiness?.benchmark_id ?? "unknown"}</code>
              {probeConfig?.select_readiness?.decision_receipt_sha256
                ? <> · receipt{" "}
                    <code>
                      {probeConfig.select_readiness.decision_receipt_sha256.slice(0, 12)}
                    </code>
                  </>
                : null}
              {" "}· this sample is authorized · immutable CT/m7 source lock verified
              {" "}· review owner assigned.
            </p>
          )}
          <p className="hint">
            {probeConfig?.note
              ?? "This deterministic micro-growth layer remains beneath the Cost-aware router."}
            {" "}
            {seedProbeMode === "select"
              ? "For a unique winner, Cost-aware deliberately takes its zero-provider deterministic lane; Fusion reasoning is bypassed for that attempt. "
              : ""}
            {probeConfig?.caveat
              ?? "A probe winner is not proof of the correct lamina; geometry and human review remain separate evidence."}
          </p>
        </fieldset>
      )}

      <div className="controls">
        <button onClick={() => setShowOptions((v) => !v)}>
          {showOptions ? "hide options" : `Options (${knobs.data?.options.length ?? 0})`}
        </button>
        {Object.values(options).filter((v) => v !== "").length > 0 && (
          <Pill kind="warn">
            {Object.values(options).filter((v) => v !== "").length} changed from default
          </Pill>
        )}
      </div>

      {showOptions && knobs.data && (
        <>
          {/* One group open at a time, with <details> rather than state: the
              browser already knows how to do this, and five sections expanded at
              once was a wall nobody read. Only the first is open, because "where
              to look" is the one most runs touch. */}
          {Array.from(new Set(knobs.data.options.map((o) => o.group))).map((group, index) => {
            const fields = knobs.data!.options.filter((o) => o.group === group);
            const set = fields.filter((o) => (options[o.field] ?? "") !== "").length;
            return (
              <details key={group} className="knobs" open={index === 0}>
                <summary>
                  {group}
                  <span className="knobs-count">
                    {set > 0 ? `${set} of ${fields.length} set` : `${fields.length} settings`}
                  </span>
                </summary>
                <div className="knobgrid">
                  {fields.map((o) => (
                    <label key={o.field}>
                      <span className="knoblabel">
                        {o.label}
                        <Explain text={o.note} />
                      </span>
                      {o.kind === "toggle" ? (
                        <select value={options[o.field] ?? ""}
                                onChange={(e) => setOptions(
                                  { ...options, [o.field]: e.target.value })}>
                          <option value="">default (on)</option>
                          <option value="on">on</option>
                          <option value="off">off</option>
                        </select>
                      ) : o.kind === "choice" ? (
                        <select value={options[o.field] ?? ""}
                                onChange={(e) => setOptions(
                                  { ...options, [o.field]: e.target.value })}>
                          <option value="">default</option>
                          {o.choices?.map((c) => <option key={c} value={c}>{c}</option>)}
                        </select>
                      ) : (
                        <input
                          type={o.kind === "text" ? "text" : "number"}
                          step={o.kind === "float" ? "any" : undefined}
                          value={options[o.field] ?? ""}
                          placeholder="default"
                          onChange={(e) => setOptions(
                            { ...options, [o.field]: e.target.value })} />
                      )}
                    </label>
                  ))}
                </div>
              </details>
            );
          })}

          {/* One line, not six tiles. Read-only numbers rendered like readouts
              made the thing you cannot change the loudest thing on the page. */}
          <p className="hint knobs-readonly">
            <b>VC3D growth, fixed per attempt:</b>{" "}
            {knobs.data.growth.parameters
              .map((g) => `${g.name} ${String(g.default)}${
                g.range.startsWith("pinned") ? " (pinned)" : ` [${g.range}]`}`)
              .join(" · ")}
            . {knobs.data.growth.note}
          </p>
        </>
      )}

      {notGrowable && (
        <p className="formerror">
          {covers} cannot be grown here: {source?.note}
          {" "}Use <b>Import surface</b> for it, or cover one of{" "}
          {(source?.growable_scrolls ?? []).join(", ")}.
        </p>
      )}

      <div className="controls">
        <input className="search" value={reason}
               placeholder={needsReason
                 ? "why this backend, given it is not adoptable…"
                 : "what you are trying (optional, kept with the run)"}
               onChange={(e) => setReason(e.target.value)} />
        <button disabled={queue.isPending || notGrowable || !inputValid
                          || (needsReason && !reason.trim())
                          || (seedFrom === "discovered"
                              && ((seedProbeMode === "select" && !probeCanSelect)
                                || seedProbeTopK < (probeConfig?.top_k.minimum ?? 1)
                                || seedProbeTopK > (probeConfig?.top_k.maximum ?? 3)
                                || seedProbeGenerations
                                  < (probeConfig?.generations.minimum ?? 10)
                                || seedProbeGenerations
                                  > (probeConfig?.generations.maximum ?? 20)))
                          || (seedFrom === "supplied"
                              && (good.length === 0 || bad.length > 0))}
                onClick={() => queue.mutate()}>
          {queue.isPending ? "queueing…"
            : seedFrom === "supplied"
              ? `Queue ${good.length || "no"} point${good.length === 1 ? "" : "s"}`
              : "Queue"}
        </button>
      </div>

      <Info label="Why a run does not name coordinates" title="Where a run looks">
        A run does not name coordinates. The fleet decides which grid cells are
        still uncovered and the planner picks a seed inside one — nothing typed
        here chooses where to look, which is what lets the result be read later.
      </Info>
      {/* A block, not a pill. A pill is an inline badge and this is a paragraph:
          it arrived as one unbroken 300-character line, clipped at the card edge
          with the rest unreachable. The truncation went too -- an error worth
          showing is worth showing all of. */}
      {queue.isError && <p className="formerror">{String(queue.error)}</p>}
    </div>
  );
}
