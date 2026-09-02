import {Suspense, useMemo, useState} from "react";
import { useParams } from "react-router";
import { SourceTile } from "../components/SourceTile";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, Empty, Pill } from "../components/Bits";
import { scoped, useMission, useSubject } from "../mission";
import { lazyRoute } from "../lazyRoute";
import { failure } from "../api";

/**
 * One phase, always the same four blocks in the same order:
 *
 *   State      what exists here now
 *   Action     what can be queued, or why nothing can
 *   Artefacts  what came out, with its gates
 *   Contract   what it consumes, produces, and how it fails
 *
 * A phase with no runner shows an empty Action block carrying the reason.
 * A phase that is absent from the interface teaches nothing; a phase that says
 * "observed, not driven, because no runner is registered" teaches exactly what
 * is missing.
 */

const Runs = lazyRoute(() => import("./Runs"));
const Intake = lazyRoute(() => import("./Intake"));
const Certification = lazyRoute(() => import("./Certification"));
const Flattening = lazyRoute(() => import("./Flattening"));
const SegmentationView = lazyRoute(() => import("./Segmentation"));
const Screening = lazyRoute(() => import("./Screening"));
const Coverage = lazyRoute(() => import("./Coverage"));
const InkLanes = lazyRoute(() => import("./InkLanes"));
const InkMaps = lazyRoute(() => import("./InkMaps"));
const InkLauncher = lazyRoute(() => import("./Command"));

type Contract = {
  id: string; name: string; slug: string; one_line: string;
  consumes: string; produces: string; lives_in: string[];
  maturity: string; distributed?: boolean;
  how_to_run: string; how_it_fails: string; gate: string | null;
  prerequisites?: {
    needs: string | null; produced_by: string[]; may_come_from_outside: boolean;
    external_source: string | null; note: string | null;
  };
};
type Component = {
  component: string; phases: string[]; status: string; remote?: string;
  viewer?: string; local_path: string; what_it_does: string;
  entry_points?: Record<string, string>; why_it_matters_here?: string;
  known_state?: string; validation?: string;
};
/**
 * How long ago, in the shortest form that is still true.
 *
 * The age matters as much as the line: a progress bar from four seconds ago is
 * a job that is working, and the identical line from nine minutes ago is a job
 * that has stopped saying anything. Rendering the line alone would make those
 * two look the same, which is the failure this column exists to end.
 */
function ago(at: string): string {
  const seconds = Math.max(0, Math.round((Date.now() - Date.parse(at)) / 1000));
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

type Job = {
  job_id: string; sample_id: string; phase: string; state: string;
  attempts: number; max_attempts: number; result: Record<string, any> | null;
  // The newest line the job wrote, carried by the heartbeat that renews its
  // lease. Absent until a worker has claimed it and it has said something.
  progress?: { line: string; source: string; at: string } | null;
};
// One rendered page, as the worker recorded it on the job.
type Plate = {
  file: string; wrap?: string; width?: number; height?: number;
  bytes?: number; sha256?: string;
};
type Profile = {
  profile_id: string; schema: string; method_id: string | null;
  adapter: string | null; path: string;
  input_contract: Record<string, any>; default_execution: Record<string, any>;
  disqualified: boolean; registry_status: string | null; registry_policy: string | null;
};
type PhaseData = {
  contract: Contract; components: Component[];
  profiles: Profile[]; profile_stage: string | null;
  state: Record<string, unknown>; artefacts: any[];
  queueable: boolean; queueable_reason: string | null; jobs: Job[];
};

const JOB_KIND: Record<string, "ok" | "run" | "crit" | "warn" | "neg"> = {
  succeeded: "ok", running: "run", leased: "run",
  failed: "crit", cancelled: "neg", pending: "warn",
};

// The parameters each queueable phase accepts, mirroring the server allowlist.
// Keeping them side by side is deliberate: a field here that the server does
// not know is refused rather than silently dropped.
/**
 * The queue's own parameter list, drawn as a form.
 *
 * This component used to read PHASE_FIELDS, a table in this file that repeated
 * the queue's parameters, their types and their wording. Every parameter added
 * on one side was invisible on the other until somebody remembered to add it
 * twice -- and the ones that mattered most were the ones added last: the
 * direction along the normal, the depth window, the chain from a render to the
 * detector. All of them reached the API and none of them reached anybody who
 * was not typing curl.
 *
 * `filled_by_deployment` fields are not drawn at all. Where a render publishes
 * is a property of the machine room and the panel fills it in; asking for it
 * here would be asking somebody to get it wrong.
 */
type SchemaField = {
  name: string; type: "text" | "integer" | "number" | "boolean" | "json";
  required: boolean; lane: string | string[] | null; label: string;
  note: string | null; placeholder: string | null;
  filled_by_deployment: boolean;
  // Which phase's jobs this field may name, and the ones this mission has.
  names_a_job_from?: string | null;
  choices?: { value: string; note: string }[];
};
type Schema = {
  available: boolean; reason?: string; fields: SchemaField[];
  lanes: {
    id: string; name: string; note: string; validated: string | null;
    profiles?: string[];
  }[];
  exactly_one_of: { lane?: string; names: string[] }[];
};

const WIDE = /path|dir|surface|url|volume|checkpoint|store|output|stack/;

function QueueForm({ phase, subject }: { phase: string; subject: string | null }) {
  const client = useQueryClient();
  const { missionId } = useMission();
  const [lane, setLane] = useState<string>("");
  const [typedSample, setTypedSample] = useState("");
  const sample = missionId ? (subject ?? "") : typedSample;
  const [values, setValues] = useState<Record<string, string | boolean>>({});

  const schema = useQuery<Schema>({
    // The mission and scroll are part of the key: a field that names another
    // job is offered this mission's jobs, so the answer is not the same for
    // every mission the way the field list is.
    queryKey: ["phase-parameters", phase, missionId ?? "", sample ?? ""],
    queryFn: async () => {
      const scope = new URLSearchParams();
      if (missionId) scope.set("mission", missionId);
      if (sample) scope.set("sample", sample);
      const query = scope.toString();
      const response = await fetch(
        `/api/phases/${phase}/parameters${query ? `?${query}` : ""}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
    staleTime: 300_000,
  });

  const lanes = schema.data?.lanes ?? [];
  const chosen = lane || lanes[0]?.id || "";
  const chosenLane = lanes.find((candidate) => candidate.id === chosen);
  const fixedProfile = chosenLane?.profiles?.length === 1
    ? chosenLane.profiles[0]
    : undefined;
  const fields = (schema.data?.fields ?? []).filter(
    (f) => f.name !== "lane" && !f.filled_by_deployment
      && (!f.lane || (Array.isArray(f.lane) ? f.lane.includes(chosen) : f.lane === chosen))
    // A lane's own fields are the ones it declares; everything unlaned is
    // offered whatever renderer is chosen, which is how depth and the cache
    // budget reach both.
  );

  const enqueue = useMutation({
    mutationFn: async () => {
      const parameters: Record<string, unknown> = {};
      if (lanes.length) parameters.lane = chosen;
      for (const field of fields) {
        const raw = values[field.name];
        if (raw === undefined || raw === "" || raw === false) continue;
        parameters[field.name] =
          field.type === "boolean" ? true
          : field.type === "integer" ? parseInt(String(raw), 10)
          : field.type === "number" ? Number(raw)
          : field.type === "json" ? JSON.parse(String(raw))
          : String(raw).trim();
      }
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sample, phase, parameters,
                               profile_id: fixedProfile,
                               mission_id: missionId }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`);
      return body as { job_id: string };
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["phase", phase] }),
  });

  // Neither is required and exactly one must be there, which "required" cannot
  // express -- so the queue states the pairs and this checks them.
  const rules = (schema.data?.exactly_one_of ?? []).filter(
    (rule) => !rule.lane || rule.lane === chosen);
  const unmet = rules.filter(
    (rule) => rule.names.filter((name) => String(values[name] ?? "").trim()).length !== 1);
  // A field in one of those pairs is required by the pair, not on its own.
  //
  // Exempting only the fields of an *unmet* rule made naming one of them turn
  // the other back into a requirement: P4's surface path is `required` and
  // pairs with the flattened surface id, so filling the id satisfied the rule
  // and re-armed the path -- and the button stayed dead however the form was
  // filled in. A flattened sheet could not be rendered from a browser at all.
  const governed = new Set(rules.flatMap((rule) => rule.names));
  const ready = sample && unmet.length === 0
    && fields.filter((f) => f.required && !governed.has(f.name))
             .every((f) => String(values[f.name] ?? "").trim());

  if (schema.isLoading) return <Empty>loading the parameters…</Empty>;
  if (schema.isError || schema.data?.available === false)
    return <Empty>{schema.data?.reason ?? String(schema.error)}</Empty>;

  return (
    <div className="body-pad">
      <div className="formgrid">
        <label>
          Scroll
          <input value={sample} onChange={(e) => setTypedSample(e.target.value)}
                 disabled={Boolean(missionId)}
                 placeholder={missionId ? "select a scroll in P0" : "PHerc0139"} />
        </label>
        {lanes.length > 1 ? (
          <label className="wide">
            Renderer
            <select value={chosen} onChange={(e) => setLane(e.target.value)}>
              {lanes.map((l) => (
                <option key={l.id} value={l.id}>{l.name}</option>
              ))}
            </select>
            <span className="dash">
              {chosenLane?.note}
              {chosenLane?.validated && ` Validated: ${chosenLane.validated}`}
              {fixedProfile && ` Frozen profile: ${fixedProfile}`}
            </span>
          </label>
        ) : chosenLane && (chosenLane.note || fixedProfile) ? (
          /* One lane, but one worth naming: it pins the profile this run is
             made against, and that id is what the queue checks. Shown rather
             than offered -- there is nothing to choose. */
          <div className="wide">
            <strong>{chosenLane.name}</strong>
            <span className="dash">
              {chosenLane.note}
              {fixedProfile && ` Frozen profile: ${fixedProfile}`}
            </span>
          </div>
        ) : null}
        {fields.map((f) => (
          <label key={f.name}
                 className={f.type === "boolean" ? "wide toggle"
                            : WIDE.test(f.name) ? "wide" : ""}>
            {f.type === "boolean" ? (
              <>
                <input type="checkbox" checked={values[f.name] === true}
                       onChange={(e) => setValues({ ...values, [f.name]: e.target.checked })} />
                {" "}{f.label}
              </>
            ) : (
              <>
                {f.label}{f.required && " *"}
                {/* A field that names another job lists the jobs it can name.
                    Free text stayed possible -- a job from outside the mission
                    is still a legitimate answer for anyone who has the id -- so
                    this is a datalist, not a select that forbids the rest. */}
                <input value={String(values[f.name] ?? "")}
                       inputMode={f.type === "integer" || f.type === "number"
                         ? "decimal" : undefined}
                       list={f.choices?.length ? `${phase}-${f.name}` : undefined}
                       onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                       placeholder={f.placeholder ?? undefined} />
                {f.choices?.length ? (
                  <datalist id={`${phase}-${f.name}`}>
                    {f.choices.map((c) => (
                      <option key={c.value} value={c.value} label={c.note} />
                    ))}
                  </datalist>
                ) : null}
              </>
            )}
            {f.note && <span className="dash">{f.note}</span>}
          </label>
        ))}
        {unmet.map((rule) => (
          <span className="dash" key={rule.names.join()}>
            name exactly one of: {rule.names.join(" or ")}
          </span>
        ))}
      </div>
      <div className="controls">
        <button onClick={() => enqueue.mutate()} disabled={!ready || enqueue.isPending}>
          {enqueue.isPending ? "queueing…" : `Queue ${phase} job`}
        </button>
        {enqueue.isError && <Pill kind="crit">{String(enqueue.error)}</Pill>}
        {enqueue.isSuccess && <Pill kind="ok">queued {enqueue.data.job_id}</Pill>}
      </div>
    </div>
  );
}

// P1 contributes its own tabs to this bar rather than nesting a second one
// inside its view. Two levels of tabs made the reader choose twice to reach
// one table, and the outer label had to name what the inner ones already did.
/**
 * State keys a phase's own view already puts on screen, and where.
 *
 * The generic strip below prints one tile per key the API returns. That is a
 * safety net -- coverage numbers once arrived in these rows with nothing to
 * show them -- but a net that catches everything also catches what the view
 * underneath has already laid out by hand. P1 rendered `attempts` in two
 * adjacent rows because of it, and put a grid-version count next to a surface
 * count that has nothing to do with it.
 *
 * "view" means the phase draws it itself and the strip must not repeat it. A
 * tab name means it is real but belongs on that tab: cells and grids describe
 * the coverage grid, and Coverage is a tab that already exists.
 */
const PLACED_BY_HAND: Record<string, Record<string, Sub | "view">> = {
  // P2 draws all four itself, and styles one of them: unmeasured turns the tile
  // amber, which the generic strip cannot do. Without this entry the page showed
  // the same four counts twice, in two different orders -- which is the failure
  // this table was added for, on the phase that was never added to it.
  P2: {
    surfaces: "view", certified: "view", unmeasured: "view", rejected: "view",
  },
  P1: {
    surfaces: "view", area_cm2: "view", tasks: "view", attempts: "view",
    cells_attempted: "coverage", cells_with_surface: "coverage",
    grid_versions: "coverage",
  },
};

type Sub = "state" | "run" | "profiles" | "artefacts" | "coverage" | "lanes"
         | "maps" | "runs" | "segments" | "new" | "import";

// Phases with their own view. The rest have nothing to put under Artefacts,
// and an empty tab is worse than an absent one: it promises a place to look.
const HAS_VIEW = new Set(["P0", "P1", "P2", "P3", "P5", "P7"]);
// "Artefacts" is right for a list of outputs and wrong for a page that counts
// two populations of surface and launches work.
const VIEW_LABEL: Record<string, string> = {
  // P1 is "Work" rather than "Segments and runs": the view separates those
  // into their own tabs now, so naming two of the three things inside it
  // from outside made the second tab bar look like a contradiction.
  P0: "Scrolls", P2: "Certification", P3: "Sheets",
  P5: "Runs", P7: "Screening",
};

// P1's tabs, flat. Runs first because the question on arriving is what the
// fleet is doing, not what it has accumulated.
const P1_TABS: [Sub, string][] = [
  ["runs", "Runs"], ["segments", "Segments"], ["coverage", "Coverage"],
  ["profiles", "Profiles"], ["new", "New run"], ["import", "Import surface"],
];

/**
 * What the phase is, on the line that already names it.
 *
 * This was a card of its own under the tabs, which put a paragraph of reference
 * between the control you just used and the thing it changed -- collapsed, so
 * mostly it was a closed box taking a row. It does not change between visits and
 * it belongs to the title, so it lives there now behind the same info button
 * Configuration uses for the same reason.
 */
function ContractInfo({ contract }: { contract: Contract }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="infowrap">
      <button className="infobtn" aria-expanded={open}
              aria-label={`What ${contract.id} consumes, produces and how it fails`}
              onClick={() => setOpen((v) => !v)}>
        i
      </button>
      {open && (
        <span className="infopop" role="tooltip">
          <b>{contract.id} · {contract.name}</b>
          <span className="infometa">consumes</span>
          <span>{contract.consumes}</span>
          <span className="infometa">produces</span>
          <span>{contract.produces}</span>
          {contract.prerequisites?.needs && (
            <>
              <span className="infometa">needs first</span>
              <span>
                {contract.prerequisites.needs}
                {contract.prerequisites.produced_by.length > 0 &&
                  ` — normally from ${contract.prerequisites.produced_by.join(", ")}`}
                {contract.prerequisites.may_come_from_outside &&
                  contract.prerequisites.external_source &&
                  ` · or from outside: ${contract.prerequisites.external_source}`}
              </span>
            </>
          )}
          <span className="infometa">how to run it</span>
          <span>{contract.how_to_run}</span>
          <span className="infometa">how it fails</span>
          <span>{contract.how_it_fails}</span>
          {contract.gate && (
            <>
              <span className="infometa warn">gate</span>
              <span>{contract.gate}</span>
            </>
          )}
        </span>
      )}
    </span>
  );
}

export default function Phase() {
  const { phaseId } = useParams();
  const id = (phaseId ?? "P0").toUpperCase();
  // A tab named in the address bar opens on it, so a link can point at the
  // form rather than at the page that contains it.
  const [sub, setSub] = useState<Sub>(
    () => (new URLSearchParams(window.location.search).get("tab") as Sub) || "state");
  const { missionId } = useMission();
  const { subject } = useSubject();
  const { data, isLoading, error } = useQuery({
    queryKey: ["phase", id, missionId, subject],
    queryFn: async () => {
      const base = scoped(`/api/phase/${id}`, missionId);
      const r = await fetch(subject
        ? base + (base.includes("?") ? "&" : "?") + `subject=${encodeURIComponent(subject)}`
        : base);
      if (!r.ok) throw await failure(r);
      return (await r.json()) as PhaseData;
    },
    refetchInterval: 8000,
  });

  const stateRows = useMemo(
    () => Object.entries(data?.state ?? {})
      .filter(([, v]) => v !== null && v !== undefined)
      .filter(([k]) => {
        const placed = (PLACED_BY_HAND[id ?? ""] ?? {})[k];
        // Unclaimed keys still get a tile: that is what this row is for.
        if (!placed) return true;
        // Claimed for a tab, so only there. Claimed by the view, so never.
        return placed === sub;
      }),
    [data, id, sub],
  );

  if (isLoading) return <Empty>loading {id}…</Empty>;
  if (error || !data) return <Empty>{String(error ?? "no data")}</Empty>;
  const c = data.contract;

  // Only offer what exists. P0 has no profiles and nothing to queue -- there is
  // no download step -- so it is one view, and the tab bar disappears entirely.
  const tabs: [Sub, string][] = [];
  const stateRowCount = Object.keys(data.state ?? {}).length;
  // A phase with a view of its own opens on it. P1 has ten profiles and so was
  // landing on the profile list, which is a reference table -- the page looked
  // like the phase had nothing to do.
  if (id === "P1") {
    tabs.push(...P1_TABS);
  } else {
    if (HAS_VIEW.has(id)) tabs.push(["artefacts", VIEW_LABEL[id] ?? "Artefacts"]);
    // What P5 actually produced. The tab above it indexes the legacy receipt
    // tree on disk, which cannot see a screening queued through the fleet --
    // so the phase's own output was reachable only over ssh.
    if (id === "P5") tabs.push(["maps", "Maps"]);
    // Which ink models exist and which of them this queue can actually run.
    if (id === "P5") tabs.push(["lanes", "Models"]);
    if (stateRowCount && !HAS_VIEW.has(id)) tabs.push(["state", "State"]);
    // Profiles before Run: Run carries margin-left:auto to sit at the far right,
    // and anything after it in the DOM goes along for the ride. The views belong
    // together on the left.
    if (data.profiles.length) tabs.push(["profiles", "Profiles"]);
    if (data.queueable || data.jobs.length) tabs.push(["run", "Run"]);
  }
  const active: Sub = tabs.some(([k]) => k === sub) ? sub : (tabs[0]?.[0] ?? "state");

  return (
    <>
      <div className="phasehead">
        <h1>
          <span className="phasehead-id">{c.id}</span>
          {c.name}
        </h1>
        <p>{c.one_line}</p>
        <ContractInfo contract={c} />
        <div className="phasehead-pills">
          <Pill kind={c.maturity === "WORKING" ? "ok" : c.maturity === "NOT_REACHED" ? "neg" : "warn"}>
            {c.maturity.replaceAll("_", " ").toLowerCase()}
          </Pill>
          {c.distributed && <Pill kind="run">distributed</Pill>}
        </div>
      </div>

      {tabs.length > 1 && (
        <nav className="subtabs">
          {/* Starting a run is the one thing on this bar that does something
              rather than showing something, so it does not queue up with the
              views: it sits at the far right and carries the accent. "new" on
              P1, "run" everywhere else -- both are the launcher. */}
          {tabs.map(([k, label]) => (
            <button key={k} onClick={() => setSub(k)}
                    className={k === "new" || k === "run" ? "cta" : undefined}
                    aria-current={active === k ? "page" : undefined}>
              {label}
              {k === "run" && data.jobs.filter((j) =>
                ["pending", "leased", "running"].includes(j.state)).length > 0 && (
                <em className="subtab-count">
                  {data.jobs.filter((j) =>
                    ["pending", "leased", "running"].includes(j.state)).length}
                </em>
              )}
            </button>
          ))}
        </nav>
      )}

      {/* Every phase draws it now. P1 and P0 were excluded because their own
          views already counted things -- and then coverage and the never-intaken
          scrolls started arriving in these rows and nothing put them on screen. */}
      {stateRows.length > 0 && (
        <div className="strip">
          {stateRows.map(([k, v]) => k === "inventory_origin" ? (
            <SourceTile key={k} value={String(v)} />
          ) : (
            <div className="tile steady" key={k}>
              <h2>{k.replaceAll("_", " ")}</h2>
              {typeof v === "number" ? (
                <div className="readout">{v}</div>
              ) : (
                <p style={{ color: "var(--ink)", fontSize: 13 }}>
                  {/* A tally is a tally, not a JSON literal. P6 counts its
                      verdicts and the tile read {"ALIVE":2}, braces and quotes
                      included, for what is two words. */}
                  {v && typeof v === "object" && !Array.isArray(v)
                    ? Object.entries(v as Record<string, unknown>)
                        .map(([key, count]) => `${key.replaceAll("_", " ")} ${
                          typeof count === "object" ? JSON.stringify(count) : String(count)}`)
                        .join(" · ")
                    : typeof v === "object" ? JSON.stringify(v) : String(v)}
                </p>
              )}
            </div>
          ))}
        </div>
      )}




      {active === "run" && (
      <Card
        title="Queue work"
        note={data.queueable ? "the worker builds the command from the phase" : undefined}
      >
        {data.queueable ? (
          id === "P5" ? (
            <Suspense fallback={<Empty>loading the lane launcher…</Empty>}>
              <InkLauncher />
            </Suspense>
          ) : (
            <QueueForm phase={id} subject={subject} />
          )
        ) : (
          <div className="body-pad">
            {id === "P0" ? (
              <>
                <p>
                  P0 has nothing to queue: there is no download step. The source is OME-Zarr over
                  HTTPS and every reader fetches only the chunks it touches.
                </p>
                <p>
                  What this phase does is <b>select</b> — which scrolls join the mission. That is
                  under Artefacts.
                </p>
              </>
            ) : (
              <>
                <p>{data.queueable_reason}</p>
                <p>
                  The components below implement it; they are indexed and vendored but not yet
                  wired to a runner this queue can dispatch.
                </p>
              </>
            )}
          </div>
        )}
      </Card>
      )}

      {active === "run" && data.jobs.length > 0 && (
        <Card title="Queue" note={`${data.jobs.length} jobs at this phase`}>
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l">Job</th>
                  <th className="l">Scroll</th>
                  <th className="l">State</th>
                  <th>Try</th>
                  {/* A running job used to be `running` and four blanks, for as
                      long as it took. Telling one that is working from one that
                      has wedged meant opening a shell on the host -- where the
                      output was buffered until the process exited anyway. */}
                  <th className="l">Doing</th>
                  <th className="l">Result</th>
                </tr>
              </thead>
              <tbody>
                {data.jobs.map((j) => (
                  <tr key={j.job_id}>
                    <td className="l wrap">{j.job_id}</td>
                    <td className="l">{j.sample_id}</td>
                    <td className="l">
                      <Pill kind={JOB_KIND[j.state] ?? "neg"}>{j.state}</Pill>
                    </td>
                    <td>
                      {j.attempts}/{j.max_attempts}
                    </td>
                    <td className="l wrap doing">
                      {j.progress?.line
                        ? <><code>{j.progress.line}</code>
                            <span className="dash"> · {ago(j.progress.at)}</span></>
                        : <span className="dash">
                            {j.state === "running" ? "nothing said yet" : ""}
                          </span>}
                    </td>
                    <td className="l wrap">
                      {j.result?.error ? String(j.result.error).slice(0, 70) : ""}
                      {/* A structure score over an upsampled render peaks at the
                          upsampling factor. When the strongest repetition in the
                          screened window is the grid, the verdict beside it is
                          reading the mesh -- so the reader sees that first. */}
                      {j.result?.grid_alarm?.alarm && (
                        <span title={String(j.result.grid_alarm.reason ?? "")}>
                          <Pill kind="warn">grid</Pill>
                        </span>
                      )}
                      {j.result?.liveness?.verdict && (
                        <Pill kind={j.result.liveness.verdict === "ALIVE" ? "ok" : "crit"}>
                          {j.result.liveness.verdict}
                        </Pill>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* What P9 made, which was reachable only over SSH.
          The phase reported "plate runs succeeded: 1" and the 38 pages it
          rendered -- the deliverable of the whole pipeline -- had nowhere to be
          looked at. The plate set is on the job's own result, so this needs no
          second index of anything. */}
      {active === "run" && data.jobs.some((j) => j.result?.plate_set?.plates?.length) && (
        <Card title="Plates"
              note={`${data.jobs.reduce((n, j) =>
                n + (j.result?.plate_set?.plates?.length ?? 0), 0)} pages rendered`}>
          {data.jobs.filter((j) => j.result?.plate_set?.plates?.length).map((j) => (
            <div className="body-pad" key={j.job_id}>
              <p className="dash">
                {j.job_id} · {j.sample_id}
                {j.result?.wrote_to ? ` · ${j.result.wrote_to}` : ""}
              </p>
              <div className="plates">
                {(j.result!.plate_set.plates as Plate[]).map((plate) => (
                  <a key={plate.file} className="plate"
                     href={`/api/jobs/${j.job_id}/plate/${encodeURIComponent(plate.file)}`}
                     target="_blank" rel="noreferrer">
                    <img loading="lazy" alt={plate.file}
                         src={`/api/jobs/${j.job_id}/plate/${encodeURIComponent(plate.file)}`} />
                    <span className="dash">
                      {plate.wrap ?? plate.file} · {plate.width}×{plate.height}
                    </span>
                  </a>
                ))}
              </div>
            </div>
          ))}
        </Card>
      )}

      {active === "profiles" && (
        <div className="strip">
          <div className="tile steady">
            <h2>profiles</h2>
            <div className="readout">{data.profiles.length}</div>
          </div>
          <div className={data.profiles.some((p) => p.disqualified) ? "tile warn" : "tile steady"}>
            <h2>disqualified</h2>
            <div className="readout">{data.profiles.filter((p) => p.disqualified).length}</div>
          </div>
          <div className="tile steady">
            <h2>schema</h2>
            <p style={{ color: "var(--ink)", fontSize: 13 }}>
              {data.profile_stage ?? "none"}
            </p>
          </div>
        </div>
      )}

      {active === "profiles" && (
        <Card
          title="Profiles"
          note={data.profile_stage
            ? `framework/profiles/${data.profile_stage} — frozen, selected by id`
            : "this phase declares no profiles"}
        >
          <div className="body-pad">
            <p>
              A profile is a frozen declaration of how this phase runs. The concept is shared
              across phases; the schema is not — these are {data.profile_stage ?? "none"}, and
              their fields mean nothing in another phase.
            </p>
            <p>
              They are read-only here. A profile's sha256 is bound into the receipts of every run
              that used it, so changing one would break that binding: a change means a new version
              committed to git, which is the same reason a detector is selected by id rather than
              edited in place.
            </p>
          </div>
          {data.profiles.length > 0 && (
            <div className="scroller">
              <table>
                <thead>
                  <tr>
                    <th className="l">Profile</th>
                    <th className="l">Schema</th>
                    {/* The column that takes the slack: the others are
                        identifiers and a pill, all content-sized, so without
                        naming one the table stopped short of its card and left
                        a third of the width empty. */}
                    <th className="l grow">Key fields</th>
                    <th className="l">Adapter</th>
                    <th className="l">State</th>
                  </tr>
                </thead>
                <tbody>
                  {data.profiles.map((p) => {
                    const c = p.input_contract ?? {};
                    const fields = [
                      c.model_type && `model ${c.model_type}`,
                      c.frames && `${c.frames} frames`,
                      c.tile_size_y_x?.[0] && `tile ${c.tile_size_y_x[0]}`,
                      c.training_pixel_um && `${c.training_pixel_um} µm`,
                      c.max_clip_value && `clip ${c.max_clip_value}`,
                    ].filter(Boolean) as string[];
                    return (
                      <tr key={p.profile_id + p.path}>
                        {/* Identifiers, so they do not wrap: `wrap` caps a column
                            at 18ch and broke a 44-character profile id across
                            three lines while the table sat half the width of the
                            page. The scroller handles the overflow case. Only
                            the key fields are prose enough to break. */}
                        <td className="l">{p.profile_id}</td>
                        <td className="l"><code>{p.schema.replace("campaignx.", "")}</code></td>
                        <td className="l wrap grow">
                          {fields.length ? fields.join(" · ") : <span className="dash">—</span>}
                        </td>
                        <td className="l">
                          {p.adapter ? p.adapter.split("/").pop() : <span className="dash">—</span>}
                        </td>
                        <td className="l">
                          {p.disqualified ? (
                            <Pill kind="crit">disqualified</Pill>
                          ) : c.model_type ? (
                            <Pill kind="ok">routable by model_type</Pill>
                          ) : (
                            <Pill kind="neg">routable by adapter</Pill>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {data.profiles.filter((p) => p.disqualified).map((p) => (
            <div className="body-pad" key={p.profile_id}>
              <p><b>{p.method_id}</b> — {p.registry_status}</p>
              <p>{p.registry_policy}</p>
            </div>
          ))}
          {data.profiles.length === 0 && <Empty>no profile declared for this phase</Empty>}
        </Card>
      )}

      {active === "coverage" && (
      <Suspense fallback={<Empty>loading…</Empty>}>
        <Coverage sample={subject ?? undefined} mission={missionId ?? undefined} />
      </Suspense>
      )}

      {active === "maps" && (
      <Suspense fallback={<Empty>loading…</Empty>}>
        <InkMaps sample={subject ?? undefined} mission={missionId ?? undefined} />
      </Suspense>
      )}

      {active === "lanes" && (
      <Suspense fallback={<Empty>loading…</Empty>}>
        <InkLanes />
      </Suspense>
      )}

      {active === "artefacts" && (
      <Suspense fallback={<Empty>loading…</Empty>}>
        {id === "P0" && <Intake />}
        {id === "P2" && <Certification mission={missionId ?? undefined}
                                        sample={subject ?? undefined} />}
        {id === "P3" && <Flattening sample={subject ?? undefined}
                                    mission={missionId ?? undefined} />}
        {id === "P5" && <Runs />}
        {id === "P7" && <Screening />}
      </Suspense>
      )}

      {/* P1's three working tabs are on the one bar above, so its view is
          handed the active one instead of drawing a second bar to pick it. */}
      {id === "P1" && active !== "profiles" && (
        <Suspense fallback={<Empty>loading…</Empty>}>
          <SegmentationView job={active as "runs" | "segments" | "new" | "import"}
                            onSwitch={setSub} />
        </Suspense>
      )}

      {/* A phase with no tabs and nothing to run has only its contract, and the
          info button is a control you have to know to press. P3 rendered a
          heading over an empty page before that contract was shown anywhere;
          hiding it behind a button everywhere would put it back. */}
      {tabs.length === 0 && (
        <Card title={`What ${c.id} is`}
              note={c.gate ? <Pill kind="warn">gated</Pill> : undefined}>
          <div className="body-pad">
            <dl className="contract">
              <dt>Consumes</dt><dd>{c.consumes}</dd>
              <dt>Produces</dt><dd>{c.produces}</dd>
              <dt>How to run it</dt><dd>{c.how_to_run}</dd>
              <dt>How it fails</dt><dd>{c.how_it_fails}</dd>
              {c.gate && <><dt>Gate</dt><dd>{c.gate}</dd></>}
            </dl>
          </div>
        </Card>
      )}

      {active === "artefacts" && id === "P6" && data.artefacts.length === 0 && (
        <Card title="Artefacts">
          <div className="body-pad">
            <p>
              Liveness is recorded in every run receipt rather than as a separate artefact. The
              counts above are read from those receipts.
            </p>
          </div>
        </Card>
      )}

    </>
  );
}
