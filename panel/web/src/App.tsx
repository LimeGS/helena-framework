import {Suspense} from "react";
import { Navigate, NavLink, Route, Routes } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { MissionFooter, MissionGate, SubjectPicker, scoped, useMission, useSubject } from "./mission";
import markColor from "./assets/brand/helena-mark-color.svg";
import markReverse from "./assets/brand/helena-mark-reverse.svg";
import { Mark } from "./components/Bits";
import { PIPELINE } from "./phases";
import { SessionFoot } from "./Login";
import { lazyRoute } from "./lazyRoute";
import { RouteError } from "./RouteError";

// Three general surfaces plus one route that serves all ten phases. Route-level
// splitting keeps the map shaders and the fleet tables off the first paint.
const Mission = lazyRoute(() => import("./routes/Mission"));
const Configuration = lazyRoute(() => import("./routes/Configuration"));
const Documentation = lazyRoute(() => import("./routes/Docs"));
const Phase = lazyRoute(() => import("./routes/Phase"));
const RunDetail = lazyRoute(() => import("./routes/RunDetail"));
const Compare = lazyRoute(() => import("./routes/Compare"));

// Two general surfaces. Queueing is not one of them: it belongs inside the
// phase that does the work, next to the state that tells you whether to.
const GENERAL = [
  ["/", "Mission"],
  ["/configuration", "Configuration"],
  ["/documentation", "Documentation"],
] as const;

type PhaseStatus =
  | "running" | "queued" | "failed" | "stopped" | "blocked"
  | "ready" | "done" | "waiting" | "elsewhere" | "no-run";

type PhaseBadge = {
  id: string; name: string; maturity: string;
  queueable: boolean; badge: string | null;
  active_jobs: number; running_jobs?: number; queued_jobs?: number;
  status: PhaseStatus;
  why: string;
};

/** A rail row before the server has spoken: the name, and no state yet. */
type RailRow = Partial<PhaseBadge> & { id: string; name: string };

const PLACEHOLDER_ROWS: RailRow[] = PIPELINE.map((p) => ({ id: p.id, name: p.name }));

/**
 * What each state looks like, and what it says when asked.
 *
 * The glyphs differ in shape, not only in colour: a rail where red and green
 * carry the whole message is a rail some people cannot read. Every mark also
 * carries the sentence the server derived, so a dot is never the only
 * explanation available.
 *
 * Two states move, and they move differently. `running` breathes, because a
 * worker is executing right now and the rail should say so. `queued` creeps —
 * work exists and nothing has picked it up, which is a different thing to know
 * and the exact shape of the bug where a phase was routed to an image with no
 * runner for it. Nothing else animates: a sidebar with six things pulsing is a
 * sidebar people stop reading.
 */
// The shapes are in the stylesheet, under .mark-*; only the words are here.
const STATUS: Record<PhaseStatus, { label: string }> = {
  running:   { label: "running now on a worker" },
  queued:    { label: "queued — no worker has claimed it yet" },
  failed:    { label: "the last attempt failed" },
  stopped:   { label: "the last attempt was cancelled" },
  done:      { label: "has produced something here" },
  ready:     { label: "prerequisites met — ready to run" },
  blocked:   { label: "prerequisites not met" },
  waiting:   { label: "prerequisites not met — nothing upstream yet" },
  elsewhere: { label: "not run from this repository" },
  "no-run":  { label: "nothing to run — a committed artefact, or a check inside another phase" },
};

function PhaseRail() {
  const { missionId } = useMission();
  const { subject } = useSubject();
  const { data } = useQuery({
    queryKey: ["phase-summary", missionId, subject],
    queryFn: async () => {
      const url = scoped("/api/phase-summary", missionId);
      const r = await fetch(subject
        ? url + (url.includes("?") ? "&" : "?") + `subject=${encodeURIComponent(subject)}`
        : url);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { phases: PhaseBadge[] };
    },
    refetchInterval: 15_000,
    staleTime: 10_000,
    // Never reuse another mission's rail while this request is in flight.  The
    // static phase names below are the honest placeholder; old progress is not.
  });

  // The names are known at build time; only the states have to be fetched. So
  // the rail draws its ten rows on the first paint and the marks appear when
  // the summary answers, instead of the whole thing arriving at once at the
  // end of a four-request chain.
  const rows: RailRow[] = data?.phases ?? PLACEHOLDER_ROWS;

  return (
    <nav className="rail" aria-label="Pipeline phases">
      <div className="rail-title">Pipeline</div>
      {rows.map((p) => (
        <NavLink key={p.id} to={`/phase/${p.id.toLowerCase()}`}
                 className={`rail-item${p.status ? ` is-${p.status}` : ""}`}
                 title={p.status ? `${STATUS[p.status].label} — ${p.why ?? ""}` : p.name}>
          <span className="rail-id">{p.id}</span>
          <span className="rail-name">{p.name}</span>
          <span className="rail-badges">
            {(p.active_jobs ?? 0) > 0 && (
              <em className={p.running_jobs ? "rail-active" : "rail-queued"}>
                {p.active_jobs}
              </em>
            )}
            {p.badge && <em>{p.badge}</em>}
          </span>
          {/* Its own column, so the mark sits in the same place on every row.
              Inside the badges it moved with the width of the count beside it,
              and a status you have to look for is a status you stop reading.
              Empty until the summary lands: no mark says "not known yet",
              where any mark would be a claim about the phase. */}
          <span className="rail-status">
            {p.status && <Mark status={p.status} />}
          </span>
          <span className="visually-hidden">
            {p.status ? STATUS[p.status].label : "state still loading"}
          </span>
        </NavLink>
      ))}
      <SubjectPicker />
      <MissionFooter />
      <SessionFoot />
    </nav>
  );
}

/**
 * The footer.
 *
 * The source link comes from a setting rather than a constant, because there is
 * no repository URL anywhere in this checkout -- the parent repo's only remote is
 * an internal GitLab -- and a guessed address in the footer of a control panel is
 * worse than no link. Set CX_SOURCE_URL and it appears.
 */
function Footer() {
  const { data } = useQuery({
    queryKey: ["build"],
    queryFn: async () => {
      const r = await fetch("/api/build");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { entry: string | null; source_url?: string };
    },
    staleTime: 5 * 60_000,
  });
  const source = data?.source_url;
  return (
    <footer className="footpiece">
      <span>Helena Exploration Framework</span>
      {source && (
        <a href={source} target="_blank" rel="noreferrer noopener">source on GitHub</a>
      )}
    </footer>
  );
}

export default function App() {
  return (
    <div className="shell">
      <header className="masthead">
        {/* The mark, then the name set in type -- not the kit's horizontal
            lockup, whose drawn wordmark reads "Helena Framework" and is not
            what this platform is called. Both marks ship and CSS picks: the
            reverse one on Obsidian, the colour one on Parchment. */}
        <a className="wordmark" href="/" aria-label="Helena Exploration Framework">
          <img src={markReverse} alt="" className="only-dark" />
          <img src={markColor} alt="" className="only-light" />
          <span className="wordmark-name">
            HELENA<span>·</span>EXPLORATION<span>·</span>FRAMEWORK
          </span>
        </a>
        <div className="clock">
          {new Date().toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}
        </div>
      </header>

      <nav className="stages" aria-label="Sections">
        {GENERAL.map(([to, label]) => (
          <NavLink key={to} to={to} end={to === "/"}>
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Two surfaces sit outside the mission gate. The user guide exists to
          tell somebody what a mission is and why they want one, and
          Configuration is settings, hosts, accounts and the audit log -- none
          of which belong to a mission. Behind the gate, both were menu entries
          that did nothing when clicked. */}
      <Routes>
        <Route path="/documentation" element={
          <main className="workarea">
            <RouteError><Suspense fallback={<div className="empty">loading…</div>}>
              <Documentation />
            </Suspense></RouteError>
          </main>
        } />
        <Route path="/configuration" element={
          <main className="workarea">
            <RouteError><Suspense fallback={<div className="empty">loading…</div>}>
              <Configuration />
            </Suspense></RouteError>
          </main>
        } />
        <Route path="*" element={<Gated />} />
      </Routes>
      {/* Inside the shell, so it sits under every page rather than only the
          ones behind the mission gate. */}
      <Footer />
    </div>
  );
}

function Gated() {
  return (
      <MissionGate>
      <div className="workspace">
        <PhaseRail />
        <main className="workarea">
          <RouteError><Suspense fallback={<div className="empty">loading…</div>}>
            <Routes>
              <Route path="/" element={<Mission />} />
              {/* Reference split into the two errands it was conflating. */}
              <Route path="/reference" element={<Navigate to="/configuration" replace />} />
              {/* Command used to be a top-level surface; queueing now lives inside
                  the phase that does the work, so old links land on Mission. */}
              <Route path="/command" element={<Navigate to="/" replace />} />
              <Route path="/phase/:phaseId" element={<Phase />} />
              <Route path="/run/:runId" element={<RunDetail />} />
              <Route path="/compare" element={<Compare />} />
            </Routes>
          </Suspense></RouteError>
        </main>
      </div>
      </MissionGate>
  );
}
