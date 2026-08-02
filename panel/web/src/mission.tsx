import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pill } from "./components/Bits";
import { SessionFoot } from "./Login";

/**
 * Which mission the interface is looking at.
 *
 * The selection lives in the browser, not on the server: two people can watch
 * two different missions at once, and nothing about the framework's state
 * depends on who is looking. Every scoped query carries the id, so a view that
 * forgets to pass it shows everything rather than the wrong thing.
 */

export type Mission = {
  mission_id: string;
  name: string;
  description: string;
  state: "active" | "paused" | "archived";
  scrolls: string[];
  scrolls_frozen_at_utc: string | null;
  created_at_utc: string | null;
  amendments: { added?: string[]; removed?: string[]; reason: string; by: string; at_utc: string }[];
  non_claims: string[];
  path: string;
  run_count: number;
  /** True once the mission has produced a run: from then on edits are amendments. */
  selection_frozen?: boolean;
  job_count?: number;
  implicit?: boolean;
};

const KEY = "campaignx.mission";
const Ctx = createContext<{
  missionId: string | null;
  setMissionId: (id: string | null) => void;
  missions: Mission[];
  current: Mission | null;
}>({ missionId: null, setMissionId: () => {}, missions: [], current: null });

/** The mission in the address bar wins over the one this browser remembers, so
 *  a link to a phase page carries its scope with it. Without this a shared URL
 *  lands the reader in whichever mission they last opened, which is a different
 *  page with the same address. */
function initialMission(): string | null {
  const asked = new URLSearchParams(window.location.search).get("mission");
  if (asked) {
    localStorage.setItem(KEY, asked);
    return asked;
  }
  return localStorage.getItem(KEY);
}

export function MissionProvider({ children }: { children: React.ReactNode }) {
  const [missionId, setId] = useState<string | null>(initialMission);
  const { data } = useQuery({
    queryKey: ["missions"],
    queryFn: async () => {
      const r = await fetch("/api/missions");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { missions: Mission[]; runs_root: string };
    },
    staleTime: 30_000,
  });

  const setMissionId = useCallback((id: string | null) => {
    if (id) localStorage.setItem(KEY, id);
    else localStorage.removeItem(KEY);
    // The query parameter is authoritative on reload and in shared links, so
    // changing only React state/localStorage leaves a time bomb in the address
    // bar: refresh reopens the previous mission. Keep all three in lockstep.
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("mission", id);
    else url.searchParams.delete("mission");
    window.history.replaceState(window.history.state, "", url);
    setId(id);
  }, []);

  const missions = data?.missions ?? [];
  const current = missions.find((m) => m.mission_id === missionId) ?? null;
  const value = useMemo(
    () => ({ missionId: current ? missionId : null, setMissionId, missions, current }),
    [missionId, current, missions, setMissionId],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useMission = () => useContext(Ctx);

/**
 * The subject is what crosses phases.
 *
 * A run is the output of one phase and does not exist in the other nine, so
 * selecting runs would leave most of the rail empty. A scroll -- and within it
 * a segment -- is born in P1, certified in P2, rendered in P4, screened in P5
 * and adjudicated in P7. Following that is what an operator is actually doing.
 */
const SubjectCtx = createContext<{
  subject: string | null;
  setSubject: (s: string | null) => void;
}>({ subject: null, setSubject: () => {} });

export function SubjectProvider({ children }: { children: React.ReactNode }) {
  const { missionId } = useMission();
  // A phase runs on one scroll. There is no "all of them" execution, so
  // offering it as a subject would name a thing you cannot act on. The
  // mission-wide view is the Mission tab; the rail is one subject at a time.
  const [selection, setSelection] = useState<{
    missionId: string | null; subject: string | null;
  }>({ missionId, subject: null });
  // Mission changes are atomic at the context boundary.  The previous scroll
  // is never exposed under the new mission, even for the render before the new
  // /api/subjects response arrives.
  const subject = selection.missionId === missionId ? selection.subject : null;
  const setSubject = useCallback((next: string | null) => {
    setSelection({ missionId, subject: next });
  }, [missionId]);
  const value = useMemo(() => ({ subject, setSubject }), [subject, setSubject]);
  return <SubjectCtx.Provider value={value}>{children}</SubjectCtx.Provider>;
}

export const useSubject = () => useContext(SubjectCtx);

export type Subject = {
  sample_id: string; pixel_um: string; energy_kev: number | null;
  surfaces: number; certified_surfaces: number; surface_area_cm2: number;
  runs: number; maps: number; reached_phase: string; segments: string[];
};

export function SubjectPicker() {
  const { missionId } = useMission();
  const { subject, setSubject } = useSubject();
  const { data } = useQuery({
    queryKey: ["subjects", missionId],
    queryFn: async () => {
      const r = await fetch(scoped("/api/subjects", missionId));
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { subjects: Subject[] };
    },
    staleTime: 20_000,
  });
  const subjects = data?.subjects ?? [];
  const current = subjects.find((s) => s.sample_id === subject);

  // Select the first subject as soon as one exists, so the rail is never
  // showing phase state for nothing in particular.
  useEffect(() => {
    if (!subjects.length && subject) setSubject(null);
    if (!subject && subjects.length) setSubject(subjects[0].sample_id);
    if (subject && subjects.length && !current) setSubject(subjects[0].sample_id);
  }, [subject, subjects, current, setSubject]);

  if (!subjects.length) {
    return (
      <div className="subjectpick">
        <div className="subjectmeta">no scroll in this mission has anything yet</div>
      </div>
    );
  }

  return (
    <div className="subjectpick">
      <select
        value={subject ?? subjects[0].sample_id}
        onChange={(e) => setSubject(e.target.value)}
        aria-label="Scroll"
      >
        {subjects.map((s) => (
          <option key={s.sample_id} value={s.sample_id}>
            {s.sample_id} — reached {s.reached_phase}
          </option>
        ))}
      </select>
      {current && (
        <span className="subjectmeta">
          {current.surfaces} surf · {current.runs} runs · {current.maps} maps
        </span>
      )}
    </div>
  );
}

/** Append the mission to a URL, or leave it alone when nothing is selected. */
export function scoped(url: string, missionId: string | null): string {
  if (!missionId) return url;
  return url + (url.includes("?") ? "&" : "?") + `mission=${encodeURIComponent(missionId)}`;
}

export function MissionFooter() {
  const { current, setMissionId } = useMission();
  if (!current) return null;
  return (
    <div className="missionfoot">
      <div className="missionfoot-label">Mission</div>
      <div className="missionfoot-name" title={current.description || current.name}>
        {current.name}
      </div>
      <div className="missionfoot-meta">
        {current.scrolls.length} scroll{current.scrolls.length === 1 ? "" : "s"} ·{" "}
        {current.run_count} run{current.run_count === 1 ? "" : "s"}
      </div>
      {/* Changing mission goes back to the table rather than opening a list
          here: there can be hundreds, and a control that fits three is a
          control that breaks at thirty. */}
      <button className="missionfoot-change" onClick={() => setMissionId(null)}>
        Change or create
      </button>
    </div>
  );
}

/**
 * Nothing renders until a mission is chosen.
 *
 * Every view below is scoped to one, so with none selected the interface would
 * either show everything -- merging scopes that exist to stay apart -- or show
 * nothing and not say why. This says why, and offers the two ways out.
 */
export function MissionGate({ children }: { children: React.ReactNode }) {
  const { current, missions, setMissionId } = useMission();
  const client = useQueryClient();
  const [creating, setCreating] = useState(false);
  const [filter, setFilter] = useState("");
  const [form, setForm] = useState({ mission_id: "", name: "" });

  const create = useMutation({
    mutationFn: async () => {
      const suggested = form.name
        .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64);
      const r = await fetch("/api/missions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Scrolls are chosen in P0, not here. A mission starts as a name.
        body: JSON.stringify({
          mission_id: form.mission_id || suggested,
          name: form.name,
          scrolls: [],
        }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body as { mission_id: string };
    },
    onSuccess: (created) => {
      client.invalidateQueries({ queryKey: ["missions"] });
      setMissionId(created.mission_id);
    },
  });

  if (current) return <>{children}</>;

  const suggested = form.name
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64);
  const ready = form.name.trim();
  const needle = filter.trim().toLowerCase();
  const shown = needle
    ? missions.filter((m) =>
        m.mission_id.toLowerCase().includes(needle) ||
        m.name.toLowerCase().includes(needle) ||
        m.scrolls.some((x) => x.toLowerCase().includes(needle)))
    : missions;

  return (
    <div className="missiongate">
      <section className="card">
        <div className="card-head">
          <h2>Missions</h2>
          <span className="note">
            {missions.length
              ? `${shown.length} of ${missions.length} · pick a row to open it`
              : "none yet"}
          </span>
          <button className="headaction" onClick={() => setCreating((v) => !v)}>
            {creating ? "Cancel" : "New mission"}
          </button>
        </div>

        {creating && (
          <div className="body-pad newmission">
            <div className="formgrid">
              <label>
                Name
                <input value={form.name} placeholder="First Letters · PHerc0826"
                       onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </label>
              <label>
                Id
                <input value={form.mission_id} placeholder={suggested || "first-letters-826"}
                       onChange={(e) => setForm({ ...form, mission_id: e.target.value })} />
              </label>
            </div>
            <div className="controls">
              <button disabled={!ready || create.isPending} onClick={() => create.mutate()}>
                {create.isPending ? "creating…" : "Create"}
              </button>
              {create.isError && <Pill kind="crit">{String(create.error)}</Pill>}
              <span className="hint">
                Scrolls are chosen in P0. Every change to that selection is recorded with a reason.
              </span>
            </div>
          </div>
        )}

        {missions.length > 4 && (
          <div className="body-pad">
            <input
              className="search" type="search" value={filter}
              placeholder="filter by name, id or scroll…"
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
        )}

        {missions.length > 0 ? (
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l grow">Mission</th>
                  <th className="l">Id</th>
                  <th>Scrolls</th>
                  <th>Runs</th>
                  <th>Jobs</th>
                  <th className="l">Created</th>
                  <th className="l">State</th>
                  <th className="l"></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((m) => (
                  <tr key={m.mission_id} className="clickable"
                      onClick={() => setMissionId(m.mission_id)}>
                    <td className="l grow">
                      <span className="scrollid">{m.name}</span>
                      {m.description && (
                        <div className="dash truncate" title={m.description}>
                          {m.description}
                        </div>
                      )}
                    </td>
                    <td className="l"><code>{m.mission_id}</code></td>
                    <td title={m.scrolls.join(", ")}>{m.scrolls.length}</td>
                    <td>{m.run_count}</td>
                    <td>{m.job_count ?? 0}</td>
                    <td className="l">
                      {m.created_at_utc
                        ? m.created_at_utc.slice(0, 16).replace("T", " ")
                        : <span className="dash">—</span>}
                    </td>
                    <td className="l">
                      {m.implicit
                        ? <Pill kind="neg">pre-existing</Pill>
                        : m.state === "active"
                          ? <Pill kind="ok">active</Pill>
                          : <Pill kind="warn">{m.state}</Pill>}
                    </td>
                    <td className="l">
                      <button onClick={(e) => { e.stopPropagation(); setMissionId(m.mission_id); }}>
                        open
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          !creating && (
            <div className="empty">
              No mission yet. Use <b>New mission</b> to name the scrolls you are attempting.
            </div>
          )
        )}
      </section>
      {/* The way out. Signing out lived only in the rail, and the rail is not
          drawn until a mission is open -- so the one screen you reach before
          choosing anything was the one screen you could not leave. */}
      <SessionFoot />
    </div>
  );
}

/**
 * The subject is what crosses phases.
 *
 * A run is the output of one phase and does not exist in the other nine, so
 * selecting runs would leave most of the rail empty. A scroll -- and within it
 * a segment -- is born in P1, certified in P2, rendered in P4, screened in P5
 * and adjudicated in P7. Following that is what an operator is actually doing.
 */
