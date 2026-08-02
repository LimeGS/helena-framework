import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMission } from "../mission";
import { Card, Empty, Pill } from "./Bits";

/**
 * What this phase produced, what it may consume, and which one is in use.
 *
 * An artifact is immutable: its id is the hash of its content, so correcting an
 * earlier phase adds a version rather than replacing one. Both stay. What moves
 * is the *selection* -- which version each phase and scroll uses -- and that is
 * versioned the way configuration is, as a whole map with its own hash, so
 * "what was selected when that run happened" has an answer.
 *
 * The point of all of it is the round trip: go back to an earlier phase, fix
 * something, come forward again, and still be able to say which results were
 * computed from what.
 */

type Artifact = {
  artifact_id: string; phase: string; sample_id: string; kind: string;
  path: string; content_sha256: string; file_count: number; total_bytes: number;
  produced_by: string | null; inputs: string[]; note: string;
  registered_at_utc: string; selected: boolean; exists: boolean;
};
type Selection = {
  version_id: string; index: number; content_sha256: string;
  choices: Record<string, string>; reason: string; at_utc: string; by: string;
  restored_from?: string;
};

const bytes = (n: number) =>
  n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(1)} MB`
  : n > 1e3 ? `${(n / 1e3).toFixed(0)} kB` : `${n} B`;

/**
 * Moved off the phase pages and into Configuration.
 *
 * It was three cards on every phase -- a register, a "what will read" list and
 * a selection history -- next to a Runs table that showed the same objects in
 * different words. Choosing an input is now part of starting a run, which is
 * when the choice is actually made; what is left here is the audit trail and
 * the one-off backfill, which are maintenance and read like it.
 */
export function Artifacts({ phase, sample }: { phase: string; sample: string | null }) {
  const client = useQueryClient();
  const { missionId, current } = useMission();
  const [affecting, setAffecting] = useState<string | null>(null);
  const [reason, setReason] = useState("");

  const enabled = Boolean(missionId) && !current?.implicit;

  const listing = useQuery({
    enabled,
    queryKey: ["artifacts", missionId, phase, sample ?? ""],
    queryFn: async () => {
      const q = new URLSearchParams({ phase });
      if (sample) q.set("sample", sample);
      const r = await fetch(`/api/missions/${missionId}/artifacts?${q}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as {
        artifacts: Artifact[]; selection: Selection | null; consumes_from: string[];
      };
    },
    staleTime: 20_000,
  });

  const upstream = useQuery({
    enabled: enabled && (listing.data?.consumes_from.length ?? 0) > 0,
    queryKey: ["artifacts-upstream", missionId, listing.data?.consumes_from, sample ?? ""],
    queryFn: async () => {
      const found: Artifact[] = [];
      for (const from of listing.data?.consumes_from ?? []) {
        const q = new URLSearchParams({ phase: from });
        if (sample) q.set("sample", sample);
        const r = await fetch(`/api/missions/${missionId}/artifacts?${q}`);
        if (r.ok) found.push(...(await r.json()).artifacts);
      }
      return found;
    },
    staleTime: 20_000,
  });

  const history = useQuery({
    enabled,
    queryKey: ["selection", missionId],
    queryFn: async () => {
      const r = await fetch(`/api/missions/${missionId}/selection`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { current: Selection | null; history: Selection[] };
    },
    staleTime: 20_000,
  });

  const affects = useQuery({
    enabled: enabled && Boolean(affecting),
    queryKey: ["affects", missionId, affecting],
    queryFn: async () => {
      const r = await fetch(
        `/api/missions/${missionId}/artifacts/${encodeURIComponent(affecting!)}/affects`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { affects: Artifact[] };
    },
  });

  const backfill = useMutation({
    mutationFn: async (write: boolean) => {
      const r = await fetch(
        `/api/missions/${missionId}/artifacts/backfill?dry_run=${!write}`,
        { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string"
        ? body.detail : JSON.stringify(body.detail));
      return body as { count: number; dry_run: boolean; caveat: string };
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["artifacts"] }),
  });

  const choose = useMutation({
    mutationFn: async (picked: Artifact) => {
      // The whole map, never a patch: one entry moving on its own would make
      // "what was selected then" unanswerable.
      const choices = { ...(history.data?.current?.choices ?? {}) };
      choices[`${picked.phase}/${picked.sample_id}`] = picked.artifact_id;
      const r = await fetch(`/api/missions/${missionId}/selection`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ choices, reason }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string"
        ? body.detail : JSON.stringify(body.detail));
      return body;
    },
    onSuccess: () => {
      setReason("");
      client.invalidateQueries({ queryKey: ["artifacts"] });
      client.invalidateQueries({ queryKey: ["selection"] });
    },
  });

  const restore = useMutation({
    mutationFn: async (versionId: string) => {
      const r = await fetch(
        `/api/missions/${missionId}/selection/${encodeURIComponent(versionId)}/restore`,
        { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string"
        ? body.detail : JSON.stringify(body.detail));
      return body;
    },
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["artifacts"] });
      client.invalidateQueries({ queryKey: ["selection"] });
    },
  });

  const produced = listing.data?.artifacts ?? [];
  const consumable = upstream.data ?? [];
  const bySample = useMemo(() => {
    const groups = new Map<string, Artifact[]>();
    for (const a of produced) {
      groups.set(a.sample_id, [...(groups.get(a.sample_id) ?? []), a]);
    }
    return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [produced]);

  if (!enabled) {
    return (
      <Card title="Artifacts" collapsed>
        <Empty>
          {current?.implicit
            ? "the unfiled view is assembled from receipts and owns no artifacts"
            : "pick a mission — artifacts and their selection belong to one"}
        </Empty>
      </Card>
    );
  }

  return (
    <>
      <Card
        title={`${phase} artifacts`}
        collapsed
        note={produced.length
          ? `${produced.length} across ${bySample.length} ${bySample.length === 1 ? "scroll" : "scrolls"}`
          : "none yet"}
      >
        <div className="body-pad">
          <p>
            An artifact is identified by its content, so a corrected version does not
            replace the one it corrects — both stay, and what moves is which one is
            selected. That is what makes going back to an earlier phase and forward again
            something you can still read afterwards.
          </p>
        </div>

        <div className="body-pad">
          <div className="controls">
            <button disabled={backfill.isPending}
                    onClick={() => backfill.mutate(false)}>
              {backfill.isPending ? "reading…" : "find runs that predate this register"}
            </button>
            {backfill.data?.dry_run && backfill.data.count > 0 && (
              <button onClick={() => backfill.mutate(true)}>
                register {backfill.data.count}
              </button>
            )}
          </div>
          {backfill.data && (
            <p className="hint">
              {backfill.data.dry_run
                ? `${backfill.data.count} runs on disk are not in the register. `
                : `${backfill.data.count} registered. `}
              {backfill.data.caveat}
            </p>
          )}
          {backfill.isError && <Pill kind="crit">{String(backfill.error)}</Pill>}
        </div>

        {bySample.length === 0 ? (
          <Empty>this phase has registered nothing in this mission</Empty>
        ) : (
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l">Scroll</th>
                  <th className="l grow">Artifact</th>
                  <th className="l">Kind</th>
                  <th>Files</th>
                  <th>Size</th>
                  <th className="l">Registered</th>
                  <th className="l">In use</th>
                  <th className="l"></th>
                </tr>
              </thead>
              <tbody>
                {bySample.flatMap(([scroll, items]) =>
                  items.map((a, index) => (
                    <tr key={a.artifact_id} className={a.selected ? "" : "muted"}>
                      <td className="l scrollid">{index === 0 ? scroll : ""}</td>
                      <td className="l grow" title={`${a.content_sha256}\n${a.path}`}>
                        <code>{a.artifact_id.split(":").pop()}</code>
                        {a.note && <span className="dash"> · {a.note}</span>}
                        {!a.exists && <> <Pill kind="warn">bytes missing</Pill></>}
                      </td>
                      <td className="l">{a.kind}</td>
                      <td>{a.file_count}</td>
                      <td>{bytes(a.total_bytes)}</td>
                      <td className="l">{a.registered_at_utc.slice(0, 16).replace("T", " ")}</td>
                      <td className="l">
                        {a.selected ? <Pill kind="ok">selected</Pill> : <span className="dash">—</span>}
                      </td>
                      <td className="l">
                        <div className="controls">
                          {!a.selected && (
                            <button disabled={choose.isPending} onClick={() => choose.mutate(a)}>
                              use this
                            </button>
                          )}
                          <button onClick={() => setAffecting(
                            affecting === a.artifact_id ? null : a.artifact_id)}>
                            {affecting === a.artifact_id ? "hide" : "affects"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        )}

        {produced.length > 1 && (
          <div className="body-pad">
            <div className="controls">
              <input className="search" value={reason}
                     placeholder="why this version, for the record…"
                     onChange={(e) => setReason(e.target.value)} />
            </div>
          </div>
        )}
        {choose.isError && <p className="hint"><Pill kind="crit">{String(choose.error)}</Pill></p>}

        {affecting && (
          <div className="body-pad">
            <h3>What was computed from <code>{affecting.split(":").pop()}</code></h3>
            {affects.isLoading ? <Empty>tracing…</Empty>
              : affects.data?.affects.length ? (
                <>
                  <p>
                    These are not wrong. They are answers computed with this input, and if
                    you replace it they are answers to a question nobody asked any more.
                  </p>
                  <div className="controls">
                    {affects.data.affects.map((a) => (
                      <Pill key={a.artifact_id} kind="warn">
                        {a.phase} {a.sample_id} · {a.artifact_id.split(":").pop()}
                      </Pill>
                    ))}
                  </div>
                </>
              ) : <Empty>nothing downstream has consumed it</Empty>}
          </div>
        )}
      </Card>

      {consumable.length > 0 && (
        <Card title={`What ${phase} will read`} collapsed
              note={`from ${listing.data?.consumes_from.join(", ")}`}>
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l">Scroll</th>
                  <th className="l grow">Artifact</th>
                  <th className="l">From</th>
                  <th className="l">In use</th>
                </tr>
              </thead>
              <tbody>
                {consumable.map((a) => (
                  <tr key={a.artifact_id} className={a.selected ? "" : "muted"}>
                    <td className="l scrollid">{a.sample_id}</td>
                    <td className="l grow"><code>{a.artifact_id.split(":").pop()}</code>
                      {a.note && <span className="dash"> · {a.note}</span>}</td>
                    <td className="l">{a.phase}</td>
                    <td className="l">
                      {a.selected ? <Pill kind="ok">selected</Pill> : <span className="dash">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(history.data?.history.length ?? 0) > 0 && (
        <Card title="Selection history" collapsed
              note={`${history.data!.history.length} versions`}>
          <div className="body-pad">
            <p>
              Append-only. Going back writes a new version equal to the old one rather than
              rewinding, so a mission that went forward, found a mistake and came back reads
              as three decisions instead of one that never happened.
            </p>
          </div>
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l">Version</th>
                  <th className="l grow">Why</th>
                  <th>Entries</th>
                  <th className="l">When</th>
                  <th className="l"></th>
                </tr>
              </thead>
              <tbody>
                {history.data!.history.map((v, index) => (
                  <tr key={v.version_id} className={index === 0 ? "" : "muted"}>
                    <td className="l scrollid">
                      {v.version_id}
                      {index === 0 && <> <Pill kind="ok">current</Pill></>}
                    </td>
                    <td className="l grow wrap">
                      {v.reason || <span className="dash">—</span>}
                      {v.restored_from && <span className="dash"> (from {v.restored_from})</span>}
                    </td>
                    <td>{Object.keys(v.choices).length}</td>
                    <td className="l">{v.at_utc.slice(0, 16).replace("T", " ")}</td>
                    <td className="l">
                      {index !== 0 && (
                        <button disabled={restore.isPending}
                                onClick={() => restore.mutate(v.version_id)}>
                          go back to this
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {restore.isError && <p className="hint"><Pill kind="crit">{String(restore.error)}</Pill></p>}
        </Card>
      )}
    </>
  );
}

/**
 * Only what a phase produced, for the phase's own page.
 *
 * The full component above moved to Configuration because choosing an input
 * belongs to starting a run. This is the other half and it did not move with
 * it: what came out. P0 has an Apply button, every Apply mints a new version,
 * and the page that mints them showed no sign that any existed -- so the one
 * place you would look to check that your edit actually produced something was
 * the one place that could not tell you.
 *
 * Read-only by construction. Selecting a version stays where the selection is
 * made; this answers "did that Apply produce anything, and which one is live".
 */
export function ProducedArtifacts({ phase, sample }: { phase: string; sample: string | null }) {
  const client = useQueryClient();
  const { missionId, current } = useMission();
  const enabled = Boolean(missionId) && !current?.implicit;
  const selected = current?.scrolls?.length ?? 0;

  // Amending needs a scroll to add, so a selection made before this register
  // existed had no way to record itself -- and the table said "nothing yet"
  // about a phase that had decided something. Idempotent by content: pressing
  // it twice returns the same artifacts, because a version means the decision
  // changed.
  const freeze = useMutation({
    mutationFn: async () => {
      const r = await fetch(`/api/missions/${missionId}/artifacts/freeze-p0`,
                            { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["produced"] }),
  });

  const produced = useQuery({
    queryKey: ["produced", missionId, phase, sample ?? ""],
    enabled,
    queryFn: async () => {
      const q = new URLSearchParams({ phase });
      if (sample) q.set("sample", sample);
      const r = await fetch(`/api/missions/${missionId}/artifacts?${q}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { artifacts: Artifact[] };
    },
    staleTime: 15_000,
  });

  if (!enabled) return null;
  const rows = produced.data?.artifacts ?? [];

  return (
    <Card title={`What ${phase} produced`}
          note={rows.length ? `${rows.length} versions` : undefined}>
      {rows.length === 0 ? (
        <div className="body-pad">
          <p>
            {selected > 0
              ? `${selected} scroll${selected === 1 ? "" : "s"} selected, and `
                + "nothing recorded for them yet. This selection predates the "
                + "register, or was made without one."
              : "Nothing to record: no scroll is selected, so this phase has not "
                + "decided anything."}
          </p>
          {selected > 0 && (
            <div className="controls">
              <button disabled={freeze.isPending} onClick={() => freeze.mutate()}>
                {freeze.isPending ? "freezing…" : "Record what P0 decided"}
              </button>
              {freeze.isError && <Pill kind="crit">{String(freeze.error)}</Pill>}
            </div>
          )}
        </div>
      ) : (
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l">Registered</th>
                <th className="l grow">Artifact</th>
                <th className="l">Scroll</th>
                <th className="l">Kind</th>
                <th>Files</th>
                <th>Size</th>
                <th className="l">In use</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.artifact_id} className={a.exists ? "" : "muted"}>
                  <td className="l">
                    {a.registered_at_utc?.slice(0, 16).replace("T", " ")
                      ?? <span className="dash">—</span>}
                  </td>
                  <td className="l grow" title={a.path}>
                    <code>{a.artifact_id.slice(0, 16)}</code>
                    {!a.exists && <span className="dash"> · file is gone</span>}
                  </td>
                  <td className="l">{a.sample_id || <span className="dash">—</span>}</td>
                  <td className="l">{a.kind}</td>
                  <td>{a.file_count}</td>
                  <td>{bytes(a.total_bytes)}</td>
                  <td className="l">
                    {a.selected ? <Pill kind="ok">selected</Pill>
                                : <Pill kind="none">superseded</Pill>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
