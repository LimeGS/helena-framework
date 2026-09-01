import { useMemo, useState } from "react";
import { Card, Empty, Info, Num, Pill, queryGate } from "../components/Bits";
import { useResizableColumns } from "../components/Table";
import { useInkRun, useInkRuns, type InkRun, type InkRunDetail } from "../api";

/**
 * What P5 produced, without a shell on the GPU host.
 *
 * `/api/runs` -- the Runs tab beside this one -- indexes the legacy CX_RUNS
 * receipt tree. A screening queued through the fleet keeps its verdict in the
 * job row and writes its map into whichever directory the worker named, so that
 * index has never been able to see one. The map was reachable by ssh and
 * `np.load`, and that is how every one of them has been looked at so far.
 *
 * The array does not come here. The picture is rendered on the server, on a
 * percentile stretch, and the stretch is printed under it -- because a viewer
 * that silently rescales is exactly how a map carrying no decision comes to
 * look like one that carries a decision, and the reader has no way to tell.
 */

const STATE_KIND: Record<string, "ok" | "run" | "crit" | "warn" | "neg"> = {
  succeeded: "ok", running: "run", leased: "run",
  failed: "crit", cancelled: "neg", pending: "warn",
};

// ALIVE is not a finding about ink. It says the distribution has a shape, so a
// screen on it can mean something; DEGENERATE says the map carries no decision
// at all. Those are the two colours, and neither of them is about writing.
const VERDICT_KIND: Record<string, "ok" | "crit" | "warn"> = {
  ALIVE: "ok", DEGENERATE: "crit", EMPTY: "warn",
};

type SortKey = "ran" | "sample_id" | "surface_id" | "profile_id" | "state"
             | "verdict" | "p50" | "p99" | "spread";

const metric = (run: InkRun, name: string): number | null => {
  const value = run.liveness?.metrics?.[name];
  return typeof value === "number" ? value : null;
};

/** When this run last did something: it finished, or it is still waiting. */
const ranAt = (run: InkRun): string | null =>
  ["succeeded", "failed", "cancelled"].includes(run.state)
    ? (run.updated_at ?? run.created_at)
    : run.created_at;

const sortValue = (run: InkRun, key: SortKey): string | number | null => {
  switch (key) {
    case "ran": return ranAt(run);
    case "verdict": return run.liveness?.verdict ?? null;
    case "p50": return metric(run, "p50");
    case "p99": return metric(run, "p99");
    case "spread": return metric(run, "spread_p99_p50");
    default: return run[key] ?? null;
  }
};

/**
 * A sortable header.
 *
 * There was no sortable table in this panel to copy, so this is the smallest
 * thing that works: a button in the cell, the direction stated in
 * `aria-sort` on the header itself rather than only in an arrow.
 */
function SortHead({ column, label, sort, onSort, className }: {
  column: SortKey; label: string; className?: string;
  sort: { key: SortKey; descending: boolean };
  onSort: (key: SortKey) => void;
}) {
  const active = sort.key === column;
  return (
    <th className={className}
        aria-sort={active ? (sort.descending ? "descending" : "ascending") : "none"}>
      <button type="button" className="colsort" onClick={() => onSort(column)}>
        {label}
        <span aria-hidden="true" className={active ? "colsort-on" : "colsort-off"}>
          {active ? (sort.descending ? "▾" : "▴") : "▪"}
        </span>
      </button>
    </th>
  );
}

export default function InkMaps({ sample, mission }:
                                { sample?: string; mission?: string }) {
  const query = useInkRuns(sample, mission);
  const [text, setText] = useState("");
  const [state, setState] = useState("all");
  const [verdict, setVerdict] = useState("all");
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>(
    { key: "ran", descending: true });
  const [selected, setSelected] = useState<string | null>(null);

  const runs = query.data?.runs ?? [];
  const rows = useMemo(() => {
    const needle = text.trim().toLowerCase();
    const kept = runs.filter((run) =>
      (state === "all" || run.state === state)
      && (verdict === "all" || (run.liveness?.verdict ?? "none") === verdict)
      && (!needle || [run.job_id, run.sample_id, run.surface_id, run.profile_id]
          .some((field) => String(field ?? "").toLowerCase().includes(needle))));
    return [...kept].sort((a, b) => {
      const left = sortValue(a, sort.key);
      const right = sortValue(b, sort.key);
      // Absent sorts last in both directions: a row with no verdict is not the
      // smallest verdict, and burying it under the ones that have one is how a
      // missing measurement stops being visible.
      if (left === null && right === null) return 0;
      if (left === null) return 1;
      if (right === null) return -1;
      const order = typeof left === "number" && typeof right === "number"
        ? left - right : String(left).localeCompare(String(right));
      return sort.descending ? -order : order;
    });
  }, [runs, text, state, verdict, sort]);

  const onSort = (key: SortKey) =>
    setSort((current) => current.key === key
      ? { key, descending: !current.descending }
      : { key, descending: key === "ran" });

  const gate = queryGate(query, "reading the screenings…");
  if (gate) return gate;
  const data = query.data!;
  if (!data.available) {
    return <Empty>{data.reason ?? "the fleet queue could not be read"}</Empty>;
  }

  const verdicts = new Set(runs.map((run) => run.liveness?.verdict).filter(Boolean));
  const readable = runs.filter((run) => run.maps.length).length;

  return (
    <>
      <div className="strip">
        <div className="tile steady">
          <h2>screenings</h2>
          <div className="readout">{runs.length}</div>
          <p>queued through the fleet</p>
        </div>
        <div className={runs.some((r) => r.liveness?.verdict === "ALIVE")
                        ? "tile steady" : "tile warn"}>
          <h2>alive</h2>
          <div className="readout">
            {runs.filter((r) => r.liveness?.verdict === "ALIVE").length}
          </div>
          <p>the map carries a decision — not a finding about ink</p>
        </div>
        <div className={runs.some((r) => r.liveness
                        && r.liveness.verdict !== "ALIVE") ? "tile warn" : "tile steady"}>
          <h2>no decision</h2>
          <div className="readout">
            {runs.filter((r) => r.liveness && r.liveness.verdict !== "ALIVE").length}
          </div>
          <p>degenerate or empty — nothing downstream may screen these</p>
        </div>
        <div className={readable ? "tile steady" : "tile warn"}>
          <h2>readable here</h2>
          <div className="readout">{readable}</div>
          <p>
            {readable === runs.length
              ? "every map is mounted on this host"
              : "the rest ran on a worker whose volume is not mounted here"}
          </p>
        </div>
      </div>

      <Card
        title={<>Maps <Info label="Where these runs come from and what a map is not"
                            title="P5 output">
          These are the P5 jobs in the fleet queue, which is where a screening
          records what it found. The Runs tab beside this one indexes the legacy
          receipt tree on disk and cannot see them. A probability map is not a
          reading: it accepts no ink, text or letters, and a DEGENERATE verdict
          says the map carries no decision rather than saying anything about ink.
        </Info></>}
        note={`${rows.length} of ${runs.length} shown`}
      >
        <div className="body-pad">
          <div className="controls">
            <input className="search" type="search" value={text}
                   aria-label="Filter screenings"
                   placeholder="filter by job, scroll, surface or lane…"
                   onChange={(event) => setText(event.target.value)} />
            <select value={state} aria-label="Filter by job state"
                    onChange={(event) => setState(event.target.value)}>
              <option value="all">any state</option>
              {[...new Set(runs.map((run) => run.state))].sort().map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
            <select value={verdict} aria-label="Filter by liveness verdict"
                    onChange={(event) => setVerdict(event.target.value)}>
              <option value="all">any verdict</option>
              {[...verdicts].sort().map((value) => (
                <option key={value} value={value!}>{value!.toLowerCase()}</option>
              ))}
              <option value="none">no verdict</option>
            </select>
          </div>
        </div>
        {rows.length === 0 ? (
          <Empty>
            {runs.length === 0
              ? "no P5 job has run in this scope yet"
              : "no screening matches these filters"}
          </Empty>
        ) : (
          <RunTable rows={rows} sort={sort} onSort={onSort}
                    selected={selected} onSelect={setSelected} />
        )}
      </Card>

      {selected && (
        <RunInspector jobId={selected} onClose={() => setSelected(null)} />
      )}
    </>
  );
}

function RunTable({ rows, sort, onSort, selected, onSelect }: {
  rows: InkRun[];
  sort: { key: SortKey; descending: boolean };
  onSort: (key: SortKey) => void;
  selected: string | null;
  onSelect: (jobId: string | null) => void;
}) {
  const tableRef = useResizableColumns<HTMLTableElement>();
  return (
    <div className="scroller">
      <table ref={tableRef}>
        <thead>
          <tr>
            <th className="l grow">Job</th>
            <SortHead className="l" column="ran" label="Ran" sort={sort} onSort={onSort} />
            <SortHead className="l" column="sample_id" label="Scroll" sort={sort} onSort={onSort} />
            <SortHead className="l" column="surface_id" label="Surface" sort={sort} onSort={onSort} />
            <SortHead className="l" column="profile_id" label="Lane" sort={sort} onSort={onSort} />
            <SortHead className="l" column="state" label="State" sort={sort} onSort={onSort} />
            <SortHead className="l" column="verdict" label="Liveness" sort={sort} onSort={onSort} />
            <SortHead column="p50" label="p50" sort={sort} onSort={onSort} />
            <SortHead column="p99" label="p99" sort={sort} onSort={onSort} />
            <SortHead column="spread" label="Spread" sort={sort} onSort={onSort} />
            <th className="l">Map</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((run) => {
            const at = ranAt(run);
            return (
              <tr key={run.job_id}
                  /* Not `.diff`: that row style is the red one, and a row you
                     chose to look at must not look like a row that failed. */
                  className={selected === run.job_id ? "is-open" : undefined}>
                <td className="l grow">
                  <button type="button" className="linky"
                          aria-expanded={selected === run.job_id}
                          onClick={() => onSelect(selected === run.job_id
                                                  ? null : run.job_id)}>
                    {run.job_id}
                  </button>
                </td>
                <td className="l" title={`queued ${run.created_at ?? "?"}`
                                         + ` · last change ${run.updated_at ?? "?"}`}>
                  {at ? at.slice(0, 16).replace("T", " ") : <span className="dash">—</span>}
                </td>
                <td className="l">{run.sample_id}</td>
                <td className="l" title={run.surface_id ?? undefined}>
                  {run.surface_id
                    ? <code>{run.surface_id.split(":").pop()?.slice(0, 12)}</code>
                    : <span className="dash">none named</span>}
                </td>
                <td className="l wrap">
                  {run.profile_id ?? <span className="dash">—</span>}
                </td>
                <td className="l">
                  <Pill kind={STATE_KIND[run.state] ?? "neg"}>{run.state}</Pill>
                </td>
                <td className="l" title={run.liveness?.reason || undefined}>
                  {run.liveness
                    ? <Pill kind={VERDICT_KIND[run.liveness.verdict] ?? "warn"}>
                        {run.liveness.verdict}
                      </Pill>
                    /* Not a dash: a P5 job with no verdict is refused by the
                       worker, so this cell means the run never got that far. */
                    : <span className="dash">not assessed</span>}
                </td>
                <td><Num v={metric(run, "p50")} digits={3} /></td>
                <td><Num v={metric(run, "p99")} digits={3} /></td>
                <td><Num v={metric(run, "spread_p99_p50")} digits={3} /></td>
                <td className="l">
                  {run.maps.length
                    ? <code>{run.maps[0]}</code>
                    : run.published
                      ? <span className="dash" title={run.published.artifact_uri ?? ""}>
                          published, not on this host
                        </span>
                      : <span className="dash">none</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** One key/value line of the provenance chain, or nothing when it is absent. */
function Link({ label, value, kind, children }: {
  label: string; value?: string | null;
  kind?: "" | "bad" | "unknown"; children?: React.ReactNode;
}) {
  return (
    <div className={`link ${kind ?? ""}`}>
      <span className="k">{label}</span>
      <span className="v">{value}</span>
      {children}
    </div>
  );
}

function RunInspector({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const [name, setName] = useState<string | null>(null);
  const query = useInkRun(jobId, name);
  const gate = queryGate(query, "reading the receipt…");
  if (gate) return <Card title={jobId}>{gate}</Card>;
  const run = query.data as InkRunDetail;
  const chosen = run.selected_map;

  return (
    <div className="split wide">
      <Card
        title={run.job_id}
        note={
          <span className="controls">
            {run.maps.length > 1 && (
              <select value={chosen ?? ""} aria-label="Which array to draw"
                      onChange={(event) => setName(event.target.value)}>
                {run.maps.map((each) => (
                  <option key={each} value={each}>{each}</option>
                ))}
              </select>
            )}
            <button type="button" className="linky" onClick={onClose}>close</button>
          </span>
        }
      >
        {chosen && run.display ? (
          <>
            <div className="inkmap">
              <img alt={`Probability map ${chosen} of ${run.job_id}`}
                   src={`/api/ink/maps/${encodeURIComponent(run.job_id)}/render.png`
                        + `?map=${encodeURIComponent(chosen)}&size=1400`} />
            </div>
            {/* The stretch, stated. Without this the picture is an unlabelled
                rescale, and the difference between a map with signal and one
                without is invisible in it. */}
            <p className="hint">
              {run.display.note} {run.display.width}×{run.display.height} px,{" "}
              {run.display.valid_pixels.toLocaleString()} valid
              {run.display.invalid_pixels > 0 && (
                <> · {run.display.invalid_pixels.toLocaleString()} pixels no tile
                  covered, drawn transparent rather than as a low probability</>
              )}
            </p>
          </>
        ) : (
          <Empty>
            {run.display_error
              ?? (run.published
                  ? "the bytes are published but not mounted on this host: "
                    + run.published.artifact_uri
                  : "this run left no readable array on this host")}
          </Empty>
        )}
      </Card>

      <div className="inspector">
        {run.liveness ? (
          <Card title="Liveness">
            <div className="body-pad">
              <Pill kind={VERDICT_KIND[run.liveness.verdict] ?? "warn"}>
                {run.liveness.verdict}
              </Pill>
              {/* ALIVE writes an empty reason, which is not a missing one. */}
              {run.liveness.reason
                ? <p>{run.liveness.reason}</p>
                : <p className="dash">
                    no failing check — the three thresholds were all met
                  </p>}
              {run.liveness.interpretation && <p>{run.liveness.interpretation}</p>}
            </div>
            {run.liveness.metrics && Object.keys(run.liveness.metrics).length > 0 && (
              <Numbers values={run.liveness.metrics} />
            )}
          </Card>
        ) : (
          <Card title="Liveness">
            <Empty>
              this run recorded no verdict. A P5 job that succeeds without one is
              failed by the worker, because an unchecked map and a live one must
              not look the same.
            </Empty>
          </Card>
        )}

        <Card title="Statistics"
              note={run.map_shape_yx
                ? `${run.map_shape_yx[1]}×${run.map_shape_yx[0]}` : undefined}>
          {run.statistics && Object.keys(run.statistics).length > 0 ? (
            <Numbers values={run.statistics} />
          ) : (
            /* Not zeros. The TimeSformer receipt has no statistics block at
               all, and a fabricated zero here reads exactly like a measured
               one. The liveness metrics above are the block every lane writes. */
            <Empty>this lane's receipt carries no statistics block</Empty>
          )}
        </Card>

        <Card title="Provenance">
          <div className="chain">
            <Link label="Scroll" value={run.sample_id} />
            {run.surface_id && <Link label="Surface" value={run.surface_id} />}
            <Link label="Lane" value={run.profile_id ?? "not declared"}
                  kind={run.profile ? "" : "unknown"}>
              {!run.profile && <Pill>no profile in the repo</Pill>}
            </Link>
            <Link label="Checkpoint"
                  kind={run.checkpoint_sha256 ? "" : "unknown"}
                  value={run.checkpoint_sha256 ?? "not declared"}>
              {run.checkpoint_sha256 && run.profile
               && run.profile.checkpoint_sha256 !== undefined && (
                run.profile.checkpoint_sha256 === run.checkpoint_sha256
                  ? <Pill kind="ok">matches the profile</Pill>
                  : <Pill kind="crit">does not match the profile</Pill>
              )}
            </Link>
            {run.input && <Link label={run.input.kind.replaceAll("_", " ")}
                                value={run.input.value} />}
          </div>
        </Card>

        <Card title="Input lineage">
          {run.lineage?.p4_job_id || run.rendered_from ? (
            <div className="chain">
              {run.lineage?.p4_job_id && (
                <Link label="P4 job" value={String(run.lineage.p4_job_id)} />
              )}
              {run.lineage?.p4_layer_artifact_sha256 && (
                <Link label="Layer stack content"
                      value={String(run.lineage.p4_layer_artifact_sha256)} />
              )}
              {run.lineage?.p4_layer_manifest_sha256 && (
                <Link label="Layer stack manifest"
                      value={String(run.lineage.p4_layer_manifest_sha256)} />
              )}
              {run.rendered_from?.rendered_by && !run.lineage?.p4_job_id && (
                <Link label="Layer stack"
                      value={String(run.rendered_from.rendered_by)} />
              )}
              {run.rendered_from?.layer_stack_artifact_sha256
               && !run.lineage?.p4_layer_artifact_sha256 && (
                <Link label="Layer stack content"
                      value={String(run.rendered_from.layer_stack_artifact_sha256)} />
              )}
              {run.rendered_from?.path && (
                <Link label="Supplied by hand" value={String(run.rendered_from.path)}
                      kind="unknown" />
              )}
            </div>
          ) : (
            /* A run entered from a public stack carries the adapter's own
               normalization block, which names no upstream job. Saying so is
               not the same as printing a blank where a job id would go. */
            <Empty>
              nothing upstream of this run is recorded by job id. It was not
              chained to a P4 render of this control plane.
            </Empty>
          )}
        </Card>

        {run.published && (
          <Card title="Published">
            <div className="chain">
              <Link label="Artifact set"
                    value={run.published.artifact_uri ?? "not published"} />
              {run.published.artifact_sha256 && (
                <Link label="Content" value={run.published.artifact_sha256} />
              )}
              {run.published.manifest_sha256 && (
                <Link label="Manifest" value={run.published.manifest_sha256} />
              )}
            </div>
          </Card>
        )}

        <Card title="Receipt">
          <div className="body-pad">
            {run.receipt_path
              ? <p><code>{run.receipt_path}</code></p>
              : <p className="dash">{run.receipt_unavailable ?? "not found"}</p>}
            {run.output_dir && (
              <p className="dash">wrote to <code>{run.output_dir}</code></p>
            )}
            {(run.error || run.refused) && (
              <p><Pill kind="crit">{run.refused ? "refused" : "error"}</Pill>{" "}
                {run.refused ?? run.error}</p>
            )}
            {run.receipt && (
              <a href={`/api/ink/maps/${encodeURIComponent(run.job_id)}`}>
                view the full JSON
              </a>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

/** A block of measured numbers, exactly the ones that are there. */
function Numbers({ values }: { values: Record<string, unknown> }) {
  return (
    <div className="scroller">
      <table>
        <tbody>
          {Object.entries(values).map(([key, value]) => (
            <tr key={key}>
              <td className="l">{key.replaceAll("_", " ")}</td>
              <td>
                {typeof value === "number"
                  ? Number(value.toPrecision(6))
                  : String(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
