import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
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
};

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
  if (!data.grids?.length) return <Empty>no cells have been attempted here yet</Empty>;

  const attempted = data.grids.reduce((total, g) => total + g.cells_attempted, 0);
  const withSurface = data.grids.reduce((total, g) => total + g.cells_with_surface, 0);

  return (
    <>
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
