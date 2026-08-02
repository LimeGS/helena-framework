import { useQuery } from "@tanstack/react-query";
import { Card, Empty, Info, Pill } from "../components/Bits";
import { RunPhase } from "../components/RunPhase";

/**
 * P3: unroll a certified surface into a flat sheet.
 *
 * This phase reported "nothing here flattens" until vc_flatten was routed to
 * it, so the page leads with the two populations rather than a list: surfaces
 * P2 certified, and surfaces P3 has unrolled. The gap between them is the
 * backlog, and it is the only number that says whether the phase is keeping up.
 *
 * The area ratio gets a column because it is the one measurement that says the
 * flattening did something reasonable. A lamina is not developable, so the
 * ratio is never 1.0; a sheet that lost a third of its area is not one anyone
 * should measure ink on.
 */

type Flattening = {
  available: boolean; reason?: string;
  certified: number; flattened: number; awaiting: number;
  awaiting_physical_qc: number;
  by_state: Record<string, number>;
  rows: {
    surface_id: string; sample_id: string; profile_id: string; state: string;
    area_ratio: number | null; artifact_uri: string | null; created_at: string | null;
  }[];
  note: string;
};

// Under 0.8 is not a failure the phase can detect, so it is not a verdict --
// it is a number worth looking at twice, and the colour says only that.
const ratioKind = (ratio: number | null): "ok" | "warn" | "crit" =>
  ratio === null ? "warn" : ratio >= 0.8 ? "ok" : "crit";

export default function Flattening({ sample, mission }:
                                   { sample?: string; mission?: string }) {
  const query = useQuery<Flattening>({
    queryKey: ["flattening", sample ?? null, mission ?? null],
    queryFn: async () => {
      const q = new URLSearchParams();
      if (sample) q.set("sample", sample);
      if (mission) q.set("mission", mission);
      const response = await fetch("/api/flattening" + (q.size ? `?${q}` : ""));
      if (!response.ok) throw new Error("the flattening ledger could not be read");
      return response.json();
    },
  });

  if (query.isLoading) return <Empty>loading…</Empty>;
  if (query.isError) return <Empty>{String(query.error)}</Empty>;
  const data = query.data!;
  if (!data.available) return <Empty>{data.reason ?? "no control plane"}</Empty>;

  return (
    <>
      <Card title={<>What P3 has unrolled <Info label="What flattening produces and what the area ratio means"
            title="Flattening">{data.note}</Info></>}>
        <div className="knobgrid">
          <div>
            <div className="big">{data.flattened}</div>
            <div className="dash">flattened sheets</div>
          </div>
          <div>
            <div className="big">{data.certified}</div>
            <div className="dash">certified surfaces, which is what P3 may consume</div>
          </div>
          <div>
            <div className="big">{data.awaiting}</div>
            <div className="dash">
              certified and not yet unrolled — the backlog
            </div>
          </div>
          <div>
            <div className="big">{data.awaiting_physical_qc ?? 0}</div>
            <div className="dash">
              certified and waiting on CT support — P3 consumes a surface the
              scan confirms, and these have no verdict on that axis yet
            </div>
          </div>
        </div>
      </Card>

      <Card title={<>Unroll the certified surfaces that are waiting <Info
              label="Why only certified surfaces" title="Eligibility">
              Only certified surfaces are eligible: flattening smooths out the seam
              that would have shown a lamina crossing.
            </Info></>}>
        <RunPhase endpoint="/api/flattening/run" label="Flatten"
                  invalidate="flattening" sample={sample} mission={mission}
                  disabled={Boolean(mission && !sample)}
                  disabledReason="Select a scroll in P0 before flattening."
                  override={{
                    name: "allow_unvalidated",
                    label: "include surfaces the CT never confirmed",
                    note: "The default takes only surfaces the scan supports. "
                        + "This admits the ones whose CT support was never "
                        + "measured, which is a comparison against what the "
                        + "old gate allowed rather than a routine run.",
                  }} />
      </Card>

      <Card title="Sheets">
        {data.rows.length === 0 ? (
          <Empty>
            nothing flattened yet. P3 takes certified surfaces only, so an empty
            page here with certified surfaces above means the phase has not run
          </Empty>
        ) : (
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th>surface</th><th>scroll</th><th>profile</th>
                  <th>state</th><th>area kept</th><th>published</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr key={`${row.surface_id}:${row.profile_id}`}>
                    <td className="mono">{row.surface_id.slice(-12)}</td>
                    <td>{row.sample_id}</td>
                    <td className="mono">{row.profile_id}</td>
                    <td>
                      <Pill kind={row.state === "FLATTENED" ? "ok" : "crit"}>
                        {row.state}
                      </Pill>
                    </td>
                    <td>
                      {row.area_ratio === null ? (
                        <span className="dash">—</span>
                      ) : (
                        <Pill kind={ratioKind(row.area_ratio)}>
                          {(row.area_ratio * 100).toFixed(1)}%
                        </Pill>
                      )}
                    </td>
                    <td className="mono">{row.artifact_uri ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
