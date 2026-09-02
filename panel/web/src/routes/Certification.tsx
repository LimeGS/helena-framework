import { useQuery } from "@tanstack/react-query";
import { Card, Empty, Info, Pill } from "../components/Bits";
import { Strips } from "../components/Strips";

/**
 * P2: is this grown surface a physically plausible lamina?
 *
 * The phase shared the Fleet view with P1, which shows tasks and workers --
 * neither of which P2 has. It has surfaces and verdicts on them.
 *
 * The first block counts the silence. Certification is fail-soft: a gate that
 * cannot load records GEOMETRY_UNMEASURED rather than losing the segmentation,
 * so a gate that never runs and a gate that runs cleanly both leave a page with
 * no rejections on it. Counting unmeasured surfaces is what distinguishes them,
 * and it is the number that goes at the top.
 */

type Geometry = {
  available: boolean; reason?: string;
  surfaces: number;
  by_geometry_state: Record<string, number>;
  by_physical_state: Record<string, number>;
  by_sample: Record<string, number>;
  unmeasured: number;
  meaning: Record<string, string>;
  note: string;
};

const KIND = (state: string): "ok" | "crit" | "warn" =>
  state === "GEOMETRY_CERTIFIED" ? "ok"
    : state.startsWith("GEOMETRY_REJECTED") ? "crit" : "warn";

export default function Certification({ mission, sample }:
                                      { mission?: string; sample?: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["geometry", mission ?? null, sample ?? null],
    queryFn: async () => {
      const q = new URLSearchParams();
      if (sample) q.set("sample", sample);
      if (mission) q.set("mission", mission);
      const r = await fetch("/api/geometry" + (q.size ? `?${q}` : ""));
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as Geometry;
    },
    refetchInterval: 15_000,
  });

  if (isLoading) return <Empty>loading certification…</Empty>;
  if (error) return <Empty>{String(error)}</Empty>;
  if (!data?.available) return <Empty>{data?.reason ?? "no control plane"}</Empty>;

  const states = Object.entries(data.by_geometry_state)
    .sort((a, b) => b[1] - a[1]);
  const certified = data.by_geometry_state.GEOMETRY_CERTIFIED ?? 0;
  const rejected = states
    .filter(([s]) => s.startsWith("GEOMETRY_REJECTED"))
    .reduce((n, [, c]) => n + c, 0);

  return (
    <>
      <div className="strip">
        <div className="tile steady">
          <h2>certified</h2>
          <div className="readout">{certified}</div>
        </div>
        <div className="tile steady">
          <h2>rejected</h2>
          <div className="readout">{rejected}</div>
        </div>
        <div className={data.unmeasured ? "tile warn" : "tile steady"}>
          <h2>unmeasured</h2>
          <div className="readout">{data.unmeasured}</div>
        </div>
        <div className="tile steady">
          <h2>surfaces</h2>
          <div className="readout">{data.surfaces}</div>
        </div>
      </div>

      <Card title={<>Geometry verdicts <Info label="How certification works" title="Geometry verdicts">
              Certification runs automatically when the fleet finalizes a surface; there is
              nothing to queue here. It is <b>fail-soft in control flow and fail-closed in
              verdict</b>: a gate that cannot load records <i>unmeasured</i>, which is not
              certification, rather than discarding a segmentation that took hours.
              {" "}The verdict is orthogonal to CT support — a surface can be CT_SUPPORTED
              and rejected for bridging at the same time, because they answer different
              questions.
            </Info></>} note={data.note}>
        {states.length === 0 ? (
          <Empty>no surfaces yet</Empty>
        ) : (
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l grow">Verdict</th>
                  <th>Surfaces</th>
                  <th className="l">What it means</th>
                </tr>
              </thead>
              <tbody>
                {states.map(([state, count]) => (
                  <tr key={state}>
                    <td className="l grow">
                      <Pill kind={KIND(state)}>
                        {state.replace("GEOMETRY_", "").replaceAll("_", " ").toLowerCase()}
                      </Pill>
                    </td>
                    <td>{count}</td>
                    <td className="l wrap">
                      {data.meaning[state] ?? <span className="dash">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </Card>

      <Card title="CT support" note="physical QC, measured separately" collapsed>
        <div className="scroller">
          <table>
            <thead>
              <tr><th className="l grow">State</th><th>Surfaces</th></tr>
            </thead>
            <tbody>
              {Object.entries(data.by_physical_state)
                .sort((a, b) => b[1] - a[1])
                .map(([state, count]) => (
                  <tr key={state}>
                    <td className="l grow">{state.replaceAll("_", " ").toLowerCase()}</td>
                    <td>{count}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* The independent check. The gate above is the fleet grading its own
          output; a strip is a reference the grower did not write. */}
      <Strips />
    </>
  );
}
