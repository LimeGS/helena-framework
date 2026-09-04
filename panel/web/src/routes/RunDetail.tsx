import { useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import { useMapMeta, useRun } from "../api";
import { MapViewer } from "../components/MapViewer";
import { Card, Empty, Pill } from "../components/Bits";

export default function RunDetail() {
  const { runId } = useParams();
  const [params, setParams] = useSearchParams();
  const { data: run, isLoading } = useRun(runId);

  const selected = params.get("map") ?? run?.maps[0];
  const { data: meta } = useMapMeta(runId, selected);
  const [threshold, setThreshold] = useState(0.5);
  const [gamma, setGamma] = useState(1);

  const marks = useMemo(() => {
    if (!meta) return [];
    const keys = ["p50", "p90", "p99"] as const;
    return keys.flatMap((k) => {
      const v = meta[k];
      return typeof v === "number" ? [{ k: k as string, v }] : [];
    });
  }, [meta]);

  if (isLoading) return <Empty>loading run…</Empty>;
  if (!run) return <Empty>no such run</Empty>;

  const profile = run.profile;
  const shaMatches = profile ? profile.checkpoint_sha256 === run.checkpoint_sha : null;

  return (
    <div className="split wide">
      <Card
        title={run.run_id}
        note={
          <select
            value={selected}
            onChange={(e) => setParams({ map: e.target.value }, { replace: true })}
          >
            {run.maps.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        }
      >
        {selected && meta ? (
          <>
            <MapViewer
              runId={run.run_id}
              name={selected}
              meta={meta}
              threshold={threshold}
              gamma={gamma}
            />
            <div className="controls">
              <label htmlFor="thr">Threshold</label>
              <input
                id="thr"
                type="range"
                min={0}
                max={1}
                step={0.005}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
              <output>{threshold.toFixed(3)}</output>
              <div className="ticks">
                {marks.map((m) => (
                  <i key={m.k} style={{ left: `${m.v * 100}%` }}>
                    <span>
                      {m.k} {m.v.toFixed(3)}
                    </span>
                  </i>
                ))}
              </div>
              <label htmlFor="gam">Gamma</label>
              <input
                id="gam"
                type="range"
                min={0.3}
                max={3}
                step={0.05}
                value={gamma}
                onChange={(e) => setGamma(Number(e.target.value))}
              />
              <output>{gamma.toFixed(2)}</output>
            </div>
            <p className="hint">
              Threshold and gamma are applied in the shader: moving them asks the server for
              nothing. Drag to pan; the wheel zooms and fetches only the visible window.
            </p>
          </>
        ) : (
          <Empty>This run left no .npy on disk.</Empty>
        )}
      </Card>

      <div className="rail">
        <Card title="Provenance">
          <div className="chain">
            <div className="link">
              <span className="k">Scroll</span>
              <span className="v">{run.sample_id}</span>
            </div>
            <div className={`link ${profile ? "" : "unknown"}`}>
              <span className="k">Lane</span>
              <span className="v">{run.lane_id}</span>
              {!profile && <Pill>no profile in the repo</Pill>}
            </div>
            <div className={`link ${shaMatches === false ? "bad" : run.checkpoint_sha ? "" : "unknown"}`}>
              <span className="k">Checkpoint</span>
              <span className="v">{run.checkpoint_sha.slice(0, 32) || "not declared"}…</span>
              {shaMatches === true && <Pill kind="ok">matches the profile</Pill>}
              {shaMatches === false && <Pill kind="crit">does not match the profile</Pill>}
            </div>
            <div className={`link ${run.contract_ok ? "" : "bad"}`}>
              <span className="k">Normalization</span>
              <span className="v">{run.normalization || "not declared"}</span>
            </div>
            <div className="link">
              <span className="k">Receipt</span>
              <span className="v">{run.schema}</span>
            </div>
          </div>
        </Card>

        {run.liveness && (
          <Card title="Lane liveness">
            <div className="body-pad">
              <Pill kind={run.liveness.verdict === "ALIVE" ? "ok" : "crit"}>
                {run.liveness.verdict}
              </Pill>
              {run.liveness.reason && <p>{run.liveness.reason}</p>}
              {run.liveness.interpretation && <p>{run.liveness.interpretation}</p>}
            </div>
          </Card>
        )}

        <Card title="Statistics" note={meta ? `${meta.width}×${meta.height}` : undefined}>
          <div className="scroller">
            <table>
              <tbody>
                {Object.entries(meta ?? run.stats).map(([k, v]) => (
                  <tr key={k}>
                    <td className="l">{k}</td>
                    <td>{typeof v === "number" ? Number(v.toPrecision(6)) : String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Receipt">
          <div className="body-pad">
            <p>
              <code>{run.receipt_path}</code>
            </p>
            <a href={`/api/run/${run.run_id}`}>view the full JSON</a>
          </div>
        </Card>
      </div>
    </div>
  );
}
