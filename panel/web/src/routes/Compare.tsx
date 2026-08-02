import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useMapMeta, useRun, useRuns } from "../api";
import { MapViewer } from "../components/MapViewer";
import { Card, Empty, Pill } from "../components/Bits";

const CONTRACT_KEYS = ["schema", "sample_id", "lane_id", "checkpoint_sha", "normalization",
  "clip_value", "divisor"] as const;

/** Contract fields first: those are the ones that should be equal if the
 *  comparison means anything. The /255 defect was exactly a contract
 *  difference that produced a plausible-looking output. */
export default function Compare() {
  const [params, setParams] = useSearchParams();
  const { data: runs } = useRuns();
  const a = params.get("a") ?? undefined;
  const b = params.get("b") ?? undefined;
  const { data: left } = useRun(a);
  const { data: right } = useRun(b);

  // One view drives both canvases, so panning either compares the same place.
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const [threshold, setThreshold] = useState(0.5);

  const leftMap = left?.maps[0];
  const rightMap = right?.maps[0];
  const { data: leftMeta } = useMapMeta(a, leftMap);
  const { data: rightMeta } = useMapMeta(b, rightMap);

  const rows = useMemo(() => {
    if (!left || !right) return [];
    const out: { key: string; left: unknown; right: unknown; same: boolean; numeric: boolean }[] =
      CONTRACT_KEYS.map((k) => ({
      key: k as string,
      left: left[k] as unknown,
      right: right[k] as unknown,
      same: left[k] === right[k],
      numeric: false,
    }));
    const stats = new Set([...Object.keys(left.stats), ...Object.keys(right.stats)]);
    for (const k of [...stats].sort()) {
      out.push({
        key: `stats.${k}`,
        left: left.stats[k],
        right: right.stats[k],
        same: left.stats[k] === right.stats[k],
        numeric: true,
      });
    }
    return out;
  }, [left, right]);

  const differing = rows.filter((r) => !r.same).length;

  return (
    <>
      <Card title="Pick two runs" note="contract fields come first">
        <div className="body-pad">
          <div className="controls">
            {(["a", "b"] as const).map((slot) => (
              <select
                key={slot}
                value={params.get(slot) ?? ""}
                onChange={(e) => {
                  const next = new URLSearchParams(params);
                  next.set(slot, e.target.value);
                  setParams(next, { replace: true });
                }}
              >
                <option value="">— run {slot.toUpperCase()} —</option>
                {runs?.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id}
                  </option>
                ))}
              </select>
            ))}
          </div>
        </div>
      </Card>

      {left && right && (
        <>
          <Card
            title="Differences"
            note={
              differing ? (
                <Pill kind="warn">{differing} of {rows.length} fields differ</Pill>
              ) : (
                <Pill kind="ok">identical</Pill>
              )
            }
          >
            <div className="scroller">
              <table>
                <thead>
                  <tr>
                    <th className="l">Field</th>
                    <th className="l">{left.run_id}</th>
                    <th className="l">{right.run_id}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.key} className={r.same ? "" : "diff"}>
                      <td className="l">{r.key}</td>
                      {([r.left, r.right] as unknown[]).map((v, i) => (
                        <td className="l wrap" key={i}>
                          {v === null || v === undefined ? (
                            <span className="dash">—</span>
                          ) : r.numeric ? (
                            Number(Number(v).toPrecision(6))
                          ) : (
                            String(v)
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="controls" style={{ margin: "12px 0" }}>
            <label htmlFor="cthr">Threshold (both)</label>
            <input
              id="cthr"
              type="range"
              min={0}
              max={1}
              step={0.005}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <output>{threshold.toFixed(3)}</output>
          </div>

          <div className="split" style={{ gridTemplateColumns: "1fr 1fr" }}>
            {[
              { run: left, map: leftMap, meta: leftMeta },
              { run: right, map: rightMap, meta: rightMeta },
            ].map(({ run, map, meta }) => (
              <Card key={run.run_id} title={run.run_id} note={map}>
                {map && meta ? (
                  <MapViewer
                    runId={run.run_id}
                    name={map}
                    meta={meta}
                    threshold={threshold}
                    height={420}
                    view={view}
                    onViewChange={setView}
                  />
                ) : (
                  <Empty>no .npy</Empty>
                )}
              </Card>
            ))}
          </div>
        </>
      )}
    </>
  );
}
