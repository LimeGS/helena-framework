import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useMapMeta, useRuns } from "../api";
import { MapViewer } from "../components/MapViewer";
import { Card, Empty, Pill } from "../components/Bits";

type Screen = {
  parameters: Record<string, number>;
  shape_count: number; row_count: number; qualifying_row_count: number;
  rows: { y: number; shapes: number; qualifies: boolean }[];
  verdict: string;
  percentiles: Record<string, number | null>;
  shape_y_x: [number, number];
  non_claims: string[];
};

/** Adjudication, not viewing. Every threshold of the strict screen is on the
 *  controls rather than compiled in, because watching what moves when they move
 *  is the only way to tell a robust verdict from one balanced on a number. */
export default function Screening() {
  const { data: runs } = useRuns();
  const screenable = useMemo(() => (runs ?? []).filter((r) => r.maps.length), [runs]);
  const [runId, setRunId] = useState("");
  const run = screenable.find((r) => r.run_id === runId) ?? screenable[0];
  const map = run?.maps[0];
  const { data: meta } = useMapMeta(run?.run_id, map);

  const [p, setP] = useState({
    threshold: 0.55, min_area: 12, max_area: 4000,
    row_gap: 60, row_min_shapes: 4, min_candidates: 10, min_rows: 2,
  });
  const query = new URLSearchParams(
    Object.entries(p).map(([k, v]) => [k, String(v)]),
  ).toString();

  const { data: screen, isFetching } = useQuery({
    queryKey: ["screen", run?.run_id, map, query],
    queryFn: async () => {
      const r = await fetch(
        `/api/run/${encodeURIComponent(run!.run_id)}/map/${encodeURIComponent(map!)}/screen?${query}`,
      );
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as Screen;
    },
    enabled: Boolean(run && map),
    placeholderData: (previous) => previous,
  });

  if (!screenable.length) return <Empty>no run has a probability map to screen</Empty>;

  const num = (k: keyof typeof p, label: string, step = 1, min = 0, max = 100000) => (
    <label key={k}>
      {label}
      <input
        type="number" step={step} min={min} max={max} value={p[k]}
        onChange={(e) => setP({ ...p, [k]: Number(e.target.value) })}
      />
    </label>
  );

  return (
    <>
      <Card
        title="Adjudication"
        note={
          screen && (
            <Pill kind={screen.verdict === "PASSES_STRICT_SCREEN" ? "ok" : "neg"}>
              {screen.verdict.replaceAll("_", " ").toLowerCase()}
            </Pill>
          )
        }
      >
        <div className="body-pad">
          <div className="controls">
            <select value={run?.run_id ?? ""} onChange={(e) => setRunId(e.target.value)}>
              {screenable.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} — {r.sample_id}
                </option>
              ))}
            </select>
            {isFetching && <span className="dash">screening…</span>}
          </div>

          <div className="controls" style={{ marginTop: 10 }}>
            <label htmlFor="sthr">Threshold</label>
            <input
              id="sthr" type="range" min={0} max={1} step={0.005}
              value={p.threshold}
              onChange={(e) => setP({ ...p, threshold: Number(e.target.value) })}
            />
            <output>{p.threshold.toFixed(3)}</output>
            <div className="ticks">
              {screen &&
                (["p50", "p90", "p99"] as const).map((k) => {
                  const v = screen.percentiles[k];
                  return typeof v === "number" ? (
                    <i key={k} style={{ left: `${v * 100}%` }}>
                      <span>{k} {v.toFixed(3)}</span>
                    </i>
                  ) : null;
                })}
            </div>
          </div>

          <div className="formgrid" style={{ marginTop: 10 }}>
            {num("min_area", "Min shape area (px)")}
            {num("max_area", "Max shape area (px)")}
            {num("row_gap", "Row gap (px)")}
            {num("row_min_shapes", "Shapes per row")}
            {num("min_candidates", "Candidates to pass")}
            {num("min_rows", "Rows to pass")}
          </div>
        </div>

        {screen && (
          <div className="scroller">
            <table>
              <tbody>
                <tr><td className="l">candidate shapes</td><td>{screen.shape_count}</td>
                    <td className="l dash">need {p.min_candidates}</td></tr>
                <tr><td className="l">rows detected</td><td>{screen.row_count}</td>
                    <td className="l dash"></td></tr>
                <tr><td className="l">qualifying rows</td><td>{screen.qualifying_row_count}</td>
                    <td className="l dash">need {p.min_rows}, at {p.row_min_shapes} shapes each</td></tr>
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {run && map && meta && (
        <Card title="Map" note={`${run.run_id} · ${map}`}>
          <MapViewer runId={run.run_id} name={map} meta={meta} threshold={p.threshold} />
        </Card>
      )}

      {screen && screen.rows.length > 0 && (
        <Card title="Rows" note={`${screen.qualifying_row_count} of ${screen.row_count} qualify`}>
          <div className="scroller">
            <table>
              <thead>
                <tr><th className="l grow">Row y</th><th>Shapes</th><th className="l">Qualifies</th></tr>
              </thead>
              <tbody>
                {screen.rows.map((r, i) => (
                  <tr key={i} className={r.qualifies ? "" : "muted"}>
                    <td className="l">{r.y}</td>
                    <td>{r.shapes}</td>
                    <td className="l">
                      {r.qualifies ? <Pill kind="ok">yes</Pill> : <Pill>no</Pill>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {screen && (
        <Card title="Before calling anything a candidate">
          <div className="body-pad">
            {screen.non_claims.map((n) => <p key={n}>{n}</p>)}
            <p>
              The vetting-card battery is vendored at{" "}
              <code>framework/vendored/vetting-card</code>; queue it from this phase's Action
              block with the window's bounding box.
            </p>
          </div>
        </Card>
      )}
    </>
  );
}
