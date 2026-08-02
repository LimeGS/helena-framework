import { memo } from "react";
import { Link } from "react-router";
import { useRuns, type Run } from "../api";
import { Card, Empty, Num, Pill } from "../components/Bits";

const Row = memo(function Row({ r }: { r: Run }) {
  return (
    <tr>
      <td className="l grow">
        <Link to={`/run/${r.run_id}`}>{r.run_id}</Link>
      </td>
      <td className="l">{r.sample_id}</td>
      <td className="l wrap">{r.lane_id}</td>
      {(["p50", "p90", "p99", "max"] as const).map((k) => (
        <td key={k}>
          <Num v={r.stats[k]} />
        </td>
      ))}
      <td className="l">
        {r.clip_value ? `clip ${r.clip_value} / div ${r.divisor ?? "?"}` : <span className="dash">not declared</span>}
      </td>
      <td className="l">
        {r.clip_value === null ? (
          <Pill>not declared</Pill>
        ) : r.contract_ok ? (
          <Pill kind="ok">matches</Pill>
        ) : (
          <Pill kind="crit">divergent</Pill>
        )}
      </td>
      <td className="l">
        {r.liveness ? (
          <Pill kind={r.liveness.verdict === "ALIVE" ? "ok" : "crit"}>{r.liveness.verdict}</Pill>
        ) : (
          <span className="dash">—</span>
        )}
      </td>
    </tr>
  );
});

export default function Runs() {
  const { data, isLoading, error } = useRuns();
  if (isLoading) return <Empty>loading runs…</Empty>;
  if (error) return <Empty>{String(error)}</Empty>;
  if (!data?.length) return <Empty>No receipts. Is CX_RUNS right?</Empty>;

  return (
    <Card title="Runs" note="indexed from the receipts on disk">
      <div className="scroller">
        <table>
          <thead>
            <tr>
              <th className="l grow">Run</th>
              <th className="l">Scroll</th>
              <th className="l">Lane</th>
              <th>p50</th>
              <th>p90</th>
              <th>p99</th>
              <th>max</th>
              <th className="l">Normalization</th>
              <th className="l">Contract</th>
              <th className="l">Liveness</th>
            </tr>
          </thead>
          <tbody>
            {data.map((r) => (
              <Row key={r.run_id} r={r} />
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
