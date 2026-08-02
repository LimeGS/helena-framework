import { memo, useDeferredValue, useMemo, useState } from "react";
import { Link } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { useScrolls, type Scroll } from "../api";
import { Card, Empty, Num, Pill, Tile } from "../components/Bits";

type Filter = "all" | "screened" | "unscreened" | "with-scale";

const Row = memo(function Row({ s }: { s: Scroll }) {
  return (
    <tr>
      <td className="scrollid grow">{s.sample_id}</td>
      <td>
        {s.pixel_um || <span className="dash">—</span>}
        {s.higher_res && <span title="has a higher-resolution sibling"> ▲</span>}
      </td>
      <td>{s.energy_kev ?? <span className="dash">—</span>}</td>
      <td>{s.runs || <span className="dash">—</span>}</td>
      <td className="l wrap">
        {s.lane && s.run_id ? <Link to={`/run/${s.run_id}`}>{s.lane}</Link> : <span className="dash">—</span>}
      </td>
      <td>
        <Num v={s.p90} />
      </td>
      <td className="l">
        {s.verdict === "SCREENED" ? <Pill kind="neg">Screened</Pill> : <Pill>Not screened</Pill>}
      </td>
    </tr>
  );
});

export default function Scrolls() {
  const { data, isLoading, error } = useScrolls();
  const client = useQueryClient();
  const [text, setText] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const deferred = useDeferredValue(text);

  const rows = useMemo(() => {
    if (!data) return [];
    const needle = deferred.trim().toLowerCase();
    return data.scrolls.filter((s) => {
      if (needle && !s.sample_id.toLowerCase().includes(needle)) return false;
      if (filter === "screened") return s.runs > 0;
      if (filter === "unscreened") return s.runs === 0;
      if (filter === "with-scale") return Boolean(s.pixel_um);
      return true;
    });
  }, [data, deferred, filter]);

  if (isLoading) return <Empty>loading the scroll inventory…</Empty>;
  if (error || !data) return <Empty>{String(error ?? "no data")}</Empty>;

  const age = data.fetched_at ? Math.round((Date.now() / 1000 - data.fetched_at) / 3600) : null;

  return (
    <>
      <div className="strip">
        <Tile title="Scrolls" value={data.total}>
          <p>every scroll the open data exposes, not only the mission cohort</p>
        </Tile>
        <Tile title="With a declared scale" value={data.with_scale}>
          <p>voxel size and energy from the frozen catalog</p>
        </Tile>
        <Tile title="Screened" tone={data.screened_count ? "steady" : "warn"} value={data.screened_count}>
          <p>at least one run on disk</p>
        </Tile>
        <Tile title="Inventory">
          <p>
            source: <b>{data.inventory_origin}</b>
            {age !== null && age >= 0 && <> · fetched {age}h ago</>}
          </p>
          <button
            onClick={async () => {
              await fetch("/api/scrolls?refresh=true");
              client.invalidateQueries({ queryKey: ["scrolls"] });
            }}
          >
            refresh
          </button>
        </Tile>
      </div>

      <Card title="Inventory" note={`${rows.length} of ${data.total} shown`}>
        <div className="body-pad">
          <div className="controls">
            <input
              className="search"
              type="search"
              placeholder="filter by scroll id…"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
              <option value="all">all</option>
              <option value="with-scale">with a declared scale</option>
              <option value="screened">screened</option>
              <option value="unscreened">not screened</option>
            </select>
          </div>
        </div>
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l grow">Scroll</th>
                <th>µm</th>
                <th>keV</th>
                <th>Runs</th>
                <th className="l">Last lane</th>
                <th>p90</th>
                <th className="l">State</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <Row key={s.sample_id} s={s} />
              ))}
            </tbody>
          </table>
        </div>
        {rows.length === 0 && <Empty>nothing matches</Empty>}
      </Card>

      <Card title="Scale note">
        <div className="body-pad">
          <p>
            µm and keV are blank for any scroll the frozen catalog does not describe. The bucket
            listing gives identity, not acquisition parameters, and reading a scan's own metadata
            is what would fill them — the panel does not do that yet. Scope is a mission's job:
            create one naming the scrolls you are attempting.
          </p>
        </div>
      </Card>
    </>
  );
}
