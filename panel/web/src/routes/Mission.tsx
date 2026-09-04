import { memo } from "react";
import { Link } from "react-router";
import { useAppState, useFleet, useHosts, type Target } from "../api";
import { Card, Empty, Num, Pill, Tile } from "../components/Bits";

const TargetRow = memo(function TargetRow({ t }: { t: Target }) {
  return (
    <tr>
      <td className="scrollid grow">{t.sample_id}</td>
      <td>
        {t.pixel_um}
        {t.higher_res && <span title="has a higher-resolution sibling"> ▲</span>}
      </td>
      <td>{t.energy_kev ?? <span className="dash">—</span>}</td>
      <td>{t.runs || <span className="dash">—</span>}</td>
      <td className="l wrap">
        {t.lane && t.run_id ? <Link to={`/run/${t.run_id}`}>{t.lane}</Link> : <span className="dash">—</span>}
      </td>
      <td>
        <Num v={t.p90} />
      </td>
      <td className="l">
        {t.verdict === "SCREENED" ? <Pill kind="neg">Screened</Pill> : <Pill>Not screened</Pill>}
      </td>
    </tr>
  );
});

/**
 * What the fleet is, added up.
 *
 * This tile used to be the panel's own nvidia-smi, and the panel runs in a
 * container with no card: on a machine with two GPUs it said "nvidia-smi not
 * available on this host" and that was the whole hardware story the mission
 * page told. The hosts table already holds every machine's reading, reported by
 * the workers that can see it, so the summary comes from there -- and it counts
 * the fleet rather than whichever host happens to serve the page.
 *
 * Only enabled hosts. A disabled one takes no work, so counting its cores as
 * capacity would overstate what the fleet can do.
 */
function Hardware() {
  const { data } = useHosts();
  const hosts = (data?.hosts ?? []).filter((h) => h.enabled);
  const cards = hosts.flatMap((h) => h.last_state?.gpus ?? []);
  const cores = hosts.reduce((a, h) => a + (h.last_state?.cores ?? 0), 0);
  const ramTotal = hosts.reduce((a, h) => a + (h.last_state?.ram_total_gb ?? 0), 0);
  const ramFree = hosts.reduce((a, h) => a + (h.last_state?.ram_free_gb ?? 0), 0);
  const busiest = cards.reduce((a, g) => Math.max(a, g.util_pct), 0);
  // Reported, not registered: a host that has never checked in has no reading to
  // add, and saying "4 cores across 2 machines" when one of them is silent is
  // the kind of number that gets believed.
  const reporting = hosts.filter((h) => h.last_state).length;

  return (
    <Tile title="Fleet hardware" tone={busiest > 5 ? "busy" : hosts.length ? "steady" : "warn"}>
      <div className="hardware">
        <span><b>{cards.length}</b> gpu</span>
        <span><b>{cores}</b> cpu</span>
        <span><b>{ramTotal ? `${Math.round(ramFree)}/${Math.round(ramTotal)}` : "—"}</b> gb mem</span>
        <span><b>{hosts.length}</b> {hosts.length === 1 ? "instance" : "instances"}</span>
      </div>
      <p>
        {hosts.length === 0
          ? "no hosts registered — register one in Configuration"
          : reporting === hosts.length
            ? `all ${hosts.length} reporting`
            : `${reporting} of ${hosts.length} reporting`}
        {cards.length > 0 && ` · busiest card ${busiest}%`}
      </p>
    </Tile>
  );
}

/**
 * Which workers are alive, and which of the GPU ones cannot see their card.
 *
 * `state` alone cannot draw this line: a worker whose GPU passthrough breaks
 * keeps polling on schedule, so POLLING looks identical to healthy. helena-
 * ink-0 did exactly that for five hours -- `docker ps` said "Up", this row
 * would have said POLLING, and nothing said a card was missing until someone
 * noticed six P5 jobs were not draining.
 *
 * gpu_visible is read fresh, per worker, per poll (ink_worker.py's
 * worker_gpu_visible()) -- deliberately not the host-wide hardware count in
 * the tile above, which would have kept showing a card present the whole
 * time, reported by whichever *other* worker on the same host could still
 * see one. A host can look fully hardware-equipped while one of its workers
 * cannot reach any of it.
 */
function Workers() {
  const { data } = useFleet();
  if (!data?.available) {
    return (
      <Tile title="Workers" tone="warn" value="—">
        <p>{data?.reason ?? "no connection to the fleet"}</p>
      </Tile>
    );
  }
  const workers = data.workers ?? [];
  const silent = workers.filter((w) => w.state === "SILENT");
  const blind = workers.filter((w) => w.gpu_visible === false);
  const troubled = Array.from(new Set([...silent, ...blind].map((w) => w.worker_id)));

  return (
    <Tile title="Workers" value={workers.length}
         tone={troubled.length ? "alert" : workers.length ? "steady" : "warn"}>
      {workers.length === 0 ? (
        <p>no worker has ever polled</p>
      ) : (
        <>
          <p>
            {workers.length - silent.length} polling
            {silent.length > 0 && <> · <b>{silent.length}</b> silent</>}
            {blind.length > 0 && <> · <b>{blind.length}</b> blind to their GPU</>}
          </p>
          {troubled.length > 0 && <p className="wrap">{troubled.join(", ")}</p>}
        </>
      )}
    </Tile>
  );
}

export default function Mission() {
  const { data, isLoading, error } = useAppState();

  if (isLoading) return <Empty>loading state…</Empty>;
  if (error || !data) return <Empty>{String(error ?? "no data")}</Empty>;

  const { fleet, integrity, targets } = data;

  return (
    <>
      <div className="strip">
        <Hardware />

        <Tile
          title="Fleet"
          tone={fleet.available ? "steady" : "warn"}
          value={fleet.available ? fleet.surfaces : "—"}
        >
          {fleet.available ? (
            <p>
              surfaces from {fleet.tasks} tasks · <b>{fleet.stale_leases}</b> stale leases
            </p>
          ) : (
            <p>{fleet.reason}</p>
          )}
        </Tile>

        <Workers />

        <Tile title="Runs" tone="steady" value={data.run_count}>
          <p>{data.lane_count} ink lanes with a declared profile</p>
        </Tile>

        <Tile
          title="Integrity"
          tone={integrity.length ? "alert" : "steady"}
          value={integrity.length}
        >
          <p>
            {integrity.length
              ? "findings that contradict the declared contract"
              : "every receipt matches its contract"}
          </p>
        </Tile>
      </div>

      <div className="split">
        <Card title="Scrolls in this mission" note="scale and energy from the frozen catalog">
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
                {targets.map((t) => (
                  <TargetRow key={t.sample_id} t={t} />
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="rail">
          {integrity.length > 0 && (
            <Card title="Findings">
              {integrity.map((f, i) => (
                <div className="lane" key={`${f.run_id}-${f.kind}-${i}`}>
                  <div className="lane-id">
                    <Link to={`/run/${f.run_id}`}>{f.run_id}</Link>
                  </div>
                  <div className="lane-meta">
                    <span>{f.sample_id}</span>
                    <span>{f.kind}</span>
                  </div>
                  <div>
                    <Pill kind={f.severity === "critical" ? "crit" : "warn"}>{f.detail}</Pill>
                  </div>
                </div>
              ))}
            </Card>
          )}

          {fleet.available && fleet.surfaces_by_sample && (
            <Card title="Surfaces by scroll">
              <div className="scroller">
                <table>
                  <tbody>
                    {fleet.surfaces_by_sample.map((s) => (
                      <tr key={s.sample_id}>
                        <td className="l">{s.sample_id}</td>
                        <td>{s.count}</td>
                        <td>{s.area_cm2.toFixed(2)} cm²</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
