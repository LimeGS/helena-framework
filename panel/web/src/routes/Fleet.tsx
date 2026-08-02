import { useFleet } from "../api";
import {Card, Tile, queryGate} from "../components/Bits";

export default function Fleet() {
  const { data, isLoading, error } = useFleet();
  const gate = queryGate({ isLoading, error, data }, "querying the fleet…");
  if (gate) return gate;
  // The gate covers every unset case; the compiler cannot see that
  // through a helper.
  if (!data) return null;
  if (!data.available)
    return (
      <Card title="No connection to the fleet">
        <div className="body-pad">
          <p>{data.reason}</p>
          <p>
            Set <code>CX_DB</code>, for example{" "}
            <code>postgresql://campaignx:…@127.0.0.1:55432/campaignx</code>.
          </p>
        </div>
      </Card>
    );

  return (
    <>
      <div className="strip">
        <Tile title="Tasks" value={data.tasks} />
        <Tile title="Attempts" value={data.attempts} />
        <Tile title="Surfaces" value={data.surfaces} />
        <Tile title="Events" value={data.events} />
      </div>
      <div className="split">
        <Card
          title="Tasks by state"
          note={`${data.leased} currently leased · ${data.stale_leases} stale`}
        >
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l grow">State</th>
                  <th>Tasks</th>
                </tr>
              </thead>
              <tbody>
                {data.task_states?.map((s) => (
                  <tr key={s.state}>
                    <td className="l">{s.state}</td>
                    <td>{s.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <div className="rail">
          <Card title="Events">
            <div className="scroller">
              <table>
                <tbody>
                  {data.events_by_type?.map((e) => (
                    <tr key={e.type}>
                      <td className="l wrap">{e.type}</td>
                      <td>{e.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Workers">
            <div className="scroller">
              <table>
                <tbody>
                  {data.workers?.length ? (
                    data.workers.map((w) => (
                      <tr key={w.worker_id}>
                        <td className="l wrap">{w.worker_id}</td>
                        <td>{w.attempts}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td>no attempt has a worker assigned</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
