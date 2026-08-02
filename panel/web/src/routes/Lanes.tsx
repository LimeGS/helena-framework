import { useLanes } from "../api";
import {Card, Pill, queryGate} from "../components/Bits";

export default function Lanes() {
  const { data, isLoading, error } = useLanes();
  const gate = queryGate({ isLoading, error, data }, "loading lanes…");
  if (gate) return gate;
  // The gate covers every unset case; the compiler cannot see that
  // through a helper.
  if (!data) return null;

  return (
    <>
      <Card title="Ink profiles" note={`the upstream contract clips and divides by ${data.upstream_clip}`}>
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l grow">Profile</th>
                <th className="l">model_type</th>
                <th>µm</th>
                <th>frames</th>
                <th>tile</th>
                <th>clip</th>
                <th className="l">Adapter</th>
                <th className="l">State</th>
              </tr>
            </thead>
            <tbody>
              {data.profiles.map((p) => {
                const c = p.input_contract as Record<string, any>;
                return (
                  <tr key={p.profile_id}>
                    <td className="l wrap">{p.profile_id}</td>
                    <td className="l">{c.model_type ?? "—"}</td>
                    <td>{c.training_pixel_um ?? "—"}</td>
                    <td>{c.frames ?? "—"}</td>
                    <td>{c.tile_size_y_x?.[0] ?? "—"}</td>
                    <td>{c.max_clip_value ?? "—"}</td>
                    <td className="l wrap">{p.adapter.split("/").pop()}</td>
                    <td className="l">
                      {p.disqualified ? (
                        <Pill kind="crit">disqualified</Pill>
                      ) : c.model_type ? (
                        <Pill kind="ok">routable by model_type</Pill>
                      ) : (
                        <Pill kind="neg">routable by adapter</Pill>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {data.profiles.filter((p) => p.disqualified).map((p) => (
        <Card key={p.profile_id} title={`Disqualified · ${p.method_id}`}>
          <div className="body-pad">
            <p><b>{p.registry_status}</b></p>
            <p>{p.registry_policy}</p>
          </div>
        </Card>
      ))}
    </>
  );
}
