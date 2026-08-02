import { useQuery } from "@tanstack/react-query";
import { Card, Empty, Pill } from "../components/Bits";

/**
 * Every ink model this deployment knows, and whether P5 can run it.
 *
 * Three populations nobody had listed together: the lane profiles, the adapters
 * that can execute one, and the method registry's record of what each checkpoint
 * is worth. A method with no lane profile cannot be queued however good it is,
 * and a lane whose adapter the queue has no command for used to be routed to
 * whichever runner came first -- which is how every TimeSformer lane was handed
 * the ResNet runner's flags and nobody noticed for as long as nothing ran them.
 *
 * Routable and validated are different axes and the page keeps them apart:
 * routable means the queue can build the command, and says nothing at all about
 * whether the model is any good on this scroll.
 */

type Lane = {
  profile_id: string | null; method_id: string | null; adapter: string | null;
  routable: boolean; reason: string | null;
  checkpoint_sha256: string | null; validation_status: string | null;
  training_pixel_um: number | null;
};
type Lanes = { available: boolean; reason?: string; lanes: Lane[];
               routable: number; note: string };

const statusKind = (status: string | null): "ok" | "warn" | "crit" | "neg" => {
  if (!status) return "neg";
  if (status.includes("DISQUALIFIED") || status.includes("FAILED")) return "crit";
  if (status.startsWith("CONTROL_AND_TARGET")) return "ok";
  return "warn";
};

export default function InkLanes() {
  const query = useQuery<Lanes>({
    queryKey: ["ink-lanes"],
    queryFn: async () => {
      const response = await fetch("/api/ink/lanes");
      if (!response.ok) throw new Error("the lane inventory could not be read");
      return response.json();
    },
    staleTime: 300_000,
  });

  if (query.isLoading) return <Empty>loading…</Empty>;
  if (query.isError) return <Empty>{String(query.error)}</Empty>;
  const data = query.data!;
  if (!data.available) return <Empty>{data.reason ?? "unavailable"}</Empty>;

  return (
    <>
      <Card title="Ink models">
        <div className="knobgrid">
          <div>
            <div className="big">{data.routable}</div>
            <div className="dash">lanes P5 can run</div>
          </div>
          <div>
            <div className="big">{data.lanes.length - data.routable}</div>
            <div className="dash">
              known and not runnable here — no lane profile, no adapter, or a
              job shape this queue cannot express
            </div>
          </div>
        </div>
        <p className="dash">{data.note}</p>
      </Card>

      <Card title="Every lane and what runs it">
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th>lane</th><th>method</th><th>runner</th>
                <th>training µm</th><th>validation</th><th></th>
              </tr>
            </thead>
            <tbody>
              {data.lanes.map((lane) => (
                <tr key={(lane.profile_id ?? "") + (lane.method_id ?? "")}>
                  <td className="mono">{lane.profile_id ?? <span className="dash">— no profile</span>}</td>
                  <td className="mono">{lane.method_id ?? "—"}</td>
                  <td className="mono">
                    {lane.adapter ? lane.adapter.split("/").pop() : "—"}
                  </td>
                  <td>{lane.training_pixel_um ?? "—"}</td>
                  <td>
                    <Pill kind={statusKind(lane.validation_status)}>
                      {(lane.validation_status ?? "unrecorded").replaceAll("_", " ").toLowerCase()}
                    </Pill>
                  </td>
                  {/* Either a one-word pill or a whole sentence explaining why
                      the queue cannot build this lane. `wrap` alone caps it at
                      18ch, which broke those sentences a word or two per line. */}
                  <td className="wrap prose dash">
                    {lane.routable ? <Pill kind="ok">routable</Pill> : lane.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
