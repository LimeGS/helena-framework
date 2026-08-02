import { useQuery } from "@tanstack/react-query";
import {Card, Pill, queryGate} from "../components/Bits";

type Phase = {
  id: string; slug: string; name: string; one_line: string;
  consumes: string; produces: string; lives_in: string[];
  maturity: string; distributed?: boolean;
  how_to_run: string; how_it_fails: string; gate: string | null;
};

const MATURITY: Record<string, "ok" | "warn" | "crit" | "neg"> = {
  WORKING: "ok",
  WORKING_WITH_A_KNOWN_LIMIT: "warn",
  PARTIAL: "warn",
  PARTIAL_LOCALLY: "warn",
  NOT_REACHED: "neg",
};

type Impl = {
  component: string; phases: string[]; status: string; remote?: string;
  viewer?: string; local_path: string; what_it_does: string;
  entry_points?: Record<string, string>; why_it_matters_here?: string;
  known_state?: string; caveat?: string;
};

const STATUS: Record<string, "ok" | "warn" | "neg"> = {
  PUBLISHED_WORKING: "ok",
  WORKING_UNPUBLISHED: "warn",
  PUBLISHED_PARTIAL: "warn",
};

export default function Phases() {
  const impls = useQuery({
    queryKey: ["implementations"],
    queryFn: async () => {
      const r = await fetch("/api/implementations");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { implementations: Impl[] };
    },
    staleTime: 10 * 60 * 1000,
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ["phases"],
    queryFn: async () => {
      const r = await fetch("/api/phases");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { phases: Phase[]; note: string };
    },
    staleTime: 10 * 60 * 1000,
  });

  const gate = queryGate({ isLoading, error, data }, "loading the phase vocabulary…");
  if (gate) return gate;
  // The gate covers every unset case; the compiler cannot see that
  // through a helper.
  if (!data) return null;

  return (
    <>
      <Card title="Pipeline phases" note={`${data.phases.length} phases, CT to a rendered papyrus`}>
        <div className="body-pad">
          <p>
            The six directories under <code>framework/stages/</code> are the coarse grouping. These
            are the discrete steps, including the ones that live inside a stage and have caused the
            most trouble: geometry certification, surface-volume depth, and lane liveness.
          </p>
          <p>Maturity is what is true today, not what is planned.</p>
        </div>
      </Card>

      {data.phases.map((p) => (
        <Card
          key={p.id}
          title={`${p.id} · ${p.name}`}
          note={
            <>
              <Pill kind={MATURITY[p.maturity] ?? "neg"}>{p.maturity.replaceAll("_", " ").toLowerCase()}</Pill>
              {p.distributed && <> <Pill kind="run">distributed</Pill></>}
            </>
          }
        >
          <div className="body-pad">
            <p style={{ color: "var(--ink)", fontSize: 13.5 }}>{p.one_line}</p>
            <div className="phasegrid">
              <div><span className="k">consumes</span><span>{p.consumes}</span></div>
              <div><span className="k">produces</span><span>{p.produces}</span></div>
              <div><span className="k">lives in</span><span>{p.lives_in.map((l) => <code key={l}>{l}</code>)}</span></div>
              {p.gate && <div><span className="k">gate</span><span>{p.gate}</span></div>}
            </div>
            <h4>How to run it</h4>
            <p>{p.how_to_run}</p>
            <h4>How it fails</h4>
            <p>{p.how_it_fails}</p>

            {(impls.data?.implementations ?? [])
              .filter((i) => i.phases.includes(p.id))
              .map((i) => (
                <div className="impl" key={i.component}>
                  <div className="impl-head">
                    <span className="impl-name">{i.component}</span>
                    <Pill kind={STATUS[i.status] ?? "neg"}>
                      {i.status.replaceAll("_", " ").toLowerCase()}
                    </Pill>
                    {i.remote && (
                      <a href={i.remote} target="_blank" rel="noreferrer">
                        repo
                      </a>
                    )}
                    {i.viewer && (
                      <a href={i.viewer} target="_blank" rel="noreferrer">
                        viewer
                      </a>
                    )}
                  </div>
                  <p>{i.what_it_does}</p>
                  {i.why_it_matters_here && <p><b>Why here:</b> {i.why_it_matters_here}</p>}
                  {i.known_state && <p><b>Known state:</b> {i.known_state}</p>}
                  {i.caveat && <p><b>Caveat:</b> {i.caveat}</p>}
                  {i.entry_points && (
                    <ul className="entrypoints">
                      {Object.entries(i.entry_points).map(([k, v]) => (
                        <li key={k}>
                          <code>{k}</code> — {v}
                        </li>
                      ))}
                    </ul>
                  )}
                  <p className="dash">{i.local_path}</p>
                </div>
              ))}
          </div>
        </Card>
      ))}
    </>
  );
}
