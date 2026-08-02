import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { failure } from "../api";
import { useState } from "react";
import { Card, queryGate, Pill } from "./Bits";

/**
 * Credentials the workers need, set here and held in the control plane.
 *
 * They used to live in a file on one host's tmpfs: lost on every reboot, absent
 * on every other machine, and placed by hand each time -- so surface QC and the
 * ink worker refused to start after a restart until somebody remembered. A
 * worker is ephemeral and has to be able to start from a database URL alone.
 *
 * Write-only. The page can say a credential is set, how long it is and who set
 * it; it cannot show it back, because a value that can be read out of an API is
 * a value in a second place.
 */

type Secret = {
  name: string; set: boolean; characters: number;
  updated_at: string | null; updated_by: string | null;
};

export function FleetSecrets() {
  const client = useQueryClient();
  const [draft, setDraft] = useState<Record<string, string>>({});

  const query = useQuery<{ secrets: Secret[]; note: string }>({
    queryKey: ["secrets"],
    queryFn: async () => {
      const r = await fetch("/api/secrets");
      if (!r.ok) throw await failure(r);
      return r.json();
    },
    // A 409 saying there is no control plane is an answer, not a hiccup.
    // Retrying it three times only delays showing somebody the reason.
    retry: false,
  });

  const save = useMutation({
    mutationFn: async ({ name, value }: { name: string; value: string }) => {
      const r = await fetch(`/api/secrets/${encodeURIComponent(name)}`, {
        method: value ? "PUT" : "DELETE",
        headers: { "Content-Type": "application/json" },
        ...(value ? { body: JSON.stringify({ value }) } : {}),
      });
      if (!r.ok) throw await failure(r);
      return r.json();
    },
    onSuccess: (_r, variables) => {
      setDraft((d) => ({ ...d, [variables.name]: "" }));
      client.invalidateQueries({ queryKey: ["secrets"] });
    },
  });

  // Guarded, not asserted. Between retries a failed query is neither loading nor
  // errored and its data is undefined, and the assertion turned a deployment
  // with no control plane -- a fresh install -- into a Configuration page that
  // rendered nothing at all.
  const gate = queryGate(query, "reading the fleet's credentials…");
  if (gate || !query.data) {
    return (
      <Card title="Credentials" collapsed>
        {gate ?? null}
      </Card>
    );
  }

  return (
    <Card title="Credentials" collapsed>
      <p className="dash">{query.data.note}</p>
      <div className="scroller">
        <table>
          <thead>
            <tr><th>name</th><th>held</th><th>set by</th><th>new value</th><th /></tr>
          </thead>
          <tbody>
            {query.data.secrets.map((s) => (
              <tr key={s.name}>
                <td className="mono">{s.name}</td>
                <td>
                  {s.set
                    ? <Pill kind="ok">{s.characters} characters</Pill>
                    : <span className="dash">not set</span>}
                </td>
                <td className="dash">
                  {s.updated_by ?? "—"}
                  {s.updated_at ? ` · ${s.updated_at.slice(0, 10)}` : ""}
                </td>
                <td>
                  <input type="password" autoComplete="off"
                         value={draft[s.name] ?? ""}
                         placeholder={s.set ? "replace it" : "paste it"}
                         onChange={(e) =>
                           setDraft({ ...draft, [s.name]: e.target.value })} />
                </td>
                <td>
                  <button
                    disabled={save.isPending || (!draft[s.name] && !s.set)}
                    onClick={() =>
                      save.mutate({ name: s.name, value: draft[s.name] ?? "" })}>
                    {draft[s.name] ? "save" : "forget"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {save.isError && <Pill kind="crit">{String(save.error)}</Pill>}
      <p className="dash">
        A worker reads these when it starts, so a change reaches the fleet at the
        next restart rather than mid-job. An environment variable already set on
        a host wins, so a key exported for one run is not silently overridden.
      </p>
    </Card>
  );
}
