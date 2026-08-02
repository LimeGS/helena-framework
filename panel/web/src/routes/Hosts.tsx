import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { failure, useHosts } from "../api";
import { Card, Empty, Pill } from "../components/Bits";

// Mirrors ASSIGNABLE_ROLES on the server, which refuses anything else. A role
// describing where infrastructure lives -- the host carrying the control-plane
// database is tagged postgres -- is a fact about the deployment rather than a
// choice, so it is not offered here and the server keeps it when this page
// saves.
const ROLES = ["segment", "render", "ink", "mesh", "build"] as const;

const ROLE_MEANING: Record<string, string> = {
  segment: "grows surfaces with VC3D — CPU only, so any host can take it",
  render: "turns a surface into a layer stack — no GPU",
  ink: "the only stage that needs a GPU worth having",
  mesh: "comparative backend, research only — its surfaces are not catalogued",
  build: "compiles the images, which is why they are built where they run",
};

export default function Hosts() {
  const client = useQueryClient();
  const { data, isLoading, error } = useHosts();
  const [form, setForm] = useState({ host_id: "", ssh_target: "", roles: "ink", notes: "" });

  const register = useMutation({
    mutationFn: async () => {
      const r = await fetch("/api/hosts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host_id: form.host_id,
          ssh_target: form.ssh_target,
          roles: form.roles.split(",").map((s) => s.trim()).filter(Boolean),
          notes: form.notes || null,
        }),
      });
      if (!r.ok) throw await failure(r);
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["hosts"] }),
  });

  const setRoles = useMutation({
    mutationFn: async ({ id, roles }: { id: string; roles: string[] }) => {
      const r = await fetch(`/api/hosts/${encodeURIComponent(id)}/roles`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roles }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["hosts"] }),
  });

  const toggle = useMutation({
    mutationFn: async ({ id, enabled }: { id: string; enabled: boolean }) => {
      const r = await fetch(`/api/hosts/${id}/enabled?enabled=${enabled}`, { method: "POST" });
      if (!r.ok) throw new Error(String(r.status));
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["hosts"] }),
  });

  if (isLoading) return <Empty>loading hosts…</Empty>;
  if (error)
    return (
      <Card title="Command plane unavailable">
        <div className="body-pad">
          <p>{String(error)}</p>
        </div>
      </Card>
    );

  return (
    <>
      <Card title="Hosts" note="a host is a place work can run">
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l">Host</th>
                <th className="l">SSH target</th>
                <th className="l">Roles</th>
                <th className="l grow">GPUs</th>
                <th>Cores</th>
                <th className="unit">RAM free / total</th>
                <th>Disk free</th>
                <th className="l">Last seen</th>
                <th className="l">State</th>
                <th className="l"></th>
              </tr>
            </thead>
            <tbody>
              {data?.hosts.map((h) => {
                const st = h.last_state;
                const gpus = st?.gpus ?? [];
                return (
                  <tr key={h.host_id}>
                    <td className="scrollid">{h.host_id}</td>
                    <td className="l">{h.ssh_target}</td>
                    <td className="l wrap">
                      <div className="rolepick">
                        {ROLES.map((role) => {
                          const on = h.roles.includes(role);
                          return (
                            <button
                              key={role}
                              className={on ? "roletag is-on" : "roletag"}
                              title={ROLE_MEANING[role]}
                              disabled={setRoles.isPending}
                              onClick={() => {
                                // Only the assignable ones. Sending the row's
                                // full list would put roles this page cannot
                                // assign into the body, and the server would
                                // refuse the whole request.
                                const mine = h.roles.filter(
                                  (r) => ROLES.includes(r as typeof ROLES[number]));
                                setRoles.mutate({
                                  id: h.host_id,
                                  roles: on ? mine.filter((r) => r !== role)
                                            : [...mine, role],
                                });
                              }}
                            >
                              {role}
                            </button>
                          );
                        })}
                      </div>
                      {/* Shown rather than hidden: the row would otherwise lose a
                          role on save and look like the page did it. */}
                      {h.roles.filter((r) => !ROLES.includes(r as typeof ROLES[number]))
                        .map((r) => (
                          <span key={r} className="dash" title="not assignable here — it says where infrastructure lives">
                            {" "}{r}
                          </span>
                        ))}
                    </td>
                    {/* One per line: eight on a host is normal and a single
                        run of them wraps into an unreadable block. */}
                    <td className="l grow">
                      {gpus.length ? (
                        <ul className="gpulist">
                          {gpus.map((g) => (
                            <li key={g.uuid ?? g.index} title={g.uuid ?? undefined}>
                              <span className="gpuindex">{g.index}</span>
                              <span className="gpuname">{g.name ?? "GPU"}</span>
                              <span className="gpubar" aria-hidden="true">
                                <i style={{ width: `${Math.min(100, (g.used_mb / g.total_mb) * 100)}%` }} />
                              </span>
                              <span className="gpunums">
                                {g.util_pct}% · {(g.used_mb / 1024).toFixed(1)}/{(g.total_mb / 1024).toFixed(0)} GiB
                              </span>
                            </li>
                          ))}
                        </ul>
                      ) : <span className="dash">not reported</span>}
                    </td>
                    <td title={st?.cores_total ? `${st.cores_total} on the machine; this worker is confined to ${st.cores}` : undefined}>
                      {st?.cores ?? <span className="dash">—</span>}
                      {st?.cores_total && st.cores_total !== st.cores ? <span className="dash">/{st.cores_total}</span> : null}
                    </td>
                    <td title="free is MemAvailable: what a new process could get, reclaimable cache included">
                      {st?.ram_total_gb
                        ? <>{st.ram_free_gb ?? "?"}<span className="dash"> / {st.ram_total_gb} GB</span></>
                        : <span className="dash">—</span>}
                    </td>
                    <td title={st?.disk_path ? `free on ${st.disk_path}, the volume runs land on` : undefined}>
                      {st?.disk_free_gb ? `${st.disk_free_gb} GB` : <span className="dash">—</span>}
                    </td>
                    <td className="l">
                      {h.state_source
                        ? <span title={h.state_source}>live</span>
                        : h.last_seen_at
                          ? h.last_seen_at.slice(0, 19).replace("T", " ")
                          : <span className="dash">never</span>}
                    </td>
                    <td className="l">
                      {h.enabled ? <Pill kind="ok">enabled</Pill> : <Pill kind="neg">disabled</Pill>}
                    </td>
                    <td className="l">
                      <button onClick={() => toggle.mutate({ id: h.host_id, enabled: !h.enabled })}>
                        {h.enabled ? "disable" : "enable"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!data?.hosts.length && <Empty>no host registered yet</Empty>}
      </Card>

      <Card title="Register a host">
        <div className="body-pad">
          <div className="formgrid">
            <label>
              Host id
              <input value={form.host_id} onChange={(e) => setForm({ ...form, host_id: e.target.value })} placeholder="gpu-2" />
            </label>
            <label>
              SSH target
              <input value={form.ssh_target} onChange={(e) => setForm({ ...form, ssh_target: e.target.value })} placeholder="gpu-2" />
            </label>
            <label>
              Roles
              <input value={form.roles} onChange={(e) => setForm({ ...form, roles: e.target.value })} placeholder="ink, segment" />
            </label>
            <label className="wide">
              Notes
              <input value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} placeholder="4x A100" />
            </label>
          </div>
          <div className="controls">
            <button onClick={() => register.mutate()} disabled={!form.host_id || !form.ssh_target || register.isPending}>
              Register
            </button>
            {register.isError && <Pill kind="crit">{String(register.error)}</Pill>}
          </div>
          <p className="hint">
            Registering a host records it. Work only reaches it once an ink worker runs there and
            claims from the same queue; the panel never reaches into a host by itself.
          </p>
        </div>
      </Card>
    </>
  );
}
