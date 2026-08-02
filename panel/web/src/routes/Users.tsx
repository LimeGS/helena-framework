import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, Empty, Pill } from "../components/Bits";
import { useSession } from "../Login";

/**
 * Accounts. No roles: everyone who can sign in can do everything, including
 * adding and removing accounts.
 *
 * That is stated on the page rather than left to be discovered, because a
 * permission model people assume exists is worse than one they know is absent.
 * The panel queues GPU work and accepts uploads; an account here is the whole
 * boundary.
 */

type Machine = {
  name: string;
  created_at_utc: string | null;
  created_by: string | null;
  last_used_utc: string | null;
};

type User = {
  username: string;
  created_at_utc: string | null;
  created_by: string | null;
  last_login_utc: string | null;
};

export default function Users() {
  const client = useQueryClient();
  const session = useSession();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [changing, setChanging] = useState<string | null>(null);
  const [replacement, setReplacement] = useState("");
  const [machineName, setMachineName] = useState("");
  // Shown once and never again: the panel stores a hash, exactly as it does for
  // passwords and session tokens. Kept in component state so it survives until
  // the operator has copied it, and lost on navigation, which is correct.
  const [minted, setMinted] = useState<string | null>(null);

  const listing = useQuery({
    queryKey: ["users"],
    queryFn: async () => {
      const r = await fetch("/api/users");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { users: User[]; minimum_password: number };
    },
  });

  const call = async (url: string, method: string, body?: unknown) => {
    const r = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const parsed = await r.json();
    if (!r.ok) throw new Error(parsed.detail ?? `HTTP ${r.status}`);
    return parsed;
  };

  const machines = useQuery({
    queryKey: ["machines"],
    queryFn: async () => {
      const r = await fetch("/api/machines");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { machines: Machine[] };
    },
  });

  const mintToken = useMutation({
    mutationFn: () => call("/api/machines", "POST", { name: machineName }),
    onSuccess: (result: { token: string }) => {
      setMinted(result.token);
      setMachineName("");
      client.invalidateQueries({ queryKey: ["machines"] });
    },
  });

  const revokeToken = useMutation({
    mutationFn: (name: string) => call(`/api/machines/${name}`, "DELETE"),
    onSuccess: () => client.invalidateQueries({ queryKey: ["machines"] }),
  });

  const add = useMutation({
    mutationFn: () => call("/api/users", "POST", { username, password }),
    onSuccess: () => {
      setUsername(""); setPassword("");
      client.invalidateQueries({ queryKey: ["users"] });
    },
  });

  const setPasswordFor = useMutation({
    mutationFn: (who: string) =>
      call(`/api/users/${encodeURIComponent(who)}/password`, "POST",
           { username: who, password: replacement }),
    onSuccess: () => { setChanging(null); setReplacement(""); },
  });

  const remove = useMutation({
    mutationFn: (who: string) => call(`/api/users/${encodeURIComponent(who)}`, "DELETE"),
    onSuccess: () => client.invalidateQueries({ queryKey: ["users"] }),
  });

  const minimum = listing.data?.minimum_password ?? 10;
  const me = session.data?.username;

  return (
    <>
      <Card title="Accounts" note={`${listing.data?.users.length ?? 0}`}>
        <div className="body-pad">
          <p>
            There are no roles. Everyone who can sign in can do everything, including
            adding and removing accounts — said here rather than left to be discovered,
            because a permission model people assume exists is worse than one they know
            is absent.
          </p>
          <p>
            An account is the whole boundary between the network and a panel that queues
            GPU work and accepts uploads. Passwords are stored as salted scrypt hashes and
            are never recoverable; a forgotten one is replaced, not read.
          </p>
        </div>

        {listing.data?.users.length ? (
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l grow">Username</th>
                  <th className="l">Added by</th>
                  <th className="l">Created</th>
                  <th className="l">Last sign-in</th>
                  <th className="l"></th>
                </tr>
              </thead>
              <tbody>
                {listing.data.users.map((u) => (
                  <tr key={u.username}>
                    <td className="l scrollid grow">
                      {u.username}
                      {u.username === me && <> <Pill kind="ok">you</Pill></>}
                    </td>
                    <td className="l">{u.created_by ?? <span className="dash">—</span>}</td>
                    <td className="l">
                      {u.created_at_utc?.slice(0, 16).replace("T", " ") ?? <span className="dash">—</span>}
                    </td>
                    <td className="l">
                      {u.last_login_utc?.slice(0, 16).replace("T", " ")
                        ?? <span className="dash">never</span>}
                    </td>
                    <td className="l">
                      <div className="controls">
                        <button onClick={() => {
                          setChanging(changing === u.username ? null : u.username);
                          setReplacement("");
                        }}>
                          {changing === u.username ? "cancel" : "set password"}
                        </button>
                        <button disabled={remove.isPending || listing.data.users.length === 1}
                                title={listing.data.users.length === 1
                                  ? "the only account; removing it locks everyone out"
                                  : undefined}
                                onClick={() => remove.mutate(u.username)}>
                          remove
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>no accounts</Empty>
        )}

        {changing && (
          <div className="body-pad">
            <div className="controls">
              <input className="search" type="password" value={replacement}
                     autoComplete="new-password"
                     placeholder={`new password for ${changing}, ${minimum}+ characters`}
                     onChange={(e) => setReplacement(e.target.value)} />
              <button disabled={replacement.length < minimum || setPasswordFor.isPending}
                      onClick={() => setPasswordFor.mutate(changing)}>
                set it
              </button>
            </div>
            <p className="hint">
              Their open sessions stay open — changing a password stops the old one being
              usable to sign in again, not the browser somebody already has.
            </p>
          </div>
        )}
        {remove.isError && <p className="hint"><Pill kind="crit">{String(remove.error)}</Pill></p>}
        {setPasswordFor.isError && (
          <p className="hint"><Pill kind="crit">{String(setPasswordFor.error)}</Pill></p>
        )}
      </Card>

      <Card title="Add an account">
        <div className="body-pad">
          <div className="controls">
            <input className="search" value={username} autoComplete="off"
                   placeholder="username — lowercase, digits, dot, dash"
                   onChange={(e) => setUsername(e.target.value)} />
            <input className="search" type="password" value={password}
                   autoComplete="new-password"
                   placeholder={`password, ${minimum}+ characters`}
                   onChange={(e) => setPassword(e.target.value)} />
            <button disabled={add.isPending || !username.trim() || password.length < minimum}
                    onClick={() => add.mutate()}>
              {add.isPending ? "adding…" : "Add"}
            </button>
          </div>
          {add.isError && <Pill kind="crit">{String(add.error).replace("Error: ", "")}</Pill>}
        </div>
      </Card>

      {/*
        Workers, not people. A worker on another host publishes its surfaces to
        this panel and needs a credential to do it; the alternative was copying
        somebody's password into an env file on every worker machine, which
        makes this page's audit trail a lie.

        A machine token reaches the artifact endpoints and nothing else.
      */}
      <Card title="Machine tokens">
        <div className="body-pad">
          <p className="hint">
            For workers on other hosts, which publish surfaces to this panel over
            the network. A machine token reaches the artifact endpoints and
            nothing else — it cannot queue work or read missions. Revoking one
            affects only that worker.
          </p>
          {minted && (
            <div className="controls" style={{ flexDirection: "column", alignItems: "stretch" }}>
              <Pill kind="warn">Copy this now — it is not shown again</Pill>
              <code style={{ userSelect: "all", wordBreak: "break-all", padding: "0.5rem 0" }}>
                {minted}
              </code>
              <p className="hint">
                Put it in the worker&apos;s env file as <code>HELENA_PANEL_TOKEN</code>.
              </p>
              <button onClick={() => setMinted(null)}>Done</button>
            </div>
          )}
          {(machines.data?.machines.length ?? 0) === 0 ? (
            <Empty>No machine tokens. Workers on this host do not need one.</Empty>
          ) : (
            <table className="grid">
              <thead>
                <tr><th>Name</th><th>Created</th><th>By</th><th>Last used</th><th /></tr>
              </thead>
              <tbody>
                {machines.data?.machines.map((m) => (
                  <tr key={m.name}>
                    <td>{m.name}</td>
                    <td>{m.created_at_utc?.slice(0, 10) ?? "—"}</td>
                    <td>{m.created_by ?? "—"}</td>
                    <td>{m.last_used_utc?.slice(0, 10) ?? "never"}</td>
                    <td>
                      <button disabled={revokeToken.isPending}
                              onClick={() => revokeToken.mutate(m.name)}>
                        Revoke
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="controls">
            <input className="search" value={machineName} autoComplete="off"
                   placeholder="worker name — e.g. gpu-2-segment"
                   onChange={(e) => setMachineName(e.target.value)} />
            <button disabled={mintToken.isPending || !machineName.trim()}
                    onClick={() => mintToken.mutate()}>
              {mintToken.isPending ? "minting…" : "Mint token"}
            </button>
          </div>
          {mintToken.isError && (
            <Pill kind="crit">{String(mintToken.error).replace("Error: ", "")}</Pill>
          )}
          {revokeToken.isError && (
            <Pill kind="crit">{String(revokeToken.error).replace("Error: ", "")}</Pill>
          )}
        </div>
      </Card>
    </>
  );
}
