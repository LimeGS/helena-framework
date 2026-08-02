import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, Empty, Pill } from "../components/Bits";

/**
 * Everything that changed something, and who changed it.
 *
 * Reads are not here. A trail that records every page load is a trail nobody
 * searches, and this platform's questions are all about mutations: who queued
 * that render, who removed that scroll from the mission, who added an account,
 * who was refused.
 *
 * Refusals are included on purpose. "Nobody did that" and "somebody tried and
 * was told no" are different answers, and only the second one is interesting.
 */

type Entry = {
  id: string;
  at: string;
  user: string;
  action: string;
  method: string;
  path: string;
  query: string | null;
  status: number;
  ms: number;
  client: string | null;
};

const verdict = (status: number): "ok" | "warn" | "crit" | "none" => {
  if (status < 300) return "ok";
  if (status === 401 || status === 403) return "crit";
  if (status < 500) return "warn";
  return "crit";
};

export default function Audit() {
  const [user, setUser] = useState("");
  const [contains, setContains] = useState("");
  const [limit, setLimit] = useState(200);

  const params = new URLSearchParams({ limit: String(limit) });
  if (user) params.set("user", user);
  if (contains) params.set("contains", contains);

  const trail = useQuery({
    queryKey: ["audit", user, contains, limit],
    queryFn: async () => {
      const r = await fetch(`/api/audit?${params}`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as {
        entries: Entry[];
        count: number;
        limit: number;
        months: string[];
        users: string[];
        root: string;
        captures: string;
      };
    },
    refetchInterval: 30_000,
  });

  const data = trail.data;

  return (
    <>
      <Card title="Audit log" note={data ? `${data.count} entries` : ""}>
        <div className="body-pad">
          <p>
            Every request that could change something, newest first — including the ones
            that were refused, because an attempt that failed is usually the half worth
            looking for. Reads are not recorded: a log of who opened a page is a log
            nobody searches.
          </p>
          <p className="dim">
            {data?.captures ??
              "timestamp, id, user, action, status, duration and client address."}
            {data?.root && <> Kept at <code>{data.root}</code>, one file per month, and
              carried by the control-plane backup.</>}
          </p>
        </div>

        <div className="controls body-pad">
          <label>
            user{" "}
            <select value={user} onChange={(e) => setUser(e.target.value)}>
              <option value="">anyone</option>
              {(data?.users ?? []).map((u) => (
                <option key={u} value={u}>{u}</option>
              ))}
            </select>
          </label>
          <label>
            action contains{" "}
            <input value={contains} placeholder="/api/jobs"
                   onChange={(e) => setContains(e.target.value)} />
          </label>
          <label>
            showing{" "}
            <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
              {[100, 200, 500, 2000].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
          <button onClick={() => trail.refetch()} disabled={trail.isFetching}>
            {trail.isFetching ? "reading…" : "refresh"}
          </button>
        </div>

        {data?.entries.length ? (
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l">When (UTC)</th>
                  <th className="l">Id</th>
                  <th className="l">User</th>
                  <th className="l grow">Action</th>
                  <th className="l">Outcome</th>
                  <th className="r">ms</th>
                  <th className="l">From</th>
                </tr>
              </thead>
              <tbody>
                {data.entries.map((e) => (
                  <tr key={e.id}>
                    <td className="l">{e.at.replace("T", " ").replace("Z", "")}</td>
                    <td className="l scrollid">{e.id}</td>
                    <td className="l">{e.user}</td>
                    <td className="l grow">
                      <code>{e.action}</code>
                      {e.query && <span className="dim"> ?{e.query}</span>}
                    </td>
                    <td className="l">
                      <Pill kind={verdict(e.status)}>{e.status}</Pill>
                    </td>
                    <td className="r">{e.ms}</td>
                    <td className="l">{e.client ?? <span className="dash">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : trail.isLoading ? (
          <Empty>reading the trail…</Empty>
        ) : (
          <Empty>
            {user || contains
              ? "nothing matches that filter"
              : "nothing has changed anything yet"}
          </Empty>
        )}

        {data && data.count >= data.limit && (
          <div className="body-pad dim">
            Showing the newest {data.limit}. Raise the limit or narrow the filter to see
            further back{data.months.length > 1 && <> — the trail holds {data.months.length} months</>}.
          </div>
        )}
      </Card>
    </>
  );
}
