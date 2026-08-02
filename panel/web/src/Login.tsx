import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import markColor from "./assets/brand/helena-mark-color.svg";
import markReverse from "./assets/brand/helena-mark-reverse.svg";

/**
 * The gate.
 *
 * Wraps the app rather than living on a route: an unauthenticated request for
 * any page gets the app shell back, and this decides whether that shell shows
 * the panel or this form. A redirect would lose the address somebody typed.
 *
 * The first account is a separate thing and says so. It can only be claimed
 * from the host itself, because a panel that is already reachable cannot let
 * whoever finds it first through the door.
 */

type Session = {
  username: string | null;
  required: boolean;
  any_users: boolean;
  bootstrap_available: boolean;
  bootstrap_note: string;
};

export function useSession() {
  return useQuery({
    queryKey: ["session"],
    queryFn: async () => {
      const r = await fetch("/api/session");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as Session;
    },
    // Cheap, and the answer changing is the whole point.
    staleTime: 10_000,
    retry: false,
  });
}

export function Gate({ children }: { children: React.ReactNode }) {
  const session = useSession();
  if (session.isLoading) return null;
  const d = session.data;
  if (!d?.required || d?.username) return <>{children}</>;
  return <SignIn session={d} />;
}

function SignIn({ session }: { session: Session }) {
  const client = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const first = !session.any_users;

  const submit = useMutation({
    mutationFn: async () => {
      const r = await fetch(first ? "/api/session/bootstrap" : "/api/session", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: () => {
      setPassword("");
      client.invalidateQueries();
    },
  });

  return (
    <div className="gate">
      <form className="gatecard" onSubmit={(e) => { e.preventDefault(); submit.mutate(); }}>
        <div className="gatemark">
          <img src={markReverse} alt="" className="only-dark" />
          <img src={markColor} alt="" className="only-light" />
        </div>
        <h1>HELENA<span>·</span>EXPLORATION<span>·</span>FRAMEWORK</h1>
        <p className="gatesub">
          {first ? "No accounts yet — claim the first one." : "control panel"}
        </p>

        {first && !session.bootstrap_available ? (
          <p className="gatenote">
            {session.bootstrap_note}. Run this on the panel host:
            <code className="gatecmd">
              curl -X POST localhost:8800/api/session/bootstrap \{"\n"}
              {"  "}-H 'Content-Type: application/json' \{"\n"}
              {"  "}-d '{"{"}"username":"you","password":"…"{"}"}'
            </code>
          </p>
        ) : (
          <>
            <label>
              <span>Username</span>
              <input value={username} autoFocus autoComplete="username"
                     onChange={(e) => setUsername(e.target.value)} />
            </label>
            <label>
              <span>Password</span>
              <input type="password" value={password}
                     autoComplete={first ? "new-password" : "current-password"}
                     onChange={(e) => setPassword(e.target.value)} />
            </label>
            <button type="submit"
                    disabled={submit.isPending || !username.trim() || !password}>
              {submit.isPending ? "…" : first ? "Create and sign in" : "Sign in"}
            </button>
            {first && (
              <p className="gatenote">
                At least 10 characters. Everyone who can sign in can do everything —
                there are no roles.
              </p>
            )}
          </>
        )}

        {submit.isError && <p className="gateerror">{String(submit.error).replace("Error: ", "")}</p>}
      </form>
    </div>
  );
}

/** Who is signed in, and the way out. Sits in the rail foot. */
export function SessionFoot() {
  const client = useQueryClient();
  const session = useSession();
  const out = useMutation({
    mutationFn: async () => { await fetch("/api/session", { method: "DELETE" }); },
    onSuccess: () => client.invalidateQueries(),
  });
  if (!session.data?.username) return null;
  return (
    <div className="sessionfoot">
      <span className="sessionwho" title="everyone who can sign in can do everything">
        {session.data.username}
      </span>
      <button className="tinylink" onClick={() => out.mutate()}>sign out</button>
    </div>
  );
}
