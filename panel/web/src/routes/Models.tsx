import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, Empty, Pill, queryGate } from "../components/Bits";

/**
 * The weights the platform needs, and getting them onto the machine.
 *
 * A profile is authoritative about a checkpoint's identity: it names a SHA-256
 * and treats the path as runtime input. So this page is not a catalogue of
 * models somebody thinks are good — it is the list of hashes the frozen
 * profiles ask for, each shown as installed or missing, with the repository
 * that publishes exactly those bytes.
 *
 * That derivation is the reason there is no hardcoded list. A typed catalogue
 * is a guess about somebody else's naming, and it ships as a button that 404s
 * the day they rename something.
 */

type HuggingFace = {
  state: "exact" | "mismatch" | "pickle_only" | "no_safetensors" | "gated"
       | "not_published" | "unreachable" | "no_family";
  why: string;
  repo?: string; revision?: string; file?: string; bytes?: number;
  safetensors?: string[]; files?: string[];
};

type Checkpoint = {
  checkpoint_sha256: string;
  model_family: string | null;
  declared_by: string[];
  installed: boolean;
  installed_at: string | null;
  hugging_face?: HuggingFace;
};

type Payload = {
  root: string;
  writable: boolean;
  checkpoints: Checkpoint[];
  note: string;
};

const size = (b?: number) =>
  b == null ? "" : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`;

// What the operator can do about each state, which is the only thing that makes
// a state worth showing.
const STATE: Record<HuggingFace["state"],
                    { kind: "ok" | "warn" | "none"; label: string }> = {
  exact: { kind: "ok", label: "available" },
  mismatch: { kind: "warn", label: "re-uploaded" },
  pickle_only: { kind: "warn", label: "pickle only" },
  no_safetensors: { kind: "warn", label: "no safetensors" },
  gated: { kind: "none", label: "not reachable" },
  not_published: { kind: "none", label: "not published" },
  unreachable: { kind: "none", label: "not reachable" },
  no_family: { kind: "none", label: "no repository" },
};

function Download({ row }: { row: Checkpoint }) {
  const client = useQueryClient();
  const hf = row.hugging_face!;
  const get = useMutation({
    mutationFn: async () => {
      const r = await fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo: hf.repo, file: hf.file, revision: hf.revision,
          name: row.model_family, expect_sha256: row.checkpoint_sha256,
        }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail?.detail ?? body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["models"] }),
  });

  return (
    <>
      <button className="primary" disabled={get.isPending} onClick={() => get.mutate()}>
        {get.isPending ? `fetching ${size(hf.bytes)}…` : `Download ${size(hf.bytes)}`}
      </button>
      {get.isError && <p className="formerror">{String(get.error)}</p>}
    </>
  );
}

function Arbitrary() {
  const client = useQueryClient();
  const [form, setForm] = useState({ repo: "", file: "model.safetensors", revision: "main" });
  const get = useMutation({
    mutationFn: async () => {
      const r = await fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo: form.repo.trim(), file: form.file.trim(),
          revision: form.revision.trim() || "main",
        }),
      });
      const body = await r.json();
      if (!r.ok) {
        const d = body.detail;
        const offered = d?.safetensors_it_does_have;
        throw new Error(
          (d?.detail ?? d ?? `HTTP ${r.status}`) +
          (offered?.length ? ` — it does have: ${offered.join(", ")}` : ""));
      }
      return body;
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["models"] }),
  });

  return (
    <Card title="Any Hugging Face repository">
      <p className="hint">
        A model that no profile names yet. It is fetched and hashed; nothing will
        use it until a profile declares that hash, which is how a new model
        arrives before there is a lane for it.
      </p>
      <div className="row">
        <label>
          Repository
          <input value={form.repo} placeholder="owner/name"
                 onChange={(e) => setForm({ ...form, repo: e.target.value })} />
        </label>
        <label>
          File
          <input value={form.file}
                 onChange={(e) => setForm({ ...form, file: e.target.value })} />
        </label>
        <label>
          Revision
          <input value={form.revision} placeholder="main, a tag, or a commit"
                 onChange={(e) => setForm({ ...form, revision: e.target.value })} />
        </label>
      </div>
      <button className="primary" disabled={!form.repo.includes("/") || get.isPending}
              onClick={() => get.mutate()}>
        {get.isPending ? "fetching…" : "Download"}
      </button>
      {get.isError && <p className="formerror">{String(get.error)}</p>}
      {get.isSuccess && (
        <p className="hint">
          {get.data.path} — {size(get.data.bytes)}, {get.data.checkpoint_sha256.slice(0, 16)}…
          {" "}{get.data.recognised
            ? `satisfies ${get.data.satisfies.join(", ")}`
            : "no profile declares this hash yet"}
        </p>
      )}
    </Card>
  );
}

export default function Models() {
  // Resolving asks Hugging Face about every missing checkpoint, so it is a
  // deliberate click rather than something that happens on every page view.
  const [resolve, setResolve] = useState(false);
  const q = useQuery<Payload>({
    queryKey: ["models", resolve],
    queryFn: async () => {
      const r = await fetch(`/api/models${resolve ? "?resolve=1" : ""}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    },
  });
  const gate = queryGate(q, "reading the profiles…");
  if (gate) return gate;
  const { root, writable, checkpoints } = q.data!;
  const missing = checkpoints.filter((c) => !c.installed).length;

  return (
    <>
      <Card title="Checkpoints">
        <p className="hint">
          {checkpoints.length - missing} of {checkpoints.length} installed in{" "}
          <code>{root}</code>
          {!writable && " — which the panel cannot write to, so nothing can be " +
            "downloaded until the models volume is mounted read-write here"}.
          {" "}Every row is a hash a frozen profile names.
        </p>
        {!resolve && missing > 0 && (
          <button onClick={() => setResolve(true)}>
            Look up the {missing} missing on Hugging Face
          </button>
        )}
        {/* Fixed layout, because two of these columns hold sentences. Auto
            layout sizes a column to its content: unconstrained it widened the
            table off the screen, and constrained with overflow-wrap it decided
            the minimum content width was one character and collapsed to a
            column of stacked letters. Neither is a layout; declaring the widths
            is. */}
        <table className="grid models">
          <colgroup>
            <col style={{ width: "26%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "24%" }} />
            <col style={{ width: "28%" }} />
            <col style={{ width: "10%" }} />
          </colgroup>
          <thead>
            <tr>
              <th>Model</th><th>State</th><th>Declared by</th><th>Source</th><th />
            </tr>
          </thead>
          <tbody>
            {checkpoints.map((row) => {
              const hf = row.hugging_face;
              return (
                <tr key={row.checkpoint_sha256}>
                  <td>
                    <code>{row.model_family ?? "(no family named)"}</code>
                    <div className="hint mono">{row.checkpoint_sha256.slice(0, 24)}…</div>
                  </td>
                  <td>
                    {row.installed
                      ? <Pill kind="ok">installed</Pill>
                      : hf
                        ? <Pill kind={STATE[hf.state].kind}>{STATE[hf.state].label}</Pill>
                        : <Pill>missing</Pill>}
                  </td>
                  {/* One per line. Comma-joined they read as one long string
                      with no natural break point. */}
                  <td className="hint">
                    {row.declared_by.map((name) => (
                      <div key={name} className="mono">{name}</div>
                    ))}
                  </td>
                  <td className="hint">
                    {row.installed
                      ? <span className="mono">{row.installed_at}</span>
                      : hf
                        ? <>
                            {hf.repo && <div className="mono">{hf.repo}</div>}
                            <div>{hf.why}</div>
                          </>
                        : "—"}
                  </td>
                  <td>
                    {!row.installed && hf?.state === "exact" && writable &&
                      <Download row={row} />}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!checkpoints.length && <Empty>no profile declares a checkpoint</Empty>}
      </Card>
      <Arbitrary />
      <Card title="Why only safetensors">
        <p className="hint">
          A <code>.bin</code>, <code>.pt</code> or <code>.ckpt</code> checkpoint is
          a Python pickle: loading one executes whatever was serialised into it,
          and these are loaded on GPU workers. Safetensors cannot carry code, so
          it is the only format the panel will fetch. A model published only as a
          pickle has to be converted and installed by hand, which is a decision
          somebody makes deliberately rather than by clicking.
        </p>
      </Card>
    </>
  );
}
