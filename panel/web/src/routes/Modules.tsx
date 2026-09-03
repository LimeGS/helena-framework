import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, Empty, Pill, queryGate } from "../components/Bits";

/**
 * What every phase can be done with, and what is switched on.
 *
 * The platform has three extension mechanisms and each is right for what it
 * does — a lane is a program, a profile is a model with its scale, a seeder
 * chooses a point. This page does not add a fourth. It reports all of them in
 * one shape so that "what can P1 run" has one answer, and switches them in one
 * place.
 */

type Module = {
  id: string; phase: string; kind: string; name: string;
  note: string | null; enabled: boolean; removable: boolean;
  runner?: string; adapter?: string; adoptable?: boolean; repeatable?: boolean;
  source?: { kind: string; repo_id?: string };
};

type Payload = {
  phases: { phase: string; modules: Module[] }[];
  kinds: Record<string, string>;
};

// The backend's own allowlist is job_store.INK_ADAPTERS; kept in step by
// hand because the picker has to render before that answer is fetched. A
// fifth adapter, run_ink_9um, existed here before this list did and was left
// off -- the same drift as the API's own refusal text, which said "the four
// here are what exist" after it had become five.
const ADAPTERS = [
  "framework/stages/03-ink/scripts/run_ink.py",
  "framework/stages/03-ink/scripts/run_ink_timesformer.py",
  "framework/stages/03-ink/scripts/run_ink_canonical2um.py",
  "framework/stages/03-ink/scripts/run_ink_3d_dino.py",
  "framework/stages/03-ink/scripts/run_ink_9um.py",
];

function AddFromHuggingFace() {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({
    repo_id: "", adapter: ADAPTERS[1], training_pixel_um: "7.91",
    frames: "26", revision: "", checkpoint_file: "model.safetensors",
    known_limits: "",
  });

  const add = useMutation({
    mutationFn: async () => {
      const r = await fetch("/api/modules/P5/huggingface", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_id: form.repo_id.trim(),
          adapter: form.adapter,
          training_pixel_um: Number(form.training_pixel_um),
          frames: Number(form.frames),
          revision: form.revision.trim() || null,
          checkpoint_file: form.checkpoint_file.trim(),
          known_limits: form.known_limits.trim() || null,
        }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail?.detail ?? body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: () => {
      setForm({ ...form, repo_id: "", known_limits: "" });
      client.invalidateQueries({ queryKey: ["modules"] });
    },
  });

  return (
    <>
      <div className="controls body-pad">
        <button onClick={() => setOpen((v) => !v)}>
          {open ? "cancel" : "Add a detector from Hugging Face"}
        </button>
        {add.isError && <span className="dash">{String(add.error).replace(/^Error:\s*/, "")}</span>}
        {add.isSuccess && <Pill kind="ok">registered</Pill>}
      </div>
      {open && (
        <div className="body-pad">
          <p className="dash">
            The weights are not downloaded now. What is written is a profile: which
            repository, at what physical scale, through which adapter. The worker
            fetches the checkpoint on first use, and the run records what it got.
          </p>
          <div className="formgrid">
            <label>
              Repository *
              <input value={form.repo_id} placeholder="scrollprize/timesformer_GP_scroll1"
                     onChange={(e) => setForm({ ...form, repo_id: e.target.value })} />
              <span className="dash">owner/name, as it appears on huggingface.co</span>
            </label>
            <label>
              Adapter *
              <select value={form.adapter}
                      onChange={(e) => setForm({ ...form, adapter: e.target.value })}>
                {ADAPTERS.map((a) => (
                  <option key={a} value={a}>{a.split("/").pop()}</option>
                ))}
              </select>
              <span className="dash">
                the command-line contract the model is run through; a model that
                fits none of these needs one written for it
              </span>
            </label>
            <label>
              Training µm per pixel *
              <input value={form.training_pixel_um}
                     onChange={(e) => setForm({ ...form, training_pixel_um: e.target.value })} />
              <span className="dash">
                what it was trained at. Get this wrong and the model sees a slab of
                the wrong physical thickness and still produces a map
              </span>
            </label>
            <label>
              Frames *
              <input value={form.frames}
                     onChange={(e) => setForm({ ...form, frames: e.target.value })} />
              <span className="dash">how many slices the model expects</span>
            </label>
            <label>
              Revision
              <input value={form.revision} placeholder="main"
                     onChange={(e) => setForm({ ...form, revision: e.target.value })} />
              <span className="dash">a tag or commit, so the run pins what it read</span>
            </label>
            <label>
              Checkpoint file
              <input value={form.checkpoint_file}
                     onChange={(e) => setForm({ ...form, checkpoint_file: e.target.value })} />
            </label>
            <label className="full">
              What it does not claim
              <textarea value={form.known_limits} rows={2}
                        placeholder="what it was trained on, and what it has never been tested against"
                        onChange={(e) => setForm({ ...form, known_limits: e.target.value })} />
              <span className="dash">
                left empty, the profile says it has been validated against nothing
                on this deployment — which is true, and is what a reader needs
              </span>
            </label>
          </div>
          <div className="controls">
            <button disabled={!form.repo_id.trim() || add.isPending}
                    onClick={() => add.mutate()}>
              {add.isPending ? "registering…" : "Register"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}

function PhaseModules({ phase, modules, kinds }: {
  phase: string; modules: Module[]; kinds: Record<string, string>;
}) {
  const client = useQueryClient();
  const [failure, setFailure] = useState<string | null>(null);

  const toggle = useMutation({
    mutationFn: async ({ id, enabled }: { id: string; enabled: boolean }) => {
      const r = await fetch(`/api/modules/${phase}/${encodeURIComponent(id)}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail?.detail ?? body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: () => { setFailure(null); client.invalidateQueries({ queryKey: ["modules"] }); },
    onError: (e) => setFailure(String(e).replace(/^Error:\s*/, "")),
  });

  const on = modules.filter((m) => m.enabled).length;

  return (
    <Card title={phase} note={`${on} of ${modules.length} enabled`} collapsed>
      {failure && <div className="body-pad"><p className="dash">{failure}</p></div>}
      {modules.length === 0 ? (
        <Empty>this phase has nothing to switch — it runs no program of its own</Empty>
      ) : (
        <div className="scroller">
          <table>
            <thead>
              {/* State and its switch lead the row. "What it is" is the widest
                  column by far and it pushed both off to the right, where the
                  one thing you came to read and the one thing you came to click
                  were the last things reachable -- and the first to leave the
                  card when the description ran long. */}
              <tr>
                <th className="l">State</th>
                <th className="l"></th>
                <th className="l">Module</th>
                <th className="l">Kind</th>
                <th className="l grow">What it is</th>
              </tr>
            </thead>
            <tbody>
              {modules.map((m) => (
                <tr key={m.id}>
                  <td className="l">
                    {m.enabled ? <Pill kind="ok">on</Pill> : <Pill kind="neg">off</Pill>}
                  </td>
                  <td className="l">
                    <button disabled={toggle.isPending}
                            onClick={() => toggle.mutate({ id: m.id, enabled: !m.enabled })}>
                      {m.enabled ? "switch off" : "switch on"}
                    </button>
                  </td>
                  <td className="l scrollid">
                    {m.name}
                    {m.name !== m.id && <div className="dim"><code>{m.id}</code></div>}
                  </td>
                  <td className="l">
                    <Pill>{m.kind}</Pill>
                    <div className="dim">{kinds[m.kind]}</div>
                  </td>
                  <td className="l grow">
                    {m.note ?? <span className="dash">—</span>}
                    {m.runner && <div className="dim"><code>{m.runner}</code></div>}
                    {m.adapter && <div className="dim">via <code>{m.adapter.split("/").pop()}</code></div>}
                    {m.source?.repo_id && (
                      <div className="dim">from <code>{m.source.repo_id}</code></div>
                    )}
                    {m.adoptable === false && (
                      <div><Pill kind="warn">not adoptable yet</Pill></div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {phase === "P5" && <AddFromHuggingFace />}
    </Card>
  );
}

export default function Modules() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["modules"],
    queryFn: async () => {
      const r = await fetch("/api/modules");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as Payload;
    },
    staleTime: 30_000,
  });

  const gate = queryGate({ isLoading, error, data }, "reading the modules…");
  if (gate) return gate;
  if (!data) return null;

  return (
    <>
      <Card title="Modules" note="what each phase can be done with">
        <div className="body-pad guide-prose">
          <p>
            A phase is a question, not a program. Each one below lists the ways this
            deployment can answer it, and whether each is switched on.
          </p>
          <p>
            Switching one off makes the queue refuse it and the forms stop offering
            it. Nothing is deleted: a run that used it keeps its receipt, and that
            receipt still names what produced it.
          </p>
          <p className="dim">
            There are three kinds because they are different contracts, not three
            names for one thing. A <b>lane</b> is a program the queue starts. A{" "}
            <b>profile</b> is a model with its weights and its physical scale,
            routed to an adapter. A <b>backend</b> grows a surface and a{" "}
            <b>seeder</b> chooses the point to grow it from.
          </p>
        </div>
      </Card>
      {data.phases.map(({ phase, modules }) => (
        <PhaseModules key={phase} phase={phase} modules={modules} kinds={data.kinds} />
      ))}
    </>
  );
}
