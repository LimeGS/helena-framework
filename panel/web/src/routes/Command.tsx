import { memo, useMemo, useState } from "react";
import { Link } from "react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { failure, useHosts, useJobs, useLanes, useScrolls, type Job } from "../api";
import { Card, Empty, Pill } from "../components/Bits";
import { useMission, useSubject } from "../mission";

const STATE_KIND: Record<string, "ok" | "run" | "crit" | "warn" | "neg"> = {
  succeeded: "ok",
  running: "run",
  leased: "run",
  failed: "crit",
  cancelled: "neg",
  pending: "warn",
};

const JobRow = memo(function JobRow({ j, onCancel }: { j: Job; onCancel: (id: string) => void }) {
  const result = j.result ?? {};
  return (
    <tr>
      <td className="l wrap">{j.job_id}</td>
      <td className="l">{j.sample_id}</td>
      <td className="l wrap">{j.profile_id}</td>
      <td className="l">
        <Pill kind={STATE_KIND[j.state] ?? "neg"}>{j.state}</Pill>
      </td>
      <td>
        {j.attempts}/{j.max_attempts}
      </td>
      <td className="l">{j.requested_host ?? <span className="dash">any</span>}</td>
      <td className="l">{j.worker_id ?? <span className="dash">—</span>}</td>
      <td className="l wrap">
        {result.liveness?.verdict && <Pill kind={result.liveness.verdict === "ALIVE" ? "ok" : "crit"}>{result.liveness.verdict}</Pill>}
        {result.error && <span className="dash">{String(result.error).slice(0, 60)}</span>}
        {result.exit_code !== undefined && result.exit_code !== 0 && !result.error && (
          <span className="dash">exit {result.exit_code}</span>
        )}
        {result.output_dir && (
          <>
            {" "}
            <Link to={`/runs`}>output</Link>
          </>
        )}
      </td>
      <td className="l">
        {j.state === "pending" && <button onClick={() => onCancel(j.job_id)}>cancel</button>}
      </td>
    </tr>
  );
});

export default function Command() {
  const client = useQueryClient();
  const { missionId } = useMission();
  const { subject } = useSubject();
  const { data: jobs, error: jobsError } = useJobs();
  const { data: lanes } = useLanes();
  const { data: hosts } = useHosts();
  const { data: scrollIndex } = useScrolls();

  const routable = useMemo(
    () => (lanes?.profiles ?? []).filter((p) => !p.disqualified && p.input_contract?.model_type),
    [lanes],
  );

  const [form, setForm] = useState({
    sample_id: "",
    profile_id: "",
    tiff_dir: "",
    checkpoint: "",
    upstream_dir: "",
    source_pixel_um: "",
    device: "cuda:0",
    batch_size: "",
    stride: "",
    depth_center: "",
    min_valid_ratio: "",
    on_degenerate: "fail",
    requested_host: "",
    priority: "0",
  });
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));
  const sample = missionId ? (subject ?? "") : form.sample_id;

  const profile = routable.find((p) => p.profile_id === form.profile_id);
  const contract = (profile?.input_contract ?? {}) as Record<string, any>;

  const enqueue = useMutation({
    mutationFn: async () => {
      const numeric = (v: string) => (v.trim() === "" ? undefined : Number(v));
      const parameters: Record<string, unknown> = {
        tiff_dir: form.tiff_dir,
        checkpoint: form.checkpoint,
        upstream_dir: form.upstream_dir,
        source_pixel_um: Number(form.source_pixel_um),
        device: form.device || undefined,
        batch_size: numeric(form.batch_size),
        stride: numeric(form.stride),
        depth_center: numeric(form.depth_center),
        min_valid_ratio: numeric(form.min_valid_ratio),
        on_degenerate: form.on_degenerate,
      };
      for (const k of Object.keys(parameters)) if (parameters[k] === undefined) delete parameters[k];
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_id: sample,
          phase: "P5",
          mission_id: missionId,
          profile_id: form.profile_id,
          parameters,
          priority: Number(form.priority) || 0,
          requested_host: form.requested_host || null,
        }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`);
      return body as { job_id: string };
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const cancel = useMutation({
    mutationFn: async (jobId: string) => {
      const r = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      if (!r.ok) throw await failure(r);
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });

  if (jobsError)
    return (
      <Card title="Command plane unavailable">
        <div className="body-pad">
          <p>{String(jobsError)}</p>
          <p>
            The queue lives in the fleet database. Set <code>CX_DB</code> and{" "}
            <code>POST /api/jobs/init</code> to create the tables.
          </p>
        </div>
      </Card>
    );

  return (
    <>
      <Card title="Queue a run" note="the worker builds the command from the profile">
        <div className="body-pad">
          <div className="formgrid">
            <label>
              Scroll
              <input
                list="scroll-ids"
                value={sample}
                onChange={set("sample_id")}
                disabled={Boolean(missionId)}
                placeholder={missionId ? "select a scroll in P0" : "PHerc0139"}
              />
              <datalist id="scroll-ids">
                {scrollIndex?.scrolls.map((s) => (
                  <option key={s.sample_id} value={s.sample_id} />
                ))}
              </datalist>
            </label>
            <label>
              Profile
              <select value={form.profile_id} onChange={set("profile_id")}>
                <option value="">— pick a lane —</option>
                {routable.map((p) => (
                  <option key={p.profile_id} value={p.profile_id}>
                    {p.profile_id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Host
              <select value={form.requested_host} onChange={set("requested_host")}>
                <option value="">any host</option>
                {hosts?.hosts.map((h) => (
                  <option key={h.host_id} value={h.host_id} disabled={!h.enabled}>
                    {h.host_id}
                    {h.enabled ? "" : " (disabled)"}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Source µm
              <input value={form.source_pixel_um} onChange={set("source_pixel_um")} placeholder="2.399" />
            </label>
            <label className="wide">
              TIFF directory
              <input value={form.tiff_dir} onChange={set("tiff_dir")} placeholder="/ssd/campaignx/PHerc0139-2399/layers" />
            </label>
            <label className="wide">
              Checkpoint
              <input value={form.checkpoint} onChange={set("checkpoint")} placeholder="/ssd/campaignx/canonical/r152.ckpt" />
            </label>
            <label className="wide">
              Upstream module directory
              <input value={form.upstream_dir} onChange={set("upstream_dir")} placeholder="/ssd/campaignx/canonical" />
            </label>
            <label>
              Device
              <input value={form.device} onChange={set("device")} />
            </label>
            <label>
              Batch size
              <input value={form.batch_size} onChange={set("batch_size")} placeholder="profile default" />
            </label>
            <label>
              Stride
              <input value={form.stride} onChange={set("stride")} placeholder="profile default" />
            </label>
            <label>
              Depth centre
              <input value={form.depth_center} onChange={set("depth_center")} placeholder="middle" />
            </label>
            <label>
              Min valid ratio
              <input value={form.min_valid_ratio} onChange={set("min_valid_ratio")} placeholder="profile default" />
            </label>
            <label>
              On degenerate
              <select value={form.on_degenerate} onChange={set("on_degenerate")}>
                <option value="fail">fail (refuse the map)</option>
                <option value="warn">warn (keep going)</option>
              </select>
            </label>
            <label>
              Priority
              <input value={form.priority} onChange={set("priority")} />
            </label>
          </div>

          {profile && (
            <div className="resolved">
              <h4>Resolved contract</h4>
              <table>
                <tbody>
                  {[
                    ["model_type", contract.model_type, "profile"],
                    ["frames", contract.frames, "profile"],
                    ["tile", contract.tile_size_y_x?.[0], "profile"],
                    ["training_pixel_um", contract.training_pixel_um, "profile"],
                    ["max_clip_value", contract.max_clip_value, "profile"],
                    ["source_pixel_um", form.source_pixel_um || "—", "this form"],
                    ["stride", form.stride || "—", form.stride ? "override" : "profile"],
                    ["batch_size", form.batch_size || "—", form.batch_size ? "override" : "profile"],
                  ].map(([k, v, from]) => (
                    <tr key={String(k)}>
                      <td className="l">
                        <code>{String(k)}</code>
                      </td>
                      <td className="l">
                        <code>{String(v ?? "—")}</code>
                      </td>
                      <td className="l">
                        <span className="dash">← {String(from)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                Every value above comes from the profile unless the row says otherwise. Showing it
                here is the same provenance the receipt records afterwards, moved to before the run.
              </p>
            </div>
          )}

          <div className="controls">
            <button
              onClick={() => enqueue.mutate()}
              disabled={
                enqueue.isPending ||
                !sample ||
                !form.profile_id ||
                !form.tiff_dir ||
                !form.checkpoint ||
                !form.upstream_dir ||
                !form.source_pixel_um
              }
            >
              {enqueue.isPending ? "queueing…" : "Queue run"}
            </button>
            {enqueue.isError && <Pill kind="crit">{String(enqueue.error)}</Pill>}
            {enqueue.isSuccess && <Pill kind="ok">queued {enqueue.data.job_id}</Pill>}
          </div>
          <p className="hint">
            A device index inside a container is not the host index: a worker started with{" "}
            <code>--gpus device=1</code> sees that card as <code>cuda:0</code>.
          </p>
        </div>
      </Card>

      <Card title="Queue" note={`${jobs?.length ?? 0} jobs`}>
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l">Job</th>
                <th className="l">Scroll</th>
                <th className="l">Profile</th>
                <th className="l">State</th>
                <th>Try</th>
                <th className="l">Host</th>
                <th className="l">Worker</th>
                <th className="l">Result</th>
                <th className="l"></th>
              </tr>
            </thead>
            <tbody>
              {jobs?.map((j) => (
                <JobRow key={j.job_id} j={j} onCancel={(id) => cancel.mutate(id)} />
              ))}
            </tbody>
          </table>
        </div>
        {!jobs?.length && <Empty>nothing queued yet</Empty>}
      </Card>
    </>
  );
}
