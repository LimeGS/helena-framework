import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, Empty, Info, Pill } from "./Bits";

/**
 * Scoring a surface against something that was not written by whoever made it.
 *
 * A strip is a small patch of scroll where consecutive wraps are recorded as
 * separate labelled point sets. It is not ground truth and the page says so:
 * it is derived from a segmentation, so it cannot judge the segment it came
 * from, and only the optional CT cross-check appeals to anything outside that
 * segmentation at all.
 *
 * What it does judge is the failure that matters for a mesher -- non-manifold
 * edges, open boundaries, and wraps welded into one surface.
 */

type Strip = {
  strip_id: string; path: string; bytes: number;
  scroll: string | null; segment_id: string | null; window: string | null;
  tier: string | null; wraps: number; points: number;
  pitch_um: Record<string, number | null> | null;
  problems?: string[];
  qualified: boolean;
  error?: string;
  qualification: { overall_pass: boolean | null;
                   checks: Record<string, boolean | null> } | null;
};

const CHECK_MEANING: Record<string, string> = {
  self_test: "plumbing: every wrap's own points return to their own wrap",
  wrong_side: "the one that catches mislabelled or shuffled wraps",
  null_baseline: "a scorer too loose to fail garbage would fail this",
  ct_check: "the only check that looks outside the segmentation — needs network",
};

export function Strips({ predPath: given }: { predPath?: string | null }) {
  const client = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [scored, setScored] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  // No caller passes a surface down today, which left the score button
  // permanently unreachable: `predPath` was always undefined. A path here is
  // a surface directory or the run that produced one, relative to the runs
  // root -- the same thing the Segments table's "open" link points at -- so
  // typing it in is a real fallback, not a placeholder for one.
  const [typed, setTyped] = useState("");
  const predPath = given ?? (typed.trim() || null);

  const strips = useQuery({
    queryKey: ["strips"],
    queryFn: async () => {
      const r = await fetch("/api/strips");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { strips: Strip[]; root: string; tools: string };
    },
    staleTime: 30_000,
  });

  const upload = useMutation({
    mutationFn: async (file: File) => {
      // The raw file as the body: the endpoint takes exactly one, and a
      // multipart envelope around one part is a dependency for nothing.
      const r = await fetch("/api/strips", { method: "POST", body: file });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string"
        ? body.detail : JSON.stringify(body.detail));
      return body;
    },
    onSuccess: () => { setError(null); client.invalidateQueries({ queryKey: ["strips"] }); },
    onError: (e) => setError(String(e)),
  });

  const qualify = useMutation({
    mutationFn: async ({ id, ct }: { id: string; ct: boolean }) => {
      const r = await fetch(`/api/strips/${encodeURIComponent(id)}/qualify?ct_check=${ct}`,
                            { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string"
        ? body.detail : JSON.stringify(body.detail));
      return body;
    },
    onSuccess: () => { setError(null); client.invalidateQueries({ queryKey: ["strips"] }); },
    onError: (e) => setError(String(e)),
  });

  const score = useMutation({
    mutationFn: async (id: string) => {
      const r = await fetch(`/api/strips/${encodeURIComponent(id)}/score`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pred_path: predPath, mode: "mesh" }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string"
        ? body.detail : JSON.stringify(body.detail));
      return body;
    },
    onSuccess: (body) => { setError(null); setScored(body); },
    onError: (e) => setError(String(e)),
  });

  const list = strips.data?.strips ?? [];

  return (
    <Card title={<>Score against a reference strip <Info
            label="What a reference strip is, and what it cannot judge"
            title="Reference strips">
            A strip records consecutive papyrus wraps as separate labelled point sets, taken
            from a segment's own geometry where it spirals — no annotation needed. It is a
            local reference, <b>not ground truth</b>: it is derived from a segmentation, so
            it cannot judge the segment it came from, and only the CT cross-check appeals to
            anything outside that segmentation.
            {" "}For a mesher it measures non-manifold edges, open boundaries, connected
            components and wraps fused into one surface.
          </Info></>} note={`${list.length} on disk`}>
      <div className="body-pad">
        <div className="controls">
          <input ref={fileInput} type="file" accept=".npz" style={{ display: "none" }}
                 onChange={(e) => {
                   const file = e.target.files?.[0];
                   if (file) upload.mutate(file);
                   e.target.value = "";
                 }} />
          <button disabled={upload.isPending} onClick={() => fileInput.current?.click()}>
            {upload.isPending ? "uploading…" : "Upload a strip (.npz)"}
          </button>
        </div>
        {error && <Pill kind="crit">{error.slice(0, 400)}</Pill>}
      </div>

      {list.length === 0 ? (
        <Empty>
          no strips yet — upload a strip-v0 .npz, or mint one from a segment with
          reference-strips/make_strip.py
        </Empty>
      ) : (
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l grow">Strip</th>
                <th className="l">Tier</th>
                <th>Wraps</th>
                <th>Points</th>
                <th className="unit">Pitch µm</th>
                <th className="l">Qualified</th>
                <th className="l"></th>
              </tr>
            </thead>
            <tbody>
              {list.map((s) => (
                <tr key={s.strip_id}>
                  <td className="l scrollid grow" title={s.path}>
                    {s.strip_id}
                    {s.error && <span className="dash"> · {s.error}</span>}
                  </td>
                  <td className="l">{s.tier ?? <span className="dash">—</span>}</td>
                  <td>{s.wraps || <span className="dash">—</span>}</td>
                  <td>{s.points || <span className="dash">—</span>}</td>
                  <td>{s.pitch_um?.median?.toFixed(0) ?? <span className="dash">—</span>}</td>
                  <td className="l">
                    {s.qualified
                      ? <Pill kind="ok">qualified</Pill>
                      : <Pill kind="warn">unqualified</Pill>}
                    {s.qualification && (
                      <span className="dash">
                        {" "}
                        {Object.entries(s.qualification.checks)
                          .map(([k, v]) => `${k.split("_")[0]}:${v === null ? "–" : v ? "✓" : "✗"}`)
                          .join(" ")}
                      </span>
                    )}
                  </td>
                  <td className="l">
                    <div className="controls">
                      <button disabled={qualify.isPending}
                              onClick={() => qualify.mutate({ id: s.strip_id, ct: false })}>
                        {qualify.isPending ? "…" : "qualify"}
                      </button>
                      {predPath && (
                        <button disabled={score.isPending}
                                onClick={() => score.mutate(s.strip_id)}>
                          score
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {list.some((s) => s.qualification && !s.qualified) && (
        <p className="hint">
          {Object.entries(CHECK_MEANING).map(([k, v]) => `${k}: ${v}`).join(" · ")}
        </p>
      )}

      {!given && list.length > 0 && (
        <div className="formgrid">
          <label>
            Surface path (under the runs root)
            <input value={typed} onChange={(e) => setTyped(e.target.value)}
                  placeholder="segmentation/PHerc826/…/surface.obj" />
            <span className="dash">
              the same path the Segments table's "open" link points at. A strip
              with no passing qualification still scores — the scorecard is
              stamped UNQUALIFIED and says so.
            </span>
          </label>
        </div>
      )}

      {scored && (
        <div className="body-pad">
          <h3>Scorecard</h3>
          <pre className="pre">{JSON.stringify(scored, null, 2).slice(0, 4000)}</pre>
        </div>
      )}
    </Card>
  );
}
