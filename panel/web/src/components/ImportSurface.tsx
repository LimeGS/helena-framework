import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useScrolls } from "../api";
import { Info } from "./Bits";

/**
 * Bringing a surface in from a laptop.
 *
 * The catalogue could already be added to, but only by a machine: one route
 * takes a URI and a digest for bytes somebody has already published and hashed
 * by hand, and the other takes a gzipped tar from a worker holding a machine
 * token. A person with a VC3D patch folder had neither, so in practice the
 * inbound half of the catalogue was closed to them.
 *
 * What this does NOT do is as much of the point as what it does. It asks for
 * the scroll and nothing else. Bounds and area are measured from the bytes on
 * arrival with the finalizer's own function -- so an uploaded surface and a
 * grown one are comparable, which is the only reason a bbox in that table is
 * worth anything. A form that asked for them would be asking somebody to
 * restate a measurement, and the restatement is what would have been recorded.
 *
 * The surface lands as IMPORTED. This fleet did not grow it and must not count
 * it as its output, and every origin split on the segments view depends on
 * that staying true however the surface arrived.
 */

type Phase = "idle" | "opening" | "sending" | "committing" | "done";

async function refuse(response: Response): Promise<never> {
  // The panel answers with {detail: ...} where detail is sometimes a string and
  // sometimes an object. Rendering [object Object] at somebody who is trying to
  // work out why their upload failed is worse than the status code alone.
  let body: unknown = null;
  try { body = await response.json(); } catch { /* not JSON; the status is what there is */ }
  const detail = (body as { detail?: unknown } | null)?.detail;
  throw new Error(
    typeof detail === "string" ? detail
      : detail ? JSON.stringify(detail)
        : `the panel refused this with ${response.status}`);
}

export function ImportSurface({ sample, missionId, onDone }: {
  sample: string | null; missionId: string | null; onDone: () => void;
}) {
  const client = useQueryClient();
  const scrolls = useScrolls();
  const picker = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [scroll, setScroll] = useState(sample ?? "");
  const [owner, setOwner] = useState("uploaded");
  const [phase, setPhase] = useState<Phase>("idle");
  const [sent, setSent] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ surface_id?: string; inserted: number } | null>(null);

  const busy = phase !== "idle" && phase !== "done";

  async function run() {
    setError(null);
    setResult(null);
    setSent(0);
    setPhase("opening");

    let uploadId: string | null = null;
    try {
      const opened = await fetch("/api/segmentation/uploads", { method: "POST" });
      if (!opened.ok) await refuse(opened);
      const { upload_id, required, optional } = await opened.json() as {
        upload_id: string; required: string[]; optional: string[];
      };
      uploadId = upload_id;

      // Which files to send, and whether this is a surface at all, are decided
      // against the set the server just named rather than a list held here.
      // Two lists would eventually disagree about what a surface is made of.
      const known = new Set([...required, ...optional]);
      const carried = files.filter((f) => known.has(f.name));
      const missing = required.filter((name) => !carried.some((f) => f.name === name));
      if (missing.length) {
        throw new Error(
          `this is not yet a surface: ${missing.join(", ")} ` +
          `${missing.length === 1 ? "is" : "are"} missing`);
      }

      setPhase("sending");
      for (const file of carried) {
        const put = await fetch(
          `/api/segmentation/uploads/${uploadId}/${encodeURIComponent(file.name)}`,
          { method: "PUT", body: file });
        if (!put.ok) await refuse(put);
        setSent((n) => n + 1);
      }

      setPhase("committing");
      const committed = await fetch(`/api/segmentation/uploads/${uploadId}/commit`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: scroll, owner, mission_id: missionId }),
      });
      if (!committed.ok) await refuse(committed);
      // Committed: the staging directory moved into the artifact volume and is
      // no longer ours to delete.
      uploadId = null;
      setResult(await committed.json());
      setPhase("done");
      setFiles([]);
      if (picker.current) picker.current.value = "";
      client.invalidateQueries({ queryKey: ["segmentation-segments"] });
      client.invalidateQueries({ queryKey: ["segmentation"] });
      onDone();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      setPhase("idle");
      // Whatever landed before the failure is bytes on the volume that is this
      // deployment's copy of record, and nothing else will ever finish this
      // upload. The server sweeps stale ones too; this is the tidy exit.
      if (uploadId) {
        await fetch(`/api/segmentation/uploads/${uploadId}`, { method: "DELETE" })
          .catch(() => { /* the sweep will get it */ });
      }
    }
  }

  const options = scrolls.data?.scrolls ?? [];

  return (
    <div className="importsurface">
      <div className="runhead">
        <label className="runhead-reads">
          <span className="runhead-label">belongs to</span>
          {options.length ? (
            <select value={scroll} onChange={(e) => setScroll(e.target.value)}
                    disabled={busy}>
              <option value="">choose a scroll…</option>
              {options.map((s) => (
                <option key={s.sample_id} value={s.sample_id}>{s.sample_id}</option>
              ))}
            </select>
          ) : (
            <input value={scroll} disabled={busy} placeholder="PHerc0332"
                   onChange={(e) => setScroll(e.target.value)} />
          )}
        </label>
        <label className="inlinecheck">
          owner
          <input value={owner} disabled={busy} maxLength={64}
                 onChange={(e) => setOwner(e.target.value)} />
        </label>
      </div>

      <div className="runsizing">
        <label className="inlinecheck">
          the surface folder
          <input ref={picker} type="file" multiple disabled={busy}
                 onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
        </label>
      </div>
      <p className="hint">
        A VC3D patch folder: <code>x.tif</code>, <code>y.tif</code>,{" "}
        <code>z.tif</code> and <code>meta.json</code>, plus{" "}
        <code>generations.tif</code> if the grow left one. Select the files
        inside the folder — anything else in the selection is left behind rather
        than uploaded.
      </p>

      {files.length > 0 && (
        <ul className="filelist">
          {files.map((f) => (
            <li key={f.name}>
              <code>{f.name}</code>
              <span className="dash"> · {(f.size / 1024).toFixed(0)} KB</span>
            </li>
          ))}
        </ul>
      )}

      <div className="controls">
        <button disabled={busy || !files.length || !scroll.trim()}
                onClick={() => { void run(); }}>
          {phase === "opening" ? "opening…"
            : phase === "sending" ? `sending ${sent + 1}…`
              : phase === "committing" ? "measuring and recording…"
                : "Import surface"}
        </button>
      </div>

      {result && (
        <p className="hint">
          Recorded as <b>IMPORTED</b>
          {result.inserted === 0
            ? " — this surface was already in the catalogue, so nothing was added."
            : ". Its bounds and area were measured from the bytes; it is"
              + " unvalidated until QC and geometry have run against it."}
        </p>
      )}
      {error && <p className="formerror">{error}</p>}

      <Info label="Why an import is never counted as ours"
            title="Origin is what the catalogue splits on">
        A surface this fleet grew carries an attempt, a lineage and a route.
        One that arrived some other way carries none of that, and counting the
        two together would report work nobody here did as this fleet's output.
        An imported surface is recorded with its authorship and its digest, and
        certification starts from it rather than behind it.
      </Info>
    </div>
  );
}
