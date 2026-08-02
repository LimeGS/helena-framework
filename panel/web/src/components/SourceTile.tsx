import { useState } from "react";
import { useSearchParams } from "react-router";
import { useConfig } from "../api";

/**
 * The inventory-origin tile, with the source behind a `Change` link.
 *
 * The chosen source lives in the query string rather than in component state.
 * The tile that changes it sits in the phase header and the table that reads it
 * is a route below, so anything local would need a context to bridge them; a
 * search param bridges them for free, survives a reload, and makes the view
 * something you can send to somebody.
 *
 * It browses only. The configured source is what persists, and it is changed in
 * Configuration -- a box in a phase header should not quietly rewrite a setting
 * that every later phase reads.
 */
export function SourceTile({ value }: { value: string }) {
  const [params, setParams] = useSearchParams();
  const [open, setOpen] = useState(false);
  const browsing = params.get("source") ?? "";
  const { data: config } = useConfig();
  const configured =
    config?.environment.find((e) => e.name === "CX_SCROLL_SOURCE")?.value ?? "";
  const [draft, setDraft] = useState(browsing || configured);

  const apply = (next: string) => {
    const trimmed = next.trim();
    setParams(
      (prev) => {
        const out = new URLSearchParams(prev);
        if (trimmed && trimmed !== configured) out.set("source", trimmed);
        else out.delete("source");
        return out;
      },
      { replace: true },
    );
    setOpen(false);
  };

  return (
    <div className={`tile ${browsing ? "warn" : "steady"}`}>
      <h2>
        Inventory origin
        <button className="tinylink" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          {open ? "close" : "change"}
        </button>
      </h2>
      <p style={{ color: "var(--ink)", fontSize: 13 }}>{value}</p>
      {open && (
        <div className="sourcebox">
          <input
            className="search" type="url" value={draft} spellCheck={false}
            placeholder="https://…s3.amazonaws.com/"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && apply(draft)}
          />
          <div className="controls">
            <button disabled={!draft.trim()} onClick={() => apply(draft)}>browse</button>
            {browsing && (
              <button onClick={() => { setDraft(configured); apply(configured); }}>
                official
              </button>
            )}
          </div>
          <p className="hint">
            Browses only — <code>CX_SCROLL_SOURCE</code> in Configuration is what persists. Any
            bucket laid out as <code>&lt;scroll&gt;/volumes/…keV.zarr/</code> enumerates here.
          </p>
        </div>
      )}
    </div>
  );
}
