import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Pill } from "./Bits";
import { useState } from "react";

/**
 * Run one phase and show what it produced.
 *
 * P2 and P3 were reachable only from a shell on the host: the phases worked and
 * nobody but their author could run them, which is the same as not having them.
 * They are synchronous -- minutes over a bounded backlog, not a queue -- so the
 * button waits and reports, rather than handing back a job id to go look up.
 */
export function RunPhase({ endpoint, label, invalidate, sample, mission, override,
                           disabled = false, disabledReason }: {
  endpoint: string; label: string; invalidate: string; sample?: string;
  mission?: string | null;
  disabled?: boolean; disabledReason?: string;
  /** A named escape hatch this phase's policy allows, drawn as a toggle. */
  override?: { name: string; label: string; note: string };
}) {
  const client = useQueryClient();
  const [limit, setLimit] = useState("");
  const [dry, setDry] = useState(true);
  const [relaxed, setRelaxed] = useState(false);

  const run = useMutation({
    mutationFn: async () => {
      const r = await fetch(endpoint, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dry_run: dry,
          ...(limit.trim() ? { limit: Number(limit) } : {}),
          ...(sample ? { sample_id: sample } : {}),
          // The mission travels, so this phase's jobs belong to a campaign
          // rather than to nobody.
          ...(mission ? { mission_id: mission } : {}),
          ...(override && relaxed ? { [override.name]: true } : {}),
        }),
      });
      const body = await r.json();
      if (!r.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail
          : body.detail?.stderr_tail?.trim().split("\n").slice(-1)[0]
            ?? `HTTP ${r.status}`);
      }
      return body as Record<string, unknown>;
    },
    onSuccess: () => client.invalidateQueries({ queryKey: [invalidate] }),
  });

  const outcome = run.data as any;
  return (
    <div className="controls">
      <label className="inline">
        how many
        <input value={limit} onChange={(e) => setLimit(e.target.value)}
               placeholder="default" size={6} />
      </label>
      <label className="inline">
        <input type="checkbox" checked={dry}
               onChange={(e) => setDry(e.target.checked)} />
        {/* On by default: it lists what would be done and touches nothing, so
            the first click cannot start half an hour of work by accident. */}
        list only, do not run
      </label>
      {override && (
        <label className="inline" title={override.note}>
          <input type="checkbox" checked={relaxed}
                 onChange={(e) => setRelaxed(e.target.checked)} />
          {override.label}
        </label>
      )}
      <button onClick={() => run.mutate()} disabled={run.isPending || disabled}>
        {run.isPending ? "running…" : label}
      </button>
      {disabled && disabledReason && <span className="dash">{disabledReason}</span>}
      {run.isError && <Pill kind="crit">{String(run.error)}</Pill>}
      {outcome && (
        <span className="dash">
          considered {String(outcome.considered ?? "?")}
          {outcome.dry_run ? " (listed only)" : ""}
          {outcome.certified ? ` · ${JSON.stringify(outcome.certified)}` : ""}
          {outcome.flattened ? ` · ${JSON.stringify(outcome.flattened)}` : ""}
        </span>
      )}
    </div>
  );
}
