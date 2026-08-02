import { memo, useDeferredValue, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useScrolls, type Scroll } from "../api";
import { useMission } from "../mission";
import { ProducedArtifacts } from "../components/Artifacts";
import { Card, Empty, Pill , queryGate} from "../components/Bits";
import { useResizableColumns } from "../components/Table";

/**
 * P0 is where a mission gets its scrolls.
 *
 * There is no download step, and that is worth saying plainly because the
 * interface would otherwise imply one. The source is an OME-Zarr over HTTPS and
 * every reader fetches the chunks it touches -- vc3d takes the URI, the
 * renderers do sparse chunk-gather. Nothing pulls a whole scroll. What lands on
 * disk is derived and far smaller: a surface layer stack is gigabytes where the
 * scan is hundreds.
 *
 * So this page selects; it does not fetch.
 */

type Row = Scroll & { selected: boolean };

const ScrollRow = memo(function ScrollRow({
  row, onToggle, disabled,
}: { row: Row; onToggle: (id: string) => void; disabled: boolean }) {
  return (
    <tr className={row.selected ? "" : "muted"}>
      <td className="l pick">
        <input
          type="checkbox"
          checked={row.selected}
          disabled={disabled}
          aria-label={`Include ${row.sample_id}`}
          onChange={() => onToggle(row.sample_id)}
        />
      </td>
      <td className="scrollid grow">{row.sample_id}</td>
      <td title={row.scale_from === "scan name"
                 ? `finest of ${row.scans} scans, read from the scan directory name`
                 : row.scale_from === "catalog" ? "from the frozen catalog" : undefined}>
        {row.pixel_um || <span className="dash">unknown</span>}
      </td>
      <td>{row.energy_kev ?? <span className="dash">—</span>}</td>
      <td>{row.scans || <span className="dash">—</span>}</td>
      {/* The earliest scan's date: the inventory is an S3 listing and carries no
          date of its own, so this is the one place the answer exists. */}
      <td className="l">{row.scanned_on ?? <span className="dash">—</span>}</td>
      <td>{row.runs || <span className="dash">—</span>}</td>
      <td className="l">
        {row.selected ? <Pill kind="ok">in mission</Pill> : <span className="dash">—</span>}
      </td>
    </tr>
  );
});

export default function Intake() {
  const client = useQueryClient();
  const { missionId, current } = useMission();
  // Set by the Change link on the inventory-origin tile, which lives in the
  // phase header above this route.
  const [params] = useSearchParams();
  const browsing = params.get("source") ?? undefined;
  const { data, isLoading, error } = useScrolls(browsing);
  const [text, setText] = useState("");
  const [onlySelected, setOnlySelected] = useState(false);
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [dropping, setDropping] = useState<Set<string>>(new Set());
  const [reason, setReason] = useState("");
  const [sourceOpen, setSourceOpen] = useState(false);
  const tableRef = useResizableColumns<HTMLTableElement>();
  const deferred = useDeferredValue(text);

  const source = useQuery({
    queryKey: ["config"],
    queryFn: async () => {
      const r = await fetch("/api/config");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { environment: { name: string; value: string }[] };
    },
    staleTime: 60_000,
  });
  const sourceUri = source.data?.environment.find((e) => e.name === "CX_SCROLL_SOURCE")?.value;

  const selected = useMemo(() => new Set(current?.scrolls ?? []), [current]);
  // The unfiled mission describes what exists rather than what was chosen, so
  // its selection is not editable and the checkboxes would lie.
  const editable = Boolean(missionId) && !current?.implicit;
  // Ceremony is owed to work, not to the calendar: a mission with no receipts
  // has claimed nothing, so its selection is a draft and edits are free.
  const frozen = Boolean(current?.selection_frozen);

  const rows: Row[] = useMemo(() => {
    if (!data) return [];
    const needle = deferred.trim().toLowerCase();
    return data.scrolls
      .map((s) => ({
        ...s,
        selected: (selected.has(s.sample_id) && !dropping.has(s.sample_id)) ||
                  pending.has(s.sample_id),
      }))
      .filter((s) => (!needle || s.sample_id.toLowerCase().includes(needle)) &&
                     (!onlySelected || s.selected));
  }, [data, deferred, selected, pending, dropping, onlySelected]);

  // Checking an unselected scroll queues an addition; unchecking a selected one
  // queues a removal. Both are amendments and both are applied together.
  const toggle = (id: string) => {
    if (selected.has(id)) {
      setDropping((d) => {
        const next = new Set(d);
        next.has(id) ? next.delete(id) : next.add(id);
        return next;
      });
      return;
    }
    setPending((p) => {
      const next = new Set(p);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const apply = useMutation({
    mutationFn: async () => {
      const call = async (path: string, scrolls: string[]) => {
        const r = await fetch(`/api/missions/${missionId}/${path}`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ add: scrolls, reason }),
        });
        const body = await r.json();
        if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
        return body;
      };
      if (pending.size) await call("amend", [...pending]);
      if (dropping.size) await call("remove", [...dropping]);
      return true;
    },
    onSuccess: () => {
      setPending(new Set());
      setDropping(new Set());
      setReason("");
      client.invalidateQueries({ queryKey: ["missions"] });
      client.invalidateQueries({ queryKey: ["subjects"] });
      client.invalidateQueries({ queryKey: ["phase", "P0"] });
    },
  });

  const gate = queryGate({ isLoading, error, data }, "reading the source…");
  if (gate) return gate;
  // The gate covers every unset case; the compiler cannot see that
  // through a helper.
  if (!data) return null;

  return (
    <>
      {!missionId && (
        <Card title="Pick a mission first">
          <div className="body-pad">
            <p>
              Scroll selection belongs to a mission — that is what freezes it, so that a result
              across the selection means something later.
            </p>
          </div>
        </Card>
      )}

      {current?.implicit && (
        <Card title="This selection cannot be edited">
          <div className="body-pad">
            <p>
              <b>{current.name}</b> is a view of runs that predate missions, assembled from their
              receipts. Its scroll list describes what exists rather than what anybody chose, so
              there is nothing to add to or remove from.
            </p>
            <p>Create a mission to select the scrolls you mean to attempt.</p>
          </div>
        </Card>
      )}

      {missionId && (pending.size > 0 || dropping.size > 0) && (
        <Card
          title="Change the selection"
          note={
            <>
              {pending.size > 0 && <Pill kind="ok">+{pending.size}</Pill>}
              {dropping.size > 0 && <> <Pill kind="crit">−{dropping.size}</Pill></>}
            </>
          }
        >
          <div className="body-pad">
            {pending.size > 0 && (
              <p><b>Adding:</b> {[...pending].join(", ")}</p>
            )}
            {dropping.size > 0 && (
              <p><b>Removing:</b> {[...dropping].join(", ")}</p>
            )}
            {frozen ? (
              <p>
                This mission has produced work, so its selection is frozen and every change is
                recorded with a reason — a selection that moves quietly makes every earlier
                result unreadable, because "we screened everything" means something different
                when everything changed halfway through. A scroll that has already produced work
                here cannot be removed at all.
              </p>
            ) : (
              <p>
                Nothing has run in this mission yet, so the selection is still a draft and this
                applies straight away. It freezes on the first run: from then on a change is an
                amendment and needs a reason.
              </p>
            )}
            <div className="controls">
              {frozen && (
                <input
                  className="search"
                  value={reason}
                  placeholder="why the selection is changing…"
                  onChange={(e) => setReason(e.target.value)}
                />
              )}
              <button disabled={(frozen && !reason.trim()) || apply.isPending}
                      onClick={() => apply.mutate()}>
                {apply.isPending ? "applying…" : "Apply"}
              </button>
              <button onClick={() => { setPending(new Set()); setDropping(new Set()); }}>
                clear
              </button>
            </div>
            {apply.isError && <Pill kind="crit">{String(apply.error)}</Pill>}
          </div>
        </Card>
      )}

      <Card
        title="Scrolls"
        note={
          <span className="notewithinfo">
            {rows.length} of {data.total} shown
            <span className="infowrap">
              <button className="infobtn" aria-label="About the source"
                      aria-expanded={sourceOpen} onClick={() => setSourceOpen((v) => !v)}>i</button>
              {sourceOpen && (
                <span className="infopop" role="tooltip">
                  <b>Source</b>
                  <span><code>{sourceUri ?? "not configured"}</code></span>
                  <span>
                    <b>Nothing is downloaded.</b> The source is OME-Zarr over HTTPS and every
                    reader fetches only the chunks it touches — vc3d takes the URI, the renderers
                    do sparse chunk-gather. What lands on disk is derived and far smaller: a
                    surface layer stack is gigabytes where the scan is hundreds.
                  </span>
                  <span>
                    <b>Discovery is by layout, not by list.</b> A top-level prefix counts as a
                    scroll when it holds{" "}
                    <code>volumes/&lt;timestamp&gt;-&lt;µm&gt;um-…-&lt;keV&gt;keV.zarr/</code>.
                    Any bucket in that layout enumerates here; one that is not reports nothing
                    rather than listing its directories.
                  </span>
                  <span className="infometa">
                    the box below browses another bucket for this session only — Configuration →{" "}
                    <code>CX_SCROLL_SOURCE</code> is what persists
                  </span>
                </span>
              )}
            </span>
          </span>
        }
      >
        <div className="body-pad">
          <div className="controls">
            <input
              className="search" type="search" value={text}
              placeholder="filter by scroll id…"
              onChange={(e) => setText(e.target.value)}
            />
            <label className="inlinecheck">
              <input
                type="checkbox" checked={onlySelected}
                onChange={(e) => setOnlySelected(e.target.checked)}
              />
              only this mission
            </label>
            <button
              onClick={async () => {
                const q = browsing ? `&source=${encodeURIComponent(browsing)}` : "";
                await fetch(`/api/scrolls?refresh=true${q}`);
                client.invalidateQueries({ queryKey: ["scrolls"] });
              }}
            >
              refresh inventory
            </button>
          </div>
          {browsing && (
            <p className="hint">
              Browsing <code>{browsing}</code>. Nothing here is saved — the configured source is
              unchanged, and a mission that selects these scrolls records the id only.
            </p>
          )}
          {data.skipped.length > 0 && (
            <p className="hint">
              {data.skipped.length} top-level {data.skipped.length === 1 ? "prefix" : "prefixes"}{" "}
              had no <code>volumes/</code> and are not listed: {data.skipped.join(", ")}.
            </p>
          )}
        </div>
        <div className="scroller">
          <table ref={tableRef}>
            <thead>
              <tr>
                <th className="l pick"><span className="visually-hidden">Include</span></th>
                <th className="l grow">Scroll</th>
                <th className="unit">µm</th>
                <th className="unit">keV</th>
                <th>Scans</th>
                <th className="l">Since</th>
                <th>Runs</th>
                <th className="l">State</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <ScrollRow key={r.sample_id} row={r} onToggle={toggle} disabled={!editable} />
              ))}
            </tbody>
          </table>
        </div>
        {rows.length === 0 && <Empty>nothing matches</Empty>}
      </Card>

      {current && current.amendments.length > 0 && (
        <Card title="How this selection changed" note={`${current.amendments.length} amendments`}>
          <div className="scroller">
            <table>
              <thead>
                <tr><th className="l">Why</th><th className="l">Change</th><th className="l">When</th></tr>
              </thead>
              <tbody>
                {current.amendments.map((a, i) => (
                  <tr key={i}>
                    <td className="l wrap">{a.reason}</td>
                    <td className="l">
                      {a.added?.length ? <Pill kind="ok">+ {a.added.join(", ")}</Pill> : null}
                      {a.removed?.length ? <Pill kind="crit">− {a.removed.join(", ")}</Pill> : null}
                    </td>
                    <td className="l">{a.at_utc.slice(0, 19).replace("T", " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <ProducedArtifacts phase="P0" sample={null} />

      <Card title="Scale is what this phase actually decides">
        <div className="body-pad">
          <p>
            µm and keV come from the frozen catalog and are blank for any scroll it does not
            describe. That blank matters more than it looks: if the voxel size is not pinned here,
            every micron figure downstream is unanchored, and P5 resamples against a number nobody
            checked.
          </p>
        </div>
      </Card>
    </>
  );
}
