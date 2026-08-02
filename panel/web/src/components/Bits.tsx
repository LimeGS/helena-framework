import { memo, useState } from "react";

/**
 * A phase status, drawn rather than typeset. The ten marks used to be Unicode
 * glyphs from four different blocks, which meant four different optical sizes
 * in one column -- and the shapes still have to agree between the rail and the
 * legend in the guide, so they live here rather than in either of them.
 * Decorative: every caller states the status in words beside it.
 */
export const Mark = ({ status }: { status: string }) => (
  <span className={`mark mark-${status}`} aria-hidden="true" />
);

export const Pill = memo(function Pill({
  kind = "none",
  children,
}: {
  kind?: "none" | "neg" | "ok" | "warn" | "crit" | "run";
  children: React.ReactNode;
}) {
  return <span className={`pill ${kind}`}>{children}</span>;
});

/**
 * `collapsed` makes the card foldable and starts it folded. Reference material
 * that is always the same -- a phase contract, a schema -- costs a screenful
 * every visit and is read once; anything that reports current state should not
 * use it.
 */
export const Card = ({
  title,
  note,
  collapsed,
  children,
}: {
  // ReactNode rather than string: a card may want an info button beside its
  // title, which is where the "i" already sits on the phase header.
  title: React.ReactNode;
  note?: React.ReactNode;
  collapsed?: boolean;
  children: React.ReactNode;
}) => {
  const [open, setOpen] = useState(!collapsed);
  return (
    <section className="card">
      <div className="card-head">
        {collapsed !== undefined ? (
          <button className="cardfold" aria-expanded={open}
                  onClick={() => setOpen((v) => !v)}>
            <span className="cardfold-mark" aria-hidden="true">{open ? "▾" : "▸"}</span>
            <h2>{title}</h2>
          </button>
        ) : (
          <h2>{title}</h2>
        )}
        {note && <span className="note">{note}</span>}
      </div>
      {open && children}
    </section>
  );
};

export const Tile = ({
  title,
  tone = "steady",
  value,
  unit,
  children,
}: {
  title: string;
  tone?: "steady" | "busy" | "warn" | "alert";
  value?: React.ReactNode;
  unit?: string;
  children?: React.ReactNode;
}) => (
  <div className={`tile ${tone}`}>
    <h2>{title}</h2>
    {value !== undefined && (
      <div className={`readout ${tone === "alert" ? "alert" : ""}`}>
        {value}
        {unit && <small> {unit}</small>}
      </div>
    )}
    {children}
  </div>
);

export const Num = ({ v, digits = 4 }: { v: number | null | undefined; digits?: number }) =>
  v === null || v === undefined ? <span className="dash">—</span> : <>{v.toFixed(digits)}</>;

export const Empty = ({ children }: { children: React.ReactNode }) => (
  <div className="empty">{children}</div>
);

/**
 * The gate every page needs before it can read `data`.
 *
 * Seven routes wrote `if (isLoading || !data) return <Empty>loading…</Empty>`,
 * which renders a failure as a message that says the opposite. A page that
 * errored sat on "reading the segmentation fleet…" indefinitely, with nothing on
 * screen naming the status code, the endpoint, or even that anything was wrong --
 * so the first sign of trouble was a user waiting.
 *
 * Returns null when the data is usable, so the caller reads:
 *
 *     const gate = queryGate(query, "reading the fleet…");
 *     if (gate) return gate;
 *
 * The three states are kept apart because they need different actions: retry,
 * wait, or fix the thing that made the query empty.
 */
export function queryGate(
  query: { isLoading: boolean; isFetching?: boolean; error: unknown; data: unknown },
  message: string,
): React.ReactElement | null {
  if (query.error) {
    return (
      <Empty>
        this did not load — {String(query.error).replace(/^Error:\s*/, "")}
      </Empty>
    );
  }
  if (query.isLoading) return <Empty>{message}</Empty>;
  if (!query.data) {
    // Not loading, no error, nothing there: a disabled query or a response the
    // parser accepted and emptied. Saying "loading" here waits for an event
    // that is not coming.
    return <Empty>nothing came back, and no error was raised</Empty>;
  }
  return null;
}

/**
 * A paragraph of explanation, folded behind an "i".
 *
 * These pages were carrying four and five sentences of standing explanation
 * above the thing they explained -- true, worth having, and read once. After
 * that it is furniture between you and the data, and the page reads as dense
 * when the density is prose rather than information.
 *
 * Click rather than hover: a hover-only tooltip is unreachable on a touch
 * screen and awkward to read from, and this text is long enough that people
 * want to keep it open while they look at the table under it. Escape and a
 * second click both close it.
 *
 * The same markup Config and the phase header already used inline, in one
 * place now instead of four copies.
 */
export function Info({ label, title, children }: {
  /** What the button announces to a screen reader. */
  label: string;
  /** Optional heading inside the popover. */
  title?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <span className="infowrap">
      <button className="infobtn" aria-label={label} aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
              onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
        i
      </button>
      {open && (
        <span className="infopop" role="tooltip">
          {title && <b>{title}</b>}
          <span>{children}</span>
        </span>
      )}
    </span>
  );
}
