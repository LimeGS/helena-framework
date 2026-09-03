import { useEffect, useMemo, useRef, useState } from "react";
import {
  HANDBOOK, SECTION_ORDER, SECTION_TITLES,
  type Block, type Page, type Span,
} from "./handbook-content";

/**
 * The handbook: a sidebar of sections, one page at a time, and a filter.
 *
 * What it replaces was four tabs, each a single scroll. That worked while the
 * whole of it fitted in a morning's reading; it stopped working at about thirty
 * kilobytes, and the answer to "where is the thing about lanes" became scroll
 * and hope. Four flat tabs cannot express fifty pages.
 *
 * So: sections down the left, pages inside them, headings of the current page
 * under it. The address bar carries the page, which is what makes a link to a
 * paragraph possible -- and a link is how one person tells another where the
 * answer is.
 */

// ------------------------------------------------------------------ rendering

function Spans({ spans }: { spans: Span[] }) {
  return (
    <>
      {spans.map((span, index) => {
        switch (span.kind) {
          case "code": return <code key={index}>{span.text}</code>;
          case "strong": return <strong key={index}>{span.text}</strong>;
          case "em": return <em key={index}>{span.text}</em>;
          case "link":
            // Internal links are hashes into this same handbook; external ones
            // open away from the panel, and say so to a screen reader.
            return span.href.startsWith("#") ? (
              <a key={index} href={span.href}>{span.text}</a>
            ) : (
              <a key={index} href={span.href} target="_blank" rel="noreferrer">
                {span.text} <span aria-hidden="true">↗</span>
              </a>
            );
          default: return <span key={index}>{span.text}</span>;
        }
      })}
    </>
  );
}

function Figure({ block }: { block: Extract<Block, { kind: "figure" }> }) {
  // Resolved through the bundler's glob so a screenshot gets a content hash and
  // a corrected one is never served from a stale cache. A missing file renders
  // as its caption rather than a broken image icon: the words are the part that
  // has to survive.
  const sources = import.meta.glob<string>("../assets/handbook/*", {
    eager: true, query: "?url", import: "default",
  });
  const src = sources[`../assets/handbook/${block.src}`];
  return (
    <figure className="hb-figure">
      {src ? <img src={src} alt={block.alt} loading="lazy" /> : null}
      {block.caption ? <figcaption>{block.caption}</figcaption> : null}
    </figure>
  );
}

// `page` is the id of the page these blocks belong to. A heading's permalink
// used to be `#<heading>` alone, which pageFromHash cannot read: following it
// (or reopening a copied link) fell back to the first page. It carries the page
// now, and the anchor after it is scrolled to by Handbook itself.
function Blocks({ blocks, page }: { blocks: Block[]; page: string }) {
  return (
    <>
      {blocks.map((block, index) => {
        switch (block.kind) {
          case "h": {
            const Tag = (`h${block.level}`) as "h1" | "h2" | "h3";
            return (
              <Tag key={index} id={block.id} className="hb-h">
                {block.text}
                <a className="hb-anchor" href={`#/docs/${page}#${block.id}`}
                   aria-label={`link to ${block.text}`}>#</a>
              </Tag>
            );
          }
          case "p":
            return <p key={index}><Spans spans={block.spans} /></p>;
          case "list":
            return block.ordered ? (
              <ol key={index}>{block.items.map((item, n) =>
                <li key={n}><Spans spans={item} /></li>)}</ol>
            ) : (
              <ul key={index}>{block.items.map((item, n) =>
                <li key={n}><Spans spans={item} /></li>)}</ul>
            );
          case "code":
            return (
              <pre key={index} className="hb-code" data-language={block.language}>
                <code>{block.text}</code>
              </pre>
            );
          case "table":
            return (
              <div key={index} className="hb-tablewrap">
                <table className="hb-table">
                  <thead><tr>{block.head.map((cell, n) =>
                    <th key={n}><Spans spans={cell} /></th>)}</tr></thead>
                  <tbody>{block.rows.map((row, r) =>
                    <tr key={r}>{row.map((cell, c) =>
                      <td key={c}><Spans spans={cell} /></td>)}</tr>)}</tbody>
                </table>
              </div>
            );
          case "callout":
            return (
              <aside key={index} className={`hb-callout hb-${block.tone}`}>
                <b>{block.tone}</b>
                <p><Spans spans={block.spans} /></p>
              </aside>
            );
          case "figure":
            return <Figure key={index} block={block} />;
          default:
            return null;
        }
      })}
    </>
  );
}

// -------------------------------------------------------------------- search

/** Every word of a page, flattened once, so filtering is a substring test. */
function haystack(page: Page): string {
  const words: string[] = [page.title, page.summary];
  const spans = (list: Span[]) => list.forEach((s) => words.push(s.text));
  for (const block of page.blocks) {
    if (block.kind === "p" || block.kind === "callout") spans(block.spans);
    else if (block.kind === "h") words.push(block.text);
    else if (block.kind === "list") block.items.forEach(spans);
    else if (block.kind === "code") words.push(block.text);
    else if (block.kind === "table") {
      block.head.forEach(spans);
      block.rows.forEach((row) => row.forEach(spans));
    }
  }
  return words.join(" ").toLowerCase();
}

// ---------------------------------------------------------------------- shell

const FALLBACK = HANDBOOK[0]?.id ?? "";

// `#/docs/start/what-helena-is#the-queue` names a heading after the page. The
// browser cannot scroll to it on its own (the fragment is the whole string), so
// the page does, once it has rendered. Guarded like scrollTo: jsdom has no
// scrollIntoView.
function scrollToAnchor(): void {
  const match = /#\/docs\/[a-z0-9-]+\/[a-z0-9-]+#([A-Za-z0-9_-]+)/.exec(window.location.hash);
  const target = match ? document.getElementById(match[1]) : null;
  if (target && typeof target.scrollIntoView === "function") target.scrollIntoView();
}

function pageFromHash(): string {
  // `#/docs/start/what-helena-is` -> `start/what-helena-is`. A heading anchor
  // may follow it, and is left to the browser.
  const match = /#\/docs\/([a-z0-9-]+\/[a-z0-9-]+)/.exec(window.location.hash);
  return match && HANDBOOK.some((p) => p.id === match[1]) ? match[1] : FALLBACK;
}

export default function Handbook() {
  const [current, setCurrent] = useState(pageFromHash);
  const [filter, setFilter] = useState("");
  const body = useRef<HTMLDivElement>(null);

  // The address bar is the source of truth, so Back works and a pasted link
  // lands where it says. Listening rather than only writing is what makes the
  // browser's own buttons behave.
  useEffect(() => {
    const onHash = () => { setCurrent(pageFromHash()); scrollToAnchor(); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const index = useMemo(
    () => HANDBOOK.map((page) => ({ page, text: haystack(page) })), []);

  const matches = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return null;
    return new Set(index.filter((e) => e.text.includes(needle))
                        .map((e) => e.page.id));
  }, [filter, index]);

  const page = HANDBOOK.find((p) => p.id === current) ?? HANDBOOK[0];

  // A new page starts at its own top. Without this, following a link from the
  // bottom of a long page lands you in the middle of the next one.
  //
  // Guarded because `scrollTo` is not a function on an element under jsdom, and
  // a documentation page that throws in a test run is a documentation page
  // nobody can assert anything about.
  useEffect(() => {
    const node = body.current;
    if (typeof node?.scrollTo === "function") node.scrollTo({ top: 0 });
    scrollToAnchor();
  }, [current]);

  const go = (id: string) => {
    window.location.hash = `#/docs/${id}`;
    setCurrent(id);
  };

  if (!page) return <div className="empty">the handbook is empty</div>;

  return (
    <div className="hb">
      <nav className="hb-nav" aria-label="Handbook">
        <input
          className="hb-filter"
          type="search"
          placeholder="Filter pages…"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          aria-label="Filter handbook pages"
        />
        {matches?.size === 0 ? (
          <p className="hb-none">Nothing matches {`"${filter}"`}.</p>
        ) : null}
        {SECTION_ORDER.map((section) => {
          const pages = HANDBOOK.filter(
            (p) => p.section === section && (!matches || matches.has(p.id)));
          if (!pages.length) return null;
          return (
            <div key={section} className="hb-section">
              <h2>{SECTION_TITLES[section]}</h2>
              <ul>
                {pages.map((p) => (
                  <li key={p.id}>
                    <button
                      className="hb-link"
                      aria-current={p.id === current ? "page" : undefined}
                      onClick={() => go(p.id)}
                      title={p.summary}
                    >
                      {p.title}
                    </button>
                    {/* The current page's own headings, in place, so the
                        sidebar answers "where am I" and "what else is on this
                        page" without a second column. */}
                    {p.id === current && p.outline.length > 1 ? (
                      <ul className="hb-outline">
                        {p.outline.map((h) => (
                          <li key={h.id}>
                            <a href={`#${h.id}`}>{h.text}</a>
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </nav>

      <article className="hb-page" ref={body}>
        <header className="hb-head">
          <p className="hb-crumb">{page.sectionTitle}</p>
          <h1>{page.title}</h1>
          <p className="hb-summary">{page.summary}</p>
        </header>
        <Blocks blocks={page.blocks} page={page.id} />
      </article>
    </div>
  );
}
