#!/usr/bin/env node
/**
 * The handbook, from Markdown to a typed module the panel can render.
 *
 * Why a build step rather than a Markdown renderer at runtime: this panel has
 * four dependencies and the handbook is not worth a fifth plus its tree. Why
 * Markdown rather than the typed content objects the old guide used: the
 * handbook is a hundred and fifty kilobytes of prose, and prose inside string
 * literals is unreviewable in a diff -- you cannot see a paragraph change, only
 * that a line moved.
 *
 * So the parsing happens once, here, and what ships is data. The subset is the
 * one documentation actually needs: headings, paragraphs, lists, fenced code,
 * pipe tables, blockquote callouts, figures, and inline emphasis/code/links.
 * Anything outside it is a build error rather than a silent passthrough, so a
 * page cannot half-render in production because somebody used a syntax nobody
 * implemented.
 */

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..", "..");
const SOURCE = join(ROOT, "docs", "handbook");
const TARGET = join(HERE, "..", "src", "routes", "handbook-content.ts");

class HandbookError extends Error {}

// ---------------------------------------------------------------- front matter

function frontMatter(text, where) {
  if (!text.startsWith("---\n")) {
    throw new HandbookError(`${where}: every page needs front matter`);
  }
  const end = text.indexOf("\n---\n", 3);
  if (end < 0) throw new HandbookError(`${where}: front matter is never closed`);
  const head = {};
  for (const line of text.slice(4, end).split("\n")) {
    if (!line.trim()) continue;
    const at = line.indexOf(":");
    if (at < 0) {
      throw new HandbookError(
        `${where}: front matter line is not key: value -- ${JSON.stringify(line)}`);
    }
    head[line.slice(0, at).trim()] = line.slice(at + 1).trim();
  }
  for (const need of ["title", "summary"]) {
    if (!head[need]) throw new HandbookError(`${where}: front matter needs ${need}`);
  }
  return [head, text.slice(end + 5)];
}

// -------------------------------------------------------------------- inline

// Emphasis, code, links and nothing else. Ordered so that code wins: a
// backtick span is opaque, which is what lets a code sample contain an asterisk
// without turning half a paragraph italic.
const INLINE = [
  [/`([^`]+)`/, (m) => ({ kind: "code", text: m[1] })],
  [/\*\*([^*]+)\*\*/, (m) => ({ kind: "strong", text: m[1] })],
  [/\[([^\]]+)\]\(([^)]+)\)/, (m) => ({ kind: "link", text: m[1], href: m[2] })],
  [/(?<![*\w])\*([^*]+)\*(?!\w)/, (m) => ({ kind: "em", text: m[1] })],
];

function inline(text) {
  const out = [];
  let rest = text;
  while (rest) {
    let best = null;
    for (const [pattern, build] of INLINE) {
      const m = pattern.exec(rest);
      if (m && (best === null || m.index < best.at)) {
        best = { at: m.index, length: m[0].length, node: build(m) };
      }
    }
    if (best === null) { out.push({ kind: "text", text: rest }); break; }
    if (best.at > 0) out.push({ kind: "text", text: rest.slice(0, best.at) });
    out.push(best.node);
    rest = rest.slice(best.at + best.length);
  }
  return out.filter((n) => n.kind !== "text" || n.text !== "");
}

// ------------------------------------------------------------------- blocks

const FIGURE = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)$/;
const CALLOUT = /^>\s*\*\*([A-Za-z ]+)\*\*\s*(.*)$/;

function slug(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function tableRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

function parse(body, where) {
  const lines = body.split("\n");
  const blocks = [];
  let i = 0;

  const paragraph = [];
  const flush = () => {
    if (paragraph.length) {
      blocks.push({ kind: "p", spans: inline(paragraph.join(" ")) });
      paragraph.length = 0;
    }
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) { flush(); i += 1; continue; }

    // Fenced code. Taken verbatim, including blank lines, until the fence
    // closes -- an unclosed one is an error rather than a page that swallows
    // everything after it.
    if (trimmed.startsWith("```")) {
      flush();
      const language = trimmed.slice(3).trim();
      const opened = i;
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]); i += 1;
      }
      if (i >= lines.length) {
        throw new HandbookError(`${where}: code fence opened at line ${opened + 1} never closes`);
      }
      blocks.push({ kind: "code", language, text: code.join("\n") });
      i += 1;
      continue;
    }

    if (trimmed.startsWith("#")) {
      flush();
      const level = trimmed.match(/^#+/)[0].length;
      if (level > 3) throw new HandbookError(`${where}: headings go to ###, not ${level}`);
      const text = trimmed.slice(level).trim();
      blocks.push({ kind: "h", level, text, id: slug(text) });
      i += 1;
      continue;
    }

    const figure = FIGURE.exec(trimmed);
    if (figure) {
      flush();
      blocks.push({ kind: "figure", alt: figure[1], src: figure[2],
                    caption: figure[3] || "" });
      i += 1;
      continue;
    }

    // A callout is a blockquote whose first line names it in bold. The name
    // decides the colour, so the vocabulary is closed: a typo becomes a build
    // error instead of an unstyled grey box nobody notices.
    if (trimmed.startsWith(">")) {
      flush();
      const head = CALLOUT.exec(trimmed);
      if (!head) throw new HandbookError(
        `${where}: a blockquote must start "> **Note**", "> **Trap**", ` +
        `"> **Cost**" or "> **Certification**"; got ${JSON.stringify(trimmed)}`);
      const tone = head[1].trim().toLowerCase();
      if (!["note", "trap", "cost", "certification"].includes(tone)) {
        throw new HandbookError(`${where}: unknown callout ${JSON.stringify(tone)}`);
      }
      const said = [head[2]];
      i += 1;
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        said.push(lines[i].trim().replace(/^>\s?/, "")); i += 1;
      }
      blocks.push({ kind: "callout", tone,
                    spans: inline(said.join(" ").trim()) });
      continue;
    }

    if (/^\|/.test(trimmed)) {
      flush();
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        rows.push(lines[i]); i += 1;
      }
      if (rows.length < 2 || !/^[\s|:-]+$/.test(rows[1])) {
        throw new HandbookError(`${where}: a table needs a header and a --- rule`);
      }
      blocks.push({
        kind: "table",
        head: tableRow(rows[0]).map(inline),
        rows: rows.slice(2).map((r) => tableRow(r).map(inline)),
      });
      continue;
    }

    const bullet = /^([-*])\s+(.*)$/.exec(trimmed);
    const number = /^(\d+)\.\s+(.*)$/.exec(trimmed);
    if (bullet || number) {
      flush();
      const ordered = Boolean(number);
      const items = [];
      while (i < lines.length) {
        const t = lines[i].trim();
        const m = ordered ? /^\d+\.\s+(.*)$/.exec(t) : /^[-*]\s+(.*)$/.exec(t);
        if (!m) {
          // A continuation line: indented, and not a new item. It belongs to
          // the item above, which is how a bullet carries a second sentence
          // without becoming a paragraph of its own.
          if (items.length && lines[i].startsWith("  ") && t) {
            items[items.length - 1] += ` ${t}`; i += 1; continue;
          }
          break;
        }
        items.push(m[1]); i += 1;
      }
      blocks.push({ kind: "list", ordered, items: items.map(inline) });
      continue;
    }

    paragraph.push(trimmed);
    i += 1;
  }
  flush();
  return blocks;
}

// -------------------------------------------------------------------- pages

function walk(directory) {
  const found = [];
  for (const name of readdirSync(directory).sort()) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) found.push(...walk(path));
    else if (name.endsWith(".md")) found.push(path);
  }
  return found;
}

// A directory is a section. The numeric prefix orders them and is not shown,
// so the order is visible in a file listing rather than in a table somebody has
// to remember to update.
const SECTIONS = {
  "00-start": "Start here",
  "10-tutorial": "Tutorial",
  "20-phases": "The phases",
  "30-panel": "The panel",
  "40-reference": "Reference",
  "50-operations": "Operations",
};

function build() {
  // The image build copies `panel/web` and not the repository, so the Markdown
  // is not there -- and it does not need to be: the generated module is
  // committed, and CI regenerates it against the Markdown and refuses a
  // difference. Without this the panel image stopped building the moment the
  // handbook joined `npm run build`.
  //
  // Absent source and an existing module is a build from a committed artifact.
  // Absent source and no module is a genuine error, because there is then
  // nothing for the app to import.
  if (!existsSync(SOURCE)) {
    if (existsSync(TARGET)) {
      console.log(`handbook: no ${relative(ROOT, SOURCE)} here; keeping the ` +
                  "committed module (this is the image build)");
      return;
    }
    throw new HandbookError(
      `neither ${relative(ROOT, SOURCE)} nor a generated module exists`);
  }
  const pages = [];
  const figures = new Set();
  for (const path of walk(SOURCE)) {
    const where = relative(ROOT, path);
    const section = basename(dirname(path));
    if (!(section in SECTIONS)) {
      throw new HandbookError(`${where}: ${section} is not a known section`);
    }
    const [head, body] = frontMatter(readFileSync(path, "utf8"), where);
    const blocks = parse(body, where);
    for (const block of blocks) {
      if (block.kind === "figure") figures.add(block.src);
    }
    // The numeric prefixes order the files and never appear in a URL: a page
    // renamed from 03- to 07- to move it must not break every link to it.
    const slug = basename(path, ".md").replace(/^\d+-/, "");
    pages.push({
      id: `${section.replace(/^\d+-/, "")}/${slug}`,
      slug,
      section,
      sectionTitle: SECTIONS[section],
      title: head.title,
      summary: head.summary,
      // Every heading, so a page can show its own contents without the
      // renderer walking the blocks a second time at display time.
      outline: blocks.filter((b) => b.kind === "h" && b.level === 2)
                     .map((b) => ({ id: b.id, text: b.text })),
      blocks,
    });
  }
  if (!pages.length) throw new HandbookError("no pages found under docs/handbook");

  const banner = `// Generated by panel/web/scripts/build-handbook.mjs from
// docs/handbook/**/*.md. Do not edit: edit the Markdown and run
// \`npm run handbook\`. The check in CI regenerates it and fails if the result
// differs, so an edit here is reverted rather than argued about.
`;
  const types = `
export type Span =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "strong"; text: string }
  | { kind: "em"; text: string }
  | { kind: "link"; text: string; href: string };

export type Block =
  | { kind: "h"; level: number; text: string; id: string }
  | { kind: "p"; spans: Span[] }
  | { kind: "list"; ordered: boolean; items: Span[][] }
  | { kind: "code"; language: string; text: string }
  | { kind: "table"; head: Span[][]; rows: Span[][][] }
  | { kind: "callout"; tone: string; spans: Span[] }
  | { kind: "figure"; alt: string; src: string; caption: string };

export type Page = {
  id: string; slug: string; section: string; sectionTitle: string;
  title: string; summary: string;
  outline: { id: string; text: string }[];
  blocks: Block[];
};

export const SECTION_ORDER: string[] = ${JSON.stringify(Object.keys(SECTIONS))};
export const SECTION_TITLES: Record<string, string> = ${JSON.stringify(SECTIONS, null, 2)};
export const FIGURES: string[] = ${JSON.stringify([...figures].sort())};

export const HANDBOOK: Page[] = ${JSON.stringify(pages, null, 1)};
`;
  writeFileSync(TARGET, banner + types);
  const words = pages.reduce((n, p) => n + JSON.stringify(p.blocks).split(/\s+/).length, 0);
  console.log(`handbook: ${pages.length} pages, ~${words} words -> ${relative(ROOT, TARGET)}`);
}

try {
  build();
} catch (error) {
  if (error instanceof HandbookError) {
    console.error(`handbook: ${error.message}`);
    process.exit(2);
  }
  throw error;
}
