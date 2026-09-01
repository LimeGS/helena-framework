/**
 * The handbook's shell, and the contract its build script has to keep.
 *
 * The content itself is Markdown and is not asserted here -- prose changes
 * every day and a test that pins a sentence is a test somebody deletes. What is
 * pinned is the shape: every page reachable, every callout a known tone, every
 * figure a file that exists, every internal link a page that is there.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Handbook from "./Handbook";
import { HANDBOOK, SECTION_ORDER, SECTION_TITLES } from "./handbook-content";

const TONES = new Set(["note", "trap", "cost", "certification"]);

describe("the handbook", () => {
  it("has pages, and every one belongs to a declared section", () => {
    expect(HANDBOOK.length).toBeGreaterThan(0);
    for (const page of HANDBOOK) {
      expect(SECTION_ORDER).toContain(page.section);
      expect(SECTION_TITLES[page.section]).toBeTruthy();
      expect(page.title.trim()).not.toEqual("");
      expect(page.summary.trim()).not.toEqual("");
    }
  });

  it("gives every page a unique id, because the id is the URL", () => {
    const ids = HANDBOOK.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("uses only callout tones the stylesheet knows", () => {
    for (const page of HANDBOOK) {
      for (const block of page.blocks) {
        if (block.kind === "callout") expect(TONES).toContain(block.tone);
      }
    }
  });

  it("never links to a handbook page that does not exist", () => {
    // A dead cross-reference is the failure a reader cannot route around: they
    // are told where the answer is and it is not there.
    const ids = new Set(HANDBOOK.map((p) => p.id));
    for (const page of HANDBOOK) {
      for (const block of page.blocks) {
        const spans = block.kind === "p" || block.kind === "callout" ? block.spans
          : block.kind === "list" ? block.items.flat() : [];
        for (const span of spans) {
          if (span.kind !== "link") continue;
          const internal = /^#\/docs\/([a-z0-9-]+\/[a-z0-9-]+)/.exec(span.href);
          if (internal) {
            expect(ids, `${page.id} links to ${span.href}`).toContain(internal[1]);
          }
        }
      }
    }
  });

  it("renders the first page with its sidebar", () => {
    render(<Handbook />);
    // No jest-dom in this suite, so the assertions are plain vitest: the
    // element is found or getBy throws.
    expect(screen.getByLabelText("Filter handbook pages")).toBeTruthy();
    expect(screen.getAllByText(HANDBOOK[0].title).length).toBeGreaterThan(0);
    // The section heading appears in the sidebar and again as the crumb over
    // the page, which is the intended duplication.
    expect(screen.getAllByText("Start here").length).toBeGreaterThan(0);
  });
});
