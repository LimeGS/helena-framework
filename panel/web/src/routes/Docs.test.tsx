import { describe, expect, it } from "vitest";

import { AREAS } from "./guide-controls";

/**
 * That the documentation says something useful about each control.
 *
 * Whether it covers every page is checked in the Python suite instead
 * (tests/test_the_guide_documents_the_panel.py): that needs to read the route
 * files from disk, and pulling node's fs into the browser build to do it costs
 * more than it is worth.
 */

describe("the panel's controls are inventoried and described", () => {
  it("says what each control does, when to use it, and what to leave it on", () => {
    // A control with a name and nothing else is an index entry, not
    // documentation. These lengths are a floor, not a style rule: they catch a
    // placeholder somebody meant to come back to.
    for (const area of AREAS) {
      expect(area.controls.length, `${area.page} documents no controls`).toBeGreaterThan(0);
      expect(area.purpose.length, `${area.page} does not say what the page is for`)
        .toBeGreaterThan(30);
      for (const c of area.controls) {
        expect(c.what.length, `${area.page}/${c.name}: no description`).toBeGreaterThan(20);
        expect(c.when.length, `${area.page}/${c.name}: does not say when to use it`)
          .toBeGreaterThan(10);
        expect(c.recommend.length, `${area.page}/${c.name}: no recommended default`)
          .toBeGreaterThan(20);
      }
    }
  });

  it("does not document the same page twice", () => {
    const pages = AREAS.map((a) => a.page);
    expect(new Set(pages).size, `duplicate page entries: ${pages.join(", ")}`)
      .toBe(pages.length);
  });
});
