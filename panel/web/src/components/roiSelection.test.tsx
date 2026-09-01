import { expect, test } from "vitest";
import { boxFromDrag, coverage, pointToMap, reviewBox } from "./roiSelection";

// The browser's only job here is to say which rectangle somebody dragged. It
// must not compute the transform, the lineage or any digest -- the panel derives
// those, and a second implementation in the browser would be obliged to agree
// with the first. Every disagreement of that kind in this pipeline has cost a
// run of GPU time to find.

const MAP = { width: 56, height: 64 };
const RECT = { left: 0, top: 0, width: 560, height: 640 };

test("a screen point lands where the pan and zoom put it", () => {
  const view = { x: 0, y: 0, scale: 1 };
  const extent = { w: 56, h: 64 };

  expect(pointToMap(280, 320, RECT, view, extent)).toEqual({ x: 28, y: 32 });
});

test("zoomed in, the same screen point is a different map pixel", () => {
  // Half the map visible, panned to its middle.
  const view = { x: 14, y: 16, scale: 2 };
  const extent = { w: 28, h: 32 };

  expect(pointToMap(280, 320, RECT, view, extent)).toEqual({ x: 28, y: 32 });
  expect(pointToMap(0, 0, RECT, view, extent)).toEqual({ x: 14, y: 16 });
});

test("dragging up-left is the same rectangle as dragging down-right", () => {
  const a = boxFromDrag({ x: 30, y: 34 }, { x: 10, y: 12 }, MAP);
  const b = boxFromDrag({ x: 10, y: 12 }, { x: 30, y: 34 }, MAP);

  expect(a).toEqual(b);
  expect(a).toEqual({ x0: 10, y0: 12, x1: 30, y1: 34 });
});

test("a drag beyond the edge is clamped to the map", () => {
  const box = boxFromDrag({ x: -12, y: -8 }, { x: 900, y: 900 }, MAP);

  expect(box).toEqual({ x0: 0, y0: 0, x1: 56, y1: 64 });
});

test("a box covering the whole map is refused, with the reason", () => {
  const verdict = reviewBox({ x0: 0, y0: 0, x1: 56, y1: 64 }, MAP);

  expect(verdict.ok).toBe(false);
  if (!verdict.ok) expect(verdict.why).toContain("asserts nothing");
});

test("a click without a drag is refused rather than sent", () => {
  expect(reviewBox({ x0: 10, y0: 12, x1: 10, y1: 12 }, MAP).ok).toBe(false);
});

test("a real selection is accepted as the bbox the route takes", () => {
  const verdict = reviewBox({ x0: 10, y0: 12, x1: 30, y1: 34 }, MAP);

  expect(verdict).toEqual({ ok: true, bbox: [10, 12, 30, 34] });
});

test("coverage is shown so a near-total selection is visible before sending", () => {
  expect(coverage({ x0: 0, y0: 0, x1: 56, y1: 64 }, MAP)).toBe(1);
  expect(coverage({ x0: 0, y0: 0, x1: 28, y1: 32 }, MAP)).toBeCloseTo(0.25);
});
