/**
 * Turning a drag on the map into the one thing the panel accepts: a rectangle
 * in map pixels.
 *
 * Nothing else is computed here on purpose. The transform, the lineage and the
 * digests are derived by the server, because a second implementation of them in
 * the browser would be obliged to agree with the first -- and every disagreement
 * of that kind in this pipeline has cost a run of GPU time to discover.
 *
 * The guards below duplicate refusals the server also makes. That is deliberate
 * and the order matters: the server's refusal is the one that decides, and these
 * exist only so somebody does not drag a box, wait, and be told no. A disabled
 * button is a courtesy, never a substitute.
 */

export type View = { x: number; y: number; scale: number };
export type Extent = { w: number; h: number };
export type Box = { x0: number; y0: number; x1: number; y1: number };

/** Where a screen point lands in map pixels, under the current pan and zoom. */
export function pointToMap(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  view: View,
  extent: Extent,
): { x: number; y: number } {
  const fx = (clientX - rect.left) / rect.width;
  const fy = (clientY - rect.top) / rect.height;
  return { x: view.x + extent.w * fx, y: view.y + extent.h * fy };
}

/**
 * The box two corners describe, snapped to whole map pixels and clamped to the
 * map. Dragging up-left is the same rectangle as dragging down-right.
 */
export function boxFromDrag(
  from: { x: number; y: number },
  to: { x: number; y: number },
  map: { width: number; height: number },
): Box {
  const x0 = Math.max(0, Math.min(Math.floor(from.x), Math.floor(to.x)));
  const y0 = Math.max(0, Math.min(Math.floor(from.y), Math.floor(to.y)));
  const x1 = Math.min(map.width, Math.max(Math.ceil(from.x), Math.ceil(to.x)));
  const y1 = Math.min(map.height, Math.max(Math.ceil(from.y), Math.ceil(to.y)));
  return { x0, y0, x1, y1 };
}

export type Refusal = { ok: false; why: string };
export type Accepted = { ok: true; bbox: [number, number, number, number] };

/** Whether this box is worth sending, and if not, what to tell the person. */
export function reviewBox(
  box: Box,
  map: { width: number; height: number },
): Accepted | Refusal {
  if (box.x1 <= box.x0 || box.y1 <= box.y0) {
    return { ok: false, why: "Drag a rectangle to select the letterforms." };
  }
  if (box.x0 === 0 && box.y0 === 0 && box.x1 === map.width && box.y1 === map.height) {
    return {
      ok: false,
      why:
        "A region covering the whole map asserts nothing — the model lights up " +
        "somewhere on every run. Select the letterforms themselves.",
    };
  }
  return { ok: true, bbox: [box.x0, box.y0, box.x1, box.y1] };
}

/** What fraction of the map the selection covers, for the person to see. */
export function coverage(box: Box, map: { width: number; height: number }): number {
  const area = Math.max(0, box.x1 - box.x0) * Math.max(0, box.y1 - box.y0);
  return area / (map.width * map.height);
}
