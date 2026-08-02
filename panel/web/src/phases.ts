/**
 * The ten phases, by name, known before the server says anything.
 *
 * The rail used to wait for /api/phase-summary, which is the fourth request in
 * a serial chain -- session, then missions, then subjects, then this -- so the
 * pipeline was missing from the page for the whole round trip and then arrived
 * all at once. The names never depended on that call: they come from
 * framework/contracts/pipeline_phases.json, which is a file in this repository
 * and changes only when the repository does. So they ship in the bundle, the
 * rail draws immediately, and the request fills in the part that is genuinely
 * per-mission: what each phase is doing.
 *
 * phases.test.tsx reads the contract and fails if this list drifts from it.
 */
export const PIPELINE = [
  { id: "P0", slug: "intake", name: "Volume intake" },
  { id: "P1", slug: "segmentation", name: "Segmentation" },
  { id: "P2", slug: "geometry-certification", name: "Geometry certification" },
  { id: "P3", slug: "flattening", name: "Flattening" },
  { id: "P4", slug: "surface-volume", name: "Surface volume rendering" },
  { id: "P5", slug: "ink-detection", name: "Ink detection" },
  { id: "P6", slug: "liveness-and-control", name: "Liveness" },
  { id: "P7", slug: "screening", name: "Screening and adjudication" },
  { id: "P8", slug: "reconstruction", name: "Reconstruction" },
  { id: "P9", slug: "reading", name: "Rendering and reading" },
] as const;
