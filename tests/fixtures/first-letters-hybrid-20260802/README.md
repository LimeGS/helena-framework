# First Letters hybrid campaign — bounded negative dossier

Status: complete bounded campaign record
Frozen evidence: [`evidence.json`](evidence.json)

This dossier records the 2026-08-02 First Letters hybrid campaign across 13 frozen P0 scroll artifacts. It is a bounded negative result: the campaign made 56 attempt records, retained no reviewable candidate, and requested no human letter review. It makes no First Letters claim.

The outcome is deliberately narrower than an absence claim. It does **not** say that any scroll lacks ink, that any scroll is exhausted, or that the target set contains no letters. It says only that no credible, reviewable candidate was produced by this bounded campaign under its recorded policy.

## Verdicts

| Question | Verdict |
| --- | --- |
| Operational campaign | **BOUNDED_THIRTEEN_SCROLL_CAMPAIGN_COMPLETE** |
| Reviewable candidate | **None** |
| Human letter review | **Not requested** |
| Geometry | One PHerc0268 surface was **GEOMETRY_CERTIFIED** |
| CT support | **NOT_SEPARATELY_ESTABLISHED** |
| Ink screen | The sole surface was terminal **INK_SCREEN_INSUFFICIENT / EMPTY** |
| Downstream admissibility | **No**; zero downstream jobs |
| First Letters claim | **No claim** |

Geometry certification and completed QC do not establish CT support. The sole surface was not admissible after an empty ink-screen result, so CT support is kept separately as `NOT_SEPARATELY_ESTABLISHED`; it is not inferred from geometry, terminal QC, or service liveness.

## Campaign scope and frozen inputs

The observed P1 surface path used growth profile `vc3d-m7-growth-v1`; campaign work used policy `first-letters-hybrid-20260802-v1`, planner `cost-aware-v2`, selection strategy `stratified-clearance-v1`, and CT material gate `ome-zarr-nearby-material-v1`, with seed probes off. Expansion was serial: a subsequent scroll was not enqueued before the preceding wave had terminal evidence. P0 identities were frozen before P1 work; the public IDs and SHA-256 content identities are in `evidence.json`.

The final readback records 13 scrolls, 56 attempts, no active attempts, no active QC jobs, no active generic jobs, no active queue tasks or leases, and no downstream jobs. Of the 56 attempt projections, 55 are `NO_SEED` and one is literally `QC_PENDING`. The `QC_PENDING` PHerc0268 projection is related to, but not replaced by, its associated terminal `COMPLETED` QC record; this historical projection/QC mismatch is preserved rather than rewritten.

The campaign made 13 P1 requests, each receipt reporting four generated and four inserted tasks: 52 reported generated and 52 reported inserted in total. Final readback observed 56 attempts. The entire +4 difference is attributed to PHerc0826: its one request reported 4/4, while eight immutable historical task/attempt records were observed. No other scroll has a receipt-to-attempt difference. The incident and its correction are preserved below and in `evidence.json`.

## Per-scroll terminal record

`NO_M7` means `NO_M7_CANDIDATES`. “Clearance” is `INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE`. Counts record every attempt without publishing internal attempt or task identifiers.

| Scroll | Attempts | Terminal result | Candidate gate result |
| --- | ---: | --- | --- |
| PHerc0125 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc0191 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc0211 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc0257 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc0268 | 4 | 3 `NO_SEED`; 1 `QC_PENDING` projection with terminal `COMPLETED` QC | 3 `NO_M7`; one geometry-certified surface later terminal `INK_SCREEN_INSUFFICIENT / EMPTY` and non-admissible |
| PHerc0358 | 4 | 4 `NO_SEED` | 3 `NO_M7`; 1 clearance rejection with raw/post-CT/usable = **8/8/0** |
| PHerc0800 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc0813 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc0826 | 8 | 8 `NO_SEED` | 8 `NO_M7`; see historical fan-out incident below |
| PHerc1203 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc1218 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc1447 | 4 | 4 `NO_SEED` | 4 `NO_M7` |
| PHerc1545 | 4 | 4 `NO_SEED` | 4 `NO_M7` |

Totals: 54 `NO_M7` rejections, one 8/8/0 interior-clearance rejection, and one PHerc0268 surface path. The surface had a 14 by 14 grid, 132 valid triangles, area 0.01983222455087575 cm², and artifact SHA-256 `d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6`. Its geometry was `GEOMETRY_CERTIFIED`, but its physical QC reached `INK_SCREEN_INSUFFICIENT` with liveness verdict `EMPTY`; it was therefore not admissible downstream. This records an empty output for this screening path, not an ink-absence conclusion about the scroll.

## Preserved incidents and corrective evidence

The PHerc0268 QC path initially requeued a terminal degenerate-or-empty liveness refusal as retryable. The repair was implemented at `2ba943ed80fb3a100e3e95261c38d3021acdbf7c`, with 169 focused tests and 1,873 full tests passing (129 skipped). Staging gates `build_ci`, `unit_tests`, `frontend`, `panel_build`, `deploy`, and `smoke` all reached `SUCCEEDED`; deployment completed at 2026-08-02T19:00:29.794Z and smoke at 2026-08-02T19:01:13.712Z. Runtime revision `2ba943ed` converged 8/8 services, with zero transient CI containers remaining. Its observed retry then reached terminal `COMPLETED` with `INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY`; the original projection was preserved.

PHerc0826 also exposed a historical source-snapshot fan-out: one API request reported four generated/inserted tasks, while eight immutable task and attempt records existed across two historical snapshots. The evidence was retained and no cleanup rewrote it. The repair resolves exactly the current mapping and fails closed on missing, duplicate, or sample-mismatched rows. It was fixed at `ebf3b568324ce0cc55947bfa2df983c85aa360c3`, with 85 focused and 1,877 full tests passing (129 skipped). Staging gates `build_ci`, `unit_tests`, `frontend`, `panel_build`, `deploy`, and `smoke` all reached `SUCCEEDED`; deployment completed at 2026-08-02T20:52:02.126912Z and smoke at 2026-08-02T20:53:28.301598Z. Runtime revision `ebf3b568` converged 8/8 services, database health was true, and zero transient CI containers remained. The correction was independently reviewed, deployed, and smoke-approved.

## Evidence boundary and limitations

`evidence.json` contains public P0 artifact identities and hashes, policy and profile names, quantitative gates, terminal state counts, approved fix commits, test counts, and SHA-256 identities of the two source records. It intentionally omits hostnames, users or accounts, worker identifiers, private paths or URIs, authentication or topology details, internal task/attempt/QC identifiers, source-snapshot identifiers, CI job identifiers, and raw logs.

The campaign does not establish sensitivity or recall outside its selected waves; no positive-control conclusion is made. No reviewable packet was formed, so there was no blinded human assessment. The next-wave decision is `NONE`: this bounded scope is complete, carries no implicit continuation, and any future expansion requires new campaign authorization rather than treating this record as evidence of ink absence or scroll exhaustion.
