# PHerc0358 8/8/0 — causal classification of the interior-clearance rejection

Status: **read-only diagnostic review. No threshold, margin, policy, queue or
artifact was changed by this work.**

Scope: task 7 of
[`docs/superpowers/plans/2026-08-02-first-letters-discovery-recovery.md`](../superpowers/plans/2026-08-02-first-letters-discovery-recovery.md),
steps 1–4. Base revision `2d25e9b3e90c7204d7722e7bac96ab5fb58f59d9`.

The 2026-08-02 First Letters hybrid campaign made 56 attempts across 13 scrolls.
Fifty-four attempts ended with no M7 proposal at all. Exactly one attempt —
on PHerc0358 — received proposals and lost all of them at the clearance screen,
with raw / post-CT / usable = **8 / 8 / 0**. This document classifies why.

---

## 1. Evidence this review reads

| Artifact | SHA-256 of the file bytes |
| --- | --- |
| `docs/first-letters/first-letters-hybrid-20260802/README.md` | `be683c27bcffd7f2c83885b7ababcf0263841d9e0d044cd42fd7c0d918dee40e` |
| `docs/first-letters/first-letters-hybrid-20260802/evidence.json` | `a2aad7a2fbdaeca590b3bf6664441db6c1b362b4bcf7e48742dc1703ba50e12e` |

The dossier itself names its two upstream records by content, and neither is in
this repository:

| Record | SHA-256 declared by the dossier | Present here |
| --- | --- | --- |
| campaign ledger | `ad3d126a387615a13da56299b8bb6a6beb26b35ff7a474eb2e36d6050b3b2131` | no |
| final API readback | `93918848d245efa93759d8a6a225f1837f710e0fe2c5825ee79e342761bfc83b` | no |

Frozen source identity for the scroll:

* P0 artifact `p0:PHerc0358:8360cd908a08`, content
  `8360cd908a0816ae1719e3af8854f426c23a9779eaea5d12bce4ee0c93b56070`.
* `workspace/catalog/eligible_volumes.json` — `shape_zyx` `[14744, 7783, 7783]`,
  voxel `9.362 µm`, M7 threshold `0.2`.
* `workspace/catalog/volume_metadata/PHerc358.json` — `stored_array_order`
  `zyx`, `consumer_coordinate_order` `xyz`, Zarr shape `[14744, 7783, 7783]`.

Frozen campaign configuration recorded by the dossier: policy
`first-letters-hybrid-20260802-v1`, growth profile `vc3d-m7-growth-v1`, planner
`cost-aware-v2`, selection strategy `stratified-clearance-v1`, CT gate
`ome-zarr-nearby-material-v1`, seed probes `off`, four tasks per scroll.

Everything needed to recompute an actual distance — candidate coordinates, cell
identity, task bounds, the volume shape carried on the task, and the numeric
gate values in force — was deliberately redacted from the public dossier and
exists nowhere in this repository. That fact is asserted by
`tests/test_pherc0358_clearance_classification.py`, not merely described here,
and it is the reason section 6 stops where it does.

---

## 2. What the rule computes

Two independent screens run after the CT material gate, in
`framework/stages/01-segmentation/fleet/planner.py`
(`normalize_candidates`, `diagnose_candidate_rejections`, `screen_candidates`):

```
cell_margin   = min over axes of ( c - low , high - c )
volume_margin = min over axes of ( c , shape - 1 - c )

reject if cell_margin   < minimum_cell_interior_clearance_voxels     (planner.py:429, :536)
reject if volume_margin < minimum_volume_interior_clearance_voxels   (planner.py:429, :538)
```

* `c` is the candidate's CT-L0 voxel coordinate. `low`/`high` are the task's
  claimed region — the recentred `effective_candidate_region` when the seed
  region policy recentres, the nominal cell otherwise.
* `shape` is the scanned volume in x/y/z. The high bound is **inclusive**:
  `shape - 1` is a valid voxel index.
* The comparison is strict `<`, so a candidate sitting exactly on the margin is
  admitted.
* The volume term uses **no** property of the cell. It is an absolute distance
  from the candidate to the six faces of the scan.

The units are frozen alongside: at PHerc0358's `9.362 µm` voxel, a 64-voxel
margin is `599.168 µm`.

The task carries both numbers from the generator
(`framework/stages/01-segmentation/fleet/generator.py:810–811`), where
`minimum_volume_interior_clearance_voxels` is set to **the same
`volume_edge_margin` that inset the grid** (`_axis_centers`, generator.py:296–311,
`margin = max(query_radius, volume_edge_margin)`).

Shipped bootstrap defaults (`framework/stages/01-segmentation/fleet/cli.py:1833–1838`):
`--query-radius 64`, `--volume-edge-margin 64`, `--candidate-interior-clearance 0`,
`--grid-step 2048`. The panel's default wave size is four
(`panel/app.py:7972`), and the provider request is capped at
`maximum_candidate_count = 8` (`generator.py:26–36`, `worker.py:278`).

---

## 3. What the frozen evidence establishes deductively

`primary_causes` in the dossier is the sorted list of causes with a nonzero
count for one NO_SEED attempt (`framework/stages/01-segmentation/fleet/worker.py:828–837`).
Across the campaign the `primary_causes` counts sum to 55, exactly the number of
NO_SEED attempts, so **each attempt contributed exactly one cause**. PHerc0358's
row is `{NO_M7_CANDIDATES: 3, INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE: 1}` over
four attempts. Therefore, for the single 8/8/0 attempt:

1. The CT material gate rejected **none** of the eight (`raw == post_ct == 8`).
2. **No** candidate was malformed.
3. **No** candidate failed the cell screen.
4. `usable == 0` means every retained-eligible count was zero, and a candidate
   with no recorded cause is retained — so **all eight failed the volume screen
   and nothing else**.

This is a derivation, not an inference. It is pinned by
`test_the_frozen_evidence_records_one_clearance_rejection_and_no_cell_rejection`.

Two things this does *not* establish, and which are easy to over-read:

* **Zero cell rejections is uninformative.** `--candidate-interior-clearance`
  ships at `0`, so the cell screen degenerates into "the provider answered
  inside the region it was asked about". Every corner of the query cube passes
  it. The absence of a cell cause says nothing about where inside the cube the
  eight candidates sat.
* **`raw = 8` is the request cap, not a population.** The provider is asked for
  `max_candidates = 8`. The gate therefore saw M7's top eight by score, ranked
  by a model that cannot see the gate. A region containing admissible
  candidates below rank eight still reports 8/8/0.

---

## 4. The structural mechanism, reproduced

The generator and the screen were driven directly, at the shipped defaults,
against PHerc0358's real catalog shape. All of the following is reproduced by
`tests/test_pherc0358_clearance_classification.py` and by
`scripts/harness/analyze_candidate_clearance.py::query_cube_volume_gate_geometry`.

**(a) A first wave on a fresh scroll always lands on the scan rim.**
PHerc0358 finished the campaign with `surface_count` 0, so for every one of its
attempts no surface existed and the cell-clearance term was infinite for every
cell (`generator.py:660`, `point_bbox_gap(..., default=inf)`). With every cell
tied, selection is decided entirely by the deterministic tie-break — lowest grid
index wins (`generator.py:692–702`, `:751`). The lowest index on an axis is by
construction the cell whose centre sits *exactly* `volume_edge_margin` from that
face. The generated wave is:

| cell | centre (x, y, z) | centre volume margin | query cube |
| --- | --- | ---: | --- |
| `r00000c00000a00000` | 64, 64, 64 | 64 | `[0,0,0] – [128,128,128]` |
| `r00000c00000a00004` | 64, 64, 8256 | 64 | `[0,0,8192] – [128,128,8320]` |
| `r00000c00003a00000` | 64, 6208, 64 | 64 | `[0,6144,0] – [128,6272,128]` |
| `r00000c00003a00004` | 64, 6208, 8256 | 64 | `[0,6144,8192] – [128,6272,8320]` |

Every cell of the wave sits exactly *on* the threshold it must clear.

**(b) Part of every one of those cubes is inadmissible by construction.**
The grid protects the *centre*; a candidate may be up to `query_radius` voxels
outward of it; the gate then applies the same number to the candidate. At the
shipped default the two numbers are equal, so the outward half-space on each
rim axis can never be admitted, no matter what M7 proposes there:

| cell | closed faces | admissible voxels / query-cube voxels | fraction |
| --- | --- | ---: | ---: |
| `r00000c00000a00000` | x_low, y_low, z_low | 274 625 / 2 146 689 | 0.128 |
| `r00000c00000a00004` | x_low, y_low | 545 025 / 2 146 689 | 0.254 |
| `r00000c00003a00000` | x_low, z_low | 545 025 / 2 146 689 | 0.254 |
| `r00000c00003a00004` | x_low | 1 081 665 / 2 146 689 | 0.504 |

Seven eighths of the corner cell's search volume is unreachable. Over the full
`5 × 5 × 8 = 200`-cell grid for this scroll, 137 cells (68.5 %) have a query
cube that extends past the gate.

**(c) The signature reproduces exactly, with nothing weakened.** Feeding eight
top-scoring candidates in the outward shell of `r00000c00000a00000` through the
production `screen_candidates` yields:

```
raw_candidate_count      8
eligible_candidate_count 0
usable_candidate_count   0
rejection_counts         {MALFORMED: 0, CELL: 0, VOLUME: 8}
clearance_policy         64 voxels = 599.168 µm
```

Translating the same eight candidates inward by their deficit — changing
nothing else, and no threshold — retains all eight. The rejection is
positional, not a broken screen.

---

## 5. Causal class

Against the four classes the plan allows:

| Class | Verdict |
| --- | --- |
| `CELL_BOUNDARY_ONLY` | **Excluded.** Zero cell rejections are recorded, and the cell minimum was `0`. |
| `IMPLEMENTATION_OR_METADATA_DEFECT` | **Not found in the clearance computation.** See below. |
| `POLICY_MARGIN_UNCALIBRATED` | **Not decidable, and not reachable at the shipped default.** See section 7. |
| `TRUE_VOLUME_BOUNDARY` | **The class this result falls in, definitionally.** |

`TRUE_VOLUME_BOUNDARY` is the classification, with one qualification stated
plainly: at the shipped configuration this verdict is nearly tautological. A
volume rejection *means* the candidate was within `volume_edge_margin` — 64
voxels, `599.168 µm` — of a face of the scan. Having found no defect in the
computation that could produce that reading falsely, the eight candidates were
genuinely inside the rim exclusion shell. **The margin is therefore not
changed, and this review does not authorize changing it.**

Defect hypotheses examined and excluded, each pinned by a test:

* **Axis-order / metadata transposition.** The catalog stores `shape_zyx`; the
  control plane reverses it into the x/y/z frame the gate measures in
  (`panel/app.py:783`, `:2751`, `:3020`, `:3105`, `:4609`, `:4733`). For
  PHerc0358, `[14744, 7783, 7783] → [7783, 7783, 14744]`. A transposition would
  have capped z at 7783 and falsely rejected the upper half of the scan; it is
  not present.
* **Off-by-one on the high face.** `shape - 1` is used as the inclusive last
  voxel index, consistently on all six faces.
* **CT/M7 frame disagreement.** The CT sampler converts x/y/z to z/y/x for the
  Zarr array (`ct_support.py:159`) and clips its window to the array
  (`ct_support.py:169`), so a candidate near a face is still sampled — which
  is why 8/8 passing the CT gate is *consistent* with rim proximity rather than
  evidence against it.
* **A broken screen.** The screen retains the same eight candidates once moved
  inward.

What *is* defective is not the clearance rule but the interaction above it:
**the task asks the provider for candidates over ground the task's own frozen
gate cannot accept, and then keeps only the provider's top eight by a score
that is blind to the gate.** On a scroll with no prior surface, the
deterministic tie-break guarantees that the first wave is drawn from exactly
the rim cells where that overlap is largest. Under the plan's taxonomy this is
not a defect in the clearance computation, so the classification above stands;
it is recorded here as a separate, testable finding about candidate supply,
and it belongs to the preflight and budgeting work of tasks 3, 4 and 6 rather
than to the margin.

---

## 6. What could not be determined, and why

The plan's step 1 asks for every distance to be recomputed for all eight
candidates and compared with the terminal receipt.
`scripts/harness/analyze_candidate_clearance.py` implements exactly that and is
covered by 49 existing tests, but it requires the attempt's
`TERMINAL_RECEIPT.json`, `SEED_CANDIDATES.json`,
`CT_MATERIAL_SUPPORT_SCREEN.json`, `CLAIMED_TASK.json`, `P0_SOURCE.json` and
`M7_CT_ALIGNMENT.json`. **None of those exist in this repository**, and the
public dossier redacted every field they would supply. The analyzer's existing
test suite exercises it against a synthetic reconstruction, not against the
2026-08-02 attempt.

Consequently the following remain open and are *not* claimed:

* the actual coordinates of the eight candidates, and their measured face
  distances in voxels or micrometres;
* which cell the attempt ran on, and whether it was one of the four modelled in
  section 4;
* the numeric `query_radius`, `volume_edge_margin`, `candidate_interior_clearance`
  and `seed_region_policy` in force for that attempt — section 4 assumes the
  shipped defaults, which the dossier neither confirms nor contradicts;
* whether a recentring seed-region policy moved the effective region, which
  would widen the reachable set beyond the rim cells enumerated above.

Section 4 therefore demonstrates that the observed signature is **reachable, and
structurally likely, under the recorded configuration**. It does not demonstrate
that the observed attempt took that path. Closing that gap requires the frozen
attempt bundle, run through the existing analyzer; nothing else in this review
substitutes for it.

---

## 7. Sensitivity (plan step 3) — diagnostic only

The candidate-level sensitivity table the plan asks for — retained counts at
100 / 75 / 50 / 25 / 0 % of the margin — **cannot be computed**, because it needs
the per-candidate face distances that were redacted. The analyzer produces it
automatically once the real bundle is supplied
(`SENSITIVITY_PERCENTAGES = (100, 75, 50, 25, 0)`).

What can be computed is the *geometry* the margin closes off, for the corner
cell of section 4. This is cube geometry, not candidate retention, and it
**cannot authorize a production threshold**:

| margin | voxels | µm | admissible fraction of the corner cube |
| ---: | ---: | ---: | ---: |
| 100 % | 64 | 599.168 | 0.1279 |
| 75 % | 48 | 449.376 | 0.2476 |
| 50 % | 32 | 299.584 | 0.4252 |
| 25 % | 16 | 149.792 | 0.6721 |
| 0 % | 0 | 0.000 | 1.0000 |

A reachability limit worth recording before anyone reads a future verdict:
the analyzer awards `POLICY_MARGIN_UNCALIBRATED` only when a rejected candidate
was still *physically safe*, meaning its own query cube fits inside the volume
(face distance ≥ `query_radius`). When the margin does not exceed the query
radius — as at the shipped `64 = 64` — every rejected candidate is unsafe by
construction, so that verdict is **unreachable** and a `TRUE_VOLUME_BOUNDARY`
result is uninformative on its own. This is exposed as
`policy_margin_verdict_reachability` and pinned by test.

---

## 8. Explicit non-claims

* **No ink claim of any kind.** This review says nothing about whether
  PHerc0358 contains ink, text or letters. A bounded negative never becomes an
  absence claim, and a rejected candidate is a screening outcome, not a
  statement about papyrus.
* **No claim that the eight candidates were bad seeds.** They passed the CT
  material gate. Nothing here evaluates whether they would have grown a
  surface.
* **No claim that the observed attempt is the attempt modelled in section 4.**
  The reproduction is of the *signature* under the recorded configuration, from
  the shipped defaults. The attempt's own parameters and coordinates are not
  available.
* **No claim that the margin is miscalibrated, or correctly calibrated.** The
  evidence needed to decide that was redacted, and at the shipped configuration
  the review taxonomy cannot express the answer anyway (section 7).
* **No claim that PHerc0358 has no usable candidate.** Four cells out of a
  200-cell grid were sampled, all in one corner. No census was run. Candidate
  scarcity measured this way is not surface absence and not ink absence.
* **No claim that the campaign's other 54 `NO_M7_CANDIDATES` outcomes share
  this cause.** They received no proposals at all; the clearance screen never
  ran on them.
* **No threshold, margin, policy version, queue, receipt or artifact was
  changed.** The shipped defaults are asserted unchanged by
  `test_the_classification_document_changes_no_threshold`.

---

## 9. The only follow-up this review authorizes (plan step 4)

1. **Retrieve and analyse the frozen attempt bundle.** Run the six documents
   through `scripts/harness/analyze_candidate_clearance.py` and record
   `campaignx.candidate_clearance_review.v1` under
   `docs/first-letters/pherc0358-clearance-review/`. Until then section 6
   stands.
2. **Do not edit the global margin.** No evidence here supports it, and the
   plan forbids it.
3. **Treat the rim-sampling overlap as a candidate-supply question**, owned by
   the preflight (task 3) and budgeting (task 4) work: measure the admissible
   fraction of a cell's query cube before spending a task on it, and report
   candidate availability separately from attempted coverage. Any change to
   cell selection is a new versioned strategy, not an edit to a gate.
4. **If, and only if, the real bundle shows physically safe rejections**, the
   plan's diagnostic-only route applies: a new diagnostic policy version and at
   most one bounded eight-cell adaptive wave. Never an in-place margin edit.

---

## 10. Verification

* `tests/test_pherc0358_clearance_classification.py` — this review's claims,
  including the frozen-evidence hashes, the deduction in section 3, the
  reproduction in section 4, the axis-order exclusion, and the reachability
  limit in section 7.
* `tests/test_candidate_clearance_analysis.py` — the pre-existing analyzer
  suite, unchanged in behaviour by this work.
* `scripts/harness/analyze_candidate_clearance.py` — gains two pure, read-only
  diagnostics (`query_cube_volume_gate_geometry`,
  `policy_margin_verdict_reachability`). Neither reads state, mutates anything,
  or changes a threshold.
