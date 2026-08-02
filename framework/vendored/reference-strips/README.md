# Reference strips: qualified multi-wrap benchmarks in a box

Curated local reference bands for the Vesuvius Challenge community, with an
automated qualification suite and a one-command scorer for **tracers**
(wrong-hop %) and **meshers** (manifoldness / boundary / cross-wrap-fusion
metrics).

The engine is the local-multi-wrap-reference methodology validated in the
[neural-tracing-audit](../neural-tracing-audit/) benchmark (see its
`docs/NT_AUDIT.md`): where a segment spirals through several revolutions,
its own geometry records where the neighboring wraps physically sit — no
human annotation needed. This repo productizes that one validated band into
a reusable format, a qualification pipeline, and a scorer anyone can run
against their own method.

## What a qualified strip is

A **strip** (`strip-v0`, one `.npz` file) is a small patch of scroll where
two or more consecutive papyrus wraps are recorded as separate labeled
point sets, plus pitch and provenance metadata. It is a *local reference*,
not ground truth: it is derived from a segmentation and validated by
internal consistency plus (optionally) an independent CT cross-check — not
by papyrological annotation.

**Green means the reference is trustworthy** because of four checks
(`qualify_strip.py`), each guarding a different failure:

| Check | Guards against | Pass criterion (PROVISIONAL) |
|---|---|---|
| (a) Self-test | pipeline plumbing bugs (indexing, xyz/zyx mixups, dtype) | every wrap's own points return to their own wrap at 0.000 distance |
| (b) Wrong-side separation | mislabeled/shuffled wraps, undersampled references | ≥ 95% of each wrap's points are far from the adjacent wrap relative to their own sampling distance |
| (c) Null baseline | a scorer too loose to fail garbage | a synthetic 2×-gap overshoot scores ≥ 99% wrong-hop |
| (d) CT intensity cross-check (optional, network) | segmentation-only artifacts (a wrong band that is self-consistent) | CT peak spacing along normals within 0.5–2.0× the strip's pitch |

Check (b) is the one that actually catches corrupted labels — and it is
deliberately *not* expressible in terms of the quantity it tests (see the
docstring in `qualify_strip.py` for why the naive formulation is a
tautology that passes everything). The test suite includes a
deliberately-shuffled strip that must fail, as a regression guard on the
suite itself.

A strip without a passing `<strip>.qualification.json` next to it is
**UNQUALIFIED** and the scorer says so on every run.

## Tiers (PROVISIONAL thresholds)

Assigned from the measured median pitch (inter-wrap spacing):

| Tier | Median pitch |
|---|---|
| `ultra` | < 120 µm |
| `medium` | 120–250 µm |
| `easy` | ≥ 250 µm |

These encode "ultra-compressed / medium / easy" as round-number bins; they
are not calibrated against a corpus and are marked PROVISIONAL in the code
(`make_strip.py`), as is every other threshold in this repo (all recorded
verbatim into each qualification report for auditability).

> **First:** `strips/UC-01/UC-01.npz` is NOT in git (it is a ~66 MB
> regenerable artifact). A fresh clone does not contain it, and the two
> `score_strip.py` commands below will fail with `FileNotFoundError` until
> you rebuild it. Rebuild it once, from public data, with the recipe in
> [`strips/UC-01/README.md`](strips/UC-01/README.md) (`convert_ntaudit_band.py`
> against the neural-tracing-audit band). Or point `--strip` at a strip you
> minted yourself (see "Minting a strip" below).

## Quickstart — tracer people

```bash
# 1. score your predicted points (M,3 xyz, key "points") for the hop
#    from wrap 1 to wrap 2 (rebuild UC-01.npz first, see the note above):
python score_strip.py --strip strips/UC-01/UC-01.npz --mode points \
    --pred my_predictions.npz --from-wrap 1 --direction front

# 2. read the scorecard (also written as .json + .md next to your file):
#    headline = wrong-hop % (fraction landing past half the local gap),
#    split into wrong-wrap % and distance-miss %; excluded points are
#    reported, not guessed.
```

## Quickstart — mesher people

```bash
# 1. score your mesh (OBJ, or a tifxyz directory) against the strip:
python score_strip.py --strip strips/UC-01/UC-01.npz --mode mesh \
    --pred my_mesh.obj

# 2. read the scorecard: non-manifold edges, open boundary edges,
#    connected components, and cross-wrap fusion triangles (bridges that
#    straddle two wraps of the reference) with a fused-pair histogram.
```

## Minting a strip from a segment

```bash
python make_strip.py path/to/tifxyz_segment/ \
    --window 0,200,4000,12000 --axis 7100,1732,1809 \
    --voxel-size-um 7.91 --scroll "PHerc0332" --segment-id 20240618142020 \
    --out strips/my-strip/my-strip.npz
python qualify_strip.py strips/my-strip/my-strip.npz
```

The window must span ≥ 2 revolutions of the scroll (that is what makes a
multi-wrap reference). `--axis z,y,x` is the scroll's rotation axis
(umbilicus); without it, make_strip estimates one from the window centroid
and *refuses to proceed* if the geometry says the estimate is invalid
(see limitations below).

## Scoring rule (points mode)

Ported from the source audit (provenance headers in `scoring_core.py`
document exactly what is unchanged vs adapted):

- one KD-tree per reference wrap; a prediction is assigned to the wrap it
  is nearest to;
- hop-correct iff assigned to the expected adjacent wrap AND within half
  the local gap (the from-wrap point's own distance to the target wrap);
- **wrong-hop % = wrong-wrap % (wrong identity) + distance-miss % (right
  wrap, ≥ half-gap)**, over included predictions only;
- predictions with no reference coverage within a radius, or whose nearest
  reference point sits on a coverage boundary (edge flags — the port of
  the audit's band row-edge rule), are **excluded, not guessed**, and
  counted in the scorecard.

## Registry

| Strip | Scroll | Tier (measured) | Status | Where |
|---|---|---|---|---|
| UC-01 | PHerc 0332 / Scroll 3, segment `20240618142020`, PPM rows 1145–1344 | medium band-wide (ultra-compressed core: 70–103 µm) | QUALIFIED (all four checks measured here; CT streamed the live volume, 13.0 vox = 102.8 µm, ratio 0.73) | `strips/UC-01/` (by recipe + converter; data not bundled) |

## Honest v0 limitations

- **Validated on synthetic geometry + one real band.** The full pipeline
  (build → qualify → score, both modes) is exercised end-to-end on an
  analytic Archimedean-spiral scroll in the tests, and the UC-01
  conversion of the real PHerc 0332 band passes qualification with class
  populations exactly matching the source audit's published table. No
  other real segment has been through it.
- **`make_strip.py` on real segments is v0-fragile, by design refusing
  rather than guessing.** Constraints actually implemented: the window
  must span ≥ 2 revolutions around the given axis; without `--axis` the
  centroid-estimated axis is rejected when the window's radius span says
  the estimate is meaningless (this is exactly what happens on the small
  near-planar `seed_segment/` bundled with the source audit — measured
  ~0.002 revolutions, correctly refused); phase unwrapping is per-line
  with no cross-line branch matching, so segments whose angle varies
  fast across the non-winding axis may mis-bin (the UC-01 converter uses
  the source audit's cross-row-matched unwrap instead); normals are only
  computed for fully-valid windows. `make_strip` is therefore validated
  on synthetic segments only; the one real strip (UC-01) was built by its
  own converter, not by `make_strip`.
- **Mesh metrics are strip-local geometric checks**, not a global
  topology certification: fusion counts are relative to THIS strip's
  wraps and coverage; a mesh can be clean here and broken elsewhere.
- **The CT check is optional and network-dependent**, never run in the
  offline test suite. It requires per-wrap normals and the CT volume
  (`zarr`/`fsspec`/`aiohttp`). It has been run once, on UC-01 (2026-07-10):
  the strip was rebuilt with normals recovered from the PPM's `nx,ny,nz`
  channels, and check (d) streamed ~35 MB of the live CT volume to measure
  13.0 vox = 102.8 µm median inter-sheet spacing (ratio 0.73 vs pitch) —
  reproducing the source audit's figure through the shipped `check_ct`.
- **Coverage holes are excluded, not fixed.** Real bands have patchy
  wrap coverage (damaged/unsegmented sheet); the qualification null
  excludes seeds facing holes (measured on UC-01: 41% of seeds for the
  outermost wrap pair, consistent with the source audit's own ~53%
  front-direction exclusion rate). Scoring near holes inherits the
  reference's sparsity — a conservative bias (toward wrong-hop) for
  predictions there.
- **All thresholds are PROVISIONAL** (tier bounds, separation factor,
  wrong-side 95%, null 99%, coverage-radius multipliers, edge margin, CT
  ratio band, subsample cap). Each is tagged in the code and recorded in
  every report.

## Layout

```
strip_format.py       strip-v0 container: save/load/validate, edge flags,
                      is_qualified()
scoring_core.py       KD-tree primitives + ported half-gap scoring rule
make_strip.py         mint a strip from a tifxyz segment window
qualify_strip.py      the 4-check qualification suite -> qualification.json/.md
score_strip.py        one-command scorer (points | mesh) -> scorecard.json/.md
mesh_metrics.py       OBJ parsing, manifoldness/boundary/components, fusion
strips/UC-01/         first registry entry: recipe, converter, metadata,
                      provenance (data files not committed)
tests/                offline deterministic unittest suite (54 tests; the
                      UC-01 real-data test is env-gated and skips by default)
```

## Running the tests

```bash
python -m unittest discover -s tests
```

Offline, deterministic, ~2 s (numpy + scipy only). The optional UC-01
real-data test runs only when `UC01_BAND_NPZ` points at a local band file.

## Requirements

Core: `numpy`, `scipy` (KD-trees via `scipy.spatial.cKDTree`). Python 3.11.
Optional, clearly separated: `tifffile` (real tifxyz segments), `zarr`
(+`fsspec`, `aiohttp`) for the network CT check. No torch, no GPU.

## License

MIT (code). Referenced scan data is CC BY-NC 4.0 from the Vesuvius
Challenge.
