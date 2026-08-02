# PHerc 0139: the columns, in reading order

**Every segmented column of PHerc 0139 (Philodemus, On Gods, book 8) as
plates in physical reading order — all 38 segments (37 body wraps + the
title): 33 with no published readings at all, 4 with isolated phrase-level
readings published in the paper's supplement, and the already-read title
column as a control. Built entirely from public data: the official ds8 ink
maps and the public segment meshes.**

What is formally published on PHerc 0139 (arXiv 2606.29085, submitted 2026-06-27):
the end title ("Φιλοδήμου περὶ θεῶν Η") and, in Supplementary Fig. 1
("Readings from PHerc. 139"), five isolated phrase readings located by
wrap: w49 χωρὶϲ προνο[ί]α̣ϲ̣, w25 ἀόρατα, w34 καὶ τὸ κατ̣ὰ̣ φύϲιν,
w47 καὶ πάν̣τ̣α̣ χωρὶϲ πόνων, w49(int) νοερόν. No continuous transcription
of any column has been published. Everything in these plates beyond the
title and those five phrases is, as of 2026-07-13, unpublished territory —
and the five located phrases double as anchor points to align this series
against the official reading.

Open `viewer/index.html` (or serve this repo and open `/viewer/`) for the
annotated reading view; `plates/` holds the clean full-resolution PNGs.

## How the reading order was obtained

### 1. Wrap order = mesh geometry, not file names

Every segment's TIFXYZ mesh (x/y/z.tif point map, public bucket) references
the same CT volume (20260102150214, 2.399 µm), so all coordinates are
directly comparable. We fit the scroll axis (first principal component of
the union of all segment point clouds, ~760k subsampled points) and
computed, per segment, the **mean radial distance of its point cloud to
that axis** — the wrap radius.

Result: **Spearman rho = 0.9993 (p = 2e-51)** between the wNNN number in
the segment names and the measured mean radius. The official w059 → w023
sequence is therefore strongly consistent with outer → inner radial order:
w023 is the innermost measured body wrap (r = 7.02 mm) and w059 the
outermost (13.26 mm), with all 37 labels consecutive. We use that official
wrap sequence, **not a strict sort of the noisy per-wrap radius estimates**:
there is one local inversion (w050/w049). The mean radial step is
**~0.17 mm — the thickness of rolled papyrus**, an independent physical
sanity check.

> **Method pitfall, documented so you don't repeat it:** the radius of a
> segment's *centroid* is useless here. Each segment is a full turn, so
> its centroid collapses onto the axis regardless of the wrap's radius
> (centroids gave Spearman 0.28 = noise). Use the mean radial distance of
> the *cloud*.

Herculaneum scrolls are read unrolling outer → inner (the end of the work,
with the subscriptio, is rolled innermost). So reading order is
w059 → w023. The title lands at r = 5.50 mm — **inside** the innermost
body wrap — exactly where a subscriptio should be. (Caveat: the title
mesh sits ~4-5 cm away along the axis from the body wraps — a physically
separate chunk in the same scanned volume — so its radius is indicative,
not as rigorous as the body monotonicity.)

### 2. One wrap ≈ one column

At these radii the circumference of one turn is 2πr = 46-83 mm — roughly
the width of one Herculaneum column. Measured plate widths track 2πr wrap
by wrap (e.g. w059: 87.7 mm map vs 83.3 mm circumference; w024: 58.0 vs
45.7, constant margin). **Each segment map is already an unrolled
column** — no stitching across segments needed.

### 3. Reading direction within a plate

Two independent checks (evidence in `data/direction_check/`):

- **Line continuity**: strips cut across contiguous review windows show
  text lines running horizontally and continuously across the full plate
  width (w047 band y=546, 1911 px through 6 windows without a seam;
  w043 band y=1904, 1610 px).
- **No mirroring**: asymmetric letterforms (lunate epsilon/sigma opening
  RIGHT) in the top-scoring windows, plus the decisive convention anchor:
  the title was officially read from this same render family — a mirrored
  family would not read.

Non-mirrored + lines along x ⇒ the greek runs left → right ⇒ within-plate
reading order is x ascending. The exact wrap-to-wrap seam position (where
one turn's text ends and the next begins) is not modeled; it does not
affect the relative order of anything shown.

### 4. The plates (three styles)

Three renderings of each column, toggled in the viewer:

- **`plates/` — pure ink maps.** The official ds8 ink-detection maps
  (~18 µm/px), contrast-stretched to the 2-98 percentile band, ink white
  on black. Nothing else: no denoising, no content edits, no resampling.
  The viewer overlays amber boxes on the reviewed windows; the PNGs are
  clean.
- **`plates_photo/` — our publication-style composite.** Papyrus texture
  from the CT surface volume (the segment's `surface-volumes` zarr,
  level-3 multiscale, mid-depth layer) as a light background, with the ink
  prediction composited in BLACK on top
  (`out = paper_texture * (1 - 0.88 * ink)`). This is our own approximation,
  in the visual spirit of the paper's Fig. 4 panel b ("ink-enhanced signal
  shown in black against the papyrus texture") — letters read as lines of
  text rather than white blobs — but **not** a byte-verified match to the
  paper's own renderer: no formula for that figure is published.
- **`plates_villa/` — the verified villa recipe.** Same papyrus base, but
  the ink is composited with the actual ink-bake formula from
  ScrollPrize/villa's `foundation/scroll-unwrap-pipeline`
  (`docs/RENDER-STYLE.md` + `configs/global.yaml [ink]`, confirmed
  byte-for-byte 2026-07-13): `opacity = smoothstep(ink, 0.42, 0.78) * 0.88`,
  composited as `tint*opacity + paper*(1-opacity)` with a neutral carbon-black
  tint `(16,16,16)` / `#101010`, no glow. That pipeline composites onto its
  own mesh-rendered RGBA texture, which isn't available here, so this still
  uses our same zarr-slice papyrus as the base — the *ink formula* is
  verified, the *base texture* is still an approximation.

  The texture is fetched with HTTP Range requests over the raw
  (uncompressed, `compressor: null`) zarr chunks so only the three mid
  layers of each chunk are downloaded, not the whole volume. Regenerate
  `plates_photo/` and `plates_villa/` together with
  `scripts/make_photo_plates.py`, which fetches the official ds8 maps and
  the textures from the public bucket once and composites both recipes
  from the same fetch.

## Human review behind the amber boxes

The windows come from a legibility index over the official maps
(proxy classifier → human review of every flagged window, one by one),
across two score bands:

- **score ≥ 0.9:** 63 windows flagged, **54 rated "clear text"** (52 outside
  the title); the 52 body windows were then re-reviewed *in physical reading
  order* with 6 visual filters each, **52/52 confirmed legible**
  (`data/atlas_readthrough.json`; raw review in `data/review_0139_human.json`).
- **relaxed 0.6 ≤ score < 0.9:** the 0.9 cut turned out to be very
  conservative, so all 73 windows in this band were reviewed the same way
  (`data/review_band_0139.json`): **53 clear / 19 uncertain / 1 noise** — a
  single false positive in the whole band, i.e. the sub-0.9 signal was mostly
  real text the threshold was discarding.

Together: **107 windows confirmed legible across 16 body columns plus the
title** (up from 14 body columns at ≥0.9; the relaxed pass added the first
confirmed text on w040 and w048). Single reviewer, no inter-rater check;
"legible" means letterforms a reader could attempt — **it is not a
transcription**.

## Reading order and content summary

| # | wrap | radius (mm) | clear windows |
|---|------|------------|---------------|
| 01 | w059 | 13.26 | 13 |
| 02 | w058 | 13.05 | 1 |
| 03 | w057 | 12.91 | — not yet reviewed |
| 04 | w056 | 12.79 | 2 |
| 05 | w055 | 12.48 | — not yet reviewed |
| 06 | w054 | 12.16 | — not yet reviewed |
| 07 | w053 | 12.07 | — not yet reviewed |
| 08 | w052 | 11.82 | — not yet reviewed |
| 09 | w051 | 11.69 | — not yet reviewed |
| 10 | w050 | 11.53 | — not yet reviewed |
| 11 | w049 | 11.62 | 7 |
| 12 | w048 | 11.59 | 1 |
| 13 | w047 | 11.37 | 12 |
| 14 | w046 | 10.95 | 8 |
| 15 | w045 | 10.92 | 6 |
| 16 | w044 | 10.62 | 14 |
| 17 | w043 | 10.43 | 14 |
| 18 | w042 | 10.25 | 6 |
| 19 | w041 | 10.15 | 1 |
| 20 | w040 | 9.94 | 4 |
| 21 | w039 | 9.73 | — not yet reviewed |
| 22 | w038 | 9.57 | — not yet reviewed |
| 23 | w037 | 9.36 | — not yet reviewed |
| 24 | w036 | 9.26 | — not yet reviewed |
| 25 | w035 | 9.19 | — not yet reviewed |
| 26 | w034 | 9.04 | 9 |
| 27 | w033 | 8.91 | 0 (reviewed, none clear) |
| 28 | w032 | 8.75 | 0 (reviewed, none clear) |
| 29 | w031 | 8.61 | 0 (reviewed, none clear) |
| 30 | w030 | 8.42 | — not yet reviewed |
| 31 | w029 | 8.26 | — not yet reviewed |
| 32 | w028 | 8.10 | — not yet reviewed |
| 33 | w027 | 7.89 | 0 (reviewed, none clear) |
| 34 | w026 | 7.72 | — not yet reviewed |
| 35 | w025 | 7.52 | 2 |
| 36 | w024 | 7.28 | 5 |
| 37 | w023 | 7.02 | 0 (reviewed, none clear) |
| 38 | title | 5.50 | 2 (control — officially read) |

Ordering follows the official wrap-number sequence, which is strongly
geometry-consistent with radial order (ρ=0.9993). The radius column is an
independent per-wrap estimate; its sub-0.17 mm scatter produces a local
inversion (w050 vs w049), so it is not used as a strict sorter. Human review so
far covers 22 of the 38 wraps: 17 with confirmed legible windows (16 body
columns plus the title control), and 5 more looked at but with no window
clear enough to confirm (w023/w027/w031/w032/w033). The other 16 are shown
but **not yet reviewed** — their plates are included so the whole segmented
scroll can be read in order. The densest confirmed text sits mid-band
(w043-w047: 54 of the 107 confirmed windows).

## How much of the scroll is this?

The atlas contains the **38 segment IDs frozen in
`data/wrap_radial.json` on 2026-07-13**: 37 body wraps and the title control.
This is a reproducible snapshot of the public bucket inventory used by this
project, not a claim that the live bucket can never gain more segments.

The sum of the 37 body-wrap circumferences is **2.36 m** (2.40 m if the title
control is included). `scripts/estimate_extent.py` recomputes this number
from the committed radial data and writes an assumption-explicit range for a
whole-scroll model: 118–159 wraps and 11.1–16.4 m, using a 25–30 mm outer
radius, 3–5 mm core and 0.17 mm pitch. That places the body-wrap circumference
sum at approximately **14–21% of modeled total papyrus length**.

No mesh-area or total-surface percentage is claimed here. Those quantities
require a versioned per-mesh valid-area calculation; the previous 1,529 cm²
and 5–8% figures were not reproducible from the committed artifacts and have
been removed.

## Two open asks (for papyrologist eyes)

1. Is any of this actually readable at ds8 resolution?
2. The paper's Supplementary Fig. 1 publishes five phrase readings WITH
   wrap locations (w25, w34, w47, w49) and images at 5 mm scale. All four
   wraps are plates in this atlas (13, 12, 05, 04). Matching those five
   published snippets to exact coordinates in our plates would anchor the
   whole series, pixel-level, against the official reading.

## Honest limits

- We flag and order; we do not read greek. No transcription is claimed.
- Four of the 37 body plates (w025, w034, w047, w049) contain the
  five isolated phrases read in the paper's supplement; they are labeled
  in the viewer. The other 33 body plates have no published readings we
  could locate (checked against the paper incl. supplement,
  scrollprize.org and community repos, 2026-07-13).
- Single reviewer for the window labels.
- ds8 resolution (~18 µm/px): letters are 80-220 px tall — visible, but a
  serious reading attempt may want the full-resolution renders.
- **The amber boxes undercount the text — by design.** They mark only the
  107 one-cm windows a human confirmed as clear text (54 from the >=0.9 band,
  53 from the relaxed 0.6-0.9 pass), out of ~4,900 gridded windows on these
  segments. The index is precision-oriented triage; the plates visibly
  contain far more text than the flagged spots (faint or partial text dilutes
  inside a 1 cm window and scores low — which is exactly why relaxing the
  threshold recovered 53 more with only one false positive). Boxes mean
  "guaranteed", not "all there is".
- This is triage/presentation of official model output, not a new ink
  detector. Whatever the official maps miss, these plates miss.
- `reparametrize_to_column.py` is a negative-result diagnostic: radius bands
  do not isolate a physical wrap. `separate_wraps_by_winding.py` is an
  **experimental** alternative validated only on one official single-wrap
  segment plus synthetic controls, not yet on a real multi-wrap mesh.
- Perishable: the official team is actively scaling transcription
  (PHerc 1667 fully read, 2026-06). This ordering is useful today.

## Reproduce (all from public data)

```bash
pip install numpy pillow boto3 scipy
python scripts/wrap_order.py --out data/wrap_radial.json  # meshes -> committed radial table (~1 GB download)
python scripts/make_plates.py           # official ds8 maps -> plates/*.png  (all 38 wraps)
python scripts/make_photo_plates.py     # + surface textures -> plates_photo/*.png, plates_villa/*.png
python scripts/build_viewer.py          # plates + human review -> viewer/index.html
python scripts/estimate_extent.py        # committed geometry -> transparent length estimate
python scripts/validate_release.py       # release integrity checks
```

`scripts/vesuvius_data.py` is the anonymous open-data-bucket helper.
`data/wrap_radial.json` and `data/scroll_extent_estimate.json` are committed
outputs, so the geometry and length numbers above are checkable without
re-downloading the meshes. Serve the repository root and open `/viewer/` (or
open `viewer/index.html` from a checkout); the viewer references the committed
plate files rather than embedding a 60+ MB HTML payload.

## License

Code is MIT. `plates/`, `plates_photo/`, `plates_villa/`, the viewer's
rendered display of them, and derivative data are governed by the source
Vesuvius Challenge data license, CC BY-NC 4.0, not MIT. The photo and villa
styles additionally derive from public CT surface-volume textures. See
`CITATION.md` for the required dataset citation and paper reference.
