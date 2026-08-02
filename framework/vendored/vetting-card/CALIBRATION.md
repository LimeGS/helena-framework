# Calibration of the v0.3 thresholds

v0.1 shipped with hand-picked PROVISIONAL thresholds and a same-map
percentile verdict. On 2026-07-11 that design was measured to fail real,
human-verified clear text on official ds8 ink maps (0/5 windows passed;
`data/index_s5_0139/vetting_diag_20260711.log` in the parent project). v0.2
replaces the verdict with **absolute thresholds calibrated on a labeled set
of real windows**, and demotes the percentile to optional non-gating
context. This file is the full account.

## Why the v0.1 design failed on real maps

Two independent problems, both structural:

1. **The same-map percentile is the wrong bar on text-dense maps.** The
   letter_energy/structure verdicts asked "does this bbox rank in the top
   1% of random same-size boxes on THIS map?". On a segment that is wall to
   wall letters, the random boxes are also full of letters, so a real
   letterform is not locally rare and fails. (This was already flagged in
   the v0.1 README's own limitations.)
2. **The v0.1 `structure` statistic was anti-predictive at ds8 scale.** It
   counted connected components whose area lands in a letter band. On ds8
   maps (~18–19 µm/px) real letters fuse into a few large components
   (below the count a real letter row was assumed to make), while fiber
   noise fragments into many small ones. Measured AUROC of the component
   count on the calibration train split: **0.11** (i.e. strongly inverted).

## The calibration set

516 windows across THREE scrolls, every one on an OFFICIAL ds8 ink map,
each with an explicit-provenance label
(`calibration/build_calibration_set.py`, `calibration/calibration_set.json`).
Thresholds were FIT on 0139 + S1 (v0.2, 2026-07-11); Scroll 4 / PHerc 1667 —
the fully-read scroll — was added 2026-07-12 as a **ground-truth validation
set**, NOT to re-fit (see "Third-scroll validation" below):

The roles are deliberately disjoint in the committed registry:

- **Fit:** 324 windows from PHerc 0139 + Scroll 1 (174 positives, 150 clean
  negatives), with the frozen 70/30 train/test assignment.
- **External validation:** 96 Scroll-4 windows (36 dense-wrap positives,
  60 lacunae negatives). They are never marked train or test and never tune
  a threshold.
- **Context/report only:** 87 S1 implied negatives and 9 PHerc 0139
  possibles. They are disclosed but never tune or score a gate.

| Group | n | Label | Source |
|---|---|---|---|
| 0139 clear | 54 | positive | `review_0139_human.json` rating 1 (single human reviewer, published) |
| S1 confirmed | 120 | positive | `train_labels.jsonl` round=model_proposed (reviewer-confirmed, published) |
| **S4 dense** | **36** | **positive** | **proxy_v4 gold on wraps w028-034, each mapped to a column with 105-159 papyrologist-transcribed letters — wrap-level ground truth** |
| 0139 possible | 9 | report-only | `review_0139_human.json` rating 2 (never fit or scored) |
| S1 fiber | 50 | negative | `fiber_negatives_50.jsonl` (GPU-verified fiber, published) |
| 0139 background | 100 | negative | `index_0139.json` score ≤ 0.05, ≥ 2 windows from any gold (heuristic) |
| **S4 lacunae** | **60** | **negative** | **wraps w011, w037-041, published as lacunae / lost-margin in the read scroll — the strongest "no text here" label available** |
| S1 implied | 87 | context | `train_labels.jsonl` implied negatives — reported, NOT used to fit thresholds |

The fitting split is a frozen registry assignment, not a call to Python's
process-salted `hash()`: rebuilding preserves it for existing IDs and uses
a stable SHA-256-derived seed only for new fit rows. The five windows from
the 2026-07-11 discovery diagnostic are forced into the **test** split, so
"the original failures now pass" is an out-of-sample statement.

**Honest limits of this set.** Three scrolls, but the fitted thresholds come
from only two (0139 + S1); for those, one reviewer, no inter-rater
adjudication; the 0139 background negatives are a heuristic (low score, far
from gold), not human-confirmed non-text; the S1 implied negatives are
excluded from fitting precisely because they are the weakest label. The S4
positives use WRAP-level ground truth (a gold window in a densely-transcribed
wrap can still land on an inter-line gap), which is why they read as a looser
positive label (see the third-scroll section). One more lineage caveat: the
positives were SURFACED by the proxy_v4 triage classifier before being
human-confirmed, so the thresholds are tuned on the kind of text proxy_v4
finds — the human confirmation breaks most of the circularity (every fitted
positive was verified by eye, not by model score), but text that proxy_v4
systematically misses was never in this set. The
vetting card itself contains no trained model: three classical image
statistics and explicit thresholds, auditable by anyone. This
recalibrates the tool from "hand-picked on synthetic data" to "fit on real
human-reviewed data" — it does not make it a statistically certified
detector, and the card still means "survived an automated first pass".

## The v0.2 statistics and thresholds

Chosen on the **train split only**, at the operating point FPR ≤ 0.05 over
the clean negatives, then evaluated once on test. All three must pass.

| Check | Statistic | Gate | Train AUROC |
|---|---|---|---|
| letter_energy | band-pass (DoG) energy / bbox variance (contrast-invariant) | `E_FRAC_MIN = 0.225` | 0.96 |
| structure | fraction of bbox pixels in letter-band components (adaptive threshold K=0.5) | `STRUCTURE_AREAFRAC_MIN = 0.13` | 0.90 |
| contrast_bimodality | Otsu between-class / total variance of the bbox | `OTSU_SEP_MIN = 0.885` | (added to kill smooth-blob pareidolia) |

The bimodality gate exists because energy + area fraction alone let a
smooth-blob field through. Gaussian-blurred noise (σ 12–30, the classic
oversmoothed-output pareidolia texture) tops out at Otsu ≈ 0.66, while real
letters sit at p10 = 0.90 — a clean separation that the first two gates do
not provide on their own.

## Results

On the two scrolls the thresholds were fit on (0139 + S1), held-out:

- **55/56 positives pass** (0139 19/20, S1 36/36), **1/45 clean negatives**.
- **All five 2026-07-11 diagnostic windows now pass** (were 0/5).
- Synthetic control, through the real tool: white noise, blurred noise
  (σ 12/20/30), and synthetic fiber lines all score **0/20**; a blank
  patch trips the degenerate gate. (`calibration/validate_tool.py`,
  reproduced as `tests/test_calibration.py`.)
- Context, not gated: 7/9 "possible" windows pass (consistent with their
  in-between status); 55/87 S1 implied negatives pass (expected — many
  "not flagged while browsing" windows do contain faint structure; this is
  exactly why they were excluded from fitting).

## Third-scroll validation — PHerc 1667, the read scroll (2026-07-12)

Scroll 4 / PHerc 1667 is the first fully-read Herculaneum scroll, so it
offers something the fitted set could not: **ground-truth labels from
published papyrology**, not one reviewer's eye. It was added as a held-out
validation scroll — the thresholds were NOT re-fit on it. All numbers below
are through the real tool (`run_all_checks`).

- **Negatives — the strong result.** 0 of 60 windows on published-lacunae
  wraps (w011, w037-041, "lost margin / traces") pass. The strongest
  "no text here" label available — confirmed blank in a scroll a papyrologist
  read end to end — and the tool rejects every one. This hardens the
  false-positive story well beyond the heuristic 0139 background negatives.
- **Positives — the honest caveat.** 27 of 36 (75%) gold-in-dense-wrap
  windows pass, vs 95%+ on the hand-reviewed scrolls. Most of this gap is
  label quality, not a regression: S4 positives are anchored at the WRAP
  level (a wrap has 105-159 transcribed letters), so a gold window inside
  it can still fall on an inter-line gap or margin and correctly not clear
  the letter-content bar. It is reported, not hidden.
- **Thresholds held.** Scroll 4 was never used to choose a threshold. Its
  separately reported results (27/36 on the wrap-level positives and 0/60 on
  the lacunae negatives) are compatible with retaining the frozen 0139 + S1
  configuration, but they do not turn this validation set into another fit
  split. The config hash therefore remains unchanged.
- **Label-resolution difference.** The fitted-scroll positives are
  window-level reviewer-confirmed letterforms, while the S4 positives are
  wrap-level. S4's lower positive pass rate is consequently a useful stress
  result, not an operating-point retune.

Do not pool the fitted test split with Scroll 4 into one pseudo-held-out
percentage: their label provenance differs. Report **55/56 positives and
1/45 negatives** on the fitted-scroll held-out split, plus **27/36** and
**0/60** separately for the external Scroll-4 positive and negative sets.

## Resolution robustness (2026-07-12 probe)

The calibration set spans only 15.3–19.2 µm/px (0139 ds8 ~18, S1 ds8 19.2),
so the absolute thresholds are *fit* in that band. To test whether one rule
holds outside it, the tool was run on the one real out-of-band point
available: the `gate0_s1_known_text` fixture — a Grand-Prize-verified
known-text region imaged at native **2.4 µm/px**.

Result: at a letter-scale window (≥ ~1 mm, i.e. ≥ one letter) it **passes
comfortably** (e_frac 0.375, area 0.36–0.40, Otsu 0.94), well clear of every
gate — evidence the contrast-invariant design generalizes across an ~8×
resolution range, not just the calibrated band. n = 1 region at 2.4 µm, so
this is encouraging, not a guarantee.

The same region **fails** when cropped to ~1 mm (400 px). That is CORRECT,
not a false negative: at 2.4 µm/px one letter is 625–1667 px, so a 400 px
window physically cannot contain a letter. This exposed the real
false-rejection risk — not resolution, but **sub-letter windows** (the #1
usage error, usually a wrong `--px-um`). v0.2 now guards it: a bbox smaller
than one minimum letter at the given px_um returns "cannot evaluate"
(status error with an actionable message), never a content FAIL that would
read as "no letters here". `MIN_BBOX_LETTER_FRACTION` in `card_config.py`;
`tests/test_vet_map.py::TestSubLetterWindowGuard`.

Validated operating range, stated honestly: thresholds fit on 15–19 µm/px,
spot-checked passing at 2.4 µm/px; unvalidated on other render families
(different ink-detection models) — the model card's "scores do not transfer
across render families" caveat applies here too.

## The render-family guard (v0.4, 2026-07-13)

The "unvalidated on other render families" caveat above stopped being
theoretical: a real claimant ran the tool against PHerc 0139 "photo-style"
plates — the official ink-detection map composited onto a papyrus-texture
background at partial opacity, built for publication-figure readability
(`release/pherc0139-column-atlas-gh/scripts/make_photo_plates.py`) — on
windows a human had already read as clear text, and got a flat FAIL. The
text was real; the input just wasn't the render family `E_FRAC_MIN` /
`STRUCTURE_AREAFRAC_MIN` / `OTSU_SEP_MIN` were fit on. That FAIL was
indistinguishable from "no letters here" — the same misleading-verdict
failure mode the sub-letter-window guard above already exists to prevent,
for a different cause.

**Signal.** Fraction of the whole submitted map's pixels below a fixed
darkness cut (`DARK_FRACTION_DARKNESS_CUT`, 40/255 normalized). Raw ds8
ink-detection renders carry real, confidently-blank background — near-black
after any reasonable contrast stretch. A photo-style composite deliberately
never gets that dark: its background is painted as light "paper" and even
fully-inked pixels are a blend against it, not true black.

**Where it's measured.** The signal needs real surrounding background,
which a single small claim crop may not have — so it is computed over
whichever array the tool is actually handed (the whole file for the CLI;
the padded crop for the web tool), gated by a minimum ratio to the claimed
bbox area (`RENDER_FAMILY_MIN_CONTEXT_RATIO = 16`). Below that ratio the
check is skipped (reported, non-gating) rather than risk a false "wrong
render family" verdict on a legitimately tight submission — which the
README's own Quickstart recommends ("a crop around your claim is enough").
In practice this means the guard is active for CLI users evaluating a
whole map/plate file (exactly the case that surfaced the bug), and usually
inactive for the web tool's own padded-crop submissions, whose context
ratio rarely clears 16x for a letter-scale claim.

**Calibration.** Measured 2026-07-13, whole-file, not per-window (a
tight window may not contain any true background regardless of family —
see "Where it's measured" above):

| Family | n | Source | Dark fraction |
|---|---|---|---|
| Raw (ds8 ink-detection maps) | 78 | 15 freshly-regenerated PHerc 0139 plates + the 63 unique source map files underlying the full 516-window calibration set, all three scrolls | min **0.420** |
| Photo-style composite | 15 | Every plate from the recipe above, one per wrap | max **0.236** |

`DARK_FRACTION_MIN = 0.30` sits with real margin on both sides (~0.12
below the raw min, ~0.06 above the photo max), biased toward protecting
real content: a missed detection just reverts to pre-v0.4 behavior
(evaluated normally against the four gates, which is where a photo
composite still usually fails anyway — see below), while a false detection
would incorrectly block a legitimate input, the worse failure of the two.

**Why not a per-window statistic.** The first, more naive version of this
check measured dark fraction on the claimed bbox itself (or a padded crop
around it), not the whole map. That version does NOT separate the two
families: real human-verified positive windows from the existing
516-window calibration set range from 0.156 to full coverage at that
granularity, overlapping the photo family's 0.130–0.234 range entirely.
The family signature only holds with enough real background in view —
whole plates, not tight per-claim windows. This was caught by validating
against the real 516-window calibration set specifically at the same crop
granularity `run_all_checks` actually uses, not by trusting an earlier,
easier-looking whole-image measurement — the same discipline that caught
v0.1's failures in the first place.

**What this does NOT do.** The four calibrated gates were never re-fit for
composites — there is no human-reviewed label set for "is there real text
in a photo-style composite" the way there is for raw maps, and inventing
thresholds without one would repeat the exact mistake `v0.1` made. A
detected render-family mismatch always returns "cannot evaluate", never a
composite-specific PASS.

**Honest limit.** This validates ONE compositing recipe against the raw
ds8 family, not composites in general — a differently-styled composite
(different opacity, different background treatment, a scan of a printed
figure) could evade detection and simply fall through to the four gates as
before, unchanged from pre-v0.4 behavior. Re-derive `DARK_FRACTION_MIN` if
a second composite style is ever confirmed.
`tests/test_vet_map.py::TestRenderFamilyGuard`.

## Speed

Removing the 200-sample same-map null from the verdict path also removed
its cost: one window went from **~200 s to ~1 s** (the DoG was being
recomputed for 200 null boxes per verdict). `--local-rarity` restores the
old percentile as reported context and pays that cost only when asked.

## Reproduce

```bash
python calibration/build_calibration_set.py       # rebuild role-aware registry
python calibration/record_source_hashes.py        # pin every real source raster
python calibration/replay_results.py              # replay all frozen real-data measurements
VETTING_CARD_REAL_CALIBRATION=1 python -m unittest tests.test_calibration
python calibration/validate_tool.py               # expensive full code sweep, release-time
```

Thresholds live in `card_config.py` (E_FRAC_MIN, STRUCTURE_AREAFRAC_MIN,
OTSU_SEP_MIN); the `config_hash()` in every card footer changes if any of
them move.

`source_manifest.json` records the exact 63 source-raster byte hashes used
by the committed registry. The repository intentionally does **not** claim
that a bare checkout can replay the image-processing sweep: it contains the
coordinates, frozen per-window measurements, and source identities, not a
redistributable copy of all rasters. A complete public replay requires a
release asset containing the reviewed crops or stable public URLs for every
source. Until then, `validate_tool.py` is reproducible on a checkout with
the byte-matching project data tree, and `replay_results.py` is an auditable
measurement replay rather than a fresh tool execution.
