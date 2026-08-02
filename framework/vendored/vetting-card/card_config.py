"""Vetting Card configuration and derived-parameter math.

The v0.3 *verdict gates* (``E_FRAC_MIN``, ``STRUCTURE_AREAFRAC_MIN``, and
``OTSU_SEP_MIN``) are calibrated on the real-data set described in
CALIBRATION.md: 324 fitting windows from Scroll 1 + PHerc 0139, with 96
external Scroll-4 validation windows. They are not synthetic, percentile,
or hand-picked gates.

Several operational constants below (for example the historical optional
same-map null sampler, input guards, and pipeline scan geometry) are design
parameters rather than fitted statistical gates. A PASS therefore means that
the supplied image survived these calibrated automated refutation checks; it
does not confirm ancient text or independently attest who generated an input.

Nothing here imports numpy/scipy/PIL -- this module is pure Python + stdlib
so it stays trivial to read, hash, and unit-test in isolation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool / schema identity
# ---------------------------------------------------------------------------

TOOL_VERSION = "0.4.0"       # v0.4: render-family guard (rejects the wrong image family, not the claim).
CARD_SCHEMA_VERSION = "v1"   # v1 adds input/evaluator provenance to verdict JSON.

# Fixed default RNG seed for all null-distribution sampling. Overridable via
# each CLI's --seed flag. PROVISIONAL: picked as an arbitrary, memorable
# constant (0), not tuned. The point is that it is FIXED, so a claimant's
# verdict is exactly reproducible by a reviewer re-running the same command.
DEFAULT_SEED = 0


def config_hash() -> str:
    """Short, stable fingerprint of this file's actual source bytes.

    Hashing the source (rather than hand-maintaining a list of constants to
    hash) means the fingerprint can never silently drift out of sync with a
    threshold change -- if this file changes at all, the hash changes.
    Truncated to 12 hex chars: collision risk is irrelevant here, this is a
    "did the config change" fingerprint for a vetting card footer, not a
    security primitive.
    """
    src = Path(__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Physical scale
# ---------------------------------------------------------------------------

# Documented default for --px-um. PROVISIONAL: matches a commonly used
# downsampled render resolution in this project's own working scans. A
# claimant MUST pass their own scan's real microns-per-pixel for the
# verdict to mean anything; this default only exists so the CLI has one.
DEFAULT_PX_UM = 8.0

# Greek letters in these ink maps run roughly 1.5-4mm tall, per the task
# brief this tool's checks were commissioned against (community convention
# for Herculaneum papyrus hands, not independently re-derived here).
# PROVISIONAL: a single fixed range for "a letter", ignoring per-scroll or
# per-hand variation.
LETTER_SIZE_MM_MIN = 1.5
LETTER_SIZE_MM_MAX = 4.0


def letter_scale_px_range(px_um: float) -> tuple[float, float]:
    """Convert the mm letter-size range to a pixel-size range at px_um."""
    if px_um <= 0:
        raise ValueError(f"px_um must be positive, got {px_um!r}")
    lo = (LETTER_SIZE_MM_MIN * 1000.0) / px_um
    hi = (LETTER_SIZE_MM_MAX * 1000.0) / px_um
    return lo, hi


# ---------------------------------------------------------------------------
# Null-distribution sampling (shared by letter_energy and structure checks)
# ---------------------------------------------------------------------------

# How many same-size random bboxes to sample from "the rest of the map" to
# build each null distribution. PROVISIONAL: 200 gives ~0.5-percentile
# resolution (1/200) at the percentiles this tool actually thresholds on
# (99th), while staying fast on a 16GB laptop with no GPU.
N_NULL_SAMPLES = 200

# Minimum valid null samples required before a verdict is trusted. Below
# this, the map is probably too small relative to the claimed bbox for the
# percentile statistic to mean anything. PROVISIONAL: 30 is the classic
# "large enough for CLT-flavored intuition" rule of thumb, not derived from
# this tool's actual statistics (which are exact rank percentiles, not
# t-tests -- kept anyway as a floor below which we refuse to answer).
MIN_NULL_SAMPLES = 30

# Cap on random-sampling attempts before giving up on gathering
# N_NULL_SAMPLES valid (in-bounds, non-overlapping-with-claim) boxes.
# PROVISIONAL: 50x oversampling is generous for any map where the claim
# bbox doesn't dominate the frame; purely a safety valve against infinite
# loops on pathological (claim-bbox-is-most-of-the-map) inputs.
MAX_NULL_SAMPLE_ATTEMPTS = N_NULL_SAMPLES * 50

# Margin around the claimed bbox excluded from null sampling, as a fraction
# of the claimed bbox's own (max) side length. PROVISIONAL: 0.5 means a
# null box has to clear the claim by half a bbox-width on every side --
# enough to avoid null boxes that mostly overlap the claim's own halo/blur
# footprint, without shrinking the eligible sampling area too much on
# modest-size maps.
EXCLUSION_MARGIN_FACTOR = 0.5

# ---------------------------------------------------------------------------
# CHECK: letter_energy (band-pass energy at letter scale)
# ---------------------------------------------------------------------------

# Percentile the claim bbox's band-pass energy must reach or exceed within
# the null distribution to PASS. PROVISIONAL, per the task brief's own
# example: 99th percentile means "more letter-scale energy than 99% of
# same-size random patches elsewhere on the same map" -- a strict bar
# chosen to keep the false-positive (pareidolia-passes) rate low at the
# cost of sensitivity; see tests/test_vet_map.py for the measured
# noise-only false-positive rate at this setting.
LETTER_ENERGY_PERCENTILE = 99.0

# Difference-of-Gaussians band-pass: blur at sigma_lo removes sub-stroke
# pixel noise, blur at sigma_hi smooths past the whole letter; their
# difference keeps energy concentrated at the letter length scale.
# PROVISIONAL: factors chosen so the pass-band bounds bracket typical
# stroke-width-to-letter-height ratios (roughly 1/6-1/3) without any
# empirical fit to real ink maps.
BANDPASS_SIGMA_LO_FACTOR = 0.25   # applied to the letter-scale MIN (px)
BANDPASS_SIGMA_HI_FACTOR = 0.50   # applied to the letter-scale MAX (px)


def bandpass_sigmas(px_um: float) -> tuple[float, float]:
    """Return (sigma_lo, sigma_hi) in pixels for the DoG band-pass filter."""
    lo_scale, hi_scale = letter_scale_px_range(px_um)
    sigma_lo = max(0.5, BANDPASS_SIGMA_LO_FACTOR * lo_scale)
    sigma_hi = max(sigma_lo * 1.5, BANDPASS_SIGMA_HI_FACTOR * hi_scale)
    return sigma_lo, sigma_hi


# ---------------------------------------------------------------------------
# CHECK: structure (letter-scale connected components)
# ---------------------------------------------------------------------------

# Percentile the claim bbox's in-range component COUNT must reach or exceed
# within the null distribution to PASS. PROVISIONAL, same reasoning and
# same numeric choice as LETTER_ENERGY_PERCENTILE (kept equal for v0
# simplicity -- one knob, not two, until real data says they should differ).
STRUCTURE_PERCENTILE = 99.0

# Adaptive threshold inside the bbox: binarize at (local mean + K * local
# std), computed from the bbox's OWN pixels (that's what makes it
# "adaptive" rather than one fixed global cut). PROVISIONAL: K=1.0 is the
# textbook "one standard deviation above the mean" starting point for
# picking out a bright minority against a background, not fit to real ink
# statistics.
STRUCTURE_THRESHOLD_K = 1.0

# A connected component's pixel AREA must fall in
# [MIN_FACTOR * letter_scale_px_min^2, MAX_FACTOR * letter_scale_px_max^2]
# to count as "letter-sized". Rationale: a single stroke fragment, once
# thresholded, covers only a fraction of the full letter bounding box
# (stroke width is much less than letter height), while a run of touching
# letters/blob covers a large multiple of it. PROVISIONAL: these factors
# are hand-picked from that geometric intuition (thin stroke vs. a couple
# of touching glyphs), not measured off real labeled letterforms --
# recalibrate once real traced-letter connected-component areas are
# available (see README Roadmap).
COMPONENT_AREA_MIN_FACTOR = 0.02
COMPONENT_AREA_MAX_FACTOR = 1.30


def component_area_px_range(px_um: float) -> tuple[float, float]:
    """Return (min_area_px, max_area_px) for a 'letter-sized' component."""
    lo_scale, hi_scale = letter_scale_px_range(px_um)
    min_area = COMPONENT_AREA_MIN_FACTOR * (lo_scale ** 2)
    max_area = COMPONENT_AREA_MAX_FACTOR * (hi_scale ** 2)
    return min_area, max_area


# ---------------------------------------------------------------------------
# CALIBRATED absolute gates (v0.2, 2026-07-11) -- these replace the same-map
# percentile as the VERDICT for the two statistical checks, and add a third
# check. Fit on windows from OFFICIAL ds8 ink maps of two scrolls
# (Scroll 1 / PHerc Paris 4 and PHerc 0139): 174 human-verified letterform
# positives vs 150 clean negatives (GPU-ranked fiber + far-from-gold
# background), 70/30 train/test split; thresholds chosen on train only at
# FPR <= 0.05. Held-out (fitted scrolls): 55/56 pass; validated on a
# third, fully-read scroll (0/60 confirmed-lacunae pass). Synthetic constraint:
# gaussian-blurred noise (sigma 12-30, the smooth-blob pareidolia texture)
# must not pass -- it is rejected by the bimodality gate (max observed 0.66
# vs threshold 0.885). Full derivation, dataset provenance and honest
# caveats: CALIBRATION.md + calibration/ in this repo.
# ---------------------------------------------------------------------------

# letter_energy gate: band-pass energy inside the bbox divided by the
# bbox's own pixel variance (contrast-invariant, so one absolute threshold
# can hold across maps with different normalization).
E_FRAC_MIN = 0.225

# structure gate: fraction of bbox pixels covered by connected components
# whose area falls in the letter band, after an adaptive threshold at
# (mean + STRUCTURE_AREAFRAC_K * std) of the bbox's own pixels. NOTE the
# v0 count-of-components statistic was measured ANTI-predictive on real
# ds8 maps (letterforms fuse into few components; fiber fragments into
# many) -- area fraction is the calibrated replacement.
STRUCTURE_AREAFRAC_K = 0.5
STRUCTURE_AREAFRAC_MIN = 0.13

# contrast_bimodality gate: Otsu between-class variance over total variance
# of the bbox. Real ink-on-papyrus windows are strongly bimodal
# (ink + background); smooth blob fields (the classic oversmoothed-output
# pareidolia texture) are unimodal and top out near 0.66.
OTSU_SEP_MIN = 0.885

# ---------------------------------------------------------------------------
# CHECK: saturation / degenerate input (hard fails, not percentile-based)
# ---------------------------------------------------------------------------

# A bbox (or the whole map) is "blank" if its normalized std is below this.
# PROVISIONAL: 1e-6 is essentially "bit-identical constant" in float64;
# chosen loose enough to still catch quantization-flat 8-bit PNG regions
# (which land on exact repeated integers after /255 normalization) without
# false-tripping on genuinely low-contrast-but-real texture.
BLANK_STD_EPS = 1e-6

# A bbox is "saturated" if more than this fraction of its (normalized)
# pixels sit within SATURATION_VALUE_EPS of 0 or of 1. PROVISIONAL: 0.98
# allows a thin real border/artifact rim while still catching maps that
# are mostly clipped at one rail (a common renderer bug in this project's
# own pipeline, per commit c1cdfd0 discussed in NEGATIVE_RESULTS.md --
# provenance note: that finding is about resolution collapse, not
# saturation, but it's the same "look at what actually reached the model"
# discipline this check exists to automate).
SATURATION_FRACTION_MAX = 0.98
SATURATION_VALUE_EPS = 1e-3

# ---------------------------------------------------------------------------
# CHECK: render_family (v0.4, 2026-07-13) -- is the SUBMITTED MAP the same
# kind of image the four verdict gates above were calibrated on (a raw,
# single-channel ink-detection probability render), or something else that
# happens to share a file format?
#
# Origin: a real claimant ran this tool against PHerc 0139 "photo-style"
# plates -- the official ink-detection map composited onto a papyrus-texture
# background at partial opacity for publication-figure readability (see
# release/pherc0139-column-atlas-gh/scripts/make_photo_plates.py) -- on
# windows a human had already read as clear text, and got a flat FAIL. The
# text was real; the input just wasn't the render family E_FRAC_MIN /
# STRUCTURE_AREAFRAC_MIN / OTSU_SEP_MIN were fit on. That FAIL was
# indistinguishable from "no letters here", exactly the misleading-verdict
# failure mode the sub-letter-window guard already exists to prevent for a
# different cause (see MIN_BBOX_LETTER_FRACTION below) -- this is the same
# fix applied to a second cause.
#
# Signal: fraction of the WHOLE submitted map's pixels below a fixed
# darkness cut. Raw ds8 ink-detection renders carry real, confidently-blank
# background (near-black after any reasonable contrast stretch); a
# photo-style composite deliberately never gets that dark -- its
# background is painted as light "paper" and even fully-inked pixels are a
# blend against it, not true black.
#
# Measured 2026-07-13, whole-file (not per-window; the signal needs real
# surrounding background, which a single small crop may not have -- see
# RENDER_FAMILY_MIN_CONTEXT_RATIO):
#   - RAW family, n=78 (15 freshly-regenerated PHerc 0139 ds8 plates + the
#     63 unique source map files underlying the full 516-window calibration
#     set, spanning all three calibration scrolls): dark fraction min 0.420.
#   - PHOTO family, n=15 (every plate from the recipe above, one per wrap):
#     dark fraction max 0.236.
# DARK_FRACTION_MIN sits with real margin on both sides (~0.12 below the
# raw min, ~0.06 above the photo max), biased toward protecting real
# content: a missed detection just reverts to the pre-v0.4 behavior
# (evaluated normally against the four gates), while a false detection
# would incorrectly block a legitimate input, which is the worse failure.
#
# HONEST LIMIT: this validates ONE compositing recipe against the raw ds8
# family, not "composites in general" -- a differently-styled composite
# (different opacity, different background treatment, a scan of a printed
# figure, etc.) could evade detection and simply fall through to the four
# gates as before. Re-derive DARK_FRACTION_MIN if a second composite style
# is ever confirmed. See CALIBRATION.md.
# ---------------------------------------------------------------------------

DARK_FRACTION_DARKNESS_CUT = 40.0 / 255.0
DARK_FRACTION_MIN = 0.30

# The render-family signal is only trusted when the submitted array has
# enough area beyond the claimed bbox to plausibly contain real background
# -- it was measured only on whole map/plate files (tens to thousands of
# times the bbox area), never on a tightly-cropped claim image. Below this
# ratio the check is skipped (reported, not gating) rather than risk a
# false "wrong render family" verdict on a legitimately small submission --
# which the tool's own Quickstart explicitly recommends ("a crop around
# your claim is enough"). PROVISIONAL: 16x (4x per side) is a guess at
# "probably has some real background in it", not independently measured at
# the boundary -- the validated regime is whole plates, ratios in the
# hundreds to thousands.
RENDER_FAMILY_MIN_CONTEXT_RATIO = 16.0

# ---------------------------------------------------------------------------
# Minimum sizes (hard errors, not verdict fails -- the tool can't run at all)
# ---------------------------------------------------------------------------

# A map smaller than this in either dimension is rejected outright.
# PROVISIONAL: 32px is roughly "big enough to hold one letter-scale bbox
# plus a sliver of surrounding context" at a coarse px_um; real scans are
# always far larger, this only guards against toy/corrupt inputs.
MIN_MAP_DIM_PX = 32

# A claimed bbox smaller than this in either dimension is rejected outright
# (band-pass filtering and connected-component statistics are not
# meaningful on a handful of pixels). PROVISIONAL, same reasoning as
# MIN_MAP_DIM_PX.
MIN_BBOX_DIM_PX = 8


# A claimed bbox smaller than ONE minimum-size letter at the given px_um
# cannot physically contain a letter, so any letter-content verdict on it is
# meaningless -- the tool returns "cannot evaluate" (status error) rather
# than a misleading FAIL. This is the calibrated guard against the #1 usage
# error (wrong px_um / sub-letter window): a real known-text window at
# native 2.4 um/px passes at letter scale but "fails" if cropped to ~1mm.
# Factor 1.0 = require at least the smallest letter (LETTER_SIZE_MM_MIN).
MIN_BBOX_LETTER_FRACTION = 1.0


def min_bbox_side_px(px_um: float) -> float:
    """Smallest allowed bbox side (px): one minimum-size letter at px_um."""
    lo, _ = letter_scale_px_range(px_um)
    return MIN_BBOX_LETTER_FRACTION * lo

# ---------------------------------------------------------------------------
# vet_pipeline.py: grid-scan grading of a claimant's whole pipeline
# ---------------------------------------------------------------------------

# Candidate scan-window side length, as a multiple of the letter-scale MAX
# (px). PROVISIONAL: 1.5x is "a little more than one big letter" -- large
# enough that a single stroke fragment landing at a window edge still has
# most of its mass inside some window, small enough that the scan stays
# fine-grained relative to typical fixture crop sizes.
PIPELINE_SCAN_WINDOW_LETTERSCALES = 1.5

# Scan stride as a fraction of the scan-window side length. PROVISIONAL:
# 0.5 (50% overlap) is the standard sliding-window compromise between scan
# density and runtime.
PIPELINE_SCAN_STRIDE_FRACTION = 0.5

# vet_pipeline pass rule (PROVISIONAL, per task brief): a claimant's
# pipeline passes only if it fires on NONE of the blank fixtures and on at
# least this fraction of the positive fixtures.
PIPELINE_MAX_PAREIDOLIA_RATE = 0.0
PIPELINE_MIN_SENSITIVITY = 0.5
