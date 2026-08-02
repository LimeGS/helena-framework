#!/usr/bin/env python3
"""vet_map.py -- anti-pareidolia checks for a single claimed letterform
bbox on a submitted 2D ink map. v0.3 verdicts are absolute gates fit on
324 labeled fitting windows from official Scroll 1 / PHerc 0139 ds8 maps,
then externally validated on 96 Scroll-4 windows (CALIBRATION.md). The v0
same-map percentile survives only as optional, non-gating context
(--local-rarity). Verdict JSON records SHA-256 hashes for exact replay.

    python vet_map.py --map claim_map.npy --bbox x0,y0,x1,y1 \\
        [--px-um 8.0] [--seed 0] [--local-rarity] --out verdict.json

Four VERDICT checks run against the claimed bbox, gated behind one
whole-map PRE-check (see card_config.py for every threshold's provenance):

  render_family        -- v0.4: does the SUBMITTED MAP look like the raw
                          ink-detection render family the four checks below
                          were calibrated on? Raises "cannot evaluate" (not
                          a misleading FAIL) on inputs like photo-style
                          composites, which fail the calibrated gates for
                          reasons that have nothing to do with whether
                          there's real text. Skipped -- not gating -- when
                          the map isn't large enough relative to the bbox
                          for the signal to be trustworthy.
  degenerate           -- hard fail if the bbox is blank, saturated, or the
                          whole map is constant.
  letter_energy        -- band-pass (difference-of-Gaussians) energy at the
                          letter length scale, divided by the bbox's own
                          variance (contrast-invariant). Pass = value >=
                          E_FRAC_MIN (calibrated).
  structure            -- fraction of bbox pixels covered by connected
                          components with letter-band areas after an
                          adaptive threshold. Pass = value >=
                          STRUCTURE_AREAFRAC_MIN (calibrated). The v0
                          component COUNT was measured anti-predictive on
                          real ds8 maps (letters fuse, fiber fragments) and
                          is reported for context only.
  contrast_bimodality  -- Otsu between-class/total variance of the bbox.
                          Real ink+background windows are bimodal; smooth
                          blob fields (oversmoothed-output pareidolia) are
                          unimodal and cannot reach the calibrated gate.

Two distinct outcomes, both always written as JSON to --out:

  status "ok"    -- the tool ran to completion; check the "overall.pass"
                     field for the actual verdict (which may be False --
                     that is a normal, informative result, not an error).
  status "error" -- the input could not be evaluated at all (bad bbox,
                     map too small, unreadable file, wrong render family).
                     CLI exit code 2. overall.pass is always False in this
                     case too, so downstream tooling that only looks at
                     overall.pass never has to special-case this.

This module is written to be imported, not just run: vet_pipeline.py
reuses check_letter_energy/sample_null_bboxes/etc. directly rather than
re-implementing the same statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

import card_config

Bbox = tuple[int, int, int, int]  # (x0, y0, x1, y1), x1/y1 exclusive


class VetMapError(Exception):
    """Raised for inputs the tool cannot evaluate at all (not a verdict)."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of exact on-disk bytes without loading them at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_provenance() -> dict:
    """Identify the evaluator source that produced a verdict.

    This is replay provenance, not a signature: a reviewer still has to run
    the published code against input bytes matching the recorded digest.
    """
    path = Path(__file__)
    return {"source": path.name, "source_sha256": sha256_file(path)}


# ---------------------------------------------------------------------------
# Loading and normalizing
# ---------------------------------------------------------------------------

def load_map(path) -> np.ndarray:
    """Load a .npy (any float/int dtype) or 8-bit .png as a raw 2D float64
    array. No normalization here -- callers need the raw dynamic range to
    detect degenerate (constant) inputs before any rescaling hides them.
    """
    path = Path(path)
    if not path.exists():
        raise VetMapError(f"map file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".png":
        # Multi-channel PNGs are flattened to 8-bit grayscale luminance.
        arr = np.array(Image.open(path).convert("L"))
    else:
        raise VetMapError(
            f"unsupported map format {suffix!r} for {path.name}: expected .npy or .png"
        )
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise VetMapError(
            f"expected a 2D map after loading {path.name}, got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise VetMapError(f"{path.name} contains NaN/Inf values; cannot evaluate")
    return arr.astype(np.float64)


def normalize01(raw: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Caller must have already handled the
    degenerate constant-map case (hi <= lo here would otherwise silently
    return all zeros).
    """
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi <= lo:
        return np.zeros_like(raw, dtype=np.float64)
    return (raw - lo) / (hi - lo)


def parse_bbox(text: str) -> Bbox:
    parts = text.split(",")
    if len(parts) != 4:
        raise VetMapError(f"--bbox must be 'x0,y0,x1,y1', got {text!r}")
    try:
        x0, y0, x1, y1 = (int(round(float(p))) for p in parts)
    except ValueError as exc:
        raise VetMapError(f"--bbox must be 4 numbers 'x0,y0,x1,y1', got {text!r}") from exc
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Validation (raises VetMapError -> CLI "error" status, exit 2)
# ---------------------------------------------------------------------------

def validate_map_size(raw: np.ndarray, cfg=card_config) -> None:
    h, w = raw.shape
    if h < cfg.MIN_MAP_DIM_PX or w < cfg.MIN_MAP_DIM_PX:
        raise VetMapError(
            f"map too small: {w}x{h}px, minimum is {cfg.MIN_MAP_DIM_PX}x{cfg.MIN_MAP_DIM_PX}px"
        )


def validate_bbox(bbox: Bbox, shape: tuple[int, int], cfg=card_config) -> None:
    x0, y0, x1, y1 = bbox
    h, w = shape
    if not (x1 > x0 and y1 > y0):
        raise VetMapError(f"invalid bbox {bbox}: require x1>x0 and y1>y0")
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise VetMapError(
            f"bbox {bbox} exceeds map bounds (map is {w}x{h}px): "
            "bbox is larger than the map, or out of range"
        )
    if (x1 - x0) < cfg.MIN_BBOX_DIM_PX or (y1 - y0) < cfg.MIN_BBOX_DIM_PX:
        raise VetMapError(f"bbox {bbox} is too small: minimum side is {cfg.MIN_BBOX_DIM_PX}px")


# ---------------------------------------------------------------------------
# Null-distribution sampling (shared by both statistical checks)
# ---------------------------------------------------------------------------

def sample_null_bboxes(
    shape: tuple[int, int],
    claim_bbox: Bbox,
    n: int,
    margin_factor: float,
    rng: np.random.Generator,
    max_attempts: int,
    min_required: int,
) -> list[Bbox]:
    """Sample up to n same-size bboxes from `shape`, excluding claim_bbox
    expanded by margin_factor * max(claim side lengths) on every side.
    """
    h, w = shape
    x0, y0, x1, y1 = claim_bbox
    bw, bh = x1 - x0, y1 - y0
    margin = margin_factor * max(bw, bh)
    ex0, ey0, ex1, ey1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin

    result: list[Bbox] = []
    attempts = 0
    max_x0, max_y0 = w - bw, h - bh
    while len(result) < n and attempts < max_attempts:
        attempts += 1
        nx0 = int(rng.integers(0, max_x0 + 1))
        ny0 = int(rng.integers(0, max_y0 + 1))
        nx1, ny1 = nx0 + bw, ny0 + bh
        overlaps = nx0 < ex1 and nx1 > ex0 and ny0 < ey1 and ny1 > ey0
        if overlaps:
            continue
        result.append((nx0, ny0, nx1, ny1))

    if len(result) < min_required:
        raise VetMapError(
            f"could not sample enough null bboxes ({len(result)}/{min_required} minimum): "
            "the map is too small relative to the claimed bbox for a reliable null distribution"
        )
    return result


def percentile_of(value: float, null_values) -> float:
    """Fraction (as a 0-100 percentile) of null_values strictly < value.

    Strict "<" (not "<=") matters for degenerate inputs: if a claim and its
    whole null population are all exactly tied (e.g. a claimant's honest
    all-zero output on a truly blank region), "<=" would score that as the
    100th percentile -- a tie-inflation artifact that would make an honest
    "nothing here" output look like a pass. With strict "<", a value tied
    with its null population scores 0 (correctly: it does not stand out
    from the null distribution, which is exactly what should not pass).
    """
    arr = np.asarray(list(null_values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(100.0 * np.mean(arr < value))


def summarize(values) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# CHECK: letter_energy
# ---------------------------------------------------------------------------

def bandpass_energy(map01: np.ndarray, bbox: Bbox, sigma_lo: float, sigma_hi: float) -> float:
    """Mean squared difference-of-Gaussians response inside bbox.

    Filters a padded crop around bbox (not the whole map, for speed on
    large inputs) so the Gaussian blur sees real neighboring pixels rather
    than being cut off exactly at the bbox edge, falling back to scipy's
    own boundary handling only at the true map border.

    That boundary handling is mode="mirror", not the arguably more obvious
    "nearest". This was measured, not assumed: mode="nearest" replicates
    the same edge pixel outward in both directions at once at a bbox that
    touches TWO map edges simultaneously (a true corner), which flattens
    the padded region the wide ("hi") filter partially averages over far
    more than it flattens what the narrow ("lo") filter sees -- widening
    their difference and inflating measured energy specifically at
    corners (empirically, on pure-noise fixtures at this module's default
    letter scale, roughly +22% at a true corner vs. a same-shaped interior
    bbox). mode="mirror" reflects the real local texture across the
    boundary instead of replicating a flat constant, cutting that same gap
    to roughly +6%. Every real map has exactly four true corners -- a
    claimant's fixture drop is no exception -- so this is a real accuracy
    fix for near-border claims, not just a synthetic-test artifact.
    """
    x0, y0, x1, y1 = bbox
    h, w = map01.shape
    pad = int(np.ceil(sigma_hi * 3))
    px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
    px1, py1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = map01[py0:py1, px0:px1]
    lo = ndimage.gaussian_filter(crop, sigma_lo, mode="mirror")
    hi = ndimage.gaussian_filter(crop, sigma_hi, mode="mirror")
    band = lo - hi
    bx0, by0 = x0 - px0, y0 - py0
    bx1, by1 = bx0 + (x1 - x0), by0 + (y1 - y0)
    band_bbox = band[by0:by1, bx0:bx1]
    return float(np.mean(band_bbox ** 2))


def check_letter_energy(
    map01: np.ndarray, bbox: Bbox, px_um: float,
    null_bboxes: Optional[list[Bbox]] = None, cfg=card_config,
) -> dict:
    """VERDICT (v0.2): absolute, calibrated gate on the contrast-invariant
    energy fraction (band-pass energy / bbox variance). The v0 same-map
    percentile is reported as optional non-gating context ("local rarity")
    when null_bboxes are provided: on maps that are letters everywhere, a
    real-text bbox is NOT locally rare, so rarity must not gate (measured;
    see CALIBRATION.md and README "Known limitations" history).
    """
    sigma_lo, sigma_hi = cfg.bandpass_sigmas(px_um)
    x0, y0, x1, y1 = bbox
    patch_var = float(map01[y0:y1, x0:x1].var())
    value = bandpass_energy(map01, bbox, sigma_lo, sigma_hi)
    e_frac = value / patch_var if patch_var > 0 else 0.0
    result = {
        "pass": bool(e_frac >= cfg.E_FRAC_MIN),
        "value": e_frac,
        "threshold": cfg.E_FRAC_MIN,
        "raw_band_energy": value,
        "patch_var": patch_var,
    }
    if null_bboxes is not None:
        null_values = []
        for nb in null_bboxes:
            nx0, ny0, nx1, ny1 = nb
            nvar = float(map01[ny0:ny1, nx0:nx1].var())
            ne = bandpass_energy(map01, nb, sigma_lo, sigma_hi)
            null_values.append(ne / nvar if nvar > 0 else 0.0)
        result["local_rarity"] = {
            "percentile": percentile_of(e_frac, null_values),
            "null_stats": summarize(null_values),
            "note": "context only, does not gate (top-1%-of-own-map is the wrong bar on text-dense maps)",
        }
    return result


# ---------------------------------------------------------------------------
# CHECK: structure
# ---------------------------------------------------------------------------

def structure_component_count(map01: np.ndarray, bbox: Bbox, px_um: float, cfg=card_config) -> int:
    """v0 statistic, kept for reporting only: measured ANTI-predictive on
    real ds8 maps (see CALIBRATION.md) -- letterforms fuse, fiber fragments."""
    x0, y0, x1, y1 = bbox
    patch = map01[y0:y1, x0:x1]
    threshold = patch.mean() + cfg.STRUCTURE_THRESHOLD_K * patch.std()
    binary = patch > threshold
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return 0
    sizes = np.bincount(labeled.ravel())[1:]  # drop background label 0
    lo_area, hi_area = cfg.component_area_px_range(px_um)
    return int(np.sum((sizes >= lo_area) & (sizes <= hi_area)))


def structure_area_fraction(map01: np.ndarray, bbox: Bbox, px_um: float, cfg=card_config) -> float:
    """v0.2 verdict statistic: fraction of bbox pixels covered by connected
    components with letter-band areas, adaptive threshold at
    mean + STRUCTURE_AREAFRAC_K * std of the bbox's own pixels."""
    x0, y0, x1, y1 = bbox
    patch = map01[y0:y1, x0:x1]
    threshold = patch.mean() + cfg.STRUCTURE_AREAFRAC_K * patch.std()
    labeled, n_components = ndimage.label(patch > threshold)
    if n_components == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    lo_area, hi_area = cfg.component_area_px_range(px_um)
    inband = (sizes >= lo_area) & (sizes <= hi_area)
    return float(sizes[inband].sum() / patch.size)


def otsu_separability(map01: np.ndarray, bbox: Bbox) -> float:
    """Otsu between-class variance / total variance of the bbox, in [0, 1].
    Bimodal ink-plus-background windows score high; smooth unimodal blob
    fields (the oversmoothed-output pareidolia texture) top out ~0.66."""
    x0, y0, x1, y1 = bbox
    patch = map01[y0:y1, x0:x1]
    total_var = float(patch.var())
    if total_var <= 0:
        return 0.0
    hist, edges = np.histogram(patch, bins=64)
    p = hist / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * (1.0 - w0))
    return float(np.nanmax(between) / total_var)


def check_structure(
    map01: np.ndarray, bbox: Bbox, px_um: float,
    null_bboxes: Optional[list[Bbox]] = None, cfg=card_config,
) -> dict:
    """VERDICT (v0.2): absolute gate on in-band component AREA FRACTION.
    The v0 count statistic is reported for context but measured
    anti-predictive on real ds8 maps (letters fuse, fiber fragments)."""
    value = structure_area_fraction(map01, bbox, px_um, cfg)
    lo_area, hi_area = cfg.component_area_px_range(px_um)
    result = {
        "pass": bool(value >= cfg.STRUCTURE_AREAFRAC_MIN),
        "value": value,
        "threshold": cfg.STRUCTURE_AREAFRAC_MIN,
        "component_area_px_range": [lo_area, hi_area],
        "v0_component_count": structure_component_count(map01, bbox, px_um, cfg),
        "v0_count_note": "reported only; anti-predictive on real ds8 maps (CALIBRATION.md)",
    }
    if null_bboxes is not None:
        null_values = [structure_area_fraction(map01, b, px_um, cfg) for b in null_bboxes]
        result["local_rarity"] = {
            "percentile": percentile_of(value, null_values),
            "null_stats": summarize(null_values),
            "note": "context only, does not gate",
        }
    return result


def check_contrast_bimodality(map01: np.ndarray, bbox: Bbox, cfg=card_config) -> dict:
    """VERDICT (v0.2): Otsu separability gate. Kills the smooth-blob
    pareidolia family that energy+structure alone let through (measured:
    blurred noise sigma 12-30 tops out at 0.66 vs real-letter p10 0.90)."""
    value = otsu_separability(map01, bbox)
    return {
        "pass": bool(value >= cfg.OTSU_SEP_MIN),
        "value": value,
        "threshold": cfg.OTSU_SEP_MIN,
    }


# ---------------------------------------------------------------------------
# CHECK: render_family (whole-map sniff test, v0.4 -- see card_config.py)
# ---------------------------------------------------------------------------

def dark_fraction(map01: np.ndarray, cfg=card_config) -> float:
    """Fraction of the WHOLE array's pixels below a fixed normalized
    darkness cut. A property of the submitted map's render family, not of
    the claimed bbox -- see card_config.DARK_FRACTION_MIN."""
    return float(np.mean(map01 < cfg.DARK_FRACTION_DARKNESS_CUT))


def check_render_family(map01: np.ndarray, bbox: Bbox, cfg=card_config) -> dict:
    """Sniffs whether the SUBMITTED MAP (not just the claimed bbox) looks
    like the raw ink-detection-map render family the four verdict gates
    were calibrated on, or something else (e.g. a photo-style composite --
    see CALIBRATION.md). Skipped when the map isn't large enough relative
    to the bbox to trust the signal (card_config.RENDER_FAMILY_MIN_CONTEXT_RATIO)."""
    x0, y0, x1, y1 = bbox
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    ratio = map01.size / bbox_area
    if ratio < cfg.RENDER_FAMILY_MIN_CONTEXT_RATIO:
        return {
            "pass": True,
            "skipped": True,
            "reason": (
                f"submitted map is only {ratio:.1f}x the claimed bbox area "
                f"(need >= {cfg.RENDER_FAMILY_MIN_CONTEXT_RATIO:.0f}x); not enough "
                "surrounding context to trust the render-family signal"
            ),
        }
    value = dark_fraction(map01, cfg)
    return {
        "pass": bool(value >= cfg.DARK_FRACTION_MIN),
        "value": value,
        "threshold": cfg.DARK_FRACTION_MIN,
        "context_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# CHECK: degenerate / saturation (hard fail, no null sampling)
# ---------------------------------------------------------------------------

def check_degenerate(map01: np.ndarray, bbox: Bbox, map_is_constant: bool, cfg=card_config) -> dict:
    x0, y0, x1, y1 = bbox
    patch = map01[y0:y1, x0:x1]
    blank = bool(patch.std() < cfg.BLANK_STD_EPS)
    frac_hi = float(np.mean(patch >= 1.0 - cfg.SATURATION_VALUE_EPS))
    frac_lo = float(np.mean(patch <= cfg.SATURATION_VALUE_EPS))
    saturated = bool(max(frac_hi, frac_lo) > cfg.SATURATION_FRACTION_MAX)

    reasons = []
    if map_is_constant:
        reasons.append("the whole map is constant (no variation anywhere)")
    if blank:
        reasons.append("the bbox is blank (near-constant within the claimed region)")
    if saturated:
        pinned = max(frac_hi, frac_lo) * 100.0
        reasons.append(f"the bbox is saturated ({pinned:.1f}% of pixels pinned at one extreme)")

    ok = not (map_is_constant or blank or saturated)
    return {
        "pass": ok,
        "map_constant": map_is_constant,
        "blank": blank,
        "saturated": saturated,
        "message": "; ".join(reasons) if reasons else "ok",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_checks(
    raw: np.ndarray, bbox: Bbox, px_um: float, cfg=card_config,
    seed: Optional[int] = None, local_rarity: bool = False,
) -> dict:
    """Run every check for one (map, bbox) pair. Raises VetMapError for
    inputs that cannot be evaluated at all; returns {"checks", "overall"}
    otherwise (overall.pass may legitimately be False).
    """
    validate_map_size(raw, cfg)
    validate_bbox(bbox, raw.shape, cfg)

    x0, y0, x1, y1 = bbox
    min_side = cfg.min_bbox_side_px(px_um)
    if min(x1 - x0, y1 - y0) < min_side:
        raise VetMapError(
            f"bbox side {min(x1 - x0, y1 - y0)}px is smaller than one letter "
            f"({min_side:.0f}px = {cfg.LETTER_SIZE_MM_MIN}mm at {px_um} um/px). "
            "This window is too small to contain a letter, so a letter-content "
            "verdict is meaningless. Check --px-um (is it your scan's real "
            "microns-per-pixel?) and use a window of at least ~1 cm."
        )

    map_is_constant = bool(np.max(raw) == np.min(raw))
    map01 = np.zeros_like(raw, dtype=np.float64) if map_is_constant else normalize01(raw)

    render_family = None
    if not map_is_constant:
        render_family = check_render_family(map01, bbox, cfg)
        if not render_family.get("skipped") and not render_family["pass"]:
            raise VetMapError(
                "cannot evaluate: this input's overall tonal profile does not match "
                "the render family the four verdict gates were calibrated on (raw "
                "single-channel ink-detection maps). Whole-map dark-pixel fraction "
                f"{render_family['value']:.3f} is below {cfg.DARK_FRACTION_MIN} "
                "(78 real raw-family maps measured >=0.42, three scrolls). This "
                "usually means the image is a photo-style composite, colorized "
                "rendering, or otherwise not the model's raw probability output -- "
                "re-run against the underlying ink-detection map if you have one. "
                f"(context ratio {render_family['context_ratio']:.1f}x; see "
                "CALIBRATION.md)"
            )

    degenerate = check_degenerate(map01, bbox, map_is_constant, cfg)
    checks = {}
    if render_family is not None:
        checks["render_family"] = render_family
    checks["degenerate"] = degenerate

    if not degenerate["pass"]:
        skip = {"pass": False, "skipped": True, "reason": "skipped: degenerate input, see 'degenerate' check"}
        checks["letter_energy"] = skip
        checks["structure"] = dict(skip)
        checks["contrast_bimodality"] = dict(skip)
        overall_pass = False
    else:
        null_bboxes = None
        if local_rarity:
            rng = np.random.default_rng(cfg.DEFAULT_SEED if seed is None else seed)
            null_bboxes = sample_null_bboxes(
                raw.shape, bbox, cfg.N_NULL_SAMPLES, cfg.EXCLUSION_MARGIN_FACTOR,
                rng, cfg.MAX_NULL_SAMPLE_ATTEMPTS, cfg.MIN_NULL_SAMPLES,
            )
        checks["letter_energy"] = check_letter_energy(map01, bbox, px_um, null_bboxes, cfg)
        checks["structure"] = check_structure(map01, bbox, px_um, null_bboxes, cfg)
        checks["contrast_bimodality"] = check_contrast_bimodality(map01, bbox, cfg)
        overall_pass = (checks["letter_energy"]["pass"]
                        and checks["structure"]["pass"]
                        and checks["contrast_bimodality"]["pass"])

    return {"checks": checks, "overall": {"pass": bool(overall_pass)}}


def build_result(map_path, bbox: Optional[Bbox], px_um: float, seed: Optional[int], cfg=card_config, local_rarity: bool = False) -> dict:
    result = {
        "schema_version": cfg.CARD_SCHEMA_VERSION,
        "tool": "vet_map.py",
        "tool_version": cfg.TOOL_VERSION,
        "config_hash": cfg.config_hash(),
        "evaluator": evaluator_provenance(),
        "input": {
            "map": str(map_path),
            "bbox": list(bbox) if bbox is not None else None,
            "px_um": px_um,
            "seed": cfg.DEFAULT_SEED if seed is None else seed,
        },
    }
    try:
        raw = load_map(map_path)
        path = Path(map_path)
        result["input"].update({
            "map_sha256": sha256_file(path),
            "map_size_bytes": path.stat().st_size,
            "map_shape": list(raw.shape),
            "map_dtype": str(raw.dtype),
        })
        outcome = run_all_checks(raw, bbox, px_um, cfg=cfg, seed=seed, local_rarity=local_rarity)
        result["status"] = "ok"
        result["checks"] = outcome["checks"]
        result["overall"] = outcome["overall"]
    except VetMapError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["overall"] = {"pass": False}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vet_map.py",
        description="Anti-pareidolia statistical checks for one claimed letterform bbox.",
    )
    parser.add_argument("--map", required=True, help="path to a .npy (float) or .png (8-bit) ink map")
    parser.add_argument("--bbox", required=True, help="claimed letterform bbox as x0,y0,x1,y1 (pixels)")
    parser.add_argument(
        "--px-um", type=float, default=card_config.DEFAULT_PX_UM,
        help=f"microns per pixel of --map (default {card_config.DEFAULT_PX_UM}; pass your scan's real value)",
    )
    parser.add_argument("--seed", type=int, default=None, help="null-sampling RNG seed (default: card_config.DEFAULT_SEED)")
    parser.add_argument("--card", default=None, metavar="PATH",
                        help="also render the Markdown vetting card here (same as piping "
                             "the verdict through make_card.py; one command instead of two)")
    parser.add_argument("--local-rarity", action="store_true",
                        help="also report the v0 same-map percentile as context (slow: samples "
                             "null windows; NEVER gates the verdict -- see CALIBRATION.md)")
    parser.add_argument("--out", required=True, help="path to write the verdict JSON")
    return parser


def main(argv=None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    try:
        bbox = parse_bbox(args.bbox)
    except VetMapError as exc:
        result = {
            "schema_version": card_config.CARD_SCHEMA_VERSION,
            "tool": "vet_map.py",
            "tool_version": card_config.TOOL_VERSION,
            "config_hash": card_config.config_hash(),
            "evaluator": evaluator_provenance(),
            "input": {"map": args.map, "bbox": args.bbox, "px_um": args.px_um, "seed": args.seed},
            "status": "error",
            "error": str(exc),
            "overall": {"pass": False},
        }
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = build_result(args.map, bbox, args.px_um, args.seed, local_rarity=args.local_rarity)
    Path(args.out).write_text(json.dumps(result, indent=2))

    if args.card:
        from make_card import render_card  # lazy: only when asked
        Path(args.card).write_text(render_card(result, None))
        print(f"card: wrote {args.card}")

    if result["status"] == "error":
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 2

    verdict = "PASS" if result["overall"]["pass"] else "FAIL"
    print(f"{verdict}: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
