#!/usr/bin/env python3
"""Quantify and visualize stability of a Phase 4 ink-screening ensemble.

This tool is deliberately diagnostic.  It never calls a pixel, component, or
crop "ink" or a "letter"; it ranks persistent model activations for raw-CT
review.  Probabilities are kept on their original sigmoid scale.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

_CONTRACT_ROOT = Path(__file__).resolve().parents[4]
if str(_CONTRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_ROOT))

from framework.contracts.slice_order import (  # noqa: E402
    NUMERIC_STEM_INDEX,
    resolve_tiff_slice,
)


def ensure_stage03_helper_on_path() -> Path:
    """Expose the declared Stage 03 helper to a standalone Stage 04 run."""
    helper_directory = Path(__file__).resolve().parents[2] / "03-ink" / "scripts"
    helper = helper_directory / "annotate_glyph_candidates.py"
    if not helper.is_file():
        raise RuntimeError(f"required Stage 03 glyph helper is missing: {helper}")
    text = str(helper_directory)
    if text not in sys.path:
        sys.path.insert(0, text)
    return helper_directory


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_maps(screening_dir: Path) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(screening_dir.glob("center-*_offset-*.npy"))
    if len(paths) < 2:
        raise RuntimeError("at least two screening maps are required")
    maps = [np.load(path).astype(np.float32) for path in paths]
    shapes = {value.shape for value in maps}
    if len(shapes) != 1:
        raise RuntimeError("screening maps do not share one shape")
    return np.stack(maps), paths


def parse_map_coordinates(path: Path) -> tuple[int, int]:
    match = re.fullmatch(
        r"center-(\d+)_offset-(\d+)\.npy",
        path.name,
    )
    if match is None:
        raise ValueError(f"unexpected screening map name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def group_maps_by_depth(
    maps: np.ndarray,
    paths: list[Path],
) -> list[tuple[int, np.ndarray]]:
    """Return the mean response across tiling offsets for each sampled depth."""

    if maps.shape[0] != len(paths):
        raise ValueError("map count and path count differ")
    grouped: dict[int, list[np.ndarray]] = {}
    for value, path in zip(maps, paths):
        center, _ = parse_map_coordinates(path)
        grouped.setdefault(center, []).append(value)
    return [
        (center, np.stack(values).mean(axis=0))
        for center, values in sorted(grouped.items())
    ]


def common_valid_mask(maps: np.ndarray) -> np.ndarray:
    if maps.ndim != 3:
        raise ValueError("maps must have shape run,y,x")
    return np.all(np.isfinite(maps) & (maps > 0), axis=0)


def group_depth_maps_by_offset(
    maps: np.ndarray,
    paths: list[Path],
) -> dict[int, np.ndarray]:
    """Return one conservative map per tiling offset.

    The pixel value is the minimum response across all sampled depths at that
    offset, so a signal disappears if it is only present at one depth.
    """

    if maps.shape[0] != len(paths):
        raise ValueError("map count and path count differ")
    grouped: dict[int, list[np.ndarray]] = {}
    for value, path in zip(maps, paths):
        _, offset = parse_map_coordinates(path)
        grouped.setdefault(offset, []).append(value)
    if len(grouped) < 2:
        raise ValueError("at least two tiling offsets are required")
    return {
        offset: np.stack(values).min(axis=0)
        for offset, values in sorted(grouped.items())
    }


def glyph_like_support(
    maps: np.ndarray,
    paths: list[Path],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Screen for bounded two-offset shapes arranged in row-like bands."""

    ensure_stage03_helper_on_path()
    from annotate_glyph_candidates import (
        STRICT_SCREEN_POLICY,
        discover_glyph_candidates,
        strict_text_like_screen,
    )

    grouped = group_depth_maps_by_offset(maps, paths)
    offsets = sorted(grouped)
    first, second = grouped[offsets[0]], grouped[offsets[1]]
    persistent, candidates, bands = discover_glyph_candidates(
        first,
        second,
        threshold=threshold,
    )
    # FIX-08: one canonical strict criterion, owned by the Stage 03 helper.
    screen = strict_text_like_screen(candidates)
    return {
        "threshold": threshold,
        "offsets_compared": offsets[:2],
        "depth_aggregation": "minimum probability at each tiling offset",
        "persistent_pixels": int(persistent.sum()),
        "glyph_like_candidate_count": screen["glyph_like_candidate_count"],
        "row_band_count": screen["row_band_count"],
        "candidate_count_by_row": screen["candidate_count_by_row"],
        "rows_with_at_least_four_candidates": screen[
            "rows_with_at_least_four_candidates"
        ],
        "strict_screen": screen,
        "detected_band_count": len(bands),
        "screening_outcome": (
            "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"
            if screen["qualifies"]
            else "INSUFFICIENT_TEXT_LIKE_SUPPORT"
        ),
        "candidates": candidates,
        "policy": [
            *STRICT_SCREEN_POLICY,
            "support at every sampled depth in each of two tiling offsets",
            "positive screening still requires raw-CT fiber-confound review",
        ],
    }


def manual_review_routing(screening: dict[str, Any]) -> dict[str, Any]:
    """Route only text-like screens to the expensive human CT review."""

    positive = (
        screening.get("screening_outcome")
        == "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"
    )
    return {
        "route": (
            "QUEUE_FOR_RAW_CT_FIBER_CONFOUND_REVIEW"
            if positive
            else "NOT_QUEUED_TEXT_LIKE_GATE_FAILED"
        ),
        "human_review_required": positive,
        "reason": (
            "morphological text-like gate passed; raw CT must still exclude "
            "fibers, laminar edges, cracks, voids and mineral inclusions"
            if positive
            else "the activation did not form enough bounded repeated shapes "
            "in at least two populated row bands"
        ),
    }


def select_hotspots(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    count: int,
    smoothing_size: int,
    exclusion_size: int,
) -> list[tuple[int, int, float]]:
    """Select deterministic, spatially separated maxima from a score map."""
    if score.shape != valid.shape:
        raise ValueError("score and valid mask shapes differ")
    if count < 1 or smoothing_size < 1 or exclusion_size < 1:
        raise ValueError("selection arguments must be positive")
    valid_float = valid.astype(np.float32)
    numerator = ndimage.uniform_filter(
        np.where(valid, score, 0).astype(np.float32),
        size=smoothing_size,
        mode="constant",
    )
    denominator = ndimage.uniform_filter(
        valid_float,
        size=smoothing_size,
        mode="constant",
    )
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -np.inf),
        where=denominator >= 0.75,
    )
    available = valid.copy()
    selected: list[tuple[int, int, float]] = []
    half = exclusion_size // 2
    for _ in range(count):
        candidate = np.where(available, smoothed, -np.inf)
        flat_index = int(np.argmax(candidate))
        best = float(candidate.flat[flat_index])
        if not np.isfinite(best):
            break
        y, x = np.unravel_index(flat_index, candidate.shape)
        selected.append((int(y), int(x), best))
        y0, y1 = max(0, y - half), min(score.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(score.shape[1], x + half + 1)
        available[y0:y1, x0:x1] = False
    return selected


def contrast_image(array: np.ndarray) -> np.ndarray:
    valid = array > 0
    if not valid.any():
        return np.zeros_like(array, dtype=np.uint8)
    lower, upper = np.percentile(array[valid], [1, 99.7])
    scaled = np.clip(
        (array.astype(np.float32) - lower) / max(float(upper - lower), 1.0),
        0,
        1,
    )
    return np.rint(scaled * 255).astype(np.uint8)


# FIX-05 — review-render display policy.
#
# Until 2026-07-24 every probability layer was rendered with a hardcoded floor
# of 0.20 and ceiling of 0.70.  Measured against the percentiles those same runs
# wrote into their own INK_STABILITY_ANALYSIS.json, the p90 of the mean map was
# 0.192-0.279: the display floor sat at or above the p90 of the data, so the top
# decile of signal was quantized to pure black before a human ever saw it.  The
# delivered PNGs were 78-94 % black and persistent_overlay.png differed from the
# bare CT in 0.79-3.74 % of pixels.
#
# contrast_image() in this same file already derives its limits from the data
# for the CT layer; probability layers now do the same.  Bounds are computed
# ONCE per layer over the full valid region and reused for every crop of that
# layer, so crops and full frames stay on one comparable scale.
#
# The legacy fixed transfer function is kept reachable (explicit lower/upper) so
# historical renders remain reproducible.
DISPLAY_PROFILE_DATA_DERIVED = "DATA_DERIVED_P50_P995_V1"
DISPLAY_PROFILE_FIXED = "EXPLICIT_FIXED_V1"
DISPLAY_LOWER_PERCENTILE = 50.0
DISPLAY_UPPER_PERCENTILE = 99.5
LEGACY_DISPLAY_LOWER = 0.20
LEGACY_DISPLAY_UPPER = 0.70
DISAGREEMENT_DISPLAY_BOUNDS = (0.0, 0.15)


def probability_image(
    array: np.ndarray,
    valid: np.ndarray,
    *,
    lower: float = LEGACY_DISPLAY_LOWER,
    upper: float = LEGACY_DISPLAY_UPPER,
) -> np.ndarray:
    """Quantize a probability layer between explicit display limits.

    The default limits are the frozen legacy pair and exist only to reproduce
    historical renders.  New review material must obtain its limits from
    probability_display_bounds() via render_probability_layer().
    """

    scaled = np.clip((array - lower) / max(upper - lower, 1e-6), 0, 1)
    return np.where(valid, np.rint(scaled * 255), 0).astype(np.uint8)


def probability_display_bounds(values: np.ndarray) -> tuple[float, float]:
    """Derive display limits from the data, as contrast_image does for CT."""

    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = (
        float(value)
        for value in np.percentile(
            finite, [DISPLAY_LOWER_PERCENTILE, DISPLAY_UPPER_PERCENTILE]
        )
    )
    if upper <= lower:
        upper = lower + 1e-6
    return lower, upper


def layer_bounds(
    array: np.ndarray,
    valid: np.ndarray,
    *,
    fixed: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return the display bounds record stamped onto every rendered PNG."""

    if fixed is not None:
        return {
            "display_profile": DISPLAY_PROFILE_FIXED,
            "display_lower": float(fixed[0]),
            "display_upper": float(fixed[1]),
            "display_lower_percentile": None,
            "display_upper_percentile": None,
        }
    values = np.asarray(array)[valid]
    lower, upper = probability_display_bounds(values)
    record = {
        "display_profile": DISPLAY_PROFILE_DATA_DERIVED,
        "display_lower": lower,
        "display_upper": upper,
        "display_lower_percentile": DISPLAY_LOWER_PERCENTILE,
        "display_upper_percentile": DISPLAY_UPPER_PERCENTILE,
    }
    record.update(assert_top_decile_visible(values, lower, upper))
    return record


def assert_top_decile_visible(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Fail closed if the display floor would erase the top decile again.

    FIX-05 invariant: the floor must sit strictly below p90 and the ceiling at
    or above p99, so a candidate near p90 can never again be byte-identical to
    background.  This is the regression that produced the delivered renders.
    """

    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p90": None, "p99": None, "grey_separation_p90_p99": None}
    p90, p99 = (float(value) for value in np.percentile(finite, [90.0, 99.0]))
    if not lower < p90:
        raise RuntimeError(
            f"display floor {lower} is not below p90 {p90}: this is the FIX-05 "
            "regression that quantized the top decile to black"
        )
    if not upper >= p99:
        raise RuntimeError(
            f"display ceiling {upper} is below p99 {p99}: the top decile would "
            "saturate instead of remaining discriminable"
        )
    span = max(upper - lower, 1e-6)
    grey = lambda value: float(  # noqa: E731
        np.rint(np.clip((value - lower) / span, 0.0, 1.0) * 255)
    )
    return {
        "p90": p90,
        "p99": p99,
        "grey_at_p90": grey(p90),
        "grey_at_p99": grey(p99),
        "grey_separation_p90_p99": grey(p99) - grey(p90),
    }


def render_probability_layer(
    array: np.ndarray,
    valid: np.ndarray,
    bounds: dict[str, Any],
) -> np.ndarray:
    return probability_image(
        array,
        valid,
        lower=float(bounds["display_lower"]),
        upper=float(bounds["display_upper"]),
    )


def zero_fraction_record(image: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Measure how much of a delivered PNG is pure black (audit acceptance)."""

    array = np.asarray(image)
    if array.ndim == 3:
        black = np.all(array == 0, axis=2)
    else:
        black = array == 0
    record = {"zero_fraction": float(black.mean())}
    if valid is not None and np.asarray(valid).shape == black.shape:
        mask = np.asarray(valid)
        record["zero_fraction_within_valid"] = (
            float(black[mask].mean()) if mask.any() else 1.0
        )
    return record


def heat_overlay(
    ct: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    bounds: dict[str, Any],
) -> Image.Image:
    grey = contrast_image(ct)
    heat = render_probability_layer(probability, valid, bounds).astype(np.float32) / 255.0
    rgb = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
    alpha = heat[..., None] * 0.72
    color = np.zeros_like(rgb)
    color[..., 0] = 255
    color[..., 1] = np.clip(255 * (1.0 - heat), 0, 255)
    output = rgb * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.rint(np.clip(output, 0, 255)).astype(np.uint8))


def build_display_policy(
    *,
    mean_map: np.ndarray,
    robust_map: np.ndarray,
    depth_maps: list[np.ndarray],
    valid: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Resolve one display scale per layer family, shared by crops and frames.

    Depth-centre panels share a single scale so that a difference between two
    depths is a difference in the data and not in the transfer function.
    """

    depth_values = (
        np.stack([np.asarray(value) for value in depth_maps])[:, valid]
        if depth_maps
        else np.asarray(mean_map)[valid]
    )
    depth_lower, depth_upper = probability_display_bounds(depth_values)
    depth_record = {
        "display_profile": DISPLAY_PROFILE_DATA_DERIVED,
        "display_lower": depth_lower,
        "display_upper": depth_upper,
        "display_lower_percentile": DISPLAY_LOWER_PERCENTILE,
        "display_upper_percentile": DISPLAY_UPPER_PERCENTILE,
    }
    depth_record.update(
        assert_top_decile_visible(depth_values, depth_lower, depth_upper)
    )
    return {
        "mean_probability": layer_bounds(mean_map, valid),
        "robust_minimum": layer_bounds(robust_map, valid),
        "replica_disagreement": layer_bounds(
            None, valid, fixed=DISAGREEMENT_DISPLAY_BOUNDS
        ),
        "depth_probability": depth_record,
    }


def crop_center(array: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    half = size // 2
    y0 = max(0, min(array.shape[-2] - size, y - half))
    x0 = max(0, min(array.shape[-1] - size, x - half))
    return array[..., y0 : y0 + size, x0 : x0 + size]


def add_label(image: Image.Image, text: str, height: int = 28) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + height), "#101827")
    output.paste(image.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(output)
    draw.text((7, 6), text, fill="#f4f7fb", font=ImageFont.load_default())
    return output


def make_montage(
    ct: np.ndarray,
    mean_map: np.ndarray,
    std_map: np.ndarray,
    robust_map: np.ndarray,
    agreement: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, dict[str, Any]],
) -> Image.Image:
    panels = [
        add_label(Image.fromarray(contrast_image(ct)), "CT central (contraste)"),
        add_label(
            Image.fromarray(
                render_probability_layer(
                    mean_map, valid, policy["mean_probability"]
                )
            ),
            "Probabilidad media cruda",
        ),
        add_label(
            Image.fromarray(
                render_probability_layer(
                    std_map, valid, policy["replica_disagreement"]
                )
            ),
            "Replica disagreement (std)",
        ),
        add_label(
            Image.fromarray(
                render_probability_layer(
                    robust_map, valid, policy["robust_minimum"]
                )
            ),
            "Minimum of the 6 replicas",
        ),
        add_label(
            Image.fromarray(
                np.where(valid, np.rint(agreement * 255), 0).astype(np.uint8)
            ),
            "Fraction of replicas > 0.5",
        ),
        add_label(
            heat_overlay(ct, robust_map, valid, policy["robust_minimum"]),
            "CT + activacion persistente",
        ),
    ]
    thumb_width = 720
    resized = [
        panel.resize(
            (thumb_width, round(panel.height * thumb_width / panel.width)),
            Image.Resampling.LANCZOS,
        )
        for panel in panels
    ]
    cell_height = max(panel.height for panel in resized)
    canvas = Image.new("RGB", (thumb_width * 2, cell_height * 3), "#080d16")
    for index, panel in enumerate(resized):
        canvas.paste(panel, ((index % 2) * thumb_width, (index // 2) * cell_height))
    return canvas


def save_comparison_layers(
    output: Path,
    *,
    ct: np.ndarray,
    maps: np.ndarray,
    map_paths: list[Path],
    mean_map: np.ndarray,
    std_map: np.ndarray,
    robust_map: np.ndarray,
    agreement: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Save like-for-like layers for control/target comparison.

    Every layer carries the display bounds actually used (FIX-05), so a reader
    can tell signal from transfer function without re-deriving anything.
    """

    layers = output / "comparison_layers"
    layers.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, tuple[Image.Image, dict[str, Any] | None]] = {
        "ct": (Image.fromarray(contrast_image(ct)), None),
        "mean_probability": (
            Image.fromarray(
                render_probability_layer(
                    mean_map, valid, policy["mean_probability"]
                )
            ),
            policy["mean_probability"],
        ),
        "robust_minimum": (
            Image.fromarray(
                render_probability_layer(
                    robust_map, valid, policy["robust_minimum"]
                )
            ),
            policy["robust_minimum"],
        ),
        "replica_disagreement": (
            Image.fromarray(
                render_probability_layer(
                    std_map, valid, policy["replica_disagreement"]
                )
            ),
            policy["replica_disagreement"],
        ),
        "replica_agreement": (
            Image.fromarray(
                np.where(valid, np.rint(agreement * 255), 0).astype(np.uint8)
            ),
            None,
        ),
        "persistent_overlay": (
            heat_overlay(ct, robust_map, valid, policy["robust_minimum"]),
            policy["robust_minimum"],
        ),
    }
    for center, value in group_maps_by_depth(maps, map_paths):
        rendered[f"depth_center_{center:03d}"] = (
            Image.fromarray(
                render_probability_layer(
                    value, valid, policy["depth_probability"]
                )
            ),
            policy["depth_probability"],
        )

    records: dict[str, dict[str, Any]] = {}
    for name, (image, bounds) in rendered.items():
        path = layers / f"{name}.png"
        image.save(path)
        record: dict[str, Any] = {
            "path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            **zero_fraction_record(np.asarray(image), valid),
        }
        if bounds is not None:
            record.update(bounds)
        records[name] = record
    return records


def make_hotspot_sheet(
    ct: np.ndarray,
    maps: np.ndarray,
    map_paths: list[Path],
    robust_map: np.ndarray,
    std_map: np.ndarray,
    valid: np.ndarray,
    hotspots: list[tuple[int, int, float]],
    *,
    crop_size: int,
    policy: dict[str, dict[str, Any]],
) -> Image.Image:
    width = 320
    row_height = width + 28
    columns = 6
    sheet = Image.new(
        "RGB",
        (columns * width, max(1, len(hotspots)) * row_height),
        "#080d16",
    )
    depth_groups = group_maps_by_depth(maps, map_paths)
    for row, (y, x, score) in enumerate(hotspots):
        ct_crop = crop_center(ct, y, x, crop_size)
        valid_crop = crop_center(valid, y, x, crop_size)
        robust_crop = crop_center(robust_map, y, x, crop_size)
        std_crop = crop_center(std_map, y, x, crop_size)
        probability_crops = [
            (
                center,
                crop_center(value, y, x, crop_size),
            )
            for center, value in depth_groups
        ]
        images = [
            add_label(
                Image.fromarray(contrast_image(ct_crop)),
                f"#{row + 1} CT y={y} x={x} score={score:.3f}",
            ),
            add_label(
                Image.fromarray(
                    render_probability_layer(
                        robust_crop, valid_crop, policy["robust_minimum"]
                    )
                ),
                "minimum of 6 replicas",
            ),
            add_label(
                Image.fromarray(
                    render_probability_layer(
                        std_crop, valid_crop, policy["replica_disagreement"]
                    )
                ),
                "replica std",
            ),
            *[
                add_label(
                    Image.fromarray(
                        render_probability_layer(
                            value, valid_crop, policy["depth_probability"]
                        )
                    ),
                    f"centro de profundidad {center}",
                )
                for center, value in probability_crops
            ],
        ]
        for column, image in enumerate(images):
            resized = image.resize((width, row_height), Image.Resampling.LANCZOS)
            sheet.paste(resized, (column * width, row * row_height))
    return sheet


def save_hotspot_assets(
    output: Path,
    ct: np.ndarray,
    maps: np.ndarray,
    map_paths: list[Path],
    robust_map: np.ndarray,
    std_map: np.ndarray,
    valid: np.ndarray,
    hotspots: list[tuple[int, int, float]],
    *,
    crop_size: int,
    policy: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    assets_dir = output / "viewer_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    depth_groups = group_maps_by_depth(maps, map_paths)
    records: list[dict[str, Any]] = []
    for rank, (y, x, score) in enumerate(hotspots, start=1):
        stem = f"hotspot-{rank:02d}"
        ct_crop = crop_center(ct, y, x, crop_size)
        valid_crop = crop_center(valid, y, x, crop_size)
        robust_crop = crop_center(robust_map, y, x, crop_size)
        std_crop = crop_center(std_map, y, x, crop_size)
        depth_maps = [
            (
                center,
                crop_center(value, y, x, crop_size),
            )
            for center, value in depth_groups
        ]
        rendered = {
            "ct": Image.fromarray(contrast_image(ct_crop)),
            "overlay": heat_overlay(
                ct_crop, robust_crop, valid_crop, policy["robust_minimum"]
            ),
            "robust": Image.fromarray(
                render_probability_layer(
                    robust_crop, valid_crop, policy["robust_minimum"]
                )
            ),
            "std": Image.fromarray(
                render_probability_layer(
                    std_crop, valid_crop, policy["replica_disagreement"]
                )
            ),
        }
        layer_policy_by_name = {
            "overlay": policy["robust_minimum"],
            "robust": policy["robust_minimum"],
            "std": policy["replica_disagreement"],
        }
        depth_names: list[str] = []
        for center, value in depth_maps:
            name = f"center{center:03d}"
            depth_names.append(name)
            layer_policy_by_name[name] = policy["depth_probability"]
            rendered[name] = Image.fromarray(
                render_probability_layer(
                    value, valid_crop, policy["depth_probability"]
                )
            )
        diagnostics = Image.new("L", (crop_size * 2, crop_size))
        diagnostics.paste(rendered["robust"], (0, 0))
        diagnostics.paste(rendered["std"], (crop_size, 0))
        rendered["diagnostics"] = diagnostics
        depths = Image.new("L", (crop_size * len(depth_names), crop_size))
        for index, name in enumerate(depth_names):
            depths.paste(rendered[name], (index * crop_size, 0))
        rendered["depths"] = depths
        files: dict[str, dict[str, Any]] = {}
        for name, image in rendered.items():
            path = assets_dir / f"{stem}-{name}.png"
            image.save(path)
            record: dict[str, Any] = {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            bounds = layer_policy_by_name.get(name)
            if bounds is not None:
                record.update(bounds)
            files[name] = record
        records.append(
            {
                "rank": rank,
                "center_y_x": [y, x],
                "smoothed_stability_score": score,
                "depth_centers": [center for center, _ in depth_maps],
                "files": files,
            }
        )
    return records


def build_viewer_html(
    output: Path,
    *,
    sample_id: str,
    hotspots: list[dict[str, Any]],
    depth_centers: list[int],
) -> Path:
    data = json.dumps(
        {"sample_id": sample_id, "hotspots": hotspots},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework · hotspots {html.escape(sample_id)}</title>
<style>
:root{{--bg:#08101b;--panel:#111d2e;--line:#2e4461;--text:#f3f6fb;--muted:#a9b7ca;--blue:#78b9ff;--amber:#f7c65c;--green:#55d59a;--red:#ff7c82}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.35 system-ui,sans-serif;overflow:hidden}}
header{{height:72px;padding:12px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:16px}}
h1{{font-size:22px;margin:0}} .muted{{color:var(--muted)}} button{{background:#18263a;color:var(--text);border:1px solid #49627f;border-radius:9px;padding:10px 14px;font:inherit;cursor:pointer}}
button:hover{{border-color:var(--blue)}} button.selected{{outline:3px solid var(--blue);background:#22466d}}
main{{height:calc(100vh - 72px);display:grid;grid-template-columns:minmax(0,1.7fr) minmax(340px,.72fr);gap:12px;padding:12px}}
.visuals,.review{{background:var(--panel);border:1px solid var(--line);border-radius:12px;min-height:0}}
.visuals{{display:grid;grid-template-columns:1.2fr 1.2fr .72fr;grid-template-rows:1fr 1fr;gap:6px;padding:8px}}
.tile{{position:relative;background:#02050a;border-radius:8px;overflow:hidden;min-height:0}}
.tile.primary{{grid-row:span 2}} .tile img{{width:100%;height:100%;object-fit:contain;display:block}}
.label{{position:absolute;left:7px;top:7px;background:#07101dd9;border-radius:5px;padding:4px 7px;font-size:13px}}
.review{{padding:18px;overflow:auto}} .rank{{color:var(--blue);font-weight:750}} h2{{font-size:24px;line-height:1.15;margin:8px 0}}
.instruction{{background:#18253a;border-left:4px solid var(--amber);padding:10px 12px;margin:14px 0;border-radius:5px}}
.answers{{display:grid;gap:9px}} .answer{{text-align:left;font-weight:700;padding:14px}}
.texture{{border-color:#5e718b}} .possible{{border-color:#c49238}} .unclear{{border-color:#68798e}}
.note{{width:100%;min-height:78px;margin-top:12px;background:#0b1524;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px;font:inherit}}
.footer{{display:flex;gap:8px;justify-content:space-between;margin-top:12px;flex-wrap:wrap}}
.status{{margin:12px 0;color:var(--muted)}} kbd{{background:#26364c;border-radius:4px;padding:2px 5px}}
@media(max-width:1100px){{body{{overflow:auto}} main{{height:auto;grid-template-columns:1fr}} .visuals{{height:72vh}}}}
</style>
</head>
<body>
<header>
 <div><h1>Helena Framework · {html.escape(sample_id)}  - hotspot review</h1><div class="muted">Persistent model activations; not accepted ink and not letters.</div></div>
 <div><button id="prev">← Anterior</button> <strong id="counter"></strong> <button id="next">Siguiente →</button></div>
</header>
<main>
 <section class="visuals">
  <div class="tile primary"><span class="label">Raw CT - is there a thin, coherent stroke?</span><img id="ct"></div>
  <div class="tile primary"><span class="label">CT + persistent activation</span><img id="overlay"></div>
  <div class="tile"><span class="label">Persistence (left) / variation (right)</span><img id="robust"></div>
  <div class="tile"><span class="label">Change with depth: {html.escape(" / ".join(map(str, depth_centers)))}</span><img id="depth"></div>
 </section>
 <aside class="review">
  <div class="rank" id="rank"></div>
  <h2>Does the activation coincide with an ink-like shape?</h2>
  <div class="instruction"><b>Look at the CT first.</b> Find a thin, continuous, relatively uniform line that is not simply the bright edge of a fibre, a crack, a hole or a region of broken material. Then check that the map's brightness sits on that same line and persists as the depth changes.</div>
  <div class="muted" id="metrics"></div>
  <div class="answers">
   <button class="answer texture" data-answer="LIKELY_TEXTURE_OR_DAMAGE"><kbd>1</kbd> Looks like texture, an edge or damage</button>
   <button class="answer possible" data-answer="POSSIBLE_INK_NEEDS_FOLLOWUP"><kbd>2</kbd> Possible ink: a thin, coherent shape</button>
   <button class="answer unclear" data-answer="INCONCLUSIVE"><kbd>3</kbd> No concluyente</button>
  </div>
  <textarea class="note" id="note" placeholder="Optional note: what shape you saw and where"></textarea>
  <div class="status" id="status"></div>
  <div class="footer"><button id="export">Export review JSON</button><button id="clear">Clear current answer</button></div>
 </aside>
</main>
<script>
const DATA={data}; const KEY="campaign-x-ink-review-"+DATA.sample_id; let index=0;
const saved=JSON.parse(localStorage.getItem(KEY)||"{{}}");
const $=id=>document.getElementById(id);
function persist(){{localStorage.setItem(KEY,JSON.stringify(saved));}}
function render(){{
 const h=DATA.hotspots[index], f=h.files;
 $("ct").src=f.ct.path; $("overlay").src=f.overlay.path; $("robust").src=f.diagnostics.path;
 $("depth").src=f.depths.path;
 $("counter").textContent=`${{index+1}} / ${{DATA.hotspots.length}}`;
 $("rank").textContent=`Hotspot #${{h.rank}} · y=${{h.center_y_x[0]}}, x=${{h.center_y_x[1]}}`;
 $("metrics").textContent=`Smoothed persistence score: ${{h.smoothed_stability_score.toFixed(3)}} - each crop is ~3.04 mm`;
 document.querySelectorAll(".answer").forEach(b=>b.classList.toggle("selected",saved[h.rank]?.answer===b.dataset.answer));
 $("note").value=saved[h.rank]?.note||"";
 $("status").textContent=saved[h.rank]?`Guardado: ${{saved[h.rank].answer}}`:"Sin respuesta";
}}
function answer(value){{const h=DATA.hotspots[index];saved[h.rank]={{answer:value,note:$("note").value,center_y_x:h.center_y_x,score:h.smoothed_stability_score,updated_at:new Date().toISOString()}};persist();render();}}
document.querySelectorAll(".answer").forEach(b=>b.onclick=()=>answer(b.dataset.answer));
$("note").onchange=()=>{{const h=DATA.hotspots[index];if(saved[h.rank]){{saved[h.rank].note=$("note").value;persist();}}}};
$("prev").onclick=()=>{{index=(index+DATA.hotspots.length-1)%DATA.hotspots.length;render()}};
$("next").onclick=()=>{{index=(index+1)%DATA.hotspots.length;render()}};
$("clear").onclick=()=>{{delete saved[DATA.hotspots[index].rank];persist();render()}};
$("export").onclick=()=>{{const payload={{kind:"campaign_x_phase4_human_ink_hotspot_review_v1",sample_id:DATA.sample_id,exported_at:new Date().toISOString(),assessments:saved}};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:"application/json"}}));a.download=DATA.sample_id+"-ink-hotspot-review.json";a.click();URL.revokeObjectURL(a.href)}};
addEventListener("keydown",e=>{{if(e.key==="ArrowLeft")$("prev").click();if(e.key==="ArrowRight")$("next").click();if(["1","2","3"].includes(e.key))document.querySelectorAll(".answer")[+e.key-1].click();}});
render();
</script>
</body></html>"""
    path = output / "INK_HOTSPOT_REVIEW.html"
    path.write_text(page, encoding="utf-8")
    return path


def build_negative_screen_html(
    output: Path,
    *,
    sample_id: str,
    screening: dict[str, Any],
) -> Path:
    """Replace the hotspot UI with a closed summary for negative screens."""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework · cribado cerrado · {html.escape(sample_id)}</title>
<style>
:root{{--bg:#08101b;--panel:#111d2e;--line:#2e4461;--text:#f3f6fb;--muted:#a9b7ca;--green:#55d59a}}
*{{box-sizing:border-box}} body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);font:18px/1.45 system-ui,sans-serif}}
main{{width:min(820px,calc(100% - 32px));background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:28px}}
h1{{margin:0 0 8px;font-size:clamp(26px,4vw,42px)}} .status{{color:var(--green);font-weight:800}}
.muted{{color:var(--muted)}} dl{{display:grid;grid-template-columns:1fr auto;gap:9px 20px;border-top:1px solid var(--line);padding-top:16px}}
dt,dd{{margin:0}} dd{{font-weight:800}} code{{font-size:.88em}}
</style></head><body><main>
<div class="status">NOT SENT TO HUMAN REVIEW</div>
<h1>{html.escape(sample_id)}  - the text-like gate did not pass</h1>
<p>Activations may be fibres, lamina edges, cracks, voids or inclusions.
Presenting hotspots as if they were candidates only adds false positives, which
is why this viewer stops short of manual review.</p>
<dl>
<dt>Formas acotadas persistentes</dt><dd>{int(screening["glyph_like_candidate_count"])}</dd>
<dt>Renglones detectados</dt><dd>{int(screening["row_band_count"])}</dd>
<dt>Rows with 4+ shapes</dt><dd>{int(screening["rows_with_at_least_four_candidates"])}</dd>
<dt>Gate exigido</dt><dd>≥10 formas y ≥2 renglones poblados</dd>
</dl>
<p class="muted">Resultado: <code>{html.escape(str(screening["screening_outcome"]))}</code>.
The diagnostic PNG sheets are kept for audit, but they require no human
decision and are not negative evidence for the scroll as a whole.</p>
</main></body></html>"""
    path = output / "INK_HOTSPOT_REVIEW.html"
    path.write_text(page, encoding="utf-8")
    return path


def component_summary(
    robust_map: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    binary = valid & (robust_map >= threshold)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    components: list[dict[str, Any]] = []
    if count:
        objects = ndimage.find_objects(labels)
        sizes = np.bincount(labels.ravel())[1:]
        order = np.argsort(sizes)[::-1][:20]
        for index in order:
            bounds = objects[int(index)]
            if bounds is None:
                continue
            components.append(
                {
                    "pixels": int(sizes[index]),
                    "bbox_y0_x0_y1_x1": [
                        int(bounds[0].start),
                        int(bounds[1].start),
                        int(bounds[0].stop),
                        int(bounds[1].stop),
                    ],
                    "mean_robust_probability": float(
                        robust_map[labels == index + 1].mean()
                    ),
                }
            )
    return {
        "threshold": threshold,
        "active_pixels": int(binary.sum()),
        "component_count": int(count),
        "largest_components": components,
    }


# FIX-09 — physical scale must come from the frozen catalog and the ink-lane
# profile, never from a CLI default.
#
# The old default was --source-pixel-um 9.362, while workspace/catalog/
# eligible_volumes.json carries four volumes at 8.64 um (PHerc0268, PHerc0800,
# PHerc1218, PHerc1447).  Omitting the flag silently rescaled those runs by
# 8.4 %.  training_pixel_um was hardcoded to 7.91 in five callers rather than
# read from the ink lane profile that actually declares it.
def helena_repo_root() -> Path:
    """Locate the Helena Framework root that owns the frozen catalog and profiles."""

    configured = os.environ.get("HELENA_REPO_ROOT", "").strip()
    candidates = (
        [Path(configured).expanduser().resolve()] if configured else []
    ) + list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "workspace" / "catalog").is_dir() and (
            candidate / "framework" / "profiles"
        ).is_dir():
            return candidate
    raise RuntimeError("cannot locate the Helena Framework repository root")


HELENA_REPO_ROOT = helena_repo_root()
DEFAULT_VOLUME_CATALOG = (
    HELENA_REPO_ROOT / "workspace" / "catalog" / "eligible_volumes.json"
)
DEFAULT_INK_PROFILE = (
    HELENA_REPO_ROOT
    / "framework"
    / "profiles"
    / "03-ink"
    / "timesformer-gp-scroll1-screening-1.0.0.json"
)
PIXEL_UM_TOLERANCE = 1e-6


def catalog_voxel_size_um(catalog_path: Path, sample_id: str) -> float | None:
    """Return the frozen voxel size for a sample, or None if uncatalogued."""

    if not catalog_path.is_file():
        raise RuntimeError(f"volume catalog is missing: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"volume catalog has no entries: {catalog_path}")
    matches = {
        float(entry["voxel_size_um"])
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("sample_id")) == sample_id
    }
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"volume catalog gives {sample_id} more than one voxel size: {matches}"
        )
    return matches.pop()


def resolve_source_pixel_um(
    *,
    sample_id: str,
    catalog_path: Path,
    requested: float | None,
) -> tuple[float, dict[str, Any]]:
    """Resolve the source pixel size, failing closed on any CLI disagreement."""

    catalogued = catalog_voxel_size_um(catalog_path, sample_id)
    if catalogued is None:
        if requested is None:
            raise RuntimeError(
                f"{sample_id} is not in {catalog_path} and no --source-pixel-um "
                "was supplied; refusing to guess the physical scale"
            )
        return float(requested), {
            "source": "CLI_UNCATALOGUED_SAMPLE",
            "catalog_path": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "catalog_voxel_size_um": None,
            "cli_source_pixel_um": float(requested),
        }
    if requested is not None and abs(float(requested) - catalogued) > PIXEL_UM_TOLERANCE:
        raise RuntimeError(
            f"--source-pixel-um {requested} disagrees with the frozen catalog "
            f"value {catalogued} for {sample_id} (tolerance {PIXEL_UM_TOLERANCE}); "
            "refusing to rescale silently"
        )
    return catalogued, {
        "source": "ELIGIBLE_VOLUMES_CATALOG",
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_voxel_size_um": catalogued,
        "cli_source_pixel_um": None if requested is None else float(requested),
    }


def resolve_training_pixel_um(
    *,
    profile_path: Path,
    requested: float | None,
) -> tuple[float, dict[str, Any]]:
    """Resolve the training pixel size from the ink lane profile."""

    if not profile_path.is_file():
        raise RuntimeError(f"ink lane profile is missing: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    contract = profile.get("input_contract")
    if not isinstance(contract, dict) or "training_pixel_um" not in contract:
        raise RuntimeError(
            f"ink lane profile declares no training_pixel_um: {profile_path}"
        )
    declared = float(contract["training_pixel_um"])
    if requested is not None and abs(float(requested) - declared) > PIXEL_UM_TOLERANCE:
        raise RuntimeError(
            f"--training-pixel-um {requested} disagrees with the ink lane profile "
            f"value {declared} ({profile.get('profile_id')}); refusing to rescale "
            "silently"
        )
    return declared, {
        "source": "INK_LANE_PROFILE",
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "profile_id": profile.get("profile_id"),
        "profile_training_pixel_um": declared,
        "cli_training_pixel_um": None if requested is None else float(requested),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--screening-dir", type=Path, required=True)
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-center", type=int, default=32)
    parser.add_argument("--source-pixel-um", type=float, default=None)
    parser.add_argument("--training-pixel-um", type=float, default=None)
    parser.add_argument(
        "--volume-catalog", type=Path, default=DEFAULT_VOLUME_CATALOG
    )
    parser.add_argument("--ink-profile", type=Path, default=DEFAULT_INK_PROFILE)
    parser.add_argument("--glyph-threshold", type=float, default=0.5)
    parser.add_argument("--hotspots", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=384)
    args = parser.parse_args()

    source_pixel_um, source_scale_provenance = resolve_source_pixel_um(
        sample_id=args.sample_id,
        catalog_path=args.volume_catalog.resolve(),
        requested=args.source_pixel_um,
    )
    training_pixel_um, training_scale_provenance = resolve_training_pixel_um(
        profile_path=args.ink_profile.resolve(),
        requested=args.training_pixel_um,
    )

    screening_dir = args.screening_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    maps, map_paths = load_maps(screening_dir)
    map_coordinates = [parse_map_coordinates(path) for path in map_paths]
    depth_centers = sorted({center for center, _ in map_coordinates})
    offsets = sorted({offset for _, offset in map_coordinates})
    valid = common_valid_mask(maps)
    if not valid.any():
        raise RuntimeError("screening maps have no common valid pixels")
    mean_map = maps.mean(axis=0)
    std_map = maps.std(axis=0)
    robust_map = maps.min(axis=0)
    agreement = (maps >= 0.5).mean(axis=0)
    stability_score = np.where(valid, robust_map - std_map, -np.inf)
    text_like_screening = glyph_like_support(
        maps,
        map_paths,
        threshold=args.glyph_threshold,
    )
    review_routing = manual_review_routing(text_like_screening)

    tiff_path = resolve_tiff_slice(
        args.tiff_dir.resolve(),
        args.source_center,
    )
    source_ct = np.asarray(Image.open(tiff_path), dtype=np.uint8)
    target_shape = maps.shape[1:]
    ct = np.asarray(
        Image.fromarray(source_ct).resize(
            (target_shape[1], target_shape[0]),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )

    hotspots = select_hotspots(
        stability_score,
        valid,
        count=args.hotspots,
        smoothing_size=128,
        exclusion_size=args.crop_size,
    )
    display_policy = build_display_policy(
        mean_map=mean_map,
        robust_map=robust_map,
        depth_maps=[value for _, value in group_maps_by_depth(maps, map_paths)],
        valid=valid,
    )
    montage_path = output / "stability_montage.png"
    hotspot_path = output / "hotspot_contact_sheet.png"
    make_montage(
        ct, mean_map, std_map, robust_map, agreement, valid, display_policy
    ).save(montage_path)
    comparison_layers = save_comparison_layers(
        output,
        ct=ct,
        maps=maps,
        map_paths=map_paths,
        mean_map=mean_map,
        std_map=std_map,
        robust_map=robust_map,
        agreement=agreement,
        valid=valid,
        policy=display_policy,
    )
    make_hotspot_sheet(
        ct,
        maps,
        map_paths,
        robust_map,
        std_map,
        valid,
        hotspots,
        crop_size=args.crop_size,
        policy=display_policy,
    ).save(hotspot_path)
    hotspot_assets = save_hotspot_assets(
        output,
        ct,
        maps,
        map_paths,
        robust_map,
        std_map,
        valid,
        hotspots,
        crop_size=args.crop_size,
        policy=display_policy,
    )
    if review_routing["human_review_required"]:
        viewer_path = build_viewer_html(
            output,
            sample_id=args.sample_id,
            hotspots=hotspot_assets,
            depth_centers=depth_centers,
        )
    else:
        viewer_path = build_negative_screen_html(
            output,
            sample_id=args.sample_id,
            screening=text_like_screening,
        )

    values = {
        "mean": mean_map[valid],
        "std": std_map[valid],
        "robust_min": robust_map[valid],
        "agreement": agreement[valid],
    }
    receipt = {
        "kind": "campaign_x_phase4_ink_stability_analysis_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "sample_id": args.sample_id,
        "scope": "PRIVATE_LOCAL_FUNCTIONAL_INK_SCREENING",
        "input": {
            "screening_directory": str(screening_dir),
            "maps": [
                {
                    "file": path.name,
                    "depth_center": center,
                    "tiling_offset": offset,
                    "sha256": sha256_file(path),
                }
                for path, (center, offset) in zip(map_paths, map_coordinates)
            ],
            "central_tiff": str(tiff_path),
            "central_tiff_sha256": sha256_file(tiff_path),
            "central_tiff_slice_index": args.source_center,
            "slice_ordering": NUMERIC_STEM_INDEX,
            "shape_y_x": list(target_shape),
            "common_valid_pixels": int(valid.sum()),
            "depth_centers": depth_centers,
            "tiling_offsets": offsets,
        },
        "physical_scale": {
            "source_pixel_um": source_pixel_um,
            "normalized_pixel_um": training_pixel_um,
            "crop_size_pixels": args.crop_size,
            "crop_size_mm": args.crop_size * training_pixel_um / 1000,
            "source_pixel_um_provenance": source_scale_provenance,
            "training_pixel_um_provenance": training_scale_provenance,
        },
        "display_policy": display_policy,
        "probability_distributions": {
            name: {
                "mean": float(value.mean()),
                "percentiles": {
                    str(percentile): float(np.percentile(value, percentile))
                    for percentile in (1, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9)
                },
            }
            for name, value in values.items()
        },
        "threshold_agreement": {
            "all_six_above_0_5_pixels": int(
                (valid & (agreement == 1.0)).sum()
            ),
            "at_least_five_above_0_5_pixels": int(
                (valid & (agreement >= 5 / 6)).sum()
            ),
            "all_six_above_0_7_pixels": int(
                (valid & np.all(maps >= 0.7, axis=0)).sum()
            ),
        },
        "robust_components": [
            component_summary(robust_map, valid, threshold=threshold)
            for threshold in (0.5, 0.6, 0.7)
        ],
        "text_like_screening": text_like_screening,
        "manual_review_routing": review_routing,
        "ranked_review_hotspots": [
            {
                "rank": index + 1,
                "center_y_x": [y, x],
                "smoothed_stability_score": score,
                "crop_size_pixels": args.crop_size,
                "crop_size_mm": args.crop_size * training_pixel_um / 1000,
                "mean_probability_at_center": float(mean_map[y, x]),
                "min_probability_at_center": float(robust_map[y, x]),
                "std_probability_at_center": float(std_map[y, x]),
            }
            for index, (y, x, score) in enumerate(hotspots)
        ],
        "artifacts": {
            montage_path.name: {
                "sha256": sha256_file(montage_path),
                "size_bytes": montage_path.stat().st_size,
            },
            hotspot_path.name: {
                "sha256": sha256_file(hotspot_path),
                "size_bytes": hotspot_path.stat().st_size,
            },
            viewer_path.name: {
                "sha256": sha256_file(viewer_path),
                "size_bytes": viewer_path.stat().st_size,
            },
            "comparison_layers": comparison_layers,
        },
        "interpretation_policy": [
            "ranked hotspots are model activations, not accepted ink",
            "a hotspot requires raw-CT morphology review",
            "a hotspot must persist across depth and tiling perturbations",
            "a First Letters claim requires a coherent group of letter-like traces",
        ],
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not a First Letters submission claim",
        ],
    }
    receipt_path = output / "INK_STABILITY_ANALYSIS.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
