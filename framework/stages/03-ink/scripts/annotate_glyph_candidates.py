#!/usr/bin/env python3
"""Annotate persistent glyph-like candidates in exact 4 cm² ink windows.

This is a screening tool, not a papyrological reading.  A candidate is emitted
only when a bright, bounded shape is supported by both independently published
ink-prediction images and belongs to a detected horizontal text-like band.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, signal


#
# FIX-08 — canonical strict text-like screen.
#
# Until 2026-07-24 this criterion existed twice with divergent terms:
#
#   * analyze_ink_stability.py  -> >=10 shapes AND >=2 rows holding >=4
#     shapes each, labelled POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW.
#   * this file                        -> >=10 shapes AND >=2 populated rows,
#     with no per-row minimum, labelled POTENTIAL_FIRST_LETTERS_REGION.
#
# The second was strictly more permissive and carried the First Letters label.
# The STRICT form is canonical: it is the criterion that produced the PHerc0139
# out-of-cohort positive control (15 shapes across 2 rows).  The per-row minimum
# is NOT relaxed.  Relaxing it was evaluated (handoff U1) and cancelled with
# evidence: on the U0 control over a typical VC3D surface the screen returned
# 1 candidate against a minimum of 10, so the candidate count -- not the row
# term -- is what binds.  Both call sites now route through
# strict_text_like_screen() so the two can no longer drift apart.
#
STRICT_SCREEN_MINIMUM_CANDIDATES = 10
STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW = 4
STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS = 2

STRICT_SCREEN_POLICY = (
    f"at least {STRICT_SCREEN_MINIMUM_CANDIDATES} bounded candidates",
    f"at least {STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS} row bands containing at "
    f"least {STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW} candidates each",
)


def candidate_row_histogram(candidates: list[dict[str, Any]]) -> dict[int, int]:
    """Count screened candidates per detected row band."""

    histogram: dict[int, int] = {}
    for item in candidates:
        row = int(item["row"])
        histogram[row] = histogram.get(row, 0) + 1
    return dict(sorted(histogram.items()))


def strict_text_like_screen(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the one canonical strict text-like criterion.

    Shared by Stage 03 (`annotate_glyph_candidates`) and Stage 04
    (`analyze_ink_stability`) so the two cannot diverge again.
    """

    histogram = candidate_row_histogram(candidates)
    qualifying_row_count = sum(
        count >= STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW
        for count in histogram.values()
    )
    qualifies = (
        len(candidates) >= STRICT_SCREEN_MINIMUM_CANDIDATES
        and qualifying_row_count >= STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS
    )
    return {
        "qualifies": bool(qualifies),
        "glyph_like_candidate_count": len(candidates),
        "row_band_count": len(histogram),
        "candidate_count_by_row": {
            str(row): count for row, count in histogram.items()
        },
        "rows_with_at_least_four_candidates": int(qualifying_row_count),
        "criterion_id": "campaignx.strict_text_like_screen.v1",
        "policy": list(STRICT_SCREEN_POLICY),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def detect_row_bands(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return non-overlapping (start, end, peak) text-like horizontal bands."""

    height = mask.shape[0]
    projection = ndimage.gaussian_filter1d(mask.sum(axis=1).astype(float), 4.0)
    peaks, properties = signal.find_peaks(
        projection,
        distance=max(24, height // 9),
        prominence=max(3.0, float(projection.std()) * 0.5),
    )
    if peaks.size == 0:
        return []

    prominences = properties["prominences"]
    order = sorted(
        range(len(peaks)),
        key=lambda index: (-float(prominences[index]), int(peaks[index])),
    )
    retained: list[int] = []
    for index in order:
        peak = int(peaks[index])
        if projection[peak] < max(4.0, float(np.percentile(projection, 65.0))):
            continue
        if all(abs(peak - other) >= max(24, height // 11) for other in retained):
            retained.append(peak)
        if len(retained) == 5:
            break
    retained.sort()
    if not retained:
        return []

    typical_spacing = (
        float(np.median(np.diff(retained)))
        if len(retained) > 1
        else max(36.0, height * 0.18)
    )
    half_height = max(14, min(36, round(typical_spacing * 0.37)))
    bands: list[tuple[int, int, int]] = []
    for peak in retained:
        start = max(0, peak - half_height)
        end = min(height, peak + half_height + 1)
        if bands and start < bands[-1][1]:
            midpoint = (bands[-1][2] + peak) // 2
            previous = bands[-1]
            bands[-1] = (previous[0], midpoint, previous[2])
            start = midpoint
        bands.append((start, end, peak))
    return bands


def discover_glyph_candidates(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold: float = 0.55,
) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[int, int, int]]]:
    """Find bounded, two-model-supported shapes organized in row bands."""

    if first.shape != second.shape:
        raise RuntimeError(
            f"model images must have the same shape: {first.shape} != {second.shape}"
        )
    persistent = (first >= threshold) & (second >= threshold)
    persistent = ndimage.binary_opening(
        persistent, structure=np.ones((2, 2), dtype=bool)
    )
    height, width = persistent.shape
    bands = detect_row_bands(persistent)
    candidates: list[dict[str, Any]] = []
    for row_index, (row_start, row_end, peak) in enumerate(bands, start=1):
        row_mask = persistent[row_start:row_end]
        minimum_column_pixels = max(2, round((row_end - row_start) * 0.03))
        active_columns = row_mask.sum(axis=0) >= minimum_column_pixels
        active_columns = ndimage.binary_closing(
            active_columns, structure=np.ones(4, dtype=bool)
        )
        labels, count = ndimage.label(active_columns)
        for label in range(1, count + 1):
            columns = np.flatnonzero(labels == label)
            if columns.size == 0:
                continue
            x0, x1 = int(columns[0]), int(columns[-1] + 1)
            interval_width = x1 - x0
            if x0 <= round(width * 0.02) or x1 >= width - round(width * 0.02):
                continue
            if interval_width < max(3, round(width * 0.012)):
                continue
            if interval_width > max(18, round(width * 0.14)):
                continue

            local = persistent[row_start:row_end, x0:x1]
            ys, xs = np.nonzero(local)
            if ys.size < 10:
                continue
            y0 = int(row_start + ys.min())
            y1 = int(row_start + ys.max() + 1)
            actual_x0 = int(x0 + xs.min())
            actual_x1 = int(x0 + xs.max() + 1)
            box_height = y1 - y0
            box_width = actual_x1 - actual_x0
            if box_height < max(5, round(height * 0.018)):
                continue
            if box_height > max(20, round(height * 0.18)):
                continue

            first_box = first[y0:y1, actual_x0:actual_x1]
            second_box = second[y0:y1, actual_x0:actual_x1]
            first_high = first_box >= threshold
            second_high = second_box >= threshold
            union = int((first_high | second_high).sum())
            intersection = int((first_high & second_high).sum())
            agreement_iou = float(intersection / union) if union else 0.0
            if agreement_iou < 0.30:
                continue
            candidates.append(
                {
                    "row": row_index,
                    "row_peak_y": peak,
                    "bbox_xyxy": [actual_x0, y0, actual_x1, y1],
                    "persistent_pixels": intersection,
                    "agreement_iou": agreement_iou,
                    "model_a_mean": float(first_box[first_high].mean())
                    if first_high.any()
                    else 0.0,
                    "model_b_mean": float(second_box[second_high].mean())
                    if second_high.any()
                    else 0.0,
                }
            )

    candidates.sort(key=lambda item: (item["row"], item["bbox_xyxy"][0]))
    for index, item in enumerate(candidates, start=1):
        item["candidate_id"] = f"G{index:02d}"
    return persistent, candidates, bands


def save_overlay(
    path: Path,
    consensus_path: Path,
    candidates: list[dict[str, Any]],
    bands: list[tuple[int, int, int]],
) -> None:
    image = Image.open(consensus_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row_start, row_end, _ in bands:
        draw.line((0, row_start, image.width - 1, row_start), fill="#21d4fd", width=1)
        draw.line((0, row_end - 1, image.width - 1, row_end - 1), fill="#21d4fd", width=1)
    for item in candidates:
        x0, y0, x1, y1 = item["bbox_xyxy"]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#ffcc33", width=2)
        label = str(item["candidate_id"])
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y0 - text_height - 3)
        draw.rectangle(
            (x0, label_y, x0 + text_width + 4, label_y + text_height + 3),
            fill="#07101d",
        )
        draw.text((x0 + 2, label_y + 1), label, fill="#ffcc33", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize(
        (image.width * 2, image.height * 2), Image.Resampling.NEAREST
    ).save(path)


def render_viewer(path: Path, receipt: dict[str, Any]) -> None:
    cards: list[str] = []
    for row in receipt["candidates"]:
        status = html.escape(str(row["screening_outcome"]))
        cards.append(
            f"""<article class="candidate">
              <header>
                <div><strong>#{row['rank']} · {html.escape(row['segment_id'])}</strong>
                <span class="status">{status}</span></div>
                <div>{row['glyph_like_candidate_count']} formas persistentes ·
                {row['row_band_count']} renglones · score {row['ranking_score']:.3f}</div>
              </header>
              <div class="images">
                <figure><img src="{html.escape(row['assets']['model_a'])}">
                  <figcaption>Modelo oficial A</figcaption></figure>
                <figure><img src="{html.escape(row['assets']['model_b'])}">
                  <figcaption>Modelo oficial B</figcaption></figure>
                <figure><img src="{html.escape(row['assets']['annotated'])}">
                  <figcaption>Consensus: yellow boxes = candidates held by both models</figcaption></figure>
              </div>
              <p><b>Limit:</b> the boxes are glyph-like shapes, not letters read nor
              papyrological confirmation. The area is exactly {row['window_area_cm2']:.4f} cm2.</p>
            </article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework - text-like signal calibration</title>
<style>
:root{{--bg:#07101d;--panel:#101d30;--line:#29415e;--text:#edf4ff;--muted:#a9bad0;--gold:#ffcc33}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.35 system-ui,sans-serif}}
main{{max-width:1900px;margin:auto;padding:18px}} h1{{margin:0 0 5px;font-size:clamp(24px,3vw,42px)}}
.lead{{color:var(--muted);margin:0 0 16px}} .warning{{background:#282310;border:1px solid #7b6420;padding:12px;border-radius:10px}}
.candidate{{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:18px 0;overflow:hidden}}
.candidate header{{display:flex;justify-content:space-between;gap:16px;padding:12px 15px;flex-wrap:wrap}}
.status{{display:inline-block;margin-left:10px;color:var(--gold)}} .images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:2px;background:var(--line)}}
figure{{margin:0;background:#000;display:flex;flex-direction:column}} img{{display:block;width:100%;min-height:0;object-fit:contain;image-rendering:auto}}
figcaption{{background:#0b1524;padding:8px;text-align:center;color:var(--muted)}} .candidate p{{margin:0;padding:12px 15px;color:var(--muted)}}
@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Helena Framework - text-like signal calibration</h1>
<p class="lead">PHerc0139 - positive control outside the cohort - exact 20x20 mm physical windows - two official predictions sobre modelos y volúmenes distintos</p>
<p class="warning"><b>What to look for:</b> a candidate ink signal must repeat in A and B,
form discrete strokes and organise into rows. Continuous fibres, lamina edges and
blobs present in only one model do not count. PHerc0139 is not one of the 13 targets
of Helena Framework: this page calibrates what to look for and does not complete the search.</p>
{''.join(cards)}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--out-of-cohort-positive-control",
        action="store_true",
        help="mark positive screens as calibration controls, not Helena Framework discoveries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking = read_json(args.ranking_json)
    rows = ranking.get("ranked_segments")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("ranking has no ranked_segments")
    output_dir = args.output_dir
    asset_dir = output_dir / "viewer_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.ranking_json.parent
    results: list[dict[str, Any]] = []
    for row in rows[: args.top_n]:
        source_assets = row["assets"]
        first_path = source_root / source_assets["first"]
        second_path = source_root / source_assets["second"]
        consensus_path = source_root / source_assets["consensus"]
        first = load_gray(first_path)
        second = load_gray(second_path)
        _, candidates, bands = discover_glyph_candidates(
            first, second, threshold=args.threshold
        )
        segment_id = str(row["segment_id"])
        copied_assets: dict[str, str] = {}
        for key, source in (("model_a", first_path), ("model_b", second_path)):
            target = asset_dir / f"{segment_id}-{key}.jpg"
            target.write_bytes(source.read_bytes())
            copied_assets[key] = target.relative_to(output_dir).as_posix()
        overlay_path = asset_dir / f"{segment_id}-annotated.png"
        save_overlay(overlay_path, consensus_path, candidates, bands)
        copied_assets["annotated"] = overlay_path.relative_to(output_dir).as_posix()
        screen = strict_text_like_screen(candidates)
        qualifies = screen["qualifies"]
        positive_outcome = (
            "TEXT_LIKE_POSITIVE_CONTROL"
            if args.out_of_cohort_positive_control
            else "POTENTIAL_FIRST_LETTERS_REGION"
        )
        results.append(
            {
                "rank": int(row["rank"]),
                "segment_id": segment_id,
                "long_id": row["long_id"],
                "ranking_score": float(row["score"]),
                "window_area_cm2": float(row["window_physical"]["area_cm2"]),
                "window_reference_preview_yxyx": row[
                    "window_reference_preview_yxyx"
                ],
                "threshold": args.threshold,
                "glyph_like_candidate_count": screen["glyph_like_candidate_count"],
                "row_band_count": screen["row_band_count"],
                "candidate_count_by_row": screen["candidate_count_by_row"],
                "rows_with_at_least_four_candidates": screen[
                    "rows_with_at_least_four_candidates"
                ],
                "strict_screen": screen,
                "screening_outcome": (
                    positive_outcome
                    if qualifies
                    else "INSUFFICIENT_GLYPH_LIKE_SUPPORT"
                ),
                "glyph_candidates": candidates,
                "source_assets": {
                    "model_a": {
                        "path": first_path.as_posix(),
                        "sha256": sha256_file(first_path),
                    },
                    "model_b": {
                        "path": second_path.as_posix(),
                        "sha256": sha256_file(second_path),
                    },
                    "consensus": {
                        "path": consensus_path.as_posix(),
                        "sha256": sha256_file(consensus_path),
                    },
                },
                "assets": copied_assets,
            }
        )

    receipt = {
        "kind": "campaign_x_phase4_glyph_candidate_screening_v1",
        "generated_at_utc": utc_now(),
        "claim_limit": (
            "OUT_OF_COHORT_POSITIVE_CONTROL_NOT_CAMPAIGN_DISCOVERY"
            if args.out_of_cohort_positive_control
            else "SCREENING_ONLY_NOT_LETTER_IDENTIFICATION_NOT_PAPYROLOGICAL_CONFIRMATION"
        ),
        "cohort_role": (
            "OUT_OF_COHORT_POSITIVE_CONTROL"
            if args.out_of_cohort_positive_control
            else "CAMPAIGN_CANDIDATE"
        ),
        "external_generalization_claim": False,
        "ranking_json": args.ranking_json.as_posix(),
        "ranking_json_sha256": sha256_file(args.ranking_json),
        "method": {
            "two_model_threshold": args.threshold,
            "requirements": [
                "bright support in both official prediction models",
                "bounded component with model IoU >= 0.30",
                "membership in a detected horizontal row band",
                *STRICT_SCREEN_POLICY,
            ],
            "strict_screen_criterion_id": "campaignx.strict_text_like_screen.v1",
        },
        "candidates": results,
    }
    write_json(output_dir / "GLYPH_CANDIDATE_RECEIPT.json", receipt)
    render_viewer(output_dir / "TEXT_SIGNAL_POSITIVE_CONTROL_REVIEW.html", receipt)
    print(
        json.dumps(
            {
                "output_dir": output_dir.as_posix(),
                "candidate_count": len(results),
                "positive_screens": sum(
                    row["screening_outcome"]
                    in {"POTENTIAL_FIRST_LETTERS_REGION", "TEXT_LIKE_POSITIVE_CONTROL"}
                    for row in results
                ),
                "top_glyph_like_count": results[0]["glyph_like_candidate_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
