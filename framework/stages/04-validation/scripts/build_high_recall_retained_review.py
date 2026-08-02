#!/usr/bin/env python3
"""Build a compact like-for-like review for high-recall CT-gate survivors."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

_CONFIGURED_ROOT = os.environ.get("HELENA_REPO_ROOT", "").strip()
_ROOT_CANDIDATES = (
    [Path(_CONFIGURED_ROOT).expanduser().resolve()] if _CONFIGURED_ROOT else []
) + list(Path(__file__).resolve().parents)
_STAGE_ROOT = next(
    candidate / "framework/stages"
    for candidate in _ROOT_CANDIDATES
    if (candidate / "framework/stages").is_dir()
)
for _stage_scripts in _STAGE_ROOT.glob("*/scripts"):
    _stage_scripts_text = str(_stage_scripts)
    if _stage_scripts_text not in sys.path:
        sys.path.insert(0, _stage_scripts_text)
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from analyze_ink_stability import (
    layer_bounds,
    probability_image,
)
from build_orthogonal_candidate_review import (
    crop_with_padding,
    labeled_panel,
    load_stack,
    map_analysis_half_size_to_source,
    map_analysis_point_to_source,
    ordered_tiff_stack_position,
    orthogonal_views,
)


def load_mean_probability_floats(screening_dir: Path) -> np.ndarray:
    """Read the original float mean map, never the delivered display PNG.

    FIX-05: mean_probability.png was quantized with a hardcoded floor of 0.20
    that sat at or above the p90 of the data, so the top decile was already
    black before this panel opened it.  Re-quantizing that PNG and blending it
    at 0.48 compounded the loss with no route back to the floats.  There is
    deliberately no PNG fallback here: if the floats are absent the panel must
    fail rather than present a truncated layer as evidence.
    """

    direct = screening_dir / "mean_probability.npy"
    if direct.is_file():
        return np.load(direct).astype(np.float32)
    replicas = sorted(screening_dir.glob("center-*_offset-*.npy"))
    if not replicas:
        raise RuntimeError(
            f"no float probability maps under {screening_dir}; refusing to "
            "re-quantize mean_probability.png for the model-context panel"
        )
    return np.stack(
        [np.load(path).astype(np.float32) for path in replicas]
    ).mean(axis=0)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"path escapes Helena Framework root: {path}") from error
    return path


def resolve_screening_directory(
    *,
    root: Path,
    tiff_dir: Path,
    analysis: dict[str, Any],
) -> Path:
    """Locate the exact screening bound by the adapter, with legacy fallback."""

    def has_float_maps(path: Path) -> bool:
        # FIX-05: a screening directory is only usable for review if the
        # original float maps survive; the display PNG is not a source.
        return (path / "mean_probability.npy").is_file() or bool(
            list(path.glob("center-*_offset-*.npy"))
        )

    bound = analysis.get("input", {}).get("screening_directory")
    if bound:
        screening = resolve_under_root(root, str(bound))
        if screening.parent.resolve() != tiff_dir.parent.resolve():
            raise RuntimeError("bound screening directory is outside its robust window")
        if not has_float_maps(screening):
            raise RuntimeError(
                f"bound screening directory has no float probability maps: {screening}"
            )
        return screening

    legacy = tiff_dir.parent / "ink_screening_gp_scroll1_all_shortlist_v1"
    if legacy.is_dir() and has_float_maps(legacy):
        return legacy
    matches = sorted(
        path
        for path in tiff_dir.parent.glob("ink_screening_*")
        if path.is_dir() and has_float_maps(path)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot locate one unambiguous screening directory: {tiff_dir.parent}"
        )
    return matches[0]


def context_bounds(
    bbox_y0_x0_y1_x1: list[int],
    shape_y_x: tuple[int, int],
    *,
    minimum_half_size: int = 128,
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = map(int, bbox_y0_x0_y1_x1)
    center_y = (y0 + y1) // 2
    center_x = (x0 + x1) // 2
    half = max(minimum_half_size, (y1 - y0) // 2 + 24, (x1 - x0) // 2 + 24)
    return (
        max(0, center_y - half),
        max(0, center_x - half),
        min(shape_y_x[0], center_y + half),
        min(shape_y_x[1], center_x + half),
    )


def model_context_panel(
    *,
    central_ct: Image.Image,
    mean_probability: np.ndarray,
    display_bounds: dict[str, Any],
    bbox_y0_x0_y1_x1: list[int],
    candidate_id: str,
    output: Path,
) -> None:
    """Render CT, probability and overlay from the original float map."""

    model_shape = (int(mean_probability.shape[0]), int(mean_probability.shape[1]))
    ct = central_ct.convert("L").resize(
        (model_shape[1], model_shape[0]),
        Image.Resampling.BILINEAR,
    )
    probability = Image.fromarray(
        probability_image(
            mean_probability,
            np.isfinite(mean_probability) & (mean_probability > 0),
            lower=float(display_bounds["display_lower"]),
            upper=float(display_bounds["display_upper"]),
        )
    )
    bounds = context_bounds(bbox_y0_x0_y1_x1, model_shape)
    y0, x0, y1, x1 = bounds
    crop_box = (x0, y0, x1, y1)
    ct_crop = ct.crop(crop_box).resize((420, 420), Image.Resampling.BILINEAR)
    p_crop = probability.crop(crop_box).resize((420, 420), Image.Resampling.BILINEAR)
    ct_rgb = Image.merge("RGB", (ct_crop, ct_crop, ct_crop))
    p_rgb = Image.merge("RGB", (p_crop, p_crop, p_crop))
    heat = Image.new("RGB", ct_rgb.size)
    heat.putdata([(int(value), 0, 0) for value in np.asarray(p_crop).ravel()])
    overlay = Image.blend(ct_rgb, heat, 0.48)

    source_width = max(1, x1 - x0)
    source_height = max(1, y1 - y0)
    by0, bx0, by1, bx1 = map(int, bbox_y0_x0_y1_x1)
    rectangle = (
        round((bx0 - x0) / source_width * 420),
        round((by0 - y0) / source_height * 420),
        round((bx1 - x0) / source_width * 420),
        round((by1 - y0) / source_height * 420),
    )
    for image in (ct_rgb, p_rgb, overlay):
        draw = ImageDraw.Draw(image)
        draw.rectangle(rectangle, outline="#ffd34d", width=3)

    panel = Image.new("RGB", (1260, 452), "#07111e")
    panel.paste(ct_rgb, (0, 32))
    panel.paste(p_rgb, (420, 32))
    panel.paste(overlay, (840, 32))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{candidate_id} · CT central", fill="#eef6ff")
    draw.text(
        (428, 8),
        "probabilidad media · escala "
        f"{float(display_bounds['display_lower']):.3f}–"
        f"{float(display_bounds['display_upper']):.3f}",
        fill="#eef6ff",
    )
    draw.text((848, 8), "overlay · caja amarilla = componente", fill="#eef6ff")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def build_review(
    *,
    root: Path,
    adapter_receipt_path: Path,
    gate_evaluation_path: Path,
    output: Path,
    half_size: int,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    adapter_receipt_path = adapter_receipt_path.resolve()
    gate_evaluation_path = gate_evaluation_path.resolve()
    receipt_path = output / "HIGH_RECALL_RETAINED_REVIEW_RECEIPT.json"
    viewer_path = output / "HIGH_RECALL_RETAINED_REVIEW.html"
    if receipt_path.exists() or viewer_path.exists():
        raise RuntimeError("refusing to overwrite retained-review evidence")

    adapter = read_json(adapter_receipt_path)
    gate = read_json(gate_evaluation_path)
    if adapter.get("status") != "READY_FOR_FROZEN_CT_FEATURE_EXTRACTION":
        raise RuntimeError("high-recall adapter is not ready")
    if gate.get("status") != "COMPLETED":
        raise RuntimeError("CT gate evaluation is not terminal")
    decisions_path = gate_evaluation_path.parent / str(
        gate["artifacts"]["decisions"]
    )
    if sha256_file(decisions_path) != gate["artifacts"]["decisions_sha256"]:
        raise RuntimeError("CT decision artifact hash mismatch")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    retained = [row for row in decisions if row.get("retained") is True]
    if len(retained) != int(gate["retained_count"]):
        raise RuntimeError("retained decision count differs from gate receipt")

    spec_path = resolve_under_root(root, str(adapter["spec"]["path"]))
    if sha256_file(spec_path) != adapter["spec"]["sha256"]:
        raise RuntimeError("adapter CT spec hash mismatch")
    spec = read_json(spec_path)
    groups = {str(row["group_id"]): row for row in spec["groups"]}
    analyses = {
        str(row["window_id"]): row for row in adapter["analyses"]
    }
    retained_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in retained:
        group_id = str(row["group_id"])
        if group_id not in groups or group_id not in analyses:
            raise RuntimeError(f"retained decision has no adapter group: {group_id}")
        retained_by_group.setdefault(group_id, []).append(row)

    artifacts: list[dict[str, Any]] = []
    for group_id in sorted(retained_by_group):
        group = groups[group_id]
        analysis_row = analyses[group_id]
        analysis_path = resolve_under_root(root, str(analysis_row["path"]))
        if sha256_file(analysis_path) != analysis_row["sha256"]:
            raise RuntimeError(f"adapter analysis hash mismatch: {group_id}")
        analysis = read_json(analysis_path)
        candidates = {
            str(row["candidate_id"]): row
            for row in analysis["text_like_screening"]["candidates"]
        }
        tiff_dir = resolve_under_root(root, str(group["tiff_directory"]))
        stack, tiff_files, slice_ordering = load_stack(tiff_dir)
        if len(tiff_files) != 65:
            raise RuntimeError(f"{group_id} does not contain 65 TIFF slices")
        central_position = ordered_tiff_stack_position(
            tiff_files, int(group["central_slice"])
        )
        model_shape = tuple(map(int, analysis["input"]["shape_y_x"]))
        source_shape = (int(stack.shape[1]), int(stack.shape[2]))
        source_half = map_analysis_half_size_to_source(
            analysis_half_size=half_size,
            analysis_shape_y_x=model_shape,
            source_shape_y_x=source_shape,
        )
        screening_dir = resolve_screening_directory(
            root=root,
            tiff_dir=tiff_dir,
            analysis=analysis,
        )
        mean_floats = load_mean_probability_floats(screening_dir)
        mean_valid = np.isfinite(mean_floats) & (mean_floats > 0)
        mean_display_bounds = layer_bounds(mean_floats, mean_valid)
        central_image = Image.fromarray(
            np.asarray(stack[central_position], dtype=np.uint8)
        )

        group_root = output / group_id
        panels: list[Image.Image] = []
        for decision in retained_by_group[group_id]:
            candidate_id = str(decision["candidate_id"])
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise RuntimeError(f"retained candidate missing from adapter: {candidate_id}")
            x0, y0, x1, y1 = map(int, candidate["bbox_xyxy"])
            analysis_y = (y0 + y1) // 2
            analysis_x = (x0 + x1) // 2
            source_y, source_x = map_analysis_point_to_source(
                analysis_y=analysis_y,
                analysis_x=analysis_x,
                analysis_shape_y_x=model_shape,
                source_shape_y_x=source_shape,
            )
            xy, xz, yz = orthogonal_views(
                stack,
                center_y=source_y,
                center_x=source_x,
                central_position=central_position,
                half_size=source_half,
                average_width=2,
            )
            panel = labeled_panel(
                xy,
                xz,
                yz,
                candidate_id=candidate_id,
                analysis_center_y=analysis_y,
                analysis_center_x=analysis_x,
                source_center_y=source_y,
                source_center_x=source_x,
                upper=200,
                depth_scale=4,
            )
            orthogonal_path = group_root / f"{candidate_id}-orthogonal-ct.png"
            orthogonal_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(orthogonal_path)
            panels.append(panel)
            context_path = group_root / f"{candidate_id}-model-context.png"
            model_context_panel(
                central_ct=central_image,
                mean_probability=mean_floats,
                display_bounds=mean_display_bounds,
                bbox_y0_x0_y1_x1=[y0, x0, y1, x1],
                candidate_id=candidate_id,
                output=context_path,
            )
            artifacts.append(
                {
                    "group_id": group_id,
                    "sample_id": analysis["sample_id"],
                    "candidate_id": candidate_id,
                    "slice_ordering": slice_ordering,
                    "routing_score": candidate["high_recall_routing_score"],
                    "bbox_xyxy": candidate["bbox_xyxy"],
                    "gate_checks": decision["checks"],
                    "orthogonal_ct": {
                        "path": str(orthogonal_path.relative_to(output)),
                        "sha256": sha256_file(orthogonal_path),
                    },
                    "model_context": {
                        "path": str(context_path.relative_to(output)),
                        "sha256": sha256_file(context_path),
                        "probability_source": "ORIGINAL_FLOAT_MAPS",
                        **mean_display_bounds,
                    },
                }
            )

    cards = []
    for row in sorted(artifacts, key=lambda value: -float(value["routing_score"])):
        cards.append(
            f"""
            <article>
              <h2>{html.escape(row['sample_id'])} · {html.escape(row['candidate_id'])}</h2>
              <p>Score high-recall: {float(row['routing_score']):.4f}. Retenido
              by CT means localised in depth, not ink.</p>
              <img src="{html.escape(row['model_context']['path'])}" alt="CT, probabilidad y overlay">
              <img src="{html.escape(row['orthogonal_ct']['path'])}" alt="vistas CT XY XZ YZ">
            </article>
            """
        )
    viewer_path.parent.mkdir(parents=True, exist_ok=True)
    viewer_path.write_text(
        """<!doctype html><meta charset="utf-8"><title>High-recall CT review</title>
        <style>
        body{margin:0;background:#08111d;color:#edf4ff;font:17px system-ui}
        header{position:sticky;top:0;background:#0e1a2b;padding:16px 24px;border-bottom:1px solid #345}
        main{padding:18px;display:grid;gap:20px}
        article{background:#101d2d;border:1px solid #345;border-radius:12px;padding:14px}
        h1,h2{margin:0 0 8px} p{color:#b8c7da}
        img{display:block;max-width:100%;margin:10px auto;background:#000}
        </style><header><h1>Helena Framework · supervivientes high-recall + CT</h1>
        <p>Review only. No card here is accepted as ink or as a letter.</p></header><main>"""
        + "\n".join(cards)
        + "</main>",
        encoding="utf-8",
    )
    receipt = {
        "kind": "campaign_x_phase4_high_recall_retained_review_v1",
        "status": "REVIEW_EVIDENCE_READY_NO_AUTOMATIC_ACCEPTANCE",
        "generated_at_utc": utc_now(),
        "adapter": {
            "path": str(adapter_receipt_path),
            "sha256": sha256_file(adapter_receipt_path),
        },
        "gate": {
            "path": str(gate_evaluation_path),
            "sha256": sha256_file(gate_evaluation_path),
            "row_count": int(gate["row_count"]),
            "retained_count": int(gate["retained_count"]),
        },
        "candidate_count": len(artifacts),
        "candidates": artifacts,
        "viewer": {
            "path": viewer_path.name,
            "sha256": sha256_file(viewer_path),
        },
        "explicit_non_claims": [
            "CT retention is not accepted ink",
            "model morphology is not a letter identity",
            "human or stronger independent evidence is still required",
            "not a First Letters submission claim",
        ],
    }
    write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--gate-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--half-size", type=int, default=128)
    args = parser.parse_args()
    receipt = build_review(
        root=args.root,
        adapter_receipt_path=args.adapter_receipt,
        gate_evaluation_path=args.gate_evaluation,
        output=args.output,
        half_size=args.half_size,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
