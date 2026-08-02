#!/usr/bin/env python3
"""Build the spatial PHerc0139 R6.1 quality map from official Villa metrics."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from flattening_gates import (  # noqa: E402
    MAXIMUM_ROI_WINDING_STEP,
    component_summaries,
    label_winding_aware_components,
)


EXPECTED_AREA_FRACTION = 0.7005780483573333
EXPECTED_CHECKPOINT_SHA256 = "13fc568e9fc90954e5d3b9db623ff7d0a4ce24facab173fdb71d618c23e26cd4"
GUARD_EXCLUSION_RADIUS_MM = 2.0
TARGET_WINDOW_AREA_CM2 = 4.0
GREEN_SATISFACTION_FRACTION = 0.90


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reverse_last_adamw_step(
    current: torch.Tensor,
    state: dict[str, Any],
    group: dict[str, Any],
    *,
    learning_rate_used: float,
) -> torch.Tensor:
    """Recover the parameter value immediately before the saved AdamW step."""

    beta1, beta2 = group["betas"]
    step = float(state["step"].item())
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denominator = state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
    denominator = denominator + float(group["eps"])
    update = learning_rate_used / bias_correction1 * state["exp_avg"] / denominator
    decay = 1.0 - learning_rate_used * float(group["weight_decay"])
    return (current + update) / decay


def load_transform(
    checkpoint_path: Path,
    *,
    spiral_root: Path,
    umbilicus_path: Path,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    sys.path.insert(0, str(spiral_root))
    import fit_spiral as fs

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(fs.default_config)
    config.update(checkpoint["cfg"])
    fs.cfg = config
    fs.spiral_outward_sense = "CW"
    fs.umbilicus_z_to_yx = lambda: fs.json_umbilicus_z_to_yx(
        str(umbilicus_path), coordinate_scale=1.0
    )
    z_begin = int(checkpoint["z_begin"])
    z_end = int(checkpoint["z_end"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_zs = np.arange(z_begin, z_end)
    umbilicus_fn = fs.umbilicus_z_to_yx()
    umbilicus_zyx = torch.from_numpy(
        np.concatenate([all_zs[:, None], umbilicus_fn(all_zs)], axis=-1).astype(np.float32)
    ).to(device)
    radius = config["flow_bounds_radius"]
    flow_min = torch.tensor(
        [z_begin - config["flow_bounds_z_margin"], -radius, -radius],
        dtype=torch.int64,
        device=device,
    )
    flow_max = torch.tensor(
        [z_end + config["flow_bounds_z_margin"], radius, radius],
        dtype=torch.int64,
        device=device,
    )
    model = fs.SpiralAndTransform(
        flow_integration_steps=config["num_flow_integration_steps"],
        flow_integration_solver=config["flow_integration_solver"],
        umbilicus_zyx=umbilicus_zyx,
        flow_min_corner_zyx=flow_min,
        flow_max_corner_zyx=flow_max,
        config=config,
        spiral_outward_sense=fs.spiral_outward_sense,
    )
    model.to(device)
    model.load_state_dict(checkpoint["spiral_and_transform"])
    model.eval()
    final_iteration = int(config["num_training_steps"]) - 1
    scale_fraction = min(
        1.0,
        max(
            0.0,
            (final_iteration - float(config["flow_field_high_res_lr_ramp_start_step"]))
            / max(1, int(config["flow_field_high_res_lr_ramp_steps"])),
        ),
    )
    model.flow_field.flow_scales[1] = min(
        1.0,
        float(config["flow_field_high_res_lr_scale_initial"])
        + scale_fraction
        * (
            float(config["flow_field_high_res_lr_scale_final"])
            - float(config["flow_field_high_res_lr_scale_initial"])
        ),
    )
    # Villa constructs the transform and dr immediately before the last
    # optimiser step, then saves the post-step checkpoint and evaluates using
    # those pre-step derived tensors. Reconstruct that exact mixed state rather
    # than evaluating the fully post-step model, which moves a few threshold
    # quads and no longer reproduces satisfied_fitted.json.
    optimiser = checkpoint["optimiser"]
    groups = optimiser["param_groups"]
    gamma = (
        float(config["lr_final_factor"]) ** (1.0 / max(1, int(config["num_training_steps"])))
        if config["exp_lr_schedule"]
        else 1.0
    )
    learning_rate_used = float(groups[0]["lr"]) / gamma
    dr_group = groups[0]
    linear_group = groups[1]
    flow_group = groups[3]
    flow_parameters = list(model.flow_field.parameters())
    if (
        len(dr_group["params"]) != 1
        or len(linear_group["params"]) != 1
        or len(flow_group["params"]) != len(flow_parameters)
    ):
        raise RuntimeError("unexpected optimiser parameter grouping")
    dr_state = optimiser["state"][dr_group["params"][0]]
    linear_state = optimiser["state"][linear_group["params"][0]]
    current_dr_logit = model.dr_per_winding_logit.detach().clone()
    current_linear = model.linear_logits.detach().clone()
    previous_dr_logit = reverse_last_adamw_step(
        current_dr_logit.cpu(), dr_state, dr_group, learning_rate_used=learning_rate_used
    ).to(device)
    previous_linear = reverse_last_adamw_step(
        current_linear.cpu(), linear_state, linear_group, learning_rate_used=learning_rate_used
    ).to(device)
    previous_flow = [
        reverse_last_adamw_step(
            parameter.detach().cpu(),
            optimiser["state"][parameter_id],
            flow_group,
            learning_rate_used=learning_rate_used,
        ).to(device)
        for parameter, parameter_id in zip(
            flow_parameters, flow_group["params"], strict=True
        )
    ]
    with torch.no_grad():
        model.dr_per_winding_logit.copy_(previous_dr_logit)
        model.linear_logits.copy_(previous_linear)
        for parameter, previous in zip(flow_parameters, previous_flow, strict=True):
            parameter.copy_(previous)
        transform = model.get_slice_to_spiral_transform()
        dr = model.get_dr_per_winding().detach()
    metadata = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "z_range": [z_begin, z_end],
        "dr_per_winding": float(dr.cpu()),
        "device": str(device),
        "erode_patches": int(config["erode_patches"]),
        "official_final_evaluation_state": "PRE_LAST_STEP_DR_LINEAR_AND_FLOW_WITH_POST_STEP_GAP_PARAMETERS",
        "reversed_adamw_step": int(dr_state["step"].item()),
        "learning_rate_used_for_reversal": learning_rate_used,
    }
    return transform, dr, metadata


def in_roi_valid_mask(patch: Any, z_begin: int, z_end: int) -> np.ndarray:
    z = patch.zyxs[..., 0]
    quad_zs = torch.stack(
        [z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:]], dim=0
    )
    touches = (quad_zs.amax(dim=0) >= z_begin) & (quad_zs.amin(dim=0) < z_end)
    return (patch.valid_quad_mask & touches).cpu().numpy().astype(bool)


def boundary_mask(valid: np.ndarray) -> np.ndarray:
    padded = np.pad(valid, 1, constant_values=False)
    all_neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return valid & ~all_neighbors


def label_components(
    mask: np.ndarray,
    cell_area_cm2: float,
    winding: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Label safe components, cutting them at winding (sheet) discontinuities.

    Plain four-connectivity joins quads that are neighbours in the flattened
    map but sit on different wraps of the spiral.  ``target_winding_index`` was
    already stored in the NPZ with no reader; it is the cut criterion here.
    """

    labels, _ = label_winding_aware_components(
        mask, winding, maximum_winding_step=MAXIMUM_ROI_WINDING_STEP
    )
    return labels.astype(np.int32), component_summaries(labels, cell_area_cm2)


def _integral(array: np.ndarray) -> np.ndarray:
    result = np.pad(array.astype(np.float64), ((1, 0), (1, 0)))
    return result.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(integral: np.ndarray, i0: int, j0: int, i1: int, j1: int) -> float:
    return float(
        integral[i1, j1] - integral[i0, j1] - integral[i1, j0] + integral[i0, j0]
    )


def exact_square_sum(array: np.ndarray, i: int, j: int, side_cells: float) -> float:
    """Area-weighted sum in an exact square with a fractional last row/column."""

    whole = int(math.floor(side_cells))
    fraction = float(side_cells - whole)
    integral = _integral(array)
    total = _rect_sum(integral, i, j, i + whole, j + whole)
    if fraction > 0:
        total += fraction * _rect_sum(integral, i + whole, j, i + whole + 1, j + whole)
        total += fraction * _rect_sum(integral, i, j + whole, i + whole, j + whole + 1)
        total += fraction * fraction * float(array[i + whole, j + whole])
    return total


def exact_window_candidates(
    *,
    valid: np.ndarray,
    safe_satisfied: np.ndarray,
    guard: np.ndarray,
    labels: np.ndarray,
    pitch_um: float,
    area_cm2: float = TARGET_WINDOW_AREA_CM2,
    keep: int = 100,
) -> tuple[list[dict[str, Any]], int]:
    side_um = math.sqrt(area_cm2 * 100_000_000.0)
    side_cells = side_um / pitch_um
    extent = int(math.ceil(side_cells))
    h, w = valid.shape
    if h < extent or w < extent:
        return [], 0
    integrals = {
        "valid": _integral(valid),
        "safe": _integral(safe_satisfied),
        "guard": _integral(guard),
    }
    whole = int(math.floor(side_cells))
    fraction = float(side_cells - whole)
    footprint_cells = side_cells * side_cells

    def fast_sum(name: str, array: np.ndarray, i: int, j: int) -> float:
        integral = integrals[name]
        total = _rect_sum(integral, i, j, i + whole, j + whole)
        if fraction > 0:
            total += fraction * _rect_sum(integral, i + whole, j, i + whole + 1, j + whole)
            total += fraction * _rect_sum(integral, i, j + whole, i + whole, j + whole + 1)
            total += fraction * fraction * float(array[i + whole, j + whole])
        return total

    ranked: list[dict[str, Any]] = []
    green_count = 0
    for i in range(0, h - extent + 1):
        for j in range(0, w - extent + 1):
            valid_sum = fast_sum("valid", valid, i, j)
            safe_sum = fast_sum("safe", safe_satisfied, i, j)
            guard_sum = fast_sum("guard", guard, i, j)
            valid_fraction = valid_sum / footprint_cells
            safe_fraction = safe_sum / footprint_cells
            is_green = (
                valid_fraction >= 1.0 - 1e-9
                and safe_fraction >= GREEN_SATISFACTION_FRACTION
                and guard_sum <= 1e-12
            )
            green_count += int(is_green)
            score = safe_fraction - 2.0 * (1.0 - valid_fraction) - 4.0 * guard_sum / footprint_cells
            ranked.append(
                {
                    "top_left_quad_ij": [i, j],
                    "side_cells": side_cells,
                    "side_mm": side_um / 1000.0,
                    "area_cm2": area_cm2,
                    "valid_fraction": valid_fraction,
                    "safe_satisfied_fraction": safe_fraction,
                    "guard_fraction": guard_sum / footprint_cells,
                    "green": is_green,
                    "score": score,
                }
            )
    ranked.sort(key=lambda item: (-item["score"], item["top_left_quad_ij"]))
    top = ranked[:keep]
    for item in top:
        i, j = item["top_left_quad_ij"]
        center_i = min(int(i + side_cells / 2), labels.shape[0] - 1)
        center_j = min(int(j + side_cells / 2), labels.shape[1] - 1)
        item["center_component_id"] = int(labels[center_i, center_j])
    return top, green_count


def load_constraint_lookup(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json_load(path)
        for item in payload.get("constraints", []):
            lookup[item["constraint_id"]] = item
    return lookup


def guard_endpoints(
    policy: dict[str, Any], lookup: dict[str, dict[str, Any]]
) -> dict[str, list[np.ndarray]]:
    ids = [
        policy["known_original_outlier"]["constraint_id"],
        policy["isolated_spatial_sign_disagreement"]["constraint_id"],
    ]
    endpoints: dict[str, list[np.ndarray]] = {}
    for constraint_id in ids:
        row = lookup.get(constraint_id)
        if row is None:
            raise RuntimeError(f"guard constraint has no coordinate source: {constraint_id}")
        endpoints.setdefault(row["source_wrap"], []).append(
            np.asarray(row["endpoint_a_xyz_l0"], dtype=np.float32)[::-1]
        )
        endpoints.setdefault(row["target_wrap"], []).append(
            np.asarray(row["endpoint_b_xyz_l0"], dtype=np.float32)[::-1]
        )
    return endpoints


def training_points(pcl_path: Path) -> np.ndarray:
    payload = json_load(pcl_path)
    points: list[list[float]] = []
    for collection in payload["collections"].values():
        for point in collection["points"].values():
            points.append(list(reversed(point["p"])))
    return np.asarray(points, dtype=np.float32)


def centers_zyx(patch: Any) -> np.ndarray:
    z = patch.zyxs
    centers = (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:]) / 4
    return centers.cpu().numpy()


def render_patch_png(
    path: Path,
    *,
    valid: np.ndarray,
    satisfied: np.ndarray,
    guard: np.ndarray,
    boundary: np.ndarray,
) -> None:
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    rgb[valid] = [125, 40, 45]
    rgb[satisfied & valid] = [35, 170, 95]
    rgb[boundary] = [238, 185, 40]
    rgb[guard] = [220, 60, 220]
    Image.fromarray(rgb).save(path)


def make_overview(output: Path, patches: list[tuple[str, Path, dict[str, Any]]]) -> None:
    panel_width = 520
    panel_height = 560
    canvas = Image.new("RGB", (panel_width * len(patches), panel_height), "#0a0f18")
    draw = ImageDraw.Draw(canvas)
    for index, (patch_id, image_path, metrics) in enumerate(patches):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((panel_width - 24, panel_height - 100), Image.Resampling.NEAREST)
        x = index * panel_width + 12
        canvas.paste(image, (x, 54))
        draw.text((x, 14), f"{patch_id} · {metrics['satisfied_fraction']:.2%}", fill="white")
        draw.text(
            (x, panel_height - 32),
            f"componente mayor {metrics['largest_safe_component_cm2']:.2f} cm²",
            fill="#b6c2d9",
        )
    canvas.save(output)


def make_viewer(output: Path, receipt: dict[str, Any]) -> None:
    cards = []
    for patch in receipt["patches"]:
        patch_id = html.escape(patch["patch_id"])
        cards.append(
            f"""<article><h2>{patch_id}</h2>
<img src="{patch_id}-quality.png" alt="Mapa de calidad {patch_id}">
<p>Satisfied: <b>{patch['satisfied_fraction']:.2%}</b> ·
componente seguro mayor: <b>{patch['largest_safe_component_cm2']:.2f} cm²</b> ·
ventanas verdes: <b>{patch['green_4cm2_window_count']}</b></p></article>"""
        )
    output.write_text(
        """<!doctype html><meta charset="utf-8"><title>Helena Framework · P4.1</title>
<style>body{margin:0;background:#090e17;color:#e8edf7;font:16px system-ui;padding:24px}
.note{background:#1a2333;border-left:5px solid #e6b83f;padding:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
article{background:#111a29;border:1px solid #2a3a55;border-radius:12px;padding:14px}
img{width:100%;image-rendering:pixelated;background:#000}
b{color:#68d99d}</style>
<h1>Helena Framework · mapa espacial R6.1 de PHerc0139</h1>
<p class="note">Geometry only. Green=satisfied; red=valid but unsatisfied;
yellow=border; magenta=guard exclusion. It contains and uses no ink.</p>
<div class="grid">"""
        + "\n".join(cards)
        + "</div>\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--spiral-root", type=Path, required=True)
    parser.add_argument("--umbilicus", type=Path, required=True)
    parser.add_argument("--train-pcl", type=Path, required=True)
    parser.add_argument("--guard-policy", type=Path, required=True)
    parser.add_argument("--guard-constraints", type=Path, action="append", required=True)
    parser.add_argument("--expected-satisfaction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-um", type=float, default=9.362)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("checkpoint hash does not match frozen R6.1 champion")
    sys.path.insert(0, str(args.spiral_root))
    from satisfaction_metrics import get_patch_satisfied_areas, metrics_config
    from spiral_helpers import erode_patch_valid_region
    from tifxyz import load_tifxyz

    transform, dr, transform_meta = load_transform(
        args.checkpoint, spiral_root=args.spiral_root, umbilicus_path=args.umbilicus
    )
    patch_ids = sorted(path.name for path in args.patches.iterdir() if path.is_dir())
    patches: dict[str, Any] = {}
    for patch_id in patch_ids:
        patch = load_tifxyz(args.patches / patch_id)
        if transform_meta["erode_patches"] > 0:
            if not erode_patch_valid_region(patch, transform_meta["erode_patches"]):
                raise RuntimeError(f"erosion removed patch {patch_id}")
        patches[patch_id] = patch
    ordered = [patches[name] for name in patch_ids]
    (
        satisfied_patches,
        satisfied_areas,
        total_areas,
        satisfied_masks,
        boundary_satisfied,
        winding_indices,
    ) = get_patch_satisfied_areas(
        transform,
        dr,
        ordered,
        transform_meta["z_range"][0],
        transform_meta["z_range"][1],
    )

    total_satisfied = float(satisfied_areas.sum())
    total_area = float(total_areas.sum())
    aggregate = total_satisfied / total_area
    expected = json_load(args.expected_satisfaction)
    expected_by_patch = {row["id"]: row for row in expected["patches"]}
    aggregate_matches = abs(aggregate - EXPECTED_AREA_FRACTION) <= 1e-15
    if not aggregate_matches:
        diagnostic = {
            patch_id: {
                "actual_satisfied_area": float(satisfied_areas[index]),
                "expected_satisfied_area": float(expected_by_patch[patch_id]["satisfied_area"]),
                "actual_total_area": float(total_areas[index]),
                "expected_total_area": float(expected_by_patch[patch_id]["total_area"]),
            }
            for index, patch_id in enumerate(patch_ids)
        }
        print(json.dumps({"aggregate": aggregate, "patches": diagnostic}, indent=2))
        raise RuntimeError(
            f"official aggregate mismatch: got {aggregate}, expected {EXPECTED_AREA_FRACTION}"
        )

    lookup = load_constraint_lookup(args.guard_constraints)
    endpoints = guard_endpoints(json_load(args.guard_policy), lookup)
    train_tree = cKDTree(training_points(args.train_pcl))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    component_payload: dict[str, Any] = {}
    window_payload: dict[str, Any] = {}
    patch_receipts: list[dict[str, Any]] = []
    overview_inputs: list[tuple[str, Path, dict[str, Any]]] = []

    for index, patch_id in enumerate(patch_ids):
        patch = patches[patch_id]
        valid = in_roi_valid_mask(
            patch, transform_meta["z_range"][0], transform_meta["z_range"][1]
        )
        satisfied = satisfied_masks[index].numpy().astype(bool)
        boundary = boundary_mask(valid)
        centers = centers_zyx(patch)
        flat_centers = centers.reshape(-1, 3)
        train_distance_vox = train_tree.query(flat_centers, workers=-1)[0].reshape(valid.shape)
        guard = np.zeros(valid.shape, dtype=bool)
        for point in endpoints.get(patch_id, []):
            distance_vox = np.linalg.norm(centers - point[None, None, :], axis=-1)
            guard |= distance_vox * args.voxel_size_um / 1000.0 <= GUARD_EXCLUSION_RADIUS_MM
        guard &= valid
        safe_satisfied = satisfied & ~guard
        border_distance_mm = (
            ndimage.distance_transform_edt(valid)
            * float((1.0 / patch.scale).mean())
            * args.voxel_size_um
            / 1000.0
        )
        pitch_um_axes = (1.0 / patch.scale.cpu().numpy()) * args.voxel_size_um
        if not np.isclose(pitch_um_axes[0], pitch_um_axes[1], rtol=0, atol=1e-5):
            raise RuntimeError(f"non-square tifxyz pitch for {patch_id}: {pitch_um_axes}")
        pitch_um = float(pitch_um_axes.mean())
        cell_area_cm2 = pitch_um * pitch_um / 100_000_000.0
        patch_winding = winding_indices[index].numpy()
        labels, components = label_components(
            safe_satisfied, cell_area_cm2, patch_winding
        )
        windows, green_count = exact_window_candidates(
            valid=valid,
            safe_satisfied=safe_satisfied,
            guard=guard,
            labels=labels,
            pitch_um=pitch_um,
        )

        official_satisfied = float(satisfied_areas[index])
        official_total = float(total_areas[index])
        expected_row = expected_by_patch[patch_id]
        if official_satisfied != float(expected_row["satisfied_area"]):
            raise RuntimeError(f"patch satisfied area mismatch: {patch_id}")
        if official_total != float(expected_row["total_area"]):
            raise RuntimeError(f"patch total area mismatch: {patch_id}")

        npz_path = output_dir / f"{patch_id}-quality.npz"
        np.savez_compressed(
            npz_path,
            valid=valid,
            satisfied=satisfied,
            guard=guard,
            boundary=boundary,
            safe_satisfied=safe_satisfied,
            component_labels=labels,
            target_winding_index=patch_winding,
            nearest_train_distance_mm=(train_distance_vox * args.voxel_size_um / 1000.0).astype(np.float32),
            border_distance_mm=border_distance_mm.astype(np.float32),
        )
        png_path = output_dir / f"{patch_id}-quality.png"
        render_patch_png(
            png_path, valid=valid, satisfied=satisfied, guard=guard, boundary=boundary
        )
        largest = components[0]["area_cm2"] if components else 0.0
        patch_receipt = {
            "patch_id": patch_id,
            "shape_quads": list(valid.shape),
            "pitch_um": pitch_um,
            "cell_area_cm2": cell_area_cm2,
            "valid_quad_count": int(valid.sum()),
            "satisfied_quad_count": int(satisfied.sum()),
            "guard_quad_count": int(guard.sum()),
            "satisfied_area_scan_vox2": official_satisfied,
            "total_area_scan_vox2": official_total,
            "satisfied_fraction": official_satisfied / official_total,
            "official_patch_satisfied": bool(satisfied_patches[index]),
            "official_boundary_satisfied": bool(boundary_satisfied[index]),
            "safe_component_count": len(components),
            "component_labelling": "FOUR_NEIGHBOUR_CUT_AT_WINDING_STEP_V1",
            "maximum_roi_winding_step": MAXIMUM_ROI_WINDING_STEP,
            "largest_safe_component_cm2": largest,
            "green_4cm2_window_count": green_count,
            "npz_path": npz_path.name,
            "npz_sha256": sha256_file(npz_path),
            "png_path": png_path.name,
            "png_sha256": sha256_file(png_path),
        }
        patch_receipts.append(patch_receipt)
        component_payload[patch_id] = components
        window_payload[patch_id] = windows
        overview_inputs.append((patch_id, png_path, patch_receipt))

    components_path = output_dir / "QUALITY_COMPONENTS.json"
    components_path.write_text(
        json.dumps(
            {
                "kind": "campaign_x_phase4_quality_components_v1",
                "patches": component_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    windows_path = output_dir / "ROI_4CM2_CANDIDATES.json"
    windows_path.write_text(
        json.dumps(
            {
                "kind": "campaign_x_phase4_exact_4cm2_windows_v1",
                "area_cm2": TARGET_WINDOW_AREA_CM2,
                "square_side_mm": 20.0,
                "classification": {
                    "green_satisfaction_fraction_minimum": GREEN_SATISFACTION_FRACTION,
                    "valid_fraction_required": 1.0,
                    "guard_fraction_required": 0.0,
                },
                "patches": window_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    overview_path = output_dir / "quality_overview.png"
    make_overview(overview_path, overview_inputs)
    receipt = {
        "kind": "campaign_x_phase4_pherc0139_quality_map_receipt_v1",
        "generated_at_utc": utc_now(),
        "status": "PASSED" if aggregate_matches else "FAILED",
        "gate": "P4_1_QUALITY_MAP_VALIDATED",
        "scope": "GEOMETRY_ONLY_NO_INK",
        "coordinate_frame": "PHerc0139/20250728140407-9.362um",
        "checkpoint": transform_meta,
        "official_metrics_config": metrics_config,
        "aggregate": {
            "satisfied_area_scan_vox2": total_satisfied,
            "total_area_scan_vox2": total_area,
            "satisfied_fraction": aggregate,
            "expected_satisfied_fraction": EXPECTED_AREA_FRACTION,
            "absolute_difference": abs(aggregate - EXPECTED_AREA_FRACTION),
            "matches_frozen_result": aggregate_matches,
        },
        "guard": {
            "policy": str(args.guard_policy),
            "exclusion_radius_mm": GUARD_EXCLUSION_RADIUS_MM,
            "constraint_ids": [
                json_load(args.guard_policy)["known_original_outlier"]["constraint_id"],
                json_load(args.guard_policy)["isolated_spatial_sign_disagreement"]["constraint_id"],
            ],
        },
        "exact_window": {
            "area_cm2": TARGET_WINDOW_AREA_CM2,
            "square_side_mm": 20.0,
            "green_satisfaction_fraction_minimum": GREEN_SATISFACTION_FRACTION,
        },
        "patches": patch_receipts,
        "artifacts": {
            "components": components_path.name,
            "components_sha256": sha256_file(components_path),
            "windows": windows_path.name,
            "windows_sha256": sha256_file(windows_path),
            "overview": overview_path.name,
            "overview_sha256": sha256_file(overview_path),
            "viewer": "quality_viewer.html",
        },
        "privacy": "PRIVATE",
        "ink_used": False,
    }
    receipt_path = output_dir / "QUALITY_MAP_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_viewer(output_dir / "quality_viewer.html", receipt)
    print(json.dumps(receipt["aggregate"], indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                item["patch_id"]: {
                    "largest_safe_component_cm2": item["largest_safe_component_cm2"],
                    "green_4cm2_window_count": item["green_4cm2_window_count"],
                }
                for item in patch_receipts
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
