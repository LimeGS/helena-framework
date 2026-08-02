#!/usr/bin/env python3
"""Build a fail-closed geometry-only quality map for one Phase 4 target fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import sys

_STAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()) / "framework/stages"
for _stage_scripts in _STAGE_ROOT.glob("*/scripts"):
    _stage_scripts_text = str(_stage_scripts)
    if _stage_scripts_text not in sys.path:
        sys.path.insert(0, _stage_scripts_text)
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from build_quality_map import (
    GREEN_SATISFACTION_FRACTION,
    GUARD_EXCLUSION_RADIUS_MM,
    TARGET_WINDOW_AREA_CM2,
    boundary_mask,
    centers_zyx,
    exact_window_candidates,
    in_roi_valid_mask,
    label_components,
    load_transform,
    make_overview,
    make_viewer,
    render_patch_png,
    sha256_file,
    training_points,
    utc_now,
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


def one_file(root: Path, name: str) -> Path:
    matches = sorted(root.glob(f"*/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} below {root}, found {matches}")
    return matches[0]


def failed_guard_points(guard: dict[str, Any]) -> np.ndarray:
    """Return every endpoint of every failed post-fit relation in ZYX order."""

    points: list[list[float]] = []
    for row in guard.get("rows", []):
        if bool(row.get("pass")):
            continue
        for key in ("endpoint_a_xyz_l0", "endpoint_b_xyz_l0"):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != 3:
                raise RuntimeError(f"failed guard row has no valid {key}")
            points.append([float(value[2]), float(value[1]), float(value[0])])
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def incorrect_sign_guard_points(
    evaluation: dict[str, Any], candidate_arm: str
) -> tuple[np.ndarray, int]:
    """Return endpoints of physically contradictory relations for one arm."""

    rows = evaluation.get(candidate_arm, {}).get("all_rows", [])
    failed_rows = [row for row in rows if not bool(row.get("correct_sign"))]
    points: list[list[float]] = []
    for row in failed_rows:
        for key in ("endpoint_a_xyz_l0", "endpoint_b_xyz_l0"):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != 3:
                raise RuntimeError(f"incorrect-sign row has no valid {key}")
            points.append([float(value[2]), float(value[1]), float(value[0])])
    if not points:
        return np.empty((0, 3), dtype=np.float32), len(failed_rows)
    return np.asarray(points, dtype=np.float32), len(failed_rows)


def classify_roi(
    *, green_count: int, best_window: dict[str, Any] | None
) -> tuple[str, bool, str]:
    if green_count > 0:
        return (
            "GREEN_QUALIFIED_ROI",
            True,
            "at least one exact 4 cm2 window is fully valid, guard-free and >=90% satisfied",
        )
    if (
        best_window is not None
        and float(best_window["valid_fraction"]) >= 1.0 - 1e-9
        and float(best_window["guard_fraction"]) <= 1e-12
        and float(best_window["safe_satisfied_fraction"]) >= 0.80
    ):
        return (
            "AMBER_HIGH_BOUNDED_REPAIR",
            True,
            "best exact 4 cm2 window is fully valid, guard-free and >=80% satisfied",
        )
    return (
        "RED_NO_QUALIFIED_ROI",
        False,
        "no exact 4 cm2 window meets the frozen GREEN or high-AMBER geometry gate",
    )


def validate_evaluation(
    *,
    evaluation: dict[str, Any],
    sample_id: str,
    candidate_arm: str,
    allow_failed_baseline_rescue: bool,
    allow_failed_experimental_rescue: bool = False,
    allow_weak_holdout_local_screen: bool = False,
) -> dict[str, Any]:
    """Validate the A/B binding without silently weakening the normal path.

    The normal rescue path is deliberately narrow.  A second explicit
    exploratory opt-in can reconstruct the frozen experimental arm after an
    ink-blind A/B failure, but it never promotes that arm as passed and it
    preserves every downstream geometry guard.
    """

    if evaluation.get("sample_id") != sample_id:
        raise RuntimeError("A/B sample binding mismatch")
    if bool(evaluation.get("ink_used")):
        raise RuntimeError("A/B evaluation used ink")
    gates = evaluation.get("gates", {})
    passed = evaluation.get("status") == "PASSED" and bool(gates) and all(
        bool(value) for value in gates.values()
    )
    if passed:
        return {
            "mode": "FROZEN_AB_PASS",
            "evaluation_status": "PASSED",
            "failed_gates": [],
        }
    if not allow_failed_baseline_rescue and not allow_failed_experimental_rescue:
        raise RuntimeError("quality map requires a completely passed frozen A/B")
    if candidate_arm == "baseline":
        if not allow_failed_baseline_rescue:
            raise RuntimeError(
                "failed-A/B baseline rescue requires its explicit opt-in"
            )
        rescue_mode = "FAILED_AB_BASELINE_LOCAL_FUNCTIONAL_RESCUE"
    elif candidate_arm == "r61":
        if not allow_failed_experimental_rescue:
            raise RuntimeError(
                "failed-A/B experimental rescue requires its explicit opt-in"
            )
        rescue_mode = "FAILED_AB_EXPERIMENTAL_LOCAL_FUNCTIONAL_RESCUE"
    else:
        raise RuntimeError(
            "failed-A/B rescue supports only baseline or r61 candidate arms"
        )
    if not bool(gates.get("satisfaction_map_generated")):
        raise RuntimeError(
            "failed-A/B rescue requires a frozen satisfaction artifact"
        )
    weak_holdout = not bool(gates.get("heldout_at_least_12"))
    weak_holdout_receipt: dict[str, Any] = {}
    if weak_holdout:
        if not allow_weak_holdout_local_screen:
            raise RuntimeError(
                "failed-A/B rescue requires heldout and satisfaction artifacts"
            )
        declared_count = int(evaluation.get("heldout_count", 0))
        heldout = evaluation.get(candidate_arm, {}).get("heldout", {})
        evaluable_count = int(heldout.get("count", 0))
        if declared_count < 12 or evaluable_count < 10 or evaluable_count >= 12:
            raise RuntimeError(
                "weak-holdout screen requires >=12 declared and 10-11 evaluable relations"
            )
        rescue_mode += "_WEAK_HOLDOUT"
        weak_holdout_receipt = {
            "weak_holdout": True,
            "declared_holdout_count": declared_count,
            "evaluable_holdout_count": evaluable_count,
            "minimum_evaluable_count": 10,
        }
    failed_gates = sorted(name for name, value in gates.items() if not bool(value))
    if not failed_gates:
        raise RuntimeError("baseline rescue requires a recorded failed A/B gate")
    return {
        "mode": rescue_mode,
        "evaluation_status": str(evaluation.get("status")),
        "failed_gates": failed_gates,
        "claim_limit": (
            "PRIVATE_WEAK_HOLDOUT_LOCAL_FUNCTIONAL_SCREEN_ONLY"
            if weak_holdout
            else "PRIVATE_LOCAL_FUNCTIONAL_DIAGNOSTIC_ONLY"
        ),
        **weak_holdout_receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--spiral-root", type=Path, required=True)
    parser.add_argument("--fit-subdir", default="fit_ab")
    parser.add_argument("--candidate-arm", default="r61")
    parser.add_argument("--output-subdir", default="quality_map")
    parser.add_argument(
        "--allow-failed-ab-baseline-rescue",
        action="store_true",
        help=(
            "permit only the community baseline after a failed ink-blind A/B; "
            "the result remains a private local-functional diagnostic"
        ),
    )
    parser.add_argument(
        "--allow-failed-ab-experimental-rescue",
        action="store_true",
        help=(
            "permit only the frozen r61 arm after a failed ink-blind A/B; "
            "incorrect-sign endpoints become guards and the result remains "
            "a private local-functional diagnostic"
        ),
    )
    parser.add_argument(
        "--allow-weak-holdout-local-screen",
        action="store_true",
        help=(
            "allow only 10-11 evaluable relations from at least 12 declared "
            "holdouts; the output is restricted to one private local screen"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    target = root / "phase4" / "targets" / args.sample_id
    fit_ab = target / args.fit_subdir
    evaluation_path = fit_ab / "AB_EVALUATION.json"
    evaluation = read_json(evaluation_path)
    evaluation_qualification = validate_evaluation(
        evaluation=evaluation,
        sample_id=args.sample_id,
        candidate_arm=args.candidate_arm,
        allow_failed_baseline_rescue=args.allow_failed_ab_baseline_rescue,
        allow_failed_experimental_rescue=args.allow_failed_ab_experimental_rescue,
        allow_weak_holdout_local_screen=args.allow_weak_holdout_local_screen,
    )
    rescue_mode = evaluation_qualification["mode"].startswith("FAILED_AB_")
    experimental_rescue_mode = (
        evaluation_qualification["mode"].startswith(
            "FAILED_AB_EXPERIMENTAL_LOCAL_FUNCTIONAL_RESCUE"
        )
    )
    weak_holdout_mode = bool(evaluation_qualification.get("weak_holdout"))

    dataset = fit_ab / "dataset"
    dataset_receipt_path = dataset / "DATASET_RECEIPT.json"
    dataset_receipt = read_json(dataset_receipt_path)
    if dataset_receipt.get("sample_id") != args.sample_id:
        raise RuntimeError("dataset sample binding mismatch")
    if dataset_receipt.get("status") != "READY":
        raise RuntimeError("dataset is not ready")
    if bool(dataset_receipt.get("ink_used")) or bool(
        dataset_receipt.get("ct_intensity_used")
    ):
        raise RuntimeError("dataset preparation used intensity or ink")
    if dataset_receipt.get("spiral_outward_sense") != "CW":
        raise RuntimeError("target quality-map reconstruction currently requires CW")

    guard_path = fit_ab / "POST_FIT_RELATION_GUARD.json"
    guard_document = read_json(guard_path)
    if guard_document.get("sample_id") != args.sample_id:
        raise RuntimeError("guard sample binding mismatch")
    if rescue_mode:
        guard_points, guard_failed_count = incorrect_sign_guard_points(
            evaluation, args.candidate_arm
        )
        guard_source_path = evaluation_path
        guard_criterion = (
            "candidate-arm relations with physically incorrect winding sign; "
            "magnitude error is not cross-arm calibrated"
        )
    else:
        guard_points = failed_guard_points(guard_document)
        guard_failed_count = int(guard_document["failed_count"])
        guard_source_path = guard_path
        guard_criterion = "frozen post-fit absolute winding-error guard"

    fit_root = fit_ab / "fits" / args.candidate_arm
    checkpoint_path = one_file(fit_root, "checkpoint_fitted.ckpt")
    expected_path = one_file(fit_root, "satisfied_fitted.json")
    expected = read_json(expected_path)
    expected_by_patch = {str(row["id"]): row for row in expected["patches"]}
    patches_root = dataset / "verified_patches"
    patch_ids = sorted(path.name for path in patches_root.iterdir() if path.is_dir())
    if patch_ids != sorted(expected_by_patch):
        raise RuntimeError("patch inventory differs from frozen satisfaction result")

    sys.path.insert(0, str(args.spiral_root.resolve()))
    from satisfaction_metrics import get_patch_satisfied_areas, metrics_config
    from spiral_helpers import erode_patch_valid_region
    from tifxyz import load_tifxyz

    transform, dr, transform_meta = load_transform(
        checkpoint_path,
        spiral_root=args.spiral_root.resolve(),
        umbilicus_path=dataset / "umbilicus.json",
    )
    if transform_meta["z_range"] != [int(value) for value in dataset_receipt["z_range"]]:
        raise RuntimeError("checkpoint z range differs from frozen dataset")

    patches: dict[str, Any] = {}
    for patch_id in patch_ids:
        patch = load_tifxyz(patches_root / patch_id)
        if transform_meta["erode_patches"] > 0 and not erode_patch_valid_region(
            patch, transform_meta["erode_patches"]
        ):
            raise RuntimeError(f"erosion removed patch {patch_id}")
        patches[patch_id] = patch
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
        [patches[name] for name in patch_ids],
        transform_meta["z_range"][0],
        transform_meta["z_range"][1],
    )

    expected_satisfied = sum(float(row["satisfied_area"]) for row in expected["patches"])
    expected_total = sum(float(row["total_area"]) for row in expected["patches"])
    actual_satisfied = float(satisfied_areas.sum())
    actual_total = float(total_areas.sum())
    if actual_satisfied != expected_satisfied or actual_total != expected_total:
        raise RuntimeError(
            "official satisfaction reconstruction does not match satisfied_fitted.json: "
            f"{actual_satisfied}/{actual_total} != {expected_satisfied}/{expected_total}"
        )

    train_tree = cKDTree(training_points(dataset / "relative_windings_train.json"))
    output = fit_ab / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    components_payload: dict[str, Any] = {}
    windows_payload: dict[str, Any] = {}
    patch_receipts: list[dict[str, Any]] = []
    overview_inputs: list[tuple[str, Path, dict[str, Any]]] = []

    for index, patch_id in enumerate(patch_ids):
        expected_row = expected_by_patch[patch_id]
        if float(satisfied_areas[index]) != float(expected_row["satisfied_area"]):
            raise RuntimeError(f"patch satisfied area mismatch: {patch_id}")
        if float(total_areas[index]) != float(expected_row["total_area"]):
            raise RuntimeError(f"patch total area mismatch: {patch_id}")

        patch = patches[patch_id]
        valid = in_roi_valid_mask(
            patch, transform_meta["z_range"][0], transform_meta["z_range"][1]
        )
        satisfied = satisfied_masks[index].numpy().astype(bool)
        boundary = boundary_mask(valid)
        centers = centers_zyx(patch)
        flat_centers = centers.reshape(-1, 3)
        train_distance_vox = train_tree.query(flat_centers, workers=-1)[0].reshape(
            valid.shape
        )
        guard = np.zeros(valid.shape, dtype=bool)
        for point in guard_points:
            distance_vox = np.linalg.norm(centers - point[None, None, :], axis=-1)
            guard |= (
                distance_vox * float(dataset_receipt["voxel_size_um"]) / 1000.0
                <= GUARD_EXCLUSION_RADIUS_MM
            )
        guard &= valid
        safe_satisfied = satisfied & ~guard
        border_distance_mm = (
            ndimage.distance_transform_edt(valid)
            * float((1.0 / patch.scale).mean())
            * float(dataset_receipt["voxel_size_um"])
            / 1000.0
        )
        pitch_um_axes = (
            1.0 / patch.scale.cpu().numpy()
        ) * float(dataset_receipt["voxel_size_um"])
        if not np.isclose(pitch_um_axes[0], pitch_um_axes[1], rtol=0, atol=1e-5):
            raise RuntimeError(f"non-square tifxyz pitch for {patch_id}")
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

        npz_path = output / f"{patch_id}-quality.npz"
        np.savez_compressed(
            npz_path,
            valid=valid,
            satisfied=satisfied,
            guard=guard,
            boundary=boundary,
            safe_satisfied=safe_satisfied,
            component_labels=labels,
            target_winding_index=patch_winding,
            nearest_train_distance_mm=(
                train_distance_vox * float(dataset_receipt["voxel_size_um"]) / 1000.0
            ).astype(np.float32),
            border_distance_mm=border_distance_mm.astype(np.float32),
        )
        png_path = output / f"{patch_id}-quality.png"
        render_patch_png(
            png_path,
            valid=valid,
            satisfied=satisfied,
            guard=guard,
            boundary=boundary,
        )
        largest = components[0]["area_cm2"] if components else 0.0
        receipt = {
            "patch_id": patch_id,
            "shape_quads": list(valid.shape),
            "pitch_um": pitch_um,
            "cell_area_cm2": cell_area_cm2,
            "valid_quad_count": int(valid.sum()),
            "satisfied_quad_count": int(satisfied.sum()),
            "guard_quad_count": int(guard.sum()),
            "satisfied_area_scan_vox2": float(satisfied_areas[index]),
            "total_area_scan_vox2": float(total_areas[index]),
            "satisfied_fraction": float(satisfied_areas[index] / total_areas[index]),
            "official_patch_satisfied": bool(satisfied_patches[index]),
            "official_boundary_satisfied": bool(boundary_satisfied[index]),
            "safe_component_count": len(components),
            "largest_safe_component_cm2": largest,
            "green_4cm2_window_count": green_count,
            "best_4cm2_window": windows[0] if windows else None,
            "npz_path": npz_path.name,
            "npz_sha256": sha256_file(npz_path),
            "png_path": png_path.name,
            "png_sha256": sha256_file(png_path),
        }
        patch_receipts.append(receipt)
        components_payload[patch_id] = components
        windows_payload[patch_id] = windows
        overview_inputs.append((patch_id, png_path, receipt))

    best_windows = [
        (patch["patch_id"], patch["best_4cm2_window"])
        for patch in patch_receipts
        if patch["best_4cm2_window"] is not None
    ]
    best_windows.sort(
        key=lambda value: (
            -float(value[1]["score"]),
            value[0],
            value[1]["top_left_quad_ij"],
        )
    )
    green_count = sum(patch["green_4cm2_window_count"] for patch in patch_receipts)
    classification, eligible, reason = classify_roi(
        green_count=green_count,
        best_window=best_windows[0][1] if best_windows else None,
    )
    write_json(
        output / "QUALITY_COMPONENTS.json",
        {
            "kind": "campaign_x_phase4_target_quality_components_v1",
            "sample_id": args.sample_id,
            "patches": components_payload,
        },
    )
    write_json(
        output / "ROI_4CM2_CANDIDATES.json",
        {
            "kind": "campaign_x_phase4_target_exact_4cm2_windows_v1",
            "sample_id": args.sample_id,
            "area_cm2": TARGET_WINDOW_AREA_CM2,
            "square_side_mm": 20.0,
            "classification": {
                "green_satisfaction_fraction_minimum": GREEN_SATISFACTION_FRACTION,
                "high_amber_satisfaction_fraction_minimum": 0.80,
                "valid_fraction_required": 1.0,
                "guard_fraction_required": 0.0,
            },
            "patches": windows_payload,
        },
    )
    overview_path = output / "quality_overview.png"
    make_overview(overview_path, overview_inputs)
    aggregate = actual_satisfied / actual_total
    receipt = {
        "kind": (
            "campaign_x_phase4_weak_holdout_exploratory_quality_map_receipt_v1"
            if weak_holdout_mode
            else "campaign_x_phase4_experimental_rescue_quality_map_receipt_v1"
            if experimental_rescue_mode
            else "campaign_x_phase4_baseline_rescue_quality_map_receipt_v1"
            if rescue_mode
            else "campaign_x_phase4_target_quality_map_receipt_v1"
        ),
        "generated_at_utc": utc_now(),
        "status": (
            "PASSED_LOCAL_FUNCTIONAL_DIAGNOSTIC"
            if eligible and rescue_mode
            else "PASSED"
            if eligible
            else "NO_QUALIFIED_ROI"
        ),
        "gate": (
            "P4_18_WEAK_HOLDOUT_LOCAL_FUNCTIONAL_SCREEN"
            if weak_holdout_mode
            else "P4_15_FAILED_AB_EXPERIMENTAL_LOCAL_FUNCTIONAL_RESCUE"
            if experimental_rescue_mode
            else "P4_10_BASELINE_LOCAL_FUNCTIONAL_RESCUE"
            if rescue_mode
            else "P4_6_GEOMETRY_ONLY_ROI_SELECTION"
        ),
        "sample_id": args.sample_id,
        "scope": (
            "GEOMETRY_ONLY_NO_INK_PRIVATE_LOCAL_FUNCTIONAL_DIAGNOSTIC"
            if rescue_mode
            else "GEOMETRY_ONLY_NO_INK"
        ),
        "candidate_arm": args.candidate_arm,
        "evaluation_qualification": evaluation_qualification,
        "classification": classification,
        "eligible_for_p4_7": eligible,
        "reason": reason,
        "dataset_receipt": str(dataset_receipt_path),
        "dataset_receipt_sha256": sha256_file(dataset_receipt_path),
        "ab_evaluation_sha256": sha256_file(evaluation_path),
        "checkpoint": transform_meta,
        "official_metrics_config": metrics_config,
        "aggregate": {
            "satisfied_area_scan_vox2": actual_satisfied,
            "total_area_scan_vox2": actual_total,
            "satisfied_fraction": aggregate,
            "expected_satisfied_fraction": expected_satisfied / expected_total,
            "matches_frozen_result": True,
        },
        "guard": {
            "source": str(guard_source_path),
            "source_sha256": sha256_file(guard_source_path),
            "criterion": guard_criterion,
            "failed_relation_count": guard_failed_count,
            "endpoint_count": int(len(guard_points)),
            "exclusion_radius_mm": GUARD_EXCLUSION_RADIUS_MM,
            "conservative_application": "each failed endpoint tested against every patch in the shared coordinate frame",
        },
        "exact_window": {
            "area_cm2": TARGET_WINDOW_AREA_CM2,
            "square_side_mm": 20.0,
            "green_satisfaction_fraction_minimum": GREEN_SATISFACTION_FRACTION,
            "high_amber_satisfaction_fraction_minimum": 0.80,
            "green_window_count": green_count,
            "best": (
                {"patch_id": best_windows[0][0], **best_windows[0][1]}
                if best_windows
                else None
            ),
        },
        "patches": patch_receipts,
        "artifacts": {
            "components": "QUALITY_COMPONENTS.json",
            "windows": "ROI_4CM2_CANDIDATES.json",
            "overview": overview_path.name,
            "overview_sha256": sha256_file(overview_path),
            "viewer": "quality_viewer.html",
        },
        "privacy": "PRIVATE",
        "ink_used": False,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
    }
    write_json(output / "QUALITY_MAP_RECEIPT.json", receipt)
    make_viewer(output / "quality_viewer.html", receipt)
    print(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "classification": classification,
                "eligible_for_p4_7": eligible,
                "aggregate_satisfied_fraction": aggregate,
                "green_4cm2_window_count": green_count,
                "best_4cm2_window": receipt["exact_window"]["best"],
                "largest_components_cm2": {
                    patch["patch_id"]: patch["largest_safe_component_cm2"]
                    for patch in patch_receipts
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
