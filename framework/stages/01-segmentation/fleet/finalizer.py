from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from .artifact_store import open_artifact_store
from .common import (
    artifact_manifest,
    content_sha256,
    is_fixture_surface,
    stable_id,
    utc_now,
    write_json_atomic,
)
from .dedup import find_duplicate_in_surfaces
from .store import DEFAULT_GEOMETRY_QC_STATE, FleetStore


REQUIRED = ("x.tif", "y.tif", "z.tif", "meta.json")
RESUME_CHANNEL = "generations.tif"

GEOMETRY_GATE_RELATIVE = (
    "framework/stages/04-validation/scripts/helena_tifxyz_geometry_gate.py"
)


def _geometry_gate_path() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / GEOMETRY_GATE_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def load_geometry_gate() -> ModuleType | None:
    """Load the 04-validation TIFXYZ geometry gate without importing a stage."""

    path = _geometry_gate_path()
    if path is None:
        return None
    spec = spec_from_file_location("helena_tifxyz_geometry_gate", path)
    if spec is None or spec.loader is None:
        return None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def certify_surface_geometry(surface_dir: Path) -> dict[str, Any]:
    """Run the geometry gate before the surface can be queued for the model.

    This is the gate that never existed: until now `GROW_SUCCEEDED` reached the
    ink model with nothing between it and TimeSformer but a sha256 and a file
    count.  It is fail-closed in verdict and fail-soft in control flow: a gate
    that runs and cannot measure this surface returns GEOMETRY_UNMEASURED, which
    is not certification, rather than losing a segmentation attempt.

    A gate that is not on disk is a different thing and no longer soft.  A moved
    or missing file used to degrade every surface in the campaign to
    GEOMETRY_UNMEASURED, silently and one at a time, with the same receipt an
    unmeasurable surface gets -- so a broken deployment was indistinguishable
    from a hard corpus, and P2 could stop working without anyone finding out.
    """

    module = load_geometry_gate()
    if module is None:
        raise RuntimeError(
            f"the geometry gate is not on disk at {GEOMETRY_GATE_RELATIVE}; "
            "this is a broken deployment, not an unmeasurable surface, and "
            "recording it as GEOMETRY_UNMEASURED would hide it")
    try:
        receipt = module.certify(Path(surface_dir))
    except Exception as failure:  # noqa: BLE001 - unmeasured is a real verdict
        return {
            "schema": "campaignx.tifxyz_geometry_certification.v1",
            "geometry_qc_state": DEFAULT_GEOMETRY_QC_STATE,
            "reason": "GATE_UNAVAILABLE",
            "status": "FAIL",
            "measurement_complete": False,
            "error": f"{type(failure).__name__}: {failure}",
            "non_claims": ["an unmeasured surface is not a certified surface"],
        }
    return receipt


def inspect_tifxyz(surface_dir: Path, voxel_size_um: float, sample_limit: int = 256) -> dict[str, Any]:
    import numpy as np
    import tifffile

    arrays = [np.asarray(tifffile.imread(surface_dir / f"{axis}.tif"), dtype=np.float64) for axis in "xyz"]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError(f"TIFXYZ shapes differ: {[array.shape for array in arrays]}")
    if arrays[0].ndim != 2 or min(arrays[0].shape) < 2:
        raise ValueError(f"TIFXYZ must be matching two-dimensional grids, got {arrays[0].shape}")
    # VC3D/TIFXYZ uses -1 as an invalid-coordinate sentinel. Coordinates also
    # cannot be negative in the CT-L0 frame. Treating those finite sentinels as
    # geometry fabricates long triangles, inflates area/bounds and poisons the
    # spatial duplicate index.
    valid = np.logical_and.reduce([
        np.isfinite(array) & (array >= 0.0) for array in arrays
    ])
    if int(valid.sum()) < 4:
        raise ValueError("TIFXYZ has fewer than four finite coordinates")
    points = np.stack([array[valid] for array in arrays], axis=1)
    low = points.min(axis=0)
    high = points.max(axis=0)
    sample_indices = np.linspace(0, len(points) - 1, min(sample_limit, len(points)), dtype=np.int64)
    sampled = points[sample_indices]
    # Sum each valid triangle independently. A cell on a mask boundary may
    # retain one real triangle even when its other triangle touches a sentinel.
    p = np.stack(arrays, axis=-1)
    v00, v10, v01, v11 = p[:-1, :-1], p[1:, :-1], p[:-1, 1:], p[1:, 1:]
    first_valid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]
    second_valid = valid[1:, 1:] & valid[1:, :-1] & valid[:-1, 1:]
    first = 0.5 * np.linalg.norm(np.cross(v10 - v00, v01 - v00), axis=-1)
    second = 0.5 * np.linalg.norm(np.cross(v11 - v10, v01 - v10), axis=-1)
    area_voxel2 = float(np.sum(first[first_valid]) + np.sum(second[second_valid]))
    area_cm2 = area_voxel2 * float(voxel_size_um) ** 2 / 100_000_000.0
    return {
        "shape": list(arrays[0].shape),
        "finite_coordinate_count": int(valid.sum()),
        "valid_triangle_count": int(first_valid.sum() + second_valid.sum()),
        "invalid_coordinate_policy": "FINITE_AND_NONNEGATIVE_CT_L0",
        "bbox_xyz": [low.tolist(), high.tolist()],
        "sample_points": sampled.tolist(),
        "area_cm2": area_cm2,
    }


def find_duplicate(store: FleetStore, source_snapshot_id: str, artifact_sha256: str, sample_points: list[list[float]], *, tolerance_voxels: float = 2.5, overlap_fraction: float = 0.92, maximum_median_voxels: float = 1.0) -> tuple[str | None, dict[str, Any]]:
    return find_duplicate_in_surfaces(
        store.surfaces_for_snapshot(source_snapshot_id),
        artifact_sha256,
        sample_points,
        tolerance_voxels=tolerance_voxels,
        overlap_fraction=overlap_fraction,
        maximum_median_voxels=maximum_median_voxels,
    )


def finalize_surface(
    store: FleetStore,
    task: dict[str, Any],
    locked_plan: dict[str, Any],
    surface_dir: Path,
    artifact_root: Path | str,
    attempt_dir: Path,
    qc_profile_id: str,
) -> dict[str, Any]:
    if not qc_profile_id or "@" not in qc_profile_id:
        raise ValueError("finalization requires a versioned semantic QC profile ID")
    inspection = inspect_tifxyz(surface_dir, float(task["source"]["voxel_size_um"]))
    geometry = certify_surface_geometry(surface_dir)
    write_json_atomic(attempt_dir / "GEOMETRY_CERTIFICATION.json", geometry)
    # Historical/imported fixtures may not carry the generation channel, so it
    # cannot be made retroactively mandatory here.  Every current VC3D executor
    # does require it, and publishing it when present is essential: downstream
    # extraction and --resume both consume generations.tif.
    manifest_files = (
        (*REQUIRED, RESUME_CHANNEL)
        if (surface_dir / RESUME_CHANNEL).is_file()
        else REQUIRED
    )
    files = artifact_manifest(surface_dir, manifest_files)
    artifact_sha = content_sha256(files)
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan),
        "files": files,
        "artifact_sha256": artifact_sha,
        **inspection,
        "ink_used": False,
    }
    write_json_atomic(attempt_dir / "ARTIFACT_SET.json", manifest)
    artifact_store = open_artifact_store(artifact_root)
    staged = artifact_store.stage(surface_dir, task["attempt_id"], manifest)
    artifact_set_id = store.add_artifact_set(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        manifest,
        staged["staging_uri"],
    )
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"], "FINALIZING")
    surface_id = stable_id("surface", {"source_snapshot_id": task["source_snapshot_id"], "artifact_sha256": artifact_sha})
    promotion = artifact_store.promote(
        staged, task["sample_id"], surface_id, manifest
    )
    fixture_only = is_fixture_surface(task) or is_fixture_surface(
        task.get("source", {})
    )
    surface = {
        "schema": "campaignx.segment_fleet_surface.v1",
        "surface_id": surface_id,
        "source_snapshot_id": task["source_snapshot_id"],
        "sample_id": task["sample_id"],
        "owner": "campaign-x",
        "artifact_sha256": artifact_sha,
        "artifact_uri": promotion["artifact_uri"],
        "bbox_xyz": inspection["bbox_xyz"],
        "sample_points": inspection["sample_points"],
        "area_cm2": inspection["area_cm2"],
        "state": "FINALIZING",
        "physical_qc_state": (
            "NOT_APPLICABLE_FIXTURE" if fixture_only else "UNVALIDATED"
        ),
        # Orthogonal to physical_qc_state: a surface can be CT_SUPPORTED and
        # GEOMETRY_REJECTED_BRIDGE at the same time.
        "geometry_qc_state": geometry.get(
            "geometry_qc_state", DEFAULT_GEOMETRY_QC_STATE
        ),
        "geometry_certification": {
            "geometry_qc_state": geometry.get("geometry_qc_state"),
            "reason": geometry.get("reason"),
            "hard_defects_observed": geometry.get("hard_defects_observed"),
            "resolution_limited": geometry.get("resolution_limited"),
            "receipt_path": str(attempt_dir / "GEOMETRY_CERTIFICATION.json"),
            "receipt_sha256": content_sha256(geometry),
        },
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan),
        "ink_used": False,
        "fixture_only": fixture_only,
    }
    finalization = store.finalize(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        surface,
        artifact_set_id,
        qc_profile_id,
    )
    receipt = {
        "schema": "campaignx.segment_fleet_finalization_receipt.v1",
        "status": finalization["status"],
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "surface": surface,
        "duplicate_of": finalization["duplicate_of"],
        "duplicate_diagnostics": finalization["duplicate_diagnostics"],
        "geometry_certification": geometry,
        "geometry_blocked_qc": finalization.get("geometry_blocked_qc", False),
        "artifact_set_id": artifact_set_id,
        "qc_profile_id": qc_profile_id,
        "artifact_stage": staged,
        "artifact_promotion": promotion,
        "generated_at_utc": utc_now(),
        "non_claim": (
            "FIXTURE_ONLY is excluded from scientific QC."
            if finalization["status"] == "FIXTURE_ONLY"
            else "QC_PENDING is not physical-sheet acceptance, ink, text, or First Letters."
        ),
    }
    write_json_atomic(attempt_dir / "FINALIZATION_RECEIPT.json", receipt)
    return receipt
