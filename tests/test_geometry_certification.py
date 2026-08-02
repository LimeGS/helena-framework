"""FIX-07: a real geometric gate between segmentation and the ink model.

Before this gate the only checks between ``GROW_SUCCEEDED`` and TimeSformer
were a sha256 and a file count, and ``qc_jobs`` had no failure state at all.
These tests hold the two halves of that fix: TIFXYZ meshes with real geometric
defects must be rejected by name, and a rejected surface must not be able to
reach the model.
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]
STAGE01 = ROOT / "framework/stages/01-segmentation"
for candidate in (ROOT, STAGE01):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fleet.store import (  # noqa: E402
    DEFAULT_GEOMETRY_QC_STATE,
    GEOMETRY_QC_STATES,
    GEOMETRY_REJECTED_STATES,
    QC_JOB_STATES,
    FleetStore,
    is_geometry_rejected,
)
from fleet.common import content_sha256  # noqa: E402

STEP = 20.0
GATE_PATH = (
    ROOT / "framework/stages/04-validation/scripts/helena_tifxyz_geometry_gate.py"
)
INTEGRITY_PATH = (
    ROOT / "framework/stages/04-validation/scripts/helena_audit_mesh_integrity.py"
)


def load(path: Path):
    spec = spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return load(GATE_PATH)


def write_tifxyz(directory: Path, points: np.ndarray) -> Path:
    """Persist an (H, W, 3) grid as VC3D TIFXYZ, -1 meaning invalid."""

    directory.mkdir(parents=True, exist_ok=True)
    for index, axis in enumerate("xyz"):
        tifffile.imwrite(
            directory / f"{axis}.tif", points[:, :, index].astype(np.float32)
        )
    return directory


def clean_patch(rows: int = 24, cols: int = 24) -> np.ndarray:
    """A gently curved single-lamina patch, arc-length uniform like VC3D."""

    row_index, col_index = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing="ij"
    )
    x = 1000.0 + col_index * STEP
    y = 2000.0 + row_index * STEP
    # Curvature well under the grid step keeps the patch arc-length uniform.
    z = 3000.0 + 0.0006 * ((col_index - cols / 2.0) * STEP) ** 2
    return np.stack([x, y, z], axis=-1)


def stitched_laminae() -> np.ndarray:
    """One grid that jumps from one lamina to another mid-traverse.

    The stitch is a step of ten grid units in a grid whose every other edge is
    one grid unit long: the segmentation left the sheet it was following.
    """

    points = clean_patch(rows=32)
    points[16:, :, 2] += 10.0 * STEP
    return points


def crossing_laminae() -> np.ndarray:
    """Two laminae in one grid that physically stab through each other."""

    rows, cols = 40, 24
    points = np.full((rows, cols, 3), -1.0)
    col_index = np.arange(cols)
    for row in range(16):
        points[row, :, 0] = 1000.0 + col_index * STEP
        points[row, :, 1] = 2000.0 + row * STEP
        points[row, :, 2] = 3000.0
    # rows 16-21 stay at the invalid sentinel so the two sheets are separated by
    # more than the grid band and no adjacent edge spans the gap.
    diagonal = STEP / np.sqrt(2.0)
    for offset, row in enumerate(range(22, rows)):
        points[row, :, 0] = 1000.0 + col_index * STEP
        points[row, :, 1] = 2000.0 + offset * diagonal
        points[row, :, 2] = 3000.0 - 120.0 + offset * diagonal
    return points


def doubled_surface() -> np.ndarray:
    """Two near-parallel sheets a fraction of a grid step apart: a false bridge."""

    rows, cols = 40, 24
    points = np.full((rows, cols, 3), -1.0)
    col_index = np.arange(cols)
    for row in range(16):
        points[row, :, 0] = 1000.0 + col_index * STEP
        points[row, :, 1] = 2000.0 + row * STEP
        points[row, :, 2] = 3000.0
    for offset, row in enumerate(range(22, rows)):
        points[row, :, 0] = 1000.0 + col_index * STEP
        points[row, :, 1] = 2000.0 + offset * STEP
        # Well inside the doubled-surface gap and far inside one lamina spacing.
        points[row, :, 2] = 3000.0 + 1.5
    return points


def fold_back() -> np.ndarray:
    """A hairpin: the sheet reverses inside a single grid step and climbs away.

    The return leg leaves the outgoing leg faster than the near-coincidence gap,
    so the only detector that can fire is the edge-adjacent fold-back.
    """

    rows, cols = 32, 24
    points = np.zeros((rows, cols, 3))
    col_index = np.arange(cols)
    points[:, :, 0] = 1000.0 + col_index * STEP
    for row in range(16):
        points[row, :, 1] = 2000.0 + row * STEP
        points[row, :, 2] = 3000.0
    crease_y = 2000.0 + 15 * STEP
    for offset, row in enumerate(range(16, rows)):
        points[row, :, 1] = crease_y - 19.0 - offset * 14.0
        points[row, :, 2] = 3000.0 + 6.0 + offset * 14.0
    return points


def curled_sheet(rows: int = 38, cols: int = 12, radius: float = 120.0) -> np.ndarray:
    """A single sheet curled almost all the way round, as a scroll winding is.

    Its two ends pass within a few voxels of each other while remaining one
    continuous lamina.  Nothing here is a defect, and a gate that calls it one
    would reject every surface that spans more than one turn.
    """

    points = np.zeros((rows, cols, 3))
    angle = np.arange(rows) * (STEP / radius)
    points[:, :, 0] = np.arange(cols) * STEP
    points[:, :, 1] = (1000.0 + radius * np.cos(angle))[:, None]
    points[:, :, 2] = (1000.0 + radius * np.sin(angle))[:, None]
    return points


def test_a_clean_tifxyz_patch_is_certified(gate, tmp_path: Path) -> None:
    receipt = gate.certify(write_tifxyz(tmp_path / "clean", clean_patch()))
    assert receipt["geometry_qc_state"] == "GEOMETRY_CERTIFIED"
    assert receipt["status"] == "PASS"
    assert receipt["hard_defects_observed"] == 0
    assert receipt["measurement_complete"] is True
    assert receipt["error"] is None
    # The receipt must publish what limits it, not only its verdict.
    assert receipt["resolution_limited"] is True
    assert receipt["grid"]["median_edge_voxels"] == pytest.approx(STEP, rel=0.05)
    assert set(receipt["inputs"]) == {"x.tif", "y.tif", "z.tif"}


def test_two_stitched_laminae_are_rejected_not_ct_supported(gate, tmp_path: Path) -> None:
    receipt = gate.certify(write_tifxyz(tmp_path / "stitched", stitched_laminae()))
    assert receipt["geometry_qc_state"] == "GEOMETRY_REJECTED_LAMINA_SWITCH"
    assert receipt["status"] == "FAIL"
    assert receipt["seam"]["lamina_step_discontinuity_edges"] > 0
    assert receipt["hard_defects_observed"] > 0
    # The geometry axis says nothing about ink or CT support, and vice versa.
    assert "CT_SUPPORTED" not in receipt["geometry_qc_state"]


def test_interpenetrating_laminae_are_a_lamina_switch(gate, tmp_path: Path) -> None:
    receipt = gate.certify(write_tifxyz(tmp_path / "crossing", crossing_laminae()))
    assert receipt["geometry_qc_state"] == "GEOMETRY_REJECTED_LAMINA_SWITCH"
    assert receipt["seam"]["interpenetration_pairs"] > 0
    assert receipt["exact_self_intersection"]["self_intersections_present"] is True


def test_a_doubled_surface_is_a_bridge(gate, tmp_path: Path) -> None:
    receipt = gate.certify(write_tifxyz(tmp_path / "doubled", doubled_surface()))
    assert receipt["geometry_qc_state"] == "GEOMETRY_REJECTED_BRIDGE"
    assert receipt["seam"]["near_coincident_overlap_pairs"] > 0
    assert receipt["seam"]["offending_triangles"] > 0


def test_a_fold_back_self_intersection_is_rejected(gate, tmp_path: Path) -> None:
    receipt = gate.certify(write_tifxyz(tmp_path / "fold", fold_back()))
    assert receipt["seam"]["fold_back_intersections"] > 0
    assert receipt["geometry_qc_state"] in GEOMETRY_REJECTED_STATES
    assert receipt["geometry_qc_state"] != "GEOMETRY_CERTIFIED"
    assert receipt["status"] == "FAIL"


def test_an_unreadable_surface_is_unmeasured_never_certified(gate, tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "x.tif").write_bytes(b"not a tiff")
    receipt = gate.certify(directory)
    assert receipt["geometry_qc_state"] == "GEOMETRY_UNMEASURED"
    assert receipt["status"] == "FAIL"
    assert receipt["error"]


def test_a_curled_winding_is_not_a_false_bridge(gate, tmp_path: Path) -> None:
    """The proximity of the next turn of a scroll is geometry, not a defect."""

    receipt = gate.certify(write_tifxyz(tmp_path / "curled", curled_sheet()))
    assert receipt["grid"]["grid_far_candidate_pair_count"] > 0
    separation = receipt["grid"]["minimum_far_separation_voxels"]
    assert 0.0 < separation < STEP
    assert receipt["seam"]["near_coincident_overlap_pairs"] == 0
    assert receipt["geometry_qc_state"] == "GEOMETRY_CERTIFIED"


def test_an_incomplete_measurement_cannot_be_certified(gate, tmp_path: Path) -> None:
    """Fail closed: a detector that did not run is not a detector that passed."""

    directory = write_tifxyz(tmp_path / "budget", curled_sheet())
    receipt = gate.certify(directory, {"maximum_candidate_pairs": 0})
    assert receipt["geometry_qc_state"] == "GEOMETRY_UNMEASURED"
    assert receipt["measurement_complete"] is False
    assert receipt["grid"]["non_local_stage"] == "BUDGET_EXCEEDED"
    assert receipt["hard_defects_observed"] is None


def test_a_local_defect_survives_an_incomplete_non_local_stage(gate, tmp_path: Path) -> None:
    """A mesh too large for the pair search still gets the verdict it earned."""

    directory = write_tifxyz(tmp_path / "budget-stitched", stitched_laminae())
    receipt = gate.certify(directory, {"maximum_candidate_pairs": 0})
    assert receipt["geometry_qc_state"] == "GEOMETRY_REJECTED_LAMINA_SWITCH"


def test_the_hard_defect_arithmetic_is_single_sourced(gate, tmp_path: Path) -> None:
    integrity = load(INTEGRITY_PATH)
    assert integrity.hard_defect_count({"a": 0, "b": 0}, False) == 0
    assert integrity.hard_defect_count({"a": 0, "b": 0}, True) == 1
    assert integrity.hard_defect_count({"a": 2, "b": 3}, False) == 5
    # The TIFXYZ port loads the frozen ScrollFiesta gate rather than forking it.
    assert callable(gate._mesh_integrity_module().hard_defect_count)
    # Every metric the frozen gate parses out of seam_audit must also be
    # reported by the TIFXYZ port, or the two routes are not comparable.
    metrics = gate.measure(write_tifxyz(tmp_path / "named", clean_patch()))
    assert set(integrity.SEAM_METRIC_NAMES) <= set(metrics["seam"])
    assert "lamina_step_discontinuity_edges" in metrics["seam"]


def test_the_geometry_axis_is_orthogonal_to_physical_qc_state() -> None:
    assert "GEOMETRY_CERTIFIED" in GEOMETRY_QC_STATES
    assert set(GEOMETRY_REJECTED_STATES) == {
        "GEOMETRY_REJECTED_BRIDGE",
        "GEOMETRY_REJECTED_LAMINA_SWITCH",
        "GEOMETRY_REJECTED_DISTORTION",
        "GEOMETRY_REJECTED_COVERAGE",
    }
    assert DEFAULT_GEOMETRY_QC_STATE == "GEOMETRY_UNMEASURED"
    assert is_geometry_rejected("GEOMETRY_REJECTED_BRIDGE") is True
    assert is_geometry_rejected("GEOMETRY_CERTIFIED") is False
    assert is_geometry_rejected(None) is False
    assert "FAILED" in QC_JOB_STATES


def _snapshot(store: FleetStore) -> str:
    return store.register_snapshot({
        "sample_id": "PHercTEST",
        "ct_uri": "file:///ct",
        "m7_uri": "file:///m7",
        "shape_xyz": [10, 10, 10],
        "voxel_size_um": 7.91,
        "coordinate_frame": "ct_l0_xyz",
    })


def _surface(source_id: str, name: str, geometry_state: str) -> dict:
    return {
        "surface_id": f"campaign-x:PHercTEST:{name}",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "artifact_sha256": name.encode().hex().ljust(64, "0")[:64],
        "artifact_uri": f"file:///artifacts/{name}",
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "sample_points": [[0.0, 0.0, 0.0]],
        "area_cm2": 1.0,
        "geometry_qc_state": geometry_state,
    }


def _prepare_finalization(
    store: FleetStore,
    task: dict,
    surface: dict,
) -> tuple[dict, str]:
    """Build the same cross-bound receipts as the real finalizer."""

    locked_plan = {"schema": "test.locked_plan.v1", "task_id": task["task_id"]}
    locked_plan_sha256 = content_sha256(locked_plan)
    bound_surface = {
        **surface,
        "schema": "campaignx.segment_fleet_surface.v1",
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "locked_plan_sha256": locked_plan_sha256,
        "ink_used": False,
    }
    store.transition(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        "RUNNING",
        locked_plan=locked_plan,
    )
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "locked_plan_sha256": locked_plan_sha256,
        "files": {},
        "artifact_sha256": bound_surface["artifact_sha256"],
        "bbox_xyz": bound_surface["bbox_xyz"],
        "sample_points": bound_surface["sample_points"],
        "area_cm2": bound_surface["area_cm2"],
        "ink_used": False,
    }
    artifact_set_id = store.add_artifact_set(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        manifest,
        "file:///staging",
    )
    store.transition(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        "FINALIZING",
    )
    return bound_surface, artifact_set_id


def _task(store: FleetStore, source_id: str, cell: str) -> dict:
    store.create_tasks([{
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "cell_id": cell,
        "grid_version": "g1",
        "policy_version": "p1",
        "bounds_xyz": [[0, 0, 0], [1, 1, 1]],
        "center_xyz": [0.5, 0.5, 0.5],
        "priority": 1.0,
        "parameter_envelope": {},
        "catalog_snapshot_sha256": "c" * 64,
    }])
    claim = store.claim(worker_id="w1", lease_seconds=600)
    assert claim is not None
    return claim


def test_a_geometry_rejection_keeps_a_surface_away_from_the_model(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite3")
    store.initialize()
    source_id = _snapshot(store)
    task = _task(store, source_id, "cell-0001")
    surface, artifact_set_id = _prepare_finalization(
        store,
        task,
        _surface(source_id, "bridged", "GEOMETRY_REJECTED_BRIDGE"),
    )
    result = store.finalize(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        surface,
        artifact_set_id,
        "geometry-screen-v1@1",
    )
    assert result["status"] == "QC_PENDING"
    assert result["geometry_qc_state"] == "GEOMETRY_REJECTED_BRIDGE"
    assert result["geometry_blocked_qc"] is True
    # The queue must refuse to hand a geometrically rejected surface to QC.
    assert store.claim_qc("qc-worker", 600) is None
    with store.connect() as connection:
        row = connection.execute("SELECT state,result_json FROM qc_jobs").fetchone()
        assert row["state"] == "FAILED"
        assert "GEOMETRY_REJECTED_BRIDGE" in row["result_json"]
        surface = connection.execute(
            "SELECT physical_qc_state,geometry_qc_state FROM surfaces"
        ).fetchone()
        # Orthogonality: the ink/CT axis is untouched by the geometry verdict.
        assert surface["physical_qc_state"] == "UNVALIDATED"
        assert surface["geometry_qc_state"] == "GEOMETRY_REJECTED_BRIDGE"


def test_a_certified_surface_still_reaches_qc(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite3")
    store.initialize()
    source_id = _snapshot(store)
    task = _task(store, source_id, "cell-0002")
    surface, artifact_set_id = _prepare_finalization(
        store,
        task,
        _surface(source_id, "clean", "GEOMETRY_CERTIFIED"),
    )
    store.finalize(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        surface,
        artifact_set_id,
        "geometry-screen-v1@1",
    )
    claim = store.claim_qc("qc-worker", 600)
    assert claim is not None
    assert claim["surface"]["geometry_qc_state"] == "GEOMETRY_CERTIFIED"


def test_finalization_rejects_foreign_artifacts_and_replays_exactly(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite3")
    store.initialize()
    source_id = _snapshot(store)
    first = _task(store, source_id, "cell-owner-a")
    first_surface, first_artifact = _prepare_finalization(
        store,
        first,
        _surface(source_id, "owner-a", "GEOMETRY_CERTIFIED"),
    )
    second = _task(store, source_id, "cell-owner-b")
    _, second_artifact = _prepare_finalization(
        store,
        second,
        _surface(source_id, "owner-b", "GEOMETRY_CERTIFIED"),
    )

    with pytest.raises(RuntimeError, match="different attempt"):
        store.finalize(
            first["task_id"],
            first["attempt_id"],
            first["lease_token"],
            first_surface,
            second_artifact,
            "geometry-screen-v1@1",
        )
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0
        assert connection.execute(
            "SELECT state FROM artifact_sets WHERE artifact_set_id=?",
            (second_artifact,),
        ).fetchone()["state"] == "UPLOADED"

    initial = store.finalize(
        first["task_id"],
        first["attempt_id"],
        first["lease_token"],
        first_surface,
        first_artifact,
        "geometry-screen-v1@1",
    )
    replay = store.finalize(
        first["task_id"],
        first["attempt_id"],
        first["lease_token"],
        first_surface,
        first_artifact,
        "geometry-screen-v1@1",
    )
    assert replay == initial
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 1


def test_ct_supported_and_geometry_rejected_can_coexist(tmp_path: Path) -> None:
    """The combination the campaign had no way to express."""

    store = FleetStore(tmp_path / "fleet.sqlite3")
    store.initialize()
    source_id = _snapshot(store)
    surface_id = store.import_surface({
        "surface_id": "campaign-x:PHercTEST:imported",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "physical_qc_state": "CT_SUPPORTED",
        "state": "QC_SCREENED",
    })
    recorded = store.record_geometry_certification(
        surface_id,
        "GEOMETRY_REJECTED_LAMINA_SWITCH",
        {"reason": "NON_PARALLEL_STAB_OR_STEP_DISCONTINUITY"},
    )
    assert recorded["physical_qc_state"] == "CT_SUPPORTED"
    assert recorded["geometry_qc_state"] == "GEOMETRY_REJECTED_LAMINA_SWITCH"
    with store.connect() as connection:
        row = connection.execute(
            "SELECT physical_qc_state,geometry_qc_state FROM surfaces WHERE surface_id=?",
            (surface_id,),
        ).fetchone()
    assert row["physical_qc_state"] == "CT_SUPPORTED"
    assert row["geometry_qc_state"] == "GEOMETRY_REJECTED_LAMINA_SWITCH"
    with pytest.raises(ValueError):
        store.record_geometry_certification(surface_id, "GEOMETRY_LOOKS_FINE")


def test_finalizer_certifies_geometry_and_records_an_unmeasured_default() -> None:
    from fleet import finalizer

    module = finalizer.load_geometry_gate()
    assert module is not None
    assert hasattr(module, "certify")
    verdict = finalizer.certify_surface_geometry(Path("/nonexistent/surface"))
    assert verdict["geometry_qc_state"] == DEFAULT_GEOMETRY_QC_STATE


def _finalize(tmp_path: Path, name: str, points: np.ndarray) -> dict:
    """Drive the real finalization path the segmentation worker uses."""

    import json

    from fleet.finalizer import finalize_surface

    store = FleetStore(tmp_path / f"{name}.sqlite3")
    store.initialize()
    source_id = _snapshot(store)
    task = _task(store, source_id, f"cell-{name}")
    locked_plan = {"plan": name}
    store.transition(
        task["task_id"],
        task["attempt_id"],
        task["lease_token"],
        "RUNNING",
        locked_plan=locked_plan,
    )
    surface_dir = write_tifxyz(tmp_path / f"{name}-surface", points)
    (surface_dir / "meta.json").write_text(
        json.dumps({"uuid": name, "format": "tifxyz"}), encoding="utf-8"
    )
    attempt_dir = tmp_path / f"{name}-attempt"
    attempt_dir.mkdir()
    receipt = finalize_surface(
        store,
        task,
        locked_plan,
        surface_dir,
        tmp_path / f"{name}-artifacts",
        attempt_dir,
        "geometry-screen-v1@1",
    )
    return {"store": store, "receipt": receipt, "attempt_dir": attempt_dir}


def test_finalization_certifies_a_clean_surface_and_still_enqueues_qc(tmp_path: Path) -> None:
    outcome = _finalize(tmp_path, "clean", clean_patch())
    receipt = outcome["receipt"]

    assert receipt["status"] == "QC_PENDING"
    assert receipt["surface"]["geometry_qc_state"] == "GEOMETRY_CERTIFIED"
    assert receipt["geometry_blocked_qc"] is False
    assert (outcome["attempt_dir"] / "GEOMETRY_CERTIFICATION.json").is_file()
    assert outcome["store"].claim_qc("qc-worker", 600) is not None


def test_finalization_stops_a_stitched_surface_before_the_model(tmp_path: Path) -> None:
    """The gate sits between finalize_surface and the QC enqueue, as specified."""

    outcome = _finalize(tmp_path, "stitched", stitched_laminae())
    receipt = outcome["receipt"]

    assert receipt["surface"]["geometry_qc_state"] == "GEOMETRY_REJECTED_LAMINA_SWITCH"
    assert receipt["geometry_blocked_qc"] is True
    assert receipt["geometry_certification"]["seam"]["lamina_step_discontinuity_edges"] > 0
    # Nothing can hand this surface to TimeSformer.
    assert outcome["store"].claim_qc("qc-worker", 600) is None
    with outcome["store"].connect() as connection:
        assert connection.execute("SELECT state FROM qc_jobs").fetchone()["state"] == "FAILED"
        event = connection.execute(
            "SELECT event_type FROM events WHERE event_type='GEOMETRY_REJECTED_BEFORE_MODEL'"
        ).fetchone()
    assert event is not None
