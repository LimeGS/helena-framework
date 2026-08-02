"""Frozen voxel metadata resolution for reusable high-recall runs."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "framework/stages/04-validation/scripts/build_high_recall_ct_application.py"
)
SPEC = spec_from_file_location("high_recall_ct_application", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_catalog(root: Path, entries: list[dict[str, object]]) -> None:
    path = root / "workspace" / "catalog" / "eligible_volumes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def test_target_voxel_um_uses_refactored_eligible_catalogue(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        [{"sample_id": "PHerc191", "voxel_size_um": 9.362}],
    )

    assert MODULE.target_voxel_um(tmp_path, "PHerc191") == pytest.approx(9.362)


def test_target_voxel_um_rejects_duplicate_sample_rows(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        [
            {"sample_id": "PHerc191", "voxel_size_um": 9.362},
            {"sample_id": "PHerc191", "voxel_size_um": 9.362},
        ],
    )

    with pytest.raises(RuntimeError, match="duplicate rows"):
        MODULE.target_voxel_um(tmp_path, "PHerc191")


def test_build_separates_artifact_root_from_frozen_metadata_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    metadata_root = tmp_path / "metadata"
    artifact_root.mkdir()
    write_catalog(
        metadata_root,
        [{"sample_id": "PHerc268", "voxel_size_um": 9.362}],
    )
    gate = metadata_root / "profiles" / "gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text("{}\n", encoding="utf-8")
    screening = artifact_root / "window" / "screening"
    screening.mkdir(parents=True)
    tiffs = screening.parent / "tiffs"
    tiffs.mkdir()
    for index in range(65):
        (tiffs / f"{index}.tif").write_bytes(b"tiff")
    candidate = {
        "candidate_id": "candidate-1",
        "window_id": "surface-1",
        "bbox_y0_x0_y1_x1": [1, 1, 4, 4],
        "routing_score": 0.5,
        "quota_reasons": ["TOP_GLOBAL"],
        "component_rank_in_window": 1,
    }
    router = artifact_root / "router.json"
    router.write_text(
        json.dumps(
            {
                "kind": MODULE.ROUTER_KIND,
                "status": MODULE.ROUTER_STATUS,
                "window_count": 1,
                "ct_review_queue_count": 1,
                "windows": [
                    {
                        "window_id": "surface-1",
                        "scroll_id": "PHerc268",
                        "shape_y_x": [8, 8],
                        "screening_dir": str(screening),
                        "components": [candidate],
                    }
                ],
                "ct_review_queue": [candidate],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = MODULE.build(
        root=artifact_root,
        metadata_root=metadata_root,
        router_path=router,
        output=artifact_root / "application",
        gate_freeze=gate,
        patch_radius_um=200.0,
        central_slice=32,
    )

    spec = json.loads(
        (artifact_root / "application/HIGH_RECALL_CT_FIBER_SPEC.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["adapted_candidate_count"] == 1
    assert spec["groups"][0]["voxel_um"] == pytest.approx(9.362)


def test_build_accepts_authoritative_qc_input_voxel_size_without_catalogue(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    screening = artifact_root / "screening"
    screening.mkdir(parents=True)
    tiffs = artifact_root / "tiffs"
    tiffs.mkdir()
    for index in range(65):
        (tiffs / f"{index:03d}.tif").write_bytes(b"fixture")
    router = artifact_root / "router.json"
    router.write_text(
        json.dumps(
            {
                "kind": MODULE.ROUTER_KIND,
                "status": MODULE.ROUTER_STATUS,
                "window_count": 1,
                "ct_review_queue_count": 1,
                "windows": [
                    {
                        "window_id": "w1",
                        "scroll_id": "PHerc826",
                        "screening_dir": str(screening),
                        "shape_y_x": [8, 8],
                        "components": [
                            {
                                "candidate_id": "c1",
                                "window_id": "w1",
                                "bbox_y0_x0_y1_x1": [1, 1, 3, 3],
                                "routing_score": 1.0,
                                "quota_reasons": ["fixture"],
                                "component_rank_in_window": 1,
                            }
                        ],
                    }
                ],
                "ct_review_queue": [
                    {
                        "candidate_id": "c1",
                        "window_id": "w1",
                        "bbox_y0_x0_y1_x1": [1, 1, 3, 3],
                        "routing_score": 1.0,
                        "quota_reasons": ["fixture"],
                        "component_rank_in_window": 1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_root = tmp_path / "metadata-absent"
    metadata_root.mkdir()
    gate = metadata_root / "gate.json"
    gate.write_text("{}\n", encoding="utf-8")
    MODULE.build(
        root=artifact_root,
        metadata_root=metadata_root,
        router_path=router,
        output=artifact_root / "application",
        gate_freeze=gate,
        patch_radius_um=200.0,
        central_slice=32,
        voxel_um=9.362,
    )
    spec = json.loads(
        (artifact_root / "application" / "HIGH_RECALL_CT_FIBER_SPEC.json").read_text()
    )
    assert spec["groups"][0]["voxel_um"] == pytest.approx(9.362)
    assert spec["groups"][0]["voxel_um_source"] == "frozen_surface_qc_input"
    assert spec["groups"][0]["tiff_directory"] == "tiffs"
    assert spec["policy"]["gate_freeze"] == "gate.json"
