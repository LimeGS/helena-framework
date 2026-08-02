"""Fail-closed contract around the upstream Volume Cartographer merge lane."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/05-reconstruction/scripts/run_vc3d_tifxyz_merge.py"


def lane_module():
    spec = importlib.util.spec_from_file_location("vc3d_merge_lane", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent(surface_id: str, **changed):
    row = {
        "surface_id": surface_id,
        "sample_id": "PHerc826",
        "source_snapshot_id": "snapshot-1",
        "artifact_uri": f"s3://surfaces/{surface_id}",
        "artifact_sha256": surface_id * 8,
        "ct_uri": "https://example.test/scroll.zarr",
        "ct_sha256": "c" * 64,
        "coordinate_frame": {"axes": "xyz", "level": 0},
        "voxel_size_um": 7.91,
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
    }
    row.update(changed)
    return row


@pytest.mark.parametrize("field,value", [
    ("sample_id", "PHerc172"),
    ("source_snapshot_id", "snapshot-2"),
    ("ct_uri", "https://example.test/other.zarr"),
    ("ct_sha256", "d" * 64),
    ("coordinate_frame", {"axes": "zyx", "level": 0}),
    ("voxel_size_um", 8.0),
    ("geometry_qc_state", "GEOMETRY_UNMEASURED"),
])
def test_parent_compatibility_refuses_every_scientific_mismatch(field, value):
    module = lane_module()
    with pytest.raises(module.MergeRefused, match=field):
        module.validate_parent_compatibility([
            parent("a"), parent("b", **{field: value})])


def test_upstream_command_is_frozen_and_names_explicit_reference_and_seed(tmp_path):
    module = lane_module()
    command = module.upstream_command(
        binary=Path("/opt/vc_merge_tifxyz"),
        obj2tifxyz=Path("/opt/vc_obj2tifxyz_legacy"),
        merge_json=tmp_path / "merge.json", paths_dir=tmp_path / "paths",
        reference="s000", ransac_seed=1729, anchor_cap=2000, strip_cols=0)
    assert command == [
        "/opt/vc_merge_tifxyz", "--merge", str(tmp_path / "merge.json"),
        "--paths-dir", str(tmp_path / "paths"),
        "--obj2tifxyz", "/opt/vc_obj2tifxyz_legacy",
        "--ref", "s000", "--ransac-seed", "1729",
        "--anchor-cap", "2000", "--strip-cols", "0",
    ]


def test_profile_freezes_the_upstream_wide_overlap_memory_controls():
    profile = json.loads((
        ROOT / "framework/profiles/05-reconstruction/"
        "vc3d-tifxyz-merge-1.0.0.json"
    ).read_text())
    assert profile["parameters"]["anchor_cap"] == 2000
    assert profile["parameters"]["strip_cols"] == 0


def test_queue_builds_only_the_registered_wrapper_argv():
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    parameters = validate_parameters({
        "lane": "vc3d-tifxyz-merge",
        "artifact_ids": ["surface-a", "surface-b"],
        "rows": [["surface-a", "surface-b"]],
        "reference_artifact_id": "surface-a", "ransac_seed": 1729,
        "anchor_cap": 2000, "strip_cols": 0,
        "artifact_store": "s3://helena/reconstruction-v1",
    }, "P8")
    argv = command_for({
        "phase": "P8", "profile_id": "vc3d-tifxyz-merge@1.0.0",
        "parameters": parameters,
    }, runner="ignored", output_dir="/runs/p8-job")
    assert argv[:2] == [
        "python3",
        "framework/stages/05-reconstruction/scripts/run_vc3d_tifxyz_merge.py",
    ]
    assert argv[-2:] == ["--artifact-store", "s3://helena/reconstruction-v1"]
    assert argv.count("--output") == 1
    assert not any(value in argv for value in ("sh", "bash", "-c", "command"))


def test_lineage_is_append_only_not_rewritten_on_conflict():
    source = (ROOT / "framework/stages/03-ink/fleet/job_store.py").read_text()
    registration = source[source.index("def register_merged_surface("):
                          source.index("\n    def events(")]
    assert "ON CONFLICT(child_surface_id,parent_surface_id) DO NOTHING" in registration
    assert "DO UPDATE" not in registration


def test_merged_child_is_transactionally_enqueued_for_the_frozen_physical_qc():
    profile = json.loads((
        ROOT / "framework/profiles/05-reconstruction/"
        "vc3d-tifxyz-merge-1.0.0.json"
    ).read_text())
    assert profile["physical_qc"] == {
        "profile_id": "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0",
        "initial_state": "PENDING",
    }

    store_source = (ROOT / "framework/stages/03-ink/fleet/job_store.py").read_text()
    registration = store_source[store_source.index("def register_merged_surface("):
                                store_source.index("\n    def events(")]
    assert "INSERT INTO segment_qc_jobs" in registration
    assert "qc_profile_id" in registration
    assert '"qc_job_id": qc_job_id' in registration

    wrapper = SCRIPT.read_text()
    assert 'qc_profile_id=profile["physical_qc"]["profile_id"]' in wrapper


def test_p3_can_target_only_the_merged_child_surface():
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    parameters = validate_parameters({
        "surface_id": "merged-child-1",
        "artifact_store": "s3://helena/flatten-v1",
    }, "P3")
    argv = command_for({
        "phase": "P3", "profile_id": "flatten-abf-v1@1.0.0",
        "parameters": parameters,
    }, runner="fleet.py", output_dir="/runs/p3-job")
    assert argv[argv.index("--surface-id") + 1] == "merged-child-1"

    cli = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    postgres = (
        ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py"
    ).read_text()
    assert "surface_id=args.surface_id" in cli
    assert "AND s.surface_id=%s" in postgres


def test_seam_gate_requires_real_overlap_and_enough_inliers_on_every_edge():
    module = lane_module()
    policy = {
        "minimum_anchor_count_per_edge": 3,
        "minimum_ransac_inliers_per_edge": 3,
        "minimum_inlier_fraction_per_edge": 0.25,
        "maximum_ransac_sigma_in_voxels": 10.0,
        "require_real_overlap_on_both_parents": True,
        "require_all_declared_edges": True,
    }
    passing = {"edges": [{
        "a": "s000", "b": "s001", "anchor_count": 9,
        "ransac_inliers": 5, "ransac_total": 9, "ransac_sigma_in": 2.0,
        "real_overlap_A": 20, "real_overlap_B": 22,
    }]}
    assert module.evaluate_seam_qc(passing, {frozenset(("s000", "s001"))}, policy)[
        "status"] == "PASS"

    passing["edges"][0]["real_overlap_B"] = 0
    with pytest.raises(module.MergeRefused, match="real overlap"):
        module.evaluate_seam_qc(
            passing, {frozenset(("s000", "s001"))}, policy)


def test_wrapper_revalidates_connected_layout_and_preserves_empty_cells():
    module = lane_module()
    assert module.validate_layout(
        [["a", None], ["b", None]], ["a", "b"]
    ) == [["a", None], ["b", None]]
    assert module.declared_edges([["a", None], ["b", None]]) == {
        frozenset(("a", "b"))
    }
    with pytest.raises(module.MergeRefused, match="disconnected"):
        module.validate_layout([["a", None, "b"]], ["a", "b"])


def test_worker_resolves_the_registered_wrapper():
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"phase": "P8", "profile_id": "vc3d-tifxyz-merge@1.0.0",
           "parameters": {"lane": "vc3d-tifxyz-merge"}}
    assert ink_worker.runner_for(job) == SCRIPT


def test_worker_passes_job_identity_and_reads_the_merge_receipt():
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"job_id": "job-merge-1", "phase": "P8",
           "profile_id": "vc3d-tifxyz-merge@1.0.0",
           "parameters": {"lane": "vc3d-tifxyz-merge"}}
    assert ink_worker.runner_environment(job)["HELENA_JOB_ID"] == "job-merge-1"
    assert "MERGE_RECEIPT.json" in ink_worker.receipt_names(job)


def test_worker_exposes_the_complete_merge_receipt_in_the_job_result():
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"phase": "P8", "profile_id": "vc3d-tifxyz-merge@1.0.0",
           "parameters": {"lane": "vc3d-tifxyz-merge"}}
    receipt = {
        "schema": "campaignx.vc3d_tifxyz_merge_receipt.v1",
        "status": "PASS", "surface_id": "merged-surface",
        "artifact_uri": "s3://evidence/merged-surface",
        "artifact_sha256": "a" * 64,
        "parents": [{"surface_id": "a"}, {"surface_id": "b"}],
    }
    publication = {
        "schema": "campaignx.vc3d_merge_evidence_publication.v1",
        "evidence_uri": "s3://evidence/evidence-merged-surface",
        "evidence_sha256": "b" * 64,
        "registration": {"surface_id": "merged-surface"},
    }
    fields = ink_worker.merge_result_from_receipt(job, receipt, publication)
    assert fields["merge_receipt"] == receipt
    assert fields["surface_id"] == "merged-surface"
    assert fields["artifact_uri"] == receipt["artifact_uri"]
    assert fields["evidence_uri"] == publication["evidence_uri"]
    assert fields["parents"] == receipt["parents"]

    invalid = {**publication, "evidence_sha256": None}
    with pytest.raises(RuntimeError, match="evidence_sha256"):
        ink_worker.merge_result_from_receipt(job, receipt, invalid)


def test_gpl_files_are_tracked_inside_the_worker_build_context():
    profile = json.loads((
        ROOT / "framework/profiles/05-reconstruction/"
        "vc3d-tifxyz-merge-1.0.0.json"
    ).read_text())
    for key in ("license_path", "notice_path"):
        relative = profile["source"][key]
        assert relative.startswith("framework/licenses/volume-cartographer/")
        assert (ROOT / relative).is_file()
    worker = (ROOT / "containers/images/Containerfile.worker").read_text()
    assert "COPY --from=repo framework /workspace/campaign-x/framework" in worker


def test_the_deployed_worker_proves_the_complete_digest_pinned_toolchain():
    villa = (ROOT / "containers/images/Containerfile.villa").read_text()
    worker = (ROOT / "containers/images/Containerfile.worker").read_text()
    builder = (ROOT / "containers/build-worker.sh").read_text()
    for binary in ("vc_merge_tifxyz", "vc_obj2tifxyz_legacy",
                   "vc_flatten", "vc_render_tifxyz"):
        assert binary in villa
        assert binary in worker
    assert "VILLA_IMAGE_DIGEST" in builder
    assert "VILLA_IMAGE_DIGEST" in worker
    assert "RepoDigests" in builder


def test_p8_merge_is_claimed_only_by_the_full_villa_worker():
    segment = (ROOT / "containers/compose/segment.compose.yaml").read_text()
    ink = (ROOT / "containers/compose/ink.compose.yaml").read_text()
    assert "HELENA_FLEET_RUNNER_PHASES:-P2,P3,P8" in segment
    assert "HELENA_INK_PHASES:-P4,P5,P7,P9" in ink
