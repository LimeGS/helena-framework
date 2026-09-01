"""Import and HTTP-boundary safety checks for the positive-control harness."""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import hashlib
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/harness"
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "tests"))

import panel_client  # noqa: E402

Panel = panel_client.Panel
AmbiguousMutationError = getattr(panel_client, "AmbiguousMutationError", RuntimeError)

from run_first_letters_positive_control import canonical_sha256, run_positive_control  # noqa: E402
import test_first_letters_positive_control as synthetic_control  # noqa: E402

MANIFEST_PATH = ROOT / "framework/profiles/01-segmentation/first-letters-control-policy-1.1.0.json"
REVISION = "1" * 40


def route_surface(store, surface_id: str, *, area_cm2: float) -> dict:
    """Give a directly inserted surface the routing receipt finalization writes.

    Human review requires the exact standard route, and the route lives in an
    immutable receipt the control plane writes when a surface is created. These
    fixtures insert their surface with SQL rather than through finalization, so
    they write the same receipt with the same builder.
    """
    from fleet import surface_routing

    receipt = surface_routing.build_receipt(
        surface_id=surface_id, area_cm2=area_cm2,
        policy=surface_routing.load_policy(), measurement={}, read_set={})
    with store.connect() as connection:
        connection.execute(
            "INSERT INTO surface_routing_receipts(surface_id,route,"
            "measured_area_cm2,minimum_area_cm2,policy_version,profile_id,"
            "receipt_sha256,receipt_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (receipt["surface_id"], receipt["route"], receipt["measured_area_cm2"],
             receipt["minimum_area_cm2"], receipt["policy_version"],
             receipt["profile_id"], receipt["receipt_sha256"],
             json.dumps(receipt, sort_keys=True, separators=(",", ":")),
             "2026-08-03T00:00:00+00:00"))
    return receipt


def test_runner_is_import_safe_and_does_not_parse_arguments_or_contact_a_panel():
    """Catches module import triggering CLI execution or external I/O."""
    path = HARNESS / "run_first_letters_positive_control.py"
    assert path.is_file(), "the stage-survival runner has not been implemented"
    spec = importlib.util.spec_from_file_location("first_letters_control_import_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.run_positive_control)


def test_panel_classifies_a_dropped_post_as_ambiguous_after_one_request():
    """Catches retrying a POST after a server consumed it but dropped the reply."""
    class DropReplyOpener:
        def __init__(self):
            self.requests = 0

        def open(self, _request, timeout):
            self.requests += 1
            raise http.client.RemoteDisconnected("server consumed request and closed")

    panel = Panel("http://panel.invalid", timeout=2)
    dropped = DropReplyOpener()
    panel.http = dropped
    with pytest.raises(AmbiguousMutationError, match="must not be retried"):
        panel.call("POST", "/mutation", {"value": 1})
    assert dropped.requests == 1


def test_panel_classifies_a_mutation_gateway_timeout_as_ambiguous_without_retry():
    class GatewayTimeoutOpener:
        def __init__(self):
            self.requests = 0

        def open(self, request, timeout):
            self.requests += 1
            raise urllib.error.HTTPError(
                request.full_url, 504, "Gateway Timeout", {},
                io.BytesIO(b'{"detail":"queueing may have completed"}'),
            )

    panel = Panel("http://panel.invalid", timeout=2)
    timed_out = GatewayTimeoutOpener()
    panel.http = timed_out
    with pytest.raises(AmbiguousMutationError, match="must not be retried"):
        panel.call("POST", "/api/segmentation/manual-seeds", {"value": 1})
    assert timed_out.requests == 1


def test_real_panel_routes_fail_closed_if_selected_p0_changes_during_orchestration(
        tmp_path, monkeypatch):
    """The runner must bind route readbacks, not merely trust a scripted response.

    A concurrent P0 selection change between the initial artifact read and the
    read-only preflight is a realistic stale-lineage race.  Every HTTP request in
    this test goes through FastAPI's real request models and route handlers.  Only
    the database/catalog summaries and remote M7/CT readers are replaced at their
    external boundaries.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import panel.app as panel_app
    from framework.contracts import artifact, auth, mission
    import fleet.candidate_preflight as candidate_preflight

    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mission_id = "first-letters-control-route-e2e"
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(panel_app, "RUNS", runs)
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setenv("CX_DEPLOYED_REVISION", REVISION)
    monkeypatch.setattr(panel_app, "fleet_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(panel_app, "mission_scrolls", lambda _mission: {"PHerc0139"})
    monkeypatch.setattr(panel_app, "integrity", lambda _mission=None: [])
    monkeypatch.setattr(panel_app, "targets", lambda _mission=None: [])
    monkeypatch.setattr(panel_app, "scrolls", lambda: {})
    monkeypatch.setattr(panel_app, "index_runs", lambda **_kwargs: [])
    monkeypatch.setattr(panel_app, "ink_profiles", lambda: [])
    monkeypatch.setattr(panel_app, "declared_checkpoints", lambda: [{
        "checkpoint_sha256": document["model_locks"][0]["checkpoint_sha256"],
        "model_family": document["model_locks"][0]["model_family"],
        "declared_by": ["control"],
        "installed": True,
        "installed_at": "/models/timesformer.ckpt",
    }])
    monkeypatch.setattr(panel_app, "verified_control_runtime",
                        lambda: synthetic_control.runtime(document))
    monkeypatch.setattr(
        panel_app, "require_write_sample",
        lambda _mission, sample, _action: sample,
    )

    mission.create(runs, mission_id=mission_id, name="control", scrolls=["PHerc0139"])
    directory = runs / mission_id
    ct = document["source_locks"]["ct"]
    m7 = document["source_locks"]["m7"]

    def frozen_source(snapshot_id: str) -> dict:
        return {
            "schema": "campaignx.p0_frozen_source.v1",
            "sample_id": "PHerc0139",
            "ct_uri": ct["uri"],
            "m7_uri": m7["uri"],
            "shape_xyz": list(reversed(ct["shape_zyx"])),
            "voxel_size_um": ct["voxel_size_um"],
            "coordinate_frame": ct["coordinate_frame"],
            "source_snapshot_id": snapshot_id,
            "control_only": True,
        }

    records = []
    snapshots = []
    p0_dir = directory / "artifacts/P0"
    p0_dir.mkdir(parents=True)
    for name in ("initial", "replacement"):
        snapshot = frozen_source(f"source-{name}")
        source_path = p0_dir / f"{name}.json"
        source_path.write_text(json.dumps(snapshot), encoding="utf-8")
        records.append(artifact.register(
            directory, phase="P0", sample_id="PHerc0139",
            kind="frozen-source", path=source_path,
        ))
        snapshots.append(snapshot)
    selection_key = artifact.selection_key("P0", "PHerc0139")
    artifact.select(
        directory, choices={selection_key: records[0]["artifact_id"]},
        reason="initial orchestrator binding",
    )

    class FleetStore:
        """The control plane, plus the worker that services its preflight queue.

        The route stopped running the measurement and started queueing it, so a
        store that only answers `snapshots` makes the route answer 503 and the
        test read as a policy failure. The job is run here by the deployed
        worker's own executor; only the seed provider, which lives on a worker's
        loopback and has no route from a test, is replaced.
        """

        def __init__(self):
            self.jobs: dict[str, dict] = {}

        def snapshots(self, _samples):
            return list(snapshots)

        def enqueue_candidate_preflight(self, request):
            from fleet.preflight_worker import ControlRegionPreflightExecutor

            job_id = f"pf-{len(self.jobs) + 1}"
            self.jobs[job_id] = {
                "state": "COMPLETED", "request": request, "attempts": 1,
                "receipt": ControlRegionPreflightExecutor(
                    provider=object()).execute(request["snapshot"], request),
            }
            return {"preflight_job_id": job_id, "state": "PENDING", "created": True}

        def preflight_job(self, job_id):
            return self.jobs.get(job_id)

    queue = FleetStore()
    monkeypatch.setattr(panel_app, "fleet_store", lambda: queue)
    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: queue)

    source_locks = document["source_locks"]
    ct_objects = [
        {"object_key": row["path"], "sha256": row["sha256"], "bytes": 11}
        for row in source_locks["ct"]["metadata"]
    ] + [{"object_key": "0/1/2/3", "sha256": "b" * 64, "bytes": 21}]
    m7_objects = [
        {"object_key": row["path"], "sha256": row["sha256"], "bytes": 13}
        for row in source_locks["m7"]["metadata"]
    ] + [{"object_key": "0/1/2/3", "sha256": "d" * 64, "bytes": 17}]
    ct_objects.sort(key=lambda row: row["object_key"])
    m7_objects.sort(key=lambda row: row["object_key"])
    surface_objects = [{"object_key": "meta.json", "sha256": "1" * 64, "bytes": 19}]
    # `provider` because the worker's executor passes one; the boundary it names
    # is exactly what this fake stands in for.
    monkeypatch.setattr(candidate_preflight, "run_control_region_preflight", lambda snapshot, _request, provider=None, frozen_root_objects=None: {
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "state": "SUCCEEDED",
        "status": "COMPLETE",
        "sample_id": snapshot["sample_id"],
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "counts": {"raw_m7": 2, "post_ct": 2, "post_clearance": 1, "packet_limited": 1},
        "closest_survivor_distance_ct_l0_voxels": 1.0,
        "ct_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": ct_objects,
            "canonical_manifest_sha256": canonical_sha256(ct_objects),
        },
        "m7_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": m7_objects,
            "canonical_manifest_sha256": canonical_sha256(m7_objects),
        },
        "surface_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": surface_objects,
            "canonical_manifest_sha256": canonical_sha256(surface_objects),
        },
        "provider_exchange": {
            "request_sha256": "e" * 64, "request_bytes": 12,
            "response_sha256": "f" * 64, "response_bytes": 34,
        },
        "resource_identity": {"provider": "external-boundary-fake"},
    })

    auth.create_user(panel_app.AUTH_ROOT, "tester", "a-long-enough-one")
    client = TestClient(panel_app.app)
    assert client.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one",
    }).status_code == 200

    class TestClientPanel:
        def __init__(self):
            self.calls = []
            self.preflight_readback = None

        def call(self, method: str, path: str, body: dict | None = None) -> dict:
            self.calls.append((method, path, body))
            if method == "POST" and urlsplit(path).path == "/api/segmentation/preflight":
                artifact.select(
                    directory, choices={selection_key: records[1]["artifact_id"]},
                    reason="concurrent replacement",
                )
            response = client.request(method, path, json=body)
            assert response.status_code < 400, response.text
            payload = response.json()
            if urlsplit(path).path == "/api/segmentation/preflight":
                self.preflight_readback = payload
            return payload

    routed_panel = TestClientPanel()
    receipt = run_positive_control(
        routed_panel,
        document,
        mission_id=mission_id,
        deployed_revision=REVISION,
        submitted_by="tester",
    )

    assert routed_panel.preflight_readback["resource_identity"]["p0_artifact_id"] == records[1]["artifact_id"]
    assert receipt["bindings"]["p0_artifact"]["artifact_id"] == records[0]["artifact_id"]
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "P1"
    assert receipt["stages"][1]["reason_code"] == "MISSING_CONTENT_BOUND_DISCOVERY_EVIDENCE"
    assert not any(urlsplit(path).path == "/api/segmentation/manual-seeds"
                   for _, path, _ in routed_panel.calls)


def _exact_lineage_jobs(document: dict, mission_id: str) -> dict[str, dict]:
    p3_surface = synthetic_control.p3_result()
    p4_result = synthetic_control.p4_result()
    p5_result = synthetic_control.p5_result()
    return {
        "p3-route": {
            "job_id": "p3-route", "phase": "P3", "state": "succeeded",
            "mission_id": mission_id, "sample_id": "PHerc0139",
            "parameters": {"surface_id": "surface-control"},
            "result": {"surfaces": [{**p3_surface,
                                      "requested_by_job_id": "p3-route"}]},
        },
        "p4-route": {
            "job_id": "p4-route", "phase": "P4", "state": "succeeded",
            "mission_id": mission_id, "sample_id": "PHerc0139",
            "parameters": {
                "flattened_surface": "surface-control",
                "flattening_id": "flat-control", "p3_job_id": "p3-route",
                "flattened_artifact_sha256": p3_surface["artifact_sha256"],
            },
            "result": {**p4_result, "lateral_metric": {
                **p4_result["lateral_metric"], "lineage": {
                    **p4_result["lateral_metric"]["lineage"],
                    "p4_job_id": "p4-route",
                },
            }},
        },
        "p5-route": {
            "job_id": "p5-route", "phase": "P5", "state": "succeeded",
            "mission_id": mission_id, "sample_id": "PHerc0139",
            "parameters": {"layer_stack": "p4-route"},
            "result": {**p5_result, "physical_normalization": {
                **p5_result["physical_normalization"], "p4_job_id": "p4-route",
            }},
        },
        "p7-route": {
            "job_id": "p7-route", "phase": "P7", "state": "succeeded",
            "mission_id": mission_id, "sample_id": "PHerc0139",
            "parameters": {
                "surface_id": "surface-control", "screening_of": "p5-route",
                "roi_receipt_sha256": "6" * 64,
            },
            "result": synthetic_control.p7_result(),
        },
    }


def test_entire_chain_actual_policy_routes_stop_before_p4_p7_and_review(
        tmp_path, monkeypatch):
    """Real server policy leaves missing scientific locks unproven and unmutated."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import panel.app as panel_app
    from framework.contracts import artifact, auth, mission
    from fleet.store import FleetStore

    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mission_id = "first-letters-real-policy-stop"
    runs = tmp_path / "runs"
    mission.create(runs, mission_id=mission_id, name="control", scrolls=["PHerc0139"])
    reviews = FleetStore(tmp_path / "reviews.sqlite")
    reviews.initialize()
    from fleet.common import content_sha256, file_sha256
    surface_dir = tmp_path / "synthetic-tifxyz"
    surface_dir.mkdir()
    xx, yy = np.meshgrid(np.arange(12, dtype=np.float32),
                         np.arange(12, dtype=np.float32), indexing="xy")
    for axis, values in zip("xyz", (xx, yy, np.zeros_like(xx)), strict=True):
        tifffile.imwrite(surface_dir / f"{axis}.tif", values)
    (surface_dir / "meta.json").write_text("{}", encoding="utf-8")
    files = {name: {"sha256": file_sha256(surface_dir / name),
                    "size_bytes": (surface_dir / name).stat().st_size}
             for name in ("meta.json", "x.tif", "y.tif", "z.tif")}
    surface_sha = content_sha256(files)
    (surface_dir / "ARTIFACT_SET.json").write_text(json.dumps({
        "schema": "campaignx.segmentation_artifact_set.v1",
        "artifact_sha256": surface_sha, "files": files,
    }), encoding="utf-8")
    document["source_locks"]["community_surface"] = {
        "uri": surface_dir.as_uri(), "grid_shape_yx": [12, 12],
        "bbox_ct_l0_xyz": {"minimum": [0.0, 0.0, 0.0],
                            "maximum": [11.0, 11.0, 0.0]},
        "artifacts": [{"path": name, "sha256": row["sha256"]}
                      for name, row in files.items()]}
    lineage = _exact_lineage_jobs(document, mission_id)
    lineage["p3-route"]["result"]["surfaces"][0][
        "source_artifact_sha256"] = surface_sha
    jobs = {job_id: lineage[job_id] for job_id in ("p3-route", "p5-route")}
    ct, m7 = document["source_locks"]["ct"], document["source_locks"]["m7"]
    source_content_lock = {
        "control_profile_id": document["profile_id"],
        "control_profile_sha256": canonical_sha256(document),
        "ct_lock_sha256": canonical_sha256(ct),
        "m7_lock_sha256": canonical_sha256(m7),
    }
    snapshot = {
        "sample_id": "PHerc0139", "ct_uri": ct["uri"], "m7_uri": m7["uri"],
        "shape_xyz": list(reversed(ct["shape_zyx"])),
        "voxel_size_um": ct["voxel_size_um"],
        "coordinate_frame": ct["coordinate_frame"],
        "source_content_lock": source_content_lock,
    }
    snapshot_id = reviews.register_snapshot(snapshot)
    snapshot["source_snapshot_id"] = snapshot_id
    frozen = {**snapshot, "schema": "campaignx.p0_frozen_source.v1",
              "control_only": True, "target_allowed": False}
    p0_path = runs / mission_id / "p0.json"
    p0_path.write_text(json.dumps(frozen), encoding="utf-8")
    p0 = artifact.register(
        runs / mission_id, phase="P0", sample_id="PHerc0139",
        kind="frozen-source", path=p0_path)
    artifact.select(runs / mission_id, choices={
        artifact.selection_key("P0", "PHerc0139"): p0["artifact_id"]},
        reason="actual-policy unproven fixture")
    with reviews.connect() as connection:
        connection.execute(
            "INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,"
            "artifact_sha256,artifact_uri,bbox_xyz_json,state,physical_qc_state,"
            "geometry_qc_state,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("surface-control", snapshot_id, "PHerc0139", "campaign-x", surface_sha,
             str(surface_dir), "[[0,0,0],[11,11,0]]", "QC_PENDING", "UNVALIDATED",
             "GEOMETRY_CERTIFIED", "{}", "2026-08-03T00:00:00+00:00"))

    class ReadOnlyJobStore:
        enqueue_calls: list[dict] = []

        def job(self, job_id):
            return jobs.get(job_id)

        def flattened_sheet(self, surface_id, profile_id):
            flattened = lineage["p3-route"]["result"]["surfaces"][0]
            assert surface_id == flattened["surface_id"]
            assert profile_id == flattened["profile_id"]
            return {
                "flattening_id": flattened["artifact_id"],
                "requested_by_job_id": flattened["requested_by_job_id"],
                "artifact_sha256": flattened["artifact_sha256"],
            }

        def enqueue(self, **request):
            self.enqueue_calls.append(request)
            raise AssertionError("actual unproven policy must not enqueue downstream work")

    store = ReadOnlyJobStore()
    monkeypatch.setattr(panel_app, "RUNS", runs)
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(panel_app, "job_store", lambda: store)
    monkeypatch.setattr(panel_app, "fleet_store", lambda: reviews)
    monkeypatch.setattr(panel_app, "mission_scrolls", lambda _mission: {"PHerc0139"})
    panel_app.app.dependency_overrides[panel_app.first_letters_control_policy] = \
        lambda: document
    auth.create_user(panel_app.AUTH_ROOT, "tester", "a-long-enough-one")
    client = TestClient(panel_app.app)
    assert client.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one",
    }).status_code == 200

    orientation = client.get(
        "/api/geometry/orientation-proof",
        params={"mission": mission_id, "sample": "PHerc0139",
                "surface": "surface-control", "p3_job": "p3-route"},
    )
    assert orientation.status_code == 200, orientation.text
    assert orientation.json()["status"] == "UNPROVEN"
    assert orientation.json()["reason_code"] == "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING"

    roi = client.get(
        "/api/validation/positive-control-roi",
        params={"mission": mission_id, "sample": "PHerc0139",
                "surface": "surface-control", "p5_job": "p5-route"},
    )
    assert roi.status_code == 409
    # This fixture deliberately holds no P4, so the P5 -> P4 -> P3 walk cannot
    # reach a surface and refuses before any binding or transform is consulted.
    assert "JOB_NOT_FOUND" in roi.text
    assert store.enqueue_calls == []
    assert "p4-route" not in jobs and "p7-route" not in jobs
    assert reviews.human_reviews("p7-route") == []
    panel_app.app.dependency_overrides.clear()


def test_entire_chain_synthetic_contract_routes_exact_review_without_acceptance(
        tmp_path, monkeypatch):
    """Synthetic proofs exercise wiring only; they are not First Letters evidence."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import panel.app as panel_app
    from framework.contracts import artifact, auth, mission
    from fleet.store import FleetStore

    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mission_id = "first-letters-synthetic-contract"
    runs = tmp_path / "runs"
    mission.create(runs, mission_id=mission_id, name="synthetic contract only",
                   scrolls=["PHerc0139"])
    directory = runs / mission_id
    reviews = FleetStore(tmp_path / "reviews.sqlite")
    reviews.initialize()
    from fleet.common import content_sha256, file_sha256
    surface_dir = tmp_path / "synthetic-tifxyz"
    surface_dir.mkdir()
    xx, yy = np.meshgrid(np.arange(12, dtype=np.float32),
                         np.arange(12, dtype=np.float32), indexing="xy")
    for axis, values in zip("xyz", (xx, yy, np.zeros_like(xx)), strict=True):
        tifffile.imwrite(surface_dir / f"{axis}.tif", values)
    (surface_dir / "meta.json").write_text("{}", encoding="utf-8")
    files = {name: {"sha256": file_sha256(surface_dir / name),
                    "size_bytes": (surface_dir / name).stat().st_size}
             for name in ("meta.json", "x.tif", "y.tif", "z.tif")}
    surface_sha = content_sha256(files)
    (surface_dir / "ARTIFACT_SET.json").write_text(json.dumps({
        "schema": "campaignx.segmentation_artifact_set.v1",
        "artifact_sha256": surface_sha, "files": files,
    }), encoding="utf-8")
    document["source_locks"]["community_surface"] = {
        "uri": surface_dir.as_uri(), "grid_shape_yx": [12, 12],
        "bbox_ct_l0_xyz": {"minimum": [0.0, 0.0, 0.0],
                            "maximum": [11.0, 11.0, 0.0]},
        "artifacts": [{"path": name, "sha256": row["sha256"]}
                      for name, row in files.items()]}
    orientation = document["checks"]["PIPELINE_CONTROL"]["orientation_parity"]
    reference = document["source_locks"]["community_surface"]
    absolute_evidence = {
        "schema": "campaignx.first_letters_absolute_orientation_evidence.v1",
        "reference_read_set": {
            "uri": reference["uri"], "objects": reference["artifacts"],
            "canonical_manifest_sha256": canonical_sha256(reference["artifacts"]),
        },
        "lineage": {
            "control_profile_id": document["profile_id"],
            "orientation_profile_id": orientation["policy"]["profile_id"],
        },
        "side_decision": {"same_winding_flip_normals": False},
    }
    absolute_evidence["receipt_sha256"] = canonical_sha256(absolute_evidence)
    absolute_path = tmp_path / "synthetic-absolute-orientation.json"
    absolute_path.write_text(json.dumps(
        absolute_evidence, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    orientation["absolute_orientation"] = {
        "verified": True, "evidence_receipt_uri": str(absolute_path),
        "evidence_receipt_sha256": hashlib.sha256(absolute_path.read_bytes()).hexdigest(),
        "same_winding_flip_normals": False}
    ct = document["source_locks"]["ct"]
    m7 = document["source_locks"]["m7"]
    snapshot = {
        "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
        "ct_uri": ct["uri"], "m7_uri": m7["uri"],
        "shape_xyz": list(reversed(ct["shape_zyx"])),
        "voxel_size_um": ct["voxel_size_um"],
        "coordinate_frame": ct["coordinate_frame"],
        "source_snapshot_id": "synthetic-control-source", "control_only": True,
        "target_allowed": False,
        "source_content_lock": {
            "control_profile_id": document["profile_id"],
            "control_profile_sha256": canonical_sha256(document),
            "ct_lock_sha256": canonical_sha256(ct),
            "m7_lock_sha256": canonical_sha256(m7),
        },
    }
    reviews.register_snapshot(snapshot)
    with reviews.connect() as connection:
        connection.execute(
            "INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,"
            "artifact_sha256,artifact_uri,bbox_xyz_json,area_cm2,state,"
            "physical_qc_state,geometry_qc_state,payload_json,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("surface-control", snapshot["source_snapshot_id"], "PHerc0139",
             "campaign-x", surface_sha, str(surface_dir), "[[0,0,0],[11,11,0]]",
             0.5, "QC_PENDING", "CT_SUPPORTED_REVIEW", "GEOMETRY_CERTIFIED", "{}",
             "2026-08-03T00:00:00+00:00"))
    route_surface(reviews, "surface-control", area_cm2=0.5)
    p0_path = directory / "p0.json"
    p0_path.write_text(json.dumps(snapshot), encoding="utf-8")
    p0_record = artifact.register(
        directory, phase="P0", sample_id="PHerc0139", kind="frozen-source",
        path=p0_path,
    )
    artifact.select(directory, choices={
        artifact.selection_key("P0", "PHerc0139"): p0_record["artifact_id"],
    }, reason="synthetic TestClient control binding")

    worker_state = {"manual_queued": False, "manual": {}}
    pipeline = synthetic_control.pipeline(submitted_by="tester",
                                          artifact_sha256=surface_sha)
    p3_surface = synthetic_control.p3_result()
    p3_surface["source_artifact_sha256"] = surface_sha
    p3_surface["receipt"]["source_artifact_sha256"] = surface_sha
    p3_surface["receipt_sha256"] = canonical_sha256(p3_surface["receipt"])
    p5_fixture = synthetic_control.p5_result()
    roi_lineage = {
        "surface_id": "surface-control", "p5_job_id": "p5-job",
        "probability_map_sha256": p5_fixture["probability_map"]["artifact_sha256"],
        "probability_map_manifest_sha256": p5_fixture[
            "probability_map"]["manifest_sha256"],
        "normalization_receipt_sha256": p5_fixture[
            "physical_normalization"]["receipt_sha256"],
        "checkpoint_sha256": p5_fixture["checkpoint_sha256"],
        "profile_id": p5_fixture["physical_normalization"]["profile_id"],
        "profile_sha256": p5_fixture["physical_normalization"]["profile_sha256"]}
    roi_provenance = {
        "schema": "campaignx.first_letters_positive_control_roi_provenance.v1",
        "source_coordinate_system": "flat-control-uv",
        "source_bbox_xyxy": [20, 30, 120, 130],
        "transform": {"scale_xy": [1.0, 1.0], "offset_xy": [0.0, 0.0]},
        "transformed_bbox_xyxy": [20, 30, 120, 130],
        "verified_training_pixel_um": 7.91, "lineage": roi_lineage}
    roi_path = tmp_path / "synthetic-roi.json"
    roi_path.write_text(json.dumps(roi_provenance, sort_keys=True,
                                   separators=(",", ":")), encoding="utf-8")
    document["checks"]["PIPELINE_CONTROL"]["positive_control_roi"] = {
        "verified": True, "provenance_artifact_uri": str(roi_path),
        "provenance_artifact_sha256": hashlib.sha256(roi_path.read_bytes()).hexdigest(),
        "source_coordinate_system": "flat-control-uv",
        "source_bbox_xyxy": [20, 30, 120, 130],
        "p5_transform_receipt_sha256": canonical_sha256(roi_provenance),
        "transformed_bbox_xyxy": [20, 30, 120, 130],
        "verified_training_pixel_um": 7.91}
    snapshot = {**snapshot,
                "source_snapshot_id": "synthetic-control-source-verified",
                "source_content_lock": {
                    "control_profile_id": document["profile_id"],
                    "control_profile_sha256": canonical_sha256(document),
                    "ct_lock_sha256": canonical_sha256(ct),
                    "m7_lock_sha256": canonical_sha256(m7),
                }}
    reviews.register_snapshot(snapshot)
    with reviews.connect() as connection:
        connection.execute(
            "UPDATE surfaces SET source_snapshot_id=? WHERE surface_id=?",
            (snapshot["source_snapshot_id"], "surface-control"),
        )
    verified_p0_path = directory / "p0-verified.json"
    verified_p0_path.write_text(json.dumps(snapshot), encoding="utf-8")
    verified_p0 = artifact.register(
        directory, phase="P0", sample_id="PHerc0139", kind="frozen-source",
        path=verified_p0_path)
    artifact.select(directory, choices={
        artifact.selection_key("P0", "PHerc0139"): verified_p0["artifact_id"],
    }, reason="synthetic verified TestClient control binding")

    class CompletedWorkerStore:
        """External-worker seam: jobs are completed with immutable fixture receipts."""

        def __init__(self):
            self.rows: dict[str, dict] = {}
            self.event_rows: dict[str, list[dict]] = {}

        def enqueue(self, **request):
            phase = request["phase"]
            job_id = f"{phase.lower()}-job"
            # Merged and declared, the way the real queue does it: the server's
            # own parameters arrive separately so validation can tell which
            # half supplied each, and a fake that validated only the request
            # half would reject a job the queue accepts.
            supplied = dict(request.get("server_parameters") or {})
            parameters = synthetic_control.validate_parameters(
                {**request["parameters"], **supplied}, phase,
                server_owned=supplied)
            if phase == "P2":
                result = synthetic_control.p2_result()
                payload = dict(result)
            elif phase == "P3":
                surface = {**p3_surface, "requested_by_job_id": job_id}
                result = {"surfaces": [surface]}
                payload = dict(result)
            elif phase == "P4":
                result = synthetic_control.p4_result()
                payload = {
                    "surface_id": "surface-control", "flattening_id": "flat-control",
                    "p3_job_id": "p3-job",
                    "flattened_artifact_sha256": p3_surface["artifact_sha256"],
                    "profile_id": "flatten-abf-v1@1.0.0",
                    "orientation_receipt_sha256": parameters["orientation_receipt_sha256"],
                    "flip_normals": parameters["flip_normals"],
                }
            elif phase == "P5":
                result = synthetic_control.p5_result()
                normalization = result["physical_normalization"]
                payload = {
                    "rendered_by": "p4-job",
                    "layer_stack_artifact_sha256": normalization[
                        "p4_layer_artifact_sha256"],
                    "layer_stack_manifest_sha256": normalization[
                        "p4_layer_manifest_sha256"],
                    "lateral_metric_receipt_sha256": normalization[
                        "lateral_metric_receipt_sha256"],
                    "source_pixel_um": normalization["source_pixel_um"],
                    "source_slice_um": normalization["source_slice_um"],
                }
            elif phase == "P7":
                result = synthetic_control.p7_result()
                result["probability_map_input"] = {
                    "screened_by": parameters["screening_of"],
                    "artifact_sha256": parameters[
                        "probability_map_artifact_sha256"],
                    "manifest_sha256": parameters[
                        "probability_map_manifest_sha256"],
                }
                payload = {
                    "screened_by": "p5-job", "surface_id": parameters["surface_id"],
                    "probability_map_artifact_sha256": parameters[
                        "probability_map_artifact_sha256"],
                    "probability_map_manifest_sha256": parameters[
                        "probability_map_manifest_sha256"],
                    "roi_receipt_sha256": parameters["roi_receipt_sha256"],
                    "bbox": parameters["bbox"], "px_um": parameters["px_um"],
                }
                packet = directory / "synthetic-p7-packet.json"
                packet.write_text(json.dumps({
                    "synthetic": True,
                    "claim": "contract plumbing only; no text accepted",
                }), encoding="utf-8")
                artifact.register(
                    directory, phase="P7", sample_id="PHerc0139",
                    kind="vetting-packet", path=packet, produced_by=f"job:{job_id}",
                )
            else:  # pragma: no cover - the harness uses only these phases
                raise AssertionError(phase)
            control_binding = {
                key: parameters[key] for key in panel_app.CONTROL_JOB_BINDING_FIELDS
                if key in parameters
            }
            if control_binding:
                result.update(control_binding)
                payload.update(control_binding)
            self.rows[job_id] = {
                "job_id": job_id, "sample_id": request["sample_id"],
                "phase": phase, "profile_id": request.get("profile_id"),
                "state": "succeeded", "parameters": parameters, "result": result,
                "output_dir": f"/synthetic/{job_id}",
                "mission_id": request.get("mission_id"), "worker_id": "fixture-worker",
            }
            self.event_rows[job_id] = [{
                "event_type": "succeeded" if phase in {"P2", "P3"} else "rendered_from",
                "payload": payload, "created_at": "2026-08-03T00:00:00+00:00",
            }]
            return job_id

        def job(self, job_id):
            return self.rows.get(job_id)

        def flattened_sheet(self, surface_id, profile_id):
            p3 = self.rows.get("p3-job") or {}
            surface = next(
                (row for row in (p3.get("result") or {}).get("surfaces", [])
                 if row.get("surface_id") == surface_id
                 and row.get("profile_id") == profile_id), None)
            if surface is None:
                raise RuntimeError("no persisted exact P3 flattening")
            return {
                "flattening_id": surface["artifact_id"],
                "requested_by_job_id": surface["requested_by_job_id"],
                "artifact_sha256": surface["artifact_sha256"],
                "artifact_uri": surface["artifact_uri"],
                "state": surface["state"],
            }

        def jobs(self, *, limit=100, states=None, mission_id=None, **_filters):
            rows = list(self.rows.values())
            if states:
                rows = [row for row in rows if row["state"] in states]
            if mission_id:
                rows = [row for row in rows if row["mission_id"] == mission_id]
            return rows[-limit:]

        def events(self, job_id, limit=50):
            return self.event_rows.get(job_id, [])[-limit:]

    completed_jobs = CompletedWorkerStore()

    class FakeCursor:
        def __init__(self, *, dict_rows=False):
            self.dict_rows = dict_rows
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, _parameters=None):
            sql = " ".join(str(query).split())
            if "SELECT a.attempt_id" in sql and "WHERE a.attempt_id" in sql:
                manual = worker_state["manual"]
                self.rows = [{
                    "attempt_id": pipeline["attempt_id"], "task_id": "task-control",
                    # The state the fleet finalizes a grow with; `SUCCEEDED` is
                    # not one an attempt reaches.
                    "state": "QC_PENDING", "worker_id": "segment-worker",
                    "created_at": None, "updated_at": None, "result": {},
                    "proposal": None, "proposal_sha256": None,
                    "locked_plan": None, "locked_plan_sha256": None,
                    "cell_id": "manual-control", "policy_version": manual["policy_version"],
                    "grid_version": manual["grid_version"], "parameter_envelope": {},
                    "catalog_snapshot_sha256": "1" * 64,
                    "payload": {
                        "source_snapshot_id": snapshot["source_snapshot_id"],
                        "grid_version": manual["grid_version"],
                        "policy_version": manual["policy_version"],
                        "manual_candidates": [{
                            "seed_origin": "human", "submitted_by": "tester",
                            "ct_l0_coordinate": dict(zip(
                                "xyz", pipeline["coordinate_ct_l0_xyz"], strict=True)),
                        }],
                    },
                    "artifact_set_id": "surface-control-set", "manifest": {},
                    "manifest_sha256": "2" * 64, "staging_uri": None,
                }]
            elif "SELECT DISTINCT f.surface_id" in sql:
                self.rows = [(
                    "surface-control", "PHerc0139", pipeline["area_cm2"], "SUCCEEDED",
                    "CT_SUPPORTED_REVIEW", "GEOMETRY_CERTIFIED",
                    "s3://synthetic/surface", pipeline["artifact_sha256"], None,
                    pipeline["attempt_id"], None, "campaign-x", None, {
                        "schema": "campaignx.tifxyz_geometry_certification.v1",
                        "surface_artifact_sha256": pipeline["artifact_sha256"],
                        "source_attempt_id": pipeline["attempt_id"],
                        "profile_id": synthetic_control.p2_result()["profile_id"],
                        "profile_sha256": synthetic_control.p2_result()["profile_sha256"],
                        "result": synthetic_control.p2_result()["result"],
                        "result_sha256": synthetic_control.p2_result()["result_sha256"],
                    },
                    # The lamina axis and the seed agreement, in the state a
                    # control that measured neither is actually in. None here
                    # rather than a passing value on purpose: this row exists to
                    # prove the chain routes, and a synthetic "measured" would
                    # be the one place the control claims something it did not
                    # do. The panel renders these as LAMINA_UNMEASURED and
                    # SEED_UNPAIRED, which is what they are.
                    None, None, None, None,
                )]
            elif "SELECT q.qc_job_id" in sql:
                qc_result = {
                    "schema": "campaignx.segment_surface_qc_result.v1",
                    "surface_id": "surface-control", "outcome": "QC_RETAINED",
                    "evidence_manifest_sha256": "7" * 64,
                    "ink_used": False, "source_result_sha256": "5" * 64,
                }
                self.rows = [(
                    "qc-control", "surface-control", "PHerc0139",
                    "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0", "COMPLETED",
                    None, None, None, 1, "QC_RETAINED", None,
                    qc_result["source_result_sha256"], "GEOMETRY_CERTIFIED",
                    "CT_SUPPORTED_REVIEW", pipeline["artifact_sha256"], None,
                    None, pipeline["attempt_id"], "true", qc_result["schema"],
                    qc_result["surface_id"], qc_result["outcome"],
                    qc_result["evidence_manifest_sha256"], qc_result["ink_used"],
                )]
            elif "SELECT count(*) FROM eligible_surfaces" in sql:
                self.rows = [(0 if "<> ALL" in sql else 1,)]
            elif "SELECT f.state, count(*) FROM surface_flattenings" in sql:
                self.rows = [("FLATTENED", 1)]
            elif "SELECT f.flattening_id" in sql:
                receipt = dict(p3_surface["receipt"])
                payload = {
                    **receipt, "receipt_sha256": p3_surface["receipt_sha256"],
                    "requested_by_job_id": "p3-job",
                    "source_artifact_sha256": p3_surface["source_artifact_sha256"],
                    "profile_file_sha256": p3_surface["profile_file_sha256"],
                    "files": p3_surface["files"], "objects": p3_surface["objects"],
                }
                self.rows = [(
                    "flat-control", "surface-control", "PHerc0139",
                    p3_surface["profile_id"], "FLATTENED", 1.0,
                    p3_surface["artifact_uri"], p3_surface["artifact_sha256"],
                    None, payload,
                )]
            elif sql.startswith("SELECT"):
                self.rows = []
            else:
                self.rows = []
            return self

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            return list(self.rows)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self, row_factory=None):
            return FakeCursor(dict_rows=row_factory is not None)

    def fake_bootstrap(argv, **_kwargs):
        receipt_path = Path(argv[argv.index("--receipt") + 1])
        worker_state["manual_queued"] = True
        worker_state["manual"] = {
            "policy_version": argv[argv.index("--policy-version") + 1],
            "grid_version": argv[argv.index("--grid-version") + 1],
        }
        receipt_path.write_text(json.dumps({
            "seed_origin": "human",
            "policy_version": worker_state["manual"]["policy_version"],
            "tasks": {"PHerc0139": {"generated": 1, "inserted": 1}},
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(panel_app, "RUNS", runs)
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setenv("CX_DEPLOYED_REVISION", REVISION)
    monkeypatch.setattr(panel_app, "DSN", "postgresql://synthetic-worker-state")
    monkeypatch.setattr(panel_app, "RENDER_STORE", "s3://synthetic/layers")
    monkeypatch.setattr(panel_app, "INK_STORE", "s3://synthetic/maps")
    monkeypatch.setattr(panel_app, "FLATTEN_STORE", "s3://synthetic/flat")
    monkeypatch.setattr(panel_app, "FIRST_LETTERS_CONTROL_CT_CACHE", "/synthetic/ct")
    monkeypatch.setattr(panel_app, "job_store", lambda: completed_jobs)
    monkeypatch.setattr(panel_app, "fleet_store", lambda: reviews)
    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: reviews)
    monkeypatch.setattr(panel_app, "mission_scrolls", lambda _mission: {"PHerc0139"})
    monkeypatch.setattr(panel_app, "require_write_sample",
                        lambda _mission, sample, _action: sample)
    monkeypatch.setattr(panel_app, "catalog_sample_id",
                        lambda sample, **_kwargs: sample)
    monkeypatch.setattr(panel_app, "fleet_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(panel_app, "integrity", lambda _mission=None: [])
    monkeypatch.setattr(panel_app, "targets", lambda _mission=None: [])
    monkeypatch.setattr(panel_app, "scrolls", lambda: {})
    monkeypatch.setattr(panel_app, "index_runs", lambda **_kwargs: [])
    monkeypatch.setattr(panel_app, "module_disabled", lambda *_args: False)
    monkeypatch.setattr(panel_app, "ink_profiles", lambda: [{
        "profile_id": "timesformer-gp-scroll1-screening@1.0.0",
        "method_id": "timesformer-gp",
    }])
    monkeypatch.setattr(panel_app, "registry_entries", lambda: {
        "timesformer-gp": {"validation_status": "VALIDATED"},
    })
    monkeypatch.setattr(panel_app, "declared_checkpoints", lambda: [{
        "checkpoint_sha256": document["model_locks"][0]["checkpoint_sha256"],
        "model_family": document["model_locks"][0]["model_family"],
        "declared_by": ["synthetic-control"], "installed": True,
        "installed_at": "/models/timesformer.ckpt",
    }])
    monkeypatch.setattr(panel_app, "segmentation_runs", lambda **_kwargs: {
        "available": True,
        "runs": [pipeline] if worker_state["manual_queued"] else [],
    })
    monkeypatch.setattr(panel_app, "verified_control_runtime",
                        lambda: synthetic_control.runtime(document))
    monkeypatch.setattr(panel_app.subprocess, "run", fake_bootstrap)
    import types
    psycopg = types.ModuleType("psycopg")
    psycopg.connect = lambda *_args, **_kwargs: FakeConnection()
    psycopg_rows = types.ModuleType("psycopg.rows")
    psycopg_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", psycopg_rows)
    import fleet.candidate_preflight as candidate_preflight
    monkeypatch.setattr(
        candidate_preflight, "run_control_region_preflight",
        # `provider` because the worker's executor passes one.
        lambda _snapshot, _request, provider=None, frozen_root_objects=None: synthetic_control.discovery(),
    )

    panel_app.app.dependency_overrides[panel_app.first_letters_control_policy] = \
        lambda: document
    auth.create_user(panel_app.AUTH_ROOT, "tester", "a-long-enough-one")
    client = TestClient(panel_app.app)
    assert client.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one",
    }).status_code == 200

    class TestClientPanel:
        def __init__(self):
            self.calls = []

        def call(self, method, path, body=None):
            self.calls.append((method, path, body))
            response = client.request(method, path, json=body)
            assert response.status_code < 400, response.text
            if method == "POST" and urlsplit(path).path == "/api/segmentation/preflight":
                # A queue nobody services is not a failing test, it is a hung
                # one: the route returns a handle and the runner polls it for
                # half an hour. This is the deployed worker, one iteration, with
                # the seed provider replaced at the boundary a test cannot reach.
                from fleet.preflight_worker import (
                    CandidatePreflightWorker, ControlRegionPreflightExecutor)

                CandidatePreflightWorker(
                    reviews, worker_id="e2e-preflight",
                    executor=ControlRegionPreflightExecutor(provider=object()),
                ).run_one()
            return response.json()

        def wait_for_job(self, job_id, **_kwargs):
            jobs = self.call("GET", "/api/jobs?limit=50")["jobs"]
            return next(row for row in jobs if row["job_id"] == job_id)

        def wait_until(self, predicate, **_kwargs):
            return predicate()

    routed_panel = TestClientPanel()
    receipt = run_positive_control(
        routed_panel, document, mission_id=mission_id,
        deployed_revision=REVISION, submitted_by="tester",
        wait_minutes=0.01,
    )
    assert receipt["control_state"] == "CONTROL_PASS", json.dumps(receipt, indent=2)
    assert receipt["control_pass_is_independent_validation"] is False
    assert receipt["automatic_letter_acceptance"] is False
    assert receipt["stages"][8]["reason_code"] == "HUMAN_REVIEW_PACKET_ROUTED"
    reinjected = client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": mission_id, "phase": "P7",
        "parameters": {"screening_of": "p5-job", "surface_id": "surface-control"},
        "max_attempts": 1,
    })
    assert reinjected.status_code == 201, reinjected.text
    injected = completed_jobs.rows["p7-job"]["parameters"]
    assert injected["bbox"] == "20,30,120,130"
    assert injected["px_um"] == 7.91
    assert injected["roi_receipt_sha256"] == receipt["stages"][7][
        "output_hashes"]["roi_receipt_sha256"]
    forged = client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": mission_id, "phase": "P7",
        "parameters": {"screening_of": "p5-job", "surface_id": "surface-control",
                       "roi_receipt_sha256": "a" * 64},
        "max_attempts": 1,
    })
    assert forged.status_code == 409
    assert len(completed_jobs.rows) == 4
    normalization = completed_jobs.rows["p5-job"]["result"][
        "physical_normalization"]
    normalization["training_pixel_um"] = 8.0
    forged_normalization = client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": mission_id, "phase": "P7",
        "parameters": {"screening_of": "p5-job", "surface_id": "surface-control"},
        "max_attempts": 1,
    })
    assert forged_normalization.status_code == 409
    assert "normalization receipt/profile/model lock mismatch" in \
        forged_normalization.text
    assert len(completed_jobs.rows) == 4
    panel_app.app.dependency_overrides.clear()
    assert all(path.startswith("/api/") for _method, path, _body in routed_panel.calls)
    assert {row["job_id"] for row in completed_jobs.rows.values()} == {
        "p3-job", "p4-job", "p5-job", "p7-job",
    }
    assert not any(path == "/api/geometry/certify"
                   for _method, path, _body in routed_panel.calls)
