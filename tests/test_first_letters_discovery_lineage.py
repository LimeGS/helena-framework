from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import canonical_lineage
from fleet.postgres_store import PostgresFleetStore
from fleet.store import FleetStore
from test_geometry_certification import (
    _prepare_finalization, _snapshot, _surface, _task,
)


BOUNDARIES = {
    "P1_FINALIZATION_INSERT", "DIRECT_SURFACE_IMPORT",
    "P2_QUEUE_ADMISSION", "P2_EXECUTION_RESOLUTION",
    "PHYSICAL_QC_DIRECT_ENQUEUE", "PHYSICAL_QC_CLAIM_RESOLUTION",
    "P3_QUEUE_ADMISSION", "P3_EXECUTION_RESOLUTION",
    "P4_QUEUE_ADMISSION", "P4_EXECUTION_RESOLUTION",
    "P5_QUEUE_ADMISSION", "P5_EXECUTION_RESOLUTION",
    "P7_QUEUE_ADMISSION", "P7_EXECUTION_RESOLUTION",
    "P8_QUEUE_ADMISSION", "P8_PARENT_MATERIALIZATION",
    "P8_DERIVED_SURFACE_REGISTRATION",
}

REASONS = {
    "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED",
    "CONTROLLED_MISSION_EXTERNAL_ADMISSION_REQUIRED",
    "CANONICAL_LINEAGE_MISSING", "CANONICAL_LINEAGE_AMBIGUOUS",
    "CANONICAL_LINEAGE_HASH_CONFLICT", "CANONICAL_SOURCE_BINDING_MISMATCH",
    "CANONICAL_SURFACE_STATE_INVALID", "ALLOW_UNVALIDATED_PROHIBITED",
    "P1_FINALIZATION_LINEAGE_INCOMPLETE", "SURFACE_IMPORT_LINEAGE_INCOMPLETE",
    "P2_LINEAGE_INCOMPLETE", "PHYSICAL_QC_LINEAGE_INCOMPLETE",
    "P3_LINEAGE_INCOMPLETE", "P4_LINEAGE_INCOMPLETE", "P5_LINEAGE_INCOMPLETE",
    "P7_LINEAGE_INCOMPLETE", "P8_INPUT_LINEAGE_INCOMPLETE",
    "P8_OUTPUT_LINEAGE_INCOMPLETE", "CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK",
}


def _lineage(**changes):
    value = {
        "schema": "campaignx.authoritative_surface_lineage.v1",
        "mission_id": "mission-a", "surface_id": "surface-a",
        "namespace": "CANONICAL_SURFACE", "artifact_identity": "artifact-set-a",
        "artifact_sha256": "1" * 64, "artifact_uri": "file:///canonical/a",
        "source_snapshot_id": "source-a", "source_binding_sha256": "2" * 64,
        "promotion_lineage_sha256": None, "route_sha256": "3" * 64,
        "surface_state": "QC_PENDING", "canonical": True,
        "external": False, "external_admission_sha256": None,
        "ambiguous": False, "hash_conflict": False,
    }
    value.update(changes)
    return value


def _decision(boundary, lineage=None, **kwargs):
    return canonical_lineage.canonical_lineage_decision(
        boundary=boundary, controlled_mission=True,
        authoritative_lineage=_lineage() if lineage is None else lineage,
        allow_unvalidated=False, **kwargs,
    )


def _discovery_surface_payload(source_id="source-a", surface_id="surface-a"):
    lineage = _lineage(
        surface_id=surface_id, source_snapshot_id=source_id,
        namespace="NONCANONICAL_DISCOVERY", canonical=False,
        artifact_uri="file:///probes/task-a/surface",
        artifact_identity="probe_artifact_sets:probe-a",
    )
    return {
        "surface_id": surface_id, "source_snapshot_id": source_id,
        "sample_id": "PHercA", "owner": "campaign-x",
        "artifact_sha256": "1" * 64,
        "artifact_uri": "file:///probes/task-a/surface",
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "state": "QC_PENDING", "physical_qc_state": "UNVALIDATED",
        "controlled_first_letters": True, "mission_id": "mission-a",
        "authoritative_lineage": lineage, "allow_unvalidated": False,
    }


def _forge_sqlite_surface(store: FleetStore, payload: dict, *, geometry="GEOMETRY_UNMEASURED"):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO surfaces
               (surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,
                artifact_uri,bbox_xyz_json,area_cm2,state,physical_qc_state,
                geometry_qc_state,payload_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (payload["surface_id"], payload["source_snapshot_id"],
             payload["sample_id"], payload["owner"],
             payload["artifact_sha256"], payload["artifact_uri"],
             json.dumps(payload["bbox_xyz"]), 1.0, payload["state"],
             payload["physical_qc_state"], geometry,
             json.dumps(payload, sort_keys=True, separators=(",", ":")), now),
        )


def test_boundary_enum_equals_complete_literal_ingress_execution_registration_set():
    assert set(canonical_lineage.CanonicalLineageBoundary) == BOUNDARIES


def test_p1_finalizer_rejects_discovery_parent_before_surface_or_artifact_insert(tmp_path):
    store = FleetStore(tmp_path / "p1.sqlite"); store.initialize()
    source_id = _snapshot(store)
    task = _task(store, source_id, "discovery-parent-cell")
    surface, artifact_set_id = _prepare_finalization(
        store, task, _surface(source_id, "discovery-parent", "GEOMETRY_CERTIFIED")
    )
    with store.connect() as connection:
        row = connection.execute(
            "SELECT payload_json FROM tasks WHERE task_id=?", (task["task_id"],)
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload.update({
            "namespace": "NONCANONICAL_DISCOVERY",
            "controlled_first_letters": True,
            "allow_unvalidated": False,
        })
        connection.execute(
            "UPDATE tasks SET payload_json=? WHERE task_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),
             task["task_id"]),
        )
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as caught:
        store.finalize(
            task["task_id"], task["attempt_id"], task["lease_token"], surface,
            artifact_set_id, "geometry-screen-v1@1",
        )
    assert caught.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM qc_jobs").fetchone()[0] == 0
        assert connection.execute(
            "SELECT state FROM artifact_sets WHERE artifact_set_id=?",
            (artifact_set_id,),
        ).fetchone()["state"] == "UPLOADED"


def test_direct_import_panel_and_both_stores_reject_discovery_before_insert(
    tmp_path, monkeypatch,
):
    store = FleetStore(tmp_path / "fleet.sqlite"); store.initialize()
    store.register_snapshot({
        "source_snapshot_id": "source-a", "sample_id": "PHercA",
        "ct_uri": "fixture://ct", "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7", "m7_sha256": "1" * 64,
        "shape_xyz": [10, 10, 10], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    # Refused earlier than it used to be, and for a wider reason. This asserted
    # that a payload carrying a *discovery* lineage was rejected -- which left
    # a payload carrying a canonical-looking one accepted, stored whole, and
    # read back out of that same row by resolve_canonical_surface_lineage as
    # the answer to whether the surface is canonical. A surface being imported
    # states no lineage at all now, so there is nothing to inspect.
    for store_under_test in (store, PostgresFleetStore("postgresql://must-not-connect")):
        with pytest.raises(ValueError, match="own lineage"):
            store_under_test.import_surface({
                "surface_id": "surface-a", "source_snapshot_id": "source-a",
                "sample_id": "PHercA", "artifact_sha256": "1" * 64,
                "artifact_uri": "file:///discovery/a",
                "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
                "controlled_first_letters": True,
                "authoritative_lineage": _lineage(namespace="NONCANONICAL_DISCOVERY"),
                "allow_unvalidated": False,
            })
        # And the one that would have passed the old gate: canonical on its
        # face, asserted by whoever sent it.
        with pytest.raises(ValueError, match="own lineage"):
            store_under_test.import_surface({
                "surface_id": "surface-a", "source_snapshot_id": "source-a",
                "sample_id": "PHercA", "artifact_sha256": "1" * 64,
                "artifact_uri": "file:///discovery/a",
                "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
                "controlled_first_letters": True,
                "authoritative_lineage": _lineage(),
            })
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0

    import panel.app as panel_app
    monkeypatch.setattr(panel_app, "DSN", "postgresql://must-not-connect")
    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _: {})
    monkeypatch.setattr(
        panel_app.mission_contract, "is_first_letters_discovery_manifest",
        lambda _: True,
    )
    request = panel_app.ImportRequest(
        sample_id="PHerc826", mission_id="mission-a",
        surfaces=[{
            "artifact_uri": "file:///probes/task-a/surface",
            "artifact_sha256": "1" * 64,
            "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
            "namespace": "NONCANONICAL_DISCOVERY",
        }],
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        panel_app.api_import(
            request, SimpleNamespace(state=SimpleNamespace(username="tester"))
        )
    assert caught.value.status_code == 400
    assert caught.value.detail == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"


def test_p2_admission_and_worker_resolution_reject_forged_discovery_job(tmp_path):
    store = FleetStore(tmp_path / "p2.sqlite"); store.initialize()
    source_id = store.register_snapshot({
        "source_snapshot_id": "source-a", "sample_id": "PHercA",
        "ct_uri": "fixture://ct", "m7_uri": "fixture://m7",
        "shape_xyz": [10, 10, 10], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    payload = _discovery_surface_payload(source_id)
    _forge_sqlite_surface(store, payload)
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as admission:
        store.surfaces_without_geometry_verdict(surface_id=payload["surface_id"])
    assert admission.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as execution:
        store.surface_artifact(
            payload["surface_id"], boundary="P2_EXECUTION_RESOLUTION"
        )
    assert execution.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT geometry_qc_state FROM surfaces WHERE surface_id=?",
            (payload["surface_id"],),
        ).fetchone()["geometry_qc_state"] == "GEOMETRY_UNMEASURED"


@pytest.mark.parametrize("forged_state", ["PENDING", "CLAIMED"])
def test_physical_qc_enqueue_claim_restart_and_retry_reject_discovery_lineage(
    tmp_path, forged_state,
):
    store = FleetStore(tmp_path / f"qc-{forged_state}.sqlite"); store.initialize()
    source_id = store.register_snapshot({
        "source_snapshot_id": "source-a", "sample_id": "PHercA",
        "ct_uri": "fixture://ct", "m7_uri": "fixture://m7",
        "shape_xyz": [10, 10, 10], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    payload = _discovery_surface_payload(source_id)
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as enqueue:
        store.enqueue_imported_surface_qc(payload, profile_id="physical-qc@1")
    assert enqueue.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    _forge_sqlite_surface(store, payload, geometry="GEOMETRY_CERTIFIED")
    now = datetime.now(timezone.utc)
    lease_expires = (
        (now - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
        if forged_state == "CLAIMED" else None
    )
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO qc_jobs
               (qc_job_id,surface_id,profile_id,state,payload_json,worker_id,
                lease_token,lease_expires_at,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            ("qc-forged", payload["surface_id"], "physical-qc@1", forged_state,
             "{}", "old-worker" if forged_state == "CLAIMED" else None,
             "old-token" if forged_state == "CLAIMED" else None, lease_expires,
             now.isoformat().replace("+00:00", "Z"),
             now.isoformat().replace("+00:00", "Z")),
        )
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as claim:
        store.claim_qc("new-worker", 60)
    assert claim.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    with store.connect() as connection:
        row = connection.execute(
            "SELECT state,worker_id FROM qc_jobs WHERE qc_job_id='qc-forged'"
        ).fetchone()
        assert row["state"] == forged_state
        assert row["worker_id"] == (
            "old-worker" if forged_state == "CLAIMED" else None
        )


def test_p3_admission_selection_and_worker_resolution_reject_discovery_lineage():
    class Cursor:
        def __init__(self): self.executed = []
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, sql, args=()): self.executed.append((sql, args))
        def fetchall(self):
            return [{
                "surface_id": "surface-a", "sample_id": "PHercA",
                "artifact_uri": "file:///probes/a", "artifact_sha256": "1" * 64,
                "geometry_qc_state": "GEOMETRY_CERTIFIED",
                "physical_qc_state": "CT_SUPPORTED", "voxel_size_um": 9.0,
                "payload": _discovery_surface_payload(),
            }]
    class Connection:
        def __init__(self): self.value = Cursor()
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def cursor(self): return self.value
    store = PostgresFleetStore("postgresql://scripted")
    connection = Connection(); store.connect = lambda: connection
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as selection:
        store.surfaces_awaiting_flattening("flatten@1", surface_id="surface-a")
    assert selection.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"

    class RecordCursor(Cursor):
        def fetchone(self): return {"payload": _discovery_surface_payload()}
    record_connection = Connection(); record_connection.value = RecordCursor()
    store.connect = lambda: record_connection
    receipt = {
        "surface_id": "surface-a", "profile_id": "flatten@1",
        "requested_by_job_id": "p3-forged", "source_artifact_sha256": "1" * 64,
        "profile_file_sha256": "2" * 64,
    }
    from fleet.common import content_sha256
    receipt["receipt_sha256"] = content_sha256(receipt)
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as execution:
        store.record_flattening(receipt)
    assert execution.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    assert not any("INSERT INTO surface_flattenings" in sql
                   for sql, _ in record_connection.value.executed)


def test_panel_p3_admission_rejects_before_job_store_mutation(monkeypatch):
    import panel.app as panel_app
    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _: {})
    monkeypatch.setattr(
        panel_app.mission_contract, "is_first_letters_discovery_manifest",
        lambda _: True,
    )
    class Segmentation:
        def resolve_canonical_surface_lineage(self, **_):
            return _lineage(namespace="NONCANONICAL_DISCOVERY", canonical=False)
    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: Segmentation())
    monkeypatch.setattr(
        panel_app, "job_store",
        lambda: pytest.fail("P3 discovery lineage reached job_store.enqueue"),
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        panel_app.enqueue_fleet_phase(
            "P3", {"surface_id": "surface-a", "allow_unvalidated": False},
            "PHerc826", "mission-a", "tester",
        )
    assert caught.value.status_code == 400
    assert caught.value.detail == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"


def test_p4_admission_and_flattened_surface_resolution_recheck_original_lineage(
    tmp_path,
):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker
    class Store:
        flattened_calls = 0
        def require_surface_lineage(self, **_):
            canonical_lineage.require_canonical_lineage(
                boundary="P4_EXECUTION_RESOLUTION", controlled_mission=True,
                authoritative_lineage=_lineage(
                    namespace="NONCANONICAL_DISCOVERY", canonical=True
                ), allow_unvalidated=False,
            )
        def flattened_sheet(self, *_):
            self.flattened_calls += 1
            return {}
    store = Store()
    job = {
        "mission_id": "mission-a",
        "parameters": {
            "flattened_surface": "surface-a",
            "flattening_profile": "flatten@1", "flattening_id": "flat-a",
            "p3_job_id": "p3-a", "flattened_artifact_sha256": "1" * 64,
        },
    }
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as caught:
        ink_worker.resolve_flattened_surface(store, job, tmp_path / "sheet")
    assert caught.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    assert store.flattened_calls == 0


@pytest.mark.parametrize("phase", ["P4", "P5", "P7", "P8"])
def test_p5_and_p7_surface_bound_admission_and_worker_resolution_reject_discovery(
    tmp_path, monkeypatch, phase,
):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker
    class Store:
        def __init__(self): self.mutations = []
        def require_job_canonical_lineage(self, job, *, execution):
            assert execution is True
            canonical_lineage.require_canonical_lineage(
                boundary=(
                    "P8_PARENT_MATERIALIZATION" if job["phase"] == "P8"
                    else f"{job['phase']}_EXECUTION_RESOLUTION"
                ),
                controlled_mission=True,
                authoritative_lineage=_lineage(namespace="NONCANONICAL_DISCOVERY"),
                allow_unvalidated=False,
            )
        def mark_running(self, *args, **kwargs): self.mutations.append("running")
        def note(self, *args, **kwargs): self.mutations.append("note")
        def finish(self, *args, **kwargs): self.mutations.append("finish")
    store = Store(); subprocess_calls = []
    monkeypatch.setattr(ink_worker, "runner_for", lambda _: Path("runner"))
    monkeypatch.setattr(ink_worker, "command_for", lambda *_, **__: ["runner"])
    monkeypatch.setattr(
        ink_worker.subprocess, "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )
    job = {
        "job_id": f"{phase.lower()}-forged", "lease_token": "lease",
        "phase": phase, "mission_id": "mission-a", "sample_id": "PHercA",
        "profile_id": "profile@1", "parameters": {},
    }
    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as caught:
        ink_worker.run_job(store, job, runs_root=tmp_path, timeout=10)
    assert caught.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    assert subprocess_calls == []
    assert store.mutations == []


@pytest.mark.parametrize("phase,profile,parameters", [
    ("P4", None, {
        "lane": "vc-render-tifxyz", "flattened_surface": "surface-a",
        "flattening_profile": "flatten@1", "flattening_id": "flat-a",
        "p3_job_id": "p3-a", "flattened_artifact_sha256": "1" * 64,
        "volume": "/volume", "scale": 1.0, "group_idx": 0,
    }),
    ("P5", "unknown-profile@1", {
        "surface_id": "surface-a", "tiff_dir": "/layers",
        "checkpoint": "/model",
        "source_pixel_um": 7.91,
    }),
    ("P7", None, {
        "surface_id": "surface-a", "screening_of": "p5-a",
        "bbox": "0,0,10,10", "px_um": 7.91,
        "probability_map_artifact_sha256": "2" * 64,
        "probability_map_manifest_sha256": "3" * 64,
    }),
])
def test_job_store_p4_p5_p7_admission_calls_shared_guard_before_insert(
    phase, profile, parameters,
):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import job_store
    class Cursor:
        def __init__(self): self.executed = []; self.mode = None
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, sql, args=()):
            self.executed.append((sql, args))
            self.mode = "controlled" if "SELECT EXISTS" in sql else (
                "surface" if "FROM segment_surfaces" in sql else "other"
            )
        def fetchone(self): return (True,)
        # A cursor that will not say which columns it returned is not the
        # driver it is imitating, and a decoder that trusts position is exactly
        # what this file exists to stop shipping.
        description = tuple((name,) for name in (
            "surface_id", "source_snapshot_id", "sample_id", "artifact_sha256",
            "artifact_uri", "state", "payload",
        ))
        def fetchall(self):
            payload = _discovery_surface_payload()
            return [(
                "surface-a", "source-a", "PHercA", "1" * 64,
                "file:///probes/a", "QC_PENDING", payload,
            )]
    class Connection:
        def __init__(self): self.cursor_value = Cursor()
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def cursor(self): return self.cursor_value
    connection = Connection()
    store = job_store.InkJobStore("postgresql://scripted")
    store._connect = lambda: connection
    with pytest.raises(Exception) as caught:
        store.enqueue(
            sample_id="PHercA", phase=phase, mission_id="mission-a",
            profile_id=profile, parameters=parameters,
        )
    assert "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED" in str(caught.value)
    assert not any("INSERT INTO ink_jobs" in sql
                   for sql, _ in connection.cursor_value.executed)


def test_p8_queue_parent_materialization_and_derived_registration_each_reject_discovery():
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import job_store
    class Cursor:
        def __init__(self): self.executed = []; self.query = ""
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, sql, args=()): self.query = sql; self.executed.append((sql, args))
        def fetchone(self): return (True,)
        @property
        def description(self):
            if "source_snapshot_id, s.sample_id" in self.query:
                return tuple((name,) for name in (
                    "surface_id", "source_snapshot_id", "sample_id",
                    "artifact_sha256", "artifact_uri", "bbox_xyz",
                    "sample_points", "area_cm2", "state", "physical_qc_state",
                    "geometry_qc_state", "payload", "ct_uri", "ct_sha256",
                    "m7_uri", "m7_sha256", "shape_xyz", "voxel_size_um",
                    "coordinate_frame", "source_payload",
                ))
            return tuple((name,) for name in (
                "surface_id", "source_snapshot_id", "sample_id",
                "artifact_sha256", "artifact_uri", "state", "payload",
            ))
        def fetchall(self):
            payload = _discovery_surface_payload()
            if "source_snapshot_id, s.sample_id" in self.query:
                return [(
                    surface_id, "source-a", "PHercA", "1" * 64,
                    f"file:///probes/{surface_id}", [[0, 0, 0], [1, 1, 1]],
                    [], 1.0, "QC_PENDING", "CT_SUPPORTED",
                    "GEOMETRY_CERTIFIED", payload, "fixture://ct", "0" * 64,
                    "fixture://m7", "1" * 64, [10, 10, 10], 9.0,
                    "ct_l0_xyz", {},
                ) for surface_id in ("surface-a", "surface-b")]
            return [(
                surface_id, "source-a", "PHercA", "1" * 64,
                f"file:///probes/{surface_id}", "QC_PENDING", payload,
            ) for surface_id in ("surface-a", "surface-b")]
    class Connection:
        def __init__(self): self.cursor_value = Cursor()
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def cursor(self): return self.cursor_value
    connection = Connection(); store = job_store.InkJobStore("postgresql://scripted")
    store._connect = lambda: connection
    parameters = {
        "lane": "vc3d-tifxyz-merge",
        "artifact_ids": ["surface-a", "surface-b"],
        "rows": [["surface-a", "surface-b"]],
        "reference_artifact_id": "surface-a", "ransac_seed": 1729,
        "anchor_cap": 2000, "strip_cols": 0,
    }
    with pytest.raises(Exception) as queue:
        store.enqueue(
            server_parameters={"artifact_store": "s3://artifacts"},
            sample_id="PHercA", phase="P8", mission_id="mission-a",
            profile_id="vc3d-tifxyz-merge@1.0.0", parameters=parameters,
        )
    assert "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED" in str(queue.value)
    assert not any("INSERT INTO ink_jobs" in sql
                   for sql, _ in connection.cursor_value.executed)

    connection.cursor_value = Cursor()
    with pytest.raises(Exception) as materialization:
        store.merge_surfaces(["surface-a", "surface-b"])
    assert "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED" in str(materialization.value)

    registration = job_store.InkJobStore("postgresql://must-not-connect")
    registration._connect = lambda: pytest.fail(
        "discovery parent reached merged-surface transaction"
    )
    parents = [{
            "surface_id": surface_id, "artifact_sha256": "1" * 64,
            "mission_id": "mission-a", "controlled_first_letters": True,
            "allow_unvalidated": False,
        "authoritative_lineage": _lineage(
            surface_id=surface_id, namespace="NONCANONICAL_DISCOVERY",
            canonical=False,
        ),
    } for surface_id in ("surface-a", "surface-b")]
    with pytest.raises(Exception) as output:
        registration.register_merged_surface(
            {
                "surface_id": "merged", "source_snapshot_id": "source-a",
                "sample_id": "PHercA", "artifact_sha256": "9" * 64,
                "artifact_uri": "s3://merged", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
                "geometry_qc_state": "GEOMETRY_CERTIFIED",
                "controlled_first_letters": True, "mission_id": "mission-a",
                "allow_unvalidated": False,
                "allow_unvalidated": False,
            }, parents, job_id="p8-a", qc_profile_id="physical-qc@1",
        )
    assert "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED" in str(output.value)


def test_p8_wrapper_rechecks_parents_before_materialization_or_upstream(
    tmp_path, monkeypatch,
):
    import importlib.util
    path = ROOT / "framework/stages/05-reconstruction/scripts/run_vc3d_tifxyz_merge.py"
    spec = importlib.util.spec_from_file_location("lineage_merge_wrapper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    parent_rows = []
    for surface_id in ("surface-a", "surface-b"):
        parent_rows.append({
            "surface_id": surface_id, "sample_id": "PHerc826",
            "source_snapshot_id": "source-a", "artifact_sha256": "1" * 64,
            "artifact_uri": f"file:///probes/{surface_id}",
            "ct_uri": "fixture://ct", "ct_sha256": "0" * 64,
            "coordinate_frame": "ct_l0_xyz", "voxel_size_um": 9.0,
            "geometry_qc_state": "GEOMETRY_CERTIFIED",
            "controlled_first_letters": True, "mission_id": "mission-a",
            "allow_unvalidated": False,
            "authoritative_lineage": _lineage(
                surface_id=surface_id, namespace="NONCANONICAL_DISCOVERY",
                canonical=False,
            ),
        })
    import job_store
    class Store:
        def __init__(self, _): pass
        def merge_surfaces(self, _): return parent_rows
    monkeypatch.setattr(job_store, "InkJobStore", Store)
    monkeypatch.setattr(
        module, "materialize_parents",
        lambda *_: pytest.fail("discovery parent reached materialization"),
    )
    subprocess_calls = []
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )
    monkeypatch.setenv("TEST_P8_DB", "postgresql://scripted")
    args = SimpleNamespace(
        output=tmp_path / "out",
        profile=ROOT / "framework/profiles/05-reconstruction/vc3d-tifxyz-merge-1.0.0.json",
        db="postgres-env://TEST_P8_DB",
        artifact_ids_json=json.dumps(["surface-a", "surface-b"]),
        rows_json=json.dumps([["surface-a", "surface-b"]]),
        reference_artifact_id="surface-a", ransac_seed=1729,
        anchor_cap=2000, strip_cols=0, artifact_store="s3://artifacts",
    )
    with pytest.raises(Exception) as caught:
        module.run(args)
    assert "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED" in str(caught.value)
    assert subprocess_calls == []


def test_controlled_external_surface_requires_explicit_immutable_canonical_admission():
    denied = _decision("DIRECT_SURFACE_IMPORT", _lineage(external=True, canonical=False))
    assert denied["reason_code"] == "CONTROLLED_MISSION_EXTERNAL_ADMISSION_REQUIRED"
    allowed = _decision("DIRECT_SURFACE_IMPORT", _lineage(external=True, canonical=True, external_admission_sha256="5" * 64))
    assert allowed["allowed"] is True


def test_generic_external_mesh_and_ordinary_canonical_positive_paths_remain_unchanged():
    generic = canonical_lineage.canonical_lineage_decision(
        boundary="DIRECT_SURFACE_IMPORT", controlled_mission=False,
        authoritative_lineage=_lineage(external=True, canonical=False),
        allow_unvalidated=True,
    )
    assert generic["allowed"] is True
    assert _decision("P3_EXECUTION_RESOLUTION")["allowed"] is True


def test_each_boundary_uses_frozen_specific_incomplete_reason_code():
    for boundary in BOUNDARIES:
        expected = canonical_lineage.INCOMPLETE_REASON_BY_BOUNDARY[boundary]
        assert _decision(boundary, {})["reason_code"] == expected


def test_allow_unvalidated_true_is_rejected_at_every_controlled_boundary_and_worker():
    for boundary in BOUNDARIES:
        row = canonical_lineage.canonical_lineage_decision(
            boundary=boundary, controlled_mission=True,
            authoritative_lineage=_lineage(), allow_unvalidated=True,
        )
        assert row["reason_code"] == "ALLOW_UNVALIDATED_PROHIBITED"


def test_exported_shared_lineage_guard_api_schema_boundaries_and_reason_map_are_frozen():
    assert set(canonical_lineage.FROZEN_REASON_CODES) == REASONS
    assert canonical_lineage.resolve_authoritative_surface_lineage
    assert canonical_lineage.require_canonical_lineage
    decision = _decision("P8_QUEUE_ADMISSION")
    assert decision["schema"] == "campaignx.canonical_lineage_decision.v1"
    assert decision["allow_unvalidated"] is False


def test_an_imported_surface_cannot_write_the_answer_the_resolver_gives_back(tmp_path):
    """The chain this closes, end to end.

    `import_surface` persists the payload whole. `resolve_canonical_surface_lineage`
    reads `authoritative_lineage` back out of that stored row. Every downstream
    gate then calls require_canonical_lineage with it. So a caller that put one
    in its payload wrote the answer to its own audit -- and the resolver's
    promise to go through a store-owned row rather than caller nesting was true
    about the row and false about how the row got there.
    """
    store = FleetStore(tmp_path / "fleet.sqlite"); store.initialize()
    store.register_snapshot({
        "source_snapshot_id": "source-a", "sample_id": "PHercA",
        "ct_uri": "fixture://ct", "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7", "m7_sha256": "1" * 64,
        "shape_xyz": [10, 10, 10], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    honest = {
        "surface_id": "surface-a", "source_snapshot_id": "source-a",
        "sample_id": "PHercA", "artifact_sha256": "1" * 64,
        "artifact_uri": "file:///imported/a", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
    }
    store.import_surface(honest)

    # What the resolver returns is derived from the row, field by field, and
    # not a document the caller wrote. The derived form still defaults to
    # canonical -- that is the pipeline's own policy for an ordinary import and
    # is not changed here -- but it is derived, so every field is one this
    # store computed.
    resolved = store.resolve_canonical_surface_lineage(
        surface_id="surface-a", mission_id="mission-a")
    assert resolved["surface_id"] == "surface-a"
    assert resolved.get("external_admission_sha256") is None

    # The document is refused outright.
    with pytest.raises(ValueError, match="own lineage"):
        store.import_surface({**honest, "surface_id": "surface-b",
                              "authoritative_lineage": _lineage()})
    # And so is the one key of the derived form that decides `canonical`.
    with pytest.raises(ValueError, match="namespace"):
        store.import_surface({**honest, "surface_id": "surface-c",
                              "namespace": "CANONICAL_SURFACE"})

    # Narrowing still works: a surface marking itself as discovery is how the
    # generator says what it made, and refusing that would remove the only way
    # to be honest.
    store.import_surface({**honest, "surface_id": "surface-d",
                         "artifact_sha256": "d" * 64,
                         "artifact_uri": "file:///imported/d",
                         "namespace": "NONCANONICAL_DISCOVERY"})
    assert store.resolve_canonical_surface_lineage(
        surface_id="surface-d", mission_id="mission-a")["canonical"] is False
