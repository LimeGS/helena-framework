"""The endpoints today's pages were built on, through the app itself.

Everything these serve was written this week, and each of them was wrong once in
a way that looked like an empty campaign rather than a broken read: the coverage
query returned KeyError and the page said "no cells attempted", the replan query
did the same and the panel reported it as a policy refusal, and the parameter
schema is the thing standing between a flag existing in the queue and anybody
being able to set it.

Driven through TestClient with no control plane behind it, so these run
anywhere. What they hold is the shape of each answer and the behaviour when the
database is absent -- which is the state a fresh deployment is in, and the one
where a page that lies is most expensive.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    # No control plane: every one of these has to answer without one.
    monkeypatch.setattr(module, "DSN", "")
    return module


@pytest.fixture
def anonymous(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


@pytest.fixture
def client(app_module, anonymous):
    """Signed in through the real login, because the gate is what stands between
    the network and a panel that queues GPU work."""
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    response = anonymous.post("/api/session",
                              json={"username": "tester", "password": "a-long-enough-one"})
    assert response.status_code == 200, response.text
    return anonymous


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

NEW_ROUTES = ["/api/phases/P4/parameters", "/api/coverage", "/api/ink/lanes"]


@pytest.mark.parametrize("route", NEW_ROUTES)
def test_a_new_route_is_closed_without_a_session(anonymous, route):
    """Deny by default: a list of what is protected goes stale the moment
    somebody adds a route, and the one they forget is the one that queues work."""
    assert anonymous.get(route).status_code == 401


def test_queueing_a_replan_is_closed_too(anonymous):
    response = anonymous.post("/api/segmentation/replan",
                              json={"grid_version": "g", "policy_version": "p"})
    assert response.status_code == 401


def test_candidate_preflight_is_closed_without_a_session(anonymous):
    response = anonymous.post("/api/segmentation/preflight", json={
        "sample_id": "PHerc0358", "mission_id": "m",
        "grid_version": "grid@1.0.0", "policy_version": "policy@1.0.0",
    })
    assert response.status_code == 401


def test_campaign_resume_identity_comes_from_the_authenticated_session(app_module):
    proposal = {
        "schema": "campaignx.first_letters_campaign_resume_authorization.v1",
        "mission_id": "first-letters",
        "prior_sample_id": "PHerc358",
        "new_sample_id": "PHerc358",
        "prior_policy_version": "search-v1",
        "new_policy_version": "search-v2",
        "prior_admission_sha256": "1" * 64,
        "new_admission_sha256": "2" * 64,
        "prior_decision_receipt_sha256": "3" * 64,
        "material_changes": [],
        "authorized_by": "caller-forged-admin",
        "authentication_context": {
            "mechanism": "HELENA_AUTHENTICATED_PANEL_SESSION",
            "principal": "caller-forged-admin",
            "session_fingerprint_sha256": "7" * 64,
            "request_method": "POST",
            "request_path": "/api/segmentation/runs",
        },
    }
    proposal["authorization_evidence_sha256"] = (
        app_module._canonical_document_sha256({
            "schema": "campaignx.first_letters_resume_authorization_evidence.v1",
            **{key: proposal[key] for key in (
                "mission_id", "prior_sample_id", "new_sample_id",
                "prior_policy_version", "new_policy_version",
                "prior_admission_sha256", "new_admission_sha256",
                "prior_decision_receipt_sha256", "material_changes",
                "authorized_by", "authentication_context",
            )},
        })
    )
    proposal["authorization_sha256"] = (
        app_module._canonical_document_sha256(proposal))
    request = SimpleNamespace(
        state=SimpleNamespace(username="tester"),
        cookies={"helena_session": "test-session-token"},
    )
    bound = app_module._bind_panel_campaign_resume_authorization(
        proposal, request)
    assert bound["authorized_by"] == "tester"
    assert bound["authentication_context"]["principal"] == "tester"
    recorded = {}

    class TrustedStore:
        def register_campaign_resume_principal_attestation(
            self, authorization, *, authenticated_principal,
        ):
            recorded.update({
                "authorization": authorization,
                "authenticated_principal": authenticated_principal,
            })
            return authorization

    app_module._attest_panel_campaign_resume_authorization(
        bound, request, TrustedStore())
    assert recorded == {
        "authorization": bound,
        "authenticated_principal": "tester",
    }
    with pytest.raises(app_module.HTTPException) as anonymous_error:
        app_module._attest_panel_campaign_resume_authorization(
            bound, SimpleNamespace(
                state=SimpleNamespace(username=None), cookies={}),
            TrustedStore())
    assert anonymous_error.value.status_code == 403


def test_candidate_preflight_models_are_discriminated_and_reject_unknown_fields(app_module):
    common = {
        "scope": "FULL_GRID", "sample_id": "PHerc0358", "mission_id": "m",
        "expected_p0_artifact_id": "p0", "expected_p0_artifact_sha256": "a" * 64,
        "expected_source_snapshot_id": "source", "expected_source_content_lock_sha256": "b" * 64,
        "catalog_snapshot_sha256": "c" * 64,
        "grid_version": "grid@1.0.0", "policy_version": "preflight@1.0.0",
        "grid_step": 1024, "query_radius": 128, "cell_clearance": 0,
        "volume_clearance": 128, "candidate_interior_clearance": 32,
        "selection_strategy": "stratified-clearance-v1",
        "candidate_selection_policy": "score-cell-volume-clearance-v1",
        "seed_region_policy": "fixed-v1", "m7_threshold": 0.2,
        "packet_candidate_limit": 8, "maximum_cells": 256, "parallelism": 2,
        "provider": "vc3d-mcp",
        "ct_material_support_gate": {"policy": "ome-zarr-nearby-material-v1",
          "level": 5, "radius_l0_voxels": 192, "minimum_nonzero_voxels": 1},
    }
    model = app_module.SegmentationCandidateCoveragePreflightRequest(**common)
    assert model.scope == "FULL_GRID"
    with pytest.raises(ValidationError):
        app_module.SegmentationCandidateCoveragePreflightRequest(**common, secret="leak")


def test_candidate_preflight_relational_bounds_and_provider_fail_as_conflict(
    app_module, monkeypatch
):
    monkeypatch.setattr(app_module, "require_write_sample", lambda *_args: "PHerc358")
    for update in ({"provider": "unknown"}, {"grid_step": 128, "query_radius": 128},
                   {"volume_clearance": 64, "query_radius": 128},
                   {"ct_material_support_gate": {"policy": "unknown"}}):
        body = _candidate_preflight_body(app_module).model_copy(update=update)
        with pytest.raises(app_module.HTTPException) as refusal:
            app_module._resolve_candidate_preflight_binding(body)
        assert refusal.value.status_code == 409


def _candidate_preflight_body(app_module):
    return app_module.SegmentationCandidateCoveragePreflightRequest(**{
        "scope": "FULL_GRID", "sample_id": "PHerc0358", "mission_id": "m",
        "expected_p0_artifact_id": "p0", "expected_p0_artifact_sha256": "a" * 64,
        "expected_source_snapshot_id": "source", "expected_source_content_lock_sha256": "b" * 64,
        "catalog_snapshot_sha256": "c" * 64,
        "grid_version": "grid@1.0.0", "policy_version": "preflight@1.0.0",
        "grid_step": 1024, "query_radius": 128, "cell_clearance": 0,
        "volume_clearance": 128, "candidate_interior_clearance": 32,
        "selection_strategy": "stratified-clearance-v1",
        "candidate_selection_policy": "score-cell-volume-clearance-v1",
        "seed_region_policy": "fixed-v1", "m7_threshold": 0.2,
        "packet_candidate_limit": 8, "maximum_cells": 256, "parallelism": 2,
        "provider": "vc3d-mcp",
        "ct_material_support_gate": {"policy": "ome-zarr-nearby-material-v1",
          "level": 5, "radius_l0_voxels": 192, "minimum_nonzero_voxels": 1},
    })


def test_identical_candidate_preflight_is_atomically_refused_while_running(
    app_module, tmp_path, monkeypatch
):
    stage = ROOT / "framework/stages/01-segmentation"
    if str(stage) not in sys.path:
        sys.path.insert(0, str(stage))
    from fleet import candidate_preflight
    from fleet.common import content_sha256

    body = _candidate_preflight_body(app_module)
    binding = {"artifact_id": "p0", "artifact_sha256": "a" * 64,
      "source_snapshot_id": "source", "source_content_lock_sha256": "b" * 64,
      "p0_selection_version": "selection", "selection_sha256": "d" * 64,
      "snapshot_sha256": "e" * 64}
    snapshot = {"source_snapshot_id": "source"}
    monkeypatch.setattr(app_module, "_resolve_candidate_preflight_binding",
                        lambda _body: ("PHerc358", tmp_path, snapshot, dict(binding)))
    monkeypatch.setattr(app_module, "_deployed_revision", lambda: "1" * 40)
    monkeypatch.setattr(app_module, "fleet_store_read_only", lambda: object())
    entered, release = threading.Event(), threading.Event()

    def blocked(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        private = {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
                   "receipt_sha256": "f" * 64, "generated_at_utc": "now"}
        public = {"schema": "campaignx.segment_candidate_coverage_preflight.sanitized.v1",
                  "private_receipt_sha256": "f" * 64, "receipt_sha256": content_sha256({})}
        return {"private_receipt": private, "sanitized_receipt": public}

    monkeypatch.setattr(candidate_preflight, "run_candidate_coverage_preflight", blocked)
    monkeypatch.setattr(candidate_preflight, "persist_candidate_preflight_receipt_pair",
                        lambda _private_path, _public_path, private, public: {
                          "private_receipt": private, "sanitized_receipt": public})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(app_module._run_candidate_preflight_api, body)
        assert entered.wait(2)
        with pytest.raises(app_module.HTTPException) as refused:
            app_module._run_candidate_preflight_api(body)
        assert refused.value.status_code == 409
        release.set()
        assert future.result().status_code == 200
    assert app_module._run_candidate_preflight_api(body).status_code == 200


def test_candidate_preflight_lock_excludes_a_separate_process(
    app_module, tmp_path
):
    handle = app_module._acquire_segmentation_preflight_lock(tmp_path, "identity")
    lock_path = tmp_path / ".locks" / "segmentation-preflight" / "identity.lock"
    script = ("import fcntl,sys; h=open(sys.argv[1],'a+'); "
              "\ntry: fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)"
              "\nexcept BlockingIOError: raise SystemExit(7)")
    blocked = subprocess.run([sys.executable, "-c", script, str(lock_path)], check=False)
    assert blocked.returncode == 7
    app_module._release_segmentation_preflight_lock(handle)
    acquired = subprocess.run([sys.executable, "-c", script, str(lock_path)], check=False)
    assert acquired.returncode == 0


def test_candidate_preflight_refuses_p0_change_after_probe_before_persisting(
    app_module, tmp_path, monkeypatch
):
    stage = ROOT / "framework/stages/01-segmentation"
    if str(stage) not in sys.path:
        sys.path.insert(0, str(stage))
    from fleet import candidate_preflight

    body = _candidate_preflight_body(app_module)
    binding = {"artifact_id": "p0", "artifact_sha256": "a" * 64,
      "source_snapshot_id": "source", "source_content_lock_sha256": "b" * 64,
      "p0_selection_version": "selection", "selection_sha256": "d" * 64,
      "snapshot_sha256": "e" * 64}
    calls = 0

    def resolve(_body):
        nonlocal calls
        calls += 1
        changed = {**binding, "p0_selection_version": "changed"} if calls == 2 else binding
        return "PHerc358", tmp_path, {"source_snapshot_id": "source"}, dict(changed)

    monkeypatch.setattr(app_module, "_resolve_candidate_preflight_binding", resolve)
    monkeypatch.setattr(app_module, "_deployed_revision", lambda: "1" * 40)
    monkeypatch.setattr(app_module, "fleet_store_read_only", lambda: object())
    monkeypatch.setattr(candidate_preflight, "run_candidate_coverage_preflight",
                        lambda *_args, **_kwargs: {"private_receipt": {"receipt_sha256": "f" * 64},
                                                  "sanitized_receipt": {}})
    with pytest.raises(app_module.HTTPException) as refusal:
        app_module._run_candidate_preflight_api(body)
    assert refusal.value.status_code == 409
    assert not list(tmp_path.rglob("*.json"))


def test_candidate_preflight_resolver_verifies_selected_p0_bytes_and_snapshot(
    app_module, tmp_path, monkeypatch
):
    lock = {"schema": "campaignx.source_content_lock.v1",
      "status": "VERIFIED_IMMUTABLE", "verification_method": "fixture-sha256-v1",
      "verified_at_utc": "2026-08-02T00:00:00Z",
      "ct_uri": "fixture://ct?versionId=ct-version-0001", "ct_sha256": "1" * 64,
      "ct_version_id": "ct-version-0001",
      "m7_uri": "fixture://m7?versionId=m7-version-0001", "m7_sha256": "2" * 64,
      "m7_version_id": "m7-version-0001"}
    snapshot = {"source_snapshot_id": "source", "sample_id": "PHerc358",
      "ct_uri": lock["ct_uri"], "ct_sha256": lock["ct_sha256"],
      "m7_uri": lock["m7_uri"], "m7_sha256": lock["m7_sha256"],
      "shape_xyz": [64, 64, 64], "voxel_size_um": 9.362,
      "coordinate_frame": "ct_l0_xyz", "m7_threshold": 0.2,
      "source_content_lock": lock}
    document = {**{key: snapshot[key] for key in (
      "ct_uri", "ct_sha256", "m7_uri", "m7_sha256", "shape_xyz", "voxel_size_um",
      "coordinate_frame", "m7_threshold", "source_snapshot_id", "source_content_lock")},
      "sample_id": "PHerc0358"}
    path = tmp_path / "p0.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    actual_sha = app_module.artifact_contract.content_hash(path)[0]
    catalog = tmp_path / "geometry.sqlite"
    catalog.write_bytes(b"frozen geometry catalog")
    monkeypatch.setattr(app_module, "GEOMETRY_CATALOG", catalog)
    body = _candidate_preflight_body(app_module).model_copy(update={
      "expected_p0_artifact_sha256": actual_sha,
      "expected_source_content_lock_sha256": app_module._canonical_document_sha256(lock),
      "catalog_snapshot_sha256": app_module._path_sha256(catalog),
    })
    choices = {app_module.artifact_contract.selection_key("P0", "PHerc0358"): "p0"}
    selection = {"version_id": "selection", "choices": choices,
                 "content_sha256": app_module.artifact_contract.selection_hash(choices)}
    monkeypatch.setattr(app_module, "require_write_sample",
                        lambda *_args: "PHerc358")
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app_module.artifact_contract, "current_selection",
                        lambda _directory: selection)
    monkeypatch.setattr(app_module.artifact_contract, "get",
                        lambda _directory, _artifact: {"phase": "P0",
                          "sample_id": "PHerc0358", "path": str(path),
                          "content_sha256": actual_sha})
    monkeypatch.setattr(app_module, "fleet_store_read_only",
                        lambda: type("Store", (), {"snapshots": lambda _self, _samples: [snapshot]})())
    sample, _, resolved, binding = app_module._resolve_candidate_preflight_binding(body)
    assert sample == "PHerc358" and resolved == snapshot
    assert binding["artifact_sha256"] == actual_sha
    selection["content_sha256"] = "0" * 64
    with pytest.raises(app_module.HTTPException) as refusal:
        app_module._resolve_candidate_preflight_binding(body)
    assert refusal.value.status_code == 409


def test_candidate_preflight_command_is_complete_and_redacts_the_database(
    app_module, tmp_path
):
    body = _candidate_preflight_body(app_module)
    command = app_module._candidate_preflight_command(
        body, {"artifact_id": "p0", "artifact_sha256": "a" * 64,
               "p0_selection_version": "selection",
               "selection_sha256": "d" * 64,
               "source_content_lock_sha256": "b" * 64},
        "1" * 40, tmp_path / "private.json", tmp_path / "public.json")
    assert "[REDACTED_CX_DB]" in command
    assert not any("password" in value.lower() for value in command)
    for required in ("--source-snapshot-id", "--p0-artifact-sha256",
                     "--p0-selection-sha256", "--source-content-lock-sha256",
                     "--catalog-snapshot-sha256",
                     "--code-revision", "--private-output", "--sanitized-output"):
        assert required in command


def test_hash_named_preflight_receipts_are_create_once_and_content_equal(
    app_module, tmp_path
):
    path = tmp_path / ("a" * 64 + ".private.json")
    value = {"schema": "example.v1", "receipt_sha256": "a" * 64}
    app_module._write_preflight_receipt(path, value, private=True)
    first = path.read_bytes()
    first_stat = path.stat()
    app_module._write_preflight_receipt(path, value, private=True)
    assert path.read_bytes() == first
    assert path.stat().st_ino == first_stat.st_ino
    with pytest.raises(app_module.HTTPException) as collision:
        app_module._write_preflight_receipt(path, {**value, "changed": True}, private=True)
    assert collision.value.status_code == 409


def test_preflight_store_is_read_only_but_normal_store_still_initializes(
    app_module, tmp_path, monkeypatch
):
    calls = []

    class FakeStore:
        def initialize(self):
            calls.append("initialize")

        def verify_read_only_schema(self):
            calls.append("verify")

    fake = FakeStore()
    import fleet.store_factory as factory
    monkeypatch.setattr(app_module, "DSN", str(tmp_path / "existing.sqlite"))
    (tmp_path / "existing.sqlite").touch()
    monkeypatch.setattr(factory, "open_fleet_store", lambda _dsn: fake)
    monkeypatch.setattr(factory, "open_fleet_store_read_only", lambda _dsn: fake,
                        raising=False)
    assert app_module.fleet_store() is fake
    assert calls == ["initialize"]
    calls.clear()
    assert app_module.fleet_store_read_only() is fake
    assert calls == []


def test_preflight_store_does_not_create_an_absent_sqlite_database(
    app_module, tmp_path, monkeypatch
):
    missing = tmp_path / "missing.sqlite"
    monkeypatch.setattr(app_module, "DSN", str(missing))
    with pytest.raises(app_module.HTTPException) as refusal:
        app_module.fleet_store_read_only()
    assert refusal.value.status_code == 409
    assert not missing.exists()


def test_coverage_marks_latest_tampered_preflight_invalid_without_falling_back(
    app_module, tmp_path, monkeypatch
):
    root = tmp_path / "evidence" / "segmentation-preflight" / "PHerc0358"
    root.mkdir(parents=True)
    private = {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
      "scientific_core": {"schema": "core.v1", "funnel": {"cells_surveyed": 8}},
      "generated_at_utc": "2026-08-02T00:00:00Z"}
    private["receipt_sha256"] = app_module._canonical_document_sha256(private["scientific_core"])
    stage = ROOT / "framework/stages/01-segmentation"
    if str(stage) not in sys.path:
        sys.path.insert(0, str(stage))
    from fleet.candidate_preflight import sanitize_candidate_coverage_receipt
    value = sanitize_candidate_coverage_receipt(private)
    (root / f"{private['receipt_sha256']}.private.json").write_text(
      json.dumps(private), encoding="utf-8")
    (root / f"{private['receipt_sha256']}.sanitized.json").write_text(
      json.dumps(value), encoding="utf-8")
    tampered = {**value, "funnel": {"cells_surveyed": 999}}
    tampered_path = root / "z-tampered.sanitized.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app_module, "read_scope", lambda *_args: {"PHerc358"})
    monkeypatch.setattr(app_module, "fleet_store", lambda: type("Store", (), {
      "coverage": lambda _self, _sample, mission_id=None: {
        "schema": "campaignx.segment_coverage.v1", "grids": [], "volumes": [],
        "non_claims": []}})())
    response = app_module.api_coverage(sample="PHerc0358", mission="first-letters")
    body = json.loads(response.body)
    assert body["candidate_preflight"]["evidence_status"] == "INVALID"
    assert "funnel" not in body["candidate_preflight"]
    assert "candidate_coordinates_xyz" not in body["candidate_preflight"]


def test_coverage_exposes_immutable_campaign_pause_even_with_no_queue_rows(
    app_module, monkeypatch,
):
    receipt = {
        "schema": "campaignx.first_letters_campaign_decision.v1",
        "decision": "PAUSE_CANDIDATE_STARVATION",
        "mission_id": "first-letters",
        "policy_version": "search-v1",
        "no_m7_numerator": 7,
        "scientific_terminal_denominator": 8,
        "excluded_attempt_count": 1,
        "excluded_attempts": [{
            "task_id": "task-source", "attempt_id": "attempt-source",
            "reason": "SOURCE_FAILURE",
        }],
        "trigger_attempt_ids": [f"attempt-{index}" for index in range(7)],
        "receipt_sha256": "d" * 64,
        "allowed_next_actions": [
            "CREATE_MATERIALLY_CHANGED_VERSIONED_STRATEGY",
            "CLOSE_CAMPAIGN",
        ],
    }
    active = {
        "schema": "campaignx.first_letters_campaign_active_decision.v1",
        "decision": "CONTINUE",
        "evidence_status": "IN_PROGRESS",
        "mission_id": "first-letters",
        "policy_version": "search-v2",
        "policy_chain": ["search-v1", "search-v2"],
        "evaluation_kind": "ACTIVE_SCIENTIFIC_TERMINAL_BLOCK",
        "evaluation_index": 1,
        "scientific_terminal_attempt_count": 0,
        "no_m7_numerator": 0,
        "scientific_terminal_denominator": 8,
        "excluded_attempt_count": 0,
        "excluded_attempts": [],
        "trigger_attempt_ids": [],
        "state_sha256": "e" * 64,
        "allowed_next_actions": ["QUEUE_NEXT_BOUND_WAVE", "CLOSE_CAMPAIGN"],
    }

    class Store:
        def coverage(self, _sample, mission_id=None):
            assert mission_id == "first-letters"
            return {
                "schema": "campaignx.segment_coverage.v1",
                "grids": [], "volumes": [], "non_claims": [],
            }

        def campaign_decisions(self, *, mission_id, policy_version=None):
            assert mission_id == "first-letters"
            assert policy_version is None
            return [receipt]

        def campaign_active_decision(self, *, mission_id):
            assert mission_id == "first-letters"
            return active

    monkeypatch.setattr(app_module, "read_scope", lambda *_args: {"PHerc358"})
    monkeypatch.setattr(app_module, "fleet_store", lambda: Store())
    monkeypatch.setattr(
        app_module, "_latest_candidate_preflight_evidence",
        lambda *_args: None,
    )
    response = app_module.api_coverage(
        sample="PHerc0358", mission="first-letters")
    body = json.loads(response.body)
    assert body["grids"] == []
    assert body["campaign_decisions"] == [receipt]
    assert body["active_campaign_decision"] == active


def test_valid_candidate_preflight_is_current_or_stale_from_live_binding(
    app_module, tmp_path, monkeypatch
):
    root = tmp_path / "evidence" / "segmentation-preflight" / "PHerc0358"
    root.mkdir(parents=True)
    bindings = {key: None for key in (
      "p0_artifact_id", "p0_artifact_sha256", "p0_selection_version",
      "p0_selection_sha256", "catalog_snapshot_sha256", "source_snapshot_id",
      "source_content_lock_sha256", "ct_sha256", "m7_sha256", "coordinate_frame",
      "voxel_size_um", "m7_threshold", "grid_version", "policy_version", "provider",
      "candidate_selection_policy", "seed_region_policy", "selection_strategy",
      "maximum_cells", "shape_xyz", "source_snapshot_sha256",
      "normalized_request_sha256", "code_revision")}
    bindings["sample_id"] = "PHerc358"
    bindings["mission_id"] = "first-letters"
    for key in ("p0_artifact_sha256", "p0_selection_sha256",
                "catalog_snapshot_sha256", "source_content_lock_sha256", "ct_sha256",
                "m7_sha256", "source_snapshot_sha256", "normalized_request_sha256"):
        bindings[key] = "a" * 64
    bindings["code_revision"] = "1" * 40
    bindings["shape_xyz"] = [1, 1, 1]
    funnel = {key: 0 for key in (
      "total_grid_cells", "grid_cells_in_design_sample",
      "geometrically_eligible_cells_estimate", "geometrically_eligible_sampled_cells",
      "cells_attempted", "cells_surveyed_successfully", "cells_failed_source",
      "cells_with_raw_m7_candidates", "raw_m7_candidates", "post_ct_candidates",
      "post_cell_clearance_candidates", "post_volume_clearance_candidates",
      "packet_retained_candidates")}
    funnel["geometrically_eligible_cells"] = 0
    core = {
      "schema": "campaignx.segment_candidate_coverage_preflight.scientific_core.v1",
      "status": "COMPLETE", "measurement_kind": "CENSUS", "sampling_design": {
        "name": "census", "measurement_kind": "CENSUS", "population_grid_cells": 0,
        "sampled_grid_cells": 0, "inclusion_fraction": 1.0,
        "ordinal_rule": "lexicographic", "ordinal_stride": 1, "ordinal_offset": 0,
        "sampled_index_sha256": "a" * 64, "cell_order_sha256": "b" * 64},
      "planned_sampling_percentage": 100.0,
      "achieved_successful_sampling_percentage": 100.0,
      "sample_id": "PHerc358", "source_snapshot_id": "source", "bindings": bindings,
      "gates": {}, "funnel": funnel, "no_candidate_causes": {},
      "score_statistics": {}, "cell_clearance_statistics": {},
      "volume_clearance_statistics": {}, "spatial_bins": [],
      "candidate_coordinates_xyz": [], "state_mutation": "NONE",
      "growth_allowed": False, "ink_used": False, "non_claim": "not absence",
      "normalized_request": {"sample_id": "PHerc358", "mission_id": "first-letters"},
      "cell_outcomes": [],
      "m7_read_set_manifest_sha256": [], "ct_read_set_manifest_sha256": [],
    }
    private = {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
               "scientific_core": core,
               **{key: core[key] for key in core if key not in {
                 "schema", "cell_outcomes", "m7_read_set_manifest_sha256",
                 "ct_read_set_manifest_sha256"}}}
    private["receipt_sha256"] = app_module._canonical_document_sha256(core)
    stage = ROOT / "framework/stages/01-segmentation"
    if str(stage) not in sys.path:
        sys.path.insert(0, str(stage))
    from fleet.candidate_preflight import sanitize_candidate_coverage_receipt
    import fleet.candidate_preflight as candidate_module
    public = sanitize_candidate_coverage_receipt(private)
    prefix = private["receipt_sha256"]
    (root / f"{prefix}.private.json").write_text(json.dumps(private), encoding="utf-8")
    (root / f"{prefix}.sanitized.json").write_text(json.dumps(public), encoding="utf-8")
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app_module, "_candidate_preflight_current_binding_status",
                        lambda *_args: ("CURRENT", "all frozen bindings match"), raising=False)
    monkeypatch.setattr(candidate_module, "validate_candidate_preflight_receipt_pair",
                        lambda private, public: (private, public))
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "CURRENT"
    monkeypatch.setattr(app_module, "_candidate_preflight_current_binding_status",
                        lambda *_args: ("STALE", "P0 selection changed"), raising=False)
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "STALE"
    assert loaded["evidence_status_reason"] == "P0 selection changed"
    (root / f"{prefix}.private.json").unlink()
    (root / f"{prefix}.sanitized.json").unlink()
    core["sample_id"] = "PHerc9999"
    core["bindings"]["sample_id"] = "PHerc9999"
    private["sample_id"] = "PHerc9999"
    private["bindings"]["sample_id"] = "PHerc9999"
    private["receipt_sha256"] = app_module._canonical_document_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    prefix = private["receipt_sha256"]
    (root / f"{prefix}.private.json").write_text(json.dumps(private), encoding="utf-8")
    (root / f"{prefix}.sanitized.json").write_text(json.dumps(public), encoding="utf-8")
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "INVALID"
    assert "different mission or sample" in loaded["evidence_status_reason"]
    assert "funnel" not in loaded
    (root / f"{prefix}.private.json").unlink()
    (root / f"{prefix}.sanitized.json").unlink()
    core["sample_id"] = "PHerc358"
    core["bindings"]["sample_id"] = "PHerc358"
    core["bindings"]["mission_id"] = "other-mission"
    core["normalized_request"] = {
      "sample_id": "PHerc358", "mission_id": "other-mission"}
    private["sample_id"] = "PHerc358"
    private["bindings"] = core["bindings"]
    private["normalized_request"] = core["normalized_request"]
    private["receipt_sha256"] = app_module._canonical_document_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    prefix = private["receipt_sha256"]
    (root / f"{prefix}.private.json").write_text(json.dumps(private), encoding="utf-8")
    (root / f"{prefix}.sanitized.json").write_text(json.dumps(public), encoding="utf-8")
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "INVALID"
    assert "different mission or sample" in loaded["evidence_status_reason"]
    assert "funnel" not in loaded


def test_candidate_preflight_rejects_unhashed_top_level_scientific_mutation(
    app_module, tmp_path, monkeypatch
):
    root = tmp_path / "evidence" / "segmentation-preflight" / "PHerc0358"
    root.mkdir(parents=True)
    core = {"schema": "core.v1", "bindings": {"mission_id": "first-letters"},
            "funnel": {"cells_surveyed": 8}}
    private = {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
               "scientific_core": core, "bindings": {"mission_id": "tampered"},
               "funnel": {"cells_surveyed": 999}}
    private["receipt_sha256"] = app_module._canonical_document_sha256(core)
    stage = ROOT / "framework/stages/01-segmentation"
    if str(stage) not in sys.path:
        sys.path.insert(0, str(stage))
    from fleet.candidate_preflight import sanitize_candidate_coverage_receipt
    public = sanitize_candidate_coverage_receipt(private)
    prefix = private["receipt_sha256"]
    (root / f"{prefix}.private.json").write_text(json.dumps(private), encoding="utf-8")
    (root / f"{prefix}.sanitized.json").write_text(json.dumps(public), encoding="utf-8")
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app_module, "_candidate_preflight_current_binding_status",
                        lambda *_args: ("CURRENT", "all frozen bindings match"))
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "INVALID"
    assert "funnel" not in loaded and "bindings" not in loaded


def test_candidate_preflight_latest_nondict_json_is_invalid_not_a_500(
    app_module, tmp_path, monkeypatch
):
    root = tmp_path / "evidence" / "segmentation-preflight" / "PHerc0358"
    root.mkdir(parents=True)
    (root / "latest.sanitized.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "INVALID" and "funnel" not in loaded


@pytest.mark.parametrize("core", [
    {"schema": "unsupported.core.v9", "bindings": {}, "funnel": {}},
    {"schema": "campaignx.segment_candidate_coverage_preflight.scientific_core.v1",
     "bindings": {}},
])
def test_candidate_preflight_rejects_signed_unsupported_or_incomplete_core(
    app_module, tmp_path, monkeypatch, core
):
    root = tmp_path / "evidence" / "segmentation-preflight" / "PHerc0358"
    root.mkdir(parents=True)
    private = {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
               "scientific_core": core,
               **{key: value for key, value in core.items() if key != "schema"}}
    private["receipt_sha256"] = app_module._canonical_document_sha256(core)
    stage = ROOT / "framework/stages/01-segmentation"
    if str(stage) not in sys.path:
        sys.path.insert(0, str(stage))
    from fleet.candidate_preflight import sanitize_candidate_coverage_receipt
    public = sanitize_candidate_coverage_receipt(private)
    prefix = private["receipt_sha256"]
    (root / f"{prefix}.private.json").write_text(json.dumps(private), encoding="utf-8")
    (root / f"{prefix}.sanitized.json").write_text(json.dumps(public), encoding="utf-8")
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app_module, "_candidate_preflight_current_binding_status",
                        lambda *_args: ("CURRENT", "all frozen bindings match"))
    loaded = app_module._latest_candidate_preflight_evidence("first-letters", "PHerc0358")
    assert loaded["evidence_status"] == "INVALID" and "funnel" not in loaded


def test_candidate_preflight_current_status_rechecks_p0_source_catalog_and_code(
    app_module, tmp_path, monkeypatch
):
    lock = {"schema": "campaignx.source_content_lock.v1", "status": "VERIFIED_IMMUTABLE"}
    snapshot = {"sample_id": "PHerc358", "source_snapshot_id": "source",
      "ct_sha256": "1" * 64, "m7_sha256": "2" * 64,
      "m7_uri": "fixture://m7",
      "coordinate_frame": "ct_l0_xyz", "voxel_size_um": 9.362,
      "m7_threshold": 0.2, "shape_xyz": [3, 4, 5], "source_content_lock": lock}
    p0_path = tmp_path / "p0.json"
    p0_path.write_text(json.dumps({"source_snapshot_id": "source"}), encoding="utf-8")
    p0_sha = app_module.artifact_contract.content_hash(p0_path)[0]
    catalog = tmp_path / "catalog.sqlite"
    catalog.write_bytes(b"catalog")
    choices = {app_module.artifact_contract.selection_key("P0", "PHerc0358"): "p0"}
    selection = {"version_id": "selection", "choices": choices,
                 "content_sha256": app_module.artifact_contract.selection_hash(choices)}
    record = {"phase": "P0", "path": str(p0_path), "content_sha256": p0_sha}
    monkeypatch.setattr(app_module, "GEOMETRY_CATALOG", catalog)
    monkeypatch.setattr(app_module, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app_module.artifact_contract, "current_selection",
                        lambda _directory: selection)
    monkeypatch.setattr(app_module.artifact_contract, "get",
                        lambda _directory, _artifact: record)
    monkeypatch.setattr(app_module, "fleet_store_read_only", lambda: type("Store", (), {
      "snapshots": lambda _self, _samples: [snapshot]})())
    monkeypatch.setattr(app_module, "_deployed_revision", lambda: "1" * 40)
    bindings = {"sample_id": "PHerc358", "mission_id": "first-letters",
      "p0_artifact_id": "p0",
      "p0_artifact_sha256": p0_sha, "p0_selection_version": "selection",
      "p0_selection_sha256": selection["content_sha256"],
      "catalog_snapshot_sha256": app_module._path_sha256(catalog),
      "source_snapshot_id": "source",
      "source_content_lock_sha256": app_module._canonical_document_sha256(lock),
      "ct_sha256": "1" * 64, "m7_sha256": "2" * 64,
      "m7_uri_sha256": hashlib.sha256(b"fixture://m7").hexdigest(),
      "coordinate_frame": "ct_l0_xyz", "voxel_size_um": 9.362,
      "m7_threshold": 0.2, "shape_xyz": [3, 4, 5],
      "source_snapshot_sha256": app_module._canonical_document_sha256(snapshot),
      "code_revision": "1" * 40}
    assert app_module._candidate_preflight_current_binding_status(
      "first-letters", "PHerc0358", bindings)[0] == "CURRENT"
    bindings["sample_id"] = "PHerc9999"
    status, reason = app_module._candidate_preflight_current_binding_status(
      "first-letters", "PHerc0358", bindings)
    assert status == "STALE" and "sample_id" in reason
    bindings["sample_id"] = "PHerc358"
    selection["version_id"] = "changed"
    status, reason = app_module._candidate_preflight_current_binding_status(
      "first-letters", "PHerc0358", bindings)
    assert status == "STALE" and "p0_selection_version" in reason


def test_p0_can_freeze_the_exact_registered_target_snapshot_for_preflight(
    app_module, monkeypatch
):
    lock = {"schema": "campaignx.source_content_lock.v1", "status": "VERIFIED_IMMUTABLE"}
    row = {"sample_id": "PHerc358", "source_snapshot_id": "source",
      "ct_uri": "https://ct/version-1", "m7_uri": "https://m7/version-1",
      "m7_threshold": 0.2, "shape_xyz": [3, 4, 5], "voxel_size_um": 9.362,
      "coordinate_frame": "ct_l0_xyz", "source_content_lock": lock}
    monkeypatch.setattr(app_module, "fleet_store_read_only", lambda: type("Store", (), {
      "snapshots": lambda _self, _samples: [row]})())
    resolved = app_module._locked_catalog_snapshot_for_p0("PHerc0358", {
      "ct_uri": row["ct_uri"], "m7_uri": row["m7_uri"], "m7_threshold": 0.2,
      "shape_xyz": [3, 4, 5], "voxel_size_um": 9.362})
    assert resolved == row


# --------------------------------------------------------------------------
# The parameter schema, which is what the form draws itself from
# --------------------------------------------------------------------------

def test_the_schema_offers_every_parameter_the_queue_accepts(client):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import PHASE_PARAMETERS

    for phase, accepted in PHASE_PARAMETERS.items():
        body = client.get(f"/api/phases/{phase}/parameters").json()
        assert body["available"], f"{phase}: {body.get('reason')}"
        assert {field["name"] for field in body["fields"]} == set(accepted)


def test_the_flag_that_took_the_control_from_0_09_to_0_885_is_offered(client):
    fields = {f["name"]: f for f in
              client.get("/api/phases/P4/parameters").json()["fields"]}
    assert fields["flip_normals"]["type"] == "boolean"
    assert not fields["flip_normals"]["filled_by_deployment"]
    # And what the deployment decides is not asked of a person.
    assert fields["artifact_store"]["filled_by_deployment"]


def test_the_pairs_that_must_be_exactly_one_travel_with_the_schema(client):
    """Neither is required and exactly one must be there, which "required"
    cannot express -- so the browser must not have to know it separately."""
    p5 = client.get("/api/phases/P5/parameters").json()
    # Three ways of naming one input. `surface_volume` was added so a control
    # can queue a ready OME-Zarr from the open-data bucket, which is what lets
    # it be reproduced without a credential.
    assert ({"tiff_dir", "layer_stack", "surface_volume"}
            == set(p5["exactly_one_of"][0]["names"]))


def test_a_phase_with_nothing_to_queue_says_so(client):
    body = client.get("/api/phases/P0/parameters").json()
    assert body["available"] is False
    assert "takes no queued parameters" in body["reason"]


def test_an_unknown_phase_does_not_pretend(client):
    body = client.get("/api/phases/P99/parameters").json()
    assert body["available"] is False


# --------------------------------------------------------------------------
# The reads, with no control plane behind them
# --------------------------------------------------------------------------

def test_coverage_without_a_control_plane_says_why(client):
    """It answered "no cells attempted" when the query was broken, which is
    indistinguishable from an unexplored scroll -- the exact silence this phase
    exists to break."""
    body = client.get("/api/coverage").json()
    assert body["available"] is False
    assert body["reason"]


def test_the_lane_inventory_survives_without_a_control_plane(client):
    """It reads the profile directory and the method registry, neither of which
    is the database, so a fresh deployment can still see what it could run."""
    body = client.get("/api/ink/lanes").json()
    assert body["available"] is True
    assert body["lanes"], "no ink lanes are described at all"
    assert body["routable"] >= 1
    assert all(lane["reason"] for lane in body["lanes"] if not lane["routable"])


def test_flattening_without_a_control_plane_says_why(client):
    body = client.get("/api/flattening").json()
    assert body["available"] is False


def test_seed_probe_status_is_rollout_safe_without_its_control_plane(client):
    body = client.get("/api/segmentation/probes").json()
    assert body["available"] is False
    assert body["reason"]
    assert body["counts"] == {
        "runs": 0, "trials": 0, "decisions": 0, "promotions": 0,
    }


def test_seed_probe_launcher_contract_is_served_with_its_bounds(client):
    probe = client.get("/api/segmentation/options").json()["probe"]
    assert [mode["id"] for mode in probe["modes"]] == ["off", "shadow", "select"]
    assert probe["default_mode"] == "off"
    # 20, not 3. Three made the probe a tie-break between m7's best candidates,
    # and the first shadow run measured 8 of 8 ELIGIBLE at that depth -- a 0%
    # rejection rate against a 34% break-even. Whether m7's ordering holds at
    # rank 20 is the open question, and the cap is what made it unaskable.
    assert probe["top_k"] == {"minimum": 1, "maximum": 20, "default": 2}
    assert probe["generations"] == {"minimum": 10, "maximum": 20, "default": 12}
    assert probe["select_readiness"]["available"] is False
    assert probe["select_readiness"]["benchmark_approved"] is False
    assert probe["select_readiness"]["review_owner_declared"] is False
    assert "benchmark receipt" in probe["select_readiness"]["reason"]
    assert "verified immutable" in probe["select_readiness"]["reason"]
    assert "review-resolution owner" in probe["select_readiness"]["reason"]
    assert "not proof" in probe["caveat"]


def test_seed_probe_select_is_scoped_to_receipt_samples(
    app_module, tmp_path, monkeypatch
):
    version = "version-1234"

    def locked_entry(sample_id: str) -> dict:
        ct_uri = f"s3://test/{sample_id}/ct?versionId={version}"
        m7_uri = f"s3://test/{sample_id}/m7?versionId={version}"
        return {
            "sample_id": sample_id,
            "ct_uri": ct_uri,
            "ct_sha256": "a" * 64,
            "surface_prediction_uri": m7_uri,
            "surface_prediction_sha256": "b" * 64,
            "source_content_lock": {
                "schema": "campaignx.source_content_lock.v1",
                "status": "VERIFIED_IMMUTABLE",
                "verification_method": "immutable-uri-manifest-sha256-v1",
                "verified_at_utc": "2026-07-30T00:00:00Z",
                "ct_uri": ct_uri,
                "ct_sha256": "a" * 64,
                "ct_version_id": version,
                "m7_uri": m7_uri,
                "m7_sha256": "b" * 64,
                "m7_version_id": version,
            },
        }

    catalog = tmp_path / "eligible.json"
    catalog.write_text(
        json.dumps(
            {"entries": [locked_entry("PHerc125"), locked_entry("PHerc191")]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "CATALOG", catalog)
    monkeypatch.setattr(
        app_module, "SEED_PROBE_BENCHMARK_RECEIPT", tmp_path / "decision.json"
    )
    monkeypatch.setattr(
        app_module, "SEED_PROBE_REVIEW_OWNER", "segmentation-review-test"
    )
    monkeypatch.setenv("HELENA_ENABLE_SEED_PROBE_SELECT", "1")

    stage_root = ROOT / "framework/stages/01-segmentation"
    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))
    from fleet import seed_probe

    monkeypatch.setattr(
        seed_probe,
        "load_seed_probe_benchmark_receipt",
        lambda _path: {
            "benchmark_id": "scope-test-v1",
            "decision_receipt_sha256": "c" * 64,
            "paired_cell_count": 60,
            "scroll_count": 1,
            "authorized_sample_ids": ["PHerc125"],
        },
    )

    authorized = app_module.seed_probe_select_readiness("PHerc0125")
    assert authorized["available"] is True
    assert authorized["benchmark_scope_allows"] is True
    assert authorized["authorized_sample_count"] == 1

    unauthorized = app_module.seed_probe_select_readiness("PHerc0191")
    assert unauthorized["available"] is False
    assert unauthorized["benchmark_approved"] is True
    assert unauthorized["source_locked"] is True
    assert unauthorized["benchmark_scope_allows"] is False
    assert "outside the approved benchmark scope" in unauthorized["reason"]


def a_mission_for(client, sample: str, mission_id: str = "probe") -> str:
    """A real mission holding one scroll.

    Queueing work needs one now -- it is the project that contains the run --
    and the checks each of these tests is actually about sit behind that.
    Without it they would pass on the mission refusal and never reach the lane,
    the probe mode or the missing control plane they claim to test.
    """
    created = client.post("/api/missions", json={
        "mission_id": mission_id, "name": mission_id, "scrolls": [sample]})
    assert created.status_code in (201, 409), created.text
    return mission_id


def test_seed_probe_select_refuses_a_lane_it_would_steer(client):
    response = client.post("/api/segmentation/runs", json={
        "sample_id": "PHerc0826",
        "mission_id": a_mission_for(client, "PHerc0826"),
        "planner": "opencode-v2",
        "seed_probe_mode": "select",
    })
    assert response.status_code == 400
    assert "cost-aware-v2 or deterministic-v2" in response.json()["detail"]


def test_seed_probe_shadow_does_not_restrict_the_planner_lane(client):
    response = client.post("/api/segmentation/runs", json={
        "sample_id": "PHerc0826",
        "mission_id": a_mission_for(client, "PHerc0826"),
        "planner": "opencode-v2",
        "seed_probe_mode": "shadow",
    })
    # It reached the ordinary deployment precondition, so the lane/probe pair
    # was accepted. There is deliberately no control plane in this fixture.
    assert response.status_code == 409
    assert "control plane" in response.json()["detail"]


def test_a_replan_needs_a_control_plane_rather_than_failing_late(client):
    response = client.post(
        "/api/segmentation/replan",
        json={"grid_version": "replan-x", "policy_version": "replan-x",
              "sample_id": "PHerc0826",
              "mission_id": a_mission_for(client, "PHerc0826")})
    assert response.status_code == 409
    assert "control plane" in response.json()["detail"]


def test_a_replan_refuses_a_version_it_cannot_use(client):
    """The versions are the whole safety of this: task identity is (snapshot,
    grid, cell, policy), so a replan under the same policy inserts nothing and
    reports success."""
    response = client.post("/api/segmentation/replan",
                           json={"grid_version": "", "policy_version": "p"})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# The documentation the panel serves instead of a directory of markdown
# --------------------------------------------------------------------------

def test_every_queueable_field_explains_itself(app_module):
    """The user guide documents each phase from the same schema the form draws
    itself from, so a field with no note is a field the guide cannot explain.
    Thirty-four of them had none: the guide would have printed "no note
    recorded" thirty-four times."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import job_store

    missing = [(phase, field["name"])
               for phase in job_store.PHASE_PARAMETERS
               for field in job_store.phase_parameter_schema(phase)["fields"]
               if not field["note"] and not field["filled_by_deployment"]]
    assert missing == []


def test_the_documentation_is_served_rather_than_shipped_as_files(app_module):
    """It moved into the panel: a user guide per phase and a developer
    reference, both reading the committed contracts rather than restating
    them. What is left under docs/ is hash-bound evidence, not prose."""
    stale = sorted(p.name for p in (ROOT / "docs").glob("*.md"))
    assert stale == [], f"markdown left behind: {stale}"

    guide = (ROOT / "panel/web/src/routes/UserGuide.tsx").read_text()
    # Built from the contract and the live schema, not from a copy of either.
    assert "/api/phases" in guide and "/parameters" in guide
    assert "details" in guide, "the per-phase modules are collapsible"
    # Segmentation is planned rather than parameterised, so its controls are the
    # seeders and the eighteen options, and both are served too.
    assert "/api/segmentation/options" in guide
    assert '"/api/segmentation"' in guide

    developer = (ROOT / "panel/web/src/routes/DeveloperReference.tsx").read_text()
    for topic in ("Adding a model or a tool", "Extending each phase",
                  "Doing a phase a different way",
                  "Adding a phase that does not exist yet",
                  "Running a worker of your own"):
        assert topic in developer, topic


def test_the_guide_is_readable_before_a_mission_exists(anonymous):
    """It explains what a mission is. Putting it behind the screen that demands
    one is a manual locked inside the machine it explains."""
    app_tsx = (ROOT / "panel/web/src/App.tsx").read_text()
    gated = app_tsx[app_tsx.index("function Gated()"):]
    assert "/documentation" not in gated


def test_two_screens_do_not_share_a_layout_class(app_module):
    """The sign-in screen and the mission picker were both `.gate`. They want
    opposite layouts -- one centres itself in the viewport, the other is a card
    list at the top of the page -- and the later rule won, so the picker
    rendered as a card floating in the middle of a 100vh box."""
    css = (ROOT / "panel/web/src/styles.css").read_text()
    mission = (ROOT / "panel/web/src/mission.tsx").read_text()
    login = (ROOT / "panel/web/src/Login.tsx").read_text()

    assert 'className="gate"' not in mission
    assert 'className="missiongate"' in mission
    assert 'className="gate"' in login

    # One declaration each, and only the sign-in one takes the viewport.
    assert css.count("\n.gate {") == 1
    assert css.count("\n.missiongate {") == 1
    assert "100vh" not in css[css.index("\n.missiongate {"):][:200]


def test_configuration_does_not_need_a_mission(app_module):
    """Settings, hosts, accounts and the audit log belong to the deployment, not
    to a campaign. Behind the mission gate, Configuration was a menu entry that
    did nothing when clicked."""
    app_tsx = (ROOT / "panel/web/src/App.tsx").read_text()
    ungated, gated = app_tsx.split("function Gated()")
    assert "<Configuration />" not in gated
    assert '<Route path="/configuration"' in ungated


def test_the_secrets_card_survives_having_no_control_plane(app_module):
    """It dereferenced the response before checking for one. Between retries a
    failed query is neither loading nor errored and its data is undefined, so a
    deployment with no control plane -- a fresh install -- got a Configuration
    page that rendered nothing at all.

    And the 409 is an answer, not a hiccup: retrying it three times only delayed
    showing somebody the reason."""
    source = (ROOT / "panel/web/src/components/FleetSecrets.tsx").read_text()
    assert "queryGate" in source
    assert "data!" not in source
    assert "retry: false" in source


def test_every_phase_is_actually_guided(app_module):
    """A module that only restates the contract is not a guide. Each phase must
    say what you are doing, what to have ready, the decisions in order, what to
    look at afterwards, and how it goes wrong quietly."""
    content = (ROOT / "panel/web/src/routes/guide-content.ts").read_text()
    for phase in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"):
        assert f"\n  {phase}: {{" in content, f"{phase} has no guide"
    for section in ("purpose:", "before:", "steps:", "reading:", "traps:"):
        assert content.count(section) >= 10, f"{section} missing from some phase"


def test_the_guide_ships_the_pictures_it_embeds(app_module):
    """A figure whose file is missing is a broken image and a caption explaining
    it. Every src named in the guide must exist under public/, which is what
    Vite copies into the build."""
    import re

    guide = (ROOT / "panel/web/src/routes/UserGuide.tsx").read_text()
    content = (ROOT / "panel/web/src/routes/guide-content.ts").read_text()
    named = set(re.findall(r'src=\{?"([a-z-]+)"', guide + content))
    named |= set(re.findall(r'src: "([a-z-]+)"', content))
    assert named, "the guide embeds no figures at all"

    # Under src/assets and imported, not under a public directory: the panel
    # serves static files from /assets only, so a picture anywhere else falls
    # through to the single-page fallback and renders broken.
    shipped = {p.stem for p in (ROOT / "panel/web/src/assets/guide").glob("*.png")}
    assert named <= shipped, f"missing: {sorted(named - shipped)}"
    for figure in sorted(named):
        assert f'assets/guide/{figure}.png"' in guide, f"{figure} is not imported"

    # And every caption teaches: a figure with no words under it is decoration.
    assert guide.count("caption=") + content.count("caption:") >= len(named)


def test_a_link_carries_the_scope_it_was_read_in(app_module):
    """A phase page is a different page in a different mission, and its tabs are
    different views. Without either in the address, a shared link lands the
    reader wherever their browser last was."""
    mission = (ROOT / "panel/web/src/mission.tsx").read_text()
    phase = (ROOT / "panel/web/src/routes/Phase.tsx").read_text()
    assert 'get("mission")' in mission
    assert 'get("tab")' in phase


def test_the_documentation_opens_folded_shut(app_module):
    """A page that opens as several screens of prose is a page people scroll
    past. Every section starts closed, so the first thing you see is an index of
    what is there."""
    import re

    for name in ("UserGuide", "DeveloperReference"):
        source = (ROOT / f"panel/web/src/routes/{name}.tsx").read_text()
        cards = re.findall(r"<Card [^>]*?>", source, re.S)
        # The one card that stays open is the list of phase modules, and every
        # module inside it is itself a closed <details>.
        open_cards = [c for c in cards if "collapsed" not in c]
        assert open_cards == [], f"{name}: {open_cards}"

    guide = (ROOT / "panel/web/src/routes/UserGuide.tsx").read_text()
    assert "<details className=\"guide-phase\">" in guide, "modules must start closed"
    assert "guide-phase\" open" not in guide


def test_the_prose_has_a_readable_measure(app_module):
    """96ch of small text is a line the eye loses its place in. Prose is capped;
    tables and figures keep the full width because they are scanned."""
    css = (ROOT / "panel/web/src/styles.css").read_text()
    block = css[css.index("---- documentation: a column you can actually read ----"):]
    assert "max-width:74ch" in block
    assert ".guide-prose p" in block and ".guide-phase p" in block


def test_the_openapi_document_is_not_swallowed_by_the_spa() -> None:
    """The API reference reads it, so where it lives is load-bearing.

    FastAPI defaults openapi_url to /openapi.json, and the SPA serves a
    catch-all at /{full_path}. The catch-all won: /openapi.json answered 200
    with index.html, which is the worst shape of wrong for a document a client
    parses as JSON -- no error, just HTML where the paths should be.
    """
    source = (ROOT / "panel/app.py").read_text()
    assert 'openapi_url="/api/openapi.json"' in source, (
        "openapi_url is not under /api/, so the SPA catch-all answers it with HTML"
    )
    page = (ROOT / "panel/web/src/routes/ApiReference.tsx").read_text()
    assert '"/api/openapi.json"' in page, "the API reference reads a different URL"
    assert '"/openapi.json"' not in page.replace('"/api/openapi.json"', ""), (
        "the page still fetches the path the SPA swallows"
    )
