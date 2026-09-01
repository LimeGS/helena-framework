from __future__ import annotations

import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from test_first_letters_discovery_evidence_store import _claim_job, _store


IDENTITY_FIELDS = (
    "request_id", "cell_id", "source_snapshot_id",
    "source_snapshot_sha256", "prediction_root_sha256", "resolution",
    "level", "model_id", "model_sha256", "provider_id",
    "provider_sha256", "cell_region_sha256", "grid_spec_sha256",
    "dependency_manifest_sha256", "maximum_candidates",
)


def _provider_response(request: dict) -> bytes:
    return json.dumps(
        {
            "prediction_identity": {
                field: copy.deepcopy(request[field])
                for field in IDENTITY_FIELDS
            },
            "candidates": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _controller_api():
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from fleet.discovery_worker import DiscoveryPreRunRetryable

    return FirstLettersDiscoveryController, DiscoveryPreRunRetryable


class _PreparedProvider:
    def __init__(self, owner, *, delay: float = 0.0, error=None):
        self.owner = owner
        self.delay = delay
        self.error = error

    def execute(self, provider_request: dict) -> bytes:
        self.owner.execute_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return _provider_response(provider_request)


class _Provider:
    def __init__(self, *, preparation_failures=0, delay=0.0, error=None):
        self.preparation_failures = preparation_failures
        self.delay = delay
        self.error = error
        self.prepare_calls = 0
        self.execute_calls = 0

    def prepare(self):
        self.prepare_calls += 1
        if self.prepare_calls <= self.preparation_failures:
            _, retryable = _controller_api()
            raise retryable("provider transport was not started")
        return _PreparedProvider(self, delay=self.delay, error=self.error)


class _PublicApiStore:
    ALLOWED = {
        "claim_first_letters_discovery_job",
        "revalidate_first_letters_discovery_job_claim",
        "begin_first_letters_discovery_evidence_run",
        "start_first_letters_discovery_evidence_run",
        "heartbeat_first_letters_discovery_evidence_run",
        "complete_first_letters_discovery_evidence_run",
        "read_first_letters_discovery_evidence_run",
        "read_first_letters_discovery_evidence_run_status",
        "mark_first_letters_discovery_evidence_run_incomplete",
        "reconcile_expired_first_letters_discovery_evidence_run",
    }

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls: list[str] = []

    def __getattr__(self, name):
        if name not in self.ALLOWED:
            raise AssertionError(f"controller used non-public store API: {name}")
        method = getattr(self.delegate, name)

        def call(*args, **kwargs):
            self.calls.append(name)
            return method(*args, **kwargs)

        return call


class _CommittedResponseLossStore(_PublicApiStore):
    def complete_first_letters_discovery_evidence_run(self, **kwargs):
        self.calls.append("complete_first_letters_discovery_evidence_run")
        return self.delegate.complete_first_letters_discovery_evidence_run(
            **kwargs, failpoint="evidence.after_commit_before_response"
        )


class _AmbiguousCompletionStore(_PublicApiStore):
    def complete_first_letters_discovery_evidence_run(self, **kwargs):
        self.calls.append("complete_first_letters_discovery_evidence_run")
        raise TimeoutError("completion outcome unknown")

    def read_first_letters_discovery_evidence_run(self, run_id):
        self.calls.append("read_first_letters_discovery_evidence_run")
        raise ConnectionError("readback unavailable")


class _InitialHeartbeatResponseLossStore(_PublicApiStore):
    def __init__(self, delegate):
        super().__init__(delegate)
        self.heartbeat_calls = 0

    def heartbeat_first_letters_discovery_evidence_run(self, **kwargs):
        self.calls.append("heartbeat_first_letters_discovery_evidence_run")
        self.heartbeat_calls += 1
        heartbeat = self.delegate.heartbeat_first_letters_discovery_evidence_run(
            **kwargs
        )
        if self.heartbeat_calls == 1:
            raise TimeoutError("initial heartbeat response lost after commit")
        return heartbeat


class _StartResponseLossStore(_PublicApiStore):
    def start_first_letters_discovery_evidence_run(self, **kwargs):
        self.calls.append("start_first_letters_discovery_evidence_run")
        self.delegate.start_first_letters_discovery_evidence_run(**kwargs)
        raise TimeoutError("start response lost after RUNNING commit")


class _TerminalWriteResponseLossStore(_PublicApiStore):
    def mark_first_letters_discovery_evidence_run_incomplete(self, **kwargs):
        self.calls.append(
            "mark_first_letters_discovery_evidence_run_incomplete"
        )
        self.delegate.mark_first_letters_discovery_evidence_run_incomplete(
            **kwargs
        )
        raise TimeoutError("terminal response lost after commit")


class _TerminalWriteUnavailableStore(_PublicApiStore):
    def mark_first_letters_discovery_evidence_run_incomplete(self, **kwargs):
        self.calls.append(
            "mark_first_letters_discovery_evidence_run_incomplete"
        )
        raise ConnectionError("terminal write unavailable")


def _run(controller, reservation, profile_bytes):
    del profile_bytes
    return controller.run_job(
        lease_seconds=60, job_id=reservation["jobs"][0]["job_id"],
    )


def test_a3_claim_lifecycle_requires_running_and_heartbeats_exact_owner(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    claimed = store.read_first_letters_discovery_evidence_run_status(handle.run_id)
    assert claimed["state"] == "CLAIMED"
    assert claimed["evidence_set_id"] is None
    with pytest.raises(ValueError, match="RUNNING"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response(handle.provider_request),
        )
    with pytest.raises(ValueError, match="RUNNING"):
        store.heartbeat_first_letters_discovery_evidence_run(
            run_handle=handle, lease_seconds=60,
        )

    running = store.start_first_letters_discovery_evidence_run(
        run_handle=handle,
    )
    assert running["state"] == "RUNNING"
    heartbeat = store.heartbeat_first_letters_discovery_evidence_run(
        run_handle=handle, lease_seconds=120,
    )
    assert heartbeat["state"] == "RUNNING"
    assert heartbeat["lease_expires_at"] > running["lease_expires_at"]
    with pytest.raises(ValueError, match="owner|claim|token"):
        store.heartbeat_first_letters_discovery_evidence_run(
            run_handle=dataclasses.replace(handle, run_token="wrong"),
            lease_seconds=120,
        )

    completed = store.complete_first_letters_discovery_evidence_run(
        run_handle=handle,
        provider_response_bytes=_provider_response(handle.provider_request),
    )
    status = store.read_first_letters_discovery_evidence_run_status(handle.run_id)
    assert status["state"] == "COMPLETED"
    assert status["evidence_set_id"] == completed["evidence_set_id"]
    with pytest.raises(ValueError, match="RUNNING"):
        store.heartbeat_first_letters_discovery_evidence_run(
            run_handle=handle, lease_seconds=120,
        )


def test_a3_running_claim_can_only_terminalize_as_permanent_incomplete(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    incomplete = store.mark_first_letters_discovery_evidence_run_incomplete(
        run_handle=handle,
        reason="PROVIDER_RESPONSE_AMBIGUOUS_AFTER_RUNNING",
    )
    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    assert incomplete["incomplete_reason"] == (
        "PROVIDER_RESPONSE_AMBIGUOUS_AFTER_RUNNING"
    )
    assert store.read_first_letters_discovery_evidence_run_status(
        handle.run_id
    ) == incomplete
    with pytest.raises(ValueError, match="RUNNING"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response(handle.provider_request),
        )


def test_a3_status_rejects_split_run_and_executor_claim_lifecycle(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with store.connect() as connection:
        connection.execute(
            """UPDATE first_letters_discovery_executor_claims
                  SET state='CONTROL_INCOMPLETE',incomplete_at=?,
                      incomplete_reason=? WHERE run_id=?""",
            (
                "2026-01-01T00:00:00Z", "COMPLETION_FAILED_AFTER_RUNNING",
                handle.run_id,
            ),
        )
    with pytest.raises(ValueError, match="lifecycle|claim|inconsistent"):
        store.read_first_letters_discovery_evidence_run_status(handle.run_id)


def test_a3_heartbeat_rejects_an_expired_running_owner_lease(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with store.connect() as connection:
        connection.execute(
            """UPDATE first_letters_discovery_evidence_runs
                  SET lease_expires_at='2000-01-01T00:00:00Z'
                WHERE run_id=?""",
            (handle.run_id,),
        )
    with pytest.raises(ValueError, match="live RUNNING lease"):
        store.heartbeat_first_letters_discovery_evidence_run(
            run_handle=handle, lease_seconds=60,
        )


def test_a3_reconciliation_rejects_live_or_tampered_running_owner(
    tmp_path, monkeypatch,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with pytest.raises(ValueError, match="expired RUNNING"):
        store.reconcile_expired_first_letters_discovery_evidence_run(
            run_id=handle.run_id,
        )

    import fleet.store as store_module
    monkeypatch.setattr(store_module, "utc_now", lambda: "2999-01-01T00:00:00Z")
    with store.connect() as connection:
        connection.execute(
            """UPDATE first_letters_discovery_executor_claims
                  SET claim_sha256='tampered' WHERE run_id=?""",
            (handle.run_id,),
        )
    with pytest.raises(ValueError, match="expired executor claim"):
        store.reconcile_expired_first_letters_discovery_evidence_run(
            run_id=handle.run_id,
        )


def test_a3_replacement_process_reconciles_persisted_expired_worker(
    tmp_path, monkeypatch,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)

    import fleet.store as store_module
    monkeypatch.setattr(store_module, "utc_now", lambda: "2999-01-01T00:00:00Z")
    replacement = store_module.FleetStore(
        store.path,
        first_letters_discovery_worker_id="replacement-process-worker",
    )
    reconciled = replacement.reconcile_expired_first_letters_discovery_evidence_run(
        run_id=handle.run_id,
    )
    assert reconciled["state"] == "CONTROL_INCOMPLETE"
    assert reconciled["incomplete_reason"] == "WORKER_LOST_AFTER_RUNNING"


def test_controller_off_and_select_never_claim_or_execute(tmp_path):
    controller_type, _ = _controller_api()
    provider = _Provider()
    off = controller_type(mode="off", store=None, provider=provider)
    result = off.run_item(
        lease_seconds=60,
        reservation_id="unused",
        item_id="unused",
        profile_bytes=b"unused",
    )
    assert result == {
        "schema": "campaignx.first_letters_discovery_controller_result.v1",
        "mode": "off",
        "state": "OFF_UNCHANGED",
        "run_id": None,
        "evidence_set_id": None,
        "canonical_admission": "PROHIBITED",
    }
    assert provider.prepare_calls == provider.execute_calls == 0

    select = controller_type(mode="select", store=None, provider=provider)
    with pytest.raises(RuntimeError, match="TASK9|dormant|DORMANT"):
        select.run_item(
            lease_seconds=60,
            reservation_id="unused",
            item_id="unused",
            profile_bytes=b"unused",
        )
    with pytest.raises(RuntimeError, match="TASK9|dormant|DORMANT"):
        select.promote_discovery_evidence(evidence_set_id="never")
    assert provider.prepare_calls == provider.execute_calls == 0


def test_real_off_baseline_and_alternative_shadow_preserve_every_canonical_row(
    tmp_path,
):
    """Compare cloned databases after the actual controller/worker paths."""

    from fleet.discovery_controller import FirstLettersDiscoveryController
    from test_first_letters_discovery_shadow_bridge import (
        _BridgeProvider,
        _DISCOVERY_SHADOW_TABLE_ALLOWLIST,
        _clone_live_bridge_store,
        _live_bridge_store,
        _sqlite_table_projection,
    )

    template, admission, _ = _live_bridge_store(tmp_path / "template")
    clones = {
        name: _clone_live_bridge_store(
            template, tmp_path / name / "fleet.sqlite"
        )
        for name in ("off", "baseline-shadow", "alternative-shadow")
    }
    template_all = _sqlite_table_projection(template.path)
    template_canonical = _sqlite_table_projection(
        template.path, excluded=_DISCOVERY_SHADOW_TABLE_ALLOWLIST,
    )

    off_provider = _BridgeProvider()
    off = FirstLettersDiscoveryController(
        mode="off", store=clones["off"], provider=off_provider,
    )
    assert off.reserve_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )["state"] == "OFF_UNCHANGED"
    assert off.reserve_alternative_shadow(
        request_id="request-alt",
        budget_admission_sha256=admission["admission_sha256"], arm_id="arm-a",
    )["state"] == "OFF_UNCHANGED"
    assert off.run_job(job_id="off-does-not-read-job", lease_seconds=60)[
        "state"
    ] == "OFF_UNCHANGED"
    assert off_provider.prepare_calls == off_provider.execute_calls == 0
    assert _sqlite_table_projection(clones["off"].path) == template_all

    completed = {}
    for name, reserve in (
        ("baseline-shadow", "baseline"),
        ("alternative-shadow", "alternative"),
    ):
        provider = _BridgeProvider()
        controller = FirstLettersDiscoveryController(
            mode="shadow", store=clones[name], provider=provider,
        )
        if reserve == "baseline":
            branch = controller.reserve_baseline_shadow(
                request_id="request-a",
                budget_admission_sha256=admission["admission_sha256"],
            )
        else:
            branch = controller.reserve_alternative_shadow(
                request_id="request-alt",
                budget_admission_sha256=admission["admission_sha256"],
                arm_id="arm-a",
            )
        result = controller.run_job(
            job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
        )
        assert result["state"] == "COMPLETED"
        assert result["canonical_admission"] == "PROHIBITED"
        assert provider.prepare_calls == provider.execute_calls == 1
        assert _sqlite_table_projection(
            clones[name].path,
            excluded=_DISCOVERY_SHADOW_TABLE_ALLOWLIST,
        ) == template_canonical
        completed[name] = result

    assert completed["baseline-shadow"]["run_id"]
    assert completed["alternative-shadow"]["run_id"]


def test_shadow_controller_retries_only_pre_run_then_uses_public_a3_apis(tmp_path):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    public_store = _PublicApiStore(store)
    provider = _Provider(preparation_failures=1)
    controller = controller_type(
        mode="shadow", store=public_store, provider=provider,
        pre_run_attempts=2, heartbeat_interval_seconds=0.01,
    )
    result = _run(controller, reservation, profile_bytes)
    assert result["state"] == "COMPLETED"
    assert result["mode"] == "shadow"
    assert result["canonical_admission"] == "PROHIBITED"
    assert result["evidence_set_id"]
    assert provider.prepare_calls == 2
    assert provider.execute_calls == 1
    assert public_store.calls[:3] == [
        "claim_first_letters_discovery_job",
        "revalidate_first_letters_discovery_job_claim",
        "start_first_letters_discovery_evidence_run",
    ]
    assert public_store.calls.count(
        "heartbeat_first_letters_discovery_evidence_run"
    ) >= 2
    assert public_store.calls[-1] == (
        "complete_first_letters_discovery_evidence_run"
    )


def test_invalid_heartbeat_configuration_fails_before_preparation_or_claim(
    tmp_path,
):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider()
    controller = controller_type(
        mode="shadow", store=store, provider=provider,
        heartbeat_interval_seconds=60,
    )
    with pytest.raises(ValueError, match="heartbeat interval"):
        _run(controller, reservation, profile_bytes)
    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0] == 0


def test_active_shadow_worker_heartbeats_without_canonical_fleet_mutation(tmp_path):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    with store.connect() as connection:
        before = {
            "tasks": [tuple(row) for row in connection.execute(
                "SELECT * FROM tasks ORDER BY task_id"
            )],
            "attempts": [tuple(row) for row in connection.execute(
                "SELECT * FROM attempts ORDER BY attempt_id"
            )],
        }
    public_store = _PublicApiStore(store)
    provider = _Provider(delay=0.05)
    result = _run(controller_type(
        mode="shadow", store=public_store, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "COMPLETED"
    assert public_store.calls.count(
        "heartbeat_first_letters_discovery_evidence_run"
    ) >= 2
    with store.connect() as connection:
        after = {
            "tasks": [tuple(row) for row in connection.execute(
                "SELECT * FROM tasks ORDER BY task_id"
            )],
            "attempts": [tuple(row) for row in connection.execute(
                "SELECT * FROM attempts ORDER BY attempt_id"
            )],
        }
    assert after == before


def test_post_running_provider_failure_executes_once_and_is_permanent(tmp_path):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider(error=TimeoutError("provider outcome unknown"))
    result = _run(controller_type(
        mode="shadow", store=store, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "CONTROL_INCOMPLETE"
    assert result["incomplete_reason"] == (
        "PROVIDER_RESPONSE_AMBIGUOUS_AFTER_RUNNING"
    )
    assert provider.execute_calls == 1
    assert store.read_first_letters_discovery_evidence_run_status(
        result["run_id"]
    )["state"] == "CONTROL_INCOMPLETE"

    with pytest.raises(ValueError, match="already claimed|already exists"):
        _run(controller_type(
            mode="shadow", store=store, provider=provider,
            heartbeat_interval_seconds=0.01,
        ), reservation, profile_bytes)
    assert provider.execute_calls == 1


def test_initial_post_start_heartbeat_loss_terminalizes_without_execution(
    tmp_path,
):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider()
    wrapped = _InitialHeartbeatResponseLossStore(store)
    result = _run(controller_type(
        mode="shadow", store=wrapped, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "CONTROL_INCOMPLETE"
    assert result["incomplete_reason"] == "ACTIVE_CLAIM_HEARTBEAT_FAILED"
    assert provider.execute_calls == 0
    assert store.read_first_letters_discovery_evidence_run_status(
        result["run_id"]
    )["state"] == "CONTROL_INCOMPLETE"

    with pytest.raises(ValueError, match="already claimed|already exists"):
        _run(controller_type(
            mode="shadow", store=wrapped, provider=provider,
            heartbeat_interval_seconds=0.01,
        ), reservation, profile_bytes)
    assert provider.execute_calls == 0


def test_heartbeat_thread_start_failure_terminalizes_without_execution(
    tmp_path, monkeypatch,
):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider()
    import fleet.discovery_worker as worker_module

    def fail_start(_heartbeat):
        raise RuntimeError("heartbeat thread unavailable")

    monkeypatch.setattr(worker_module._ClaimHeartbeat, "start", fail_start)
    result = _run(controller_type(
        mode="shadow", store=store, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "CONTROL_INCOMPLETE"
    assert result["incomplete_reason"] == "ACTIVE_CLAIM_HEARTBEAT_FAILED"
    assert provider.execute_calls == 0
    assert store.read_first_letters_discovery_evidence_run_status(
        result["run_id"]
    )["state"] == "CONTROL_INCOMPLETE"


def test_committed_start_response_loss_terminalizes_without_execution(tmp_path):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider()
    wrapped = _StartResponseLossStore(store)
    result = _run(controller_type(
        mode="shadow", store=wrapped, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "CONTROL_INCOMPLETE"
    assert result["incomplete_reason"] == (
        "START_RESPONSE_AMBIGUOUS_AFTER_RUNNING"
    )
    assert provider.execute_calls == 0
    assert store.read_first_letters_discovery_evidence_run_status(
        result["run_id"]
    )["state"] == "CONTROL_INCOMPLETE"


def test_terminal_write_response_loss_requires_confirmed_readback(tmp_path):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider(error=TimeoutError("provider outcome unknown"))
    wrapped = _TerminalWriteResponseLossStore(store)
    result = _run(controller_type(
        mode="shadow", store=wrapped, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "CONTROL_INCOMPLETE"
    assert wrapped.calls[-1] == (
        "read_first_letters_discovery_evidence_run_status"
    )
    assert store.read_first_letters_discovery_evidence_run_status(
        result["run_id"]
    )["state"] == "CONTROL_INCOMPLETE"


def test_unconfirmed_terminal_write_raises_until_expired_run_is_reconciled(
    tmp_path, monkeypatch,
):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider(error=TimeoutError("provider outcome unknown"))
    wrapped = _TerminalWriteUnavailableStore(store)
    with pytest.raises(RuntimeError, match="terminal state could not be confirmed"):
        _run(controller_type(
            mode="shadow", store=wrapped, provider=provider,
            heartbeat_interval_seconds=0.01,
        ), reservation, profile_bytes)
    assert provider.execute_calls == 1
    with store.connect() as connection:
        run_id = connection.execute(
            "SELECT run_id FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0]
    assert store.read_first_letters_discovery_evidence_run_status(
        run_id
    )["state"] == "RUNNING"

    import fleet.store as store_module
    monkeypatch.setattr(store_module, "utc_now", lambda: "2999-01-01T00:00:00Z")
    reconciled = store.reconcile_expired_first_letters_discovery_evidence_run(
        run_id=run_id,
    )
    assert reconciled["state"] == "CONTROL_INCOMPLETE"
    assert reconciled["incomplete_reason"] == "WORKER_LOST_AFTER_RUNNING"
    assert store.read_first_letters_discovery_evidence_run_status(
        run_id
    )["state"] == "CONTROL_INCOMPLETE"

    with pytest.raises(ValueError, match="already claimed|already exists"):
        _run(controller_type(
            mode="shadow", store=store, provider=provider,
            heartbeat_interval_seconds=0.01,
        ), reservation, profile_bytes)
    assert provider.execute_calls == 1


def test_committed_completion_response_loss_recovers_exact_run_readback(tmp_path):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider()
    wrapped = _CommittedResponseLossStore(store)
    result = _run(controller_type(
        mode="shadow", store=wrapped, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "COMPLETED"
    assert result["recovered_after_response_loss"] is True
    assert provider.execute_calls == 1
    assert wrapped.calls[-1] == "read_first_letters_discovery_evidence_run"


def test_ambiguous_post_running_readback_is_terminalized_without_reexecution(
    tmp_path,
):
    controller_type, _ = _controller_api()
    store, profile_bytes, reservation = _store(tmp_path)
    provider = _Provider()
    wrapped = _AmbiguousCompletionStore(store)
    result = _run(controller_type(
        mode="shadow", store=wrapped, provider=provider,
        heartbeat_interval_seconds=0.01,
    ), reservation, profile_bytes)
    assert result["state"] == "CONTROL_INCOMPLETE"
    assert result["incomplete_reason"] == (
        "COMPLETION_READBACK_AMBIGUOUS_AFTER_RUNNING"
    )
    assert provider.execute_calls == 1
    assert wrapped.calls.count(
        "complete_first_letters_discovery_evidence_run"
    ) == 1
    assert store.read_first_letters_discovery_evidence_run_status(
        result["run_id"]
    )["state"] == "CONTROL_INCOMPLETE"
