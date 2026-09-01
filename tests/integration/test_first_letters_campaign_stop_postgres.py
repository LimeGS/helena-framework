"""Real PostgreSQL serialization for the candidate-starvation gate.

Run with HELENA_TEST_DSN pointing at a throwaway PostgreSQL database.  The
test uses unique mission/sample ids and leaves immutable evidence behind.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "tests"))

from fleet import campaign_decision  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from test_first_letters_campaign_decision import (  # noqa: E402
    budget_admission,
    budget_task,
)
from test_first_letters_campaign_stop import (  # noqa: E402
    no_seed_result,
    resized_admission,
    resume_authorization,
)


DSN = os.environ.get("HELENA_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN,
    reason=(
        "HELENA_TEST_DSN is not set; set it to a throwaway PostgreSQL to run "
        "the real campaign pause/create serialization test"
    ),
)


def _snapshot(store: PostgresFleetStore, sample_id: str, suffix: str) -> str:
    return store.register_snapshot({
        "sample_id": sample_id,
        "ct_uri": f"fixture://ct/{suffix}",
        "ct_sha256": "2" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "3" * 64,
        "shape_xyz": [32768, 32768, 32768],
        "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
        "m7_threshold": 0.2,
        "source_content_lock": {"schema": "fixture-source-lock"},
    })


def _scoped_authority(
    template: dict, *, mission_id: str, sample_id: str,
    source_snapshot_id: str, receipt_sha256: str,
) -> dict:
    authority = copy.deepcopy(template)
    authority.update({
        "mission_id": mission_id,
        "sample_id": sample_id,
        "receipt_sha256": receipt_sha256,
    })
    authority["execution_bindings"].update({
        "mission_id": mission_id,
        "sample_id": sample_id,
        "source_snapshot_id": source_snapshot_id,
    })
    authority["admission_sha256"] = content_sha256({
        key: value for key, value in authority.items()
        if key != "admission_sha256"
    })
    return authority


def _mission_claim_order(store: PostgresFleetStore, mission_id: str) -> list[str]:
    """The order `claim` would take if this mission owned the whole queue.

    The database is shared: a plain `store.claim(worker, lease)` takes the
    highest-priority pending task in the fleet, which in this file is not
    always one of the caller's -- the pause race in the first test can leave a
    pending task behind.  Claiming by ID keeps each test inside its own
    mission without changing the order the scheduler would have used.
    """
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT task_id FROM segment_tasks
                    WHERE mission_id=%s AND state='PENDING'
                    ORDER BY priority DESC,task_id""",
                (mission_id,),
            )
            return [str(row["task_id"]) for row in cursor.fetchall()]


def _tasks(authority: dict, source_snapshot_id: str) -> list[dict]:
    rows = [
        budget_task(source_snapshot_id, cell_id, authority)
        for cell_id in authority["prefix_cell_ids"]
    ]
    for row in rows:
        row.update({
            "mission_id": authority["mission_id"],
            "sample_id": authority["sample_id"],
            "priority": 1000.0,
        })
    return campaign_decision.bind_campaign_budget_to_tasks(rows, authority)


def test_pause_and_competing_task_creation_have_a_postgres_boundary(
    tmp_path: Path,
) -> None:
    store = PostgresFleetStore(DSN)
    store.initialize()
    nonce = uuid.uuid4().hex[:12]
    mission_id = f"campaign-stop-pg-{nonce}"
    first_sample = f"TESTA{nonce}"
    second_sample = f"TESTB{nonce}"
    first_source = _snapshot(store, first_sample, f"a-{nonce}")
    second_source = _snapshot(store, second_sample, f"b-{nonce}")
    template = budget_admission(tmp_path / "mission", first_source)
    first = _scoped_authority(
        template,
        mission_id=mission_id,
        sample_id=first_sample,
        source_snapshot_id=first_source,
        receipt_sha256=("1" + nonce).ljust(64, "1")[:64],
    )
    second = _scoped_authority(
        template,
        mission_id=mission_id,
        sample_id=second_sample,
        source_snapshot_id=second_source,
        receipt_sha256=("2" + nonce).ljust(64, "2")[:64],
    )
    first_tasks = _tasks(first, first_source)
    second_tasks = _tasks(second, second_source)
    store.register_campaign_budget_admission(first)
    store.register_campaign_budget_admission(second)
    assert store.create_tasks(first_tasks) == (8, 8)

    claim_order = _mission_claim_order(store, mission_id)
    assert len(claim_order) == 8
    for index, task_id in enumerate(claim_order[:7]):
        claim = store.claim(
            f"campaign-stop-pg-{nonce}-{index}", 60, task_id=task_id,
        )
        assert claim is not None and claim["mission_id"] == mission_id
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 0),
        )
    eighth = store.claim(
        f"campaign-stop-pg-{nonce}-eighth", 60, task_id=claim_order[7],
    )
    assert eighth is not None and eighth["mission_id"] == mission_id
    boundary = Barrier(2)

    def terminalize() -> str:
        boundary.wait()
        store.mark_terminal(
            eighth["task_id"], eighth["attempt_id"],
            eighth["lease_token"], "NO_SEED", no_seed_result(eighth, 2),
        )
        return "terminal"

    def create() -> str:
        boundary.wait()
        try:
            assert store.create_tasks(second_tasks[:1]) == (1, 1)
        except ValueError as error:
            assert "campaign decision blocks" in str(error)
            return "blocked"
        return "inserted-before-terminal"

    with ThreadPoolExecutor(max_workers=2) as executor:
        terminal_future = executor.submit(terminalize)
        create_future = executor.submit(create)
        assert terminal_future.result() == "terminal"
        creation_result = create_future.result()

    decisions = store.campaign_decisions(
        mission_id=mission_id,
        policy_version=first["execution_bindings"]["policy_version"],
    )
    assert decisions[-1]["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert decisions[-1]["governing_admission_sha256s"] == [
        first["admission_sha256"]]
    assert creation_result in {"blocked", "inserted-before-terminal"}


def test_two_authorized_successors_race_without_forking_the_postgres_chain(
    tmp_path: Path,
) -> None:
    store = PostgresFleetStore(DSN)
    store.initialize()
    nonce = uuid.uuid4().hex[:12]
    mission_id = f"campaign-resume-pg-{nonce}"
    first_sample = f"TESTP{nonce}"
    first_source = _snapshot(store, first_sample, f"p-{nonce}")
    template = budget_admission(tmp_path / "mission", first_source)
    first = _scoped_authority(
        template,
        mission_id=mission_id,
        sample_id=first_sample,
        source_snapshot_id=first_source,
        receipt_sha256=("3" + nonce).ljust(64, "3")[:64],
    )
    first_tasks = _tasks(first, first_source)
    for row in first_tasks:
        row["priority"] = 1001.0
    store.register_campaign_budget_admission(first)
    assert store.create_tasks(first_tasks) == (8, 8)
    claim_order = _mission_claim_order(store, mission_id)
    assert len(claim_order) == 8
    for index, task_id in enumerate(claim_order):
        claim = store.claim(
            f"campaign-resume-pg-{nonce}-{index}", 60, task_id=task_id,
        )
        assert claim is not None and claim["mission_id"] == mission_id
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 2 if index == 7 else 0),
        )
    pause = store.campaign_decisions(
        mission_id=mission_id,
        policy_version=first["execution_bindings"]["policy_version"],
    )[-1]

    successors: list[tuple[dict, dict]] = []
    for marker, provider in (("4", "vc3d-mcp-v2a"), ("5", "vc3d-mcp-v2b")):
        sample = f"TEST{marker}{nonce}"
        source = _snapshot(store, sample, f"{marker}-{nonce}")
        authority = _scoped_authority(
            template,
            mission_id=mission_id,
            sample_id=sample,
            source_snapshot_id=source,
            receipt_sha256=(marker + nonce).ljust(64, marker)[:64],
        )
        authority["execution_bindings"].update({
            "policy_version": f"search-v2-{marker}",
            "provider": provider,
        })
        authority["admission_sha256"] = content_sha256({
            key: value for key, value in authority.items()
            if key != "admission_sha256"
        })
        authorization = resume_authorization(
            first, authority, pause, field="discovery_provider")
        store.register_campaign_resume_principal_attestation(
            authorization, authenticated_principal="campaign-owner")
        successors.append((authority, authorization))

    boundary = Barrier(2)

    def register(candidate: tuple[dict, dict]) -> tuple[str, str]:
        authority, authorization = candidate
        boundary.wait()
        try:
            store.register_campaign_budget_admission(
                authority, resume_authorization=authorization)
        except ValueError as error:
            return "conflict", str(error)
        return "winner", authority["admission_sha256"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(register, successors))
    assert sorted(status for status, _detail in results) == [
        "conflict", "winner"]
    winner_digest = next(
        detail for status, detail in results if status == "winner")
    winner, winner_authorization = next(
        candidate for candidate in successors
        if candidate[0]["admission_sha256"] == winner_digest)
    assert store.register_campaign_budget_admission(
        winner, resume_authorization=winner_authorization) == winner

    active = store.campaign_active_decision(mission_id=mission_id)
    assert active is not None
    assert active["policy_chain"] == [
        first["execution_bindings"]["policy_version"],
        winner["execution_bindings"]["policy_version"],
    ]
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT COUNT(*) AS count
                     FROM segment_campaign_resume_authorizations
                    WHERE mission_id=%s""",
                (mission_id,),
            )
            assert cursor.fetchone()["count"] == 1


def test_postgres_resume_rederives_pause_inside_the_mission_advisory_lock(
    tmp_path: Path, monkeypatch,
) -> None:
    store = PostgresFleetStore(DSN)
    store.initialize()
    nonce = uuid.uuid4().hex[:12]
    mission_id = f"campaign-provenance-pg-{nonce}"
    first_sample = f"TESTT{nonce}"
    second_sample = f"TESTD{nonce}"
    first_source = _snapshot(store, first_sample, f"t-{nonce}")
    second_source = _snapshot(store, second_sample, f"d-{nonce}")
    template = budget_admission(tmp_path / "mission", first_source)
    trigger_authority = resized_admission(_scoped_authority(
        template,
        mission_id=mission_id,
        sample_id=first_sample,
        source_snapshot_id=first_source,
        receipt_sha256=("6" + nonce).ljust(64, "6")[:64],
    ), 7)
    denominator_authority = resized_admission(_scoped_authority(
        template,
        mission_id=mission_id,
        sample_id=second_sample,
        source_snapshot_id=second_source,
        receipt_sha256=("7" + nonce).ljust(64, "7")[:64],
    ), 1)
    store.register_campaign_budget_admission(trigger_authority)
    store.register_campaign_budget_admission(denominator_authority)
    assert store.create_tasks(_tasks(trigger_authority, first_source)) == (7, 7)
    assert store.create_tasks(_tasks(denominator_authority, second_source)) == (1, 1)
    claim_order = _mission_claim_order(store, mission_id)
    assert len(claim_order) == 8
    for index, task_id in enumerate(claim_order):
        claim = store.claim(
            f"campaign-provenance-pg-{nonce}-{index}", 60, task_id=task_id,
        )
        assert claim is not None and claim["mission_id"] == mission_id
        raw_m7 = 2 if claim["sample_id"] == second_sample else 0
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, raw_m7),
        )
    pause = store.campaign_decisions(
        mission_id=mission_id,
        policy_version=trigger_authority["execution_bindings"][
            "policy_version"],
    )[-1]

    forged_pause = copy.deepcopy(pause)
    for bound_attempt in forged_pause["scientific_terminal_attempts"]:
        if bound_attempt["attempt_id"] in forged_pause["trigger_attempt_ids"]:
            bound_attempt.update({
                "sample_id": second_sample,
                "admission_sha256": denominator_authority[
                    "admission_sha256"],
                "budget_receipt_sha256": denominator_authority[
                    "receipt_sha256"],
            })
    forged_pause["trigger_governing_admission_sha256s"] = [
        denominator_authority["admission_sha256"]]
    forged_pause["governing_admission_sha256s"] = [
        denominator_authority["admission_sha256"]]
    forged_pause["receipt_sha256"] = content_sha256({
        key: value for key, value in forged_pause.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE segment_campaign_decisions
                      SET receipt_sha256=%s,receipt=%s::jsonb
                    WHERE receipt_sha256=%s""",
                (
                    forged_pause["receipt_sha256"],
                    json.dumps(forged_pause, sort_keys=True,
                               separators=(",", ":")),
                    pause["receipt_sha256"],
                ),
            )

    successor = copy.deepcopy(denominator_authority)
    successor["receipt_sha256"] = ("8" + nonce).ljust(64, "8")[:64]
    successor["execution_bindings"].update({
        "policy_version": f"search-v2-{nonce}",
        "provider": "vc3d-mcp-v2",
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key != "admission_sha256"
    })
    authorization = resume_authorization(
        denominator_authority, successor, forged_pause,
        field="discovery_provider",
    )
    store.register_campaign_resume_principal_attestation(
        authorization, authenticated_principal="campaign-owner")
    original_inputs = store._campaign_decision_inputs
    lock_observations: list[bool] = []

    def observed_inputs(cursor, **scope):
        cursor.execute(
            """SELECT COUNT(*) AS count FROM pg_locks
                WHERE pid=pg_backend_pid() AND locktype='advisory'
                  AND mode='ExclusiveLock' AND granted"""
        )
        # register_campaign_budget_admission owns both its mission and exact
        # admission xact locks before authoritative provenance is read.
        lock_observations.append(cursor.fetchone()["count"] >= 2)
        return original_inputs(cursor, **scope)

    monkeypatch.setattr(store, "_campaign_decision_inputs", observed_inputs)
    with pytest.raises(ValueError, match="authoritative persisted pause"):
        store.register_campaign_budget_admission(
            successor, resume_authorization=authorization)
    assert lock_observations == [True]
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT 1 FROM segment_campaign_budget_admissions
                    WHERE admission_sha256=%s""",
                (successor["admission_sha256"],),
            )
            assert cursor.fetchone() is None
            cursor.execute(
                """SELECT COUNT(*) AS count
                     FROM segment_campaign_budget_admissions
                    WHERE mission_id=%s""",
                (mission_id,),
            )
            assert cursor.fetchone()["count"] == 2
