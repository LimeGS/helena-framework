"""Concurrent finalization against the real PostgreSQL control plane.

The SQLite contention test proves the invariant on a single file. The question
that gates reordering `finalize_surface` is whether it survives real
`SELECT ... FOR UPDATE SKIP LOCKED` semantics with two workers racing to
register the same physical surface.

Run with SEGMENT_FLEET_DATABASE_URL pointing at a throwaway database.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "tests"))

from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from test_segment_search_fleet import source, task, worker  # noqa: E402

URL = os.environ.get("SEGMENT_FLEET_DATABASE_URL", "")


def _fresh_store() -> PostgresFleetStore:
    """A clean schema per test: this database exists only for these runs."""
    store = PostgresFleetStore(URL, identity="postgres-env://SEGMENT_FLEET_DATABASE_URL")
    with store.connect() as connection:  # noqa: SLF001 - test-only reset
        with connection.cursor() as cursor:
            cursor.execute("drop schema public cascade; create schema public;")
        connection.commit()
    store.initialize()
    return store


@pytest.mark.skipif(not URL, reason="SEGMENT_FLEET_DATABASE_URL is not set")
def test_concurrent_finalizers_register_only_one_surface_on_postgres(
    tmp_path: Path,
) -> None:
    """Two workers, two tasks, one physical surface: exactly one registration.

    Mirrors tests/test_segment_search_fleet.py::
    test_concurrent_finalizers_register_only_one_surface, but on the real
    control plane, so the deduplication rides genuine row locks rather than
    SQLite's single-writer serialization.
    """

    store = _fresh_store()
    source_id = source(store)
    store.create_tasks([task(source_id, "cell-a"), task(source_id, "cell-b")])

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(
                lambda name: worker(store, tmp_path, name).run_one(),
                ("worker-a", "worker-b"),
            )
        )

    statuses = sorted(receipt["status"] for receipt in receipts if receipt)
    assert statuses == ["DUPLICATE_SURFACE", "QC_PENDING"], statuses
    status = store.status()
    assert status["surfaces"] == 1, status
    assert status["qc_jobs"] == 1, status


@pytest.mark.skipif(not URL, reason="SEGMENT_FLEET_DATABASE_URL is not set")
def test_no_artifact_set_is_left_pointing_only_at_staging(tmp_path: Path) -> None:
    """After a successful finalization no row references only the staging URI.

    `s3_lifecycle.json` expires `segment-fleet-v1/staging/` after 14 days.
    `finalize_surface` registers the staging URI and only then promotes, so a
    worker interrupted between those calls leaves the database pointing at an
    object that will be deleted. This pins the post-condition of the happy
    path, which is the guard a reordering must preserve.
    """

    store = _fresh_store()
    source_id = source(store)
    store.create_tasks([task(source_id, "cell-a")])
    receipt = worker(store, tmp_path, "worker-a").run_one()
    assert receipt is not None and receipt["status"] == "QC_PENDING", receipt

    with store.connect() as connection:  # noqa: SLF001 - test-only inspection
        with connection.cursor() as cursor:
            cursor.execute(
                "select artifact_uri from segment_surfaces where state is not null"
            )
            surface_uris = [dict(row)["artifact_uri"] for row in cursor.fetchall()]

    assert surface_uris, "a promoted surface must exist"
    for uri in surface_uris:
        assert "/staging/" not in str(uri), (
            f"a promoted surface still points into the expiring staging prefix: {uri}"
        )


@pytest.mark.skipif(not URL, reason="SEGMENT_FLEET_DATABASE_URL is not set")
def test_a_stale_lease_cannot_register_an_artifact_set(tmp_path: Path) -> None:
    """Why promotion must NOT be moved before registration.

    `s3_lifecycle.json` expires the staging prefix after 14 days, and
    `finalize_surface` registers before promoting, so an interrupted worker
    leaves a row pointing at an object that will be deleted. The obvious remedy
    -- promote first -- is wrong: `add_artifact_set` is also the lease gate, so
    promoting first would publish an artifact grown under a lease another
    worker had already taken over.

    This pins the gate. The residual exposure is therefore one lost grow
    attempt, recoverable by retry, not lost scientific evidence: nothing is
    promoted, so no surface row is ever created.
    """

    store = _fresh_store()
    source_id = source(store)
    store.create_tasks([task(source_id, "cell-a")])

    claim = store.claim("worker-a", 60)
    assert claim is not None
    manifest = {"files": {}, "artifact_sha256": "0" * 64}

    with pytest.raises(RuntimeError, match="stale lease|no longer owns the task"):
        store.add_artifact_set(
            claim["task_id"], claim["attempt_id"], "not-the-real-token", manifest, "s3://b/staging/x"
        )
