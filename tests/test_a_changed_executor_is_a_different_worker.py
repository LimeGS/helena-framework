"""A legitimate change to the executor must be deployable.

The discovery executor registry is immutable per worker_id: the row is inserted
with ON CONFLICT DO NOTHING and then read back, and a registration whose digest
differs from the stored one raises DISCOVERY_EXECUTOR_REGISTRATION_CONFLICT.
That control is right -- a worker must not be able to swap its executor and keep
its identity.

But the worker id was derived from hostname, pid and interpreter path, and in a
container those are `gpu-1`, `1` and `/usr/local/bin/python3` on every deploy.
So the id was stable while the executor was not, and the registry had no update
path in either store. The consequence, seen rather than imagined: adding a retry
to `OmeZarrCtSupportSampler.sample` changed `executor_sha256` from 8f42f833 to
768059d8, the init container raised, the panel never started, and the deployment
served nothing until someone looked.

The fix keeps the control exactly as strict and stops it firing on the honest
case: the executor's digest is part of the worker's identity. A different
executor is a different worker -- new row, old row untouched, nothing mutated,
nothing superseded. A silent swap is still impossible, because the id moves with
the code and the old registration stays exactly as it was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import discovery_executor as de  # noqa: E402


class _Sampler:
    """A sampler whose source can be told apart from the real one."""

    def sample(self, *args, **kwargs):
        return {}


class _OtherSampler:
    def sample(self, *args, **kwargs):
        return {"different source entirely": True}


def test_two_executors_are_two_workers(monkeypatch) -> None:
    monkeypatch.delenv("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID", raising=False)

    one = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())
    other = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_OtherSampler())

    assert (de.runtime_discovery_executor_sha256(one)
            != de.runtime_discovery_executor_sha256(other)), (
        "the two fixtures hash the same; this test cannot prove anything")

    assert (de.production_discovery_worker_id(executor=one)
            != de.production_discovery_worker_id(executor=other)), (
        "a changed executor kept its worker id, so its registration collides "
        "with the stored one and the deployment cannot start"
    )


def test_the_same_executor_keeps_its_worker(monkeypatch) -> None:
    """The other half. If the id moved on its own, every restart would insert a
    new row and the immutability check would never mean anything."""
    monkeypatch.delenv("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID", raising=False)

    one = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())
    again = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())

    assert (de.production_discovery_worker_id(executor=one)
            == de.production_discovery_worker_id(executor=again))


def test_a_pinned_worker_id_still_wins(monkeypatch) -> None:
    """An operator who names the worker gets the worker they named."""
    monkeypatch.setenv("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID", "named-by-hand")

    executor = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())

    assert de.production_discovery_worker_id(executor=executor) == "named-by-hand"


def test_the_registration_agrees_with_the_id_it_was_built_for(monkeypatch) -> None:
    """The id and the registration have to be derived from the same executor, or
    the identity says one thing and the row says another."""
    monkeypatch.delenv("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID", raising=False)

    executor = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())
    worker_id = de.production_discovery_worker_id(executor=executor)
    registration = de.production_discovery_executor_registration(
        executor, worker_id=worker_id)

    assert registration["worker_id"] == worker_id
    assert registration["executor_sha256"] == de.runtime_discovery_executor_sha256(
        executor)


def test_a_changed_executor_registers_without_conflict(tmp_path, monkeypatch) -> None:
    """End to end against a real store: the situation that took the panel down.

    A registry already holding the old executor's row must accept the new one,
    and must still refuse a registration that reuses an id with different
    contents -- the control this exists to preserve."""
    monkeypatch.delenv("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID", raising=False)

    from fleet.store import FleetStore

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()

    old = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())
    old_id = de.production_discovery_worker_id(executor=old)
    store.register_first_letters_discovery_executor(
        de.production_discovery_executor_registration(old, worker_id=old_id))

    new = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_OtherSampler())
    new_id = de.production_discovery_worker_id(executor=new)
    store.register_first_letters_discovery_executor(
        de.production_discovery_executor_registration(new, worker_id=new_id))

    # The old row is still exactly what it was: nothing was updated to make room.
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT worker_id,executor_sha256 FROM "
            "first_letters_discovery_executor_registry ORDER BY worker_id"
        ).fetchall()
    stored = {row["worker_id"]: row["executor_sha256"] for row in rows}
    assert stored[old_id] == de.runtime_discovery_executor_sha256(old)
    assert stored[new_id] == de.runtime_discovery_executor_sha256(new)


def test_the_immutability_control_still_refuses_a_swap(tmp_path, monkeypatch) -> None:
    """The point of the registry, unchanged: one id cannot come back carrying a
    different executor."""
    monkeypatch.delenv("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID", raising=False)

    from fleet.store import FleetStore

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()

    old = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_Sampler())
    pinned = "pinned-worker"
    store.register_first_letters_discovery_executor(
        de.production_discovery_executor_registration(old, worker_id=pinned))

    new = de.ProductionFirstLettersDiscoveryExecutor(ct_sampler=_OtherSampler())
    with pytest.raises(ValueError, match="DISCOVERY_EXECUTOR_REGISTRATION_CONFLICT"):
        store.register_first_letters_discovery_executor(
            de.production_discovery_executor_registration(new, worker_id=pinned))
