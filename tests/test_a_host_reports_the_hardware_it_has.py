"""The Hosts page reported no hardware on a machine with two cards.

A host runs more than one worker and they do not see the same hardware: the ink
worker may be given a card while the P2/P3 runner is not, and both record the
host's state under the same host_id -- so whichever wrote last decided what the
panel showed. The one without a card wrote `gpus: []` over two cards, minutes
after the machine had rendered a layer stack on one of them.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import GPU_OBSERVATION_TTL_SECONDS, merge_gpu_observations  # noqa: E402

FIRST = {"uuid": "GPU-225f9a42", "name": "NVIDIA GeForce GTX 1660", "index": 0,
         "total_mb": 6144}
SECOND = {"uuid": "GPU-7c1180ab", "name": "NVIDIA GeForce GTX 1660", "index": 1,
          "total_mb": 6144}


def test_a_worker_that_sees_no_card_does_not_erase_one(monkeypatch):
    now = time.time()
    stored = {"gpus": [{**FIRST, "seen_at": now - 30}]}
    merged = merge_gpu_observations(stored, {"gpus": [], "hostname": "worker-host"}, now=now)
    assert [card["uuid"] for card in merged] == ["GPU-225f9a42"]


def test_two_workers_pinned_to_different_cards_make_two_cards():
    """Each container calls its own card index 0, so merging on index would
    collapse a two-GPU machine into one."""
    now = time.time()
    stored = merge_gpu_observations({}, {"gpus": [{**FIRST, "index": 0}],
                                         "hostname": "worker-host"}, now=now)
    merged = merge_gpu_observations({"gpus": stored},
                                    {"gpus": [{**SECOND, "index": 0}],
                                     "hostname": "worker-host"}, now=now)
    assert len(merged) == 2
    assert {card["uuid"] for card in merged} == {"GPU-225f9a42", "GPU-7c1180ab"}


def test_a_card_that_is_gone_ages_out():
    """Never forgetting is its own lie: a machine whose card was pulled would
    report it forever."""
    now = time.time()
    stale = {"gpus": [{**FIRST, "seen_at": now - GPU_OBSERVATION_TTL_SECONDS - 1}]}
    assert merge_gpu_observations(stale, {"gpus": []}, now=now) == []


def test_a_fresh_observation_replaces_the_reading_it_repeats():
    now = time.time()
    stored = {"gpus": [{**FIRST, "used_mb": 10, "seen_at": now - 60}]}
    merged = merge_gpu_observations(stored,
                                    {"gpus": [{**FIRST, "used_mb": 2632}],
                                     "hostname": "worker-host"}, now=now)
    assert len(merged) == 1
    assert merged[0]["used_mb"] == 2632
    assert merged[0]["seen_by"] == "worker-host"


def test_an_unregistered_host_registers_itself_by_reporting(monkeypatch) -> None:
    """This ran every minute on a fresh install: "no host row named 'ubuntu';
    add it under Configuration -> Hosts", while two workers on that machine
    took every job and the mission page counted no hardware at all.

    A host that reports exists. Its first report writes the row, with no ssh
    target -- that is what registering by hand adds, and only provisioning
    needs it -- and says so in the row's notes, naming where the target goes.
    """
    import sys
    import types

    from framework.contracts import host_report

    executed: list[tuple[str, tuple]] = []

    class Cursor:
        rowcount = 0

        def execute(self, sql, params=()):
            executed.append((" ".join(sql.split()), params))

        def fetchone(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class Connection(Cursor):
        def cursor(self):
            return Cursor()

    fake = types.ModuleType("psycopg2")
    fake.connect = lambda *_, **__: Connection()
    monkeypatch.setitem(sys.modules, "psycopg2", fake)
    monkeypatch.setattr(host_report, "host_state",
                        lambda _disk: {"hostname": "ubuntu", "cores": 15,
                                       "gpus": [], "ram_free_gb": 40.0,
                                       "ram_total_gb": 47.0})

    said = host_report.report_once("postgresql://x/y", "ubuntu", None)

    assert "registered 'ubuntu'" in said
    inserts = [(sql, params) for sql, params in executed
               if sql.startswith("INSERT INTO ink_hosts")]
    assert len(inserts) == 1, executed
    sql, params = inserts[0]
    assert params[0] == "ubuntu"
    assert "ssh_target" in sql and "''" in sql, "the row needs an ssh_target column value"
    assert "ON CONFLICT (host_id)" in sql, "two reporters racing must not fail one"
    assert "Configuration -> Hosts" in params[1], (
        "the note has to say where the ssh target is added")

    # The ink worker's heartbeat writes the same table and had the same gap.
    root = Path(__file__).resolve().parents[1]
    store = (root / "framework/stages/03-ink/fleet/job_store.py").read_text()
    body = store[store.index("def record_host_state("):store.index("def hosts(")]
    assert "INSERT INTO ink_hosts" in body and "ON CONFLICT (host_id)" in body, (
        "the heartbeat still only updates a row somebody else has to create")
    configuration = (root / "panel/web/src/routes/Configuration.tsx").read_text()
    assert 'tab === "hosts"' in configuration, (
        "the note sends people to a Hosts tab that Configuration no longer renders")
