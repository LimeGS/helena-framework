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


def test_an_unregistered_host_is_told_where_to_register() -> None:
    """This ran every minute on a fresh install and named no way forward.

    "no host row named 'work-3'; register it to see this" is true and useless:
    the reader has the panel open in front of them and no reason to guess that
    Configuration holds a Hosts tab. Saying where turns a log line that repeats
    forever into one step.

    The location is asserted against the panel, not spelled twice: a message
    naming a tab that has been moved is worse than the one that named nothing.
    """
    root = Path(__file__).resolve().parents[1]
    message = (root / "framework/contracts/host_report.py").read_text()
    assert "no host row named" in message, "the message is gone; drop this test"

    said = re.search(r'add it under\s*"?\s*\n?\s*f?"([^"]*Hosts[^"]*)', message)
    assert said, "the message no longer says where a host is registered"

    configuration = (root / "panel/web/src/routes/Configuration.tsx").read_text()
    assert 'tab === "hosts"' in configuration, (
        "the message sends people to a Hosts tab that Configuration no longer "
        "renders")
    hosts_page = (root / "panel/web/src/routes/Hosts.tsx").read_text()
    assert 'title="Hosts"' in hosts_page, "the page it names is not called Hosts"
