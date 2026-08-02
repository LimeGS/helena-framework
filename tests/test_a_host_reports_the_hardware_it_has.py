"""The Hosts page reported no hardware on a machine with two cards.

A host runs more than one worker and they do not see the same hardware: the ink
worker may be given a card while the P2/P3 runner is not, and both record the
host's state under the same host_id -- so whichever wrote last decided what the
panel showed. The one without a card wrote `gpus: []` over two cards, minutes
after the machine had rendered a layer stack on one of them.
"""

from __future__ import annotations

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
