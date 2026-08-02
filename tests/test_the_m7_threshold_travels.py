"""The threshold the m7 prediction was published at reaches the seed screen.

It decides which voxels of the prediction count as sheet, so it decides which
seeds exist at all -- and it sat in the frozen catalogue reaching nothing.
seed_candidates carried `DEFAULT_THRESHOLD = 0.2` as a constant, with a comment
asking for exactly this. A catalogue publishing 0.3 would have been screened at
0.2, in silence, under an identical task identity.

The chain is catalogue -> snapshot -> task.candidate_discovery -> the MCP request
-> find_candidates, and each link is checked here because the value is useless if
any one of them drops it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (ROOT / "framework/stages/01-segmentation/fleet/generator.py").read_text()
WORKER = (ROOT / "framework/stages/01-segmentation/fleet/worker.py").read_text()
SERVER = (ROOT / "framework/stages/01-segmentation/mcp/server.py").read_text()
CANDIDATES = (ROOT / "framework/stages/01-segmentation/mcp/seed_candidates.py").read_text()
CATALOG = ROOT / "workspace/catalog/eligible_volumes.json"


def test_the_catalogue_still_publishes_one() -> None:
    """If it stops, the rest of this is plumbing for a field nobody sets."""
    entries = json.loads(CATALOG.read_text())
    entries = entries.get("entries", entries if isinstance(entries, list) else [])
    published = [e.get("surface_prediction_threshold") for e in entries]
    assert any(v is not None for v in published), (
        "no catalogue entry publishes surface_prediction_threshold any more"
    )


def test_the_snapshot_records_it() -> None:
    snapshot = GENERATOR.split("def bootstrap_sources")[1]
    snapshot = snapshot[: snapshot.index("\ndef ")]
    assert '"m7_threshold"' in snapshot
    assert "surface_prediction_threshold" in snapshot


def test_the_task_carries_it_where_the_seed_provider_looks() -> None:
    """candidate_discovery, which is the dict the provider reads."""
    # Only the builders that screen candidates. There are two
    # candidate_discovery blocks and the second names the manual provider: a point
    # somebody supplied by hand is not searched for, so no threshold applies to
    # it. Counting both was this test's first version, and it was the assertion
    # that was wrong rather than the code.
    searching = GENERATOR.count('"provider": "vc3d-mcp"')
    carried = GENERATOR.count('**({"m7_threshold": snapshot["m7_threshold"]}')
    assert searching >= 1
    assert carried == searching, (
        f"{searching} builders search the prediction and only {carried} carry the "
        "threshold, so some of them screen at the provider's constant instead"
    )


def test_the_worker_sends_it_and_the_server_uses_it() -> None:
    assert 'discovery.get("m7_threshold")' in WORKER, (
        "the MCP request drops the threshold the task carries"
    )
    assert 'arguments.get("threshold")' in SERVER, (
        "the server ignores the threshold the worker sends"
    )
    # And the constant stays as the fallback, because tasks queued before this
    # carry no threshold at all.
    assert "DEFAULT_THRESHOLD = 0.2" in CANDIDATES


def test_a_changed_threshold_is_refused_rather_than_ignored() -> None:
    """A snapshot is immutable, so a changed catalogue would grow at the old value.

    And the identity is deliberately not touched: source_snapshot_id is hashed
    from it and is part of task identity, so adding a field would give every
    existing source a new id and re-queue the entire backlog as new work.
    """
    snapshot = GENERATOR.split("def bootstrap_sources")[1]
    snapshot = snapshot[: snapshot.index("\ndef ")]
    assert "is a different set of seeds" in snapshot, (
        "nothing refuses a catalogue whose threshold moved under a frozen source"
    )
    assert snapshot.index("m7_threshold") < snapshot.index("register_snapshot"), (
        "the check runs after the source is registered, which is too late"
    )

    identity = GENERATOR  # the tuple lives in the stores, not here
    for store in ("postgres_store.py", "store.py"):
        text = (ROOT / "framework/stages/01-segmentation/fleet" / store).read_text()
        block = text[text.index("def register_snapshot"):]
        block = block[: block.index("\n    def ")]
        assert "m7_threshold" not in block, (
            f"{store} hashes the threshold into source_snapshot_id, which changes "
            "the id of every existing source and re-queues the backlog"
        )
    assert identity  # keeps the name meaningful to a reader
