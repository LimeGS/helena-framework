"""Fourteen checkpoints, not one, and each of them nameable on its own.

`scrollprize/ink_9um` publishes a training run, not a model: two seeds by seven
steps. A frozen profile names exactly one of them -- seed 42 at step 75000 --
which is the right thing for a profile to do and the wrong thing for the
platform to know. The other thirteen were not missing and not installed; they
were absent from the question, and comparing step 10000 against step 75000, or
seed 42 against seed 43, is most of the reason to have a training run at all.

The manifest is the second declarer. A profile says how a lane runs; the
manifest says what exists to be fetched, with the digest upstream published for
each file. Keeping the two apart is what lets the models page say "upstream has
fourteen of these and you have one" rather than quietly equating them.

Every digest in it was read from upstream's own LFS metadata. A file that
matches proves the bytes are upstream's -- not merely that two copies we made
agree with each other, which is what hashing a download against itself proves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "framework/registries/ink-weights-0.1.0.json"


@pytest.fixture(name="panel")
def _panel():
    sys.path.insert(0, str(ROOT / "panel"))
    import app  # noqa: PLC0415
    return app


@pytest.fixture(name="entries")
def _entries():
    return json.loads(MANIFEST.read_text())["entries"]


def test_every_entry_carries_a_digest_and_a_size(entries):
    for entry in entries:
        assert len(entry["sha256"]) == 64, entry["destination"]
        assert int(entry["size_bytes"]) > 0, entry["destination"]


def test_no_two_destinations_collide(entries):
    """Two upstream files landing on one path would make the later one shadow
    the earlier, and the page would report both installed from one download."""
    seen: dict[str, str] = {}
    for entry in entries:
        clash = seen.get(entry["destination"])
        assert clash is None, (
            f"{entry['destination']} is claimed by {clash} and "
            f"{entry['repo']}/{entry['upstream_path']}")
        seen[entry["destination"]] = f"{entry['repo']}/{entry['upstream_path']}"


def test_no_destination_escapes_the_models_root(entries):
    for entry in entries:
        destination = Path(entry["destination"])
        assert not destination.is_absolute(), entry["destination"]
        assert ".." not in destination.parts, entry["destination"]


def test_the_fourteen_ink_9um_checkpoints_are_separately_addressable(entries):
    """Upstream's layout is preserved, so a seed and a step are readable from
    the path rather than encoded in a name somebody invented."""
    nine_um = sorted(e["destination"] for e in entries
                     if e["repo"] == "scrollprize/ink_9um")
    assert len(nine_um) == 14, nine_um
    assert len({e for e in nine_um}) == 14
    for seed in ("seed42", "seed43"):
        steps = [d for d in nine_um if f"hybrid_3d2d-{seed}/" in d]
        assert len(steps) == 7, (seed, steps)
    assert "ink_9um/hybrid_3d2d-seed42/step-075000.pth" in nine_um


def test_the_digests_are_distinct_so_a_seed_is_not_a_copy_of_the_other(entries):
    """If two of the fourteen shared a digest they would be one file under two
    names, and a comparison between them would be measuring nothing -- the
    failure `flip_normals` already produced once here at r=1.0000."""
    nine_um = [e for e in entries if e["repo"] == "scrollprize/ink_9um"]
    assert len({e["sha256"] for e in nine_um}) == 14


def test_the_profile_checkpoint_is_the_one_the_manifest_names(entries):
    """The declared 9 um checkpoint has to be a file that exists upstream, or
    the profile pins a digest nothing can ever satisfy."""
    profile = json.loads(
        (ROOT / "framework/profiles/03-ink/ink-9um-hybrid-3d2d-screening-1.0.0.json").read_text())
    match = [e for e in entries if e["sha256"] == profile["checkpoint_sha256"]]
    assert len(match) == 1, profile["checkpoint_sha256"]
    assert match[0]["upstream_path"] == "hybrid_3d2d-seed42/step-075000.pth"


def test_the_panel_offers_all_of_them(panel, monkeypatch, tmp_path):
    """On disk is not wired. The models page is where a checkpoint becomes
    something a person can see, and it derived its whole list from the profiles.
    """
    monkeypatch.setattr(panel, "MODELS_ROOT", tmp_path / "empty")
    rows = panel.declared_checkpoints()
    offered = {r["checkpoint_sha256"] for r in rows}
    for entry in json.loads(MANIFEST.read_text())["entries"]:
        assert entry["sha256"] in offered, entry["destination"]
    addressable = [r for r in rows
                   if str(r.get("expected_path", "")).startswith("ink_9um/")]
    assert len(addressable) == 14


def test_a_missing_manifest_does_not_take_the_page_down(panel, monkeypatch, tmp_path):
    """The profiles are what a job can actually name; the manifest is a
    convenience on top. A deployment shipping one and not the other still works.
    """
    monkeypatch.setattr(panel, "REPO", tmp_path)
    assert panel.weight_manifest_entries() == []
