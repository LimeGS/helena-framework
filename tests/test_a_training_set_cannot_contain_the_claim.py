"""Step five of the post, with the one rule it states in bold made enforceable.

The tutorial's iterate loop is: run inference, label conservatively, expand the
supervision mask, retrain, test on held-out regions, repeat. Among its key
principles is one that is not a matter of taste:

    Never train on region intended for prize claim

A model trained on the window you then claim has memorised the answer, and the
map it produces over that window is not evidence of anything. Upstream's own
training command cannot know which window you mean to claim; this platform
does, because a claim is a bbox on a surface it already tracks.

So the training set is assembled here rather than by hand, and assembling it
is where the rule is enforced: a tile that overlaps a declared holdout is
refused, and a build with no declaration at all is refused too. An absent
holdout must never read as "nothing is reserved" -- that is the same failure
as a skipped test reporting success, which this suite already refuses
elsewhere.

What this does not do is train. Upstream ships that, and the registry records
what happened the last time this platform reimplemented an upstream runner
instead of calling it. What comes back from training is a checkpoint, and a
checkpoint enters by digest through the method registry like every other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

from build_ink_training_set import (  # noqa: E402
    HoldoutViolation,
    UndeclaredHoldout,
    build_training_set,
    overlaps,
)


def _window(surface: str, box, path: Path | None = None) -> dict:
    return {"surface_id": surface, "bbox_xy": list(box),
            "image": str(path or "img.tif"), "label": str(path or "lbl.tif")}


def _files(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "image.tif").write_bytes(b"image-bytes")
    (directory / "label.tif").write_bytes(b"label-bytes")
    return directory


# -- the geometry ----------------------------------------------------------

@pytest.mark.parametrize(("a", "b", "expected"), [
    ((0, 0, 10, 10), (5, 5, 15, 15), True),      # corner overlap
    ((0, 0, 10, 10), (0, 0, 10, 10), True),      # identical
    ((0, 0, 10, 10), (2, 2, 4, 4), True),        # contained
    ((0, 0, 10, 10), (10, 0, 20, 10), False),    # edge-adjacent, not overlapping
    ((0, 0, 10, 10), (20, 20, 30, 30), False),   # disjoint
])
def test_overlap_is_decided_on_the_boxes(a, b, expected):
    assert overlaps(a, b) is expected


def test_a_touching_edge_is_not_an_overlap():
    """Half-open boxes: a tile that ends where the holdout begins shares no
    pixel with it. Treating that as contamination would reserve twice the
    ground somebody actually reserved."""
    assert overlaps((0, 0, 10, 10), (10, 10, 20, 20)) is False


# -- the rule --------------------------------------------------------------

def test_a_window_overlapping_the_claim_is_refused(tmp_path):
    source = _files(tmp_path, "w1")
    with pytest.raises(HoldoutViolation) as refused:
        build_training_set(
            windows=[_window("surface-a", (0, 0, 100, 100), source)],
            holdout=[{"surface_id": "surface-a", "bbox_xy": [50, 50, 150, 150],
                      "reason": "First Letters claim window"}],
            output=tmp_path / "set")
    message = str(refused.value)
    assert "surface-a" in message
    assert "First Letters claim window" in message, (
        "the refusal does not say which reservation was hit")


def test_the_same_box_on_a_different_surface_is_fine(tmp_path):
    """A holdout is a region of one surface, not a coordinate range in the
    abstract."""
    source = _files(tmp_path, "w1")
    manifest = build_training_set(
        windows=[_window("surface-b", (0, 0, 100, 100), source)],
        holdout=[{"surface_id": "surface-a", "bbox_xy": [0, 0, 100, 100],
                  "reason": "claim"}],
        output=tmp_path / "set")
    assert manifest["window_count"] == 1


def test_a_build_with_no_declaration_at_all_is_refused(tmp_path):
    """Fail closed. Forgetting to declare must not read as nothing reserved."""
    source = _files(tmp_path, "w1")
    with pytest.raises(UndeclaredHoldout):
        build_training_set(
            windows=[_window("surface-a", (0, 0, 10, 10), source)],
            holdout=None, output=tmp_path / "set")


def test_declaring_nothing_reserved_is_allowed_but_must_be_explicit(tmp_path):
    """An empty list is a statement somebody made. It is recorded as one."""
    source = _files(tmp_path, "w1")
    manifest = build_training_set(
        windows=[_window("surface-a", (0, 0, 10, 10), source)],
        holdout=[], output=tmp_path / "set")
    assert manifest["holdout_count"] == 0
    assert manifest["holdout_declared"] is True


def test_one_bad_window_refuses_the_whole_set(tmp_path):
    """Not "skip the contaminated tile and carry on": a set that quietly
    dropped a window is a set whose contents nobody stated."""
    good, bad = _files(tmp_path, "w1"), _files(tmp_path, "w2")
    with pytest.raises(HoldoutViolation):
        build_training_set(
            windows=[_window("surface-a", (0, 0, 10, 10), good),
                     _window("surface-a", (60, 60, 90, 90), bad)],
            holdout=[{"surface_id": "surface-a", "bbox_xy": [50, 50, 150, 150],
                      "reason": "claim"}],
            output=tmp_path / "set")
    assert not (tmp_path / "set").exists(), (
        "a refused build left a partial training set behind")


# -- what it hands to training --------------------------------------------

def test_the_manifest_binds_every_file_by_digest(tmp_path):
    source = _files(tmp_path, "w1")
    manifest = build_training_set(
        windows=[_window("surface-a", (0, 0, 10, 10), source)],
        holdout=[], output=tmp_path / "set")

    row = manifest["windows"][0]
    assert len(row["image_sha256"]) == 64
    assert len(row["label_sha256"]) == 64
    assert row["image_sha256"] != row["label_sha256"]


def test_the_manifest_records_the_holdout_it_was_checked_against(tmp_path):
    """Rebuilding the argument later needs the reservation, not just the
    assertion that one was applied."""
    source = _files(tmp_path, "w1")
    holdout = [{"surface_id": "surface-z", "bbox_xy": [0, 0, 5, 5],
                "reason": "claim"}]
    manifest = build_training_set(
        windows=[_window("surface-a", (0, 0, 10, 10), source)],
        holdout=holdout, output=tmp_path / "set")

    assert manifest["holdout"] == holdout
    assert len(manifest["holdout_sha256"]) == 64
    written = json.loads((tmp_path / "set" / "TRAINING_SET.json").read_text())
    assert written["holdout_sha256"] == manifest["holdout_sha256"]


def test_the_manifest_says_it_is_not_a_model(tmp_path):
    source = _files(tmp_path, "w1")
    manifest = build_training_set(
        windows=[_window("surface-a", (0, 0, 10, 10), source)],
        holdout=[], output=tmp_path / "set")
    claims = " ".join(manifest["non_claims"]).lower()
    assert "train" in claims
