"""The anchor appears twice in the control manifest, and both must agree.

`known_region.anchor_*` is what the panel turns into the seed's coordinate.
`checks.PIPELINE_CONTROL.seed_provenance.*` is what the run is checked against
afterwards. They are the same fact written in two places, so moving the anchor
means moving both -- and moving one is worse than moving neither, because the
seed would then be planted at one point and audited against another.

That nearly happened. Version 1.1.0 was written by copying 1.0.0 and editing
`known_region`, and a field-by-field diff of the two files reported exactly the
intended change: the second copy was stale in both, so it did not appear as a
difference at all. Only grepping for the old numbers found it.

Every checked-in version is held to this, not just the current one: a stale copy
in an old version is a trap for whoever reads it next.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROFILES = Path(__file__).resolve().parents[1] / "framework/profiles/01-segmentation"
CONTROLS = sorted(PROFILES.glob("first-letters-control-policy-*.json"))


def test_there_is_a_control_profile_to_check() -> None:
    """A glob that matches nothing passes forever."""
    assert CONTROLS, f"no control profile found under {PROFILES}"


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_both_copies_of_the_anchor_say_the_same_thing(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    known = document["known_region"]
    provenance = document["checks"]["PIPELINE_CONTROL"]["seed_provenance"]

    assert known["anchor_surface_cell_yx"] == provenance["surface_cell_yx"], (
        "the seed would be planted at one cell and audited against another")
    assert known["anchor_ct_l0_xyz"] == provenance["coordinate_ct_l0_xyz"], (
        "the seed would be planted at one coordinate and audited against another")


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_the_profile_id_matches_the_file_it_lives_in(path: Path) -> None:
    """A file named 1.1.0 that calls itself 1.0.0 is two versions of one name."""
    document = json.loads(path.read_text(encoding="utf-8"))
    version = path.stem.rsplit("-", 1)[-1]
    assert document["profile_id"] == f"first-letters-control-policy@{version}"


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_the_anchor_is_inside_the_surface_it_is_an_anchor_on(path: Path) -> None:
    """The anchor is a point of the source-locked surface, and the manifest
    declares that surface's own bounding box. A coordinate outside it is not a
    cell of the surface whatever the cell index says."""
    document = json.loads(path.read_text(encoding="utf-8"))
    known = document["known_region"]
    bbox = known["surface_bbox_ct_l0_xyz"]
    rows, cols = known["surface_grid_shape_yx"]
    row, col = known["anchor_surface_cell_yx"]

    assert 0 <= row < rows and 0 <= col < cols, "the anchor cell is off the grid"
    for value, low, high in zip(known["anchor_ct_l0_xyz"],
                                bbox["minimum"], bbox["maximum"], strict=True):
        assert low <= value <= high, "the anchor is outside the locked surface"
