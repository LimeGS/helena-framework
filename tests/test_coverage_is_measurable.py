"""The question the framework is named for.

"How much of this scroll have we explored" had no answer. Coverage existed only
as a ranking input inside the bootstrap -- how far a candidate cell is from the
surfaces already grown -- and was never reported, so progress was read off a
surface count, which rises whether the fleet is finding new ground or re-treading
old.

The arithmetic below is the part worth pinning: a grid's step comes from the
cell centres its tasks carry, and two plausible shortcuts are both wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.postgres_store import _cells_in_volume, _grid_step  # noqa: E402

# Three cells of the stratified grid on PHerc826, as their tasks record them.
CENTERS = [{"x": 2560, "y": 7656, "z": 2560},
           {"x": 3584, "y": 7656, "z": 2560},
           {"x": 2560, "y": 8680, "z": 3584}]


def test_the_step_comes_from_the_centres():
    assert _grid_step(CENTERS) == [1024.0, 1024.0, 1024.0]


def test_bounds_are_not_the_cell():
    """`bounds_xyz` is the candidate discovery region around the centre -- 256
    voxels across where the grid steps by 1024 -- so a cell count derived from
    it is wrong by a factor of 64. Recorded here because I derived it from that
    field first and got 67,300 cells for a volume that holds 1,052."""
    assert _cells_in_volume([8169, 8169, 16920], [1024.0, 1024.0, 1024.0]) == 1052
    assert _cells_in_volume([8169, 8169, 16920], [256.0, 256.0, 256.0]) == 67300


def test_one_cell_does_not_reveal_a_step():
    """A grid with a single attempted cell has no spacing to measure, and
    guessing one would put a made-up denominator under a coverage fraction."""
    assert _grid_step([CENTERS[0]]) is None
    assert _cells_in_volume([8169, 8169, 16920], None) is None


def test_a_missing_shape_is_not_a_zero():
    assert _cells_in_volume(None, [1024.0, 1024.0, 1024.0]) is None
