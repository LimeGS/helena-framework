"""The winding invariant, and the numbers it was calibrated against.

Measured on 37 reconstructed wraps of PHerc826 -- w023 to w059, 4.4M points --
before the contract was written:

    radius recovers the wrap ordering    Spearman rho +0.9993, 35/36 monotone
    spacing between wraps                17 vx = 0.14 mm, which is papyrus
    pairs behaving, of 666 comparable    664
    the two that did not                 0.2 vx and 1.0 vx apart

Those last two are why UNDETERMINED exists. The full run is in
vesubius-challenge/winding-invariant.
"""

from __future__ import annotations

import math

from framework.contracts import winding


def ring(radius: float, z: float = 0.0, n: int = 64):
    """A circle of points at one radius: the simplest thing shaped like a wrap."""
    return [(radius * math.cos(2 * math.pi * i / n),
             radius * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def test_it_finds_where_a_surface_sits() -> None:
    at = winding.locate(ring(1000.0, z=25.0), centre=(0, 0, 0), axis=(0, 0, 1))
    assert at is not None
    assert abs(at.radius - 1000.0) < 1e-6
    assert abs(at.z - 25.0) < 1e-6
    assert at.points == 64


def test_two_sheets_a_papyrus_apart_are_consistent() -> None:
    """165 um was the control on ground truth -- w023 against w024, which the
    ordering check gets right 99.6% of the time."""
    a = winding.locate(ring(1000.0), (0, 0, 0), (0, 0, 1))
    b = winding.locate(ring(1000.0 + 165 / winding.VOXEL_UM), (0, 0, 0), (0, 0, 1))
    verdict, _ = winding.compare(a, b)
    assert verdict == winding.CONSISTENT


def test_closer_than_a_sheet_is_a_contradiction() -> None:
    """w045/w046 sit 34 um apart in the reconstruction, and both are
    GEOMETRY_CERTIFIED. Nothing physical fits between them, and no per-surface
    check can say so, because it is a statement about the pair."""
    a = winding.locate(ring(1000.0), (0, 0, 0), (0, 0, 1))
    b = winding.locate(ring(1000.0 + 34 / winding.VOXEL_UM), (0, 0, 0), (0, 0, 1))
    verdict, evidence = winding.compare(a, b)
    assert verdict == winding.CONTRADICTED
    assert evidence["separation_um"] == 34.0
    assert "papyrus" in evidence["why"]


def test_it_declines_where_the_order_was_never_recoverable() -> None:
    """The two pairs that failed on ground truth sat 0.2 and 1.0 voxels apart
    and inverted on 36-43% of rays -- a coin flip, because the data does not
    carry the answer. Reporting confidently there would be inventing."""
    a = winding.locate(ring(1000.0), (0, 0, 0), (0, 0, 1))
    b = winding.locate(ring(1001.0), (0, 0, 0), (0, 0, 1))
    verdict, evidence = winding.compare(a, b)
    assert verdict == winding.UNDETERMINED, (
        "a separation the ground truth could not resolve is being reported as "
        "though it could"
    )
    assert "recoverable" in evidence["why"]


def test_an_expected_order_can_be_contradicted() -> None:
    """Where something upstream already claims which wrap is outer, the check
    can disagree with it."""
    inner = winding.locate(ring(1000.0), (0, 0, 0), (0, 0, 1))
    outer = winding.locate(ring(1200.0), (0, 0, 0), (0, 0, 1))
    assert winding.compare(inner, outer, expected="b")[0] == winding.CONSISTENT
    assert winding.compare(inner, outer, expected="a")[0] == winding.CONTRADICTED


def test_surfaces_that_do_not_meet_are_not_judged() -> None:
    a = winding.locate(ring(1000.0), (0, 0, 0), (0, 0, 1))
    b = winding.locate(ring(1400.0), (0, 0, 0), (0, 0, 1))
    verdict, _ = winding.compare(a, b, shared_bins=2)
    assert verdict == winding.NOT_COMPARABLE


def test_the_thresholds_are_the_measured_ones() -> None:
    """These came from the run, not from taste. Changing one without a new
    measurement changes what the verdict means."""
    assert winding.VOXEL_UM == 8.0
    assert winding.SHEET_UM == 100.0
    assert winding.UNDETERMINED_VOXELS == 3.0
    assert winding.SHEET_VOXELS == 12.5


def test_a_stray_point_does_not_move_a_surface() -> None:
    """Median, not mean: a handful of points dragged onto a neighbouring wrap
    should not relocate the patch they belong to."""
    clean = ring(1000.0)
    strayed = clean + [(3000.0, 0.0, 0.0)] * 3
    a = winding.locate(clean, (0, 0, 0), (0, 0, 1))
    b = winding.locate(strayed, (0, 0, 0), (0, 0, 1))
    assert abs(a.radius - b.radius) < 1.0
