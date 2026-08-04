"""Where a surface sits in the roll, and whether two surfaces can both be true.

A scroll is one sheet wound outward, so two things hold that no check inside a
single surface can reach:

  * every point on a lamina has a radius and a winding angle about the scroll
    axis, and along one sheet the radius grows with the angle;
  * two distinct laminae cannot occupy the same place. Papyrus is 100-200 um
    thick, so two surfaces passing closer than that are not two sheets.

Measured on 37 reconstructed wraps of PHerc826 before this was written:

    radius recovers the wrap ordering      Spearman rho +0.9993, 35/36 monotone
    spacing between consecutive wraps      17 vx = 0.14 mm, which is papyrus
    pairs behaving, of 666 comparable      664
    the two that do not                    0.2 vx and 1.0 vx apart

That last line is the calibration and the reason this returns a verdict of
UNDETERMINED rather than a boolean. Where wraps are 15-25 vx apart the ordering
is right 95-99.6% of the time; where they are 1-3 vx apart it is a coin flip,
because at that separation the order is not in the data. A check that reports
confidently there would be inventing.

Deliberately not a gate. It answers a different question from geometry
certification, CT support, model response and human review, so it is a fifth
judgement beside them rather than a veto over any of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# 8 um per voxel at L0; papyrus is 100-200 um. Two surfaces closer than half a
# sheet are not two sheets, whatever they are labelled.
VOXEL_UM = 8.0
SHEET_UM = 100.0
SHEET_VOXELS = SHEET_UM / VOXEL_UM

# Below this the radial order was not recoverable on ground truth: the two pairs
# that failed there sat 0.2 and 1.0 voxels apart and inverted on 36-43% of rays.
UNDETERMINED_VOXELS = 3.0

CONSISTENT = "WINDING_CONSISTENT"
CONTRADICTED = "WINDING_CONTRADICTED"
UNDETERMINED = "WINDING_UNDETERMINED"
NOT_COMPARABLE = "WINDING_NOT_COMPARABLE"


@dataclass(frozen=True)
class Cylindrical:
    """One surface reduced to where it sits in the roll.

    Three numbers per surface rather than a mesh: this is what makes the check
    O(1) at publication time instead of a read of every artifact that might
    overlap.
    """

    radius: float
    theta: float
    z: float
    points: int


def axis_frame(axis: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Two directions across the scroll, given the one along it."""
    ax = _unit(axis)
    ref = (1.0, 0.0, 0.0) if abs(_dot(ax, (1.0, 0.0, 0.0))) < 0.9 else (0.0, 1.0, 0.0)
    e1 = _unit(_cross(ax, ref))
    return ax, e1, _unit(_cross(ax, e1))


def locate(points: Iterable[Sequence[float]], centre: Sequence[float],
           axis: Sequence[float]) -> Cylindrical | None:
    """Reduce a point cloud to its median position in the roll.

    Median rather than mean: a patch with a few points dragged onto a
    neighbouring wrap should not have its whole position moved by them.
    """
    ax, e1, e2 = axis_frame(axis)
    radii: list[float] = []
    thetas: list[float] = []
    zs: list[float] = []
    for p in points:
        rel = (p[0] - centre[0], p[1] - centre[1], p[2] - centre[2])
        z = _dot(rel, ax)
        u, v = _dot(rel, e1), _dot(rel, e2)
        radii.append(math.hypot(u, v))
        thetas.append(math.atan2(v, u))
        zs.append(z)
    if not radii:
        return None
    return Cylindrical(_median(radii), _median(thetas), _median(zs), len(radii))


def compare(a: Cylindrical, b: Cylindrical, *, expected: str | None = None,
            shared_bins: int = 0) -> tuple[str, dict]:
    """Can these two both be laminae of the same roll?

    `expected` names which of the two should be further out, when something
    upstream already claims to know -- a wrap number, a reconstruction order.
    Without it the check can still report that they are too close to be two
    sheets, which is the finding that does not need an ordering to state.
    """
    separation = abs(a.radius - b.radius)
    evidence = {
        "separation_voxels": round(separation, 2),
        "separation_um": round(separation * VOXEL_UM, 1),
        "sheet_voxels": SHEET_VOXELS,
        "shared_bins": shared_bins,
    }

    if shared_bins and shared_bins < 4:
        return NOT_COMPARABLE, evidence | {
            "why": "the two barely overlap; there is no shared ground to judge on"}

    if separation < UNDETERMINED_VOXELS:
        return UNDETERMINED, evidence | {
            "why": f"{separation * VOXEL_UM:.0f} um apart, under the "
                   f"{UNDETERMINED_VOXELS * VOXEL_UM:.0f} um at which the order "
                   "stopped being recoverable on ground truth"}

    if separation < SHEET_VOXELS:
        return CONTRADICTED, evidence | {
            "why": f"{separation * VOXEL_UM:.0f} um apart, closer than a sheet of "
                   f"papyrus is thick ({SHEET_UM:.0f}-200 um); nothing physical "
                   "fits between them"}

    if expected is not None:
        outer = expected
        seen = "a" if a.radius > b.radius else "b"
        if seen != outer:
            return CONTRADICTED, evidence | {
                "why": f"{seen} sits further out, and {outer} was expected to"}

    return CONSISTENT, evidence


# -- small helpers, kept local so this contract imports nothing ---------------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(a: Sequence[float]) -> tuple[float, float, float]:
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
