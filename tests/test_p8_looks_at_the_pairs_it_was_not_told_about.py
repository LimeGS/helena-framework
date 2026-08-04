"""The merge certifies the seams it was given. This is about the ones it was not.

`evaluate_seam_qc` iterates `expected_edges` -- the layout's declared
neighbours -- and checks each for anchors, inliers and real overlap on both
parents. Thorough, and blind by construction to any pair the layout did not
name.

A scroll is one sheet, so two surfaces five wraps apart in the stitching order
can still occupy the same place. In the PHerc826 reconstruction two of them do:
w045 and w046 pass within 34 um across 62% of their extent, w041/w042 within
74 um, against a control pair of genuine neighbours at 165 um. Papyrus is
100-200 um thick. Every surface involved is GEOMETRY_CERTIFIED and every
declared seam passed.
"""

from __future__ import annotations

import ast
from pathlib import Path

MERGE = Path(__file__).resolve().parents[1] / (
    "framework/stages/05-reconstruction/scripts/run_vc3d_tifxyz_merge.py")
SOURCE = MERGE.read_text()
TREE = ast.parse(SOURCE)


def function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from the merge")


def test_seam_qc_still_only_walks_the_declared_edges() -> None:
    """Not a complaint -- it is what seam QC is for. The test pins the gap so
    the collision check below cannot be deleted as redundant."""
    source = ast.get_source_segment(SOURCE, function("evaluate_seam_qc")) or ""
    assert "expected_edges" in source
    assert "for key in sorted(expected_edges" in source, (
        "seam QC no longer iterates the declared edge set; if it now examines "
        "every pair, this file has the wrong premise"
    )


def test_every_pair_is_examined_after_the_merge() -> None:
    collision = function("evaluate_collision_qc")
    source = ast.get_source_segment(SOURCE, collision) or ""
    # An all-pairs loop, not a walk over declared edges.
    assert "for b in names[i + 1:]" in source, (
        "the collision check does not compare every pair, which is the whole "
        "reason it exists"
    )
    assert "declared_neighbours" in source, (
        "a contradiction between declared neighbours and one between strangers "
        "are different findings and must be distinguishable"
    )


def test_it_runs_on_every_merge() -> None:
    assert "evaluate_collision_qc(" in SOURCE.split("def evaluate_collision_qc")[-1], (
        "the check is defined and never called"
    )
    assert 'write_json(output / "COLLISION_QC.json"' in SOURCE, (
        "the verdict is computed and not written down, so nothing downstream "
        "can read it"
    )


def test_it_reports_rather_than_refusing_by_default() -> None:
    """A layout is a stitching order, not a claim about geometry: two surfaces
    may legitimately be adjacent without being declared neighbours. What cannot
    be legitimate is two sheets closer than one sheet is thick -- so the finding
    is recorded, and refusing is opt-in."""
    source = ast.get_source_segment(SOURCE, function("evaluate_collision_qc")) or ""
    assert 'policy.get("refuse_on_undeclared_collision", False)' in source, (
        "refusal is not opt-in; a merge that was fine yesterday would start "
        "failing on a check nobody has calibrated on their own data"
    )
    assert "MergeRefused(" in source, "there is no way to make it refuse at all"


def test_it_uses_the_measured_contract_rather_than_its_own_numbers() -> None:
    """The thresholds were calibrated on 37 reconstructed wraps. A second copy
    of them here is a second thing to keep in step."""
    source = ast.get_source_segment(SOURCE, function("evaluate_collision_qc")) or ""
    assert "from framework.contracts import winding" in source
    assert "winding.compare" in source
    assert "100" not in source.split("findings = []")[-1], (
        "a sheet thickness appears to be hardcoded here; it belongs to the "
        "winding contract, which was measured"
    )


def test_a_parent_that_cannot_be_read_does_not_fail_the_merge() -> None:
    """Geometry is read from disk and disks disappoint. A merge that is
    otherwise sound must not fail because one artifact could not be opened --
    it must say it could not evaluate."""
    source = ast.get_source_segment(SOURCE, function("evaluate_collision_qc")) or ""
    assert "NOT_EVALUATED" in source
    reader = ast.get_source_segment(SOURCE, function("_read_tifxyz_points")) or ""
    assert "return None" in reader and "except Exception" in reader
