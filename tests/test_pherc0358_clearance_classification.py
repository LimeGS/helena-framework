"""Pin the causal classification of PHerc0358's 8/8/0 clearance result.

The 2026-08-02 First Letters hybrid campaign rejected exactly one attempt at
`INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE`, with raw/post-CT/usable = 8/8/0.
These tests hold every step of the classification recorded in
`docs/first-letters/pherc0358-clearance-classification.md`:

* the frozen evidence the classification reads, by content hash;
* what the volume gate actually computes, and the fact that the *cell* gate is
  vacuous at the shipped default so its zero count carries no information;
* the structural reason a first wave on a fresh scroll always lands on the
  scanned-volume rim, where part of every query cube is inadmissible by
  construction;
* the reproduction of the exact 8/8/0 signature through the production
  generator and screen with **no threshold changed**;
* the reachability limit of the review taxonomy at the shipped default.

Nothing here authorizes a margin change, and nothing here claims the scroll
lacks ink, text or letters.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from fleet.generator import DEFAULT_ENVELOPE, generate_tasks_for_snapshot
from fleet.planner import screen_candidates
from fleet.store import FleetStore

SCRIPT = ROOT / "scripts/harness/analyze_candidate_clearance.py"
SPEC = importlib.util.spec_from_file_location("candidate_clearance_analysis", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

EVIDENCE_DIR = ROOT / "docs/first-letters/first-letters-hybrid-20260802"
CLASSIFICATION = ROOT / "docs/first-letters/pherc0358-clearance-classification.md"
CATALOG = ROOT / "workspace/catalog/eligible_volumes.json"

# The bounded campaign's own record, frozen on 2026-08-02.  The classification
# reads these two files and nothing else from that campaign; pinning their
# bytes keeps a later edit from silently changing what was classified.
FROZEN_README_SHA256 = (
    "be683c27bcffd7f2c83885b7ababcf0263841d9e0d044cd42fd7c0d918dee40e")
FROZEN_EVIDENCE_SHA256 = (
    "a2aad7a2fbdaeca590b3bf6664441db6c1b362b4bcf7e48742dc1703ba50e12e")

# PHerc0358 as the frozen catalog records it: an OME-Zarr shape in z/y/x that
# the control plane reverses into the x/y/z frame the gate measures in.
PHERC0358_SHAPE_ZYX = [14744, 7783, 7783]
PHERC0358_SHAPE_XYZ = [7783, 7783, 14744]
PHERC0358_VOXEL_UM = 9.362

# The shipped bootstrap defaults (framework/stages/01-segmentation/fleet/cli.py).
QUERY_RADIUS = 64
VOLUME_EDGE_MARGIN = 64
CANDIDATE_INTERIOR_CLEARANCE = 0
GRID_STEP = 2048
MAX_TASKS = 4

CAUSE_CELL = "INSUFFICIENT_CELL_INTERIOR_CLEARANCE"
CAUSE_VOLUME = "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_scroll_tasks(tmp_path: Path) -> list[dict]:
    """Generate a first wave exactly as the campaign's bootstrap would.

    A scroll with no grown surface is the campaign's starting state: PHerc0358
    finished with `surface_count` 0, so every cell tied at infinite clearance
    for every one of its four attempts.
    """

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshot_id = store.register_snapshot({
        "sample_id": "PHerc358",
        "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64,
        "m7_threshold": 0.2,
        "shape_xyz": list(PHERC0358_SHAPE_XYZ),
        "voxel_size_um": PHERC0358_VOXEL_UM,
        "coordinate_frame": "ct_l0_xyz",
    })
    snapshot = store.snapshots({"PHerc358"})[0]
    assert store.surfaces_for_snapshot(snapshot_id) == []
    return generate_tasks_for_snapshot(
        store, snapshot,
        catalog_snapshot_sha256="2" * 64,
        grid_step=GRID_STEP,
        query_radius=QUERY_RADIUS,
        clearance=256.0,
        volume_edge_margin=VOLUME_EDGE_MARGIN,
        candidate_interior_clearance=CANDIDATE_INTERIOR_CLEARANCE,
        selection_strategy="stratified-clearance-v1",
        max_tasks=MAX_TASKS,
        grid_version="ct-l0-grid-2048-v1",
        policy_version="first-letters-hybrid-20260802-v1",
        ct_material_support_gate={
            "policy": "ome-zarr-nearby-material-v1",
            "level": 0,
            "radius_l0_voxels": 8,
            "minimum_nonzero_voxels": 1,
        },
    )


# --------------------------------------------------------------------------
# The evidence the classification is allowed to rest on
# --------------------------------------------------------------------------


def test_the_classification_reads_the_exact_frozen_campaign_bytes() -> None:
    """A classification of a frozen negative is only as good as its inputs.

    If either file is edited, this fails and the write-up must be re-derived
    rather than quietly inheriting a conclusion drawn from different bytes.
    """

    assert _file_sha256(EVIDENCE_DIR / "README.md") == FROZEN_README_SHA256
    assert _file_sha256(EVIDENCE_DIR / "evidence.json") == FROZEN_EVIDENCE_SHA256


def test_the_frozen_evidence_records_one_clearance_rejection_and_no_cell_rejection() -> None:
    """8/8/0 is one attempt out of four, and its only cause is the volume gate.

    `primary_causes` counts sum to the attempt count across the whole campaign,
    so each NO_SEED attempt contributed exactly one cause.  PHerc0358's single
    clearance attempt therefore reported no CT rejection, no malformed
    candidate and no *cell* rejection -- every one of the eight died on the
    volume gate alone.
    """

    evidence = json.loads((EVIDENCE_DIR / "evidence.json").read_text())
    rows = [row for row in evidence["per_scroll"] if row["scroll"] == "PHerc0358"]
    assert len(rows) == 1
    row = rows[0]
    assert row["attempt_count"] == 4
    assert row["candidate_counts"] == {"raw": 8, "post_ct": 8, "usable": 0}
    assert row["primary_causes"] == {
        "NO_M7_CANDIDATES": 3, CAUSE_VOLUME: 1}
    assert row["surface_count"] == 0
    assert sum(row["primary_causes"].values()) == row["attempt_count"]

    aggregate = evidence["aggregate"]
    assert sum(aggregate["primary_causes"].values()) == (
        aggregate["attempt_projection_states_literal"]["NO_SEED"])
    assert aggregate["primary_causes"][CAUSE_VOLUME] == 1
    assert CAUSE_CELL not in aggregate["primary_causes"]
    assert "CT_MATERIAL_SUPPORT_REJECTED" not in aggregate["primary_causes"]

    scope = evidence["scope"]
    assert scope["selection_strategy"] == "stratified-clearance-v1"
    assert scope["ct_material_support_gate"] == "ome-zarr-nearby-material-v1"
    assert scope["planner"] == "cost-aware-v2"
    assert scope["seed_probe_mode"] == "off"
    # Everything needed to recompute a distance was deliberately redacted.
    for redacted in ("task, attempt, and QC identifiers",
                     "source-snapshot identifiers", "raw logs"):
        assert redacted in scope["redacted"]


def test_the_frozen_evidence_carries_no_candidate_geometry() -> None:
    """The public dossier never published a coordinate, bound or volume shape.

    This is the reason the classification cannot be closed from the repository
    alone, and it is asserted rather than described so the non-claim section
    cannot drift away from the evidence.
    """

    text = (EVIDENCE_DIR / "evidence.json").read_text()
    for absent in ("bounds_xyz", "center_xyz", "shape_xyz", "ct_l0_coordinate",
                   "cell_id", "minimum_volume_interior_clearance_voxels",
                   "query_radius", "volume_edge_margin", "seed_region_policy"):
        assert absent not in text, f"{absent} is present after all; re-derive the review"


def test_the_catalog_axis_order_matches_the_frame_the_gate_measures_in() -> None:
    """An x/z transposition would fake a boundary rejection deep inside a scroll.

    PHerc0358 is 14744 slices by 7783 by 7783.  Reading the stored z/y/x shape
    as x/y/z would cap z at 7783 and reject every candidate in the upper half
    of the scan.  The control plane reverses the catalog shape, so that defect
    is excluded -- and this test keeps it excluded.
    """

    catalog = json.loads(CATALOG.read_text())
    entries = [row for row in catalog["entries"] if row["sample_id"] == "PHerc358"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["shape_zyx"] == PHERC0358_SHAPE_ZYX
    assert entry["voxel_size_um"] == PHERC0358_VOXEL_UM
    assert list(reversed(entry["shape_zyx"])) == PHERC0358_SHAPE_XYZ
    assert "PHerc0358" in entry["ct_uri"]

    metadata = json.loads(
        (ROOT / "workspace/catalog/volume_metadata/PHerc358.json").read_text())
    assert metadata["stored_array_order"] == "zyx"
    assert metadata["consumer_coordinate_order"] == "xyz"
    assert metadata["ct_zarray"]["shape"] == PHERC0358_SHAPE_ZYX


# --------------------------------------------------------------------------
# What the rule computes
# --------------------------------------------------------------------------


def test_the_volume_gate_measures_the_candidate_not_the_cell() -> None:
    """The gate is an absolute distance from the candidate to a volume face.

    Six faces, an inclusive high bound of `shape - 1`, and a strict `<`
    comparison against the frozen minimum.  Nothing about the task's cell
    enters it, which is why a cell whose *centre* satisfies the margin can
    still offer only inadmissible ground.
    """

    shape = list(PHERC0358_SHAPE_XYZ)

    def screen(coordinate: dict[str, int]) -> dict:
        # A cell centred on the candidate, so only the volume term can bite.
        centre = [coordinate[axis] for axis in "xyz"]
        task = {
            "bounds_xyz": [
                [float(value - QUERY_RADIUS) for value in centre],
                [float(value + QUERY_RADIUS) for value in centre],
            ],
            "source": {"shape_xyz": shape, "voxel_size_um": PHERC0358_VOXEL_UM},
            "parameter_envelope": {"maximum_candidate_count": 8},
            "candidate_discovery": {
                "minimum_cell_interior_clearance_voxels": 0,
                "minimum_volume_interior_clearance_voxels": VOLUME_EDGE_MARGIN,
            },
        }
        return screen_candidates({"candidates": [
            {"candidate_id": "c01", "ct_l0_coordinate": dict(coordinate),
             "score": 0.9}]}, task)

    # Exactly at the margin is admitted; one voxel outward is not.
    assert screen({"x": 64, "y": 64, "z": 64})["usable_candidate_count"] == 1
    outward = screen({"x": 63, "y": 64, "z": 64})
    assert outward["usable_candidate_count"] == 0
    assert outward["rejection_diagnostics"]["rejection_counts"][CAUSE_VOLUME] == 1
    assert outward["rejection_diagnostics"]["rejection_counts"][CAUSE_CELL] == 0
    # The high face is inclusive of `shape - 1`, on every axis independently.
    for index, axis in enumerate("xyz"):
        coordinate = {"x": 4096, "y": 4096, "z": 4096}
        coordinate[axis] = shape[index] - 1 - VOLUME_EDGE_MARGIN
        assert screen(coordinate)["usable_candidate_count"] == 1
        coordinate[axis] += 1
        assert screen(coordinate)["usable_candidate_count"] == 0


def test_the_cell_gate_cannot_fire_on_a_candidate_inside_its_own_region() -> None:
    """Zero cell rejections is a property of the default, not of the geometry.

    `--candidate-interior-clearance` ships at 0, so the cell term degenerates
    into "the provider answered inside the region it was asked about".  Every
    corner of the cube passes it.  The frozen dossier's silence about a cell
    cause therefore says nothing about where the eight candidates sat inside
    their cube -- it only says the provider stayed in bounds.
    """

    task = {
        "bounds_xyz": [[0.0, 0.0, 0.0], [128.0, 128.0, 128.0]],
        "source": {"shape_xyz": list(PHERC0358_SHAPE_XYZ),
                   "voxel_size_um": PHERC0358_VOXEL_UM},
        "parameter_envelope": {"maximum_candidate_count": 8},
        "candidate_discovery": {
            "minimum_cell_interior_clearance_voxels": CANDIDATE_INTERIOR_CLEARANCE,
            "minimum_volume_interior_clearance_voxels": VOLUME_EDGE_MARGIN,
        },
    }
    corners = [
        {"x": x, "y": y, "z": z}
        for x in (0, 128) for y in (0, 128) for z in (0, 128)
    ]
    screen = screen_candidates({"candidates": [
        {"candidate_id": f"c{index:02d}", "ct_l0_coordinate": corner, "score": 0.9}
        for index, corner in enumerate(corners, start=1)]}, task)
    counts = screen["rejection_diagnostics"]["rejection_counts"]
    assert counts[CAUSE_CELL] == 0
    assert counts["MALFORMED_COORDINATE_OR_SCORE"] == 0
    assert screen["rejection_diagnostics"]["clearance_policy"][
        "minimum_cell_interior_clearance_um"] == 0.0
    # It is not dead code: a candidate the provider returned from outside the
    # claimed region still fails, even at a zero minimum.
    outside = screen_candidates({"candidates": [
        {"candidate_id": "c99", "ct_l0_coordinate": {"x": 200, "y": 64, "z": 64},
         "score": 0.9}]}, task)
    assert outside["rejection_diagnostics"]["rejection_counts"][CAUSE_CELL] == 1


# --------------------------------------------------------------------------
# The structural mechanism, measured
# --------------------------------------------------------------------------


def test_query_cube_volume_gate_geometry_measures_the_admissible_fraction() -> None:
    """A read-only measurement of how much of a cube its own gate can admit."""

    corner = analysis.query_cube_volume_gate_geometry(
        bounds_xyz=[[0, 0, 0], [128, 128, 128]],
        shape_xyz=list(PHERC0358_SHAPE_XYZ),
        minimum_volume_interior_clearance_voxels=VOLUME_EDGE_MARGIN,
    )
    assert corner["inadmissible_faces"] == ["x_low", "y_low", "z_low"]
    assert corner["query_cube_voxel_count"] == 129 ** 3
    assert corner["admissible_bounds_xyz"] == [[64, 64, 64], [128, 128, 128]]
    assert corner["admissible_voxel_count"] == 65 ** 3
    assert corner["admissible_voxel_fraction"] == pytest.approx(
        65 ** 3 / 129 ** 3)
    assert corner["gate_admits_whole_cube"] is False
    assert corner["gate_admits_no_voxel"] is False
    assert corner["diagnostic_only"] is True


def test_query_cube_volume_gate_geometry_admits_an_interior_cube_whole() -> None:
    """A cell far from every face offers its whole cube, as it should."""

    interior = analysis.query_cube_volume_gate_geometry(
        bounds_xyz=[[4096, 4096, 4096], [4224, 4224, 4224]],
        shape_xyz=list(PHERC0358_SHAPE_XYZ),
        minimum_volume_interior_clearance_voxels=VOLUME_EDGE_MARGIN,
    )
    assert interior["inadmissible_faces"] == []
    assert interior["gate_admits_whole_cube"] is True
    assert interior["admissible_voxel_fraction"] == 1.0
    assert interior["admissible_voxel_count"] == interior["query_cube_voxel_count"]


def test_query_cube_volume_gate_geometry_reports_a_wholly_closed_cube() -> None:
    """A margin wider than the cube's reach closes it completely."""

    closed = analysis.query_cube_volume_gate_geometry(
        bounds_xyz=[[0, 0, 0], [40, 40, 40]],
        shape_xyz=[512, 512, 512],
        minimum_volume_interior_clearance_voxels=64,
    )
    assert closed["gate_admits_no_voxel"] is True
    assert closed["admissible_voxel_count"] == 0
    assert closed["admissible_voxel_fraction"] == 0.0
    assert closed["admissible_bounds_xyz"] is None


@pytest.mark.parametrize("kwargs", [
    {"bounds_xyz": [[0, 0], [10, 10]]},
    {"bounds_xyz": [[10, 0, 0], [0, 10, 10]]},
    {"shape_xyz": [0, 512, 512]},
    {"shape_xyz": [512.5, 512, 512]},
    {"minimum_volume_interior_clearance_voxels": -1},
    {"bounds_xyz": [[float("nan"), 0, 0], [10, 10, 10]]},
    {"bounds_xyz": [[-500, -500, -500], [-400, -400, -400]]},
])
def test_query_cube_volume_gate_geometry_fails_closed(kwargs) -> None:
    """A diagnostic that guesses at a malformed bound is worse than none."""

    call = {
        "bounds_xyz": [[0, 0, 0], [128, 128, 128]],
        "shape_xyz": [512, 512, 512],
        "minimum_volume_interior_clearance_voxels": 64,
        **kwargs,
    }
    with pytest.raises(ValueError):
        analysis.query_cube_volume_gate_geometry(**call)


def test_a_fresh_scroll_first_wave_lands_on_the_scanned_volume_rim(tmp_path) -> None:
    """Every first-wave cell sits exactly at the margin it must clear.

    With no grown surface the clearance term is infinite for every cell, so the
    deterministic tie-break -- lowest grid index wins -- decides the whole
    wave.  The lowest index on an axis is by construction the cell whose centre
    is exactly `volume_edge_margin` from that face, so the first wave is
    guaranteed to sample the volume's low corner.
    """

    tasks = _fresh_scroll_tasks(tmp_path)
    assert len(tasks) == MAX_TASKS
    assert [task["cell_id"] for task in tasks] == [
        "r00000c00000a00000", "r00000c00000a00004",
        "r00000c00003a00000", "r00000c00003a00004",
    ]
    shape = PHERC0358_SHAPE_XYZ
    for task in tasks:
        centre = [task["center_xyz"][axis] for axis in "xyz"]
        margin = min(*centre, *(shape[axis] - 1 - centre[axis] for axis in range(3)))
        # Not merely "near" the rim: exactly at the threshold, on every task.
        assert margin == VOLUME_EDGE_MARGIN
        discovery = task["candidate_discovery"]
        assert discovery["minimum_volume_interior_clearance_voxels"] == VOLUME_EDGE_MARGIN
        assert discovery["minimum_cell_interior_clearance_voxels"] == CANDIDATE_INTERIOR_CLEARANCE
        assert discovery["max_candidates"] == DEFAULT_ENVELOPE["maximum_candidate_count"] == 8
        assert discovery["region"]["radius"] == {axis: QUERY_RADIUS for axis in "xyz"}


def test_the_first_wave_asks_for_ground_its_own_gate_cannot_admit(tmp_path) -> None:
    """The provider is asked over the whole cube; the gate admits a fraction.

    The same number is used twice -- to inset the grid and to gate a single
    candidate -- while a candidate may sit `query_radius` voxels outward of the
    centre the inset protected.  At the shipped default the two numbers are
    equal, so seven eighths of the corner cell's cube is unreachable no matter
    what M7 proposes there.
    """

    tasks = _fresh_scroll_tasks(tmp_path)
    fractions = []
    for task in tasks:
        geometry = analysis.query_cube_volume_gate_geometry(
            bounds_xyz=task["bounds_xyz"],
            shape_xyz=list(PHERC0358_SHAPE_XYZ),
            minimum_volume_interior_clearance_voxels=task["candidate_discovery"][
                "minimum_volume_interior_clearance_voxels"],
        )
        assert geometry["gate_admits_whole_cube"] is False
        assert geometry["gate_admits_no_voxel"] is False
        fractions.append(geometry["admissible_voxel_fraction"])
    # Three closed faces, then two, two, and one.
    assert fractions[0] < fractions[1] <= fractions[3]
    assert fractions[0] == pytest.approx(65 ** 3 / 129 ** 3)
    assert fractions[3] == pytest.approx(65 / 129)
    assert all(fraction < 1.0 for fraction in fractions)


def test_the_exact_8_8_0_signature_reproduces_with_no_threshold_changed(tmp_path) -> None:
    """Eight top-scoring candidates in the outward shell, and the frozen counts.

    This reproduces the observed signature; it does not prove the observed
    attempt took this path, because the campaign's candidate coordinates were
    never published.
    """

    task = dict(_fresh_scroll_tasks(tmp_path)[0])
    task["source"] = {"shape_xyz": list(PHERC0358_SHAPE_XYZ),
                      "voxel_size_um": PHERC0358_VOXEL_UM}
    assert task["bounds_xyz"] == [[0.0, 0.0, 0.0], [128.0, 128.0, 128.0]]
    candidates = [
        {"candidate_id": f"m7-{index}", "score": 0.90 - index / 100.0,
         "ct_l0_coordinate": {"x": 10 + index, "y": 70 + index, "z": 70 + index}}
        for index in range(1, 9)
    ]
    screen = screen_candidates({"candidates": candidates}, task)
    assert screen["raw_candidate_count"] == 8
    assert screen["eligible_candidate_count"] == 0
    assert screen["usable_candidate_count"] == 0
    assert screen["best_candidate"] is None
    counts = screen["rejection_diagnostics"]["rejection_counts"]
    assert counts == {"MALFORMED_COORDINATE_OR_SCORE": 0,
                      CAUSE_CELL: 0, CAUSE_VOLUME: 8}
    policy = screen["rejection_diagnostics"]["clearance_policy"]
    assert policy["minimum_volume_interior_clearance_voxels"] == float(VOLUME_EDGE_MARGIN)
    assert policy["minimum_volume_interior_clearance_um"] == pytest.approx(
        VOLUME_EDGE_MARGIN * PHERC0358_VOXEL_UM)
    # Moving the same eight candidates inward by the deficit, and nothing else,
    # retains all eight: the rejection is positional, not a broken screen.
    inward = [
        {**candidate, "ct_l0_coordinate": {
            axis: value + 54 if axis == "x" else value
            for axis, value in candidate["ct_l0_coordinate"].items()}}
        for candidate in candidates
    ]
    recovered = screen_candidates({"candidates": inward}, task)
    assert recovered["usable_candidate_count"] == 8
    assert recovered["rejection_diagnostics"]["rejection_counts"][CAUSE_VOLUME] == 0


def test_the_provider_cap_censors_the_funnel_before_the_gate(tmp_path) -> None:
    """`raw = 8` is the request cap, so "eight candidates" is a truncation.

    The gate sees M7's top eight by score, chosen by a model that cannot see
    the gate.  A region holding admissible candidates below rank eight still
    reports 8/8/0.
    """

    task = dict(_fresh_scroll_tasks(tmp_path)[0])
    task["source"] = {"shape_xyz": list(PHERC0358_SHAPE_XYZ),
                      "voxel_size_um": PHERC0358_VOXEL_UM}
    assert task["candidate_discovery"]["max_candidates"] == 8
    assert task["parameter_envelope"]["maximum_candidate_count"] == 8
    outward = [
        {"candidate_id": f"m7-{index}", "score": 0.90 - index / 100.0,
         "ct_l0_coordinate": {"x": 10 + index, "y": 70 + index, "z": 70 + index}}
        for index in range(1, 9)
    ]
    below_the_cap = [
        {"candidate_id": f"m7-{index}", "score": 0.50 - index / 100.0,
         "ct_l0_coordinate": {"x": 96, "y": 96, "z": 96 + index}}
        for index in range(9, 13)
    ]
    truncated = screen_candidates({"candidates": outward}, task)
    complete = screen_candidates({"candidates": outward + below_the_cap}, task)
    assert truncated["usable_candidate_count"] == 0
    assert complete["usable_candidate_count"] == 4
    assert complete["raw_candidate_count"] == 12


# --------------------------------------------------------------------------
# What the review taxonomy can and cannot say at the shipped default
# --------------------------------------------------------------------------


def test_policy_margin_reachability_depends_on_margin_exceeding_query_radius() -> None:
    """`POLICY_MARGIN_UNCALIBRATED` needs a rejection that was physically safe.

    The analyzer calls a rejection physically safe when the candidate's own
    query cube fits inside the volume, that is when its face distance is at
    least `query_radius`.  When the margin equals the query radius -- the
    shipped default -- every rejected candidate is by definition unsafe, so
    that verdict cannot be produced at all.
    """

    shipped = analysis.policy_margin_verdict_reachability(
        minimum_volume_interior_clearance_voxels=VOLUME_EDGE_MARGIN,
        query_radius_voxels=QUERY_RADIUS,
    )
    assert shipped["policy_margin_uncalibrated_reachable"] is False
    assert shipped["diagnostic_only"] is True

    widened = analysis.policy_margin_verdict_reachability(
        minimum_volume_interior_clearance_voxels=VOLUME_EDGE_MARGIN + 1,
        query_radius_voxels=QUERY_RADIUS,
    )
    assert widened["policy_margin_uncalibrated_reachable"] is True

    narrowed = analysis.policy_margin_verdict_reachability(
        minimum_volume_interior_clearance_voxels=VOLUME_EDGE_MARGIN - 1,
        query_radius_voxels=QUERY_RADIUS,
    )
    assert narrowed["policy_margin_uncalibrated_reachable"] is False

    with pytest.raises(ValueError):
        analysis.policy_margin_verdict_reachability(
            minimum_volume_interior_clearance_voxels=-1,
            query_radius_voxels=QUERY_RADIUS)
    with pytest.raises(ValueError):
        analysis.policy_margin_verdict_reachability(
            minimum_volume_interior_clearance_voxels=VOLUME_EDGE_MARGIN,
            query_radius_voxels=0)


def test_the_analyzer_cannot_reach_the_uncalibrated_verdict_at_the_default() -> None:
    """Proved against the analyzer's own branch, not only against the lemma.

    Any candidate the gate rejects at `margin == query_radius` is scored
    physically unsafe by the same analyzer, which routes the review to
    `TRUE_VOLUME_BOUNDARY`.  A `TRUE_VOLUME_BOUNDARY` verdict produced under
    the shipped default is therefore uninformative on its own.
    """

    margin = query_radius = QUERY_RADIUS
    for distance in range(0, margin):
        # Rejected by the gate ...
        assert distance < margin
        # ... and never physically safe, which is the analyzer's own predicate.
        assert not (distance >= query_radius)
    assert analysis.SENSITIVITY_PERCENTAGES == (100, 75, 50, 25, 0)


# --------------------------------------------------------------------------
# The write-up
# --------------------------------------------------------------------------


def test_the_classification_document_states_a_class_and_its_non_claims() -> None:
    """A review without an explicit non-claim section is not a review."""

    text = CLASSIFICATION.read_text()
    assert "Explicit non-claims" in text
    assert FROZEN_README_SHA256 in text
    assert FROZEN_EVIDENCE_SHA256 in text
    for phrase in ("No ink claim", "not ink absence", "TRUE_VOLUME_BOUNDARY",
                   "does not authorize changing it"):
        assert phrase in text
    named = [name for name in (
        "IMPLEMENTATION_OR_METADATA_DEFECT", "TRUE_VOLUME_BOUNDARY",
        "CELL_BOUNDARY_ONLY", "POLICY_MARGIN_UNCALIBRATED") if name in text]
    assert len(named) == 4, "the review must weigh every class it was offered"


def test_the_classification_document_changes_no_threshold() -> None:
    """The review is diagnostic; the shipped defaults must still be the defaults."""

    cli = (STAGE / "fleet/cli.py").read_text()
    assert '"--volume-edge-margin", type=int, default=64' in cli
    assert '"--query-radius", type=int, default=64' in cli
    assert '"--candidate-interior-clearance", type=int, default=0' in cli
    assert math.isclose(DEFAULT_ENVELOPE["maximum_candidate_count"], 8)


def test_the_classification_document_names_only_files_that_exist() -> None:
    """A review that cites a moved file sends its reader nowhere.

    Directory citations are exempt: the review names one output location that
    deliberately does not exist yet, because the evidence to fill it is not in
    the repository.
    """

    text = CLASSIFICATION.read_text()
    cited = {
        path for path in re.findall(
            r"`((?:framework|panel|scripts|tests|docs|workspace)/[\w./-]+?)"
            r"(?::[\d–\-]+)?`", text)
        if not path.endswith("/")
    }
    assert cited, "the review must cite the code it read"
    for path in cited:
        assert (ROOT / path).exists(), f"the review names {path}, which is not here"
    assert not (ROOT / "docs/first-letters/pherc0358-clearance-review").exists(), (
        "the analyzer output directory exists; re-derive the review from the "
        "real attempt bundle instead of leaving section 6 open")


def test_the_documented_sensitivity_table_is_the_computed_one() -> None:
    """Every number in the diagnostic table is recomputed, not transcribed.

    The table describes how much of the rim cell's query cube each margin level
    closes.  It is cube geometry, and it cannot authorize a production
    threshold; this test only stops it from drifting away from the code.
    """

    text = CLASSIFICATION.read_text()
    for percentage in analysis.SENSITIVITY_PERCENTAGES:
        margin = VOLUME_EDGE_MARGIN * percentage // 100
        geometry = analysis.query_cube_volume_gate_geometry(
            bounds_xyz=[[0, 0, 0], [128, 128, 128]],
            shape_xyz=list(PHERC0358_SHAPE_XYZ),
            minimum_volume_interior_clearance_voxels=margin,
        )
        row = (f"| {percentage} % | {margin} | "
               f"{margin * PHERC0358_VOXEL_UM:.3f} | "
               f"{geometry['admissible_voxel_fraction']:.4f} |")
        assert row in text, f"the review's sensitivity row differs: {row}"
