"""The measurement P2's third axis is made of.

The question it answers is the one neither existing axis asks: geometry
certifies mesh integrity and says in its own non-claims that it is *not* a claim
the segmentation followed the correct lamina, and CT support asks only whether
scanned material is there at all. This asks whether the density profile along
the normal holds two air/papyrus interfaces one sheet apart -- which is what
decides whether a 29-minute render is worth starting.

Thickness is the discriminating number. The bands come from a frozen profile, so
these tests state their own and never depend on the calibration file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.lamina import (  # noqa: E402
    assess_lamina, column_material_depth, half_maximum_crossings,
    histogram_bimodality, interface_level,
)

PROFILE = {
    "profile_id": "test-bands@1.0.0",
    "thickness_um": {"sheet_low": 15.0, "sheet_high": 70.0},
    "minimum_clean_fraction": 0.90,
    "bimodality_ceiling": 1.0,
}


def sheet(width_samples: int, *, length: int = 40, peak: float = 200.0) -> list[float]:
    """Air, a band of papyrus, air -- the shape a single lamina makes."""
    middle = length // 2
    low = width_samples / 2
    return [peak if abs(index - middle) <= low else 10.0 for index in range(length)]


def test_a_crossing_pair_is_interpolated_not_snapped() -> None:
    """At 9.362 um a sheet is four samples thick; rounding each crossing to a
    sample is a quarter of the quantity being measured."""
    profile = [0.0, 20.0, 100.0, 100.0, 0.0, 0.0]

    lower, upper = half_maximum_crossings(profile)

    assert 1.0 < lower < 2.0 and 3.0 < upper < 4.0
    # A width that is not a whole number of samples, which snapping cannot give.
    assert upper - lower != round(upper - lower)


def test_a_profile_with_no_interface_pair_is_not_a_thin_one() -> None:
    assert half_maximum_crossings([5.0, 5.0, 5.0, 5.0]) is None
    # Rising and never coming back down: one interface, not a pair.
    assert half_maximum_crossings([0.0, 10.0, 100.0, 100.0, 100.0]) is None


def test_one_sheet_measures_a_sheet() -> None:
    column = column_material_depth(sheet(8), sample_step_um=4.0)

    assert column["state"] == "CLEAN"
    assert 30.0 <= column["thickness_um"] <= 40.0


def test_a_column_with_a_hole_is_dropped_rather_than_measured() -> None:
    """A missing chunk reads as air, which is exactly what a thickness
    measurement is looking for -- so it must not be measured at all."""
    values = sheet(8)
    holes = [False] * len(values)
    holes[3] = True

    column = column_material_depth(values, sample_step_um=4.0, missing=holes)

    assert column["state"] == "HOLED"
    assert column["thickness_um"] is None


def test_a_sheet_passes_and_a_slab_does_not() -> None:
    thin = [column_material_depth(sheet(8), sample_step_um=4.0) for _ in range(64)]
    fused = [column_material_depth(sheet(32), sample_step_um=4.0) for _ in range(64)]

    passed = assess_lamina(thin, profile=PROFILE, bimodality=0.45)
    slab = assess_lamina(fused, profile=PROFILE, bimodality=0.65)

    assert passed["state"] == "LAMINA_SINGLE_SHEET"
    assert slab["state"] == "LAMINA_FUSED"
    # And says why in the units the reader thinks in.
    assert "um" in slab["reason"]


def test_holes_are_an_absence_of_verdict_not_a_verdict() -> None:
    """Too few clean columns is not a thin sheet and not a fused one. Calling
    it either would put a judgement on a hole."""
    values = sheet(8)
    holed = [column_material_depth(values, sample_step_um=4.0,
                                   missing=[True] + [False] * (len(values) - 1))
             for _ in range(50)]
    clean = [column_material_depth(values, sample_step_um=4.0) for _ in range(50)]

    outcome = assess_lamina(holed + clean, profile=PROFILE, bimodality=0.4)

    assert outcome["state"] == "LAMINA_INSUFFICIENT_COLUMNS"
    assert outcome["clean_fraction"] == 0.5


def test_no_valley_between_the_modes_is_its_own_answer() -> None:
    columns = [column_material_depth(sheet(8), sample_step_um=4.0) for _ in range(64)]

    outcome = assess_lamina(columns, profile=PROFILE, bimodality=1.4)

    assert outcome["state"] == "LAMINA_UNRESOLVED"


def test_bimodality_falls_below_one_for_two_materials() -> None:
    """Air and papyrus: two populations with a dip between them."""
    two_materials = [10.0] * 500 + [200.0] * 500

    assert histogram_bimodality(two_materials) < 1.0
    # One material has no second mode, which is an absent measurement rather
    # than a large number.
    assert histogram_bimodality([100.0] * 1000) is None


def test_the_frozen_profile_says_what_it_is_and_what_it_is_not() -> None:
    """The bands are a calibration somebody else measured. The file has to
    carry that, because a number with no provenance is a number nobody can
    argue with."""
    profile = json.loads(
        (ROOT / "framework/profiles/01-segmentation/lamina-gate-1.0.0.json").read_text())

    assert profile["thickness_um"]["sheet_high"] < 120.0, (
        "the band must exclude the fused population measured at 122-169 um")
    assert profile["calibration"]["reimplemented_here"] is True
    assert "not been compared" in profile["calibration"]["reimplementation_note"]
    assert any("not a claim about content" in claim
               for claim in profile["non_claims"])


def test_the_profile_bands_read_the_calibration_they_came_from() -> None:
    """Each observation the profile records must land on the side its reading
    says -- otherwise the bands and the evidence in the same file disagree."""
    profile = json.loads(
        (ROOT / "framework/profiles/01-segmentation/lamina-gate-1.0.0.json").read_text())
    low = float(profile["thickness_um"]["sheet_low"])
    high = float(profile["thickness_um"]["sheet_high"])

    for row in profile["calibration"]["observations"]:
        thickness = float(row["median_thickness_um"])
        inside = low <= thickness <= high
        assert inside == (row["reading"] == "one sheet"), row


def test_a_column_that_never_leaves_the_material_is_a_slab() -> None:
    """The clearest fused surface is the one a column cannot get out of.

    Against a level measured over the whole surface -- which is the only place
    it can come from, since a column of constant material has no two levels of
    its own to sit between.
    """
    inside = [200.0] * 40

    column = column_material_depth(inside, sample_step_um=4.0, level=105.0)

    assert column["state"] == "SATURATED"
    assert column["window_um"] == 160.0


def test_mostly_saturated_columns_are_fused_not_unmeasurable() -> None:
    saturated = [column_material_depth([200.0] * 40, sample_step_um=4.0, level=105.0)
                 for _ in range(40)]
    measured = [column_material_depth(sheet(8), sample_step_um=4.0, level=105.0)
                for _ in range(10)]

    outcome = assess_lamina(saturated + measured, profile=PROFILE, bimodality=0.5)

    assert outcome["state"] == "LAMINA_FUSED"
    assert outcome["saturated_fraction"] == 0.8
    assert "160 um" in outcome["reason"]


def test_the_level_is_the_middle_of_the_valley() -> None:
    """Between two separated materials the valley is wide and empty, and its
    first bin sits hard against the air mode -- which measures the outer edge
    of every interface ramp and reads a sheet several microns too thick."""
    two_materials = [10.0] * 500 + [200.0] * 500

    level = interface_level(two_materials)

    assert 80.0 < level < 130.0


def test_one_population_has_no_level_and_says_so() -> None:
    assert interface_level([200.0] * 500) is None
