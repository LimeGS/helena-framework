"""Running the fit twice is the only uncertainty this geometry has.

The spiral fit publishes none and its paper reports no run-to-run variability,
so a fitted surface is a number without an error bar. The second run costs one
override -- `random_seed` is one of upstream's own config keys -- and what comes
back has to be reported carefully, because four ways of reporting it were got
wrong on this corpus first.

Averaged rather than summed: the two Chamfer conventions differ by exactly two,
and 258 um was carried for half a campaign where the value was 121.

Decomposed: 75 um along a sheet does not leave it and 35 um through it does. The
normal is the headline because it decides whether a render samples the right
lamina; the lateral stays visible because it is the error bar on P8 and P9
claiming that these letters sit at this place on the page.

Normalized by lamina thickness rather than the winding pitch: dividing by 371 um
produced "0.33 laminae" when in depth it was a third of one sheet.

Per z band: on PHerc0826 the agreement ran 94 to 157 um across one surface, and
a mean hides which turns are worth trusting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

import seed_agreement as agreement  # noqa: E402

VOXEL_UM = 9.362


def sheet(directory: Path, *, rows: int = 40, columns: int = 40,
          normal_offset: float = 0.0, lateral_offset: float = 0.0,
          z0: float = 500.0) -> Path:
    """A flat patch in the plane z = z0, whose normal is the z axis.

    So a shift in z is purely normal and a shift in x is purely lateral, and the
    decomposition has a right answer rather than a plausible one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    r, c = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    tifffile.imwrite(directory / "x.tif",
                     (200.0 + 4.0 * c + lateral_offset).astype(np.float32))
    tifffile.imwrite(directory / "y.tif", (300.0 + 4.0 * r).astype(np.float32))
    tifffile.imwrite(directory / "z.tif",
                     np.full((rows, columns), z0 + normal_offset, np.float32))
    return directory


# -- the decomposition -----------------------------------------------------

def test_a_pure_depth_shift_is_all_normal(tmp_path):
    """Two voxels through the sheet: the number that decides whether a render
    samples the right lamina."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=2.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert out["normal_um"]["median"] == pytest.approx(2 * VOXEL_UM, rel=0.02)
    assert out["lateral_um"]["median"] < 0.5


def test_a_pure_slide_along_the_sheet_is_all_lateral(tmp_path):
    """Sliding does not take you off the sheet, and reporting it as depth is
    how a harmless run reads as a failure."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", lateral_offset=2.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert out["normal_um"]["median"] < 0.5
    assert out["lateral_um"]["median"] > 0.0


def test_the_two_components_are_the_legs_of_the_total(tmp_path):
    """Lateral is what is left once the normal part is removed, not the total."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=3.0, lateral_offset=4.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)
    normal = out["normal_um"]["median"]
    lateral = out["lateral_um"]["median"]
    total = out["total_um"]["chamfer_um"]

    assert (normal ** 2 + lateral ** 2) ** 0.5 == pytest.approx(total, rel=0.1)


# -- the conventions -------------------------------------------------------

def test_the_chamfer_is_averaged_not_summed(tmp_path):
    """The two conventions differ by two, and the campaign reported 258 where
    the averaged value was 121."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=2.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    # Each direction is 2 voxels, so the average is 2 and the sum would be 4.
    assert out["total_um"]["chamfer_voxels"] == pytest.approx(2.0, rel=0.02)
    assert "averaged not" in out["total_um"]["convention"]


def test_it_normalizes_by_lamina_thickness_not_by_winding_pitch(tmp_path):
    """Dividing by the 371 um pitch produced "0.33 laminae" for something that
    was a third of one sheet."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=2.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert out["lamina_thickness_um"] == agreement.SHEET_REFERENCE_UM
    assert out["normal_um"]["normal_in_sheet_thicknesses"] == pytest.approx(
        out["normal_um"]["median"] / agreement.SHEET_REFERENCE_UM, rel=1e-6)
    # And a surface that measured its own thickness uses that instead.
    measured = agreement.measure(a, b, voxel_um=VOXEL_UM,
                                 lamina_thickness_um=44.7)
    assert measured["lamina_thickness_measured"] is True
    assert (measured["normal_um"]["normal_in_sheet_thicknesses"]
            < out["normal_um"]["normal_in_sheet_thicknesses"])


def test_the_median_never_travels_without_its_tail(tmp_path):
    """Everything this campaign measured had a good median and a poor tail:
    interface localisation is 1.4 um at the median and 128 at p90."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=2.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert {"median", "p90", "p99",
            "normal_in_sheet_thicknesses"} == set(out["normal_um"])
    assert {"median", "p90"} == set(out["lateral_um"])


# -- per band --------------------------------------------------------------

def test_the_bands_are_reported_separately(tmp_path):
    """On PHerc0826 the agreement ran 94 to 157 um across one surface."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for directory, offset in ((a, 0.0), (b, 0.0)):
        directory.mkdir(parents=True)
        r, c = np.meshgrid(np.arange(60), np.arange(40), indexing="ij")
        tifffile.imwrite(directory / "x.tif", (200.0 + 4.0 * c).astype(np.float32))
        tifffile.imwrite(directory / "y.tif", (300.0 + 4.0 * r).astype(np.float32))
        # z varies down the rows, so the bands are real slabs of the volume.
        tifffile.imwrite(directory / "z.tif",
                         (500.0 + 4.0 * r + offset).astype(np.float32))
    # The second seed drifts only in the lower half.
    z = tifffile.imread(b / "z.tif")
    z[30:, :] += 6.0
    tifffile.imwrite(b / "z.tif", z)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM, z_bands=3)
    bands = [band for band in out["by_z_band"] if "normal_um" in band]

    assert len(bands) == 3
    assert bands[0]["normal_um"] < bands[-1]["normal_um"], (
        "a drift confined to one band has to show up in that band")


# -- the failure that looks like success ----------------------------------

def test_two_identical_surfaces_are_a_seed_that_did_not_take(tmp_path):
    """The one metric here whose failure disguises itself as its best result.

    Every other measurement in this framework fails downward. This one fails
    upward: an override that reached nothing produces two identical fits and an
    agreement of zero, which reads as perfect reproducibility. Two independent
    stochastic optimizations do not agree bit for bit.
    """
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b")            # same geometry, different directory

    with pytest.raises(agreement.AgreementUnmeasurable,
                       match="seed override did not take"):
        agreement.measure(a, b, voxel_um=VOXEL_UM)


def test_the_refusal_names_the_key_that_was_renamed(tmp_path):
    """`random_seed` at 05dcf034 and `optimizer_random_seed` later. Which one
    reaches the optimizer depends on the pinned commit, and the wrong one is
    exactly how two fits come out identical."""
    a, b = sheet(tmp_path / "a"), sheet(tmp_path / "b")

    with pytest.raises(agreement.AgreementUnmeasurable) as refused:
        agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert "optimizer_random_seed" in str(refused.value)


def test_agreement_far_closer_than_two_fits_ever_land_is_refused(tmp_path):
    """Not only bit-identical. The campaign's own pairs sit at 9 to 17 voxels;
    a tenth of one is a run that did not vary, however it got there."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=0.01)

    with pytest.raises(agreement.AgreementUnmeasurable, match="closer than"):
        agreement.measure(a, b, voxel_um=VOXEL_UM)


def test_a_real_pair_is_not_caught_by_that_guard(tmp_path):
    """The guard has to sit below where real pairs land, or it refuses the
    measurement it exists to protect."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=1.3)   # about the campaign's own

    assert agreement.measure(a, b, voxel_um=VOXEL_UM)["state"] == agreement.MEASURED


def test_the_seed_key_is_read_from_the_commit_that_will_run(tmp_path):
    """Naming one in a profile pins the profile to a commit range nobody wrote
    down, and the wrong name is a KeyError today and a silent identical pair the
    day that validation is loosened.

    At 23adee04 the seed lives on spiral-fitting/config.py's Config class,
    read by importing it directly rather than parsing a default_config dict
    out of fit_spiral.py's AST -- config.py does no dataset I/O, so importing
    it is as cheap as parsing was."""
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))
    import adapter

    pinned = ROOT / "vendor/villa/spiral-fitting"
    if (pinned / "config.py").is_file():
        assert adapter.seed_key(pinned) == "optimizer_random_seed"

    later = tmp_path / "later"
    later.mkdir()
    (later / "config.py").write_text(
        "class Config:\n"
        "    def __init__(self, overrides=None):\n"
        "        self.optimizer_random_seed = 1\n"
        "    def as_dict(self):\n"
        "        return vars(self).copy()\n")
    assert adapter.seed_key(later) == "optimizer_random_seed"

    neither = tmp_path / "neither"
    neither.mkdir()
    (neither / "config.py").write_text(
        "class Config:\n"
        "    def __init__(self, overrides=None):\n"
        "        self.input_disable_patches = False\n"
        "    def as_dict(self):\n"
        "        return vars(self).copy()\n")
    with pytest.raises(adapter.UnknownOverrideKey, match="named something else"):
        adapter.seed_key(neither)


# -- what it refuses to answer --------------------------------------------

def test_the_same_surface_twice_is_refused(tmp_path):
    """A glob over a directory holding both seeds can pick one twice and answer
    a very convincing zero."""
    a = sheet(tmp_path / "a")

    with pytest.raises(agreement.AgreementUnmeasurable, match="same surface"):
        agreement.measure(a, a, voxel_um=VOXEL_UM)


def test_a_surface_with_no_second_run_is_unpaired_not_zero(tmp_path):
    """Never measured is different from measured and small."""
    out = agreement.unpaired(tmp_path / "a")

    assert out["state"] == agreement.UNPAIRED
    assert "UNPAIRED" in agreement.headline(out)
    # A state, not a missing field: no error bar is a different thing from a
    # large one. The large one can be defended with its number beside it.
    assert out["defensible"] is False
    assert "different thing from having a large one" in out["why_not"]


def test_a_comparison_without_a_voxel_size_is_refused(tmp_path):
    a, b = sheet(tmp_path / "a"), sheet(tmp_path / "b", normal_offset=1.0)

    with pytest.raises(agreement.AgreementUnmeasurable, match="voxel size"):
        agreement.measure(a, b, voxel_um=0)


# -- how it reads ----------------------------------------------------------

def test_the_headline_leads_with_the_normal_and_never_shows_the_total_alone(tmp_path):
    """The total is the actively misleading number: 121 um against a 35 um
    sheet reads as failure when the answer to the question is 12."""
    a = sheet(tmp_path / "a")
    b = sheet(tmp_path / "b", normal_offset=1.0, lateral_offset=8.0)

    line = agreement.headline(agreement.measure(a, b, voxel_um=VOXEL_UM))

    assert line.index("normal") < line.index("lateral") < line.index("total")
    assert "reproducibility, not correctness" in line


def test_the_depth_tolerance_travels_as_a_number_not_a_verdict(tmp_path):
    """+-51 um is the offset at which the ink score fell 43%: a degradation
    curve. Nothing here turns it into a pass mark."""
    a, b = sheet(tmp_path / "a"), sheet(tmp_path / "b", normal_offset=1.0)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert out["ink_depth_tolerance_um"] == 51.0
    assert not any(key in out for key in ("passed", "within_tolerance", "ok"))
    assert any("degradation curve" in claim for claim in out["non_claims"])


def test_it_says_out_loud_that_it_is_not_a_defect_detector(tmp_path):
    """Measured on this corpus: the rows-160-250 band of w015, the one with 830
    fold-backs and real self-contact, had the *best* seed agreement of the three
    bands -- 90.9 um against 93.1 and 95.5."""
    a, b = sheet(tmp_path / "a"), sheet(tmp_path / "b", normal_offset=1.0)

    claims = agreement.measure(a, b, voxel_um=VOXEL_UM)["non_claims"]

    assert any("reproducibility, not correctness" in claim for claim in claims)


# -- how it reaches a table ------------------------------------------------

def test_the_table_cell_is_a_state_and_one_number(tmp_path):
    """Four scannable judgements beside a cell of four numbers is a table
    nobody can sweep. The decomposition lives in the surface's detail."""
    a, b = sheet(tmp_path / "a"), sheet(tmp_path / "b", normal_offset=1.3)

    line = agreement.cell(agreement.measure(a, b, voxel_um=VOXEL_UM))

    assert line.endswith("um normal")
    assert "lateral" not in line and "total" not in line and "p90" not in line


def test_an_unpaired_surface_reads_as_a_state_in_the_cell(tmp_path):
    assert agreement.cell(agreement.unpaired(tmp_path / "a")) == "UNPAIRED"


def test_the_normalised_form_is_named_for_its_divisor(tmp_path):
    """A collision worth naming. The campaign published "0.33 laminae" for the
    *total* over the *winding pitch* (121/371 = 0.326); this is the *normal*
    over the *sheet thickness* (12/35.5 = 0.338). They round alike, share no
    term, and mean different things -- so the bare word is not used.
    """
    a, b = sheet(tmp_path / "a"), sheet(tmp_path / "b", normal_offset=1.3)

    out = agreement.measure(a, b, voxel_um=VOXEL_UM)

    assert "normal_in_sheet_thicknesses" in out["normal_um"]
    assert "in_laminae" not in out["normal_um"]
    assert out["normalisation"]["divided_by"] == "lamina thickness"
    assert out["normalisation"]["not"] == "the winding pitch"
    assert "0.326" in out["normalisation"]["collision"]
    # And it does not reach the cell at all: there is no room there to say
    # which divisor it was.
    assert "thickness" not in agreement.cell(out)


def test_the_fifth_judgement_is_a_state_of_its_own_in_the_store():
    """Separate from the other four so it can contradict them, which on this
    corpus it does."""
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.store import DEFAULT_SEED_AGREEMENT_STATE, SEED_AGREEMENT_STATES

    assert DEFAULT_SEED_AGREEMENT_STATE == agreement.UNPAIRED
    assert set(SEED_AGREEMENT_STATES) == {
        agreement.MEASURED, agreement.UNPAIRED, agreement.UNMEASURED,
        agreement.NOT_A_PAIR}


def test_the_migration_backfills_to_unpaired_rather_than_to_a_pass():
    """Every row that exists had one run. Saying so is different from saying
    the error bar is small."""
    sql = (ROOT / "framework/stages/01-segmentation/fleet/migrations"
           / "001_postgresql.sql").read_text(encoding="utf-8")

    assert "seed_agreement_state text" in sql
    assert "DEFAULT 'SEED_UNPAIRED'" in sql
    assert "VALUES (26," in sql
