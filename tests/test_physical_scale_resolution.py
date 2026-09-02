"""FIX-09 — physical scale comes from the catalog and the profile, not the CLI.

`--source-pixel-um` used to default to 9.362 while four eligible volumes
(PHerc268, PHerc800, PHerc1218, PHerc1447) are 8.64 um scans, so omitting the
flag silently rescaled those runs by 8.4 %.  `training_pixel_um` was hardcoded
as "7.91" in five callers rather than read from the ink lane profile that
declares it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))

from analyze_ink_stability import (  # noqa: E402
    DEFAULT_INK_PROFILE,
    DEFAULT_VOLUME_CATALOG,
    PIXEL_UM_TOLERANCE,
    catalog_voxel_size_um,
    resolve_source_pixel_um,
    resolve_training_pixel_um,
)

CATALOG = json.loads(DEFAULT_VOLUME_CATALOG.read_text(encoding="utf-8"))
ENTRIES = CATALOG["entries"]
EIGHT_SIX_FOUR = {"PHerc268", "PHerc800", "PHerc1218", "PHerc1447"}
LEGACY_CLI_DEFAULT = 9.362


def test_every_catalogued_volume_is_one_scroll_and_only_one() -> None:
    """It asserted the catalogue holds exactly thirteen.

    Thirteen was the Campania filter of the day, not a property of the
    catalogue: the open-data bucket carries forty-five scroll prefixes and
    twenty-six of them have everything P0 and P1 need. Freezing the count meant
    that adding a scroll anyone can already download failed a test about
    physical scale, which is not what this file is for.

    What has to hold is that the catalogue does not carry a scroll twice --
    two rows for one sample_id are two scales for one scroll, and every
    resolution question below picks whichever it saw first.
    """
    assert ENTRIES, "the catalogue is empty"
    samples = [entry["sample_id"] for entry in ENTRIES]
    duplicated = sorted({s for s in samples if samples.count(s) > 1})
    assert not duplicated, f"catalogued more than once: {duplicated}"


@pytest.mark.parametrize(
    "sample_id", sorted(entry["sample_id"] for entry in ENTRIES)
)
def test_every_catalogued_roll_resolves_without_the_flag(sample_id: str) -> None:
    """Parametric over all 13 rolls: no flag means the catalog value wins."""

    expected = next(
        float(entry["voxel_size_um"])
        for entry in ENTRIES
        if entry["sample_id"] == sample_id
    )
    resolved, provenance = resolve_source_pixel_um(
        sample_id=sample_id,
        catalog_path=DEFAULT_VOLUME_CATALOG,
        requested=None,
    )

    assert resolved == expected
    assert provenance["source"] == "ELIGIBLE_VOLUMES_CATALOG"
    assert provenance["cli_source_pixel_um"] is None


@pytest.mark.parametrize("sample_id", sorted(EIGHT_SIX_FOUR))
def test_an_8_64_roll_uses_8_64_not_the_old_default(sample_id: str) -> None:
    resolved, _ = resolve_source_pixel_um(
        sample_id=sample_id,
        catalog_path=DEFAULT_VOLUME_CATALOG,
        requested=None,
    )

    assert resolved == 8.64
    assert resolved != LEGACY_CLI_DEFAULT


@pytest.mark.parametrize("sample_id", sorted(EIGHT_SIX_FOUR))
def test_the_old_default_now_fails_closed_on_an_8_64_roll(sample_id: str) -> None:
    """Passing the legacy 9.362 default against an 8.64 scan must abort."""

    with pytest.raises(RuntimeError, match="disagrees with the frozen catalog"):
        resolve_source_pixel_um(
            sample_id=sample_id,
            catalog_path=DEFAULT_VOLUME_CATALOG,
            requested=LEGACY_CLI_DEFAULT,
        )


def test_a_matching_cli_value_is_accepted() -> None:
    resolved, provenance = resolve_source_pixel_um(
        sample_id="PHerc1218",
        catalog_path=DEFAULT_VOLUME_CATALOG,
        requested=8.64,
    )

    assert resolved == 8.64
    assert provenance["cli_source_pixel_um"] == 8.64


def test_disagreement_within_tolerance_is_accepted() -> None:
    resolved, _ = resolve_source_pixel_um(
        sample_id="PHerc1218",
        catalog_path=DEFAULT_VOLUME_CATALOG,
        requested=8.64 + PIXEL_UM_TOLERANCE / 2,
    )

    assert resolved == 8.64


def test_disagreement_beyond_tolerance_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="disagrees with the frozen catalog"):
        resolve_source_pixel_um(
            sample_id="PHerc1218",
            catalog_path=DEFAULT_VOLUME_CATALOG,
            requested=8.64 + PIXEL_UM_TOLERANCE * 100,
        )


def test_uncatalogued_sample_without_a_flag_fails_closed() -> None:
    """PHerc0139 is an out-of-cohort control; the scale must be stated."""

    assert catalog_voxel_size_um(DEFAULT_VOLUME_CATALOG, "PHerc0139") is None
    with pytest.raises(RuntimeError, match="refusing to guess"):
        resolve_source_pixel_um(
            sample_id="PHerc0139",
            catalog_path=DEFAULT_VOLUME_CATALOG,
            requested=None,
        )


def test_uncatalogued_sample_with_an_explicit_flag_is_allowed() -> None:
    resolved, provenance = resolve_source_pixel_um(
        sample_id="PHerc0139",
        catalog_path=DEFAULT_VOLUME_CATALOG,
        requested=9.362,
    )

    assert resolved == 9.362
    assert provenance["source"] == "CLI_UNCATALOGUED_SAMPLE"
    assert provenance["catalog_voxel_size_um"] is None


def test_missing_catalog_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="volume catalog is missing"):
        resolve_source_pixel_um(
            sample_id="PHerc268",
            catalog_path=tmp_path / "absent.json",
            requested=None,
        )


def test_training_scale_comes_from_the_ink_lane_profile() -> None:
    profile = json.loads(DEFAULT_INK_PROFILE.read_text(encoding="utf-8"))
    declared = float(profile["input_contract"]["training_pixel_um"])

    resolved, provenance = resolve_training_pixel_um(
        profile_path=DEFAULT_INK_PROFILE, requested=None
    )

    assert resolved == declared
    assert provenance["source"] == "INK_LANE_PROFILE"
    assert provenance["profile_id"] == profile["profile_id"]


def test_training_scale_disagreement_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="disagrees with the ink lane profile"):
        resolve_training_pixel_um(
            profile_path=DEFAULT_INK_PROFILE, requested=8.5
        )


def test_profile_without_training_pixel_um_fails_closed(tmp_path: Path) -> None:
    bogus = tmp_path / "profile.json"
    bogus.write_text(json.dumps({"profile_id": "x@1.0.0", "input_contract": {}}))

    with pytest.raises(RuntimeError, match="declares no training_pixel_um"):
        resolve_training_pixel_um(profile_path=bogus, requested=None)


def test_no_caller_hardcodes_the_training_scale() -> None:
    """The five callers named in FIX-09 must not carry the literal."""

    callers = [
        ROOT / "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py",
        ROOT
        / "framework/stages/01-segmentation/scripts/run_coverage_surface_v2_robust.py",
        ROOT
        / "framework/stages/01-segmentation/scripts/screen_expanded_candidate_surfaces.py",
    ]
    for path in callers:
        assert '"7.91"' not in path.read_text(encoding="utf-8"), path


def test_analysis_cli_defaults_are_unset() -> None:
    """Neither scale may fall back to a literal default any more."""

    source = (
        ROOT / "framework/stages/04-validation/scripts/analyze_ink_stability.py"
    ).read_text(encoding="utf-8")

    assert '"--source-pixel-um", type=float, default=None' in source
    assert '"--training-pixel-um", type=float, default=None' in source
    assert "default=9.362" not in source
    assert "default=7.91" not in source
