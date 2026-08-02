"""Regression checks for the shared CT slice-order contract (FIX-02).

The bug these tests exist for: an unpadded ``0.tif … 64.tif`` render sorted
lexicographically yields ``0, 1, 10, 11, …, 19, 2, 20, …`` so position 32 is
``38.tif``.  Every reader that indexed a stack by position therefore rendered,
hashed and reported a depth six slices away from the one it declared.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import (  # noqa: E402
    EMPTY_TIFF_DIRECTORY,
    LEXICOGRAPHIC_FILENAME,
    NUMERIC_STEM_ASCENDING,
    NUMERIC_STEM_CONTIGUOUS_ASCENDING,
    SLICE_ORDERING_POLICIES,
    SliceOrderError,
    ordered_tiff_files,
    ordered_tiff_stack_position,
    resolve_tiff_slice,
)


PRODUCTION_STACK_READERS = (
    "framework/stages/04-validation/scripts/build_orthogonal_candidate_review.py",
    "framework/stages/04-validation/scripts/build_strict_signal_ct_spec.py",
    "framework/stages/04-validation/scripts/helena_build_surface_qc_ledger.py",
    "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py",
    "framework/stages/04-validation/scripts/build_high_recall_ct_application.py",
    "framework/stages/04-validation/scripts/restore_geometry_high_recall_maps.py",
    "framework/stages/04-validation/scripts/extract_ct_fiber_features.py",
    "framework/stages/01-segmentation/scripts/scan_large_surface_windows.py",
    "framework/stages/01-segmentation/scripts/run_coverage_surface_v2.py",
    "framework/stages/03-ink/scripts/rank_coarse_ink_windows.py",
    "framework/stages/03-ink/scripts/rank_coarse_ink_windows_v2.py",
    "framework/stages/03-ink/scripts/run_ink_timesformer.py",
    "framework/stages/03-ink/scripts/run_official_gp_scroll1_robust_chain.py",
    "framework/stages/03-ink/scripts/run_official_gp_scroll1_rescreen.py",
    "framework/stages/02-flattening/scripts/crop_render_window.py",
    "framework/stages/02-flattening/scripts/run.py",
    "framework/stages/06-discovery/scripts/compact_robust_window.py",
    "framework/stages/06-discovery/scripts/run_pherc1667_iteration0.py",
    "scripts/harness/run_geometry_recovery_screen.py",
)


def touch(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def unpadded_65(directory: Path) -> None:
    touch(directory, [f"{index}.tif" for index in range(65)])


def test_unpadded_65_slice_render_indexes_the_physical_slice(
    tmp_path: Path,
) -> None:
    """The exact failure of the orthogonal adjudication panel."""

    unpadded_65(tmp_path)

    files, ordering = ordered_tiff_files(tmp_path)

    assert ordering == NUMERIC_STEM_ASCENDING
    assert len(files) == 65
    assert files[32].name == "32.tif"
    assert [path.name for path in files[:4]] == [
        "0.tif",
        "1.tif",
        "2.tif",
        "3.tif",
    ]
    assert files[-1].name == "64.tif"


def test_the_old_lexicographic_reader_is_off_by_six_slices(
    tmp_path: Path,
) -> None:
    """Pin the defect so the fix cannot be reverted silently.

    This is the implementation that shipped in
    ``build_orthogonal_candidate_review.py:41``.  It must disagree with
    the contract at the declared central slice.
    """

    unpadded_65(tmp_path)

    legacy = sorted(tmp_path.glob("*.tif"))
    contract, _ = ordered_tiff_files(tmp_path)

    assert legacy[32].name == "38.tif"
    assert contract[32].name == "32.tif"
    assert legacy[32] != contract[32]


def test_padding_ambiguity_fails_closed(tmp_path: Path) -> None:
    touch(tmp_path, ["1.tif", "01.tif"])

    with pytest.raises(SliceOrderError, match="ambiguous"):
        ordered_tiff_files(tmp_path)


def test_one_non_numeric_stem_fails_instead_of_taking_a_reserve_bucket(
    tmp_path: Path,
) -> None:
    """No ``else 1_000_000`` / ``10_000`` / ``10**9`` fallback survives."""

    touch(tmp_path, ["0.tif", "1.tif", "2.tif", "mask.tif"])

    with pytest.raises(SliceOrderError, match="mixes numeric and non-numeric"):
        ordered_tiff_files(tmp_path)


def test_a_wholly_non_numeric_directory_keeps_filename_order(
    tmp_path: Path,
) -> None:
    touch(tmp_path, ["slice-b.tif", "slice-a.tif"])

    files, ordering = ordered_tiff_files(tmp_path)

    assert [path.name for path in files] == ["slice-a.tif", "slice-b.tif"]
    assert ordering == LEXICOGRAPHIC_FILENAME


def test_require_numeric_rejects_a_filename_ordered_directory(
    tmp_path: Path,
) -> None:
    touch(tmp_path, ["slice-a.tif", "slice-b.tif"])

    with pytest.raises(SliceOrderError, match="requires numeric TIFF stems"):
        ordered_tiff_files(tmp_path, require_numeric=True)


def test_contiguity_is_opt_in_and_fails_closed(tmp_path: Path) -> None:
    touch(tmp_path, ["0.tif", "2.tif"])

    files, ordering = ordered_tiff_files(tmp_path)
    assert [path.name for path in files] == ["0.tif", "2.tif"]
    assert ordering == NUMERIC_STEM_ASCENDING

    with pytest.raises(SliceOrderError, match="contiguous"):
        ordered_tiff_files(tmp_path, require_contiguous=True)


def test_contiguous_numeric_render_reports_its_own_policy(
    tmp_path: Path,
) -> None:
    unpadded_65(tmp_path)

    _, ordering = ordered_tiff_files(tmp_path, require_contiguous=True)

    assert ordering == NUMERIC_STEM_CONTIGUOUS_ASCENDING


def test_empty_directory_fails_closed_unless_allowed(tmp_path: Path) -> None:
    tmp_path.joinpath("empty").mkdir()

    with pytest.raises(SliceOrderError, match="no TIFF slices"):
        ordered_tiff_files(tmp_path / "empty")

    files, ordering = ordered_tiff_files(tmp_path / "empty", allow_empty=True)
    assert files == []
    assert ordering == EMPTY_TIFF_DIRECTORY


def test_extra_suffixes_are_opt_in(tmp_path: Path) -> None:
    touch(tmp_path, ["0.tif", "1.tiff"])

    files, _ = ordered_tiff_files(tmp_path)
    assert [path.name for path in files] == ["0.tif"]

    files, _ = ordered_tiff_files(tmp_path, suffixes=(".tif", ".tiff"))
    assert [path.name for path in files] == ["0.tif", "1.tiff"]


def test_every_policy_name_is_declared() -> None:
    for policy in (
        NUMERIC_STEM_ASCENDING,
        NUMERIC_STEM_CONTIGUOUS_ASCENDING,
        LEXICOGRAPHIC_FILENAME,
        EMPTY_TIFF_DIRECTORY,
    ):
        assert policy in SLICE_ORDERING_POLICIES


def test_resolve_tiff_slice_is_immune_to_zero_padding(tmp_path: Path) -> None:
    padded = tmp_path / "padded"
    unpadded = tmp_path / "unpadded"
    touch(padded, [f"{index:03d}.tif" for index in range(65)])
    touch(unpadded, [f"{index}.tif" for index in range(65)])

    assert resolve_tiff_slice(padded, 32).name == "032.tif"
    assert resolve_tiff_slice(unpadded, 32).name == "32.tif"
    assert int(resolve_tiff_slice(padded, 32).stem) == int(
        resolve_tiff_slice(unpadded, 32).stem
    )


def test_resolve_tiff_slice_fails_closed_on_ambiguity_and_gaps(
    tmp_path: Path,
) -> None:
    ambiguous = tmp_path / "ambiguous"
    touch(ambiguous, ["32.tif", "032.tif"])
    with pytest.raises(SliceOrderError, match="ambiguous"):
        resolve_tiff_slice(ambiguous, 32)

    mixed = tmp_path / "mixed"
    touch(mixed, ["0.tif", "mask.tif"])
    with pytest.raises(SliceOrderError, match="non-numeric"):
        resolve_tiff_slice(mixed, 0)

    sparse = tmp_path / "sparse"
    touch(sparse, ["0.tif", "1.tif"])
    with pytest.raises(SliceOrderError, match="missing TIFF slice index 32"):
        resolve_tiff_slice(sparse, 32)


def test_stack_position_maps_a_physical_slice_onto_the_array_axis(
    tmp_path: Path,
) -> None:
    unpadded_65(tmp_path)
    files, _ = ordered_tiff_files(tmp_path)

    assert ordered_tiff_stack_position(files, 32) == 32
    assert files[ordered_tiff_stack_position(files, 32)].name == "32.tif"

    padded = tmp_path / "padded"
    touch(padded, [f"{index:03d}.tif" for index in range(65)])
    padded_files, _ = ordered_tiff_files(padded)
    assert padded_files[ordered_tiff_stack_position(padded_files, 32)].name == (
        "032.tif"
    )


def test_stack_position_fails_closed_on_a_missing_physical_slice(
    tmp_path: Path,
) -> None:
    touch(tmp_path, ["10.tif", "11.tif"])
    files, _ = ordered_tiff_files(tmp_path)

    with pytest.raises(SliceOrderError, match="missing TIFF slice index 32"):
        ordered_tiff_stack_position(files, 32)


@pytest.mark.parametrize("relative", PRODUCTION_STACK_READERS)
def test_no_production_stack_reader_keeps_a_local_ordering_policy(
    relative: str,
) -> None:
    """Acceptance criterion of FIX-02: one contract, no private copies."""

    source = (ROOT / relative).read_text(encoding="utf-8")

    assert 'sorted(directory.glob("*.tif"))' not in source
    assert 'sorted(tiff_dir.glob("*.tif"))' not in source
    assert 'sorted(input_dir.glob("*.tif"))' not in source
    assert 'sorted(tiff_root.glob("*.tif"))' not in source
    assert 'sorted(render.glob("*.tif"))' not in source
    for bucket in ("1_000_000", "10_000", "10**9"):
        assert f"else {bucket}" not in source
    assert "framework.contracts.slice_order" in source


def test_the_three_former_copies_now_re_export_the_contract() -> None:
    """``ordered_tiff_files`` must be the same object everywhere."""

    for stage in ("01-segmentation", "03-ink", "04-validation", "06-discovery"):
        path = str(ROOT / "framework/stages" / stage / "scripts")
        if path not in sys.path:
            sys.path.insert(0, path)

    timesformer = importlib.import_module("run_ink_timesformer")
    fiber = importlib.import_module("extract_ct_fiber_features")

    assert timesformer.ordered_tiff_files is ordered_tiff_files
    assert fiber.ordered_tiff_files is ordered_tiff_files
