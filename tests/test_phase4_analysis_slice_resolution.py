import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "framework/stages/04-validation/scripts"),
)

from analyze_ink_stability import ensure_stage03_helper_on_path, resolve_tiff_slice


def touch_tiffs(directory: Path, names: list[str]) -> None:
    directory.mkdir()
    for name in names:
        (directory / name).touch()


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["16.tif", "17.tif", "18.tif"], "17.tif"),
        (["0016.tif", "0017.tif", "0018.tif"], "0017.tif"),
        (["016.tif", "0017.tif", "00018.tif"], "0017.tif"),
    ],
)
def test_resolve_tiff_slice_ignores_zero_padding(
    tmp_path: Path,
    names: list[str],
    expected: str,
) -> None:
    source = tmp_path / "tiffs"
    touch_tiffs(source, names)

    assert resolve_tiff_slice(source, 17).name == expected


def test_resolve_tiff_slice_rejects_ambiguous_indices(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiffs"
    touch_tiffs(source, ["17.tif", "0017.tif"])

    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_tiff_slice(source, 17)


def test_resolve_tiff_slice_reports_missing_index(tmp_path: Path) -> None:
    source = tmp_path / "tiffs"
    touch_tiffs(source, ["0016.tif", "0018.tif"])

    with pytest.raises(RuntimeError, match="slice index 17"):
        resolve_tiff_slice(source, 17)


def test_validation_declares_stage03_glyph_helper_dependency() -> None:
    directory = ensure_stage03_helper_on_path()
    assert (directory / "annotate_glyph_candidates.py").is_file()
