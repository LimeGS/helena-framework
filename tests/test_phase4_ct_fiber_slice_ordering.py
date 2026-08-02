import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "framework/stages/04-validation/scripts"),
)

import extract_ct_fiber_features as extractor
from extract_ct_fiber_features import ordered_tiff_files


def touch_tiffs(directory: Path, names: list[str]) -> None:
    directory.mkdir()
    for name in names:
        (directory / name).touch()


def test_numeric_tiff_stems_are_ordered_as_slice_indices(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numeric"
    touch_tiffs(source, ["10.tif", "2.tif", "1.tif"])

    files, ordering = ordered_tiff_files(source)

    assert [path.name for path in files] == ["1.tif", "2.tif", "10.tif"]
    assert ordering == "NUMERIC_STEM_ASCENDING"


def test_ambiguous_numeric_tiff_stems_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous"
    touch_tiffs(source, ["1.tif", "01.tif"])

    with pytest.raises(RuntimeError, match="ambiguous"):
        ordered_tiff_files(source)


def test_non_numeric_tiff_stems_keep_filename_order(tmp_path: Path) -> None:
    source = tmp_path / "named"
    touch_tiffs(source, ["slice-b.tif", "slice-a.tif"])

    files, ordering = ordered_tiff_files(source)

    assert [path.name for path in files] == [
        "slice-a.tif",
        "slice-b.tif",
    ]
    assert ordering == "LEXICOGRAPHIC_FILENAME"


def test_empty_morphology_candidate_set_writes_a_valid_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    tiffs = root / "tiffs"
    tiffs.mkdir(parents=True)
    for index in range(3):
        Image.fromarray(
            np.full((8, 8), index, dtype=np.uint8)
        ).save(tiffs / f"{index}.tif")
    analysis = root / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "input": {"shape_y_x": [8, 8]},
                "text_like_screening": {"candidates": []},
            }
        )
    )
    spec = root / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "patch_radius_um": 10.0,
                "groups": [
                    {
                        "group_id": "negative-control",
                        "class": "HARD_NEGATIVE",
                        "tiff_directory": "tiffs",
                        "analysis": "analysis.json",
                        "central_slice": 1,
                        "voxel_um": 1.0,
                    }
                ],
            }
        )
    )
    output = root / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_ct_fiber_features.py",
            "--root",
            str(root),
            "--spec",
            str(spec),
            "--output",
            str(output),
        ],
    )

    assert extractor.main() == 0

    receipt = json.loads(
        (output / "CT_FIBER_FEATURE_BENCHMARK.json").read_text()
    )
    assert receipt["row_count"] == 0
    assert receipt["sources"][0]["candidate_count"] == 0
    assert receipt["sources"][0]["slice_ordering"] == (
        "NUMERIC_STEM_ASCENDING"
    )
    with (output / "CT_FIBER_FEATURES.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
