"""Containment behaviour for geometry high-recall receipt paths."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "framework/stages/04-validation/scripts/build_geometry_high_recall_manifest.py"
)
SPEC = spec_from_file_location("geometry_high_recall_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_under_accepts_relative_and_contained_absolute(tmp_path: Path) -> None:
    root = tmp_path / "root"
    artifact = root / "runtime" / "analysis.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    assert MODULE.resolve_under(root, "runtime/analysis.json", "analysis") == artifact
    assert MODULE.resolve_under(root, str(artifact), "analysis") == artifact


def test_resolve_under_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes root"):
        MODULE.resolve_under(root, str(outside), "analysis")


def test_build_rejects_screen_summary_outside_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "summary.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="screen execution summary escapes root"):
        MODULE.build(root, outside, root / "manifest.json")


def test_terminal_ct_insufficient_receipts_are_excluded_from_routing() -> None:
    completed = {
        "sample_id": "PHerc191",
        "seed_id": "supported",
        "state": "COMPLETED_DIAGNOSTIC_ONLY",
    }
    insufficient = {
        "sample_id": "PHerc191",
        "seed_id": "edge",
        "state": "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
    }
    receipts, partial = MODULE.completed_receipts_for_recovery(
        {
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "selected_count": 2,
            "completed_count": 1,
            "receipts": [insufficient, completed],
        },
        allow_disk_guard_subset=False,
    )

    assert receipts == [completed]
    assert partial is None


def test_terminal_batch_rejects_incorrect_completed_count() -> None:
    with pytest.raises(RuntimeError, match="completed receipt count mismatch"):
        MODULE.completed_receipts_for_recovery(
            {
                "status": "COMPLETED_DIAGNOSTIC_ONLY",
                "selected_count": 1,
                "completed_count": 1,
                "receipts": [
                    {
                        "sample_id": "PHerc191",
                        "seed_id": "edge",
                        "state": "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
                    }
                ],
            },
            allow_disk_guard_subset=False,
        )
