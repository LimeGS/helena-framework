from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "workspace/campaigns/campaign-x-2026/plans"
    / "PLN-EXP-20260722-SEG-HYBRID-scrollfiesta-vc3d-r01-SOURCE_LOCK.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


@needs_campaign_evidence
def test_hybrid_source_lock_is_result_blind_and_preserves_vc3d_default() -> None:
    document = _document()
    assert document["state"] == "INPUTS_LOCKED"
    assert document["ink_used_for_selection"] is False
    constraints = document["execution_constraints"]
    assert constraints["vc3d_remains_default"] is True
    assert constraints["physical_mesh_fusion"] is False
    assert constraints["extended_regions_require_prior_local_pass"] is True
    assert constraints["no_parameter_or_roi_change_after_backend_results"] is True


@needs_campaign_evidence
def test_locked_profile_hashes_match_bytes() -> None:
    document = _document()
    for profile in document["profiles"]:
        path = ROOT / profile["path"]
        assert path.is_file()
        assert _sha256(path) == profile["sha256"]


@needs_campaign_evidence
def test_locked_rois_are_unique_aligned_and_inside_level_zero_sources() -> None:
    document = _document()
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for region in document["roi_selection"]["regions"]:
        sample = region["sample_id"]
        roi = tuple(region["roi_level0_zyx"])
        assert len(roi) == 6
        assert all(value % 128 == 0 for value in roi)
        assert all(roi[index] < roi[index + 3] for index in range(3))
        shape = document["sources"][sample]["m7"]["level0_shape_zyx"]
        assert all(0 <= roi[axis] < roi[axis + 3] <= shape[axis] for axis in range(3))
        cli = region["scrollfiesta_bbox_cli_zyx_pairs"]
        assert cli == [roi[0], roi[3], roi[1], roi[4], roi[2], roi[5]]
        seed_xyz = region["vc3d_seed_xyz"]
        seed_zyx = list(reversed(seed_xyz))
        assert all(roi[axis] <= seed_zyx[axis] < roi[axis + 3] for axis in range(3))
        key = (sample, roi)
        assert key not in seen
        seen.add(key)


@needs_campaign_evidence
def test_local_and_extended_target_rois_contain_prelocked_candidate_center() -> None:
    document = _document()
    regions = {region["stage"]: region for region in document["roi_selection"]["regions"]}
    center_xyz = regions["S4"]["selection_evidence"]["candidate_center_xyz"]
    center_zyx = list(reversed(center_xyz))
    for stage in ("S4", "S5"):
        roi = regions[stage]["roi_level0_zyx"]
        assert all(roi[axis] <= center_zyx[axis] < roi[axis + 3] for axis in range(3))
