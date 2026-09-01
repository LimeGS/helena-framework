"""The route between a dragged rectangle and hash-bound evidence.

The viewer sends the box and nothing else. The panel walks P5 -> P4 -> P3 for
the lineage, derives the transform, writes the provenance artifact and returns
the manifest fragment. Every field the check later compares is produced on this
side, from one implementation, because the alternative is two producers obliged
to agree -- which is how this control found four defects in a week.

What the route must refuse is as much of its job as what it accepts: a box that
is the whole map, a box outside it, and a sample that is not the frozen control
scroll. The last one matters because a locked "known positive" for a target
scroll would be a claim about a scroll nobody has read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

MISSION = "control-mission"
SAMPLE = "PHerc0139"
P5_JOB = "p5-1"

WALKED = {
    "surface_id": "s-1",
    "p5_job": P5_JOB,
    "lineage": {
        "surface_id": "s-1", "p5_job_id": P5_JOB,
        "probability_map_sha256": "a" * 64,
        "probability_map_manifest_sha256": "b" * 64,
        "normalization_receipt_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "profile_id": "timesformer-gp-scroll1-screening@1.0.0",
        "profile_sha256": "e" * 64,
    },
    "map_shape_yx": [64, 56],
    "sheet_shape_yx": [1024, 896],
    "probability_map": {"artifact_sha256": "a" * 64, "manifest_sha256": "b" * 64},
    "expected_binding": {"control_p0_artifact_id": "p0:x"},
}


@pytest.fixture()
def route(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "resolve_control_roi_lineage",
                        lambda *_a, **_k: dict(WALKED))
    monkeypatch.setattr(panel_app, "read_scope", lambda _m, _s: {SAMPLE})
    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    return panel_app


def _control(scroll_id: str = SAMPLE) -> dict:
    return {"profile_id": "first-letters-control-policy@1.2.0",
            "control_cohort": {"scroll_id": scroll_id},
            "checks": {"PIPELINE_CONTROL": {"p5_physical_normalization": {
                "verified_training_pixel_um": 7.91}}}}


def _request(bbox):
    from panel.app import PositiveControlRoiRequest

    return PositiveControlRoiRequest(
        mission=MISSION, sample=SAMPLE, p5_job=P5_JOB, drawn_bbox_xyxy=bbox,
        note="the visible letterforms on w025")


def test_the_route_returns_the_document_and_the_lock_fragment(route) -> None:
    answer = json.loads(bytes(route.api_record_positive_control_roi(
        _request([10, 12, 30, 34]), control=_control()).body))

    assert answer["provenance"]["schema"] == (
        "campaignx.first_letters_positive_control_roi_provenance.v1")
    assert answer["provenance"]["transformed_bbox_xyxy"] == [10, 12, 30, 34]
    assert answer["lock"]["verified"] is True
    assert answer["lock"]["p5_transform_receipt_sha256"] == (
        route._canonical_document_sha256(answer["provenance"]))


def test_the_lineage_is_the_walk_s_not_the_client_s(route) -> None:
    """The request has no place to put a lineage, and the answer carries the
    one the panel derived."""
    answer = json.loads(bytes(route.api_record_positive_control_roi(
        _request([10, 12, 30, 34]), control=_control()).body))

    assert answer["provenance"]["lineage"] == WALKED["lineage"]
    assert "lineage" not in _request([10, 12, 30, 34]).model_dump()


def test_the_artifact_is_written_and_its_digest_is_the_lock_s(route, tmp_path) -> None:
    """The lock names a URI and a digest of bytes, so the bytes have to exist."""
    answer = json.loads(bytes(route.api_record_positive_control_roi(
        _request([10, 12, 30, 34]), control=_control()).body))

    import hashlib
    written = Path(answer["artifact_path"])
    assert written.is_file(), "the lock points at bytes nobody wrote"
    assert hashlib.sha256(written.read_bytes()).hexdigest() == (
        answer["lock"]["provenance_artifact_sha256"])
    assert json.loads(written.read_text()) == answer["provenance"]


def test_the_whole_map_is_refused(route) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refused:
        route.api_record_positive_control_roi(
            _request([0, 0, 56, 64]), control=_control())
    assert refused.value.status_code == 409
    assert "whole map" in str(refused.value.detail)


def test_a_sample_that_is_not_the_control_scroll_is_refused(route) -> None:
    """A locked known positive for a target scroll would be a claim about a
    scroll nobody has read."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refused:
        route.api_record_positive_control_roi(
            _request([10, 12, 30, 34]), control=_control(scroll_id="PHerc0172"))
    assert refused.value.status_code == 409


def test_a_refusal_says_what_was_wrong(route) -> None:
    """The 409s this control produces have cost hours because they named no
    field. This one says what it compared."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refused:
        route.api_record_positive_control_roi(
            _request([10, 12, 900, 34]), control=_control())
    assert "outside the map" in str(refused.value.detail)
