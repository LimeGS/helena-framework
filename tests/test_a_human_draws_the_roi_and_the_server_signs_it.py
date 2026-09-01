"""The human contributes a judgement; the server contributes the provenance.

P7 needs a locked ROI: where, in the P5 probability map, the known writing of
w025 actually is. That is ground truth about the world -- it comes from the
community attribution the manifest already cites -- and no detector may supply
it, because a detector deciding where the letters are would be the control
verifying itself.

Everything else is arithmetic, and the browser must not do any of it. The
provenance artifact has to match a lineage the panel derives by walking
P5 -> P4 -> P3, and its transform is recomputed and required to be pixel exact.
A client that computed either would be a second implementation that has to agree
with the first -- the "producer and verifier read different places" defect that
cost this program four runs in one week.

So the request carries one thing: the rectangle somebody dragged, in map pixels.

The source frame is the P4 rendered sheet rather than the map, because the map
is the model's downscaled output and that scale is a property of the run. A box
recorded in sheet pixels still means the same place if the output scale ever
changes; a box recorded in map pixels silently would not.

When the sheet does not divide the map evenly the request is refused rather
than rounded: the panel demands that `source * scale + offset` land on whole
map pixels, and no rounding rule inside a proof is a number anybody could
defend. With the even ratio the render actually produces, every drawn box
already lands on the grid and nothing is adjusted at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from panel.app import roi_provenance_from_drawn_box  # noqa: E402

# The shapes the server already knows: P4's rendered sheet and P5's map.
SHEET_YX = [1024, 896]
MAP_YX = [64, 56]          # exactly 1/16 on both axes
LINEAGE = {
    "surface_id": "s-1", "p5_job_id": "p5-1",
    "probability_map_sha256": "a" * 64,
    "probability_map_manifest_sha256": "b" * 64,
    "normalization_receipt_sha256": "c" * 64,
    "checkpoint_sha256": "d" * 64,
    "profile_id": "timesformer-gp-scroll1-screening@1.0.0",
    "profile_sha256": "e" * 64,
}


def _draw(bbox, *, sheet=None, map_yx=None, pixel_um=7.91):
    return roi_provenance_from_drawn_box(
        drawn_bbox_xyxy=bbox, map_shape_yx=map_yx or MAP_YX,
        sheet_shape_yx=sheet or SHEET_YX, lineage=LINEAGE,
        verified_training_pixel_um=pixel_um)


def test_the_document_carries_the_schema_the_check_requires() -> None:
    document = _draw([10, 12, 30, 34])

    assert document["schema"] == (
        "campaignx.first_letters_positive_control_roi_provenance.v1")
    assert document["lineage"] == LINEAGE, "the lineage must be the server's, verbatim"


def test_the_transform_reproduces_the_drawn_box_exactly() -> None:
    """The panel recomputes source * scale + offset and demands integers that
    equal the declared transformed box. This is that arithmetic, from the
    producing side."""
    document = _draw([10, 12, 30, 34])

    source = document["source_bbox_xyxy"]
    scale = document["transform"]["scale_xy"]
    offset = document["transform"]["offset_xy"]
    recomputed = [source[0] * scale[0] + offset[0],
                  source[1] * scale[1] + offset[1],
                  source[2] * scale[0] + offset[0],
                  source[3] * scale[1] + offset[1]]

    assert all(float(value).is_integer() for value in recomputed)
    assert [int(value) for value in recomputed] == document["transformed_bbox_xyxy"]
    assert document["transformed_bbox_xyxy"] == [10, 12, 30, 34]


def test_the_source_frame_is_the_rendered_sheet_not_the_map() -> None:
    """A box in map pixels stops meaning the same place if the model's output
    scale changes. In sheet pixels it does not."""
    document = _draw([10, 12, 30, 34])

    assert document["source_coordinate_system"] == "p4_layer_stack_pixels"
    assert document["source_bbox_xyxy"] == [160, 192, 480, 544]  # 16x


def test_a_sheet_that_does_not_divide_the_map_is_refused() -> None:
    """Refused rather than rounded.

    This began as an outward-expansion rule for boxes that miss the source
    grid, which was designing for a case that cannot be recorded honestly: when
    the sheet does not divide the map evenly, no choice of source box makes
    `source * scale + offset` land on whole map pixels, and the panel demands
    exactly that. A rounding rule inside a proof would be a number nobody could
    defend, so the mismatch is named instead.

    With an even ratio -- what the render actually produces -- every drawn box
    already lands on the grid, so nothing needs adjusting at all.
    """
    with pytest.raises(ValueError, match="whole map pixels"):
        _draw([10, 12, 30, 34], sheet=[1000, 900])


def test_an_even_ratio_needs_no_adjustment() -> None:
    document = _draw([10, 12, 30, 34])

    assert document["adjusted_from_drawn"] is False
    assert document["transformed_bbox_xyxy"] == document["drawn_bbox_xyxy"]


def test_the_whole_map_is_refused() -> None:
    """A ROI that is the entire map asserts nothing: the model lights up
    somewhere every time. The panel refuses it and so does this."""
    with pytest.raises(ValueError, match="whole map"):
        _draw([0, 0, MAP_YX[1], MAP_YX[0]])


def test_a_box_outside_the_map_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the map"):
        _draw([10, 12, MAP_YX[1] + 1, 34])


def test_a_degenerate_box_is_refused() -> None:
    for bbox in ([10, 12, 10, 34], [10, 12, 30, 12], [30, 34, 10, 12]):
        with pytest.raises(ValueError, match="empty|inverted"):
            _draw(bbox)


def test_the_training_pixel_size_is_carried_for_the_lock_to_match() -> None:
    """The lock and the artifact must agree on it, so the producer records the
    value rather than leaving the two to be filled in separately."""
    document = _draw([10, 12, 30, 34], pixel_um=7.91)

    assert document["verified_training_pixel_um"] == 7.91


def test_the_lock_fragment_matches_the_document_it_describes() -> None:
    """What an operator pastes into the manifest is derived here, not retyped:
    every mismatch between the two is a 409 that costs a run to discover."""
    from panel.app import roi_lock_fragment, _canonical_document_sha256

    document = _draw([10, 12, 30, 34])
    fragment = roi_lock_fragment(document, artifact_uri="file:///roi.json",
                                 artifact_sha256="f" * 64)

    assert fragment["verified"] is True
    assert fragment["provenance_artifact_uri"] == "file:///roi.json"
    assert fragment["provenance_artifact_sha256"] == "f" * 64
    assert fragment["source_bbox_xyxy"] == document["source_bbox_xyxy"]
    assert fragment["transformed_bbox_xyxy"] == document["transformed_bbox_xyxy"]
    assert fragment["source_coordinate_system"] == document["source_coordinate_system"]
    assert fragment["verified_training_pixel_um"] == document["verified_training_pixel_um"]
    assert fragment["p5_transform_receipt_sha256"] == _canonical_document_sha256(document)
