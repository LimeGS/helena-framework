"""A human review is not a QC verdict, and the code has to keep them apart.

Approved/Defective/Reviewed/Inspect come from whoever is logged in. Geometry
certification and CT support come from the fleet, live in their own columns, and
are what P3 and P4 ask before consuming a surface. If a person calling a surface
"approved" could reach either of those, one click would forge a scientific claim.

So the review goes into the surface payload under its own key, and downstream
admission keeps reading the columns.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()
STORE = (ROOT / "framework/stages/01-segmentation/fleet/store.py").read_text()


def _handler(path: str) -> str:
    start = APP.index(f'@app.post("{path}")')
    return APP[start:][: APP[start:].index("\n@app.", 1)]


def test_a_review_cannot_write_a_qc_column() -> None:
    handler = _handler("/api/segmentation/surface/{surface_id}/review")
    assert "'human_review'" in handler
    for column in ("physical_qc_state", "geometry_qc_state", "state"):
        assert f"SET {column}" not in handler and f"{column} =" not in handler, (
            f"the review endpoint writes {column}, which is the fleet's verdict "
            "and not a person's"
        )


def test_the_verdicts_are_a_closed_set() -> None:
    handler = _handler("/api/segmentation/surface/{surface_id}/review")
    assert "HUMAN_REVIEW_VERDICTS" in handler
    verdicts = APP[APP.index("HUMAN_REVIEW_VERDICTS = ("):]
    verdicts = verdicts[: verdicts.index(")")]
    for word in ("APPROVED", "DEFECTIVE", "REVIEWED", "INSPECT"):
        assert word in verdicts
    # And none of them collide with the vocabulary the fleet uses, or a reader
    # scanning two columns would see the same word meaning two things.
    assert "CERTIFIED" not in verdicts
    assert "CT_SUPPORTED" not in verdicts


def test_downstream_admission_still_asks_the_columns() -> None:
    """The gate P3/P4 use must not have learned about human_review."""
    gate = STORE[STORE.index("def is_downstream_admissible"):]
    gate = gate[: gate.index("\ndef ")]
    assert "geometry_state" in gate and "physical_state" in gate
    assert "human" not in gate.lower(), (
        "downstream admission consults a human review, so an operator can admit "
        "a surface the fleet never certified"
    )


def test_the_bundle_points_rather_than_copies() -> None:
    """A CT volume here is 20840x8387x8387 voxels behind an HTTP URL."""
    start = APP.index('@app.get("/api/segmentation/surface/{surface_id}/vc3d")')
    bundle = APP[start:][: APP[start:].index("\n@app.", 1)]
    assert "ct_uri" in bundle and "m7_uri" in bundle
    assert "coordinate_frame" in bundle and "voxel_size_um" in bundle
    assert "vc_grow_seg_from_seed" in bundle
    # And it says so when the volume was never hashed, rather than implying the
    # bundle pins content.
    assert "HASH_UNAVAILABLE" in bundle
