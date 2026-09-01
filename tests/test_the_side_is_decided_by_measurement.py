"""Which side of the sheet faces the ink, settled by a number.

The orientation proof can prove that a grown mesh winds the same way as the
reference. It cannot say which side carries the writing, and it refuses to
choose without absolute evidence -- `ABSOLUTE_ORIENTATION_EVIDENCE_MISSING`,
which is where the control stops today.

That evidence has looked like a judgement call and it is not. Render the locked
public surface both ways, screen each with the model the control locks, and
correlate both against the community's published map for the same segment. One
side wins by a margin or the experiment did not settle anything.

There is prior art and it is not usable as it stands: a 2026-07-28 reproduction
reached r = 0.885 with `--flip-normals` against r = 0.09 without. It ran at
2.399 um with a different model, on a lane whose own receipt says it "must not
be routed to 8.64/9.362 um targets without its own control", and the number
lives in a registry note rather than any receipt. Convincing prose from another
resolution is not evidence for this one, so the measurement is made again here,
at the control's scale, with the control's model, and written down.

What this produces is the receipt the manifest's `absolute_orientation` lock
points at. It does not decide the lock: an operator still reads the margin and
commits it, because a run that both measures and accepts its own result is the
control certifying itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")
pytest.importorskip("fastapi")

from panel.app import absolute_orientation_evidence  # noqa: E402

REFERENCE = {
    "uri": "https://example.invalid/w025.tifxyz",
    "artifacts": [{"path": "x.tif", "sha256": "a" * 64}],
}
LINEAGE = {
    "control_profile_id": "first-letters-control-policy@1.3.0",
    "orientation_profile_id": "first-letters-orientation-parity@1.0.0",
}


def _evidence(unflipped: float, flipped: float, **kwargs):
    return absolute_orientation_evidence(
        reference=REFERENCE, lineage=LINEAGE,
        correlations={"unflipped": unflipped, "flipped": flipped},
        community_map_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
        **kwargs)


def test_the_side_follows_the_larger_correlation() -> None:
    evidence = _evidence(0.09, 0.885)

    assert evidence["side_decision"]["same_winding_flip_normals"] is True
    assert evidence["side_decision"]["margin"] == pytest.approx(0.795)


def test_the_other_way_round_decides_the_other_way() -> None:
    evidence = _evidence(0.885, 0.09)

    assert evidence["side_decision"]["same_winding_flip_normals"] is False


def test_a_thin_margin_settles_nothing() -> None:
    """Two sides that score alike is a failed experiment, not a coin toss. The
    whole point of measuring is that the answer is not a preference."""
    with pytest.raises(ValueError, match="did not separate"):
        _evidence(0.44, 0.46)


def test_a_side_that_correlates_with_nothing_is_refused() -> None:
    """If neither render matches the published map, the winner is the better of
    two failures and means nothing.

    The margin here is wide -- 0.23 -- so this reaches the second gate rather
    than the first. Both are needed: sides that separate cleanly can still both
    be wrong."""
    with pytest.raises(ValueError, match="neither side"):
        _evidence(0.02, 0.25)


def test_the_receipt_carries_what_the_loader_checks() -> None:
    """The panel compares these field by field against the manifest before it
    will use the receipt at all."""
    from panel.app import _canonical_document_sha256

    evidence = _evidence(0.09, 0.885)

    assert evidence["schema"] == (
        "campaignx.first_letters_absolute_orientation_evidence.v1")
    assert evidence["lineage"] == LINEAGE
    assert evidence["reference_read_set"] == {
        "uri": REFERENCE["uri"],
        "objects": REFERENCE["artifacts"],
        "canonical_manifest_sha256": _canonical_document_sha256(REFERENCE["artifacts"]),
    }
    body = {k: v for k, v in evidence.items() if k != "receipt_sha256"}
    assert evidence["receipt_sha256"] == _canonical_document_sha256(body)


def test_the_measurement_is_recorded_beside_the_decision() -> None:
    """A bare boolean would be an assertion. The numbers that produced it, the
    map they were scored against and the checkpoint that made them are what let
    somebody else disagree."""
    evidence = _evidence(0.09, 0.885)

    assert evidence["measurement"]["correlations"] == {
        "unflipped": 0.09, "flipped": 0.885}
    assert evidence["measurement"]["community_map_sha256"] == "d" * 64
    assert evidence["measurement"]["checkpoint_sha256"] == "e" * 64


def test_the_receipt_claims_nothing_about_ink() -> None:
    """A correlation with a published map is evidence about geometry. It is not
    a reading, and the receipt says so where somebody quoting it will see it."""
    evidence = _evidence(0.09, 0.885)

    assert "non_claim" in evidence
    assert "not" in evidence["non_claim"].lower()
