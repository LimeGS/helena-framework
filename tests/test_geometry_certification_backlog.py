"""P2 applied to surfaces that already existed.

The gate ran inside the finalizer, so it only ever saw surfaces on their way
out. Everything grown before it landed stayed GEOMETRY_UNMEASURED -- all 43 of
them -- and there was no way to go back. These are the properties that have to
hold for the way back, because the verdict has to mean the same thing whichever
side it came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.certifier import certify_one  # noqa: E402


class RecordingStore:
    def __init__(self):
        self.recorded = []

    def record_geometry_certification(self, surface_id, state, receipt=None):
        self.recorded.append((surface_id, state, receipt))
        return {"surface_id": surface_id, "geometry_qc_state": state,
                "physical_qc_state": "UNVALIDATED", "blocked_qc_jobs": 0}


class Adapter:
    """Stands in for the QC adapter's materialiser."""

    def __init__(self, failure: Exception | None = None):
        self.failure = failure
        self.calls = 0

    def materialize_surface(self, uri, digest, destination, s3_client=None):
        self.calls += 1
        if self.failure:
            raise self.failure
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"files": {}}


SURFACE = {"surface_id": "s-1", "artifact_uri": "s3://bucket/s-1",
           "artifact_sha256": "abc"}


def test_an_artifact_that_cannot_be_fetched_is_unmeasured_not_rejected(tmp_path):
    """Rejected blocks a surface from the ink model for a reason in its
    geometry. A network error is not one, and recording it as rejected would
    retire a surface over an outage."""
    store = RecordingStore()
    outcome = certify_one(store, SURFACE, tmp_path,
                          adapter=Adapter(RuntimeError("connection reset")))
    assert outcome["geometry_qc_state"] == "GEOMETRY_UNMEASURED"
    assert outcome["reason"] == "ARTIFACT_UNAVAILABLE"


def test_the_reason_is_recorded_with_the_verdict(tmp_path):
    """"Unmeasured" with no reason attached is a word.

    It cost a debugging session: 43 surfaces read UNMEASURED and the cause --
    first a missing module, then absent credentials, then scipy -- was only
    visible in the terminal of whoever ran it."""
    store = RecordingStore()
    certify_one(store, SURFACE, tmp_path, adapter=Adapter(RuntimeError("no creds")))
    _, _, receipt = store.recorded[0]
    assert receipt["reason"] == "ARTIFACT_UNAVAILABLE"
    assert "no creds" in receipt["error"]


def test_the_staging_directory_does_not_survive_the_surface(tmp_path):
    """43 surfaces at a time, each a full TIFXYZ, on a disk with other jobs."""
    certify_one(RecordingStore(), SURFACE, tmp_path, adapter=Adapter())
    assert list(tmp_path.iterdir()) == []


def test_a_verdict_is_always_written_even_when_nothing_could_be_measured(tmp_path):
    """A surface that silently keeps its old state looks like one nobody has
    got to yet, so the backlog never shrinks and never explains itself."""
    store = RecordingStore()
    certify_one(store, SURFACE, tmp_path, adapter=Adapter(RuntimeError("gone")))
    assert len(store.recorded) == 1
