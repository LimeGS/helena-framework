"""P9's pages are reachable from the browser that queued them.

The phase whose whole point is a page somebody can read reported "plate runs
succeeded: 1" and stopped there: the 38 plates it rendered were on the host, and
looking at one meant opening a shell. This is the endpoint the gallery reads.

A name from a request never becomes a path here. It is matched against the plate
set the worker recorded on the job, so `../../etc/passwd` is not refused by a
rule about dots -- it is simply not one of the files this job says it wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402


class _Store:
    def __init__(self, job):
        self._job = job

    def job(self, job_id):
        return self._job if self._job and self._job["job_id"] == job_id else None


def _job(tmp_path: Path, *, wrote_to: Path | None = None):
    return {
        "job_id": "p9-5f255767b7054b",
        "sample_id": "PHerc0139",
        "output_dir": str(tmp_path / "run"),
        "result": {
            "exit_code": 0,
            "wrote_to": str(wrote_to) if wrote_to else None,
            "plate_set": {"plates": [
                {"file": "01_w059.png", "wrap": "w059", "width": 5180,
                 "height": 3733},
            ]},
        },
    }


def test_a_recorded_plate_is_served_from_where_the_job_wrote(monkeypatch, tmp_path):
    import panel.app as app

    plates = tmp_path / "plates"
    plates.mkdir()
    # A PNG only in the sense that matters here: bytes at the recorded name.
    (plates / "01_w059.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(app, "job_store",
                        lambda: _Store(_job(tmp_path, wrote_to=plates)))

    response = app.api_job_plate("p9-5f255767b7054b", "01_w059.png")
    assert Path(response.path) == plates / "01_w059.png"
    assert response.media_type == "image/png"


def test_a_name_the_job_did_not_record_is_not_a_path(monkeypatch, tmp_path):
    import panel.app as app

    secret = tmp_path / "secret.png"
    secret.write_bytes(b"not yours")
    monkeypatch.setattr(app, "job_store",
                        lambda: _Store(_job(tmp_path, wrote_to=tmp_path)))

    for name in ("secret.png", "../secret.png", "01_w059.png/../secret.png"):
        with pytest.raises(HTTPException) as refusal:
            app.api_job_plate("p9-5f255767b7054b", name)
        assert refusal.value.status_code == 404
        # And it says what this job did record, so the caller can ask again.
        assert refusal.value.detail["plates"] == ["01_w059.png"]


def test_a_plate_recorded_but_absent_says_where_it_should_be(monkeypatch, tmp_path):
    """A job that ran on another worker is not a 404.

    The record is right and the bytes are elsewhere; saying "no such plate"
    would send somebody looking for a bug in P9.
    """
    import panel.app as app

    monkeypatch.setattr(app, "job_store",
                        lambda: _Store(_job(tmp_path, wrote_to=tmp_path / "gone")))
    with pytest.raises(HTTPException) as refusal:
        app.api_job_plate("p9-5f255767b7054b", "01_w059.png")
    assert refusal.value.status_code == 409
    assert "gone/01_w059.png" in refusal.value.detail["expected_at"]


def test_a_job_from_before_the_worker_recorded_it_still_resolves(
        monkeypatch, tmp_path):
    """The job named the directory; the record just did not repeat it.

    Every P9 queued before the worker started recording `wrote_to` has an empty
    run directory and its pages under the `out_dir` it was given.
    """
    import panel.app as app

    plates = tmp_path / "named"
    plates.mkdir()
    (plates / "01_w059.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    job = _job(tmp_path)
    job["parameters"] = {"out_dir": str(plates)}
    monkeypatch.setattr(app, "job_store", lambda: _Store(job))

    response = app.api_job_plate("p9-5f255767b7054b", "01_w059.png")
    assert Path(response.path) == plates / "01_w059.png"


def test_an_unknown_job_is_an_unknown_job(monkeypatch):
    import panel.app as app

    monkeypatch.setattr(app, "job_store", lambda: _Store(None))
    with pytest.raises(HTTPException) as refusal:
        app.api_job_plate("p9-nothing", "01_w059.png")
    assert refusal.value.status_code == 404
