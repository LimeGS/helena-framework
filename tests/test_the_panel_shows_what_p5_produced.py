"""What a P5 screening produced, reachable without a shell on the GPU host.

`/api/runs` indexes the legacy CX_RUNS receipt tree. A P5 job queued through the
fleet keeps its verdict in `ink_jobs.result` and writes its map beside a receipt
in the directory the worker named, so nothing that index serves could see it and
the only way to look at a map was to ssh in and render the array by hand.

These drive the three endpoints that end that through the app itself, with a
stand-in queue and real .npy files on disk. What they hold is the shape of each
answer, the refusals, and -- the reason the render happens on the server at all
-- that a map carrying no decision is not displayed as though it carried one.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# A queue, without a queue
# --------------------------------------------------------------------------

class Store:
    """The two reads these endpoints make of the fleet queue."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def jobs(self, *, limit=100, states=None, mission_id=None, phase=None,
             sample_id=None) -> list[dict]:
        rows = [job for job in self.rows
                if (phase is None or job.get("phase") == phase)
                and (mission_id is None or job.get("mission_id") == mission_id)
                and (sample_id is None or job.get("sample_id") == sample_id)]
        return rows[:limit]

    def job(self, job_id: str) -> dict | None:
        return next((job for job in self.rows if job["job_id"] == job_id), None)


def receipt_document(statistics: dict | None, liveness: dict) -> dict:
    document = {
        "schema": "campaignx.ink_profile_screening_receipt.v1",
        "generated_at_utc": "2026-08-30T11:04:00Z",
        "sample_id": "PHerc0172",
        "lane": {"profile_id": "ink-2um-canonical",
                 "method_id": "youssef-2023",
                 "checkpoint_sha256": "c" * 64},
        "input": {"tiff_dir": "/ssd/vc3d/artifacts/layers/p4-9931",
                  "normalization": "clip(0,200) then divide by 200"},
        "liveness": liveness,
        "non_claims": ["A probability map is not OCR."],
    }
    if statistics is not None:
        document["statistics"] = statistics
    return document


ALIVE = {
    "verdict": "ALIVE",
    "reason": "",
    "metrics": {"p50": 0.3121, "p99": 0.8840, "spread_p99_p50": 0.5719,
                "std": 0.1904, "fraction_near_half": 0.2210,
                "valid_pixels": 262144},
}
DEGENERATE = {
    "verdict": "DEGENERATE",
    "reason": "p99-p50 0.0004 < 0.05; std 0.0002 < 0.02",
    "metrics": {"p50": 0.5001, "p99": 0.5005, "spread_p99_p50": 0.0004,
                "std": 0.0002, "fraction_near_half": 1.0,
                "valid_pixels": 262144},
    "interpretation": "the map carries no decision. Do not screen this map.",
}

# What the worker records under `physical_normalization` when the render it read
# is a P4 job of this control plane: the input lineage, by job id and digest.
LINEAGE = {
    "schema": "campaignx.first_letters_p5_normalization.v1",
    "p4_job_id": "p4-9931",
    "p4_layer_artifact_sha256": "a" * 64,
    "p4_layer_manifest_sha256": "b" * 64,
    "source_layer_objects": [{"object_key": "00.tif", "sha256": "d" * 64}],
    "profile_id": "ink-2um-canonical",
    "profile_sha256": "e" * 64,
    "checkpoint_sha256": "c" * 64,
}


def write_run(directory: Path, array: np.ndarray, *,
              name: str = "probability.npy",
              statistics: dict | None = None,
              liveness: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / name, array)
    (directory / "INK_PROFILE_RECEIPT.json").write_text(json.dumps(
        receipt_document(statistics, liveness or ALIVE), indent=2))


def p5_job(job_id: str, output: Path, *, state: str = "succeeded",
           statistics: dict | None = None, liveness: dict | None = None,
           surface_id: str | None = "surface:PHerc0172:0041") -> dict:
    parameters: dict = {"layer_stack": "/ssd/vc3d/artifacts/layers/p4-9931"}
    if surface_id:
        parameters["surface_id"] = surface_id
    return {
        "job_id": job_id, "sample_id": "PHerc0172", "phase": "P5",
        "profile_id": "ink-2um-canonical", "mission_id": "first-letters",
        "state": state, "attempts": 1, "max_attempts": 3,
        "parameters": parameters,
        "output_dir": str(output),
        "created_at": "2026-08-30T11:00:00+00:00",
        "updated_at": "2026-08-30T11:04:00+00:00",
        "result": {
            "exit_code": 0, "runtime_seconds": 214.6,
            "statistics": statistics,
            "liveness": liveness or ALIVE,
            "output_dir": str(output),
            "checkpoint_sha256": "c" * 64,
            "map_shape_yx": [512, 512],
            "physical_normalization": LINEAGE,
            "probability_map": {
                "artifact_uri": "s3://helena/ink-maps/p5-0001",
                "artifact_sha256": "f" * 64,
                "manifest_sha256": "0" * 64,
                "files": 2,
                "objects": [{"object_key": "probability.npy",
                             "sha256": "f" * 64, "bytes": 1048576}],
            },
        },
    }


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    # The map readers are cached on (path, mtime), and tmp_path is a new path
    # per test, but the cache is process-wide and these tests are cheap.
    module._ink_map_array.cache_clear()
    module._ink_map_display.cache_clear()
    module._ink_map_png.cache_clear()
    return module


@pytest.fixture
def anonymous(panel):
    from fastapi.testclient import TestClient

    return TestClient(panel.app)


@pytest.fixture
def client(panel, anonymous):
    from framework.contracts import auth

    auth.create_user(panel.AUTH_ROOT, "tester", "a-long-enough-one")
    assert anonymous.post("/api/session", json={"username": "tester",
                                                "password": "a-long-enough-one"}
                          ).status_code == 200
    return anonymous


def with_queue(panel, monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr(panel, "DSN", "postgresql://stand-in")
    monkeypatch.setattr(panel, "job_store", lambda: Store(rows))


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("route", [
    "/api/ink/maps",
    "/api/ink/maps/p5-0001",
    "/api/ink/maps/p5-0001/render.png",
])
def test_the_new_routes_are_closed_without_a_session(anonymous, route):
    """Deny by default, like every other read on this panel."""
    assert anonymous.get(route).status_code == 401


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------

def test_an_unreachable_queue_says_so_instead_of_serving_an_empty_table(
        panel, client, monkeypatch):
    """"No P5 runs" and "the database did not answer" are the same picture and
    opposite problems, so the second one has to say its own name."""
    monkeypatch.setattr(panel, "DSN", "")
    body = client.get("/api/ink/maps").json()
    assert body["available"] is False
    assert "CX_DB" in body["reason"]


def test_the_table_carries_every_column_it_promises(panel, client, monkeypatch,
                                                    tmp_path):
    output = tmp_path / "runs" / "first-letters" / "p5-0001"
    write_run(output, np.linspace(0.05, 0.95, 4096, dtype=np.float32).reshape(64, 64),
              statistics={"p50": 0.3121, "p90": 0.7, "p99": 0.884, "max": 0.99})
    with_queue(panel, monkeypatch, [p5_job("p5-0001", output)])

    body = client.get("/api/ink/maps").json()
    assert body["available"] is True
    (row,) = body["runs"]
    assert row["job_id"] == "p5-0001"
    assert row["sample_id"] == "PHerc0172"
    assert row["surface_id"] == "surface:PHerc0172:0041"
    assert row["profile_id"] == "ink-2um-canonical"
    assert row["state"] == "succeeded"
    assert row["created_at"].startswith("2026-08-30")
    assert row["liveness"]["verdict"] == "ALIVE"
    assert row["liveness"]["metrics"]["p50"] == pytest.approx(0.3121)
    assert row["liveness"]["metrics"]["p99"] == pytest.approx(0.8840)
    assert row["liveness"]["metrics"]["spread_p99_p50"] == pytest.approx(0.5719)
    assert row["maps"] == ["probability.npy"]
    assert row["published"]["artifact_sha256"] == "f" * 64


def test_a_lane_that_writes_no_statistics_reports_none_rather_than_zero(
        panel, client, monkeypatch, tmp_path):
    """The TimeSformer receipt has no `statistics` block at all. A zero there is
    a measurement nobody made, and it would read exactly like a real one."""
    output = tmp_path / "runs" / "unfiled" / "p5-tsf"
    write_run(output, np.full((32, 32), 0.4, dtype=np.float32),
              name="mean_probability.npy")
    with_queue(panel, monkeypatch, [p5_job("p5-tsf", output, statistics=None)])

    (row,) = client.get("/api/ink/maps").json()["runs"]
    assert row["statistics"] is None
    assert row["maps"] == ["mean_probability.npy"]


def test_a_failed_run_keeps_its_row_and_its_reason(panel, client, monkeypatch,
                                                   tmp_path):
    """A P5 job with no verdict is failed by the worker, and the refusal is the
    only thing on the page that explains why the map beside it must not be read."""
    output = tmp_path / "runs" / "unfiled" / "p5-dead"
    write_run(output, np.full((32, 32), 0.5, dtype=np.float32), liveness=DEGENERATE)
    job = p5_job("p5-dead", output, state="failed", liveness=DEGENERATE)
    job["result"]["refused"] = "DEGENERATE map: the lane produced no decision"
    with_queue(panel, monkeypatch, [job])

    (row,) = client.get("/api/ink/maps").json()["runs"]
    assert row["state"] == "failed"
    assert row["liveness"]["verdict"] == "DEGENERATE"
    assert "no decision" in row["refused"]


def test_the_table_says_a_map_is_not_on_this_host_rather_than_hiding_the_run(
        panel, client, monkeypatch, tmp_path):
    """Workers are ephemeral. A run whose bytes are not mounted here is still a
    run, and its published artifact set is where the bytes went."""
    with_queue(panel, monkeypatch,
               [p5_job("p5-elsewhere", tmp_path / "not-mounted")])

    (row,) = client.get("/api/ink/maps").json()["runs"]
    assert row["maps"] == []
    assert row["published"]["artifact_uri"] == "s3://helena/ink-maps/p5-0001"


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def test_the_detail_serves_the_receipt_the_digest_and_the_lineage(
        panel, client, monkeypatch, tmp_path):
    output = tmp_path / "runs" / "first-letters" / "p5-0001"
    write_run(output, np.linspace(0.05, 0.95, 4096, dtype=np.float32).reshape(64, 64),
              statistics={"p50": 0.3121, "p90": 0.7, "p99": 0.884, "max": 0.99})
    with_queue(panel, monkeypatch, [p5_job("p5-0001", output)])

    body = client.get("/api/ink/maps/p5-0001").json()
    assert body["receipt"]["schema"] == "campaignx.ink_profile_screening_receipt.v1"
    assert body["receipt"]["statistics"]["p99"] == pytest.approx(0.884)
    assert body["receipt_path"].endswith("INK_PROFILE_RECEIPT.json")
    assert body["checkpoint_sha256"] == "c" * 64
    assert body["lineage"]["p4_job_id"] == "p4-9931"
    assert body["lineage"]["p4_layer_artifact_sha256"] == "a" * 64
    assert body["liveness"]["reason"] == ""
    assert body["selected_map"] == "probability.npy"


def test_a_missing_receipt_is_named_rather_than_left_blank(
        panel, client, monkeypatch, tmp_path):
    output = tmp_path / "runs" / "unfiled" / "p5-bare"
    output.mkdir(parents=True)
    np.save(output / "probability.npy", np.full((16, 16), 0.3, dtype=np.float32))
    with_queue(panel, monkeypatch, [p5_job("p5-bare", output)])

    body = client.get("/api/ink/maps/p5-bare").json()
    assert body["receipt"] is None
    assert body["receipt_unavailable"] == "no receipt beside the map"


def test_a_job_from_another_phase_is_refused_by_name(panel, client, monkeypatch,
                                                     tmp_path):
    job = p5_job("p9-plates", tmp_path / "plates")
    job["phase"] = "P9"
    with_queue(panel, monkeypatch, [job])

    response = client.get("/api/ink/maps/p9-plates")
    assert response.status_code == 409
    assert "not a P5 screening" in response.json()["detail"]


# --------------------------------------------------------------------------
# The picture
# --------------------------------------------------------------------------

def test_the_map_is_served_as_a_png_and_never_as_the_array(
        panel, client, monkeypatch, tmp_path):
    output = tmp_path / "runs" / "first-letters" / "p5-0001"
    write_run(output, np.linspace(0.05, 0.95, 4096, dtype=np.float32).reshape(64, 64))
    with_queue(panel, monkeypatch, [p5_job("p5-0001", output)])

    response = client.get("/api/ink/maps/p5-0001/render.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image

    assert Image.open(io.BytesIO(response.content)).size == (64, 64)


def test_the_response_states_the_stretch_it_applied(panel, client, monkeypatch,
                                                    tmp_path):
    """A viewer that silently rescales is how a map with no signal comes to look
    like one with signal, so the normalisation travels with the picture."""
    output = tmp_path / "runs" / "first-letters" / "p5-0001"
    write_run(output, np.linspace(0.05, 0.95, 4096, dtype=np.float32).reshape(64, 64))
    with_queue(panel, monkeypatch, [p5_job("p5-0001", output)])

    response = client.get("/api/ink/maps/p5-0001/render.png")
    assert "percentile stretch" in response.headers["X-Map-Normalisation"]
    assert response.headers["X-Map-Name"] == "probability.npy"

    display = client.get("/api/ink/maps/p5-0001").json()["display"]
    assert display["normalisation"] == "percentile"
    assert display["low_percentile"] == 1.0 and display["high_percentile"] == 99.0
    assert display["low_value"] == pytest.approx(0.059, abs=0.01)
    assert display["high_value"] == pytest.approx(0.941, abs=0.01)
    assert display["width"] == 64 and display["height"] == 64
    assert "not comparable between runs" in display["note"]


def test_a_map_with_no_range_is_drawn_flat_instead_of_stretched(
        panel, client, monkeypatch, tmp_path):
    """The whole reason the stretch is applied here. A constant map is what a
    DEGENERATE verdict looks like; stretching it turns float noise into a
    picture of structure, and the reader cannot tell the difference."""
    output = tmp_path / "runs" / "unfiled" / "p5-dead"
    write_run(output, np.full((32, 32), 0.5001, dtype=np.float32),
              liveness=DEGENERATE)
    with_queue(panel, monkeypatch,
               [p5_job("p5-dead", output, state="failed", liveness=DEGENERATE)])

    display = client.get("/api/ink/maps/p5-dead").json()["display"]
    assert display["flat"] is True
    assert "constant" in display["note"]

    from PIL import Image

    png = client.get("/api/ink/maps/p5-dead/render.png").content
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    assert pixels.min() == pixels.max() == 0


def test_a_stretch_that_fell_back_to_the_range_does_not_call_itself_percentile(
        panel, client, monkeypatch, tmp_path):
    """Nearly every pixel one value, a handful outside it.

    Falling back to (min, max) is right -- there is something to draw. Printing
    "p1=... is black" underneath it afterwards is not: those are not the numbers
    the picture was made from, and the whole point of stating the stretch is
    that the statement is true.
    """
    array = np.full((32, 32), 0.5, dtype=np.float32)
    array[0, 0] = 0.99
    output = tmp_path / "runs" / "unfiled" / "p5-spike"
    write_run(output, array, liveness=DEGENERATE)
    with_queue(panel, monkeypatch, [p5_job("p5-spike", output, liveness=DEGENERATE)])

    display = client.get("/api/ink/maps/p5-spike").json()["display"]
    assert display["normalisation"] == "range"
    assert display["flat"] is False
    assert display["low_value"] == pytest.approx(0.5)
    assert display["high_value"] == pytest.approx(0.99)
    assert "fell back to the full range" in display["note"]
    assert "percentile stretch" not in display["note"]


def test_a_name_the_job_did_not_write_is_refused(panel, client, monkeypatch,
                                                 tmp_path):
    """The name is matched against the directory's own listing, so it never
    becomes a path -- `../` and a symlink out of the volume are not in it."""
    output = tmp_path / "runs" / "first-letters" / "p5-0001"
    write_run(output, np.full((16, 16), 0.4, dtype=np.float32))
    (tmp_path / "secret.npy").write_bytes(b"not yours")
    with_queue(panel, monkeypatch, [p5_job("p5-0001", output)])

    assert client.get(
        "/api/ink/maps/p5-0001/render.png?map=../../secret.npy").status_code == 404
    assert client.get(
        "/api/ink/maps/p5-0001/render.png?map=stability_std.npy").status_code == 404


def test_a_map_this_host_cannot_read_refuses_with_where_it_expected_it(
        panel, client, monkeypatch, tmp_path):
    with_queue(panel, monkeypatch,
               [p5_job("p5-elsewhere", tmp_path / "not-mounted")])

    response = client.get("/api/ink/maps/p5-elsewhere/render.png")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "not-mounted" in detail["expected_at"]
    assert "another worker" in detail["why"]


def test_the_evidence_arrays_beside_the_aggregate_are_offered_but_not_chosen(
        panel, client, monkeypatch, tmp_path):
    """The TimeSformer lane writes per-window maps and a stability map beside its
    aggregate. They are evidence and they are not interchangeable with it, so the
    aggregate is what a run opens on."""
    output = tmp_path / "runs" / "unfiled" / "p5-tsf"
    write_run(output, np.linspace(0.1, 0.9, 1024, dtype=np.float32).reshape(32, 32),
              name="mean_probability.npy")
    np.save(output / "center-016_offset-00.npy",
            np.full((32, 32), 0.2, dtype=np.float32))
    np.save(output / "stability_std.npy", np.full((32, 32), 0.05, dtype=np.float32))
    with_queue(panel, monkeypatch, [p5_job("p5-tsf", output)])

    body = client.get("/api/ink/maps/p5-tsf").json()
    assert body["maps"][0] == "mean_probability.npy"
    assert set(body["maps"]) == {"mean_probability.npy", "stability_std.npy",
                                 "center-016_offset-00.npy"}
    assert body["selected_map"] == "mean_probability.npy"

    chosen = client.get(
        "/api/ink/maps/p5-tsf/render.png?map=stability_std.npy")
    assert chosen.headers["X-Map-Name"] == "stability_std.npy"
