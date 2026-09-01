"""The seed search, rebuilt from a contract because its source is gone.

Every assertion here is something the worker or the planner depends on, so a
reimplementation that drifts from the original in these respects is a
reimplementation that breaks the fleet quietly.
"""

from __future__ import annotations

import json
import hashlib
import http.server
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "framework" / "stages" / "01-segmentation" / "mcp"))
import seed_candidates as seeds  # noqa: E402
import server as mcp_server  # noqa: E402

URI = "s3://bucket/PHerc0139/surface-m7-L0-th0.2.zarr"


def region(size: int = 64, origin: int = 0) -> dict:
    return {f"{a}_{b}": (origin if b == "min" else origin + size)
            for a in "xyz" for b in ("min", "max")}


def empty(size: int = 64) -> np.ndarray:
    return np.zeros((size, size, size), dtype=np.uint8)


class CountingVolume:
    def __init__(self, array: np.ndarray, *, chunks=(128, 128, 128)):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.chunks = chunks
        self.read_count = 0

    def __getitem__(self, key):
        self.read_count += 1
        return self.array[key]


def test_a_region_with_no_prediction_returns_nothing_rather_than_failing():
    """This is the NO_SEED case, and it is not an error: the model says there
    is no sheet there, which is an answer."""
    assert seeds.find_candidates(empty(), region(), prediction_uri=URI) == []


def test_real_mcp_service_returns_the_actual_zarr_metadata_and_chunk_read_inventory(tmp_path):
    """Removing the production store tracker must remove this evidence."""
    zarr = pytest.importorskip("zarr")
    target = tmp_path / "surface-m7-L0-th0.2.zarr"
    group = zarr.open_group(str(target), mode="w", zarr_format=2)
    group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[4, 5, 6] = 220
    group.create_array("0", data=data, chunks=(8, 8, 8), compressor=None)

    result = mcp_server.Service(tmp_path).find_seed_candidates({
        "prediction_uri": URI,
        "prediction_space": "ct_l0_xyz",
        "region": region(size=8),
        "max_candidates": 2,
        "minimum_separation_voxels": 1,
        "threshold": 0.2,
    })

    receipt = result["source_read_set"]
    assert receipt["schema"] == "campaignx.first_letters_source_read_set.v1"
    keys = [row["object_key"] for row in receipt["objects"]]
    assert keys == sorted(set(keys))
    assert ".zattrs" in keys
    assert "0/.zarray" in keys
    assert any(key.startswith("0/") and not key.endswith(".zarray") for key in keys)
    for row in receipt["objects"]:
        payload = (target / row["object_key"]).read_bytes()
        assert row["bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()
    canonical = json.dumps(
        receipt["objects"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode()
    assert receipt["canonical_manifest_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_production_mcp_reads_zarr_v2_through_remote_http_store_with_ranges(tmp_path):
    zarr = pytest.importorskip("zarr")
    target = tmp_path / "remote.zarr"
    group = zarr.open_group(str(target), mode="w", zarr_format=2)
    group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[4, 5, 6] = 220
    group.create_array("0", data=data, chunks=(8, 8, 8), compressor=None)
    requests = []

    class RangeHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            requests.append((self.path, self.headers.get("Range")))
            return super().do_GET()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/remote.zarr"
        result = mcp_server.Service(None).find_seed_candidates({
            "prediction_uri": url, "prediction_space": "ct_l0_xyz",
            "region": region(size=8), "max_candidates": 2,
            "minimum_separation_voxels": 1, "threshold": 0.2,
        })
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert result["candidate_count"] == 1
    receipt = result["source_read_set"]
    assert {".zattrs", "0/.zarray"} <= {
        row["object_key"] for row in receipt["objects"]}
    assert requests
    assert any(path.endswith("0/.zarray") for path, _header in requests)
    for row in receipt["objects"]:
        if "#range=" not in row["object_key"]:
            payload = (target / row["object_key"]).read_bytes()
            assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_production_fsspec_store_records_actual_http_partial_ranges(tmp_path):
    zarr = pytest.importorskip("zarr")
    target = tmp_path / "remote-range.zarr"
    group = zarr.open_group(str(target), mode="w", zarr_format=2)
    group.create_array("0", data=np.zeros((4, 4, 4), dtype=np.uint8), chunks=(2, 2, 2))
    requests = []

    class RangeHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            header = self.headers.get("Range")
            requests.append((self.path, header))
            if not header:
                return super().do_GET()
            start_text, end_text = header.removeprefix("bytes=").split("-", 1)
            start, end = int(start_text), int(end_text)
            path = Path(self.translate_path(self.path))
            payload = path.read_bytes()[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{path.stat().st_size}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/remote-range.zarr"
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.sync import sync
        store, tracker = mcp_server.prediction_store_with_read_set(url)
        value = sync(store.get(
            "0/.zarray", default_buffer_prototype(),
            RangeByteRequest(start=1, end=8)))
        repeated = sync(store.get(
            "0/.zarray", default_buffer_prototype(),
            RangeByteRequest(start=1, end=8)))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    expected = (target / "0/.zarray").read_bytes()[1:8]
    assert value.to_bytes() == expected
    assert repeated.to_bytes() == expected
    assert any(header == "bytes=1-7" for _path, header in requests)
    assert tracker.receipt()["objects"] == [{
        "object_key": "0/.zarray#range=1:8",
        "sha256": hashlib.sha256(expected).hexdigest(), "bytes": len(expected),
        "byte_range": {"kind": "range", "start": 1, "end": 8},
    }]


def test_prediction_open_forces_v2_and_refuses_consolidated_only_substitution(tmp_path):
    zarr = pytest.importorskip("zarr")
    v3 = tmp_path / "v3.zarr"
    zarr.open_array(str(v3), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), zarr_format=3)
    with pytest.raises(Exception):
        mcp_server.open_prediction_with_read_set(str(v3), None)

    consolidated = tmp_path / "consolidated-only.zarr"
    group = zarr.open_group(str(consolidated), mode="w", zarr_format=2)
    group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    group.create_array("0", data=np.zeros((4, 4, 4), dtype=np.uint8), chunks=(2, 2, 2))
    zarr.consolidate_metadata(str(consolidated))
    (consolidated / "0/.zarray").unlink()
    with pytest.raises(Exception):
        mcp_server.open_prediction_with_read_set(str(consolidated), None)


def test_range_receipts_use_a_stable_numeric_identity_not_object_repr():
    from zarr.abc.store import RangeByteRequest

    class Payload:
        def to_bytes(self):
            return b"abc"

    tracker = mcp_server.PredictionReadTracker()
    tracker.record("0/0", Payload(), RangeByteRequest(start=2, end=5))
    row = tracker.receipt()["objects"][0]
    assert row == {
        "object_key": "0/0#range=2:5", "sha256": hashlib.sha256(b"abc").hexdigest(),
        "bytes": 3, "byte_range": {"kind": "range", "start": 2, "end": 5},
    }


def test_it_finds_what_is_above_threshold():
    volume = empty()
    volume[10, 20, 30] = 200          # 0.78, well above 0.2
    volume[40, 40, 40] = 12           # 0.05, below
    found = seeds.find_candidates(volume, region(), prediction_uri=URI)
    assert len(found) == 1
    assert found[0]["ct_l0_coordinate"] == {"x": 30, "y": 20, "z": 10}
    assert found[0]["surface_score"] == pytest.approx(200 / 255)


def test_candidates_are_kept_apart(): 
    """Two points a voxel apart are the same place to a grower."""
    volume = empty()
    for offset in range(5):
        volume[10, 20, 30 + offset] = 200
    found = seeds.find_candidates(volume, region(), prediction_uri=URI,
                                  minimum_separation_voxels=16)
    assert len(found) == 1, "a cluster produced more than one seed"


def test_separation_is_euclidean_not_per_axis():
    """Diagonally adjacent is still adjacent."""
    volume = empty()
    volume[10, 10, 10] = 200
    volume[19, 19, 19] = 200          # 15.6 apart, inside a 16 radius
    found = seeds.find_candidates(volume, region(), prediction_uri=URI,
                                  minimum_separation_voxels=16)
    assert len(found) == 1


def test_the_strongest_survives_a_cluster():
    """The separation filter must not keep whichever the scan reached first."""
    volume = empty()
    volume[10, 10, 10] = 90
    volume[10, 10, 12] = 250
    found = seeds.find_candidates(volume, region(), prediction_uri=URI,
                                  minimum_separation_voxels=16)
    assert len(found) == 1
    assert found[0]["ct_l0_coordinate"]["x"] == 12


def test_equal_scores_have_a_stable_global_xyz_order():
    """Tie order is part of candidate-set identity; numpy's incidental sort
    order cannot be allowed to vary it by version or platform."""
    volume = empty()
    volume[2, 50, 30] = 200
    volume[50, 2, 10] = 200
    volume[1, 40, 10] = 200
    found = seeds.find_candidates(
        volume,
        region(),
        prediction_uri=URI,
        minimum_separation_voxels=1,
    )
    assert [
        (row["x"], row["y"], row["z"]) for row in found
    ] == [
        (10, 2, 50),
        (10, 40, 1),
        (30, 50, 2),
    ]


def test_ids_are_stable_for_the_same_point():
    """The planner records the candidate it chose and the validator checks the
    choice copies one that was offered. A fresh id per probe breaks both."""
    volume = empty()
    volume[10, 20, 30] = 200
    first = seeds.find_candidates(volume, region(), prediction_uri=URI)
    second = seeds.find_candidates(volume, region(), prediction_uri=URI)
    assert first[0]["candidate_id"] == second[0]["candidate_id"]
    # And different volumes do not collide on the same coordinate.
    other = seeds.find_candidates(volume, region(), prediction_uri=URI + "-rescan")
    assert other[0]["candidate_id"] != first[0]["candidate_id"]


def test_max_candidates_is_respected():
    volume = empty(size=128)
    for index in range(10):
        volume[10, 10, index * 12] = 200
    found = seeds.find_candidates(volume, region(size=128), prediction_uri=URI,
                                  max_candidates=3, minimum_separation_voxels=8)
    assert len(found) == 3


def test_the_chunk_cap_from_the_lost_patch_is_kept():
    """27 is a 3x3x3 cube, which is what a radius-128 probe needs. The patch
    that raised it from 8 is one of the three that outlived the source."""
    with pytest.raises(seeds.SeedSearchError, match="more than the"):
        seeds.chunk_span(region(size=4 * 128))
    assert seeds.chunk_span(region(size=128)) <= seeds.MAX_CANDIDATE_CHUNKS


def test_chunk_accounting_uses_exclusive_high_bounds_and_actual_chunks():
    volume = CountingVolume(empty(), chunks=(32, 16, 8))
    box = {
        "x_min": 0, "x_max": 16,
        "y_min": 0, "y_max": 16,
        "z_min": 0, "z_max": 32,
    }
    # x touches two 8-wide chunks; y and z touch one each. Boundary-aligned
    # exclusive highs do not add phantom chunks.
    assert seeds.chunk_span(box, volume) == 2


def test_the_cap_cannot_be_raised_by_the_environment(monkeypatch):
    """It could be lowered temporarily and never raised, which is what the
    patch's own hard maximum said."""
    monkeypatch.setenv("VC_MCP_MAX_SEED_CANDIDATE_CHUNKS", "64")
    with pytest.raises(seeds.SeedSearchError, match="must be 1 through 27"):
        seeds.chunk_span(region(size=128))


def test_voxel_and_byte_caps_are_checked_before_the_volume_is_read(monkeypatch):
    volume = CountingVolume(empty(size=16))
    monkeypatch.setenv("VC_MCP_MAX_SEED_CANDIDATE_VOXELS", "100")
    with pytest.raises(seeds.SeedSearchError, match="voxels"):
        seeds.find_candidates(volume, region(size=16), prediction_uri=URI)
    assert volume.read_count == 0

    monkeypatch.delenv("VC_MCP_MAX_SEED_CANDIDATE_VOXELS")
    monkeypatch.setenv("VC_MCP_MAX_SEED_CANDIDATE_READ_BYTES", "100")
    with pytest.raises(seeds.SeedSearchError, match="bytes"):
        seeds.find_candidates(volume, region(size=16), prediction_uri=URI)
    assert volume.read_count == 0


def test_actual_small_chunks_can_refuse_a_read_before_it_happens():
    volume = CountingVolume(empty(), chunks=(16, 16, 16))
    with pytest.raises(seeds.SeedSearchError, match="64 chunks"):
        seeds.find_candidates(volume, region(), prediction_uri=URI)
    assert volume.read_count == 0


def test_unsupported_integer_dtype_is_refused_before_read():
    volume = CountingVolume(np.zeros((16, 16, 16), dtype=np.uint16))
    with pytest.raises(seeds.SeedSearchError, match="unsupported"):
        seeds.find_candidates(volume, region(size=16), prediction_uri=URI)
    assert volume.read_count == 0


def test_float_predictions_must_be_finite_and_normalized():
    volume = empty(size=16).astype(np.float32)
    volume[1, 1, 1] = np.nan
    with pytest.raises(seeds.SeedSearchError, match="finite"):
        seeds.find_candidates(volume, region(size=16), prediction_uri=URI)

    volume[1, 1, 1] = 1.1
    with pytest.raises(seeds.SeedSearchError, match=r"normalized to \[0,1\]"):
        seeds.find_candidates(volume, region(size=16), prediction_uri=URI)


def test_local_structure_evidence_is_measured_from_the_single_query_read():
    array = empty(size=48)
    # A 5x5 one-voxel-thick component has a clean planar PCA signature.
    array[24, 20:25, 20:25] = 200
    # A disconnected, weaker local peak supplies a measured competitor.
    array[35, 35, 35] = 180
    volume = CountingVolume(array)

    found = seeds.find_candidates(
        volume,
        region(size=48),
        prediction_uri=URI,
        max_candidates=1,
    )
    assert volume.read_count == 1
    evidence = found[0]["candidate_evidence"]
    assert evidence["schema"] == "campaignx.m7_seed_candidate_evidence.v1"
    assert evidence["policy_id"] == "m7-local-structure-v1"
    assert evidence["ink_used"] is False
    assert evidence["score_semantics"] == "normalized_m7_intensity_not_probability"

    component = evidence["threshold_component"]
    assert component["connectivity"] == 26
    assert component["positive_component_count"] == 2
    assert component["seed_component_voxel_count"] == 25
    assert component["bounding_box_extent_xyz_voxels"] == {"x": 5, "y": 5, "z": 1}
    assert component["nearest_disconnected_positive_distance_voxels"] is not None

    shape = evidence["shape_descriptor"]
    assert shape["point_count"] == 25
    assert shape["eigenvalues_descending_voxels_squared"] == [2.0, 2.0, 0.0]
    assert shape["linearity"] == 0.0
    assert shape["planarity"] == 1.0
    assert shape["scattering"] == 0.0

    competition = evidence["spatial_peak_competition"]
    assert competition["point_is_local_maximum"] is True
    assert competition["distinct_peak_count"] == 2
    assert competition["competing_peak_count"] == 1
    assert competition["strongest_competitor_normalized_m7_intensity"] == pytest.approx(
        180 / 255, abs=1e-8)
    assert competition[
        "point_minus_strongest_competitor_normalized_m7_intensity"
    ] == pytest.approx(20 / 255, abs=1e-8)
    schema = json.loads((
        ROOT / "framework" / "contracts" / "schemas"
        / "m7-seed-candidate-evidence-v1.schema.json"
    ).read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(evidence)


def test_evidence_marks_query_truncation_and_component_censoring_explicitly():
    volume = empty(size=32)
    volume[0, 0, 0:6] = 220
    evidence = seeds.find_candidates(
        volume,
        region(size=32),
        prediction_uri=URI,
        max_candidates=1,
    )[0]["candidate_evidence"]

    assert evidence["window"]["coverage"] == "QUERY_BOUNDARY_TRUNCATED"
    assert evidence["window"]["query_faces_truncated"] == ["x_min", "y_min", "z_min"]
    assert evidence["threshold_component"]["observation_state"] == "WINDOW_CENSORED"
    assert set(evidence["threshold_component"]["touched_window_faces"]) >= {
        "x_min", "y_min", "z_min",
    }


def test_evidence_policy_bounds_are_enforced_before_read():
    volume = CountingVolume(empty())
    with pytest.raises(seeds.SeedSearchError, match="evidence_window_radius"):
        seeds.find_candidates(
            volume, region(), prediction_uri=URI, evidence_window_radius_voxels=3)
    assert volume.read_count == 0

    with pytest.raises(seeds.SeedSearchError, match="peak_neighborhood_radius"):
        seeds.find_candidates(
            volume, region(), prediction_uri=URI, peak_neighborhood_radius_voxels=9)
    assert volume.read_count == 0


def test_clearance_is_null_rather_than_zero():
    """This service does not know what is already segmented. Reporting 0 would
    let an unmeasured distance be read as a measured one."""
    volume = empty()
    volume[10, 20, 30] = 200
    found = seeds.find_candidates(volume, region(), prediction_uri=URI)[0]
    assert found["clearance_voxels"] is None
    assert found["cell_interior_clearance_voxels"] is None


def test_an_inverted_region_is_refused():
    bad = region()
    bad["x_max"] = bad["x_min"]
    with pytest.raises(seeds.SeedSearchError, match="no extent"):
        seeds.chunk_span(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
