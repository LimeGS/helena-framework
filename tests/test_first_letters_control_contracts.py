"""Real contract tests for the First Letters control's new platform seams."""

from __future__ import annotations

import json
import hashlib
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from fleet.candidate_preflight import (  # noqa: E402
    _bilinear_distance,
    _load_locked_tifxyz,
    _surface_regions,
    run_control_region_preflight,
)
from fleet.common import content_sha256  # noqa: E402
from fleet.ct_support import OmeZarrCtSupportSampler  # noqa: E402
from fleet.worker import McpSeedProvider, SourceProviderUnavailable  # noqa: E402
from job_store import command_for, validate_parameters  # noqa: E402
import ink_worker  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

# Derived from the version the panel actually loads, not written out again here.
# Spelling it a second time makes "which manifest is in force" two facts that
# happen to agree, and the day they stop agreeing the panel seals a P0 under one
# profile while the run declares another -- a 409 that costs a whole run to
# discover. Read from the panel's source rather than its loader, so this still
# compares two independent things.
_IN_FORCE = re.search(r"first-letters-control-policy-(\d+\.\d+\.\d+)\.json",
                      (ROOT / "panel/app.py").read_text(encoding="utf-8")).group(1)
MANIFEST = (ROOT / "framework/profiles/01-segmentation"
            / f"first-letters-control-policy-{_IN_FORCE}.json")


def read_set(key: str, byte: str = "a") -> dict:
    objects = [{"object_key": key, "sha256": byte * 64, "bytes": 10}]
    return {
        "schema": "campaignx.first_letters_source_read_set.v1",
        "objects": objects,
        "canonical_manifest_sha256": content_sha256(objects),
    }


def preflight_request() -> dict:
    return {
        "region_center_xyz": {"x": 20, "y": 20, "z": 20},
        "region_radius_xyz": {"x": 10, "y": 10, "z": 10},
        "known_coordinate_xyz": {"x": 20, "y": 20, "z": 20},
        "tolerance_ct_l0_voxels": 2,
        "m7_threshold": 0.2,
        "max_candidates": 8,
        "packet_candidate_limit": 8,
        "minimum_separation_voxels": 0,
        "minimum_cell_clearance_voxels": 0,
        "minimum_volume_clearance_voxels": 0,
        "seed_region_policy": "fixed-v1",
        "ct_material_support_gate": {
            "policy": "ome-zarr-nearby-material-v1", "level": 0,
            "radius_l0_voxels": 1, "minimum_nonzero_voxels": 1,
        },
    }


def route_preflight_request() -> dict:
    control = json.loads(MANIFEST.read_text())
    known = control["known_region"]
    minimum = known["surface_bbox_ct_l0_xyz"]["minimum"]
    maximum = known["surface_bbox_ct_l0_xyz"]["maximum"]
    discovery = control["checks"]["DISCOVERY_CONTROL"]
    execution = discovery["execution_parameters"]
    return {
        "region_center_xyz": {axis: (minimum[index] + maximum[index]) / 2
                              for index, axis in enumerate("xyz")},
        "region_radius_xyz": {axis: (maximum[index] - minimum[index]) / 2
                              for index, axis in enumerate("xyz")},
        "known_coordinate_xyz": dict(zip("xyz", known["anchor_ct_l0_xyz"], strict=True)),
        "tolerance_ct_l0_voxels": known["control_tolerance_ct_l0_voxels"],
        "m7_threshold": discovery["inputs"]["m7_level_set_iso_value"],
        "max_candidates": execution["max_candidates"],
        "packet_candidate_limit": execution["packet_candidate_limit"],
        "minimum_separation_voxels": execution["minimum_separation_voxels"],
        "minimum_cell_clearance_voxels": execution["minimum_cell_clearance_voxels"],
        "minimum_volume_clearance_voxels": execution["minimum_volume_clearance_voxels"],
        "seed_region_policy": discovery["inputs"]["seed_region_policy"],
        "ct_material_support_gate": execution["ct_material_support_gate"],
        "maximum_surface_probe_regions": execution["maximum_surface_probe_regions"],
    }


def snapshot() -> dict:
    return {
        "source_snapshot_id": "source-current", "sample_id": "PHerc0139",
        "ct_uri": "memory://ct", "m7_uri": "memory://m7",
        "shape_xyz": [100, 100, 100], "voxel_size_um": 9.362,
        "coordinate_frame": "frame",
    }


# -- where the hour goes -------------------------------------------------------
#
# A preflight took 4184 seconds and visited twelve regions. Profiling the parts
# in isolation accounted for about fifty: four seconds of candidate search per
# region, four to fetch and verify the locked surface, a fifth of a second to
# plan 3052 regions, and nothing measurable for the CT gate. The other 98% could
# not be attributed, because the one link left -- the provider call as the loop
# actually makes it -- is only reachable with credentials the entrypoint mints
# inside the container and never writes down.
#
# So the receipt carries the timing itself. Not to make it faster: to stop the
# next person having to reconstruct it from wall clocks and arithmetic.

def test_the_receipt_says_how_long_each_region_took():
    class Provider:
        def discover(self, _task):
            return {
                "candidates": [{"candidate_id": "c1",
                                "ct_l0_coordinate": {"x": 20, "y": 20, "z": 20},
                                "score": 1}],
                "source_read_set": read_set("m7/0", "b"),
                "provider_exchange": {
                    "request_sha256": "c" * 64, "request_bytes": 12,
                    "response_sha256": "d" * 64, "response_bytes": 34,
                },
            }

    class Ct:
        def sample(self, *_args, **_kwargs):
            return {"nonzero_voxel_count": 1,
                    "source_read_set": read_set("0/chunk-1", "1")}

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=Provider(), ct_sampler=Ct())

    timings = result["region_timings"]
    assert len(timings) == result["visited_region_count"], (
        "a region was walked and not timed, or timed and not walked")
    for row in timings:
        for field in ("discover_seconds", "ct_gate_seconds", "screen_seconds"):
            assert isinstance(row[field], (int, float)) and row[field] >= 0.0, row
    assert isinstance(result["elapsed_seconds"], (int, float))


def test_the_timing_is_ink_blind_so_the_worker_will_persist_it():
    """The finishing gate walks the whole document. A field it refuses is a
    field that turns every preflight into PREFLIGHT_RECEIPT_NOT_INK_BLIND."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "framework/stages/01-segmentation"))
    from fleet.preflight_worker import _refusal

    class Provider:
        def discover(self, _task):
            return {
                "candidates": [],
                "source_read_set": read_set("m7/0", "b"),
                "provider_exchange": {
                    "request_sha256": "c" * 64, "request_bytes": 12,
                    "response_sha256": "d" * 64, "response_bytes": 34,
                },
            }

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=Provider(), ct_sampler=object())

    assert "region_timings" in result
    assert _refusal(result) is None, _refusal(result)


def test_a_slow_region_is_visible_in_its_own_row():
    """One row per region is the point: a mean would hide the region that cost
    the hour among eleven that cost seconds."""
    import time

    class SlowProvider:
        def __init__(self):
            self.calls = 0

        def discover(self, _task):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.05)
            return {
                "candidates": [],
                "source_read_set": read_set("m7/0", "b"),
                "provider_exchange": {
                    "request_sha256": "c" * 64, "request_bytes": 12,
                    "response_sha256": "d" * 64, "response_bytes": 34,
                },
            }

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=SlowProvider(), ct_sampler=object())

    assert result["region_timings"][0]["discover_seconds"] >= 0.05


def test_preflight_merges_all_candidate_ct_reads_and_never_grows():
    class Provider:
        def discover(self, _task):
            return {
                "candidates": [
                    {"candidate_id": "c1", "ct_l0_coordinate": {"x": 20, "y": 20, "z": 20}, "score": 1},
                    {"candidate_id": "c2", "ct_l0_coordinate": {"x": 21, "y": 20, "z": 20}, "score": .9},
                ],
                "source_read_set": read_set("m7/0", "b"),
                "provider_exchange": {
                    "request_sha256": "c" * 64, "request_bytes": 12,
                    "response_sha256": "d" * 64, "response_bytes": 34,
                },
            }

    class Ct:
        calls = 0

        def sample(self, *_args, **_kwargs):
            self.calls += 1
            return {"nonzero_voxel_count": 1,
                    "source_read_set": read_set(f"0/chunk-{self.calls}", str(self.calls))}

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=Provider(), ct_sampler=Ct())
    assert result["status"] == "COMPLETE"
    assert [row["object_key"] for row in result["ct_read_set"]["objects"]] == [
        "0/chunk-1", "0/chunk-2"]
    assert result["state_mutation"] == "NONE"
    assert result["growth_allowed"] is False and result["ink_used"] is False


def test_preflight_fails_closed_when_provider_read_evidence_is_missing():
    class Provider:
        def discover(self, _task):
            return {"candidates": [], "provider_exchange": {
                "request_sha256": "c" * 64, "request_bytes": 1,
                "response_sha256": "d" * 64, "response_bytes": 1,
            }}

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=Provider(), ct_sampler=object())
    assert result["status"] == "INCOMPLETE_SOURCE_BINDING"


# -- the frozen root objects ---------------------------------------------------
#
# The manifest freezes `.zattrs` and `0/.zarray` for CT and for M7. They are not
# data: they are what says *this* volume. The panel refuses a receipt whose read
# sets do not contain them, because a measurement that never touched them cannot
# show it ran against the volume the control was frozen to.
#
# The measurement did not read them. It opens the group and works at level 5, so
# a real run on the deployment produced `.zattrs, .zgroup, 5/.zarray, 5/.zattrs,
# 5/1/0/0, 5/1/0/1` and the control stopped at P1 with
# FROZEN_ROOT_OBJECT_EVIDENCE_MISSING -- with the measurement itself complete.
#
# So it reads them, byte-verifies them, and records them. This adds evidence; it
# changes no threshold, no sampling level and no count. The alternative was to
# edit the frozen manifest to ask for the level the code happens to read, which
# is the same thing as deciding the answer.

def _frozen(path: str, payload: bytes) -> dict:
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}


class _ObjectStore:
    """Stands in for the store the frozen roots are fetched from."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.asked: list[str] = []

    def __call__(self, url: str, timeout: float | None = None):
        self.asked.append(url)
        key = url.split("://", 1)[-1].split("/", 1)[-1]
        if key not in self.payloads:
            raise OSError(f"no such object: {url}")
        return io.BytesIO(self.payloads[key])


def _provider_and_ct():
    class Provider:
        def discover(self, _task):
            return {
                "candidates": [{"candidate_id": "c1",
                                "ct_l0_coordinate": {"x": 20, "y": 20, "z": 20},
                                "score": 1}],
                "source_read_set": read_set("m7/0", "b"),
                "provider_exchange": {
                    "request_sha256": "c" * 64, "request_bytes": 12,
                    "response_sha256": "d" * 64, "response_bytes": 34,
                },
            }

    class Ct:
        def sample(self, *_args, **_kwargs):
            return {"nonzero_voxel_count": 1,
                    "source_read_set": read_set("5/1/0/0", "e")}

    return Provider(), Ct()


def _roots_and_store():
    payloads = {".zattrs": b"{\"multiscales\": []}", "0/.zarray": b"{\"shape\": [1]}"}
    frozen = {
        "ct": [_frozen(".zattrs", payloads[".zattrs"]),
               _frozen("0/.zarray", payloads["0/.zarray"])],
        "m7": [_frozen(".zattrs", payloads[".zattrs"]),
               _frozen("0/.zarray", payloads["0/.zarray"])],
    }
    return frozen, _ObjectStore(payloads)


def test_the_frozen_root_objects_are_read_and_recorded():
    provider, ct = _provider_and_ct()
    frozen, store = _roots_and_store()

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=provider, ct_sampler=ct,
        frozen_root_objects=frozen, root_object_reader=store)

    assert result["status"] == "COMPLETE"
    for name in ("ct_read_set", "m7_read_set"):
        keys = [row["object_key"] for row in result[name]["objects"]]
        assert ".zattrs" in keys and "0/.zarray" in keys, f"{name}: {keys}"
    # What the run actually read is still there; the roots are added, not
    # substituted.
    assert "5/1/0/0" in [row["object_key"] for row in result["ct_read_set"]["objects"]]
    assert "memory://ct/0/.zarray" in store.asked
    assert "memory://m7/0/.zarray" in store.asked


def test_the_receipt_satisfies_the_check_that_refused_it():
    """Both sides of the contract in one test, using the panel's own function.

    Asserting the read set "contains the frozen objects" in the test's own words
    is how these two halves drifted apart in the first place.
    """
    pytest.importorskip("fastapi")
    from panel.app import _read_set_contains_frozen_metadata

    provider, ct = _provider_and_ct()
    frozen, store = _roots_and_store()

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=provider, ct_sampler=ct,
        frozen_root_objects=frozen, root_object_reader=store)

    assert _read_set_contains_frozen_metadata(result["ct_read_set"], frozen["ct"])
    assert _read_set_contains_frozen_metadata(result["m7_read_set"], frozen["m7"])


def test_a_root_object_whose_bytes_drifted_is_refused():
    """The read is a verification, not a formality. Recording a hash the
    manifest did not freeze would make the binding a decoration."""
    provider, ct = _provider_and_ct()
    frozen, store = _roots_and_store()
    store.payloads["0/.zarray"] = b"{\"shape\": [2]}"

    with pytest.raises(ValueError, match="0/.zarray"):
        run_control_region_preflight(
            snapshot(), preflight_request(), provider=provider, ct_sampler=ct,
            frozen_root_objects=frozen, root_object_reader=store)


def test_without_frozen_roots_the_measurement_is_what_it_was():
    """Callers that lock no root objects are unchanged, and nothing is fetched."""
    provider, ct = _provider_and_ct()
    _, store = _roots_and_store()

    result = run_control_region_preflight(
        snapshot(), preflight_request(), provider=provider, ct_sampler=ct,
        root_object_reader=store)

    assert result["status"] == "COMPLETE"
    assert store.asked == []
    assert [row["object_key"] for row in result["ct_read_set"]["objects"]] == ["5/1/0/0"]


def test_control_surface_preflight_tiles_all_valid_bilinear_regions_and_uses_surface_distance():
    x = np.asarray([[10.0, 160.0, 310.0], [10.0, 160.0, 310.0]])
    y = np.asarray([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    z = np.full_like(x, 30.0)
    surface_objects = [{"object_key": "meta.json", "sha256": "f" * 64, "bytes": 9}]
    surface_read_set = {
        "schema": "campaignx.first_letters_source_read_set.v1",
        "objects": surface_objects,
        "canonical_manifest_sha256": content_sha256(surface_objects),
    }

    class Provider:
        def __init__(self):
            self.centers = []

        def discover(self, task):
            center = dict(task["center_xyz"])
            self.centers.append(center)
            index = len(self.centers)
            coordinate = ({"x": 999.0, "y": 999.0, "z": 999.0} if index < 3
                          else {"x": 310.0, "y": 15.0, "z": 30.0})
            return {
                "candidates": [{"candidate_id": (f"overlap-candidate-{index}" if index < 3 else "surface-hit"),
                                "ct_l0_coordinate": coordinate, "score": 1.0}],
                "source_read_set": read_set(f"m7/{index}", "a"),
                "provider_exchange": {
                    "request_sha256": f"{index:x}" * 64, "request_bytes": 10,
                    "response_sha256": "e" * 64, "response_bytes": 20,
                },
            }

    class Ct:
        def __init__(self):
            self.calls = 0

        def sample(self, *_args, **_kwargs):
            self.calls += 1
            return {"nonzero_voxel_count": 1,
                    "source_read_set": read_set(f"ct/{self.calls}", "b")}

    request = preflight_request()
    request.update({
        "control_surface": {"uri": "https://locked/surface", "artifacts": [],
                            "grid_shape_yx": [2, 3]},
        "maximum_surface_probe_regions": 32,
    })
    provider = Provider()
    result = run_control_region_preflight(
        {**snapshot(), "shape_xyz": [1000, 1000, 1000]},
        request, provider=provider, ct_sampler=Ct(),
        surface_loader=lambda _spec: ((x, y, z), surface_read_set),
    )
    assert len(provider.centers) >= 3
    assert result["planned_region_count"] >= len(provider.centers) >= 3
    assert result["visited_region_count"] == len(provider.centers)
    assert result["counts"]["raw_m7"] == 2
    assert result["duplicate_candidate_count"] == 1
    assert result["candidate_identity_aliases"] == [{
        "coordinate_xyz": [999.0, 999.0, 999.0],
        "observed_candidate_ids": ["overlap-candidate-1", "overlap-candidate-2"],
    }]
    assert result["surface_read_set"] == surface_read_set
    assert result["within_tolerance"] is True
    assert result["closest_survivor_distance_ct_l0_voxels"] < 1e-6

    limited = {**request, "maximum_surface_probe_regions": 2}
    incomplete = run_control_region_preflight(
        {**snapshot(), "shape_xyz": [2000, 2000, 2000]},
        limited, provider=Provider(), ct_sampler=Ct(),
        surface_loader=lambda _spec: ((x, y, z), surface_read_set),
    )
    assert incomplete["status"] == "INCOMPLETE_REGION_COVERAGE"
    assert incomplete["visited_region_count"] == 2
    assert incomplete["coverage_complete"] is False
    assert incomplete["coverage_fraction"] < 1.0


def test_locked_tifxyz_loader_verifies_all_four_source_artifact_hashes(monkeypatch):
    import tifffile

    payloads = {"meta.json": b'{"width":2,"height":2}'}
    for name, value in (("x.tif", 1), ("y.tif", 2), ("z.tif", 3)):
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, np.full((2, 2), value, dtype=np.float32))
        payloads[name] = buffer.getvalue()

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *_args): return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=60: Response(
        payloads[str(url).rsplit("/", 1)[-1]]))
    spec = {
        "uri": "https://locked/surface.tifxyz", "grid_shape_yx": [2, 2],
        "artifacts": [{"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
                      for name, payload in payloads.items()],
    }
    arrays, receipt = _load_locked_tifxyz(spec)
    assert arrays[0].shape == (2, 2)
    assert [row["object_key"] for row in receipt["objects"]] == [
        "meta.json", "x.tif", "y.tif", "z.tif"]

    spec["artifacts"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="byte hash drifted at x.tif"):
        _load_locked_tifxyz(spec)


def test_bilinear_surface_distance_is_finite_for_a_far_candidate():
    x = np.asarray([[0.0, 1.0], [0.0, 1.0]])
    y = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    z = np.zeros((2, 2))
    distance = _bilinear_distance({"x": 100.0, "y": 100.0, "z": 100.0}, (x, y, z))
    assert np.isfinite(distance)
    assert distance > 2.0


def test_finite_negative_one_tifxyz_sentinel_is_not_a_valid_surface_vertex():
    x = np.asarray([[-1.0, -1.0, -1.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    y = np.asarray([[-1.0, -1.0, -1.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    z = np.asarray([[-1.0, -1.0, -1.0], [30.0, 30.0, 30.0], [30.0, 30.0, 30.0]])
    regions = _surface_regions((x, y, z))
    assert regions
    assert all(center[axis] >= 0 for center in regions for axis in "xyz")


def test_mcp_provider_binds_canonical_request_response_and_server_read_set(monkeypatch):
    response = {
        "candidates": [{"candidate_id": "c", "ct_l0_coordinate": {"x": 1, "y": 2, "z": 3}}],
        "source_read_set": read_set("m7/chunk", "e"),
    }

    class Client:
        def __init__(self, *_args): pass
        def initialize(self): pass
        def call(self, *_args): return response

    monkeypatch.setitem(sys.modules, "campaign_x", SimpleNamespace(
        McpClient=Client, structured=lambda value: value))
    task = {
        "task_id": "t", "attempt_id": "a",
        "candidate_discovery": {
            "prediction_uri": "locked-m7", "prediction_space": "ct_l0_xyz",
            "region": {"center": {"x": 1, "y": 2, "z": 3},
                       "radius": {"x": 1, "y": 1, "z": 1}},
            "max_candidates": 1, "seed_region_policy": "fixed-v1",
        },
    }
    found = McpSeedProvider("https://mcp.invalid", "secret").discover(task)
    assert found["source_read_set"] == response["source_read_set"]
    assert found["provider_exchange"]["request_bytes"] > 0
    assert found["provider_exchange"]["response_bytes"] > 0
    assert "secret" not in json.dumps(found)


def test_mcp_provider_refuses_candidate_results_without_production_source_evidence(monkeypatch):
    class Client:
        def __init__(self, *_args): pass
        def initialize(self): pass
        def call(self, *_args): return {"candidates": []}

    monkeypatch.setitem(sys.modules, "campaign_x", SimpleNamespace(
        McpClient=Client, structured=lambda value: value))
    task = {
        "task_id": "t", "attempt_id": "a",
        "candidate_discovery": {
            "prediction_uri": "locked-m7", "prediction_space": "ct_l0_xyz",
            "region": {"center": {"x": 1, "y": 2, "z": 3},
                       "radius": {"x": 1, "y": 1, "z": 1}},
            "max_candidates": 1, "seed_region_policy": "fixed-v1",
        },
    }
    with pytest.raises(SourceProviderUnavailable, match="source read evidence"):
        McpSeedProvider("https://mcp.invalid", "secret").discover(task)


def test_ct_sampler_records_zarr_metadata_and_chunk_bytes(tmp_path):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "ct.zarr"
    group = zarr.open_group(str(root), mode="w", zarr_format=2)
    group.attrs["multiscales"] = [{"datasets": [{
        "path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}],
    }]}]
    group.create_array("0", data=np.ones((4, 4, 4), dtype=np.uint8), chunks=(2, 2, 2))
    sampled = OmeZarrCtSupportSampler().sample(
        str(root), {"x": 2, "y": 2, "z": 2}, level=0, radius_l0_voxels=1)
    keys = [row["object_key"] for row in sampled["source_read_set"]["objects"]]
    assert any(key.endswith(".zattrs") for key in keys)
    assert any("0/" in key and not key.endswith(".zarray") for key in keys)
    assert sampled["source_read_set"]["canonical_manifest_sha256"] == content_sha256(
        sampled["source_read_set"]["objects"])


def test_p2_and_p4_command_builders_preserve_exact_lineage_and_frozen_render_parameters():
    p2 = command_for({"job_id": "p2-current-job", "phase": "P2",
        "sample_id": "PHerc0139", "parameters": {
        "limit": 1, "sample": "PHerc0139", "surface_id": "surface-control",
    }}, runner="fleet.py", output_dir="/tmp/out")
    assert p2[p2.index("--surface-id") + 1] == "surface-control"
    assert p2[p2.index("--requested-by-job-id") + 1] == "p2-current-job"

    p4_parameters = validate_parameters({
        "lane": "vc-render-tifxyz", "flattened_surface": "surface-control",
        "flattening_profile": "flatten-abf-v1@1.0.0", "volume": "/control-cache/ct",
        "flattening_id": "flat-control", "p3_job_id": "p3-current",
        "flattened_artifact_sha256": "4" * 64,
        "remote_url": "https://locked/ct.zarr", "scale": 1.0, "group_idx": 0,
        "num_slices": 33, "slice_step": 1.0, "flip_normals": False,
        "orientation_receipt_sha256": "6" * 64,
    }, "P4")
    p4_parameters["segmentation"] = "/tmp/materialized-flat"
    argv = command_for({"phase": "P4", "sample_id": "PHerc0139",
                        "parameters": p4_parameters},
                       runner="vc_render_tifxyz", output_dir="/tmp/out")
    assert "/control-cache/ct" in argv and "https://locked/ct.zarr" in argv
    assert "33" in argv


def test_actual_control_orientation_route_is_content_bound_and_unproven(
        panel_session, monkeypatch):
    control = json.loads(MANIFEST.read_text())
    policy = control["checks"]["PIPELINE_CONTROL"]["orientation_parity"]["policy"]
    p3_receipt_sha = "5" * 64
    flattened_sha = "4" * 64

    class OrientationStore:
        def job(self, job_id):
            assert job_id == "p3-current"
            return {
                "job_id": job_id, "phase": "P3", "state": "succeeded",
                "mission_id": "control-mission", "sample_id": "PHerc0139",
                "parameters": {"surface_id": "surface-control"},
                "result": {"surfaces": [{
                    "surface_id": "surface-control",
                    "requested_by_job_id": job_id,
                    "source_artifact_sha256": "3" * 64,
                    "artifact_id": "flat-control",
                    "artifact_sha256": flattened_sha,
                    "profile_id": "flatten-abf-v1@1.0.0",
                    "profile_file_sha256": next(
                        row["sha256"] for row in control["profile_locks"]
                        if row["profile_id"] == "flatten-abf-v1@1.0.0"),
                    "receipt_sha256": p3_receipt_sha,
                }]},
            }

        def flattened_sheet(self, _surface, _profile):
            return {"flattening_id": "flat-control",
                    "requested_by_job_id": "p3-current",
                    "artifact_sha256": flattened_sha}

    monkeypatch.setattr(panel_session.panel, "job_store", lambda: OrientationStore())
    invoked = []
    monkeypatch.setattr(panel_session.panel, "load_locked_orientation_reference",
                        lambda _spec: ("reference-grid", {"objects": []}))
    monkeypatch.setattr(panel_session.panel, "load_hash_bound_grown_mesh",
                        lambda _surface, _sha, **_expected: (
                            "grown-vertices", "grown-faces"))
    def proving(reference, vertices, faces, lineage, selected_policy):
        invoked.append((reference, vertices, faces, lineage, selected_policy))
        receipt = {
            "schema": "campaignx.first_letters_orientation_parity.v1",
            "profile_id": selected_policy["profile_id"],
            "profile_sha256": content_sha256(selected_policy), "lineage": lineage,
            "policy": selected_policy, "status": "UNPROVEN",
            "reason_code": "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING",
            "parity_state": "PROVEN_SAME_WINDING", "selected_flip_normals": None}
        return {**receipt, "receipt_sha256": content_sha256(receipt)}
    monkeypatch.setattr(panel_session.panel, "prove_control_orientation", proving)
    response = panel_session.client.get(
        "/api/geometry/orientation-proof?mission=control-mission&sample=PHerc0139"
        "&surface=surface-control&p3_job=p3-current")
    assert response.status_code == 200, response.text
    proof = response.json()
    assert proof["status"] == "UNPROVEN"
    assert proof["reason_code"] == "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING"
    assert proof["selected_flip_normals"] is None
    assert proof["profile_sha256"] == content_sha256(policy)
    assert proof["lineage"]["reference"]["objects"] == \
        control["source_locks"]["community_surface"]["artifacts"]
    assert proof["lineage"]["flattened_artifact"] == {
        "artifact_id": "flat-control", "sha256": flattened_sha}
    assert proof["lineage"]["p3"]["receipt_sha256"] == p3_receipt_sha
    assert proof["receipt_sha256"] == content_sha256({
        key: value for key, value in proof.items() if key != "receipt_sha256"})
    assert len(invoked) == 1


def test_orientation_route_invokes_real_prover_and_rejects_grown_hash_drift(
        panel_session, monkeypatch):
    control = json.loads(MANIFEST.read_text())
    flattened = {
        "surface_id": "surface-control", "requested_by_job_id": "p3-current",
        "source_artifact_sha256": "3" * 64, "artifact_id": "flat-control",
        "artifact_sha256": "4" * 64, "profile_id": "flatten-abf-v1@1.0.0",
        "profile_file_sha256": next(row["sha256"] for row in control["profile_locks"]
                                    if row["profile_id"] == "flatten-abf-v1@1.0.0"),
        "receipt_sha256": "5" * 64}
    class Store:
        def job(self, _job_id):
            return {"phase": "P3", "state": "succeeded",
                    "mission_id": "control-mission", "sample_id": "PHerc0139",
                    "parameters": {"surface_id": "surface-control"},
                    "result": {"surfaces": [flattened]}}

        def flattened_sheet(self, _surface, _profile):
            return {"flattening_id": "flat-control",
                    "requested_by_job_id": "p3-current",
                    "artifact_sha256": "4" * 64}
    monkeypatch.setattr(panel_session.panel, "job_store", lambda: Store())
    invoked = []
    monkeypatch.setattr(panel_session.panel, "load_locked_orientation_reference",
                        lambda _spec: ("reference-grid", {"objects": []}))
    monkeypatch.setattr(panel_session.panel, "load_hash_bound_grown_mesh",
                        lambda surface, sha, **_expected: (_ for _ in ()).throw(
                            ValueError(f"grown artifact hash drift: {surface}:{sha}")))
    monkeypatch.setattr(panel_session.panel, "prove_control_orientation",
                        lambda *args: invoked.append(args))
    response = panel_session.client.get(
        "/api/geometry/orientation-proof?mission=control-mission&sample=PHerc0139"
        "&surface=surface-control&p3_job=p3-current")
    assert response.status_code == 409
    assert "hash drift" in response.text
    assert invoked == []


def test_orientation_route_walks_to_the_surface_and_refuses_decoy_chains(
        panel_session, monkeypatch):
    """The P3 job says which surface it flattened. Nothing else gets a vote.

    The job's own result carries a row for a second surface -- the shape a decoy
    takes in real data, where one job emitted more than one thing. Naming it does
    not select it, and two rows for the one surface is a fork, not a preference.
    """
    control = json.loads(MANIFEST.read_text())
    real = {
        "surface_id": "surface-control", "requested_by_job_id": "p3-current",
        "source_artifact_sha256": "3" * 64, "artifact_id": "flat-control",
        "artifact_sha256": "4" * 64, "profile_id": "flatten-abf-v1@1.0.0",
        "profile_file_sha256": next(row["sha256"] for row in control["profile_locks"]
                                    if row["profile_id"] == "flatten-abf-v1@1.0.0"),
        "receipt_sha256": "5" * 64}
    decoy = {**real, "surface_id": "surface-tiny", "artifact_id": "flat-tiny",
             "artifact_sha256": "7" * 64}
    job = {"job_id": "p3-current", "phase": "P3", "state": "succeeded",
           "mission_id": "control-mission", "sample_id": "PHerc0139",
           "parameters": {"surface_id": "surface-control"},
           "result": {"surfaces": [decoy, real]}}

    class Store:
        def job(self, _job_id):
            return job

        def flattened_sheet(self, _surface, _profile):
            return {"flattening_id": "flat-control",
                    "requested_by_job_id": "p3-current",
                    "artifact_sha256": "4" * 64}

    loaded: list[tuple] = []
    monkeypatch.setattr(panel_session.panel, "job_store", lambda: Store())
    monkeypatch.setattr(panel_session.panel, "load_locked_orientation_reference",
                        lambda _spec: ("reference-grid", {"objects": []}))
    monkeypatch.setattr(
        panel_session.panel, "load_hash_bound_grown_mesh",
        lambda surface, sha, **_expected: (
            loaded.append((surface, sha)) or ("grown-vertices", "grown-faces")))
    monkeypatch.setattr(
        panel_session.panel, "prove_control_orientation",
        lambda reference, vertices, faces, lineage, policy: {
            "status": "UNPROVEN", "lineage": lineage, "receipt_sha256": "0" * 64})
    base = ("/api/geometry/orientation-proof?mission=control-mission"
            "&sample=PHerc0139&p3_job=p3-current")

    # The caller names no surface at all, and the walk still finds the one this
    # P3 was asked for rather than the first row in its result.
    answered = panel_session.client.get(base)
    assert answered.status_code == 200, answered.text
    assert answered.json()["lineage"]["flattened_artifact"] == {
        "artifact_id": "flat-control", "sha256": "4" * 64}
    assert loaded == [("surface-control", "3" * 64)]

    named = panel_session.client.get(base + "&surface=surface-tiny")
    assert named.status_code == 409
    assert "CLIENT_SURFACE_ASSERTION_REJECTED" in named.text

    job["result"]["surfaces"] = [decoy, real, {**real, "artifact_id": "flat-second",
                                               "artifact_sha256": "9" * 64}]
    forked = panel_session.client.get(base)
    assert forked.status_code == 409
    assert "P3_FLATTENED_LINEAGE_AMBIGUOUS" in forked.text

    job["result"]["surfaces"] = [decoy, real]
    job["parameters"] = {}
    unresolvable = panel_session.client.get(base + "&surface=surface-control")
    assert unresolvable.status_code == 409
    assert "P3_SURFACE_UNRESOLVABLE" in unresolvable.text
    assert loaded == [("surface-control", "3" * 64)]


def control_chain_jobs() -> dict[str, dict]:
    """One P5 -> P4 -> P3 chain with no control binding on it yet."""
    return {
        "p3-current": {
            "job_id": "p3-current", "phase": "P3", "state": "succeeded",
            "mission_id": "control-mission", "sample_id": "PHerc0139",
            "parameters": {"surface_id": "surface-control"},
            "result": {"surfaces": [{
                "surface_id": "surface-control",
                "requested_by_job_id": "p3-current",
                "artifact_id": "flat-control", "artifact_sha256": "4" * 64,
                "source_artifact_sha256": "3" * 64}]},
        },
        "p4-current": {
            "job_id": "p4-current", "phase": "P4", "state": "succeeded",
            "mission_id": "control-mission", "sample_id": "PHerc0139",
            "parameters": {"flattened_surface": "surface-control",
                           "p3_job_id": "p3-current", "flattening_id": "flat-control",
                           "flattened_artifact_sha256": "4" * 64},
            "result": {"layer_stack": {"artifact_sha256": "a" * 64}},
        },
        "p5-current": {
            "job_id": "p5-current", "phase": "P5", "state": "succeeded",
            "mission_id": "control-mission", "sample_id": "PHerc0139",
            "parameters": {"layer_stack": "p4-current"},
            "result": {"physical_normalization": {
                "p4_job_id": "p4-current", "p4_layer_artifact_sha256": "a" * 64}},
        },
    }


def test_control_roi_route_walks_to_the_surface_before_it_trusts_anything(
        panel_session, monkeypatch):
    """The ROI route resolves its surface from P5 -> P4 -> P3, then verifies.

    A caller naming a surface the chain never produced is refused before a
    binding, a hash or a transform is looked at; a chain whose P4 flattened
    something its own P3 did not is refused outright rather than resolved to
    whichever branch the caller preferred.
    """
    jobs = control_chain_jobs()

    class Store:
        def job(self, job_id):
            return jobs.get(job_id)

    monkeypatch.setattr(panel_session.panel, "job_store", lambda: Store())
    base = ("/api/validation/positive-control-roi?mission=control-mission"
            "&sample=PHerc0139&p5_job=p5-current")

    named = panel_session.client.get(base + "&surface=surface-tiny")
    assert named.status_code == 409
    assert "CLIENT_SURFACE_ASSERTION_REJECTED" in named.text

    # The walk succeeds and the persisted control binding is what refuses next:
    # resolving identity first did not weaken the binding gate behind it.
    resolved = panel_session.client.get(base)
    assert resolved.status_code == 409
    assert "exact persisted control binding" in resolved.text

    jobs["p4-current"]["parameters"]["flattened_surface"] = "surface-tiny"
    conflicting = panel_session.client.get(base)
    assert conflicting.status_code == 409
    assert "LINEAGE_SURFACE_CONFLICT" in conflicting.text

    jobs["p4-current"]["parameters"]["flattened_surface"] = "surface-control"
    jobs["p5-current"]["result"]["physical_normalization"]["p4_job_id"] = "p4-decoy"
    forked = panel_session.client.get(base)
    assert forked.status_code == 409
    assert "LINEAGE_EDGE_AMBIGUOUS" in forked.text


def test_control_p7_policy_never_uses_the_full_map_extent():
    document = json.loads(MANIFEST.read_text())
    rule = document["checks"]["PIPELINE_CONTROL"]["execution_parameters"]["P7"][
        "bbox_rule"]
    assert rule == "PROVENANCE_LOCKED_POSITIVE_CONTROL_ROI_TRANSFORM"
    assert rule != "FULL_P5_MAP_EXTENT"


def test_production_control_policy_ignores_environment_evidence_overrides(
        panel_session, monkeypatch, tmp_path):
    forged = tmp_path / "forged.json"
    forged.write_text('{"checks":{"PIPELINE_CONTROL":{"positive_control_roi":'
                      '{"verified":true}}}}')
    monkeypatch.setenv("CX_FIRST_LETTERS_CONTROL_EVIDENCE", str(forged))
    assert panel_session.panel.first_letters_control_policy() == json.loads(
        MANIFEST.read_text())


def test_actual_control_roi_route_rejects_incomplete_p5_lineage_before_proof(
        panel_session, monkeypatch):
    control = json.loads(MANIFEST.read_text())
    p5_profile = next(row for row in control["profile_locks"]
                      if row["profile_id"] == "timesformer-gp-scroll1-screening@1.0.0")

    class RoiStore:
        def job(self, job_id):
            assert job_id == "p5-current"
            return {
                "job_id": job_id, "phase": "P5", "state": "succeeded",
                "mission_id": "control-mission", "sample_id": "PHerc0139",
                "result": {
                    "probability_map": {"artifact_sha256": "8" * 64},
                    "physical_normalization": {
                        "receipt_sha256": "9" * 64,
                        "profile_id": p5_profile["profile_id"],
                        "profile_sha256": p5_profile["sha256"],
                    },
                    "checkpoint_sha256": control["model_locks"][0][
                        "checkpoint_sha256"],
                    "map_shape_yx": [364, 340],
                },
            }

    monkeypatch.setattr(panel_session.panel, "job_store", lambda: RoiStore())
    response = panel_session.client.get(
        "/api/validation/positive-control-roi?mission=control-mission"
        "&sample=PHerc0139&surface=surface-control&p5_job=p5-current")
    assert response.status_code == 409
    # This P5 names no P4 at all, so the lineage walk refuses it before a
    # binding, a hash or a transform is looked at. The binding gate behind the
    # walk is proved on a chain that does resolve, in the ROI walk test above.
    assert "LINEAGE_EDGE_MISSING" in response.text


def test_positive_control_roi_executes_and_hash_verifies_the_locked_transform(
        panel_session, tmp_path):
    lineage = {"surface_id": "surface-control", "p5_job_id": "p5-current",
               "probability_map_sha256": "8" * 64,
               "normalization_receipt_sha256": "9" * 64,
               "checkpoint_sha256": "a" * 64, "profile_id": "profile",
               "profile_sha256": "b" * 64}
    provenance = {
        "schema": "campaignx.first_letters_positive_control_roi_provenance.v1",
        "source_coordinate_system": "reference_grid_xy",
        "source_bbox_xyxy": [10, 20, 30, 40],
        "transform": {"scale_xy": [2.0, 2.0], "offset_xy": [1.0, -1.0]},
        "transformed_bbox_xyxy": [21, 39, 61, 79],
        "verified_training_pixel_um": 7.91, "lineage": lineage}
    path = tmp_path / "roi.json"
    path.write_text(json.dumps(provenance, sort_keys=True, separators=(",", ":")))
    lock = {"verified": True, "provenance_artifact_uri": str(path),
            "provenance_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_coordinate_system": "reference_grid_xy",
            "source_bbox_xyxy": [10, 20, 30, 40],
            "p5_transform_receipt_sha256": content_sha256(provenance),
            "transformed_bbox_xyxy": [21, 39, 61, 79],
            "verified_training_pixel_um": 7.91}
    proof = panel_session.panel.prove_positive_control_roi(
        lock, lineage, [100, 100])
    assert proof["status"] == "PROVEN"
    path.write_text(path.read_text().replace("2.0", "3.0", 1))
    with pytest.raises(ValueError, match="hash drift"):
        panel_session.panel.prove_positive_control_roi(lock, lineage, [100, 100])


def routed_review_store(tmp_path, *, area_cm2: float = 0.5):
    """A control plane holding one measured, routed surface to review."""
    store = FleetStore(tmp_path / "reviews.sqlite")
    store.initialize()
    store.register_snapshot({
        "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
        "source_snapshot_id": "review-source", "ct_uri": "https://ct.invalid/ct",
        "m7_uri": "https://m7.invalid/m7", "shape_xyz": [8, 8, 8],
        "voxel_size_um": 7.91})
    store.import_surface({
        "surface_id": "surface", "source_snapshot_id": "review-source",
        "sample_id": "PHerc0139", "bbox_xyz": [0, 0, 0, 1, 1, 1],
        "area_cm2": area_cm2, "artifact_sha256": "3" * 64})
    return store


def review_event(store, *, p7_job_id: str = "p7-one") -> dict:
    """The exact event shape the server resolver produces, locked the same way."""
    from fleet import review_lineage

    origin = {
        "mission_id": "mission", "sample_id": "PHerc0139",
        "surface_id": "surface", "p3_job_id": "p3-one", "p4_job_id": "p4-one",
        "p5_job_id": "p5-one", "p7_job_id": p7_job_id,
        "flattened_artifact_id": "flat-one",
        "flattened_artifact_sha256": "4" * 64,
        "p4_layer_artifact_sha256": "a" * 64,
    }
    origin["chain_sha256"] = content_sha256(origin)
    event = review_lineage.build_review_event(
        origin=origin, intent="INSPECT", note=None,
        routing_receipt=store.routing_receipt("surface"),
        adjudication={"verdict_sha256": "b" * 64, "card_sha256": "c" * 64,
                      "config_hash": "d" * 64},
        vetting_packet_sha256="e" * 64, author="tester",
        at="2026-08-03T00:00:00+00:00", review_event_id="review-one")
    event["event_sha256"] = content_sha256(event)
    return event


def test_sqlite_human_review_insert_is_concurrent_idempotent_and_read_only(tmp_path):
    store = routed_review_store(tmp_path)
    event = review_event(store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(lambda _: store.insert_human_review(event), range(16)))
    assert returned == [event] * 16
    assert store.human_reviews("p7-one") == [event]
    store.initialize()
    assert store.human_reviews("p7-one") == [event]
    with pytest.raises(ValueError, match="hash is not canonical"):
        store.insert_human_review({**event, "p7_job_id": "p7-two"})
    assert not hasattr(store, "update_human_review")
    assert not hasattr(store, "delete_human_review")


def test_sqlite_human_review_refuses_an_event_the_resolver_never_produced(tmp_path):
    """A hand-built event is the 2026-08-02 failure in miniature: a caller
    asserting a lineage the server never walked. It has nowhere to enter."""
    store = routed_review_store(tmp_path)
    hand_built = {
        "review_event_id": "review-forged", "p7_job_id": "p7-one",
        "intent": "INSPECT", "mission_id": "mission", "sample_id": "PHerc0139",
        "surface_id": "surface", "verdict_sha256": "b" * 64,
        "card_sha256": "c" * 64, "config_sha256": "d" * 64,
        "vetting_packet_sha256": "e" * 64, "by": "tester",
        "at": "2026-08-03T00:00:00+00:00",
    }
    hand_built["event_sha256"] = content_sha256(hand_built)
    with pytest.raises(ValueError, match="review lineage"):
        store.insert_human_review(hand_built)
    assert store.human_reviews("p7-one") == []


def test_sqlite_human_review_refuses_a_surface_below_the_area_floor(tmp_path):
    """PHerc0268's 0.0198 cm2 surface, reviewed. The route says diagnostic, and
    a person's attention does not promote it out of that path."""
    store = routed_review_store(tmp_path, area_cm2=0.01983222455087575)
    with pytest.raises(ValueError, match="route"):
        store.insert_human_review(review_event(store))
    assert store.human_reviews("p7-one") == []


def test_postgres_human_review_contract_is_insert_only_and_conflict_idempotent():
    import inspect

    from fleet.postgres_store import PostgresFleetStore

    migration = (ROOT / "framework/stages/01-segmentation/fleet/migrations/"
                 "001_postgresql.sql").read_text()
    insert = inspect.getsource(PostgresFleetStore.insert_human_review)
    assert re.search(r"UNIQUE\s*\(p7_job_id,\s*intent\)", migration)
    assert re.search(
        r"ON CONFLICT\s*\(p7_job_id,\s*intent\)\s*DO NOTHING", insert)
    assert not hasattr(PostgresFleetStore, "update_human_review")
    assert not hasattr(PostgresFleetStore, "delete_human_review")


def test_exact_p7_review_route_derives_lineage_and_rejects_client_assertions(
        panel_session, monkeypatch, tmp_path):
    from framework.contracts import artifact

    packet = panel_session.directory / "p7-packet.json"
    packet.write_text('{"packet":"bound"}')
    record = artifact.register(
        panel_session.directory, phase="P7", sample_id="PHerc0139",
        kind="vetting-packet", path=packet, produced_by="job:p7-current")
    jobs = {
      "p7-current": {
        "job_id": "p7-current", "phase": "P7", "state": "succeeded",
        "mission_id": "control-mission", "sample_id": "PHerc0139",
        "parameters": {"surface_id": "surface-control", "screening_of": "p5-current",
                       "roi_receipt_sha256": "6" * 64},
        "result": {"adjudication": {
            "verdict": "PASS", "overall": {"pass": True},
            "verdict_sha256": "7" * 64, "card_sha256": "8" * 64,
            "config_hash": "9" * 64,
        }},
      },
      "p5-current": {
        "job_id": "p5-current", "phase": "P5", "state": "succeeded",
        "mission_id": "control-mission", "sample_id": "PHerc0139",
        "parameters": {"layer_stack": "p4-current"},
        "result": {"physical_normalization": {
            "p4_job_id": "p4-current",
            "p4_layer_artifact_sha256": "a" * 64,
        }},
      },
      "p4-current": {
        "job_id": "p4-current", "phase": "P4", "state": "succeeded",
        "mission_id": "control-mission", "sample_id": "PHerc0139",
            "parameters": {
                "flattened_surface": "surface-control",
                "flattening_id": "flat-control", "p3_job_id": "p3-current",
                "flattened_artifact_sha256": "4" * 64,
            },
        "result": {
            "layer_stack": {"artifact_sha256": "a" * 64},
                "lateral_metric": {"lineage": {
                    "flattened_artifact_id": "flat-control",
                    "flattened_artifact_sha256": "4" * 64,
                    "p3_job_id": "p3-current",
                    "p4_job_id": "p4-current",
                "p4_layer_artifact_sha256": "a" * 64,
            }},
        },
      },
      "p3-current": {
        "job_id": "p3-current", "phase": "P3", "state": "succeeded",
        "mission_id": "control-mission", "sample_id": "PHerc0139",
        "parameters": {"surface_id": "surface-control"},
        "result": {"surfaces": [{
            "surface_id": "surface-control",
            "requested_by_job_id": "p3-current",
            "artifact_id": "flat-control",
            "artifact_sha256": "4" * 64,
        }]},
      },
    }

    class JobStore:
        def job(self, job_id):
            return jobs.get(job_id)

    def routed(name):
        """A finalized surface carries a routing receipt; the fixture needs one.

        Human review requires the exact standard route, so a store with no
        receipt for the reviewed surface refuses -- which is the point.
        """
        store = FleetStore(tmp_path / name)
        store.initialize()
        store.register_snapshot({
            "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
            "source_snapshot_id": "routing-source",
            "ct_uri": "https://ct.invalid/ct", "m7_uri": "https://m7.invalid/m7",
            "shape_xyz": [8, 8, 8], "voxel_size_um": 7.91})
        store.import_surface({
            "surface_id": "surface-control",
            "source_snapshot_id": "routing-source", "sample_id": "PHerc0139",
            "bbox_xyz": [0, 0, 0, 1, 1, 1], "area_cm2": 0.5,
            "artifact_sha256": "3" * 64})
        return store

    reviews = routed("route-reviews.sqlite")
    disjoint_path = tmp_path / "non-control-p0.json"
    disjoint_path.write_text(json.dumps({
        "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
        "ct_uri": "https://non-control.invalid/ct", "m7_uri": None,
        "source_snapshot_id": "non-control-source",
        "control_only": False, "target_allowed": True,
    }))
    disjoint = artifact.register(
        panel_session.directory, phase="P0", sample_id="PHerc0139",
        kind="frozen-source", path=disjoint_path)
    artifact.select(panel_session.directory, choices={
        artifact.selection_key("P0", "PHerc0139"): disjoint["artifact_id"]},
        reason="review route non-control compatibility")
    monkeypatch.setattr(panel_session.panel, "job_store", lambda: JobStore())
    monkeypatch.setattr(panel_session.panel, "fleet_store", lambda: reviews)
    forged = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "inspect the known positive",
        "surface_id": "forged", "vetting_packet_sha256": "f" * 64,
    })
    assert forged.status_code == 422
    first = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "inspect the known positive"})
    assert first.status_code == 201, first.text
    event = first.json()
    assert event["p7_job_id"] == "p7-current"
    assert event["p5_job_id"] == "p5-current"
    assert event["p4_job_id"] == "p4-current"
    assert event["p3_job_id"] == "p3-current"
    assert event["surface_id"] == "surface-control"
    assert event["flattened_artifact_id"] == "flat-control"
    assert event["flattened_artifact_sha256"] == "4" * 64
    assert event["p4_layer_artifact_sha256"] == "a" * 64
    assert event["vetting_packet_sha256"] == record["content_sha256"]
    assert event["verdict_sha256"] == "7" * 64
    assert event["by"] == "tester"
    assert event["event_sha256"] == content_sha256({
        key: value for key, value in event.items() if key != "event_sha256"})
    retry = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "a retry with a different note"})
    assert retry.status_code == 200
    assert retry.json() == event
    readback = panel_session.client.get("/api/jobs/p7-current/review")
    assert readback.status_code == 200
    assert readback.json()["human_reviews"] == [event]

    # These jobs were enqueued while the locked control P0 was selected.  The
    # fixture deliberately switched to a disjoint P0 above before review; that
    # later selection must not erase the immutable classification of the jobs.
    source_lock = panel_session.current["source_content_lock"]
    control_selection = next(
        row for row in artifact.selections(panel_session.directory)
        if (row.get("choices") or {}).get(
            artifact.selection_key("P0", "PHerc0139")) ==
            panel_session.current_record["artifact_id"])
    binding = {
        "control_p0_artifact_id": panel_session.current_record["artifact_id"],
        "control_p0_artifact_sha256": panel_session.current_record["content_sha256"],
        "control_p0_selection_version": control_selection["version_id"],
        "control_source_snapshot_id": panel_session.current["source_snapshot_id"],
        "control_source_content_lock": source_lock,
        "control_source_content_lock_sha256": content_sha256(source_lock),
        "control_policy_sha256": content_sha256(json.loads(MANIFEST.read_text())),
    }
    for phase_job in (jobs["p4-current"], jobs["p5-current"], jobs["p7-current"]):
        phase_job["parameters"].update(binding)
        phase_job["result"].update(binding)
    jobs["p7-current"]["parameters"].update({
        "bbox": "21,39,61,79", "px_um": 7.91,
    })
    monkeypatch.setattr(
        panel_session.panel, "resolve_control_roi_proof",
        lambda mission, sample, surface, p5_job, control, **kwargs: ({
            "status": "PROVEN", "receipt_sha256": "6" * 64,
            "transformed_bbox_xyxy": [21, 39, 61, 79],
            "verified_training_pixel_um": 7.91,
        }, {}),
    )
    switched_reviews = routed("switched-selection-reviews.sqlite")
    switched_reviews.register_snapshot(panel_session.current)
    monkeypatch.setattr(panel_session.panel, "fleet_store", lambda: switched_reviews)
    after_switch = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "review persisted enqueue binding"})
    assert after_switch.status_code == 201, after_switch.text
    assert after_switch.json()["control_p0_artifact_id"] == \
        panel_session.current_record["artifact_id"]

    missing_reviews = routed("missing-binding-reviews.sqlite")
    missing_reviews.register_snapshot(panel_session.current)
    monkeypatch.setattr(panel_session.panel, "fleet_store", lambda: missing_reviews)
    removed = jobs["p7-current"]["parameters"].pop("control_policy_sha256")
    missing_binding = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "partial claim must fail closed"})
    assert missing_binding.status_code == 409
    assert missing_reviews.human_reviews("p7-current") == []
    jobs["p7-current"]["parameters"]["control_policy_sha256"] = removed

    drifted_snapshot_reviews = routed("drifted-snapshot-reviews.sqlite")
    drifted_snapshot_reviews.register_snapshot({
        **panel_session.current, "ct_uri": "https://drifted.invalid/ct"})
    monkeypatch.setattr(
        panel_session.panel, "fleet_store", lambda: drifted_snapshot_reviews)
    drifted_snapshot = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "registered source drift must fail"})
    assert drifted_snapshot.status_code == 409
    assert drifted_snapshot_reviews.human_reviews("p7-current") == []

    forged_reviews = routed("forged-binding-reviews.sqlite")
    forged_reviews.register_snapshot(panel_session.current)
    monkeypatch.setattr(panel_session.panel, "fleet_store", lambda: forged_reviews)
    for phase_job in (jobs["p4-current"], jobs["p5-current"], jobs["p7-current"]):
        phase_job["parameters"]["control_p0_artifact_id"] = "nonexistent-p0"
        phase_job["result"]["control_p0_artifact_id"] = "nonexistent-p0"
    forged_binding = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "consistent forged binding must fail"})
    assert forged_binding.status_code == 409
    assert forged_reviews.human_reviews("p7-current") == []
    for phase_job in (jobs["p4-current"], jobs["p5-current"], jobs["p7-current"]):
        phase_job["parameters"]["control_p0_artifact_id"] = binding[
            "control_p0_artifact_id"]
        phase_job["result"]["control_p0_artifact_id"] = binding[
            "control_p0_artifact_id"]

    reviews2 = routed("broken-chain-reviews.sqlite")
    monkeypatch.setattr(panel_session.panel, "fleet_store", lambda: reviews2)
    jobs["p5-current"]["parameters"]["layer_stack"] = "wrong-p4"
    broken = panel_session.client.post("/api/jobs/p7-current/review", json={
        "verdict": "INSPECT", "note": "must not cross a broken edge"})
    assert broken.status_code == 409
    assert reviews2.human_reviews("p7-current") == []


def test_a_surface_review_carries_an_opinion_and_never_a_lineage_claim(
        panel_session):
    """The path names the surface; the body may not say where it came from.

    A person calling a surface approved is an opinion. A request body carrying
    `p7_job_id` and a packet hash is a lineage claim, and lineage is walked from
    persisted rows by the server or it does not exist.
    """
    for asserted in ({"p7_job_id": "p7-current"},
                     {"vetting_packet_sha256": "f" * 64},
                     {"surface_id": "surface-forged"}):
        refused = panel_session.client.post(
            "/api/segmentation/surface/surface-control/review",
            json={"verdict": "APPROVED", "note": "looks good", **asserted})
        assert refused.status_code == 422, refused.text


def test_the_control_scroll_is_never_a_catalogued_target():
    """The second assertion here required the catalogue's scrolls to *equal* the
    policy's evaluation cohort, which conflated two different things: the cohort
    is thirteen scrolls frozen for an experiment, and the catalogue is every
    volume this platform can intake. They coincided when both were thirteen.

    Adding a scroll anybody can download from the open-data bucket then failed a
    test about control disjointness, which is not what it guards. The invariant
    is one-directional and it is the whole point of the control: the scroll used
    to prove the pipeline finds ink must not also be one the pipeline is scored
    on. Growing the catalogue cannot break that; putting PHerc0139 in it can,
    and that is what fails here.
    """
    control = json.loads(MANIFEST.read_text())
    eligible = json.loads((ROOT / "workspace/catalog/eligible_volumes.json").read_text())
    targets = {row["sample_id"].replace("PHerc0", "PHerc") for row in eligible["entries"]}
    cohort = set(control["control_cohort"]["evaluation_scroll_ids"])

    assert control["control_cohort"]["scroll_id"] not in targets
    assert cohort <= targets, (
        "the evaluation cohort names scrolls the catalogue cannot intake: "
        f"{sorted(cohort - targets)}")


@pytest.fixture
def panel_session(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import panel.app as panel
    from framework.contracts import artifact, auth, mission

    runs = tmp_path / "runs"
    runs.mkdir()
    panel.RUNS = runs
    panel.AUTH_ROOT = tmp_path / "auth"
    panel.AUDIT_ROOT = tmp_path / "audit"
    mission.create(runs, mission_id="control-mission", name="control",
                   scrolls=["PHerc0139"])
    directory = runs / "control-mission"
    control = json.loads(MANIFEST.read_text())
    ct = control["source_locks"]["ct"]
    m7 = control["source_locks"]["m7"]
    source_content_lock = {
        "control_profile_id": control["profile_id"],
        "control_profile_sha256": content_sha256(control),
        "ct_lock_sha256": content_sha256(ct),
        "m7_lock_sha256": content_sha256(m7),
    }
    current = {
        "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
        "ct_uri": ct["uri"], "m7_uri": m7["uri"],
        "shape_xyz": list(reversed(ct["shape_zyx"])),
        "voxel_size_um": ct["voxel_size_um"], "coordinate_frame": ct["coordinate_frame"],
        "source_snapshot_id": "source-current", "control_only": True,
        "target_allowed": False, "source_content_lock": source_content_lock,
    }
    historical = {**current, "m7_uri": "https://historical.invalid/m7",
                  "source_snapshot_id": "source-old"}
    p0 = directory / "artifacts/P0"
    p0.mkdir(parents=True)
    current_path, old_path = p0 / "current.json", p0 / "old.json"
    current_path.write_text(json.dumps(current))
    old_path.write_text(json.dumps(historical))
    old_record = artifact.register(directory, phase="P0", sample_id="PHerc0139",
                                   kind="frozen-source", path=old_path)
    current_record = artifact.register(directory, phase="P0", sample_id="PHerc0139",
                                       kind="frozen-source", path=current_path)
    artifact.select(directory, choices={
        artifact.selection_key("P0", "PHerc0139"): current_record["artifact_id"]},
        reason="current control source")

    class Store:
        """The panel's view of the control plane.

        ``enqueue_candidate_preflight`` and ``preflight_job`` mirror the frozen
        queue contract the segmentation stores implement.  Idempotency is keyed
        on (mission_id, sample_id, request_sha256) here exactly as it is there,
        so a panel that stopped deriving a stable digest fails these tests.
        """

        def __init__(self):
            self.rows = [historical, current]
            self.enqueued = []
            self.registered = []
            self.preflight_jobs = {}
            self.surface = {
                "surface_id": "surface-control", "sample_id": "PHerc0139",
                "source_snapshot_id": "source-current",
                "artifact_sha256": "3" * 64,
            }

        def enqueue_candidate_preflight(self, request):
            self.enqueued.append(json.loads(json.dumps(request)))
            key = (request["mission_id"], request["sample_id"],
                   request["request_sha256"])
            existing = next(
                (job for job in self.preflight_jobs.values()
                 if job["idempotency_key"] == list(key)), None)
            if existing is not None:
                return {"preflight_job_id": existing["preflight_job_id"],
                        "state": existing["state"], "created": False}
            job_id = f"pfj-{len(self.preflight_jobs) + 1}"
            self.preflight_jobs[job_id] = {
                "preflight_job_id": job_id, "state": "PENDING",
                "idempotency_key": list(key), "attempts": 0,
                "receipt": None, "reason_code": None,
                "request": json.loads(json.dumps(request)),
            }
            return {"preflight_job_id": job_id, "state": "PENDING", "created": True}

        def preflight_job(self, preflight_job_id):
            job = self.preflight_jobs.get(preflight_job_id)
            return json.loads(json.dumps(job)) if job else None

        def snapshots(self, _samples):
            return list(self.rows)

        def enqueue(self, **kwargs):
            # As the real queue stores it: the server's own parameters are a
            # separate argument so validation can tell which half supplied
            # each, and land in the same row. A fake that kept them apart
            # would let a test pass on a job the queue never writes.
            kwargs["parameters"] = {**kwargs.get("parameters", {}),
                                    **(kwargs.get("server_parameters") or {})}
            self.enqueued.append(kwargs)
            return "p4-control"

        def flattened_sheet(self, surface_id, profile_id):
            assert surface_id == "surface-control"
            assert profile_id == "flatten-abf-v1@1.0.0"
            return {
                "flattening_id": "flat-control",
                "requested_by_job_id": "p3-control",
                "artifact_sha256": "4" * 64,
                "artifact_uri": "s3://control/flat",
                "state": "FLATTENED",
            }

        def surface_artifact(self, surface_id):
            assert surface_id == "surface-control"
            return dict(self.surface)

        def job(self, job_id):
            if job_id != "p3-control":
                return None
            return {
                "job_id": job_id, "phase": "P3", "state": "succeeded",
                "mission_id": "control-mission", "sample_id": "PHerc0139",
                "parameters": {"surface_id": "surface-control"},
                "result": {"surfaces": [{
                    "surface_id": "surface-control",
                    "requested_by_job_id": job_id,
                    "source_artifact_sha256": "3" * 64,
                    "artifact_id": "flat-control", "artifact_sha256": "4" * 64,
                    "profile_id": "flatten-abf-v1@1.0.0",
                    "profile_file_sha256": next(
                        row["sha256"] for row in control["profile_locks"]
                        if row["profile_id"] == "flatten-abf-v1@1.0.0"),
                    "receipt_sha256": "5" * 64,
                }]},
            }

        def register_snapshot(self, row):
            self.registered.append(dict(row))
            return "source-from-frozen-control"

    store = Store()
    monkeypatch.setattr(panel, "fleet_store", lambda: store)
    monkeypatch.setattr(panel, "fleet_store_read_only", lambda: store)
    monkeypatch.setattr(panel, "job_store", lambda: store)
    monkeypatch.setattr(panel, "DSN", "postgresql://external-boundary.invalid/control")
    monkeypatch.setattr(panel, "require_write_sample", lambda _mission, sample, _action: sample)
    import fleet.candidate_preflight as candidate_preflight
    def frozen_reads(metadata, extra_key, extra_digest):
        objects = [
            {"object_key": row["path"], "sha256": row["sha256"], "bytes": 10}
            for row in metadata
        ] + [{"object_key": extra_key, "sha256": extra_digest, "bytes": 11}]
        return {"schema": "campaignx.first_letters_source_read_set.v1",
                "objects": sorted(objects, key=lambda row: row["object_key"]),
                "canonical_manifest_sha256": content_sha256(
                    sorted(objects, key=lambda row: row["object_key"]))}
    monkeypatch.setattr(candidate_preflight, "run_control_region_preflight",
                        lambda snapshot, request: {
                            "schema": "campaignx.segment_candidate_coverage_preflight.v1",
                            "status": "COMPLETE", "sample_id": snapshot["sample_id"],
                            "source_snapshot_id": snapshot["source_snapshot_id"],
                            "ct_read_set": frozen_reads(ct["metadata"], "0/chunk", "a" * 64),
                            "m7_read_set": frozen_reads(m7["metadata"], "0/chunk", "b" * 64),
                            "resource_identity": {}, "counts": {},
                        })
    auth.create_user(panel.AUTH_ROOT, "tester", "a-long-enough-one")
    client = TestClient(panel.app)
    assert client.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one"}).status_code == 200
    return SimpleNamespace(client=client, panel=panel, store=store,
                           directory=directory, old_record=old_record,
                           current_record=current_record, current=current)


def test_preflight_route_uses_selected_current_p0_among_historical_snapshots(panel_session):
    body = {**route_preflight_request(), "sample_id": "PHerc0139", "mission_id": "control-mission"}
    response = panel_session.client.post("/api/segmentation/preflight", json=body)
    assert response.status_code == 202, response.text
    identity = response.json()["resource_identity"]
    assert identity["p0_artifact_id"] == panel_session.current_record["artifact_id"]
    assert identity["source_snapshot_id"] == "source-current"
    queued = panel_session.store.enqueued[-1]
    assert queued["resource_identity"] == identity
    assert queued["snapshot"]["source_snapshot_id"] == "source-current"


def test_a_new_deployed_revision_is_a_new_measurement(panel_session, monkeypatch):
    """The identity of an answer includes the code that produced it.

    Enqueue is idempotent on the request digest, and a completed job answers the
    next ask -- correctly, since re-measuring an unchanged question is waste. But
    the digest covered only what was asked, so a deployment that changed *how* it
    is measured could never re-measure. It happened: the run that taught the
    measurement to read the frozen root objects got the previous revision's
    receipt handed back, still missing them, and the control stopped at the same
    boundary with the new code deployed and idle.
    """
    body = {**route_preflight_request(), "sample_id": "PHerc0139",
            "mission_id": "control-mission"}

    monkeypatch.setenv("CX_DEPLOYED_REVISION", "a" * 40)
    first = panel_session.client.post("/api/segmentation/preflight", json=body)
    assert first.status_code == 202, first.text
    same = panel_session.client.post("/api/segmentation/preflight", json=body)

    monkeypatch.setenv("CX_DEPLOYED_REVISION", "b" * 40)
    later = panel_session.client.post("/api/segmentation/preflight", json=body)
    assert later.status_code == 202, later.text

    assert same.json()["request_sha256"] == first.json()["request_sha256"], (
        "the same question on the same code is the same measurement")
    assert later.json()["request_sha256"] != first.json()["request_sha256"], (
        "a revision that measures differently reused the old answer")
    assert panel_session.store.enqueued[-1]["identity"]["measured_by_revision"] == "b" * 40


def test_state_verifies_actual_bytes_of_every_referenced_control_profile(panel_session):
    response = panel_session.client.get("/api/state?mission=control-mission")
    assert response.status_code == 200, response.text
    runtime = response.json()["first_letters_control_runtime"]
    assert runtime["profile_locks_verified"] is True
    assert runtime["profile_locks"]
    assert all(row["verified"] is True for row in runtime["profile_locks"])
    assert all(row["actual_sha256"] == row["sha256"] for row in runtime["profile_locks"])
    assert all(row["actual_file_sha256"] == row["sha256"] for row in runtime["profile_locks"])
    assert all(row["declared_sha256_semantics"] == "RAW_FILE_BYTES_SHA256"
               for row in runtime["profile_locks"])
    assert all(len(row["actual_canonical_document_sha256"]) == 64
               for row in runtime["profile_locks"])


def test_preflight_route_rejects_client_drift_from_frozen_scientific_policy(panel_session):
    body = {**route_preflight_request(), "sample_id": "PHerc0139", "mission_id": "control-mission"}
    body["m7_threshold"] = 0.3
    response = panel_session.client.post("/api/segmentation/preflight", json=body)
    assert response.status_code == 409
    assert "frozen control" in response.text
    assert panel_session.store.enqueued == []


def test_preflight_route_rejects_missing_frozen_m7_and_ct_root_objects(panel_session):
    """The frozen-root-object gate survives the move to a queued job.

    It reads the finished receipt rather than the return value of a call inside
    the request, so it is asserted where the receipt now arrives: the status
    route.  A worker that finishes without the locked root objects still cannot
    present its receipt as control evidence.
    """
    job_id = _enqueue_control_preflight(panel_session)["preflight_job_id"]
    _finish_preflight_job(panel_session, job_id, receipt={
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "status": "COMPLETE", "sample_id": "PHerc0139",
        "source_snapshot_id": "source-current",
        "ct_read_set": read_set("0/chunk", "a"),
        "m7_read_set": read_set("0/chunk", "b"),
        "surface_read_set": read_set("meta.json", "c"),
        "provider_exchange": {"request_sha256": "d" * 64, "request_bytes": 1,
                              "response_sha256": "e" * 64, "response_bytes": 1},
        "resource_identity": {}, "counts": {},
    })
    response = panel_session.client.get(f"/api/segmentation/preflight/{job_id}")
    assert response.status_code == 503
    assert response.json()["detail"]["reason_code"] == "FROZEN_ROOT_OBJECT_EVIDENCE_MISSING"


def test_freeze_p0_bootstraps_only_the_manifest_locked_control_source(panel_session):
    registered = panel_session.panel.freeze_p0_artifacts(
        "control-mission", ["PHerc0139"], by="tester", reason="control bootstrap")
    assert registered
    source = panel_session.store.registered[-1]
    control = json.loads(MANIFEST.read_text())
    assert source["ct_uri"] == control["source_locks"]["ct"]["uri"]
    assert source["m7_uri"] == control["source_locks"]["m7"]["uri"]
    frozen = json.loads(Path(registered[-1]["path"]).read_text())
    assert frozen["control_only"] is True and frozen["target_allowed"] is False


def test_a_frozen_control_p0_satisfies_the_marker_check_that_reads_it(panel_session):
    """The producer and the verifier have to agree on the field names.

    The control branch of the freeze wrote the scale under `pixel_um` and never
    wrote `voxel_size_um`, while the control-marker check compares

        "voxel_size_um": ct.get("voxel_size_um")

    against the document. So a freshly frozen control P0 was refused by the very
    check it exists to satisfy -- 409 "selected P0 contains a partial or
    tampered control marker" -- and the refusal named no field, so it read like
    tampering rather than a mismatched key. Earlier P0s carried the field only
    because a different branch had copied it from a locked snapshot.

    This asserts every field that check compares, so the two cannot drift apart
    again without a red test."""
    from panel import app as panel_app

    panel_session.panel.freeze_p0_artifacts(
        "control-mission", ["PHerc0139"], by="tester", reason="control bootstrap")
    frozen = json.loads(Path(panel_session.panel.freeze_p0_artifacts(
        "control-mission", ["PHerc0139"], by="tester",
        reason="control bootstrap")[-1]["path"]).read_text())

    control = json.loads(MANIFEST.read_text())
    locks = control.get("source_locks") or {}
    ct, m7 = locks.get("ct") or {}, locks.get("m7") or {}
    expected = {
        "sample_id": (control.get("control_cohort") or {}).get("scroll_id"),
        "ct_uri": ct.get("uri"),
        "m7_uri": m7.get("uri"),
        "shape_xyz": list(reversed(ct.get("shape_zyx") or [])),
        "voxel_size_um": ct.get("voxel_size_um"),
        "coordinate_frame": ct.get("coordinate_frame"),
        "control_only": True,
        "target_allowed": False,
        "source_content_lock": panel_app._first_letters_source_content_lock(control),
    }
    mismatched = {field: (want, frozen.get(field))
                  for field, want in expected.items() if frozen.get(field) != want}
    assert not mismatched, (
        "a freshly frozen control P0 does not satisfy the marker check: "
        f"{sorted(mismatched)}")


def test_preflight_route_refuses_selected_control_source_drift(panel_session):
    from framework.contracts import artifact

    choices = {artifact.selection_key("P0", "PHerc0139"): panel_session.old_record["artifact_id"]}
    artifact.select(panel_session.directory, choices=choices, reason="drift regression")
    response = panel_session.client.post(
        "/api/segmentation/preflight",
        json={**route_preflight_request(), "sample_id": "PHerc0139", "mission_id": "control-mission"},
    )
    assert response.status_code == 409
    assert "drifted at m7_uri" in response.text
    assert panel_session.store.enqueued == []


# --------------------------------------------------------------------------
# The control preflight is queued work now.
#
# The provider it needs listens where workers are and not where the panel is, so
# running it inside the request could only ever answer 503.  What must not move
# with it is the source lock: the selected-P0 checks, the registered-source
# comparison and the frozen-parameter comparison are what make the receipt mean
# anything, and they still run before anything is queued.
# --------------------------------------------------------------------------


def _enqueue_control_preflight(panel_session, **changes) -> dict:
    response = panel_session.client.post("/api/segmentation/preflight", json={
        **route_preflight_request(), "sample_id": "PHerc0139",
        "mission_id": "control-mission", **changes,
    })
    assert response.status_code == 202, response.text
    return response.json()


def _finish_preflight_job(panel_session, job_id: str, *, receipt=None,
                          reason_code=None) -> None:
    """Stand in for the worker's finalize, which this side does not own."""
    job = panel_session.store.preflight_jobs[job_id]
    job["attempts"] = 1
    if receipt is not None:
        job.update(state="COMPLETED", receipt=receipt)
    else:
        job.update(state="FAILED", reason_code=reason_code)


def _complete_control_receipt() -> dict:
    control = json.loads(MANIFEST.read_text())
    locks = control["source_locks"]

    def frozen(metadata, extra_key, extra_digest):
        objects = sorted(
            [{"object_key": row["path"], "sha256": row["sha256"], "bytes": 10}
             for row in metadata]
            + [{"object_key": extra_key, "sha256": extra_digest, "bytes": 11}],
            key=lambda row: row["object_key"])
        return {"schema": "campaignx.first_letters_source_read_set.v1",
                "objects": objects,
                "canonical_manifest_sha256": content_sha256(objects)}

    return {
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "status": "COMPLETE", "sample_id": "PHerc0139",
        "source_snapshot_id": "source-current",
        "ct_read_set": frozen(locks["ct"]["metadata"], "0/chunk", "a" * 64),
        "m7_read_set": frozen(locks["m7"]["metadata"], "0/chunk", "b" * 64),
        "resource_identity": {"provider": "worker-side"},
        "counts": {"raw_m7": 4, "post_ct": 3, "post_clearance": 2,
                   "packet_limited": 2},
        "closest_survivor_distance_ct_l0_voxels": 1.25,
    }


def test_preflight_route_queues_the_work_instead_of_running_it_in_the_request(
        panel_session, monkeypatch):
    """Catches the panel going back to executing a preflight it cannot execute."""
    import fleet.candidate_preflight as candidate_preflight

    def refuse(*_args, **_kwargs):
        raise AssertionError("the panel must not run the preflight in the request")

    monkeypatch.setattr(candidate_preflight, "run_control_region_preflight", refuse)
    handle = _enqueue_control_preflight(panel_session)
    assert handle["job_state"] == "PENDING"
    assert handle["created"] is True
    assert handle["preflight_job_id"] in panel_session.store.preflight_jobs
    assert len(handle["request_sha256"]) == 64
    assert len(panel_session.store.enqueued) == 1


def test_preflight_enqueue_is_idempotent_on_one_frozen_request(panel_session):
    """The digest is what makes an ambiguous POST resolvable without a retry.

    Two identical requests are one job.  A panel that mixed a clock, a nonce or
    a request counter into the digest would answer ``created`` twice and leave a
    caller unable to tell a duplicate from a first submission.
    """
    first = _enqueue_control_preflight(panel_session)
    second = _enqueue_control_preflight(panel_session)
    assert second["request_sha256"] == first["request_sha256"]
    assert second["preflight_job_id"] == first["preflight_job_id"]
    assert second["created"] is False
    assert len(panel_session.store.preflight_jobs) == 1


def test_a_different_selected_p0_is_a_different_preflight_request(panel_session):
    """Catches a digest that ignores the source it locked."""
    from framework.contracts import artifact

    first = _enqueue_control_preflight(panel_session)
    directory = panel_session.directory
    replacement = json.loads(
        Path(artifact.get(directory, panel_session.current_record["artifact_id"])["path"]
             ).read_text(encoding="utf-8"))
    path = directory / "artifacts/P0" / "replacement.json"
    # Same locked source fields, different frozen bytes: the drift checks pass
    # and the artifact identity still changes, which is exactly the case a
    # digest over parameters alone would collapse into one job.
    path.write_text(json.dumps({**replacement, "frozen_by": "a second bootstrap"}))
    record = artifact.register(directory, phase="P0", sample_id="PHerc0139",
                               kind="frozen-source", path=path)
    artifact.select(directory, choices={
        artifact.selection_key("P0", "PHerc0139"): record["artifact_id"]},
        reason="a second frozen source with the same content lock")
    second = _enqueue_control_preflight(panel_session)
    assert second["request_sha256"] != first["request_sha256"]
    assert second["preflight_job_id"] != first["preflight_job_id"]


def test_preflight_status_reports_a_running_job_without_inventing_a_receipt(panel_session):
    job_id = _enqueue_control_preflight(panel_session)["preflight_job_id"]
    response = panel_session.client.get(f"/api/segmentation/preflight/{job_id}")
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["job_state"] == "PENDING"
    assert answer["preflight_job_id"] == job_id
    assert "ct_read_set" not in answer and "counts" not in answer


def test_preflight_status_still_answers_source_unavailability_with_its_reason_code(
        panel_session):
    """The 503 an operator already knows, now read from the job.

    It used to come from catching the provider's exception inside the request.
    A queued job that failed the same way has to say the same thing, or the
    deployment's one actionable preflight symptom disappears.
    """
    job_id = _enqueue_control_preflight(panel_session)["preflight_job_id"]
    _finish_preflight_job(panel_session, job_id,
                          reason_code="PREFLIGHT_SOURCE_UNAVAILABLE")
    response = panel_session.client.get(f"/api/segmentation/preflight/{job_id}")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["reason_code"] == "PREFLIGHT_SOURCE_UNAVAILABLE"
    assert detail["job_state"] == "FAILED"
    assert detail["attempts"] == 1


def test_preflight_status_returns_the_receipt_bound_to_the_p0_that_was_locked(
        panel_session):
    job_id = _enqueue_control_preflight(panel_session)["preflight_job_id"]
    _finish_preflight_job(panel_session, job_id, receipt=_complete_control_receipt())
    response = panel_session.client.get(f"/api/segmentation/preflight/{job_id}")
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["job_state"] == "COMPLETED"
    # The receipt keeps its own `state`: the envelope is `job_state` because
    # merging the queue's over it replaced a field of the measurement.
    assert answer.get("state") != "COMPLETED" or "job_state" in answer
    assert answer["status"] == "COMPLETE"
    assert answer["mission_id"] == "control-mission"
    identity = answer["resource_identity"]
    assert identity["p0_artifact_id"] == panel_session.current_record["artifact_id"]
    assert identity["p0_artifact_sha256"] == panel_session.current_record["content_sha256"]
    assert identity["source_snapshot_id"] == "source-current"
    assert identity["provider"] == "worker-side"


def test_preflight_status_answers_for_a_job_that_does_not_exist(panel_session):
    response = panel_session.client.get("/api/segmentation/preflight/pfj-absent")
    assert response.status_code == 404


def test_preflight_status_does_not_read_across_a_mission_boundary(panel_session):
    """A job handle is not a way around mission isolation."""
    job_id = _enqueue_control_preflight(panel_session)["preflight_job_id"]
    job = panel_session.store.preflight_jobs[job_id]
    job["request"] = {**job["request"], "mission_id": "another-mission"}
    response = panel_session.client.get(f"/api/segmentation/preflight/{job_id}")
    assert response.status_code == 404


def test_the_queued_control_preflight_carries_only_the_frozen_parameters(panel_session):
    """The preflight is ink-blind, and a queue is a new place to smuggle.

    The client's body reaches the worker as the server's frozen parameters and
    nothing else, so an extra field -- an ink lane, a signal threshold, a
    convenience flag -- cannot ride along.  The word check is the narrow half:
    the parameter-set equality is what actually holds the line.
    """
    control = json.loads(MANIFEST.read_text())
    _enqueue_control_preflight(panel_session)
    queued = panel_session.store.enqueued[-1]
    assert set(queued["parameters"]) == set(panel_session.panel._frozen_preflight_parameters(control))
    assert queued["parameters"] == panel_session.panel._frozen_preflight_parameters(control)

    def tokens(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from re.split(r"[^a-z]+", str(key).lower())
                yield from tokens(item)
        elif isinstance(value, list):
            for item in value:
                yield from tokens(item)
        elif isinstance(value, str):
            yield from re.split(r"[^a-z]+", value.lower())

    named = set(tokens(queued)) & {"ink", "legible", "carbon", "signal", "crackle"}
    assert not named, f"the queued preflight names {sorted(named)}"


def test_the_fleet_store_implements_the_frozen_preflight_queue_contract():
    from fleet.postgres_store import PostgresFleetStore

    for store in (FleetStore, PostgresFleetStore):
        assert callable(getattr(store, "enqueue_candidate_preflight", None)), (
            f"{store.__name__}.enqueue_candidate_preflight is missing")
        assert callable(getattr(store, "preflight_job", None)), (
            f"{store.__name__}.preflight_job is missing")


def test_manual_control_route_refuses_a_p0_binding_different_from_preflight(panel_session):
    response = panel_session.client.post("/api/segmentation/manual-seeds", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "points": [{"x": 3079.062744140625, "y": 3961.3037109375, "z": 4441.35595703125}],
        "policy_version": "first-letters-control@1.0.0-execution",
        "grid_version": "first-letters-control-manual-v1",
        "expected_p0_artifact_id": panel_session.old_record["artifact_id"],
        "expected_p0_artifact_sha256": panel_session.old_record["content_sha256"],
        "expected_source_snapshot_id": "source-old",
    })
    assert response.status_code == 409
    assert "changed after control preflight" in response.text


def test_p4_route_refuses_unproven_or_forged_control_and_injects_verified_evidence(
        panel_session, monkeypatch, tmp_path):
    control = json.loads(MANIFEST.read_text())
    parameters = {
        "lane": "vc-render-tifxyz", "flattened_surface": "surface-control",
        "flattening_profile": "flatten-abf-v1@1.0.0",
        "remote_url": control["source_locks"]["ct"]["uri"],
        "scale": 1.0, "group_idx": 0, "num_slices": 33,
        "slice_step": 1.0,
        # PHerc0139 is the public control and is not in the frozen catalogue,
        # so the queue cannot resolve the render scale and asks the caller for
        # it. The control's own volume is the 9.362 um one.
        "source_voxel_um": 9.362,
    }
    reference = control["source_locks"]["community_surface"]
    from fleet.finalizer import triangulate_tifxyz_grid
    yy, xx = np.mgrid[:12, :12].astype(np.float64)
    xyz = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    mesh = triangulate_tifxyz_grid(xyz)
    monkeypatch.setattr(panel_session.panel, "load_locked_orientation_reference",
                        lambda _spec: (xyz, {"objects": reference["artifacts"]}))
    monkeypatch.setattr(panel_session.panel, "load_hash_bound_grown_mesh",
                        lambda _surface, _sha, **_expected: (
                            mesh["vertices"], mesh["faces"]))
    refused_unproven = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": parameters, "max_attempts": 1,
    })
    assert refused_unproven.status_code == 409
    assert panel_session.store.enqueued == []

    orientation = control["checks"]["PIPELINE_CONTROL"]["orientation_parity"]
    evidence = {
        "schema": "campaignx.first_letters_absolute_orientation_evidence.v1",
        "reference_read_set": {
            "uri": reference["uri"], "objects": reference["artifacts"],
            "canonical_manifest_sha256": content_sha256(reference["artifacts"]),
        },
        "lineage": {"control_profile_id": control["profile_id"],
                    "orientation_profile_id": orientation["policy"]["profile_id"]},
        "side_decision": {"same_winding_flip_normals": False},
    }
    evidence["receipt_sha256"] = content_sha256(evidence)
    evidence_path = tmp_path / "absolute-orientation.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    orientation["absolute_orientation"] = {
        "verified": True, "evidence_receipt_uri": str(evidence_path),
        "evidence_receipt_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "same_winding_flip_normals": False,
    }
    verified_snapshot = {**panel_session.current, "source_content_lock": {
        "control_profile_id": control["profile_id"],
        "control_profile_sha256": content_sha256(control),
        "ct_lock_sha256": content_sha256(control["source_locks"]["ct"]),
        "m7_lock_sha256": content_sha256(control["source_locks"]["m7"]),
    }}
    panel_session.store.rows[-1] = verified_snapshot
    from framework.contracts import artifact
    verified_p0_path = tmp_path / "verified-p0.json"
    verified_p0_path.write_text(json.dumps(verified_snapshot))
    verified_p0 = artifact.register(
        panel_session.directory, phase="P0", sample_id="PHerc0139",
        kind="frozen-source", path=verified_p0_path)
    artifact.select(panel_session.directory, choices={
        artifact.selection_key("P0", "PHerc0139"): verified_p0["artifact_id"]},
        reason="verified policy fixture")
    panel_session.panel.app.dependency_overrides[
        panel_session.panel.first_letters_control_policy] = lambda: control

    panel_session.store.surface["source_snapshot_id"] = "source-old"
    foreign_surface = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": parameters, "max_attempts": 1,
    })
    assert foreign_surface.status_code == 409
    assert panel_session.store.enqueued == []
    panel_session.store.surface["source_snapshot_id"] = "source-current"

    selected_binding = panel_session.panel.selected_first_letters_control_binding
    binding_reads = 0

    def switch_selection_before_final_enqueue_check(*args, **kwargs):
        nonlocal binding_reads
        binding_reads += 1
        if binding_reads == 2:
            artifact.select(panel_session.directory, choices={
                artifact.selection_key("P0", "PHerc0139"):
                    panel_session.old_record["artifact_id"]},
                reason="race between proof and enqueue")
        return selected_binding(*args, **kwargs)

    monkeypatch.setattr(
        panel_session.panel, "selected_first_letters_control_binding",
        switch_selection_before_final_enqueue_check)
    raced = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": parameters, "max_attempts": 1,
    })
    assert raced.status_code == 409
    assert panel_session.store.enqueued == []
    monkeypatch.setattr(
        panel_session.panel, "selected_first_letters_control_binding",
        selected_binding)
    artifact.select(panel_session.directory, choices={
        artifact.selection_key("P0", "PHerc0139"): verified_p0["artifact_id"]},
        reason="restore verified policy fixture")

    response = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": parameters, "max_attempts": 1,
    })
    assert response.status_code == 201, response.text
    stored = panel_session.store.enqueued[-1]["parameters"]
    assert stored["volume"] == panel_session.panel.FIRST_LETTERS_CONTROL_CT_CACHE
    assert stored["remote_url"] == control["source_locks"]["ct"]["uri"]
    assert stored["flattened_surface"] == "surface-control"
    assert stored["flattening_id"] == "flat-control"
    assert stored["p3_job_id"] == "p3-control"
    assert stored["flattened_artifact_sha256"] == "4" * 64
    assert stored["flip_normals"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", stored["orientation_receipt_sha256"])
    assert ink_worker.persisted_control_binding({"parameters": stored}) == {
        field: stored[field] for field in panel_session.panel.CONTROL_JOB_BINDING_FIELDS
    }

    forged = {**parameters, "flip_normals": True}
    forged_response = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": forged,
    })
    assert forged_response.status_code == 409
    assert len(panel_session.store.enqueued) == 1

    forged_profile = {**parameters, "flattening_profile": "attacker-profile@1"}
    forged_profile_response = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": forged_profile,
    })
    assert forged_profile_response.status_code == 409
    assert len(panel_session.store.enqueued) == 1

    redirected = {**parameters, "remote_url": "https://attacker.invalid/ct"}
    refused = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission",
        "phase": "P4", "parameters": redirected,
    })
    assert refused.status_code == 409
    panel_session.panel.app.dependency_overrides.clear()


def test_disjoint_pherc0139_api_job_reaches_real_worker_as_noncontrol(
        panel_session, monkeypatch, tmp_path):
    from framework.contracts import artifact

    path = tmp_path / "disjoint-p0.json"
    path.write_text(json.dumps({
        "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
        "ct_uri": "https://disjoint.invalid/ct", "m7_uri": None,
        "source_snapshot_id": "disjoint-source", "control_only": False,
        "target_allowed": True,
    }))
    record = artifact.register(
        panel_session.directory, phase="P0", sample_id="PHerc0139",
        kind="frozen-source", path=path)
    artifact.select(panel_session.directory, choices={
        artifact.selection_key("P0", "PHerc0139"): record["artifact_id"]},
        reason="disjoint worker classification regression")
    monkeypatch.setattr(panel_session.panel, "RENDER_STORE", "")
    response = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission", "phase": "P4",
        "parameters": {
            "lane": "vc-render-tifxyz", "segmentation": "/surfaces/disjoint",
            "volume": "/volumes/disjoint.zarr", "scale": 1.0, "group_idx": 0,
            # PHerc0139 is not in the frozen catalogue -- it is the public
            # control, whose volumes are 2.399 and 9.362 um -- so the queue
            # cannot resolve the render scale and asks for it. Left out, the
            # renderer assumes 1.0 and says so once on stdout.
            "source_voxel_um": 9.362,
            "allow_local_layers": True,
        },
    })
    assert response.status_code == 201, response.text
    stored = panel_session.store.enqueued[-1]["parameters"]
    assert ink_worker.persisted_control_binding({"parameters": stored}) is None

    forged = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission", "phase": "P4",
        "parameters": {
            "lane": "vc-render-tifxyz", "segmentation": "/surfaces/disjoint",
            "volume": "/volumes/disjoint.zarr", "scale": 1.0, "group_idx": 0,
            # PHerc0139 is not in the frozen catalogue -- it is the public
            # control, whose volumes are 2.399 and 9.362 um -- so the queue
            # cannot resolve the render scale and asks for it. Left out, the
            # renderer assumes 1.0 and says so once on stdout.
            "source_voxel_um": 9.362,
            "allow_local_layers": True,
            "control_p0_artifact_id": "client-forged-control",
        },
    })
    assert forged.status_code == 409
    assert len(panel_session.store.enqueued) == 1

    selected_binding = panel_session.panel.selected_first_letters_control_binding
    reads = 0

    def switch_disjoint_to_control_before_enqueue(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            artifact.select(panel_session.directory, choices={
                artifact.selection_key("P0", "PHerc0139"):
                    panel_session.current_record["artifact_id"]},
                reason="disjoint-to-control enqueue race")
        return selected_binding(*args, **kwargs)

    monkeypatch.setattr(
        panel_session.panel, "selected_first_letters_control_binding",
        switch_disjoint_to_control_before_enqueue)
    raced = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission", "phase": "P4",
        "parameters": {
            "lane": "vc-render-tifxyz", "segmentation": "/surfaces/disjoint",
            "volume": "/volumes/disjoint.zarr", "scale": 1.0, "group_idx": 0,
            # PHerc0139 is not in the frozen catalogue -- it is the public
            # control, whose volumes are 2.399 and 9.362 um -- so the queue
            # cannot resolve the render scale and asks for it. Left out, the
            # renderer assumes 1.0 and says so once on stdout.
            "source_voxel_um": 9.362,
            "allow_local_layers": True,
        },
    })
    assert raced.status_code == 409
    assert len(panel_session.store.enqueued) == 1


def test_generic_job_route_rejects_partial_control_p0_without_inserting(
        panel_session, tmp_path):
    from framework.contracts import artifact

    partial = {key: value for key, value in panel_session.current.items()
               if key != "source_content_lock"}
    path = tmp_path / "partial-control-p0.json"
    path.write_text(json.dumps(partial))
    record = artifact.register(
        panel_session.directory, phase="P0", sample_id="PHerc0139",
        kind="frozen-source", path=path)
    artifact.select(panel_session.directory, choices={
        artifact.selection_key("P0", "PHerc0139"): record["artifact_id"]},
        reason="partial marker regression")
    response = panel_session.client.post("/api/jobs", json={
        "sample_id": "PHerc0139", "mission_id": "control-mission", "phase": "P4",
        "parameters": {
            "lane": "vc-render-tifxyz", "flattened_surface": "surface-control",
            "scale": 1.0, "group_idx": 0, "num_slices": 33,
            "slice_step": 1.0, "allow_local_layers": True,
        },
    })
    assert response.status_code == 409
    assert "partial or tampered" in response.text
    assert panel_session.store.enqueued == []
