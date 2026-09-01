"""Candidate availability is measured without creating segmentation work."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from fleet.candidate_preflight import (  # noqa: E402
    aggregate_candidate_survival,
    persist_candidate_preflight_receipt_pair,
    SCIENTIFIC_DUPLICATE_FIELDS,
    run_candidate_coverage_preflight,
    sanitize_candidate_coverage_receipt,
    survey_candidate_task,
    validate_candidate_preflight_receipt_pair,
)
from fleet.common import content_sha256  # noqa: E402
from fleet.generator import bounded_preflight_grid_design, generate_tasks_for_snapshot  # noqa: E402
from fleet.cli import build_parser  # noqa: E402
from fleet.store import FleetStore  # noqa: E402
from fleet.store_factory import open_fleet_store_read_only  # noqa: E402


LOCK = {
    "schema": "campaignx.source_content_lock.v1",
    "status": "VERIFIED_IMMUTABLE",
    "verification_method": "fixture-sha256-v1",
    "verified_at_utc": "2026-08-02T00:00:00Z",
    "ct_uri": "fixture://ct?versionId=ct-version-0001",
    "ct_sha256": "a" * 64,
    "ct_version_id": "ct-version-0001",
    "m7_uri": "fixture://m7?versionId=m7-version-0001",
    "m7_sha256": "b" * 64,
    "m7_version_id": "m7-version-0001",
}


def snapshot(*, shape=(33, 33, 33)):
    return {
        "source_snapshot_id": "source-locked-1",
        "sample_id": "PHerc0358",
        "ct_uri": LOCK["ct_uri"],
        "ct_sha256": LOCK["ct_sha256"],
        "m7_uri": LOCK["m7_uri"],
        "m7_sha256": LOCK["m7_sha256"],
        "m7_threshold": 0.2,
        "shape_xyz": list(shape),
        "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
        "source_content_lock": copy.deepcopy(LOCK),
    }


def request(**overrides):
    value = {
        "mission_id": "first-letters",
        "provider": "vc3d-mcp",
        "catalog_snapshot_sha256": "c" * 64,
        "grid_version": "first-letters-grid@1.0.0",
        "policy_version": "first-letters-preflight@1.0.0",
        "grid_step": 16,
        "query_radius": 8,
        "cell_clearance": 0,
        "volume_clearance": 8,
        "candidate_interior_clearance": 2,
        "selection_strategy": "stratified-clearance-v1",
        "candidate_selection_policy": "score-cell-volume-clearance-v1",
        "seed_region_policy": "fixed-v1",
        "packet_candidate_limit": 2,
        "maximum_cells": 100,
        "parallelism": 2,
        "ct_material_support_gate": {
            "policy": "ome-zarr-nearby-material-v1",
            "level": 1,
            "radius_l0_voxels": 4,
            "minimum_nonzero_voxels": 1,
        },
        "p0_artifact_id": "p0-1",
        "p0_artifact_sha256": "d" * 64,
        "p0_selection_version": "selection-1",
        "p0_selection_sha256": "e" * 64,
        "source_content_lock_sha256": content_sha256(LOCK),
    }
    value.update(overrides)
    return value


class ReadOnlySurfaces:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.reads = 0
        self.mutations = 0

    def surfaces_for_snapshot(self, source_snapshot_id):
        self.reads += 1
        assert source_snapshot_id == "source-locked-1"
        return copy.deepcopy(self.rows)

    def __getattr__(self, name):
        if name.startswith(("insert", "create", "claim", "lease", "complete")):
            self.mutations += 1
            raise AssertionError(f"preflight attempted fleet mutation: {name}")
        raise AttributeError(name)


class Provider:
    def discover(self, task):
        center = task["center_xyz"]
        coordinate = {axis: int(center[axis]) for axis in "xyz"}
        return {
            "candidates": [{
                "candidate_id": f"m7-{center['x']}-{center['y']}-{center['z']}",
                "ct_l0_coordinate": coordinate,
                "score": 0.75,
            }],
            "provider_exchange": {
                "request_sha256": "e" * 64, "request_bytes": 12,
                "response_sha256": content_sha256(coordinate), "response_bytes": 24,
            },
            "source_read_set": {
                "schema": "campaignx.first_letters_source_read_set.v1",
                "objects": [{"object_key": "m7/chunk", "sha256": "f" * 64, "bytes": 24}],
                "canonical_manifest_sha256": content_sha256([
                    {"object_key": "m7/chunk", "sha256": "f" * 64, "bytes": 24}
                ]),
            },
        }


class Sampler:
    def sample(self, _ct_uri, coordinate_xyz, *, level, radius_l0_voxels):
        assert level == 1 and radius_l0_voxels == 4
        row = {"object_key": f"ct/{coordinate_xyz['x']}", "sha256": "1" * 64, "bytes": 8}
        return {
            "nonzero_voxel_count": 2,
            "source_read_set": {
                "schema": "campaignx.first_letters_source_read_set.v1",
                "objects": [row],
                "canonical_manifest_sha256": content_sha256([row]),
            },
        }


def test_shared_survey_primitive_runs_provider_ct_clearance_and_packet_limit():
    task = {
        "task_id": "t1", "sample_id": "PHerc0358", "cell_id": "c1", "priority": 1,
        "source_snapshot_id": "source-locked-1", "source": snapshot(),
        "bounds_xyz": [[0, 0, 0], [32, 32, 32]],
        "center_xyz": {"x": 16, "y": 16, "z": 16},
        "candidate_discovery": {
            "ct_material_support_gate": request()["ct_material_support_gate"],
            "minimum_cell_interior_clearance_voxels": 2,
            "minimum_volume_interior_clearance_voxels": 8,
        },
        "parameter_envelope": {"maximum_candidate_count": 1},
    }
    row = survey_candidate_task(task, Provider(), ct_sampler=Sampler())
    assert row["counts"] == {"raw_m7": 1, "post_ct": 1,
                             "post_cell_clearance": 1,
                             "post_volume_clearance": 1, "packet_retained": 1}
    assert row["source_error"] is None
    assert row["ink_used"] is False


def test_clearance_survival_is_measured_before_packet_truncation():
    task = {
        "task_id": "t-many", "sample_id": "PHerc0358", "cell_id": "c-many", "priority": 1,
        "source_snapshot_id": "source-locked-1", "source": snapshot(),
        "bounds_xyz": [[0, 0, 0], [32, 32, 32]],
        "center_xyz": {"x": 16, "y": 16, "z": 16},
        "candidate_discovery": {
            "ct_material_support_gate": request()["ct_material_support_gate"],
            "minimum_cell_interior_clearance_voxels": 2,
            "minimum_volume_interior_clearance_voxels": 8,
        },
        "parameter_envelope": {"maximum_candidate_count": 1},
    }

    class ManyProvider(Provider):
        def discover(self, _task):
            return {"candidates": [
                {"candidate_id": f"c{i}", "ct_l0_coordinate": {"x": 12 + i, "y": 16, "z": 16},
                 "score": 0.9 - i / 10} for i in range(3)
            ]}

    row = survey_candidate_task(task, ManyProvider(), ct_sampler=Sampler())
    assert row["counts"] == {"raw_m7": 3, "post_ct": 3, "post_cell_clearance": 3,
                             "post_volume_clearance": 3, "packet_retained": 1}
    assert len(row["volume_clearance_candidates"]) == 3
    assert len(row["packet_candidates"]) == 1


def test_preflight_is_a_census_when_every_geometrically_eligible_cell_is_surveyed():
    store = ReadOnlySurfaces()
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=store, provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40,
    )
    private, public = result["private_receipt"], result["sanitized_receipt"]
    assert private["schema"] == "campaignx.segment_candidate_coverage_preflight.v1"
    assert private["measurement_kind"] == "CENSUS"
    assert private["funnel"]["total_grid_cells"] == 8
    assert private["funnel"]["geometrically_eligible_cells"] == 8
    assert private["funnel"]["cells_surveyed"] == 8
    assert private["funnel"]["packet_retained_candidates"] == 8
    assert private["bindings"]["source_content_lock_sha256"] == content_sha256(LOCK)
    assert private["bindings"]["code_revision"] == "1" * 40
    assert private["receipt_sha256"] == content_sha256(private["scientific_core"])
    assert public["private_receipt_sha256"] == private["receipt_sha256"]
    assert "candidate_coordinates_xyz" not in public
    assert store.reads == 1 and store.mutations == 0


def test_generated_receipts_hash_but_never_expose_credentialed_m7_uri():
    secret_uri = (
        "https://reader:secret@example.invalid/private-m7.zarr"
        "?versionId=m7-version-0001"
    )
    locked_snapshot = snapshot()
    locked_snapshot["m7_uri"] = secret_uri
    locked_snapshot["source_content_lock"]["m7_uri"] = secret_uri
    locked_snapshot["source_content_lock"]["verification_method"] = (
        "immutable-uri-manifest-sha256-v1"
    )
    result = run_candidate_coverage_preflight(
        locked_snapshot,
        request(source_content_lock_sha256=content_sha256(
            locked_snapshot["source_content_lock"])),
        surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40,
    )
    expected_digest = hashlib.sha256(secret_uri.encode("utf-8")).hexdigest()
    for receipt in (result["private_receipt"], result["sanitized_receipt"]):
        assert secret_uri not in json.dumps(receipt)
        assert receipt["bindings"]["m7_uri_sha256"] == expected_digest


def test_bounded_preflight_is_labeled_estimated_and_reports_sampling_percentage():
    result = run_candidate_coverage_preflight(
        snapshot(shape=(65, 65, 65)), request(maximum_cells=5),
        surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40,
    )["private_receipt"]
    assert result["measurement_kind"] == "ESTIMATE"
    assert result["sampling_design"]["name"] == "deterministic-golden-coprime-rank1-grid-sample-v1"
    assert "gcd(stride,N)=1" in result["sampling_design"]["ordinal_rule"]
    assert result["funnel"]["geometrically_eligible_cells"] is None
    assert result["funnel"]["geometrically_eligible_cells_estimate"] > 5
    assert result["funnel"]["cells_surveyed"] == 5
    assert result["planned_sampling_percentage"] == pytest.approx(
        100 * 5 / result["funnel"]["total_grid_cells"]
    )


def test_generator_reports_exact_population_at_the_cap_boundary():
    for cap, expected_selected in ((8, 8), (7, 7)):
        population = {}
        tasks = generate_tasks_for_snapshot(
            ReadOnlySurfaces(), snapshot(), catalog_snapshot_sha256="c" * 64,
            grid_step=16, query_radius=8, clearance=0, volume_edge_margin=8,
            candidate_interior_clearance=2,
            selection_strategy="stratified-clearance-v1", max_tasks=cap,
            grid_version="grid@1.0.0", policy_version="policy@1.0.0",
            population_count_out=population,
        )
        assert len(tasks) == expected_selected
        assert population == {
            "total_grid_cells": 8, "geometrically_eligible_cells": 8,
        }


def test_anisotropic_complete_enumeration_does_not_collapse_stratified_buckets():
    result = run_candidate_coverage_preflight(
        snapshot(shape=(33, 33, 321)), request(maximum_cells=100),
        surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40,
    )["private_receipt"]
    assert result["funnel"]["geometrically_eligible_cells"] == 80
    assert result["funnel"]["cells_attempted"] == 80


def test_bounded_anisotropic_design_cycles_thin_axes_and_spans_long_axis():
    options = dict(shape_xyz=[2001, 5, 5], query_radius=1, grid_step=2,
                   volume_edge_margin=1, hard_cell_limit=16)
    first = bounded_preflight_grid_design(**options)
    second = bounded_preflight_grid_design(**options)
    assert first == second
    assert first["total_grid_cells"] == 1000 * 2 * 2
    assert len(first["indices"]) == len(set(first["indices"])) == 16
    assert {(y, z) for _, y, z in first["indices"]} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}
    long_axis = [x for x, _, _ in first["indices"]]
    assert min(long_axis) < 100 and max(long_axis) >= 900


@pytest.mark.parametrize("cap", [500, 250])
def test_bounded_rank1_prefix_covers_every_long_axis_quartile(cap):
    result = bounded_preflight_grid_design(
        [16001, 17, 17], query_radius=8, grid_step=16,
        volume_edge_margin=8, hard_cell_limit=cap)
    assert result["total_grid_cells"] == 1000
    indices = result["indices"]
    assert len(indices) == len(set(indices)) == cap
    quartiles = [0, 0, 0, 0]
    for x, y, z in indices:
        assert y == z == 0
        quartiles[min(3, x * 4 // 1000)] += 1
    assert all(count > 0 for count in quartiles)
    assert max(quartiles) - min(quartiles) <= max(8, cap // 20)


def test_zero_eligible_cells_still_emit_a_complete_bound_receipt():
    result = run_candidate_coverage_preflight(
        snapshot(shape=(10, 10, 10)), request(maximum_cells=8),
        surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40,
    )["private_receipt"]
    assert result["funnel"]["total_grid_cells"] == 0
    assert result["funnel"]["geometrically_eligible_cells"] == 0
    assert result["bindings"]["candidate_selection_policy"] == "score-cell-volume-clearance-v1"
    assert result["bindings"]["seed_region_policy"] == "fixed-v1"
    assert result["receipt_sha256"] == content_sha256(result["scientific_core"])


def test_overlapping_cell_deduplication_retains_candidate_if_any_observation_survives():
    coordinate = {"x": 16, "y": 16, "z": 16}
    rows = [
        {"task_id": "a", "raw_candidates": [{**coordinate, "score": 0.9}],
         "ct_candidates": [{**coordinate, "score": 0.9}], "clearance_candidates": [],
         "cell_clearance_candidates": [], "volume_clearance_candidates": [],
         "packet_candidates": [], "source_error": None, "rejection_counts": {"INSUFFICIENT_CELL_INTERIOR_CLEARANCE": 1}},
        {"task_id": "b", "raw_candidates": [{**coordinate, "score": 0.9}],
         "ct_candidates": [{**coordinate, "score": 0.9}],
         "clearance_candidates": [{**coordinate, "score": 0.9, "cell_interior_clearance_voxels": 8,
                                    "volume_interior_clearance_voxels": 16}],
         "cell_clearance_candidates": [{**coordinate, "score": 0.9, "cell_interior_clearance_voxels": 8,
                                         "volume_interior_clearance_voxels": 16}],
         "volume_clearance_candidates": [{**coordinate, "score": 0.9, "cell_interior_clearance_voxels": 8,
                                           "volume_interior_clearance_voxels": 16}],
         "packet_candidates": [{**coordinate, "score": 0.9, "cell_interior_clearance_voxels": 8,
                                 "volume_interior_clearance_voxels": 16}],
         "source_error": None, "rejection_counts": {}},
    ]
    aggregate = aggregate_candidate_survival(rows)
    assert aggregate["unique_raw"] == 1
    assert aggregate["unique_post_clearance"] == 1
    assert aggregate["unique_packet_retained"] == 1
    assert aggregate["duplicate_observations"] == 1


def test_sanitized_receipt_keeps_only_bins_counts_and_non_claim():
    private = {
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "receipt_sha256": "a" * 64,
        "candidate_coordinates_xyz": [{"x": 1, "y": 2, "z": 3}],
        "cells": [{"center_xyz": {"x": 1, "y": 2, "z": 3}}],
        "spatial_bins": [{"bin_xyz": [0, 0, 0], "usable_candidate_cells": 1}],
        "funnel": {"raw_m7_candidates": 1},
        "non_claim": "Candidate scarcity is not surface or ink absence.",
    }
    public = sanitize_candidate_coverage_receipt(private)
    assert "candidate_coordinates_xyz" not in public and "cells" not in public
    assert public["spatial_bins"] == private["spatial_bins"]
    assert public["non_claim"] == private["non_claim"]


def test_private_receipt_binds_catalog_and_selected_p0_version_hash():
    private = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)["private_receipt"]
    assert private["bindings"]["catalog_snapshot_sha256"] == "c" * 64
    assert private["bindings"]["p0_selection_sha256"] == "e" * 64


def test_receipt_pair_rerun_reuses_first_timestamp_and_exact_bytes(tmp_path):
    first = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private_path = tmp_path / "private.json"
    public_path = tmp_path / "public.json"
    persisted = persist_candidate_preflight_receipt_pair(
        private_path, public_path, first["private_receipt"], first["sanitized_receipt"])
    original = (private_path.read_bytes(), public_path.read_bytes())
    rerun_private = copy.deepcopy(first["private_receipt"])
    rerun_private["generated_at_utc"] = "2099-01-01T00:00:00Z"
    rerun_public = sanitize_candidate_coverage_receipt(rerun_private)
    rerun = persist_candidate_preflight_receipt_pair(
        private_path, public_path, rerun_private, rerun_public)
    assert (private_path.read_bytes(), public_path.read_bytes()) == original
    assert rerun == persisted
    assert rerun["private_receipt"]["generated_at_utc"] != "2099-01-01T00:00:00Z"


def test_receipt_pair_rolls_back_first_half_if_second_publish_fails(
    tmp_path, monkeypatch
):
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private_path = tmp_path / "private.json"
    public_path = tmp_path / "public.json"
    import fleet.candidate_preflight as module
    original_link = module.os.link
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated public publish failure")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", fail_second)
    with pytest.raises(OSError, match="simulated public"):
        persist_candidate_preflight_receipt_pair(
            private_path, public_path, result["private_receipt"],
            result["sanitized_receipt"])
    assert not private_path.exists() and not public_path.exists()


def test_receipt_pair_repairs_a_valid_private_only_interrupted_pair(tmp_path):
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private_path = tmp_path / "private.json"
    public_path = tmp_path / "public.json"
    first = persist_candidate_preflight_receipt_pair(
        private_path, public_path, result["private_receipt"],
        result["sanitized_receipt"])
    expected_public = public_path.read_bytes()
    public_path.unlink()
    repaired = persist_candidate_preflight_receipt_pair(
        private_path, public_path, result["private_receipt"],
        result["sanitized_receipt"])
    assert public_path.read_bytes() == expected_public
    assert repaired == first


@pytest.mark.parametrize("field,mutate", [
    ("planned_sampling_percentage", lambda _value: True),
    ("planned_sampling_percentage", lambda _value: float("nan")),
    ("achieved_successful_sampling_percentage", lambda _value: float("inf")),
    ("spatial_bins", lambda _value: [{}]),
    ("spatial_bins", lambda value: [{**value[0], "bin_xyz": [0, True, 2]}]),
    ("funnel", lambda value: {**value, "cells_attempted": True}),
    ("funnel", lambda value: {**value,
                               "post_ct_candidates": value["raw_m7_candidates"] + 1}),
    ("funnel", lambda value: {**value, "packet_retained_candidates":
                               value["post_volume_clearance_candidates"] + 1}),
    ("funnel", lambda value: {**value, "duplicate_candidate_observations":
                               value["duplicate_candidate_observations"] + 1}),
    ("candidate_coordinates_xyz", lambda value: value[:-1]),
    ("candidate_coordinates_xyz", lambda value: [value[0] for _row in value]),
    ("sampling_design", lambda value: {**value, "inclusion_fraction": "25%"}),
    ("no_candidate_causes", lambda _value: {"SOURCE_ERROR": True}),
    ("score_statistics", lambda value: {**value, "median": float("inf")}),
    ("bindings", lambda value: {**value, "maximum_cells": True}),
])
def test_pair_validator_rejects_browser_crashing_signed_core_shapes(field, mutate):
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    core = private["scientific_core"]
    core[field] = mutate(core[field])
    private[field] = copy.deepcopy(core[field])
    private["receipt_sha256"] = content_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


def test_pair_validator_rejects_estimate_relabelled_as_exact_census():
    result = run_candidate_coverage_preflight(
        snapshot(shape=(65, 65, 65)), request(maximum_cells=5),
        surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    private["scientific_core"]["measurement_kind"] = "CENSUS"
    private["measurement_kind"] = "CENSUS"
    private["receipt_sha256"] = content_sha256(private["scientific_core"])
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


def test_pair_validator_rejects_falsified_eligible_population_estimate():
    result = run_candidate_coverage_preflight(
        snapshot(shape=(65, 65, 65)), request(maximum_cells=5),
        surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
        code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    private["scientific_core"]["funnel"]["geometrically_eligible_cells_estimate"] = 0
    private["funnel"]["geometrically_eligible_cells_estimate"] = 0
    private["receipt_sha256"] = content_sha256(private["scientific_core"])
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


def test_pair_validator_rejects_source_failures_relabelled_complete_census():
    class HalfFailingProvider(Provider):
        def discover(self, task):
            if task["center_xyz"]["x"] == 8:
                raise RuntimeError("source offline")
            return super().discover(task)

    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(),
        provider=HalfFailingProvider(), ct_sampler=Sampler(), code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    core = private["scientific_core"]
    core["status"] = "COMPLETE"
    core["measurement_kind"] = "CENSUS"
    private["status"] = "COMPLETE"
    private["measurement_kind"] = "CENSUS"
    private["receipt_sha256"] = content_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


def test_pair_validator_rejects_zero_packet_with_usable_spatial_cells():
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    core = private["scientific_core"]
    funnel = core["funnel"]
    for key in ("raw_m7_candidates", "post_ct_candidates",
                "post_cell_clearance_candidates", "post_volume_clearance_candidates",
                "packet_retained_candidates", "raw_m7_candidate_observations",
                "duplicate_candidate_observations"):
        funnel[key] = 0
    core["candidate_coordinates_xyz"] = []
    private["funnel"] = copy.deepcopy(funnel)
    private["candidate_coordinates_xyz"] = []
    private["receipt_sha256"] = content_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


@pytest.mark.parametrize("mutate", [
    lambda core: core["normalized_request"].update({"parallelism": 7}),
    lambda core: core.update({"sample_id": "PHerc9999"}),
    lambda core: core.update({"state_mutation": "WRITE"}),
    lambda core: core["candidate_coordinates_xyz"].reverse(),
    lambda core: core["candidate_coordinates_xyz"][0].update({"x": 999999}),
    lambda core: core["score_statistics"].update({"minimum": 2.0, "p95": 1.0}),
    lambda core: core["m7_read_set_manifest_sha256"].append(
        core["m7_read_set_manifest_sha256"][0]),
    lambda core: core["spatial_bins"].reverse(),
    lambda core: core["cell_outcomes"][0]["counts"].update({
        "post_ct": core["cell_outcomes"][0]["counts"]["raw_m7"] + 1}),
    lambda core: core["sampling_design"].update({"name": "pretend-census"}),
])
def test_pair_validator_rejects_signed_core_derivation_drift(mutate):
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    core = private["scientific_core"]
    mutate(core)
    for field in SCIENTIFIC_DUPLICATE_FIELDS:
        private[field] = copy.deepcopy(core[field])
    private["receipt_sha256"] = content_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


@pytest.mark.parametrize("mutate", [
    lambda core: core["normalized_request"].update({"grid_step": 32}),
    lambda core: core["normalized_request"].update({"parallelism": "eight"}),
])
def test_pair_validator_rejects_rehashed_request_that_does_not_bind_execution(mutate):
    result = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=Provider(),
        ct_sampler=Sampler(), code_revision="1" * 40)
    private = copy.deepcopy(result["private_receipt"])
    core = private["scientific_core"]
    mutate(core)
    core["bindings"]["normalized_request_sha256"] = content_sha256(
        core["normalized_request"])
    for field in SCIENTIFIC_DUPLICATE_FIELDS:
        private[field] = copy.deepcopy(core[field])
    private["receipt_sha256"] = content_sha256(core)
    public = sanitize_candidate_coverage_receipt(private)
    with pytest.raises(ValueError):
        validate_candidate_preflight_receipt_pair(private, public)


def test_verified_sqlite_preflight_view_reads_without_mutation(tmp_path):
    path = tmp_path / "fleet.sqlite"
    writable = FleetStore(path)
    writable.initialize()
    writable.register_snapshot(snapshot())
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = path.read_bytes()
    view = open_fleet_store_read_only(path)
    assert view.snapshots({"PHerc0358"})[0]["source_snapshot_id"] == "source-locked-1"
    assert view.surfaces_for_snapshot("source-locked-1") == []
    assert path.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError, match="readonly database"):
        view.register_snapshot({**snapshot(), "source_snapshot_id": "forbidden"})
    assert path.read_bytes() == before


def test_verified_sqlite_preflight_view_rejects_existing_file_without_schema(tmp_path):
    path = tmp_path / "uninitialized.sqlite"
    path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="schema is not initialized"):
        open_fleet_store_read_only(path)
    assert path.read_bytes() == b""


def test_scientific_receipt_hash_ignores_nested_observation_timestamps():
    class VolatileProvider(Provider):
        calls = 0

        def discover(self, task):
            value = super().discover(task)
            self.calls += 1
            value["generated_at_utc"] = f"2026-08-02T00:00:0{self.calls}Z"
            return value

    provider = VolatileProvider()
    first = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=provider,
        ct_sampler=Sampler(), code_revision="1" * 40)["private_receipt"]
    second = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(), provider=provider,
        ct_sampler=Sampler(), code_revision="1" * 40)["private_receipt"]
    assert first["scientific_core"] == second["scientific_core"]
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_source_failures_are_not_counted_as_successful_or_exact_census():
    class HalfFailingProvider(Provider):
        def discover(self, task):
            if task["center_xyz"]["x"] == 8:
                raise RuntimeError("source offline")
            return super().discover(task)

    receipt = run_candidate_coverage_preflight(
        snapshot(), request(), surface_view=ReadOnlySurfaces(),
        provider=HalfFailingProvider(), ct_sampler=Sampler(), code_revision="1" * 40,
    )["private_receipt"]
    assert receipt["measurement_kind"] == "INCOMPLETE_CENSUS"
    assert receipt["planned_sampling_percentage"] == 100
    assert receipt["achieved_successful_sampling_percentage"] == 50
    assert receipt["funnel"]["cells_attempted"] == 8
    assert receipt["funnel"]["cells_surveyed_successfully"] == 4
    assert receipt["funnel"]["cells_failed_source"] == 4
    assert receipt["no_candidate_causes"] == {"SOURCE_ERROR": 4}


@pytest.mark.parametrize("field,value", [
    ("parallelism", 0), ("parallelism", 9), ("maximum_cells", 0),
    ("source_content_lock_sha256", "0" * 64),
])
def test_preflight_fails_closed_on_invalid_bounds_or_source_lock(field, value):
    with pytest.raises(ValueError):
        run_candidate_coverage_preflight(
            snapshot(), request(**{field: value}), surface_view=ReadOnlySurfaces(),
            provider=Provider(), ct_sampler=Sampler(), code_revision="1" * 40,
        )


def test_preflight_rejects_a_hash_consistent_but_unverified_source_lock():
    unlocked = snapshot()
    unlocked["source_content_lock"]["status"] = "HASH_DECLARED_ONLY"
    with pytest.raises(ValueError, match="not verified immutable"):
        run_candidate_coverage_preflight(
            unlocked,
            request(source_content_lock_sha256=content_sha256(unlocked["source_content_lock"])),
            surface_view=ReadOnlySurfaces(), provider=Provider(), ct_sampler=Sampler(),
            code_revision="1" * 40,
        )


def test_cli_exposes_bounded_read_only_preflight_command():
    parser = build_parser(ROOT)
    args = parser.parse_args([
        "preflight", "--db", "fleet.sqlite", "--sample", "PHerc358",
        "--mission-id", "first-letters", "--source-snapshot-id", "source-locked-1",
        "--p0-artifact-id", "p0-1", "--p0-artifact-sha256", "d" * 64,
        "--p0-selection-version", "selection-1", "--p0-selection-sha256", "e" * 64,
        "--source-content-lock-sha256", "e" * 64,
        "--catalog-snapshot-sha256", "c" * 64, "--grid-version", "grid@1.0.0",
        "--policy-version", "policy@1.0.0", "--grid-step", "16",
        "--query-radius", "8", "--cell-clearance", "0", "--volume-clearance", "8",
        "--candidate-interior-clearance", "2", "--selection-strategy",
        "stratified-clearance-v1", "--m7-threshold", "0.2", "--maximum-cells", "8",
        "--parallelism", "2", "--private-output", "private.json",
        "--sanitized-output", "public.json", "--code-revision", "1" * 40,
    ])
    assert args.command == "preflight"
    assert args.maximum_cells == 8 and args.parallelism == 2
