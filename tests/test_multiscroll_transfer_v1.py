from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/helena_multiscroll_transfer_v1.py"
)
PREPARE_SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_prepare_multiscroll_ct_adjudication.py"
)
VIEWER_SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_build_multiscroll_adjudication_viewer.py"
)
OFFICIAL_LABEL_PLAN_SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_build_official_ink_label_plan.py"
)
OFFICIAL_CONTROL_SELECTOR_SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_select_official_ink_controls.py"
)
OFFICIAL_CONTROL_MATERIALIZER_SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_materialize_official_control_patches.py"
)


def load(path: Path = SCRIPT):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def asset():
    return {"uri": "s3://public/example.tif", "sha256": "a" * 64}


def source(scroll: str, *, role: str = "EVALUATION", positives: int = 50, confounds: int = 50):
    return {
        "scroll_id": scroll,
        "benchmark_role": role,
        "label_authority": "PUBLIC_EXPERT_LABEL",
        "certified_positive_components": positives,
        "certified_confound_components": confounds,
        "independent_surface_groups": 5,
        "assets": [asset()],
    }


def manifest(*, status: str = "FROZEN_BEFORE_RESULTS"):
    return {
        "benchmark_id": "MULTISCROLL_TRANSFER_V1",
        "status": status,
        "development_scrolls": ["PHerc0139"],
        "policy": {
            "minimum_independent_positive_scrolls": 3,
            "minimum_independent_confound_scrolls": 2,
            "minimum_positive_components_per_scroll": 50,
            "minimum_confound_components_per_scroll": 50,
            "minimum_surface_groups_per_scroll": 5,
            "minimum_positive_recall_per_scroll": 0.95,
            "bootstrap_iterations": 100,
            "bootstrap_seed": 7,
        },
        "scroll_sources": [source("P1"), source("P2"), source("P3")],
        "controls": [{"frozen": True}],
    }


def control(scroll: str, index: int, expected: str, tier: str):
    row = {
        "scroll_id": scroll,
        "surface_group_id": f"{scroll}-surface-{index % 5}",
        "component_id": f"{expected}-{index:03d}",
        "expected_class": expected,
        "label_source": {
            "label_authority": "PUBLIC_EXPERT_LABEL",
            "assets": [asset()],
        },
        "ct_coordinate_xyz": [100.0 + index, 200.0, 300.0],
        "voxel_size_um": [7.91, 7.91, 7.91],
        "slice_order": "ZYX_ASCENDING",
        "scanner_domain": f"scanner-{scroll}",
        "decision_sources": {
            "v3": {
                "uri": f"file:///frozen/{scroll}/{expected}-{index:03d}/v3.json",
                "sha256": "b" * 64,
            },
            "v4": {
                "uri": f"file:///frozen/{scroll}/{expected}-{index:03d}/v4.json",
                "sha256": "c" * 64,
            },
        },
        "v3_retained": tier == "TIER_A_V3_RETAINED_REVIEW",
        "v4_tier": tier,
        "v4_not_discarded": True,
    }
    if expected == "CONFOUND":
        row["confound_subtype"] = "FIBER"
    return row


def test_preflight_excludes_development_scroll_by_scroll_identity():
    module = load()
    draft = manifest(status="DRAFT_SOURCE_AUDIT")
    draft["controls"] = []
    draft["scroll_sources"].append(source("PHerc0139", role="EVALUATION"))
    result = module.preflight_manifest(draft)
    assert "DEVELOPMENT_SCROLL_NOT_EXCLUDED:PHerc0139" in result["blocking_reasons"]


def test_preflight_rejects_model_predictions_as_ground_truth():
    module = load()
    draft = manifest(status="DRAFT_SOURCE_AUDIT")
    draft["controls"] = []
    draft["scroll_sources"][0]["label_authority"] = "MODEL_PREDICTION"
    result = module.preflight_manifest(draft)
    assert any(
        reason.endswith("PROHIBITED_LABEL_AUTHORITY:MODEL_PREDICTION")
        for reason in result["blocking_reasons"]
    )


def test_preflight_requires_curated_label_alignment_assets():
    module = load()
    draft = manifest(status="DRAFT_SOURCE_AUDIT")
    draft["controls"] = []
    curated = draft["scroll_sources"][0]
    curated["label_authority"] = "PUBLIC_CURATED_SURFACE_LABEL"
    curated["assets"] = [
        {**asset(), "role": "INK_LABEL"},
        {**asset(), "role": "SURFACE_CT"},
    ]
    curated["prediction_used_as_ground_truth"] = False
    curated["coordinate_frame_id"] = "scroll:surface:canvas-v1"
    result = module.preflight_manifest(draft)
    assert any(
        reason.endswith("MISSING_CURATED_LABEL_ASSET:SUPERVISION_MASK")
        for reason in result["blocking_reasons"]
    )


def test_benchmark_passes_three_scrolls_without_discarding_tier_c():
    module = load()
    controls = []
    for scroll in ["P1", "P2", "P3"]:
        controls.extend(
            control(scroll, index, "POSITIVE", "TIER_A_V3_RETAINED_REVIEW")
            for index in range(48)
        )
        controls.extend(
            control(scroll, index + 48, "POSITIVE", "TIER_B_SHADOW_REVIEW")
            for index in range(2)
        )
        controls.extend(
            control(scroll, index, "CONFOUND", "TIER_B_SHADOW_REVIEW")
            for index in range(50)
        )
    result = module.evaluate_benchmark(manifest(), controls)
    assert result["status"] == "MULTISCROLL_TRANSFER_V1_PASSED"
    assert (
        result["metrics_by_scroll"]["P1"]["positive_v4_direct_review_recall"]["rate"]
        == 1.0
    )


def test_tier_c_positive_is_preserved_but_can_fail_direct_review_gate():
    module = load()
    controls = []
    for scroll in ["P1", "P2", "P3"]:
        controls.extend(
            control(scroll, index, "POSITIVE", "TIER_A_V3_RETAINED_REVIEW")
            for index in range(47)
        )
        controls.extend(
            control(scroll, index + 47, "POSITIVE", "TIER_C_EXTEND_OR_RESEGMENT")
            for index in range(3)
        )
        controls.extend(
            control(scroll, index, "CONFOUND", "TIER_B_SHADOW_REVIEW")
            for index in range(50)
        )
    result = module.evaluate_benchmark(manifest(), controls)
    assert result["status"] == "MULTISCROLL_TRANSFER_V1_BLOCKED_OR_FAILED"
    assert "P1:V4_DIRECT_REVIEW_RECALL_BELOW_GATE" in result[
        "blocking_or_failure_reasons"
    ]
    assert (
        result["metrics_by_scroll"]["P1"]["positive_v4_preservation_recall"]["rate"]
        == 1.0
    )


def test_stage_registry_resolves_benchmark_script():
    from scripts.harness.stage_script_registry import resolve_stage_script

    for name in [
        "helena_multiscroll_transfer_v1.py",
        "helena_build_multiscroll_control_acquisition.py",
        "helena_prepare_multiscroll_ct_adjudication.py",
        "helena_build_multiscroll_adjudication_viewer.py",
        "helena_build_official_ink_label_plan.py",
        "helena_select_official_ink_controls.py",
        "helena_materialize_official_control_patches.py",
        "helena_execute_multiscroll_gates_once.py",
    ]:
        assert resolve_stage_script(ROOT, name).name == name


def test_level0_materializer_enumerates_every_intersecting_chunk():
    module = load(OFFICIAL_CONTROL_MATERIALIZER_SCRIPT)
    controls = [
        {"component_id": "center", "surface_y_x_level0": [128, 128]},
        {"component_id": "edge", "surface_y_x_level0": [0, 0]},
    ]
    names = module._chunks_for_controls(
        controls,
        shape=(65, 512, 512),
        chunks=(65, 128, 128),
        radius=83,
    )
    assert names == {
        "0.0.0",
        "0.0.1",
        "0.1.0",
        "0.1.1",
    }


def test_level0_materializer_pads_edge_without_moving_center():
    module = load(OFFICIAL_CONTROL_MATERIALIZER_SCRIPT)
    stack = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    # The production stack always has 65 planes; duplicate the fixture so the
    # helper's fixed scientific depth contract is exercised.
    stack = np.concatenate([stack, np.zeros((63, 4, 4), dtype=np.uint8)])
    patch = module._crop_with_padding(
        stack,
        center_y=0,
        center_x=0,
        radius=2,
    )
    assert patch.shape == (65, 5, 5)
    assert patch[0, 2, 2] == stack[0, 0, 0]
    assert patch[0, 4, 4] == stack[0, 2, 2]
    assert np.count_nonzero(patch[0, :2]) == 0
    assert np.count_nonzero(patch[0, :, :2]) == 0


def test_level0_materializer_honors_declared_tifxyz_scale(tmp_path):
    module = load(OFFICIAL_CONTROL_MATERIALIZER_SCRIPT)
    coordinates = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "x.tif"
    tifffile.imwrite(path, coordinates)
    values, sampled_y_x = module._read_points(
        path,
        [
            {"component_id": "origin", "surface_y_x_level0": [0, 0]},
            {"component_id": "last", "surface_y_x_level0": [39, 59]},
        ],
        surface_shape_y_x=(40, 60),
        tifxyz_scale_y_x=(0.05, 0.05),
    )
    assert values == [0.0, 5.0]
    assert sampled_y_x == [[0, 0], [1, 2]]


def test_official_selector_requires_full_positive_supervision_margin():
    module = load(OFFICIAL_CONTROL_SELECTOR_SCRIPT)
    supervision = np.ones((11, 11), dtype=bool)
    supervision[0, :] = False
    candidates = [
        (1.0, 2, 5),
        (1.0, 5, 5),
    ]
    assert module._inside_supervision_margin(
        candidates,
        supervision,
        margin=2,
    ) == [(1.0, 5, 5)]


def test_official_selector_excludes_overlapping_prior_audit_patches():
    module = load(OFFICIAL_CONTROL_SELECTOR_SCRIPT)
    candidates = [
        (1.0, 10, 10),
        (0.9, 20, 20),
        (0.8, 40, 40),
    ]
    kept = module._outside_frozen_control_exclusions(
        candidates,
        selection_level=0,
        excluded_level0_y_x=[(10, 10)],
        minimum_distance_level0=20.0,
    )
    assert kept == [(0.8, 40, 40)]


def test_benchmark_v2_reports_v2_status_without_changing_v1_contract():
    module = load()
    frozen = manifest()
    frozen["benchmark_id"] = "MULTISCROLL_TRANSFER_V2"
    controls = []
    for scroll in ["P1", "P2", "P3"]:
        controls.extend(
            control(scroll, index, "POSITIVE", "TIER_B_SHADOW_REVIEW")
            for index in range(50)
        )
        controls.extend(
            control(scroll, index, "CONFOUND", "TIER_B_SHADOW_REVIEW")
            for index in range(50)
        )
    result = module.evaluate_benchmark(frozen, controls)
    assert result["status"] == "MULTISCROLL_TRANSFER_V2_PASSED"


def test_official_label_plan_requires_aligned_surface_assets():
    module = load(OFFICIAL_LABEL_PLAN_SCRIPT)

    def directory(path: str):
        return {"type": "directory", "path": path}

    def file(path: str):
        return {
            "type": "file",
            "path": path,
            "size": 123,
            "xet_hash": "a" * 64,
        }

    roots = {
        "ink/1667": ["w1", "w2", "w3", "w4", "w5"],
        "ink/phercparis4": ["p1", "p2", "p3", "p4", "p5"],
        "ink/814": ["i1"],
    }

    def lister(path: str):
        if path in roots:
            return [directory(f"{path}/{name}") for name in roots[path]]
        name = Path(path).name
        return [
            file(f"{path}/meta.json"),
            directory(f"{path}/{name}.zarr"),
            directory(f"{path}/preds"),
            file(f"{path}/{name}_inklabels.tif"),
            file(f"{path}/{name}_supervision_mask.tif"),
            file(f"{path}/x.tif"),
            file(f"{path}/y.tif"),
            file(f"{path}/z.tif"),
        ]

    result = module.build_plan(lister=lister)
    assert result["status"] == "OFFICIAL_LABEL_SOURCES_READY"
    assert result["by_scroll"]["PHerc0814"]["ready_surface_count"] == 1
    assert result["by_scroll"]["PHerc0814"]["prospective_region_group_count"] == 5
    assert all(
        not row["prediction_assets_used_as_truth"] for row in result["surfaces"]
    )
    assert all(
        "/preds" not in str(row["assets"]["surface_ct"])
        for row in result["surfaces"]
    )


def test_official_label_plan_fails_closed_when_supervision_is_missing():
    module = load(OFFICIAL_LABEL_PLAN_SCRIPT)

    def lister(path: str):
        if path in {"ink/1667", "ink/phercparis4", "ink/814"}:
            return [{"type": "directory", "path": f"{path}/surface"}]
        return [
            {"type": "file", "path": f"{path}/meta.json"},
            {"type": "directory", "path": f"{path}/surface.zarr"},
            {"type": "file", "path": f"{path}/surface_inklabels.tif"},
            {"type": "file", "path": f"{path}/x.tif"},
            {"type": "file", "path": f"{path}/y.tif"},
            {"type": "file", "path": f"{path}/z.tif"},
        ]

    result = module.build_plan(lister=lister)
    assert result["status"] == "BLOCKED_INSUFFICIENT_ALIGNED_GROUPS"
    assert all(
        row["status"] == "BLOCKED_MISSING_ALIGNED_ASSETS"
        for row in result["surfaces"]
    )
    assert all("supervision_mask" in row["missing_assets"] for row in result["surfaces"])


def test_official_control_selector_balances_three_scrolls_and_five_groups():
    module = load(OFFICIAL_CONTROL_SELECTOR_SCRIPT)

    def surface(scroll: str, name: str, partitions: int = 1):
        return {
            "scroll_id": scroll,
            "official_surface_id": name,
            "status": "READY",
            "region_partition_count": partitions,
            "assets": {
                axis: {
                    "uri": f"hf://buckets/test/{scroll}/{name}/{axis}.tif",
                    "size_bytes": 100,
                }
                for axis in (
                    "ink_label",
                    "supervision_mask",
                    "surface_ct",
                    "tifxyz_x",
                    "tifxyz_y",
                    "tifxyz_z",
                )
            },
        }

    plan = {
        "benchmark_id": "MULTISCROLL_TRANSFER_V1",
        "status": "OFFICIAL_LABEL_SOURCES_READY",
        "content_sha256": "d" * 64,
        "surfaces": [
            *[surface("P1", f"s{index}") for index in range(5)],
            *[surface("P2", f"s{index}") for index in range(5)],
            surface("P3", "s0", partitions=5),
        ],
    }

    class FakeLoader:
        def load(self, row):
            height, width = (320, 320)
            label = np.zeros((height, width), dtype=bool)
            supervision = np.ones((height, width), dtype=bool)
            ct = np.zeros((height, width), dtype=np.float32)
            partitions = int(row["region_partition_count"])
            for region in range(partitions):
                left = region * width // partitions
                right = (region + 1) * width // partitions
                label[8:312, left + 4 : right - 4] = True
                for local_index in range(32):
                    y = 8 + (local_index * 9) % 300
                    x = left + 6 + (local_index * 11) % max(8, right - left - 12)
                    ct[y, x] = float(1000 - local_index)
            return {
                "label_level0": label,
                "supervision_level0": supervision,
                "label_reduced": label,
                "supervision_reduced": supervision,
                "ct": ct,
                "central_slice": 1,
                "shapes": {
                    "label_yx": [height, width],
                    "supervision_yx": [height, width],
                    "ct_zyx": [3, height, width],
                },
                "mask_alignment": {
                    "method": "TEST_IDENTITY",
                    "label_level0_shape": [height, width],
                    "supervision_level0_shape": [height, width],
                    "surface_ct_plane_shape": [height, width],
                },
                "assets": {
                    name: {
                        "uri": f"hf://buckets/test/{row['scroll_id']}/{row['official_surface_id']}/{name}.zarr/5/",
                        "sha256": "e" * 64,
                        "role": role,
                    }
                    for name, role in (
                        ("label", "INK_LABEL"),
                        ("supervision", "SUPERVISION_MASK"),
                        ("ct", "SURFACE_CT"),
                    )
                },
            }

    result = module.select_controls(plan, FakeLoader(), target_per_class=10)
    assert (
        result["status"]
        == "OFFICIAL_CONTROL_CANDIDATES_READY_FOR_LEVEL0_MATERIALIZATION"
    )
    assert len(result["controls"]) == 60
    for scroll in ("P1", "P2", "P3"):
        assert result["by_scroll"][scroll]["counts"] == {
            "POSITIVE": 10,
            "CONFOUND": 10,
        }
        assert result["by_scroll"][scroll]["surface_groups"] == {
            "POSITIVE": 5,
            "CONFOUND": 5,
        }
    assert all(
        row["label_source"]["prediction_used_as_ground_truth"] is False
        for row in result["controls"]
    )
    assert all(
        row["v3_v4_outputs_read_during_selection"] is False
        for row in result["controls"]
    )


def test_official_control_selector_prefers_unused_surfaces_and_combines_exclusions():
    module = load(OFFICIAL_CONTROL_SELECTOR_SCRIPT)

    def surface(name: str):
        return {
            "scroll_id": "P1",
            "official_surface_id": name,
            "status": "READY",
            "region_partition_count": 5,
            "assets": {
                axis: {
                    "uri": f"hf://buckets/test/P1/{name}/{axis}.tif",
                    "size_bytes": 100,
                }
                for axis in (
                    "ink_label",
                    "supervision_mask",
                    "surface_ct",
                    "tifxyz_x",
                    "tifxyz_y",
                    "tifxyz_z",
                )
            },
        }

    plan = {
        "benchmark_id": "MULTISCROLL_TRANSFER_V1",
        "status": "OFFICIAL_LABEL_SOURCES_READY",
        "content_sha256": "d" * 64,
        "surfaces": [surface("used"), surface("unused")],
    }

    class FakeLoader:
        def load(self, row):
            height = width = 512
            label = np.zeros((height, width), dtype=bool)
            supervision = np.ones((height, width), dtype=bool)
            ct = np.zeros((height, width), dtype=np.float32)
            label[16:496, 16:496] = True
            for index in range(200):
                y = 20 + (index * 37) % 470
                x = 20 + (index * 53) % 470
                ct[y, x] = float(1000 - index)
            return {
                "label_level0": label,
                "supervision_level0": supervision,
                "label_reduced": label,
                "supervision_reduced": supervision,
                "ct": ct,
                "central_slice": 1,
                "shapes": {
                    "label_yx": [height, width],
                    "supervision_yx": [height, width],
                    "ct_zyx": [3, height, width],
                },
                "mask_alignment": {
                    "method": "TEST_IDENTITY",
                    "label_level0_shape": [height, width],
                    "supervision_level0_shape": [height, width],
                    "surface_ct_plane_shape": [height, width],
                },
                "assets": {
                    name: {
                        "uri": f"hf://buckets/test/P1/{row['official_surface_id']}/{name}",
                        "sha256": "e" * 64,
                        "role": role,
                    }
                    for name, role in (
                        ("label", "INK_LABEL"),
                        ("supervision", "SUPERVISION_MASK"),
                        ("ct", "SURFACE_CT"),
                    )
                },
            }

    exclusions = [
        {
            "scroll_id": "P1",
            "official_surface_id": "used",
            "surface_y_x_level0": [100, 100],
        },
        {
            "scroll_id": "P1",
            "official_surface_id": "used",
            "surface_y_x_level0": [300, 300],
        },
    ]
    result = module.select_controls(
        plan,
        FakeLoader(),
        target_per_class=5,
        excluded_controls=exclusions,
        exclusion_radius_level0=64,
        benchmark_id="MULTISCROLL_TRANSFER_V3",
    )
    assert result["benchmark_id"] == "MULTISCROLL_TRANSFER_V3"
    assert result["selection_policy"]["excluded_frozen_control_count"] == 2
    assert {row["official_surface_id"] for row in result["controls"]} == {"unused"}
    assert all(
        row["component_id"].startswith("MULTISCROLL_TRANSFER_V3:")
        for row in result["controls"]
    )


def test_v3_priority_metrics_preserve_b2_and_bootstrap_by_surface_group():
    path = (
        ROOT
        / "framework/stages/04-validation/scripts/"
        "helena_execute_ct_priority_transfer_v3_once.py"
    )
    sys.path.insert(0, str(path.parent))
    try:
        module = load(path)
    finally:
        sys.path.remove(str(path.parent))
    rows = [
        {
            "surface_group_id": "g1",
            "positive_high_priority": True,
            "not_discarded": True,
        },
        {
            "surface_group_id": "g1",
            "positive_high_priority": False,
            "not_discarded": True,
        },
        {
            "surface_group_id": "g2",
            "positive_high_priority": True,
            "not_discarded": True,
        },
    ]
    metric = module._metric(
        rows,
        "positive_high_priority",
        seed=module.BOOTSTRAP_SEED,
    )
    assert metric["components"] == 3
    assert metric["surface_groups"] == 2
    assert metric["successes"] == 2
    assert abs(metric["rate"] - (2 / 3)) < 1e-12
    assert len(metric["surface_group_bootstrap_95"]) == 2
    assert module._base_tier(
        {"shadow_tier": "TIER_B_SHADOW_REVIEW"}
    ) == "TIER_B_SHADOW_REVIEW"
    assert module._base_tier(
        {"shadow_tier": "TIER_C_EXTEND_OR_RESEGMENT"}
    ) == "TIER_C_EXTEND_OR_RESEGMENT"


def test_acquisition_queue_uses_predictions_only_as_locators():
    path = (
        ROOT
        / "framework/stages/04-validation/scripts/"
        "helena_build_multiscroll_control_acquisition.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {}
    for index in range(6):
        data[str(index)] = {
            "long_id": f"segment-{index}",
            "original_volume_id": "volume-1",
            "properties": {"width": 100, "height": 100},
            "data": [
                {
                    "type": "ink-detection",
                    "origins": [{"path": f"prediction-{index}.tif"}],
                },
                {
                    "type": "ink-detection-downsampled",
                    "origins": [{"path": f"preview-{index}.jpg"}],
                },
            ],
        }
    queue = module.build_queue(
        {"samples": {"P1": {"segments": data}}},
        scrolls=("P1",),
        regions_per_scroll=5,
        metadata_sha256="d" * 64,
    )
    assert queue["task_count"] == 5
    assert all(
        task["selection"]["prediction_used_only_as_candidate_locator"]
        for task in queue["tasks"]
    )
    assert all("expected_class" not in task for task in queue["tasks"])


def test_label_blind_proposals_are_spaced_and_unlabeled():
    module = load(PREPARE_SCRIPT)
    image = np.zeros((512, 512), dtype=np.uint8)
    for index, (y, x) in enumerate(
        [(96, 96), (96, 240), (96, 384), (240, 96), (240, 240), (384, 384)]
    ):
        image[y - 5 : y + 6, x - 5 : x + 6] = 220 - index * 10
    image[64:448:16, 64:448:16] = 80
    proposals = module.sample_proposals(
        image,
        task_id="MULTISCROLL_TRANSFER_V1:test",
        high_count=4,
        mid_count=3,
    )
    assert 4 <= len(proposals) <= 7
    assert all(row["expected_class"] is None for row in proposals)
    assert all(row["label_authority"] is None for row in proposals)
    assert all(row["adjudication_status"] == "UNREVIEWED" for row in proposals)
    assert {row["proposal_stratum"] for row in proposals} <= {
        "HIGH_RESPONSE_LOCATOR",
        "MID_RESPONSE_LOCATOR",
    }
    assert all(
        row["surface_y_x_level0"]
        == [row["preview_y_x"][0] * 8, row["preview_y_x"][1] * 8]
        for row in proposals
    )


def test_asset_resolution_selects_registered_2p4um_pair():
    module = load(PREPARE_SCRIPT)
    segment = {
        "data": [
            {
                "type": "ink-detection-downsampled",
                "origins": [
                    {"path": "x/1.129um-volume-111-ds8.jpg"},
                    {"path": "x/2.399um-volume-222-ds8.jpg"},
                ],
            },
            {
                "type": "layers-zarr",
                "origins": [
                    {"path": "x/1.129um-volume-111.zarr/"},
                    {"path": "x/2.399um-volume-222.zarr/"},
                ],
            },
            {
                "type": "tifxyz-transformed",
                "origins": [
                    {"path": "x/segment-on-111-1.129um.tifxyz/"},
                    {"path": "x/segment-on-222-2.399um.tifxyz/"},
                ],
            },
        ]
    }
    metadata = {
        "samples": {
            "PHercTEST": {
                "segments": {"seg": segment},
                "volumes": {
                    "222": {
                        "properties": {
                            "pixel_size_um": 2.399,
                            "energy_keV": 78,
                            "z_direction_is_top_to_bottom": False,
                            "left_handed_coordinates": False,
                        }
                    }
                },
            }
        }
    }
    _, assets = module.resolve_surface_assets(metadata, "PHercTEST", "seg")
    assert "2.399um" in assets.preview_path
    assert assets.zarr_path == "x/2.399um-volume-222.zarr/"
    assert assets.tifxyz_path == "x/segment-on-222-2.399um.tifxyz/"
    assert assets.volume_id == "222"


def test_tifxyz_coordinates_are_mapped_from_registered_canvas(tmp_path):
    module = load(PREPARE_SCRIPT)
    tifxyz = tmp_path / "tifxyz"
    tifxyz.mkdir()
    base = np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    for index, axis in enumerate(("x", "y", "z")):
        tifffile.imwrite(tifxyz / f"{axis}.tif", base + index * 100.0)
    proposal = {"surface_y_x_level0": [3, 3]}
    coordinates = module.load_tifxyz_coordinates(
        tifxyz,
        proposal,
        canvas_size_xy=(7, 7),
    )
    assert coordinates == [15.0, 115.0, 215.0]
    assert proposal["tifxyz_coordinate_pixel_y_x"] == [0.5, 0.5]


def test_viewer_refuses_prelabelled_input(tmp_path):
    module = load(VIEWER_SCRIPT)
    proposals = tmp_path / "LABEL_BLIND_CT_PROPOSALS.json"
    proposals.write_text('[{"proposal_id":"x","expected_class":"POSITIVE"}]')
    try:
        module.build(proposals, tmp_path / "viewer.html")
    except ValueError as error:
        assert "label-blind" in str(error)
    else:
        raise AssertionError("prelabelled viewer input must fail closed")
