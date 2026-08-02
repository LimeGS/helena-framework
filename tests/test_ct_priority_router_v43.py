from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_develop_ct_priority_router_v43.py"
)
SELECTOR_SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/"
    "helena_select_official_ink_controls.py"
)


def load():
    spec = importlib.util.spec_from_file_location("helena_v43", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_selector():
    spec = importlib.util.spec_from_file_location(
        "helena_control_selector",
        SELECTOR_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialization(
    root: Path,
    benchmark_id: str,
    *,
    scrolls: list[str],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    controls = []
    patches = []
    for scroll_index, scroll in enumerate(scrolls):
        for surface_index in range(3):
            for item_index in range(8):
                positive = item_index % 2 == 0
                patch = rng.integers(
                    20,
                    170,
                    size=(65, 35, 35),
                    dtype=np.uint8,
                )
                if positive:
                    patch[29:36, 15:20, 7:28] = 245
                tensor_index = len(patches)
                patches.append(patch)
                controls.append(
                    {
                        "patch_tensor_index": tensor_index,
                        "analysis_bbox_xyxy": [0, 0, 35, 35],
                        "component_id": (
                            f"{benchmark_id}-{scroll_index}-{surface_index}-"
                            f"{item_index}"
                        ),
                        "surface_group_id": (
                            f"{scroll}:surface-{surface_index}:region-0"
                        ),
                        "official_surface_id": f"surface-{surface_index}",
                        "scroll_id": scroll,
                        "expected_class": (
                            "POSITIVE" if positive else "CONFOUND"
                        ),
                        "voxel_size_um": [2.4, 2.4, 2.4],
                        "patch_xy_spacing_um": 2.4,
                    }
                )
    root.mkdir(parents=True)
    tensor_path = root / "CONTROL_CT_PATCHES.npy"
    controls_path = root / "FROZEN_CONTROLS.json"
    np.save(tensor_path, np.stack(patches))
    controls_path.write_text(json.dumps(controls) + "\n", encoding="utf-8")
    (root / "MATERIALIZATION_RECEIPT.json").write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "artifacts": {
                    "patch_tensor_sha256": digest(tensor_path),
                    "frozen_controls_sha256": digest(controls_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_compact_features_are_physical_fixed_and_finite():
    module = load()
    patch = np.arange(65 * 35 * 35, dtype=np.float32).reshape(65, 35, 35)
    features = module.extract_compact_physical_features(
        patch,
        [0, 0, 35, 35],
        voxel_size_um=[2.4, 2.4, 2.4],
        patch_xy_spacing_um=2.4,
    )
    assert features.shape == (len(module.V43_FEATURE_NAMES),)
    assert features.shape == (163,)
    assert np.isfinite(features).all()


def test_v43_uses_complete_surfaces_and_never_v3(tmp_path):
    module = load()
    datasets = []
    definitions = [
        ("MULTISCROLL_TRANSFER_V1", ["PHerc0814", "PHerc1667"]),
        ("MULTISCROLL_TRANSFER_V2", ["PHercParis4"]),
        (
            "CT_PRIORITY_ROUTER_V43_DEVELOPMENT",
            ["PHerc0139", "PHerc0841"],
        ),
    ]
    for index, (benchmark_id, scrolls) in enumerate(definitions):
        root = tmp_path / f"dataset-{index}"
        materialization(root, benchmark_id, scrolls=scrolls, seed=index + 1)
        datasets.append((benchmark_id, root))
    output = tmp_path / "output"
    receipt = module.develop(datasets, output, minimum_recall=0.95)
    assert receipt["model"]["feature_count"] == 163
    assert (
        receipt["model"]["kind"]
        == "L2_REGULARIZED_HISTOGRAM_GRADIENT_BOOSTING"
    )
    assert receipt["model"]["hyperparameter_search_performed"] is False
    assert receipt["development_complete_surface_count"] == 15
    assert receipt["development_protocol"]["scroll_disjoint"] is True
    assert (
        receipt["development_protocol"]["calibration_scrolls"]
        == ["PHerc0139", "PHerc0841"]
    )
    assert (
        receipt["contamination_controls"][
            "multiscroll_transfer_v3_used_for_training"
        ]
        is False
    )
    assert receipt["routing"]["automatic_discard"] is False
    assert (output / "CT_PRIORITY_ROUTER_V43.joblib").is_file()


def test_v43_rejects_v3_as_development(tmp_path):
    module = load()
    root = tmp_path / "v3"
    materialization(
        root,
        "MULTISCROLL_TRANSFER_V3",
        scrolls=["P1"],
        seed=1,
    )
    try:
        module._load_dataset("MULTISCROLL_TRANSFER_V3", root)
    except RuntimeError as error:
        assert "prohibited" in str(error)
    else:
        raise AssertionError("v4.3 accepted V3 development data")


def test_surface_relative_routing_is_deterministic_and_non_destructive():
    module = load()
    scores = np.asarray(list(range(25)) + list(range(10)), dtype=np.float64)
    groups = np.asarray(["large"] * 25 + ["small"] * 10)
    component_ids = np.asarray(
        [f"large-{index:02d}" for index in range(25)]
        + [f"small-{index:02d}" for index in range(10)]
    )
    ranks = module.surface_relative_ranks(scores, groups, component_ids)
    assert np.sum(ranks[:25] <= 0.20) == 5
    assert np.all(ranks[25:] == 1.0)
    # B2 is a priority route only. Every input remains represented.
    b1 = ranks > 0.20
    assert len(b1) == len(scores)


def test_official_mask_padding_is_origin_preserving_and_not_selectable():
    selector = load_selector()
    label = np.ones((64, 64), dtype=bool)
    supervision = np.ones((64, 64), dtype=bool)
    reduced_label, reduced_supervision, alignment = (
        selector.HfLevelLoader._downsample_masks(
            label,
            supervision,
            (3, 4),
        )
    )
    assert reduced_label.shape == (3, 4)
    assert reduced_supervision.shape == (3, 4)
    assert reduced_label[:2, :2].all()
    assert reduced_supervision[:2, :2].all()
    assert not reduced_label[2, :].any()
    assert not reduced_label[:, 2:].any()
    assert not reduced_supervision[2, :].any()
    assert not reduced_supervision[:, 2:].any()
    assert alignment["kind"] == "TOP_LEFT_ORIGIN_BOTTOM_RIGHT_ZERO_PADDING"
    assert alignment["bottom_padding_cells"] == 1
    assert alignment["right_padding_cells"] == 2
    assert alignment["padded_cells_are_selectable"] is False
