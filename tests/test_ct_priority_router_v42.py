import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework"
    / "stages"
    / "04-validation"
    / "scripts"
    / "helena_develop_ct_priority_router_v42.py"
)
APPLY_SCRIPT = (
    ROOT
    / "framework"
    / "stages"
    / "04-validation"
    / "scripts"
    / "helena_apply_ct_priority_router_v42.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("helena_v42", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_apply_module():
    import sys

    script_directory = str(APPLY_SCRIPT.parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    spec = importlib.util.spec_from_file_location("helena_v42_apply", APPLY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _materialization(root: Path, benchmark_id: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    patches = rng.integers(20, 180, size=(40, 65, 35, 35), dtype=np.uint8)
    controls = []
    for index in range(40):
        positive = index % 2 == 0
        group = index % 5
        if positive:
            # A centered, depth-localized stroke that the morphology extractor
            # should represent without reading the label itself.
            patches[index, 29:36, 15:20, 8:27] = 245
        controls.append(
            {
                "patch_tensor_index": index,
                "analysis_bbox_xyxy": [0, 0, 35, 35],
                "component_id": f"{benchmark_id}-{index:03d}",
                "surface_group_id": f"PHercSynthetic:surface:region-{group}",
                "scroll_id": "PHercSynthetic",
                "expected_class": "POSITIVE" if positive else "CONFOUND",
            }
        )
    root.mkdir(parents=True)
    tensor_path = root / "CONTROL_CT_PATCHES.npy"
    controls_path = root / "FROZEN_CONTROLS.json"
    np.save(tensor_path, patches)
    controls_path.write_text(json.dumps(controls) + "\n", encoding="utf-8")
    receipt = {
        "benchmark_id": benchmark_id,
        "artifacts": {
            "patch_tensor_sha256": _sha256(tensor_path),
            "frozen_controls_sha256": _sha256(controls_path),
        },
    }
    (root / "MATERIALIZATION_RECEIPT.json").write_text(
        json.dumps(receipt) + "\n",
        encoding="utf-8",
    )


def test_texture_feature_schema_is_fixed_and_finite():
    module = _load_module()
    patch = np.arange(65 * 35 * 35, dtype=np.float32).reshape(65, 35, 35)
    features = module.extract_ct_texture_features(patch, [0, 0, 35, 35])
    assert features.shape == (5236,)
    assert np.isfinite(features).all()


def test_development_router_never_discards(tmp_path):
    module = _load_module()
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    _materialization(v1, "MULTISCROLL_TRANSFER_V1", 1)
    _materialization(v2, "MULTISCROLL_TRANSFER_V2", 2)
    output = tmp_path / "output"
    receipt = module.develop(
        [
            ("MULTISCROLL_TRANSFER_V1", v1),
            ("MULTISCROLL_TRANSFER_V2", v2),
        ],
        output,
        minimum_recall=0.95,
    )
    assert receipt["status"] == "DEVELOPMENT_OPTIMIZED_NOT_EXTERNALLY_VALIDATED"
    assert receipt["routing"]["automatic_discard"] is False
    assert receipt["model"]["feature_count"] == 5236
    routes = json.loads((output / "DEVELOPMENT_ROUTES.json").read_text())
    assert len(routes) == 80
    assert all(row["not_discarded"] is True for row in routes)
    assert {
        row["priority_route"] for row in routes
    } <= {
        "TIER_B1_HIGH_PRIORITY_REVIEW",
        "TIER_B2_PRESERVED_LOW_PRIORITY",
    }

    apply_module = _load_apply_module()
    items = json.loads((v1 / "FROZEN_CONTROLS.json").read_text())
    items[0]["base_v4_tier"] = "TIER_A_V3_RETAINED_REVIEW"
    items[1]["base_v4_tier"] = "TIER_C_EXTEND_OR_RESEGMENT"
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps(items) + "\n", encoding="utf-8")
    application_path = tmp_path / "application.json"
    application = apply_module.apply_router(
        output / "DEVELOPMENT_RECEIPT.json",
        v1 / "CONTROL_CT_PATCHES.npy",
        items_path,
        application_path,
    )
    assert application["status"] == "ROUTED_NONDESTRUCTIVELY"
    assert application["routes"][0]["priority_route"] == "TIER_A_V3_RETAINED_REVIEW"
    assert (
        application["routes"][1]["priority_route"]
        == "TIER_C_EXTEND_OR_RESEGMENT"
    )
    assert all(row["not_discarded"] is True for row in application["routes"])
