#!/usr/bin/env python3
"""Calibrate reproduced A8 seeds on the frozen PHerc0332 validation strip."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
WORKSPACE = ROOT.parent
PHASE2 = ROOT / "phase2"
GEOWRAP = WORKSPACE / "geowrap-ssl-lab"
sys.path[:0] = [str(PHASE2 / "src"), str(GEOWRAP / "src")]

from campaign_x_phase2.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    select_same_threshold,
)
from geowrap.experiment import canonicalize_edge_dataset, load_artifact  # noqa: E402
from geowrap.real_strips import load_strip_edge_dataset  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    strip_dir = WORKSPACE / "release" / "scroll-tracing-benchmark-gh" / "strips" / "pherc0332_pack"
    paths = sorted(strip_dir.glob("strip_s*.npz"))
    data = load_strip_edge_dataset(paths, seed=20260812, max_anchors_per_strip=5000)
    canonical = canonicalize_edge_dataset(data)
    crop_ids = np.asarray(canonical.row_metadata["trace.crop_id"], dtype=np.int64)
    validation = np.flatnonzero(crop_ids == 4)
    if not len(validation):
        raise RuntimeError("empty frozen validation strip")
    validation_data = canonical.subset(validation)
    output: dict[str, object] = {
        "kind": "campaign_x_phase2_a8_calibration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_status": "WEAK_GEOMETRIC_REFERENCE_NOT_PARIS4_GROUND_TRUTH",
        "validation_rows": len(validation_data.X),
        "validation_crop_id": 4,
        "seeds": {},
    }
    for seed in (13, 37, 73):
        checkpoint = PHASE2 / "runs" / "m1-cross-scroll-reproduction-20260715" / "runs" / f"A8_seed{seed}" / "checkpoint"
        artifact = load_artifact(checkpoint)
        raw = artifact.predict_proba(validation_data, device="cpu")
        temperature = fit_temperature(raw, validation_data.y)
        calibrated = apply_temperature(raw, temperature)
        selection = select_same_threshold(calibrated, validation_data.y)
        output["seeds"][str(seed)] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "metadata_sha256": sha256(checkpoint / "metadata.json"),
            "weights_sha256": sha256(checkpoint / "weights.npz"),
            "temperature": temperature,
            "ece_raw": expected_calibration_error(raw, validation_data.y),
            "ece_calibrated": expected_calibration_error(calibrated, validation_data.y),
            "same_threshold": selection,
        }
    destination = PHASE2 / "benchmark" / "A8_CALIBRATION.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
