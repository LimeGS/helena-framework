#!/usr/bin/env python3
"""Build the frozen CT-gate spec for Large TimeSformer hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_spec(
    *,
    root: Path,
    manifest_path: Path,
    amendment_path: Path,
    runtime: Path,
    gate_freeze: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    groups: list[dict[str, Any]] = []
    morphology: list[dict[str, Any]] = []
    for control in manifest["controls"]:
        if control["expected_role"] != "HARD_NEGATIVE_CT_DEPTH_DIFFUSE":
            continue
        control_id = str(control["id"])
        analysis_path = (
            runtime / control_id / "analysis" / "INK_STABILITY_ANALYSIS.json"
        )
        if not analysis_path.is_file():
            raise RuntimeError(f"missing hard-negative analysis: {control_id}")
        analysis = read_json(analysis_path)
        candidates = analysis["text_like_screening"]["candidates"]
        morphology.append(
            {
                "control_id": control_id,
                "candidate_count": len(candidates),
                "analysis_sha256": sha256_file(analysis_path),
            }
        )
        if not candidates:
            continue
        depth_centers = list(map(int, control["depth_centers"]))
        groups.append(
            {
                "group_id": control_id,
                "class": "HARD_NEGATIVE_CT_DEPTH_DIFFUSE",
                "tiff_directory": str(control["tiff_directory"]),
                "analysis": str(analysis_path.relative_to(root)),
                "central_slice": depth_centers[len(depth_centers) // 2],
                "voxel_um": float(control["source_slice_um"]),
            }
        )
    return {
        "kind": (
            "campaign_x_phase4_timesformer_large_control_ct_gate_spec_v1"
        ),
        "status": "FROZEN_BEFORE_CT_FEATURE_EXTRACTION",
        "generated_at_utc": utc_now(),
        "manifest": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256_file(manifest_path),
        "amendment": str(amendment_path.relative_to(root)),
        "amendment_sha256": sha256_file(amendment_path),
        "gate_freeze": str(gate_freeze.relative_to(root)),
        "gate_freeze_sha256": sha256_file(gate_freeze),
        "patch_radius_um": 200.0,
        "morphology_observations": morphology,
        "groups": groups,
        "policy": {
            "no_threshold_change": True,
            "zero_morphology_candidates_require_no_ct_component_gate": True,
            "any_retained_hard_negative_rejects_model_adoption": True,
            "retained_does_not_establish_ink_or_letters": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--gate-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = build_spec(
        root=root,
        manifest_path=args.manifest.resolve(),
        amendment_path=args.amendment.resolve(),
        runtime=args.runtime.resolve(),
        gate_freeze=args.gate_freeze.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hard_negative_count": len(
                    payload["morphology_observations"]
                ),
                "ct_group_count": len(payload["groups"]),
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
