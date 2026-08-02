#!/usr/bin/env python3
"""Prepare a complete immutable ScrollFiesta cube dump from a locked spec."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKENDS = Path(__file__).resolve().parents[1] / "backends"
sys.path.insert(0, str(BACKENDS))

from scrollfiesta.cube_stage import CubeStageConfig, CubeStageError, run_cube_stage  # noqa: E402


SCHEMA = "campaignx.scrollfiesta_cube_stage_request.v1"
REQUIRED = {
    "schema",
    "surface_prediction_uri",
    "surface_prediction_etag",
    "level",
    "roi_level0_zyx",
    "output_dir",
    "cube_mesh_bin",
    "halo",
    "threshold",
    "cube_edge_voxels",
    "max_concurrent",
    "threads_per_cube",
    "cube_timeout_seconds",
    "s3_anonymous",
}


def load_spec(path: Path) -> CubeStageConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != REQUIRED:
        keys = set(value) if isinstance(value, dict) else set()
        raise CubeStageError(
            f"request keys mismatch: missing={sorted(REQUIRED-keys)}, unknown={sorted(keys-REQUIRED)}"
        )
    if value["schema"] != SCHEMA:
        raise CubeStageError(f"expected schema {SCHEMA}")
    return CubeStageConfig(
        zarr_uri=str(value["surface_prediction_uri"]),
        source_etag=str(value["surface_prediction_etag"]),
        level=int(value["level"]),
        roi_level0_zyx=tuple(int(item) for item in value["roi_level0_zyx"]),
        output_dir=Path(value["output_dir"]),
        cube_mesh_bin=Path(value["cube_mesh_bin"]),
        halo=int(value["halo"]),
        threshold=value["threshold"],
        cube_edge_voxels=int(value["cube_edge_voxels"]),
        max_concurrent=int(value["max_concurrent"]),
        threads_per_cube=int(value["threads_per_cube"]),
        cube_timeout_seconds=float(value["cube_timeout_seconds"]),
        s3_anonymous=bool(value["s3_anonymous"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_cube_stage(load_spec(args.run_spec.resolve(strict=True)))
    except (CubeStageError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
