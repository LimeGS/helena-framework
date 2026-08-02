#!/usr/bin/env python3
"""Build ScrollFiesta cubes serially and archive intermediates per cube.

This is the low-disk companion to ``helena_prepare_scrollfiesta_cubes.py``.
It keeps the exact final step-12 OBJ tree needed by the welder, while storing
every preceding upstream OBJ byte-for-byte in a verified ``tar.zst`` archive.
No partial set is ever presented as a complete cube stage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


BACKENDS = Path(__file__).resolve().parents[1] / "backends"
sys.path.insert(0, str(BACKENDS))

from scrollfiesta.cube_stage import (  # noqa: E402
    CubeStageConfig,
    CubeStageError,
    _absolute_executable,
    _mesh_one,
    plan_cubes,
    require_complete,
    resolve_threshold,
)
from scrollfiesta.receipt import file_artifact, write_new_json  # noqa: E402


SCHEMA = "campaignx.scrollfiesta_compact_cube_stage_request.v1"
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
    "threads_per_cube",
    "cube_timeout_seconds",
    "s3_anonymous",
    "tar_bin",
}


def load_spec(path: Path) -> tuple[CubeStageConfig, Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != REQUIRED:
        keys = set(value) if isinstance(value, dict) else set()
        raise CubeStageError(
            f"request keys mismatch: missing={sorted(REQUIRED-keys)}, "
            f"unknown={sorted(keys-REQUIRED)}"
        )
    if value["schema"] != SCHEMA:
        raise CubeStageError(f"expected schema {SCHEMA}")
    tar_bin = _absolute_executable(Path(value["tar_bin"]))
    return (
        CubeStageConfig(
            zarr_uri=str(value["surface_prediction_uri"]),
            source_etag=str(value["surface_prediction_etag"]),
            level=int(value["level"]),
            roi_level0_zyx=tuple(int(item) for item in value["roi_level0_zyx"]),
            output_dir=Path(value["output_dir"]),
            cube_mesh_bin=Path(value["cube_mesh_bin"]),
            halo=int(value["halo"]),
            threshold=value["threshold"],
            cube_edge_voxels=int(value["cube_edge_voxels"]),
            max_concurrent=1,
            threads_per_cube=int(value["threads_per_cube"]),
            cube_timeout_seconds=float(value["cube_timeout_seconds"]),
            s3_anonymous=bool(value["s3_anonymous"]),
        ),
        tar_bin,
    )


def archive_and_compact_cube(
    *, dump_dir: Path, cube_id: str, archives_dir: Path, tar_bin: Path
) -> Path:
    """Archive a complete cube tree, then retain only its final OBJ tree."""

    cube_dir = dump_dir / cube_id
    final_name = f"{cube_id}_step12_final"
    final_dir = cube_dir / final_name
    final_obj = final_dir / f"{final_name}_all.obj"
    if not final_obj.is_file() or final_obj.stat().st_size == 0:
        raise CubeStageError(f"cannot compact incomplete cube {cube_id}")

    archive = archives_dir / f"{cube_id}.tar.zst"
    completed = subprocess.run(
        [str(tar_bin), "--zstd", "-cf", str(archive), "-C", str(dump_dir), cube_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or not archive.is_file() or archive.stat().st_size == 0:
        raise CubeStageError(
            f"archive failed for {cube_id}: rc={completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace')[-1000:]}"
        )
    verified = subprocess.run(
        [str(tar_bin), "--zstd", "-tf", str(archive)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if verified.returncode != 0:
        raise CubeStageError(f"archive verification failed for {cube_id}")

    held = dump_dir / f".{cube_id}.final"
    if held.exists():
        raise CubeStageError(f"unexpected compaction staging path: {held}")
    final_dir.rename(held)
    shutil.rmtree(cube_dir)
    cube_dir.mkdir()
    held.rename(cube_dir / final_name)
    return archive


def run_compact_stage(config: CubeStageConfig, tar_bin: Path) -> Path:
    output = Path(config.output_dir)
    if not output.is_absolute() or output.exists():
        raise CubeStageError(f"immutable absolute output_dir required: {output}")
    if config.threads_per_cube < 1 or config.cube_timeout_seconds <= 0:
        raise CubeStageError("thread and timeout values must be positive")
    cube_mesh = _absolute_executable(config.cube_mesh_bin)
    planned = plan_cubes(
        config.roi_level0_zyx,
        level=config.level,
        cube_edge_voxels=config.cube_edge_voxels,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    dump_dir = output / "dump"
    logs_dir = output / "logs"
    archives_dir = output / "intermediate-obj-archives"
    dump_dir.mkdir()
    logs_dir.mkdir()
    archives_dir.mkdir()
    results = []
    archives: dict[str, Path] = {}
    try:
        from scrollunwrap.zarr_source import open_volume

        volume = open_volume(config.zarr_uri, config.level, anon=config.s3_anonymous)
        for plan in planned:
            result = _mesh_one(
                volume,
                plan,
                dump_dir=dump_dir,
                logs_dir=logs_dir,
                cube_mesh_bin=cube_mesh,
                halo=config.halo,
                threshold=config.threshold,
                cube_edge_voxels=config.cube_edge_voxels,
                threads_per_cube=config.threads_per_cube,
                timeout_seconds=config.cube_timeout_seconds,
            )
            results.append(result)
            if result.returncode != 0 or not result.produced_obj:
                raise CubeStageError(
                    f"cube failed; refusing partial weld: {result.cube_id}:"
                    f"rc={result.returncode}:obj={result.produced_obj}"
                )
            archives[result.cube_id] = archive_and_compact_cube(
                dump_dir=dump_dir,
                cube_id=result.cube_id,
                archives_dir=archives_dir,
                tar_bin=tar_bin,
            )
        require_complete(results, planned)
        receipt = output / "CUBE_STAGE_RECEIPT.json"
        write_new_json(
            receipt,
            {
                "schema": "campaignx.scrollfiesta_compact_cube_stage.v1",
                "status": "SUCCEEDED",
                "source": {
                    "uri": config.zarr_uri,
                    "sha256_or_etag": config.source_etag,
                    "level": config.level,
                    "shape_zyx": list(volume.shape),
                    "chunks_zyx": list(volume.chunks),
                },
                "roi_level0_zyx": list(config.roi_level0_zyx),
                "parameters": {
                    "halo": config.halo,
                    "threshold": config.threshold,
                    "resolved_upstream_threshold": resolve_threshold(config.threshold),
                    "cube_edge_voxels": config.cube_edge_voxels,
                    "max_concurrent": 1,
                    "threads_per_cube": config.threads_per_cube,
                    "cube_timeout_seconds": config.cube_timeout_seconds,
                },
                "runtime": {
                    "cube_mesh": file_artifact(cube_mesh),
                    "tar": file_artifact(tar_bin),
                },
                "planned_cubes": [asdict(plan) for plan in planned],
                "results": [
                    {
                        "cube_id": result.cube_id,
                        "origin_zyx": list(result.origin_zyx),
                        "returncode": result.returncode,
                        "produced_obj": result.produced_obj,
                        "obj": file_artifact(result.expected_obj),
                        "intermediate_obj_archive": file_artifact(archives[result.cube_id]),
                        "stdout": file_artifact(result.stdout_path),
                        "stderr": file_artifact(result.stderr_path),
                    }
                    for result in results
                ],
                "complete_fraction": 1.0,
                "intermediates_archived_before_compaction": True,
                "archive_format": "tar.zst",
                "welder_dump_contains_final_step12_only": True,
                "partial_weld_allowed": False,
                "physical_mesh_fusion_performed": False,
                "ink_used": False,
            },
        )
        return receipt
    except Exception as exc:
        failure = output / "CUBE_STAGE_FAILURE.json"
        if not failure.exists():
            write_new_json(
                failure,
                {
                    "schema": "campaignx.scrollfiesta_compact_cube_stage_failure.v1",
                    "status": "FAILED",
                    "error_class": type(exc).__name__,
                    "message": str(exc)[:4096],
                    "planned_cubes": [asdict(plan) for plan in planned],
                    "successful_cube_ids": [
                        result.cube_id
                        for result in results
                        if result.returncode == 0 and result.produced_obj
                    ],
                    "attempted_cube_results": [
                        {
                            "cube_id": result.cube_id,
                            "returncode": result.returncode,
                            "produced_obj": result.produced_obj,
                        }
                        for result in results
                    ],
                    "runtime": {
                        "cube_mesh": file_artifact(cube_mesh),
                        "tar": file_artifact(tar_bin),
                    },
                    "partial_weld_allowed": False,
                    "physical_mesh_fusion_performed": False,
                    "ink_used": False,
                },
            )
        if isinstance(exc, CubeStageError):
            raise
        raise CubeStageError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config, tar_bin = load_spec(args.run_spec.resolve(strict=True))
        receipt = run_compact_stage(config, tar_bin)
    except (CubeStageError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
