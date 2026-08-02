"""Fail-closed, immutable ScrollFiesta cube-meshing stage.

The upstream driver intentionally welds partial output when individual cubes
fail.  That is useful for exploration, but it is not acceptable for a locked
Helena Framework A/B.  This stage streams every planned m7 cube, requires every
``cube_mesh`` invocation to succeed and emit its documented OBJ, and only then
hands the complete dump directory to the immutable adapter/welder.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .receipt import file_artifact, write_new_json


class CubeStageError(RuntimeError):
    """Raised when the cube stage cannot produce a complete dump set."""


@dataclass(frozen=True)
class CubePlan:
    cube_id: str
    origin_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class CubeResult:
    cube_id: str
    origin_zyx: tuple[int, int, int]
    returncode: int
    produced_obj: bool
    expected_obj: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class CubeStageConfig:
    zarr_uri: str
    source_etag: str
    level: int
    roi_level0_zyx: tuple[int, int, int, int, int, int]
    output_dir: Path
    cube_mesh_bin: Path
    halo: int = 13
    threshold: str | None = None
    cube_edge_voxels: int = 128
    max_concurrent: int = 8
    threads_per_cube: int = 1
    cube_timeout_seconds: float = 600.0
    s3_anonymous: bool = True


def resolve_threshold(value: str | None) -> str | None:
    """Translate Helena Framework's locked semantic aliases to upstream syntax."""

    if value == "nonzero":
        return ">=1"
    return value


def plan_cubes(
    roi_level0_zyx: tuple[int, int, int, int, int, int],
    *,
    level: int,
    cube_edge_voxels: int = 128,
) -> list[CubePlan]:
    """Plan complete, aligned cube ownership for canonical min/max ROI."""

    if level < 0 or cube_edge_voxels <= 0:
        raise CubeStageError("level must be non-negative and cube edge positive")
    if len(roi_level0_zyx) != 6:
        raise CubeStageError("ROI must be [z0,y0,x0,z1,y1,x1]")
    z0, y0, x0, z1, y1, x1 = roi_level0_zyx
    if any(lower < 0 or lower >= upper for lower, upper in zip((z0, y0, x0), (z1, y1, x1))):
        raise CubeStageError("ROI bounds must be finite positive extents")
    stride = cube_edge_voxels * (2**level)
    if any(value % stride for value in roi_level0_zyx):
        raise CubeStageError(f"ROI bounds must align to the level-{level} cube stride {stride}")
    plans: list[CubePlan] = []
    for oz0 in range(z0, z1, stride):
        for oy0 in range(y0, y1, stride):
            for ox0 in range(x0, x1, stride):
                oz, oy, ox = oz0 // (2**level), oy0 // (2**level), ox0 // (2**level)
                # This exact string is parsed by ScrollFiesta's halo/dump C
                # code; changing it silently moves cubes in world space.
                cube_id = f"z{oz:05d}_y{oy:05d}_x{ox:05d}"
                plans.append(CubePlan(cube_id=cube_id, origin_zyx=(oz, oy, ox)))
    if not plans:
        raise CubeStageError("ROI produced no cubes")
    return plans


def _absolute_executable(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise CubeStageError(f"cube_mesh_bin must be absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CubeStageError(f"cannot resolve cube_mesh_bin {path}: {exc}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CubeStageError(f"cube_mesh_bin is not executable: {resolved}")
    return resolved


def _expected_obj(dump_dir: Path, cube_id: str) -> Path:
    return dump_dir / cube_id / f"{cube_id}_step12_final" / f"{cube_id}_step12_final_all.obj"


def cube_environment(threads_per_cube: int) -> dict[str, str]:
    """Return the bounded subprocess environment expected by ScrollFiesta.

    ScrollFiesta's public ``cube_mesh`` reads ``VESUVIUS_THREADS`` and then
    calls ``omp_set_num_threads``.  ``OMP_NUM_THREADS`` alone is insufficient;
    keeping both variables identical also bounds any dependency-level OpenMP
    region that reads the standard variable directly.
    """

    if threads_per_cube < 1:
        raise CubeStageError("threads_per_cube must be positive")
    environment = os.environ.copy()
    value = str(threads_per_cube)
    environment["VESUVIUS_THREADS"] = value
    environment["OMP_NUM_THREADS"] = value
    return environment


def require_complete(results: list[CubeResult], planned: list[CubePlan]) -> None:
    """Fail unless results are one-to-one, successful and materialized."""

    expected = {plan.cube_id for plan in planned}
    observed = [result.cube_id for result in results]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise CubeStageError("cube results do not correspond one-to-one with the frozen plan")
    failures = [
        result
        for result in results
        if result.returncode != 0 or not result.produced_obj
    ]
    if failures:
        compact = ", ".join(
            f"{result.cube_id}:rc={result.returncode}:obj={result.produced_obj}"
            for result in failures[:8]
        )
        raise CubeStageError(f"incomplete cube set; refusing partial weld: {compact}")


def _mesh_one(
    volume: Any,
    plan: CubePlan,
    *,
    dump_dir: Path,
    logs_dir: Path,
    cube_mesh_bin: Path,
    halo: int,
    threshold: str | None,
    cube_edge_voxels: int,
    threads_per_cube: int,
    timeout_seconds: float,
) -> CubeResult:
    from scrollunwrap.zarr_source import apply_threshold, read_padded_cube

    oz, oy, ox = plan.origin_zyx
    padded = cube_edge_voxels + 2 * halo
    argv = [
        str(cube_mesh_bin),
        "--stdin-raw",
        str(padded),
        str(oz),
        str(oy),
        str(ox),
        "--halo",
        str(halo),
        "--dump-obj",
        str(dump_dir),
    ]
    environment = cube_environment(threads_per_cube)
    stdout_path = logs_dir / f"{plan.cube_id}.stdout.log"
    stderr_path = logs_dir / f"{plan.cube_id}.stderr.log"
    try:
        cube = read_padded_cube(
            volume, oz, oy, ox, size=cube_edge_voxels, halo=halo
        )
        mask = apply_threshold(cube, resolve_threshold(threshold))
        completed = subprocess.run(
            argv,
            input=mask.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env=environment,
            check=False,
        )
        stdout_path.write_bytes(completed.stdout or b"")
        stderr_path.write_bytes(completed.stderr or b"")
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_bytes(exc.stdout or b"")
        stderr_path.write_bytes((exc.stderr or b"") + b"\nTIMEOUT\n")
        returncode = -9
    except Exception as exc:  # Preserve a deterministic diagnostic; fail later.
        stdout_path.write_bytes(b"")
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        returncode = -1
    expected = _expected_obj(dump_dir, plan.cube_id)
    return CubeResult(
        cube_id=plan.cube_id,
        origin_zyx=plan.origin_zyx,
        returncode=returncode,
        produced_obj=expected.is_file() and expected.stat().st_size > 0,
        expected_obj=expected,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def run_cube_stage(config: CubeStageConfig) -> Path:
    """Stream and mesh every frozen cube, returning the immutable receipt."""

    output = Path(config.output_dir)
    if not output.is_absolute():
        raise CubeStageError(f"output_dir must be absolute: {output}")
    if output.exists():
        raise CubeStageError(f"immutable output_dir already exists: {output}")
    if config.halo < 0 or config.max_concurrent < 1 or config.threads_per_cube < 1:
        raise CubeStageError("halo/concurrency/thread values are invalid")
    if config.cube_timeout_seconds <= 0:
        raise CubeStageError("cube timeout must be positive")
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
    dump_dir.mkdir()
    logs_dir.mkdir()
    try:
        from scrollunwrap.zarr_source import open_volume

        volume = open_volume(
            config.zarr_uri, config.level, anon=config.s3_anonymous
        )
        results: list[CubeResult] = []
        with ThreadPoolExecutor(max_workers=config.max_concurrent) as executor:
            futures = [
                executor.submit(
                    _mesh_one,
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
                for plan in planned
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: result.cube_id)
        require_complete(results, planned)
        receipt = output / "CUBE_STAGE_RECEIPT.json"
        write_new_json(
            receipt,
            {
                "schema": "campaignx.scrollfiesta_cube_stage.v1",
                "status": "SUCCEEDED",
                "source": {
                    "uri": config.zarr_uri,
                    "sha256_or_etag": config.source_etag,
                    "level": config.level,
                    "shape_zyx": list(volume.shape),
                    "chunks_zyx": list(volume.chunks),
                },
                "roi_level0_zyx": list(config.roi_level0_zyx),
                "scrollfiesta_bbox_cli_zyx_pairs": [
                    config.roi_level0_zyx[0],
                    config.roi_level0_zyx[3],
                    config.roi_level0_zyx[1],
                    config.roi_level0_zyx[4],
                    config.roi_level0_zyx[2],
                    config.roi_level0_zyx[5],
                ],
                "parameters": {
                    "halo": config.halo,
                    "threshold": config.threshold,
                    "resolved_upstream_threshold": resolve_threshold(config.threshold),
                    "cube_edge_voxels": config.cube_edge_voxels,
                    "max_concurrent": config.max_concurrent,
                    "threads_per_cube": config.threads_per_cube,
                    "cube_timeout_seconds": config.cube_timeout_seconds,
                },
                "planned_cubes": [asdict(plan) for plan in planned],
                "results": [
                    {
                        "cube_id": result.cube_id,
                        "origin_zyx": list(result.origin_zyx),
                        "returncode": result.returncode,
                        "produced_obj": result.produced_obj,
                        "obj": file_artifact(result.expected_obj),
                        "stdout": file_artifact(result.stdout_path),
                        "stderr": file_artifact(result.stderr_path),
                    }
                    for result in results
                ],
                "complete_fraction": 1.0,
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
                    "schema": "campaignx.scrollfiesta_cube_stage_failure.v1",
                    "status": "FAILED",
                    "error_class": type(exc).__name__,
                    "message": str(exc)[:4096],
                    "planned_cubes": [asdict(plan) for plan in planned],
                    "partial_weld_allowed": False,
                    "physical_mesh_fusion_performed": False,
                    "ink_used": False,
                },
            )
        if isinstance(exc, CubeStageError):
            raise
        raise CubeStageError(str(exc)) from exc
