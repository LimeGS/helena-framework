"""Immutable ScrollFiesta welded-mesh to TIFXYZ adapter.

The upstream Python driver currently treats some nonzero ``grid_weld`` exits
as warnings.  Helena Framework cannot do that: a nonzero weld exit is a hard failure
even if an OBJ was emitted.  This wrapper also makes all coordinate and binary
boundaries explicit and preserves diagnostics in a unique output directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .component_selection import select_component_nearest_point, write_triangle_obj
from .coordinate_transform import (
    coordinate_matrix,
    load_triangle_obj,
    transform_native_zyx_to_canonical_xyz,
)
from .receipt import build_tifxyz_manifest, file_artifact, write_new_json
from .topology import topology_metrics
from .orientation import orient_with_conflict_quarantine
from .uv_initialization import (
    UvDistortionError,
    load_uv_mapped_obj,
    uv_distortion_metrics,
    write_pca_uv_obj,
    write_tutte_uv_obj,
)


class AdapterError(RuntimeError):
    """Raised when the adapter cannot produce an unambiguous artifact set."""


@dataclass(frozen=True)
class AdapterConfig:
    dump_dir: Path
    output_dir: Path
    grid_weld_bin: Path
    flatboi_bin: Path
    obj2tifxyz_bin: Path
    level: int
    voxel_size_um_xyz: tuple[float, float, float]
    component_seed_level0_xyz: tuple[float, float, float] | None = None
    flatboi_iterations: int = 20
    flatboi_energy: str = "symmetric_dirichlet"
    tifxyz_step_size: int = 20
    maximum_orientation_quarantine_fraction: float = 0.0001
    timeout_seconds: float = 3600.0
    # segmentation.hybrid-scrollfiesta-vc3d@0.1.2 hard_gates.uv_flipped_triangles.
    maximum_uv_flipped_triangle_count: int = 0
    # segmentation.hybrid-scrollfiesta-vc3d@0.1.3 hard_gates
    # .maximum_absolute_stretch_p95.  The frozen 0.1.2 gate is only a *relative*
    # 5% regression against VC3D, so two equally distorted backends both pass.
    # This absolute companion caps the p95 quasi-isometric distortion: at 2.0 a
    # glyph's aspect ratio is already doubled over 5% of the sheet, which is
    # past the point where letter shape survives the flattening.
    maximum_absolute_stretch_p95: float = 2.0


@dataclass(frozen=True)
class AdapterResult:
    output_dir: Path
    source_mesh_obj: Path
    welded_source_mesh_obj: Path
    canonical_mesh_obj: Path
    flattened_mesh_obj: Path
    tifxyz_dir: Path
    tifxyz_manifest: Path
    weld_report: Path
    topology_report: Path
    component_selection_report: Path
    orientation_quarantine_report: Path
    uv_initialization_report: Path
    uv_distortion_report: Path
    adapter_receipt: Path
    topology: dict[str, Any]
    uv_distortion: dict[str, Any]
    elapsed_seconds: float


def _absolute_executable(path: Path, *, label: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise AdapterError(f"{label} must be an explicit absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"{label} cannot be resolved: {path}: {exc}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AdapterError(f"{label} is not an executable file: {resolved}")
    return resolved


def _validate_config(config: AdapterConfig) -> tuple[Path, Path, Path, Path]:
    output = Path(config.output_dir)
    if not output.is_absolute():
        raise AdapterError(f"output_dir must be absolute: {output}")
    if output.exists():
        raise AdapterError(f"immutable output_dir already exists: {output}")
    dump = Path(config.dump_dir)
    if not dump.is_absolute():
        raise AdapterError(f"dump_dir must be absolute: {dump}")
    try:
        dump = dump.resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"dump_dir cannot be resolved: {dump}: {exc}") from exc
    if not dump.is_dir():
        raise AdapterError(f"dump_dir is not a directory: {dump}")
    if config.level < 0:
        raise AdapterError("level must be non-negative")
    if config.flatboi_iterations < 1 or config.tifxyz_step_size < 1:
        raise AdapterError("flatboi_iterations and tifxyz_step_size must be positive")
    if config.timeout_seconds <= 0:
        raise AdapterError("timeout_seconds must be positive")
    if config.maximum_uv_flipped_triangle_count < 0:
        raise AdapterError("maximum_uv_flipped_triangle_count must be non-negative")
    if (
        not np.isfinite(config.maximum_absolute_stretch_p95)
        or config.maximum_absolute_stretch_p95 < 1.0
    ):
        raise AdapterError(
            "maximum_absolute_stretch_p95 must be a finite value of at least 1.0; "
            "1.0 is a perfectly isometric flattening"
        )
    if (
        not np.isfinite(config.maximum_orientation_quarantine_fraction)
        or config.maximum_orientation_quarantine_fraction < 0
        or config.maximum_orientation_quarantine_fraction > 0.0001
    ):
        raise AdapterError(
            "maximum_orientation_quarantine_fraction must be within the "
            "frozen [0, 0.0001] cleanup cap"
        )
    if len(config.voxel_size_um_xyz) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in config.voxel_size_um_xyz
    ):
        raise AdapterError("voxel_size_um_xyz must contain three finite positive values")
    if config.component_seed_level0_xyz is not None and (
        len(config.component_seed_level0_xyz) != 3
        or any(not np.isfinite(value) for value in config.component_seed_level0_xyz)
    ):
        raise AdapterError("component_seed_level0_xyz must contain three finite values")
    return (
        dump,
        _absolute_executable(config.grid_weld_bin, label="grid_weld_bin"),
        _absolute_executable(config.flatboi_bin, label="flatboi_bin"),
        _absolute_executable(config.obj2tifxyz_bin, label="obj2tifxyz_bin"),
    )


def _run_command(
    *,
    name: str,
    argv: list[str],
    log_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not argv or not Path(argv[0]).is_absolute():
        raise AdapterError(f"{name}: argv[0] must be an absolute binary path")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(f"{name} timed out after {timeout_seconds:g}s") from exc
    elapsed = time.monotonic() - started
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    record = {
        "name": name,
        "argv": argv,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": file_artifact(stdout_path),
        "stderr": file_artifact(stderr_path),
    }
    return record


def _require_zero(record: dict[str, Any]) -> None:
    if record["returncode"] != 0:
        raise AdapterError(
            f"{record['name']} failed closed with rc={record['returncode']}; "
            f"see {record['stderr']['uri']}"
        )


def _load_weld_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"invalid or missing grid_weld report {path}: {exc}") from exc
    try:
        audit = report["manifold_audit"]
        required = (
            report["cubes_processed"],
            report["total_unique_verts"],
            report["total_unique_faces"],
            audit["non_manifold"],
            audit["same_dir_pairs"],
        )
    except (KeyError, TypeError) as exc:
        raise AdapterError(f"grid_weld report has an unknown shape: {path}") from exc
    if any(not isinstance(value, int) or value < 0 for value in required):
        raise AdapterError(f"grid_weld report contains invalid counts: {path}")
    return report


def read_tifxyz_coordinates(tifxyz_dir: Path) -> np.ndarray:
    """Read finite XYZ coordinates from a generated TIFXYZ directory."""

    arrays = []
    for name in ("x.tif", "y.tif", "z.tif"):
        path = Path(tifxyz_dir) / name
        try:
            array = np.asarray(tifffile.imread(path), dtype=np.float64)
        except Exception as exc:
            raise AdapterError(f"cannot read {path}: {exc}") from exc
        if array.size == 0 or np.any(~np.isfinite(array)):
            raise AdapterError(f"{path}: empty, NaN, or infinite coordinates are forbidden")
        arrays.append(array)
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise AdapterError("TIFXYZ x/y/z rasters have different shapes")
    return np.stack(arrays, axis=-1)


def _validate_tifxyz(tifxyz_dir: Path) -> None:
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        path = tifxyz_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise AdapterError(f"obj2tifxyz did not produce {path}")
    try:
        metadata = json.loads((tifxyz_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"invalid TIFXYZ metadata: {exc}") from exc
    if metadata.get("format") != "tifxyz":
        raise AdapterError(f"unexpected TIFXYZ meta.format={metadata.get('format')!r}")
    read_tifxyz_coordinates(tifxyz_dir)


def _write_failure(output_dir: Path, exc: Exception, commands: list[dict[str, Any]]) -> None:
    path = output_dir / "ADAPTER_FAILURE.json"
    if path.exists():
        return
    write_new_json(
        path,
        {
            "schema": "campaignx.scrollfiesta_adapter_failure.v1",
            "error_class": type(exc).__name__,
            "message": str(exc),
            "commands": commands,
            "physical_mesh_fusion_performed": False,
            "ink_used": False,
        },
    )


def run_adapter(config: AdapterConfig) -> AdapterResult:
    """Run weld → one coordinate transform → flatten → TIFXYZ, immutably."""

    dump, grid_weld, flatboi, obj2tifxyz = _validate_config(config)
    output = Path(config.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755)
    logs = output / "logs"
    logs.mkdir()
    commands: list[dict[str, Any]] = []
    started = time.monotonic()
    welded_source_obj = output / "welded_source_mesh_zyx.obj"
    source_obj = output / "source_mesh_zyx.obj"
    canonical_obj = output / "canonical_mesh_xyz.obj"
    flatboi_input_obj = output / "canonical_mesh_xyz_flatboi_input.obj"
    flattened_obj = output / "canonical_mesh_xyz_flatboi_input_flatboi.obj"
    tifxyz_dir = output / "tifxyz"

    try:
        record = _run_command(
            name="grid-weld",
            argv=[str(grid_weld), str(dump), str(welded_source_obj)],
            log_dir=logs,
            timeout_seconds=config.timeout_seconds,
        )
        commands.append(record)
        _require_zero(record)
        if not welded_source_obj.is_file() or welded_source_obj.stat().st_size <= 0:
            raise AdapterError("grid_weld exited zero but produced no source mesh")
        weld_report = Path(f"{welded_source_obj}.weld_report.json")
        weld_data = _load_weld_report(weld_report)
        native_mesh = load_triangle_obj(
            welded_source_obj,
            reject_transform_marker=True,
            drop_degenerate_triangles=True,
        )
        source_triangle_count = native_mesh.source_triangle_count
        dropped_degenerate_count = native_mesh.dropped_degenerate_triangle_count
        degenerate_fraction = dropped_degenerate_count / source_triangle_count
        if degenerate_fraction > 0.0001:
            raise AdapterError(
                "native degenerate triangle fraction "
                f"{degenerate_fraction:.8f} exceeds the frozen 0.01% gate"
            )
        if config.component_seed_level0_xyz is None:
            selected_native_mesh = native_mesh
            component_selection = {
                "schema": "campaignx.scrollfiesta_component_selection.v1",
                "policy": "ALL_COMPONENTS_FIXTURE_ONLY",
                "component_count": topology_metrics(
                    native_mesh,
                    voxel_size_um_xyz=config.voxel_size_um_xyz,
                    weld_report=weld_data,
                )["component_count"],
                "physical_mesh_fusion_performed": False,
            }
        else:
            scale = float(2**config.level)
            seed_native_zyx = tuple(
                float(value) / scale
                for value in reversed(config.component_seed_level0_xyz)
            )
            selected_native_mesh, component_selection = select_component_nearest_point(
                native_mesh, seed_native_zyx
            )
        pre_orientation_topology = topology_metrics(
            selected_native_mesh,
            voxel_size_um_xyz=config.voxel_size_um_xyz,
            weld_report=weld_data,
        )
        manifold = weld_data["manifold_audit"]
        whole_weld_selected = len(selected_native_mesh.faces) == len(native_mesh.faces)
        if whole_weld_selected:
            if pre_orientation_topology["non_manifold_edge_count"] != manifold[
                "non_manifold"
            ]:
                raise AdapterError(
                    "grid_weld and adapter disagree on pre-repair non-manifold "
                    f"edge count: {manifold['non_manifold']} != "
                    f"{pre_orientation_topology['non_manifold_edge_count']}"
                )
            if pre_orientation_topology["inconsistent_winding_edge_count"] != manifold[
                "same_dir_pairs"
            ]:
                raise AdapterError(
                    "grid_weld and adapter disagree on pre-repair inconsistent "
                    f"winding edges: {manifold['same_dir_pairs']} != "
                    f"{pre_orientation_topology['inconsistent_winding_edge_count']}"
                )
        remaining_cleanup_fraction = max(
            0.0,
            config.maximum_orientation_quarantine_fraction - degenerate_fraction,
        )
        selected_native_mesh, orientation_quarantine = (
            orient_with_conflict_quarantine(
                selected_native_mesh,
                maximum_quarantined_triangle_fraction=remaining_cleanup_fraction,
            )
        )
        total_removed_count = (
            dropped_degenerate_count
            + orientation_quarantine["quarantined_triangle_count"]
        )
        total_removed_fraction = total_removed_count / source_triangle_count
        if total_removed_fraction > config.maximum_orientation_quarantine_fraction:
            raise AdapterError(
                "combined degenerate and orientation cleanup fraction "
                f"{total_removed_fraction:.8f} exceeds frozen cap "
                f"{config.maximum_orientation_quarantine_fraction:.8f}"
            )
        write_triangle_obj(
            source_obj,
            selected_native_mesh,
            header="campaignx selected native ZYX component; no physical fusion",
        )
        component_selection_report = output / "COMPONENT_SELECTION.json"
        write_new_json(component_selection_report, component_selection)
        orientation_quarantine_report = output / "ORIENTATION_QUARANTINE.json"
        write_new_json(orientation_quarantine_report, orientation_quarantine)
        transform_native_zyx_to_canonical_xyz(
            source_obj,
            canonical_obj,
            level=config.level,
        )
        canonical_mesh = load_triangle_obj(canonical_obj)
        try:
            uv_initialization = write_tutte_uv_obj(canonical_mesh, flatboi_input_obj)
        except ValueError as exc:
            if "boundary" not in str(exc) and "multiple boundary loops" not in str(exc):
                raise
            uv_initialization = write_pca_uv_obj(canonical_mesh, flatboi_input_obj)
            uv_initialization["fallback_reason"] = str(exc)
            uv_initialization["topological_disk_required"] = False
        uv_initialization_report = output / "UV_INITIALIZATION.json"
        write_new_json(uv_initialization_report, uv_initialization)
        topology = topology_metrics(
            canonical_mesh,
            voxel_size_um_xyz=config.voxel_size_um_xyz,
            weld_report=weld_data,
        )
        topology["degenerate_triangle_fraction"] = degenerate_fraction
        deterministic_cleanup = {
            "operation": "BOUNDED_DEGENERATE_DROP_AND_ORIENTATION_QUARANTINE",
            "source_triangle_count": source_triangle_count,
            "dropped_degenerate_triangle_count": dropped_degenerate_count,
            "quarantined_orientation_triangle_count": orientation_quarantine[
                "quarantined_triangle_count"
            ],
            "total_removed_triangle_count": total_removed_count,
            "total_removed_fraction": total_removed_fraction,
            "gate_max_fraction": 0.0001,
            "orientation_quarantine": orientation_quarantine,
        }
        if topology["non_manifold_edge_count"] or topology[
            "inconsistent_winding_edge_count"
        ] or not topology["orientable"]:
            raise AdapterError(
                "post-cleanup topology still fails the frozen manifold and "
                "orientation gate"
            )

        record = _run_command(
            name="flatboi",
            argv=[
                str(flatboi),
                str(flatboi_input_obj),
                str(config.flatboi_iterations),
                config.flatboi_energy,
            ],
            log_dir=logs,
            timeout_seconds=config.timeout_seconds,
        )
        commands.append(record)
        _require_zero(record)
        if not flattened_obj.is_file() or not any(
            line.startswith("vt ")
            for line in flattened_obj.read_text(encoding="utf-8").splitlines()
        ):
            raise AdapterError("flatboi exited zero but produced no UV-mapped OBJ")

        # The presence of ``vt`` records is not a distortion measurement.  The
        # profile declares hard gates on post-SLIM flips and stretch, so the
        # optimized OBJ is measured here and the gates fail closed before any
        # TIFXYZ raster can inherit an unmeasured flattening.
        try:
            flattened_vertices, flattened_faces, flattened_uv = load_uv_mapped_obj(
                flattened_obj
            )
            uv_distortion = uv_distortion_metrics(
                flattened_vertices, flattened_faces, flattened_uv
            )
        except UvDistortionError as exc:
            raise AdapterError(f"flattened OBJ cannot be measured: {exc}") from exc
        uv_distortion["gates"] = {
            "maximum_uv_flipped_triangle_count": int(
                config.maximum_uv_flipped_triangle_count
            ),
            "maximum_absolute_stretch_p95": float(config.maximum_absolute_stretch_p95),
        }
        uv_distortion_report = output / "UV_DISTORTION.json"
        write_new_json(uv_distortion_report, uv_distortion)
        if uv_distortion["uv_degenerate_triangle_count"]:
            raise AdapterError(
                "flatboi produced "
                f"{uv_distortion['uv_degenerate_triangle_count']} UV-degenerate "
                "triangles; a collapsed UV triangle has no measurable stretch"
            )
        if (
            uv_distortion["uv_flipped_triangle_count"]
            > config.maximum_uv_flipped_triangle_count
        ):
            raise AdapterError(
                "post-SLIM UV flipped triangle count "
                f"{uv_distortion['uv_flipped_triangle_count']} exceeds the frozen "
                f"gate {config.maximum_uv_flipped_triangle_count}"
            )
        if uv_distortion["stretch_p95"] > config.maximum_absolute_stretch_p95:
            raise AdapterError(
                "post-SLIM UV stretch p95 "
                f"{uv_distortion['stretch_p95']:.6f} exceeds the absolute gate "
                f"{config.maximum_absolute_stretch_p95:.6f}"
            )
        # The distortion summary must ride the surface artifact across the stage
        # boundary; a consumer that only receives topology counts cannot tell a
        # 1.05 flattening from a 3.0 one.
        for field in (
            "uv_flipped_triangle_count",
            "uv_degenerate_triangle_count",
            "stretch_p50",
            "stretch_p95",
            "stretch_max",
        ):
            topology[field] = uv_distortion[field]
        if tifxyz_dir.exists():
            raise AdapterError(f"TIFXYZ output unexpectedly exists: {tifxyz_dir}")
        record = _run_command(
            name="obj2tifxyz",
            argv=[
                str(obj2tifxyz),
                str(flattened_obj),
                str(tifxyz_dir),
                str(config.tifxyz_step_size),
            ],
            log_dir=logs,
            timeout_seconds=config.timeout_seconds,
        )
        commands.append(record)
        _require_zero(record)
        _validate_tifxyz(tifxyz_dir)

        tifxyz_manifest = output / "TIFXYZ_MANIFEST.json"
        build_tifxyz_manifest(tifxyz_dir, tifxyz_manifest)
        topology_report = output / "TOPOLOGY_REPORT.json"
        write_new_json(
            topology_report,
            {
                "schema": "campaignx.scrollfiesta_topology_report.v1",
                "validation_state": "PROVISIONAL",
                "metrics": topology,
                "uv_distortion": uv_distortion,
                "deterministic_cleanup": deterministic_cleanup,
                "component_selection": component_selection,
                "weld_report": weld_data,
                "self_intersection_audit_completed": False,
                "ct_validation_completed": False,
            },
        )
        elapsed = time.monotonic() - started
        adapter_receipt = output / "ADAPTER_RESULT.json"
        write_new_json(
            adapter_receipt,
            {
                "schema": "campaignx.scrollfiesta_adapter_result.v1",
                "status": "SUCCEEDED",
                "coordinate_transform": {
                    "source_order": "ZYX",
                    "canonical_order": "XYZ",
                    "winding_flipped": True,
                    "application_count": 1,
                    "matrix": coordinate_matrix(level=config.level),
                },
                "commands": commands,
                "artifacts": {
                    "source_mesh_obj": file_artifact(source_obj),
                    "welded_source_mesh_obj": file_artifact(welded_source_obj),
                    "canonical_mesh_obj": file_artifact(canonical_obj),
                    "flatboi_input_obj": file_artifact(flatboi_input_obj),
                    "flattened_mesh_obj": file_artifact(flattened_obj),
                    "tifxyz_manifest": file_artifact(tifxyz_manifest),
                    "weld_report": file_artifact(weld_report),
                    "topology_report": file_artifact(topology_report),
                    "component_selection_report": file_artifact(
                        component_selection_report
                    ),
                    "orientation_quarantine_report": file_artifact(
                        orientation_quarantine_report
                    ),
                    "uv_initialization_report": file_artifact(
                        uv_initialization_report
                    ),
                    "uv_distortion_report": file_artifact(uv_distortion_report),
                },
                "topology": topology,
                "uv_distortion": uv_distortion,
                "deterministic_cleanup": deterministic_cleanup,
                "elapsed_seconds": elapsed,
                "physical_mesh_fusion_performed": False,
                "ink_used": False,
            },
        )
        return AdapterResult(
            output_dir=output,
            source_mesh_obj=source_obj,
            welded_source_mesh_obj=welded_source_obj,
            canonical_mesh_obj=canonical_obj,
            flattened_mesh_obj=flattened_obj,
            tifxyz_dir=tifxyz_dir,
            tifxyz_manifest=tifxyz_manifest,
            weld_report=weld_report,
            topology_report=topology_report,
            component_selection_report=component_selection_report,
            orientation_quarantine_report=orientation_quarantine_report,
            uv_initialization_report=uv_initialization_report,
            uv_distortion_report=uv_distortion_report,
            adapter_receipt=adapter_receipt,
            topology=topology,
            uv_distortion=uv_distortion,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        _write_failure(output, exc, commands)
        if isinstance(exc, AdapterError):
            raise
        raise AdapterError(str(exc)) from exc
