#!/usr/bin/env python3
"""Run the immutable Helena Framework ScrollFiesta OBJ→TIFXYZ adapter.

The input is a locked JSON request.  The script emits both
``campaignx.surface_artifact.v2`` and
``campaignx.segmentation_backend_run.v1`` documents and validates them before
they become durable.  It never physically fuses ScrollFiesta and VC3D meshes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BACKENDS = Path(__file__).resolve().parents[1] / "backends"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKENDS))

from framework.contracts.hybrid_surface_contracts import (  # noqa: E402
    HybridContractValidationError,
    validate_hybrid_contract,
)
from scrollfiesta.adapter import AdapterConfig, AdapterError, run_adapter  # noqa: E402
from scrollfiesta.coordinate_transform import coordinate_matrix  # noqa: E402
from scrollfiesta.receipt import (  # noqa: E402
    file_artifact,
    sha256_file,
    write_new_json,
)


REQUEST_SCHEMA = "campaignx.scrollfiesta_adapter_request.v1"
REQUIRED_KEYS = {
    "schema",
    "run_id",
    "surface_id",
    "sample_id",
    "backend_profile",
    "campaignx_git_commit",
    "scrollfiesta_repository",
    "scrollfiesta_git_commit",
    "volume_cartographer_repository",
    "volume_cartographer_git_commit",
    "runtime",
    "ct",
    "surface_prediction",
    "roi_level0_zyx",
    "level",
    "voxel_size_um_xyz",
    "handedness",
    "dump_dir",
    "output_dir",
    "binaries",
    "cpu_threads",
    "flatboi_iterations",
    "flatboi_energy",
    "tifxyz_step_size",
    "timeout_seconds",
}
OPTIONAL_KEYS = {
    "component_seed_level0_xyz",
    "maximum_orientation_quarantine_fraction",
    "maximum_uv_flipped_triangle_count",
    "maximum_absolute_stretch_p95",
}


class RequestError(ValueError):
    """Raised when a run request is incomplete or ambiguous."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_request(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RequestError(f"cannot load request {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RequestError("request must be a JSON object")
    keys = set(value)
    if not REQUIRED_KEYS.issubset(keys) or not keys.issubset(REQUIRED_KEYS | OPTIONAL_KEYS):
        missing = sorted(REQUIRED_KEYS - keys)
        unknown = sorted(keys - REQUIRED_KEYS - OPTIONAL_KEYS)
        raise RequestError(f"request keys mismatch: missing={missing}, unknown={unknown}")
    if value["schema"] != REQUEST_SCHEMA:
        raise RequestError(f"expected schema {REQUEST_SCHEMA!r}")
    for key in (
        "campaignx_git_commit",
        "scrollfiesta_git_commit",
        "volume_cartographer_git_commit",
    ):
        item = value[key]
        if not isinstance(item, str) or len(item) != 40 or any(
            char not in "0123456789abcdef" for char in item
        ):
            raise RequestError(f"{key} must be a full lowercase 40-hex commit")
    roi = value["roi_level0_zyx"]
    if (
        not isinstance(roi, list)
        or len(roi) != 6
        or any(not isinstance(item, int) or item < 0 for item in roi)
        or any(roi[index] >= roi[index + 3] for index in range(3))
    ):
        raise RequestError("roi_level0_zyx must be [z0,y0,x0,z1,y1,x1] with positive extents")
    binaries = value["binaries"]
    if not isinstance(binaries, dict) or set(binaries) != {
        "grid_weld",
        "flatboi",
        "obj2tifxyz",
    }:
        raise RequestError("binaries must contain exactly grid_weld, flatboi, obj2tifxyz")
    if value["handedness"] not in {"LEFT_HANDED", "RIGHT_HANDED"}:
        raise RequestError("handedness must be LEFT_HANDED or RIGHT_HANDED")
    seed = value.get("component_seed_level0_xyz")
    if seed is not None and (
        not isinstance(seed, list)
        or len(seed) != 3
        or any(not isinstance(item, (int, float)) for item in seed)
    ):
        raise RequestError("component_seed_level0_xyz must contain X/Y/Z values")
    if not isinstance(value["cpu_threads"], int) or value["cpu_threads"] < 1:
        raise RequestError("cpu_threads must be positive")
    return value


def _adapter_config(request: dict[str, Any]) -> AdapterConfig:
    binaries = request["binaries"]
    voxel = request["voxel_size_um_xyz"]
    if not isinstance(voxel, list) or len(voxel) != 3:
        raise RequestError("voxel_size_um_xyz must contain X/Y/Z values")
    return AdapterConfig(
        dump_dir=Path(request["dump_dir"]),
        output_dir=Path(request["output_dir"]),
        grid_weld_bin=Path(binaries["grid_weld"]),
        flatboi_bin=Path(binaries["flatboi"]),
        obj2tifxyz_bin=Path(binaries["obj2tifxyz"]),
        level=int(request["level"]),
        voxel_size_um_xyz=tuple(float(value) for value in voxel),
        component_seed_level0_xyz=(
            tuple(float(value) for value in request["component_seed_level0_xyz"])
            if request.get("component_seed_level0_xyz") is not None
            else None
        ),
        flatboi_iterations=int(request["flatboi_iterations"]),
        flatboi_energy=str(request["flatboi_energy"]),
        tifxyz_step_size=int(request["tifxyz_step_size"]),
        maximum_orientation_quarantine_fraction=float(
            request.get("maximum_orientation_quarantine_fraction", 0.0001)
        ),
        # Profile hard gates on the flattening result.  Absent keys keep the
        # adapter defaults, which mirror
        # segmentation.hybrid-scrollfiesta-vc3d@0.1.3; they are never disabled.
        maximum_uv_flipped_triangle_count=int(
            request.get("maximum_uv_flipped_triangle_count", 0)
        ),
        maximum_absolute_stretch_p95=float(
            request.get("maximum_absolute_stretch_p95", 2.0)
        ),
        timeout_seconds=float(request["timeout_seconds"]),
    )


def _parameter_rows(request: dict[str, Any]) -> list[dict[str, str]]:
    values = {
        "level": request["level"],
        "flatboi_iterations": request["flatboi_iterations"],
        "flatboi_energy": request["flatboi_energy"],
        "tifxyz_step_size": request["tifxyz_step_size"],
        "timeout_seconds": request["timeout_seconds"],
        "coordinate_transform_application_count": 1,
        "physical_mesh_fusion_performed": False,
        "component_seed_level0_xyz": request.get("component_seed_level0_xyz"),
        "maximum_orientation_quarantine_fraction": request.get(
            "maximum_orientation_quarantine_fraction", 0.0001
        ),
        "maximum_uv_flipped_triangle_count": request.get(
            "maximum_uv_flipped_triangle_count", 0
        ),
        "maximum_absolute_stretch_p95": request.get(
            "maximum_absolute_stretch_p95", 2.0
        ),
    }
    rows = []
    for name, value in sorted(values.items()):
        encoded = canonical_json(value)
        rows.append(
            {"name": name, "value_json": encoded, "value_sha256": sha256_text(encoded)}
        )
    return rows


def _base_run_receipt(
    request: dict[str, Any],
    *,
    started_at: str,
    finished_at: str,
    argv: list[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "campaignx.segmentation_backend_run.v1",
        "run_id": request["run_id"],
        "status": "FAILED",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "campaignx_git_commit": request["campaignx_git_commit"],
        "backend": {
            "family": "scrollfiesta",
            "profile": request["backend_profile"],
            "repository": request["scrollfiesta_repository"],
            "git_commit": request["scrollfiesta_git_commit"],
            "auxiliary_tools": [
                {
                    "name": "volume-cartographer",
                    "repository": request["volume_cartographer_repository"],
                    "git_commit": request["volume_cartographer_git_commit"],
                }
            ],
        },
        "runtime": request["runtime"],
        "command": {
            "argv": argv,
            "argv_sha256": sha256_text(canonical_json(argv)),
            "working_directory": str(Path.cwd().resolve()),
        },
        "environment_non_sensitive": [
            {"name": "OMP_NUM_THREADS", "value": str(request["cpu_threads"])}
        ],
        "resources": {
            "cpu_threads": request["cpu_threads"],
            "peak_ram_bytes": 0,
            "gpu_devices": [],
            "elapsed_seconds": max(0.0, elapsed_seconds),
            "scratch_bytes": 0,
        },
        "inputs": {
            "sample_id": request["sample_id"],
            "ct": request["ct"],
            "surface_prediction": request["surface_prediction"],
            "roi_level0_zyx": request["roi_level0_zyx"],
            "coordinate_order": "ZYX",
        },
        "parameters": _parameter_rows(request),
        "durable_outputs": [],
        "surface_artifact": None,
        "error": None,
        "ink_used": False,
    }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def execute_request(request_path: Path, *, argv: list[str] | None = None) -> Path:
    """Execute one locked request and return the validated run receipt path."""

    request_path = Path(request_path).resolve(strict=True)
    request = _load_request(request_path)
    output = Path(request["output_dir"])
    command_argv = argv or [str(Path(__file__).resolve()), "--run-spec", str(request_path)]
    started_at = utc_now()
    started_monotonic = datetime.now(UTC)
    os.environ["OMP_NUM_THREADS"] = str(request["cpu_threads"])

    try:
        result = run_adapter(_adapter_config(request))
    except (AdapterError, RequestError) as exc:
        if output.is_dir():
            finished_at = utc_now()
            elapsed = (datetime.now(UTC) - started_monotonic).total_seconds()
            failure = output / "ADAPTER_FAILURE.json"
            if failure.is_file() and failure.stat().st_size > 0:
                receipt = _base_run_receipt(
                    request,
                    started_at=started_at,
                    finished_at=finished_at,
                    argv=command_argv,
                    elapsed_seconds=elapsed,
                )
                receipt["resources"]["scratch_bytes"] = _directory_bytes(output)
                receipt["error"] = {
                    "class": type(exc).__name__,
                    "message": str(exc)[:4096],
                    "log_uri": failure.resolve().as_uri(),
                    "log_sha256": sha256_file(failure),
                }
                validate_hybrid_contract(
                    receipt, expected_contract="campaignx.segmentation_backend_run.v1"
                )
                write_new_json(output / "RUN_RECEIPT.json", receipt)
        raise

    adapter_artifact = file_artifact(result.adapter_receipt)
    tifxyz_artifact = file_artifact(result.tifxyz_manifest)
    surface = {
        "schema": "campaignx.surface_artifact.v2",
        "surface_id": request["surface_id"],
        "created_at_utc": started_at,
        "backend_family": "scrollfiesta",
        "backend_profile": request["backend_profile"],
        "sample_id": request["sample_id"],
        "source": {
            "ct_uri": request["ct"]["uri"],
            "ct_sha256_or_etag": request["ct"]["sha256_or_etag"],
            "surface_prediction_uri": request["surface_prediction"]["uri"],
            "surface_prediction_sha256_or_etag": request["surface_prediction"][
                "sha256_or_etag"
            ],
            "level": request["level"],
            "voxel_size_um": request["voxel_size_um_xyz"],
            "handedness": request["handedness"],
        },
        "roi_level0_zyx": request["roi_level0_zyx"],
        "coordinate_transform": {
            "source_order": "ZYX",
            "canonical_order": "XYZ",
            "winding_flipped": True,
            "matrix": coordinate_matrix(level=request["level"]),
        },
        "artifacts": {
            "tifxyz": tifxyz_artifact,
            "source_mesh_obj": file_artifact(result.source_mesh_obj),
            "canonical_mesh_obj": file_artifact(result.canonical_mesh_obj),
        },
        "metrics": result.topology,
        "validation_state": "PROVISIONAL",
        "receipt_uri": adapter_artifact["uri"],
        "receipt_sha256": adapter_artifact["sha256"],
        "ink_used": False,
    }
    validate_hybrid_contract(surface, expected_contract="campaignx.surface_artifact.v2")
    surface_path = result.output_dir / "SURFACE_ARTIFACT.json"
    write_new_json(surface_path, surface)

    finished_at = utc_now()
    receipt = _base_run_receipt(
        request,
        started_at=started_at,
        finished_at=finished_at,
        argv=command_argv,
        elapsed_seconds=result.elapsed_seconds,
    )
    receipt["status"] = "SUCCEEDED"
    receipt["resources"]["scratch_bytes"] = _directory_bytes(result.output_dir)
    receipt["durable_outputs"] = [
        file_artifact(result.source_mesh_obj, role="SOURCE_MESH_OBJ"),
        file_artifact(result.welded_source_mesh_obj, role="WELDED_SOURCE_MESH_OBJ"),
        file_artifact(result.canonical_mesh_obj, role="CANONICAL_MESH_OBJ"),
        file_artifact(result.tifxyz_manifest, role="TIFXYZ"),
        file_artifact(result.weld_report, role="WELD_REPORT"),
        file_artifact(result.topology_report, role="TOPOLOGY_REPORT"),
        file_artifact(
            result.component_selection_report, role="COMPONENT_SELECTION_REPORT"
        ),
        file_artifact(result.uv_initialization_report, role="UV_INITIALIZATION_REPORT"),
        file_artifact(result.uv_distortion_report, role="UV_DISTORTION_REPORT"),
        file_artifact(result.adapter_receipt, role="RECEIPT"),
    ]
    receipt["surface_artifact"] = {
        "surface_id": request["surface_id"],
        "uri": surface_path.resolve().as_uri(),
        "sha256": sha256_file(surface_path),
    }
    receipt["error"] = None
    validate_hybrid_contract(
        receipt, expected_contract="campaignx.segmentation_backend_run.v1"
    )
    receipt_path = result.output_dir / "RUN_RECEIPT.json"
    write_new_json(receipt_path, receipt)
    return receipt_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = execute_request(args.run_spec)
    except (AdapterError, RequestError, HybridContractValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
