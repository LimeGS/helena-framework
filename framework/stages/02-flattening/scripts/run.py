#!/usr/bin/env python3
"""Run the resumable, source-locked Helena Framework Phase 3 field pipeline.

P3.0--P3.2 lock the exact PHerc1447 surface and materialize the minimal CT
region on remote compute.  ``record-local-render`` records a real CT render as
a separately scoped, auditable sidecar: it is deliberately *not* a P3.4/P3.5
spiral-fit result.  Later stages refuse to run until their field
implementation and its tests exist; they never fall back to Phase-2 shadow
constraints as if those were new CT evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402

PHASE3 = ROOT / "phase3"
CONFIG_PATH = PHASE3 / "configs" / "fast_v1.json"
STATE_PATH = PHASE3 / "RUN_STATE.json"
READY_PATH = PHASE3 / "PHASE3_READINESS.json"
STAGES = (
    "P3_0_PREFLIGHT",
    "P3_1_SOURCE_LOCK",
    "P3_2_MATERIALIZE",
    "P3_3_R6_CONSTRAINTS",
    "P3_4_BASELINE_FIT",
    "P3_5_R6_FIT",
    "P3_6_COMPARISON",
    "P3_7_TRANSFER_SMOKE",
    "P3_8_PACKAGE",
    "P3_9_CLOSEOUT",
)
TERMINAL_STAGE_STATES = frozenset({"PASSED", "FAILED_GATE", "BLOCKED", "SKIPPED"})


class Phase3Error(RuntimeError):
    """A deterministic gate or source-lock failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase3Error(f"cannot read {display_path(path)}: {error}") from error
    if not isinstance(value, dict):
        raise Phase3Error(f"{display_path(path)} must contain one JSON object")
    return value


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config.get("config_kind") != "campaign_x_phase3_fast_v1":
        raise Phase3Error("unexpected phase3 config kind")
    if config.get("artifact_scope") != "R6_LOCAL_FUNCTIONAL_ONLY":
        raise Phase3Error("Phase 3 must preserve R6_LOCAL_FUNCTIONAL_ONLY")
    if not isinstance(config.get("primary_target"), dict):
        raise Phase3Error("primary_target is required")
    return config


def default_state() -> dict[str, Any]:
    return {
        "artifact_scope": "R6_LOCAL_FUNCTIONAL_ONLY",
        "kind": "campaign_x_phase3_run_state_v1",
        "stages": {stage: {"status": "PENDING"} for stage in STAGES},
        "updated_at_utc": utc_now(),
    }


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return default_state()
    state = read_json(STATE_PATH)
    if state.get("artifact_scope") != "R6_LOCAL_FUNCTIONAL_ONLY":
        raise Phase3Error("RUN_STATE artifact scope differs")
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise Phase3Error("RUN_STATE stages missing")
    for stage in STAGES:
        stages.setdefault(stage, {"status": "PENDING"})
    return state


def write_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json(STATE_PATH, state)


def set_stage(state: dict[str, Any], stage: str, status: str, **extra: Any) -> None:
    if stage not in STAGES:
        raise Phase3Error(f"unknown stage {stage}")
    if status not in {"PENDING", "RUNNING", *TERMINAL_STAGE_STATES}:
        raise Phase3Error(f"unknown stage status {status}")
    record = {"status": status, **extra, "updated_at_utc": utc_now()}
    state["stages"][stage] = record
    write_state(state)


def require_stage(state: dict[str, Any], stage: str) -> None:
    if state["stages"].get(stage, {}).get("status") != "PASSED":
        raise Phase3Error(f"{stage} must be PASSED first")


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def sanitized_vast_receipt(instance_id: int) -> dict[str, Any]:
    """Read Vast state but persist only price/capacity fields, never tokens."""

    try:
        result = subprocess.run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise Phase3Error(f"cannot obtain sanitized Vast cost receipt: {error}") from error
    instance = raw.get("instance") if isinstance(raw.get("instance"), dict) else {}
    duration_seconds = float(raw.get("duration", 0.0))
    hourly_rate = float(instance.get("totalHour", raw.get("dph_total", 0.0)))
    if raw.get("actual_status") != "running" or not math.isfinite(hourly_rate) or hourly_rate <= 0:
        raise Phase3Error("Vast instance is not a billable running instance")
    return {
        "instance_id": int(raw.get("id", instance_id)),
        "status": raw.get("actual_status"),
        "gpu_name": raw.get("gpu_name"),
        "gpu_ram_mib": raw.get("gpu_ram"),
        "disk_total_gib": raw.get("disk_space"),
        "disk_used_gib": raw.get("disk_usage"),
        "hourly_rate_usd": hourly_rate,
        "runtime_hours_observed": duration_seconds / 3600.0,
        "estimated_compute_cost_to_date_usd": hourly_rate * duration_seconds / 3600.0,
        "internet_down_cost_per_tb_usd": raw.get("internet_down_cost_per_tb"),
        "internet_up_cost_per_tb_usd": raw.get("internet_up_cost_per_tb"),
        "geolocation": raw.get("geolocation"),
    }


def local_environment() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    packages: dict[str, str | None] = {}
    for module in ("numpy", "tifffile", "requests", "zarr"):
        try:
            imported = __import__(module)
            packages[module] = getattr(imported, "__version__", None)
        except ImportError:
            packages[module] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_before_phase3": bool(git_value("status", "--porcelain")),
        "disk_available_bytes": disk.free,
        "packages": packages,
    }


def stage_preflight(_: argparse.Namespace) -> int:
    config = load_config()
    state = load_state()
    stage = "P3_0_PREFLIGHT"
    if state["stages"][stage].get("status") == "PASSED":
        print(json.dumps(state["stages"][stage], indent=2, sort_keys=True))
        return 0
    set_stage(state, stage, "RUNNING")
    try:
        result = subprocess.run(
            [sys.executable, "framework/stages/02-flattening/scripts/preflight.py", "--no-write"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        readiness = json.loads(result.stdout)
        if readiness.get("status") != "READY_FOR_EXPLICIT_PHASE3_AUTHORIZATION":
            raise Phase3Error("Phase-2 closeout did not authorize Phase 3")
        if readiness.get("completion_mode") != "R6_LOCAL_FUNCTIONAL_ONLY":
            raise Phase3Error("Phase-3 completion mode differs from frozen R6 scope")
        integrity = subprocess.run(
            [sys.executable, "scripts/verify_campaign_closeout.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        closeout = json.loads(integrity.stdout)
        if closeout.get("status") != "PHASE0_PHASE1_PHASE2_R6_LOCAL_FUNCTIONAL_INTEGRITY_PASSED":
            raise Phase3Error("cross-phase integrity audit failed")
        vast = sanitized_vast_receipt(int(config["vast"]["instance_id"]))
        ceiling = 30.0
        projected = float(vast["estimated_compute_cost_to_date_usd"]) + (
            float(vast["hourly_rate_usd"]) * float(config["timebox_hours"])
        )
        cost = {
            "artifact_scope": config["artifact_scope"],
            "budget_ceiling_usd": ceiling,
            "timebox_hours": config["timebox_hours"],
            "vast": vast,
            "projected_worst_case_compute_cost_usd": projected,
            "projected_within_cap": projected <= ceiling,
            "generated_at_utc": utc_now(),
        }
        if not cost["projected_within_cap"]:
            raise Phase3Error("worst-case 24-hour projection exceeds existing USD 30 cap")
        environment = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_environment_lock_v1",
            "generated_at_utc": utc_now(),
            "local": local_environment(),
            "vast": vast,
            "config_sha256": sha256_file(CONFIG_PATH),
        }
        receipt = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_preflight_receipt_v1",
            "generated_at_utc": utc_now(),
            "readiness": readiness,
            "closeout_status": closeout["status"],
            "environment_lock": "phase3/ENVIRONMENT_LOCK.json",
            "cost_ledger": "phase3/COST_LEDGER.json",
            "status": "PASSED",
        }
        atomic_json(PHASE3 / "COST_LEDGER.json", cost)
        atomic_json(PHASE3 / "ENVIRONMENT_LOCK.json", environment)
        atomic_json(PHASE3 / "PREFLIGHT_RECEIPT.json", receipt)
        set_stage(state, stage, "PASSED", receipt="phase3/PREFLIGHT_RECEIPT.json")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        set_stage(state, stage, "BLOCKED", error=f"{type(error).__name__}: {error}")
        raise


def fetch_json(url: str) -> tuple[dict[str, Any], str]:
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as error:
        raise Phase3Error(f"cannot fetch source metadata {url}: {error}") from error
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise Phase3Error(f"invalid source metadata at {url}") from error
    if not isinstance(parsed, dict):
        raise Phase3Error(f"source metadata at {url} is not an object")
    return parsed, sha256_bytes(payload)


def zarr_lock(uri: str) -> dict[str, Any]:
    root = uri.rstrip("/")
    group, group_hash = fetch_json(f"{root}/.zgroup")
    attributes, attributes_hash = fetch_json(f"{root}/.zattrs")
    array, array_hash = fetch_json(f"{root}/0/.zarray")
    if group.get("zarr_format") != 2 or array.get("zarr_format") != 2:
        raise Phase3Error("only zarr v2 source groups are supported")
    shape = array.get("shape")
    chunks = array.get("chunks")
    if not (
        isinstance(shape, list)
        and isinstance(chunks, list)
        and len(shape) == len(chunks) == 3
        and all(isinstance(value, int) and value > 0 for value in [*shape, *chunks])
    ):
        raise Phase3Error("source zarr must define positive 3-D shape/chunks")
    return {
        "uri": root,
        "zarr_group": group,
        "zarr_attributes": attributes,
        "zarr_array_level_0": array,
        "metadata_sha256": {
            ".zgroup": group_hash,
            ".zattrs": attributes_hash,
            "0/.zarray": array_hash,
        },
    }


def tifxyz_lock(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected = ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json")
    files: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    common_shape: tuple[int, ...] | None = None
    for name in expected:
        path = directory / name
        if not path.is_file():
            raise Phase3Error(f"missing TIFXYZ input {display_path(path)}")
        record: dict[str, Any] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if name.endswith(".tif"):
            value = tifffile.imread(path)
            if value.ndim != 2 or not np.isfinite(value).all():
                raise Phase3Error(f"invalid non-finite or non-2D {display_path(path)}")
            if common_shape is None:
                common_shape = value.shape
            elif value.shape != common_shape:
                raise Phase3Error("TIFXYZ coordinate rasters have inconsistent shapes")
            record.update({"shape": list(value.shape), "dtype": str(value.dtype)})
            arrays[name] = value
        files[name] = record
    meta = read_json(directory / "meta.json")
    return {"path": display_path(directory), "files": files, "meta": meta}, arrays


def bounds_from_bbox(
    bbox: list[list[float]], shape_zyx: list[int], padding: int, chunk_zyx: list[int]
) -> dict[str, Any]:
    if not (
        isinstance(bbox, list)
        and len(bbox) == 2
        and all(isinstance(row, list) and len(row) == 3 for row in bbox)
    ):
        raise Phase3Error("TIFXYZ metadata bbox must be [[x,y,z],[x,y,z]]")
    if padding < 0:
        raise Phase3Error("CT padding cannot be negative")
    minimum_xyz = [math.floor(min(bbox[0][axis], bbox[1][axis]) - padding) for axis in range(3)]
    maximum_xyz = [math.ceil(max(bbox[0][axis], bbox[1][axis]) + padding) for axis in range(3)]
    minimum_zyx = [max(0, minimum_xyz[2]), max(0, minimum_xyz[1]), max(0, minimum_xyz[0])]
    maximum_zyx = [
        min(int(shape_zyx[0]) - 1, maximum_xyz[2]),
        min(int(shape_zyx[1]) - 1, maximum_xyz[1]),
        min(int(shape_zyx[2]) - 1, maximum_xyz[0]),
    ]
    if any(lower > upper for lower, upper in zip(minimum_zyx, maximum_zyx)):
        raise Phase3Error("slab bounds do not intersect CT")
    chunk_min = [lower // chunk for lower, chunk in zip(minimum_zyx, chunk_zyx)]
    chunk_max = [upper // chunk for upper, chunk in zip(maximum_zyx, chunk_zyx)]
    counts = [upper - lower + 1 for lower, upper in zip(minimum_zyx, maximum_zyx)]
    chunk_counts = [upper - lower + 1 for lower, upper in zip(chunk_min, chunk_max)]
    return {
        "coordinate_space": "ct_l0_xyz",
        "bbox_xyz_l0": bbox,
        "padding_voxels": padding,
        "inclusive_bounds_zyx": {"min": minimum_zyx, "max": maximum_zyx},
        "shape_zyx": counts,
        "chunk_indices_zyx": {"min": chunk_min, "max": chunk_max, "shape": chunk_counts},
        "required_chunk_count": int(np.prod(chunk_counts)),
    }


def stage_lock(_: argparse.Namespace) -> int:
    config = load_config()
    state = load_state()
    require_stage(state, "P3_0_PREFLIGHT")
    stage = "P3_1_SOURCE_LOCK"
    if state["stages"][stage].get("status") == "PASSED":
        print(json.dumps(state["stages"][stage], indent=2, sort_keys=True))
        return 0
    set_stage(state, stage, "RUNNING")
    try:
        target = config["primary_target"]
        job_path = ROOT / config["source_job_record"]
        job_record = read_json(job_path)
        job = job_record.get("job")
        if not isinstance(job, dict) or job.get("state") != "succeeded":
            raise Phase3Error("primary VC3D job is not succeeded")
        normalised = job.get("normalized_input")
        surface = job.get("surface")
        if not isinstance(normalised, dict) or not isinstance(surface, dict):
            raise Phase3Error("primary VC3D record lacks normalized input/surface")
        request_id = normalised.get("client_request_id")
        if request_id not in {target["seed_id"], f'campaign-x-{target["seed_id"]}'}:
            raise Phase3Error("job identity differs from locked target")
        tif_directory = ROOT / target["surface_tifxyz_dir"]
        tif_lock, arrays = tifxyz_lock(tif_directory)
        meta = tif_lock["meta"]
        bbox = meta.get("bbox")
        if bbox != surface.get("bbox"):
            raise Phase3Error("TIFXYZ meta bbox differs from VC3D job bbox")
        ct = zarr_lock(str(normalised["prediction_uri"]).replace("/representations/predictions/surfaces/", "/volumes/").replace("-surface-20260413222639-surface-m7-L0-th0.2.zarr", "-8.640um-1.2m-116keV-masked.zarr"))
        # The explicit frozen Phase-0 CT URI is authoritative; reject heuristic divergence.
        frozen_inventory = read_json(ROOT / "phase0/eligible_volumes.json")
        entry = next(
            (
                item
                for item in frozen_inventory.get("entries", [])
                if item.get("sample_id") == target["scroll_id"]
            ),
            None,
        )
        if not isinstance(entry, dict):
            raise Phase3Error("target scroll absent from Phase-0 inventory")
        if ct["uri"] != str(entry.get("ct_uri", "")).rstrip("/"):
            raise Phase3Error("heuristic CT URI differs from Phase-0 frozen inventory")
        prediction = zarr_lock(str(entry["surface_prediction_uri"]))
        array = ct["zarr_array_level_0"]
        slab = bounds_from_bbox(
            bbox,
            array["shape"],
            int(config["ct_padding_voxels"]),
            array["chunks"],
        )
        coordinate_contract = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_coordinate_contract_v1",
            "target_seed_id": target["seed_id"],
            "native_surface_order": "xyz",
            "ct_zarr_order": "zyx",
            "conversion": "[x,y,z]_xyz -> [z,y,x]_zyx",
            "source_job_coordinate_space": normalised.get("coordinates", {}).get("ct_l0", {}).get("space"),
            "identity_scale": normalised.get("coordinates", {}).get("transform"),
            "roundtrip_max_error_voxels": 0.0,
            "status": "PASSED",
            "generated_at_utc": utc_now(),
        }
        source_lock = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_source_lock_v1",
            "generated_at_utc": utc_now(),
            "target": target,
            "entry": entry,
            "source_job_record": {
                "path": display_path(job_path),
                "sha256": sha256_file(job_path),
                "job_id": job.get("job_id"),
                "finished_at": job.get("finished_at"),
            },
            "tifxyz": tif_lock,
            "ct": ct,
            "surface_prediction": prediction,
            "slab_bounds": slab,
            "coordinate_contract": "phase3/COORDINATE_CONTRACT.json",
            "status": "PASSED",
        }
        target_lock = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_target_lock_v1",
            "generated_at_utc": utc_now(),
            "target": target,
            "fallbacks": ["PHerc1447-a02-inner", "PHerc1447-a08-middle", "PHerc268-a05-inner"],
            "fallback_activation": "technical_only_before_fit",
            "status": "LOCKED",
        }
        inventory = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_source_byte_inventory_v1",
            "generated_at_utc": utc_now(),
            "tifxyz_files": tif_lock["files"],
            "metadata_only_files": {
                "ct": ct["metadata_sha256"],
                "surface_prediction": prediction["metadata_sha256"],
            },
            "required_ct_chunk_count": slab["required_chunk_count"],
        }
        atomic_json(PHASE3 / "COORDINATE_CONTRACT.json", coordinate_contract)
        atomic_json(PHASE3 / "SLAB_BOUNDS.json", slab)
        atomic_json(PHASE3 / "SOURCE_BYTE_INVENTORY.json", inventory)
        atomic_json(PHASE3 / "TARGET_LOCK.json", target_lock)
        atomic_json(PHASE3 / "SOURCE_LOCK.json", source_lock)
        set_stage(state, stage, "PASSED", receipt="phase3/SOURCE_LOCK.json")
        print(json.dumps({"status": "PASSED", "required_ct_chunk_count": slab["required_chunk_count"], "slab_shape_zyx": slab["shape_zyx"]}, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        set_stage(state, stage, "BLOCKED", error=f"{type(error).__name__}: {error}")
        raise


def chunk_coordinates(bounds: dict[str, Any]) -> list[tuple[int, int, int]]:
    indices = bounds.get("chunk_indices_zyx")
    if not isinstance(indices, dict):
        raise Phase3Error("slab bounds lack chunk indices")
    lower = indices.get("min")
    upper = indices.get("max")
    if not (
        isinstance(lower, list)
        and isinstance(upper, list)
        and len(lower) == len(upper) == 3
        and all(isinstance(value, int) for value in [*lower, *upper])
    ):
        raise Phase3Error("slab chunk bounds are malformed")
    return [(z, y, x) for z in range(lower[0], upper[0] + 1) for y in range(lower[1], upper[1] + 1) for x in range(lower[2], upper[2] + 1)]


def download_file(url: str, destination: Path) -> dict[str, Any]:
    if destination.is_file() and destination.stat().st_size > 0:
        return {"status": "reused", "path": destination, "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size}
    last_error: Exception | None = None
    for delay in (0, 5, 20):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            atomic_bytes(destination, payload)
            return {"status": "downloaded", "path": destination, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {"status": "missing", "path": destination, "http_status": 404}
            last_error = error
        except (OSError, urllib.error.URLError) as error:
            last_error = error
    return {"status": "failed", "path": destination, "error": f"{type(last_error).__name__}: {last_error}"}


def stage_materialize(args: argparse.Namespace) -> int:
    config = load_config()
    state = load_state()
    require_stage(state, "P3_1_SOURCE_LOCK")
    stage = "P3_2_MATERIALIZE"
    if state["stages"][stage].get("status") == "PASSED":
        print(json.dumps(state["stages"][stage], indent=2, sort_keys=True))
        return 0
    set_stage(state, stage, "RUNNING")
    try:
        source = read_json(PHASE3 / "SOURCE_LOCK.json")
        bounds = read_json(PHASE3 / "SLAB_BOUNDS.json")
        root = args.cache_root.resolve()
        if root == ROOT or root == PHASE3:
            raise Phase3Error("cache root must not be a repository root")
        ct = source["ct"]
        zarr_array = ct["zarr_array_level_0"]
        chunks = chunk_coordinates(bounds)
        ct_root = root / "ct.zarr"
        for relative in (".zgroup", ".zattrs", "0/.zarray"):
            url = f'{ct["uri"]}/{relative}'
            result = download_file(url, ct_root / relative)
            if result["status"] not in {"downloaded", "reused"}:
                raise Phase3Error(f"cannot materialize required CT metadata {relative}")

        def one_chunk(coordinate: tuple[int, int, int]) -> tuple[str, dict[str, Any]]:
            relative = f"0/{coordinate[0]}/{coordinate[1]}/{coordinate[2]}"
            return relative, download_file(f'{ct["uri"]}/{relative}', ct_root / relative)

        rows: dict[str, dict[str, Any]] = {}
        workers = min(max(1, int(args.workers or config["ct_workers"])), 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for relative, result in pool.map(one_chunk, chunks):
                relative_result = dict(result)
                relative_result["path"] = relative
                rows[relative] = relative_result
        succeeded = [record for record in rows.values() if record["status"] in {"downloaded", "reused"}]
        failed = [record for record in rows.values() if record["status"] == "failed"]
        missing = [record for record in rows.values() if record["status"] == "missing"]
        coverage = len(succeeded) / max(1, len(chunks))
        manifest = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_materialization_receipt_v1",
            "generated_at_utc": utc_now(),
            "cache_root": str(root),
            "ct_uri": ct["uri"],
            "ct_shape_zyx": zarr_array["shape"],
            "ct_chunks_zyx": zarr_array["chunks"],
            "required_chunk_count": len(chunks),
            "successful_chunk_count": len(succeeded),
            "missing_chunk_count": len(missing),
            "failed_chunk_count": len(failed),
            "ct_coverage_fraction": coverage,
            "downloaded_or_reused_bytes": sum(int(record.get("size_bytes", 0)) for record in succeeded),
            "chunks": rows,
            "status": "PASSED" if coverage >= 0.90 and not failed else "FAILED_GATE",
        }
        atomic_json(PHASE3 / "MATERIALIZATION_PLAN.json", {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_materialization_plan_v1",
            "cache_root": str(root),
            "slab_bounds_sha256": sha256_file(PHASE3 / "SLAB_BOUNDS.json"),
            "required_chunk_count": len(chunks),
            "workers": workers,
        })
        atomic_json(PHASE3 / "MATERIALIZATION_RECEIPT.json", manifest)
        if manifest["status"] != "PASSED":
            set_stage(state, stage, "FAILED_GATE", receipt="phase3/MATERIALIZATION_RECEIPT.json")
            return 2
        set_stage(state, stage, "PASSED", receipt="phase3/MATERIALIZATION_RECEIPT.json")
        print(json.dumps({key: manifest[key] for key in ("status", "required_chunk_count", "successful_chunk_count", "ct_coverage_fraction", "downloaded_or_reused_bytes")}, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        set_stage(state, stage, "BLOCKED", error=f"{type(error).__name__}: {error}")
        raise


def hashed_tree(root: Path) -> list[dict[str, Any]]:
    """Return a deterministic byte inventory for a small, final render tree."""

    if not root.is_dir():
        raise Phase3Error(f"render root is absent: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append({
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    if not entries:
        raise Phase3Error("render root contains no files")
    return entries


def stage_record_local_render(args: argparse.Namespace) -> int:
    """Validate and receipt an official-renderer output without advancing P3.4.

    This is intentionally a sidecar because the single locked surface has not
    yet supplied the independent CT-backed relative constraints required for a
    genuine baseline-vs-R6 spiral fit comparison.
    """

    config = load_config()
    state = load_state()
    require_stage(state, "P3_2_MATERIALIZE")
    root = args.render_root.resolve()
    tiff_root = root / "tiffs"
    zarr_root = root / "normal_stack.zarr"
    tiffs, slice_ordering = ordered_tiff_files(tiff_root)
    expected_slices = int(args.expected_slices)
    if len(tiffs) != expected_slices:
        raise Phase3Error(f"expected {expected_slices} TIFF slices, found {len(tiffs)}")
    voxel_size_um = float(args.voxel_size_um)
    slice_step = float(args.slice_step)
    if not math.isfinite(voxel_size_um) or voxel_size_um <= 0:
        raise Phase3Error("voxel_size_um must be finite and positive")
    if not math.isfinite(slice_step) or slice_step <= 0:
        raise Phase3Error("slice_step must be finite and positive")
    depth_extent_um = expected_slices * slice_step * voxel_size_um
    shapes: list[list[int]] = []
    for path in tiffs:
        # Read only TIFF metadata.  The official renderer may choose LZW; a
        # receipt must remain portable even where optional imagecodecs is not
        # installed, and byte hashes below cover the actual compressed data.
        with tifffile.TiffFile(path) as image:
            if len(image.pages) != 1:
                raise Phase3Error(f"TIFF slice has unexpected page count: {path.name}")
            page = image.pages[0]
            shape = page.shape
            dtype = page.dtype
        if len(shape) != 2 or dtype.kind not in {"u", "i", "f"}:
            raise Phase3Error(f"invalid TIFF slice: {path.name}")
        shapes.append([int(value) for value in shape])
    if len({tuple(shape) for shape in shapes}) != 1:
        raise Phase3Error("TIFF slices do not have one consistent shape")
    zarr_array_path = zarr_root / "0" / ".zarray"
    if not zarr_array_path.is_file():
        raise Phase3Error("normal_stack.zarr lacks level-0 metadata")
    zarr_array = read_json(zarr_array_path)
    zarr_shape = zarr_array.get("shape")
    expected_shape = [expected_slices, *shapes[0]]
    if zarr_shape != expected_shape:
        raise Phase3Error(f"Zarr shape {zarr_shape} differs from TIFF stack {expected_shape}")
    renderer = args.renderer.resolve()
    if not renderer.is_file() or not os.access(renderer, os.X_OK):
        raise Phase3Error(f"official renderer is unavailable: {renderer}")
    source_root = args.source_root.resolve()
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # A build directory inside the worktree is intentionally untracked.  Only
    # tracked or staged source edits would compromise the source lock.
    for diff_args in (("diff", "--exit-code"), ("diff", "--cached", "--exit-code")):
        result = subprocess.run(
            ["git", "-C", str(source_root), *diff_args],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Phase3Error("renderer source has tracked or staged edits")
    compiler = subprocess.run(
        [args.compiler, "--version"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()[0]
    receipt = {
        "artifact_scope": config["artifact_scope"],
        "kind": "campaign_x_phase3_baseline_local_render_receipt_v1",
        "generated_at_utc": utc_now(),
        "status": "PASSED_LOCAL_RENDER_ONLY",
        "explicit_non_claims": [
            "not a spiral fit",
            "not an R6 constrained fit",
            "not a baseline-vs-R6 comparison",
            "not a text or ink claim",
        ],
        "target_seed_id": config["primary_target"]["seed_id"],
        "source_lock_sha256": sha256_file(PHASE3 / "SOURCE_LOCK.json"),
        "materialization_receipt_sha256": sha256_file(PHASE3 / "MATERIALIZATION_RECEIPT.json"),
        "renderer": {
            "path": str(renderer),
            "sha256": sha256_file(renderer),
            "source_commit": source_commit,
            "compiler": compiler,
        },
        "render_command": args.render_command,
        "expected_slices": expected_slices,
        "slice_ordering": slice_ordering,
        "tiff_shape_yx": shapes[0],
        "zarr_shape_zyx": zarr_shape,
        # Without these three the receipt cannot answer the only physical
        # question that matters about a normal stack: how thick is it, and how
        # many papyrus windings does it therefore span.  65 slices x 9.362 um
        # is 608.5 um, against a measured dr_per_winding of 15.53-15.79 vox
        # (145-148 um), i.e. about 4.1 turns inside one stack.
        "voxel_size_um": voxel_size_um,
        "slice_step": slice_step,
        "depth_extent_um": depth_extent_um,
        "output_files": hashed_tree(root),
    }
    output_receipt = root / "RUN_RECEIPT.json"
    atomic_json(output_receipt, receipt)
    receipt["output_files"] = hashed_tree(root)
    atomic_json(PHASE3 / "BASELINE_LOCAL_RENDER_RECEIPT.json", receipt)
    state.setdefault("sidecar_artifacts", {})["BASELINE_LOCAL_RENDER"] = {
        "status": receipt["status"],
        "receipt": "phase3/BASELINE_LOCAL_RENDER_RECEIPT.json",
        "updated_at_utc": utc_now(),
    }
    write_state(state)
    print(json.dumps({
        "status": receipt["status"],
        "expected_slices": expected_slices,
        "tiff_shape_yx": shapes[0],
        "depth_extent_um": depth_extent_um,
        "output_file_count": len(receipt["output_files"]),
    }, indent=2, sort_keys=True))
    return 0


def stage_constraints(_: argparse.Namespace) -> int:
    """Run the fail-closed R6 input-capacity audit for the locked target.

    The R6 policy is a relation policy: an ``ADJACENT`` edge requires two
    non-incident physical surface instances.  A single TIFXYZ is sufficient
    for the local normal-stack render, but cannot manufacture an independent
    relative relation.  This audit records that distinction before any
    algorithm can accidentally convert Phase-2 shadow constraints into new
    Phase-3 evidence.
    """

    config = load_config()
    state = load_state()
    require_stage(state, "P3_2_MATERIALIZE")
    stage = "P3_3_R6_CONSTRAINTS"
    previous = state["stages"].get(stage, {})
    if previous.get("status") in TERMINAL_STAGE_STATES:
        print(json.dumps(previous, indent=2, sort_keys=True))
        return 0 if previous.get("status") == "PASSED" else 2
    set_stage(state, stage, "RUNNING")
    try:
        source = read_json(PHASE3 / "SOURCE_LOCK.json")
        materialization = read_json(PHASE3 / "MATERIALIZATION_RECEIPT.json")
        target = config["primary_target"]
        required_edges = 8
        required_relative_edges = 2
        surface_instances = [
            {
                "seed_id": target["seed_id"],
                "tifxyz_file_sha256": {
                    name: record["sha256"]
                    for name, record in sorted(source["tifxyz"]["files"].items())
                },
                "role": "locked_primary_surface",
            }
        ]
        # The policy only regards non-incident, distinct surface instances as
        # candidates for an ADJACENT relation.  With one locked segmentation,
        # its combinatorial maximum is exactly zero; CT intensity cannot turn
        # two points on this same mesh into a second physical sheet.
        max_relative_candidates = 0
        receipt = {
            "artifact_scope": config["artifact_scope"],
            "kind": "campaign_x_phase3_r6_input_capacity_audit_v1",
            "generated_at_utc": utc_now(),
            "target_seed_id": target["seed_id"],
            "status": "SPARSE_R6",
            "classification": "R6_INPUT_CAPACITY_FAILED_BEFORE_DEPLOYMENT",
            "ct_materialization": {
                "receipt_sha256": sha256_file(PHASE3 / "MATERIALIZATION_RECEIPT.json"),
                "coverage_fraction": materialization["ct_coverage_fraction"],
                "failed_chunk_count": materialization["failed_chunk_count"],
            },
            "locked_surface_instances": surface_instances,
            "locked_surface_instance_count": len(surface_instances),
            "accepted_edges": 0,
            "accepted_relative_edges": 0,
            "max_physical_relative_candidates": max_relative_candidates,
            "capacity_gate": {
                "required_accepted_edges": required_edges,
                "required_relative_edges": required_relative_edges,
                "passed": False,
            },
            "safety_gate": {
                "status": "NOT_EVALUABLE_NO_INDEPENDENT_SURFACE_PAIR",
                "shadow_constraints_used": False,
                "synthetic_or_inferred_relative_edges_used": False,
            },
            "reason": (
                "The locked Phase-3 target contains one TIFXYZ surface. "
                "R6 ADJACENT edges require two non-incident physical surface "
                "instances, so the frozen minimum of two relative edges cannot "
                "be met without changing the locked input topology."
            ),
            "allowed_next_actions": [
                "preserve BASELINE_LOCAL_RENDER for Phase-4 preparation",
                "declare the R6 A/B demo inconclusive for fast_v1",
                "design a new source-locked multi-surface Phase-3 contract before rerunning R6",
            ],
            "forbidden_actions": [
                "reuse Phase-2 shadow constraints as CT-backed Phase-3 evidence",
                "infer a second sheet from CT intensity alone",
                "silently swap to a sibling target after observing this capacity result",
            ],
        }
        output = PHASE3 / "constraints" / target["seed_id"] / "R6_DEPLOYMENT_RECEIPT.json"
        atomic_json(output, receipt)
        set_stage(
            state,
            stage,
            "FAILED_GATE",
            receipt=display_path(output),
            classification=receipt["classification"],
        )
        print(json.dumps({
            "status": receipt["status"],
            "classification": receipt["classification"],
            "accepted_relative_edges": 0,
            "required_relative_edges": required_relative_edges,
        }, indent=2, sort_keys=True))
        return 2
    except Exception as error:
        set_stage(state, stage, "BLOCKED", error=f"{type(error).__name__}: {error}")
        raise


def stage_closeout(_: argparse.Namespace) -> int:
    """Close fast_v1 honestly when its preregistered R6 gate is sparse."""

    config = load_config()
    state = load_state()
    for stage in ("P3_0_PREFLIGHT", "P3_1_SOURCE_LOCK", "P3_2_MATERIALIZE"):
        require_stage(state, stage)
    constraint_stage = state["stages"].get("P3_3_R6_CONSTRAINTS", {})
    if constraint_stage.get("status") != "FAILED_GATE":
        raise Phase3Error("closeout is only valid after the explicit R6 capacity gate fails")
    capacity_receipt = read_json(
        PHASE3 / "constraints" / config["primary_target"]["seed_id"] / "R6_DEPLOYMENT_RECEIPT.json"
    )
    if capacity_receipt.get("status") != "SPARSE_R6":
        raise Phase3Error("closeout requires SPARSE_R6 rather than another constraint failure")
    render = state.get("sidecar_artifacts", {}).get("BASELINE_LOCAL_RENDER", {})
    if render.get("status") != "PASSED_LOCAL_RENDER_ONLY":
        raise Phase3Error("closeout requires a receipted local render sidecar")

    skipped = {
        "P3_4_BASELINE_FIT": "A global spiral fit was not defined by the one-surface input.",
        "P3_5_R6_FIT": "R6 capacity gate failed; no constrained arm may be fabricated.",
        "P3_6_COMPARISON": "No comparable baseline/R6 pair exists.",
        "P3_7_TRANSFER_SMOKE": "Not meaningful until a multi-surface R6 source contract exists.",
        "P3_8_PACKAGE": "A Progress Prize package would overstate the incomplete R6 comparison.",
    }
    for stage, reason in skipped.items():
        if state["stages"].get(stage, {}).get("status") == "PENDING":
            set_stage(state, stage, "SKIPPED", reason=reason)

    closeout = {
        "artifact_scope": config["artifact_scope"],
        "kind": "campaign_x_phase3_fast_v1_closeout_v1",
        "generated_at_utc": utc_now(),
        "status": "PHASE3_COMPLETED_INCONCLUSIVE",
        "target_seed_id": config["primary_target"]["seed_id"],
        "successful_artifacts": {
            "source_lock": "phase3/SOURCE_LOCK.json",
            "ct_materialization": "phase3/MATERIALIZATION_RECEIPT.json",
            "baseline_local_render": "phase3/BASELINE_LOCAL_RENDER_RECEIPT.json",
        },
        "r6_result": {
            "status": "SPARSE_R6",
            "receipt": display_path(
                PHASE3 / "constraints" / config["primary_target"]["seed_id"] / "R6_DEPLOYMENT_RECEIPT.json"
            ),
            "reason": "single locked surface yields zero possible physical relative edges",
        },
        "claims_not_made": [
            "R6 improves spiral fitting",
            "baseline-vs-R6 comparison",
            "text, ink, or first-letter discovery",
            "external generalization",
        ],
        "handoff": {
            "allowed": True,
            "scope": "LOCAL_PIPELINE_CONTINUATION_ONLY",
            "input": "receipted PHerc1447 normal stack",
            "restriction": "Phase 4 must not present the stack as a validated R6 improvement.",
        },
        "required_for_a_new_r6_phase3_contract": [
            "at least two independently materialized non-incident surface instances",
            "pre-locked outer-shell or equivalent physical context",
            "a source geometry layout capable of two relative R6 edges before any fit",
        ],
    }
    atomic_json(PHASE3 / "PHASE3_CLOSEOUT.json", closeout)
    set_stage(
        state,
        "P3_9_CLOSEOUT",
        "PASSED",
        receipt="phase3/PHASE3_CLOSEOUT.json",
        classification=closeout["status"],
    )
    print(json.dumps({
        "status": closeout["status"],
        "r6_status": closeout["r6_result"]["status"],
        "handoff_allowed": closeout["handoff"]["allowed"],
    }, indent=2, sort_keys=True))
    return 0


def unavailable(stage: str) -> int:
    print(json.dumps({
        "status": "NOT_YET_ENABLED",
        "stage": stage,
        "reason": "Field implementation is not present; refusing to substitute Phase-2 shadow data for CT-backed evidence.",
    }, indent=2, sort_keys=True))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("lock")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--cache-root", type=Path, required=True)
    materialize.add_argument("--workers", type=int)
    render_receipt = subparsers.add_parser("record-local-render")
    render_receipt.add_argument("--render-root", type=Path, required=True)
    render_receipt.add_argument("--renderer", type=Path, required=True)
    render_receipt.add_argument("--source-root", type=Path, required=True)
    render_receipt.add_argument("--compiler", default="g++-13")
    render_receipt.add_argument("--expected-slices", type=int, default=65)
    render_receipt.add_argument(
        "--voxel-size-um",
        type=float,
        required=True,
        help=(
            "physical size of one CT voxel in micrometres. Required because a "
            "receipt that records only a slice count cannot reconstruct the "
            "stack's physical thickness, and therefore cannot say how many "
            "papyrus windings the stack spans"
        ),
    )
    render_receipt.add_argument(
        "--slice-step",
        type=float,
        default=1.0,
        help="spacing between rendered slices, in voxels along the normal",
    )
    render_receipt.add_argument("--command", dest="render_command", required=True)
    subparsers.add_parser("constraints")
    for command in ("evaluate", "transfer-smoke", "package", "verify", "all", "closeout"):
        subparsers.add_parser(command)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--arm", choices=("baseline", "r6"), required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            return stage_preflight(args)
        if args.command == "lock":
            return stage_lock(args)
        if args.command == "materialize":
            return stage_materialize(args)
        if args.command == "record-local-render":
            return stage_record_local_render(args)
        if args.command == "constraints":
            return stage_constraints(args)
        if args.command == "closeout":
            return stage_closeout(args)
        return unavailable(args.command)
    except Phase3Error as error:
        print(json.dumps({"status": "ERROR", "error": str(error)}, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - terminal safety receipt is in RUN_STATE
        print(json.dumps({"status": "ERROR", "error": f"{type(error).__name__}: {error}"}, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
