#!/usr/bin/env python3
"""Plan or execute the isolated GP-Scroll1 screen of 21 official surfaces.

Dry-run performs no network, rendering, inference, ranking, or filesystem
mutation unless ``--plan-output`` is explicitly supplied.  Execute is
fail-closed: it requires the frozen manifest, exact runtime hashes, no
competing Phase 4 GPU process, and exact source/render provenance before a
render can be reused.

The command stops after producing separate v1/v2 rankings and a frozen robust
queue.  It deliberately does not execute the six-replica robust screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


MANIFEST_KIND = "campaign_x_phase4_official_gp_scroll1_rescreen_manifest_v1"
PLAN_KIND = "campaign_x_phase4_official_gp_scroll1_rescreen_plan_v1"
BATCH_KIND = "campaign_x_phase4_official_gp_scroll1_coarse_batch_v1"
QUEUE_KIND = "campaign_x_phase4_official_gp_scroll1_robust_queue_v1"
EXPECTED_SURFACE_COUNT = 21
TIFF_NAMES = [f"{index:02d}.tif" for index in range(65)]
COMPETING_PATTERNS = (
    "run_gp_shortlist_after_current.sh",
    "run_ink_timesformer.py",
    "vc_render_tifxyz",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RuntimeError(f"manifest path must be relative: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"manifest path escapes root: {relative}") from error
    return resolved


def official_uri(official_path: str) -> str:
    return "s3://vesuvius-challenge-open-data/" + official_path.strip("/") + "/"


def source_hashes(source: Path) -> dict[str, str] | None:
    paths = [source / name for name in ("x.tif", "y.tif", "z.tif")]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return None
    return {path.name: sha256_file(path) for path in paths}


def exact_tiff_stack(render: Path) -> tuple[list[Path], str] | None:
    """Return the exact 00..64 render stack, or ``None`` if it is not exact.

    The previous implementation pushed any non-numeric stem into a silent
    ``10_000`` reserve bucket at the end of the stack instead of failing.
    """

    paths, slice_ordering = ordered_tiff_files(
        render,
        require_numeric=True,
        allow_empty=True,
    )
    if [path.name for path in paths] != TIFF_NAMES:
        return None
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return None
    return paths, slice_ordering


def render_endpoint_hashes(paths: list[Path]) -> dict[str, str]:
    selected = (paths[0], paths[len(paths) // 2], paths[-1])
    return {path.name: sha256_file(path) for path in selected}


def valid_render_log(
    log_path: Path,
    *,
    ct_uri: str,
    source: Path,
    render: Path,
    voxel_um: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not log_path.is_file():
        return False, ["render log missing"]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    required = (
        f"Remote zarr streaming: {ct_uri}",
        f"Rendering: {source.resolve()} -> {render.resolve()} (tif)",
        f"Voxel size (from CLI): {voxel_um:g} micrometer",
    )
    for phrase in required:
        if phrase not in text:
            reasons.append(f"render log lacks {phrase!r}")
    if not re.search(r"band \d+/\d+ \(100%\)", text):
        reasons.append("render log lacks a 100% completion marker")
    return not reasons, reasons


def flatten_manifest_tasks(
    root: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if manifest.get("kind") != MANIFEST_KIND:
        raise RuntimeError("unexpected official GP manifest kind")
    if manifest.get("status") != "FROZEN_INPUT_MANIFEST":
        raise RuntimeError("official GP manifest is not frozen")
    inventory_binding = manifest["official_inventory"]
    inventory_path = safe_path(root, str(inventory_binding["path"]))
    if sha256_file(inventory_path) != inventory_binding["sha256"]:
        raise RuntimeError("official product inventory hash mismatch")
    inventory = read_json(inventory_path)
    if inventory.get("kind") != "campaign_x_phase4_official_target_product_inventory_v1":
        raise RuntimeError("unexpected official product inventory kind")

    inventory_sets: dict[str, set[str]] = {}
    inventory_targets: dict[str, dict[str, Any]] = {}
    for target in inventory["targets"]:
        campaign_id = str(target["campaign_sample_id"])
        inventory_targets[campaign_id] = target
        inventory_sets[campaign_id] = set(
            target["products"]["official_tifxyz_surfaces"]["paths"]
        )

    tasks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for target in manifest["targets"]:
        campaign_id = str(target["campaign_sample_id"])
        if campaign_id not in inventory_targets:
            raise RuntimeError(f"manifest target absent from inventory: {campaign_id}")
        frozen = inventory_targets[campaign_id]
        for key in ("official_sample_id", "exact_volume_id"):
            if target[key] != frozen[key]:
                raise RuntimeError(f"{campaign_id} {key} differs from inventory")
        if target["ct_uri"] != frozen["eligible_ct_uri"]:
            raise RuntimeError(f"{campaign_id} CT URI differs from inventory")
        manifest_paths = {
            str(surface["official_tifxyz_path"])
            for surface in target["surfaces"]
        }
        if manifest_paths != inventory_sets[campaign_id]:
            missing = sorted(inventory_sets[campaign_id] - manifest_paths)
            extra = sorted(manifest_paths - inventory_sets[campaign_id])
            raise RuntimeError(
                f"{campaign_id} exact official surface set mismatch: "
                f"missing={missing}, extra={extra}"
            )
        target_lock = safe_path(
            root, f"phase4/targets/{campaign_id}/TARGET_LOCK.json"
        )
        lock: dict[str, Any] | None = read_json(target_lock) if target_lock.is_file() else None
        if lock is not None and (
            lock.get("sample_id") != campaign_id
            or lock.get("ct_uri") != target["ct_uri"]
            or float(lock.get("voxel_size_um")) != float(target["voxel_size_um"])
        ):
            raise RuntimeError(f"target lock mismatch for {campaign_id}")
        for surface in target["surfaces"]:
            key = (campaign_id, str(surface["surface_id"]))
            if key in seen:
                raise RuntimeError(f"duplicate manifest surface: {key}")
            seen.add(key)
            task = {
                **surface,
                "campaign_sample_id": campaign_id,
                "official_sample_id": str(target["official_sample_id"]),
                "exact_volume_id": str(target["exact_volume_id"]),
                "ct_uri": str(target["ct_uri"]),
                "voxel_size_um": float(target["voxel_size_um"]),
                "official_tifxyz_uri": official_uri(
                    str(surface["official_tifxyz_path"])
                ),
            }
            tasks.append(task)
    tasks.sort(key=lambda row: (row["campaign_sample_id"], row["surface_id"]))
    if len(tasks) != EXPECTED_SURFACE_COUNT:
        raise RuntimeError(
            f"manifest must contain exactly {EXPECTED_SURFACE_COUNT} surfaces"
        )
    return manifest, tasks


def materialized_paths(
    root: Path,
    manifest: dict[str, Any],
    task: dict[str, Any],
) -> tuple[Path, Path, Path]:
    base = safe_path(
        root,
        str(manifest["runtime"]["materialized_root"]),
    ) / task["campaign_sample_id"] / task["surface_id"]
    return base / "tifxyz", base / "tiffs", base / "render.log"


def audit_task(
    root: Path,
    manifest: dict[str, Any],
    task: dict[str, Any],
    *,
    frozen_reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    material_source, material_render, material_log = materialized_paths(
        root, manifest, task
    )
    candidates: list[tuple[Path, Path, Path, str]] = []
    if all(key in task for key in ("reuse_source", "reuse_render", "reuse_render_log")):
        reuse_source = safe_path(root, str(task["reuse_source"]))
        candidates.append(
            (
                reuse_source,
                safe_path(root, str(task["reuse_render"])),
                safe_path(root, str(task["reuse_render_log"])),
                "PREEXISTING_EXACT_PRODUCT_RENDER",
            )
        )
        candidates.append(
            (
                reuse_source,
                material_render,
                material_log,
                "ISOLATED_RENDER_FROM_PREEXISTING_TIFXYZ",
            )
        )
    candidates.append(
        (material_source, material_render, material_log, "ISOLATED_MATERIALIZED_RENDER")
    )

    audit_failures: list[dict[str, Any]] = []
    first_existing_source: tuple[Path, dict[str, str]] | None = None
    drifted_sources: set[Path] = set()
    for source, render, render_log, provenance in candidates:
        hashes = source_hashes(source)
        if hashes is None:
            audit_failures.append(
                {
                    "provenance": provenance,
                    "source": str(source),
                    "reason": "source x/y/z absent or incomplete",
                }
            )
            continue
        if source.resolve() in drifted_sources:
            audit_failures.append(
                {
                    "provenance": provenance,
                    "source": str(source),
                    "reason": "source rejected by frozen reuse audit",
                }
            )
            continue
        if first_existing_source is None:
            first_existing_source = (source, hashes)
        stack = exact_tiff_stack(render)
        if stack is None:
            audit_failures.append(
                {
                    "provenance": provenance,
                    "source": str(source),
                    "render": str(render),
                    "reason": "render is not exact 00..64 TIFF stack",
                }
            )
            continue
        tiffs, slice_ordering = stack
        log_ok, log_reasons = valid_render_log(
            render_log,
            ct_uri=task["ct_uri"],
            source=source,
            render=render,
            voxel_um=task["voxel_size_um"],
        )
        if not log_ok:
            audit_failures.append(
                {
                    "provenance": provenance,
                    "source": str(source),
                    "render": str(render),
                    "reason": "; ".join(log_reasons),
                }
            )
            continue
        endpoints = render_endpoint_hashes(tiffs)
        if provenance == "PREEXISTING_EXACT_PRODUCT_RENDER" and frozen_reuse:
            frozen_fields = {
                "official_tifxyz_uri": task["official_tifxyz_uri"],
                "source": str(source),
                "source_hashes": hashes,
                "render": str(render),
                "render_endpoint_hashes": endpoints,
                "render_provenance": provenance,
            }
            mismatches = [
                key
                for key, value in frozen_fields.items()
                if frozen_reuse.get(key) != value
            ]
            if mismatches:
                audit_failures.append(
                    {
                        "provenance": provenance,
                        "source": str(source),
                        "render": str(render),
                        "reason": (
                            "frozen reuse audit drift: "
                            + ",".join(sorted(mismatches))
                        ),
                    }
                )
                if {"official_tifxyz_uri", "source", "source_hashes"} & set(
                    mismatches
                ):
                    # A drifted pre-existing source must not become the source
                    # of an isolated rerender. Acquire the exact official
                    # object again instead.
                    drifted_sources.add(source.resolve())
                    if first_existing_source and (
                        first_existing_source[0].resolve() == source.resolve()
                    ):
                        first_existing_source = None
                continue
        return {
            **task,
            "action": "REUSE_EXACT_RENDER",
            "source": str(source),
            "source_hashes": hashes,
            "render": str(render),
            "render_log": str(render_log),
            "render_endpoint_hashes": endpoints,
            "slice_ordering": slice_ordering,
            "render_provenance": provenance,
            "audit_failures": audit_failures,
        }

    incompatible: dict[str, Any] | None = None
    if task.get("known_incompatible_layer_render"):
        path = safe_path(root, str(task["known_incompatible_layer_render"]))
        incompatible = {
            "path": str(path),
            "tiff_count": (
                len(ordered_tiff_files(path, allow_empty=True)[0])
                if path.is_dir()
                else 0
            ),
            "reason": "official layer stack is not the required exact 65 slices",
        }
    if first_existing_source is not None:
        source, hashes = first_existing_source
        action = "RENDER_EXISTING_TIFXYZ"
    else:
        source, hashes = material_source, None
        action = "ACQUIRE_TIFXYZ_AND_RENDER"
    return {
        **task,
        "action": action,
        "source": str(source),
        "source_hashes": hashes,
        "render": str(material_render),
        "render_log": str(material_log),
        "render_provenance": "ISOLATED_MATERIALIZED_RENDER",
        "known_incompatible_render": incompatible,
        "audit_failures": audit_failures,
    }


def runtime_audit(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = manifest["runtime"]
    records: dict[str, Any] = {}
    bindings = (
        ("checkpoint", runtime["checkpoint_path"], runtime["checkpoint_sha256"]),
        ("ct_fiber_gate", runtime["ct_fiber_gate_path"], runtime["ct_fiber_gate_sha256"]),
    )
    for label, relative, expected in bindings:
        path = safe_path(root, str(relative))
        actual = sha256_file(path) if path.is_file() else None
        records[label] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
    renderer = Path(str(runtime["renderer_path"]))
    records["renderer"] = {
        "path": str(renderer),
        "exists": renderer.is_file(),
        "executable": os.access(renderer, os.X_OK),
    }
    records["root"] = {
        "actual": str(root.resolve()),
        "expected": str(runtime["expected_root"]),
        "matches": str(root.resolve()) == str(runtime["expected_root"]),
    }
    return records


def frozen_reuse_tasks(
    root: Path,
    manifest: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    binding = manifest["reuse_audit_binding"]
    path = safe_path(root, str(binding["path"]))
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise RuntimeError("frozen remote reuse audit receipt hash mismatch")
    receipt = read_json(path)
    if (
        receipt.get("kind") != PLAN_KIND
        or receipt.get("status") != "READY_TO_EXECUTE"
        or receipt.get("surface_count") != EXPECTED_SURFACE_COUNT
        or receipt.get("action_counts", {}).get("REUSE_EXACT_RENDER")
        != binding["expected_reusable_count"]
    ):
        raise RuntimeError("frozen remote reuse audit receipt is invalid")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for task in receipt["tasks"]:
        key = (str(task["campaign_sample_id"]), str(task["surface_id"]))
        if key in result:
            raise RuntimeError(f"duplicate task in frozen reuse audit: {key}")
        if task.get("action") == "REUSE_EXACT_RENDER":
            result[key] = task
    if len(result) != int(binding["expected_reusable_count"]):
        raise RuntimeError("frozen reusable surface count mismatch")
    return result


def build_plan(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_tasks = flatten_manifest_tasks(root, manifest_path)
    frozen = frozen_reuse_tasks(root, manifest)
    tasks = [
        audit_task(
            root,
            manifest,
            task,
            frozen_reuse=frozen.get(
                (task["campaign_sample_id"], task["surface_id"])
            ),
        )
        for task in manifest_tasks
    ]
    runtime = runtime_audit(root, manifest)
    runtime_ready = (
        runtime["checkpoint"]["matches"]
        and runtime["ct_fiber_gate"]["matches"]
        and runtime["renderer"]["exists"]
        and runtime["renderer"]["executable"]
    )
    counts = {
        action: sum(task["action"] == action for task in tasks)
        for action in (
            "REUSE_EXACT_RENDER",
            "RENDER_EXISTING_TIFXYZ",
            "ACQUIRE_TIFXYZ_AND_RENDER",
        )
    }
    return {
        "kind": PLAN_KIND,
        "status": "READY_TO_EXECUTE" if runtime_ready else "DRY_RUN_RUNTIME_BLOCKED",
        "generated_at_utc": utc_now(),
        "root": str(root.resolve()),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "runtime_audit": runtime,
        "surface_count": len(tasks),
        "sample_count": len({task["campaign_sample_id"] for task in tasks}),
        "action_counts": counts,
        "tasks": tasks,
        "next_stages": [
            "materialize missing exact TIFXYZ and 65-slice renders",
            "bind all 21 exact renders into the isolated coarse batch",
            "run GP Scroll1 one-replica coarse inference on all 21",
            "write separate v1 and v2 global rankings",
            "freeze their deterministic union as a robust-ready queue",
        ],
        "explicit_non_claims": [
            "dry-run performs no GPU work",
            "coarse ranking does not prove ink or letters",
            "the prepared queue has not run six-replica or CT review",
            "no negative result proves absence of ink",
        ],
    }


def check_no_competing_gpu_work() -> None:
    conflicts: list[str] = []
    own_pid = os.getpid()
    for pattern in COMPETING_PATTERNS:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            text=True,
            capture_output=True,
            check=False,
        )
        pids = {
            int(value)
            for value in result.stdout.split()
            if value.isdigit() and int(value) != own_pid
        }
        if pids:
            conflicts.append(f"{pattern}: {sorted(pids)}")
    if conflicts:
        raise RuntimeError(
            "competing GPU/render work detected; refusing to interfere: "
            + "; ".join(conflicts)
        )


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def fetch_zarr_metadata(ct_uri: str, cache: Path) -> None:
    for suffix in (".zgroup", ".zattrs", "0/.zarray"):
        destination = cache / suffix
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{ct_uri.rstrip('/')}/{suffix}",
            headers={"User-Agent": "Campaign-X-official-GP-rescreen/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())


def materialize_task(
    root: Path,
    manifest: dict[str, Any],
    task: dict[str, Any],
    *,
    cache_gb: int,
) -> None:
    source = Path(task["source"])
    render = Path(task["render"])
    render_log = Path(task["render_log"])
    if task["action"] == "ACQUIRE_TIFXYZ_AND_RENDER":
        source.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                "aws",
                "s3",
                "cp",
                task["official_tifxyz_uri"],
                str(source) + "/",
                "--recursive",
                "--no-sign-request",
                "--only-show-errors",
            ],
            render_log.parent / "download.stdout.log",
        )
    if source_hashes(source) is None:
        raise RuntimeError(f"exact TIFXYZ source remains incomplete: {source}")
    if task["action"] == "REUSE_EXACT_RENDER":
        return
    cache = (
        safe_path(root, str(manifest["runtime"]["output_root"]))
        / "ct_cache"
        / f"{task['campaign_sample_id']}.zarr"
    )
    fetch_zarr_metadata(task["ct_uri"], cache)
    render.mkdir(parents=True, exist_ok=True)
    parameters = manifest["parameters"]["render"]
    run_logged(
        [
            str(manifest["runtime"]["renderer_path"]),
            "--segmentation",
            str(source),
            "--volume",
            str(cache),
            "--remote-url",
            task["ct_uri"],
            "--prefetch-remote",
            "--scale",
            "1",
            "--group-idx",
            "0",
            "--auto-crop",
            "--flatten",
            "--flatten-iterations",
            str(parameters["flatten_iterations"]),
            "--num-slices",
            str(parameters["num_slices"]),
            "--slice-step",
            str(parameters["slice_step"]),
            "--cache-gb",
            str(cache_gb),
            "--timeout",
            "90",
            "--voxel-size",
            str(task["voxel_size_um"]),
            "--voxel-unit",
            str(parameters["voxel_unit"]),
            "--tif-output",
            str(render),
            "--log-path",
            str(render_log),
        ],
        render_log.parent / "render.stdout.log",
    )


def replace_with_exact_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing batch path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=True)


def stage_batch(
    root: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> Path:
    batch_root = safe_path(root, str(manifest["runtime"]["batch_root"]))
    for task in plan["tasks"]:
        surface_root = (
            batch_root / task["campaign_sample_id"] / task["surface_id"]
        )
        replace_with_exact_symlink(surface_root / "tiffs", Path(task["render"]))
        binding = {
            "kind": "campaign_x_phase4_official_surface_runtime_binding_v1",
            "status": "BOUND_EXACT_RENDER",
            "campaign_sample_id": task["campaign_sample_id"],
            "official_sample_id": task["official_sample_id"],
            "surface_id": task["surface_id"],
            "official_tifxyz_uri": task["official_tifxyz_uri"],
            "ct_uri": task["ct_uri"],
            "source": task["source"],
            "source_hashes": task["source_hashes"],
            "render": task["render"],
            "render_endpoint_hashes": task["render_endpoint_hashes"],
            "render_provenance": task["render_provenance"],
        }
        binding_path = surface_root / "SOURCE_BINDING.json"
        if binding_path.is_file() and read_json(binding_path) != binding:
            raise RuntimeError(f"existing source binding differs: {binding_path}")
        write_json(binding_path, binding)
    return batch_root


def validate_coarse_receipt(
    receipt: dict[str, Any],
    *,
    task: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    coarse = manifest["parameters"]["coarse"]
    if (
        receipt.get("kind") != "campaign_x_phase4_timesformer_private_screening_v1"
        or receipt.get("status") != "COMPLETED_DIAGNOSTIC_ONLY"
        or receipt.get("sample_id") != task["campaign_sample_id"]
    ):
        raise RuntimeError(f"invalid coarse receipt for {task['surface_id']}")
    checkpoint = receipt["checkpoint"]
    if (
        checkpoint.get("sha256")
        != manifest["runtime"]["checkpoint_sha256"]
        or checkpoint.get("model_family")
        != manifest["runtime"]["checkpoint_model_family"]
    ):
        raise RuntimeError(f"coarse checkpoint mismatch for {task['surface_id']}")
    input_record = receipt["input"]
    if (
        input_record.get("slice_count") != 65
        or input_record.get("source_slice_hashes")
        != task["render_endpoint_hashes"]
    ):
        raise RuntimeError(f"coarse input mismatch for {task['surface_id']}")
    physical = receipt["physical_normalization"]
    inference = receipt["inference"]
    expected_physical = {
        "source_pixel_um": task["voxel_size_um"],
        "training_pixel_um": coarse["training_pixel_um"],
        "source_slice_um": task["voxel_size_um"],
        "training_slice_um": coarse["training_slice_um"],
        "frames": coarse["frames"],
    }
    for key, expected in expected_physical.items():
        if float(physical.get(key)) != float(expected):
            raise RuntimeError(
                f"coarse physical normalization mismatch: {task['surface_id']}:{key}"
            )
    expected_inference = {
        "depth_centers": coarse["depth_centers"],
        "tiling_offsets": coarse["tiling_offsets"],
        "tile_size": coarse["tile_size"],
        "stride": coarse["stride"],
        "min_valid_ratio": coarse["min_valid_ratio"],
    }
    for key, expected in expected_inference.items():
        if inference.get(key) != expected:
            raise RuntimeError(
                f"coarse inference mismatch: {task['surface_id']}:{key}"
            )


def run_coarse_batch(
    root: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    batch_root: Path,
    *,
    inference_batch_size: int,
) -> Path:
    checkpoint = safe_path(root, str(manifest["runtime"]["checkpoint_path"]))
    coarse = manifest["parameters"]["coarse"]
    results: list[dict[str, Any]] = []
    for index, task in enumerate(plan["tasks"], start=1):
        surface_root = (
            batch_root / task["campaign_sample_id"] / task["surface_id"]
        )
        output = surface_root / coarse["screening_name"]
        receipt_path = output / "INK_SCREENING_RECEIPT.json"
        if not receipt_path.is_file():
            print(
                f"COARSE_START:{index}/{len(plan['tasks'])}:"
                f"{task['campaign_sample_id']}:{task['surface_id']}",
                flush=True,
            )
            run_logged(
                [
                    sys.executable,
                    str(root / "scripts" / "run_ink_timesformer.py"),
                    "--sample-id",
                    task["campaign_sample_id"],
                    "--tiff-dir",
                    str(surface_root / "tiffs"),
                    "--checkpoint",
                    str(checkpoint),
                    "--model-family",
                    str(manifest["runtime"]["checkpoint_model_family"]),
                    "--output",
                    str(output),
                    "--depth-centers",
                    ",".join(map(str, coarse["depth_centers"])),
                    "--tiling-offsets",
                    ",".join(map(str, coarse["tiling_offsets"])),
                    "--frames",
                    str(coarse["frames"]),
                    "--source-pixel-um",
                    str(task["voxel_size_um"]),
                    "--training-pixel-um",
                    str(coarse["training_pixel_um"]),
                    "--source-slice-um",
                    str(task["voxel_size_um"]),
                    "--training-slice-um",
                    str(coarse["training_slice_um"]),
                    "--tile-size",
                    str(coarse["tile_size"]),
                    "--stride",
                    str(coarse["stride"]),
                    "--batch-size",
                    str(inference_batch_size),
                    "--min-valid-ratio",
                    str(coarse["min_valid_ratio"]),
                    "--device",
                    "cuda",
                ],
                surface_root / "coarse_gp_scroll1.stdout.log",
            )
        receipt = read_json(receipt_path)
        validate_coarse_receipt(receipt, task=task, manifest=manifest)
        results.append(
            {
                "sample_id": task["campaign_sample_id"],
                "seed_id": task["surface_id"],
                "status": "COARSE_COMPLETED",
                "output": str(surface_root),
                "coarse_receipt": str(receipt_path),
                "coarse_receipt_sha256": sha256_file(receipt_path),
                "source_binding": str(surface_root / "SOURCE_BINDING.json"),
                "source_binding_sha256": sha256_file(
                    surface_root / "SOURCE_BINDING.json"
                ),
            }
        )
        print(
            f"COARSE_DONE:{index}/{len(plan['tasks'])}:"
            f"{task['campaign_sample_id']}:{task['surface_id']}",
            flush=True,
        )
    batch_receipt = batch_root / "OFFICIAL_GP_SCROLL1_COARSE_BATCH_RECEIPT.json"
    write_json(
        batch_receipt,
        {
            "kind": BATCH_KIND,
            "status": "COMPLETED_PRIORITIZATION_ONLY",
            "updated_at_utc": utc_now(),
            "task_count": len(results),
            "completed_count": len(results),
            "failed_count": 0,
            "screening_name": coarse["screening_name"],
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "model_family": manifest["runtime"]["checkpoint_model_family"],
            },
            "source_plan": {
                "path": str(
                    safe_path(root, str(manifest["runtime"]["output_root"]))
                    / "PREFLIGHT_PLAN.json"
                ),
                "sha256": sha256_file(
                    safe_path(root, str(manifest["runtime"]["output_root"]))
                    / "PREFLIGHT_PLAN.json"
                ),
            },
            "tasks": results,
            "explicit_non_claims": [
                "coarse outputs prioritize robust compute only",
                "not automatic ink or letter acceptance",
                "not evidence of absence from unselected windows",
            ],
        },
    )
    return batch_receipt


def run_rankings(
    root: Path,
    manifest: dict[str, Any],
    batch_root: Path,
    batch_receipt: Path,
) -> tuple[Path, Path]:
    coarse_name = str(manifest["parameters"]["coarse"]["screening_name"])
    v1 = manifest["parameters"]["ranking_v1"]
    v2 = manifest["parameters"]["ranking_v2"]
    v1_path = batch_root / v1["output_name"]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "rank_expanded_candidate_windows.py"),
            "--root",
            str(root),
            "--batch-root",
            str(batch_root),
            "--batch-receipt-name",
            batch_receipt.name,
            "--screening-name",
            coarse_name,
            "--per-sample-ranking-name",
            str(v1["per_sample_namespace"]),
            "--ranking-output-name",
            str(v1["output_name"]),
            "--minimum-valid-ratio",
            str(v1["minimum_valid_ratio"]),
            "--global-top-n",
            str(v1["global_top_n"]),
            "--max-per-sample",
            str(v1["max_per_sample"]),
        ],
        check=True,
    )
    v2_path = batch_root / v2["output_name"]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "rank_expanded_candidate_windows_v2.py"),
            "--root",
            str(root),
            "--batch-root",
            str(batch_root),
            "--batch-receipt-name",
            batch_receipt.name,
            "--screening-name",
            coarse_name,
            "--per-sample-ranking-name",
            str(v2["per_sample_namespace"]),
            "--ranking-output-name",
            str(v2["output_name"]),
            "--minimum-valid-ratio",
            str(v2["minimum_valid_ratio"]),
            "--global-top-n",
            str(v2["global_top_n"]),
            "--max-per-sample",
            str(v2["max_per_sample"]),
            "--global-legacy-rescue-fraction",
            str(v2["global_legacy_rescue_fraction"]),
            "--per-sample-legacy-rescue-fraction",
            str(v2["per_sample_legacy_rescue_fraction"]),
            "--v1-global-ranking",
            str(v1_path),
        ],
        check=True,
    )
    return v1_path, v2_path


def robust_union_rows(
    ranking_v1: dict[str, Any],
    ranking_v2: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    first = list(ranking_v1["global_priority"])
    second = list(ranking_v2["global_priority_v2"])
    selected: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    maximum = max(len(first), len(second))
    for index in range(maximum):
        for origin, rows in (("V2", second), ("V1", first)):
            if index >= len(rows):
                continue
            source = rows[index]
            key = (
                str(source["sample_id"]),
                str(source["surface_id"]),
                tuple(int(value) for value in source["source_crop_xyxy"]),
            )
            if key in by_key:
                existing = by_key[key]
                if origin not in existing["selection_origins"]:
                    existing["selection_origins"].append(origin)
                continue
            if len(selected) >= limit:
                continue
            score = (
                float(source["global_score_v2"])
                if origin == "V2"
                else float(source["score"])
            )
            row = dict(source)
            row["score"] = score
            row["selection_origins"] = [origin]
            row.pop("global_rank", None)
            row.pop("global_rank_v2", None)
            selected.append(row)
            by_key[key] = row
    for rank, row in enumerate(selected, start=1):
        row["global_rank"] = rank
    return selected


def freeze_robust_queue(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    batch_root: Path,
    v1_path: Path,
    v2_path: Path,
) -> Path:
    v1 = read_json(v1_path)
    v2 = read_json(v2_path)
    robust = manifest["parameters"]["robust_preparation"]
    selected = robust_union_rows(
        v1,
        v2,
        limit=int(robust["maximum_union_windows"]),
    )
    if not selected:
        raise RuntimeError("v1/v2 rankings produced no robust queue")
    queue_path = batch_root / robust["queue_name"]
    scripts = (
        "run_ink_timesformer.py",
        "rank_expanded_candidate_windows.py",
        "rank_expanded_candidate_windows_v2.py",
        "run_expanded_robust_windows.py",
    )
    write_json(
        queue_path,
        {
            "kind": QUEUE_KIND,
            # The existing robust runner accepts this status.  The more
            # specific queue_state below makes clear that no robust inference
            # has happened yet.
            "status": "COMPLETED_PRIORITIZATION_ONLY",
            "queue_state": "READY_FOR_ROBUST_EXECUTION",
            "frozen_at_utc": utc_now(),
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "ranking_v1": {
                "path": str(v1_path),
                "sha256": sha256_file(v1_path),
                "selected_count": len(v1["global_priority"]),
            },
            "ranking_v2": {
                "path": str(v2_path),
                "sha256": sha256_file(v2_path),
                "selected_count": len(v2["global_priority_v2"]),
            },
            "checkpoint": {
                "path": str(
                    safe_path(root, str(manifest["runtime"]["checkpoint_path"]))
                ),
                "sha256": manifest["runtime"]["checkpoint_sha256"],
                "model_family": manifest["runtime"]["checkpoint_model_family"],
            },
            "ct_fiber_gate": {
                "path": str(
                    safe_path(root, str(manifest["runtime"]["ct_fiber_gate_path"]))
                ),
                "sha256": manifest["runtime"]["ct_fiber_gate_sha256"],
            },
            "robust_parameters": robust,
            "script_hashes": {
                name: sha256_file(root / "scripts" / name) for name in scripts
            },
            "global_priority": selected,
            "search": {
                "selected_window_count": len(selected),
                "maximum_union_windows": robust["maximum_union_windows"],
                "union_order": "V2_THEN_V1_AT_EACH_SOURCE_RANK",
                "deduplication_key": [
                    "sample_id",
                    "surface_id",
                    "source_crop_xyxy",
                ],
            },
            "policy": [
                "the queue is a deterministic union of separate v1 and v2 rankings",
                "duplicate physical crops retain both selection origins",
                "the queue is compatible with the existing robust runner schema",
                "six-replica inference and CT review have not run",
            ],
            "explicit_non_claims": [
                "not automatic ink acceptance",
                "not automatic letter acceptance",
                "not a First Letters claim",
            ],
        },
    )
    return queue_path


def execute(
    root: Path,
    manifest_path: Path,
    *,
    cache_gb: int,
    inference_batch_size: int,
) -> dict[str, Any]:
    manifest, _ = flatten_manifest_tasks(root, manifest_path)
    if str(root.resolve()) != str(manifest["runtime"]["expected_root"]):
        raise RuntimeError("execute requires the exact frozen Vast repo root")
    plan = build_plan(root, manifest_path)
    if plan["status"] != "READY_TO_EXECUTE":
        raise RuntimeError("runtime hashes or binaries are not ready")
    check_no_competing_gpu_work()
    output_root = safe_path(root, str(manifest["runtime"]["output_root"]))
    lock = output_root / ".execution_lock"
    try:
        lock.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"official GP rescreen lock already exists: {lock}") from error
    try:
        for index, task in enumerate(plan["tasks"], start=1):
            print(
                f"MATERIALIZE:{index}/{len(plan['tasks'])}:"
                f"{task['campaign_sample_id']}:{task['surface_id']}:"
                f"{task['action']}",
                flush=True,
            )
            materialize_task(
                root,
                manifest,
                task,
                cache_gb=cache_gb,
            )
        ready_plan = build_plan(root, manifest_path)
        if (
            ready_plan["surface_count"] != EXPECTED_SURFACE_COUNT
            or ready_plan["action_counts"]["REUSE_EXACT_RENDER"]
            != EXPECTED_SURFACE_COUNT
        ):
            raise RuntimeError("post-materialization audit did not bind all 21 renders")
        plan_path = output_root / "PREFLIGHT_PLAN.json"
        write_json(plan_path, ready_plan)
        batch_root = stage_batch(root, manifest, ready_plan)
        batch_receipt = run_coarse_batch(
            root,
            manifest,
            ready_plan,
            batch_root,
            inference_batch_size=inference_batch_size,
        )
        v1_path, v2_path = run_rankings(
            root,
            manifest,
            batch_root,
            batch_receipt,
        )
        queue_path = freeze_robust_queue(
            root,
            manifest_path,
            manifest,
            batch_root,
            v1_path,
            v2_path,
        )
        result = {
            "kind": "campaign_x_phase4_official_gp_scroll1_rescreen_execution_v1",
            "status": "COMPLETED_COARSE_AND_RANKING_ROBUST_QUEUE_READY",
            "completed_at_utc": utc_now(),
            "surface_count": EXPECTED_SURFACE_COUNT,
            "batch_receipt": str(batch_receipt),
            "batch_receipt_sha256": sha256_file(batch_receipt),
            "ranking_v1": str(v1_path),
            "ranking_v1_sha256": sha256_file(v1_path),
            "ranking_v2": str(v2_path),
            "ranking_v2_sha256": sha256_file(v2_path),
            "robust_queue": str(queue_path),
            "robust_queue_sha256": sha256_file(queue_path),
            "robust_executed": False,
        }
        write_json(output_root / "EXECUTION_RECEIPT.json", result)
        return result
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--cache-gb", type=int, default=12)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else (
            root
            / "phase4"
            / "official_gp_scroll1_rescreen_v1"
            / "OFFICIAL_GP_SCROLL1_RESCREEN_MANIFEST.json"
        )
    )
    if args.dry_run:
        result = build_plan(root, manifest_path)
        if args.plan_output:
            write_json(args.plan_output.resolve(), result)
    else:
        result = execute(
            root,
            manifest_path,
            cache_gb=args.cache_gb,
            inference_batch_size=args.inference_batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
