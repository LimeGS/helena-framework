#!/usr/bin/env python3
"""Run the additive robust/CT/high-recall chain for official GP surfaces.

The upstream official rescreen intentionally stops after coarse inference and
freezes ``OFFICIAL_GP_SCROLL1_ROBUST_QUEUE_V1.json``.  This command is the
isolated second leg.  It waits for that upstream execution to become terminal,
runs the existing six-replica runner one frozen window at a time, applies the
already-frozen CT fiber gate, executes the additive high-recall router, and
writes one combined provenance/result receipt.

Dry-run is read-only unless ``--plan-output`` is supplied.  Execute is
fail-closed on source/hash/status drift, competing GPU work, rank gaps, partial
child receipts, or insufficient dynamic disk headroom.  It never changes the
historical expanded-candidate namespaces and never accepts ink or letters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


MANIFEST_KIND = "campaign_x_phase4_official_gp_scroll1_robust_chain_manifest_v1"
UPSTREAM_EXECUTION_KIND = "campaign_x_phase4_official_gp_scroll1_rescreen_execution_v1"
UPSTREAM_BATCH_KIND = "campaign_x_phase4_official_gp_scroll1_coarse_batch_v1"
QUEUE_KIND = "campaign_x_phase4_official_gp_scroll1_robust_queue_v1"
CHILD_KIND = "campaign_x_phase4_expanded_window_robust_batch_v1"
# The combined receipt deliberately remains on the established robust schema.
# Downstream CT and high-recall builders already fail closed on this kind; the
# official provenance is carried by ``scope`` and ``source_bindings``.
COMBINED_KIND = CHILD_KIND
RESULT_KIND = "campaign_x_phase4_official_gp_scroll1_robust_chain_result_v1"
RANKING_KIND = "campaign_x_phase4_expanded_surface_global_ranking_v1"
TERMINAL_CHILD_STATUSES = {
    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE",
    "COMPLETED_DIAGNOSTIC_ONLY",
}
COMPETING_GPU_PATTERNS = (
    "run_gp_shortlist_after_current.sh",
    "run_ink_timesformer.py",
    "run_expanded_robust_windows.py",
    "vc_render_tifxyz",
)
GIB = 1024**3


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError(f"configuration path must be relative: {value}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"configuration path escapes root: {value}") from error
    return result


def canonical_repository_root(root: Path) -> Path:
    """Resolve the archive's compatibility root to the real repository root."""

    if (root / ".git").exists():
        return root
    framework_link = root / "framework"
    if framework_link.exists():
        candidate = framework_link.resolve().parent
        if (candidate / ".git").exists():
            return candidate
    return root


def resolve_bound_runtime_path(
    root: Path,
    value: str,
    *,
    expected_root: str,
    relative_base: Path,
) -> Path:
    """Resolve receipts created at the frozen remote root on any repo clone."""

    candidate = Path(value)
    if not candidate.is_absolute():
        return (relative_base / candidate).resolve()
    frozen_root = Path(expected_root)
    try:
        relative = candidate.relative_to(frozen_root)
    except ValueError:
        if candidate.resolve().is_relative_to(root.resolve()):
            return candidate.resolve()
        raise RuntimeError(
            f"absolute receipt path is outside the frozen root: {value}"
        )
    return safe_path(root, str(relative))


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def load_configuration(
    root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    root = canonical_repository_root(root)
    manifest = read_json(manifest_path)
    if manifest.get("kind") != MANIFEST_KIND:
        raise RuntimeError("unexpected official robust chain manifest kind")
    if manifest.get("status") != "FROZEN_ADDITIVE_CHAIN_CONFIGURATION":
        raise RuntimeError("official robust chain manifest is not frozen")

    bindings: dict[str, Any] = {}
    for name, record in manifest["script_bindings"].items():
        path = safe_path(root, str(record["path"]))
        actual = sha256_file(path) if path.is_file() else None
        bindings[name] = {
            "path": str(path),
            "expected_sha256": record["sha256"],
            "actual_sha256": actual,
            "matches": actual == record["sha256"],
        }
    upstream_manifest = manifest["upstream"]["rescreen_manifest"]
    upstream_path = safe_path(root, str(upstream_manifest["path"]))
    upstream_actual = sha256_file(upstream_path) if upstream_path.is_file() else None
    gate = manifest["runtime"]["ct_fiber_gate"]
    gate_path = safe_path(root, str(gate["path"]))
    gate_actual = sha256_file(gate_path) if gate_path.is_file() else None
    checkpoint = manifest["runtime"]["checkpoint"]
    checkpoint_path = safe_path(root, str(checkpoint["path"]))
    checkpoint_actual = (
        sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    )
    audit = {
        "scripts": bindings,
        "upstream_manifest": {
            "path": str(upstream_path),
            "expected_sha256": upstream_manifest["sha256"],
            "actual_sha256": upstream_actual,
            "matches": upstream_actual == upstream_manifest["sha256"],
        },
        "ct_fiber_gate": {
            "path": str(gate_path),
            "expected_sha256": gate["sha256"],
            "actual_sha256": gate_actual,
            "matches": gate_actual == gate["sha256"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "expected_sha256": checkpoint["sha256"],
            "actual_sha256": checkpoint_actual,
            "exists": checkpoint_path.is_file(),
            "matches": checkpoint_actual == checkpoint["sha256"],
        },
    }
    required_local = [
        audit["upstream_manifest"]["matches"],
        audit["ct_fiber_gate"]["matches"],
        *(record["matches"] for record in bindings.values()),
    ]
    if not all(required_local):
        raise RuntimeError("frozen local script/input binding mismatch")
    return {"manifest": manifest, "runtime_audit": audit}


def _validate_queue_rows(
    queue: dict[str, Any],
    *,
    maximum: int,
) -> list[dict[str, Any]]:
    rows = queue.get("global_priority")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("official robust queue has no windows")
    if len(rows) > maximum:
        raise RuntimeError("official robust queue exceeds frozen maximum")
    ranks = [int(row["global_rank"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if ranks != expected:
        raise RuntimeError(f"official robust queue rank gap: {ranks} != {expected}")
    identities: set[tuple[str, str, tuple[int, ...]]] = set()
    for row in rows:
        crop = tuple(int(value) for value in row["source_crop_xyxy"])
        if len(crop) != 4:
            raise RuntimeError("queue crop must have four coordinates")
        identity = (
            str(row["sample_id"]),
            str(row["surface_id"]),
            crop,
        )
        if identity in identities:
            raise RuntimeError(f"duplicate official robust queue identity: {identity}")
        identities.add(identity)
        if "score" not in row:
            raise RuntimeError("queue row lacks robust-runner score adapter")
    return rows


def inspect_upstream(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    upstream = manifest["upstream"]
    execution_path = safe_path(root, str(upstream["coarse_execution_receipt"]))
    batch_path = safe_path(root, str(upstream["coarse_batch_receipt"]))
    queue_path = safe_path(root, str(upstream["queue"]))
    missing = [
        str(path)
        for path in (execution_path, batch_path, queue_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "state": "WAITING_UPSTREAM",
            "missing": missing,
            "execution_receipt": str(execution_path),
            "coarse_batch_receipt": str(batch_path),
            "queue": str(queue_path),
        }

    execution = read_json(execution_path)
    if (
        execution.get("kind") != UPSTREAM_EXECUTION_KIND
        or execution.get("status") != "COMPLETED_COARSE_AND_RANKING_ROBUST_QUEUE_READY"
        or int(execution.get("surface_count", -1))
        != int(upstream["required_surface_count"])
        or execution.get("robust_executed") is not False
    ):
        raise RuntimeError("official coarse execution receipt is not terminal-valid")
    batch = read_json(batch_path)
    if (
        batch.get("kind") != UPSTREAM_BATCH_KIND
        or batch.get("status") != "COMPLETED_PRIORITIZATION_ONLY"
        or int(batch.get("task_count", -1)) != int(upstream["required_surface_count"])
        or int(batch.get("completed_count", -1))
        != int(upstream["required_surface_count"])
        or int(batch.get("failed_count", -1)) != 0
    ):
        raise RuntimeError("official coarse batch receipt is not terminal-valid")
    queue = read_json(queue_path)
    if (
        queue.get("kind") != QUEUE_KIND
        or queue.get("status") != "COMPLETED_PRIORITIZATION_ONLY"
        or queue.get("queue_state") != "READY_FOR_ROBUST_EXECUTION"
    ):
        raise RuntimeError("official robust queue is not terminal-ready")

    expected_queue_hash = execution.get("robust_queue_sha256")
    actual_queue_hash = sha256_file(queue_path)
    execution_queue = resolve_bound_runtime_path(
        root,
        str(execution["robust_queue"]),
        expected_root=str(manifest["runtime"].get("expected_root", root)),
        relative_base=execution_path.parent,
    )
    if (
        execution_queue.resolve() != queue_path.resolve()
        or expected_queue_hash != actual_queue_hash
    ):
        raise RuntimeError("execution receipt does not bind the exact robust queue")
    if (
        batch["checkpoint"]["sha256"] != manifest["runtime"]["checkpoint"]["sha256"]
        or queue["checkpoint"]["sha256"] != manifest["runtime"]["checkpoint"]["sha256"]
        or queue["checkpoint"]["model_family"]
        != manifest["runtime"]["checkpoint"]["model_family"]
        or queue["ct_fiber_gate"]["sha256"]
        != manifest["runtime"]["ct_fiber_gate"]["sha256"]
    ):
        raise RuntimeError("upstream checkpoint or CT gate binding mismatch")
    robust_parameters = queue.get("robust_parameters", {})
    frozen_robust = manifest["parameters"]["robust"]
    for key in (
        "depth_centers",
        "tiling_offsets",
        "frames",
        "tile_size",
        "stride",
        "min_valid_ratio",
    ):
        if robust_parameters.get(key) != frozen_robust[key]:
            raise RuntimeError(f"queue robust parameter mismatch: {key}")
    queue_runner_hash = queue.get("script_hashes", {}).get(
        "run_expanded_robust_windows.py"
    )
    if queue_runner_hash != manifest["script_bindings"]["robust_runner"]["sha256"]:
        raise RuntimeError("queue robust-runner script binding mismatch")

    queue_manifest = queue["manifest"]
    frozen_upstream = manifest["upstream"]["rescreen_manifest"]
    if (
        queue_manifest["sha256"] != frozen_upstream["sha256"]
        or sha256_file(safe_path(root, frozen_upstream["path"]))
        != frozen_upstream["sha256"]
    ):
        raise RuntimeError("queue upstream manifest binding mismatch")
    for key in ("ranking_v1", "ranking_v2"):
        record = queue[key]
        path = resolve_bound_runtime_path(
            root,
            str(record["path"]),
            expected_root=str(manifest["runtime"].get("expected_root", root)),
            relative_base=queue_path.parent,
        )
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"queue {key} hash binding mismatch")
    rows = _validate_queue_rows(
        queue,
        maximum=int(upstream["maximum_queue_windows"]),
    )
    return {
        "state": "READY",
        "execution_receipt": str(execution_path),
        "execution_receipt_sha256": sha256_file(execution_path),
        "coarse_batch_receipt": str(batch_path),
        "coarse_batch_receipt_sha256": sha256_file(batch_path),
        "queue": str(queue_path),
        "queue_sha256": actual_queue_hash,
        "queue_window_count": len(rows),
        "rows": rows,
    }


def build_plan(root: Path, manifest_path: Path) -> dict[str, Any]:
    configuration = load_configuration(root, manifest_path)
    manifest = configuration["manifest"]
    upstream = inspect_upstream(root, manifest)
    checkpoint_ready = configuration["runtime_audit"]["checkpoint"]["matches"]
    if upstream["state"] == "WAITING_UPSTREAM":
        status = "WAITING_FOR_OFFICIAL_COARSE_QUEUE"
    elif not checkpoint_ready:
        status = "RUNTIME_CHECKPOINT_NOT_MATERIALIZED"
    else:
        status = "READY_FOR_ISOLATED_ROBUST_CHAIN"
    return {
        "kind": "campaign_x_phase4_official_gp_scroll1_robust_chain_plan_v1",
        "status": status,
        "generated_at_utc": utc_now(),
        "root": str(root.resolve()),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_audit": configuration["runtime_audit"],
        "upstream": upstream,
        "will_write_only_under": str(
            safe_path(root, manifest["runtime"]["chain_root"])
        ),
        "robust_output_root": str(
            safe_path(root, manifest["runtime"]["batch_root"])
            / manifest["runtime"]["robust_root_name"]
        ),
        "stages": [
            "wait for terminal official coarse execution, batch receipt and queue",
            "validate exact manifests, hashes, checkpoint, CT gate and contiguous queue",
            "run one six-replica robust window per child receipt with a disk guard",
            "combine child receipts without changing their strict v1 decisions",
            "apply the frozen CT depth-localization gate",
            "route additive 2-of-3 high-recall components",
            "write one combined review-only result and provenance receipt",
        ],
        "dry_run_mutates_runtime": False,
        "explicit_non_claims": manifest["explicit_non_claims"],
    }


def wait_for_upstream(
    root: Path,
    manifest: dict[str, Any],
    *,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = inspect_upstream(root, manifest)
        if state["state"] == "READY":
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out waiting for official coarse/queue terminal receipts"
            )
        print(
            "WAITING_UPSTREAM:"
            + ",".join(Path(path).name for path in state["missing"]),
            flush=True,
        )
        time.sleep(poll_seconds)


def check_no_competing_gpu_work() -> None:
    own_pid = os.getpid()
    conflicts: list[str] = []
    for pattern in COMPETING_GPU_PATTERNS:
        completed = subprocess.run(
            ["pgrep", "-f", pattern],
            text=True,
            capture_output=True,
            check=False,
        )
        pids = sorted(
            {
                int(value)
                for value in completed.stdout.split()
                if value.isdigit() and int(value) != own_pid
            }
        )
        if pids:
            conflicts.append(f"{pattern}:{pids}")
    if conflicts:
        raise RuntimeError(
            "competing GPU/render work detected; refusing to overlap: "
            + "; ".join(conflicts)
        )


def source_tiff_stack(
    batch_root: Path,
    row: dict[str, Any],
) -> tuple[list[Path], str]:
    directory = batch_root / str(row["sample_id"]) / str(row["surface_id"]) / "tiffs"
    paths, slice_ordering = ordered_tiff_files(directory)
    if len(paths) != 65 or not all(path.stat().st_size > 0 for path in paths):
        raise RuntimeError(f"source stack is not exact 65 TIFFs: {directory}")
    return paths, slice_ordering


def estimate_window_output_bytes(
    *,
    batch_root: Path,
    robust_root: Path,
    row: dict[str, Any],
    voxel_um: float,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    source, slice_ordering = source_tiff_stack(batch_root, row)
    with Image.open(source[0]) as image:
        width, height = image.size
    source_bytes = sum(path.stat().st_size for path in source)
    crop_pixels = max(1, round(20_000.0 / voxel_um))
    area_ratio = min(1.0, (crop_pixels * crop_pixels) / max(1, width * height))
    projected = int(
        source_bytes * area_ratio * float(parameters["source_crop_estimate_multiplier"])
    )
    fallback = int(float(parameters["fallback_window_output_gib"]) * GIB)
    estimate = max(projected, fallback)
    rank = int(row["global_rank"])
    output = robust_root / f"rank-{rank:02d}-{row['sample_id']}-{row['surface_id']}"
    existing = directory_size(output)
    remaining = max(0, estimate - existing)
    return {
        "source_stack_bytes": source_bytes,
        "slice_ordering": slice_ordering,
        "source_shape_y_x": [height, width],
        "crop_pixels": crop_pixels,
        "area_ratio": area_ratio,
        "estimated_total_window_bytes": estimate,
        "existing_window_bytes": existing,
        "estimated_remaining_window_bytes": remaining,
        "output": str(output),
    }


def disk_guard(
    *,
    filesystem_path: Path,
    estimate: dict[str, Any],
    minimum_free_after_gib: float,
    minimum_working_headroom_gib: float,
    stage: str,
) -> dict[str, Any]:
    usage = shutil.disk_usage(filesystem_path)
    reserve = int(minimum_free_after_gib * GIB)
    working = int(minimum_working_headroom_gib * GIB)
    projected_new = int(estimate.get("estimated_remaining_window_bytes", 0))
    required_now = reserve + max(working, projected_new)
    result = {
        "stage": stage,
        "checked_at_utc": utc_now(),
        "filesystem_path": str(filesystem_path),
        "free_bytes": usage.free,
        "free_gib": usage.free / GIB,
        "minimum_free_after_bytes": reserve,
        "minimum_working_headroom_bytes": working,
        "projected_new_bytes": projected_new,
        "required_free_now_bytes": required_now,
        "passes": usage.free >= required_now,
        "estimate": estimate,
    }
    if not result["passes"]:
        raise RuntimeError(
            "disk guard blocked "
            f"{stage}: free={usage.free / GIB:.2f} GiB, "
            f"required={required_now / GIB:.2f} GiB"
        )
    return result


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            check=True,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def child_receipt_path(
    robust_root: Path,
    rank: int,
) -> Path:
    return robust_root / f"RANK_{rank:03d}_ROBUST_RECEIPT.json"


def write_ranking_adapter(
    *,
    output: Path,
    queue_path: Path,
    queue: dict[str, Any],
) -> dict[str, Any]:
    """Adapt the official queue to the frozen robust/routing ranking schema."""

    payload = {
        "kind": RANKING_KIND,
        "status": "COMPLETED_PRIORITIZATION_ONLY",
        "scope": "SCHEMA_ADAPTER_FOR_OFFICIAL_GP_SCROLL1_ROBUST_QUEUE_V1",
        "source_official_queue": {
            "path": str(queue_path),
            "sha256": sha256_file(queue_path),
            "kind": queue["kind"],
            "queue_state": queue["queue_state"],
        },
        "global_priority": queue["global_priority"],
        "search": queue["search"],
        "policy": [
            "global_priority rows are copied without modification",
            "the adapter changes schema kind only for existing robust/router consumers",
            "the source official queue remains the scientific prioritization authority",
        ],
        "explicit_non_claims": queue["explicit_non_claims"],
    }
    if output.is_file():
        existing = read_json(output)
        if existing != payload:
            raise RuntimeError("existing official ranking adapter differs")
    else:
        write_json(output, payload)
    return payload


def validate_child_receipt(
    path: Path,
    *,
    rank: int,
    queue_path: Path,
    checkpoint_sha256: str,
    screening_name: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"robust child receipt is missing: {path}")
    receipt = read_json(path)
    if (
        receipt.get("kind") != CHILD_KIND
        or receipt.get("status") not in TERMINAL_CHILD_STATUSES
        or int(receipt.get("completed_count", -1)) != 1
        or receipt.get("selected_global_ranks") != [rank]
        or receipt.get("screening_name") != screening_name
        or receipt["checkpoint"]["sha256"] != checkpoint_sha256
        or Path(str(receipt["global_ranking"])).resolve() != queue_path.resolve()
        or receipt["global_ranking_sha256"] != sha256_file(queue_path)
        or len(receipt.get("results", [])) != 1
        or int(receipt["results"][0]["global_rank"]) != rank
    ):
        raise RuntimeError(f"robust child receipt is not terminal-valid: {path}")
    return receipt


def combine_child_receipts(
    *,
    output: Path,
    queue_path: Path,
    upstream: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    child_paths: list[Path],
    checkpoint_path: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    checkpoint = manifest["runtime"]["checkpoint"]
    screening = str(manifest["runtime"]["screening_name"])
    for rank, path in enumerate(child_paths, start=1):
        receipt = validate_child_receipt(
            path,
            rank=rank,
            queue_path=queue_path,
            checkpoint_sha256=str(checkpoint["sha256"]),
            screening_name=screening,
        )
        results.append(receipt["results"][0])
        children.append(
            {
                "global_rank": rank,
                "path": str(path),
                "sha256": sha256_file(path),
                "status": receipt["status"],
            }
        )
    if [int(row["global_rank"]) for row in results] != list(range(1, len(results) + 1)):
        raise RuntimeError("combined robust child ranks are not contiguous")
    positives = sum(row["route"] == "RAW_CT_REVIEW_REQUIRED" for row in results)
    if output.is_file():
        existing = read_json(output)
        if (
            existing.get("kind") != COMBINED_KIND
            or existing.get("global_ranking_sha256") != sha256_file(queue_path)
            or existing.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]
            or existing.get("completed_count") != len(results)
            or existing.get("selected_global_ranks") != list(range(1, len(results) + 1))
            or existing.get("results") != results
            or existing.get("child_receipts") != children
            or existing.get("source_bindings", {})
            .get("chain_manifest", {})
            .get("sha256")
            != sha256_file(manifest_path)
        ):
            raise RuntimeError("existing combined robust receipt differs")
        return existing
    payload = {
        "kind": COMBINED_KIND,
        "scope": "OFFICIAL_GP_SCROLL1_COMBINED_ROBUST_BATCH_V1",
        "status": (
            "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE"
            if positives
            else "COMPLETED_DIAGNOSTIC_ONLY"
        ),
        "updated_at_utc": utc_now(),
        "global_ranking": str(queue_path),
        "global_ranking_sha256": sha256_file(queue_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint["sha256"],
            "model_family": checkpoint["model_family"],
        },
        "screening_name": screening,
        "requested_start_rank": 1,
        "requested_limit": len(results),
        "selected_global_ranks": list(range(1, len(results) + 1)),
        "completed_count": len(results),
        "text_like_positive_count": positives,
        "stopped_on_text_like": False,
        "continue_after_text_like": True,
        "results": results,
        "child_receipts": children,
        "source_bindings": {
            "chain_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "upstream_execution_receipt": {
                "path": upstream["execution_receipt"],
                "sha256": upstream["execution_receipt_sha256"],
            },
            "upstream_coarse_batch_receipt": {
                "path": upstream["coarse_batch_receipt"],
                "sha256": upstream["coarse_batch_receipt_sha256"],
            },
            "upstream_queue": {
                "path": upstream["queue"],
                "sha256": upstream["queue_sha256"],
            },
        },
        "policy": [
            "three depths and two offsets remain frozen",
            "strict row-gate results are preserved byte-for-byte from child receipts",
            "all queued windows complete even if a strict text-like route is observed",
            "CT gate and high-recall routing are later additive review channels",
        ],
        "explicit_non_claims": manifest["explicit_non_claims"],
    }
    write_json(output, payload)
    return payload


def load_ct_decisions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError("CT gate decision artifact must be a JSON list")
    return value


def count_strict_candidates(combined: dict[str, Any]) -> int:
    count = 0
    for result in combined["results"]:
        analysis_path = Path(str(result["analysis"])).resolve()
        if (
            not analysis_path.is_file()
            or sha256_file(analysis_path) != result["analysis_sha256"]
        ):
            raise RuntimeError("strict analysis changed before CT application")
        analysis = read_json(analysis_path)
        candidates = analysis.get("text_like_screening", {}).get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError("strict analysis candidate inventory is missing")
        count += len(candidates)
    return count


def write_empty_ct_application(
    *,
    spec_path: Path,
    features_root: Path,
    gate_root: Path,
    gate_path: Path,
) -> None:
    """Record the vacuous frozen-gate application when strict found no forms."""

    features_root.mkdir(parents=True, exist_ok=True)
    gate_root.mkdir(parents=True, exist_ok=True)
    csv_path = features_root / "CT_FIBER_FEATURES.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["group_id", "class", "candidate_id"])
    write_json(
        features_root / "CT_FIBER_FEATURE_BENCHMARK.json",
        {
            "kind": "campaign_x_phase4_ct_fiber_feature_benchmark_v1",
            "status": "COMPLETED_NO_STRICT_COMPONENTS",
            "generated_at_utc": utc_now(),
            "spec": str(spec_path),
            "spec_sha256": sha256_file(spec_path),
            "row_count": 0,
            "feature_names": [],
            "sources": [],
            "summary": {},
            "artifacts": {
                "csv": csv_path.name,
                "csv_sha256": sha256_file(csv_path),
                "csv_size_bytes": csv_path.stat().st_size,
            },
            "non_claims": [
                "no strict components existed to which the CT rule could apply",
                "high-recall routing remains required",
                "not evidence of absence of ink",
            ],
        },
    )
    decisions_path = gate_root / "CT_FIBER_GATE_DECISIONS.json"
    decisions_path.write_text("[]\n", encoding="utf-8")
    write_json(
        gate_root / "CT_FIBER_GATE_EVALUATION.json",
        {
            "kind": "campaign_x_phase4_ct_fiber_gate_evaluation_v1",
            "status": "COMPLETED",
            "generated_at_utc": utc_now(),
            "features": str(csv_path),
            "features_sha256": sha256_file(csv_path),
            "rule": str(gate_path),
            "rule_sha256": sha256_file(gate_path),
            "row_count": 0,
            "retained_count": 0,
            "downranked_count": 0,
            "by_group": {},
            "by_class": {},
            "artifacts": {
                "decisions": decisions_path.name,
                "decisions_sha256": sha256_file(decisions_path),
            },
            "interpretation": (
                "The frozen CT rule had no strict-v1 components to evaluate. "
                "This is a vacuous application, not evidence that the windows "
                "lack ink; the additive high-recall channel still runs."
            ),
        },
    )


def validate_postprocess_bindings(
    *,
    root: Path,
    manifest: dict[str, Any],
    combined_receipt_path: Path,
    ct_spec_path: Path,
    ct_features_path: Path,
    ct_feature_receipt_path: Path,
    ct_evaluation_path: Path,
    ct_decisions_path: Path,
    router_manifest_path: Path,
    router_receipt_path: Path,
) -> None:
    """Reject stale or partially resumed CT/router artifacts."""

    spec = read_json(ct_spec_path)
    feature_receipt = read_json(ct_feature_receipt_path)
    evaluation = read_json(ct_evaluation_path)
    decisions = load_ct_decisions(ct_decisions_path)
    router_manifest = read_json(router_manifest_path)
    router_receipt = read_json(router_receipt_path)
    combined_hash = sha256_file(combined_receipt_path)
    gate_path = safe_path(root, manifest["runtime"]["ct_fiber_gate"]["path"])
    gate_hash = sha256_file(gate_path)
    if (
        spec.get("source_batch_receipt_sha256") != combined_hash
        or spec.get("policy", {}).get("gate_freeze_sha256") != gate_hash
        or feature_receipt.get("spec_sha256") != sha256_file(ct_spec_path)
        or feature_receipt.get("artifacts", {}).get("csv_sha256")
        != sha256_file(ct_features_path)
        or evaluation.get("features_sha256") != sha256_file(ct_features_path)
        or evaluation.get("rule_sha256") != gate_hash
        or int(evaluation.get("row_count", -1)) != len(decisions)
        or int(evaluation.get("retained_count", -1))
        != sum(bool(row.get("retained")) for row in decisions)
        or int(evaluation.get("downranked_count", -1))
        != sum(not bool(row.get("retained")) for row in decisions)
    ):
        raise RuntimeError("CT postprocess provenance binding mismatch")
    source_robust = router_manifest.get("source_robust_receipt", {})
    if (
        router_manifest.get("status") != "READY_FOR_HIGH_RECALL_CT_ROUTING"
        or source_robust.get("sha256") != combined_hash
        or router_receipt.get("input_manifest_sha256")
        != sha256_file(router_manifest_path)
        or int(router_receipt.get("window_count", -1))
        != int(source_robust.get("completed_count", -2))
        or int(router_receipt.get("ct_review_queue_count", -1))
        != len(router_receipt.get("ct_review_queue", []))
    ):
        raise RuntimeError("high-recall router provenance binding mismatch")


def build_combined_result(
    *,
    output: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    combined_receipt_path: Path,
    ct_spec_path: Path,
    ct_features_path: Path,
    ct_evaluation_path: Path,
    ct_decisions_path: Path,
    router_manifest_path: Path,
    router_receipt_path: Path,
    disk_audit_path: Path,
) -> dict[str, Any]:
    combined = read_json(combined_receipt_path)
    ct_evaluation = read_json(ct_evaluation_path)
    ct_decisions = load_ct_decisions(ct_decisions_path)
    router = read_json(router_receipt_path)
    if (
        ct_evaluation.get("status") != "COMPLETED"
        or router.get("status") != "COMPLETED_DIAGNOSTIC_ROUTING_ONLY"
        or int(router.get("window_count", -1)) != int(combined["completed_count"])
    ):
        raise RuntimeError("CT gate or high-recall router is not terminal-valid")

    ct_by_group: dict[str, list[dict[str, Any]]] = {}
    for decision in ct_decisions:
        ct_by_group.setdefault(str(decision["group_id"]), []).append(decision)
    high_by_window: dict[str, list[dict[str, Any]]] = {}
    for item in router["ct_review_queue"]:
        high_by_window.setdefault(str(item["window_id"]), []).append(item)

    expected_groups = {
        f"rank-{int(row['global_rank']):02d}-{row['sample_id']}-{row['surface_id']}"
        for row in combined["results"]
    }
    expected_windows = set(expected_groups)
    unknown_groups = set(ct_by_group) - expected_groups
    unknown_windows = set(high_by_window) - expected_windows
    if unknown_groups or unknown_windows:
        raise RuntimeError(
            "postprocess result references an unknown robust window: "
            f"ct={sorted(unknown_groups)}, high_recall={sorted(unknown_windows)}"
        )

    windows: list[dict[str, Any]] = []
    combined_queue: list[dict[str, Any]] = []
    for result in combined["results"]:
        rank = int(result["global_rank"])
        group = f"rank-{rank:02d}-{result['sample_id']}-{result['surface_id']}"
        window_id = f"rank-{rank:02d}-{result['sample_id']}-{result['surface_id']}"
        decisions = ct_by_group.get(group, [])
        retained = [row for row in decisions if row["retained"]]
        high = high_by_window.get(window_id, [])
        for decision in retained:
            combined_queue.append(
                {
                    "channel": "STRICT_V1_COMPONENT_RETAINED_BY_FROZEN_CT_GATE",
                    "global_rank": rank,
                    "sample_id": result["sample_id"],
                    "surface_id": result["surface_id"],
                    "candidate_id": decision["candidate_id"],
                    "review_outcome": "QUEUE_FOR_ORTHOGONAL_CT_REVIEW",
                }
            )
        for item in high:
            combined_queue.append(
                {
                    "channel": "ADDITIVE_HIGH_RECALL_2_OF_3",
                    "global_rank": rank,
                    "sample_id": result["sample_id"],
                    "surface_id": result["surface_id"],
                    "candidate_id": item["candidate_id"],
                    "center_y_x": item["center_y_x"],
                    "bbox_y0_x0_y1_x1": item["bbox_y0_x0_y1_x1"],
                    "routing_score": item["routing_score"],
                    "review_outcome": "QUEUE_FOR_RAW_CT_LOCALIZATION_REVIEW",
                }
            )
        windows.append(
            {
                "global_rank": rank,
                "sample_id": result["sample_id"],
                "surface_id": result["surface_id"],
                "source_crop_xyxy": result["source_crop_xyxy"],
                "strict_v1_route": result["route"],
                "strict_candidate_count": len(decisions),
                "ct_retained_count": len(retained),
                "ct_downranked_count": len(decisions) - len(retained),
                "high_recall_routed_count": len(high),
            }
        )

    artifacts = {
        "combined_robust_receipt": combined_receipt_path,
        "ct_application_spec": ct_spec_path,
        "ct_features": ct_features_path,
        "ct_gate_evaluation": ct_evaluation_path,
        "ct_gate_decisions": ct_decisions_path,
        "high_recall_manifest": router_manifest_path,
        "high_recall_receipt": router_receipt_path,
        "disk_guard_audit": disk_audit_path,
    }
    payload = {
        "kind": RESULT_KIND,
        "status": "COMPLETED_DIAGNOSTIC_REVIEW_QUEUES_ONLY",
        "completed_at_utc": utc_now(),
        "scope": "OFFICIAL_GP_SCROLL1_ROBUST_CT_AND_HIGH_RECALL",
        "chain_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
        "window_count": len(windows),
        "strict_text_like_window_count": int(combined["text_like_positive_count"]),
        "ct_gate": {
            "component_count": int(ct_evaluation["row_count"]),
            "retained_count": int(ct_evaluation["retained_count"]),
            "downranked_count": int(ct_evaluation["downranked_count"]),
        },
        "high_recall": {
            "routed_component_count": int(router["ct_review_queue_count"]),
            "window_count": int(router["window_count"]),
            "scroll_count": int(router["scroll_count"]),
        },
        "combined_review_queue_count": len(combined_queue),
        "combined_review_queue": combined_queue,
        "windows": windows,
        "interpretation": (
            "The two channels are intentionally not deduplicated by approximate "
            "geometry. Every entry is a review route only. CT retention does not "
            "accept ink, and CT downranking does not reject a scroll."
        ),
        "explicit_non_claims": manifest["explicit_non_claims"],
    }
    write_json(output, payload)
    return payload


def execute(
    root: Path,
    manifest_path: Path,
    *,
    wait_timeout_seconds: int,
    poll_seconds: int,
    inference_batch_size: int | None,
) -> dict[str, Any]:
    configuration = load_configuration(root, manifest_path)
    manifest = configuration["manifest"]
    if str(root.resolve()) != str(manifest["runtime"]["expected_root"]):
        raise RuntimeError("execute requires the exact frozen Vast root")
    if not configuration["runtime_audit"]["checkpoint"]["matches"]:
        raise RuntimeError("frozen checkpoint is absent or drifted")
    upstream = wait_for_upstream(
        root,
        manifest,
        timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
    )
    check_no_competing_gpu_work()

    chain_root = safe_path(root, str(manifest["runtime"]["chain_root"]))
    batch_root = safe_path(root, str(manifest["runtime"]["batch_root"]))
    robust_root = batch_root / str(manifest["runtime"]["robust_root_name"])
    queue_path = Path(upstream["queue"]).resolve()
    queue = read_json(queue_path)
    checkpoint = safe_path(root, str(manifest["runtime"]["checkpoint"]["path"]))
    lock = chain_root / ".execution_lock"
    chain_root.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"official robust chain lock exists: {lock}") from error

    disk_audits: list[dict[str, Any]] = []
    disk_audit_path = chain_root / "DISK_GUARD_AUDIT.json"
    try:
        ranking_adapter_path = (
            chain_root / "OFFICIAL_GP_SCROLL1_ROBUST_RANKING_ADAPTER.json"
        )
        write_ranking_adapter(
            output=ranking_adapter_path,
            queue_path=queue_path,
            queue=queue,
        )
        rows = upstream["rows"]
        child_paths: list[Path] = []
        disk_parameters = manifest["parameters"]["disk_guard"]
        batch_size = (
            inference_batch_size
            if inference_batch_size is not None
            else int(manifest["parameters"]["robust"]["inference_batch_size"])
        )
        for row in rows:
            rank = int(row["global_rank"])
            child_path = child_receipt_path(robust_root, rank)
            try:
                validate_child_receipt(
                    child_path,
                    rank=rank,
                    queue_path=ranking_adapter_path,
                    checkpoint_sha256=manifest["runtime"]["checkpoint"]["sha256"],
                    screening_name=manifest["runtime"]["screening_name"],
                )
                child_paths.append(child_path)
                print(f"ROBUST_REUSE:rank-{rank:02d}", flush=True)
                continue
            except RuntimeError:
                if child_path.exists():
                    raise
            lock_record = read_json(
                root / "phase4" / "targets" / str(row["sample_id"]) / "TARGET_LOCK.json"
            )
            estimate = estimate_window_output_bytes(
                batch_root=batch_root,
                robust_root=robust_root,
                row=row,
                voxel_um=float(lock_record["voxel_size_um"]),
                parameters=disk_parameters,
            )
            audit = disk_guard(
                filesystem_path=chain_root,
                estimate=estimate,
                minimum_free_after_gib=float(
                    disk_parameters["minimum_free_after_window_gib"]
                ),
                minimum_working_headroom_gib=float(
                    disk_parameters["minimum_working_headroom_gib"]
                ),
                stage=f"ROBUST_RANK_{rank:03d}",
            )
            disk_audits.append(audit)
            write_json(
                disk_audit_path,
                {
                    "kind": "campaign_x_phase4_dynamic_disk_guard_audit_v1",
                    "status": "PASS_SO_FAR",
                    "updated_at_utc": utc_now(),
                    "checks": disk_audits,
                },
            )
            check_no_competing_gpu_work()
            print(f"ROBUST_START:rank-{rank:02d}", flush=True)
            run_logged(
                [
                    sys.executable,
                    str(
                        safe_path(
                            root,
                            manifest["script_bindings"]["robust_runner"]["path"],
                        )
                    ),
                    "--root",
                    str(root),
                    "--batch-root",
                    str(batch_root),
                    "--ranking-path",
                    str(ranking_adapter_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--model-family",
                    str(manifest["runtime"]["checkpoint"]["model_family"]),
                    "--screening-name",
                    str(manifest["runtime"]["screening_name"]),
                    "--batch-receipt-name",
                    child_path.name,
                    "--robust-root-name",
                    str(manifest["runtime"]["robust_root_name"]),
                    "--continue-after-text-like",
                    "--inference-batch-size",
                    str(batch_size),
                    "--start-rank",
                    str(rank),
                    "--limit",
                    "1",
                ],
                chain_root / "logs" / f"rank-{rank:03d}.stdout.log",
            )
            validate_child_receipt(
                child_path,
                rank=rank,
                queue_path=ranking_adapter_path,
                checkpoint_sha256=manifest["runtime"]["checkpoint"]["sha256"],
                screening_name=manifest["runtime"]["screening_name"],
            )
            child_paths.append(child_path)
            print(f"ROBUST_DONE:rank-{rank:02d}", flush=True)

        write_json(
            disk_audit_path,
            {
                "kind": "campaign_x_phase4_dynamic_disk_guard_audit_v1",
                "status": "ROBUST_WINDOWS_COMPLETE",
                "updated_at_utc": utc_now(),
                "checks": disk_audits,
            },
        )
        combined_receipt_path = (
            robust_root / manifest["runtime"]["combined_robust_receipt_name"]
        )
        combined = combine_child_receipts(
            output=combined_receipt_path,
            queue_path=ranking_adapter_path,
            upstream=upstream,
            manifest_path=manifest_path,
            manifest=manifest,
            child_paths=child_paths,
            checkpoint_path=checkpoint,
        )

        postprocess_audit = disk_guard(
            filesystem_path=chain_root,
            estimate={"estimated_remaining_window_bytes": 0},
            minimum_free_after_gib=float(
                disk_parameters["minimum_free_after_window_gib"]
            ),
            minimum_working_headroom_gib=float(
                disk_parameters["postprocess_working_headroom_gib"]
            ),
            stage="CT_AND_HIGH_RECALL_POSTPROCESS",
        )
        disk_audits.append(postprocess_audit)
        write_json(
            disk_audit_path,
            {
                "kind": "campaign_x_phase4_dynamic_disk_guard_audit_v1",
                "status": "ALL_CHECKS_PASSED",
                "updated_at_utc": utc_now(),
                "checks": disk_audits,
            },
        )

        ct_root = chain_root / "ct_fiber_application_v1"
        ct_spec_path = ct_root / "OFFICIAL_GP_SCROLL1_CT_FIBER_SPEC.json"
        ct_features_root = ct_root / "features"
        ct_features_path = ct_features_root / "CT_FIBER_FEATURES.csv"
        ct_feature_receipt_path = ct_features_root / "CT_FIBER_FEATURE_BENCHMARK.json"
        ct_gate_root = ct_root / "gate"
        ct_evaluation_path = ct_gate_root / "CT_FIBER_GATE_EVALUATION.json"
        ct_decisions_path = ct_gate_root / "CT_FIBER_GATE_DECISIONS.json"
        gate_path = safe_path(
            root,
            manifest["runtime"]["ct_fiber_gate"]["path"],
        )
        if not ct_spec_path.is_file():
            run_logged(
                [
                    sys.executable,
                    str(
                        safe_path(
                            root,
                            manifest["script_bindings"]["ct_spec_builder"]["path"],
                        )
                    ),
                    "--root",
                    str(root),
                    "--batch-root",
                    str(batch_root),
                    "--batch-receipt",
                    str(combined_receipt_path),
                    "--gate-freeze",
                    str(gate_path),
                    "--output",
                    str(ct_spec_path),
                    "--application-name",
                    str(manifest["parameters"]["ct_fiber"]["application_name"]),
                    "--robust-root-name",
                    str(manifest["runtime"]["robust_root_name"]),
                    "--central-slice",
                    str(manifest["parameters"]["ct_fiber"]["central_slice"]),
                    "--patch-radius-um",
                    str(manifest["parameters"]["ct_fiber"]["patch_radius_um"]),
                ],
                chain_root / "logs" / "ct_spec.stdout.log",
            )
        if not ct_evaluation_path.is_file():
            if count_strict_candidates(combined) == 0:
                write_empty_ct_application(
                    spec_path=ct_spec_path,
                    features_root=ct_features_root,
                    gate_root=ct_gate_root,
                    gate_path=gate_path,
                )
            else:
                run_logged(
                    [
                        sys.executable,
                        str(
                            safe_path(
                                root,
                                manifest["script_bindings"]["ct_feature_extractor"][
                                    "path"
                                ],
                            )
                        ),
                        "--root",
                        str(root),
                        "--spec",
                        str(ct_spec_path),
                        "--output",
                        str(ct_features_root),
                    ],
                    chain_root / "logs" / "ct_features.stdout.log",
                )
                run_logged(
                    [
                        sys.executable,
                        str(
                            safe_path(
                                root,
                                manifest["script_bindings"]["ct_gate_application"][
                                    "path"
                                ],
                            )
                        ),
                        "--features",
                        str(ct_features_path),
                        "--rule",
                        str(gate_path),
                        "--output",
                        str(ct_gate_root),
                    ],
                    chain_root / "logs" / "ct_gate.stdout.log",
                )
        if not all(
            path.is_file()
            for path in (
                ct_spec_path,
                ct_features_path,
                ct_feature_receipt_path,
                ct_evaluation_path,
                ct_decisions_path,
            )
        ):
            raise RuntimeError("CT gate postprocess artifacts are incomplete")

        router_root = chain_root / "high_recall_router_v1"
        router_manifest_path = router_root / "HIGH_RECALL_ROUTER_MANIFEST.json"
        router_receipt_path = router_root / "HIGH_RECALL_CT_ROUTER_RECEIPT.json"
        if not router_receipt_path.is_file():
            builder = safe_path(
                root,
                manifest["script_bindings"]["high_recall_manifest_builder"]["path"],
            )
            builder_command = [
                sys.executable,
                str(builder),
                "--robust-receipt",
                str(combined_receipt_path),
                "--output",
                str(router_manifest_path),
                "--expected-window-count",
                str(combined["completed_count"]),
                "--expected-model-family",
                str(manifest["runtime"]["checkpoint"]["model_family"]),
                "--expected-screening-name",
                str(manifest["runtime"]["screening_name"]),
                "--expected-receipt-sha256",
                sha256_file(combined_receipt_path),
                "--expected-checkpoint-sha256",
                str(manifest["runtime"]["checkpoint"]["sha256"]),
                "--link-strict-analysis",
            ]
            run_logged(
                [*builder_command, "--dry-run"],
                chain_root / "logs" / "high_recall_manifest.dry_run.stdout.log",
            )
            if not router_manifest_path.exists():
                run_logged(
                    builder_command,
                    chain_root / "logs" / "high_recall_manifest.stdout.log",
                )
            router_parameters = manifest["parameters"]["high_recall_router"]
            run_logged(
                [
                    sys.executable,
                    str(
                        safe_path(
                            root,
                            manifest["script_bindings"]["high_recall_router"]["path"],
                        )
                    ),
                    "--manifest",
                    str(router_manifest_path),
                    "--output",
                    str(router_root),
                    "--fixed-threshold",
                    str(router_parameters["fixed_threshold"]),
                    "--relative-percentile",
                    str(router_parameters["relative_percentile"]),
                    "--minimum-relative-threshold",
                    str(router_parameters["minimum_relative_threshold"]),
                    "--minimum-component-pixels",
                    str(router_parameters["minimum_component_pixels"]),
                    "--top-k-per-window",
                    str(router_parameters["top_k_per_window"]),
                    "--top-k-per-scroll",
                    str(router_parameters["top_k_per_scroll"]),
                ],
                chain_root / "logs" / "high_recall_router.stdout.log",
            )
        if not all(
            path.is_file() for path in (router_manifest_path, router_receipt_path)
        ):
            raise RuntimeError("high-recall router artifacts are incomplete")
        validate_postprocess_bindings(
            root=root,
            manifest=manifest,
            combined_receipt_path=combined_receipt_path,
            ct_spec_path=ct_spec_path,
            ct_features_path=ct_features_path,
            ct_feature_receipt_path=ct_feature_receipt_path,
            ct_evaluation_path=ct_evaluation_path,
            ct_decisions_path=ct_decisions_path,
            router_manifest_path=router_manifest_path,
            router_receipt_path=router_receipt_path,
        )

        result_path = chain_root / "OFFICIAL_GP_SCROLL1_ROBUST_CHAIN_RESULT.json"
        result = build_combined_result(
            output=result_path,
            manifest_path=manifest_path,
            manifest=manifest,
            combined_receipt_path=combined_receipt_path,
            ct_spec_path=ct_spec_path,
            ct_features_path=ct_features_path,
            ct_evaluation_path=ct_evaluation_path,
            ct_decisions_path=ct_decisions_path,
            router_manifest_path=router_manifest_path,
            router_receipt_path=router_receipt_path,
            disk_audit_path=disk_audit_path,
        )
        return {
            "status": result["status"],
            "window_count": result["window_count"],
            "combined_review_queue_count": result["combined_review_queue_count"],
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
        }
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
    parser.add_argument("--wait-timeout-seconds", type=int, default=21_600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--inference-batch-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wait_timeout_seconds < 0 or args.poll_seconds < 1:
        raise RuntimeError("wait timeout must be non-negative and poll positive")
    if args.inference_batch_size is not None and args.inference_batch_size < 1:
        raise RuntimeError("inference batch size must be positive")
    root = canonical_repository_root(args.root.resolve())
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else (
            root
            / "phase4"
            / "official_gp_scroll1_rescreen_v1"
            / "OFFICIAL_GP_SCROLL1_ROBUST_CHAIN_MANIFEST.json"
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
            wait_timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
            inference_batch_size=args.inference_batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
