#!/usr/bin/env python3
"""Render and coarsely screen every already-expanded Phase 4 seed surface.

The batch is deliberately only a compute-prioritization pass. It never calls a
surface ink-positive, a letter, or a First Letters candidate. Its outputs are
later ranked into physical windows of at most 4 cm2 and only those windows run
through the frozen six-replica screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# FIX-09: the training scale is declared by the ink lane profile; a literal here
# silently disagrees with the profile the checkpoint was actually trained under.
INK_PROFILE_PATH = (
    Path(__file__).resolve().parents[3]
    / "profiles/03-ink/timesformer-gp-scroll1-screening-1.0.0.json"
)


def training_pixel_um() -> str:
    profile = json.loads(INK_PROFILE_PATH.read_text(encoding="utf-8"))
    return str(float(profile["input_contract"]["training_pixel_um"]))


def training_slice_um() -> str:
    profile = json.loads(INK_PROFILE_PATH.read_text(encoding="utf-8"))
    return str(float(profile["input_contract"]["training_slice_um"]))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_tasks(root: Path, sample_ids: set[str] | None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for target in sorted((root / "phase4" / "targets").glob("PHerc*")):
        if not target.is_dir() or (sample_ids and target.name not in sample_ids):
            continue
        lock_path = target / "TARGET_LOCK.json"
        if not lock_path.is_file():
            continue
        lock = read_json(lock_path)
        if lock.get("sample_id") != target.name:
            raise RuntimeError(f"target lock mismatch: {lock_path}")
        for meta_path in sorted(
            (target / "candidate_surfaces").glob("*/expanded/meta.json")
        ):
            surface = meta_path.parent
            seed_id = surface.parent.name
            meta = read_json(meta_path)
            required = [surface / name for name in ("x.tif", "y.tif", "z.tif")]
            if not all(path.is_file() and path.stat().st_size for path in required):
                raise RuntimeError(f"incomplete expanded tifxyz: {surface}")
            tasks.append(
                {
                    "sample_id": target.name,
                    "seed_id": seed_id,
                    "target": target,
                    "lock_path": lock_path,
                    "ct_uri": str(lock["ct_uri"]),
                    "voxel_um": float(lock["voxel_size_um"]),
                    "surface": surface,
                    "surface_area_cm2": float(meta["area_cm2"]),
                    "source_hashes": {
                        path.name: sha256_file(path)
                        for path in [meta_path, *required]
                    },
                }
            )
    return tasks


def fetch_zarr_metadata(ct_uri: str, cache: Path) -> None:
    for suffix in (".zgroup", ".zattrs", "0/.zarray"):
        destination = cache / suffix
        if destination.is_file() and destination.stat().st_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{ct_uri.rstrip('/')}/{suffix}",
            headers={"User-Agent": "Campaign-X-phase4-expanded-screen/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def checkpoint_receipt(
    path: Path,
    *,
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    renderer: Path,
    checkpoint: Path,
    model_family: str,
    screening_name: str,
) -> None:
    completed = sum(state["status"] == "COARSE_COMPLETED" for state in states.values())
    failed = sum(state["status"] == "FAILED" for state in states.values())
    write_json(
        path,
        {
            "kind": "campaign_x_phase4_expanded_surface_coarse_batch_v1",
            "status": (
                "COMPLETED_PRIORITIZATION_ONLY"
                if completed == len(tasks) and not failed
                else "RUNNING_OR_PARTIAL"
            ),
            "updated_at_utc": utc_now(),
            "task_count": len(tasks),
            "completed_count": completed,
            "failed_count": failed,
            "renderer": {
                "path": str(renderer),
                "sha256": sha256_file(renderer),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "model_family": model_family,
            },
            "screening_name": screening_name,
            "policy": [
                "one depth and one tiling offset only prioritize compute",
                "no coarse activation is accepted as ink or letters",
                "only physical windows at or below 4 cm2 may advance",
                "advancing windows require the frozen six-replica screen",
                "a positive six-replica screen still requires raw-CT review",
            ],
            "tasks": [
                {
                    "sample_id": task["sample_id"],
                    "seed_id": task["seed_id"],
                    "surface_area_cm2": task["surface_area_cm2"],
                    "source_hashes": task["source_hashes"],
                    **states[task["seed_id"]],
                }
                for task in tasks
            ],
            "explicit_non_claims": [
                "not automatic ink acceptance",
                "not automatic letter acceptance",
                "not a First Letters submission claim",
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        default="timesformer_scroll5_july_retreat",
    )
    parser.add_argument(
        "--screening-name",
        default="coarse_screen_v1",
    )
    parser.add_argument(
        "--batch-receipt-name",
        default="EXPANDED_SURFACE_BATCH_RECEIPT.json",
    )
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--cache-gb", type=int, default=12)
    args = parser.parse_args()

    root = args.root.resolve()
    renderer = args.renderer.resolve()
    checkpoint = args.checkpoint.resolve()
    if not renderer.is_file() or not checkpoint.is_file():
        raise RuntimeError("renderer and checkpoint must be existing files")
    for value, label in (
        (args.model_family, "model family"),
        (args.screening_name, "screening name"),
        (args.batch_receipt_name, "batch receipt name"),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"unsafe {label}: {value!r}")

    sample_ids = set(args.sample_id) if args.sample_id else None
    tasks = discover_tasks(root, sample_ids)
    if not tasks:
        raise RuntimeError("no expanded candidate surfaces found")

    output_root = root / "phase4" / "expanded_candidate_surface_screen_v1"
    receipt_path = output_root / args.batch_receipt_name
    states: dict[str, dict[str, Any]] = {
        task["seed_id"]: {
            "status": "PENDING",
            "output": str(
                output_root / task["sample_id"] / task["seed_id"]
            ),
        }
        for task in tasks
    }

    for index, task in enumerate(tasks, start=1):
        sample_id = task["sample_id"]
        seed_id = task["seed_id"]
        output = output_root / sample_id / seed_id
        tiffs = output / "tiffs"
        coarse = output / args.screening_name
        cache = output_root / sample_id / "ct_remote_cache.zarr"
        try:
            print(f"START:{index}/{len(tasks)}:{seed_id}", flush=True)
            fetch_zarr_metadata(task["ct_uri"], cache)
            if len(list(tiffs.glob("*.tif"))) != 65:
                run_logged(
                    [
                        str(renderer),
                        "--segmentation",
                        str(task["surface"]),
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
                        "10",
                        "--num-slices",
                        "65",
                        "--slice-step",
                        "1",
                        "--cache-gb",
                        str(args.cache_gb),
                        "--timeout",
                        "90",
                        "--voxel-size",
                        str(task["voxel_um"]),
                        "--voxel-unit",
                        "micrometer",
                        "--tif-output",
                        str(tiffs),
                        "--log-path",
                        str(output / "render.log"),
                    ],
                    output / "render.stdout.log",
                )
            if len(list(tiffs.glob("*.tif"))) != 65:
                raise RuntimeError("renderer did not produce exactly 65 TIFF slices")

            screening_receipt = coarse / "INK_SCREENING_RECEIPT.json"
            if screening_receipt.is_file():
                frozen_screen = read_json(screening_receipt)
                frozen_checkpoint = frozen_screen.get("checkpoint", {})
                if (
                    frozen_checkpoint.get("sha256") != sha256_file(checkpoint)
                    or frozen_checkpoint.get("model_family")
                    != args.model_family
                ):
                    raise RuntimeError(
                        "existing coarse screen belongs to a different "
                        "checkpoint or model family"
                    )
            if not screening_receipt.is_file():
                run_logged(
                    [
                        "python3",
                        str(root / "scripts" / "run_ink_timesformer.py"),
                        "--sample-id",
                        sample_id,
                        "--tiff-dir",
                        str(tiffs),
                        "--checkpoint",
                        str(checkpoint),
                        "--model-family",
                        args.model_family,
                        "--output",
                        str(coarse),
                        "--depth-centers",
                        "32",
                        "--tiling-offsets",
                        "0",
                        "--frames",
                        "26",
                        "--source-pixel-um",
                        str(task["voxel_um"]),
                        "--training-pixel-um",
                        training_pixel_um(),
                        "--source-slice-um",
                        str(task["voxel_um"]),
                        "--training-slice-um",
                        training_slice_um(),
                        "--tile-size",
                        "64",
                        "--stride",
                        "32",
                        "--batch-size",
                        "64",
                        "--min-valid-ratio",
                        "0.60",
                        "--device",
                        "cuda",
                    ],
                    output / "coarse.stdout.log",
                )
            states[seed_id] = {
                "status": "COARSE_COMPLETED",
                "output": str(output),
                "rendered_tiff_count": 65,
                "coarse_receipt": str(screening_receipt),
                "coarse_receipt_sha256": sha256_file(screening_receipt),
            }
            print(f"DONE:{index}/{len(tasks)}:{seed_id}", flush=True)
        except Exception as error:
            states[seed_id] = {
                "status": "FAILED",
                "output": str(output),
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"FAILED:{index}/{len(tasks)}:{seed_id}:{error}", flush=True)
        checkpoint_receipt(
            receipt_path,
            tasks=tasks,
            states=states,
            renderer=renderer,
            checkpoint=checkpoint,
            model_family=args.model_family,
            screening_name=args.screening_name,
        )

    failed = [seed_id for seed_id, state in states.items() if state["status"] == "FAILED"]
    if failed:
        raise RuntimeError(f"expanded-surface batch failures: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
