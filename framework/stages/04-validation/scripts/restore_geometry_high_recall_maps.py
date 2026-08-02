#!/usr/bin/env python3
"""Restore strict-screen evidence needed by the high-recall CT follow-up.

This is a cache-recovery tool, not a second screen.  It rebuilds a missing map
in an isolated staging directory and copies it back only if its SHA-256 equals
the hash committed in the original strict analysis and screening receipt.

When explicitly requested, it also retains the 65-slice CT render in the
location consumed by the high-recall CT adapter.  Those TIFFs were not
historically hash-bound by the strict screen, so the receipt calls them
``REHYDRATED_FROM_HASH_BOUND_SURFACE`` rather than claiming byte identity.
The six restored maps remain the byte-identical binding to the strict run.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


REQUIRED_SURFACE = ("x.tif", "y.tif", "z.tif", "meta.json")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False, env=env)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def archive_surface(surface: Path, archive: Path) -> dict[str, dict[str, Any]]:
    archive.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SURFACE:
        source, destination = surface / name, archive / name
        digest = sha256(source)
        if destination.is_file():
            if sha256(destination) != digest:
                raise RuntimeError(f"non-matching archived TIFXYZ: {destination}")
        else:
            shutil.copy2(source, destination)
        output[name] = {"sha256": digest, "size_bytes": source.stat().st_size}
    return output


def verify_surface(surface: Path, expected: dict[str, Any], *, seed_id: str) -> None:
    """Require the preserved V15 surface to match its terminal growth receipt."""
    for name in REQUIRED_SURFACE:
        path = surface / name
        row = expected.get(name) if isinstance(expected, dict) else None
        if not path.is_file() or not isinstance(row, dict) or sha256(path) != str(row.get("sha256", "")):
            raise RuntimeError(f"preserved surface hash mismatch for {seed_id} {name}")


def tiff_inventory(directory: Path) -> tuple[list[dict[str, Any]], str]:
    """Return a stable, complete inventory of a 65-slice CT render."""
    files, slice_ordering = ordered_tiff_files(directory, require_numeric=True)
    if len(files) != 65:
        raise RuntimeError(f"expected exactly 65 TIFF slices, found {len(files)}: {directory}")
    if [int(path.stem) for path in files] != list(range(65)):
        raise RuntimeError(f"TIFF slices must be named 0.tif through 64.tif: {directory}")
    rows = [
        {"name": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in files
    ]
    return rows, slice_ordering


def retain_rehydrated_tiffs(*, source: Path, destination: Path, renderer: Path) -> dict[str, Any]:
    """Copy an exact-count render into the adapter's canonical location.

    Never overwrites an existing incomplete directory.  Existing complete
    evidence is preserved and simply inventoried; the caller records which
    state occurred in its restoration receipt.
    """
    source_inventory, slice_ordering = tiff_inventory(source)
    existing = (
        ordered_tiff_files(destination, allow_empty=True)[0]
        if destination.exists()
        else []
    )
    if existing:
        destination_inventory, destination_ordering = tiff_inventory(destination)
        return {
            "state": "ALREADY_PRESENT",
            "source_inventory": source_inventory,
            "destination_inventory": destination_inventory,
            "slice_ordering": slice_ordering,
            "destination_slice_ordering": destination_ordering,
            "renderer_sha256": sha256(renderer),
        }
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing to populate a non-empty TIFF destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for row in source_inventory:
        shutil.copy2(source / str(row["name"]), destination / str(row["name"]))
    destination_inventory, destination_ordering = tiff_inventory(destination)
    if destination_inventory != source_inventory:
        raise RuntimeError("copied rehydrated TIFF inventory does not match staging render")
    return {
        "state": "REHYDRATED_FROM_HASH_BOUND_SURFACE",
        "source_inventory": source_inventory,
        "destination_inventory": destination_inventory,
        "slice_ordering": slice_ordering,
        "destination_slice_ordering": destination_ordering,
        "renderer_sha256": sha256(renderer),
    }


def restore_one(
    *, root: Path, eligible: dict[str, dict[str, Any]], row: dict[str, Any], archive_root: Path | None,
    retain_tiffs_for_high_recall: bool, attempt_label: str,
) -> dict[str, Any]:
    analysis_path = (root / str(row["analysis"])).resolve()
    analysis, screening = load(analysis_path), load((analysis_path.parents[1] / "INK_SCREENING_RECEIPT.json"))
    screening_dir, base = analysis_path.parents[1], analysis_path.parents[2]
    expected_analysis = {str(item["file"]): str(item["sha256"]) for item in analysis["input"]["maps"]}
    expected_screening = {str(item["npy"]): str(item["npy_sha256"]) for item in screening["inference"]["runs"]}
    if expected_analysis != expected_screening or len(expected_analysis) != 6:
        raise RuntimeError(f"original replica bindings are invalid: {row['seed_id']}")
    missing: list[str] = []
    for name, digest in sorted(expected_analysis.items()):
        existing = screening_dir / name
        if not existing.is_file():
            missing.append(name)
        elif sha256(existing) != digest:
            raise RuntimeError(f"existing replica hash mismatch for {row['seed_id']} {name}")
    if not missing and not retain_tiffs_for_high_recall:
        return {"seed_id": row["seed_id"], "state": "ALREADY_PRESENT", "restored_maps": []}
    growth = load(base / "GROWTH_RECEIPT.json")
    if growth.get("status") != "PASSED":
        raise RuntimeError(f"growth was not passed: {row['seed_id']}")
    stage = base / f"_high_recall_map_restore_staging_{attempt_label}"
    if stage.exists():
        raise RuntimeError(f"staging already exists; refusing overwrite: {stage}")
    stage_surface, stage_tiffs, stage_output = stage / "surface", stage / "tiffs", stage / "inference"
    recovery = base / f"map_restoration_{attempt_label}"
    recovery.mkdir(parents=True, exist_ok=True)
    command = list(growth["command"])
    command[command.index("--target-dir") + 1] = str(stage_surface)
    env = os.environ.copy()
    env["VC_GROWPATCH_RNG_SEED"] = hashlib.sha256(str(row["seed_id"]).encode()).hexdigest()[:16]
    try:
        expected_surface = growth.get("files", {})
        preserved_surface = base / "surface"
        if all((preserved_surface / name).is_file() for name in REQUIRED_SURFACE):
            verify_surface(preserved_surface, expected_surface, seed_id=str(row["seed_id"]))
            render_surface = preserved_surface
            surface_source = "PRESERVED_HASH_BOUND_SURFACE"
        else:
            run(command, recovery / "grow.stdout.log", env=env)
            verify_surface(stage_surface, expected_surface, seed_id=str(row["seed_id"]))
            render_surface = stage_surface
            surface_source = "REGROWN_HASH_BOUND_SURFACE"
        entry = eligible[str(row["sample_id"])]
        renderer = Path("/workspace/villa-phase3/build-phase3-gcc13/bin/vc_render_tifxyz")
        if not renderer.is_file():
            renderer = (root / "../villa-phase3/build-phase3-gcc13/bin/vc_render_tifxyz").resolve()
        if not renderer.is_file():
            raise RuntimeError(f"renderer is unavailable: {renderer}")
        metadata_cache = base.parents[1] / "ct_metadata_cache" / f"{row['sample_id']}.zarr"
        render = [
            str(renderer),
            "--segmentation", str(render_surface), "--volume", str(metadata_cache),
            "--remote-url", str(entry["ct_uri"]), "--prefetch-remote", "--scale", "1", "--group-idx", "0", "--auto-crop", "--flatten", "--flatten-iterations", "10",
            "--num-slices", "65", "--slice-step", "1", "--cache-gb", "4", "--timeout", "90", "--voxel-size", str(entry["voxel_size_um"]), "--voxel-unit", "micrometer",
            "--tif-output", str(stage_tiffs), "--log-path", str(recovery / "ct-render.log"),
        ]
        run(render, recovery / "ct-render.stdout.log")
        if len(ordered_tiff_files(stage_tiffs, allow_empty=True)[0]) != 65:
            raise RuntimeError(f"restored CT render count is not 65: {row['seed_id']}")
        physical, inference = screening["physical_normalization"], screening["inference"]
        infer = [
            sys.executable, str(root / "framework/stages/03-ink/scripts/run_ink_timesformer.py"), "--sample-id", str(row["sample_id"]), "--tiff-dir", str(stage_tiffs),
            "--checkpoint", str(root / "models/timesformer_GP_scroll1/model.safetensors"), "--model-family", "timesformer_GP_scroll1", "--output", str(stage_output),
            "--depth-centers", ",".join(str(value) for value in inference["depth_centers"]), "--tiling-offsets", ",".join(str(value) for value in inference["tiling_offsets"]),
            "--frames", str(physical["frames"]), "--source-pixel-um", str(physical["source_pixel_um"]), "--training-pixel-um", str(physical["training_pixel_um"]),
            "--source-slice-um", str(physical["source_slice_um"]), "--training-slice-um", str(physical["training_slice_um"]), "--tile-size", str(inference["tile_size"]),
            "--stride", str(inference["stride"]), "--batch-size", str(inference["batch_size"]), "--min-valid-ratio", str(inference["min_valid_ratio"]), "--device", str(inference["device"]),
        ]
        run(infer, recovery / "inference.stdout.log")
        for name, digest in expected_analysis.items():
            generated = stage_output / name
            if not generated.is_file() or sha256(generated) != digest:
                raise RuntimeError(f"restored replica hash mismatch for {row['seed_id']} {name}")
        for name in missing:
            shutil.copy2(stage_output / name, screening_dir / name)
        tiff_restore = None
        if retain_tiffs_for_high_recall:
            tiff_restore = retain_rehydrated_tiffs(
                source=stage_tiffs,
                destination=base / "tiffs",
                renderer=renderer,
            )
        archive = archive_surface(render_surface, archive_root / str(row["sample_id"]) / str(row["seed_id"])) if archive_root else None
        receipt = {
            "kind": "campaign_x_phase4_high_recall_map_restoration_v1",
            "generated_at_utc": utc_now(), "seed_id": row["seed_id"], "sample_id": row["sample_id"],
            "state": "RESTORED_BYTE_IDENTICAL", "restored_maps": missing,
            "expected_replica_hashes": expected_analysis, "archive_tifxyz": archive,
            "surface_source": surface_source, "attempt_label": attempt_label,
            "rehydrated_tiffs": tiff_restore,
            "policy": [
                "original strict summary was not overwritten",
                "only maps whose hashes match the original analysis were restored",
                "rehydrated TIFFs are explicitly not claimed byte-identical because the strict screen did not commit TIFF hashes",
                "no threshold or model selection changed",
            ],
        }
        (recovery / "HIGH_RECALL_MAP_RESTORATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return receipt
    finally:
        if stage.exists() and all((screening_dir / name).is_file() for name in expected_analysis):
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--archive-surface-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--retain-tiffs-for-high-recall",
        action="store_true",
        help="retain a rehydrated 65-slice CT render for the downstream high-recall adapter",
    )
    parser.add_argument(
        "--attempt-label",
        default="v1",
        help="plain local suffix for a never-overwritten staging/log attempt",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt_label):
        raise RuntimeError("attempt label must contain only letters, digits, _ or -")
    summary = load(args.screen_summary.resolve())
    eligible = {str(row["sample_id"]): row for row in load(root / "phase0/eligible_volumes.json")["entries"]}
    rows = summary.get("receipts")
    if summary.get("status") != "COMPLETED_DIAGNOSTIC_ONLY" or not isinstance(rows, list):
        raise RuntimeError("restoration requires a completed strict screen")
    restored: list[dict[str, Any]] = []
    payload = {
        "kind": "campaign_x_phase4_high_recall_map_restoration_batch_v1",
        "generated_at_utc": utc_now(),
        "source_screen_summary": str(args.screen_summary.resolve()),
        "status": "RUNNING",
        "rows": restored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        try:
            restored.append(
                restore_one(
                    root=root,
                    eligible=eligible,
                    row=row,
                    archive_root=args.archive_surface_root.resolve(),
                    retain_tiffs_for_high_recall=args.retain_tiffs_for_high_recall,
                    attempt_label=args.attempt_label,
                )
            )
        except Exception as error:
            payload.update({"status": "FAILED_CLOSED", "failed_seed_id": row.get("seed_id"), "error": str(error)})
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
    payload["status"] = "COMPLETED_BYTE_IDENTICAL"
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(restored), "restored": sum(row["state"] == "RESTORED_BYTE_IDENTICAL" for row in restored)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
