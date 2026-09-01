#!/usr/bin/env python3
"""Download a bounded, checksummed public PHercParis4 spiral-input subset.

The public Hugging Face bucket is much larger than the curated spiral-input
release.  This script never performs a recursive bucket sync.  ``metadata``
downloads the complete relation graphs; ``smoke`` additionally downloads all
fibers, the shell, and four deterministic verified patches.
``full_spiral_input`` is the explicit curated release only; it never traverses
any other bucket prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
PHASE2 = ROOT / "phase2"
CONFIG_PATH = PHASE2 / "configs" / "paris4_spiral_input.json"
API_ROOT = "https://huggingface.co/api/buckets/scrollprize/datasets/tree"
RESOLVE_ROOT = "https://huggingface.co/buckets/scrollprize/datasets/resolve"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_files(prefix: str) -> list[dict[str, Any]]:
    """List one small public subtree, following pagination deterministically."""

    url = f"{API_ROOT}/{prefix}?recursive=true&expand=true&limit=1000"
    files: list[dict[str, Any]] = []
    while url:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
            link = response.headers.get("Link", "")
        files.extend(item for item in payload if item.get("type") == "file")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip().strip("<>")
                break
        url = next_url
    return files


def download(path: str, destination: Path, expected_size: int | None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return {"path": path, "bytes": destination.stat().st_size, "sha256": sha256(destination), "reused": True}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    url = f"{RESOLVE_ROOT}/{path}"
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise ValueError(f"size mismatch: {temporary.stat().st_size} != {expected_size}")
            os.replace(temporary, destination)
            return {"path": path, "bytes": destination.stat().st_size, "sha256": sha256(destination), "reused": False}
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def files_for_profile(config: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    prefix = config["source"]["prefix"]
    metadata = [{"path": f"{prefix}/{name}", "size": None} for name in config["profiles"]["metadata"]]
    if profile == "metadata":
        return metadata
    if profile == "calibration":
        files = list(metadata)
        for directory in config["profiles"]["calibration"]["include_directories"]:
            files.extend(list_files(f"{prefix}/{directory}"))
        unique = {item["path"]: item for item in files}
        return [unique[path] for path in sorted(unique)]
    if profile == "full_spiral_input":
        files = list(metadata)
        for directory in config["profiles"]["full_spiral_input"]["include_directories"]:
            files.extend(list_files(f"{prefix}/{directory}"))
        unique = {item["path"]: item for item in files}
        return [unique[path] for path in sorted(unique)]
    smoke = config["profiles"]["smoke"]
    files = list(metadata)
    for directory in smoke["include_directories"]:
        files.extend(list_files(f"{prefix}/{directory}"))
    for directory, limit in smoke.get("limited_directories", {}).items():
        limited = sorted(list_files(f"{prefix}/{directory}"), key=lambda item: item["path"])[: int(limit)]
        files.extend(limited)
    for patch in smoke["verified_patch_directories"]:
        files.extend(list_files(f"{prefix}/verified_patches/{patch}"))
    unique = {item["path"]: item for item in files}
    return [unique[path] for path in sorted(unique)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("metadata", "smoke", "calibration", "full_spiral_input"), default="metadata")
    parser.add_argument("--destination", type=Path, help="override the local input directory")
    parser.add_argument("--workers", type=int, default=8, help="bounded parallel download workers")
    parser.add_argument("--plan", action="store_true", help="list the bounded profile without downloading it")
    parser.add_argument("--progress-path", type=Path, help="write resumable, non-secret download progress JSON here")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers must be between 1 and 32")
    config = json.loads(CONFIG_PATH.read_text())
    destination_root = (args.destination or PHASE2 / "inputs" / "paris4").resolve()
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    files = files_for_profile(config, args.profile)
    known_bytes = sum(int(item.get("size") or 0) for item in files)
    if args.plan:
        print(json.dumps({"profile": args.profile, "file_count": len(files), "known_total_bytes": known_bytes}, indent=2))
        return 0
    profile_disk_gb = config["profiles"].get(args.profile, {}).get("required_free_disk_gb")
    required = known_bytes + 2 * 1024**3  # temp files plus a conservative safety margin
    if profile_disk_gb:
        required = max(required, int(profile_disk_gb) * 1024**3)
    free = shutil.disk_usage(destination_root.parent).free
    if free < required:
        raise SystemExit(f"need at least {required} free bytes; only {free} available")
    progress_path = (args.progress_path or destination_root / "PARIS4_DOWNLOAD_PROGRESS.json").resolve()
    progress = {
        "kind": "campaign_x_phase2_paris4_download_progress_v1",
        "profile": args.profile,
        "planned_file_count": len(files),
        "known_total_bytes": known_bytes,
        "completed_file_count": 0,
        "completed_bytes": 0,
        "reused_file_count": 0,
        "status": "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    atomic_json(progress_path, progress)

    def one(item: dict[str, Any]) -> dict[str, Any]:
        relative = Path(item["path"]).relative_to(config["source"]["prefix"])
        return download(item["path"], destination_root / relative, item.get("size"))

    records_by_path: dict[str, dict[str, Any]] = {}
    completed_bytes = 0
    reused_file_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, item): item["path"] for item in files}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records_by_path[record["path"]] = record
            completed_bytes += int(record["bytes"])
            reused_file_count += int(bool(record["reused"]))
            if completed == len(files) or completed % 100 == 0:
                progress.update(
                    {
                        "completed_file_count": completed,
                        "completed_bytes": completed_bytes,
                        "reused_file_count": reused_file_count,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                )
                atomic_json(progress_path, progress)
    records = [records_by_path[item["path"]] for item in files]
    manifest = {
        "kind": "campaign_x_phase2_paris4_input_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "profile": args.profile,
        "source": config["source"],
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
        "full_dataset_not_downloaded": True,
        "profile_exclusions": config["profiles"].get(args.profile, {}).get("excluded_directories", []),
    }
    output = destination_root / "PARIS4_INPUT_MANIFEST.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    progress.update(
        {
            "completed_file_count": len(records),
            "completed_bytes": manifest["total_bytes"],
            "reused_file_count": sum(bool(record["reused"]) for record in records),
            "status": "COMPLETE",
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    atomic_json(progress_path, progress)
    print(json.dumps({key: manifest[key] for key in ("profile", "file_count", "total_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
