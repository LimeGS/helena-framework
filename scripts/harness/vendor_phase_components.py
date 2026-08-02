#!/usr/bin/env python3
"""Vendor the code of the phase components into the framework, with provenance.

Code only. The findings these repositories carry -- plates, indices, reviews,
checkpoints, fixtures -- stay where they are. What is imported is the technique.

Each component gets a VENDOR.json recording where it came from, which commit,
and the sha256 of every file taken, so a vendored copy can always be compared
against its origin.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent
DEST = ROOT / "framework" / "vendored"

CODE_SUFFIXES = {".py", ".sh", ".toml", ".cfg", ".txt", ".md", ".sql", ".json"}
# Data, results and weights are findings, not technique.
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "plates", "plates_photo",
             "plates_villa", "figures", "fixtures", "work_maps", "docs", "data",
             "calibration_data", "evidence_0139", ".pytest_cache"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".npy", ".npz", ".pt", ".pth", ".safetensors",
                 ".tif", ".tiff", ".zip", ".jsonl", ".pdf", ".log"}
MAX_BYTES = 512 * 1024

# macOS evicts files to iCloud and leaves a dataless stub behind. Reading one
# blocks until the download completes, and when iCloud is not serving it blocks
# indefinitely -- shutil.copy2 sits in fcopyfile with no timeout. Checking the
# flag is instant, so unavailable files are recorded rather than waited on.
SF_DATALESS = 0x40000000


def is_dataless(path: Path) -> bool:
    try:
        return bool(getattr(path.stat(), "st_flags", 0) & SF_DATALESS)
    except OSError:
        return True

COMPONENTS = {
    "scroll-streaming-tools": ["P0", "P1", "P4", "P5"],
    "vetting-card": ["P7"],
    "scroll-tracing-benchmark-v4": ["P1", "P6"],
    "reference-strips": ["P1", "P2"],
    "pherc0139-column-atlas-gh": ["P8", "P9"],
    "helena-framework": [],
    "hf-proxy-v4-dataset": ["P7"],
}
LOOSE_FILES = {"ppm_from_tifxyz": ("vendor/ppm_from_tifxyz.py", ["P3", "P4"])}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(directory: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_remote(directory: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(directory), "remote", "get-url", "origin"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def wanted(absolute: Path, relative: Path) -> bool:
    if any(part in SKIP_DIRS for part in relative.parts):
        return False
    if relative.suffix.lower() in SKIP_SUFFIXES:
        return False
    if relative.suffix.lower() not in CODE_SUFFIXES:
        return False
    return absolute.stat().st_size <= MAX_BYTES


def vendor(name: str, source: Path, phases: list[str]) -> dict:
    target = DEST / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    files = {}
    unavailable: list[str] = []
    # os.walk with in-place pruning: rglob descends into .git and the plate
    # directories before any filter sees them, which on these repositories means
    # walking hundreds of megabytes to copy a few hundred kilobytes.
    import os

    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for filename in sorted(filenames):
            path = pathlib.Path(dirpath) / filename
            rel = path.relative_to(source)
            if not path.is_file() or not wanted(path, rel):
                continue
            if is_dataless(path):
                unavailable.append(str(rel))
                continue
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            files[str(rel)] = sha256(path)
    manifest = {
        "schema": "campaignx.vendored_component.v1",
        "component": name,
        "phases": phases,
        "vendored_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "source_path": str(source.relative_to(SOURCE)),
        "source_remote": git_remote(source),
        "source_commit": git_head(source),
        "policy": "code only; plates, indices, reviews, checkpoints and fixtures are "
                  "findings and stay at the source",
        "file_count": len(files),
        "files": files,
        "unavailable_dataless": unavailable,
    }
    (target / "VENDOR.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    manifests = []
    for name, phases in COMPONENTS.items():
        source = SOURCE / "release" / name
        if not source.exists():
            print(f"  skip {name}: not at {source}", file=sys.stderr)
            continue
        manifests.append(vendor(name, source, phases))
        m = manifests[-1]
        missing = len(m["unavailable_dataless"])
        note = f"  ({missing} evicted to iCloud)" if missing else ""
        print(f"  {name:38s} {m['file_count']:4d} files  {phases}{note}")
    for name, (rel, phases) in LOOSE_FILES.items():
        src = SOURCE / rel
        if not src.exists():
            continue
        target = DEST / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target / src.name)
        manifest = {
            "schema": "campaignx.vendored_component.v1", "component": name, "phases": phases,
            "vendored_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
            "source_path": rel, "source_remote": None, "source_commit": None,
            "policy": "single file", "file_count": 1, "files": {src.name: sha256(src)},
            "unavailable_dataless": [],
        }
        (target / "VENDOR.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests.append(manifest)
        print(f"  {name:38s}    1 files  {phases}")

    index = {
        "schema": "campaignx.vendored_index.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "components": [{k: m[k] for k in
                        ("component", "phases", "source_path", "source_remote",
                         "source_commit", "file_count")} for m in manifests],
        "total_files": sum(m["file_count"] for m in manifests),
        "total_unavailable": sum(len(m["unavailable_dataless"]) for m in manifests),
        "unavailable_note": "files evicted to iCloud with the dataless flag; materialise them "
                            "and re-run to complete the vendoring",
    }
    (DEST / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\n  total: {index['total_files']} files across {len(manifests)} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
