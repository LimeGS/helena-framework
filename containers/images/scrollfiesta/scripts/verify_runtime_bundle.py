#!/usr/bin/env python3
"""Verify an immutable ScrollFiesta native/OCI runtime payload."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


SCHEMA = "campaignx.scrollfiesta_runtime_bundle.v1"
REQUIRED = (
    "bin/cube_mesh",
    "bin/grid_weld",
    "bin/obj_components",
    "bin/pinhole_verdict",
    "bin/seam_audit",
    "bin/flatboi",
    "bin/vc_obj2tifxyz_legacy",
    "bin/scrollunwrap",
    "SOURCE_LOCK.json",
    "LICENSE_INVENTORY.json",
    "SBOM.spdx.json",
    "BUILD_RECEIPT.json",
    "PYTHON_REQUIREMENTS.lock",
    "python-packages/scrollunwrap/__init__.py",
    "share/licenses/scrollfiesta/LICENSE",
    "share/licenses/scrollfiesta/THIRD_PARTY_LICENSES.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> None:
    if (root / "BUNDLE_SCHEMA").read_text(encoding="utf-8").strip() != SCHEMA:
        raise ValueError("runtime bundle schema mismatch")
    for relative in REQUIRED:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing bundle artifact: {relative}")
    for relative in REQUIRED[:8]:
        if not ((root / relative).stat().st_mode & 0o111):
            raise ValueError(f"runtime command is not executable: {relative}")
    sums_path = root / "SHA256SUMS"
    declared: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in declared:
            raise ValueError(f"duplicate checksum path: {relative}")
        declared[relative] = expected
    actual_files = []
    for path in root.rglob("*"):
        if path.name == "SHA256SUMS" or not (path.is_file() or path.is_symlink()):
            continue
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"bundle symlink escapes the runtime root: {path.relative_to(root)}") from exc
        actual_files.append(path.relative_to(root).as_posix())
    actual_files.sort()
    if sorted(declared) != actual_files:
        raise ValueError("SHA256SUMS inventory does not match the bundle files")
    for relative in actual_files:
        if sha256(root / relative) != declared[relative]:
            raise ValueError(f"bundle hash mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.root.resolve())
    except (OSError, ValueError) as exc:
        print(f"runtime bundle verification failed: {exc}", file=sys.stderr)
        return 2
    print("SCROLLFIESTA_RUNTIME_BUNDLE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
