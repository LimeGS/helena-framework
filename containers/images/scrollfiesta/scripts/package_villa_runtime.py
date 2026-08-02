#!/usr/bin/env python3
"""Package the exact Villa tools required by the ScrollFiesta runtime.

The packager is intentionally Linux-oriented: it records ``ldd`` output and
fails if either executable resolves a dependency inside the source/build tree.
Only the two executables plus generated JSON metadata enter the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "campaignx.villa_runtime_bundle.v1"
RECEIPT_SCHEMA = "campaignx.villa_runtime_packaging_receipt.v1"
TOOLCHAIN_SCHEMA = "campaignx.villa_toolchain_receipt.v1"
TOOLS = ("flatboi", "vc_obj2tifxyz_legacy")
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(/[^\s()]+)")


class PackagingError(ValueError):
    """An input failed a fail-closed packaging gate."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode and not allow_failure:
        raise PackagingError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed


def git(source: Path, *arguments: str) -> str:
    return run(["git", "-C", str(source), *arguments]).stdout.strip()


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def load_source_lock(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "campaignx.scrollfiesta_source_lock.v1":
        raise PackagingError("unsupported ScrollFiesta source-lock schema")
    villa = payload.get("volume_cartographer")
    if not isinstance(villa, dict) or not isinstance(villa.get("commit"), str):
        raise PackagingError("source lock does not freeze Volume Cartographer")
    return payload, sha256(path)


def verify_source(source: Path, expected_commit: str) -> dict[str, str]:
    actual_commit = git(source, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise PackagingError(f"Villa HEAD is not frozen: {actual_commit}")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise PackagingError("Villa tracked worktree is dirty")
    return {"commit": actual_commit, "tree": git(source, "rev-parse", "HEAD^{tree}")}


def verify_toolchain_receipt(path: Path, source_identity: dict[str, str], build_root: Path) -> tuple[dict[str, Any], str]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != TOOLCHAIN_SCHEMA:
        raise PackagingError("unsupported Villa toolchain receipt schema")
    if receipt.get("source_commit") != source_identity["commit"]:
        raise PackagingError("toolchain receipt source_commit mismatch")
    if receipt.get("source_tree") != source_identity["tree"]:
        raise PackagingError("toolchain receipt source_tree mismatch")
    toolchain = receipt.get("toolchain")
    required = ("c_compiler", "cxx_compiler", "cmake_version", "build_type", "build_command")
    if not isinstance(toolchain, dict) or any(not isinstance(toolchain.get(key), str) or not toolchain[key] for key in required):
        raise PackagingError("toolchain receipt is missing required toolchain fields")
    declared_root = receipt.get("build_root")
    if not isinstance(declared_root, str) or Path(declared_root).resolve() != build_root.resolve():
        raise PackagingError("toolchain receipt build_root mismatch")
    return receipt, sha256(path)


def verify_executable(path: Path, build_root: Path, expected_name: str) -> None:
    if path.name != expected_name:
        raise PackagingError(f"unexpected executable name for {expected_name}: {path.name}")
    if path.is_symlink() or not path.is_file():
        raise PackagingError(f"{expected_name} must be a regular non-symlink file")
    if not os.access(path, os.X_OK):
        raise PackagingError(f"{expected_name} is not executable")
    if not inside(path, build_root):
        raise PackagingError(f"{expected_name} is not inside the declared build root")


def inspect_dependencies(
    executable: Path,
    inspector: Path,
    forbidden_roots: tuple[Path, ...],
) -> dict[str, Any]:
    completed = run([str(inspector), str(executable)])
    output = completed.stdout.strip()
    lowered = output.lower()
    if "not found" in lowered:
        raise PackagingError(f"unresolved dynamic dependency for {executable.name}")
    resolved_paths = sorted(set(ABSOLUTE_PATH.findall(output)))
    violations = []
    for raw in resolved_paths:
        candidate = Path(raw)
        for root in forbidden_roots:
            if inside(candidate, root):
                violations.append({"path": raw, "forbidden_root": str(root)})
    if violations:
        rendered = ", ".join(item["path"] for item in violations)
        raise PackagingError(f"{executable.name} depends on forbidden build/source paths: {rendered}")
    return {
        "command": [inspector.name, executable.name],
        "returncode": completed.returncode,
        "stdout": output,
        "resolved_absolute_paths": resolved_paths,
        "forbidden_path_violations": [],
    }


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod((mode & 0o555) | stat.S_IRUSR | stat.S_IXUSR)
        else:
            path.chmod(mode & ~0o222)
    root.chmod(0o555)


def package(
    *,
    source: Path,
    build_root: Path,
    flatboi: Path,
    obj2tifxyz: Path,
    toolchain_receipt: Path,
    source_lock: Path,
    inspector: Path,
    output: Path,
) -> dict[str, Any]:
    source = source.resolve()
    build_root = build_root.resolve()
    # Keep the final path components unresolved so a symlink cannot disguise
    # itself as a regular input during verify_executable().
    flatboi = Path(os.path.abspath(flatboi))
    obj2tifxyz = Path(os.path.abspath(obj2tifxyz))
    inspector = inspector.resolve()
    output = output.resolve()
    if output.exists():
        raise PackagingError("output already exists")
    if not output.parent.is_dir():
        raise PackagingError("output parent does not exist")
    if not inspector.is_file() or not os.access(inspector, os.X_OK):
        raise PackagingError("dependency inspector is missing or not executable")

    lock, lock_hash = load_source_lock(source_lock)
    source_identity = verify_source(source, lock["volume_cartographer"]["commit"])
    toolchain, toolchain_hash = verify_toolchain_receipt(toolchain_receipt, source_identity, build_root)
    binaries = {"flatboi": flatboi, "vc_obj2tifxyz_legacy": obj2tifxyz}
    for name, path in binaries.items():
        verify_executable(path, build_root, name)
    input_hashes = {name: sha256(path) for name, path in binaries.items()}

    inspector_version = run([str(inspector), "--version"], allow_failure=True).stdout.strip()
    dependency_receipts = {
        name: inspect_dependencies(path, inspector, (source, build_root))
        for name, path in binaries.items()
    }
    for name, path in binaries.items():
        if sha256(path) != input_hashes[name]:
            raise PackagingError(f"{name} changed during dependency inspection")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        bin_dir = staging / "bin"
        bin_dir.mkdir()
        artifacts: dict[str, dict[str, str]] = {}
        for name, path in binaries.items():
            target = bin_dir / name
            shutil.copyfile(path, target)
            target.chmod(0o555)
            copied_hash = sha256(target)
            if copied_hash != input_hashes[name]:
                raise PackagingError(f"{name} changed while being copied")
            artifacts[name] = {"path": f"bin/{name}", "sha256": copied_hash}

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PACKAGED_AND_LINKAGE_VERIFIED",
            "source": source_identity,
            "source_lock_sha256": lock_hash,
            "toolchain_receipt_sha256": toolchain_hash,
            "toolchain": toolchain["toolchain"],
            "build_root_identity_sha256": hashlib.sha256(str(build_root).encode()).hexdigest(),
            "dependency_inspector": {"path": str(inspector), "version_output": inspector_version},
            "dependency_inspection": dependency_receipts,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "artifacts": artifacts,
            "contains_only_required_executables": True,
            "scientific_data_used": False,
        }
        (staging / "VILLA_RUNTIME_PACKAGING_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema": SCHEMA,
            "source_commit": source_identity["commit"],
            "source_tree": source_identity["tree"],
            "artifacts": artifacts,
            "packaging_receipt": {
                "path": "VILLA_RUNTIME_PACKAGING_RECEIPT.json",
                "sha256": sha256(staging / "VILLA_RUNTIME_PACKAGING_RECEIPT.json"),
            },
        }
        (staging / "VILLA_RUNTIME_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        make_read_only(staging)
        os.replace(staging, output)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--villa-source", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--flatboi", type=Path, required=True)
    parser.add_argument("--obj2tifxyz", type=Path, required=True)
    parser.add_argument("--toolchain-receipt", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--dependency-inspector", type=Path, default=Path("/usr/bin/ldd"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        package(
            source=args.villa_source,
            build_root=args.build_root,
            flatboi=args.flatboi,
            obj2tifxyz=args.obj2tifxyz,
            toolchain_receipt=args.toolchain_receipt,
            source_lock=args.source_lock,
            inspector=args.dependency_inspector,
            output=args.output,
        )
    except (OSError, PackagingError, json.JSONDecodeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Villa runtime packaging failed: {exc}", file=sys.stderr)
        return 2
    print("VILLA_RUNTIME_PACKAGED_AND_LINKAGE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
