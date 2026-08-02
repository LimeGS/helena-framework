#!/usr/bin/env python3
"""Fail closed unless a ScrollFiesta checkout matches the frozen I1 lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def verify(source: Path, lock_path: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    frozen = lock["scrollfiesta"]
    errors: list[str] = []
    if git(source, "rev-parse", "HEAD") != frozen["commit"]:
        errors.append("ScrollFiesta HEAD does not match the frozen commit")
    if git(source, "rev-parse", "HEAD^{tree}") != frozen["tree"]:
        errors.append("ScrollFiesta tree does not match the frozen tree")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        errors.append("ScrollFiesta tracked worktree is dirty")
    for relative, expected in frozen["required_file_sha256"].items():
        candidate = source / relative
        if not candidate.is_file():
            errors.append(f"missing locked source file: {relative}")
        elif sha256(candidate) != expected:
            errors.append(f"source hash mismatch: {relative}")
    for name, expected in lock["vendored_dependency_trees"].items():
        path_by_name = {
            "triangle": "deps/src/triangle",
            "clipper2": "deps/src/Clipper2",
            "andres_graph": "deps/src/graph",
            "libtiff_4_7_1": "deps/src/tiff-4.7.1",
            "zlib": "deps/src/zlib",
            "poissonrecon": "deps/src/PoissonRecon",
            "probabilistic_quadrics": "deps/src/probabilistic-quadrics",
        }
        relative = path_by_name[name]
        try:
            actual = git(source, "rev-parse", f"HEAD:{relative}")
        except subprocess.CalledProcessError:
            errors.append(f"unable to resolve frozen dependency tree: {relative}")
        else:
            if actual != expected:
                errors.append(f"dependency tree mismatch: {name}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema": "campaignx.scrollfiesta_source_verification.v1",
        "status": "VERIFIED",
        "commit": frozen["commit"],
        "tree": frozen["tree"],
        "lock_sha256": sha256(lock_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify(args.source.resolve(), args.lock.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"source lock verification failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
