#!/usr/bin/env python3
"""Write a deterministic GNU sha256sum inventory in one process."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def generate(root: Path, output: Path) -> int:
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.parent != root:
        raise ValueError("output must be directly inside root")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path == output or not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(root).as_posix()
        target = path.resolve(strict=True) if path.is_symlink() else path
        rows.append(f"{digest(target)}  {relative}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = generate(args.root, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"SHA256SUMS_READY files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
