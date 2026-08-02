#!/usr/bin/env python3
"""Create a sterile, deterministic BuildKit context from a verified bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from verify_runtime_bundle import verify


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, root: Path, path: Path) -> None:
    relative = path.relative_to(root).as_posix()
    info = archive.gettarinfo(str(path), arcname=relative)
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    if path.is_file():
        with path.open("rb") as handle:
            archive.addfile(info, handle)
    else:
        archive.addfile(info)


def create_context(bundle: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError("OCI context output already exists")
    verify(bundle)
    if not output.parent.is_dir():
        raise ValueError("OCI context output parent does not exist")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        target = staging / "scrollfiesta-runtime.tgz"
        with target.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for path in sorted(bundle.rglob("*")):
                        add_file(archive, bundle, path)
        receipt = {
            "schema": "campaignx.scrollfiesta_oci_context_receipt.v1",
            "status": "VERIFIED_CONTEXT_READY",
            "bundle_schema": "campaignx.scrollfiesta_runtime_bundle.v1",
            "archive": "scrollfiesta-runtime.tgz",
            "archive_sha256": sha256(target),
            "contains_scientific_data": False,
            "distribution": "INTERNAL_RESEARCH_ONLY",
        }
        (staging / "OCI_CONTEXT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_context(args.bundle.resolve(), args.output.resolve())
    print("SCROLLFIESTA_OCI_CONTEXT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
