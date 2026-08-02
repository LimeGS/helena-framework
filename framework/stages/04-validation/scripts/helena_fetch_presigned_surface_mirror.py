#!/usr/bin/env python3
"""Receive a credential-free S3 surface mirror over presigned HTTPS URLs.

Input is JSONL on stdin with exactly ``relative_path`` and ``url``.  URLs are
never written to disk or copied into the receipt.  The caller should pipe the
stream directly from an authenticated control plane to this process on a GPU
worker.  The scientific QC adapter subsequently verifies ARTIFACT_SET.json,
every TIFXYZ file hash, and the catalogue-level artifact digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable
from urllib.parse import urlparse


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def safe_destination(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError("presigned mirror path must be relative and traversal-free")
    destination = root.joinpath(*relative.parts).resolve()
    resolved_root = root.resolve()
    if resolved_root not in destination.parents:
        raise RuntimeError("presigned mirror path escapes the mirror root")
    return destination


def copy_and_hash(source: BinaryIO, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return size, digest.hexdigest()


def receive(
    records: Iterable[str],
    root: Path,
    *,
    expected_count: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(records, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or set(payload) != {"relative_path", "url"}:
            raise RuntimeError(f"invalid presigned record at line {line_number}")
        relative_path = str(payload["relative_path"])
        if relative_path in seen:
            raise RuntimeError(f"duplicate presigned mirror path: {relative_path}")
        seen.add(relative_path)
        url = str(payload["url"])
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError("presigned mirror URL must use HTTPS")
        destination = safe_destination(root, relative_path)
        request = urllib.request.Request(url, headers={"User-Agent": "helena-mirror/1"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            size, digest = copy_and_hash(response, destination)
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if len(rows) != expected_count:
        raise RuntimeError(
            f"presigned mirror received {len(rows)} files; expected {expected_count}"
        )
    return {
        "schema": "campaignx.presigned_surface_mirror_receipt.v1",
        "generated_at_utc": utc_now(),
        "file_count": len(rows),
        "files": sorted(rows, key=lambda row: row["relative_path"]),
        "presigned_urls_persisted": False,
        "credentials_received": False,
        "scientific_validation_deferred_to_surface_qc_adapter": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.expected_count < 1:
        raise RuntimeError("expected count must be positive")
    receipt = receive(
        sys.stdin,
        args.root.expanduser().resolve(),
        expected_count=args.expected_count,
        timeout_seconds=args.timeout_seconds,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.partial")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.receipt)
    print(json.dumps({"status": "COMPLETE", "file_count": receipt["file_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
