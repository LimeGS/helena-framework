#!/usr/bin/env python3
"""Relay a frozen private S3 TIFXYZ batch to a worker without credentials.

The control plane authenticates to S3, validates every artifact in memory,
and sends one deterministic tar stream over SSH.  The worker receives only
the planned immutable files.  AWS credentials, presigned URLs, and local
artifact staging are deliberately absent.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse


PLAN_SCHEMA = "campaignx.presigned_surface_mirror_plan.v1"
RECEIPT_SCHEMA = "campaignx.credential_free_surface_relay_receipt.v1"
REQUIRED_FILES = ("ARTIFACT_SET.json", "x.tif", "y.tif", "z.tif", "meta.json")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:@+-]+$")
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


REMOTE_EXTRACTOR = r'''
import base64, hashlib, json, shutil, sys, tarfile
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
expected = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
if not root.is_absolute():
    raise SystemExit("remote mirror root must be absolute")
if root.exists():
    raise SystemExit("remote mirror root already exists")
root.mkdir(parents=True)
try:
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isreg() or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("unsafe tar member")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("missing tar member body")
            destination = root.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as stream:
                shutil.copyfileobj(source, stream)
    actual = sorted(
        str(path.relative_to(root).as_posix())
        for path in root.rglob("*") if path.is_file()
    )
    wanted = sorted(item["path"] for item in expected)
    if actual != wanted:
        raise RuntimeError("remote mirror file inventory mismatch")
    for item in expected:
        path = root.joinpath(*PurePosixPath(item["path"]).parts)
        if path.stat().st_size != item["size_bytes"]:
            raise RuntimeError("remote mirror size mismatch: " + item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError("remote mirror hash mismatch: " + item["path"])
except Exception:
    shutil.rmtree(root, ignore_errors=True)
    raise
print(json.dumps({
    "status": "VERIFIED",
    "remote_root": str(root),
    "file_count": len(expected),
    "total_bytes": sum(item["size_bytes"] for item in expected),
}, sort_keys=True))
'''


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def parse_s3_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise RuntimeError(f"invalid S3 artifact prefix: {uri}")
    return parsed.netloc, parsed.path.strip("/")


def validate_plan(plan: dict[str, Any]) -> list[dict[str, str]]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "FROZEN":
        raise RuntimeError("surface mirror plan is not a frozen supported plan")
    surfaces = plan.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise RuntimeError("surface mirror plan has no surfaces")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in surfaces:
        if not isinstance(raw, dict):
            raise RuntimeError("invalid surface mirror row")
        sample_id = str(raw.get("sample_id", ""))
        surface_id = str(raw.get("surface_id", ""))
        digest = str(raw.get("artifact_sha256", ""))
        uri = str(raw.get("artifact_uri", ""))
        if not SAFE_IDENTIFIER.fullmatch(sample_id) or not SAFE_IDENTIFIER.fullmatch(surface_id):
            raise RuntimeError("unsafe sample or surface identifier")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("invalid catalogue artifact digest")
        bucket, prefix = parse_s3_prefix(uri)
        key = (bucket, prefix)
        if key in seen:
            raise RuntimeError("duplicate surface artifact prefix")
        seen.add(key)
        rows.append(
            {
                "sample_id": sample_id,
                "surface_id": surface_id,
                "artifact_sha256": digest,
                "artifact_uri": uri,
                "bucket": bucket,
                "prefix": prefix,
            }
        )
    return rows


def aws_fetch(uri: str) -> bytes:
    completed = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--only-show-errors"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"S3 read failed for {uri}: {message}")
    return completed.stdout


def build_archive(
    plan: dict[str, Any],
    fetch: Callable[[str], bytes] = aws_fetch,
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    expected: list[dict[str, Any]] = []
    surface_receipts: list[dict[str, Any]] = []
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for row in validate_plan(plan):
            prefix_uri = row["artifact_uri"].rstrip("/")
            manifest_body = fetch(f"{prefix_uri}/ARTIFACT_SET.json")
            manifest = json.loads(manifest_body.decode("utf-8"))
            if manifest.get("schema") != "campaignx.segmentation_artifact_set.v1":
                raise RuntimeError("unexpected TIFXYZ artifact manifest schema")
            if manifest.get("artifact_sha256") != row["artifact_sha256"]:
                raise RuntimeError("catalogue and artifact manifest digests differ")
            inventory = manifest.get("files")
            if not isinstance(inventory, dict) or set(inventory) != {
                "x.tif", "y.tif", "z.tif", "meta.json"
            }:
                raise RuntimeError("unexpected TIFXYZ artifact file inventory")
            bodies = {"ARTIFACT_SET.json": manifest_body}
            for name in REQUIRED_FILES[1:]:
                body = fetch(f"{prefix_uri}/{name}")
                metadata = inventory[name]
                if len(body) != int(metadata.get("size_bytes", -1)):
                    raise RuntimeError(f"artifact size mismatch: {name}")
                if sha256_bytes(body) != metadata.get("sha256"):
                    raise RuntimeError(f"artifact hash mismatch: {name}")
                bodies[name] = body
            relative_root = PurePosixPath(row["bucket"]) / row["prefix"]
            for name in REQUIRED_FILES:
                body = bodies[name]
                relative = str(relative_root / name)
                info = tarfile.TarInfo(relative)
                info.size = len(body)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(body))
                expected.append(
                    {
                        "path": relative,
                        "size_bytes": len(body),
                        "sha256": sha256_bytes(body),
                    }
                )
            surface_receipts.append(
                {
                    "sample_id": row["sample_id"],
                    "surface_id": row["surface_id"],
                    "artifact_sha256": row["artifact_sha256"],
                    "file_count": len(REQUIRED_FILES),
                    "total_bytes": sum(len(body) for body in bodies.values()),
                }
            )
    body = archive_buffer.getvalue()
    if len(body) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("relay archive exceeds safety limit")
    return body, expected, surface_receipts


def relay(
    *,
    archive: bytes,
    expected: list[dict[str, Any]],
    ssh_host: str,
    ssh_port: int,
    ssh_key: Path,
    remote_root: str,
) -> dict[str, Any]:
    if not SAFE_IDENTIFIER.fullmatch(ssh_host):
        raise RuntimeError("unsafe SSH host")
    if not remote_root.startswith("/") or "\n" in remote_root:
        raise RuntimeError("remote mirror root must be one absolute path")
    payload = base64.b64encode(canonical_bytes(expected)).decode("ascii")
    remote_command = " ".join(
        [
            "python3",
            "-c",
            shlex.quote(REMOTE_EXTRACTOR),
            shlex.quote(remote_root),
            shlex.quote(payload),
        ]
    )
    completed = subprocess.run(
        [
            "ssh",
            "-i",
            str(ssh_key.expanduser()),
            "-p",
            str(ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            ssh_host,
            remote_command,
        ],
        input=archive,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"credential-free SSH relay failed: {message}")
    return json.loads(completed.stdout.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    archive, expected, surfaces = build_archive(plan)
    remote = relay(
        archive=archive,
        expected=expected,
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        ssh_key=args.ssh_key,
        remote_root=args.remote_root,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_VERIFIED",
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_bytes(args.plan.read_bytes()),
        "transport": "IN_MEMORY_VALIDATED_TAR_OVER_BATCHMODE_SSH",
        "credentials_sent_to_worker": False,
        "presigned_urls_created": False,
        "local_artifact_staging": False,
        "archive_sha256": sha256_bytes(archive),
        "archive_bytes": len(archive),
        "surface_count": len(surfaces),
        "file_count": len(expected),
        "surfaces": surfaces,
        "remote_verification": remote,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
