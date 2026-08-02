#!/usr/bin/env python3
"""Remove regenerable QC staging only after durable evidence is verified.

The command is intentionally conservative: it reads only COMPLETED jobs,
requires a local file:// evidence URI, verifies both source and durable
manifest hashes, confines paths to the declared run/evidence roots, and writes
one immutable cleanup receipt per attempt.  PNG/JSON evidence is never removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse


SCHEMA = "campaignx.completed_qc_staging_cleanup.v1"
REGENERABLE_DIRECTORIES = ("surface", "tiffs", "ct_metadata_cache.zarr")


class CleanupError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise CleanupError("only local file:// durable evidence is eligible")
    return Path(unquote(parsed.path)).resolve()


def clean_completed(
    database: Path,
    run_root: Path,
    evidence_root: Path,
    *,
    apply: bool,
) -> dict:
    database = database.resolve()
    run_root = run_root.resolve()
    evidence_root = evidence_root.resolve()
    connection = sqlite3.connect(database)
    rows = list(
        connection.execute(
            "SELECT qc_job_id, surface_id, result_json FROM qc_jobs "
            "WHERE state = 'COMPLETED' ORDER BY qc_job_id"
        )
    )
    connection.close()

    attempts = []
    total_bytes = 0
    for qc_job_id, surface_id, raw_result in rows:
        result = json.loads(raw_result)
        expected = result.get("evidence_manifest_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise CleanupError(f"{qc_job_id}: missing durable manifest hash")
        durable = _file_uri(str(result.get("evidence_uri", "")))
        if not _inside(durable, evidence_root) or not durable.is_file():
            raise CleanupError(f"{qc_job_id}: durable manifest is missing or out of root")
        executor = result.get("executor_receipt") or {}
        source = Path(str(executor.get("evidence_manifest_path", ""))).resolve()
        if not _inside(source, run_root) or not source.is_file():
            raise CleanupError(f"{qc_job_id}: source manifest is missing or out of root")
        if sha256(source) != expected or sha256(durable) != expected:
            raise CleanupError(f"{qc_job_id}: evidence manifest hash mismatch")
        output = source.parent
        attempt_dir = output.parent
        receipt = attempt_dir / "POSTHOC_REGENERABLE_CLEANUP.json"
        removable = []
        for name in REGENERABLE_DIRECTORIES:
            path = output / name
            if path.exists():
                size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
                removable.append((path, size))
        for path in sorted(output.rglob("*.npy")):
            removable.append((path, path.stat().st_size))
        bytes_for_attempt = sum(size for _, size in removable)
        if apply and removable:
            for path, _ in removable:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
        row = {
            "qc_job_id": qc_job_id,
            "surface_id": surface_id,
            "attempt_dir": str(attempt_dir),
            "evidence_uri": result["evidence_uri"],
            "evidence_manifest_sha256": expected,
            "apply": apply,
            "removed_bytes": bytes_for_attempt if apply else 0,
            "eligible_bytes": bytes_for_attempt,
            "removed": [str(path.relative_to(output)) for path, _ in removable],
            "durable_evidence_unchanged": True,
        }
        if apply and not receipt.exists():
            receipt.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "created_at_utc": datetime.now(UTC).isoformat(),
                        **row,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        attempts.append(row)
        total_bytes += bytes_for_attempt
    return {
        "schema": SCHEMA,
        "mode": "APPLY" if apply else "DRY_RUN",
        "completed_job_count": len(rows),
        "attempts_with_regenerable_payloads": sum(bool(row["removed"]) for row in attempts),
        "eligible_bytes": total_bytes,
        "removed_bytes": total_bytes if apply else 0,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise CleanupError(f"refusing to overwrite: {args.output}")
    result = clean_completed(
        args.database,
        args.run_root,
        args.evidence_root,
        apply=args.apply,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("mode", "completed_job_count", "eligible_bytes", "removed_bytes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
