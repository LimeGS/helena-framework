#!/usr/bin/env python3
"""The control plane, in object storage, on a schedule.

Everything this platform produces already outlives the machine that made it: a
surface, a flattened sheet, a layer stack and a probability map are all published
with a digest. The record of *which* of them is certified, which cell was
attempted and what every verdict was lives in one PostgreSQL, on one host, and
until this existed there was no copy of it anywhere.

That is not a hypothetical. On 2026-07-28 the disk filled and PostgreSQL died
mid-recovery, and earlier the same week an rsync with --delete destroyed the
panel's accounts. Both were recoverable by luck rather than by design.

What it does, every interval and once at start:

    pg_dump -Fc            the whole database, custom format so it restores
    pg_restore --list      read it back, because a dump nobody has opened is a
                           hope rather than a backup
    aws s3 cp              to <prefix>/postgres/<utc>.dump
    tar + upload           the panel's state directory, when one is mounted:
                           accounts, overrides and their version history

Restored on 2026-07-28, which is the only evidence that matters: the newest dump
was pulled from the bucket into a clean PostgreSQL and came back with 67
surfaces, 252 tasks, 52 jobs and 50 flattenings -- the counts production held at
that moment.

Non-claims
----------
* A dump that parses is not a proven recovery of the *current* database. The
  restore above was one round on one day; the per-round check below is that the
  dump parses and lists its objects, which is the strongest thing available
  without a second database standing by.
* Retention is the bucket's business. This uploads and forgets; a lifecycle rule
  decides what to keep, because a deleter with credentials is a different risk
  from a writer with credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "campaignx.control_plane_backup.v1"


def utc_stamp(now: datetime | None = None) -> str:
    """The name a dump is filed under. Sorts lexicographically by time, which is
    what makes "the most recent backup" answerable with a list."""
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_database(dsn: str, destination: Path) -> Path:
    """The whole database in PostgreSQL's own custom format.

    -Fc rather than plain SQL: it restores selectively, it compresses, and
    pg_restore can list what is in it without a server to restore into.
    """
    subprocess.run(["pg_dump", "--format=custom", "--no-owner", "--no-acl",
                    "--file", str(destination), dsn],
                   check=True, capture_output=True, text=True)
    return destination


def verify_dump(path: Path) -> int:
    """Read the dump back and count what it holds.

    A dump written and never opened is a file, not a backup. This is what can be
    checked without a second database: that pg_restore parses it and finds
    objects. Zero objects from a database with tables means the dump is empty
    and the upload would be a lie.
    """
    listing = subprocess.run(["pg_restore", "--list", str(path)],
                             check=True, capture_output=True, text=True)
    entries = [line for line in listing.stdout.splitlines()
               if line.strip() and not line.startswith(";")]
    if not entries:
        raise RuntimeError(f"{path.name} lists no objects: the dump is empty")
    return len(entries)


def archive_directory(source: Path, destination: Path) -> Path:
    """The panel's state, which is small and is not in the database.

    Accounts, configuration overrides and their version history. Losing it does
    not lose an artefact; it loses everybody's ability to sign in, which this
    project has already done once.
    """
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=source.name)
    return destination


def upload(path: Path, target: str) -> None:
    subprocess.run(["aws", "s3", "cp", str(path), target],
                   check=True, capture_output=True, text=True)


def backup_once(*, dsn: str, prefix: str, state_dir: Path | None,
                runs_dir: Path | None, workspace: Path) -> dict:
    """One round. Returns what was written, for the log and for a receipt."""
    stamp = utc_stamp()
    receipt: dict = {"schema": SCHEMA, "stamp": stamp, "prefix": prefix,
                     "artefacts": []}

    dump = dump_database(dsn, workspace / f"control-plane-{stamp}.dump")
    receipt["artefacts"].append({
        "kind": "postgres", "objects": verify_dump(dump),
        "bytes": dump.stat().st_size, "sha256": sha256_of(dump),
        "uri": f"{prefix.rstrip('/')}/postgres/{stamp}.dump"})
    upload(dump, receipt["artefacts"][-1]["uri"])
    dump.unlink(missing_ok=True)

    if state_dir and state_dir.is_dir():
        archive = archive_directory(state_dir, workspace / f"panel-state-{stamp}.tgz")
        receipt["artefacts"].append({
            "kind": "panel_state", "bytes": archive.stat().st_size,
            "sha256": sha256_of(archive),
            "uri": f"{prefix.rstrip('/')}/panel-state/{stamp}.tgz"})
        upload(archive, receipt["artefacts"][-1]["uri"])
        archive.unlink(missing_ok=True)

    # Missions and the P0 selections they froze. The compose has mounted this
    # read-only for a while and nothing read it: the mount made the directory
    # reachable and this function still backed up two things. A mount is not a
    # backup, and the gap was invisible because the receipt only ever listed
    # what had been written.
    #
    # These are decisions rather than derivable state -- which scroll, which
    # cells, which selection was frozen -- so losing them loses the reasoning,
    # not just the bytes.
    if runs_dir and runs_dir.is_dir():
        archive = archive_directory(runs_dir, workspace / f"runs-{stamp}.tgz")
        receipt["artefacts"].append({
            "kind": "runs", "bytes": archive.stat().st_size,
            "sha256": sha256_of(archive),
            "uri": f"{prefix.rstrip('/')}/runs/{stamp}.tgz"})
        upload(archive, receipt["artefacts"][-1]["uri"])
        archive.unlink(missing_ok=True)

    # The receipt goes up too, so "what was in the backup" is answerable without
    # downloading the backup.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(receipt, handle, indent=1)
        note = Path(handle.name)
    upload(note, f"{prefix.rstrip('/')}/receipts/{stamp}.json")
    note.unlink(missing_ok=True)
    return receipt


def main() -> int:
    dsn = os.environ.get("HELENA_BACKUP_DSN") or os.environ.get("CX_DB", "")
    prefix = os.environ.get("HELENA_BACKUP_S3", "")
    if not dsn:
        print("no database: set HELENA_BACKUP_DSN or CX_DB", file=sys.stderr)
        return 2
    if not prefix.startswith("s3://"):
        print("no destination: set HELENA_BACKUP_S3 to an s3:// prefix",
              file=sys.stderr)
        return 2
    interval = float(os.environ.get("HELENA_BACKUP_INTERVAL_HOURS", "24"))
    state_dir = Path(os.environ["HELENA_BACKUP_STATE"]) \
        if os.environ.get("HELENA_BACKUP_STATE") else None
    # Where the compose mounts the runs root read-only. Absent is allowed and
    # skipped rather than fatal: a deployment that keeps missions elsewhere, or
    # has none yet, should still get its database backed up.
    runs_dir = Path(os.environ["HELENA_BACKUP_RUNS"]) \
        if os.environ.get("HELENA_BACKUP_RUNS") else None

    print(f"control-plane backup every {interval}h to {prefix}", flush=True)
    while True:
        started = time.time()
        try:
            with tempfile.TemporaryDirectory(prefix="helena-backup-") as workspace:
                receipt = backup_once(dsn=dsn, prefix=prefix, state_dir=state_dir,
                                      runs_dir=runs_dir, workspace=Path(workspace))
            for artefact in receipt["artefacts"]:
                print(f"  {artefact['kind']}: {artefact['bytes']} bytes -> "
                      f"{artefact['uri']}", flush=True)
        except subprocess.CalledProcessError as failure:
            # Loud and alive: a backup that dies on one bad round stops being a
            # backup silently, which is the failure mode it exists to prevent.
            print(f"backup failed: {failure.cmd[0]} exited {failure.returncode}: "
                  f"{(failure.stderr or '')[-400:]}", file=sys.stderr, flush=True)
        except Exception as failure:  # noqa: BLE001
            print(f"backup failed: {type(failure).__name__}: {failure}",
                  file=sys.stderr, flush=True)
        if interval <= 0:
            return 0
        time.sleep(max(interval * 3600 - (time.time() - started), 60))


if __name__ == "__main__":
    raise SystemExit(main())
