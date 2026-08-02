#!/usr/bin/env python3
"""Read a six-map Phase 4 replica grid from raw NPYs or a verified archive.

The storage-safe robust workflow may replace the eight large NPY cache files
with one lossless ZIP after analysis and visual export have completed.  This
module gives downstream diagnostic tools a single fail-closed reader:

* raw NPYs are preferred when all six canonical maps are present;
* a mixture of raw and archived canonical maps is rejected;
* archive fallback requires a completed storage-compaction receipt;
* the archive, every member, and every NPY payload are hash-checked.

It deliberately does not interpret model output or make an ink decision.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAP_PATTERN = re.compile(r"center-(\d+)_offset-(\d+)\.npy")
ARCHIVE_NAME = "REPLICA_MAPS_EVIDENCE.zip"
COMPACTION_RECEIPT_NAME = "STORAGE_COMPACTION_RECEIPT.json"
COMPACTION_KIND = "campaign_x_phase4_robust_storage_compaction_v1"
COMPACTION_COMPLETE_STATUS = "COMPLETED_DERIVED_CACHE_COMPACTION"
MAX_ARCHIVED_MEMBER_BYTES = 2 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON receipt: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"receipt is not a JSON object: {path}")
    return value


def parse_map_coordinates(name: str) -> tuple[int, int]:
    match = MAP_PATTERN.fullmatch(name)
    if match is None:
        raise RuntimeError(f"unexpected replica-map name: {name}")
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class ReplicaMapArtifact:
    """One immutable map input, either a file or a verified ZIP member."""

    name: str
    source: str
    sha256: str
    size_bytes: int
    path: Path | None = None
    archive_path: Path | None = None
    archive_member: str | None = None

    def load(self) -> np.ndarray:
        if self.path is not None:
            if not self.path.is_file():
                raise RuntimeError(f"raw replica is missing: {self.path}")
            if self.path.stat().st_size != self.size_bytes:
                raise RuntimeError(f"raw replica size changed: {self.path}")
            if sha256_file(self.path) != self.sha256:
                raise RuntimeError(f"raw replica hash changed: {self.path}")
            try:
                value = np.load(self.path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"replica is not a readable NPY array: {self.path}"
                ) from error
        else:
            if self.archive_path is None or self.archive_member is None:
                raise RuntimeError("archived replica lacks archive provenance")
            with zipfile.ZipFile(self.archive_path, "r") as archive:
                try:
                    payload = archive.read(self.archive_member)
                except (KeyError, OSError, zipfile.BadZipFile) as error:
                    raise RuntimeError(
                        f"cannot read archived replica {self.source}"
                    ) from error
            if len(payload) != self.size_bytes:
                raise RuntimeError(f"archived replica size changed: {self.source}")
            if sha256_bytes(payload) != self.sha256:
                raise RuntimeError(f"archived replica hash changed: {self.source}")
            try:
                value = np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"archived replica is not a readable NPY: {self.source}"
                ) from error
        if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
            raise RuntimeError(
                f"replica must be one two-dimensional numeric array: {self.source}"
            )
        return np.asarray(value)


def _validate_grid(artifacts: list[ReplicaMapArtifact]) -> None:
    if len(artifacts) != 6:
        raise RuntimeError(
            f"replica evidence must contain exactly six canonical maps; "
            f"found {len(artifacts)}"
        )
    coordinates = [parse_map_coordinates(item.name) for item in artifacts]
    if len(set(coordinates)) != 6:
        raise RuntimeError("replica evidence contains duplicate coordinates")
    depths = sorted({depth for depth, _ in coordinates})
    offsets = sorted({offset for _, offset in coordinates})
    if len(depths) != 3 or len(offsets) != 2:
        raise RuntimeError(
            "replica evidence is not a 3-depth by 2-offset grid: "
            f"depths={depths}, offsets={offsets}"
        )
    expected = {(depth, offset) for depth in depths for offset in offsets}
    if set(coordinates) != expected:
        raise RuntimeError("replica evidence Cartesian grid is incomplete")


def _raw_artifacts(screening_dir: Path) -> list[ReplicaMapArtifact]:
    paths = sorted(screening_dir.glob("center-*_offset-*.npy"))
    artifacts = [
        ReplicaMapArtifact(
            name=path.name,
            source=str(path),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            path=path,
        )
        for path in paths
    ]
    _validate_grid(artifacts)
    return artifacts


def _archived_artifacts(screening_dir: Path) -> list[ReplicaMapArtifact]:
    receipt_path = screening_dir.parent / COMPACTION_RECEIPT_NAME
    receipt = _read_json(receipt_path)
    if receipt.get("kind") != COMPACTION_KIND:
        raise RuntimeError("unexpected storage-compaction receipt kind")
    if receipt.get("status") != COMPACTION_COMPLETE_STATUS:
        raise RuntimeError("storage compaction is not complete")
    archive_record = receipt.get("npy_archive")
    if not isinstance(archive_record, dict):
        raise RuntimeError("storage receipt lacks NPY archive provenance")
    relative = str(archive_record.get("relative_to_window", "")).strip()
    if not relative:
        raise RuntimeError("storage receipt lacks archive relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("unsafe archive path in storage receipt")
    archive_path = (screening_dir.parent / relative_path).resolve()
    if archive_path != (screening_dir / ARCHIVE_NAME).resolve():
        raise RuntimeError("storage receipt points at a noncanonical archive")
    if not archive_path.is_file():
        raise RuntimeError(f"replica archive is missing: {archive_path}")
    expected_archive_hash = str(archive_record.get("sha256", ""))
    if sha256_file(archive_path) != expected_archive_hash:
        raise RuntimeError("replica archive SHA-256 mismatch")

    rows = archive_record.get("members")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("storage receipt lacks archive member inventory")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("archive member inventory row is not an object")
        name = str(row.get("name", ""))
        if Path(name).name != name or name in by_name:
            raise RuntimeError(f"unsafe or duplicate archive member: {name!r}")
        by_name[name] = row

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len({info.filename for info in infos}) != len(infos):
                raise RuntimeError("replica archive contains duplicate members")
            info_by_name = {info.filename: info for info in infos}
            if set(info_by_name) != set(by_name):
                raise RuntimeError("replica archive members differ from its receipt")
            for name, row in by_name.items():
                info = info_by_name[name]
                expected_size = int(row.get("size_bytes", -1))
                if (
                    expected_size < 1
                    or expected_size > MAX_ARCHIVED_MEMBER_BYTES
                    or info.file_size != expected_size
                ):
                    raise RuntimeError(f"invalid archived member size for {name}")
                with archive.open(info, "r") as stream:
                    digest = hashlib.sha256()
                    observed = 0
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        observed += len(block)
                        digest.update(block)
                if observed != expected_size:
                    raise RuntimeError(f"short archived member: {name}")
                if digest.hexdigest() != str(row.get("sha256", "")):
                    raise RuntimeError(f"archived member SHA-256 mismatch: {name}")
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"invalid replica archive: {archive_path}") from error

    artifacts = [
        ReplicaMapArtifact(
            name=name,
            source=f"{archive_path}#{name}",
            sha256=str(row["sha256"]),
            size_bytes=int(row["size_bytes"]),
            archive_path=archive_path,
            archive_member=name,
        )
        for name, row in sorted(by_name.items())
        if MAP_PATTERN.fullmatch(name)
    ]
    _validate_grid(artifacts)
    return artifacts


def discover_replica_maps(screening_dir: Path) -> list[ReplicaMapArtifact]:
    """Return exactly six maps from raw, archived, or exact rehydrated storage."""

    screening_dir = screening_dir.resolve()
    raw = sorted(screening_dir.glob("center-*_offset-*.npy"))
    archive = screening_dir / ARCHIVE_NAME
    if raw and len(raw) != 6:
        raise RuntimeError(
            "raw replica grid must contain exactly six maps; " f"found {len(raw)}"
        )
    if raw and archive.is_file():
        # ``rehydrate`` intentionally retains the immutable archive.  Coexistence
        # is accepted only when every raw byte hash equals the archived member;
        # any drift or partial extraction still fails closed.
        raw_artifacts = _raw_artifacts(screening_dir)
        archived_artifacts = _archived_artifacts(screening_dir)
        archived_by_name = {item.name: item for item in archived_artifacts}
        if {item.name: item.sha256 for item in raw_artifacts} != {
            name: item.sha256 for name, item in archived_by_name.items()
        }:
            raise RuntimeError(
                "rehydrated raw replicas differ from the verified archive"
            )
        return raw_artifacts
    if raw:
        return _raw_artifacts(screening_dir)
    if archive.is_file():
        return _archived_artifacts(screening_dir)
    raise RuntimeError(f"no raw or archived replica evidence found in {screening_dir}")
