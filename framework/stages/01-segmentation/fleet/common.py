from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "campaignx.segment_fleet"
ID_NAMESPACE = uuid.UUID("9ce0cf14-330c-54da-a175-26329924fc89")
FIXTURE_SAMPLE_PREFIXES = ("PHercDISTRIBUTEDTEST-",)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(kind: str, value: Any) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{kind}:{content_sha256(value)}"))


def is_fixture_surface(value: dict[str, Any]) -> bool:
    """Return whether a surface is explicitly non-scientific test material.

    The boolean marker is authoritative.  The narrow historical sample prefix
    keeps older distributed-probe receipts isolated without classifying
    arbitrary sample names as fixtures.
    """

    if value.get("fixture_only") is True:
        return True
    sample_id = value.get("sample_id")
    return isinstance(sample_id, str) and sample_id.startswith(
        FIXTURE_SAMPLE_PREFIXES
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def artifact_manifest(directory: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"required artifact is missing: {path}")
        result[name] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return result
