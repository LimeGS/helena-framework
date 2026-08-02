#!/usr/bin/env python3
"""Build an additive, hash-authoritative overlay for ink receipts.

Historical receipts are immutable evidence.  This tool never rewrites them.
It identifies the executed method by checkpoint SHA-256, records declared
family disagreements, and can fail closed for new production namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RECEIPT_NAME = "INK_SCREENING_RECEIPT.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def stable_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def iter_receipts(roots: Iterable[Path]) -> list[Path]:
    found: dict[Path, Path] = {}
    for root in roots:
        resolved = root.resolve()
        if resolved.is_file():
            if resolved.name != RECEIPT_NAME:
                raise RuntimeError(f"not an {RECEIPT_NAME}: {resolved}")
            found[resolved] = resolved
            continue
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        for path in resolved.rglob(RECEIPT_NAME):
            found[path.resolve()] = path.resolve()
    return sorted(found.values(), key=lambda path: path.as_posix())


def checkpoint_index(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in registry.get("entries", []):
        if not isinstance(entry, dict):
            raise RuntimeError("registry entry is not an object")
        digest = entry.get("known_checkpoint_sha256")
        if not digest:
            continue
        if digest in result:
            raise RuntimeError(f"checkpoint hash appears twice in registry: {digest}")
        result[str(digest)] = entry
    return result


def audit_receipt(
    path: Path,
    *,
    base: Path,
    by_checkpoint: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        receipt = read_object(path)
    except Exception as error:
        return {
            "receipt_path": stable_path(path, base),
            "receipt_sha256": sha256_file(path),
            "identity_status": "INVALID_RECEIPT_JSON",
            "error": str(error),
        }
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return {
            "receipt_path": stable_path(path, base),
            "receipt_sha256": sha256_file(path),
            "receipt_status": receipt.get("status"),
            "identity_status": "MISSING_CHECKPOINT_OBJECT",
        }
    digest = checkpoint.get("sha256")
    declared_family = checkpoint.get("model_family")
    if not isinstance(digest, str) or len(digest) != 64:
        return {
            "receipt_path": stable_path(path, base),
            "receipt_sha256": sha256_file(path),
            "receipt_status": receipt.get("status"),
            "declared_model_family": declared_family,
            "declared_checkpoint_sha256": digest,
            "identity_status": "MISSING_OR_INVALID_CHECKPOINT_SHA256",
        }
    method = by_checkpoint.get(digest)
    if method is None:
        return {
            "receipt_path": stable_path(path, base),
            "receipt_sha256": sha256_file(path),
            "receipt_status": receipt.get("status"),
            "declared_model_family": declared_family,
            "declared_checkpoint_sha256": digest,
            "identity_status": "UNKNOWN_CHECKPOINT_SHA256",
        }
    aliases = list(method.get("receipt_model_family_aliases", []))
    family_matches = declared_family in aliases
    return {
        "receipt_path": stable_path(path, base),
        "receipt_sha256": sha256_file(path),
        "receipt_status": receipt.get("status"),
        "declared_model_family": declared_family,
        "declared_checkpoint_sha256": digest,
        "canonical_method_id": method["method_id"],
        "canonical_model_family_aliases": aliases,
        "family_matches_checkpoint": family_matches,
        "identity_status": (
            "KNOWN_CHECKPOINT_FAMILY_MATCH"
            if family_matches
            else "KNOWN_CHECKPOINT_FAMILY_MISMATCH"
        ),
    }


def build_overlay(
    *, registry_path: Path, roots: list[Path], base: Path
) -> dict[str, Any]:
    registry = read_object(registry_path)
    rows = [
        audit_receipt(path, base=base, by_checkpoint=checkpoint_index(registry))
        for path in iter_receipts(roots)
    ]
    counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["identity_status"])
        counts[status] = counts.get(status, 0) + 1
        method = row.get("canonical_method_id")
        if method:
            method_counts[str(method)] = method_counts.get(str(method), 0) + 1
    invalid = sum(
        count
        for status, count in counts.items()
        if status != "KNOWN_CHECKPOINT_FAMILY_MATCH"
    )
    return {
        "schema": "campaignx.ink_receipt_provenance_overlay.v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETE_ALL_IDENTITIES_MATCH" if invalid == 0 else "COMPLETED_WITH_IDENTITY_EXCEPTIONS",
        "registry": {
            "path": stable_path(registry_path, base),
            "sha256": sha256_file(registry_path),
            "version": registry.get("registry_version"),
        },
        "scan_roots": [stable_path(root, base) for root in roots],
        "receipt_count": len(rows),
        "identity_status_counts": dict(sorted(counts.items())),
        "canonical_method_counts": dict(sorted(method_counts.items())),
        "receipts": rows,
        "policy": [
            "checkpoint SHA-256 is authoritative",
            "historical receipts are never rewritten",
            "family mismatches are corrected only through this additive overlay",
            "strict mode rejects every mismatch, unknown hash, or malformed receipt",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, action="append", required=True)
    parser.add_argument("--path-base", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    overlay = build_overlay(
        registry_path=args.registry.resolve(),
        roots=[root.resolve() for root in args.scan_root],
        base=args.path_base.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: overlay[key] for key in ("status", "receipt_count", "identity_status_counts", "canonical_method_counts")}, indent=2, sort_keys=True))
    if args.strict and overlay["status"] != "COMPLETE_ALL_IDENTITIES_MATCH":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
