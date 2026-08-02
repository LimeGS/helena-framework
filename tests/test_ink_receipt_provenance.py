from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework"
    / "stages"
    / "03-ink"
    / "scripts"
    / "audit_ink_receipt_provenance.py"
)
SPEC = importlib.util.spec_from_file_location("ink_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def registry(path: Path) -> None:
    write_json(
        path,
        {
            "registry_version": "1.0.0",
            "entries": [
                {
                    "method_id": "known@1.0.0",
                    "known_checkpoint_sha256": "a" * 64,
                    "receipt_model_family_aliases": ["known-family"],
                }
            ],
        },
    )


def receipt(path: Path, digest: str, family: str) -> None:
    write_json(
        path,
        {
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "checkpoint": {"sha256": digest, "model_family": family},
        },
    )


def test_overlay_uses_hash_and_records_family_mismatch(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry(registry_path)
    receipt(tmp_path / "one" / "INK_SCREENING_RECEIPT.json", "a" * 64, "wrong")
    overlay = MODULE.build_overlay(
        registry_path=registry_path,
        roots=[tmp_path],
        base=tmp_path,
    )
    assert overlay["receipt_count"] == 1
    assert overlay["canonical_method_counts"] == {"known@1.0.0": 1}
    row = overlay["receipts"][0]
    assert row["identity_status"] == "KNOWN_CHECKPOINT_FAMILY_MISMATCH"
    assert row["canonical_method_id"] == "known@1.0.0"


def test_overlay_is_complete_only_when_every_identity_matches(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry(registry_path)
    receipt(tmp_path / "one" / "INK_SCREENING_RECEIPT.json", "a" * 64, "known-family")
    overlay = MODULE.build_overlay(
        registry_path=registry_path,
        roots=[tmp_path / "one"],
        base=tmp_path,
    )
    assert overlay["status"] == "COMPLETE_ALL_IDENTITIES_MATCH"


def test_unknown_checkpoint_fails_identity_audit(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry(registry_path)
    receipt(tmp_path / "one" / "INK_SCREENING_RECEIPT.json", "b" * 64, "new")
    overlay = MODULE.build_overlay(
        registry_path=registry_path,
        roots=[tmp_path],
        base=tmp_path,
    )
    assert overlay["status"] == "COMPLETED_WITH_IDENTITY_EXCEPTIONS"
    assert overlay["identity_status_counts"] == {
        "UNKNOWN_CHECKPOINT_SHA256": 1
    }
