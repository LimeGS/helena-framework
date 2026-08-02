from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/helena_relay_s3_surface_mirror.py"
SPEC = importlib.util.spec_from_file_location("surface_relay", SCRIPT)
assert SPEC and SPEC.loader
relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relay)


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def fixture() -> tuple[dict, dict[str, bytes]]:
    prefix = "s3://private-bucket/surfaces/PHerc1/surface-1"
    bodies = {
        "x.tif": b"x",
        "y.tif": b"yy",
        "z.tif": b"zzz",
        "meta.json": b"{}\n",
    }
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "artifact_sha256": "a" * 64,
        "files": {
            name: {"size_bytes": len(body), "sha256": _sha(body)}
            for name, body in bodies.items()
        },
    }
    objects = {
        f"{prefix}/ARTIFACT_SET.json": json.dumps(manifest).encode(),
        **{f"{prefix}/{name}": body for name, body in bodies.items()},
    }
    plan = {
        "schema": relay.PLAN_SCHEMA,
        "status": "FROZEN",
        "surfaces": [
            {
                "sample_id": "PHerc1",
                "surface_id": "surface-1",
                "artifact_uri": prefix,
                "artifact_sha256": "a" * 64,
            }
        ],
    }
    return plan, objects


def test_archive_is_verified_and_preserves_bucket_key_hierarchy() -> None:
    plan, objects = fixture()
    archive, expected, surfaces = relay.build_archive(plan, objects.__getitem__)
    assert len(expected) == 5
    assert surfaces[0]["file_count"] == 5
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as stream:
        names = stream.getnames()
    assert names == [
        f"private-bucket/surfaces/PHerc1/surface-1/{name}"
        for name in relay.REQUIRED_FILES
    ]


def test_archive_fails_closed_on_hash_mismatch() -> None:
    plan, objects = fixture()
    objects["s3://private-bucket/surfaces/PHerc1/surface-1/x.tif"] = b"wrong"
    with pytest.raises(RuntimeError, match="artifact size mismatch"):
        relay.build_archive(plan, objects.__getitem__)


def test_plan_rejects_duplicate_prefixes() -> None:
    plan, _ = fixture()
    plan["surfaces"].append(dict(plan["surfaces"][0], surface_id="surface-2"))
    with pytest.raises(RuntimeError, match="duplicate surface artifact prefix"):
        relay.validate_plan(plan)
