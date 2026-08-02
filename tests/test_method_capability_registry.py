from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "framework/registries/method-capabilities-0.1.0.json"
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
METHOD_ID = re.compile(r"^[a-z0-9][a-z0-9._-]+@[0-9]+\.[0-9]+\.[0-9]+$")


def test_method_capability_registry_is_unique_and_versioned() -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert document["schema"] == "campaignx.method_capability_registry.v1"
    assert SEMVER.fullmatch(document["registry_version"])
    ids = [entry["method_id"] for entry in document["entries"]]
    assert len(ids) == len(set(ids))
    assert all(METHOD_ID.fullmatch(method_id) for method_id in ids)


def test_registry_distinguishes_availability_execution_and_transfer() -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = {entry["method_id"]: entry for entry in document["entries"]}
    assert entries["vc3d-m7-seed-grow@1.0.0"]["integration_status"] == "RUNNABLE_PRIMARY"
    assert entries["scrollfiesta-m7-mesh@0.1.1"]["validation_status"] == "FAILED_REFERENCE_CONTROL"
    assert entries["ink-3d-dino-guided@1.0.0"]["integration_status"] == "RUNNABLE_DIAGNOSTIC"
    assert entries["ink-3d-dino-guided@1.0.0"]["validation_status"] == "LOCAL_FUNCTIONAL_ONLY"
    assert entries["timesformer-gp-scroll1@1.0.0"]["validation_status"] == "CONTROL_SUPPORTED_TARGET_UNCALIBRATED"


def test_checkpointed_methods_are_identified_by_hash_when_locally_frozen() -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = {entry["method_id"]: entry for entry in document["entries"]}
    assert entries["timesformer-gp-scroll1@1.0.0"]["known_checkpoint_sha256"] == (
        "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488"
    )
    assert entries["timesformer-scroll5-pherc172@1.0.0"]["known_checkpoint_sha256"] == (
        "b5f9f48c231728548e906ec52b56434aa2bbe86e90927212be1322314d35b331"
    )


def test_registry_does_not_embed_active_campaign_target_routing() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    for target in (
        "PHerc125",
        "PHerc191",
        "PHerc211",
        "PHerc257",
        "PHerc268",
        "PHerc358",
        "PHerc800",
        "PHerc813",
        "PHerc826",
        "PHerc1203",
        "PHerc1218",
        "PHerc1447",
        "PHerc1545",
    ):
        assert target not in text


def _profiles(root) -> list[dict]:
    """Every ink lane profile, for the supersession check."""
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "framework/profiles/03-ink").glob("*.json"))]


def _checkpoint_digests(value: object) -> list[str]:
    """Every ``*checkpoint_sha256`` at any depth, so none can hide in a nest."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("checkpoint_sha256") and isinstance(item, str):
                found.append(item)
            found.extend(_checkpoint_digests(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_checkpoint_digests(item))
    return found


def test_ink_profiles_bind_registered_method_and_checkpoint() -> None:
    """Mirror both halves of the audit's rule at ``audit_business_logic.py:632-667``.

    Not every file under ``framework/profiles/03-ink`` is an executable lane
    profile.  A threshold calibration declaration binds a *criterion*, not a
    model, so it names no method and carries no checkpoint — demanding a
    ``method_id`` from it would be wrong.

    Skipping such a file entirely would be a weakening, though: a real model
    profile that simply forgot its ``method_id`` would then smuggle an
    unregistered checkpoint through.  So the binding check is conditional while
    the checkpoint check is not.
    """

    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = {entry["method_id"]: entry for entry in document["entries"]}
    known_digests = {
        entry["known_checkpoint_sha256"]
        for entry in document["entries"]
        if entry.get("known_checkpoint_sha256")
    }
    profiles = sorted((ROOT / "framework/profiles/03-ink").glob("*.json"))
    assert profiles
    profile_ids: set[str] = set()
    bound = 0
    for path in profiles:
        profile = json.loads(path.read_text(encoding="utf-8"))
        assert profile["profile_id"] not in profile_ids
        profile_ids.add(profile["profile_id"])

        # Half two, unconditional: no profile may carry an unknown checkpoint.
        for digest in _checkpoint_digests(profile):
            assert digest in known_digests, f"{path.name} names an unregistered checkpoint"

        # Half one, conditional: only a profile that names a method is bound.
        if "method_id" not in profile:
            assert "checkpoint_sha256" not in profile, (
                f"{path.name} carries a checkpoint but names no method"
            )
            continue
        method = entries[profile["method_id"]]
        assert profile["checkpoint_sha256"] == method["known_checkpoint_sha256"]
        # The adapter must be the registry's, unless a later profile supersedes
        # this one. A superseded profile records what it ran with -- its hash is
        # in receipts and editing it would orphan them -- and the registry
        # records what the method should be run under now. The canonical 2 um
        # lane is exactly that case: @1.0.0 named the generic runner, which
        # normalises and resamples differently from the recipe's own, and the
        # community control measured what that costs.
        superseded = any(other.get("supersedes") == profile["profile_id"]
                         for other in _profiles(ROOT))
        if not superseded:
            assert profile["adapter"] == method["local_adapter"]
        assert (ROOT / profile["adapter"]).is_file()
        bound += 1

    assert bound >= 4, "the executable ink lanes must still be bound"
