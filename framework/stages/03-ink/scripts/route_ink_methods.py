#!/usr/bin/env python3
"""Resolve campaign-approved ink lanes for one scan, without target tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def load_profiles(profiles_root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    profiles: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(profiles_root.glob("*.json")):
        profile = read_object(path)
        profile_id = profile.get("profile_id")
        if not profile_id:
            raise RuntimeError(f"profile has no profile_id: {path}")
        if profile_id in profiles:
            raise RuntimeError(f"duplicate profile_id: {profile_id}")
        profiles[profile_id] = (profile, path)
    return profiles


def stable_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def verify_control_receipt(
    control: dict[str, Any], *, path_base: Path, method_id: str
) -> dict[str, str]:
    receipt_path = Path(control["path"])
    if not receipt_path.is_absolute():
        receipt_path = path_base / receipt_path
    if not receipt_path.is_file():
        raise RuntimeError(f"control receipt does not exist for {method_id}: {control['path']}")
    actual_hash = sha256_file(receipt_path)
    if actual_hash != control["sha256"]:
        raise RuntimeError(
            f"control receipt SHA-256 mismatch for {method_id}: "
            f"{actual_hash} != {control['sha256']}"
        )
    receipt = read_object(receipt_path)
    if receipt.get("status") != control["status"]:
        raise RuntimeError(
            f"control receipt status mismatch for {method_id}: "
            f"{receipt.get('status')} != {control['status']}"
        )
    return {
        "path": stable_path(receipt_path, path_base),
        "sha256": actual_hash,
        "status": control["status"],
    }


def route(
    *,
    registry_path: Path,
    policy_path: Path,
    profiles_root: Path,
    sample_id: str,
    path_base: Path | None = None,
) -> dict[str, Any]:
    registry = read_object(registry_path)
    policy = read_object(policy_path)
    methods = {
        entry["method_id"]: entry for entry in registry.get("entries", [])
    }
    profiles = load_profiles(profiles_root)
    targets = {
        target["sample_id"]: target for target in policy.get("targets", [])
    }
    if sample_id not in targets:
        raise RuntimeError(
            f"sample is not declared by this campaign policy: {sample_id}"
        )
    target = targets[sample_id]
    voxel_size = float(target["voxel_size_um"])
    base = (path_base or Path.cwd()).resolve()
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lane in policy.get("lanes", []):
        method_id = lane["method_id"]
        if method_id in seen:
            raise RuntimeError(f"duplicate method in policy: {method_id}")
        seen.add(method_id)
        if method_id not in methods:
            raise RuntimeError(f"policy references unknown method: {method_id}")
        compatibility = lane.get("compatible_voxel_size_um")
        compatible = True
        if compatibility:
            minimum = float(compatibility["minimum"])
            maximum = float(compatibility["maximum"])
            if minimum > maximum:
                raise RuntimeError(f"invalid voxel range for {method_id}")
            compatible = minimum <= voxel_size <= maximum
        row = {
            "method_id": method_id,
            "profile_id": lane.get("profile_id"),
            "disposition": lane["disposition"],
            "voxel_compatible": compatible,
            "reason": lane["reason"],
        }
        control = lane.get("control_receipt")
        if control:
            row["control_receipt"] = verify_control_receipt(
                control, path_base=base, method_id=method_id
            )
        if lane["disposition"] == "ENABLED_SCREENING" and compatible:
            if methods[method_id].get("integration_status") not in {
                "RUNNABLE_PRIMARY",
                "RUNNABLE_AUXILIARY",
                "RUNNABLE_DIAGNOSTIC",
            }:
                raise RuntimeError(
                    f"enabled method has no runnable adapter: {method_id}"
                )
            profile_id = lane.get("profile_id")
            if not profile_id:
                raise RuntimeError(f"enabled method has no profile: {method_id}")
            if profile_id not in profiles:
                raise RuntimeError(
                    f"enabled method references unknown profile: {profile_id}"
                )
            profile, profile_path = profiles[profile_id]
            if profile.get("method_id") != method_id:
                raise RuntimeError(
                    f"profile method mismatch: {profile_id} != {method_id}"
                )
            if profile.get("safety", {}).get("requires_campaign_control") and not control:
                raise RuntimeError(
                    f"enabled method requires a campaign control receipt: {method_id}"
                )
            expected_checkpoint = methods[method_id].get("known_checkpoint_sha256")
            if expected_checkpoint and profile.get("checkpoint_sha256") != expected_checkpoint:
                raise RuntimeError(f"profile checkpoint mismatch: {profile_id}")
            expected_adapter = methods[method_id].get("local_adapter")
            if expected_adapter and profile.get("adapter") != expected_adapter:
                raise RuntimeError(f"profile adapter mismatch: {profile_id}")
            adapter = base / profile["adapter"]
            if not adapter.is_file():
                raise RuntimeError(f"profile adapter does not exist: {profile['adapter']}")
            row["profile"] = {
                "path": stable_path(profile_path, base),
                "sha256": sha256_file(profile_path),
                "adapter": stable_path(adapter, base),
                "adapter_sha256": sha256_file(adapter),
                "checkpoint_sha256": profile["checkpoint_sha256"],
            }
            selected.append(row)
        else:
            excluded.append(row)
    if not selected:
        raise RuntimeError(f"no enabled compatible ink lane for {sample_id}")
    return {
        "schema": "campaignx.ink_method_route_decision.v1",
        "generated_at_utc": utc_now(),
        "status": "ROUTED_SCREENING_ONLY",
        "campaign_id": policy["campaign_id"],
        "sample": target,
        "registry": {
            "path": stable_path(registry_path, base),
            "sha256": sha256_file(registry_path),
        },
        "policy": {
            "path": stable_path(policy_path, base),
            "sha256": sha256_file(policy_path),
            "policy_id": policy["policy_id"],
        },
        "selected_lanes": selected,
        "excluded_lanes": excluded,
        "non_claims": [
            "routing does not accept ink",
            "routing does not identify letters",
            "a silent lane does not prove absence",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--path-base", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = route(
        registry_path=args.registry.resolve(),
        policy_path=args.policy.resolve(),
        profiles_root=args.profiles_root.resolve(),
        sample_id=args.sample_id,
        path_base=args.path_base.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
