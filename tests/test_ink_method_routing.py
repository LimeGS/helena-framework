from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework"
    / "stages"
    / "03-ink"
    / "scripts"
    / "route_ink_methods.py"
)
SPEC = importlib.util.spec_from_file_location("ink_router", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    registry = tmp_path / "registry.json"
    policy = tmp_path / "policy.json"
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    adapter = tmp_path / "adapter.py"
    adapter.write_text("# fixture adapter\n", encoding="utf-8")
    control = tmp_path / "control.json"
    write_json(control, {"status": "CONTROL_PASSED"})
    write_json(
        registry,
        {
            "entries": [
                {
                    "method_id": "enabled@1.0.0",
                    "integration_status": "RUNNABLE_PRIMARY",
                    "known_checkpoint_sha256": "a" * 64,
                    "local_adapter": "adapter.py",
                },
                {
                    "method_id": "pending@1.0.0",
                    "integration_status": "KNOWN_NOT_INTEGRATED",
                },
            ]
        },
    )
    write_json(
        policy,
        {
            "campaign_id": "test",
            "policy_id": "test@1.0.0",
            "targets": [{"sample_id": "roll-a", "voxel_size_um": 9.0}],
            "lanes": [
                {
                    "method_id": "enabled@1.0.0",
                    "profile_id": "enabled-profile@1.0.0",
                    "disposition": "ENABLED_SCREENING",
                    "compatible_voxel_size_um": {
                        "minimum": 8.0,
                        "maximum": 10.0,
                    },
                    "reason": "control passed",
                    "control_receipt": {
                        "path": "control.json",
                        "sha256": MODULE.sha256_file(control),
                        "status": "CONTROL_PASSED",
                    },
                },
                {
                    "method_id": "pending@1.0.0",
                    "profile_id": None,
                    "disposition": "PENDING_ADAPTER_AND_CONTROL",
                    "reason": "not ready",
                },
            ],
        },
    )
    write_json(
        profiles / "enabled.json",
        {
            "profile_id": "enabled-profile@1.0.0",
            "method_id": "enabled@1.0.0",
            "checkpoint_sha256": "a" * 64,
            "adapter": "adapter.py",
            "safety": {"requires_campaign_control": True},
        },
    )
    return registry, policy, profiles


def test_router_selects_only_enabled_compatible_lane(tmp_path: Path) -> None:
    registry, policy, profiles = fixture(tmp_path)
    result = MODULE.route(
        registry_path=registry,
        policy_path=policy,
        profiles_root=profiles,
        sample_id="roll-a",
        path_base=tmp_path,
    )
    assert [row["method_id"] for row in result["selected_lanes"]] == [
        "enabled@1.0.0"
    ]
    assert result["excluded_lanes"][0]["method_id"] == "pending@1.0.0"
    selected = result["selected_lanes"][0]
    assert selected["control_receipt"]["status"] == "CONTROL_PASSED"
    assert selected["profile"]["adapter_sha256"] == MODULE.sha256_file(
        tmp_path / "adapter.py"
    )


def test_router_rejects_sample_outside_policy(tmp_path: Path) -> None:
    registry, policy, profiles = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="not declared"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-b",
            path_base=tmp_path,
        )


def test_router_rejects_enabled_non_runnable_method(tmp_path: Path) -> None:
    registry, policy, profiles = fixture(tmp_path)
    value = json.loads(policy.read_text(encoding="utf-8"))
    value["lanes"][0]["method_id"] = "pending@1.0.0"
    value["lanes"] = value["lanes"][:1]
    write_json(policy, value)
    with pytest.raises(RuntimeError, match="no runnable adapter"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=tmp_path,
        )


def test_router_rejects_missing_or_mismatched_profile(tmp_path: Path) -> None:
    registry, policy, profiles = fixture(tmp_path)
    (profiles / "enabled.json").unlink()
    with pytest.raises(RuntimeError, match="unknown profile"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=tmp_path,
        )


def test_router_rejects_checkpoint_or_adapter_drift(tmp_path: Path) -> None:
    registry, policy, profiles = fixture(tmp_path)
    profile_path = profiles / "enabled.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["checkpoint_sha256"] = "b" * 64
    write_json(profile_path, profile)
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=tmp_path,
        )

    profile["checkpoint_sha256"] = "a" * 64
    profile["adapter"] = "different.py"
    write_json(profile_path, profile)
    with pytest.raises(RuntimeError, match="adapter mismatch"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=tmp_path,
        )

    write_json(
        profiles / "enabled.json",
        {
            "profile_id": "enabled-profile@1.0.0",
            "method_id": "pending@1.0.0",
            "checkpoint_sha256": "a" * 64,
            "adapter": "adapter.py",
        },
    )
    with pytest.raises(RuntimeError, match="profile method mismatch"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=tmp_path,
        )


def test_router_rejects_missing_or_drifted_required_control(tmp_path: Path) -> None:
    registry, policy, profiles = fixture(tmp_path)
    value = json.loads(policy.read_text(encoding="utf-8"))
    del value["lanes"][0]["control_receipt"]
    write_json(policy, value)
    with pytest.raises(RuntimeError, match="requires a campaign control"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=tmp_path,
        )

    drift_root = tmp_path / "drift"
    drift_root.mkdir()
    registry, policy, profiles = fixture(drift_root)
    (drift_root / "control.json").write_text('{"status":"CHANGED"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="control receipt SHA-256 mismatch"):
        MODULE.route(
            registry_path=registry,
            policy_path=policy,
            profiles_root=profiles,
            sample_id="roll-a",
            path_base=drift_root,
        )


@needs_campaign_evidence
def test_campaign_policy_references_registered_methods_and_real_receipts() -> None:
    registry = json.loads(
        (ROOT / "framework/registries/method-capabilities-0.1.0.json").read_text(
            encoding="utf-8"
        )
    )
    methods = {entry["method_id"] for entry in registry["entries"]}
    policy = json.loads(
        (
            ROOT
            / "workspace/campaigns/campaign-x-2026/plans/INK_METHOD_ROUTING_POLICY_0.1.0.json"
        ).read_text(encoding="utf-8")
    )
    assert {lane["method_id"] for lane in policy["lanes"]} <= methods
    for lane in policy["lanes"]:
        control = lane.get("control_receipt")
        if not control:
            continue
        path = ROOT / control["path"]
        assert path.is_file()
        assert MODULE.sha256_file(path) == control["sha256"]
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == control[
            "status"
        ]
