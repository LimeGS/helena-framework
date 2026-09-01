from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .common import content_sha256, stable_id
from .seed_probe import validate_experimental_arm_admission


BASELINE_RECONCILIATION_SCHEMA = (
    "campaignx.first_letters_discovery_baseline_reconciliation.v1"
)
NATIVE_ADAPTER_SCHEMA = "campaignx.first_letters_discovery_native_adapter.v1"
DISPATCH_SCHEMA = "campaignx.first_letters_discovery_dispatch.v1"
JOB_SCHEMA = "campaignx.first_letters_discovery_job.v1"
JOB_CLAIM_SCHEMA = "campaignx.first_letters_discovery_job_claim.v1"


@dataclass(frozen=True, slots=True)
class FirstLettersDiscoveryJobClaim:
    """Sealed job-rooted packet; the opaque v18 handle never crosses APIs."""

    job_id: str
    job_sha256: str
    dispatch_id: str
    dispatch_sha256: str
    adapter_sha256: str
    reservation_id: str
    reservation_sha256: str
    item_id: str
    profile_file_sha256: str
    source_snapshot_sha256: str
    run_id: str
    worker_id: str
    provider_request_sha256: str
    claim_sha256: str
    _run_handle: Any = field(repr=False, compare=False)

_RECONCILIATION_FIELDS = {
    "schema", "mission_id", "request_id", "sample_id",
    "budget_admission_sha256", "source_snapshot_id",
    "source_snapshot_sha256", "source_content_lock_sha256",
    "accepted_p0_artifact_id", "accepted_p0_artifact_sha256",
    "grid_version", "ordered_item_ids", "ordered_item_ids_sha256",
    "ordered_item_bindings", "ordered_item_bindings_sha256",
    "cap_authority_id", "cap_authority_sha256", "profile_file_sha256",
    "profile_scientific_core_sha256", "policy_sha256",
    "deployed_revision", "history_manifest_sha256", "mode", "namespace",
    "canonical_admission", "top_k", "probe_generations",
    "maximum_attempts_per_candidate", "units_per_item",
    "allow_unvalidated", "reconciliation_sha256",
}

_RECONCILIATION_BINDING_FIELDS = {
    "schema", "item_id", "selection_rank", "sample_id",
    "source_snapshot_id", "source_snapshot_sha256", "cell_region",
    "cell_region_sha256", "grid_version", "grid_spec_sha256",
    "scientific_opportunity_id", "accepted_p0_artifact_id",
    "accepted_p0_artifact_sha256", "parent_task_id", "parent_attempt_id",
    "allow_unvalidated",
}

_GENERIC_BINDING_FIELDS = _RECONCILIATION_BINDING_FIELDS - {"selection_rank"}


def _closed(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} is not a closed object")
    return copy.deepcopy(value)


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def validate_first_letters_baseline_reconciliation(
    value: Any,
) -> dict[str, Any]:
    reconciliation = _closed(
        value, _RECONCILIATION_FIELDS, "baseline reconciliation",
    )
    if (
        reconciliation["schema"] != BASELINE_RECONCILIATION_SCHEMA
        or reconciliation["mode"] != "shadow"
        or reconciliation["namespace"] != "NONCANONICAL_DISCOVERY"
        or reconciliation["canonical_admission"] != "PROHIBITED"
        or reconciliation["top_k"] != 2
        or reconciliation["probe_generations"] != 12
        or reconciliation["maximum_attempts_per_candidate"] != 1
        or reconciliation["units_per_item"] != 24
        or reconciliation["allow_unvalidated"] is not False
    ):
        raise ValueError("baseline reconciliation safety contract is invalid")
    for field in (
        "mission_id", "request_id", "sample_id", "source_snapshot_id",
        "accepted_p0_artifact_id", "grid_version", "cap_authority_id",
    ):
        _nonempty(reconciliation[field], field)
    for field in (
        "budget_admission_sha256", "source_snapshot_sha256",
        "source_content_lock_sha256", "accepted_p0_artifact_sha256",
        "ordered_item_ids_sha256", "ordered_item_bindings_sha256",
        "cap_authority_sha256", "profile_file_sha256",
        "profile_scientific_core_sha256", "policy_sha256",
        "history_manifest_sha256", "reconciliation_sha256",
    ):
        _sha256(reconciliation[field], field)
    if re.fullmatch(r"[0-9a-f]{40}", reconciliation["deployed_revision"]) is None:
        raise ValueError("baseline deployed revision is invalid")
    items = reconciliation["ordered_item_ids"]
    bindings = reconciliation["ordered_item_bindings"]
    if (
        not isinstance(items, list)
        or not items
        or any(not isinstance(item, str) or not item for item in items)
        or items != list(dict.fromkeys(items))
        or reconciliation["ordered_item_ids_sha256"] != content_sha256(items)
        or not isinstance(bindings, list)
        or len(bindings) != len(items)
        or reconciliation["ordered_item_bindings_sha256"]
            != content_sha256(bindings)
    ):
        raise ValueError("baseline reconciliation cohort is invalid")
    for rank, (item, raw) in enumerate(zip(items, bindings, strict=True)):
        binding = _closed(
            raw, _RECONCILIATION_BINDING_FIELDS,
            "baseline reconciliation item binding",
        )
        if (
            binding["schema"]
                != "campaignx.first_letters_discovery_work_item_binding.v1"
            or binding["item_id"] != item
            or binding["selection_rank"] != rank
            or binding["sample_id"] != reconciliation["sample_id"]
            or binding["source_snapshot_id"]
                != reconciliation["source_snapshot_id"]
            or binding["source_snapshot_sha256"]
                != reconciliation["source_snapshot_sha256"]
            or binding["grid_version"] != reconciliation["grid_version"]
            or binding["accepted_p0_artifact_id"]
                != reconciliation["accepted_p0_artifact_id"]
            or binding["accepted_p0_artifact_sha256"]
                != reconciliation["accepted_p0_artifact_sha256"]
            or binding["allow_unvalidated"] is not False
            or binding["cell_region_sha256"]
                != content_sha256(binding["cell_region"])
        ):
            raise ValueError("baseline reconciliation item binding drift")
        for field in (
            "source_snapshot_sha256", "cell_region_sha256",
            "grid_spec_sha256", "accepted_p0_artifact_sha256",
        ):
            _sha256(binding[field], field)
    core = {
        key: row for key, row in reconciliation.items()
        if key != "reconciliation_sha256"
    }
    if reconciliation["reconciliation_sha256"] != content_sha256(core):
        raise ValueError("baseline reconciliation hash is invalid")
    return reconciliation


def _generic_authority(
    reconciliation: dict[str, Any], *, work_kind: str,
    source_snapshot_id: str, source_snapshot_sha256: str,
    profile_sha256: str, policy_sha256: str,
    parentless: bool,
) -> dict[str, Any]:
    bindings = []
    for raw in reconciliation["ordered_item_bindings"]:
        binding = {
            key: copy.deepcopy(value) for key, value in raw.items()
            if key in _GENERIC_BINDING_FIELDS
        }
        binding["source_snapshot_id"] = source_snapshot_id
        binding["source_snapshot_sha256"] = source_snapshot_sha256
        if parentless:
            binding["parent_task_id"] = None
            binding["parent_attempt_id"] = None
        bindings.append(binding)
    schema = {
        "BASELINE_ARM":
            "campaignx.first_letters_discovery_baseline_work_admission.v1",
        "ALTERNATIVE_SOURCE_ARM":
            "campaignx.first_letters_experimental_arm_admission.v1",
    }[work_kind]
    identity = {
        "producer_reconciliation_sha256":
            reconciliation["reconciliation_sha256"],
        "work_kind": work_kind,
        "source_snapshot_sha256": source_snapshot_sha256,
        "profile_sha256": profile_sha256,
    }
    core = {
        "schema": schema,
        "work_authority_id": stable_id(
            "first-letters-discovery-work-authority", identity,
        ),
        "mission_id": reconciliation["mission_id"],
        "work_kind": work_kind,
        "ordered_item_ids": copy.deepcopy(reconciliation["ordered_item_ids"]),
        "ordered_item_ids_sha256":
            reconciliation["ordered_item_ids_sha256"],
        "ordered_item_bindings": bindings,
        "ordered_item_bindings_sha256": content_sha256(bindings),
        "cap_authority_id": reconciliation["cap_authority_id"],
        "cap_authority_sha256": reconciliation["cap_authority_sha256"],
        "profile_sha256": profile_sha256,
        "policy_sha256": policy_sha256,
        "source_sha256": source_snapshot_sha256,
        "deployed_revision": reconciliation["deployed_revision"],
        "requested_item_count": len(reconciliation["ordered_item_ids"]),
        "requested_units": len(reconciliation["ordered_item_ids"]) * 24,
        "allow_unvalidated": False,
    }
    return {**core, "work_authority_sha256": content_sha256(core)}


def _adapter(
    *, reconciliation: dict[str, Any], producer_kind: str,
    native_schema: str, native_authority: dict[str, Any],
    native_authority_sha256: str, generic: dict[str, Any],
    source_snapshot_authority: dict[str, Any],
) -> dict[str, Any]:
    source = copy.deepcopy(source_snapshot_authority)
    if (
        not isinstance(source, dict)
        or source.get("source_snapshot_id") is None
        or source.get("source_snapshot_id")
            != generic["ordered_item_bindings"][0]["source_snapshot_id"]
        or source.get("source_snapshot_sha256") != generic["source_sha256"]
    ):
        raise ValueError("discovery adapter source authority is invalid")
    core = {
        "schema": NATIVE_ADAPTER_SCHEMA,
        "producer_kind": producer_kind,
        "mission_id": reconciliation["mission_id"],
        "request_id": reconciliation["request_id"],
        "work_kind": generic["work_kind"],
        "reservation_mode": "EXACT",
        "source": "RESERVED_BEFORE_EXECUTION",
        "native_schema": native_schema,
        "native_authority": copy.deepcopy(native_authority),
        "native_authority_sha256": native_authority_sha256,
        "source_snapshot_authority": source,
        "generic_work_authority": copy.deepcopy(generic),
        "generic_work_authority_sha256":
            generic["work_authority_sha256"],
        "profile_file_sha256": generic["profile_sha256"],
        "history_manifest_sha256":
            reconciliation["history_manifest_sha256"],
        "allow_unvalidated": False,
    }
    return {**core, "adapter_sha256": content_sha256(core)}


def adapt_first_letters_baseline_shadow(
    reconciliation: dict[str, Any],
    source_snapshot_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reconciliation = validate_first_letters_baseline_reconciliation(
        reconciliation
    )
    generic = _generic_authority(
        reconciliation,
        work_kind="BASELINE_ARM",
        source_snapshot_id=reconciliation["source_snapshot_id"],
        source_snapshot_sha256=reconciliation["source_snapshot_sha256"],
        profile_sha256=reconciliation["profile_file_sha256"],
        policy_sha256=reconciliation["policy_sha256"],
        parentless=False,
    )
    if source_snapshot_authority is None:
        source_snapshot_authority = {
            "source_snapshot_id": reconciliation["source_snapshot_id"],
            "source_snapshot_sha256": reconciliation[
                "source_snapshot_sha256"
            ],
            "source_content_lock_sha256": reconciliation[
                "source_content_lock_sha256"
            ],
        }
    return _adapter(
        reconciliation=reconciliation,
        producer_kind="BASELINE_RECONCILIATION",
        native_schema=BASELINE_RECONCILIATION_SCHEMA,
        native_authority=reconciliation,
        native_authority_sha256=reconciliation["reconciliation_sha256"],
        generic=generic,
        source_snapshot_authority=source_snapshot_authority,
    )


def adapt_first_letters_alternative_shadow(
    reconciliation: dict[str, Any], arm_admission: dict[str, Any],
    alternative_source_snapshot: dict[str, Any],
) -> dict[str, Any]:
    reconciliation = validate_first_letters_baseline_reconciliation(
        reconciliation
    )
    arm = validate_experimental_arm_admission(arm_admission)
    source = copy.deepcopy(alternative_source_snapshot)
    required_source = {
        "source_snapshot_id", "source_snapshot_sha256",
        "source_content_lock_sha256", "ct_metadata_sha256",
        "ct_read_set_manifest_sha256", "m7_metadata_sha256",
        "m7_read_set_manifest_sha256", "m7_model_id", "m7_resolution",
        "m7_level", "m7_transform_sha256", "m7_threshold",
    }
    if not isinstance(source, dict) or not required_source <= set(source):
        raise ValueError("alternative source snapshot is incomplete")
    comparisons = {
        "mission_id": reconciliation["mission_id"],
        "accepted_p0_id": reconciliation["accepted_p0_artifact_id"],
        "accepted_p0_sha256": reconciliation["accepted_p0_artifact_sha256"],
        "ordered_cell_ids": reconciliation["ordered_item_ids"],
        "ordered_cell_set_sha256": reconciliation["ordered_item_ids_sha256"],
        "mission_compute_cap_authority_id":
            reconciliation["cap_authority_id"],
        "mission_compute_cap_authority_sha256":
            reconciliation["cap_authority_sha256"],
        "requested_units": len(reconciliation["ordered_item_ids"]) * 24,
        "active_policy_chain_sha256": reconciliation["policy_sha256"],
        "deployed_revision": reconciliation["deployed_revision"],
    }
    if any(arm.get(field) != expected for field, expected in comparisons.items()):
        raise ValueError("experimental arm differs from baseline reconciliation")
    for field in required_source:
        arm_field = field
        if arm.get(arm_field) != source.get(field):
            raise ValueError("experimental arm source snapshot drift")
    if (
        arm["discovery_profile_sha256"]
            != reconciliation["profile_file_sha256"]
        or arm["may_update_accepted_p0"] is not False
        or arm["statistical_budget_delta"] != 0
        or arm["allow_unvalidated"] is not False
    ):
        raise ValueError("experimental arm is not safe for shadow execution")
    generic = _generic_authority(
        reconciliation,
        work_kind="ALTERNATIVE_SOURCE_ARM",
        source_snapshot_id=arm["source_snapshot_id"],
        source_snapshot_sha256=arm["source_snapshot_sha256"],
        profile_sha256=arm["discovery_profile_sha256"],
        policy_sha256=arm["active_policy_chain_sha256"],
        parentless=True,
    )
    native = {
        "schema":
            "campaignx.first_letters_discovery_alternative_work_authority.v1",
        "baseline_reconciliation": reconciliation,
        "arm_admission": arm,
        "alternative_source_snapshot": source,
        "allow_unvalidated": False,
    }
    native_sha = content_sha256(native)
    return _adapter(
        reconciliation=reconciliation,
        producer_kind="EXPERIMENTAL_ARM_ADMISSION",
        native_schema=native["schema"],
        native_authority=native,
        native_authority_sha256=native_sha,
        generic=generic,
        source_snapshot_authority=source,
    )


def validate_first_letters_discovery_native_adapter(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("discovery native adapter must be an object")
    adapter = copy.deepcopy(value)
    digest = adapter.pop("adapter_sha256", None)
    if (
        adapter.get("schema") != NATIVE_ADAPTER_SCHEMA
        or adapter.get("producer_kind") not in {
            "BASELINE_RECONCILIATION", "EXPERIMENTAL_ARM_ADMISSION",
        }
        or adapter.get("work_kind") not in {
            "BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM",
        }
        or adapter.get("reservation_mode") != "EXACT"
        or adapter.get("source") != "RESERVED_BEFORE_EXECUTION"
        or adapter.get("allow_unvalidated") is not False
        or digest != content_sha256(adapter)
    ):
        raise ValueError("discovery native adapter hash/contract is invalid")
    generic = adapter.get("generic_work_authority") or {}
    generic_digest = generic.get("work_authority_sha256")
    if (
        generic_digest != adapter.get("generic_work_authority_sha256")
        or generic_digest != content_sha256({
            key: row for key, row in generic.items()
            if key != "work_authority_sha256"
        })
        or generic.get("work_kind") != adapter.get("work_kind")
        or generic.get("mission_id") != adapter.get("mission_id")
        or generic.get("profile_sha256")
            != adapter.get("profile_file_sha256")
    ):
        raise ValueError("discovery adapter generic projection is invalid")
    native = adapter.get("native_authority")
    if adapter.get("producer_kind") == "BASELINE_RECONCILIATION":
        expected = adapt_first_letters_baseline_shadow(
            native, adapter.get("source_snapshot_authority")
        )
    else:
        if not isinstance(native, dict):
            raise ValueError("alternative native authority is invalid")
        expected = adapt_first_letters_alternative_shadow(
            native.get("baseline_reconciliation"),
            native.get("arm_admission"),
            native.get("alternative_source_snapshot"),
        )
    actual = {**adapter, "adapter_sha256": digest}
    if actual != expected:
        raise ValueError("discovery native adapter differs from its producer")
    return actual


def build_first_letters_discovery_dispatch(
    reservation: dict[str, Any], adapter: dict[str, Any],
) -> dict[str, Any]:
    adapter = validate_first_letters_discovery_native_adapter(adapter)
    generic = adapter["generic_work_authority"]
    if (
        reservation.get("mission_id") != adapter["mission_id"]
        or reservation.get("request_id") != adapter["request_id"]
        or reservation.get("work_kind") != adapter["work_kind"]
        or reservation.get("work_authority_sha256")
            != adapter["generic_work_authority_sha256"]
        or reservation.get("ordered_item_ids")
            != generic["ordered_item_ids"]
    ):
        raise ValueError("reservation differs from native adapter")
    dispatch_id = stable_id("first-letters-discovery-dispatch", {
        "reservation_id": reservation["reservation_id"],
        "reservation_sha256": reservation["reservation_sha256"],
        "adapter_sha256": adapter["adapter_sha256"],
    })
    core = {
        "schema": DISPATCH_SCHEMA,
        "dispatch_id": dispatch_id,
        "reservation_id": reservation["reservation_id"],
        "reservation_sha256": reservation["reservation_sha256"],
        "mission_id": reservation["mission_id"],
        "request_id": reservation["request_id"],
        "work_kind": reservation["work_kind"],
        "adapter_sha256": adapter["adapter_sha256"],
        "profile_file_sha256": adapter["profile_file_sha256"],
        "source_snapshot_sha256": generic["source_sha256"],
        "ordered_item_ids": copy.deepcopy(reservation["ordered_item_ids"]),
        "ordered_item_ids_sha256": reservation["ordered_item_ids_sha256"],
        "item_count": reservation["item_count"],
        "mode": "shadow",
        "namespace": "NONCANONICAL_DISCOVERY",
        "canonical_admission": "PROHIBITED",
        "allow_unvalidated": False,
    }
    return {**core, "dispatch_sha256": content_sha256(core)}


def build_first_letters_discovery_jobs(
    dispatch: dict[str, Any], adapter: dict[str, Any],
) -> list[dict[str, Any]]:
    adapter = validate_first_letters_discovery_native_adapter(adapter)
    generic = adapter["generic_work_authority"]
    if (
        dispatch.get("schema") != DISPATCH_SCHEMA
        or dispatch.get("dispatch_sha256") != content_sha256({
            key: row for key, row in dispatch.items()
            if key != "dispatch_sha256"
        })
        or dispatch.get("adapter_sha256") != adapter["adapter_sha256"]
        or dispatch.get("ordered_item_ids") != generic["ordered_item_ids"]
    ):
        raise ValueError("discovery dispatch differs from adapter")
    jobs = []
    for order, binding in enumerate(generic["ordered_item_bindings"]):
        binding_sha = content_sha256(binding)
        job_id = stable_id("first-letters-discovery-job", {
            "dispatch_id": dispatch["dispatch_id"],
            "item_order": order,
            "item_id": binding["item_id"],
            "work_item_binding_sha256": binding_sha,
        })
        core = {
            "schema": JOB_SCHEMA,
            "job_id": job_id,
            "dispatch_id": dispatch["dispatch_id"],
            "dispatch_sha256": dispatch["dispatch_sha256"],
            "reservation_id": dispatch["reservation_id"],
            "mission_id": dispatch["mission_id"],
            "request_id": dispatch["request_id"],
            "work_kind": dispatch["work_kind"],
            "item_order": order,
            "item_id": binding["item_id"],
            "work_item_binding": copy.deepcopy(binding),
            "work_item_binding_sha256": binding_sha,
            "profile_file_sha256": dispatch["profile_file_sha256"],
            "source_snapshot_sha256": dispatch["source_snapshot_sha256"],
            "namespace": "NONCANONICAL_DISCOVERY",
            "canonical_admission": "PROHIBITED",
            "allow_unvalidated": False,
        }
        jobs.append({**core, "job_sha256": content_sha256(core)})
    return jobs


def build_first_letters_discovery_job_claim(
    *, job: dict[str, Any], dispatch: dict[str, Any],
    adapter: dict[str, Any], reservation: dict[str, Any], run_handle: Any,
) -> FirstLettersDiscoveryJobClaim:
    """Seal the store-resolved job graph together with its opaque v18 claim."""

    if (
        not isinstance(job, dict)
        or job.get("schema") != JOB_SCHEMA
        or job.get("job_sha256") != content_sha256({
            key: value for key, value in job.items() if key != "job_sha256"
        })
        or job.get("dispatch_id") != dispatch.get("dispatch_id")
        or job.get("dispatch_sha256") != dispatch.get("dispatch_sha256")
        or dispatch.get("adapter_sha256") != adapter.get("adapter_sha256")
        or job.get("reservation_id") != reservation.get("reservation_id")
        or dispatch.get("reservation_sha256")
            != reservation.get("reservation_sha256")
        or job.get("item_id") != getattr(run_handle, "cell_id", None)
        or not isinstance(getattr(run_handle, "provider_request", None), dict)
    ):
        raise ValueError("discovery job claim graph is invalid")
    core = {
        "schema": JOB_CLAIM_SCHEMA,
        "job_id": job["job_id"],
        "job_sha256": job["job_sha256"],
        "dispatch_id": dispatch["dispatch_id"],
        "dispatch_sha256": dispatch["dispatch_sha256"],
        "adapter_sha256": adapter["adapter_sha256"],
        "reservation_id": reservation["reservation_id"],
        "reservation_sha256": reservation["reservation_sha256"],
        "item_id": job["item_id"],
        "profile_file_sha256": job["profile_file_sha256"],
        "source_snapshot_sha256": job["source_snapshot_sha256"],
        "run_id": run_handle.run_id,
        "worker_id": run_handle.worker_id,
        "provider_request_sha256": content_sha256(
            run_handle.provider_request
        ),
    }
    return FirstLettersDiscoveryJobClaim(
        **{key: value for key, value in core.items() if key != "schema"},
        claim_sha256=content_sha256(core), _run_handle=run_handle,
    )


def validate_first_letters_discovery_job_claim(
    value: Any,
) -> FirstLettersDiscoveryJobClaim:
    """Purely validate a claim sealed by the store's job-rooted transaction."""

    if type(value) is not FirstLettersDiscoveryJobClaim:
        raise ValueError("discovery job claim must be a sealed packet")
    core = {
        "schema": JOB_CLAIM_SCHEMA,
        **{
            name: getattr(value, name) for name in (
                "job_id", "job_sha256", "dispatch_id", "dispatch_sha256",
                "adapter_sha256", "reservation_id", "reservation_sha256",
                "item_id", "profile_file_sha256", "source_snapshot_sha256",
                "run_id", "worker_id", "provider_request_sha256",
            )
        },
    }
    for field_name in (
        "job_sha256", "dispatch_sha256", "adapter_sha256",
        "reservation_sha256", "profile_file_sha256",
        "source_snapshot_sha256", "provider_request_sha256",
    ):
        _sha256(core[field_name], field_name)
    handle = value._run_handle
    if (
        value.claim_sha256 != content_sha256(core)
        or getattr(handle, "run_id", None) != value.run_id
        or getattr(handle, "worker_id", None) != value.worker_id
        or getattr(handle, "cell_id", None) != value.item_id
        or content_sha256(getattr(handle, "provider_request", None))
            != value.provider_request_sha256
    ):
        raise ValueError("discovery job claim seal is invalid")
    return value
