"""Deterministic closed-loop seed trials for Stage 01.

The probe is deliberately not a second planner and not a surface.  It executes
the same small, frozen VC3D experiment for the first K candidates, measures the
result with the existing TIFXYZ geometry gate, and either identifies one
unambiguous winner or abstains.

Probe bytes and rows live in their own namespace.  They must never be inserted
into ``surfaces``, ``artifact_sets`` or downstream QC.  Only the later full grow
is canonical and it still passes normal finalization, deduplication, geometry
and physical QC.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlparse, urlsplit

from .artifact_store import open_artifact_store
from .common import (
    artifact_manifest,
    content_sha256,
    file_sha256,
    read_json,
    stable_id,
    utc_now,
    write_json_atomic,
)
from .executor import InsufficientGpuMemoryError
from .finalizer import certify_surface_geometry, inspect_tifxyz
from .planner import candidate_rank_key


PROBE_REQUIRED_ARTIFACTS = (
    "x.tif",
    "y.tif",
    "z.tif",
    "generations.tif",
    "meta.json",
)
PROBE_PROFILE_ID = "vc3d-m7-probe-v1"
PROBE_PROFILE_SHA256 = (
    "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33"
)
PROBE_EVALUATION_PROFILE_ID = "tifxyz-geometry-gate-probe-v1"
# Moved on 2026-08-28 with the gate's triangulation winding fix; the
# profile records why, and the verdicts it produces are unchanged.
PROBE_EVALUATION_PROFILE_SHA256 = (
    "7bdb99c30241f0c0b750a0405ebdfc0ab919b429e737a15b568706f4cb7ed6a5"
)
PROBE_TERMINAL_TRIAL_STATES = frozenset(
    {"SUCCEEDED", "REJECTED", "UNMEASURED", "FAILED"}
)
PROBE_DECISION_ACTIONS = frozenset(
    {"CONTINUE_WINNER", "HUMAN_REVIEW", "REJECT_ALL"}
)
SOURCE_CONTENT_LOCK_SCHEMA = "campaignx.source_content_lock.v1"
BENCHMARK_SPEC_SCHEMA = "campaignx.seed_probe_benchmark_spec.v1"
BENCHMARK_DECISION_SCHEMA = "campaignx.seed_probe_benchmark_decision.v1"

# How far down m7's candidate ordering a probe may look. Mirrored by the fleet
# CLI's argparse choices, the panel's request model and the launcher contract
# it serves; a test compares them, because this bound has been wrong in four
# places at once.
TOP_K_MAXIMUM = 20
BENCHMARK_AUTHORIZATION_SCHEMA = (
    "campaignx.seed_probe_benchmark_authorization.v1"
)
BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA = (
    "campaignx.seed_probe_benchmark_execution_authorization.v1"
)
BENCHMARK_EXECUTION_SCHEMA = "campaignx.seed_probe_benchmark_execution.v1"
BENCHMARK_RNG_PROTOCOL = (
    "sha256-benchmark-spec-sample-cell-prefix16-v1"
)
# Canonical SHA-256 of generator.SEED_PROBE_CONTINUATION_ENVELOPE.  Keeping the
# digest here avoids an import cycle (generator imports this module) while
# making any future envelope drift fail closed in the benchmark builder.
BENCHMARK_FULL_GROW_ENVELOPE_SHA256 = (
    "aa5cf6b030e3b8f2d5f90009c332f4e22e120fd1d8f87cb6f6f11aa66aba8990"
)
BENCHMARK_REQUIRED_CHECK_IDS = frozenset(
    {
        "MATCHED_EXECUTION_IDENTITY",
        "INVARIANTS",
        "YIELD_IMPROVEMENT",
        "PAIRED_SUPERIORITY",
        "REVIEWER_RATE",
        "COMPUTE_BUDGET",
        "NEW_INCORRECT_LAMINA_SAFETY",
        "SCROLL_COVERAGE",
        "SCROLL_NONREGRESSION",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RNG_RE = re.compile(r"^[0-9a-f]{16}$")


class ProbeWinnerMaterializationError(RuntimeError):
    """A retained winner could not be verified into the continuation sandbox."""

    def __init__(
        self,
        *,
        probe_run_id: str,
        decision: dict[str, Any],
        winner_trial_id: str,
        artifact_uri: str,
        error: Exception,
    ) -> None:
        super().__init__(
            "retained probe winner could not be materialized: "
            f"{type(error).__name__}: {str(error)[:1000]}"
        )
        self.probe_run_id = probe_run_id
        self.decision = copy.deepcopy(decision)
        self.winner_trial_id = winner_trial_id
        self.artifact_uri = artifact_uri


def default_seed_probe_policy(
    *,
    mode: str,
    top_k: int = 2,
    generations: int = 12,
    step_size: int = 12,
    use_cuda: bool = False,
    maximum_attempts_per_candidate: int = 2,
    benchmark_authorization: dict[str, Any] | None = None,
    review_owner: str | None = None,
) -> dict[str, Any]:
    """Return the complete immutable v1 policy stored on every probe task."""

    value = {
        "schema": "campaignx.seed_probe_policy.v1",
        "policy_id": "seed-probe-v1",
        "mode": str(mode).lower(),
        "top_k": int(top_k),
        "probe_profile_id": PROBE_PROFILE_ID,
        "probe_profile_sha256": PROBE_PROFILE_SHA256,
        "probe_parameters": {
            "generations": int(generations),
            "step_size": int(step_size),
            "min_area_cm": 0.0,
            "use_cuda": bool(use_cuda),
        },
        "maximum_total_probe_generations": (
            int(top_k)
            * int(generations)
            * int(maximum_attempts_per_candidate)
        ),
        "maximum_attempts_per_candidate": int(maximum_attempts_per_candidate),
        "evaluation_profile_id": PROBE_EVALUATION_PROFILE_ID,
        "evaluation_profile_sha256": PROBE_EVALUATION_PROFILE_SHA256,
        "decision_policy_id": "unique-geometry-certified-v1",
        "inconclusive_action": "HUMAN_REVIEW",
        "ink_used": False,
        **(
            {"benchmark_authorization": benchmark_authorization}
            if benchmark_authorization is not None
            else {}
        ),
        **({"review_owner": review_owner} if review_owner is not None else {}),
    }
    return normalize_seed_probe_policy(value)


def _require_lowercase_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


# Task 6 discovery contracts.  These are deliberately closed and kept beside
# the historical seed-probe contracts so v1 benchmark readback remains intact.
DISCOVERY_PROFILE_SCHEMA = "campaignx.first_letters_discovery_profile.v1"
DISCOVERY_INPUTS_SCHEMA = "campaignx.first_letters_discovery_inputs.v1"
DISCOVERY_ARTIFACT_SCHEMA = "campaignx.first_letters_discovery_artifact.v1"
DISCOVERY_RECEIPT_SCHEMA = "campaignx.first_letters_discovery_receipt.v1"
COORDINATE_SCHEMA = "campaignx.ct_l0_xyz_coordinate.v1"
COORDINATE_RULE_ID = "REJECT_NONINTEGRAL_FOR_PROMOTION_V1"
DISCOVERY_NAMESPACE = "NONCANONICAL_DISCOVERY"
DISCOVERY_ARM_KINDS = frozenset({"BASELINE", "ALTERNATIVE_SOURCE_ARM"})
NORMAL_FULL_GROW_PROFILE_ID = "first-letters-normal-full-grow@1.0.0"
NORMAL_GROWTH_PROFILE_ID = "vc3d-m7-growth-v1"
NORMAL_GROWTH_PROFILE_SHA256 = (
    "0a7747549011adae87c417d782747ee65196134a71d0309fc6adfd2c9c037217"
)
DISCOVERY_INPUT_ROLES = frozenset({
    "SOURCE_SNAPSHOT_METADATA",
    "CT_VOLUME",
    "M7_PREDICTION_VOLUME",
    "CANDIDATE_PROVIDER_REQUEST",
    "CANDIDATE_PROVIDER_RESPONSE",
    "CANONICAL_GRID_SPEC",
    "CT_MATERIAL_SUPPORT_POLICY",
    "CELL_AND_VOLUME_CLEARANCE_POLICY",
    "NONCANONICAL_PROBE_GEOMETRY",
})
_CONTENT_INFORMED_TOKENS = frozenset({
    "p5", "p7", "ink", "ocr", "lexical", "letter", "glyph", "text",
    "phrase", "transcription", "language", "human", "review", "reading",
    "annotation", "adjudication",
})
_SENSITIVE_CREDENTIAL_KEYS = frozenset({
    "authorization", "proxy_authorization", "cookie", "set_cookie",
    "password", "passwd", "secret", "client_secret", "token",
    "access_token", "refresh_token", "api_key", "apikey", "access_key",
    "secret_access_key", "session_key", "session_token", "credential",
})
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?:\b(?:bearer|basic)\s+[A-Za-z0-9+/._=-]+|\bAKIA[0-9A-Z]{16}\b|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FirstLettersDiscoveryCandidateMeasurement:
    """Exact read evidence; terminal booleans are derived, never asserted."""

    candidate_id: str
    ct_read_evidence_bytes: bytes | None
    probe_evidence_bytes: bytes | None


_DISCOVERY_EXECUTOR_CLAIM_FIELDS = {
    "schema", "claim_id", "run_id", "worker_id", "executor_id",
    "executor_sha256", "capability", "provider_request_sha256",
    "executor_registration_sha256", "claim_attempt_number",
    "execution_lease_token_sha256", "lease_expires_at",
    "allow_unvalidated", "claim_sha256",
}


def _derive_first_letters_discovery_run_authority(
    *, profile_bytes: bytes, reservation: dict[str, Any], work: dict[str, Any],
    binding: dict[str, Any], source: dict[str, Any],
) -> dict[str, Any]:
    """Derive backend-neutral run bytes from independently resolved rows."""

    import hashlib

    profile = load_first_letters_discovery_profile_bytes(profile_bytes)
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    authority = work.get("work_authority") or {}
    item_id = binding["item_id"]
    persisted_scope = source.get("first_letters_discovery_authority") or {}
    persisted_opportunities = persisted_scope.get("scientific_opportunities")
    cell_clearance = persisted_scope.get("minimum_cell_clearance_voxels")
    volume_clearance = persisted_scope.get("minimum_volume_clearance_voxels")
    source_snapshot_sha = source.get("source_snapshot_sha256")
    if (item_id not in reservation.get("ordered_item_ids", [])
            or item_id not in work.get("ordered_item_ids", [])
            or reservation.get("reservation_sha256") !=
                work.get("reservation_sha256")
            or reservation.get("work_authority_sha256") !=
                authority.get("work_authority_sha256")
            or authority.get("profile_sha256") != profile_sha
            or source_snapshot_sha != binding.get("source_snapshot_sha256")
            or source_snapshot_sha != authority.get("source_sha256")
            or source.get("sample_id") != binding.get("sample_id")
            or persisted_scope.get("mission_id") != reservation.get("mission_id")
            or not isinstance(persisted_opportunities, dict)
            or persisted_opportunities.get(item_id) !=
                binding.get("scientific_opportunity_id")
            or persisted_scope.get("accepted_p0_artifact_id") !=
                binding.get("accepted_p0_artifact_id")
            or persisted_scope.get("accepted_p0_artifact_sha256") !=
                binding.get("accepted_p0_artifact_sha256")
            or isinstance(cell_clearance, bool)
            or not isinstance(cell_clearance, int) or cell_clearance < 0
            or isinstance(volume_clearance, bool)
            or not isinstance(volume_clearance, int) or volume_clearance < 0):
        raise ValueError("discovery run resolved authority differs")
    if (profile["source_snapshot_id"] != source["source_snapshot_id"]
            or profile["source_snapshot_sha256"] != source_snapshot_sha
            or profile["source_content_lock_sha256"] !=
                source.get("source_content_lock_sha256")
            or profile["ct_metadata_sha256"] != source.get("ct_metadata_sha256")
            or profile["ct_read_set_manifest_sha256"] !=
                source.get("ct_read_set_manifest_sha256")
            or profile["m7_read_set_manifest_sha256"] !=
                source.get("m7_read_set_manifest_sha256")
            or profile["m7_model_id"] != source.get("m7_model_id")
            or profile["m7_resolution"] != source.get("m7_resolution")
            or profile["m7_level"] != source.get("m7_level")
            or profile["m7_threshold"] != source.get("m7_threshold")
            or profile["m7_transform_sha256"] != source.get("m7_transform_sha256")
            or profile["canonical_ordered_cell_set_sha256"] !=
                _task6_sha256(reservation.get("ordered_item_ids"))
            or profile["mission_compute_cap_authority_id"] !=
                reservation.get("cap_authority_id")
            or profile["mission_compute_cap_authority_sha256"] !=
                reservation.get("cap_authority_sha256")):
        raise ValueError("registered discovery profile differs from work authority")
    region = copy.deepcopy(binding["cell_region"])
    region_sha = binding["cell_region_sha256"]
    grid_sha = binding["grid_spec_sha256"]
    dependencies = [
        {
            "role": "CT_VOLUME",
            "artifact_sha256": profile["ct_metadata_sha256"],
            "read_set_manifest_sha256": profile["ct_read_set_manifest_sha256"],
            "cell_id": item_id, "cell_region_sha256": region_sha,
            "grid_spec_sha256": grid_sha,
        },
        {
            "role": "M7_PREDICTION_VOLUME",
            "artifact_sha256": source["m7_sha256"],
            "read_set_manifest_sha256": profile["m7_read_set_manifest_sha256"],
            "cell_id": item_id, "cell_region_sha256": region_sha,
            "grid_spec_sha256": grid_sha,
        },
        {
            "role": "CANONICAL_GRID_SPEC", "artifact_sha256": grid_sha,
            "read_set_manifest_sha256": grid_sha, "cell_id": item_id,
            "cell_region_sha256": region_sha, "grid_spec_sha256": grid_sha,
        },
    ]
    for role, policy in (
        ("CT_MATERIAL_SUPPORT_POLICY", profile["ct_material_policy"]),
        ("CELL_AND_VOLUME_CLEARANCE_POLICY", profile["clearance_policy"]),
    ):
        policy_sha = _task6_sha256({"policy_id": policy})
        dependencies.append({
            "role": role, "artifact_sha256": policy_sha,
            "read_set_manifest_sha256": policy_sha, "cell_id": item_id,
            "cell_region_sha256": region_sha, "grid_spec_sha256": grid_sha,
        })
    provider_request = {
        "request_id": reservation["request_id"], "cell_id": item_id,
        "source_snapshot_id": source["source_snapshot_id"],
        "source_snapshot_sha256": source_snapshot_sha,
        "prediction_root_sha256": source["m7_sha256"],
        "resolution": profile["m7_resolution"], "level": profile["m7_level"],
        "threshold": profile["m7_threshold"], "ct_l0_region": region,
        "cell_region_sha256": region_sha, "grid_spec_sha256": grid_sha,
        "coordinate_frame": source["coordinate_frame"],
        "maximum_candidates": profile["top_k"],
        "minimum_separation": source.get("discovery_minimum_separation"),
        "model_id": profile["m7_model_id"],
        "model_sha256": source.get("m7_model_sha256"),
        "provider_id": source.get("candidate_provider_id"),
        "provider_sha256": source.get("candidate_provider_sha256"),
        "coordinate_admission_rule_id": profile["coordinate_admission_rule_id"],
        "coordinate_admission_rule_sha256":
            profile["coordinate_admission_rule_sha256"],
        "dependency_manifest_sha256": _task6_sha256(dependencies),
    }
    source_dependency = {
        "role": "SOURCE_SNAPSHOT_METADATA",
        "source_snapshot_id": source["source_snapshot_id"],
        "source_snapshot_sha256": source_snapshot_sha,
        "source_content_lock_sha256": source["source_content_lock_sha256"],
        "cell_id": item_id, "cell_region_sha256": region_sha,
        "grid_spec_sha256": grid_sha,
    }
    run_id = stable_id("first-letters-discovery-evidence-run", {
        "reservation_id": reservation["reservation_id"], "item_id": item_id,
        "profile_file_sha256": profile_sha,
    })
    run_core = {
        "schema": "campaignx.first_letters_discovery_run_authority.v1",
        "run_id": run_id, "mission_id": reservation["mission_id"],
        "sample_id": source["sample_id"],
        "parent_task_id": binding["parent_task_id"],
        "parent_attempt_id": binding["parent_attempt_id"],
        "claim_attempt_number": 1,
        "scientific_opportunity_id": binding["scientific_opportunity_id"],
        "accepted_p0_artifact_id": binding["accepted_p0_artifact_id"],
        "accepted_p0_artifact_sha256": binding["accepted_p0_artifact_sha256"],
        "cell_id": item_id, "source_snapshot_id": source["source_snapshot_id"],
        "source_snapshot_sha256": source_snapshot_sha,
        "source_content_lock_sha256": source["source_content_lock_sha256"],
        "source_shape_xyz": copy.deepcopy(source["shape_xyz"]),
        "minimum_cell_clearance_voxels": cell_clearance,
        "minimum_volume_clearance_voxels": volume_clearance,
        "source_snapshot_dependency": source_dependency,
        "dependencies": dependencies, "profile_file_sha256": profile_sha,
        "reservation_id": reservation["reservation_id"],
        "reservation_sha256": reservation["reservation_sha256"],
        "reservation_request_id": reservation["request_id"],
        "reservation_work_kind": reservation["work_kind"],
        "reservation_work_authority_id": reservation["work_authority_id"],
        "reservation_work_authority_sha256":
            reservation["work_authority_sha256"],
        "reservation_source": reservation["source"],
        "reservation_ordered_item_ids": reservation["ordered_item_ids"],
        "provider_request_sha256": _task6_sha256(provider_request),
        "allow_unvalidated": False,
    }
    return {
        "profile_file_sha256": profile_sha,
        "provider_request": provider_request,
        "run_authority": {
            **run_core, "run_authority_sha256": _task6_sha256(run_core),
        },
    }


def _bind_first_letters_discovery_executor_claim(
    *, derived: dict[str, Any], registration: dict[str, Any],
    lease_expires_at: str,
) -> dict[str, Any]:
    """Bind a store-issued executor lease to the derived run authority."""

    import hashlib
    import secrets

    provider_request = copy.deepcopy(derived["provider_request"])
    provisional = copy.deepcopy(derived["run_authority"])
    claim_token = secrets.token_urlsafe(32)
    claim_core = {
        "schema": "campaignx.first_letters_discovery_executor_claim.v2",
        "claim_id": stable_id("first-letters-discovery-executor-claim", {
            "run_id": provisional["run_id"],
            "registration_sha256": registration["registration_sha256"],
            "claim_attempt_number": 1,
        }),
        "run_id": provisional["run_id"],
        "worker_id": registration["worker_id"],
        "executor_id": registration["executor_id"],
        "executor_sha256": registration["executor_sha256"],
        "capability": "FIRST_LETTERS_DISCOVERY_CT_PROBE_V1",
        "provider_request_sha256": _task6_sha256(provider_request),
        "executor_registration_sha256":
            registration["registration_sha256"],
        "claim_attempt_number": 1,
        "execution_lease_token_sha256": hashlib.sha256(
            claim_token.encode("utf-8")
        ).hexdigest(),
        "lease_expires_at": lease_expires_at,
        "allow_unvalidated": False,
    }
    claim = {**claim_core, "claim_sha256": _task6_sha256(claim_core)}
    run_core = {
        key: value for key, value in provisional.items()
        if key != "run_authority_sha256"
    }
    run_core.update({
        "worker_id": claim["worker_id"],
        "executor_claim": claim,
    })
    return {
        **copy.deepcopy(derived),
        "_executor_claim_token": claim_token,
        "run_authority": {
            **run_core,
            "run_authority_sha256": _task6_sha256(run_core),
        },
    }


def _accept_first_letters_discovery_executor_claim(
    *, executor: Any, run_id: str, claim_token: str,
) -> None:
    accept_method = getattr(
        executor, "accept_first_letters_discovery_claim", None
    )
    if not callable(accept_method):
        raise ValueError("AUTHORITATIVE_DISCOVERY_EXECUTOR_REQUIRED")
    accept_method(run_id=run_id, claim_token=claim_token)


def _validated_first_letters_discovery_executor_claim(
    value: Any, *, run_authority: dict[str, Any],
    registration: dict[str, Any], provider_request: dict[str, Any],
) -> dict[str, Any]:
    claim = copy.deepcopy(_closed_dict(
        value, _DISCOVERY_EXECUTOR_CLAIM_FIELDS,
        "discovery executor claim",
    ))
    core = {key: row for key, row in claim.items() if key != "claim_sha256"}
    if (claim["schema"] !=
            "campaignx.first_letters_discovery_executor_claim.v2"
            or claim["run_id"] != run_authority.get("run_id")
            or claim["worker_id"] != registration["worker_id"]
            or claim["executor_id"] != registration["executor_id"]
            or claim["executor_sha256"] != registration["executor_sha256"]
            or claim["executor_registration_sha256"] !=
                registration["registration_sha256"]
            or claim["capability"] !=
                "FIRST_LETTERS_DISCOVERY_CT_PROBE_V1"
            or claim["provider_request_sha256"] !=
                _task6_sha256(provider_request)
            or claim["claim_attempt_number"] != 1
            or claim["allow_unvalidated"] is not False
            or claim["claim_sha256"] != _task6_sha256(core)):
        raise ValueError("discovery executor claim is invalid or cross-bound")
    for field in (
        "executor_sha256", "executor_registration_sha256",
        "execution_lease_token_sha256", "provider_request_sha256",
        "claim_sha256",
    ):
        _require_lowercase_sha256(claim[field], field)
    for field in (
        "claim_id", "worker_id", "executor_id", "lease_expires_at",
    ):
        if not isinstance(claim[field], str) or not claim[field]:
            raise ValueError("discovery executor claim identity is invalid")
    return claim


def _measure_first_letters_discovery_with_executor(
    *, executor: Any, run_authority: dict[str, Any],
    provider_request: dict[str, Any], provider_response_bytes: bytes,
    source_snapshot: dict[str, Any],
) -> tuple[FirstLettersDiscoveryCandidateMeasurement, ...]:
    """Obtain observations only from the server-configured claimed executor."""

    measure_method = getattr(
        executor, "measure_first_letters_discovery_run", None
    )
    if not callable(measure_method):
        raise ValueError("AUTHORITATIVE_DISCOVERY_EXECUTOR_REQUIRED")
    claim = run_authority.get("executor_claim")
    if (not isinstance(claim, dict)
            or claim.get("worker_id") != run_authority.get("worker_id")
            or claim.get("run_id") != run_authority.get("run_id")):
        raise ValueError("discovery run has no bound executor claim")
    ownership_method = getattr(
        executor, "first_letters_discovery_claim_token", None
    )
    claim_token = (
        ownership_method(run_id=run_authority["run_id"])
        if callable(ownership_method) else None
    )
    import hashlib
    if (not isinstance(claim_token, str)
            or hashlib.sha256(claim_token.encode("utf-8")).hexdigest() !=
                claim.get("execution_lease_token_sha256")):
        raise ValueError("EXECUTOR_CLAIM_OWNERSHIP_REQUIRED")
    measurements = measure_method(
        executor_claim=copy.deepcopy(claim),
        run_authority=copy.deepcopy(run_authority),
        provider_request=copy.deepcopy(provider_request),
        provider_response_bytes=provider_response_bytes,
        source_snapshot=copy.deepcopy(source_snapshot),
    )
    if (not isinstance(measurements, tuple)
            or any(type(row) is not FirstLettersDiscoveryCandidateMeasurement
                   for row in measurements)):
        raise ValueError("authoritative discovery executor returned invalid reads")
    return measurements


def _task6_canonical_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _task6_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_task6_canonical_bytes(value)).hexdigest()


def _strict_json_loads_bytes(raw: Any, label: str) -> Any:
    """Decode one unambiguous UTF-8 JSON document."""

    import json

    if not isinstance(raw, bytes):
        raise ValueError(f"{label} must be exact bytes")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{label} contains non-JSON number {value}")

    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not exact UTF-8 JSON bytes") from error


def _closed_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{label} differs from the closed contract: expected "
            f"{sorted(fields)}, got {got}"
        )
    return value


def _false_override(value: dict[str, Any], label: str) -> None:
    if value.get("allow_unvalidated") is not False:
        raise ValueError(f"{label} requires allow_unvalidated=false")


def _reject_content_informed(value: Any, path: str = "discovery") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            tokens = {
                token for token in re.split(r"[^a-z0-9]+", str(key).lower())
                if token
            }
            if tokens & _CONTENT_INFORMED_TOKENS:
                raise ValueError(f"content-informed discovery field: {path}.{key}")
            _reject_content_informed(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_content_informed(child, f"{path}[{index}]")


def _credential_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _reject_credential_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _credential_key(key)
            sensitive = (
                normalized in _SENSITIVE_CREDENTIAL_KEYS
                or any(normalized.endswith(f"_{name}")
                       for name in _SENSITIVE_CREDENTIAL_KEYS)
            )
            if (sensitive
                    and not normalized.endswith(("_sha256", "_hash"))):
                raise ValueError(f"credential-bearing key is prohibited at {path}.{key}")
            _reject_credential_value(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_credential_value(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if _CREDENTIAL_VALUE_RE.search(value):
            raise ValueError(f"credential-bearing value is prohibited at {path}")
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            if parsed.username is not None or parsed.password is not None:
                raise ValueError(f"credential-bearing URI is prohibited at {path}")
            for key, _child in parse_qsl(parsed.query, keep_blank_values=True):
                normalized = _credential_key(key)
                if (normalized in _SENSITIVE_CREDENTIAL_KEYS
                        or any(normalized.endswith(f"_{name}")
                               for name in _SENSITIVE_CREDENTIAL_KEYS)):
                    raise ValueError(
                        f"credential-bearing URI query is prohibited at {path}"
                    )


def coordinate_sha256_v1(coordinate: Any) -> str:
    """Hash the sole Task 6 serialized CT-L0 coordinate projection."""

    validated = validate_task6_coordinate(coordinate)
    return _task6_sha256({
        "schema": COORDINATE_SCHEMA,
        "coordinate_frame": "ct_l0_xyz",
        "value": validated,
    })


def validate_task6_coordinate(
    value: Any, *, require_integral: bool = False,
    expected_coordinate: list[Any] | None = None,
) -> list[Any]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Task 6 coordinate must be a length-three JSON array [x,y,z]")
    for item in value:
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(item)):
            raise ValueError("Task 6 coordinate items must be finite JSON numbers")
        if require_integral and not isinstance(item, int):
            raise ValueError("Task 6 promotion coordinates must be JSON integers")
    if expected_coordinate is not None and value != expected_coordinate:
        raise ValueError("Task 6 coordinate axis order/value differs from its authority")
    return copy.deepcopy(value)


def load_first_letters_normal_growth_lock(
    *, source_snapshot_id: str, coordinate: Any,
    coordinate_sha256: str, deployed_revision: str, retry_budget: int,
) -> dict[str, Any]:
    """Load and bind the immutable fresh ordinary-grow profile.

    Promotion is intentionally unable to inherit a probe checkpoint or a
    mutable generator default.  The returned bytes are the complete authority
    consumed by the promotion child and ordinary worker.
    """

    profile_path = (
        Path(__file__).resolve().parent
        / "profiles/first-letters-normal-full-grow-v1.json"
    )
    ordinary_path = (
        Path(__file__).resolve().parent / "profiles/vc3d-m7-growth-v1.json"
    )
    profile = read_json(profile_path)
    ordinary = read_json(ordinary_path)
    if (
        profile.get("schema") !=
            "campaignx.first_letters_normal_full_grow_profile.v1"
        or profile.get("profile_id") != NORMAL_FULL_GROW_PROFILE_ID
        or profile.get("allow_unvalidated") is not False
        or profile.get("fresh_start") is not True
        or (profile.get("ordinary_growth_profile") or {}).get("profile_id")
            != NORMAL_GROWTH_PROFILE_ID
        or (profile.get("ordinary_growth_profile") or {}).get("sha256")
            != NORMAL_GROWTH_PROFILE_SHA256
        or file_sha256(ordinary_path) != NORMAL_GROWTH_PROFILE_SHA256
        or ordinary.get("profile_id") != NORMAL_GROWTH_PROFILE_ID
        or profile.get("growth_envelope") != {
            "generations": {
                "type": "integer", "minimum": 20, "maximum": 45,
                "default": 35,
            },
            "step_size": {
                "type": "integer", "minimum": 12, "maximum": 24,
                "default": 20,
            },
            "min_area_cm": {"type": "number", "const": 0.0, "default": 0.0},
            "use_cuda": {"type": "boolean", "const": False, "default": False},
            "maximum_candidate_count": 8,
            "ink_used": False,
        }
    ):
        raise RuntimeError("normal full-grow profile bytes/contract have drifted")
    coordinate = validate_task6_coordinate(coordinate, require_integral=True)
    if coordinate_sha256 != coordinate_sha256_v1(coordinate):
        raise ValueError("normal full-grow coordinate hash drift")
    if (not isinstance(deployed_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", deployed_revision) is None):
        raise ValueError("deployed_revision must be a lowercase git SHA")
    if (not isinstance(retry_budget, int) or isinstance(retry_budget, bool)
            or not 0 <= retry_budget <= 16):
        raise ValueError("normal full-grow retry budget is invalid")
    finalization = profile.get("finalization_envelope")
    if (not isinstance(finalization, dict)
            or finalization.get("allow_unvalidated") is not False
            or finalization.get("discovery_artifact_role_allowed") is not False):
        raise RuntimeError("normal finalization profile is not fail closed")
    core = {
        "schema": "campaignx.first_letters_normal_growth_lock.v1",
        "normal_full_grow_profile_id": NORMAL_FULL_GROW_PROFILE_ID,
        "normal_full_grow_profile_sha256": file_sha256(profile_path),
        "ordinary_growth_profile_id": NORMAL_GROWTH_PROFILE_ID,
        "ordinary_growth_profile_sha256": NORMAL_GROWTH_PROFILE_SHA256,
        "growth_envelope": copy.deepcopy(profile["growth_envelope"]),
        "growth_parameter_envelope_sha256": _task6_sha256(
            profile["growth_envelope"]
        ),
        "finalization_envelope": copy.deepcopy(finalization),
        "finalization_envelope_sha256": _task6_sha256(finalization),
        "source_snapshot_id": source_snapshot_id,
        "coordinate_frame": "ct_l0_xyz",
        "raw_coordinate_ct_l0_xyz": coordinate,
        "raw_coordinate_sha256": coordinate_sha256,
        "promotion_coordinate_ct_l0_xyz": copy.deepcopy(coordinate),
        "promotion_coordinate_sha256": coordinate_sha256,
        "coordinate_schema": COORDINATE_SCHEMA,
        "coordinate_admission_rule_id": COORDINATE_RULE_ID,
        "deployed_revision": deployed_revision,
        "required_worker_capability": "ordinary_full_grow_v1",
        "retry_budget": retry_budget,
        "fresh_start": True,
        "resume_from": None,
        "resume_artifact": None,
        "probe_checkpoint": None,
        "allow_unvalidated": False,
    }
    return {**core, "normal_growth_lock_sha256": _task6_sha256(core)}


def validate_first_letters_normal_growth_lock(value: Any) -> dict[str, Any]:
    """Reconstruct and compare the normal lock before claim/grow/finalize."""

    if not isinstance(value, dict) or value.get("allow_unvalidated") is not False:
        raise ValueError("normal growth lock must require validation")
    expected = load_first_letters_normal_growth_lock(
        source_snapshot_id=value.get("source_snapshot_id"),
        coordinate=value.get("promotion_coordinate_ct_l0_xyz"),
        coordinate_sha256=value.get("promotion_coordinate_sha256"),
        deployed_revision=value.get("deployed_revision"),
        retry_budget=value.get("retry_budget"),
    )
    if value != expected:
        raise ValueError("normal growth lock bytes/hash drift")
    return copy.deepcopy(value)


def project_provider_candidate_v1(
    value: Any, *, provider_response_sha256: str, provider_order: int = 0,
    prediction_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform the only provider-native {x,y,z} -> [x,y,z] projection."""

    _require_lowercase_sha256(provider_response_sha256, "provider_response_sha256")
    row = _closed_dict(
        value,
        {"candidate_id", "cell_id", "ct_l0_coordinate", "score"},
        "provider candidate",
    )
    if not isinstance(row["candidate_id"], str) or not row["candidate_id"]:
        raise ValueError("provider candidate_id must be a nonempty string")
    if not isinstance(row["cell_id"], str) or not row["cell_id"]:
        raise ValueError("provider candidate cell_id must be a nonempty string")
    if (isinstance(provider_order, bool) or not isinstance(provider_order, int)
            or provider_order < 0):
        raise ValueError("provider candidate order must be a nonnegative integer")
    identity = prediction_identity or {
        "cell_id": row["cell_id"],
        "cell_region_sha256": "0" * 64,
        "grid_spec_sha256": "0" * 64,
        "dependency_manifest_sha256": "0" * 64,
    }
    if row["cell_id"] != identity.get("cell_id"):
        raise ValueError("provider candidate cell differs from prediction identity")
    coordinate = _closed_dict(
        row["ct_l0_coordinate"], {"x", "y", "z"}, "provider coordinate"
    )
    raw = validate_task6_coordinate([
        coordinate["x"], coordinate["y"], coordinate["z"]
    ])
    score = row["score"]
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("provider score must be finite and within [0,1]")
    promotable = all(isinstance(item, int) and not isinstance(item, bool) for item in raw)
    raw_sha = coordinate_sha256_v1(raw)
    result = {
        "candidate_id": row["candidate_id"],
        "cell_id": row["cell_id"],
        "provider_order": provider_order,
        "raw_coordinate_ct_l0_xyz": raw,
        "raw_coordinate_sha256": raw_sha,
        "coordinate_frame": "ct_l0_xyz",
        "coordinate_schema": COORDINATE_SCHEMA,
        "coordinate_admission_rule_id": COORDINATE_RULE_ID,
        "promotion_coordinate_ct_l0_xyz": copy.deepcopy(raw) if promotable else None,
        "promotion_coordinate_sha256": raw_sha if promotable else None,
        "coordinate_admission_state": (
            "PROMOTABLE_INTEGRAL_COORDINATE_V1"
            if promotable else "REJECTED_NONINTEGRAL_COORDINATE_V1"
        ),
        "provider_score": score,
        "provider_response_sha256": provider_response_sha256,
        "cell_region_sha256": identity["cell_region_sha256"],
        "grid_spec_sha256": identity["grid_spec_sha256"],
        "dependency_manifest_sha256": identity[
            "dependency_manifest_sha256"
        ],
    }
    result["candidate_evidence_sha256"] = _task6_sha256(result)
    return result


def project_provider_response_v1(response_bytes: Any) -> dict[str, Any]:
    """Retain provider bytes verbatim and expose only canonical derived arrays."""

    import hashlib
    native = _strict_json_loads_bytes(response_bytes, "provider response")
    native = _closed_dict(
        native, {"prediction_identity", "candidates"}, "provider response bytes"
    )
    _reject_content_informed(native, "provider_response_bytes")
    identity = _closed_dict(
        native["prediction_identity"],
        {
            "request_id", "cell_id",
            "source_snapshot_id", "source_snapshot_sha256",
            "prediction_root_sha256", "resolution", "level", "model_id",
            "model_sha256", "provider_id", "provider_sha256",
            "cell_region_sha256", "grid_spec_sha256",
            "dependency_manifest_sha256", "maximum_candidates",
        },
        "provider prediction identity",
    )
    _reject_credential_value(identity, "provider_response_bytes.prediction_identity")
    for field in (
        "source_snapshot_sha256", "prediction_root_sha256", "model_sha256",
        "provider_sha256", "cell_region_sha256", "grid_spec_sha256",
        "dependency_manifest_sha256",
    ):
        _require_lowercase_sha256(identity[field], f"prediction identity {field}")
    for field in (
        "request_id", "cell_id", "source_snapshot_id", "model_id", "provider_id"
    ):
        if not isinstance(identity[field], str) or not identity[field]:
            raise ValueError(f"prediction identity {field} must be a nonempty string")
    for field in ("resolution", "level"):
        if (isinstance(identity[field], bool)
                or not isinstance(identity[field], int)
                or identity[field] < 0):
            raise ValueError(f"prediction identity {field} must be an integer")
    maximum = identity["maximum_candidates"]
    if (isinstance(maximum, bool) or not isinstance(maximum, int)
            or not 0 <= maximum <= 2):
        raise ValueError("prediction identity maximum_candidates is invalid")
    if not isinstance(native["candidates"], list):
        raise ValueError("provider response has no candidate array")
    response_sha = hashlib.sha256(response_bytes).hexdigest()
    if len(native["candidates"]) > maximum:
        raise ValueError("provider response exceeds the frozen candidate limit")
    candidates = [
        project_provider_candidate_v1(
            row, provider_response_sha256=response_sha, provider_order=index,
            prediction_identity=identity,
        )
        for index, row in enumerate(native["candidates"])
    ]
    return {
        "schema": "campaignx.first_letters_provider_response_projection.v1",
        "response_bytes": response_bytes,
        "response_sha256": response_sha,
        "prediction_identity": copy.deepcopy(identity),
        "candidates": candidates,
    }


_DISCOVERY_PROFILE_FIELDS = {
    "schema", "discovery_profile_id", "discovery_policy_id", "mode",
    "namespace", "noncanonical", "canonical_admission", "arm_kind",
    "experimental_arm_admission_id", "experimental_arm_admission_sha256",
    "source_snapshot_id", "source_snapshot_sha256", "source_content_lock_sha256",
    "ct_metadata_sha256", "ct_read_set_manifest_sha256", "m7_model_id",
    "m7_resolution", "m7_level", "m7_threshold", "m7_transform_sha256",
    "m7_read_set_manifest_sha256", "grid_spec", "canonical_ordered_cell_set_sha256",
    "candidate_policy", "ct_material_policy", "clearance_policy",
    "coordinate_admission_rule_id", "coordinate_admission_rule_sha256",
    "discovery_inputs_sha256", "probe_execution_profile_id",
    "probe_execution_profile_sha256", "top_k", "probe_generations",
    "adaptive_policy", "compute_unit", "mission_compute_cap_authority_id",
    "mission_compute_cap_authority_sha256", "mission_compute_cap_units",
    "deployed_revision", "allow_unvalidated", "scientific_core_sha256",
}


def validate_first_letters_discovery_profile(value: Any) -> dict[str, Any]:
    profile = copy.deepcopy(_closed_dict(
        value, _DISCOVERY_PROFILE_FIELDS, "First Letters discovery profile"
    ))
    _false_override(profile, "First Letters discovery profile")
    if (profile["schema"] != DISCOVERY_PROFILE_SCHEMA
            or profile["namespace"] != DISCOVERY_NAMESPACE
            or profile["noncanonical"] is not True
            or profile["canonical_admission"] != "PROHIBITED"
            or profile["mode"] != "shadow"
            or profile["top_k"] != 2
            or profile["probe_generations"] != 12
            or profile["probe_execution_profile_id"] != PROBE_PROFILE_ID
            or profile["probe_execution_profile_sha256"] != PROBE_PROFILE_SHA256
            or profile["coordinate_admission_rule_id"] != COORDINATE_RULE_ID
            or profile["compute_unit"] != "probe_generation_units"):
        raise ValueError("First Letters shadow discovery profile is unsupported")
    if profile["arm_kind"] not in DISCOVERY_ARM_KINDS:
        raise ValueError("unsupported Task 6 experimental arm kind")
    if profile["arm_kind"] == "BASELINE":
        if (profile["experimental_arm_admission_id"] is not None
                or profile["experimental_arm_admission_sha256"] is not None):
            raise ValueError("baseline cannot carry experimental-arm authority")
    else:
        if not profile["experimental_arm_admission_id"]:
            raise ValueError("alternative source arm requires its own admission")
        _require_lowercase_sha256(
            profile["experimental_arm_admission_sha256"],
            "experimental_arm_admission_sha256",
        )
    for field in (
        "source_snapshot_sha256", "source_content_lock_sha256",
        "ct_metadata_sha256", "ct_read_set_manifest_sha256",
        "m7_transform_sha256", "m7_read_set_manifest_sha256",
        "canonical_ordered_cell_set_sha256", "coordinate_admission_rule_sha256",
        "discovery_inputs_sha256", "mission_compute_cap_authority_sha256",
    ):
        _require_lowercase_sha256(profile[field], field)
    core = {key: row for key, row in profile.items() if key != "scientific_core_sha256"}
    if profile["scientific_core_sha256"] != _task6_sha256(core):
        raise ValueError("discovery profile scientific core hash is invalid")
    return profile


def load_first_letters_discovery_profile_bytes(raw: Any) -> dict[str, Any]:
    """Load registered profile bytes while retaining their exact identity."""

    import hashlib
    if not isinstance(raw, bytes):
        raise ValueError("discovery profile must be retained as exact bytes")
    value = _strict_json_loads_bytes(raw, "discovery profile")
    profile = validate_first_letters_discovery_profile(value)
    profile["profile_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return profile


def load_first_letters_discovery_profile(path: str | Path) -> dict[str, Any]:
    """Load a discovery profile while retaining its exact raw file identity."""

    return load_first_letters_discovery_profile_bytes(Path(path).read_bytes())


_INPUT_FIELDS = {
    "schema", "source_snapshot", "dependencies", "provider_request",
    "provider_response", "allow_unvalidated", "discovery_inputs_sha256",
}
_REQUEST_FIELDS = {
    "request_id", "cell_id", "source_snapshot_id", "source_snapshot_sha256",
    "prediction_root_sha256", "resolution", "level", "threshold",
    "ct_l0_region", "cell_region_sha256", "grid_spec_sha256",
    "coordinate_frame", "maximum_candidates",
    "minimum_separation", "model_id", "model_sha256", "provider_id",
    "provider_sha256", "coordinate_admission_rule_id",
    "coordinate_admission_rule_sha256", "dependency_manifest_sha256",
}
_RESPONSE_FIELDS = {
    "response_sha256", "response_bytes", "prediction_identity", "candidates",
}
_SOURCE_SNAPSHOT_DEPENDENCY_FIELDS = {
    "role", "source_snapshot_id", "source_snapshot_sha256",
    "source_content_lock_sha256", "cell_id", "cell_region_sha256",
    "grid_spec_sha256",
}
_DEPENDENCY_FIELDS = {
    "role", "artifact_sha256", "read_set_manifest_sha256", "cell_id",
    "cell_region_sha256", "grid_spec_sha256",
}
_REQUIRED_DEPENDENCY_ROLES = (
    "CT_VOLUME", "M7_PREDICTION_VOLUME", "CANONICAL_GRID_SPEC",
    "CT_MATERIAL_SUPPORT_POLICY", "CELL_AND_VOLUME_CLEARANCE_POLICY",
)


def _discovery_inputs_sha256(value: dict[str, Any]) -> str:
    """Hash the closed JSON authority and its one retained byte field."""

    import hashlib

    core = copy.deepcopy({
        key: row for key, row in value.items()
        if key != "discovery_inputs_sha256"
    })
    response_bytes = core["provider_response"].pop("response_bytes")
    return hashlib.sha256(
        _task6_canonical_bytes(core) + b"\0" + response_bytes
    ).hexdigest()


def validate_first_letters_discovery_inputs(value: Any) -> dict[str, Any]:
    result = copy.deepcopy(_closed_dict(value, _INPUT_FIELDS, "discovery inputs"))
    _false_override(result, "discovery inputs")
    if result["schema"] != DISCOVERY_INPUTS_SCHEMA:
        raise ValueError("unsupported discovery-input schema")
    _reject_content_informed(result)
    _reject_credential_value(result, "discovery_inputs")
    source = _closed_dict(
        result["source_snapshot"],
        _SOURCE_SNAPSHOT_DEPENDENCY_FIELDS,
        "source snapshot dependency",
    )
    if source["role"] != "SOURCE_SNAPSHOT_METADATA":
        raise ValueError("unsupported source snapshot role")
    dependencies = result["dependencies"]
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list")
    if len(dependencies) != len(_REQUIRED_DEPENDENCY_ROLES):
        raise ValueError("dependencies must contain the exact required role set")
    for dependency, expected_role in zip(
        dependencies, _REQUIRED_DEPENDENCY_ROLES, strict=True
    ):
        row = _closed_dict(dependency, _DEPENDENCY_FIELDS, "dependency")
        if row["role"] != expected_role:
            raise ValueError("discovery dependency roles/order differ from authority")
        for field in (
            "artifact_sha256", "read_set_manifest_sha256",
            "cell_region_sha256", "grid_spec_sha256",
        ):
            _require_lowercase_sha256(row[field], f"dependency {field}")
    request = _closed_dict(result["provider_request"], _REQUEST_FIELDS, "provider request")
    response = _closed_dict(result["provider_response"], _RESPONSE_FIELDS, "provider response")
    for field in (
        "request_id", "cell_id", "source_snapshot_id", "model_id", "provider_id"
    ):
        _require_nonempty_string(request[field], f"provider request {field}")
    for field in (
        "source_snapshot_sha256", "prediction_root_sha256", "model_sha256",
        "provider_sha256", "cell_region_sha256", "grid_spec_sha256",
        "coordinate_admission_rule_sha256", "dependency_manifest_sha256",
    ):
        _require_lowercase_sha256(request[field], f"provider request {field}")
    if (request["coordinate_admission_rule_id"] != COORDINATE_RULE_ID
            or request["coordinate_frame"] != "ct_l0_xyz"):
        raise ValueError("provider request uses unsupported coordinate authority")
    for field in ("resolution", "level"):
        item = request[field]
        if (isinstance(item, bool) or not isinstance(item, int)
                or not 0 <= item <= 64):
            raise ValueError(f"provider request {field} is invalid")
    threshold = request["threshold"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("provider request threshold is invalid")
    maximum = request["maximum_candidates"]
    separation = request["minimum_separation"]
    if (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum != 2
            or isinstance(separation, bool) or not isinstance(separation, int)
            or not 0 <= separation <= 2**31 - 1):
        raise ValueError("provider request candidate limits are invalid")
    region = _closed_dict(
        request["ct_l0_region"], {"minimum", "maximum"}, "CT-L0 region"
    )
    for name in ("minimum", "maximum"):
        bound = region[name]
        if (not isinstance(bound, list) or len(bound) != 3
                or any(isinstance(item, bool) or not isinstance(item, int)
                       or not 0 <= item <= 2**31 - 1 for item in bound)):
            raise ValueError("CT-L0 region bounds must be finite nonnegative integers")
    if any(lower >= upper for lower, upper in zip(
        region["minimum"], region["maximum"], strict=True
    )):
        raise ValueError("CT-L0 region bounds must have positive extent")
    if request["cell_region_sha256"] != _task6_sha256(region):
        raise ValueError("provider request region hash is invalid")
    for field in (
        "source_snapshot_sha256", "source_content_lock_sha256",
        "cell_region_sha256", "grid_spec_sha256",
    ):
        _require_lowercase_sha256(source[field], f"source snapshot {field}")
    if (source["source_snapshot_id"] != request["source_snapshot_id"]
            or source["source_snapshot_sha256"] != request["source_snapshot_sha256"]
            or source["cell_id"] != request["cell_id"]
            or source["cell_region_sha256"] != request["cell_region_sha256"]
            or source["grid_spec_sha256"] != request["grid_spec_sha256"]):
        raise ValueError("source snapshot dependency is not bound to request cell")
    for dependency in dependencies:
        if (dependency["cell_id"] != request["cell_id"]
                or dependency["cell_region_sha256"] !=
                    request["cell_region_sha256"]
                or dependency["grid_spec_sha256"] != request["grid_spec_sha256"]):
            raise ValueError("discovery dependency is not bound to request cell")
    if request["dependency_manifest_sha256"] != _task6_sha256(dependencies):
        raise ValueError("provider request dependency manifest hash is invalid")
    if not isinstance(response["response_bytes"], bytes):
        raise ValueError("provider response must be retained as exact bytes")
    parsed = project_provider_response_v1(response["response_bytes"])
    _require_lowercase_sha256(response["response_sha256"], "provider response SHA-256")
    if response["response_sha256"] != parsed["response_sha256"]:
        raise ValueError("provider response SHA-256 differs from exact raw bytes")
    if response["prediction_identity"] != parsed["prediction_identity"]:
        raise ValueError("provider prediction identity differs from parsed bytes")
    native_response = _strict_json_loads_bytes(
        response["response_bytes"], "provider response"
    )
    if response["candidates"] != native_response["candidates"]:
        raise ValueError("provider candidates differ from parsed bytes")
    identity = parsed["prediction_identity"]
    identity_bindings = {
        "request_id": request["request_id"],
        "cell_id": request["cell_id"],
        "source_snapshot_id": request["source_snapshot_id"],
        "source_snapshot_sha256": request["source_snapshot_sha256"],
        "prediction_root_sha256": request["prediction_root_sha256"],
        "resolution": request["resolution"],
        "level": request["level"],
        "model_id": request["model_id"],
        "model_sha256": request["model_sha256"],
        "provider_id": request["provider_id"],
        "provider_sha256": request["provider_sha256"],
        "cell_region_sha256": request["cell_region_sha256"],
        "grid_spec_sha256": request["grid_spec_sha256"],
        "dependency_manifest_sha256": request["dependency_manifest_sha256"],
        "maximum_candidates": request["maximum_candidates"],
    }
    if identity != identity_bindings:
        raise ValueError("provider prediction identity differs from request authority")
    projected = parsed["candidates"]
    if len(projected) > request["maximum_candidates"]:
        raise ValueError("provider candidates exceed request maximum")
    if len({row["candidate_id"] for row in projected}) != len(projected):
        raise ValueError("duplicate candidate IDs are prohibited")
    if len({tuple(row["raw_coordinate_ct_l0_xyz"]) for row in projected}) != len(projected):
        raise ValueError("duplicate raw candidate coordinates are prohibited")
    for row in projected:
        if (row["cell_id"] != request["cell_id"]
                or any(coordinate < lower or coordinate >= upper
                       for coordinate, lower, upper in zip(
                           row["raw_coordinate_ct_l0_xyz"], region["minimum"],
                           region["maximum"], strict=True
                       ))):
            raise ValueError("provider candidate lies outside the request cell")
    for index, left in enumerate(projected):
        for right in projected[index + 1:]:
            distance = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(
                    left["raw_coordinate_ct_l0_xyz"],
                    right["raw_coordinate_ct_l0_xyz"], strict=True,
                )
            ))
            if distance < separation:
                raise ValueError("provider candidates violate minimum separation")
    expected_hash = _discovery_inputs_sha256(result)
    if result["discovery_inputs_sha256"] != expected_hash:
        raise ValueError("discovery-input hash is invalid")
    result["projected_candidates"] = projected
    return result


def discovery_scientific_projection(value: Any) -> dict[str, Any]:
    validated = validate_first_letters_discovery_inputs(value)
    return {
        "candidate_ids": [row["candidate_id"] for row in validated["projected_candidates"]],
        "candidate_evidence_sha256s": [
            row["candidate_evidence_sha256"] for row in validated["projected_candidates"]
        ],
        "ordered_raw_coordinates": [
            row["raw_coordinate_ct_l0_xyz"] for row in validated["projected_candidates"]
        ],
    }


_DISCOVERY_NON_CLAIMS = [
    "discovery evidence is not a canonical surface or evidence of absence",
    "candidate scarcity is not evidence that a scroll lacks papyrus, a surface, ink, text, or letters",
    "automated discovery evidence does not establish ink, text, or letters",
]
_RETAINED_FILE_ROLES = frozenset({
    "CANDIDATE_PROVIDER_REQUEST",
    "CANDIDATE_PROVIDER_RESPONSE",
    "CT_MATERIAL_READ_EVIDENCE",
    "NONCANONICAL_PROBE_GEOMETRY",
    "DISCOVERY_SELECTION_POLICY_RECEIPT",
})
_REQUIRED_RETAINED_SINGLETON_ROLES = frozenset({
    "CANDIDATE_PROVIDER_REQUEST", "CANDIDATE_PROVIDER_RESPONSE",
    "DISCOVERY_SELECTION_POLICY_RECEIPT",
})
_RETAINED_FILE_FIELDS = {"relative_path", "role", "bytes"}
_FILE_MANIFEST_FIELDS = {"relative_path", "byte_count", "sha256", "role"}
_EXECUTION_AUTHORITY_FIELDS = {
    "schema", "mission_id", "sample_id", "parent_task_id",
    "parent_attempt_id", "worker_id", "run_id", "run_authority_sha256",
    "scientific_opportunity_id",
    "accepted_p0_artifact_id", "accepted_p0_artifact_sha256",
    "source_snapshot_id", "source_snapshot_sha256",
    "reservation_id", "reservation_sha256", "reservation_request_id",
    "reservation_work_kind", "reservation_work_authority_id",
    "reservation_work_authority_sha256", "reservation_source",
    "reservation_ordered_item_ids",
    "ordered_cell_ids", "allow_unvalidated", "execution_authority_sha256",
}
_RESERVATION_FIELDS = {
    "schema", "reservation_id", "mission_id", "request_id", "work_kind",
    "work_authority_id", "work_authority_sha256", "ordered_item_ids",
    "ordered_item_ids_sha256", "item_count", "compute_unit", "top_k",
    "probe_generations", "maximum_attempts_per_candidate", "units_per_item",
    "reserved_units", "cap_authority_id", "cap_authority_sha256",
    "reserved_before_units", "reserved_after_units", "source",
    "allow_unvalidated", "reservation_sha256", "created_at",
}
_SELECTION_FIELDS = {
    "schema", "outcome", "selected_candidate_id",
    "selection_policy_receipt", "selection_policy_receipt_sha256",
    "allow_unvalidated", "selection_sha256",
}
_OUTCOME_FIELDS = {
    "candidate_id", "ct_terminal", "clearance_terminal", "probe_evidence",
    "contributing_source_rows",
}
_CT_TERMINAL_FIELDS = {
    "schema", "candidate_id", "cell_id", "raw_coordinate_sha256",
    "cell_region_sha256", "grid_spec_sha256", "ct_metadata_sha256",
    "ct_read_set_manifest_sha256", "ct_material_policy",
    "ct_read_evidence_sha256", "state",
    "ct_terminal_sha256",
}
_CLEARANCE_TERMINAL_FIELDS = {
    "schema", "candidate_id", "cell_id", "raw_coordinate_sha256",
    "cell_region_sha256", "grid_spec_sha256", "ct_terminal_sha256",
    "clearance_policy", "state", "clearance_terminal_sha256",
}
_PROBE_EVIDENCE_FIELDS = {
    "schema", "candidate_id", "cell_id", "cell_region_sha256",
    "grid_spec_sha256", "clearance_terminal_sha256",
    "probe_execution_profile_sha256", "state", "used_units",
    "probe_artifact_sha256", "probe_evidence_sha256",
}
_SOURCE_ROW_FIELDS = {
    "schema", "role", "artifact_sha256", "read_set_manifest_sha256",
    "cell_id", "cell_region_sha256", "grid_spec_sha256", "source_row_sha256",
}
_PROJECTED_CANDIDATE_FIELDS = {
    "candidate_id", "cell_id", "provider_order", "raw_coordinate_ct_l0_xyz",
    "raw_coordinate_sha256", "cell_region_sha256", "grid_spec_sha256",
    "dependency_manifest_sha256",
    "coordinate_frame", "coordinate_schema", "coordinate_admission_rule_id",
    "promotion_coordinate_ct_l0_xyz", "promotion_coordinate_sha256",
    "coordinate_admission_state", "provider_score",
    "provider_response_sha256", "candidate_evidence_sha256",
}
_ARTIFACT_CANDIDATE_FIELDS = _PROJECTED_CANDIDATE_FIELDS | {
    "coordinate_admission_rule_sha256", "ct_terminal",
    "ct_terminal_sha256", "clearance_terminal", "clearance_terminal_sha256",
    "probe_evidence", "probe_evidence_sha256", "contributing_source_rows",
}
_FUNNEL_FIELDS = {
    "raw_candidates", "ct_supported_candidates",
    "clearance_supported_candidates", "probe_measurable_candidates",
}
_ARTIFACT_FIELDS = {
    "schema", "artifact_id", "evidence_set_id", "execution_authority_sha256",
    "run_id", "run_authority_sha256", "mission_id", "sample_id", "parent_task_id",
    "parent_attempt_id", "scientific_opportunity_id", "mode", "arm_kind",
    "namespace", "canonical_admission", "accepted_p0_artifact_id",
    "accepted_p0_artifact_sha256", "profile_file_sha256",
    "profile_scientific_core_sha256", "discovery_inputs_sha256",
    "source_snapshot_id", "source_snapshot_sha256",
    "source_content_lock_sha256", "ct_metadata_sha256",
    "ct_read_set_manifest_sha256", "m7_read_set_manifest_sha256",
    "m7_prediction_artifact_sha256", "ct_material_policy",
    "clearance_policy", "coordinate_admission_rule_sha256",
    "provider_request_id", "provider_request_sha256",
    "provider_response_sha256", "prediction_identity", "ordered_cell_ids",
    "cell_id", "cell_region_sha256", "grid_spec_sha256",
    "dependency_manifest_sha256",
    "canonical_ordered_cell_set_sha256", "file_manifest",
    "file_manifest_sha256", "candidates", "funnel_counts",
    "compute_cap_authority_id", "compute_cap_authority_sha256",
    "reservation_id", "reservation_sha256", "reserved_before_units",
    "reservation_request_id", "reservation_work_kind",
    "reservation_work_authority_id", "reservation_work_authority_sha256",
    "reservation_source", "reservation_ordered_item_ids",
    "reserved_after_units", "reserved_units", "used_units",
    "selection_outcome", "selected_candidate_id", "selection_policy_receipt",
    "selection_policy_receipt_sha256", "selection_sha256",
    "deployed_revision", "allow_unvalidated",
    "non_claims", "artifact_sha256",
}
_RECEIPT_FIELDS = {
    "schema", "receipt_id", "evidence_set_id", "execution_authority_sha256",
    "run_id", "run_authority_sha256", "mission_id", "sample_id", "parent_task_id",
    "parent_attempt_id", "scientific_opportunity_id", "mode", "arm_kind",
    "profile_file_sha256", "profile_scientific_core_sha256",
    "discovery_inputs_sha256", "accepted_p0_artifact_id",
    "accepted_p0_artifact_sha256", "source_snapshot_id",
    "source_snapshot_sha256", "source_content_lock_sha256",
    "ct_metadata_sha256", "ct_read_set_manifest_sha256",
    "m7_read_set_manifest_sha256", "provider_request_id",
    "provider_request_sha256", "provider_response_sha256",
    "prediction_identity", "ordered_cell_ids",
    "cell_id", "cell_region_sha256", "grid_spec_sha256",
    "dependency_manifest_sha256",
    "canonical_ordered_cell_set_sha256", "funnel_counts", "artifact_id",
    "artifact_sha256", "file_manifest", "file_manifest_sha256",
    "compute_cap_authority_id", "compute_cap_authority_sha256",
    "reservation_id", "reservation_sha256", "reserved_before_units",
    "reservation_request_id", "reservation_work_kind",
    "reservation_work_authority_id", "reservation_work_authority_sha256",
    "reservation_source", "reservation_ordered_item_ids",
    "reserved_after_units", "reserved_units", "used_units",
    "selection_outcome", "selected_candidate_id",
    "selected_candidate_evidence_sha256",
    "selected_candidate_raw_coordinate_sha256", "ct_terminal_sha256",
    "clearance_terminal_sha256", "probe_evidence_sha256",
    "selection_policy_receipt", "selection_policy_receipt_sha256", "selection_sha256",
    "deployed_revision", "namespace", "canonical_admission",
    "allow_unvalidated", "outcome", "non_claims", "receipt_sha256",
}


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _validate_execution_authority(value: Any) -> dict[str, Any]:
    authority = copy.deepcopy(_closed_dict(
        value, _EXECUTION_AUTHORITY_FIELDS, "discovery execution authority"
    ))
    _false_override(authority, "discovery execution authority")
    _reject_credential_value(authority, "discovery_execution_authority")
    if authority["schema"] != "campaignx.first_letters_discovery_execution_authority.v1":
        raise ValueError("unsupported discovery execution authority")
    for field in (
        "mission_id", "sample_id",
        "worker_id", "run_id", "scientific_opportunity_id",
        "accepted_p0_artifact_id", "source_snapshot_id", "reservation_id",
        "reservation_request_id", "reservation_work_kind",
        "reservation_work_authority_id", "reservation_source",
    ):
        _require_nonempty_string(authority[field], field)
    for field in ("parent_task_id", "parent_attempt_id"):
        if authority[field] is not None:
            _require_nonempty_string(authority[field], field)
    if authority["parent_attempt_id"] is not None and authority["parent_task_id"] is None:
        raise ValueError("discovery parent attempt requires parent task lineage")
    _require_lowercase_sha256(
        authority["accepted_p0_artifact_sha256"], "accepted P0 artifact"
    )
    for field in (
        "run_authority_sha256", "source_snapshot_sha256", "reservation_sha256",
        "reservation_work_authority_sha256",
    ):
        _require_lowercase_sha256(authority[field], field)
    cells = authority["ordered_cell_ids"]
    if (not isinstance(cells, list) or not cells
            or any(not isinstance(row, str) or not row for row in cells)
            or len(cells) != len(set(cells))):
        raise ValueError("ordered cell IDs must be a nonempty unique string list")
    reservation_items = authority["reservation_ordered_item_ids"]
    if (not isinstance(reservation_items, list) or not reservation_items
            or any(not isinstance(row, str) or not row for row in reservation_items)
            or len(reservation_items) != len(set(reservation_items))
            or any(cell not in reservation_items for cell in cells)):
        raise ValueError("execution cells differ from reservation items")
    expected = _task6_sha256({
        key: row for key, row in authority.items()
        if key != "execution_authority_sha256"
    })
    if authority["execution_authority_sha256"] != expected:
        raise ValueError("discovery execution authority hash is invalid")
    return authority


def _validate_reservation(
    value: Any, *, authority: dict[str, Any], profile: dict[str, Any],
) -> dict[str, Any]:
    reservation = copy.deepcopy(_closed_dict(
        value, _RESERVATION_FIELDS, "discovery compute reservation"
    ))
    _false_override(reservation, "discovery compute reservation")
    integer_fields = (
        "item_count", "top_k", "probe_generations",
        "maximum_attempts_per_candidate", "units_per_item", "reserved_units",
        "reserved_before_units", "reserved_after_units",
    )
    if any(isinstance(reservation[field], bool)
           or not isinstance(reservation[field], int) for field in integer_fields):
        raise ValueError("discovery reservation units/counts must be integers")
    for field in (
        "reservation_id", "mission_id", "request_id", "work_kind",
        "work_authority_id", "compute_unit", "cap_authority_id", "source",
        "created_at",
    ):
        _require_nonempty_string(reservation[field], field)
    ordered_items = reservation["ordered_item_ids"]
    if (not isinstance(ordered_items, list) or not ordered_items
            or any(not isinstance(row, str) or not row for row in ordered_items)
            or len(ordered_items) != len(set(ordered_items))):
        raise ValueError("reservation ordered items must be unique nonempty IDs")
    expected_kind = (
        "BASELINE_ARM" if profile["arm_kind"] == "BASELINE"
        else "ALTERNATIVE_SOURCE_ARM"
    )
    if (reservation["schema"] !=
            "campaignx.first_letters_discovery_compute_reservation.v1"
            or reservation["mission_id"] != authority["mission_id"]
            or reservation["reservation_id"] != authority["reservation_id"]
            or reservation["reservation_sha256"] != authority["reservation_sha256"]
            or reservation["request_id"] != authority["reservation_request_id"]
            or reservation["work_kind"] != expected_kind
            or reservation["work_authority_id"] !=
                authority["reservation_work_authority_id"]
            or reservation["work_authority_sha256"] !=
                authority["reservation_work_authority_sha256"]
            or reservation["source"] != authority["reservation_source"]
            or reservation["ordered_item_ids"] !=
                authority["reservation_ordered_item_ids"]
            or any(cell not in reservation["ordered_item_ids"]
                   for cell in authority["ordered_cell_ids"])
            or reservation["ordered_item_ids_sha256"] !=
                _task6_sha256(reservation["ordered_item_ids"])
            or reservation["item_count"] != len(reservation["ordered_item_ids"])
            or reservation["compute_unit"] != "probe_generation_units"
            or reservation["top_k"] != 2
            or reservation["probe_generations"] != 12
            or reservation["maximum_attempts_per_candidate"] != 1
            or reservation["units_per_item"] != 24
            or reservation["reserved_units"] != reservation["item_count"] * 24
            or reservation["reserved_before_units"] < 0
            or reservation["reserved_after_units"] !=
                reservation["reserved_before_units"] + reservation["reserved_units"]
            or reservation["reserved_after_units"] >
                profile["mission_compute_cap_units"]
            or reservation["cap_authority_id"] !=
                profile["mission_compute_cap_authority_id"]
            or reservation["cap_authority_sha256"] !=
                profile["mission_compute_cap_authority_sha256"]):
        raise ValueError("discovery compute reservation differs from authority")
    for field in (
        "work_authority_sha256", "ordered_item_ids_sha256",
        "cap_authority_sha256", "reservation_sha256",
    ):
        _require_lowercase_sha256(reservation[field], field)
    core = {
        key: row for key, row in reservation.items()
        if key not in {"reservation_sha256", "created_at"}
    }
    if reservation["reservation_sha256"] != _task6_sha256(core):
        raise ValueError("discovery reservation hash is invalid")
    return reservation


_SELECTION_POLICY_RECEIPT_FIELDS = {
    "schema", "policy_id", "producer_run_id", "profile_file_sha256",
    "eligibility_rule", "tie_rule", "ordered_candidate_inputs", "outcome",
    "selected_candidate_id", "allow_unvalidated", "policy_receipt_sha256",
}
_SELECTION_POLICY_INPUT_FIELDS = {
    "candidate_id", "provider_order", "provider_score", "promotable",
    "ct_state", "clearance_state", "probe_state", "candidate_evidence_sha256",
    "ct_terminal_sha256", "clearance_terminal_sha256", "probe_evidence_sha256",
    "eligible",
}


def _derive_selection(
    candidates: list[dict[str, Any]], *, profile_file_sha256: str, run_id: str,
) -> dict[str, Any]:
    ordered_inputs: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if candidate["provider_order"] != index:
            raise ValueError("candidate provider order differs from retained response")
        eligible = _promotion_eligible_discovery_candidate(candidate)
        ordered_inputs.append({
            "candidate_id": candidate["candidate_id"],
            "provider_order": candidate["provider_order"],
            "provider_score": candidate["provider_score"],
            "promotable": candidate["coordinate_admission_state"] ==
                "PROMOTABLE_INTEGRAL_COORDINATE_V1",
            "ct_state": candidate["ct_terminal"]["state"],
            "clearance_state": candidate["clearance_terminal"]["state"],
            "probe_state": candidate["probe_evidence"]["state"],
            "candidate_evidence_sha256": candidate["candidate_evidence_sha256"],
            "ct_terminal_sha256": candidate["ct_terminal_sha256"],
            "clearance_terminal_sha256": candidate["clearance_terminal_sha256"],
            "probe_evidence_sha256": candidate["probe_evidence_sha256"],
            "eligible": eligible,
        })
    eligible = [row for row in ordered_inputs if row["eligible"]]
    selected: str | None = None
    if not candidates:
        outcome = "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER"
    elif not eligible:
        outcome = "DISCOVERY_REJECTED_CANDIDATES"
    else:
        highest = max(row["provider_score"] for row in eligible)
        winners = [row for row in eligible if row["provider_score"] == highest]
        if len(winners) == 1:
            outcome = "DISCOVERY_WINNER_RETAINED"
            selected = winners[0]["candidate_id"]
        else:
            outcome = "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER"
    receipt_core = {
        "schema": "campaignx.first_letters_discovery_selection_policy_receipt.v1",
        "policy_id": "provider-order-score-within-arm-v1",
        "producer_run_id": run_id,
        "profile_file_sha256": profile_file_sha256,
        "eligibility_rule":
            "integral-ct-supported-clearance-supported-measurable-v1",
        "tie_rule": "abstain-on-equal-highest-provider-score-v1",
        "ordered_candidate_inputs": ordered_inputs,
        "outcome": outcome,
        "selected_candidate_id": selected,
        "allow_unvalidated": False,
    }
    policy_receipt = {
        **receipt_core,
        "policy_receipt_sha256": _task6_sha256(receipt_core),
    }
    selection_core = {
        "schema": "campaignx.first_letters_discovery_selection.v1",
        "outcome": outcome,
        "selected_candidate_id": selected,
        "selection_policy_receipt": policy_receipt,
        "selection_policy_receipt_sha256":
            policy_receipt["policy_receipt_sha256"],
        "allow_unvalidated": False,
    }
    return {**selection_core, "selection_sha256": _task6_sha256(selection_core)}


def _validate_selection(
    value: Any, *, candidates: list[dict[str, Any]] | None = None,
    profile_file_sha256: str | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    selection = copy.deepcopy(_closed_dict(
        value, _SELECTION_FIELDS, "discovery selection"
    ))
    _false_override(selection, "discovery selection")
    if selection["schema"] != "campaignx.first_letters_discovery_selection.v1":
        raise ValueError("unsupported discovery selection")
    receipt = copy.deepcopy(_closed_dict(
        selection["selection_policy_receipt"],
        _SELECTION_POLICY_RECEIPT_FIELDS,
        "discovery selection-policy receipt",
    ))
    _false_override(receipt, "discovery selection-policy receipt")
    if (receipt["schema"] !=
            "campaignx.first_letters_discovery_selection_policy_receipt.v1"
            or receipt["policy_id"] != "provider-order-score-within-arm-v1"
            or receipt["eligibility_rule"] !=
                "integral-ct-supported-clearance-supported-measurable-v1"
            or receipt["tie_rule"] !=
                "abstain-on-equal-highest-provider-score-v1"):
        raise ValueError("unsupported discovery selection-policy receipt")
    rows = receipt["ordered_candidate_inputs"]
    if not isinstance(rows, list):
        raise ValueError("selection-policy candidate inputs must be ordered")
    for index, row in enumerate(rows):
        _closed_dict(row, _SELECTION_POLICY_INPUT_FIELDS, "selection-policy input")
        if row["provider_order"] != index:
            raise ValueError("selection-policy provider order is invalid")
    _require_lowercase_sha256(receipt["policy_receipt_sha256"], "policy receipt")
    if receipt["policy_receipt_sha256"] != _task6_sha256({
        key: row for key, row in receipt.items() if key != "policy_receipt_sha256"
    }):
        raise ValueError("selection-policy receipt hash is invalid")
    outcomes = {
        "DISCOVERY_WINNER_RETAINED",
        "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER",
        "DISCOVERY_REJECTED_CANDIDATES",
    }
    if selection["outcome"] not in outcomes:
        raise ValueError("unsupported discovery selection outcome")
    selected = selection["selected_candidate_id"]
    if ((selection["outcome"] == "DISCOVERY_WINNER_RETAINED") !=
            (isinstance(selected, str) and bool(selected))):
        raise ValueError("discovery winner outcome and selected candidate disagree")
    if selected is not None and (not isinstance(selected, str) or not selected):
        raise ValueError("selected candidate ID is invalid")
    _require_lowercase_sha256(
        selection["selection_policy_receipt_sha256"], "selection policy receipt"
    )
    expected = _task6_sha256({
        key: row for key, row in selection.items() if key != "selection_sha256"
    })
    if selection["selection_sha256"] != expected:
        raise ValueError("discovery selection hash is invalid")
    if (selection["selection_policy_receipt_sha256"] !=
            receipt["policy_receipt_sha256"]
            or selection["outcome"] != receipt["outcome"]
            or selection["selected_candidate_id"] !=
                receipt["selected_candidate_id"]):
        raise ValueError("selection differs from its retained policy receipt")
    if candidates is not None:
        expected_selection = _derive_selection(
            candidates,
            profile_file_sha256=str(profile_file_sha256), run_id=str(run_id),
        )
        if selection != expected_selection:
            raise ValueError("selection differs from deterministic retained policy")
    return selection


def _safe_probe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("discovery retained path must be a POSIX relative path")
    parsed = urlparse(value)
    path = PurePosixPath(value)
    if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or parsed.username or parsed.password or path.is_absolute()
            or value != path.as_posix() or not value.startswith("probes/")
            or any(part in {"", ".", ".."} for part in path.parts)
            or "@" in value or ":" in value):
        raise ValueError("discovery retained path is unsafe or credential-bearing")
    return value


def _normalize_retained_files(value: Any) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    import hashlib

    if not isinstance(value, list):
        raise ValueError("retained discovery files must be a list")
    manifest: list[dict[str, Any]] = []
    actual: dict[str, bytes] = {}
    roles: set[str] = set()
    singleton_roles: set[str] = set()
    for item in value:
        row = _closed_dict(item, _RETAINED_FILE_FIELDS, "retained discovery file")
        path = _safe_probe_relative_path(row["relative_path"])
        role = row["role"]
        payload = row["bytes"]
        if role not in _RETAINED_FILE_ROLES:
            raise ValueError("retained discovery file role is unsupported")
        if (path in actual
                or (role in _REQUIRED_RETAINED_SINGLETON_ROLES
                    and role in singleton_roles)):
            raise ValueError("retained discovery file paths and singleton roles must be unique")
        if not isinstance(payload, bytes):
            raise ValueError("retained discovery file content must be exact bytes")
        if role in _REQUIRED_RETAINED_SINGLETON_ROLES:
            parsed_payload = _strict_json_loads_bytes(
                payload, f"retained discovery file {path}"
            )
            _reject_credential_value(parsed_payload, f"retained_files.{path}")
        try:
            parsed_payload = _strict_json_loads_bytes(
                payload, f"retained discovery file {path}"
            )
        except ValueError:
            parsed_payload = None
        if parsed_payload is not None:
            _reject_credential_value(parsed_payload, f"retained_files.{path}")
            _reject_content_informed(parsed_payload, f"retained_files.{path}")
        roles.add(role)
        if role != "NONCANONICAL_PROBE_GEOMETRY":
            singleton_roles.add(role)
        actual[path] = payload
        manifest.append({
            "relative_path": path,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "role": role,
        })
    if not _REQUIRED_RETAINED_SINGLETON_ROLES.issubset(roles):
        raise ValueError("retained discovery inventory is incomplete or unexpected")
    return manifest, actual


def _manifest_role(manifest: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [row for row in manifest if row["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"discovery inventory requires one {role} file")
    return matches[0]


def _validate_hashed_row(
    value: Any, fields: set[str], label: str, hash_field: str,
) -> dict[str, Any]:
    row = copy.deepcopy(_closed_dict(value, fields, label))
    _require_lowercase_sha256(row[hash_field], hash_field)
    expected = _task6_sha256({
        key: child for key, child in row.items() if key != hash_field
    })
    if row[hash_field] != expected:
        raise ValueError(f"{label} hash is invalid")
    return row


def _validate_artifact_candidate(
    value: Any, *, artifact: dict[str, Any],
) -> dict[str, Any]:
    candidate = copy.deepcopy(_closed_dict(
        value, _ARTIFACT_CANDIDATE_FIELDS, "discovery artifact candidate"
    ))
    _reject_content_informed(candidate, "artifact.candidate")
    candidate_id = _require_nonempty_string(candidate["candidate_id"], "candidate_id")
    raw = validate_task6_coordinate(candidate["raw_coordinate_ct_l0_xyz"])
    raw_sha = coordinate_sha256_v1(raw)
    promotable = all(isinstance(item, int) and not isinstance(item, bool) for item in raw)
    expected_state = (
        "PROMOTABLE_INTEGRAL_COORDINATE_V1"
        if promotable else "REJECTED_NONINTEGRAL_COORDINATE_V1"
    )
    if (candidate["raw_coordinate_sha256"] != raw_sha
            or candidate["cell_id"] != artifact["cell_id"]
            or candidate["cell_region_sha256"] != artifact["cell_region_sha256"]
            or candidate["grid_spec_sha256"] != artifact["grid_spec_sha256"]
            or candidate["dependency_manifest_sha256"] !=
                artifact["dependency_manifest_sha256"]
            or candidate["coordinate_frame"] != "ct_l0_xyz"
            or candidate["coordinate_schema"] != COORDINATE_SCHEMA
            or candidate["coordinate_admission_rule_id"] != COORDINATE_RULE_ID
            or candidate["coordinate_admission_rule_sha256"] !=
                artifact["coordinate_admission_rule_sha256"]):
        raise ValueError("artifact candidate coordinate authority is invalid")
    if (candidate["coordinate_admission_state"] != expected_state
            or candidate["promotion_coordinate_ct_l0_xyz"] !=
                (raw if promotable else None)
            or candidate["promotion_coordinate_sha256"] !=
                (raw_sha if promotable else None)):
        raise ValueError("artifact candidate promotion coordinate is invalid")
    score = candidate["provider_score"]
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1
            or isinstance(candidate["provider_order"], bool)
            or not isinstance(candidate["provider_order"], int)
            or candidate["provider_order"] < 0
            or candidate["provider_response_sha256"] !=
                artifact["provider_response_sha256"]):
        raise ValueError("artifact candidate provider evidence is invalid")
    projection = {
        key: candidate[key] for key in _PROJECTED_CANDIDATE_FIELDS
        if key != "candidate_evidence_sha256"
    }
    if candidate["candidate_evidence_sha256"] != _task6_sha256(projection):
        raise ValueError("artifact candidate evidence hash is invalid")

    ct = _validate_hashed_row(
        candidate["ct_terminal"], _CT_TERMINAL_FIELDS,
        "candidate CT terminal", "ct_terminal_sha256",
    )
    expected_ct_states = (
        {"CT_SUPPORTED", "CT_REJECTED"}
        if promotable else {"CT_NOT_RUN_NONINTEGRAL_COORDINATE"}
    )
    if (ct["schema"] != "campaignx.first_letters_ct_terminal.v1"
            or ct["candidate_id"] != candidate_id
            or ct["cell_id"] != candidate["cell_id"]
            or ct["raw_coordinate_sha256"] != raw_sha
            or ct["cell_region_sha256"] != candidate["cell_region_sha256"]
            or ct["grid_spec_sha256"] != candidate["grid_spec_sha256"]
            or ct["ct_metadata_sha256"] != artifact["ct_metadata_sha256"]
            or ct["ct_read_set_manifest_sha256"] !=
                artifact["ct_read_set_manifest_sha256"]
            or ct["ct_material_policy"] != artifact["ct_material_policy"]
            or (promotable !=
                isinstance(ct["ct_read_evidence_sha256"], str))
            or ct["state"] not in expected_ct_states
            or candidate["ct_terminal_sha256"] != ct["ct_terminal_sha256"]):
        raise ValueError("candidate CT terminal differs from artifact authority")
    if promotable:
        _require_lowercase_sha256(
            ct["ct_read_evidence_sha256"], "CT read evidence"
        )
    clearance = _validate_hashed_row(
        candidate["clearance_terminal"], _CLEARANCE_TERMINAL_FIELDS,
        "candidate clearance terminal", "clearance_terminal_sha256",
    )
    expected_clearance = (
        {"CLEARANCE_SUPPORTED", "CLEARANCE_REJECTED"}
        if ct["state"] == "CT_SUPPORTED"
        else {"CLEARANCE_NOT_RUN_DUE_TO_CT"}
    )
    if (clearance["schema"] != "campaignx.first_letters_clearance_terminal.v1"
            or clearance["candidate_id"] != candidate_id
            or clearance["cell_id"] != candidate["cell_id"]
            or clearance["raw_coordinate_sha256"] != raw_sha
            or clearance["cell_region_sha256"] !=
                candidate["cell_region_sha256"]
            or clearance["grid_spec_sha256"] != candidate["grid_spec_sha256"]
            or clearance["ct_terminal_sha256"] != ct["ct_terminal_sha256"]
            or clearance["clearance_policy"] != artifact["clearance_policy"]
            or clearance["state"] not in expected_clearance
            or candidate["clearance_terminal_sha256"] !=
                clearance["clearance_terminal_sha256"]):
        raise ValueError("candidate clearance terminal differs from CT authority")
    probe = _validate_hashed_row(
        candidate["probe_evidence"], _PROBE_EVIDENCE_FIELDS,
        "candidate probe evidence", "probe_evidence_sha256",
    )
    used = probe["used_units"]
    ran = clearance["state"] == "CLEARANCE_SUPPORTED"
    if (probe["schema"] != "campaignx.first_letters_probe_evidence.v1"
            or probe["candidate_id"] != candidate_id
            or probe["cell_id"] != candidate["cell_id"]
            or probe["cell_region_sha256"] != candidate["cell_region_sha256"]
            or probe["grid_spec_sha256"] != candidate["grid_spec_sha256"]
            or probe["clearance_terminal_sha256"] !=
                clearance["clearance_terminal_sha256"]
            or probe["probe_execution_profile_sha256"] != PROBE_PROFILE_SHA256
            or isinstance(used, bool) or not isinstance(used, int)
            or (ran and (probe["state"] not in {"MEASURABLE", "UNMEASURABLE"}
                         or used != 12
                         or not isinstance(probe["probe_artifact_sha256"], str)))
            or (not ran and (probe["state"] != "NOT_RUN_DUE_TO_CLEARANCE"
                             or used != 0
                             or probe["probe_artifact_sha256"] is not None))
            or candidate["probe_evidence_sha256"] !=
                probe["probe_evidence_sha256"]):
        raise ValueError("candidate probe evidence differs from clearance authority")
    if ran:
        _require_lowercase_sha256(probe["probe_artifact_sha256"], "probe artifact")

    rows = candidate["contributing_source_rows"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("candidate requires exact CT and M7 contributing rows")
    expected_rows = (
        ("CT_VOLUME", artifact["ct_metadata_sha256"],
         artifact["ct_read_set_manifest_sha256"]),
        ("M7_PREDICTION_VOLUME", artifact["m7_prediction_artifact_sha256"],
         artifact["m7_read_set_manifest_sha256"]),
    )
    for raw_row, expected_row in zip(rows, expected_rows, strict=True):
        row = _validate_hashed_row(
            raw_row, _SOURCE_ROW_FIELDS, "candidate contributing source row",
            "source_row_sha256",
        )
        if (row["schema"] !=
                "campaignx.first_letters_contributing_source_row.v1"
                or (row["role"], row["artifact_sha256"],
                    row["read_set_manifest_sha256"]) != expected_row):
            raise ValueError("candidate contributing source row is unbound")
        if (row["cell_id"] != candidate["cell_id"]
                or row["cell_region_sha256"] != candidate["cell_region_sha256"]
                or row["grid_spec_sha256"] != candidate["grid_spec_sha256"]):
            raise ValueError("candidate contributing source cell is unbound")
    return candidate


def validate_first_letters_discovery_artifact(
    value: Any, *, retained_files: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact = copy.deepcopy(_closed_dict(value, _ARTIFACT_FIELDS, "discovery artifact"))
    _false_override(artifact, "discovery artifact")
    _reject_content_informed(artifact, "artifact")
    _reject_credential_value(artifact, "artifact")
    if (artifact["schema"] != DISCOVERY_ARTIFACT_SCHEMA
            or artifact["namespace"] != DISCOVERY_NAMESPACE
            or artifact["canonical_admission"] != "PROHIBITED"
            or artifact["mode"] != "shadow"
            or artifact["arm_kind"] not in DISCOVERY_ARM_KINDS
            or artifact["non_claims"] != _DISCOVERY_NON_CLAIMS):
        raise ValueError("discovery artifact identity/namespace is invalid")
    for field in (
        "execution_authority_sha256", "run_authority_sha256",
        "accepted_p0_artifact_sha256", "profile_file_sha256",
        "profile_scientific_core_sha256", "discovery_inputs_sha256",
        "source_snapshot_sha256", "source_content_lock_sha256",
        "ct_metadata_sha256", "ct_read_set_manifest_sha256",
        "m7_read_set_manifest_sha256", "m7_prediction_artifact_sha256",
        "coordinate_admission_rule_sha256", "provider_request_sha256",
        "provider_response_sha256", "cell_region_sha256", "grid_spec_sha256",
        "dependency_manifest_sha256", "canonical_ordered_cell_set_sha256",
        "file_manifest_sha256", "compute_cap_authority_sha256",
        "reservation_sha256", "reservation_work_authority_sha256",
        "selection_policy_receipt_sha256", "selection_sha256",
        "artifact_sha256",
    ):
        _require_lowercase_sha256(artifact[field], field)
    for field in (
        "artifact_id", "evidence_set_id", "run_id", "mission_id",
        "sample_id", "scientific_opportunity_id",
        "accepted_p0_artifact_id", "provider_request_id",
        "compute_cap_authority_id", "reservation_id", "ct_material_policy",
        "clearance_policy", "cell_id", "reservation_request_id",
        "reservation_work_kind", "reservation_work_authority_id",
        "reservation_source",
    ):
        _require_nonempty_string(artifact[field], field)
    for field in ("parent_task_id", "parent_attempt_id"):
        if artifact[field] is not None:
            _require_nonempty_string(artifact[field], field)
    cells = artifact["ordered_cell_ids"]
    if (not isinstance(cells, list) or not cells
            or any(not isinstance(row, str) or not row for row in cells)
            or len(cells) != len(set(cells))
            or artifact["canonical_ordered_cell_set_sha256"] !=
                _task6_sha256(cells)
            or not isinstance(artifact["deployed_revision"], str)
            or re.fullmatch(r"[0-9a-f]{40}", artifact["deployed_revision"]) is None):
        raise ValueError("artifact ordered cells or deployed revision are invalid")
    manifest, actual_files = _normalize_retained_files(retained_files)
    if (artifact["file_manifest"] != manifest
            or artifact["file_manifest_sha256"] != _task6_sha256(manifest)):
        raise ValueError("artifact manifest differs from exact retained files")
    if (_manifest_role(manifest, "CANDIDATE_PROVIDER_REQUEST")["sha256"] !=
            artifact["provider_request_sha256"]
            or _manifest_role(manifest, "CANDIDATE_PROVIDER_RESPONSE")["sha256"] !=
                artifact["provider_response_sha256"]):
        raise ValueError("artifact provider files differ from provider bindings")
    response_file = _manifest_role(manifest, "CANDIDATE_PROVIDER_RESPONSE")
    request_file = _manifest_role(manifest, "CANDIDATE_PROVIDER_REQUEST")
    response_projection = project_provider_response_v1(
        actual_files[response_file["relative_path"]]
    )
    if (artifact["prediction_identity"] != response_projection["prediction_identity"]
            or artifact["provider_response_sha256"] !=
                response_projection["response_sha256"]):
        raise ValueError("artifact prediction identity differs from retained response")
    retained_request = _strict_json_loads_bytes(
        actual_files[request_file["relative_path"]],
        "retained provider request",
    )
    request = _closed_dict(
        retained_request, _REQUEST_FIELDS, "retained provider request"
    )
    identity = response_projection["prediction_identity"]
    if (request["request_id"] != artifact["provider_request_id"]
            or request["cell_id"] != artifact["cell_id"]
            or request["cell_region_sha256"] != artifact["cell_region_sha256"]
            or request["grid_spec_sha256"] != artifact["grid_spec_sha256"]
            or request["dependency_manifest_sha256"] !=
                artifact["dependency_manifest_sha256"]
            or request["source_snapshot_id"] != artifact["source_snapshot_id"]
            or request["source_snapshot_sha256"] != artifact["source_snapshot_sha256"]
            or any(identity[field] != request[field] for field in (
                "source_snapshot_id", "source_snapshot_sha256",
                "prediction_root_sha256", "resolution", "level", "model_id",
                "model_sha256", "provider_id", "provider_sha256", "request_id",
                "cell_id", "cell_region_sha256", "grid_spec_sha256",
                "dependency_manifest_sha256", "maximum_candidates",
            ))):
        raise ValueError("artifact provider exchange identity is cross-unbound")
    candidates = artifact["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("artifact candidates must be an ordered list")
    validated_candidates = [
        _validate_artifact_candidate(row, artifact=artifact) for row in candidates
    ]
    projected_rows = [
        {key: candidate[key] for key in _PROJECTED_CANDIDATE_FIELDS}
        for candidate in validated_candidates
    ]
    if projected_rows != response_projection["candidates"]:
        raise ValueError("artifact candidates differ from retained response projection")
    if (len({row["candidate_id"] for row in validated_candidates}) !=
            len(validated_candidates)
            or len({tuple(row["raw_coordinate_ct_l0_xyz"])
                    for row in validated_candidates}) != len(validated_candidates)):
        raise ValueError("artifact candidate IDs and coordinates must be unique")
    from collections import Counter

    ct_manifest_shas = Counter(
        row["sha256"] for row in manifest
        if row["role"] == "CT_MATERIAL_READ_EVIDENCE"
    )
    used_ct_shas = Counter(
        row["ct_terminal"]["ct_read_evidence_sha256"]
        for row in validated_candidates
        if row["ct_terminal"]["ct_read_evidence_sha256"] is not None
    )
    if used_ct_shas != ct_manifest_shas:
        raise ValueError(
            "retained CT read inventory differs from candidate terminals"
        )
    probe_manifest_shas = Counter(
        row["sha256"] for row in manifest
        if row["role"] == "NONCANONICAL_PROBE_GEOMETRY"
    )
    used_probe_shas: Counter[str] = Counter()
    for candidate in validated_candidates:
        probe_sha = candidate["probe_evidence"]["probe_artifact_sha256"]
        if probe_sha is not None and probe_sha not in probe_manifest_shas:
            raise ValueError("candidate probe evidence differs from retained probe bytes")
        if probe_sha is not None:
            used_probe_shas[probe_sha] += 1
    if used_probe_shas != probe_manifest_shas:
        raise ValueError("retained probe inventory contains missing or unused evidence")
    selection_file = _manifest_role(
        manifest, "DISCOVERY_SELECTION_POLICY_RECEIPT"
    )
    retained_selection = _strict_json_loads_bytes(
        actual_files[selection_file["relative_path"]],
        "retained selection-policy receipt",
    )
    if retained_selection != artifact["selection_policy_receipt"]:
        raise ValueError("artifact selection policy differs from retained bytes")
    _validate_selection({
        "schema": "campaignx.first_letters_discovery_selection.v1",
        "outcome": artifact["selection_outcome"],
        "selected_candidate_id": artifact["selected_candidate_id"],
        "selection_policy_receipt": artifact["selection_policy_receipt"],
        "selection_policy_receipt_sha256":
            artifact["selection_policy_receipt_sha256"],
        "allow_unvalidated": False,
        "selection_sha256": artifact["selection_sha256"],
    }, candidates=validated_candidates,
       profile_file_sha256=artifact["profile_file_sha256"],
       run_id=artifact["run_id"])
    funnel = _closed_dict(
        artifact["funnel_counts"], _FUNNEL_FIELDS, "artifact funnel counts"
    )
    expected_funnel = {
        "raw_candidates": len(validated_candidates),
        "ct_supported_candidates": sum(
            row["ct_terminal"]["state"] == "CT_SUPPORTED"
            for row in validated_candidates
        ),
        "clearance_supported_candidates": sum(
            row["clearance_terminal"]["state"] == "CLEARANCE_SUPPORTED"
            for row in validated_candidates
        ),
        "probe_measurable_candidates": sum(
            row["probe_evidence"]["state"] == "MEASURABLE"
            for row in validated_candidates
        ),
    }
    used_units = sum(row["probe_evidence"]["used_units"] for row in validated_candidates)
    unit_fields = (
        "reserved_before_units", "reserved_after_units", "reserved_units",
        "used_units",
    )
    if (any(isinstance(funnel[field], bool) or not isinstance(funnel[field], int)
            or funnel[field] < 0 for field in _FUNNEL_FIELDS)
            or funnel != expected_funnel
            or any(isinstance(artifact[field], bool)
                   or not isinstance(artifact[field], int) for field in unit_fields)
            or artifact["used_units"] != used_units
            or not 0 <= used_units <= artifact["reserved_units"]
            or artifact["reserved_after_units"] !=
                artifact["reserved_before_units"] + artifact["reserved_units"]):
        raise ValueError("artifact funnel or compute units are invalid")
    expected = _task6_sha256({
        key: row for key, row in artifact.items() if key != "artifact_sha256"
    })
    if artifact["artifact_sha256"] != expected:
        raise ValueError("discovery artifact hash is invalid")
    return artifact


def _dependency_by_role(inputs: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [row for row in inputs["dependencies"] if row["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"discovery inputs require exactly one {role} dependency")
    return matches[0]


def _build_artifact_candidates(
    *, projected_candidates: list[dict[str, Any]], candidate_outcomes: Any,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    if (not isinstance(candidate_outcomes, list)
            or len(candidate_outcomes) != len(projected_candidates)):
        raise ValueError("candidate outcomes must cover the exact projected candidate set")
    candidates: list[dict[str, Any]] = []
    for projected, raw_outcome in zip(
        projected_candidates, candidate_outcomes, strict=True
    ):
        outcome = copy.deepcopy(_closed_dict(
            raw_outcome, _OUTCOME_FIELDS, "discovery candidate outcome"
        ))
        if outcome["candidate_id"] != projected["candidate_id"]:
            raise ValueError("candidate outcomes differ from retained response order")
        ct = _validate_hashed_row(
            outcome["ct_terminal"], _CT_TERMINAL_FIELDS,
            "candidate CT terminal", "ct_terminal_sha256",
        )
        clearance = _validate_hashed_row(
            outcome["clearance_terminal"], _CLEARANCE_TERMINAL_FIELDS,
            "candidate clearance terminal", "clearance_terminal_sha256",
        )
        probe = _validate_hashed_row(
            outcome["probe_evidence"], _PROBE_EVIDENCE_FIELDS,
            "candidate probe evidence", "probe_evidence_sha256",
        )
        candidates.append({
            **copy.deepcopy(projected),
            "coordinate_admission_rule_sha256":
                profile["coordinate_admission_rule_sha256"],
            "ct_terminal": ct,
            "ct_terminal_sha256": ct["ct_terminal_sha256"],
            "clearance_terminal": clearance,
            "clearance_terminal_sha256": clearance["clearance_terminal_sha256"],
            "probe_evidence": probe,
            "probe_evidence_sha256": probe["probe_evidence_sha256"],
            "contributing_source_rows": copy.deepcopy(
                outcome["contributing_source_rows"]
            ),
        })
    return candidates


def _derive_candidate_read_states(
    *, candidate: dict[str, Any],
    measurement: FirstLettersDiscoveryCandidateMeasurement,
    run_authority: dict[str, Any],
) -> tuple[
    bool | None, bool | None, bool | None, bytes | None, bytes | None,
]:
    """Derive terminal states from exact producer read evidence bytes."""

    promotable = (
        candidate["coordinate_admission_state"] ==
        "PROMOTABLE_INTEGRAL_COORDINATE_V1"
    )
    if not promotable:
        if (measurement.ct_read_evidence_bytes is not None
                or measurement.probe_evidence_bytes is not None):
            raise ValueError(
                "nonintegral candidate cannot carry measured read evidence"
            )
        return None, None, None, None, None

    ct_read = _strict_json_loads_bytes(
        measurement.ct_read_evidence_bytes, "candidate CT read evidence"
    )
    ct_read = _closed_dict(ct_read, {
        "schema", "candidate_id", "source_snapshot_id",
        "raw_coordinate_sha256", "ct_metadata_sha256",
        "ct_read_set_manifest_sha256", "sampled_voxel_count",
        "nonzero_voxel_count", "allow_unvalidated",
    }, "candidate CT read evidence")
    if (ct_read["schema"] !=
            "campaignx.first_letters_ct_material_read_evidence.v1"
            or ct_read["candidate_id"] != candidate["candidate_id"]
            or ct_read["source_snapshot_id"] !=
                run_authority["source_snapshot_id"]
            or ct_read["raw_coordinate_sha256"] !=
                candidate["raw_coordinate_sha256"]
            or ct_read["ct_metadata_sha256"] !=
                next(row["artifact_sha256"] for row in
                     run_authority["dependencies"]
                     if row["role"] == "CT_VOLUME")
            or ct_read["ct_read_set_manifest_sha256"] !=
                next(row["read_set_manifest_sha256"] for row in
                     run_authority["dependencies"]
                     if row["role"] == "CT_VOLUME")
            or ct_read["allow_unvalidated"] is not False
            or isinstance(ct_read["sampled_voxel_count"], bool)
            or not isinstance(ct_read["sampled_voxel_count"], int)
            or ct_read["sampled_voxel_count"] <= 0
            or isinstance(ct_read["nonzero_voxel_count"], bool)
            or not isinstance(ct_read["nonzero_voxel_count"], int)
            or not 0 <= ct_read["nonzero_voxel_count"] <=
                ct_read["sampled_voxel_count"]):
        raise ValueError("candidate CT read evidence differs from run authority")
    _reject_credential_value(ct_read, "candidate_ct_read_evidence")
    ct_supported = ct_read["nonzero_voxel_count"] > 0
    if not ct_supported:
        if measurement.probe_evidence_bytes is not None:
            raise ValueError("CT rejection cannot carry probe read evidence")
        return False, None, None, measurement.ct_read_evidence_bytes, None

    coordinate = candidate["promotion_coordinate_ct_l0_xyz"]
    region = run_authority["provider_request"]["ct_l0_region"]
    minimum = region["minimum"]
    maximum = region["maximum"]
    cell_clearance = min(
        *(coordinate[index] - minimum[index] for index in range(3)),
        *(maximum[index] - 1 - coordinate[index] for index in range(3)),
    )
    shape = run_authority["source_shape_xyz"]
    volume_clearance = min(
        *coordinate,
        *(shape[index] - 1 - coordinate[index] for index in range(3)),
    )
    clearance_supported = (
        cell_clearance >= run_authority["minimum_cell_clearance_voxels"]
        and volume_clearance >=
            run_authority["minimum_volume_clearance_voxels"]
    )
    if not clearance_supported:
        if measurement.probe_evidence_bytes is not None:
            raise ValueError(
                "clearance rejection cannot carry probe read evidence"
            )
        return True, False, None, measurement.ct_read_evidence_bytes, None

    probe_bytes = measurement.probe_evidence_bytes
    probe_read = _strict_json_loads_bytes(
        probe_bytes, "candidate probe read evidence"
    )
    probe_read = _closed_dict(probe_read, {
        "schema", "candidate_id", "raw_coordinate_sha256",
        "probe_execution_profile_sha256", "measurement_complete",
        "geometry_qc_state", "allow_unvalidated",
    }, "candidate probe read evidence")
    if (probe_read["schema"] !=
            "campaignx.first_letters_probe_geometry_read_evidence.v1"
            or probe_read["candidate_id"] != candidate["candidate_id"]
            or probe_read["raw_coordinate_sha256"] !=
                candidate["raw_coordinate_sha256"]
            or probe_read["probe_execution_profile_sha256"] !=
                PROBE_PROFILE_SHA256
            or type(probe_read["measurement_complete"]) is not bool
            or probe_read["geometry_qc_state"] not in {
                "GEOMETRY_CERTIFIED", "GEOMETRY_UNMEASURED",
                "GEOMETRY_REJECTED_BRIDGE",
                "GEOMETRY_REJECTED_LAMINA_SWITCH",
                "GEOMETRY_REJECTED_DISTORTION",
                "GEOMETRY_REJECTED_COVERAGE",
            }
            or probe_read["allow_unvalidated"] is not False):
        raise ValueError(
            "candidate probe read evidence differs from run authority"
        )
    _reject_credential_value(probe_read, "candidate_probe_read_evidence")
    measurable = (
        probe_read["measurement_complete"] is True
        and probe_read["geometry_qc_state"] == "GEOMETRY_CERTIFIED"
    )
    return (
        True, True, measurable, measurement.ct_read_evidence_bytes,
        probe_bytes,
    )


def _produce_first_letters_discovery_evidence_set(
    *, run_authority: dict[str, Any], profile_bytes: bytes,
    provider_request: dict[str, Any], provider_response_bytes: bytes,
    measurements: tuple[FirstLettersDiscoveryCandidateMeasurement, ...],
    reservation: dict[str, Any],
) -> dict[str, Any]:
    """Derive all receipts from a store-owned run and raw worker measurements."""

    import hashlib

    profile = load_first_letters_discovery_profile_bytes(profile_bytes)
    parsed_response = project_provider_response_v1(provider_response_bytes)
    native_response = _strict_json_loads_bytes(
        provider_response_bytes, "provider response"
    )
    response = {
        "response_sha256": parsed_response["response_sha256"],
        "response_bytes": provider_response_bytes,
        "prediction_identity": parsed_response["prediction_identity"],
        "candidates": native_response["candidates"],
    }
    inputs_core = {
        "schema": DISCOVERY_INPUTS_SCHEMA,
        "source_snapshot": copy.deepcopy(
            run_authority["source_snapshot_dependency"]
        ),
        "dependencies": copy.deepcopy(run_authority["dependencies"]),
        "provider_request": copy.deepcopy(provider_request),
        "provider_response": response,
        "allow_unvalidated": False,
    }
    inputs = {
        **inputs_core,
        "discovery_inputs_sha256": _discovery_inputs_sha256(inputs_core),
    }
    validated_inputs = validate_first_letters_discovery_inputs(inputs)
    projected = validated_inputs["projected_candidates"]
    if (not isinstance(measurements, tuple)
            or len(measurements) != len(projected)
            or any(type(row) is not FirstLettersDiscoveryCandidateMeasurement
                   for row in measurements)
            or [row.candidate_id for row in measurements] !=
                [row["candidate_id"] for row in projected]):
        raise ValueError(
            "raw measurements must cover the exact provider candidate order"
        )
    ct_dependency = _dependency_by_role(validated_inputs, "CT_VOLUME")
    m7_dependency = _dependency_by_role(
        validated_inputs, "M7_PREDICTION_VOLUME"
    )
    candidate_outcomes: list[dict[str, Any]] = []
    ct_read_files: list[dict[str, Any]] = []
    probe_files: list[dict[str, Any]] = []
    for candidate, measurement in zip(projected, measurements, strict=True):
        promotable = (
            candidate["coordinate_admission_state"] ==
            "PROMOTABLE_INTEGRAL_COORDINATE_V1"
        )
        (
            ct_supported, clearance_supported, probe_measurable, ct_read_bytes,
            probe_bytes,
        ) = _derive_candidate_read_states(
            candidate=candidate, measurement=measurement,
            run_authority={
                **run_authority,
                "provider_request": provider_request,
            },
        )
        if not promotable:
            ct_state = "CT_NOT_RUN_NONINTEGRAL_COORDINATE"
            clearance_state = "CLEARANCE_NOT_RUN_DUE_TO_CT"
            probe_state, used_units, probe_sha = (
                "NOT_RUN_DUE_TO_CLEARANCE", 0, None
            )
        elif not ct_supported:
            ct_state = "CT_REJECTED"
            clearance_state = "CLEARANCE_NOT_RUN_DUE_TO_CT"
            probe_state, used_units, probe_sha = (
                "NOT_RUN_DUE_TO_CLEARANCE", 0, None
            )
        elif not clearance_supported:
            ct_state = "CT_SUPPORTED"
            clearance_state = "CLEARANCE_REJECTED"
            probe_state, used_units, probe_sha = (
                "NOT_RUN_DUE_TO_CLEARANCE", 0, None
            )
        else:
            ct_state = "CT_SUPPORTED"
            clearance_state = "CLEARANCE_SUPPORTED"
            probe_state = "MEASURABLE" if probe_measurable else "UNMEASURABLE"
            used_units = 12
            probe_sha = hashlib.sha256(probe_bytes).hexdigest()
            probe_files.append({
                "relative_path":
                    f"probes/{run_authority['run_id']}/{candidate['candidate_id']}-geometry.bin",
                "role": "NONCANONICAL_PROBE_GEOMETRY",
                "bytes": probe_bytes,
            })
        ct_read_sha = (
            hashlib.sha256(ct_read_bytes).hexdigest()
            if ct_read_bytes is not None else None
        )
        if ct_read_bytes is not None:
            ct_read_files.append({
                "relative_path": (
                    f"probes/{run_authority['run_id']}/"
                    f"{candidate['candidate_id']}-ct-read.json"
                ),
                "role": "CT_MATERIAL_READ_EVIDENCE",
                "bytes": ct_read_bytes,
            })
        binding = {
            "cell_id": candidate["cell_id"],
            "cell_region_sha256": candidate["cell_region_sha256"],
            "grid_spec_sha256": candidate["grid_spec_sha256"],
        }
        ct_core = {
            "schema": "campaignx.first_letters_ct_terminal.v1",
            "candidate_id": candidate["candidate_id"], **binding,
            "raw_coordinate_sha256": candidate["raw_coordinate_sha256"],
            "ct_metadata_sha256": ct_dependency["artifact_sha256"],
            "ct_read_set_manifest_sha256":
                ct_dependency["read_set_manifest_sha256"],
            "ct_material_policy": profile["ct_material_policy"],
            "ct_read_evidence_sha256": ct_read_sha,
            "state": ct_state,
        }
        ct = {**ct_core, "ct_terminal_sha256": _task6_sha256(ct_core)}
        clearance_core = {
            "schema": "campaignx.first_letters_clearance_terminal.v1",
            "candidate_id": candidate["candidate_id"], **binding,
            "raw_coordinate_sha256": candidate["raw_coordinate_sha256"],
            "ct_terminal_sha256": ct["ct_terminal_sha256"],
            "clearance_policy": profile["clearance_policy"],
            "state": clearance_state,
        }
        clearance = {
            **clearance_core,
            "clearance_terminal_sha256": _task6_sha256(clearance_core),
        }
        probe_core = {
            "schema": "campaignx.first_letters_probe_evidence.v1",
            "candidate_id": candidate["candidate_id"], **binding,
            "clearance_terminal_sha256":
                clearance["clearance_terminal_sha256"],
            "probe_execution_profile_sha256": PROBE_PROFILE_SHA256,
            "state": probe_state, "used_units": used_units,
            "probe_artifact_sha256": probe_sha,
        }
        probe = {
            **probe_core, "probe_evidence_sha256": _task6_sha256(probe_core),
        }
        source_rows = []
        for role, dependency in (
            ("CT_VOLUME", ct_dependency),
            ("M7_PREDICTION_VOLUME", m7_dependency),
        ):
            row_core = {
                "schema": "campaignx.first_letters_contributing_source_row.v1",
                "role": role,
                "artifact_sha256": dependency["artifact_sha256"],
                "read_set_manifest_sha256":
                    dependency["read_set_manifest_sha256"],
                **binding,
            }
            source_rows.append({
                **row_core, "source_row_sha256": _task6_sha256(row_core),
            })
        candidate_outcomes.append({
            "candidate_id": candidate["candidate_id"],
            "ct_terminal": ct, "clearance_terminal": clearance,
            "probe_evidence": probe, "contributing_source_rows": source_rows,
        })
    candidates = _build_artifact_candidates(
        projected_candidates=projected,
        candidate_outcomes=candidate_outcomes,
        profile=profile,
    )
    selection = _derive_selection(
        candidates, profile_file_sha256=profile["profile_file_sha256"],
        run_id=run_authority["run_id"],
    )
    retained_files = [
        {
            "relative_path":
                f"probes/{run_authority['run_id']}/provider-request.json",
            "role": "CANDIDATE_PROVIDER_REQUEST",
            "bytes": _task6_canonical_bytes(provider_request),
        },
        *ct_read_files,
        {
            "relative_path":
                f"probes/{run_authority['run_id']}/provider-response.json",
            "role": "CANDIDATE_PROVIDER_RESPONSE",
            "bytes": provider_response_bytes,
        },
        *probe_files,
        {
            "relative_path":
                f"probes/{run_authority['run_id']}/selection-policy-receipt.json",
            "role": "DISCOVERY_SELECTION_POLICY_RECEIPT",
            "bytes": _task6_canonical_bytes(
                selection["selection_policy_receipt"]
            ),
        },
    ]
    execution_core = {
        "schema": "campaignx.first_letters_discovery_execution_authority.v1",
        **{key: copy.deepcopy(run_authority[key]) for key in (
            "mission_id", "sample_id", "parent_task_id", "parent_attempt_id",
            "worker_id", "run_id", "run_authority_sha256",
            "scientific_opportunity_id", "accepted_p0_artifact_id",
            "accepted_p0_artifact_sha256", "source_snapshot_id",
            "source_snapshot_sha256", "reservation_id", "reservation_sha256",
            "reservation_request_id", "reservation_work_kind",
            "reservation_work_authority_id",
            "reservation_work_authority_sha256", "reservation_source",
            "reservation_ordered_item_ids",
        )},
        "ordered_cell_ids": copy.deepcopy(
            run_authority["reservation_ordered_item_ids"]
        ),
        "allow_unvalidated": False,
    }
    execution_authority = {
        **execution_core,
        "execution_authority_sha256": _task6_sha256(execution_core),
    }
    evidence_set_id = stable_id("first-letters-discovery-evidence-set", {
        "run_authority_sha256": run_authority["run_authority_sha256"],
        "discovery_inputs_sha256": inputs["discovery_inputs_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "terminal_sha256s": [
            row["probe_evidence"]["probe_evidence_sha256"]
            for row in candidate_outcomes
        ],
    })
    return {
        "evidence_set_id": evidence_set_id,
        "execution_authority": execution_authority,
        "profile_bytes": profile_bytes,
        "inputs": inputs,
        "candidate_outcomes": candidate_outcomes,
        "retained_files": retained_files,
        "reservation": copy.deepcopy(reservation),
        "selection": selection,
    }


def _promotion_eligible_discovery_candidate(value: dict[str, Any]) -> bool:
    return (
        value.get("coordinate_admission_state") ==
            "PROMOTABLE_INTEGRAL_COORDINATE_V1"
        and value.get("promotion_coordinate_ct_l0_xyz") ==
            value.get("raw_coordinate_ct_l0_xyz")
        and value.get("promotion_coordinate_sha256") ==
            value.get("raw_coordinate_sha256")
        and (value.get("ct_terminal") or {}).get("state") == "CT_SUPPORTED"
        and (value.get("clearance_terminal") or {}).get("state") ==
            "CLEARANCE_SUPPORTED"
        and (value.get("probe_evidence") or {}).get("state") == "MEASURABLE"
        and isinstance(
            (value.get("probe_evidence") or {}).get("probe_artifact_sha256"), str
        )
    )


def validate_first_letters_discovery_receipt(
    value: Any, *, artifact: dict[str, Any],
    retained_files: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = copy.deepcopy(_closed_dict(
        value, _RECEIPT_FIELDS, "discovery receipt"
    ))
    _false_override(receipt, "discovery receipt")
    _reject_content_informed(receipt, "receipt")
    _reject_credential_value(receipt, "receipt")
    validated_artifact = validate_first_letters_discovery_artifact(
        artifact, retained_files=retained_files
    )
    if (receipt["schema"] != DISCOVERY_RECEIPT_SCHEMA
            or receipt["namespace"] != DISCOVERY_NAMESPACE
            or receipt["canonical_admission"] != "PROHIBITED"
            or receipt["non_claims"] != _DISCOVERY_NON_CLAIMS):
        raise ValueError("discovery receipt identity/namespace is invalid")
    artifact_bindings = {
        "evidence_set_id", "execution_authority_sha256", "run_id",
        "run_authority_sha256", "mission_id", "sample_id",
        "parent_task_id", "parent_attempt_id",
        "scientific_opportunity_id", "mode", "arm_kind",
        "profile_file_sha256", "profile_scientific_core_sha256",
        "discovery_inputs_sha256", "accepted_p0_artifact_id",
        "accepted_p0_artifact_sha256", "source_snapshot_id",
        "source_snapshot_sha256", "source_content_lock_sha256",
        "ct_metadata_sha256", "ct_read_set_manifest_sha256",
        "m7_read_set_manifest_sha256", "provider_request_id",
        "provider_request_sha256", "provider_response_sha256",
        "prediction_identity", "ordered_cell_ids", "cell_id",
        "cell_region_sha256", "grid_spec_sha256",
        "dependency_manifest_sha256",
        "canonical_ordered_cell_set_sha256", "funnel_counts", "artifact_id",
        "artifact_sha256", "file_manifest", "file_manifest_sha256",
        "compute_cap_authority_id", "compute_cap_authority_sha256",
        "reservation_id", "reservation_sha256", "reservation_request_id",
        "reservation_work_kind", "reservation_work_authority_id",
        "reservation_work_authority_sha256", "reservation_source",
        "reservation_ordered_item_ids", "reserved_before_units",
        "reserved_after_units", "reserved_units", "used_units",
        "selection_outcome", "selected_candidate_id",
        "selection_policy_receipt", "selection_policy_receipt_sha256",
        "selection_sha256", "deployed_revision", "namespace",
        "canonical_admission", "non_claims",
    }
    for field in artifact_bindings:
        if receipt[field] != validated_artifact[field]:
            raise ValueError(f"discovery receipt {field} differs from artifact")
    for field in (
        "profile_file_sha256", "profile_scientific_core_sha256",
        "discovery_inputs_sha256", "accepted_p0_artifact_sha256",
        "source_snapshot_sha256", "source_content_lock_sha256",
        "ct_metadata_sha256", "ct_read_set_manifest_sha256",
        "m7_read_set_manifest_sha256", "provider_request_sha256",
        "provider_response_sha256", "canonical_ordered_cell_set_sha256",
        "artifact_sha256", "file_manifest_sha256",
        "compute_cap_authority_sha256", "reservation_sha256",
        "selection_policy_receipt_sha256", "selection_sha256",
        "receipt_sha256",
    ):
        _require_lowercase_sha256(receipt[field], field)
    unit_fields = (
        "reserved_before_units", "reserved_after_units", "reserved_units",
        "used_units",
    )
    if any(isinstance(receipt[field], bool)
           or not isinstance(receipt[field], int) for field in unit_fields):
        raise ValueError("receipt compute units must be integers")
    selection_core = {
        "schema": "campaignx.first_letters_discovery_selection.v1",
        "outcome": receipt["selection_outcome"],
        "selected_candidate_id": receipt["selected_candidate_id"],
        "selection_policy_receipt": receipt["selection_policy_receipt"],
        "selection_policy_receipt_sha256":
            receipt["selection_policy_receipt_sha256"],
        "allow_unvalidated": False,
    }
    if receipt["selection_sha256"] != _task6_sha256(selection_core):
        raise ValueError("receipt selection authority is invalid")
    selected = [
        row for row in validated_artifact["candidates"]
        if row["candidate_id"] == receipt["selected_candidate_id"]
    ]
    if receipt["selection_outcome"] == "DISCOVERY_WINNER_RETAINED":
        if len(selected) != 1:
            raise ValueError("winner receipt does not select one artifact candidate")
        candidate = selected[0]
        if not _promotion_eligible_discovery_candidate(candidate):
            raise ValueError("selected discovery candidate is not promotion eligible")
        expected_selection = {
            "selected_candidate_evidence_sha256":
                candidate["candidate_evidence_sha256"],
            "selected_candidate_raw_coordinate_sha256":
                candidate["raw_coordinate_sha256"],
            "ct_terminal_sha256": candidate["ct_terminal_sha256"],
            "clearance_terminal_sha256": candidate["clearance_terminal_sha256"],
            "probe_evidence_sha256": candidate["probe_evidence_sha256"],
        }
        if any(receipt[field] != expected for field, expected in expected_selection.items()):
            raise ValueError("receipt selected evidence differs from artifact candidate")
    else:
        if receipt["selected_candidate_id"] is not None or any(
            receipt[field] is not None for field in (
                "selected_candidate_evidence_sha256",
                "selected_candidate_raw_coordinate_sha256", "ct_terminal_sha256",
                "clearance_terminal_sha256", "probe_evidence_sha256",
            )
        ):
            raise ValueError("non-winner receipt cannot retain selected evidence")
    if receipt["outcome"] != receipt["selection_outcome"]:
        raise ValueError("receipt outcome differs from selection authority")
    expected_hash = _task6_sha256({
        key: row for key, row in receipt.items() if key != "receipt_sha256"
    })
    if receipt["receipt_sha256"] != expected_hash:
        raise ValueError("discovery receipt hash is invalid")
    return receipt


def _build_first_letters_discovery_artifact_and_receipt_from_registry(
    *, evidence_set_id: str, execution_authority: dict[str, Any], profile_bytes: bytes,
    inputs: dict[str, Any], candidate_outcomes: list[dict[str, Any]],
    retained_files: list[dict[str, Any]], reservation: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = _validate_execution_authority(execution_authority)
    profile = load_first_letters_discovery_profile_bytes(profile_bytes)
    validated_inputs = validate_first_letters_discovery_inputs(inputs)
    reservation = _validate_reservation(
        reservation, authority=authority, profile=profile
    )
    selection = _validate_selection(selection)
    manifest, actual_files = _normalize_retained_files(retained_files)
    request = validated_inputs["provider_request"]
    response = validated_inputs["provider_response"]
    ct_dependency = _dependency_by_role(validated_inputs, "CT_VOLUME")
    m7_dependency = _dependency_by_role(validated_inputs, "M7_PREDICTION_VOLUME")
    profile_input_bindings = {
        "source_snapshot_id": validated_inputs["source_snapshot"]["source_snapshot_id"],
        "source_snapshot_sha256":
            validated_inputs["source_snapshot"]["source_snapshot_sha256"],
        "m7_model_id": request["model_id"],
        "m7_resolution": request["resolution"],
        "m7_level": request["level"],
        "m7_threshold": request["threshold"],
        "coordinate_admission_rule_sha256":
            request["coordinate_admission_rule_sha256"],
    }
    if any(profile[field] != expected for field, expected in profile_input_bindings.items()):
        raise ValueError("discovery profile differs from validated input authority")
    if (profile["ct_metadata_sha256"] != ct_dependency["artifact_sha256"]
            or request["prediction_root_sha256"] !=
                m7_dependency["artifact_sha256"]
            or profile["canonical_ordered_cell_set_sha256"] !=
                _task6_sha256(reservation["ordered_item_ids"])):
        raise ValueError("profile source/read/cell bindings differ from execution")
    request_manifest = _manifest_role(manifest, "CANDIDATE_PROVIDER_REQUEST")
    response_manifest = _manifest_role(manifest, "CANDIDATE_PROVIDER_RESPONSE")
    if (actual_files[request_manifest["relative_path"]] !=
            _task6_canonical_bytes(request)
            or actual_files[response_manifest["relative_path"]] !=
                response["response_bytes"]):
        raise ValueError("retained provider request/response bytes differ from inputs")
    candidates = _build_artifact_candidates(
        projected_candidates=validated_inputs["projected_candidates"],
        candidate_outcomes=candidate_outcomes,
        profile=profile,
    )
    selection = _validate_selection(
        selection, candidates=candidates,
        profile_file_sha256=profile["profile_file_sha256"],
        run_id=authority["run_id"],
    )
    funnel = {
        "raw_candidates": len(candidates),
        "ct_supported_candidates": sum(
            row["ct_terminal"]["state"] == "CT_SUPPORTED" for row in candidates
        ),
        "clearance_supported_candidates": sum(
            row["clearance_terminal"]["state"] == "CLEARANCE_SUPPORTED"
            for row in candidates
        ),
        "probe_measurable_candidates": sum(
            row["probe_evidence"]["state"] == "MEASURABLE" for row in candidates
        ),
    }
    used_units = sum(row["probe_evidence"]["used_units"] for row in candidates)
    selected_ids = {row["candidate_id"] for row in candidates}
    if (selection["outcome"] == "DISCOVERY_WINNER_RETAINED"
            and selection["selected_candidate_id"] not in selected_ids):
        raise ValueError("selection does not resolve to a projected candidate")
    if selection["outcome"] == "DISCOVERY_WINNER_RETAINED":
        selected_candidate = next(
            row for row in candidates
            if row["candidate_id"] == selection["selected_candidate_id"]
        )
        if not _promotion_eligible_discovery_candidate(selected_candidate):
            raise ValueError("selected discovery candidate is not promotion eligible")
    artifact_core = {
        "schema": DISCOVERY_ARTIFACT_SCHEMA,
        "evidence_set_id": evidence_set_id,
        "execution_authority_sha256":
            authority["execution_authority_sha256"],
        "run_id": authority["run_id"],
        "run_authority_sha256": authority["run_authority_sha256"],
        "artifact_id": stable_id("first-letters-discovery-artifact", {
            "execution_authority_sha256": authority["execution_authority_sha256"],
            "discovery_inputs_sha256": validated_inputs["discovery_inputs_sha256"],
            "profile_file_sha256": profile["profile_file_sha256"],
            "file_manifest_sha256": _task6_sha256(manifest),
        }),
        "mission_id": authority["mission_id"],
        "sample_id": authority["sample_id"],
        "parent_task_id": authority["parent_task_id"],
        "parent_attempt_id": authority["parent_attempt_id"],
        "scientific_opportunity_id": authority["scientific_opportunity_id"],
        "mode": profile["mode"],
        "arm_kind": profile["arm_kind"],
        "namespace": DISCOVERY_NAMESPACE,
        "canonical_admission": "PROHIBITED",
        "accepted_p0_artifact_id": authority["accepted_p0_artifact_id"],
        "accepted_p0_artifact_sha256": authority["accepted_p0_artifact_sha256"],
        "profile_file_sha256": profile["profile_file_sha256"],
        "profile_scientific_core_sha256": profile["scientific_core_sha256"],
        "discovery_inputs_sha256": validated_inputs["discovery_inputs_sha256"],
        "source_snapshot_id": profile["source_snapshot_id"],
        "source_snapshot_sha256": profile["source_snapshot_sha256"],
        "source_content_lock_sha256": profile["source_content_lock_sha256"],
        "ct_metadata_sha256": profile["ct_metadata_sha256"],
        "ct_read_set_manifest_sha256": profile["ct_read_set_manifest_sha256"],
        "m7_read_set_manifest_sha256": profile["m7_read_set_manifest_sha256"],
        "m7_prediction_artifact_sha256": m7_dependency["artifact_sha256"],
        "ct_material_policy": profile["ct_material_policy"],
        "clearance_policy": profile["clearance_policy"],
        "coordinate_admission_rule_sha256":
            profile["coordinate_admission_rule_sha256"],
        "provider_request_id": request["request_id"],
        "provider_request_sha256": request_manifest["sha256"],
        "provider_response_sha256": response["response_sha256"],
        "prediction_identity": copy.deepcopy(response["prediction_identity"]),
        "ordered_cell_ids": copy.deepcopy(authority["ordered_cell_ids"]),
        "cell_id": request["cell_id"],
        "cell_region_sha256": request["cell_region_sha256"],
        "grid_spec_sha256": request["grid_spec_sha256"],
        "dependency_manifest_sha256": request["dependency_manifest_sha256"],
        "canonical_ordered_cell_set_sha256":
            profile["canonical_ordered_cell_set_sha256"],
        "file_manifest": manifest,
        "file_manifest_sha256": _task6_sha256(manifest),
        "candidates": candidates,
        "funnel_counts": funnel,
        "compute_cap_authority_id": reservation["cap_authority_id"],
        "compute_cap_authority_sha256": reservation["cap_authority_sha256"],
        "reservation_id": reservation["reservation_id"],
        "reservation_sha256": reservation["reservation_sha256"],
        "reservation_request_id": reservation["request_id"],
        "reservation_work_kind": reservation["work_kind"],
        "reservation_work_authority_id": reservation["work_authority_id"],
        "reservation_work_authority_sha256":
            reservation["work_authority_sha256"],
        "reservation_source": reservation["source"],
        "reservation_ordered_item_ids":
            copy.deepcopy(reservation["ordered_item_ids"]),
        "reserved_before_units": reservation["reserved_before_units"],
        "reserved_after_units": reservation["reserved_after_units"],
        "reserved_units": reservation["reserved_units"],
        "used_units": used_units,
        "selection_outcome": selection["outcome"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "selection_policy_receipt": copy.deepcopy(
            selection["selection_policy_receipt"]
        ),
        "selection_policy_receipt_sha256":
            selection["selection_policy_receipt_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "deployed_revision": profile["deployed_revision"],
        "allow_unvalidated": False,
        "non_claims": copy.deepcopy(_DISCOVERY_NON_CLAIMS),
    }
    artifact = {**artifact_core, "artifact_sha256": _task6_sha256(artifact_core)}
    validate_first_letters_discovery_artifact(
        artifact, retained_files=retained_files
    )
    selected = next(
        (row for row in candidates
         if row["candidate_id"] == selection["selected_candidate_id"]),
        None,
    )
    receipt_core = {
        "schema": DISCOVERY_RECEIPT_SCHEMA,
        "receipt_id": stable_id("first-letters-discovery-receipt", {
            "artifact_sha256": artifact["artifact_sha256"],
            "selection_sha256": selection["selection_sha256"],
        }),
        **{
            field: copy.deepcopy(artifact[field]) for field in (
                "evidence_set_id", "execution_authority_sha256", "run_id",
                "run_authority_sha256", "mission_id", "sample_id",
                "parent_task_id", "parent_attempt_id",
                "scientific_opportunity_id", "mode", "arm_kind",
                "profile_file_sha256", "profile_scientific_core_sha256",
                "discovery_inputs_sha256", "accepted_p0_artifact_id",
                "accepted_p0_artifact_sha256", "source_snapshot_id",
                "source_snapshot_sha256", "source_content_lock_sha256",
                "ct_metadata_sha256", "ct_read_set_manifest_sha256",
                "m7_read_set_manifest_sha256", "provider_request_id",
                "provider_request_sha256", "provider_response_sha256",
                "prediction_identity", "ordered_cell_ids", "cell_id",
                "cell_region_sha256", "grid_spec_sha256",
                "dependency_manifest_sha256",
                "canonical_ordered_cell_set_sha256", "funnel_counts",
                "artifact_id", "artifact_sha256", "file_manifest",
                "file_manifest_sha256", "compute_cap_authority_id",
                "compute_cap_authority_sha256", "reservation_id",
                "reservation_sha256", "reservation_request_id",
                "reservation_work_kind", "reservation_work_authority_id",
                "reservation_work_authority_sha256", "reservation_source",
                "reservation_ordered_item_ids", "reserved_before_units",
                "reserved_after_units", "reserved_units", "used_units",
                "selection_sha256", "deployed_revision", "namespace",
                "canonical_admission", "non_claims",
            )
        },
        "selection_outcome": selection["outcome"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_candidate_evidence_sha256": (
            selected["candidate_evidence_sha256"] if selected else None
        ),
        "selected_candidate_raw_coordinate_sha256": (
            selected["raw_coordinate_sha256"] if selected else None
        ),
        "ct_terminal_sha256": selected["ct_terminal_sha256"] if selected else None,
        "clearance_terminal_sha256": (
            selected["clearance_terminal_sha256"] if selected else None
        ),
        "probe_evidence_sha256": (
            selected["probe_evidence_sha256"] if selected else None
        ),
        "selection_policy_receipt_sha256":
            selection["selection_policy_receipt_sha256"],
        "selection_policy_receipt": copy.deepcopy(
            selection["selection_policy_receipt"]
        ),
        "allow_unvalidated": False,
        "outcome": selection["outcome"],
    }
    receipt = {**receipt_core, "receipt_sha256": _task6_sha256(receipt_core)}
    validate_first_letters_discovery_receipt(
        receipt, artifact=artifact, retained_files=retained_files
    )
    return artifact, receipt


def _build_first_letters_discovery_artifact_and_receipt_from_evidence_set(
    registered: Any, *, evidence_set_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure assembler used only after a concrete store transaction readback."""

    registered = _closed_dict(registered, {
        "evidence_set_id", "execution_authority", "profile_bytes", "inputs",
        "candidate_outcomes", "retained_files", "reservation", "selection",
    }, "registered discovery evidence set")
    if registered["evidence_set_id"] != evidence_set_id:
        raise ValueError("registered discovery evidence-set identity drift")
    return _build_first_letters_discovery_artifact_and_receipt_from_registry(
        evidence_set_id=evidence_set_id,
        execution_authority=registered["execution_authority"],
        profile_bytes=registered["profile_bytes"],
        inputs=registered["inputs"],
        candidate_outcomes=registered["candidate_outcomes"],
        retained_files=registered["retained_files"],
        reservation=registered["reservation"],
        selection=registered["selection"],
    )


def _resolve_discovery_promotion_evidence_from_evidence_set(
    registered: dict[str, Any], *, evidence_set_id: str,
) -> dict[str, Any]:
    artifact, receipt = (
        _build_first_letters_discovery_artifact_and_receipt_from_evidence_set(
            registered, evidence_set_id=evidence_set_id,
        )
    )
    validated_receipt = validate_first_letters_discovery_receipt(
        receipt, artifact=artifact,
        retained_files=registered["retained_files"],
    )
    if validated_receipt["selection_outcome"] != "DISCOVERY_WINNER_RETAINED":
        raise ValueError("discovery receipt has no promotion candidate")
    selected = [
        row for row in artifact["candidates"]
        if row["candidate_id"] == validated_receipt["selected_candidate_id"]
    ]
    if len(selected) != 1:
        raise ValueError("discovery promotion evidence is ambiguous")
    return {
        "receipt": validated_receipt,
        "selected_candidate": copy.deepcopy(selected[0]),
    }


_ARM_FIELDS = {
    "schema", "arm_id", "mission_id", "accepted_p0_id", "accepted_p0_sha256",
    "source_snapshot_id", "source_snapshot_sha256", "source_content_lock_sha256",
    "ct_metadata_sha256", "ct_read_set_manifest_sha256", "m7_metadata_sha256",
    "m7_read_set_manifest_sha256", "m7_model_id", "m7_resolution", "m7_level",
    "m7_transform_sha256", "m7_threshold", "discovery_policy_id",
    "discovery_profile_sha256", "deployed_revision", "preflight_private_sha256",
    "preflight_sanitized_sha256", "ordered_cell_ids", "ordered_cell_set_sha256",
    "mission_compute_cap_authority_id", "mission_compute_cap_authority_sha256",
    "requested_units", "active_policy_chain_sha256", "may_update_accepted_p0",
    "statistical_budget_delta", "allow_unvalidated", "admission_sha256",
}


def validate_experimental_arm_admission(value: Any) -> dict[str, Any]:
    result = copy.deepcopy(_closed_dict(value, _ARM_FIELDS, "experimental arm admission"))
    _false_override(result, "experimental arm admission")
    if (result["schema"] != "campaignx.first_letters_experimental_arm_admission.v1"
            or result["may_update_accepted_p0"] is not False
            or result["statistical_budget_delta"] != 0):
        raise ValueError("experimental arm authority is unsafe")
    if (not isinstance(result["ordered_cell_ids"], list)
            or result["ordered_cell_ids"] != list(dict.fromkeys(result["ordered_cell_ids"]))
            or result["ordered_cell_set_sha256"] != _task6_sha256(result["ordered_cell_ids"])):
        raise ValueError("experimental arm ordered cohort is invalid")
    for field in (name for name in result if name.endswith("sha256")):
        _require_lowercase_sha256(result[field], field)
    expected = _task6_sha256({key: row for key, row in result.items() if key != "admission_sha256"})
    if result["admission_sha256"] != expected:
        raise ValueError("experimental arm admission hash is invalid")
    return result


def compare_experimental_arms(
    arms: Any, *, ordered_cell_ids: list[str],
) -> dict[str, Any]:
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("at least two arms are required")
    membership: dict[str, set[str]] = {}
    survival: dict[str, dict[str, int]] = {}
    for arm in arms:
        if not isinstance(arm, dict) or set(arm) != {"arm_id", "cells"}:
            raise ValueError("arm comparison rows are closed")
        cells = arm["cells"]
        if [row.get("cell_id") for row in cells] != ordered_cell_ids:
            raise ValueError("arm cell membership/order differs from preregistration")
        if any(row.get("status") not in {
            "PRESENT", "MISSING_WITH_PREREGISTERED_REASON",
            "EXCLUDED_WITH_PREREGISTERED_REASON",
        } for row in cells):
            raise ValueError("arm row status is unsupported")
        candidates = [candidate for row in cells for candidate in row.get("candidates", [])]
        ids = {str(candidate["candidate_id"]) for candidate in candidates}
        membership[str(arm["arm_id"])] = ids
        survival[str(arm["arm_id"])] = {
            "raw": len(candidates),
            "post_ct": sum(candidate.get("ct_retained") is True for candidate in candidates),
            "post_clearance": sum(candidate.get("clearance_retained") is True for candidate in candidates),
            "measurable": sum(candidate.get("probe_measurable") is True for candidate in candidates),
        }
    all_sets = list(membership.values())
    union = set().union(*all_sets)
    intersection = set.intersection(*all_sets)
    return {
        "schema": "campaignx.first_letters_experimental_arm_comparison.v1",
        "union": sorted(union),
        "intersection": sorted(intersection),
        "unique": {
            arm_id: sorted(values - set().union(*[
                other for other_id, other in membership.items() if other_id != arm_id
            ]))
            for arm_id, values in membership.items()
        },
        "survival": survival,
    }


_BENCHMARK_V2_LOCK_FIELDS = (
    "sample_id", "cell_id", "cell_order", "coordinate_frame",
    "grid_spec_sha256", "source_snapshot_id", "source_snapshot_sha256",
    "source_content_lock_sha256", "ct_metadata_sha256",
    "ct_read_set_manifest_sha256", "m7_metadata_sha256",
    "m7_read_set_manifest_sha256", "model_id", "provider_id", "resolution",
    "level", "transform_sha256", "threshold",
    "discovery_profile_file_sha256", "discovery_scientific_core_sha256",
    "policy_version", "deployed_revision",
)
_BENCHMARK_V2_ROW_FIELDS = set(_BENCHMARK_V2_LOCK_FIELDS) | {
    "baseline_result_sha256", "select_result_sha256", "baseline_status",
    "select_status", "baseline_lock_sha256", "select_lock_sha256",
}


def validate_seed_probe_benchmark_execution_manifest_v2(
    value: Any,
) -> dict[str, Any]:
    manifest = copy.deepcopy(_closed_dict(value, {
        "schema", "benchmark_id", "execution_scope", "deployed_revision",
        "ordered_cohort", "ordered_cohort_sha256", "arms", "manifest_sha256",
    }, "seed-probe benchmark execution manifest v2"))
    if (manifest["schema"] != "campaignx.seed_probe_benchmark_execution_manifest.v2"
            or manifest["execution_scope"] != "ISOLATED_NONPRODUCTION"
            or re.fullmatch(r"[0-9a-f]{40}", str(manifest["deployed_revision"])) is None):
        raise ValueError("benchmark-v2 execution scope/revision is invalid")
    arms = _closed_dict(manifest["arms"], {"baseline", "select"}, "benchmark arms")
    for name, mode in (("baseline", "off"), ("select", "select")):
        arm = _closed_dict(arms[name], {"mode", "profile_sha256"}, f"benchmark {name} arm")
        if arm["mode"] != mode:
            raise ValueError("benchmark v2 arms are not paired off/select")
        _require_lowercase_sha256(arm["profile_sha256"], f"{name} profile")
    cohort = manifest["ordered_cohort"]
    if not isinstance(cohort, list) or not cohort:
        raise ValueError("benchmark v2 has no complete cohort")
    identities: list[tuple[str, str]] = []
    for index, row_value in enumerate(cohort):
        row = _closed_dict(row_value, _BENCHMARK_V2_ROW_FIELDS, "benchmark cohort row")
        if row["cell_order"] != index:
            raise ValueError("benchmark cohort order is not canonical")
        if row["coordinate_frame"] != "ct_l0_xyz":
            raise ValueError("benchmark coordinate frame is unsupported")
        if row["deployed_revision"] != manifest["deployed_revision"]:
            raise ValueError("benchmark cohort deployed revision drift")
        if row["baseline_status"] not in {"PRESENT", "MISSING", "EXCLUDED"} or row["select_status"] not in {"PRESENT", "MISSING", "EXCLUDED"}:
            raise ValueError("benchmark row missing/excluded status is invalid")
        for field in (
            "grid_spec_sha256", "source_snapshot_sha256", "source_content_lock_sha256",
            "ct_metadata_sha256", "ct_read_set_manifest_sha256", "m7_metadata_sha256",
            "m7_read_set_manifest_sha256", "transform_sha256",
            "discovery_profile_file_sha256", "discovery_scientific_core_sha256",
            "baseline_result_sha256", "select_result_sha256", "baseline_lock_sha256",
            "select_lock_sha256",
        ):
            _require_lowercase_sha256(row[field], field)
        lock = {field: row[field] for field in _BENCHMARK_V2_LOCK_FIELDS}
        if (row["baseline_lock_sha256"] != _task6_sha256({"arm": "baseline", **lock})
                or row["select_lock_sha256"] != _task6_sha256({"arm": "select", **lock})):
            raise ValueError("benchmark paired-arm common lock drift")
        identity = (str(row["sample_id"]), str(row["cell_id"]))
        if identity in identities:
            raise ValueError("benchmark cohort contains duplicate membership")
        identities.append(identity)
    if manifest["ordered_cohort_sha256"] != _task6_sha256(cohort):
        raise ValueError("benchmark ordered cohort hash is invalid")
    expected = _task6_sha256({
        key: row for key, row in manifest.items() if key != "manifest_sha256"
    })
    if manifest["manifest_sha256"] != expected:
        raise ValueError("benchmark execution-manifest hash is invalid")
    return manifest


def validate_seed_probe_benchmark_receipt_v2(
    value: Any, *, execution_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = validate_seed_probe_benchmark_execution_manifest_v2(execution_manifest)
    decision = copy.deepcopy(_closed_dict(value, {
        "schema", "benchmark_id", "status", "execution_scope",
        "execution_manifest_sha256", "spec_sha256", "results_sha256",
        "deployed_revision", "ordered_cohort_sha256", "checks", "decision_sha256",
    }, "seed-probe benchmark decision v2"))
    if (decision["schema"] != "campaignx.seed_probe_benchmark_decision.v2"
            or decision["status"] != "APPROVED_SELECT"
            or decision["execution_scope"] != "ISOLATED_NONPRODUCTION"
            or decision["benchmark_id"] != manifest["benchmark_id"]
            or decision["execution_manifest_sha256"] != manifest["manifest_sha256"]
            or decision["deployed_revision"] != manifest["deployed_revision"]
            or decision["ordered_cohort_sha256"] != manifest["ordered_cohort_sha256"]):
        raise ValueError("benchmark v2 decision does not bind the execution manifest")
    checks = decision["checks"]
    required = {"COHORT_COMPLETE", "SOURCE_LOCKS_MATCH", "PAIRED_RESULTS_COMPLETE"}
    if (not isinstance(checks, list)
            or {row.get("check_id") for row in checks} != required
            or any(set(row) != {"check_id", "status"} or row["status"] != "PASS" for row in checks)):
        raise ValueError("benchmark v2 required checks did not pass")
    for field in ("execution_manifest_sha256", "spec_sha256", "results_sha256", "ordered_cohort_sha256", "decision_sha256"):
        _require_lowercase_sha256(decision[field], field)
    if decision["decision_sha256"] != _task6_sha256({
        key: row for key, row in decision.items() if key != "decision_sha256"
    }):
        raise ValueError("benchmark v2 decision hash is invalid")
    core = {
        "schema": "campaignx.seed_probe_benchmark_authorization.v2",
        "benchmark_id": manifest["benchmark_id"],
        "execution_manifest": manifest,
        "decision": decision,
        "ordered_cohort": copy.deepcopy(manifest["ordered_cohort"]),
        "deployed_revision": manifest["deployed_revision"],
    }
    return {**core, "authorization_sha256": _task6_sha256(core)}


def validate_benchmark_authorization_for_promotion(
    authorization: Any, *, task: dict[str, Any], profile: dict[str, Any],
    source: dict[str, Any], deployed_revision: str,
) -> dict[str, Any]:
    value = copy.deepcopy(_closed_dict(authorization, {
        "schema", "benchmark_id", "execution_manifest", "decision",
        "ordered_cohort", "deployed_revision", "authorization_sha256",
    }, "seed-probe benchmark promotion authorization"))
    if value["schema"] != "campaignx.seed_probe_benchmark_authorization.v2":
        raise ValueError("Task 6 promotion requires benchmark authorization v2")
    revalidated = validate_seed_probe_benchmark_receipt_v2(
        value["decision"], execution_manifest=value["execution_manifest"]
    )
    if value != revalidated:
        raise ValueError("benchmark authorization retained bytes/hash drift")
    if value["deployed_revision"] != deployed_revision:
        raise ValueError("benchmark authorization deployed revision drift")
    matches = [
        row for row in value["ordered_cohort"]
        if row["sample_id"] == task.get("sample_id")
        and row["cell_id"] == task.get("cell_id")
    ]
    if len(matches) != 1:
        raise ValueError("promotion task is outside the benchmark cohort")
    row = matches[0]
    if (row["source_snapshot_id"] != source.get("source_snapshot_id")
            or row["source_snapshot_sha256"] != source.get("source_snapshot_sha256")
            or row["discovery_profile_file_sha256"] != profile.get("profile_file_sha256")
            or row["policy_version"] != profile.get("discovery_policy_id")):
        raise ValueError("promotion source/profile differs from benchmark locks")
    return value


_ADAPTIVE_CT_STATES = frozenset({
    "CT_RETAINED", "CT_REJECTED_NO_NEARBY_MATERIAL",
    "CT_NOT_RUN_NONINTEGRAL_COORDINATE", "CT_NOT_RUN_MALFORMED_COORDINATE",
    "CT_INCOMPLETE_PLATFORM_OR_SOURCE",
})
_ADAPTIVE_CLEARANCE_STATES = frozenset({
    "CLEARANCE_PASSED", "CLEARANCE_REJECTED_CELL_INTERIOR",
    "CLEARANCE_REJECTED_VOLUME_INTERIOR", "CLEARANCE_NOT_RUN_DUE_TO_CT",
    "CLEARANCE_INCOMPLETE",
})
_ADAPTIVE_PROBE_STATES = frozenset({
    "PROBE_MEASURABLE_NONCANONICAL_GEOMETRY", "PROBE_NOT_MEASURABLE",
    "PROBE_NOT_RUN_DUE_TO_UPSTREAM", "PROBE_INCOMPLETE",
})


def reconcile_adaptive_causal_receipt(value: Any) -> dict[str, Any]:
    required = {
        "schema", "parent_task_id", "parent_attempt_id",
        "scientific_opportunity_id", "reason", "raw_candidate_count",
        "candidates", "namespace", "allow_unvalidated", "receipt_sha256",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") !=
                "campaignx.first_letters_discovery_causal_receipt.v1"
            or value.get("namespace") != DISCOVERY_NAMESPACE
            or value.get("allow_unvalidated") is not False
            or value.get("receipt_sha256") != _task6_sha256({
                key: row for key, row in value.items() if key != "receipt_sha256"
            })):
        raise ValueError("CONTROL_INCOMPLETE: malformed adaptive causal receipt")
    candidates = value["candidates"]
    raw_count = value["raw_candidate_count"]
    if (isinstance(raw_count, bool) or not isinstance(raw_count, int)
            or raw_count < 0 or not isinstance(candidates, list)
            or raw_count != len(candidates)):
        raise ValueError("CONTROL_INCOMPLETE: adaptive candidate count disagreement")
    ids: list[str] = []
    pairs: list[tuple[str, str]] = []
    measurable_count = 0
    incomplete = False
    row_hashes: list[str] = []
    for candidate in candidates:
        if (not isinstance(candidate, dict)
                or set(candidate) != {
                    "candidate_id", "ct_terminal", "clearance_terminal",
                    "probe_terminal",
                }):
            raise ValueError("CONTROL_INCOMPLETE: malformed adaptive candidate row")
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in ids:
            raise ValueError("CONTROL_INCOMPLETE: duplicate adaptive candidate")
        ids.append(candidate_id)
        ct = candidate["ct_terminal"]
        clearance = candidate["clearance_terminal"]
        probe = candidate["probe_terminal"]
        if (not isinstance(ct, dict) or not isinstance(clearance, dict)
                or not isinstance(probe, dict)
                or ct.get("candidate_id") != candidate_id
                or clearance.get("candidate_id") != candidate_id
                or probe.get("candidate_id") != candidate_id
                or ct.get("status") not in _ADAPTIVE_CT_STATES
                or clearance.get("status") not in _ADAPTIVE_CLEARANCE_STATES
                or probe.get("status") not in _ADAPTIVE_PROBE_STATES):
            raise ValueError("CONTROL_INCOMPLETE: malformed adaptive terminal row")
        for row in (ct, clearance, probe):
            _require_lowercase_sha256(row.get("row_sha256"), "adaptive terminal row")
            row_hashes.append(row["row_sha256"])
        ct_state = ct["status"]
        clearance_state = clearance["status"]
        probe_state = probe["status"]
        if ct_state == "CT_RETAINED":
            valid_clearance = {
                "CLEARANCE_PASSED", "CLEARANCE_REJECTED_CELL_INTERIOR",
                "CLEARANCE_REJECTED_VOLUME_INTERIOR",
            }
            if clearance_state not in valid_clearance:
                raise ValueError("CONTROL_INCOMPLETE: impossible CT/clearance pair")
        elif ct_state == "CT_INCOMPLETE_PLATFORM_OR_SOURCE":
            if clearance_state != "CLEARANCE_INCOMPLETE":
                raise ValueError("CONTROL_INCOMPLETE: incomplete CT pairing")
            incomplete = True
        elif clearance_state != "CLEARANCE_NOT_RUN_DUE_TO_CT":
            raise ValueError("CONTROL_INCOMPLETE: CT rejection requires clearance not-run")
        pairs.append((ct_state, clearance_state))
        if value["reason"] == "MEASURABLE_NONCANONICAL_PROBE_GEOMETRY":
            if clearance_state == "CLEARANCE_PASSED":
                if probe_state not in {
                    "PROBE_MEASURABLE_NONCANONICAL_GEOMETRY",
                    "PROBE_NOT_MEASURABLE",
                }:
                    incomplete = True
                if probe_state == "PROBE_MEASURABLE_NONCANONICAL_GEOMETRY":
                    if (probe.get("namespace") != DISCOVERY_NAMESPACE
                            or not isinstance(probe.get("measurement_manifest_sha256"), str)
                            or len(probe["measurement_manifest_sha256"]) != 64):
                        incomplete = True
                    else:
                        measurable_count += 1
            elif probe_state != "PROBE_NOT_RUN_DUE_TO_UPSTREAM":
                incomplete = True
            if probe_state == "PROBE_INCOMPLETE":
                incomplete = True
    cause_vector: str | None = None
    eligible = False
    reason = value["reason"]
    if reason == "RAW_CANDIDATES_FAILED_CT_OR_CLEARANCE" and raw_count > 0:
        ct_rejected = (
            "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_NOT_RUN_DUE_TO_CT"
        )
        clearance_rejected = {
            ("CT_RETAINED", "CLEARANCE_REJECTED_CELL_INTERIOR"),
            ("CT_RETAINED", "CLEARANCE_REJECTED_VOLUME_INTERIOR"),
        }
        if all(pair == ct_rejected for pair in pairs):
            cause_vector, eligible = "ALL_CT_REJECTED", True
        elif all(pair in clearance_rejected for pair in pairs):
            cause_vector, eligible = "ALL_CLEARANCE_REJECTED", True
        elif (all(pair == ct_rejected or pair in clearance_rejected for pair in pairs)
              and any(pair == ct_rejected for pair in pairs)
              and any(pair in clearance_rejected for pair in pairs)):
            cause_vector, eligible = "MIXED_CT_AND_CLEARANCE_REJECTED", True
    elif reason == "MEASURABLE_NONCANONICAL_PROBE_GEOMETRY":
        cause_vector = "MEASURABLE_NONCANONICAL_PROBE_GEOMETRY"
        eligible = measurable_count > 0 and not incomplete
    result = {
        "schema": "campaignx.first_letters_discovery_adaptive_reconciliation.v1",
        "parent_task_id": value["parent_task_id"],
        "parent_attempt_id": value["parent_attempt_id"],
        "scientific_opportunity_id": value["scientific_opportunity_id"],
        "reason": reason,
        "cause_vector_id": cause_vector,
        "eligible": eligible,
        "candidate_ids": ids,
        "terminal_row_set_sha256": _task6_sha256(row_hashes),
        "allow_unvalidated": False,
    }
    result["reconciliation_sha256"] = _task6_sha256(result)
    return result


def validate_adaptive_profile_v1(value: Any) -> dict[str, Any]:
    required = {
        "top_k", "probe_generations", "maximum_attempts_per_candidate",
        "probe_profile_sha256", "allow_unvalidated",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value.get("top_k") != 2 or value.get("probe_generations") != 12
            or value.get("maximum_attempts_per_candidate") != 1
            or value.get("probe_profile_sha256") != PROBE_PROFILE_SHA256
            or value.get("allow_unvalidated") is not False):
        raise ValueError("ADAPTIVE_PROFILE_UNSUPPORTED_V1")
    return copy.deepcopy(value)


def adaptive_item_prefix_count(
    *, cap_units: int, committed_units: int, available_neighbor_count: int,
) -> int:
    for name, value in (
        ("cap_units", cap_units), ("committed_units", committed_units),
        ("available_neighbor_count", available_neighbor_count),
    ):
        if (isinstance(value, bool) or not isinstance(value, int)
                or not 0 <= value <= 2**63 - 1):
            raise ValueError(f"adaptive {name} must be a checked nonnegative integer")
    remaining = max(0, cap_units - committed_units)
    return min(8, remaining // 24, available_neighbor_count)


def build_adaptive_proposal_v1(
    *, parent_reconciliation: dict[str, Any], grid_spec: dict[str, Any],
    parent_cell_id: str, profile: dict[str, Any], cap_units: int,
    committed_units: int,
) -> dict[str, Any]:
    from .generator import canonical_grid_neighbors

    validate_adaptive_profile_v1(profile)
    if (parent_reconciliation.get("schema") !=
            "campaignx.first_letters_discovery_adaptive_reconciliation.v1"
            or parent_reconciliation.get("eligible") is not True
            or parent_reconciliation.get("reconciliation_sha256") !=
                _task6_sha256({
                    key: row for key, row in parent_reconciliation.items()
                    if key != "reconciliation_sha256"
                })):
        raise ValueError("adaptive parent is not eligible")
    neighbors = canonical_grid_neighbors(grid_spec, parent_cell_id)
    selected_count = adaptive_item_prefix_count(
        cap_units=cap_units, committed_units=committed_units,
        available_neighbor_count=len(neighbors),
    )
    core = {
        "schema": "campaignx.first_letters_discovery_adaptive.v1",
        "parent_task_id": parent_reconciliation["parent_task_id"],
        "parent_attempt_id": parent_reconciliation["parent_attempt_id"],
        "scientific_opportunity_id": parent_reconciliation["scientific_opportunity_id"],
        "parent_reconciliation_sha256": parent_reconciliation["reconciliation_sha256"],
        "cause_vector_id": parent_reconciliation["cause_vector_id"],
        "generation": 1, "parent_cell_id": parent_cell_id,
        "grid_spec_sha256": _task6_sha256(grid_spec),
        "complete_neighbor_universe": neighbors,
        "complete_neighbor_universe_sha256": _task6_sha256(neighbors),
        "selected_neighbor_ids": neighbors[:selected_count],
        "units_per_cell": 24, "reserved_units": selected_count * 24,
        "statistical_budget_delta": 0, "allow_unvalidated": False,
    }
    return {**core, "adaptive_sha256": _task6_sha256(core)}


def normalize_seed_probe_benchmark_authorization(
    value: Any,
) -> dict[str, Any]:
    """Validate the compact approval bound into a select task.

    The task intentionally carries no local receipt path.  It carries the
    benchmark identity and canonical decision digest so queued work remains
    portable and auditable after the bootstrap host disappears.
    """

    expected = {
        "schema",
        "benchmark_id",
        "decision_receipt_sha256",
        "spec_sha256",
        "results_sha256",
        "paired_cell_count",
        "scroll_count",
        "authorized_sample_ids",
        "execution_scope",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark authorization differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported seed probe benchmark authorization schema")
    benchmark_id = value["benchmark_id"]
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("benchmark authorization benchmark_id must be non-empty")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "benchmark authorization must come from ISOLATED_NONPRODUCTION"
        )
    for field in (
        "decision_receipt_sha256",
        "spec_sha256",
        "results_sha256",
    ):
        _require_lowercase_sha256(value[field], field)
    for field, minimum, maximum in (
        ("paired_cell_count", 40, 60),
        ("scroll_count", 3, None),
    ):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"benchmark authorization {field} must be an integer")
        if raw < minimum or (maximum is not None and raw > maximum):
            bounds = (
                f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
            )
            raise ValueError(
                f"benchmark authorization {field} must be {bounds}"
            )
    authorized_sample_ids = value["authorized_sample_ids"]
    if (
        not isinstance(authorized_sample_ids, list)
        or any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in authorized_sample_ids
        )
        or authorized_sample_ids != sorted(set(authorized_sample_ids))
        or len(authorized_sample_ids) != value["scroll_count"]
    ):
        raise ValueError(
            "benchmark authorization authorized_sample_ids must be the "
            "sorted unique approved scroll cohort"
        )
    return copy.deepcopy(value)


def normalize_seed_probe_benchmark_execution_authorization(
    value: Any,
) -> dict[str, Any]:
    """Validate a pre-result authority for only the frozen causal cohort."""

    expected = {
        "schema",
        "benchmark_id",
        "benchmark_spec_sha256",
        "execution_scope",
        "arm",
        "baseline_policy_version",
        "policy_version",
        "planner",
        "seed_probe_mode",
        "cells",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark execution authorization differs from v1 "
            f"contract: expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA:
        raise ValueError(
            "unsupported seed probe benchmark execution authorization schema"
        )
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "benchmark execution authorization must be ISOLATED_NONPRODUCTION"
        )
    if (
        value["arm"] != "closed_loop"
        or value["planner"] != "deterministic-v2"
        or value["seed_probe_mode"] != "select"
    ):
        raise ValueError(
            "benchmark execution authorization must name the "
            "deterministic-v2/select closed_loop arm"
        )
    for field in (
        "benchmark_id",
        "baseline_policy_version",
        "policy_version",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"benchmark execution {field} must be non-empty")
    if value["baseline_policy_version"] == value["policy_version"]:
        raise ValueError(
            "benchmark execution arm policy versions must be distinct"
        )
    _require_lowercase_sha256(
        value["benchmark_spec_sha256"], "benchmark_spec_sha256"
    )
    cells = value["cells"]
    if not isinstance(cells, list) or not 40 <= len(cells) <= 60:
        raise ValueError(
            "benchmark execution authorization must bind 40..60 cells"
        )
    seen: set[tuple[str, str]] = set()
    seen_blocks: set[str] = set()
    normalized_cells: list[dict[str, str]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or set(cell) != {
            "cell_id",
            "sample_id",
            "independence_block_id",
        }:
            raise ValueError(
                f"benchmark execution cells[{index}] differs from v1 contract"
            )
        if any(
            not isinstance(cell[field], str) or not cell[field]
            for field in (
                "cell_id",
                "sample_id",
                "independence_block_id",
            )
        ):
            raise ValueError(
                f"benchmark execution cells[{index}] identity is incomplete"
            )
        identity = (cell["sample_id"], cell["cell_id"])
        if identity in seen:
            raise ValueError(
                "benchmark execution authorization contains duplicate cells"
            )
        block = cell["independence_block_id"]
        if block in seen_blocks:
            raise ValueError(
                "benchmark execution authorization contains duplicate "
                "independence blocks"
            )
        seen.add(identity)
        seen_blocks.add(block)
        normalized_cells.append(dict(cell))
    if len({cell["sample_id"] for cell in normalized_cells}) < 3:
        raise ValueError(
            "benchmark execution authorization must bind at least three scrolls"
        )
    return {**copy.deepcopy(value), "cells": normalized_cells}


def validate_seed_probe_benchmark_spec(value: Any) -> dict[str, Any]:
    """Validate a frozen causal spec before its select arm has any results."""

    expected = {
        "schema",
        "benchmark_id",
        "frozen_at_utc",
        "execution_scope",
        "baseline",
        "closed_loop",
        "minimum_cells",
        "maximum_cells",
        "minimum_scrolls",
        "minimum_relative_yield_improvement",
        "maximum_relative_reviewer_rate_regression",
        "maximum_incremental_compute_wall_hours_per_cell",
        "maximum_new_incorrect_lamina_rate_upper_bound",
        "paired_test_alpha",
        "minimum_pairs_per_scroll",
        "review_protocol_id",
        "cells",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark spec differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_SPEC_SCHEMA:
        raise ValueError("unsupported seed probe benchmark spec schema")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError("benchmark spec must be ISOLATED_NONPRODUCTION")
    if not isinstance(value["benchmark_id"], str) or not value[
        "benchmark_id"
    ].strip():
        raise ValueError("benchmark spec benchmark_id must be non-empty")
    frozen_at = value["frozen_at_utc"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("Z"):
        raise ValueError("benchmark spec frozen_at_utc must be UTC")
    try:
        parsed_at = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("benchmark spec frozen_at_utc is invalid") from error
    if parsed_at.utcoffset() != timezone.utc.utcoffset(parsed_at):
        raise ValueError("benchmark spec frozen_at_utc must be UTC")

    expected_arm_keys = {"policy_version", "planner", "seed_probe_mode"}
    for name in ("baseline", "closed_loop"):
        arm = value[name]
        if not isinstance(arm, dict) or set(arm) != expected_arm_keys:
            raise ValueError(f"benchmark spec {name} arm differs from v1 contract")
        if any(not isinstance(arm[field], str) or not arm[field] for field in arm):
            raise ValueError(f"benchmark spec {name} arm identity is incomplete")
    if (
        value["baseline"]["planner"] != "deterministic-v2"
        or value["baseline"]["seed_probe_mode"] != "off"
        or value["closed_loop"]["planner"] != "deterministic-v2"
        or value["closed_loop"]["seed_probe_mode"] != "select"
        or value["baseline"]["policy_version"]
        == value["closed_loop"]["policy_version"]
    ):
        raise ValueError(
            "benchmark spec must compare distinct deterministic-v2/off and "
            "deterministic-v2/select arms"
        )
    for field, minimum, maximum in (
        ("minimum_cells", 40, 60),
        ("maximum_cells", 40, 60),
        ("minimum_scrolls", 3, None),
    ):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"benchmark spec {field} must be an integer")
        if raw < minimum or (maximum is not None and raw > maximum):
            raise ValueError(f"benchmark spec {field} is outside v1 bounds")
    if value["minimum_cells"] > value["maximum_cells"]:
        raise ValueError("benchmark spec cell bounds are inverted")
    for field in (
        "minimum_relative_yield_improvement",
        "maximum_relative_reviewer_rate_regression",
        "maximum_incremental_compute_wall_hours_per_cell",
        "maximum_new_incorrect_lamina_rate_upper_bound",
    ):
        raw = value[field]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise ValueError(f"benchmark spec {field} must be non-negative")
    safety_upper_bound = float(
        value["maximum_new_incorrect_lamina_rate_upper_bound"]
    )
    if not 0.0 < safety_upper_bound <= 1.0:
        raise ValueError(
            "benchmark spec "
            "maximum_new_incorrect_lamina_rate_upper_bound must be in (0, 1]"
        )
    paired_alpha = value["paired_test_alpha"]
    if (
        isinstance(paired_alpha, bool)
        or not isinstance(paired_alpha, (int, float))
        or not math.isfinite(float(paired_alpha))
        or not 0.0 < float(paired_alpha) <= 0.10
    ):
        raise ValueError(
            "benchmark spec paired_test_alpha must be in (0, 0.10]"
        )
    minimum_pairs = value["minimum_pairs_per_scroll"]
    if (
        isinstance(minimum_pairs, bool)
        or not isinstance(minimum_pairs, int)
        or minimum_pairs < 5
    ):
        raise ValueError(
            "benchmark spec minimum_pairs_per_scroll must be an integer >= 5"
        )
    if not isinstance(value["review_protocol_id"], str) or not value[
        "review_protocol_id"
    ]:
        raise ValueError("benchmark spec review_protocol_id must be non-empty")
    cells = value["cells"]
    if (
        not isinstance(cells, list)
        or not value["minimum_cells"] <= len(cells) <= value["maximum_cells"]
    ):
        raise ValueError("benchmark spec cohort is outside its frozen bounds")

    authorization = {
        "schema": BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA,
        "benchmark_id": value["benchmark_id"],
        "benchmark_spec_sha256": content_sha256(value),
        "execution_scope": value["execution_scope"],
        "arm": "closed_loop",
        "baseline_policy_version": value["baseline"]["policy_version"],
        "policy_version": value["closed_loop"]["policy_version"],
        "planner": value["closed_loop"]["planner"],
        "seed_probe_mode": value["closed_loop"]["seed_probe_mode"],
        "cells": cells,
    }
    normalized = normalize_seed_probe_benchmark_execution_authorization(
        authorization
    )
    if len({cell["sample_id"] for cell in normalized["cells"]}) < value[
        "minimum_scrolls"
    ]:
        raise ValueError("benchmark spec cohort has too few scrolls")
    pairs_by_scroll: dict[str, int] = {}
    for cell in normalized["cells"]:
        pairs_by_scroll[cell["sample_id"]] = (
            pairs_by_scroll.get(cell["sample_id"], 0) + 1
        )
    if any(
        count < value["minimum_pairs_per_scroll"]
        for count in pairs_by_scroll.values()
    ):
        raise ValueError(
            "benchmark spec cohort has too few pairs on at least one scroll"
        )
    return normalized


def load_seed_probe_benchmark_spec(path: Path) -> dict[str, Any]:
    """Load a preregistered isolated spec for its causal select arm."""

    try:
        value = read_json(Path(path))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read seed probe benchmark spec: {error}") from error
    return validate_seed_probe_benchmark_spec(value)


def benchmark_pair_rng_seed(
    authorization: dict[str, Any],
    *,
    sample_id: str,
    cell_id: str,
) -> str:
    """Derive the common pair seed from preregistered, arm-neutral identity."""

    normalized = normalize_seed_probe_benchmark_execution_authorization(
        authorization
    )
    identity = {
        "schema": "campaignx.seed_probe_benchmark_pair_rng.v1",
        "benchmark_id": normalized["benchmark_id"],
        "benchmark_spec_sha256": normalized["benchmark_spec_sha256"],
        "sample_id": str(sample_id),
        "cell_id": str(cell_id),
    }
    return content_sha256(identity)[:16]


def normalize_seed_probe_benchmark_execution(
    value: Any,
    *,
    sample_id: str | None = None,
    cell_id: str | None = None,
    policy_version: str | None = None,
    planner: str | None = None,
    seed_probe_mode: str | None = None,
    parameter_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one arm/cell execution contract and optional task context."""

    expected = {
        "schema",
        "execution_scope",
        "benchmark_id",
        "benchmark_spec_sha256",
        "authorization",
        "authorization_sha256",
        "arm",
        "policy_version",
        "planner",
        "seed_probe_mode",
        "sample_id",
        "cell_id",
        "pair_rng_seed",
        "rng_protocol",
        "full_grow_envelope_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark_execution differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_EXECUTION_SCHEMA:
        raise ValueError("unsupported seed probe benchmark_execution schema")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "benchmark_execution must be ISOLATED_NONPRODUCTION"
        )
    authorization = normalize_seed_probe_benchmark_execution_authorization(
        value["authorization"]
    )
    if (
        value["authorization_sha256"] != content_sha256(authorization)
        or value["benchmark_id"] != authorization["benchmark_id"]
        or value["benchmark_spec_sha256"]
        != authorization["benchmark_spec_sha256"]
    ):
        raise ValueError(
            "benchmark_execution is not bound to its preregistered "
            "authorization"
        )
    _require_lowercase_sha256(
        value["authorization_sha256"], "authorization_sha256"
    )
    _require_lowercase_sha256(
        value["benchmark_spec_sha256"], "benchmark_spec_sha256"
    )

    arm = value["arm"]
    if arm == "baseline":
        expected_policy = authorization["baseline_policy_version"]
        expected_mode = "off"
    elif arm == "closed_loop":
        expected_policy = authorization["policy_version"]
        expected_mode = "select"
    else:
        raise ValueError(
            "benchmark_execution arm must be baseline or closed_loop"
        )
    if (
        value["policy_version"] != expected_policy
        or value["planner"] != "deterministic-v2"
        or value["seed_probe_mode"] != expected_mode
    ):
        raise ValueError(
            "benchmark_execution differs from its preregistered arm"
        )

    execution_sample = value["sample_id"]
    execution_cell = value["cell_id"]
    if any(
        not isinstance(item, str) or not item
        for item in (execution_sample, execution_cell)
    ):
        raise ValueError(
            "benchmark_execution sample_id and cell_id must be non-empty"
        )
    if not any(
        row["sample_id"] == execution_sample
        and row["cell_id"] == execution_cell
        for row in authorization["cells"]
    ):
        raise ValueError(
            "benchmark_execution task is outside the preregistered cohort"
        )
    expected_seed = benchmark_pair_rng_seed(
        authorization,
        sample_id=execution_sample,
        cell_id=execution_cell,
    )
    if (
        not isinstance(value["pair_rng_seed"], str)
        or _PAIR_RNG_RE.fullmatch(value["pair_rng_seed"]) is None
        or value["pair_rng_seed"] != expected_seed
    ):
        raise ValueError(
            "benchmark_execution pair_rng_seed does not match its frozen cell"
        )
    if value["rng_protocol"] != BENCHMARK_RNG_PROTOCOL:
        raise ValueError("unsupported benchmark_execution RNG protocol")
    if (
        value["full_grow_envelope_sha256"]
        != BENCHMARK_FULL_GROW_ENVELOPE_SHA256
    ):
        raise ValueError(
            "benchmark_execution must use the common resume-compatible "
            "full-grow envelope"
        )
    if parameter_envelope is not None and content_sha256(
        parameter_envelope
    ) != value["full_grow_envelope_sha256"]:
        raise ValueError(
            "benchmark_execution full-grow envelope hash does not match "
            "the task envelope"
        )

    contextual = {
        "sample_id": sample_id,
        "cell_id": cell_id,
        "policy_version": policy_version,
        "planner": planner,
        "seed_probe_mode": seed_probe_mode,
    }
    for field, expected_value in contextual.items():
        if (
            expected_value is not None
            and value[field] != expected_value
        ):
            raise ValueError(
                f"benchmark_execution {field} differs from its task"
            )
    return {
        **copy.deepcopy(value),
        "authorization": authorization,
    }


def build_seed_probe_benchmark_execution(
    authorization: dict[str, Any],
    *,
    arm: str,
    sample_id: str,
    cell_id: str,
    parameter_envelope: dict[str, Any],
) -> dict[str, Any]:
    """Build the only explicit-RNG contract accepted by a benchmark task."""

    normalized = normalize_seed_probe_benchmark_execution_authorization(
        authorization
    )
    if arm == "baseline":
        policy_version = normalized["baseline_policy_version"]
        seed_probe_mode = "off"
    elif arm == "closed_loop":
        policy_version = normalized["policy_version"]
        seed_probe_mode = "select"
    else:
        raise ValueError("benchmark execution arm must be baseline or closed_loop")
    value = {
        "schema": BENCHMARK_EXECUTION_SCHEMA,
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "benchmark_id": normalized["benchmark_id"],
        "benchmark_spec_sha256": normalized["benchmark_spec_sha256"],
        "authorization": normalized,
        "authorization_sha256": content_sha256(normalized),
        "arm": arm,
        "policy_version": policy_version,
        "planner": "deterministic-v2",
        "seed_probe_mode": seed_probe_mode,
        "sample_id": str(sample_id),
        "cell_id": str(cell_id),
        "pair_rng_seed": benchmark_pair_rng_seed(
            normalized,
            sample_id=str(sample_id),
            cell_id=str(cell_id),
        ),
        "rng_protocol": BENCHMARK_RNG_PROTOCOL,
        "full_grow_envelope_sha256": content_sha256(parameter_envelope),
    }
    return normalize_seed_probe_benchmark_execution(
        value,
        sample_id=str(sample_id),
        cell_id=str(cell_id),
        policy_version=policy_version,
        planner="deterministic-v2",
        seed_probe_mode=seed_probe_mode,
        parameter_envelope=parameter_envelope,
    )


def validate_seed_probe_benchmark_execution_task(
    task: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a task's explicit execution contract or historical absence."""

    if task.get("execution_rng_seed") is not None:
        raise ValueError(
            "execution_rng_seed is not a standalone override; use an exact "
            "isolated benchmark_execution contract"
        )
    raw = task.get("benchmark_execution")
    if raw is None:
        return None
    policy = task.get("seed_probe")
    mode = (
        str(policy.get("mode"))
        if isinstance(policy, dict)
        else "off"
    )
    return normalize_seed_probe_benchmark_execution(
        raw,
        sample_id=str(task.get("sample_id") or ""),
        cell_id=str(task.get("cell_id") or ""),
        policy_version=str(task.get("policy_version") or ""),
        planner=str(task.get("planner") or ""),
        seed_probe_mode=mode,
        parameter_envelope=task.get("parameter_envelope"),
    )


def validate_seed_probe_benchmark_receipt(
    value: Any,
) -> dict[str, Any]:
    """Validate an exact causal approval and return its compact task binding."""

    expected = {
        "schema",
        "benchmark_id",
        "status",
        "execution_scope",
        "spec_sha256",
        "results_sha256",
        "paired_cell_count",
        "scroll_count",
        "authorized_sample_ids",
        "arms",
        "metrics",
        "checks",
        "generated_at_utc",
        "non_claims",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark receipt differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_DECISION_SCHEMA:
        raise ValueError("unsupported seed probe benchmark decision schema")
    if value["status"] != "APPROVED_SELECT":
        raise ValueError("seed probe benchmark receipt is not APPROVED_SELECT")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "seed probe select approval must come from ISOLATED_NONPRODUCTION"
        )
    benchmark_id = value["benchmark_id"]
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("benchmark receipt benchmark_id must be non-empty")
    for field in ("spec_sha256", "results_sha256", "receipt_sha256"):
        _require_lowercase_sha256(value[field], field)

    arms = value["arms"]
    if not isinstance(arms, dict) or set(arms) != {"baseline", "closed_loop"}:
        raise ValueError("benchmark receipt arms differ from v1 contract")
    expected_arm_keys = {"policy_version", "planner", "seed_probe_mode"}
    for name in ("baseline", "closed_loop"):
        arm = arms[name]
        if not isinstance(arm, dict) or set(arm) != expected_arm_keys:
            raise ValueError(f"benchmark receipt {name} arm differs from v1 contract")
        if any(not isinstance(arm[field], str) or not arm[field] for field in arm):
            raise ValueError(f"benchmark receipt {name} arm identity is incomplete")
    if (
        arms["baseline"]["planner"] != "deterministic-v2"
        or arms["baseline"]["seed_probe_mode"] != "off"
        or arms["closed_loop"]["planner"] != "deterministic-v2"
        or arms["closed_loop"]["seed_probe_mode"] != "select"
        or arms["baseline"]["policy_version"]
        == arms["closed_loop"]["policy_version"]
    ):
        raise ValueError(
            "benchmark receipt must compare distinct deterministic-v2/off and "
            "deterministic-v2/select arms"
        )

    checks = value["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError("benchmark receipt checks must be a non-empty list")
    seen_checks: set[str] = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) != {
            "check_id",
            "status",
            "detail",
            "evidence",
        }:
            raise ValueError(
                f"benchmark receipt checks[{index}] differs from v1 contract"
            )
        check_id = check["check_id"]
        if not isinstance(check_id, str) or not check_id:
            raise ValueError(f"benchmark receipt checks[{index}] has no check_id")
        if check_id in seen_checks:
            raise ValueError(f"benchmark receipt has duplicate check {check_id}")
        seen_checks.add(check_id)
        if check["status"] != "PASS":
            raise ValueError(f"benchmark receipt check {check_id} did not PASS")
        if not isinstance(check["detail"], str) or not check["detail"]:
            raise ValueError(f"benchmark receipt check {check_id} has no detail")
    if seen_checks != BENCHMARK_REQUIRED_CHECK_IDS:
        raise ValueError(
            "benchmark receipt check set differs from v1 approval contract"
        )
    if not isinstance(value["metrics"], dict):
        raise ValueError("benchmark receipt metrics must be an object")
    per_scroll = value["metrics"].get("per_scroll")
    if not isinstance(per_scroll, dict) or any(
        not isinstance(sample_id, str) or not sample_id
        for sample_id in per_scroll
    ):
        raise ValueError(
            "benchmark receipt metrics.per_scroll must bind the approved "
            "scroll cohort"
        )
    if not isinstance(value["non_claims"], list) or not all(
        isinstance(item, str) for item in value["non_claims"]
    ):
        raise ValueError("benchmark receipt non_claims must be a string list")
    generated_at = value["generated_at_utc"]
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise ValueError("benchmark receipt generated_at_utc must be UTC")
    try:
        parsed_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "benchmark receipt generated_at_utc is invalid"
        ) from error
    if parsed_at.utcoffset() != timezone.utc.utcoffset(parsed_at):
        raise ValueError("benchmark receipt generated_at_utc must be UTC")

    authorization = {
        "schema": BENCHMARK_AUTHORIZATION_SCHEMA,
        "benchmark_id": benchmark_id,
        "decision_receipt_sha256": value["receipt_sha256"],
        "spec_sha256": value["spec_sha256"],
        "results_sha256": value["results_sha256"],
        "paired_cell_count": value["paired_cell_count"],
        "scroll_count": value["scroll_count"],
        "authorized_sample_ids": value["authorized_sample_ids"],
        "execution_scope": value["execution_scope"],
    }
    normalized_authorization = normalize_seed_probe_benchmark_authorization(
        authorization
    )
    if normalized_authorization["authorized_sample_ids"] != sorted(per_scroll):
        raise ValueError(
            "benchmark receipt authorized_sample_ids differs from its "
            "per-scroll evidence"
        )
    body = {
        key: item for key, item in value.items() if key != "receipt_sha256"
    }
    if content_sha256(body) != value["receipt_sha256"]:
        raise ValueError("benchmark receipt canonical SHA-256 does not match")
    return normalized_authorization


def load_seed_probe_benchmark_receipt(path: Path) -> dict[str, Any]:
    """Load one explicit receipt path and return its validated task binding."""

    try:
        value = read_json(Path(path))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read seed probe benchmark receipt: {error}") from error
    return validate_seed_probe_benchmark_receipt(value)


def normalize_seed_probe_policy(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact v1 policy instead of accepting ignored knobs."""

    if not isinstance(value, dict):
        raise ValueError("seed_probe must be an object")
    expected = {
        "schema",
        "policy_id",
        "mode",
        "top_k",
        "probe_profile_id",
        "probe_profile_sha256",
        "probe_parameters",
        "maximum_total_probe_generations",
        "maximum_attempts_per_candidate",
        "evaluation_profile_id",
        "evaluation_profile_sha256",
        "decision_policy_id",
        "inconclusive_action",
        "ink_used",
    }
    actual = set(value)
    optional = {"benchmark_authorization", "review_owner"}
    if not expected.issubset(actual) or actual.difference(expected) - optional:
        raise ValueError(
            "seed_probe fields differ from v1 contract: "
            f"expected {sorted(expected)} with conditional "
            "benchmark_authorization/review_owner, "
            f"got {sorted(value)}"
        )
    if value["schema"] != "campaignx.seed_probe_policy.v1":
        raise ValueError("unsupported seed probe schema")
    if value["policy_id"] != "seed-probe-v1":
        raise ValueError("unsupported seed probe policy")
    mode = str(value["mode"]).lower()
    if mode not in {"shadow", "select"}:
        raise ValueError("seed probe mode must be shadow or select")
    top_k = int(value["top_k"])
    # The fourth place this bound lived. Three made the probe a tie-break
    # between m7's best candidates, and the first shadow run ever executed
    # measured 8 of 8 ELIGIBLE at that depth -- a 0% rejection rate against a
    # 34% break-even, so probing paid for nothing. Whether m7's ordering holds
    # at rank 20 is the open question, and this is what made it unaskable.
    #
    # maximum_attempts below is a different number and stays at 3: it bounds
    # retries per candidate, not how far down the ordering to look.
    if not 1 <= top_k <= TOP_K_MAXIMUM:
        raise ValueError(f"seed probe top_k must be between 1 and {TOP_K_MAXIMUM}")
    if mode == "select" and top_k < 2:
        raise ValueError(
            "seed probe select mode requires at least two candidates; "
            "top_k=1 is a geometry preflight, not a comparison"
        )
    if value["probe_profile_id"] != PROBE_PROFILE_ID:
        raise ValueError("unsupported seed probe profile")
    if value["probe_profile_sha256"] != PROBE_PROFILE_SHA256:
        raise ValueError("seed probe profile SHA-256 does not match v1")
    parameters = value["probe_parameters"]
    if not isinstance(parameters, dict) or set(parameters) != {
        "generations",
        "step_size",
        "min_area_cm",
        "use_cuda",
    }:
        raise ValueError("seed probe parameters differ from the v1 profile")
    generations = int(parameters["generations"])
    step_size = int(parameters["step_size"])
    if not 10 <= generations <= 20:
        raise ValueError("seed probe generations must be between 10 and 20")
    if not 8 <= step_size <= 15:
        raise ValueError("seed probe step_size must be between 8 and 15")
    if float(parameters["min_area_cm"]) != 0.0:
        raise ValueError("seed probe min_area_cm is frozen at 0")
    if parameters["use_cuda"] is not False:
        raise ValueError("seed probe v1 use_cuda is frozen at false")
    maximum_attempts = int(value["maximum_attempts_per_candidate"])
    if not 1 <= maximum_attempts <= 3:
        raise ValueError("maximum probe attempts per candidate must be 1..3")
    maximum_total = int(value["maximum_total_probe_generations"])
    required_total = top_k * generations * maximum_attempts
    if maximum_total != required_total or maximum_total > 180:
        raise ValueError(
            "maximum_total_probe_generations must equal top_k * generations "
            "* maximum_attempts_per_candidate and may not exceed 180"
        )
    if value["evaluation_profile_id"] != PROBE_EVALUATION_PROFILE_ID:
        raise ValueError("unsupported seed probe evaluation profile")
    if (
        value["evaluation_profile_sha256"]
        != PROBE_EVALUATION_PROFILE_SHA256
    ):
        raise ValueError("seed probe evaluation profile SHA-256 does not match v1")
    if value["decision_policy_id"] != "unique-geometry-certified-v1":
        raise ValueError("unsupported seed probe decision policy")
    if value["inconclusive_action"] != "HUMAN_REVIEW":
        raise ValueError("seed probe v1 must send inconclusive evidence to review")
    if value["ink_used"] is not False:
        raise ValueError("seed probes must be ink-blind")
    authorization = value.get("benchmark_authorization")
    if authorization is not None and mode != "select":
        raise ValueError(
            "benchmark authorization may only be bound to seed probe select mode"
        )
    normalized_authorization = None
    review_owner = value.get("review_owner")
    if authorization is not None:
        authorization_schema = (
            authorization.get("schema")
            if isinstance(authorization, dict)
            else None
        )
        if authorization_schema == BENCHMARK_AUTHORIZATION_SCHEMA:
            normalized_authorization = (
                normalize_seed_probe_benchmark_authorization(authorization)
            )
        elif authorization_schema == BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA:
            normalized_authorization = (
                normalize_seed_probe_benchmark_execution_authorization(
                    authorization
                )
            )
        else:
            raise ValueError(
                "unsupported seed probe benchmark authorization schema"
            )
    if (
        isinstance(normalized_authorization, dict)
        and normalized_authorization.get("schema")
        == BENCHMARK_AUTHORIZATION_SCHEMA
    ):
        if (
            not isinstance(review_owner, str)
            or not review_owner.strip()
            or len(review_owner) > 200
            or any(ord(character) < 32 for character in review_owner)
        ):
            raise ValueError(
                "production-approved seed probe select requires a non-empty "
                "review_owner"
            )
        review_owner = review_owner.strip()
    elif review_owner is not None:
        raise ValueError(
            "review_owner is only valid with production benchmark authorization"
        )
    return {
        **value,
        "mode": mode,
        "top_k": top_k,
        "probe_parameters": {
            "generations": generations,
            "step_size": step_size,
            "min_area_cm": 0.0,
            "use_cuda": parameters["use_cuda"],
        },
        "maximum_total_probe_generations": maximum_total,
        "maximum_attempts_per_candidate": maximum_attempts,
        **(
            {"benchmark_authorization": normalized_authorization}
            if normalized_authorization is not None
            else {}
        ),
        **({"review_owner": review_owner} if review_owner is not None else {}),
    }


def verify_probe_evaluation_profile() -> dict[str, Any]:
    """Fail closed if deployed gate code differs from the frozen probe profile."""

    profile_path = (
        Path(__file__).resolve().parent
        / "profiles/tifxyz-geometry-gate-probe-v1.json"
    )
    if file_sha256(profile_path) != PROBE_EVALUATION_PROFILE_SHA256:
        raise RuntimeError("seed probe evaluation profile bytes have drifted")
    profile = read_json(profile_path)
    repository = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "framework").is_dir()
        ),
        None,
    )
    if repository is None:
        raise RuntimeError("cannot locate repository root for probe gate audit")
    for field in ("gate", "integrity_dependency"):
        descriptor = profile[field]
        implementation = repository / descriptor["path"]
        if file_sha256(implementation) != descriptor["sha256"]:
            raise RuntimeError(
                f"seed probe {field} implementation differs from its "
                "frozen evaluation profile"
            )
    return profile


def verify_probe_growth_profile() -> dict[str, Any]:
    """Fail closed if the deployed VC3D probe profile bytes drift."""

    profile_path = (
        Path(__file__).resolve().parent / "profiles/vc3d-m7-probe-v1.json"
    )
    if file_sha256(profile_path) != PROBE_PROFILE_SHA256:
        raise RuntimeError("seed probe VC3D profile bytes have drifted")
    profile = read_json(profile_path)
    if (
        profile.get("profile_id") != PROBE_PROFILE_ID
        or profile.get("noncanonical") is not True
        or profile.get("ink_used") is not False
    ):
        raise RuntimeError("seed probe VC3D profile contract is invalid")
    return profile


def normalize_source_content_lock(source: dict[str, Any]) -> dict[str, Any]:
    """Validate that select mode reads immutable, content-verified inputs.

    A digest beside a mutable URI is not a lock: the grower would still read
    whatever bytes happen to live at that URI.  This contract therefore binds
    the digest, the URI actually read, and a version identifier embedded in
    that URI.  Source intake is responsible for producing the verification
    receipt; the probe path only consumes an exact frozen receipt.
    """

    if not isinstance(source, dict):
        raise ValueError("seed probe source must be an object")
    lock = source.get("source_content_lock")
    expected = {
        "schema",
        "status",
        "verification_method",
        "verified_at_utc",
        "ct_uri",
        "ct_sha256",
        "ct_version_id",
        "m7_uri",
        "m7_sha256",
        "m7_version_id",
    }
    if not isinstance(lock, dict) or set(lock) != expected:
        raise ValueError(
            "seed probe select mode requires an exact source_content_lock v1"
        )
    if (
        lock["schema"] != SOURCE_CONTENT_LOCK_SCHEMA
        or lock["status"] != "VERIFIED_IMMUTABLE"
    ):
        raise ValueError("seed probe source content lock is not verified immutable")
    fixture = str(source.get("ct_uri", "")).startswith("fixture://") and str(
        source.get("m7_uri", "")
    ).startswith("fixture://")
    expected_method = (
        "fixture-sha256-v1"
        if fixture
        else "immutable-uri-manifest-sha256-v1"
    )
    if lock["verification_method"] != expected_method:
        raise ValueError("seed probe source lock uses an unsupported verifier")
    verified_at = str(lock["verified_at_utc"])
    try:
        verified_time = datetime.fromisoformat(
            verified_at.replace("Z", "+00:00")
        )
    except ValueError:
        verified_time = None
    if (
        len(verified_at) < 20
        or not verified_at.endswith("Z")
        or verified_time is None
        or verified_time.utcoffset() != timezone.utc.utcoffset(verified_time)
    ):
        raise ValueError("seed probe source lock needs a UTC verification time")
    for prefix in ("ct", "m7"):
        uri = str(source.get(f"{prefix}_uri") or "")
        digest = str(source.get(f"{prefix}_sha256") or "")
        version_id = str(lock[f"{prefix}_version_id"] or "")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"seed probe select mode needs a lowercase {prefix} SHA-256"
            )
        if (
            lock[f"{prefix}_uri"] != uri
            or lock[f"{prefix}_sha256"] != digest
        ):
            raise ValueError(
                f"seed probe {prefix} lock does not match the source snapshot"
            )
        if len(version_id) < 8 or version_id not in uri:
            raise ValueError(
                f"seed probe {prefix} URI does not embed its immutable version"
            )
    return copy.deepcopy(lock)


def frozen_probe_candidates(
    candidates: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Take a stable, bounded candidate set without changing legacy ranking."""

    normalized = normalize_seed_probe_policy(policy)
    ordered = sorted(candidates, key=candidate_rank_key)
    frozen: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ordered[: normalized["top_k"]], start=1):
        if any(key not in candidate for key in ("candidate_id", "x", "y", "z")):
            raise ValueError("seed probe candidate lacks exact id/XYZ")
        frozen.append(
            {
                "candidate_rank": rank,
                "candidate_id": str(candidate["candidate_id"]),
                "x": int(candidate["x"]),
                "y": int(candidate["y"]),
                "z": int(candidate["z"]),
                "score": float(candidate.get("score", 0.0)),
                "cell_interior_clearance_voxels": candidate.get(
                    "cell_interior_clearance_voxels"
                ),
                "volume_interior_clearance_voxels": candidate.get(
                    "volume_interior_clearance_voxels"
                ),
                "source": candidate.get("source"),
            }
        )
    if not frozen:
        raise ValueError("a seed probe run requires at least one candidate")
    return frozen


def executor_fingerprint(executor: Any) -> dict[str, Any]:
    """Bind probe reuse to the implementation that produced the bytes."""

    binary = getattr(executor, "binary", None)
    if binary is not None:
        path = Path(binary)
        if not path.is_file():
            raise FileNotFoundError(f"VC3D grow binary is unavailable: {path}")
        return {
            "executor": "vc3d",
            "binary_sha256": file_sha256(path),
        }
    return {
        "executor": type(executor).__name__,
        "fixture_only": type(executor).__name__ == "FixtureGrowExecutor",
    }


def build_probe_locked_plan(
    task: dict[str, Any],
    trial: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-planner plan used for one identical micro-experiment."""

    normalized = normalize_seed_probe_policy(policy)
    benchmark_execution = validate_seed_probe_benchmark_execution_task(task)
    candidate = trial["candidate"]
    return {
        "schema": "campaignx.seed_probe_locked_plan.v1",
        "status": "LOCKED_READY",
        "task_id": task["task_id"],
        # Stable logical trial identity, not retry identity.  The executor uses
        # this to derive VC_GROWPATCH_RNG_SEED, so recovery is byte-repeatable.
        "attempt_id": trial["probe_trial_id"],
        "probe_run_id": trial["probe_run_id"],
        "probe_trial_id": trial["probe_trial_id"],
        "source_snapshot_id": task["source"]["source_snapshot_id"],
        "sample_id": task["sample_id"],
        "cell": {
            "cell_id": task["cell_id"],
            "bounds_xyz": task["bounds_xyz"],
            "center_xyz": task["center_xyz"],
        },
        "source": {
            "source_snapshot_id": task["source"]["source_snapshot_id"],
            "ct_uri": task["source"]["ct_uri"],
            "ct_sha256": task["source"].get("ct_sha256"),
            "m7_uri": task["source"]["m7_uri"],
            "m7_sha256": task["source"].get("m7_sha256"),
            "shape_xyz": task["source"]["shape_xyz"],
            "voxel_size_um": task["source"]["voxel_size_um"],
            "coordinate_frame": task["source"].get(
                "coordinate_frame", "ct_l0_xyz"
            ),
            **(
                {
                    "source_content_lock": normalize_source_content_lock(
                        task["source"]
                    )
                }
                if normalized["mode"] == "select"
                else {}
            ),
        },
        "selected_seed": {
            key: candidate[key] for key in ("candidate_id", "x", "y", "z")
        },
        "profile_id": normalized["probe_profile_id"],
        "profile_sha256": normalized["probe_profile_sha256"],
        "parameters": dict(normalized["probe_parameters"]),
        "policy_sha256": content_sha256(normalized),
        **(
            {
                "benchmark_execution": benchmark_execution,
                "parameter_envelope_sha256": benchmark_execution[
                    "full_grow_envelope_sha256"
                ],
            }
            if benchmark_execution is not None
            else {}
        ),
        "noncanonical": True,
        "ink_used": False,
        "non_claims": [
            "this plan can only produce a noncanonical seed probe",
            "a seed probe is not physical-sheet or correct-lamina acceptance",
        ],
    }


def validate_probe_locked_plan(
    *,
    task: dict[str, Any],
    trial: dict[str, Any],
    policy: dict[str, Any],
    locked_plan: dict[str, Any],
) -> dict[str, Any]:
    """Return the only plan the ledger may accept for this logical trial."""

    expected = build_probe_locked_plan(task, trial, policy)
    if locked_plan != expected:
        raise ValueError(
            "probe locked plan differs from the authoritative task/trial/policy plan"
        )
    return expected


def expected_probe_artifact_uri(
    *,
    namespace_identity: dict[str, Any],
    sample_id: str,
    probe_run_id: str,
    probe_trial_id: str,
    artifact_sha256: str,
) -> str:
    """Derive the only URI a persisted namespace may publish for this trial."""

    components = tuple(
        str(value)
        for value in (
            sample_id,
            probe_run_id,
            probe_trial_id,
            artifact_sha256,
        )
    )
    if any(
        not value or "/" in value or "\\" in value or value in {".", ".."}
        for value in components
    ):
        raise ValueError("probe artifact identity contains an unsafe path component")
    backend = (
        namespace_identity.get("backend")
        if isinstance(namespace_identity, dict)
        and namespace_identity.get("schema")
        == "campaignx.seed_probe_namespace.v1"
        else None
    )
    if backend == "local" and set(namespace_identity) == {
        "schema",
        "backend",
        "probe_root",
    }:
        root = Path(str(namespace_identity["probe_root"]))
        if not root.is_absolute():
            raise ValueError("local probe namespace identity is invalid")
        return str(root.joinpath(*components).resolve())
    if backend == "s3" and set(namespace_identity) == {
        "schema",
        "backend",
        "bucket",
        "probe_prefix",
    }:
        bucket = str(namespace_identity["bucket"])
        prefix = str(namespace_identity["probe_prefix"]).strip("/")
        if not bucket or not prefix:
            raise ValueError("probe S3 namespace identity is invalid")
        expected_path = "/".join((prefix, *components))
        return f"s3://{bucket}/{expected_path}"
    raise ValueError("probe executor fingerprint has no valid artifact namespace")


def validate_probe_artifact_uri(
    *,
    artifact_uri: str,
    namespace_identity: dict[str, Any],
    sample_id: str,
    probe_run_id: str,
    probe_trial_id: str,
    artifact_sha256: str,
) -> None:
    """Accept only the exact local/S3 noncanonical publication namespace."""

    parsed = urlparse(str(artifact_uri))
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("probe artifact URI may not contain parameters")
    expected = expected_probe_artifact_uri(
        namespace_identity=namespace_identity,
        sample_id=sample_id,
        probe_run_id=probe_run_id,
        probe_trial_id=probe_trial_id,
        artifact_sha256=artifact_sha256,
    )
    if str(artifact_uri) != expected:
        raise ValueError(
            "probe artifact URI is outside the configured publication namespace"
        )


def evaluate_probe_surface(
    surface_dir: Path,
    *,
    task: dict[str, Any],
    probe_run_id: str,
    probe_trial_id: str,
    probe_artifact_set_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure a probe with the normal gate and map it to a probe verdict."""

    voxel_um = float(task["source"]["voxel_size_um"])
    inspection = inspect_tifxyz(surface_dir, voxel_um)
    # The same scale the normal gate gets. A probe that says "this micro-grow
    # would certify" has to mean what P2 means by it, and the comparison between
    # candidates is unaffected either way -- every probe in one run reads the
    # same scroll.
    geometry = certify_surface_geometry(surface_dir, voxel_um=voxel_um)
    state = str(geometry.get("geometry_qc_state") or "GEOMETRY_UNMEASURED")
    complete = bool(geometry.get("measurement_complete"))
    resolution_limited = geometry.get("resolution_limited")
    if (
        state == "GEOMETRY_CERTIFIED"
        and complete
        and resolution_limited is False
    ):
        verdict = "ELIGIBLE"
    elif state.startswith("GEOMETRY_REJECTED_") and complete:
        verdict = "REJECTED"
    else:
        verdict = "UNMEASURED"
    # The geometry gate records the local directory it inspected.  That is
    # useful in an operator receipt but is deliberately not scientific
    # evidence: a crash retry writes identical bytes under a new attempt
    # directory.  Bind the decision to reproducible measurements, not that
    # transient path (or a path-bearing exception string).
    stable_geometry = stable_probe_geometry(geometry)
    evaluation = {
        "schema": "campaignx.seed_probe_evaluation.v1",
        "probe_run_id": probe_run_id,
        "probe_trial_id": probe_trial_id,
        "probe_artifact_set_id": probe_artifact_set_id,
        "profile_id": "tifxyz-geometry-gate-probe-v1",
        "profile_sha256": PROBE_EVALUATION_PROFILE_SHA256,
        "verdict": verdict,
        "geometry_qc_state": state,
        "measurement_complete": complete,
        "resolution_limited": resolution_limited,
        "inspection": {
            key: inspection[key]
            for key in (
                "shape",
                "finite_coordinate_count",
                "valid_triangle_count",
                "bbox_xyz",
                "area_cm2",
            )
        },
        "geometry_certification_sha256": content_sha256(stable_geometry),
        "ink_used": False,
        "non_claims": [
            "ELIGIBLE means only that this micro-patch passed the frozen geometry gate",
            "the geometry gate does not prove that a patch follows the correct physical lamina",
        ],
    }
    return evaluation, geometry


def stable_probe_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Strip only execution-location fields from geometry evidence."""

    return {
        key: value
        for key, value in geometry.items()
        if key not in {"surface_dir", "surface_name", "error"}
    }


def validate_probe_completion_evidence(
    *,
    evaluation: dict[str, Any],
    geometry: dict[str, Any],
    growth_receipt: dict[str, Any],
    artifact_uri: str,
    artifact_manifest: dict[str, Any],
    task_id: str,
    probe_run_id: str,
    probe_trial_id: str,
    probe_artifact_set_id: str,
    locked_plan_sha256: str,
    sample_id: str,
    namespace_identity: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    """Cross-bind every receipt before the ledger can assign a verdict."""

    expected_evaluation_keys = {
        "schema",
        "probe_run_id",
        "probe_trial_id",
        "probe_artifact_set_id",
        "profile_id",
        "profile_sha256",
        "verdict",
        "geometry_qc_state",
        "measurement_complete",
        "resolution_limited",
        "inspection",
        "geometry_certification_sha256",
        "ink_used",
        "non_claims",
    }
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) != expected_evaluation_keys
        or evaluation.get("schema")
        != "campaignx.seed_probe_evaluation.v1"
        or evaluation.get("probe_run_id") != probe_run_id
        or evaluation.get("probe_trial_id") != probe_trial_id
        or evaluation.get("probe_artifact_set_id")
        != probe_artifact_set_id
        or evaluation.get("profile_id") != policy["evaluation_profile_id"]
        or evaluation.get("profile_sha256")
        != policy["evaluation_profile_sha256"]
        or evaluation.get("ink_used") is not False
    ):
        raise ValueError("probe evaluation is not bound to this frozen trial")
    inspection = evaluation.get("inspection")
    if not isinstance(inspection, dict) or set(inspection) != {
        "shape",
        "finite_coordinate_count",
        "valid_triangle_count",
        "bbox_xyz",
        "area_cm2",
    }:
        raise ValueError("probe evaluation inspection has unexpected fields")
    if (
        evaluation["geometry_qc_state"]
        != geometry.get("geometry_qc_state")
        or evaluation["measurement_complete"]
        is not geometry.get("measurement_complete")
        or evaluation["resolution_limited"]
        is not geometry.get("resolution_limited")
        or evaluation["geometry_certification_sha256"]
        != content_sha256(stable_probe_geometry(geometry))
    ):
        raise ValueError("probe evaluation does not match geometry evidence")
    state = str(evaluation["geometry_qc_state"])
    complete = evaluation["measurement_complete"] is True
    resolution_limited = evaluation["resolution_limited"]
    expected_verdict = (
        "ELIGIBLE"
        if state == "GEOMETRY_CERTIFIED"
        and complete
        and resolution_limited is False
        else "REJECTED"
        if state.startswith("GEOMETRY_REJECTED_") and complete
        else "UNMEASURED"
    )
    if evaluation.get("verdict") != expected_verdict:
        raise ValueError("probe verdict contradicts its geometry state")
    if (
        growth_receipt.get("schema")
        != "campaignx.segment_fleet_growth_receipt.v1"
        or growth_receipt.get("status") != "GROW_SUCCEEDED"
        or growth_receipt.get("task_id") != task_id
        or growth_receipt.get("attempt_id") != probe_trial_id
        or growth_receipt.get("locked_plan_sha256") != locked_plan_sha256
        or growth_receipt.get("ink_used") is not False
    ):
        raise ValueError("probe growth receipt is not bound to the locked trial")
    if (
        artifact_manifest.get("probe_run_id") != probe_run_id
        or artifact_manifest.get("probe_trial_id") != probe_trial_id
        or artifact_manifest.get("locked_plan_sha256")
        != locked_plan_sha256
    ):
        raise ValueError("probe artifact manifest is not bound to the locked trial")
    artifact_sha = str(artifact_manifest.get("artifact_sha256") or "")
    validate_probe_artifact_uri(
        artifact_uri=artifact_uri,
        namespace_identity=namespace_identity,
        sample_id=sample_id,
        probe_run_id=probe_run_id,
        probe_trial_id=probe_trial_id,
        artifact_sha256=artifact_sha,
    )
    return expected_verdict


def decide_probe_run(run: dict[str, Any]) -> dict[str, Any]:
    """Apply the categorical v1 decision matrix; no weighted pseudo-score."""

    policy = normalize_seed_probe_policy(run["policy"])
    trials = sorted(run["trials"], key=lambda row: int(row["candidate_rank"]))
    if not trials:
        raise ValueError("cannot decide an empty probe run")
    if any(str(row["state"]) not in PROBE_TERMINAL_TRIAL_STATES for row in trials):
        raise ValueError("all probe trials must be terminal before decision")
    outcomes = [
        {
            "probe_trial_id": str(row["probe_trial_id"]),
            "candidate_id": str(row["candidate"]["candidate_id"]),
            "candidate_rank": int(row["candidate_rank"]),
            "state": str(row["state"]),
            "verdict": (
                str(row["evaluation"]["verdict"])
                if isinstance(row.get("evaluation"), dict)
                else None
            ),
            "geometry_qc_state": (
                row["evaluation"].get("geometry_qc_state")
                if isinstance(row.get("evaluation"), dict)
                else None
            ),
            "evaluation_sha256": (
                content_sha256(row["evaluation"])
                if isinstance(row.get("evaluation"), dict)
                else None
            ),
        }
        for row in trials
    ]
    eligible = [row for row in outcomes if row["verdict"] == "ELIGIBLE"]
    rejected = [row for row in outcomes if row["verdict"] == "REJECTED"]
    inconclusive = [
        row
        for row in outcomes
        if row["verdict"] not in {"ELIGIBLE", "REJECTED"}
    ]
    winner: str | None = None
    if len(outcomes) < 2:
        action = "HUMAN_REVIEW"
        reason = "INSUFFICIENT_CANDIDATES_FOR_COMPARISON"
    elif inconclusive:
        action = "HUMAN_REVIEW"
        reason = "AT_LEAST_ONE_PROBE_WAS_UNMEASURED_OR_FAILED"
    elif len(eligible) == 1 and len(rejected) == len(outcomes) - 1:
        action = "CONTINUE_WINNER"
        winner = eligible[0]["probe_trial_id"]
        reason = "EXACTLY_ONE_GEOMETRY_ELIGIBLE_PROBE"
    elif len(eligible) > 1:
        action = "HUMAN_REVIEW"
        reason = "MULTIPLE_GEOMETRY_ELIGIBLE_PROBES"
    elif len(rejected) == len(outcomes):
        action = "REJECT_ALL"
        reason = "ALL_PROBED_CANDIDATES_GEOMETRY_REJECTED"
    else:
        action = "HUMAN_REVIEW"
        reason = "INCONCLUSIVE_PROBE_EVIDENCE"
    evidence_set_sha256 = content_sha256(outcomes)
    policy_sha256 = content_sha256(policy)
    decision = {
        "schema": "campaignx.seed_probe_decision.v1",
        "probe_run_id": str(run["probe_run_id"]),
        "policy_id": policy["decision_policy_id"],
        "policy_sha256": policy_sha256,
        "evidence_set_sha256": evidence_set_sha256,
        "action": action,
        "winner_trial_id": winner,
        "reason": reason,
        "trial_outcomes": outcomes,
        "ink_used": False,
        "non_claims": [
            "a probe winner is not a correct-sheet or correct-lamina certificate",
            "REJECT_ALL applies only to the candidates this bounded run probed",
        ],
    }
    if action not in PROBE_DECISION_ACTIONS:
        raise RuntimeError("seed probe decision escaped its frozen action set")
    return decision


def seed_probe_resume_envelope(
    parameter_envelope: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Return the only full-grow envelope that can resume this select probe."""

    normalized = normalize_seed_probe_policy(policy)
    if not isinstance(parameter_envelope, dict):
        raise ValueError("full growth parameter_envelope must be an object")
    envelope = copy.deepcopy(parameter_envelope)
    if normalized["mode"] != "select":
        return envelope
    if not isinstance(envelope.get("parameters"), dict):
        raise ValueError("full growth envelope must contain parameters")
    parameters = envelope["parameters"]
    probe_parameters = normalized["probe_parameters"]
    generation_rule = dict(parameters.get("generations") or {})
    if generation_rule.get("type") != "integer":
        raise ValueError("full growth generations must be an integer envelope")
    reached_generations = int(probe_parameters["generations"])
    minimum_generations = max(
        int(generation_rule.get("minimum", 0)),
        reached_generations + 1,
    )
    maximum_generations = int(generation_rule.get("maximum", -1))
    if minimum_generations > maximum_generations:
        raise ValueError(
            "full growth envelope has no generation target after the probe"
        )
    generation_rule["minimum"] = minimum_generations
    default_generations = int(
        generation_rule.get("default", minimum_generations)
    )
    generation_rule["default"] = min(
        maximum_generations,
        max(minimum_generations, default_generations),
    )
    parameters["generations"] = generation_rule
    for name in ("step_size", "min_area_cm", "use_cuda"):
        if name not in parameters:
            raise ValueError(
                f"full growth envelope lacks resume-compatibility parameter {name}"
            )
        frozen = probe_parameters[name]
        rule = parameters[name]
        if "minimum" in rule and frozen < rule["minimum"]:
            raise ValueError(f"probe {name} is below the full envelope")
        if "maximum" in rule and frozen > rule["maximum"]:
            raise ValueError(f"probe {name} is above the full envelope")
        if "const" in rule and frozen != rule["const"]:
            raise ValueError(f"probe {name} conflicts with the full envelope")
        parameters[name] = {
            "type": "boolean" if isinstance(frozen, bool) else (
                "integer" if isinstance(frozen, int) else "number"
            ),
            "const": frozen,
            "default": frozen,
        }
    return envelope


def validate_seed_probe_task_contract(task: dict[str, Any]) -> dict[str, Any]:
    """Validate source and continuation inputs before any candidate bytes are read."""

    policy = normalize_seed_probe_policy(task.get("seed_probe"))
    candidate_rank = task.get("candidate_rank", 1)
    if (
        not isinstance(candidate_rank, int)
        or isinstance(candidate_rank, bool)
        or candidate_rank < 1
    ):
        raise ValueError("seed probe candidate_rank must be a positive integer")
    if policy["mode"] == "select" and candidate_rank != 1:
        raise ValueError(
            "seed probe select requires candidate_rank 1; its continuation "
            "is bound to one persisted winner"
        )
    benchmark_execution = validate_seed_probe_benchmark_execution_task(task)
    source = task.get("source")
    if not isinstance(source, dict):
        raise ValueError("seed probe task has no bound source snapshot")
    source_m7_uri = str(source.get("m7_uri") or "")
    if not source_m7_uri:
        raise ValueError("seed probe source has no m7_uri")

    lock: dict[str, Any] | None = None
    if policy["mode"] == "select":
        lock = normalize_source_content_lock(source)
        seed_probe_resume_envelope(task.get("parameter_envelope"), policy)
        authorization = policy.get("benchmark_authorization")
        if authorization is None:
            if benchmark_execution is not None:
                raise ValueError(
                    "benchmark_execution requires its preregistered "
                    "authorization in the select policy"
                )
            if lock.get("verification_method") != "fixture-sha256-v1":
                raise ValueError(
                    "seed probe select on a non-fixture source requires a "
                    "production approval or preregistered isolated benchmark "
                    "authorization"
                )
        elif (
            authorization.get("schema")
            == BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA
        ):
            identity = {
                "sample_id": str(task.get("sample_id") or ""),
                "cell_id": str(task.get("cell_id") or ""),
            }
            if not any(
                cell["sample_id"] == identity["sample_id"]
                and cell["cell_id"] == identity["cell_id"]
                for cell in authorization["cells"]
            ):
                raise ValueError(
                    "seed probe benchmark task is outside the preregistered "
                    "isolated cohort"
                )
            if (
                task.get("policy_version")
                != authorization["policy_version"]
                or task.get("planner") != authorization["planner"]
            ):
                raise ValueError(
                    "seed probe benchmark task differs from its preregistered "
                    "closed_loop arm"
                )
            if (
                benchmark_execution is None
                or benchmark_execution["arm"] != "closed_loop"
                or benchmark_execution["authorization"] != authorization
            ):
                raise ValueError(
                    "preregistered select requires its exact isolated "
                    "benchmark_execution binding"
                )
        else:
            if str(task.get("sample_id") or "") not in authorization[
                "authorized_sample_ids"
            ]:
                raise ValueError(
                    "production-approved select task sample is outside the "
                    "authorized non-regressing scroll cohort"
                )
            if benchmark_execution is not None:
                raise ValueError(
                    "a production-approved select task may not carry an "
                    "isolated benchmark_execution contract"
                )

    discovery = task.get("candidate_discovery")
    if isinstance(discovery, dict):
        provider = str(discovery.get("provider") or "vc3d-mcp")
        if provider == "vc3d-mcp":
            if str(discovery.get("prediction_uri") or "") != source_m7_uri:
                raise ValueError(
                    "seed probe candidate discovery URI does not match the "
                    "content-locked m7 URI"
                )
            if str(discovery.get("prediction_space") or "ct_l0_xyz") != "ct_l0_xyz":
                raise ValueError(
                    "seed probe candidate discovery must use ct_l0_xyz"
                )
        elif policy["mode"] == "select":
            raise ValueError(
                "seed probe select requires vc3d-mcp candidate discovery"
            )
    elif policy["mode"] == "select":
        fixture_candidates = task.get("recorded_candidates")
        if not (
            isinstance(lock, dict)
            and lock.get("verification_method") == "fixture-sha256-v1"
            and isinstance(fixture_candidates, list)
            and fixture_candidates
        ):
            raise ValueError(
                "seed probe select requires source-bound m7 candidate discovery"
            )
    return policy


def _resume_compatible_task(
    task: dict[str, Any],
    *,
    decision: dict[str, Any],
    winner: dict[str, Any],
    materialized_surface: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Constrain a full planner packet to the winner and compatible topology."""

    normalized = normalize_seed_probe_policy(policy)
    value = copy.deepcopy(task)
    value["parameter_envelope"] = seed_probe_resume_envelope(
        value["parameter_envelope"], normalized
    )
    reached_generations = int(
        normalized["probe_parameters"]["generations"]
    )
    value["resume_from"] = str(materialized_surface)
    value["resume_artifact"] = {
        "schema": "campaignx.seed_probe_resume_artifact.v1",
        "probe_run_id": decision["probe_run_id"],
        "probe_trial_id": winner["probe_trial_id"],
        "probe_artifact_set_id": winner["probe_artifact_set_id"],
        "artifact_uri": winner["artifact_uri"],
        "manifest_sha256": winner["manifest_sha256"],
        "materialized_path": str(materialized_surface),
        "reached_generations": reached_generations,
        "noncanonical": True,
    }
    value["seed_probe_decision"] = decision
    return value


def build_probe_continuation_contract(
    *,
    task_id: str,
    continuation_attempt_id: str,
    probe_run_id: str,
    source_snapshot_id: str,
    decision_id: str,
    decision_sha256: str,
    winner_trial_id: str,
    winner_probe_artifact_set_id: str,
    winner_manifest_sha256: str,
    winner_artifact_uri: str,
    winner_candidate: dict[str, Any],
    locked_plan: dict[str, Any],
) -> dict[str, Any]:
    """Bind a canonical continuation to stable winner evidence."""

    if (
        locked_plan.get("schema") != "campaignx.segmentation_locked_plan.v2"
        or locked_plan.get("status") != "LOCKED_READY"
        or locked_plan.get("task_id") != task_id
        or locked_plan.get("attempt_id") != continuation_attempt_id
        or locked_plan.get("source_snapshot_id") != source_snapshot_id
        or locked_plan.get("ink_used") is not False
    ):
        raise ValueError(
            "probe continuation is not an exact locked v2 plan for its task"
        )
    decision = locked_plan.get("seed_probe_decision")
    if (
        not isinstance(decision, dict)
        or decision.get("schema") != "campaignx.seed_probe_decision.v1"
        or decision.get("probe_run_id") != probe_run_id
        or decision.get("action") != "CONTINUE_WINNER"
        or decision.get("winner_trial_id") != winner_trial_id
        or decision.get("ink_used") is not False
        or content_sha256(decision) != decision_sha256
        or locked_plan.get("seed_probe_decision_sha256")
        != decision_sha256
    ):
        raise ValueError(
            "locked continuation does not carry the persisted probe decision"
        )
    winner_outcomes = [
        row
        for row in decision.get("trial_outcomes", [])
        if isinstance(row, dict)
        and row.get("probe_trial_id") == winner_trial_id
    ]
    candidate_id = str(winner_candidate.get("candidate_id") or "")
    if (
        len(winner_outcomes) != 1
        or winner_outcomes[0].get("candidate_id") != candidate_id
        or winner_outcomes[0].get("verdict") != "ELIGIBLE"
    ):
        raise ValueError("persisted winner and continuation candidate differ")

    expected_seed = {
        "candidate_id": candidate_id,
        **{
            axis: float(winner_candidate[axis])
            for axis in "xyz"
        },
    }
    selected_seed = locked_plan.get("selected_seed")
    if (
        not isinstance(selected_seed, dict)
        or selected_seed.get("candidate_id") != expected_seed["candidate_id"]
        or any(
            float(selected_seed.get(axis)) != expected_seed[axis]
            for axis in "xyz"
        )
    ):
        raise ValueError(
            "locked continuation did not select the exact probe winner"
        )

    resume = locked_plan.get("resume_artifact")
    stable_resume = {
        "probe_run_id": probe_run_id,
        "probe_trial_id": winner_trial_id,
        "probe_artifact_set_id": winner_probe_artifact_set_id,
        "artifact_uri": winner_artifact_uri,
        "manifest_sha256": winner_manifest_sha256,
    }
    if (
        not isinstance(resume, dict)
        or any(resume.get(key) != value for key, value in stable_resume.items())
        or resume.get("noncanonical") is not True
        or not isinstance(resume.get("reached_generations"), int)
        or int(resume["reached_generations"]) < 1
        or locked_plan.get("resume_from") != resume.get("materialized_path")
    ):
        raise ValueError(
            "locked continuation does not carry the exact winner artifact"
        )
    parameters = locked_plan.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("locked continuation has no full-grow parameters")
    topology_parameters = {}
    for name in ("step_size", "min_area_cm", "use_cuda"):
        if name not in parameters:
            raise ValueError(
                f"locked continuation has no topology parameter {name}"
            )
        topology_parameters[name] = parameters[name]
    return {
        "schema": "campaignx.seed_probe_continuation_contract.v1",
        "continuation_task_id": task_id,
        "source_snapshot_id": source_snapshot_id,
        "probe_run_id": probe_run_id,
        "decision_id": decision_id,
        "decision_sha256": decision_sha256,
        "winner_trial_id": winner_trial_id,
        "winner_probe_artifact_set_id": winner_probe_artifact_set_id,
        "winner_seed": expected_seed,
        "resume_artifact": {
            **stable_resume,
            "reached_generations": int(resume["reached_generations"]),
            "noncanonical": True,
        },
        "full_growth_profile_id": str(locked_plan.get("profile_id") or ""),
        "topology_parameters": topology_parameters,
        "ink_used": False,
    }


def validate_probe_continuation_contract(
    contract: dict[str, Any],
    locked_plan: dict[str, Any],
    *,
    task_id: str,
    attempt_id: str,
) -> None:
    """Reject finalization unless its plan still matches the promoted winner."""

    if (
        not isinstance(contract, dict)
        or contract.get("schema")
        != "campaignx.seed_probe_continuation_contract.v1"
        or contract.get("continuation_task_id") != task_id
        or contract.get("ink_used") is not False
        or locked_plan.get("task_id") != task_id
        or locked_plan.get("attempt_id") != attempt_id
        or locked_plan.get("source_snapshot_id")
        != contract.get("source_snapshot_id")
        or locked_plan.get("ink_used") is not False
    ):
        raise RuntimeError(
            "final grow is not bound to the probe continuation task"
        )
    decision = locked_plan.get("seed_probe_decision")
    if (
        not isinstance(decision, dict)
        or content_sha256(decision) != contract.get("decision_sha256")
        or locked_plan.get("seed_probe_decision_sha256")
        != contract.get("decision_sha256")
        or decision.get("probe_run_id") != contract.get("probe_run_id")
        or decision.get("winner_trial_id")
        != contract.get("winner_trial_id")
        or decision.get("action") != "CONTINUE_WINNER"
    ):
        raise RuntimeError(
            "final grow lost the selected probe decision binding"
        )
    winner_seed = contract.get("winner_seed")
    selected_seed = locked_plan.get("selected_seed")
    if (
        not isinstance(winner_seed, dict)
        or not isinstance(selected_seed, dict)
        or selected_seed.get("candidate_id")
        != winner_seed.get("candidate_id")
        or any(
            float(selected_seed.get(axis)) != float(winner_seed.get(axis))
            for axis in "xyz"
        )
    ):
        raise RuntimeError("final grow did not use the selected probe seed")
    expected_resume = contract.get("resume_artifact")
    actual_resume = locked_plan.get("resume_artifact")
    stable_keys = (
        "probe_run_id",
        "probe_trial_id",
        "probe_artifact_set_id",
        "artifact_uri",
        "manifest_sha256",
        "reached_generations",
        "noncanonical",
    )
    if (
        not isinstance(expected_resume, dict)
        or not isinstance(actual_resume, dict)
        or any(
            actual_resume.get(key) != expected_resume.get(key)
            for key in stable_keys
        )
        or locked_plan.get("resume_from")
        != actual_resume.get("materialized_path")
    ):
        raise RuntimeError(
            "final grow did not resume the selected probe artifact"
        )
    if (
        locked_plan.get("profile_id")
        != contract.get("full_growth_profile_id")
    ):
        raise RuntimeError("final grow changed the continuation profile")
    parameters = locked_plan.get("parameters")
    topology = contract.get("topology_parameters")
    if (
        not isinstance(parameters, dict)
        or not isinstance(topology, dict)
        or any(parameters.get(name) != value for name, value in topology.items())
    ):
        raise RuntimeError(
            "final grow changed probe-compatible topology parameters"
        )


def validate_probe_finalization_authority(
    *,
    task_payload: dict[str, Any],
    seed_probe_required: bool,
    probe_run: dict[str, Any] | None,
    probe_decision: dict[str, Any] | None,
    promotion: dict[str, Any] | None,
    locked_plan: dict[str, Any],
    replay: bool,
) -> None:
    """Make the persisted select decision—not plan omission—the authority.

    A caller-controlled locked plan cannot opt a select-mode task out of its
    probe stop.  Shadow tasks continue through the ordinary lane, while select
    tasks may finalize only after one persisted CONTINUE_WINNER decision has
    been bound to the exact promotion and continuation plan.
    """

    policy_value = task_payload.get("seed_probe")
    policy = (
        normalize_seed_probe_policy(policy_value)
        if policy_value is not None
        else None
    )
    if bool(seed_probe_required) != (policy is not None):
        raise RuntimeError(
            "authoritative task probe requirement and policy disagree"
        )
    mode = policy["mode"] if policy is not None else None
    plan_decision = locked_plan.get("seed_probe_decision")
    has_plan_winner = (
        isinstance(plan_decision, dict)
        and plan_decision.get("action") == "CONTINUE_WINNER"
    )

    if mode != "select":
        if promotion is not None or has_plan_winner:
            raise RuntimeError(
                "only an authoritative select-mode task may finalize a "
                "probe-selected continuation"
            )
        return

    if not isinstance(probe_run, dict) or not isinstance(
        probe_decision, dict
    ):
        raise RuntimeError(
            "select-mode finalization requires a persisted probe decision"
        )
    run_policy = normalize_seed_probe_policy(probe_run.get("policy"))
    expected_run_state = "PROMOTED" if replay else "CONTINUING"
    if (
        run_policy["mode"] != "select"
        or content_sha256(run_policy) != content_sha256(policy)
        or probe_run.get("state") != expected_run_state
    ):
        raise RuntimeError(
            "select-mode probe run is not the authoritative continuation"
        )

    receipt = probe_decision.get("receipt")
    receipt_sha256 = probe_decision.get("receipt_sha256")
    winner_trial_id = probe_decision.get("winner_trial_id")
    if (
        not isinstance(receipt, dict)
        or content_sha256(receipt) != receipt_sha256
        or receipt.get("probe_run_id") != probe_run.get("probe_run_id")
        or receipt.get("action") != "CONTINUE_WINNER"
        or receipt.get("winner_trial_id") != winner_trial_id
        or probe_decision.get("action") != "CONTINUE_WINNER"
        or not isinstance(winner_trial_id, str)
        or not winner_trial_id
    ):
        raise RuntimeError(
            "select-mode finalization is stopped by its persisted probe "
            "decision"
        )
    if (
        promotion is None
        or promotion.get("decision_id") != probe_decision.get("decision_id")
        or promotion.get("winner_trial_id") != winner_trial_id
    ):
        raise RuntimeError(
            "select-mode winner has no exact persisted promotion"
        )
    if (
        not has_plan_winner
        or content_sha256(plan_decision) != receipt_sha256
        or locked_plan.get("seed_probe_decision_sha256") != receipt_sha256
        or plan_decision.get("probe_run_id") != probe_run.get("probe_run_id")
        or plan_decision.get("winner_trial_id") != winner_trial_id
    ):
        raise RuntimeError(
            "final grow omitted or changed the authoritative select decision"
        )


class SeedProbeCoordinator:
    """Run/recover probe trials while the parent worker keeps its task lease."""

    def __init__(
        self,
        store: Any,
        grow_executor: Any,
        artifact_root: Path | str,
        worker_id: str,
        lease_seconds: int,
        worker_capabilities: dict[str, Any],
    ):
        self.store = store
        self.grow_executor = grow_executor
        self.artifact_store = open_artifact_store(artifact_root)
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.worker_capabilities = worker_capabilities

    def run(
        self,
        task: dict[str, Any],
        candidates: list[dict[str, Any]],
        attempt_dir: Path,
    ) -> dict[str, Any]:
        policy = validate_seed_probe_task_contract(task)
        verify_probe_growth_profile()
        verify_probe_evaluation_profile()
        frozen = frozen_probe_candidates(candidates, policy)
        fingerprint = {
            **executor_fingerprint(self.grow_executor),
            "probe_namespace": self.artifact_store.probe_namespace_identity(),
        }
        probe_run = self.store.prepare_probe_run(
            task,
            frozen,
            policy,
            fingerprint,
        )
        probe_root = attempt_dir / "seed-probe"
        probe_root.mkdir(exist_ok=True)
        write_json_atomic(probe_root / "PROBE_POLICY.json", policy)
        write_json_atomic(probe_root / "PROBE_CANDIDATES.json", frozen)

        while probe_run.get("decision") is None:
            trial = self.store.claim_probe_trial(
                task["task_id"],
                task["attempt_id"],
                task["lease_token"],
                probe_run["probe_run_id"],
                self.worker_id,
                self.lease_seconds,
                self.worker_capabilities,
            )
            if trial is None:
                snapshot = self.store.probe_run(probe_run["probe_run_id"])
                if all(
                    row["state"] in PROBE_TERMINAL_TRIAL_STATES
                    for row in snapshot["trials"]
                ):
                    decision = decide_probe_run(snapshot)
                    self.store.record_probe_decision(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        probe_run["probe_run_id"],
                        decision,
                    )
                    probe_run = self.store.probe_run(probe_run["probe_run_id"])
                    break
                raise RuntimeError(
                    "seed probe run has unfinished trials but none can be claimed"
                )

            trial_dir = (
                probe_root
                / trial["probe_trial_id"]
                / trial["probe_attempt_id"]
            )
            trial_dir.mkdir(parents=True, exist_ok=False)
            plan = build_probe_locked_plan(task, trial, policy)
            write_json_atomic(trial_dir / "PROBE_LOCKED_PLAN.json", plan)
            self.store.transition_probe_trial(
                task["task_id"],
                task["attempt_id"],
                task["lease_token"],
                trial["probe_trial_id"],
                trial["probe_attempt_id"],
                trial["lease_token"],
                "RUNNING",
                locked_plan=plan,
            )
            try:
                grown = self.grow_executor.execute(plan, trial_dir)
                surface_dir = Path(grown["surface_dir"])
                files = artifact_manifest(surface_dir, PROBE_REQUIRED_ARTIFACTS)
                manifest = {
                    "schema": "campaignx.seed_probe_artifact_set.v1",
                    "probe_run_id": probe_run["probe_run_id"],
                    "probe_trial_id": trial["probe_trial_id"],
                    "locked_plan_sha256": content_sha256(plan),
                    "files": files,
                    "artifact_sha256": content_sha256(files),
                    "noncanonical": True,
                    "ink_used": False,
                }
                write_json_atomic(trial_dir / "PROBE_ARTIFACT_SET.json", manifest)
                artifact_set_id = self.store.reserve_probe_artifact(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    manifest,
                )
                published = self.artifact_store.publish_probe(
                    surface_dir,
                    task["sample_id"],
                    probe_run["probe_run_id"],
                    trial["probe_trial_id"],
                    artifact_set_id,
                    manifest,
                )
                evaluation, geometry = evaluate_probe_surface(
                    surface_dir,
                    task=task,
                    probe_run_id=probe_run["probe_run_id"],
                    probe_trial_id=trial["probe_trial_id"],
                    probe_artifact_set_id=artifact_set_id,
                )
                write_json_atomic(
                    trial_dir / "PROBE_GEOMETRY_CERTIFICATION.json", geometry
                )
                write_json_atomic(
                    trial_dir / "PROBE_EVALUATION.json", evaluation
                )
                self.store.complete_probe_trial(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    artifact_set_id,
                    published["artifact_uri"],
                    grown["receipt"],
                    evaluation,
                    geometry,
                )
            except InsufficientGpuMemoryError as error:
                self.store.fail_probe_trial(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    {
                        "status": "RETRY_ON_LARGER_GPU",
                        "growth_receipt": error.receipt,
                        "error_type": type(error).__name__,
                        "ink_used": False,
                    },
                    retryable=True,
                )
                raise
            except Exception as error:
                failure = {
                    "status": "PROBE_TECHNICAL_FAILURE",
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "an operational failure is not evidence against a seed",
                }
                write_json_atomic(trial_dir / "PROBE_FAILURE.json", failure)
                self.store.fail_probe_trial(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    failure,
                    retryable=True,
                )
            probe_run = self.store.probe_run(probe_run["probe_run_id"])

        decision = probe_run.get("decision")
        if not isinstance(decision, dict):
            raise RuntimeError("seed probe run reached no immutable decision")
        write_json_atomic(probe_root / "PROBE_DECISION.json", decision)
        if policy["mode"] == "shadow":
            return {
                "status": "PROBE_SHADOW_COMPLETE",
                "probe_run_id": probe_run["probe_run_id"],
                "decision": decision,
                "planner_candidates": candidates,
                "planner_task": task,
            }
        if decision["action"] != "CONTINUE_WINNER":
            return {
                "status": (
                    "PROBE_REJECTED_ALL"
                    if decision["action"] == "REJECT_ALL"
                    else "PROBE_REVIEW_PENDING"
                ),
                "probe_run_id": probe_run["probe_run_id"],
                "decision": decision,
            }
        winner = next(
            row
            for row in probe_run["trials"]
            if row["probe_trial_id"] == decision["winner_trial_id"]
        )
        materialized = probe_root / "winner-resume"
        try:
            self.artifact_store.materialize_probe(
                winner["artifact_uri"],
                materialized,
                winner["manifest"],
            )
        except Exception as error:
            raise ProbeWinnerMaterializationError(
                probe_run_id=probe_run["probe_run_id"],
                decision=decision,
                winner_trial_id=winner["probe_trial_id"],
                artifact_uri=str(winner["artifact_uri"]),
                error=error,
            ) from error
        planner_candidate = next(
            row
            for row in candidates
            if row["candidate_id"] == winner["candidate"]["candidate_id"]
        )
        planner_task = _resume_compatible_task(
            task,
            decision=decision,
            winner=winner,
            materialized_surface=materialized,
            policy=policy,
        )
        return {
            "status": "PROBE_WINNER",
            "probe_run_id": probe_run["probe_run_id"],
            "decision": decision,
            "planner_candidates": [planner_candidate],
            "planner_task": planner_task,
            "winner": winner,
        }
