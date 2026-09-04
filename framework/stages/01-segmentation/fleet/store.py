from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import content_sha256, is_fixture_surface, stable_id, utc_now
from .dedup import find_duplicate_in_surfaces
from .planner_history import (
    ACTIONABLE_SEGMENTATION_HISTORY_STATES,
    build_regional_attempt_history,
)


ACTIVE_STATES = (
    "CLAIMED",
    "PLANNING",
    "PROBING",
    "LOCKED_READY",
    "RUNNING",
    "UPLOADED",
    "FINALIZING",
)
TERMINAL_STATES = (
    "ARCHIVED",
    "FIXTURE_ONLY",
    "QC_PENDING",
    "NO_SEED",
    "GROW_FAILED",
    "DUPLICATE_SURFACE",
    "FINALIZATION_FAILED",
    "POLICY_REJECTED",
    "BLOCKED_SOURCE_UNAVAILABLE",
    "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE",
    "PROBE_TECHNICAL_FAILURE",
    "PROBE_REVIEW_PENDING",
    "PROBE_REJECTED_ALL",
    "DISCOVERY_PROMOTED",
    "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER",
    "DISCOVERY_REJECTED_CANDIDATES",
)

QC_OUTCOME_STATES = {
    "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS": ("QC_CT_INSUFFICIENT", "CT_INSUFFICIENT"),
    "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL": ("QC_SCREENED", "CT_SUPPORTED"),
    "CT_SUPPORTED_RETAINED_FOR_REVIEW": ("QC_REVIEW_PENDING", "CT_SUPPORTED_REVIEW"),
    "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY": (
        "QC_INK_SCREEN_INSUFFICIENT",
        "INK_SCREEN_INSUFFICIENT",
    ),
}

# The geometry verdict is a second, orthogonal axis.  It deliberately does NOT
# live inside QC_OUTCOME_STATES: those four outcomes are the ink/CT axis that
# the surface-QC adapter reports, and a geometry verdict must be able to
# coexist with any of them.  A surface can be CT_SUPPORTED and
# GEOMETRY_REJECTED_BRIDGE at the same time -- that combination is exactly the
# one the campaign never had a way to express.
GEOMETRY_QC_STATES = (
    "GEOMETRY_CERTIFIED",
    "GEOMETRY_REJECTED_BRIDGE",
    "GEOMETRY_REJECTED_LAMINA_SWITCH",
    "GEOMETRY_REJECTED_DISTORTION",
    "GEOMETRY_REJECTED_COVERAGE",
    "GEOMETRY_UNMEASURED",
)
GEOMETRY_REJECTED_STATES = tuple(
    state for state in GEOMETRY_QC_STATES if state.startswith("GEOMETRY_REJECTED_")
)
DEFAULT_GEOMETRY_QC_STATE = "GEOMETRY_UNMEASURED"

# A third axis, orthogonal to both: does the CT resolve a single lamina under
# this surface? Geometry says the mesh is a plausible sheet and says in its own
# non-claims that it is not a claim the segmentation followed the correct
# lamina; the physical axis says scanned material is there. Neither reads the
# density profile along the normal, which is what decides whether a render is
# worth its cost. See fleet/lamina.py and the frozen bands beside it.
LAMINA_QC_STATES = (
    "LAMINA_SINGLE_SHEET",
    "LAMINA_FUSED",
    "LAMINA_TOO_THIN",
    "LAMINA_UNRESOLVED",
    "LAMINA_INSUFFICIENT_COLUMNS",
    "LAMINA_UNMEASURED",
)
DEFAULT_LAMINA_QC_STATE = "LAMINA_UNMEASURED"

# The fifth judgement. SEED_UNPAIRED is the default because it is the truth for
# every surface that exists: one run, so no error bar -- which is a different
# thing from a large one, and is why this is a state rather than a null.
SEED_AGREEMENT_STATES = (
    "SEED_AGREEMENT_MEASURED",
    "SEED_UNPAIRED",
    "SEED_AGREEMENT_UNMEASURED",
    # The one that exists because this metric fails upward: an override that
    # reached nothing produces two identical fits and an agreement of zero,
    # which reads as perfect reproducibility.
    "SEED_OVERRIDE_DID_NOT_TAKE",
)
DEFAULT_SEED_AGREEMENT_STATE = "SEED_UNPAIRED"

# QC jobs previously had no way to record a job that must never run.  A surface
# rejected by the geometry gate keeps an auditable, FAILED job row instead of
# silently disappearing from the queue.
QC_JOB_STATES = ("PENDING", "CLAIMED", "COMPLETED", "FAILED")


# Which surfaces may be consumed by P3 and P4.
#
# The two QC axes are orthogonal on purpose -- a surface can be CT_SUPPORTED and
# GEOMETRY_REJECTED_BRIDGE at once -- but nothing said which combinations may
# proceed, so P3 gated on geometry alone and ten surfaces whose CT support was
# never measured were flattened and rendered into the detector.
#
# Geometry says the shape is a plausible lamina. The physical axis says there is
# papyrus there at all. Both, or the sheet is a shape with nothing in it.
#
# UNVALIDATED is not a verdict, it is the absence of one: those surfaces are
# waiting for a QC job, not failing it, and they become admissible the moment it
# runs. NOT_APPLICABLE_FIXTURE stays out because a fixture is not evidence.
ADMISSIBLE_PHYSICAL_QC_STATES = ("CT_SUPPORTED", "CT_SUPPORTED_REVIEW")


def is_downstream_admissible(geometry_state: str | None,
                             physical_state: str | None) -> bool:
    """Whether P3 and P4 may consume this surface."""
    return (str(geometry_state or DEFAULT_GEOMETRY_QC_STATE) == "GEOMETRY_CERTIFIED"
            and str(physical_state or "UNVALIDATED") in ADMISSIBLE_PHYSICAL_QC_STATES)


def is_geometry_rejected(state: str | None) -> bool:
    """True when a geometry verdict must keep a surface away from the model."""

    return str(state or DEFAULT_GEOMETRY_QC_STATE) in GEOMETRY_REJECTED_STATES


# A QC job that exists, is auditable, and cannot be claimed.
#
# claim_qc takes only PENDING, so two states were enough while the question was
# "rejected or not". It is not: unmeasured is neither. A surface whose geometry
# has never been measured was getting a PENDING job and could be claimed by the
# coupled ink/CT adapter, which spends model time before the geometry gate the
# stage says comes first. Certification usually runs at finalization, so this
# stayed theoretical -- but "usually" is the whole exposure.
QC_WAITING_GEOMETRY = "WAITING_GEOMETRY"

# A surface below the effort floor keeps a durable, auditable job row that
# claim_qc -- which takes only PENDING -- can never hand to the ink model. It is
# not FAILED: nothing failed. It is not WAITING_GEOMETRY: no verdict is coming.
# The surface is too small for the standard path, which is a fact about its
# size and about nothing else.
QC_SMALL_SURFACE_DIAGNOSTIC = "SMALL_SURFACE_DIAGNOSTIC"


@dataclass(frozen=True, slots=True)
class _FirstLettersDiscoveryRunHandle:
    """Opaque worker handle; durable authority remains in the store."""

    run_id: str
    run_token: str
    worker_id: str
    cell_id: str
    provider_request: dict[str, Any]


def qc_job_state_for(geometry_state: str | None) -> str:
    """What a fresh QC job may do, given what geometry says about the surface.

    Three answers, because there are three situations. Rejected keeps a durable
    FAILED row so the refusal is auditable. Certified is claimable. Unmeasured
    waits -- and record_geometry_certification promotes it the moment a verdict
    arrives, which is what keeps waiting from meaning stranded.
    """
    if is_geometry_rejected(geometry_state):
        return "FAILED"
    if str(geometry_state or DEFAULT_GEOMETRY_QC_STATE) == "GEOMETRY_CERTIFIED":
        return "PENDING"
    return QC_WAITING_GEOMETRY


def normalize_resource_requirements(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return the frozen scheduling contract for one task.

    Missing requirements deliberately preserve V1 behaviour: historical
    tasks remain eligible for every worker.  A task becomes GPU constrained
    only through an explicit requirement or after a recorded GPU OOM.
    """

    raw = dict(value or {})
    minimum_vram_gb = float(raw.get("minimum_vram_gb", 0.0))
    if minimum_vram_gb < 0:
        raise ValueError("minimum_vram_gb must be non-negative")
    gpu_required = bool(raw.get("gpu_required", minimum_vram_gb > 0))
    if minimum_vram_gb > 0 and not gpu_required:
        raise ValueError("a positive minimum_vram_gb requires a GPU")
    return {
        "schema": "campaignx.task_resource_requirements.v1",
        "gpu_required": gpu_required,
        "minimum_vram_gb": minimum_vram_gb,
        "seed_probe_required": bool(raw.get("seed_probe_required", False)),
    }


def normalize_worker_capabilities(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize non-secret worker capabilities used for task admission."""

    raw = dict(value or {})
    gpu_vram_gb = float(raw.get("gpu_vram_gb", 0.0))
    if gpu_vram_gb < 0:
        raise ValueError("gpu_vram_gb must be non-negative")
    cuda_available = bool(raw.get("cuda_available", False))
    gpu_model = raw.get("gpu_model")
    if gpu_model is not None and (not isinstance(gpu_model, str) or not gpu_model.strip()):
        raise ValueError("gpu_model must be a non-empty string when supplied")
    device_index = raw.get("cuda_device_index")
    if device_index is not None and int(device_index) < 0:
        raise ValueError("cuda_device_index must be non-negative")
    if not cuda_available and gpu_vram_gb > 0:
        raise ValueError("gpu_vram_gb requires cuda_available=true")
    benchmark_spec_sha256 = raw.get("benchmark_spec_sha256")
    if benchmark_spec_sha256 is not None and (
        not isinstance(benchmark_spec_sha256, str)
        or len(benchmark_spec_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in benchmark_spec_sha256
        )
    ):
        raise ValueError(
            "benchmark_spec_sha256 must be a lowercase SHA-256 when supplied"
        )
    return {
        "schema": "campaignx.worker_capabilities.v1",
        "cuda_available": cuda_available,
        "gpu_model": gpu_model.strip() if isinstance(gpu_model, str) else None,
        "gpu_vram_gb": gpu_vram_gb,
        "cuda_device_index": int(device_index) if device_index is not None else None,
        "seed_probe_v1": bool(raw.get("seed_probe_v1", False)),
        "benchmark_spec_sha256": benchmark_spec_sha256,
    }


def validate_qc_result_contract(surface_id: str, outcome: str, result: dict[str, Any]) -> None:
    """Fail closed unless a QC result contains durable, attributable evidence."""
    required = {"schema", "surface_id", "outcome", "evidence_manifest_sha256", "evidence_uri", "ink_used"}
    if not required.issubset(result):
        raise ValueError("QC result is missing its frozen evidence contract")
    if result["schema"] != "campaignx.segment_qc_result.v1":
        raise ValueError("QC result schema is not supported")
    if result["surface_id"] != surface_id:
        raise ValueError("QC result surface_id does not match its claimed job")
    if result["outcome"] != outcome:
        raise ValueError("QC result outcome does not match finalization")
    digest = result["evidence_manifest_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("QC evidence manifest must have a lowercase SHA-256")
    if not isinstance(result["evidence_uri"], str) or not result["evidence_uri"].strip():
        raise ValueError("QC evidence URI is required")
    if not isinstance(result["ink_used"], bool):
        raise ValueError("QC result ink_used must be boolean")


FINALIZATION_OUTCOME_STATES = frozenset(
    {"QC_PENDING", "FIXTURE_ONLY", "DUPLICATE_SURFACE"}
)


def validate_finalization_evidence(
    *,
    task_id: str,
    attempt_id: str,
    task_state: str,
    attempt_state: str,
    task_source_snapshot_id: str,
    task_sample_id: str,
    attempt_locked_plan_sha256: str | None,
    artifact_attempt_id: str,
    artifact_state: str,
    artifact_manifest: dict[str, Any],
    artifact_manifest_sha256: str,
    surface: dict[str, Any],
    replay: bool = False,
) -> None:
    """Bind the canonical catalogue write to one uploaded attempt.

    ``finalize`` is the only boundary at which worker-supplied bytes become a
    canonical surface.  Every identity carried independently by the task,
    attempt, artifact manifest, and surface receipt therefore has to agree
    before that boundary is crossed.
    """

    if replay:
        if (
            task_state not in FINALIZATION_OUTCOME_STATES
            or attempt_state != task_state
        ):
            raise RuntimeError("finalization replay is not terminal and immutable")
        expected_artifact_states = (
            {"DUPLICATE"}
            if task_state == "DUPLICATE_SURFACE"
            else {"PROMOTED"}
        )
    else:
        if task_state != "FINALIZING" or attempt_state != "FINALIZING":
            raise RuntimeError(
                "finalization requires task and attempt state FINALIZING"
            )
        expected_artifact_states = {"UPLOADED"}
    if artifact_attempt_id != attempt_id:
        raise RuntimeError("artifact set belongs to a different attempt")
    if artifact_state not in expected_artifact_states:
        raise RuntimeError(
            "artifact set is not in the state required for finalization"
        )
    if (
        not isinstance(artifact_manifest, dict)
        or artifact_manifest.get("schema")
        != "campaignx.segmentation_artifact_set.v1"
        or artifact_manifest.get("task_id") != task_id
        or artifact_manifest.get("attempt_id") != attempt_id
        or artifact_manifest.get("ink_used") is not False
        or not isinstance(artifact_manifest.get("files"), dict)
        or content_sha256(artifact_manifest) != artifact_manifest_sha256
    ):
        raise ValueError("artifact manifest is not bound to the finalizing attempt")
    locked_plan_sha256 = str(attempt_locked_plan_sha256 or "")
    if (
        len(locked_plan_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in locked_plan_sha256
        )
        or artifact_manifest.get("locked_plan_sha256") != locked_plan_sha256
    ):
        raise ValueError("artifact manifest is not bound to the locked plan")
    artifact_sha256 = artifact_manifest.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_sha256
        )
    ):
        raise ValueError("artifact manifest needs a lowercase SHA-256")
    if (
        not isinstance(surface, dict)
        or surface.get("schema") != "campaignx.segment_fleet_surface.v1"
        or surface.get("task_id") != task_id
        or surface.get("attempt_id") != attempt_id
        or surface.get("source_snapshot_id") != task_source_snapshot_id
        or surface.get("sample_id") != task_sample_id
        or surface.get("locked_plan_sha256") != locked_plan_sha256
        or surface.get("artifact_sha256") != artifact_sha256
        or surface.get("ink_used") is not False
        or not isinstance(surface.get("artifact_uri"), str)
        or not surface["artifact_uri"].strip()
    ):
        raise ValueError("surface receipt is not bound to its task and artifact")
    for field in ("bbox_xyz", "sample_points", "area_cm2"):
        if artifact_manifest.get(field) != surface.get(field):
            raise ValueError(
                f"surface {field} differs from its immutable artifact manifest"
            )
    # Both sides must carry a real measurement, not merely the same one.
    #
    # The size gate is written from area_cm2, so a surface finalized without one
    # is a surface nothing can classify. The gate cannot fail closed on that
    # without quarantining surfaces for a gap in the measurement path rather
    # than for anything about their size -- so the gap is closed here instead,
    # at the one boundary both stores share.
    #
    # `inspect_tifxyz` already raises rather than returning an unmeasured area
    # and `finalize_surface` copies it into both receipts, so this constrains
    # callers that bypass the finalizer, not the fleet.
    for source, value in (("artifact manifest", artifact_manifest.get("area_cm2")),
                          ("surface receipt", surface.get("area_cm2"))):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{source} has no measured area_cm2: {value!r}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"{source} area_cm2 is not a usable measurement: {value!r}")


SQLITE_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS source_snapshots (
  source_snapshot_id TEXT PRIMARY KEY,
  sample_id TEXT NOT NULL,
  ct_uri TEXT NOT NULL,
  ct_sha256 TEXT,
  m7_uri TEXT NOT NULL,
  m7_sha256 TEXT,
  shape_xyz_json TEXT NOT NULL,
  voxel_size_um REAL NOT NULL,
  coordinate_frame TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS source_snapshots_by_sample ON source_snapshots(sample_id);

CREATE TABLE IF NOT EXISTS campaign_budget_admissions (
  mission_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  receipt_sha256 TEXT NOT NULL,
  admission_json TEXT NOT NULL,
  admission_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(mission_id, sample_id, receipt_sha256)
);

CREATE TABLE IF NOT EXISTS campaign_decisions (
  receipt_sha256 TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  evaluation_kind TEXT NOT NULL,
  evaluation_index INTEGER NOT NULL,
  decision TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id, policy_version, evaluation_kind, evaluation_index)
);
CREATE INDEX IF NOT EXISTS campaign_decisions_by_scope
  ON campaign_decisions(mission_id, policy_version, evaluation_index);

CREATE TABLE IF NOT EXISTS campaign_resume_authorizations (
  authorization_sha256 TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  prior_policy_version TEXT NOT NULL,
  new_policy_version TEXT NOT NULL,
  new_admission_sha256 TEXT NOT NULL,
  authorization_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id, sample_id, new_admission_sha256)
);
CREATE UNIQUE INDEX IF NOT EXISTS campaign_resume_authorizations_by_predecessor
  ON campaign_resume_authorizations(mission_id, prior_policy_version);

CREATE TABLE IF NOT EXISTS campaign_resume_principal_attestations (
  authorization_sha256 TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  principal TEXT NOT NULL,
  authorization_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS surfaces (
  surface_id TEXT PRIMARY KEY,
  source_snapshot_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  owner TEXT NOT NULL,
  artifact_sha256 TEXT,
  artifact_uri TEXT,
  bbox_xyz_json TEXT NOT NULL,
  sample_points_json TEXT,
  area_cm2 REAL,
  state TEXT NOT NULL,
  physical_qc_state TEXT NOT NULL,
  geometry_qc_state TEXT NOT NULL DEFAULT 'GEOMETRY_UNMEASURED',
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(source_snapshot_id, artifact_sha256),
  FOREIGN KEY(source_snapshot_id) REFERENCES source_snapshots(source_snapshot_id)
);
CREATE INDEX IF NOT EXISTS surfaces_by_sample ON surfaces(sample_id, state);

CREATE TABLE IF NOT EXISTS human_review_events (
  review_event_id TEXT PRIMARY KEY,
  p7_job_id TEXT NOT NULL,
  intent TEXT NOT NULL,
  mission_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  surface_id TEXT NOT NULL,
  verdict_sha256 TEXT NOT NULL,
  card_sha256 TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  vetting_packet_sha256 TEXT NOT NULL,
  author TEXT NOT NULL,
  event_json TEXT NOT NULL,
  event_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(p7_job_id, intent)
);
CREATE INDEX IF NOT EXISTS human_reviews_by_p7
  ON human_review_events(p7_job_id, created_at, review_event_id);

CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL DEFAULT 'unfiled',
  source_snapshot_id TEXT NOT NULL,
  cell_id TEXT NOT NULL,
  grid_version TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  bounds_xyz_json TEXT NOT NULL,
  center_xyz_json TEXT NOT NULL,
  priority REAL NOT NULL,
  parameter_envelope_json TEXT NOT NULL,
  catalog_snapshot_sha256 TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  state TEXT NOT NULL,
  worker_id TEXT,
  lease_token TEXT,
  lease_expires_at TEXT,
  retry_after TEXT,
  gpu_required INTEGER NOT NULL DEFAULT 0,
  minimum_vram_gb REAL NOT NULL DEFAULT 0,
  seed_probe_required INTEGER NOT NULL DEFAULT 0,
  active_attempt_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(mission_id, source_snapshot_id, grid_version, cell_id, policy_version),
  FOREIGN KEY(source_snapshot_id) REFERENCES source_snapshots(source_snapshot_id)
);
CREATE INDEX IF NOT EXISTS tasks_ready ON tasks(state, priority DESC, task_id);

CREATE TABLE IF NOT EXISTS worker_capabilities (
  worker_id TEXT PRIMARY KEY,
  capabilities_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
  attempt_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL,
  worker_id TEXT NOT NULL,
  lease_token TEXT NOT NULL,
  state TEXT NOT NULL,
  proposal_json TEXT,
  proposal_sha256 TEXT,
  locked_plan_json TEXT,
  locked_plan_sha256 TEXT,
  result_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(task_id, attempt_number),
  FOREIGN KEY(task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS artifact_sets (
  artifact_set_id TEXT PRIMARY KEY,
  attempt_id TEXT NOT NULL UNIQUE,
  manifest_json TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  staging_uri TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
);

-- Noncanonical closed-loop seed evidence.  None of these tables references
-- surfaces or QC: only a later ordinary full grow may cross that boundary.
CREATE TABLE IF NOT EXISTS probe_runs (
  probe_run_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  created_by_attempt_id TEXT NOT NULL,
  source_snapshot_id TEXT NOT NULL,
  candidate_set_json TEXT NOT NULL,
  candidate_set_sha256 TEXT NOT NULL,
  policy_id TEXT NOT NULL,
  policy_json TEXT NOT NULL,
  policy_sha256 TEXT NOT NULL,
  executor_fingerprint_json TEXT NOT NULL,
  executor_fingerprint_sha256 TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(task_id,candidate_set_sha256,policy_sha256,executor_fingerprint_sha256),
  FOREIGN KEY(task_id) REFERENCES tasks(task_id),
  FOREIGN KEY(created_by_attempt_id) REFERENCES attempts(attempt_id),
  FOREIGN KEY(source_snapshot_id) REFERENCES source_snapshots(source_snapshot_id)
);
CREATE INDEX IF NOT EXISTS probe_runs_by_task
  ON probe_runs(task_id,state,updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS probe_runs_one_per_task
  ON probe_runs(task_id);

CREATE TABLE IF NOT EXISTS probe_trials (
  probe_trial_id TEXT PRIMARY KEY,
  probe_run_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  candidate_rank INTEGER NOT NULL,
  candidate_json TEXT NOT NULL,
  locked_plan_json TEXT,
  locked_plan_sha256 TEXT,
  state TEXT NOT NULL,
  result_json TEXT,
  worker_id TEXT,
  lease_token TEXT,
  lease_expires_at TEXT,
  retry_after TEXT,
  gpu_required INTEGER NOT NULL DEFAULT 0,
  minimum_vram_gb REAL NOT NULL DEFAULT 0,
  active_probe_attempt_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(probe_run_id,candidate_id),
  UNIQUE(probe_run_id,candidate_rank),
  UNIQUE(probe_run_id,probe_trial_id),
  FOREIGN KEY(probe_run_id) REFERENCES probe_runs(probe_run_id)
);
CREATE INDEX IF NOT EXISTS probe_trials_ready
  ON probe_trials(probe_run_id,state,retry_after,candidate_rank);

CREATE TABLE IF NOT EXISTS probe_attempts (
  probe_attempt_id TEXT PRIMARY KEY,
  probe_trial_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL,
  worker_id TEXT NOT NULL,
  state TEXT NOT NULL,
  growth_receipt_json TEXT,
  result_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(probe_trial_id,attempt_number),
  FOREIGN KEY(probe_trial_id) REFERENCES probe_trials(probe_trial_id)
);

CREATE TABLE IF NOT EXISTS probe_artifact_sets (
  probe_artifact_set_id TEXT PRIMARY KEY,
  probe_trial_id TEXT NOT NULL UNIQUE,
  probe_attempt_id TEXT NOT NULL UNIQUE,
  manifest_json TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  artifact_uri TEXT,
  state TEXT NOT NULL,
  retain_until TEXT,
  deleted_at TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(probe_trial_id) REFERENCES probe_trials(probe_trial_id),
  FOREIGN KEY(probe_attempt_id) REFERENCES probe_attempts(probe_attempt_id)
);

CREATE TABLE IF NOT EXISTS probe_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  probe_trial_id TEXT NOT NULL UNIQUE,
  probe_artifact_set_id TEXT NOT NULL UNIQUE,
  profile_id TEXT NOT NULL,
  profile_sha256 TEXT NOT NULL,
  verdict TEXT NOT NULL,
  result_json TEXT NOT NULL,
  result_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(probe_trial_id) REFERENCES probe_trials(probe_trial_id),
  FOREIGN KEY(probe_artifact_set_id)
    REFERENCES probe_artifact_sets(probe_artifact_set_id)
);

CREATE TABLE IF NOT EXISTS probe_decisions (
  decision_id TEXT PRIMARY KEY,
  probe_run_id TEXT NOT NULL UNIQUE,
  policy_id TEXT NOT NULL,
  policy_sha256 TEXT NOT NULL,
  evidence_set_sha256 TEXT NOT NULL,
  action TEXT NOT NULL,
  winner_trial_id TEXT,
  receipt_json TEXT NOT NULL,
  receipt_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(probe_run_id) REFERENCES probe_runs(probe_run_id),
  FOREIGN KEY(probe_run_id,winner_trial_id)
    REFERENCES probe_trials(probe_run_id,probe_trial_id)
);

CREATE TABLE IF NOT EXISTS probe_promotions (
  promotion_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL UNIQUE,
  winner_trial_id TEXT NOT NULL,
  winner_probe_artifact_set_id TEXT NOT NULL,
  continuation_task_id TEXT UNIQUE,
  continuation_attempt_id TEXT NOT NULL,
  continuation_contract_sha256 TEXT NOT NULL,
  continuation_locked_plan_sha256 TEXT NOT NULL,
  canonical_artifact_set_id TEXT,
  surface_id TEXT,
  state TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  receipt_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(decision_id) REFERENCES probe_decisions(decision_id),
  FOREIGN KEY(winner_trial_id) REFERENCES probe_trials(probe_trial_id),
  FOREIGN KEY(winner_probe_artifact_set_id)
    REFERENCES probe_artifact_sets(probe_artifact_set_id),
  FOREIGN KEY(continuation_task_id) REFERENCES tasks(task_id),
  FOREIGN KEY(continuation_attempt_id) REFERENCES attempts(attempt_id),
  FOREIGN KEY(canonical_artifact_set_id) REFERENCES artifact_sets(artifact_set_id),
  FOREIGN KEY(surface_id) REFERENCES surfaces(surface_id)
);

CREATE TABLE IF NOT EXISTS qc_jobs (
  qc_job_id TEXT PRIMARY KEY,
  surface_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  state TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  worker_id TEXT,
  lease_token TEXT,
  lease_expires_at TEXT,
  retry_after TEXT,
  result_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(surface_id, profile_id),
  FOREIGN KEY(surface_id) REFERENCES surfaces(surface_id)
);

-- v20. One routing decision per surface, written in the transaction that
-- creates the surface. The receipt is the evidence a diagnostic surface was
-- classified rather than discarded, so it is immutable: the triggers below
-- refuse UPDATE and DELETE outright rather than trusting every future caller.
CREATE TABLE IF NOT EXISTS surface_routing_receipts (
  surface_id TEXT PRIMARY KEY,
  route TEXT NOT NULL,
  measured_area_cm2 REAL NOT NULL,
  minimum_area_cm2 REAL NOT NULL,
  policy_version TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  receipt_sha256 TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(surface_id) REFERENCES surfaces(surface_id)
);

CREATE TRIGGER IF NOT EXISTS surface_routing_receipts_are_immutable
BEFORE UPDATE ON surface_routing_receipts
BEGIN
  SELECT RAISE(ABORT, 'a surface routing receipt is immutable');
END;

CREATE TRIGGER IF NOT EXISTS surface_routing_receipts_are_permanent
BEFORE DELETE ON surface_routing_receipts
BEGIN
  SELECT RAISE(ABORT, 'a surface routing receipt is permanent');
END;

-- v21. The only way out of the diagnostic path: which diagnostic surface a new
-- surface continues, resolved against a locked catalogue inside the transaction
-- that creates the successor. The uniqueness constraint is the contract -- one
-- expansion of a predecessor per policy version -- and the primary key is the
-- successor, because an authority permits making a new surface and never
-- editing an old one.
CREATE TABLE IF NOT EXISTS surface_expansion_authorities (
  successor_surface_id TEXT PRIMARY KEY,
  expands_surface_id TEXT NOT NULL,
  predecessor_route TEXT NOT NULL,
  predecessor_receipt_sha256 TEXT NOT NULL,
  prior_policy_version TEXT,
  new_policy_version TEXT,
  authority_sha256 TEXT NOT NULL,
  authority_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(expands_surface_id, new_policy_version),
  FOREIGN KEY(successor_surface_id) REFERENCES surfaces(surface_id),
  FOREIGN KEY(expands_surface_id) REFERENCES surfaces(surface_id)
);

CREATE TRIGGER IF NOT EXISTS surface_expansion_authorities_are_immutable
BEFORE UPDATE ON surface_expansion_authorities
BEGIN
  SELECT RAISE(ABORT, 'a surface expansion authority is immutable');
END;

CREATE TRIGGER IF NOT EXISTS surface_expansion_authorities_are_permanent
BEFORE DELETE ON surface_expansion_authorities
BEGIN
  SELECT RAISE(ABORT, 'a surface expansion authority is permanent');
END;

-- The preflight is work, not a query: it asks M7 through a service that lives
-- where workers live, so it is enqueued where the source lock is checked and
-- executed where the sources are reachable. Same lifecycle as qc_jobs, because
-- that queue works and a second shape for one idea is a second thing to break.
CREATE TABLE IF NOT EXISTS preflight_jobs (
  preflight_job_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  source_snapshot_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('PENDING','CLAIMED','COMPLETED','FAILED')),
  request_json TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  worker_id TEXT,
  lease_token TEXT,
  lease_expires_at TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  receipt_json TEXT,
  reason_code TEXT,
  -- Beside the code, because FAILED alone sends an operator to a worker's
  -- stdout to learn why, and a reason that lives only there is not evidence.
  detail TEXT,
  -- An outage the worker itself calls recoverable sends the job back here
  -- rather than ending it, held until `retry_after` so the next claim does not
  -- just re-read a source that is still down.  `requeues` is the bound: without
  -- it a source that is genuinely gone hides behind an endless retry.
  retry_after TEXT,
  requeues INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- One live job per request, with any number of failed attempts behind it.
--
-- A plain UNIQUE over the three columns made a failure permanent. The control
-- enqueues a frozen request, so its digest never changes, and the run after a
-- transient outage was handed the FAILED row back as its answer. A terminal job
-- is never claimed again, so nothing could clear it and no measurement could
-- ever happen. The failed rows stay -- an attempt is a record -- they just stop
-- being the answer.
CREATE UNIQUE INDEX IF NOT EXISTS preflight_jobs_one_live_per_request
  ON preflight_jobs(mission_id, sample_id, request_sha256)
  WHERE state<>'FAILED';

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT,
  attempt_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_compute_caps (
  mission_id TEXT PRIMARY KEY,
  cap_authority_id TEXT NOT NULL,
  authority_sha256 TEXT NOT NULL UNIQUE,
  cap_units INTEGER NOT NULL CHECK(cap_units >= 0),
  authority_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_compute_reservations (
  reservation_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  work_kind TEXT NOT NULL CHECK(work_kind IN
    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM','ADAPTIVE_CHILD')),
  work_authority_id TEXT NOT NULL,
  work_authority_sha256 TEXT NOT NULL,
  ordered_item_ids_sha256 TEXT NOT NULL,
  item_count INTEGER NOT NULL CHECK(item_count > 0),
  units_per_item INTEGER NOT NULL CHECK(units_per_item = 24),
  reserved_units INTEGER NOT NULL CHECK(reserved_units > 0),
  reserved_before_units INTEGER NOT NULL CHECK(reserved_before_units >= 0),
  reserved_after_units INTEGER NOT NULL CHECK(reserved_after_units >= reserved_before_units),
  source TEXT NOT NULL CHECK(source IN
    ('RESERVED_BEFORE_EXECUTION','IMPORTED_HISTORICAL_EXACT')),
  reservation_json TEXT NOT NULL,
  reservation_sha256 TEXT NOT NULL UNIQUE,
  request_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id,request_id),
  UNIQUE(mission_id,work_kind,work_authority_sha256),
  FOREIGN KEY(mission_id) REFERENCES first_letters_discovery_compute_caps(mission_id)
);
CREATE INDEX IF NOT EXISTS first_letters_discovery_compute_by_mission
  ON first_letters_discovery_compute_reservations(mission_id,created_at,reservation_id);

CREATE TABLE IF NOT EXISTS first_letters_discovery_work_bindings (
  reservation_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  work_kind TEXT NOT NULL,
  dispatch_kind TEXT NOT NULL,
  work_json TEXT NOT NULL,
  work_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_evidence_runs (
  run_id TEXT PRIMARY KEY,
  reservation_id TEXT NOT NULL,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  parent_task_id TEXT,
  parent_attempt_id TEXT,
  worker_id TEXT NOT NULL,
  cell_id TEXT NOT NULL,
  source_snapshot_id TEXT NOT NULL,
  run_token_sha256 TEXT NOT NULL UNIQUE,
  lease_expires_at TEXT NOT NULL,
  profile_bytes BLOB NOT NULL,
  profile_file_sha256 TEXT NOT NULL,
  provider_request_json TEXT NOT NULL,
  run_authority_json TEXT NOT NULL,
  run_authority_sha256 TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL CHECK(state IN ('CLAIMED','COMPLETED')),
  created_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE(reservation_id,cell_id),
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id),
  FOREIGN KEY(parent_task_id) REFERENCES tasks(task_id),
  FOREIGN KEY(parent_attempt_id) REFERENCES attempts(attempt_id),
  FOREIGN KEY(source_snapshot_id) REFERENCES source_snapshots(source_snapshot_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_executor_registry (
  worker_id TEXT PRIMARY KEY,
  executor_id TEXT NOT NULL,
  executor_sha256 TEXT NOT NULL,
  capabilities_json TEXT NOT NULL,
  registration_json TEXT NOT NULL,
  registration_sha256 TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
  created_at TEXT NOT NULL,
  UNIQUE(worker_id,executor_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_executor_claims (
  claim_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE,
  worker_id TEXT NOT NULL,
  executor_id TEXT NOT NULL,
  executor_sha256 TEXT NOT NULL,
  capability TEXT NOT NULL,
  claim_attempt_number INTEGER NOT NULL CHECK(claim_attempt_number > 0),
  execution_lease_token_sha256 TEXT NOT NULL UNIQUE,
  lease_expires_at TEXT NOT NULL,
  claim_json TEXT NOT NULL,
  claim_sha256 TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL CHECK(state IN ('CLAIMED','COMPLETED')),
  created_at TEXT NOT NULL,
  completed_at TEXT,
  FOREIGN KEY(run_id) REFERENCES first_letters_discovery_evidence_runs(run_id),
  FOREIGN KEY(worker_id)
    REFERENCES first_letters_discovery_executor_registry(worker_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_evidence_sets (
  evidence_set_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE,
  evidence_json TEXT NOT NULL,
  evidence_set_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES first_letters_discovery_evidence_runs(run_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_evidence_files (
  evidence_set_id TEXT NOT NULL,
  file_order INTEGER NOT NULL CHECK(file_order >= 0),
  relative_path TEXT NOT NULL,
  role TEXT NOT NULL,
  payload BLOB NOT NULL,
  byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
  sha256 TEXT NOT NULL,
  PRIMARY KEY(evidence_set_id,relative_path),
  UNIQUE(evidence_set_id,file_order),
  FOREIGN KEY(evidence_set_id)
    REFERENCES first_letters_discovery_evidence_sets(evidence_set_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_compute_outcomes (
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  outcome TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(mission_id,request_id,outcome)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_compute_blocks (
  mission_id TEXT PRIMARY KEY,
  reason TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_promotions (
  promotion_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  scientific_opportunity_id TEXT NOT NULL,
  parent_task_id TEXT NOT NULL,
  child_task_id TEXT NOT NULL UNIQUE,
  admission_sha256 TEXT NOT NULL,
  authority_json TEXT NOT NULL,
  authority_sha256 TEXT NOT NULL UNIQUE,
  request_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id,request_id),
  UNIQUE(mission_id,scientific_opportunity_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_promotion_attempt_bindings (
  promotion_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
  attempt_id TEXT NOT NULL UNIQUE,
  binding_json TEXT NOT NULL,
  binding_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  PRIMARY KEY(promotion_id,attempt_number),
  FOREIGN KEY(promotion_id)
    REFERENCES first_letters_discovery_promotions(promotion_id)
);
"""


# Additive SQLite counterpart of PostgreSQL migration 18.  Keep the v17 table
# declarations above frozen: existing databases already contain those exact
# constraints, so the lifecycle expansion must be a real data-preserving
# migration rather than a fresh-database-only edit.
SQLITE_DISCOVERY_LIFECYCLE_V18 = """
CREATE TABLE first_letters_discovery_evidence_runs_v18 (
  run_id TEXT PRIMARY KEY,
  reservation_id TEXT NOT NULL,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  parent_task_id TEXT,
  parent_attempt_id TEXT,
  worker_id TEXT NOT NULL,
  cell_id TEXT NOT NULL,
  source_snapshot_id TEXT NOT NULL,
  run_token_sha256 TEXT NOT NULL UNIQUE,
  lease_expires_at TEXT NOT NULL,
  profile_bytes BLOB NOT NULL,
  profile_file_sha256 TEXT NOT NULL,
  provider_request_json TEXT NOT NULL,
  run_authority_json TEXT NOT NULL,
  run_authority_sha256 TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL CHECK(state IN
    ('CLAIMED','RUNNING','COMPLETED','CONTROL_INCOMPLETE')),
  created_at TEXT NOT NULL,
  started_at TEXT,
  last_heartbeat_at TEXT,
  completed_at TEXT,
  incomplete_at TEXT,
  incomplete_reason TEXT,
  UNIQUE(reservation_id,cell_id),
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id),
  FOREIGN KEY(parent_task_id) REFERENCES tasks(task_id),
  FOREIGN KEY(parent_attempt_id) REFERENCES attempts(attempt_id),
  FOREIGN KEY(source_snapshot_id) REFERENCES source_snapshots(source_snapshot_id)
);

CREATE TABLE first_letters_discovery_executor_claims_v18 (
  claim_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE,
  worker_id TEXT NOT NULL,
  executor_id TEXT NOT NULL,
  executor_sha256 TEXT NOT NULL,
  capability TEXT NOT NULL,
  claim_attempt_number INTEGER NOT NULL CHECK(claim_attempt_number > 0),
  execution_lease_token_sha256 TEXT NOT NULL UNIQUE,
  lease_expires_at TEXT NOT NULL,
  claim_json TEXT NOT NULL,
  claim_sha256 TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL CHECK(state IN
    ('CLAIMED','RUNNING','COMPLETED','CONTROL_INCOMPLETE')),
  created_at TEXT NOT NULL,
  started_at TEXT,
  last_heartbeat_at TEXT,
  completed_at TEXT,
  incomplete_at TEXT,
  incomplete_reason TEXT,
  FOREIGN KEY(run_id) REFERENCES first_letters_discovery_evidence_runs(run_id),
  FOREIGN KEY(worker_id)
    REFERENCES first_letters_discovery_executor_registry(worker_id)
);
"""


SQLITE_DISCOVERY_BRIDGE_V19 = """
CREATE TABLE IF NOT EXISTS first_letters_discovery_history_reconciliations_v19 (
  reconciliation_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('COMPLETE','CONTROL_INCOMPLETE')),
  watermark_sha256 TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  fixed_units INTEGER NOT NULL CHECK(fixed_units >= 0),
  reason TEXT,
  reconciliation_json TEXT NOT NULL,
  reconciliation_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id,manifest_sha256,state)
);
CREATE INDEX IF NOT EXISTS first_letters_discovery_history_by_mission_v19
  ON first_letters_discovery_history_reconciliations_v19(
    mission_id,created_at,reconciliation_id
  );

CREATE TABLE IF NOT EXISTS first_letters_discovery_historical_imports_v19 (
  import_id TEXT PRIMARY KEY,
  reservation_id TEXT NOT NULL,
  mission_id TEXT NOT NULL,
  logical_execution_id TEXT NOT NULL,
  producer_kind TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  profile_file_sha256 TEXT NOT NULL,
  item_id TEXT NOT NULL,
  fixed_units INTEGER NOT NULL CHECK(fixed_units = 24),
  retained_row_ids_json TEXT NOT NULL,
  retained_projection_sha256 TEXT NOT NULL,
  history_manifest_sha256 TEXT NOT NULL,
  import_json TEXT NOT NULL,
  import_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id,logical_execution_id),
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_native_adapters_v19 (
  reservation_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  work_kind TEXT NOT NULL CHECK(work_kind IN
    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')),
  producer_kind TEXT NOT NULL CHECK(producer_kind IN
    ('BASELINE_RECONCILIATION','EXPERIMENTAL_ARM_ADMISSION')),
  native_schema TEXT NOT NULL,
  native_authority_json TEXT NOT NULL,
  native_authority_sha256 TEXT NOT NULL UNIQUE,
  generic_work_authority_json TEXT NOT NULL,
  generic_work_authority_sha256 TEXT NOT NULL,
  profile_bytes BLOB NOT NULL,
  adapter_json TEXT NOT NULL,
  adapter_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id,request_id),
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_dispatches_v19 (
  dispatch_id TEXT PRIMARY KEY,
  reservation_id TEXT NOT NULL UNIQUE,
  mission_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  work_kind TEXT NOT NULL,
  adapter_sha256 TEXT NOT NULL,
  profile_file_sha256 TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  ordered_item_ids_sha256 TEXT NOT NULL,
  item_count INTEGER NOT NULL CHECK(item_count > 0),
  dispatch_json TEXT NOT NULL,
  dispatch_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id)
);

CREATE TABLE IF NOT EXISTS first_letters_discovery_jobs_v19 (
  job_id TEXT PRIMARY KEY,
  dispatch_id TEXT NOT NULL,
  reservation_id TEXT NOT NULL,
  item_order INTEGER NOT NULL CHECK(item_order >= 0),
  item_id TEXT NOT NULL,
  work_item_binding_sha256 TEXT NOT NULL,
  profile_file_sha256 TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  job_json TEXT NOT NULL,
  job_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(dispatch_id,item_order),
  UNIQUE(reservation_id,item_id),
  FOREIGN KEY(dispatch_id)
    REFERENCES first_letters_discovery_dispatches_v19(dispatch_id),
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id)
);
CREATE INDEX IF NOT EXISTS first_letters_discovery_jobs_ready_v19
  ON first_letters_discovery_jobs_v19(reservation_id,item_order,job_id);
"""


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _deadline(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_outage_detail(receipt: dict[str, Any]) -> str:
    """The outage sentence, redacted, at the boundary where it becomes durable.

    An outage sentence can carry a token, a DSN or an internal host name -- a
    presigned URL in a `ServerDisconnected` message is the ordinary case, not the
    exotic one -- and this row is read by anyone who can read the queue.
    """
    from framework.contracts import qc_diagnostics

    raw = receipt.get("error") if isinstance(receipt, dict) else None
    return qc_diagnostics.safe_message(
        str(raw if raw is not None else receipt),
        "RuntimeError: preflight outage had no safe detail",
    )


def _instant(value: str) -> str:
    """One caller-supplied time, in the shape every stored timestamp has.

    Compared as text against `utc_now()`, so a caller passing `+00:00` where the
    column holds `Z` would sort wrong and a deferral would end at the wrong
    moment -- or never.
    """
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class FleetStore:
    """Single-host reference control plane.

    SQLite uses BEGIN IMMEDIATE for atomic claims and is suitable for tests or
    several local processes. Distributed workers must use the PostgreSQL
    migration and service before V2 deployment.
    """

    def __init__(
        self, path: Path | str, *,
        task9_discovery_gate_resolver=None,
        first_letters_discovery_profile_resolver=None,
        first_letters_experimental_arm_resolver=None,
        first_letters_discovery_executor=None,
        first_letters_discovery_worker_id: str | None = None,
        first_letters_discovery_executor_id: str | None = None,
        first_letters_discovery_executor_registration: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        # Task 9 will provide this server-owned resolver.  A caller-supplied
        # gate never becomes authority; absent resolver keeps adaptive writes
        # dormant exactly as the Task 6 handoff requires.
        self._task9_discovery_gate_resolver = task9_discovery_gate_resolver
        self._first_letters_discovery_profile_resolver = (
            first_letters_discovery_profile_resolver
        )
        self._first_letters_experimental_arm_resolver = (
            first_letters_experimental_arm_resolver
        )
        self._first_letters_discovery_executor = (
            first_letters_discovery_executor
        )
        self._first_letters_discovery_worker_id = (
            first_letters_discovery_worker_id
        )
        self._first_letters_discovery_executor_id = (
            first_letters_discovery_executor_id
        )
        self._first_letters_discovery_executor_registration = copy.deepcopy(
            first_letters_discovery_executor_registration
        )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _migrate_discovery_lifecycle_v18(
        connection: sqlite3.Connection,
    ) -> None:
        columns = {
            str(row[1]) for row in connection.execute(
                "PRAGMA table_info(first_letters_discovery_evidence_runs)"
            )
        }
        claim_columns = {
            str(row[1]) for row in connection.execute(
                "PRAGMA table_info(first_letters_discovery_executor_claims)"
            )
        }
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            ("first_letters_discovery_evidence_runs",),
        ).fetchone()
        claim_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            ("first_letters_discovery_executor_claims",),
        ).fetchone()
        lifecycle_columns = {
            "started_at", "last_heartbeat_at", "incomplete_at",
            "incomplete_reason",
        }
        if (lifecycle_columns <= columns
                and lifecycle_columns <= claim_columns
                and table_sql is not None
                and "CONTROL_INCOMPLETE" in str(table_sql[0])
                and claim_sql is not None
                and "CONTROL_INCOMPLETE" in str(claim_sql[0])):
            if connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall():
                raise RuntimeError(
                    "SQLite discovery lifecycle v18 foreign-key drift"
                )
            return
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                + SQLITE_DISCOVERY_LIFECYCLE_V18
                + """
INSERT INTO first_letters_discovery_evidence_runs_v18 (
  run_id,reservation_id,mission_id,request_id,parent_task_id,
  parent_attempt_id,worker_id,cell_id,source_snapshot_id,
  run_token_sha256,lease_expires_at,profile_bytes,profile_file_sha256,
  provider_request_json,run_authority_json,run_authority_sha256,state,
  created_at,completed_at
)
SELECT run_id,reservation_id,mission_id,request_id,parent_task_id,
       parent_attempt_id,worker_id,cell_id,source_snapshot_id,
       run_token_sha256,lease_expires_at,profile_bytes,profile_file_sha256,
       provider_request_json,run_authority_json,run_authority_sha256,state,
       created_at,completed_at
  FROM first_letters_discovery_evidence_runs;

INSERT INTO first_letters_discovery_executor_claims_v18 (
  claim_id,run_id,worker_id,executor_id,executor_sha256,capability,
  claim_attempt_number,execution_lease_token_sha256,lease_expires_at,
  claim_json,claim_sha256,state,created_at,completed_at
)
SELECT claim_id,run_id,worker_id,executor_id,executor_sha256,capability,
       claim_attempt_number,execution_lease_token_sha256,lease_expires_at,
       claim_json,claim_sha256,state,created_at,completed_at
  FROM first_letters_discovery_executor_claims;

DROP TABLE first_letters_discovery_executor_claims;
DROP TABLE first_letters_discovery_evidence_runs;
ALTER TABLE first_letters_discovery_evidence_runs_v18
  RENAME TO first_letters_discovery_evidence_runs;
ALTER TABLE first_letters_discovery_executor_claims_v18
  RENAME TO first_letters_discovery_executor_claims;
"""
            )
            if connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall():
                raise RuntimeError(
                    "SQLite discovery lifecycle v18 foreign-key drift"
                )
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _migrate_discovery_import_cardinality_v19(
        connection: sqlite3.Connection,
    ) -> None:
        """Allow one retained reservation to import each item execution."""

        indexes = connection.execute(
            "PRAGMA index_list(first_letters_discovery_historical_imports_v19)"
        ).fetchall()
        has_legacy_reservation_unique = any(
            bool(index["unique"])
            and [
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA index_info({index['name']})"
                ).fetchall()
            ] == ["reservation_id"]
            for index in indexes
        )
        if not has_legacy_reservation_unique:
            return
        connection.executescript(
            """BEGIN IMMEDIATE;
CREATE TABLE first_letters_discovery_historical_imports_v19_upgrade (
  import_id TEXT PRIMARY KEY,
  reservation_id TEXT NOT NULL,
  mission_id TEXT NOT NULL,
  logical_execution_id TEXT NOT NULL,
  producer_kind TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  profile_file_sha256 TEXT NOT NULL,
  item_id TEXT NOT NULL,
  fixed_units INTEGER NOT NULL CHECK(fixed_units = 24),
  retained_row_ids_json TEXT NOT NULL,
  retained_projection_sha256 TEXT NOT NULL,
  history_manifest_sha256 TEXT NOT NULL,
  import_json TEXT NOT NULL,
  import_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(mission_id,logical_execution_id),
  FOREIGN KEY(reservation_id)
    REFERENCES first_letters_discovery_compute_reservations(reservation_id)
);
INSERT INTO first_letters_discovery_historical_imports_v19_upgrade
SELECT * FROM first_letters_discovery_historical_imports_v19;
DROP TABLE first_letters_discovery_historical_imports_v19;
ALTER TABLE first_letters_discovery_historical_imports_v19_upgrade
  RENAME TO first_letters_discovery_historical_imports_v19;
COMMIT;
"""
        )

    @staticmethod
    def _migrate_preflight_retry(connection: sqlite3.Connection) -> None:
        """Let a failed preflight stop being the answer to the next ask.

        `CREATE TABLE IF NOT EXISTS` will not remove a constraint from a table
        that already exists, and SQLite cannot drop one in place, so a database
        made before this keeps the UNIQUE that made a failure permanent. The
        rows are copied as they are: an attempt is a record.
        """
        indexes = connection.execute("PRAGMA index_list(preflight_jobs)").fetchall()
        legacy = any(
            bool(index["unique"])
            and not str(index["name"]).startswith("preflight_jobs_one_live")
            and [str(row["name"]) for row in connection.execute(
                f"PRAGMA index_info({index['name']})").fetchall()]
            == ["mission_id", "sample_id", "request_sha256"]
            for index in indexes
        )
        if not legacy:
            return
        connection.executescript(
            """BEGIN IMMEDIATE;
CREATE TABLE preflight_jobs_upgrade (
  preflight_job_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  source_snapshot_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('PENDING','CLAIMED','COMPLETED','FAILED')),
  request_json TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  worker_id TEXT,
  lease_token TEXT,
  lease_expires_at TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  receipt_json TEXT,
  reason_code TEXT,
  detail TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
INSERT INTO preflight_jobs_upgrade SELECT * FROM preflight_jobs;
DROP TABLE preflight_jobs;
ALTER TABLE preflight_jobs_upgrade RENAME TO preflight_jobs;
CREATE UNIQUE INDEX IF NOT EXISTS preflight_jobs_one_live_per_request
  ON preflight_jobs(mission_id, sample_id, request_sha256)
  WHERE state<>'FAILED';
COMMIT;"""
        )

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SQLITE_SCHEMA)
            self._migrate_discovery_lifecycle_v18(connection)
            connection.executescript(SQLITE_DISCOVERY_BRIDGE_V19)
            self._migrate_discovery_import_cardinality_v19(connection)
            self._migrate_preflight_retry(connection)
            # V1 databases predate retry scheduling.  The additive migration
            # preserves every frozen task and receipt while allowing a
            # provider-outage attempt to release its cell for a later worker.
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)")}
            if "retry_after" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN retry_after TEXT")
            if "gpu_required" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN gpu_required INTEGER NOT NULL DEFAULT 0")
            if "minimum_vram_gb" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN minimum_vram_gb REAL NOT NULL DEFAULT 0")
            if "seed_probe_required" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN seed_probe_required INTEGER "
                    "NOT NULL DEFAULT 0"
                )
            if "mission_id" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN mission_id TEXT NOT NULL "
                    "DEFAULT 'unfiled'"
                )
            # Added after `_migrate_preflight_retry`, not inside it: that
            # migration copies rows with `INSERT INTO ... SELECT *`, so widening
            # the table it builds would break the copy on column count.
            preflight_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(preflight_jobs)")
            }
            if "retry_after" not in preflight_columns:
                connection.execute("ALTER TABLE preflight_jobs ADD COLUMN retry_after TEXT")
            if "requeues" not in preflight_columns:
                connection.execute(
                    "ALTER TABLE preflight_jobs ADD COLUMN requeues INTEGER NOT NULL DEFAULT 0"
                )
            qc_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(qc_jobs)")}
            for name, declaration in (
                ("worker_id", "TEXT"),
                ("lease_token", "TEXT"),
                ("lease_expires_at", "TEXT"),
                ("retry_after", "TEXT"),
                ("result_json", "TEXT"),
                ("updated_at", "TEXT"),
            ):
                if name not in qc_columns:
                    connection.execute(f"ALTER TABLE qc_jobs ADD COLUMN {name} {declaration}")
            connection.execute("UPDATE qc_jobs SET updated_at=created_at WHERE updated_at IS NULL")
            # V1 databases predate the geometry axis.  Existing surfaces are
            # backfilled as GEOMETRY_UNMEASURED, which is the truth: nothing has
            # ever certified their geometry.  It is not GEOMETRY_CERTIFIED.
            surface_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(surfaces)")
            }
            if "geometry_qc_state" not in surface_columns:
                connection.execute(
                    "ALTER TABLE surfaces ADD COLUMN geometry_qc_state TEXT NOT NULL "
                    f"DEFAULT '{DEFAULT_GEOMETRY_QC_STATE}'"
                )
            # And the lamina axis, backfilled the same way and for the same
            # reason: the gate has never run on these rows, which is not a pass.
            if "lamina_qc_state" not in surface_columns:
                connection.execute(
                    "ALTER TABLE surfaces ADD COLUMN lamina_qc_state TEXT NOT NULL "
                    f"DEFAULT '{DEFAULT_LAMINA_QC_STATE}'"
                )
            # And the fifth judgement. SEED_UNPAIRED is the truth for every row
            # that exists -- one run, so no error bar -- and that is a different
            # thing from a large one, which is why it is a state and not a null.
            if "seed_agreement_state" not in surface_columns:
                connection.execute(
                    "ALTER TABLE surfaces ADD COLUMN seed_agreement_state TEXT "
                    f"NOT NULL DEFAULT '{DEFAULT_SEED_AGREEMENT_STATE}'"
                )
            promotion_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(probe_promotions)"
                )
            }
            for name in (
                "continuation_attempt_id",
                "continuation_contract_sha256",
                "continuation_locked_plan_sha256",
            ):
                if name not in promotion_columns:
                    connection.execute(
                        f"ALTER TABLE probe_promotions ADD COLUMN {name} TEXT"
                    )
            legacy_unbound = int(
                connection.execute(
                    """SELECT COUNT(*) FROM probe_promotions
                        WHERE continuation_attempt_id IS NULL
                           OR continuation_contract_sha256 IS NULL
                           OR continuation_locked_plan_sha256 IS NULL"""
                ).fetchone()[0]
            )
            if legacy_unbound:
                raise RuntimeError(
                    "seed-probe v8 cannot auto-bind legacy promotion rows; "
                    f"{legacy_unbound} promotion(s) require an explicit "
                    "operator evidence review before this database can run"
                )
        if self._first_letters_discovery_executor_registration is not None:
            self.register_first_letters_discovery_executor(
                self._first_letters_discovery_executor_registration
            )

    def event(self, connection: sqlite3.Connection, event_type: str, payload: Any, task_id: str | None = None, attempt_id: str | None = None) -> None:
        connection.execute(
            "INSERT INTO events(task_id,attempt_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
            (task_id, attempt_id, event_type, _dump(payload), utc_now()),
        )

    @staticmethod
    def _validated_first_letters_discovery_executor_registration(
        value: Any,
    ) -> dict[str, Any]:
        required = {
            "schema", "worker_id", "executor_id", "executor_sha256",
            "capabilities", "enabled", "allow_unvalidated",
            "registration_sha256",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("discovery executor registration is not closed")
        capabilities = value["capabilities"]
        if (value["schema"] !=
                "campaignx.first_letters_discovery_executor_registration.v1"
                or any(not isinstance(value[field], str) or not value[field]
                       for field in ("worker_id", "executor_id"))
                or re.fullmatch(r"[0-9a-f]{64}", value["executor_sha256"])
                    is None
                or not isinstance(capabilities, list)
                or any(not isinstance(item, str) or not item
                       for item in capabilities)
                or len(capabilities) != len(set(capabilities))
                or capabilities != sorted(capabilities)
                or type(value["enabled"]) is not bool
                or value["allow_unvalidated"] is not False
                or value["registration_sha256"] != content_sha256({
                    key: row for key, row in value.items()
                    if key != "registration_sha256"
                })):
            raise ValueError("discovery executor registration is invalid")
        return copy.deepcopy(value)

    def register_first_letters_discovery_executor(
        self, registration: dict[str, Any],
    ) -> dict[str, Any]:
        value = self._validated_first_letters_discovery_executor_registration(
            registration
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT registration_sha256 FROM "
                "first_letters_discovery_executor_registry WHERE worker_id=?",
                (value["worker_id"],),
            ).fetchone()
            if existing is not None:
                if existing["registration_sha256"] != value[
                    "registration_sha256"
                ]:
                    raise ValueError("DISCOVERY_EXECUTOR_REGISTRATION_CONFLICT")
                return value
            connection.execute(
                """INSERT INTO first_letters_discovery_executor_registry
                   (worker_id,executor_id,executor_sha256,capabilities_json,
                    registration_json,registration_sha256,enabled,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    value["worker_id"], value["executor_id"],
                    value["executor_sha256"], _dump(value["capabilities"]),
                    _dump(value), value["registration_sha256"],
                    int(value["enabled"]), utc_now(),
                ),
            )
        return value

    def _persisted_discovery_executor_registration_from_connection(
        self, connection: sqlite3.Connection, *, worker_id: str,
    ) -> dict[str, Any]:
        from .discovery_executor import DISCOVERY_EXECUTOR_CAPABILITY

        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        row = connection.execute(
            "SELECT * FROM first_letters_discovery_executor_registry "
            "WHERE worker_id=?",
            (worker_id,),
        ).fetchone()
        if row is None:
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        registration = self._validated_first_letters_discovery_executor_registration(
            _load(row["registration_json"])
        )
        if (row["registration_sha256"] !=
                registration["registration_sha256"]
                or row["executor_id"] != registration["executor_id"]
                or row["executor_sha256"] !=
                    registration["executor_sha256"]
                or _load(row["capabilities_json"]) !=
                    registration["capabilities"]
                or bool(row["enabled"]) is not registration["enabled"]
                or registration["worker_id"] != worker_id
                or registration["enabled"] is not True):
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        if DISCOVERY_EXECUTOR_CAPABILITY not in registration["capabilities"]:
            raise ValueError("DISCOVERY_EXECUTOR_CAPABILITY_REQUIRED")
        return registration

    def _discovery_executor_registration_from_connection(
        self, connection: sqlite3.Connection,
    ) -> dict[str, Any]:
        from .discovery_executor import runtime_discovery_executor_sha256

        worker_id = self._first_letters_discovery_worker_id
        executor_id = self._first_letters_discovery_executor_id
        executor = self._first_letters_discovery_executor
        if (not isinstance(worker_id, str) or not worker_id
                or not isinstance(executor_id, str) or not executor_id
                or executor is None):
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        registration = (
            self._persisted_discovery_executor_registration_from_connection(
                connection, worker_id=worker_id,
            )
        )
        if registration["executor_id"] != executor_id:
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        if (runtime_discovery_executor_sha256(executor) !=
                registration["executor_sha256"]):
            raise ValueError("DISCOVERY_EXECUTOR_CODE_HASH_MISMATCH")
        return registration

    @staticmethod
    def _validated_discovery_compute_cap(value: Any) -> dict[str, Any]:
        required = {
            "schema", "mission_id", "cap_authority_id", "compute_unit",
            "mission_compute_cap_units", "top_k", "probe_generations",
            "maximum_attempts_per_candidate", "probe_profile_id",
            "probe_profile_file_sha256", "deployed_revision", "policy_chain_id",
            "policy_chain_sha256", "allow_unvalidated", "authority_sha256",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("discovery compute cap differs from its closed contract")
        integer = value["mission_compute_cap_units"]
        if (value["schema"] != "campaignx.first_letters_discovery_compute_cap.v1"
                or value["compute_unit"] != "probe_generation_units"
                or value["top_k"] != 2
                or value["probe_generations"] != 12
                or value["maximum_attempts_per_candidate"] != 1
                or value["probe_profile_id"] != "vc3d-m7-probe-v1"
                or value["probe_profile_file_sha256"] !=
                    "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33"
                or value["allow_unvalidated"] is not False
                or isinstance(integer, bool) or not isinstance(integer, int)
                or not 0 <= integer <= 2**63 - 1
                or value["authority_sha256"] != content_sha256({
                    key: row for key, row in value.items()
                    if key != "authority_sha256"
                })):
            raise ValueError("discovery compute cap authority is invalid")
        return copy.deepcopy(value)

    @staticmethod
    def _validated_discovery_work_authority(
        value: Any, *, mission_id: str, work_kind: str,
        work_authority_id: str, work_authority_sha256: str,
        ordered_item_ids: list[str], cap_authority_id: str,
        cap_authority_sha256: str,
    ) -> dict[str, Any]:
        required = {
            "schema", "work_authority_id", "mission_id", "work_kind",
            "ordered_item_ids", "ordered_item_ids_sha256",
            "ordered_item_bindings", "ordered_item_bindings_sha256",
            "cap_authority_id",
            "cap_authority_sha256", "profile_sha256", "policy_sha256",
            "source_sha256", "deployed_revision", "requested_item_count",
            "requested_units", "allow_unvalidated", "work_authority_sha256",
        }
        schemas = {
            "BASELINE_ARM":
                "campaignx.first_letters_discovery_baseline_work_admission.v1",
            "ALTERNATIVE_SOURCE_ARM":
                "campaignx.first_letters_experimental_arm_admission.v1",
            "ADAPTIVE_CHILD": "campaignx.first_letters_discovery_adaptive.v1",
        }
        if (not isinstance(value, dict) or set(value) != required
                or work_kind not in schemas
                or value.get("schema") != schemas[work_kind]
                or value.get("mission_id") != mission_id
                or value.get("work_kind") != work_kind
                or value.get("work_authority_id") != work_authority_id
                or value.get("work_authority_sha256") != work_authority_sha256
                or value.get("ordered_item_ids") != ordered_item_ids
                or value.get("ordered_item_ids_sha256") !=
                    content_sha256(ordered_item_ids)
                or value.get("ordered_item_bindings_sha256") !=
                    content_sha256(value.get("ordered_item_bindings"))
                or value.get("cap_authority_id") != cap_authority_id
                or value.get("cap_authority_sha256") != cap_authority_sha256
                or value.get("requested_item_count") != len(ordered_item_ids)
                or value.get("requested_units") != len(ordered_item_ids) * 24
                or value.get("allow_unvalidated") is not False
                or work_authority_sha256 != content_sha256({
                    key: row for key, row in value.items()
                    if key != "work_authority_sha256"
                })):
            raise ValueError("discovery work authority is invalid or drifted")
        bindings = value["ordered_item_bindings"]
        binding_fields = {
            "schema", "item_id", "sample_id", "source_snapshot_id",
            "source_snapshot_sha256", "cell_region", "cell_region_sha256",
            "grid_version", "grid_spec_sha256", "scientific_opportunity_id",
            "accepted_p0_artifact_id", "accepted_p0_artifact_sha256",
            "parent_task_id", "parent_attempt_id", "allow_unvalidated",
        }
        if (not isinstance(bindings, list) or len(bindings) != len(ordered_item_ids)
                or [row.get("item_id") for row in bindings
                    if isinstance(row, dict)] != ordered_item_ids):
            raise ValueError("discovery work item bindings differ from item order")
        for binding in bindings:
            if (not isinstance(binding, dict) or set(binding) != binding_fields
                    or binding.get("schema") !=
                        "campaignx.first_letters_discovery_work_item_binding.v1"
                    or binding.get("allow_unvalidated") is not False):
                raise ValueError("discovery work item binding is invalid")
            for field in (
                "item_id", "sample_id", "source_snapshot_id", "grid_version",
                "scientific_opportunity_id", "accepted_p0_artifact_id",
            ):
                if not isinstance(binding[field], str) or not binding[field]:
                    raise ValueError("discovery work item identity is invalid")
            for field in (
                "source_snapshot_sha256", "cell_region_sha256",
                "grid_spec_sha256", "accepted_p0_artifact_sha256",
            ):
                digest = binding[field]
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(character not in "0123456789abcdef"
                               for character in digest)):
                    raise ValueError("discovery work item hash is invalid")
            region = binding["cell_region"]
            if (not isinstance(region, dict)
                    or set(region) != {"minimum", "maximum"}
                    or any(not isinstance(region[name], list)
                           or len(region[name]) != 3 for name in region)
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           or item < 0 for name in region for item in region[name])
                    or any(lower >= upper for lower, upper in zip(
                        region["minimum"], region["maximum"], strict=True
                    ))
                    or binding["cell_region_sha256"] != content_sha256(region)
                    or binding["grid_spec_sha256"] != content_sha256({
                        "grid_version": binding["grid_version"],
                        "cell_id": binding["item_id"],
                        "ct_l0_region": region,
                    })):
                raise ValueError("discovery work item region/grid binding is invalid")
            for field in ("parent_task_id", "parent_attempt_id"):
                if binding[field] is not None and (
                    not isinstance(binding[field], str) or not binding[field]
                ):
                    raise ValueError("discovery parent lineage identity is invalid")
        if (not ordered_item_ids
                or any(not isinstance(item, str) or not item for item in ordered_item_ids)
                or ordered_item_ids != list(dict.fromkeys(ordered_item_ids))):
            raise ValueError("discovery work items must be ordered unique IDs")
        return copy.deepcopy(value)

    def register_discovery_compute_cap(self, authority: dict[str, Any]) -> dict[str, Any]:
        value = self._validated_discovery_compute_cap(authority)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT authority_json FROM first_letters_discovery_compute_caps WHERE mission_id=?",
                (value["mission_id"],),
            ).fetchone()
            if existing is not None:
                prior = _load(existing["authority_json"])
                if prior != value:
                    raise ValueError("MISSION_COMPUTE_CAP_AUTHORITY_CONFLICT")
                return prior
            connection.execute(
                """INSERT INTO first_letters_discovery_compute_caps
                   (mission_id,cap_authority_id,authority_sha256,cap_units,authority_json,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (value["mission_id"], value["cap_authority_id"],
                 value["authority_sha256"], value["mission_compute_cap_units"],
                 _dump(value), utc_now()),
            )
        return value

    def discovery_compute_cap(self, mission_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT authority_json FROM first_letters_discovery_compute_caps WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
        return _load(row["authority_json"]) if row is not None else None

    def discovery_compute_total(self, mission_id: str) -> int:
        with self.connect() as connection:
            return int(connection.execute(
                "SELECT COALESCE(SUM(reserved_units),0) FROM first_letters_discovery_compute_reservations WHERE mission_id=?",
                (mission_id,),
            ).fetchone()[0])

    def _block_discovery_compute_ledger(
        self, mission_id: str, evidence: Any,
    ) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO first_letters_discovery_compute_blocks
                   (mission_id,reason,evidence_json,created_at) VALUES(?,?,?,?)
                   ON CONFLICT(mission_id) DO NOTHING""",
                (mission_id, "CONTROL_INCOMPLETE_COMPUTE_LEDGER",
                 _dump(evidence), utc_now()),
            )

    def _persist_discovery_history_incomplete_tx(
        self, connection: sqlite3.Connection, *, mission_id: str,
        manifest: dict[str, Any], reason: str,
    ) -> dict[str, Any]:
        """Seal a claim-time history failure in the caller's write txn."""

        manifest_sha = content_sha256(manifest)
        watermark = {
            "mission_id": mission_id,
            "legacy_probe_run_ids": manifest.get("legacy_probe_run_ids", []),
            "legacy_v16_reservation_ids": manifest.get(
                "legacy_v16_reservation_ids", []
            ),
            "retained_graph_count": len(
                manifest.get("retained_execution_graphs", [])
            ),
            "retained_projection_sha256s": sorted(
                content_sha256(graph)
                for graph in manifest.get("retained_execution_graphs", [])
            ),
        }
        watermark_sha = content_sha256(watermark)
        reconciliation_id = stable_id(
            "first-letters-discovery-history-reconciliation",
            {
                "mission_id": mission_id,
                "manifest_sha256": manifest_sha,
                "state": "CONTROL_INCOMPLETE",
            },
        )
        created_at = utc_now()
        core = {
            "schema":
                "campaignx.first_letters_discovery_history_reconciliation.v1",
            "reconciliation_id": reconciliation_id,
            "mission_id": mission_id,
            "state": "CONTROL_INCOMPLETE",
            "watermark": watermark,
            "watermark_sha256": watermark_sha,
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "fixed_units": 0,
            "reason": reason,
            "allow_unvalidated": False,
        }
        reconciliation = {
            **core,
            "reconciliation_sha256": content_sha256(core),
            "created_at": created_at,
        }
        connection.execute(
            """INSERT INTO
               first_letters_discovery_history_reconciliations_v19
               (reconciliation_id,mission_id,state,watermark_sha256,
                manifest_json,manifest_sha256,fixed_units,reason,
                reconciliation_json,reconciliation_sha256,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(mission_id,manifest_sha256,state) DO NOTHING""",
            (
                reconciliation_id, mission_id, "CONTROL_INCOMPLETE",
                watermark_sha, _dump(manifest), manifest_sha, 0, reason,
                _dump(reconciliation), reconciliation["reconciliation_sha256"],
                created_at,
            ),
        )
        connection.execute(
            """INSERT INTO first_letters_discovery_compute_blocks
               (mission_id,reason,evidence_json,created_at)
               VALUES(?,?,?,?) ON CONFLICT(mission_id) DO UPDATE SET
                 reason=excluded.reason,
                 evidence_json=excluded.evidence_json,
                 created_at=excluded.created_at""",
            (mission_id, reason, _dump(reconciliation), created_at),
        )
        return reconciliation

    @staticmethod
    def _history_row_projection(row: sqlite3.Row) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key in row.keys():
            cell = row[key]
            if isinstance(cell, datetime):
                if cell.tzinfo is None or cell.utcoffset() is None:
                    raise ValueError(
                        "retained history timestamp must be timezone-aware"
                    )
                cell = (
                    cell.astimezone(timezone.utc).isoformat()
                    .replace("+00:00", "Z")
                )
            elif key.endswith("_json") and isinstance(cell, str):
                try:
                    cell = _load(cell)
                except (TypeError, json.JSONDecodeError):
                    pass
            elif isinstance(cell, (bytes, bytearray, memoryview)):
                payload = bytes(cell)
                cell = {
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            value[str(key)] = cell
        return value

    def _derive_first_letters_discovery_history_tx(
        self, connection: sqlite3.Connection, *, mission_id: str,
    ) -> tuple[dict[str, Any], bool, str | None]:
        task_rows = connection.execute(
            """SELECT * FROM tasks
                WHERE mission_id=? AND seed_probe_required=1
                ORDER BY task_id""",
            (mission_id,),
        ).fetchall()
        task_by_id = {str(row["task_id"]): row for row in task_rows}
        run_rows: list[sqlite3.Row] = []
        for candidate_run in connection.execute(
            "SELECT * FROM probe_runs ORDER BY probe_run_id"
        ).fetchall():
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?",
                (candidate_run["task_id"],),
            ).fetchone()
            source = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (candidate_run["source_snapshot_id"],),
            ).fetchone()
            source_payload: dict[str, Any] = {}
            if source is not None:
                try:
                    loaded_source = _load(source["payload_json"])
                    if isinstance(loaded_source, dict):
                        source_payload = loaded_source
                except (TypeError, json.JSONDecodeError):
                    pass
            source_authority = source_payload.get(
                "first_letters_discovery_authority", {}
            )
            if (
                task is not None and task["mission_id"] == mission_id
            ) or (
                isinstance(source_authority, dict)
                and source_authority.get("mission_id") == mission_id
            ):
                run_rows.append(candidate_run)
        retained: list[dict[str, Any]] = []
        complete = True
        reason = None
        run_task_ids = {str(row["task_id"]) for row in run_rows}
        for task in task_rows:
            if str(task["task_id"]) in run_task_ids:
                continue
            complete = False
            reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            retained.append({
                "graph_kind": "LEGACY_MISSING_PROBE_RUN",
                "task": self._history_row_projection(task),
                "logical_units": 0,
            })
        for run in run_rows:
            run_id = str(run["probe_run_id"])
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (run["task_id"],),
            ).fetchone()
            task_attempts = connection.execute(
                "SELECT * FROM attempts WHERE task_id=? "
                "ORDER BY attempt_number,attempt_id", (run["task_id"],),
            ).fetchall()
            created_by_attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id=?",
                (run["created_by_attempt_id"],),
            ).fetchone()
            source_row = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (run["source_snapshot_id"],),
            ).fetchone()
            cap_row = connection.execute(
                "SELECT * FROM first_letters_discovery_compute_caps "
                "WHERE mission_id=?", (mission_id,),
            ).fetchone()
            trials = connection.execute(
                "SELECT * FROM probe_trials WHERE probe_run_id=? "
                "ORDER BY candidate_rank,probe_trial_id",
                (run_id,),
            ).fetchall()
            attempts = connection.execute(
                """SELECT a.* FROM probe_attempts a
                     JOIN probe_trials t ON t.probe_trial_id=a.probe_trial_id
                    WHERE t.probe_run_id=?
                    ORDER BY t.candidate_rank,a.attempt_number,a.probe_attempt_id""",
                (run_id,),
            ).fetchall()
            artifacts = connection.execute(
                """SELECT a.* FROM probe_artifact_sets a
                     JOIN probe_trials t ON t.probe_trial_id=a.probe_trial_id
                    WHERE t.probe_run_id=?
                    ORDER BY t.candidate_rank,a.probe_artifact_set_id""",
                (run_id,),
            ).fetchall()
            evaluations = connection.execute(
                """SELECT e.* FROM probe_evaluations e
                     JOIN probe_trials t ON t.probe_trial_id=e.probe_trial_id
                    WHERE t.probe_run_id=?
                    ORDER BY t.candidate_rank,e.evaluation_id""",
                (run_id,),
            ).fetchall()
            decisions = connection.execute(
                "SELECT * FROM probe_decisions WHERE probe_run_id=? "
                "ORDER BY decision_id",
                (run_id,),
            ).fetchall()
            try:
                policy = _load(run["policy_json"]) or {}
                candidate_set = _load(run["candidate_set_json"])
                executor_fingerprint = _load(
                    run["executor_fingerprint_json"]
                )
                task_payload = (
                    _load(task["payload_json"]) if task is not None else None
                )
                source = (
                    _load(source_row["payload_json"])
                    if source_row is not None else None
                )
            except (TypeError, json.JSONDecodeError):
                policy = {}
                candidate_set = None
                executor_fingerprint = None
                task_payload = None
                source = None
            profile = (
                policy.get("discovery_profile")
                if isinstance(policy, dict) else None
            )
            profile_sha = (
                hashlib.sha256(_dump(profile).encode("utf-8")).hexdigest()
                if isinstance(profile, dict) else None
            )
            authority = (
                source.get("first_letters_discovery_authority", {})
                if isinstance(source, dict) else {}
            )
            task_region = (
                task_payload.get("candidate_discovery", {}).get("region")
                if isinstance(task_payload, dict) else None
            )
            source_fields = (
                "source_snapshot_sha256", "source_content_lock_sha256",
                "ct_metadata_sha256", "ct_read_set_manifest_sha256",
                "m7_read_set_manifest_sha256", "m7_model_id",
                "m7_resolution", "m7_level", "m7_threshold",
                "m7_transform_sha256",
            )
            root_complete = (
                task is not None
                and task["mission_id"] == mission_id
                and task["task_id"] == run["task_id"]
                and int(task["seed_probe_required"]) == 1
                and source_row is not None
                and run["source_snapshot_id"] == task["source_snapshot_id"]
                and isinstance(source, dict)
                and source.get("source_snapshot_id")
                    == run["source_snapshot_id"]
                and isinstance(authority, dict)
                and authority.get("mission_id") == mission_id
                and len(task_attempts) == 1
                and created_by_attempt is not None
                and created_by_attempt["attempt_id"]
                    == run["created_by_attempt_id"]
                and created_by_attempt["task_id"] == run["task_id"]
                and created_by_attempt["state"] == "COMPLETED"
                and isinstance(task_payload, dict)
                and task_payload.get("sample_id") == source.get("sample_id")
                and task_payload.get("scientific_opportunity_id")
                    == authority.get("scientific_opportunities", {}).get(
                        task["cell_id"]
                    )
                and task_payload.get("accepted_p0_artifact_id")
                    == authority.get("accepted_p0_artifact_id")
                and task_payload.get("accepted_p0_artifact_sha256")
                    == authority.get("accepted_p0_artifact_sha256")
                and task_region is not None
                and isinstance(policy, dict)
                and content_sha256(policy) == run["policy_sha256"]
                and isinstance(candidate_set, list)
                and content_sha256(candidate_set)
                    == run["candidate_set_sha256"]
                and isinstance(executor_fingerprint, dict)
                and content_sha256(executor_fingerprint)
                    == run["executor_fingerprint_sha256"]
                and isinstance(profile, dict)
                and profile_sha
                    == policy.get("discovery_profile_file_sha256")
                and profile.get("scientific_core_sha256")
                    == content_sha256({
                        key: value for key, value in profile.items()
                        if key != "scientific_core_sha256"
                    })
                and profile.get("source_snapshot_id")
                    == run["source_snapshot_id"]
                and all(profile.get(field) == source.get(field)
                        for field in source_fields)
                and cap_row is not None
                and profile.get("mission_compute_cap_authority_id")
                    == cap_row["cap_authority_id"]
                and profile.get("mission_compute_cap_authority_sha256")
                    == cap_row["authority_sha256"]
                and profile.get("mission_compute_cap_units")
                    == cap_row["cap_units"]
            )
            run_complete = (
                root_complete
                and
                run["state"] in {
                    "DECIDED", "PROMOTED", "REVIEW_PENDING",
                    "ABSTAINED", "REJECTED",
                }
                and policy.get("mode") == "shadow"
                and policy.get("top_k") == 2
                and policy.get("probe_generations") == 12
                and policy.get("maximum_attempts_per_candidate") == 1
                and policy.get("arm_kind", "BASELINE") in {
                    "BASELINE", "ALTERNATIVE_SOURCE_ARM",
                }
                and len(trials) == 2
                and len(attempts) == 2
                and len(artifacts) == 2
                and len(evaluations) == 2
                and len(decisions) == 1
                and len({str(row["candidate_rank"]) for row in trials}) == 2
                and all(row["state"] not in {
                    "PENDING", "CLAIMED", "RUNNING", "UPLOADED",
                    "EVALUATING",
                } for row in trials)
                and all(row["state"] not in {
                    "CLAIMED", "RUNNING",
                } for row in attempts)
            )
            ordered_evaluation_hashes: list[str] = []
            for rank, trial in enumerate(trials):
                try:
                    candidate = _load(trial["candidate_json"])
                    locked_plan = _load(trial["locked_plan_json"])
                    trial_result = _load(trial["result_json"])
                except (TypeError, json.JSONDecodeError):
                    candidate = locked_plan = trial_result = None
                trial_attempts = [
                    row for row in attempts
                    if row["probe_trial_id"] == trial["probe_trial_id"]
                ]
                trial_artifacts = [
                    row for row in artifacts
                    if row["probe_trial_id"] == trial["probe_trial_id"]
                ]
                trial_evaluations = [
                    row for row in evaluations
                    if row["probe_trial_id"] == trial["probe_trial_id"]
                ]
                trial_ok = (
                    rank < len(candidate_set or [])
                    and trial["candidate_rank"] == rank
                    and isinstance(candidate, dict)
                    and candidate == candidate_set[rank]
                    and trial["candidate_id"] == candidate.get("candidate_id")
                    and isinstance(locked_plan, dict)
                    and content_sha256(locked_plan)
                        == trial["locked_plan_sha256"]
                    and locked_plan.get("probe_run_id") == run_id
                    and locked_plan.get("probe_trial_id")
                        == trial["probe_trial_id"]
                    and locked_plan.get("candidate_id")
                        == trial["candidate_id"]
                    and locked_plan.get("profile_file_sha256") == profile_sha
                    and locked_plan.get("allow_unvalidated") is False
                    and trial["state"] == "COMPLETED"
                    and isinstance(trial_result, dict)
                    and trial_result.get("probe_trial_id")
                        == trial["probe_trial_id"]
                    and trial_result.get("candidate_id")
                        == trial["candidate_id"]
                    and trial_result.get("state") == "COMPLETED"
                    and len(trial_attempts) == 1
                    and len(trial_artifacts) == 1
                    and len(trial_evaluations) == 1
                )
                if trial_ok:
                    probe_attempt = trial_attempts[0]
                    artifact = trial_artifacts[0]
                    evaluation = trial_evaluations[0]
                    try:
                        growth = _load(probe_attempt["growth_receipt_json"])
                        attempt_result = _load(probe_attempt["result_json"])
                        artifact_manifest = _load(artifact["manifest_json"])
                        evaluation_result = _load(evaluation["result_json"])
                    except (TypeError, json.JSONDecodeError):
                        growth = attempt_result = artifact_manifest = None
                        evaluation_result = None
                    trial_ok = (
                        probe_attempt["attempt_number"] == 1
                        and probe_attempt["state"] == "COMPLETED"
                        and isinstance(growth, dict)
                        and growth.get("probe_run_id") == run_id
                        and growth.get("probe_trial_id")
                            == trial["probe_trial_id"]
                        and growth.get("probe_attempt_id")
                            == probe_attempt["probe_attempt_id"]
                        and growth.get("locked_plan_sha256")
                            == trial["locked_plan_sha256"]
                        and isinstance(attempt_result, dict)
                        and attempt_result.get("probe_attempt_id")
                            == probe_attempt["probe_attempt_id"]
                        and attempt_result.get("outcome") == "COMPLETED"
                        and artifact["probe_attempt_id"]
                            == probe_attempt["probe_attempt_id"]
                        and artifact["state"] == "RETAINED"
                        and isinstance(artifact["artifact_uri"], str)
                        and bool(artifact["artifact_uri"])
                        and isinstance(artifact_manifest, dict)
                        and content_sha256(artifact_manifest)
                            == artifact["manifest_sha256"]
                        and artifact_manifest.get("probe_run_id") == run_id
                        and artifact_manifest.get("probe_trial_id")
                            == trial["probe_trial_id"]
                        and artifact_manifest.get("locked_plan_sha256")
                            == trial["locked_plan_sha256"]
                        and isinstance(artifact_manifest.get("files"), dict)
                        and artifact_manifest.get("artifact_sha256")
                            == content_sha256(artifact_manifest.get("files"))
                        and artifact_manifest.get("noncanonical") is True
                        and artifact_manifest.get("ink_used") is False
                        and evaluation["probe_artifact_set_id"]
                            == artifact["probe_artifact_set_id"]
                        and evaluation["profile_sha256"] == profile_sha
                        and evaluation["verdict"] == "ELIGIBLE"
                        and isinstance(evaluation_result, dict)
                        and content_sha256(evaluation_result)
                            == evaluation["result_sha256"]
                        and evaluation_result.get("evaluation_id")
                            == evaluation["evaluation_id"]
                        and evaluation_result.get("probe_trial_id")
                            == trial["probe_trial_id"]
                        and evaluation_result.get("probe_artifact_set_id")
                            == artifact["probe_artifact_set_id"]
                        and evaluation_result.get("artifact_sha256")
                            == artifact_manifest.get("artifact_sha256")
                        and evaluation_result.get("profile_sha256")
                            == profile_sha
                        and evaluation_result.get("verdict") == "ELIGIBLE"
                        and evaluation_result.get("ink_used") is False
                    )
                    if trial_ok:
                        ordered_evaluation_hashes.append(
                            str(evaluation["result_sha256"])
                        )
                run_complete = run_complete and trial_ok
            expected_evidence_sha = content_sha256({
                "probe_run_id": run_id,
                "ordered_evaluation_sha256s": ordered_evaluation_hashes,
            })
            if len(decisions) == 1:
                decision = decisions[0]
                try:
                    receipt = _load(decision["receipt_json"])
                except (TypeError, json.JSONDecodeError):
                    receipt = None
                run_complete = run_complete and (
                    decision["policy_id"] == run["policy_id"]
                    and decision["policy_sha256"] == run["policy_sha256"]
                    and decision["evidence_set_sha256"]
                        == expected_evidence_sha
                    and decision["action"] == "ABSTAIN"
                    and decision["winner_trial_id"] is None
                    and isinstance(receipt, dict)
                    and content_sha256(receipt) == decision["receipt_sha256"]
                    and receipt.get("probe_run_id") == run_id
                    and receipt.get("action") == decision["action"]
                    and receipt.get("winner_trial_id") is None
                    and receipt.get("policy_id") == run["policy_id"]
                    and receipt.get("policy_sha256") == run["policy_sha256"]
                    and receipt.get("evidence_set_sha256")
                        == expected_evidence_sha
                )
            else:
                run_complete = False
            if policy.get("arm_kind") == "ALTERNATIVE_SOURCE_ARM" and not (
                policy.get("experimental_arm_admission_id")
                and policy.get("experimental_arm_admission_sha256")
            ):
                run_complete = False
            if not run_complete:
                complete = False
                reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            retained.append({
                "graph_kind": "LEGACY_PROBE_RUN",
                "task": (
                    self._history_row_projection(task)
                    if task is not None else None
                ),
                "created_by_attempt": (
                    self._history_row_projection(created_by_attempt)
                    if created_by_attempt is not None else None
                ),
                "source_snapshot": (
                    self._history_row_projection(source_row)
                    if source_row is not None else None
                ),
                "profile_file_sha256": profile_sha,
                "probe_run": self._history_row_projection(run),
                "probe_trials": [
                    self._history_row_projection(row) for row in trials
                ],
                "probe_attempts": [
                    self._history_row_projection(row) for row in attempts
                ],
                "probe_artifact_sets": [
                    self._history_row_projection(row) for row in artifacts
                ],
                "probe_evaluations": [
                    self._history_row_projection(row) for row in evaluations
                ],
                "probe_decisions": [
                    self._history_row_projection(row) for row in decisions
                ],
                "retained_row_ids": {
                    "task_id": run["task_id"],
                    "attempt_id": run["created_by_attempt_id"],
                    "probe_run_id": run_id,
                    "probe_trial_ids": [
                        str(row["probe_trial_id"]) for row in trials
                    ],
                    "probe_attempt_ids": [
                        str(row["probe_attempt_id"]) for row in attempts
                    ],
                    "probe_artifact_set_ids": [
                        str(row["probe_artifact_set_id"])
                        for row in artifacts
                    ],
                    "evaluation_ids": [
                        str(row["evaluation_id"]) for row in evaluations
                    ],
                    "decision_ids": [
                        str(row["decision_id"]) for row in decisions
                    ],
                },
                "logical_execution_id": run_id,
                "producer_kind": "LEGACY_PROBE_RUN",
                "source_snapshot_sha256": (
                    source.get("source_snapshot_sha256")
                    if isinstance(source, dict) else None
                ),
                "item_id": task["cell_id"] if task is not None else None,
                "work_kind": (
                    policy.get("arm_kind", "BASELINE")
                    if isinstance(policy, dict) else None
                ),
                "logical_units": 24 if run_complete else 0,
            })
        legacy_orphan_queries = {
            "LEGACY_ORPHAN_ARTIFACT": """
                SELECT a.* FROM probe_artifact_sets a
                LEFT JOIN probe_trials t
                  ON t.probe_trial_id=a.probe_trial_id
                LEFT JOIN probe_attempts p
                  ON p.probe_attempt_id=a.probe_attempt_id
                WHERE t.probe_trial_id IS NULL
                   OR p.probe_attempt_id IS NULL
                ORDER BY a.probe_artifact_set_id
            """,
            "LEGACY_ORPHAN_EVALUATION": """
                SELECT e.* FROM probe_evaluations e
                LEFT JOIN probe_trials t
                  ON t.probe_trial_id=e.probe_trial_id
                LEFT JOIN probe_artifact_sets a
                  ON a.probe_artifact_set_id=e.probe_artifact_set_id
                WHERE t.probe_trial_id IS NULL
                   OR a.probe_artifact_set_id IS NULL
                ORDER BY e.evaluation_id
            """,
            "LEGACY_ORPHAN_DECISION": """
                SELECT d.* FROM probe_decisions d
                LEFT JOIN probe_runs r ON r.probe_run_id=d.probe_run_id
                WHERE r.probe_run_id IS NULL
                ORDER BY d.decision_id
            """,
        }
        for graph_kind, query in legacy_orphan_queries.items():
            for orphan in connection.execute(query).fetchall():
                complete = False
                reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
                retained.append({
                    "graph_kind": graph_kind,
                    "row": self._history_row_projection(orphan),
                    "logical_units": 0,
                })
        v16_root_rows = connection.execute(
            """SELECT r.reservation_id,w.reservation_id AS work_reservation_id,
                      r.*
                 FROM first_letters_discovery_compute_reservations r
                 LEFT JOIN first_letters_discovery_work_bindings w
                   ON w.reservation_id=r.reservation_id
                 LEFT JOIN first_letters_discovery_native_adapters_v19 a
                   ON a.reservation_id=r.reservation_id
                WHERE r.mission_id=? AND a.reservation_id IS NULL
                  AND r.source!='IMPORTED_HISTORICAL_EXACT'
                  AND r.work_kind IN ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')
                ORDER BY r.reservation_id""",
            (mission_id,),
        ).fetchall()
        orphan_work_rows = connection.execute(
            """SELECT w.* FROM first_letters_discovery_work_bindings w
                 LEFT JOIN first_letters_discovery_compute_reservations r
                   ON r.reservation_id=w.reservation_id
                WHERE w.mission_id=? AND r.reservation_id IS NULL
                  AND w.work_kind IN ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')
                ORDER BY w.reservation_id""",
            (mission_id,),
        ).fetchall()
        for root in v16_root_rows:
            if root["work_reservation_id"] is not None:
                continue
            complete = False
            reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            retained.append({
                "graph_kind": "V16_ORPHAN_RESERVATION",
                "reservation": self._history_row_projection(root),
                "logical_units": 0,
            })
        for root in orphan_work_rows:
            complete = False
            reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            retained.append({
                "graph_kind": "V16_ORPHAN_WORK",
                "work": self._history_row_projection(root),
                "logical_units": 0,
            })
        v16_rows = connection.execute(
            """SELECT r.*,w.work_json,w.work_sha256,w.mission_id AS w_mission_id,
                      w.request_id AS w_request_id,w.work_kind AS w_work_kind,
                      w.dispatch_kind,w.reservation_id AS w_reservation_id
                 FROM first_letters_discovery_compute_reservations r
                 JOIN first_letters_discovery_work_bindings w
                   ON w.reservation_id=r.reservation_id
                 LEFT JOIN first_letters_discovery_native_adapters_v19 a
                   ON a.reservation_id=r.reservation_id
                WHERE r.mission_id=? AND a.reservation_id IS NULL
                  AND r.source!='IMPORTED_HISTORICAL_EXACT'
                  AND r.work_kind IN ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')
                ORDER BY r.reservation_id""",
            (mission_id,),
        ).fetchall()
        for reservation_row in v16_rows:
            reservation = _load(reservation_row["reservation_json"])
            work = _load(reservation_row["work_json"])
            reservation_complete = (
                isinstance(reservation, dict)
                and isinstance(work, dict)
                and reservation.get("reservation_id")
                    == reservation_row["reservation_id"]
                and reservation.get("mission_id") == mission_id
                and reservation.get("request_id")
                    == reservation_row["request_id"]
                and reservation.get("work_kind")
                    == reservation_row["work_kind"]
                and reservation.get("reservation_sha256")
                    == reservation_row["reservation_sha256"]
                and reservation.get("reservation_sha256") == content_sha256({
                    key: value for key, value in reservation.items()
                    if key not in {"reservation_sha256", "created_at"}
                })
                and work.get("reservation_id")
                    == reservation_row["reservation_id"]
                and work.get("reservation_sha256")
                    == reservation["reservation_sha256"]
                and work.get("mission_id") == reservation["mission_id"]
                and work.get("request_id") == reservation["request_id"]
                and work.get("work_kind") == reservation["work_kind"]
                and work.get("work_sha256") == reservation_row["work_sha256"]
                and work.get("work_sha256") == content_sha256({
                    key: value for key, value in work.items()
                    if key != "work_sha256"
                })
                and work.get("ordered_item_ids")
                    == reservation.get("ordered_item_ids")
                and reservation_row["w_reservation_id"]
                    == reservation_row["reservation_id"]
                and reservation_row["w_mission_id"] == mission_id
                and reservation_row["w_request_id"]
                    == reservation_row["request_id"]
                and reservation_row["w_work_kind"]
                    == reservation_row["work_kind"]
                and reservation_row["dispatch_kind"] == work.get(
                    "dispatch_kind"
                )
            )
            evidence_runs = connection.execute(
                """SELECT * FROM first_letters_discovery_evidence_runs
                    WHERE reservation_id=? ORDER BY cell_id,run_id""",
                (reservation_row["reservation_id"],),
            ).fetchall()
            reservation_complete = reservation_complete and (
                len(evidence_runs) == reservation.get("item_count")
                and [str(row["cell_id"]) for row in evidence_runs]
                    == sorted(reservation.get("ordered_item_ids") or [])
            )
            if not evidence_runs:
                complete = False
                reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
                retained.append({
                    "graph_kind": "V16_DISCOVERY_EVIDENCE",
                    "reservation": self._history_row_projection(reservation_row),
                    "run": None, "executor_claims": [],
                    "evidence_sets": [], "evidence_files": [],
                    "retained_row_ids": {
                        "reservation_id": reservation_row["reservation_id"],
                        "run_id": None,
                    },
                    "logical_execution_id": None,
                    "producer_kind": (
                        "BASELINE_RECONCILIATION"
                        if reservation_row["work_kind"] == "BASELINE_ARM"
                        else "EXPERIMENTAL_ARM_ADMISSION"
                    ),
                    "source_snapshot_sha256": None,
                    "profile_file_sha256": None,
                    "item_id": None, "logical_units": 0,
                })
                continue
            for evidence_run in evidence_runs:
                claims = connection.execute(
                    """SELECT * FROM first_letters_discovery_executor_claims
                        WHERE run_id=? ORDER BY claim_id""",
                    (evidence_run["run_id"],),
                ).fetchall()
                evidence_sets = connection.execute(
                    """SELECT * FROM first_letters_discovery_evidence_sets
                        WHERE run_id=? ORDER BY evidence_set_id""",
                    (evidence_run["run_id"],),
                ).fetchall()
                files = []
                if len(evidence_sets) == 1:
                    files = connection.execute(
                        """SELECT * FROM first_letters_discovery_evidence_files
                            WHERE evidence_set_id=?
                            ORDER BY file_order,relative_path""",
                        (evidence_sets[0]["evidence_set_id"],),
                    ).fetchall()
                run_authority = _load(evidence_run["run_authority_json"])
                claim = _load(claims[0]["claim_json"]) if len(claims) == 1 else {}
                evidence = (
                    _load(evidence_sets[0]["evidence_json"])
                    if len(evidence_sets) == 1 else {}
                )
                file_roles = [str(row["role"]) for row in files]
                expected_paths = [
                    f"probes/{evidence_run['run_id']}/provider-request.json",
                    f"probes/{evidence_run['run_id']}/provider-response.json",
                    f"probes/{evidence_run['run_id']}/selection-policy-receipt.json",
                ]
                files_complete = (
                    file_roles == [
                        "CANDIDATE_PROVIDER_REQUEST",
                        "CANDIDATE_PROVIDER_RESPONSE",
                        "DISCOVERY_SELECTION_POLICY_RECEIPT",
                    ]
                    and [str(row["relative_path"]) for row in files]
                        == expected_paths
                    and [int(row["file_order"]) for row in files] == [0, 1, 2]
                    and all(
                        len(bytes(row["payload"])) == int(row["byte_count"])
                        and hashlib.sha256(bytes(row["payload"])).hexdigest()
                            == row["sha256"]
                        for row in files
                    )
                )
                provider_request = _load(evidence_run["provider_request_json"])
                authority = work.get("work_authority") or {}
                bindings = [
                    value for value in authority.get("ordered_item_bindings", [])
                    if value.get("item_id") == evidence_run["cell_id"]
                ]
                source_row = connection.execute(
                    "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                    (evidence_run["source_snapshot_id"],),
                ).fetchone()
                source = self._snapshot(source_row) if source_row is not None else {}
                task_row = None
                attempt_row = None
                if len(bindings) == 1 and bindings[0].get("parent_task_id"):
                    task_row = connection.execute(
                        "SELECT * FROM tasks WHERE task_id=?",
                        (bindings[0]["parent_task_id"],),
                    ).fetchone()
                if len(bindings) == 1 and bindings[0].get("parent_attempt_id"):
                    attempt_row = connection.execute(
                        "SELECT * FROM attempts WHERE attempt_id=?",
                        (bindings[0]["parent_attempt_id"],),
                    ).fetchone()
                try:
                    from .seed_probe import (
                        load_first_letters_discovery_profile_bytes,
                    )
                    profile = load_first_letters_discovery_profile_bytes(
                        bytes(evidence_run["profile_bytes"])
                    )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    profile = {}
                binding = bindings[0] if len(bindings) == 1 else {}
                try:
                    current_profile = (
                        self._first_letters_discovery_profile_resolver(
                            mission_id, evidence_run["source_snapshot_id"]
                        )
                        if self._first_letters_discovery_profile_resolver is not None
                        else None
                    )
                except Exception:
                    current_profile = None
                profile_complete = (
                    current_profile == bytes(evidence_run["profile_bytes"])
                    and profile.get("source_snapshot_id")
                        == evidence_run["source_snapshot_id"]
                    and profile.get("source_snapshot_sha256")
                        == source.get("source_snapshot_sha256")
                    and profile.get("source_content_lock_sha256")
                        == source.get("source_content_lock_sha256")
                    and profile.get("ct_metadata_sha256")
                        == source.get("ct_metadata_sha256")
                    and profile.get("ct_read_set_manifest_sha256")
                        == source.get("ct_read_set_manifest_sha256")
                    and profile.get("m7_read_set_manifest_sha256")
                        == source.get("m7_read_set_manifest_sha256")
                    and profile.get("m7_model_id") == source.get("m7_model_id")
                    and profile.get("m7_resolution") == source.get("m7_resolution")
                    and profile.get("m7_level") == source.get("m7_level")
                    and profile.get("m7_threshold") == source.get("m7_threshold")
                    and profile.get("m7_transform_sha256")
                        == source.get("m7_transform_sha256")
                    and profile.get("canonical_ordered_cell_set_sha256")
                        == content_sha256(reservation.get("ordered_item_ids"))
                    and profile.get("mission_compute_cap_authority_id")
                        == reservation.get("cap_authority_id")
                    and profile.get("mission_compute_cap_authority_sha256")
                        == reservation.get("cap_authority_sha256")
                )
                if reservation_row["work_kind"] == "ALTERNATIVE_SOURCE_ARM":
                    arm_id = profile.get("experimental_arm_admission_id")
                    try:
                        from .seed_probe import (
                            validate_experimental_arm_admission,
                        )
                        arm = (
                            validate_experimental_arm_admission(
                                self._first_letters_experimental_arm_resolver(
                                    arm_id
                                )
                            )
                            if self._first_letters_experimental_arm_resolver
                                is not None
                            and isinstance(arm_id, str) else None
                        )
                    except Exception:
                        arm = None
                    profile_complete = profile_complete and (
                        isinstance(arm, dict)
                        and arm.get("admission_sha256")
                            == profile.get("experimental_arm_admission_sha256")
                        and arm.get("source_snapshot_id")
                            == evidence_run["source_snapshot_id"]
                        and arm.get("source_snapshot_sha256")
                            == source.get("source_snapshot_sha256")
                    )
                parent_complete = (
                    len(bindings) == 1
                    and source.get("source_snapshot_sha256")
                        == binding.get("source_snapshot_sha256")
                    and source.get("source_snapshot_sha256")
                        == authority.get("source_sha256")
                    and source.get("sample_id") == binding.get("sample_id")
                    and (
                        binding.get("parent_task_id") is None
                        or (
                            task_row is not None
                            and task_row["mission_id"] == mission_id
                            and task_row["source_snapshot_id"]
                                == evidence_run["source_snapshot_id"]
                            and task_row["cell_id"] == evidence_run["cell_id"]
                            and (_load(task_row["payload_json"]) or {}).get(
                                "scientific_opportunity_id"
                            ) == binding.get("scientific_opportunity_id")
                            and (_load(task_row["payload_json"]) or {}).get(
                                "accepted_p0_artifact_id",
                                (_load(task_row["payload_json"]) or {}).get(
                                    "p0_artifact_id"
                                ),
                            ) == binding.get("accepted_p0_artifact_id")
                            and (_load(task_row["payload_json"]) or {}).get(
                                "accepted_p0_artifact_sha256",
                                (_load(task_row["payload_json"]) or {}).get(
                                    "p0_artifact_sha256"
                                ),
                            ) == binding.get("accepted_p0_artifact_sha256")
                            and (
                                (_load(task_row["payload_json"]) or {}).get(
                                    "candidate_discovery"
                                ) or {}
                            ).get("region") == binding.get("cell_region")
                        )
                    )
                    and (
                        binding.get("parent_attempt_id") is None
                        or (
                            attempt_row is not None
                            and attempt_row["task_id"]
                                == binding.get("parent_task_id")
                        )
                    )
                )
                try:
                    from .seed_probe import (
                        _derive_first_letters_discovery_run_authority,
                    )
                    current_derived = (
                        _derive_first_letters_discovery_run_authority(
                            profile_bytes=bytes(evidence_run["profile_bytes"]),
                            reservation=reservation, work=work,
                            binding=binding, source=source,
                        )
                    )
                    retained_run_core = {
                        key: value for key, value in run_authority.items()
                        if key not in {
                            "run_authority_sha256", "worker_id",
                            "executor_claim",
                        }
                    }
                    current_run_core = {
                        key: value for key, value in
                        current_derived["run_authority"].items()
                        if key != "run_authority_sha256"
                    }
                    current_science_complete = (
                        provider_request == current_derived["provider_request"]
                        and retained_run_core == current_run_core
                    )
                except Exception:
                    current_science_complete = False
                run_complete = (
                    reservation_complete
                    and profile_complete and parent_complete
                    and current_science_complete
                    and evidence_run["state"] == "COMPLETED"
                    and evidence_run["reservation_id"]
                        == reservation_row["reservation_id"]
                    and evidence_run["mission_id"] == mission_id
                    and evidence_run["request_id"]
                        == reservation_row["request_id"]
                    and evidence_run["cell_id"] == run_authority.get("cell_id")
                    and evidence_run["source_snapshot_id"]
                        == run_authority.get("source_snapshot_id")
                    and evidence_run["profile_file_sha256"]
                        == run_authority.get("profile_file_sha256")
                    and run_authority.get("provider_request_sha256")
                        == content_sha256(provider_request)
                    and len(claims) == 1 and claims[0]["state"] == "COMPLETED"
                    and claim.get("claim_id") == claims[0]["claim_id"]
                    and claim.get("claim_sha256") == claims[0]["claim_sha256"]
                    and claim.get("claim_sha256") == content_sha256({
                        key: value for key, value in claim.items()
                        if key != "claim_sha256"
                    })
                    and claims[0]["run_id"] == evidence_run["run_id"]
                    and claims[0]["worker_id"] == claim.get("worker_id")
                    and claims[0]["executor_id"] == claim.get("executor_id")
                    and claims[0]["executor_sha256"]
                        == claim.get("executor_sha256")
                    and claims[0]["capability"] == claim.get("capability")
                    and claims[0]["claim_attempt_number"]
                        == claim.get("claim_attempt_number")
                    and claims[0]["execution_lease_token_sha256"]
                        == claim.get("execution_lease_token_sha256")
                    and len(evidence_sets) == 1
                    and evidence.get("evidence_set_id")
                        == evidence_sets[0]["evidence_set_id"]
                    and evidence_sets[0]["run_id"] == evidence_run["run_id"]
                    and content_sha256(evidence)
                        == evidence_sets[0]["evidence_set_sha256"]
                    and run_authority.get("run_id") == evidence_run["run_id"]
                    and run_authority.get("run_authority_sha256")
                        == evidence_run["run_authority_sha256"]
                    and run_authority.get("run_authority_sha256")
                        == content_sha256({
                            key: value for key, value in run_authority.items()
                            if key != "run_authority_sha256"
                        })
                    and hashlib.sha256(
                        bytes(evidence_run["profile_bytes"])
                    ).hexdigest() == evidence_run["profile_file_sha256"]
                    and files_complete
                )
                if not run_complete:
                    complete = False
                    reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
                retained.append({
                    "graph_kind": "V16_DISCOVERY_EVIDENCE",
                    "reservation": self._history_row_projection(reservation_row),
                    "run": self._history_row_projection(evidence_run),
                    "executor_claims": [
                        self._history_row_projection(row) for row in claims
                    ],
                    "evidence_sets": [
                        self._history_row_projection(row) for row in evidence_sets
                    ],
                    "evidence_files": [
                        self._history_row_projection(row) for row in files
                    ],
                    "source_snapshot": (
                        self._history_row_projection(source_row)
                        if source_row is not None else None
                    ),
                    "parent_task": (
                        self._history_row_projection(task_row)
                        if task_row is not None else None
                    ),
                    "parent_attempt": (
                        self._history_row_projection(attempt_row)
                        if attempt_row is not None else None
                    ),
                    "profile_file_sha256": evidence_run[
                        "profile_file_sha256"
                    ],
                    "retained_row_ids": {
                        "reservation_id": reservation_row["reservation_id"],
                        "run_id": evidence_run["run_id"],
                        "claim_ids": [str(row["claim_id"]) for row in claims],
                        "evidence_set_ids": [
                            str(row["evidence_set_id"]) for row in evidence_sets
                        ],
                        "evidence_file_paths": [
                            str(row["relative_path"]) for row in files
                        ],
                    },
                    "logical_execution_id": evidence_run["run_id"],
                    "producer_kind": (
                        "BASELINE_RECONCILIATION"
                        if reservation_row["work_kind"] == "BASELINE_ARM"
                        else "EXPERIMENTAL_ARM_ADMISSION"
                    ),
                    "source_snapshot_sha256": source.get(
                        "source_snapshot_sha256"
                    ),
                    "profile_file_sha256": evidence_run[
                        "profile_file_sha256"
                    ],
                    "item_id": evidence_run["cell_id"],
                    "logical_units": 24 if run_complete else 0,
                })
        manifest = {
            "schema":
                "campaignx.first_letters_discovery_history_manifest.v1",
            "mission_id": mission_id,
            "legacy_probe_run_ids": [
                str(row["probe_run_id"]) for row in run_rows
            ],
            "legacy_v16_reservation_ids": [
                str(row["reservation_id"]) for row in v16_rows
            ],
            "retained_execution_graphs": retained,
            "allow_unvalidated": False,
        }
        return manifest, complete, reason

    def _discovery_historical_materialization_complete_tx(
        self, connection: sqlite3.Connection, *, mission_id: str,
        manifest_sha256: str, graphs: list[dict[str, Any]],
    ) -> bool:
        imported_graphs = [
            graph for graph in graphs
            if graph.get("graph_kind") in {
                "LEGACY_PROBE_RUN", "V16_DISCOVERY_EVIDENCE",
            }
        ]
        import_rows = connection.execute(
            "SELECT * FROM first_letters_discovery_historical_imports_v19 "
            "WHERE mission_id=? ORDER BY logical_execution_id,import_id",
            (mission_id,),
        ).fetchall()
        if len(import_rows) != len(imported_graphs):
            return False
        imports_by_execution = {
            str(row["logical_execution_id"]): row for row in import_rows
        }
        if len(imports_by_execution) != len(import_rows):
            return False
        legacy_reservation_ids: set[str] = set()
        for graph in imported_graphs:
            logical_execution_id = str(graph["logical_execution_id"])
            row = imports_by_execution.get(logical_execution_id)
            if row is None:
                return False
            projection_sha = content_sha256(graph)
            expected_reservation_id = (
                str(row["reservation_id"])
                if graph["graph_kind"] == "LEGACY_PROBE_RUN"
                else str(graph["retained_row_ids"]["reservation_id"])
            )
            expected_import_id = stable_id(
                "first-letters-discovery-historical-import",
                {
                    "mission_id": mission_id,
                    "logical_execution_id": logical_execution_id,
                    "retained_projection_sha256": projection_sha,
                },
            )
            try:
                imported = _load(row["import_json"])
            except (TypeError, json.JSONDecodeError):
                return False
            import_core = {
                key: value for key, value in imported.items()
                if key not in {"import_sha256", "created_at"}
            } if isinstance(imported, dict) else {}
            if (
                not isinstance(imported, dict)
                or row["import_id"] != expected_import_id
                or imported.get("import_id") != expected_import_id
                or row["reservation_id"] != expected_reservation_id
                or imported.get("reservation_id") != expected_reservation_id
                or row["mission_id"] != mission_id
                or imported.get("mission_id") != mission_id
                or row["logical_execution_id"] != logical_execution_id
                or imported.get("logical_execution_id")
                    != logical_execution_id
                or row["producer_kind"] != graph["producer_kind"]
                or imported.get("producer_kind") != graph["producer_kind"]
                or row["source_snapshot_sha256"]
                    != graph["source_snapshot_sha256"]
                or imported.get("source_snapshot_sha256")
                    != graph["source_snapshot_sha256"]
                or row["profile_file_sha256"]
                    != graph["profile_file_sha256"]
                or imported.get("profile_file_sha256")
                    != graph["profile_file_sha256"]
                or row["item_id"] != graph["item_id"]
                or imported.get("item_id") != graph["item_id"]
                or row["fixed_units"] != 24
                or imported.get("fixed_units") != 24
                or _load(row["retained_row_ids_json"])
                    != graph["retained_row_ids"]
                or imported.get("retained_row_ids")
                    != graph["retained_row_ids"]
                or row["retained_projection_sha256"] != projection_sha
                or imported.get("retained_projection_sha256")
                    != projection_sha
                or row["history_manifest_sha256"] != manifest_sha256
                or imported.get("history_manifest_sha256")
                    != manifest_sha256
                or row["import_sha256"] != imported.get("import_sha256")
                or row["import_sha256"] != content_sha256(import_core)
            ):
                return False
            if graph["graph_kind"] != "LEGACY_PROBE_RUN":
                continue
            legacy_reservation_ids.add(expected_reservation_id)
            materialized = connection.execute(
                """SELECT r.*,w.mission_id AS w_mission_id,
                          w.request_id AS w_request_id,
                          w.work_kind AS w_work_kind,
                          w.dispatch_kind,w.work_json,w.work_sha256
                     FROM first_letters_discovery_compute_reservations r
                     JOIN first_letters_discovery_work_bindings w
                       ON w.reservation_id=r.reservation_id
                    WHERE r.reservation_id=?""",
                (expected_reservation_id,),
            ).fetchall()
            if len(materialized) != 1:
                return False
            materialized_row = materialized[0]
            try:
                reservation = _load(materialized_row["reservation_json"])
                work = _load(materialized_row["work_json"])
            except (TypeError, json.JSONDecodeError):
                return False
            if (
                not isinstance(reservation, dict)
                or not isinstance(work, dict)
                or materialized_row["source"]
                    != "IMPORTED_HISTORICAL_EXACT"
                or reservation.get("source")
                    != "IMPORTED_HISTORICAL_EXACT"
                or materialized_row["mission_id"] != mission_id
                or reservation.get("mission_id") != mission_id
                or materialized_row["reservation_id"]
                    != reservation.get("reservation_id")
                or materialized_row["reservation_sha256"]
                    != reservation.get("reservation_sha256")
                or materialized_row["reservation_sha256"]
                    != content_sha256({
                        key: value for key, value in reservation.items()
                        if key not in {"reservation_sha256", "created_at"}
                    })
                or materialized_row["item_count"] != 1
                or reservation.get("item_count") != 1
                or materialized_row["units_per_item"] != 24
                or reservation.get("units_per_item") != 24
                or materialized_row["reserved_units"] != 24
                or reservation.get("reserved_units") != 24
                or reservation.get("reserved_after_units")
                    - reservation.get("reserved_before_units") != 24
                or reservation.get("ordered_item_ids") != [graph["item_id"]]
                or materialized_row["ordered_item_ids_sha256"]
                    != content_sha256([graph["item_id"]])
                or materialized_row["w_mission_id"] != mission_id
                or materialized_row["w_request_id"]
                    != materialized_row["request_id"]
                or materialized_row["w_work_kind"]
                    != materialized_row["work_kind"]
                or materialized_row["dispatch_kind"]
                    != "HISTORICAL_IMPORT_BINDING"
                or work.get("dispatch_kind")
                    != "HISTORICAL_IMPORT_BINDING"
                or work.get("reservation_id") != expected_reservation_id
                or work.get("reservation_sha256")
                    != reservation.get("reservation_sha256")
                or work.get("mission_id") != mission_id
                or work.get("request_id") != reservation.get("request_id")
                or work.get("work_kind") != reservation.get("work_kind")
                or work.get("ordered_item_ids") != [graph["item_id"]]
                or materialized_row["work_sha256"]
                    != work.get("work_sha256")
                or materialized_row["work_sha256"] != content_sha256({
                    key: value for key, value in work.items()
                    if key != "work_sha256"
                })
            ):
                return False
        imported_reservation_ids = {
            str(row["reservation_id"])
            for row in connection.execute(
                """SELECT reservation_id FROM
                   first_letters_discovery_compute_reservations
                   WHERE mission_id=? AND source='IMPORTED_HISTORICAL_EXACT'""",
                (mission_id,),
            ).fetchall()
        }
        return imported_reservation_ids == legacy_reservation_ids

    def reconcile_first_letters_discovery_history(
        self, *, mission_id: str,
    ) -> dict[str, Any]:
        if not isinstance(mission_id, str) or not mission_id:
            raise ValueError("discovery history mission ID is invalid")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            manifest, complete, reason = (
                self._derive_first_letters_discovery_history_tx(
                    connection, mission_id=mission_id,
                )
            )
            manifest_sha = content_sha256(manifest)
            prior = connection.execute(
                """SELECT reconciliation_json,manifest_sha256,state
                     FROM first_letters_discovery_history_reconciliations_v19
                    WHERE mission_id=?
                    ORDER BY created_at,rowid LIMIT 1""",
                (mission_id,),
            ).fetchone()
            if (prior is not None
                    and prior["state"] == "COMPLETE"
                    and prior["manifest_sha256"] != manifest_sha):
                complete = False
                reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            if (
                prior is not None
                and prior["state"] == "COMPLETE"
                and prior["manifest_sha256"] == manifest_sha
                and not self._discovery_historical_materialization_complete_tx(
                    connection, mission_id=mission_id,
                    manifest_sha256=manifest_sha,
                    graphs=manifest["retained_execution_graphs"],
                )
            ):
                complete = False
                reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            legacy_graphs = [
                graph for graph in manifest["retained_execution_graphs"]
                if graph.get("graph_kind") == "LEGACY_PROBE_RUN"
            ]
            if complete and legacy_graphs:
                imported_execution_ids = {
                    str(row["logical_execution_id"])
                    for row in connection.execute(
                        "SELECT logical_execution_id FROM "
                        "first_letters_discovery_historical_imports_v19 "
                        "WHERE mission_id=?", (mission_id,),
                    ).fetchall()
                }
                pending_legacy_graphs = [
                    graph for graph in legacy_graphs
                    if str(graph["logical_execution_id"])
                        not in imported_execution_ids
                ]
                cap_row = connection.execute(
                    "SELECT cap_units FROM "
                    "first_letters_discovery_compute_caps WHERE mission_id=?",
                    (mission_id,),
                ).fetchone()
                already_reserved = int(connection.execute(
                    "SELECT COALESCE(SUM(reserved_units),0) FROM "
                    "first_letters_discovery_compute_reservations "
                    "WHERE mission_id=?", (mission_id,),
                ).fetchone()[0])
                if (
                    cap_row is None
                    or already_reserved + 24 * len(pending_legacy_graphs)
                        > int(cap_row["cap_units"])
                ):
                    complete = False
                    reason = "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            state = "COMPLETE" if complete else "CONTROL_INCOMPLETE"
            fixed_units = sum(
                int(row["logical_units"])
                for row in manifest["retained_execution_graphs"]
            ) if complete else 0
            retained_graphs = manifest["retained_execution_graphs"]
            v16_graphs = [
                graph for graph in retained_graphs
                if graph.get("graph_kind") == "V16_DISCOVERY_EVIDENCE"
            ]
            retained_membership = {
                "reservation_ids": sorted({
                    str(graph["retained_row_ids"]["reservation_id"])
                    for graph in v16_graphs
                }),
                "work_binding_ids": sorted({
                    str(graph["retained_row_ids"]["reservation_id"])
                    for graph in v16_graphs
                }),
                "run_ids": sorted({
                    str(graph["retained_row_ids"]["run_id"])
                    for graph in v16_graphs
                    if graph["retained_row_ids"].get("run_id") is not None
                }),
                "claim_ids": sorted({
                    str(value) for graph in v16_graphs
                    for value in graph["retained_row_ids"].get("claim_ids", [])
                }),
                "evidence_set_ids": sorted({
                    str(value) for graph in v16_graphs
                    for value in graph["retained_row_ids"].get(
                        "evidence_set_ids", []
                    )
                }),
                "evidence_files": sorted([
                    {
                        "evidence_set_id": str(row["evidence_set_id"]),
                        "file_order": int(row["file_order"]),
                        "relative_path": str(row["relative_path"]),
                        "sha256": str(row["sha256"]),
                    }
                    for graph in v16_graphs
                    for row in graph.get("evidence_files", [])
                ], key=lambda value: (
                    value["evidence_set_id"], value["file_order"],
                    value["relative_path"], value["sha256"],
                )),
                "source_snapshot_ids": sorted({
                    str(graph["source_snapshot"]["source_snapshot_id"])
                    for graph in v16_graphs
                    if graph.get("source_snapshot") is not None
                }),
                "parent_task_ids": sorted({
                    str(graph["parent_task"]["task_id"])
                    for graph in v16_graphs
                    if graph.get("parent_task") is not None
                }),
                "parent_attempt_ids": sorted({
                    str(graph["parent_attempt"]["attempt_id"])
                    for graph in v16_graphs
                    if graph.get("parent_attempt") is not None
                }),
                "profile_file_sha256s": sorted({
                    str(graph["profile_file_sha256"])
                    for graph in v16_graphs
                }),
                "producer_authorities": sorted([
                    {
                        "producer_kind": str(graph["producer_kind"]),
                        "logical_execution_id": str(
                            graph["logical_execution_id"]
                        ),
                    } for graph in v16_graphs
                ], key=lambda value: (
                    value["producer_kind"], value["logical_execution_id"]
                )),
                "retained_projection_sha256s": sorted(
                    content_sha256(graph) for graph in retained_graphs
                ),
                "legacy_task_ids": sorted({
                    str(graph["retained_row_ids"]["task_id"])
                    for graph in legacy_graphs
                }),
                "legacy_attempt_ids": sorted({
                    str(graph["retained_row_ids"]["attempt_id"])
                    for graph in legacy_graphs
                }),
                "legacy_trial_ids": sorted({
                    str(value) for graph in legacy_graphs
                    for value in graph["retained_row_ids"][
                        "probe_trial_ids"
                    ]
                }),
                "legacy_probe_attempt_ids": sorted({
                    str(value) for graph in legacy_graphs
                    for value in graph["retained_row_ids"][
                        "probe_attempt_ids"
                    ]
                }),
                "legacy_artifact_set_ids": sorted({
                    str(value) for graph in legacy_graphs
                    for value in graph["retained_row_ids"][
                        "probe_artifact_set_ids"
                    ]
                }),
                "legacy_evaluation_ids": sorted({
                    str(value) for graph in legacy_graphs
                    for value in graph["retained_row_ids"]["evaluation_ids"]
                }),
                "legacy_decision_ids": sorted({
                    str(value) for graph in legacy_graphs
                    for value in graph["retained_row_ids"]["decision_ids"]
                }),
            }
            watermark = {
                "mission_id": mission_id,
                "legacy_probe_run_ids": manifest["legacy_probe_run_ids"],
                "legacy_v16_reservation_ids":
                    manifest["legacy_v16_reservation_ids"],
                "retained_graph_count": len(
                    manifest["retained_execution_graphs"]
                ),
                "retained_membership": retained_membership,
            }
            watermark_sha = content_sha256(watermark)
            reconciliation_id = stable_id(
                "first-letters-discovery-history-reconciliation",
                {"mission_id": mission_id, "manifest_sha256": manifest_sha,
                 "state": state},
            )
            created_at = utc_now()
            core = {
                "schema":
                    "campaignx.first_letters_discovery_history_reconciliation.v1",
                "reconciliation_id": reconciliation_id,
                "mission_id": mission_id,
                "state": state,
                "watermark": watermark,
                "watermark_sha256": watermark_sha,
                "manifest": manifest,
                "manifest_sha256": manifest_sha,
                "fixed_units": fixed_units,
                "reason": reason,
                "allow_unvalidated": False,
            }
            reconciliation = {
                **core, "reconciliation_sha256": content_sha256(core),
                "created_at": created_at,
            }
            existing = connection.execute(
                """SELECT reconciliation_json
                     FROM first_letters_discovery_history_reconciliations_v19
                    WHERE mission_id=? AND manifest_sha256=? AND state=?""",
                (mission_id, manifest_sha, state),
            ).fetchone()
            if existing is not None:
                return _load(existing["reconciliation_json"])
            connection.execute(
                """INSERT INTO
                   first_letters_discovery_history_reconciliations_v19
                   (reconciliation_id,mission_id,state,watermark_sha256,
                    manifest_json,manifest_sha256,fixed_units,reason,
                    reconciliation_json,reconciliation_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    reconciliation_id, mission_id, state, watermark_sha,
                    _dump(manifest), manifest_sha, fixed_units, reason,
                    _dump(reconciliation),
                    reconciliation["reconciliation_sha256"], created_at,
                ),
            )
            if complete:
                for graph in manifest["retained_execution_graphs"]:
                    graph_kind = graph.get("graph_kind")
                    if graph_kind not in {
                        "V16_DISCOVERY_EVIDENCE", "LEGACY_PROBE_RUN",
                    }:
                        continue
                    projection_sha = content_sha256(graph)
                    if graph_kind == "LEGACY_PROBE_RUN":
                        task = graph["task"]
                        task_payload = task["payload_json"]
                        source = graph["source_snapshot"]["payload_json"]
                        policy = graph["probe_run"]["policy_json"]
                        profile = policy["discovery_profile"]
                        cap_row = connection.execute(
                            "SELECT * FROM "
                            "first_letters_discovery_compute_caps "
                            "WHERE mission_id=?", (mission_id,),
                        ).fetchone()
                        work_kind = (
                            "ALTERNATIVE_SOURCE_ARM"
                            if graph["work_kind"]
                                == "ALTERNATIVE_SOURCE_ARM"
                            else "BASELINE_ARM"
                        )
                        item_id = str(graph["item_id"])
                        region = task_payload["candidate_discovery"]["region"]
                        work_authority_id = stable_id(
                            "first-letters-discovery-historical-authority",
                            {
                                "mission_id": mission_id,
                                "logical_execution_id":
                                    graph["logical_execution_id"],
                                "retained_projection_sha256": projection_sha,
                            },
                        )
                        binding = {
                            "schema": "campaignx.first_letters_discovery_"
                                "work_item_binding.v1",
                            "item_id": item_id,
                            "selection_rank": 0,
                            "sample_id": source["sample_id"],
                            "source_snapshot_id":
                                source["source_snapshot_id"],
                            "source_snapshot_sha256":
                                source["source_snapshot_sha256"],
                            "cell_region": region,
                            "cell_region_sha256": content_sha256(region),
                            "grid_version": task["grid_version"],
                            "grid_spec_sha256": content_sha256({
                                "grid_version": task["grid_version"],
                                "cell_id": item_id,
                                "ct_l0_region": region,
                            }),
                            "scientific_opportunity_id": task_payload[
                                "scientific_opportunity_id"
                            ],
                            "accepted_p0_artifact_id": task_payload[
                                "accepted_p0_artifact_id"
                            ],
                            "accepted_p0_artifact_sha256": task_payload[
                                "accepted_p0_artifact_sha256"
                            ],
                            "parent_task_id": task["task_id"],
                            "parent_attempt_id":
                                graph["created_by_attempt"]["attempt_id"],
                            "allow_unvalidated": False,
                        }
                        authority_core = {
                            "schema": {
                                "BASELINE_ARM": "campaignx.first_letters_"
                                    "discovery_baseline_work_admission.v1",
                                "ALTERNATIVE_SOURCE_ARM": "campaignx.first_"
                                    "letters_experimental_arm_admission.v1",
                            }[work_kind],
                            "work_authority_id": work_authority_id,
                            "mission_id": mission_id,
                            "work_kind": work_kind,
                            "ordered_item_ids": [item_id],
                            "ordered_item_ids_sha256": content_sha256([item_id]),
                            "ordered_item_bindings": [binding],
                            "ordered_item_bindings_sha256":
                                content_sha256([binding]),
                            "cap_authority_id": cap_row["cap_authority_id"],
                            "cap_authority_sha256": cap_row["authority_sha256"],
                            "profile_sha256": graph["profile_file_sha256"],
                            "policy_sha256": graph["probe_run"][
                                "policy_sha256"
                            ],
                            "source_sha256":
                                graph["source_snapshot_sha256"],
                            "deployed_revision": profile[
                                "deployed_revision"
                            ],
                            "requested_item_count": 1,
                            "requested_units": 24,
                            "historical_logical_execution_id":
                                graph["logical_execution_id"],
                            "retained_projection_sha256": projection_sha,
                            "allow_unvalidated": False,
                        }
                        work_authority = {
                            **authority_core,
                            "work_authority_sha256":
                                content_sha256(authority_core),
                        }
                        request_id = stable_id(
                            "first-letters-discovery-historical-request",
                            {
                                "mission_id": mission_id,
                                "logical_execution_id":
                                    graph["logical_execution_id"],
                                "retained_projection_sha256": projection_sha,
                            },
                        )
                        request_core = {
                            "mission_id": mission_id,
                            "request_id": request_id,
                            "work_kind": work_kind,
                            "work_authority": work_authority,
                            "ordered_item_ids": [item_id],
                            "cap_authority_id": cap_row["cap_authority_id"],
                            "cap_authority_sha256": cap_row["authority_sha256"],
                            "source": "IMPORTED_HISTORICAL_EXACT",
                            "reservation_mode": "EXACT",
                            "task9_gate": None,
                        }
                        used = int(connection.execute(
                            "SELECT COALESCE(SUM(reserved_units),0) FROM "
                            "first_letters_discovery_compute_reservations "
                            "WHERE mission_id=?", (mission_id,),
                        ).fetchone()[0])
                        reservation_core = {
                            "schema": "campaignx.first_letters_discovery_"
                                "compute_reservation.v1",
                            "reservation_id": stable_id(
                                "first-letters-discovery-reservation",
                                request_core,
                            ),
                            "mission_id": mission_id,
                            "request_id": request_id,
                            "work_kind": work_kind,
                            "work_authority_id": work_authority_id,
                            "work_authority_sha256": work_authority[
                                "work_authority_sha256"
                            ],
                            "ordered_item_ids": [item_id],
                            "ordered_item_ids_sha256": content_sha256([item_id]),
                            "item_count": 1,
                            "compute_unit": "probe_generation_units",
                            "top_k": 2,
                            "probe_generations": 12,
                            "maximum_attempts_per_candidate": 1,
                            "units_per_item": 24,
                            "reserved_units": 24,
                            "cap_authority_id": cap_row["cap_authority_id"],
                            "cap_authority_sha256": cap_row["authority_sha256"],
                            "reserved_before_units": used,
                            "reserved_after_units": used + 24,
                            "source": "IMPORTED_HISTORICAL_EXACT",
                            "allow_unvalidated": False,
                        }
                        reservation = {
                            **reservation_core,
                            "reservation_sha256":
                                content_sha256(reservation_core),
                            "created_at": created_at,
                        }
                        connection.execute(
                            """INSERT INTO
                               first_letters_discovery_compute_reservations
                               (reservation_id,mission_id,request_id,work_kind,
                                work_authority_id,work_authority_sha256,
                                ordered_item_ids_sha256,item_count,units_per_item,
                                reserved_units,reserved_before_units,
                                reserved_after_units,source,reservation_json,
                                reservation_sha256,request_sha256,created_at)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                reservation["reservation_id"], mission_id,
                                request_id, work_kind, work_authority_id,
                                work_authority["work_authority_sha256"],
                                reservation["ordered_item_ids_sha256"], 1, 24,
                                24, used, used + 24,
                                "IMPORTED_HISTORICAL_EXACT",
                                _dump(reservation),
                                reservation["reservation_sha256"],
                                content_sha256(request_core), created_at,
                            ),
                        )
                        work_core = {
                            "schema": "campaignx.first_letters_discovery_"
                                "work_binding.v1",
                            "reservation_id": reservation["reservation_id"],
                            "reservation_sha256":
                                reservation["reservation_sha256"],
                            "mission_id": mission_id,
                            "request_id": request_id,
                            "work_kind": work_kind,
                            "dispatch_kind": "HISTORICAL_IMPORT_BINDING",
                            "work_authority": work_authority,
                            "ordered_item_ids": [item_id],
                            "allow_unvalidated": False,
                        }
                        work = {
                            **work_core,
                            "work_sha256": content_sha256(work_core),
                        }
                        connection.execute(
                            """INSERT INTO
                               first_letters_discovery_work_bindings
                               (reservation_id,mission_id,request_id,work_kind,
                                dispatch_kind,work_json,work_sha256,created_at)
                               VALUES(?,?,?,?,?,?,?,?)""",
                            (
                                reservation["reservation_id"], mission_id,
                                request_id, work_kind,
                                "HISTORICAL_IMPORT_BINDING", _dump(work),
                                work["work_sha256"], created_at,
                            ),
                        )
                        historical_reservation_id = reservation[
                            "reservation_id"
                        ]
                    else:
                        historical_reservation_id = graph[
                            "retained_row_ids"
                        ]["reservation_id"]
                    import_id = stable_id(
                        "first-letters-discovery-historical-import",
                        {
                            "mission_id": mission_id,
                            "logical_execution_id":
                                graph["logical_execution_id"],
                            "retained_projection_sha256": projection_sha,
                        },
                    )
                    import_core = {
                        "schema":
                            "campaignx.first_letters_discovery_historical_import.v1",
                        "import_id": import_id,
                        "reservation_id": historical_reservation_id,
                        "mission_id": mission_id,
                        "logical_execution_id": graph["logical_execution_id"],
                        "producer_kind": graph["producer_kind"],
                        "source_snapshot_sha256":
                            graph["source_snapshot_sha256"],
                        "profile_file_sha256": graph["profile_file_sha256"],
                        "item_id": graph["item_id"],
                        "fixed_units": 24,
                        "retained_row_ids": graph["retained_row_ids"],
                        "retained_projection_sha256": projection_sha,
                        "history_manifest_sha256": manifest_sha,
                        "allow_unvalidated": False,
                    }
                    historical_import = {
                        **import_core,
                        "import_sha256": content_sha256(import_core),
                        "created_at": created_at,
                    }
                    connection.execute(
                        """INSERT INTO
                           first_letters_discovery_historical_imports_v19
                           (import_id,reservation_id,mission_id,
                            logical_execution_id,producer_kind,
                            source_snapshot_sha256,profile_file_sha256,item_id,
                            fixed_units,retained_row_ids_json,
                            retained_projection_sha256,history_manifest_sha256,
                            import_json,import_sha256,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            import_id, historical_import["reservation_id"],
                            mission_id,
                            historical_import["logical_execution_id"],
                            historical_import["producer_kind"],
                            historical_import["source_snapshot_sha256"],
                            historical_import["profile_file_sha256"],
                            historical_import["item_id"], 24,
                            _dump(historical_import["retained_row_ids"]),
                            projection_sha, manifest_sha,
                            _dump(historical_import),
                            historical_import["import_sha256"], created_at,
                        ),
                    )
            if not complete:
                connection.execute(
                    """INSERT INTO first_letters_discovery_compute_blocks
                       (mission_id,reason,evidence_json,created_at)
                       VALUES(?,?,?,?) ON CONFLICT(mission_id) DO NOTHING""",
                    (mission_id, reason, _dump(reconciliation), created_at),
                )
            return reconciliation

    def _derive_first_letters_baseline_reconciliation_tx(
        self, connection: sqlite3.Connection, *, request_id: str,
        budget_admission_sha256: str, history: dict[str, Any],
        profile_bytes: bytes, profile_source_snapshot_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        from .discovery_bridge import (
            validate_first_letters_baseline_reconciliation,
        )
        from .seed_probe import load_first_letters_discovery_profile_bytes

        admission_rows = connection.execute(
            """SELECT admission_json FROM campaign_budget_admissions
                WHERE admission_sha256=?""",
            (budget_admission_sha256,),
        ).fetchall()
        if len(admission_rows) != 1:
            raise ValueError("baseline budget admission is missing or ambiguous")
        admission = _load(admission_rows[0]["admission_json"])
        if (
            admission.get("schema")
                != "campaignx.first_letters_task_budget_admission.v1"
            or admission.get("admission_sha256") != budget_admission_sha256
            or budget_admission_sha256 != content_sha256({
                key: row for key, row in admission.items()
                if key != "admission_sha256"
            })
        ):
            raise ValueError("baseline budget admission hash is invalid")
        execution = admission.get("execution_bindings") or {}
        source_row = connection.execute(
            "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
            (execution.get("source_snapshot_id"),),
        ).fetchone()
        if source_row is None:
            raise ValueError("baseline source snapshot is missing")
        source = self._snapshot(source_row)
        profile_source = source
        if profile_source_snapshot_id is not None:
            profile_source_row = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (profile_source_snapshot_id,),
            ).fetchone()
            if profile_source_row is None:
                raise ValueError("discovery profile source snapshot is missing")
            profile_source = self._snapshot(profile_source_row)
        profile = load_first_letters_discovery_profile_bytes(profile_bytes)
        if (
            profile.get("mode") != "shadow"
            or profile.get("source_snapshot_id")
                != profile_source["source_snapshot_id"]
            or profile.get("source_snapshot_sha256")
                != profile_source.get("source_snapshot_sha256")
            or profile.get("source_content_lock_sha256")
                != profile_source.get("source_content_lock_sha256")
            or profile.get("ct_metadata_sha256")
                != profile_source.get("ct_metadata_sha256")
            or profile.get("ct_read_set_manifest_sha256")
                != profile_source.get("ct_read_set_manifest_sha256")
            or profile.get("m7_read_set_manifest_sha256")
                != profile_source.get("m7_read_set_manifest_sha256")
            or profile.get("m7_model_id")
                != profile_source.get("m7_model_id")
            or profile.get("m7_resolution")
                != profile_source.get("m7_resolution")
            or profile.get("m7_level") != profile_source.get("m7_level")
            or profile.get("m7_threshold")
                != profile_source.get("m7_threshold")
            or profile.get("m7_transform_sha256")
                != profile_source.get("m7_transform_sha256")
        ):
            raise ValueError("baseline profile differs from registered source")
        cap_row = connection.execute(
            """SELECT * FROM first_letters_discovery_compute_caps
                WHERE mission_id=?""",
            (admission["mission_id"],),
        ).fetchone()
        if cap_row is None:
            raise ValueError("baseline compute cap is missing")
        cap = self._validated_discovery_compute_cap(
            _load(cap_row["authority_json"])
        )
        if (
            cap_row["mission_id"] != cap["mission_id"]
            or cap_row["cap_authority_id"] != cap["cap_authority_id"]
            or cap_row["authority_sha256"] != cap["authority_sha256"]
            or int(cap_row["cap_units"])
                != cap["mission_compute_cap_units"]
            or profile.get("mission_compute_cap_authority_id")
                != cap["cap_authority_id"]
            or profile.get("mission_compute_cap_authority_sha256")
                != cap["authority_sha256"]
            or profile.get("deployed_revision") != cap["deployed_revision"]
        ):
            raise ValueError("baseline profile differs from registered cap")
        items = admission.get("prefix_cell_ids")
        if (
            not isinstance(items, list) or not items
            or len(items) != admission.get("approved_task_count")
            or items != list(dict.fromkeys(items))
            or admission.get("prefix_sha256") != content_sha256(items)
            or profile.get("canonical_ordered_cell_set_sha256")
                != content_sha256(items)
        ):
            raise ValueError("baseline admission cohort is invalid")
        source_discovery = source.get("first_letters_discovery_authority") or {}
        if (
            source_discovery.get("mission_id") != admission["mission_id"]
            or source_discovery.get("accepted_p0_artifact_id")
                != execution.get("p0_artifact_id")
            or source_discovery.get("accepted_p0_artifact_sha256")
                != execution.get("p0_artifact_sha256")
            or not isinstance(
                source_discovery.get("scientific_opportunities"), dict
            )
        ):
            raise ValueError("baseline source discovery authority is invalid")
        bindings = []
        for rank, item_id in enumerate(items):
            rows = connection.execute(
                """SELECT * FROM tasks
                    WHERE mission_id=? AND cell_id=?""",
                (admission["mission_id"], item_id),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("baseline task cohort is missing or ambiguous")
            task = rows[0]
            payload = _load(task["payload_json"]) or {}
            if (
                task["source_snapshot_id"] != source["source_snapshot_id"]
                or task["grid_version"] != execution.get("grid_version")
                or task["policy_version"] != execution.get("policy_version")
                or payload.get("sample_id") != admission["sample_id"]
                or payload.get("selection_rank") != rank
                or payload.get("campaign_budget_admission_sha256")
                    != budget_admission_sha256
            ):
                raise ValueError("baseline task differs from budget authority")
            region = (payload.get("candidate_discovery") or {}).get("region")
            if not isinstance(region, dict):
                bounds = _load(task["bounds_xyz_json"])
                region = {"minimum": bounds[0], "maximum": bounds[1]}
            p0_id = payload.get(
                "accepted_p0_artifact_id", payload.get("p0_artifact_id")
            )
            p0_sha = payload.get(
                "accepted_p0_artifact_sha256", payload.get("p0_artifact_sha256")
            )
            opportunity = payload.get("scientific_opportunity_id")
            if (
                p0_id != execution.get("p0_artifact_id")
                or p0_sha != execution.get("p0_artifact_sha256")
                or not isinstance(opportunity, str) or not opportunity
                or source_discovery["scientific_opportunities"].get(item_id)
                    != opportunity
            ):
                raise ValueError("baseline task opportunity/P0 authority differs")
            bindings.append({
                "schema":
                    "campaignx.first_letters_discovery_work_item_binding.v1",
                "item_id": item_id,
                "selection_rank": rank,
                "sample_id": admission["sample_id"],
                "source_snapshot_id": source["source_snapshot_id"],
                "source_snapshot_sha256": source["source_snapshot_sha256"],
                "cell_region": region,
                "cell_region_sha256": content_sha256(region),
                "grid_version": task["grid_version"],
                "grid_spec_sha256": content_sha256({
                    "grid_version": task["grid_version"],
                    "cell_id": item_id, "ct_l0_region": region,
                }),
                "scientific_opportunity_id": opportunity,
                "accepted_p0_artifact_id": p0_id,
                "accepted_p0_artifact_sha256": p0_sha,
                "parent_task_id": task["task_id"],
                "parent_attempt_id": task["active_attempt_id"],
                "allow_unvalidated": False,
            })
        core = {
            "schema":
                "campaignx.first_letters_discovery_baseline_reconciliation.v1",
            "mission_id": admission["mission_id"],
            "request_id": request_id,
            "sample_id": admission["sample_id"],
            "budget_admission_sha256": budget_admission_sha256,
            "source_snapshot_id": source["source_snapshot_id"],
            "source_snapshot_sha256": source["source_snapshot_sha256"],
            "source_content_lock_sha256":
                source["source_content_lock_sha256"],
            "accepted_p0_artifact_id": execution["p0_artifact_id"],
            "accepted_p0_artifact_sha256": execution["p0_artifact_sha256"],
            "grid_version": execution["grid_version"],
            "ordered_item_ids": copy.deepcopy(items),
            "ordered_item_ids_sha256": content_sha256(items),
            "ordered_item_bindings": bindings,
            "ordered_item_bindings_sha256": content_sha256(bindings),
            "cap_authority_id": cap["cap_authority_id"],
            "cap_authority_sha256": cap["authority_sha256"],
            "profile_file_sha256": profile["profile_file_sha256"],
            "profile_scientific_core_sha256":
                profile["scientific_core_sha256"],
            "policy_sha256": cap["policy_chain_sha256"],
            "deployed_revision": cap["deployed_revision"],
            "history_manifest_sha256": history["manifest_sha256"],
            "mode": "shadow", "namespace": "NONCANONICAL_DISCOVERY",
            "canonical_admission": "PROHIBITED",
            "top_k": 2, "probe_generations": 12,
            "maximum_attempts_per_candidate": 1, "units_per_item": 24,
            "allow_unvalidated": False,
        }
        reconciliation = validate_first_letters_baseline_reconciliation({
            **core, "reconciliation_sha256": content_sha256(core),
        })
        return reconciliation, source, profile

    def _reserve_first_letters_shadow(
        self, *, request_id: str, budget_admission_sha256: str,
        arm_id: str | None, failpoint: str | None,
    ) -> dict[str, Any]:
        from .discovery_bridge import (
            adapt_first_letters_alternative_shadow,
            adapt_first_letters_baseline_shadow,
            build_first_letters_discovery_dispatch,
            build_first_letters_discovery_jobs,
        )

        allowed = {
            None, "bridge.before_reservation",
            "bridge.after_reservation_before_adapter",
            "bridge.after_adapter_before_dispatch",
            "bridge.after_dispatch_before_jobs", "bridge.after_each_job",
            "bridge.after_jobs_before_commit", "bridge.before_commit",
            "bridge.commit_outcome_unknown",
            "bridge.after_commit_before_response",
        }
        if failpoint not in allowed:
            raise ValueError("unknown bridge failpoint")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("bridge request ID is invalid")
        resolver = self._first_letters_discovery_profile_resolver
        if resolver is None:
            raise ValueError("DISCOVERY_SERVER_PROFILE_AUTHORITY_REQUIRED")
        with self.connect() as lookup:
            admission_row = lookup.execute(
                """SELECT admission_json FROM campaign_budget_admissions
                    WHERE admission_sha256=?""",
                (budget_admission_sha256,),
            ).fetchone()
        if admission_row is None:
            raise ValueError("baseline budget admission is missing")
        admission = _load(admission_row["admission_json"])
        source_id = (admission.get("execution_bindings") or {}).get(
            "source_snapshot_id"
        )
        arm = None
        if arm_id is not None:
            arm_resolver = self._first_letters_experimental_arm_resolver
            arm = arm_resolver(arm_id) if arm_resolver is not None else None
            if not isinstance(arm, dict):
                raise ValueError("DISCOVERY_SERVER_ARM_AUTHORITY_REQUIRED")
        profile_source_id = (
            arm.get("source_snapshot_id") if arm is not None else source_id
        )
        profile_bytes = resolver(admission["mission_id"], profile_source_id)
        if not isinstance(profile_bytes, bytes):
            raise ValueError("DISCOVERY_SERVER_PROFILE_AUTHORITY_REQUIRED")
        history = self.reconcile_first_letters_discovery_history(
            mission_id=admission["mission_id"]
        )
        if history["state"] != "COMPLETE":
            raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            manifest, complete, _ = self._derive_first_letters_discovery_history_tx(
                connection, mission_id=admission["mission_id"],
            )
            if not complete or content_sha256(manifest) != history["manifest_sha256"]:
                raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
            reconciliation, baseline_source, _ = (
                self._derive_first_letters_baseline_reconciliation_tx(
                    connection, request_id=request_id,
                    budget_admission_sha256=budget_admission_sha256,
                    history=history, profile_bytes=profile_bytes,
                    profile_source_snapshot_id=(
                        profile_source_id if arm is not None else None
                    ),
                )
            )
            if arm_id is None:
                adapter = adapt_first_letters_baseline_shadow(
                    reconciliation, baseline_source,
                )
            else:
                source_row = connection.execute(
                    "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                    (arm.get("source_snapshot_id"),),
                ).fetchone()
                if source_row is None:
                    raise ValueError("experimental arm source is missing")
                adapter = adapt_first_letters_alternative_shadow(
                    reconciliation, arm, self._snapshot(source_row),
                )
            generic = adapter["generic_work_authority"]
            mission_id = adapter["mission_id"]
            request_sha = content_sha256(adapter)
            existing = connection.execute(
                """SELECT request_sha256
                     FROM first_letters_discovery_compute_reservations
                    WHERE mission_id=? AND request_id=?""",
                (mission_id, request_id),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_sha:
                    raise ValueError("DISCOVERY_COMPUTE_RESERVATION_CONFLICT")
                connection.commit()
                return self.read_first_letters_discovery_request(
                    mission_id, request_id
                )
            if connection.execute(
                "SELECT 1 FROM first_letters_discovery_compute_blocks "
                "WHERE mission_id=?", (mission_id,),
            ).fetchone() is not None:
                raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
            cap_row = connection.execute(
                """SELECT * FROM first_letters_discovery_compute_caps
                    WHERE mission_id=?""", (mission_id,),
            ).fetchone()
            if cap_row is None:
                raise ValueError("mission discovery compute cap authority mismatch")
            used = int(connection.execute(
                """SELECT COALESCE(SUM(reserved_units),0)
                     FROM first_letters_discovery_compute_reservations
                    WHERE mission_id=?""", (mission_id,),
            ).fetchone()[0])
            items = generic["ordered_item_ids"]
            reserved = len(items) * 24
            if used + reserved > int(cap_row["cap_units"]):
                raise ValueError("mission discovery compute cap exhausted")
            reservation_core = {
                "schema":
                    "campaignx.first_letters_discovery_compute_reservation.v1",
                "reservation_id": stable_id(
                    "first-letters-discovery-reservation", adapter,
                ),
                "mission_id": mission_id, "request_id": request_id,
                "work_kind": adapter["work_kind"],
                "work_authority_id": generic["work_authority_id"],
                "work_authority_sha256": generic["work_authority_sha256"],
                "ordered_item_ids": copy.deepcopy(items),
                "ordered_item_ids_sha256": content_sha256(items),
                "item_count": len(items),
                "compute_unit": "probe_generation_units",
                "top_k": 2, "probe_generations": 12,
                "maximum_attempts_per_candidate": 1,
                "units_per_item": 24, "reserved_units": reserved,
                "cap_authority_id": generic["cap_authority_id"],
                "cap_authority_sha256": generic["cap_authority_sha256"],
                "reserved_before_units": used,
                "reserved_after_units": used + reserved,
                "source": "RESERVED_BEFORE_EXECUTION",
                "allow_unvalidated": False,
            }
            reservation = {
                **reservation_core,
                "reservation_sha256": content_sha256(reservation_core),
                "created_at": utc_now(),
            }
            if failpoint == "bridge.before_reservation":
                raise RuntimeError(failpoint)
            connection.execute(
                """INSERT INTO first_letters_discovery_compute_reservations
                   (reservation_id,mission_id,request_id,work_kind,
                    work_authority_id,work_authority_sha256,
                    ordered_item_ids_sha256,item_count,units_per_item,
                    reserved_units,reserved_before_units,reserved_after_units,
                    source,reservation_json,reservation_sha256,request_sha256,
                    created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    reservation["reservation_id"], mission_id, request_id,
                    reservation["work_kind"], generic["work_authority_id"],
                    generic["work_authority_sha256"],
                    reservation["ordered_item_ids_sha256"], len(items), 24,
                    reserved, used, used + reserved,
                    "RESERVED_BEFORE_EXECUTION", _dump(reservation),
                    reservation["reservation_sha256"], request_sha,
                    reservation["created_at"],
                ),
            )
            work_core = {
                "schema": "campaignx.first_letters_discovery_work_binding.v1",
                "reservation_id": reservation["reservation_id"],
                "reservation_sha256": reservation["reservation_sha256"],
                "mission_id": mission_id, "request_id": request_id,
                "work_kind": adapter["work_kind"],
                "dispatch_kind": {
                    "BASELINE_ARM": "BASELINE_DISPATCH",
                    "ALTERNATIVE_SOURCE_ARM": "ALTERNATIVE_SOURCE_DISPATCH",
                }[adapter["work_kind"]],
                "work_authority": generic,
                "ordered_item_ids": copy.deepcopy(items),
                "allow_unvalidated": False,
            }
            work = {**work_core, "work_sha256": content_sha256(work_core)}
            connection.execute(
                """INSERT INTO first_letters_discovery_work_bindings
                   (reservation_id,mission_id,request_id,work_kind,
                    dispatch_kind,work_json,work_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    reservation["reservation_id"], mission_id, request_id,
                    adapter["work_kind"], work["dispatch_kind"], _dump(work),
                    work["work_sha256"], utc_now(),
                ),
            )
            if failpoint == "bridge.after_reservation_before_adapter":
                raise RuntimeError(failpoint)
            connection.execute(
                """INSERT INTO first_letters_discovery_native_adapters_v19
                   (reservation_id,mission_id,request_id,work_kind,
                    producer_kind,native_schema,native_authority_json,
                    native_authority_sha256,generic_work_authority_json,
                    generic_work_authority_sha256,profile_bytes,adapter_json,
                    adapter_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    reservation["reservation_id"], mission_id, request_id,
                    adapter["work_kind"], adapter["producer_kind"],
                    adapter["native_schema"],
                    _dump(adapter["native_authority"]),
                    adapter["native_authority_sha256"], _dump(generic),
                    adapter["generic_work_authority_sha256"], profile_bytes,
                    _dump(adapter), adapter["adapter_sha256"], utc_now(),
                ),
            )
            if failpoint == "bridge.after_adapter_before_dispatch":
                raise RuntimeError(failpoint)
            dispatch = build_first_letters_discovery_dispatch(
                reservation, adapter,
            )
            connection.execute(
                """INSERT INTO first_letters_discovery_dispatches_v19
                   (dispatch_id,reservation_id,mission_id,request_id,work_kind,
                    adapter_sha256,profile_file_sha256,source_snapshot_sha256,
                    ordered_item_ids_sha256,item_count,dispatch_json,
                    dispatch_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    dispatch["dispatch_id"], reservation["reservation_id"],
                    mission_id, request_id, adapter["work_kind"],
                    adapter["adapter_sha256"],
                    dispatch["profile_file_sha256"],
                    dispatch["source_snapshot_sha256"],
                    dispatch["ordered_item_ids_sha256"],
                    dispatch["item_count"], _dump(dispatch),
                    dispatch["dispatch_sha256"], utc_now(),
                ),
            )
            if failpoint == "bridge.after_dispatch_before_jobs":
                raise RuntimeError(failpoint)
            jobs = build_first_letters_discovery_jobs(dispatch, adapter)
            for job in jobs:
                connection.execute(
                    """INSERT INTO first_letters_discovery_jobs_v19
                       (job_id,dispatch_id,reservation_id,item_order,item_id,
                        work_item_binding_sha256,profile_file_sha256,
                        source_snapshot_sha256,job_json,job_sha256,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        job["job_id"], dispatch["dispatch_id"],
                        reservation["reservation_id"], job["item_order"],
                        job["item_id"], job["work_item_binding_sha256"],
                        job["profile_file_sha256"],
                        job["source_snapshot_sha256"], _dump(job),
                        job["job_sha256"], utc_now(),
                    ),
                )
                if failpoint == "bridge.after_each_job":
                    raise RuntimeError(failpoint)
            if failpoint == "bridge.after_jobs_before_commit":
                raise RuntimeError(failpoint)
            if failpoint == "bridge.before_commit":
                raise RuntimeError(failpoint)
            connection.commit()
        if failpoint == "bridge.commit_outcome_unknown":
            raise RuntimeError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        return self.read_first_letters_discovery_request(
            admission["mission_id"], request_id
        )

    def reserve_first_letters_baseline_shadow(
        self, *, request_id: str, budget_admission_sha256: str,
        failpoint: str | None = None,
    ) -> dict[str, Any]:
        return self._reserve_first_letters_shadow(
            request_id=request_id,
            budget_admission_sha256=budget_admission_sha256,
            arm_id=None, failpoint=failpoint,
        )

    def reserve_first_letters_alternative_shadow(
        self, *, request_id: str, budget_admission_sha256: str,
        arm_id: str, failpoint: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("experimental arm ID is invalid")
        return self._reserve_first_letters_shadow(
            request_id=request_id,
            budget_admission_sha256=budget_admission_sha256,
            arm_id=arm_id, failpoint=failpoint,
        )

    def _current_first_letters_discovery_adapter_tx(
        self, connection: sqlite3.Connection, *, adapter: dict[str, Any],
        persisted_profile_bytes: bytes,
    ) -> dict[str, Any]:
        """Rebuild one v19 adapter exclusively from current server authority."""

        from .discovery_bridge import (
            adapt_first_letters_alternative_shadow,
            adapt_first_letters_baseline_shadow,
        )

        native = adapter.get("native_authority") or {}
        alternative = adapter.get("producer_kind") == (
            "EXPERIMENTAL_ARM_ADMISSION"
        )
        reconciliation = (
            native.get("baseline_reconciliation") if alternative else native
        )
        if not isinstance(reconciliation, dict):
            raise ValueError("discovery reconciliation authority is missing")
        arm = None
        profile_source_id = reconciliation.get("source_snapshot_id")
        if alternative:
            persisted_arm = native.get("arm_admission") or {}
            arm_id = persisted_arm.get("arm_id")
            resolver = self._first_letters_experimental_arm_resolver
            if not isinstance(arm_id, str) or resolver is None:
                raise ValueError("DISCOVERY_SERVER_ARM_AUTHORITY_REQUIRED")
            arm = resolver(arm_id)
            if not isinstance(arm, dict):
                raise ValueError("DISCOVERY_SERVER_ARM_AUTHORITY_REQUIRED")
            profile_source_id = arm.get("source_snapshot_id")
        profile_resolver = self._first_letters_discovery_profile_resolver
        if profile_resolver is None:
            raise ValueError("DISCOVERY_SERVER_PROFILE_AUTHORITY_REQUIRED")
        profile_bytes = profile_resolver(
            reconciliation.get("mission_id"), profile_source_id,
        )
        if (
            not isinstance(profile_bytes, bytes)
            or profile_bytes != persisted_profile_bytes
        ):
            raise ValueError("DISCOVERY_SERVER_PROFILE_AUTHORITY_REQUIRED")
        current_reconciliation, baseline_source, _ = (
            self._derive_first_letters_baseline_reconciliation_tx(
                connection,
                request_id=reconciliation.get("request_id"),
                budget_admission_sha256=reconciliation.get(
                    "budget_admission_sha256"
                ),
                history={
                    "manifest_sha256": reconciliation.get(
                        "history_manifest_sha256"
                    )
                },
                profile_bytes=profile_bytes,
                profile_source_snapshot_id=(
                    profile_source_id if alternative else None
                ),
            )
        )
        if not alternative:
            current = adapt_first_letters_baseline_shadow(
                current_reconciliation, baseline_source,
            )
        else:
            source_row = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (profile_source_id,),
            ).fetchone()
            if source_row is None:
                raise ValueError("experimental arm source is missing")
            current = adapt_first_letters_alternative_shadow(
                current_reconciliation, arm, self._snapshot(source_row),
            )
        if current != adapter:
            raise ValueError("current discovery adapter authority differs")
        return current

    def read_first_letters_discovery_request(
        self, mission_id: str, request_id: str,
    ) -> dict[str, Any]:
        from .discovery_bridge import (
            build_first_letters_discovery_dispatch,
            build_first_letters_discovery_jobs,
            validate_first_letters_discovery_native_adapter,
        )

        with self.connect() as connection:
            reservation_row = connection.execute(
                """SELECT * FROM first_letters_discovery_compute_reservations
                    WHERE mission_id=? AND request_id=?""",
                (mission_id, request_id),
            ).fetchone()
            if reservation_row is None:
                raise KeyError(request_id)
            work_row = connection.execute(
                """SELECT * FROM first_letters_discovery_work_bindings
                    WHERE reservation_id=?""",
                (reservation_row["reservation_id"],),
            ).fetchone()
            adapter_row = connection.execute(
                """SELECT * FROM first_letters_discovery_native_adapters_v19
                    WHERE reservation_id=?""",
                (reservation_row["reservation_id"],),
            ).fetchone()
            dispatch_rows = connection.execute(
                """SELECT * FROM first_letters_discovery_dispatches_v19
                    WHERE reservation_id=?""",
                (reservation_row["reservation_id"],),
            ).fetchall()
            job_rows = connection.execute(
                """SELECT * FROM first_letters_discovery_jobs_v19
                    WHERE reservation_id=? ORDER BY item_order,job_id""",
                (reservation_row["reservation_id"],),
            ).fetchall()
        if work_row is None or adapter_row is None or len(dispatch_rows) != 1:
            raise ValueError("CONTROL_INCOMPLETE_DISCOVERY_DISPATCH")
        reservation = _load(reservation_row["reservation_json"])
        work = _load(work_row["work_json"])
        adapter = validate_first_letters_discovery_native_adapter(
            _load(adapter_row["adapter_json"])
        )
        dispatch = _load(dispatch_rows[0]["dispatch_json"])
        jobs = [_load(row["job_json"]) for row in job_rows]
        expected_dispatch = build_first_letters_discovery_dispatch(
            reservation, adapter,
        )
        expected_jobs = build_first_letters_discovery_jobs(
            expected_dispatch, adapter,
        )
        dispatch_row = dispatch_rows[0]
        scalar_jobs_valid = all(
            row["job_id"] == job["job_id"]
            and row["dispatch_id"] == job["dispatch_id"]
            and row["reservation_id"] == job["reservation_id"]
            and row["item_order"] == job["item_order"]
            and row["item_id"] == job["item_id"]
            and row["work_item_binding_sha256"]
                == job["work_item_binding_sha256"]
            and row["profile_file_sha256"]
                == job["profile_file_sha256"]
            and row["source_snapshot_sha256"]
                == job["source_snapshot_sha256"]
            and row["job_sha256"] == job["job_sha256"]
            for row, job in zip(job_rows, jobs, strict=True)
        ) if len(job_rows) == len(jobs) else False
        if (
            dispatch != expected_dispatch
            or jobs != expected_jobs
            or work.get("reservation_sha256")
                != reservation.get("reservation_sha256")
            or work.get("work_authority")
                != adapter["generic_work_authority"]
            or reservation_row["reservation_sha256"]
                != reservation.get("reservation_sha256")
            or work_row["work_sha256"] != work.get("work_sha256")
            or adapter_row["reservation_id"]
                != reservation["reservation_id"]
            or adapter_row["mission_id"] != reservation["mission_id"]
            or adapter_row["request_id"] != reservation["request_id"]
            or adapter_row["work_kind"] != adapter["work_kind"]
            or adapter_row["producer_kind"] != adapter["producer_kind"]
            or adapter_row["native_schema"] != adapter["native_schema"]
            or _load(adapter_row["native_authority_json"])
                != adapter["native_authority"]
            or adapter_row["native_authority_sha256"]
                != adapter["native_authority_sha256"]
            or _load(adapter_row["generic_work_authority_json"])
                != adapter["generic_work_authority"]
            or adapter_row["generic_work_authority_sha256"]
                != adapter["generic_work_authority_sha256"]
            or adapter_row["adapter_sha256"] != adapter["adapter_sha256"]
            or hashlib.sha256(bytes(adapter_row["profile_bytes"])).hexdigest()
                != adapter["profile_file_sha256"]
            or dispatch_row["dispatch_id"] != dispatch["dispatch_id"]
            or dispatch_row["reservation_id"]
                != reservation["reservation_id"]
            or dispatch_row["mission_id"] != reservation["mission_id"]
            or dispatch_row["request_id"] != reservation["request_id"]
            or dispatch_row["work_kind"] != adapter["work_kind"]
            or dispatch_row["adapter_sha256"] != adapter["adapter_sha256"]
            or dispatch_row["profile_file_sha256"]
                != dispatch["profile_file_sha256"]
            or dispatch_row["source_snapshot_sha256"]
                != dispatch["source_snapshot_sha256"]
            or dispatch_row["ordered_item_ids_sha256"]
                != dispatch["ordered_item_ids_sha256"]
            or dispatch_row["dispatch_sha256"] != dispatch["dispatch_sha256"]
            or dispatch_row["item_count"] != dispatch["item_count"]
            or not scalar_jobs_valid
        ):
            raise ValueError("CONTROL_INCOMPLETE_DISCOVERY_DISPATCH")
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current_adapter = (
                    self._current_first_letters_discovery_adapter_tx(
                        connection, adapter=adapter,
                        persisted_profile_bytes=bytes(
                            adapter_row["profile_bytes"]
                        ),
                    )
                )
            if current_adapter != adapter:
                raise ValueError("current discovery adapter authority differs")
        except Exception as error:
            native = adapter.get("native_authority") or {}
            reconciliation = (
                native
                if adapter.get("producer_kind") == "BASELINE_RECONCILIATION"
                else native.get("baseline_reconciliation") or {}
            )
            history_failure = None
            try:
                with self.connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        current_manifest, current_complete, _ = (
                            self._derive_first_letters_discovery_history_tx(
                                connection, mission_id=mission_id,
                            )
                        )
                    except Exception as history_error:
                        current_manifest = {
                            "schema": "campaignx.first_letters_discovery_"
                                "history_derivation_failure.v1",
                            "mission_id": mission_id,
                            "error_type": type(history_error).__name__,
                            "retained_execution_graphs": [],
                            "allow_unvalidated": False,
                        }
                        current_complete = False
                    if (
                        not current_complete
                        or content_sha256(current_manifest)
                            != reconciliation.get("history_manifest_sha256")
                    ):
                        self._persist_discovery_history_incomplete_tx(
                            connection, mission_id=mission_id,
                            manifest=current_manifest,
                            reason="CONTROL_INCOMPLETE_COMPUTE_LEDGER",
                        )
                        connection.commit()
                        history_failure = ValueError(
                            "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
                        )
            except Exception as persistence_error:
                history_failure = persistence_error
            if history_failure is not None:
                if str(history_failure) == "CONTROL_INCOMPLETE_COMPUTE_LEDGER":
                    raise history_failure from error
                raise ValueError(
                    "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
                ) from history_failure
            self._block_discovery_compute_ledger(mission_id, {
                "schema":
                    "campaignx.first_letters_discovery_authority_failure.v1",
                "mission_id": mission_id, "request_id": request_id,
                "error_type": type(error).__name__,
                "allow_unvalidated": False,
            })
            raise ValueError(
                "CONTROL_INCOMPLETE_DISCOVERY_DISPATCH"
            ) from error
        return {
            "reservation": reservation, "work": work, "adapter": adapter,
            "dispatch": dispatch, "jobs": jobs,
        }

    def reserve_discovery_compute(
        self, *, mission_id: str, request_id: str, work_kind: str,
        work_authority: dict[str, Any], work_authority_id: str,
        work_authority_sha256: str, ordered_item_ids: list[str],
        cap_authority_id: str, cap_authority_sha256: str,
        source: str = "RESERVED_BEFORE_EXECUTION",
        reservation_mode: str = "EXACT", task9_gate=None,
        failpoint: str | None = None,
    ) -> dict[str, Any] | None:
        if work_kind in {"BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM"}:
            raise ValueError("DISCOVERY_NATIVE_PRODUCER_REQUIRED")
        allowed_failpoints = {
            None, "compute.before_reservation_insert",
            "compute.after_reservation_insert_before_work_insert",
            "compute.after_work_insert_before_commit", "compute.before_commit",
            "compute.commit_outcome_unknown", "compute.after_commit_before_response",
        }
        if failpoint not in allowed_failpoints:
            raise ValueError("unknown compute failpoint")
        if source == "IMPORTED_HISTORICAL_EXACT":
            raise ValueError("DISCOVERY_NATIVE_PRODUCER_REQUIRED")
        if source != "RESERVED_BEFORE_EXECUTION":
            raise ValueError("unsupported discovery reservation source")
        if reservation_mode not in {"EXACT", "PREFIX_TO_CAP"}:
            raise ValueError("unsupported discovery reservation mode")
        if work_kind == "ADAPTIVE_CHILD":
            resolver = self._task9_discovery_gate_resolver
            if resolver is None:
                raise ValueError("TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED")
            authoritative_gate = resolver(mission_id)
            if (not isinstance(authoritative_gate, dict)
                    or authoritative_gate != task9_gate
                    or authoritative_gate.get("schema") !=
                        "campaignx.first_letters_task9_discovery_gate.v1"
                    or authoritative_gate.get("allow_unvalidated") is not False):
                raise ValueError("TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED")
        authority = self._validated_discovery_work_authority(
            work_authority, mission_id=mission_id, work_kind=work_kind,
            work_authority_id=work_authority_id,
            work_authority_sha256=work_authority_sha256,
            ordered_item_ids=ordered_item_ids, cap_authority_id=cap_authority_id,
            cap_authority_sha256=cap_authority_sha256,
        )
        request_core = {
            "mission_id": mission_id, "request_id": request_id,
            "work_kind": work_kind, "work_authority": authority,
            "ordered_item_ids": ordered_item_ids,
            "cap_authority_id": cap_authority_id,
            "cap_authority_sha256": cap_authority_sha256,
            "source": source, "reservation_mode": reservation_mode,
            "task9_gate": task9_gate,
        }
        request_sha = content_sha256(request_core)
        connection = self.connect()
        committed = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            block = connection.execute(
                "SELECT reason FROM first_letters_discovery_compute_blocks WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
            if block is not None:
                raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
            cap_row = connection.execute(
                "SELECT * FROM first_letters_discovery_compute_caps WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
            if (cap_row is None or cap_row["cap_authority_id"] != cap_authority_id
                    or cap_row["authority_sha256"] != cap_authority_sha256):
                raise ValueError("mission discovery compute cap authority mismatch")
            existing = connection.execute(
                """SELECT request_sha256 FROM first_letters_discovery_compute_reservations
                    WHERE mission_id=? AND request_id=?""",
                (mission_id, request_id),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_sha:
                    raise ValueError("DISCOVERY_COMPUTE_RESERVATION_CONFLICT")
                connection.commit()
                committed = True
                return self.read_discovery_compute_request(mission_id, request_id)
            used = int(connection.execute(
                "SELECT COALESCE(SUM(reserved_units),0) FROM first_letters_discovery_compute_reservations WHERE mission_id=?",
                (mission_id,),
            ).fetchone()[0])
            cap_units = int(cap_row["cap_units"])
            available_items = max(0, (cap_units - used) // 24)
            if reservation_mode == "EXACT":
                if len(ordered_item_ids) > available_items:
                    raise ValueError("mission discovery compute cap exhausted")
                selected = list(ordered_item_ids)
            else:
                selected = list(ordered_item_ids[:min(len(ordered_item_ids), available_items)])
                if not selected:
                    connection.commit()
                    committed = True
                    return None
            reserved = len(selected) * 24
            reservation_core = {
                "schema": "campaignx.first_letters_discovery_compute_reservation.v1",
                "reservation_id": stable_id("first-letters-discovery-reservation", request_core),
                "mission_id": mission_id, "request_id": request_id,
                "work_kind": work_kind, "work_authority_id": work_authority_id,
                "work_authority_sha256": work_authority_sha256,
                "ordered_item_ids": selected,
                "ordered_item_ids_sha256": content_sha256(selected),
                "item_count": len(selected), "compute_unit": "probe_generation_units",
                "top_k": 2, "probe_generations": 12,
                "maximum_attempts_per_candidate": 1, "units_per_item": 24,
                "reserved_units": reserved, "cap_authority_id": cap_authority_id,
                "cap_authority_sha256": cap_authority_sha256,
                "reserved_before_units": used, "reserved_after_units": used + reserved,
                "source": source, "allow_unvalidated": False,
            }
            reservation = {
                **reservation_core,
                "reservation_sha256": content_sha256(reservation_core),
                "created_at": utc_now(),
            }
            if failpoint == "compute.before_reservation_insert":
                raise RuntimeError(failpoint)
            connection.execute(
                """INSERT INTO first_letters_discovery_compute_reservations
                   (reservation_id,mission_id,request_id,work_kind,work_authority_id,
                    work_authority_sha256,ordered_item_ids_sha256,item_count,
                    units_per_item,reserved_units,reserved_before_units,
                    reserved_after_units,source,reservation_json,reservation_sha256,
                    request_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (reservation["reservation_id"], mission_id, request_id, work_kind,
                 work_authority_id, work_authority_sha256,
                 reservation["ordered_item_ids_sha256"], len(selected), 24,
                 reserved, used, used + reserved, source, _dump(reservation),
                 reservation["reservation_sha256"], request_sha,
                 reservation["created_at"]),
            )
            if failpoint == "compute.after_reservation_insert_before_work_insert":
                raise RuntimeError(failpoint)
            dispatch_kind = {
                "BASELINE_ARM": "BASELINE_DISPATCH",
                "ALTERNATIVE_SOURCE_ARM": "ALTERNATIVE_SOURCE_DISPATCH",
                "ADAPTIVE_CHILD": "ADAPTIVE_CHILDREN",
            }[work_kind]
            if source == "IMPORTED_HISTORICAL_EXACT":
                dispatch_kind = "HISTORICAL_IMPORT_BINDING"
            work_core = {
                "schema": "campaignx.first_letters_discovery_work_binding.v1",
                "reservation_id": reservation["reservation_id"],
                "reservation_sha256": reservation["reservation_sha256"],
                "mission_id": mission_id, "request_id": request_id,
                "work_kind": work_kind, "dispatch_kind": dispatch_kind,
                "work_authority": authority, "ordered_item_ids": selected,
                "allow_unvalidated": False,
            }
            work = {**work_core, "work_sha256": content_sha256(work_core)}
            connection.execute(
                """INSERT INTO first_letters_discovery_work_bindings
                   (reservation_id,mission_id,request_id,work_kind,dispatch_kind,
                    work_json,work_sha256,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (reservation["reservation_id"], mission_id, request_id, work_kind,
                 dispatch_kind, _dump(work), work["work_sha256"], utc_now()),
            )
            if failpoint == "compute.after_work_insert_before_commit":
                raise RuntimeError(failpoint)
            if failpoint == "compute.before_commit":
                raise RuntimeError(failpoint)
            connection.commit()
            committed = True
        except BaseException:
            if not committed and connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        if failpoint == "compute.commit_outcome_unknown":
            raise RuntimeError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        return self.read_discovery_compute_request(mission_id, request_id)

    def read_discovery_compute_request(
        self, mission_id: str, request_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT r.reservation_json,w.work_json
                     FROM first_letters_discovery_compute_reservations r
                     JOIN first_letters_discovery_work_bindings w
                       ON w.reservation_id=r.reservation_id
                    WHERE r.mission_id=? AND r.request_id=?""",
                (mission_id, request_id),
            ).fetchall()
        if len(rows) != 1:
            raise ValueError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        return {
            "reservation": _load(rows[0]["reservation_json"]),
            "work": _load(rows[0]["work_json"]),
        }

    def begin_first_letters_discovery_evidence_run(
        self, *, lease_seconds: int, reservation_id: str,
        item_id: str, profile_bytes: bytes,
    ) -> _FirstLettersDiscoveryRunHandle:
        del lease_seconds, reservation_id, item_id, profile_bytes
        raise ValueError("DISCOVERY_JOB_ID_REQUIRED")

    def _begin_first_letters_discovery_evidence_run(
        self, *, lease_seconds: int, reservation_id: str,
        item_id: str, profile_bytes: bytes, _job_id: str | None,
    ) -> _FirstLettersDiscoveryRunHandle:
        """Atomically claim the noncanonical producer seam for one work item."""

        from .seed_probe import (
            _accept_first_letters_discovery_executor_claim,
            _bind_first_letters_discovery_executor_claim,
            _derive_first_letters_discovery_run_authority,
            load_first_letters_discovery_profile_bytes,
        )

        if (not isinstance(item_id, str) or not item_id
                or isinstance(lease_seconds, bool)
                or not isinstance(lease_seconds, int) or lease_seconds < 30):
            raise ValueError("discovery producer claim arguments are invalid")
        if not isinstance(profile_bytes, bytes):
            raise ValueError("discovery profile must be exact registered bytes")
        load_first_letters_discovery_profile_bytes(profile_bytes)
        profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            adapter_probe = connection.execute(
                "SELECT 1 FROM first_letters_discovery_native_adapters_v19 "
                "WHERE reservation_id=?", (reservation_id,),
            ).fetchone()
            if adapter_probe is not None and _job_id is None:
                raise ValueError(
                    "controlled discovery reservations require a job ID claim"
                )
            if _job_id is not None:
                graph = connection.execute(
                    """SELECT j.*,d.dispatch_json,d.dispatch_sha256 AS d_sha,
                              a.adapter_json,a.adapter_sha256 AS a_sha,
                              a.profile_bytes
                         FROM first_letters_discovery_jobs_v19 j
                         JOIN first_letters_discovery_dispatches_v19 d
                           ON d.dispatch_id=j.dispatch_id
                         JOIN first_letters_discovery_native_adapters_v19 a
                           ON a.reservation_id=j.reservation_id
                        WHERE j.job_id=? AND j.reservation_id=?""",
                    (_job_id, reservation_id),
                ).fetchone()
                if graph is None:
                    raise ValueError("discovery job graph is missing")
                persisted_job = _load(graph["job_json"])
                if (
                    persisted_job.get("job_id") != _job_id
                    or persisted_job.get("job_sha256") != graph["job_sha256"]
                    or persisted_job.get("job_sha256") != content_sha256({
                        key: value for key, value in persisted_job.items()
                        if key != "job_sha256"
                    })
                    or persisted_job.get("item_id") != item_id
                    or graph["item_id"] != item_id
                    or persisted_job.get("dispatch_id") != graph["dispatch_id"]
                    or persisted_job.get("dispatch_sha256") != graph["d_sha"]
                    or persisted_job.get("profile_file_sha256")
                        != hashlib.sha256(profile_bytes).hexdigest()
                    or bytes(graph["profile_bytes"]) != profile_bytes
                ):
                    raise ValueError("discovery job graph differs before claim")
            reservation_row = connection.execute(
                """SELECT r.reservation_json,
                          r.reservation_sha256 AS registered_reservation_sha256,
                          w.work_json,w.work_sha256 AS registered_work_sha256
                     FROM first_letters_discovery_compute_reservations r
                     JOIN first_letters_discovery_work_bindings w
                       ON w.reservation_id=r.reservation_id
                    WHERE r.reservation_id=?""",
                (reservation_id,),
            ).fetchone()
            if reservation_row is None:
                raise ValueError("discovery reservation/work authority is missing")
            reservation = _load(reservation_row["reservation_json"])
            work = _load(reservation_row["work_json"])
            if _job_id is not None:
                mission_id = reservation.get("mission_id")
                if connection.execute(
                    "SELECT 1 FROM first_letters_discovery_compute_blocks "
                    "WHERE mission_id=?", (mission_id,),
                ).fetchone() is not None:
                    raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
                try:
                    current_manifest, current_complete, _ = (
                        self._derive_first_letters_discovery_history_tx(
                            connection, mission_id=mission_id,
                        )
                    )
                    adapter = _load(graph["adapter_json"])
                    native = adapter.get("native_authority") or {}
                    historical_authority = (
                        native
                        if adapter.get("producer_kind")
                            == "BASELINE_RECONCILIATION"
                        else native.get("baseline_reconciliation") or {}
                    )
                    expected_manifest_sha = historical_authority.get(
                        "history_manifest_sha256"
                    )
                    current_history_valid = (
                        current_complete
                        and content_sha256(current_manifest)
                            == expected_manifest_sha
                    )
                except Exception as error:
                    current_manifest = {
                        "schema": "campaignx.first_letters_discovery_"
                            "history_derivation_failure.v1",
                        "mission_id": mission_id,
                        "error_type": type(error).__name__,
                        "retained_execution_graphs": [],
                        "allow_unvalidated": False,
                    }
                    current_history_valid = False
                if not current_history_valid:
                    self._persist_discovery_history_incomplete_tx(
                        connection, mission_id=mission_id,
                        manifest=current_manifest,
                        reason="CONTROL_INCOMPLETE_COMPUTE_LEDGER",
                    )
                    connection.commit()
                    raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
            if (
                reservation.get("source") == "IMPORTED_HISTORICAL_EXACT"
                or work.get("dispatch_kind") == "HISTORICAL_IMPORT_BINDING"
            ):
                raise ValueError(
                    "historical discovery imports are non-executable"
                )
            if (reservation.get("reservation_sha256") !=
                    reservation_row["registered_reservation_sha256"]
                    or reservation.get("reservation_sha256") != content_sha256({
                    key: value for key, value in reservation.items()
                    if key not in {"reservation_sha256", "created_at"}
                })
                    or work.get("work_sha256") !=
                        reservation_row["registered_work_sha256"]
                    or work.get("work_sha256") != content_sha256({
                        key: value for key, value in work.items()
                        if key != "work_sha256"
                    })):
                raise ValueError("discovery persisted reservation/work authority drift")
            authority = work.get("work_authority") or {}
            matches = [
                row for row in authority.get("ordered_item_bindings", [])
                if row.get("item_id") == item_id
            ]
            if (len(matches) != 1 or item_id not in reservation["ordered_item_ids"]
                    or item_id not in work["ordered_item_ids"]
                    or reservation["reservation_sha256"] !=
                        work["reservation_sha256"]
                    or reservation["work_authority_sha256"] !=
                        authority.get("work_authority_sha256")
                    or authority.get("profile_sha256") != profile_sha):
                raise ValueError(
                    "discovery reservation/work item or profile binding differs"
                )
            binding = matches[0]
            source_row = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (binding["source_snapshot_id"],),
            ).fetchone()
            if source_row is None:
                raise ValueError("discovery work-item source snapshot is unavailable")
            source = self._snapshot(source_row)
            source_snapshot_sha = source.get("source_snapshot_sha256")
            if (source_snapshot_sha != binding["source_snapshot_sha256"]
                    or source_snapshot_sha != authority.get("source_sha256")
                    or source["sample_id"] != binding["sample_id"]):
                raise ValueError("discovery work-item source identity is unregistered")
            parent_task_id = binding["parent_task_id"]
            parent_attempt_id = binding["parent_attempt_id"]
            if parent_task_id is not None:
                parent = connection.execute(
                    "SELECT * FROM tasks WHERE task_id=?", (parent_task_id,),
                ).fetchone()
                if (parent is None or parent["mission_id"] != reservation["mission_id"]
                        or parent["source_snapshot_id"] != source["source_snapshot_id"]
                        or parent["cell_id"] != item_id):
                    raise ValueError("optional discovery parent lineage is unregistered")
                parent_payload = _load(parent["payload_json"])
                expected_p0_id = parent_payload.get(
                    "accepted_p0_artifact_id",
                    parent_payload.get("p0_artifact_id"),
                )
                expected_p0_sha = parent_payload.get(
                    "accepted_p0_artifact_sha256",
                    parent_payload.get("p0_artifact_sha256"),
                )
                if (binding["sample_id"] != parent_payload.get("sample_id")
                        or binding["grid_version"] != parent["grid_version"]
                        or binding["scientific_opportunity_id"] !=
                            parent_payload.get("scientific_opportunity_id")
                        or binding["accepted_p0_artifact_id"] != expected_p0_id
                        or binding["accepted_p0_artifact_sha256"] !=
                            expected_p0_sha):
                    raise ValueError(
                        "discovery opportunity/P0 authority differs from "
                        "persisted parent task"
                    )
            if parent_attempt_id is not None:
                attempt = connection.execute(
                    "SELECT task_id FROM attempts WHERE attempt_id=?",
                    (parent_attempt_id,),
                ).fetchone()
                if attempt is None or attempt["task_id"] != parent_task_id:
                    raise ValueError("optional discovery parent attempt is unregistered")
            derived = _derive_first_letters_discovery_run_authority(
                profile_bytes=profile_bytes, reservation=reservation, work=work,
                binding=binding, source=source,
            )
            run_id = derived["run_authority"]["run_id"]
            if connection.execute(
                "SELECT 1 FROM first_letters_discovery_evidence_runs WHERE run_id=?",
                (run_id,),
            ).fetchone() is not None:
                raise ValueError("discovery evidence run already claimed")
            registration = self._discovery_executor_registration_from_connection(
                connection
            )
            lease_expires_at = _deadline(lease_seconds)
            derived = _bind_first_letters_discovery_executor_claim(
                derived=derived,
                registration=registration,
                lease_expires_at=lease_expires_at,
            )
            executor_claim_token = derived.pop("_executor_claim_token")
            profile_sha = derived["profile_file_sha256"]
            provider_request = derived["provider_request"]
            run_authority = derived["run_authority"]
            worker_id = run_authority["worker_id"]
            run_token = secrets.token_urlsafe(32)
            connection.execute(
                """INSERT INTO first_letters_discovery_evidence_runs
                   (run_id,reservation_id,mission_id,request_id,parent_task_id,
                    parent_attempt_id,worker_id,cell_id,source_snapshot_id,
                    run_token_sha256,lease_expires_at,profile_bytes,
                    profile_file_sha256,provider_request_json,
                    run_authority_json,run_authority_sha256,state,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'CLAIMED',?)""",
                (
                    run_id, reservation_id, reservation["mission_id"],
                    reservation["request_id"], parent_task_id,
                    parent_attempt_id, worker_id, item_id,
                    source["source_snapshot_id"],
                    hashlib.sha256(run_token.encode("utf-8")).hexdigest(),
                    lease_expires_at, profile_bytes, profile_sha,
                    _dump(provider_request), _dump(run_authority),
                    run_authority["run_authority_sha256"], utc_now(),
                ),
            )
            claim = run_authority["executor_claim"]
            connection.execute(
                """INSERT INTO first_letters_discovery_executor_claims
                   (claim_id,run_id,worker_id,executor_id,executor_sha256,
                    capability,claim_attempt_number,
                    execution_lease_token_sha256,lease_expires_at,claim_json,
                    claim_sha256,state,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,'CLAIMED',?)""",
                (
                    claim["claim_id"], run_id, claim["worker_id"],
                    claim["executor_id"], claim["executor_sha256"],
                    claim["capability"], claim["claim_attempt_number"],
                    claim["execution_lease_token_sha256"],
                    claim["lease_expires_at"], _dump(claim),
                    claim["claim_sha256"], utc_now(),
                ),
            )
            _accept_first_letters_discovery_executor_claim(
                executor=self._first_letters_discovery_executor,
                run_id=run_id, claim_token=executor_claim_token,
            )
        return _FirstLettersDiscoveryRunHandle(
            run_id=run_id, run_token=run_token, worker_id=worker_id,
            cell_id=item_id, provider_request=copy.deepcopy(provider_request),
        )

    def claim_first_letters_discovery_job(
        self, *, job_id: str, lease_seconds: int,
    ):
        """Claim one immutable v19 job without caller content authority."""

        from .discovery_bridge import build_first_letters_discovery_job_claim

        if not isinstance(job_id, str) or not job_id:
            raise ValueError("discovery job ID is invalid")
        with self.connect() as connection:
            row = connection.execute(
                """SELECT j.reservation_id,j.item_id,a.profile_bytes,
                          r.mission_id,r.request_id
                     FROM first_letters_discovery_jobs_v19 j
                     JOIN first_letters_discovery_native_adapters_v19 a
                       ON a.reservation_id=j.reservation_id
                     JOIN first_letters_discovery_compute_reservations r
                       ON r.reservation_id=j.reservation_id
                    WHERE j.job_id=?""",
                (job_id,),
            ).fetchone()
        if row is None:
            raise ValueError("discovery job graph is missing")
        branch = self.read_first_letters_discovery_request(
            row["mission_id"], row["request_id"],
        )
        jobs = [job for job in branch["jobs"] if job["job_id"] == job_id]
        if len(jobs) != 1:
            raise ValueError("discovery job graph is ambiguous")
        run_handle = self._begin_first_letters_discovery_evidence_run(
            lease_seconds=lease_seconds, reservation_id=row["reservation_id"],
            item_id=row["item_id"], profile_bytes=bytes(row["profile_bytes"]),
            _job_id=job_id,
        )
        claim = build_first_letters_discovery_job_claim(
            job=jobs[0], dispatch=branch["dispatch"],
            adapter=branch["adapter"], reservation=branch["reservation"],
            run_handle=run_handle,
        )
        self.revalidate_first_letters_discovery_job_claim(claim=claim)
        return claim

    def revalidate_first_letters_discovery_job_claim(self, *, claim) -> None:
        """Re-resolve a sealed active claim immediately before provider access."""

        from .discovery_bridge import (
            validate_first_letters_discovery_job_claim,
        )

        claim = validate_first_letters_discovery_job_claim(claim)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT j.job_json,j.job_sha256,j.item_id,
                          d.dispatch_json,d.dispatch_sha256,
                          a.adapter_json,a.adapter_sha256,a.profile_bytes,
                          r.reservation_json,r.reservation_sha256,
                          r.mission_id,r.request_id
                     FROM first_letters_discovery_jobs_v19 j
                     JOIN first_letters_discovery_dispatches_v19 d
                       ON d.dispatch_id=j.dispatch_id
                     JOIN first_letters_discovery_native_adapters_v19 a
                       ON a.reservation_id=j.reservation_id
                     JOIN first_letters_discovery_compute_reservations r
                       ON r.reservation_id=j.reservation_id
                    WHERE j.job_id=?""",
                (claim.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("discovery claim graph is missing")
            if (
                row["job_sha256"] != claim.job_sha256
                or row["item_id"] != claim.item_id
                or row["dispatch_sha256"] != claim.dispatch_sha256
                or row["adapter_sha256"] != claim.adapter_sha256
                or row["reservation_sha256"] != claim.reservation_sha256
                or hashlib.sha256(bytes(row["profile_bytes"])).hexdigest()
                    != claim.profile_file_sha256
            ):
                raise ValueError("discovery claim graph differs")
            branch_identity = (row["mission_id"], row["request_id"])
            if connection.execute(
                "SELECT 1 FROM first_letters_discovery_compute_blocks "
                "WHERE mission_id=?", (row["mission_id"],),
            ).fetchone() is not None:
                raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
            adapter = _load(row["adapter_json"])
            native = adapter.get("native_authority") or {}
            reconciliation = (
                native if adapter.get("producer_kind")
                    == "BASELINE_RECONCILIATION"
                else native.get("baseline_reconciliation") or {}
            )
            try:
                current_manifest, current_complete, _ = (
                    self._derive_first_letters_discovery_history_tx(
                        connection, mission_id=row["mission_id"],
                    )
                )
                current_history_valid = (
                    current_complete
                    and content_sha256(current_manifest)
                        == reconciliation.get("history_manifest_sha256")
                )
            except Exception as error:
                current_manifest = {
                    "schema": "campaignx.first_letters_discovery_"
                        "history_derivation_failure.v1",
                    "mission_id": row["mission_id"],
                    "error_type": type(error).__name__,
                    "retained_execution_graphs": [],
                    "allow_unvalidated": False,
                }
                current_history_valid = False
            if not current_history_valid:
                self._persist_discovery_history_incomplete_tx(
                    connection, mission_id=row["mission_id"],
                    manifest=current_manifest,
                    reason="CONTROL_INCOMPLETE_COMPUTE_LEDGER",
                )
                connection.commit()
                raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
            self._first_letters_discovery_lifecycle_claim(
                connection, run_handle=claim._run_handle,
                expected_state="CLAIMED", require_live_lease=True,
            )
        branch = self.read_first_letters_discovery_request(*branch_identity)
        if [job["job_id"] for job in branch["jobs"]].count(claim.job_id) != 1:
            raise ValueError("discovery claim dispatch graph differs")

    def _first_letters_discovery_lifecycle_claim(
        self, connection: sqlite3.Connection, *,
        run_handle: _FirstLettersDiscoveryRunHandle,
        expected_state: str, require_live_lease: bool,
    ) -> tuple[sqlite3.Row, sqlite3.Row, dict[str, Any], dict[str, Any]]:
        from .seed_probe import (
            _validated_first_letters_discovery_executor_claim,
        )

        if type(run_handle) is not _FirstLettersDiscoveryRunHandle:
            raise ValueError("discovery lifecycle requires the sealed run handle")
        row = connection.execute(
            "SELECT * FROM first_letters_discovery_evidence_runs WHERE run_id=?",
            (run_handle.run_id,),
        ).fetchone()
        if (row is None or row["state"] != expected_state
                or row["worker_id"] != run_handle.worker_id
                or row["cell_id"] != run_handle.cell_id
                or row["run_token_sha256"] != hashlib.sha256(
                    run_handle.run_token.encode("utf-8")
                ).hexdigest()
                or require_live_lease and row["lease_expires_at"] <= utc_now()):
            raise ValueError(
                f"discovery claim owner/token must hold a live {expected_state} lease"
            )
        authority = _load(row["run_authority_json"])
        provider_request = _load(row["provider_request_json"])
        if (authority.get("run_authority_sha256") !=
                row["run_authority_sha256"]
                or content_sha256({
                    key: value for key, value in authority.items()
                    if key != "run_authority_sha256"
                }) != row["run_authority_sha256"]):
            raise ValueError("discovery persisted run authority drift")
        registration = self._discovery_executor_registration_from_connection(
            connection
        )
        claim = _validated_first_letters_discovery_executor_claim(
            authority.get("executor_claim"), run_authority=authority,
            registration=registration, provider_request=provider_request,
        )
        claim_row = connection.execute(
            "SELECT * FROM first_letters_discovery_executor_claims WHERE run_id=?",
            (run_handle.run_id,),
        ).fetchone()
        if (claim_row is None or claim_row["state"] != expected_state
                or _load(claim_row["claim_json"]) != claim
                or claim_row["claim_sha256"] != claim["claim_sha256"]
                or claim_row["worker_id"] != row["worker_id"]
                or claim_row["lease_expires_at"] != row["lease_expires_at"]
                or claim_row["lease_expires_at"] != claim["lease_expires_at"]
                or require_live_lease and
                    claim_row["lease_expires_at"] <= utc_now()):
            raise ValueError("discovery executor claim owner/lease is invalid")
        ownership = getattr(
            self._first_letters_discovery_executor,
            "first_letters_discovery_claim_token", None,
        )
        claim_token = (
            ownership(run_id=run_handle.run_id) if callable(ownership) else None
        )
        if (not isinstance(claim_token, str)
                or hashlib.sha256(claim_token.encode("utf-8")).hexdigest() !=
                    claim["execution_lease_token_sha256"]):
            raise ValueError("discovery executor claim ownership is required")
        return row, claim_row, authority, claim

    def start_first_letters_discovery_evidence_run(
        self, *, run_handle: _FirstLettersDiscoveryRunHandle,
    ) -> dict[str, Any]:
        """Atomically enter RUNNING immediately before provider execution."""

        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._first_letters_discovery_lifecycle_claim(
                connection, run_handle=run_handle, expected_state="CLAIMED",
                require_live_lease=True,
            )
            connection.execute(
                """UPDATE first_letters_discovery_evidence_runs
                      SET state='RUNNING',started_at=?,last_heartbeat_at=?
                    WHERE run_id=?""",
                (now, now, run_handle.run_id),
            )
            connection.execute(
                """UPDATE first_letters_discovery_executor_claims
                      SET state='RUNNING',started_at=?,last_heartbeat_at=?
                    WHERE run_id=?""",
                (now, now, run_handle.run_id),
            )
        return self.read_first_letters_discovery_evidence_run_status(
            run_handle.run_id
        )

    def heartbeat_first_letters_discovery_evidence_run(
        self, *, run_handle: _FirstLettersDiscoveryRunHandle,
        lease_seconds: int,
    ) -> dict[str, Any]:
        """Extend one active claim while preserving every content binding."""

        if (isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int)
                or lease_seconds < 30):
            raise ValueError("discovery heartbeat lease must be at least 30 seconds")
        from .seed_probe import _task6_sha256

        now = utc_now()
        lease_expires_at = _deadline(lease_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _, _, authority, claim = self._first_letters_discovery_lifecycle_claim(
                connection, run_handle=run_handle, expected_state="RUNNING",
                require_live_lease=True,
            )
            claim_core = {
                key: copy.deepcopy(value) for key, value in claim.items()
                if key != "claim_sha256"
            }
            claim_core["lease_expires_at"] = lease_expires_at
            renewed_claim = {
                **claim_core, "claim_sha256": _task6_sha256(claim_core),
            }
            authority_core = {
                key: copy.deepcopy(value) for key, value in authority.items()
                if key != "run_authority_sha256"
            }
            authority_core["executor_claim"] = renewed_claim
            renewed_authority = {
                **authority_core,
                "run_authority_sha256": _task6_sha256(authority_core),
            }
            connection.execute(
                """UPDATE first_letters_discovery_evidence_runs
                      SET lease_expires_at=?,run_authority_json=?,
                          run_authority_sha256=?,last_heartbeat_at=?
                    WHERE run_id=? AND state='RUNNING'""",
                (
                    lease_expires_at, _dump(renewed_authority),
                    renewed_authority["run_authority_sha256"], now,
                    run_handle.run_id,
                ),
            )
            connection.execute(
                """UPDATE first_letters_discovery_executor_claims
                      SET lease_expires_at=?,claim_json=?,claim_sha256=?,
                          last_heartbeat_at=?
                    WHERE run_id=? AND state='RUNNING'""",
                (
                    lease_expires_at, _dump(renewed_claim),
                    renewed_claim["claim_sha256"], now, run_handle.run_id,
                ),
            )
        return self.read_first_letters_discovery_evidence_run_status(
            run_handle.run_id
        )

    def mark_first_letters_discovery_evidence_run_incomplete(
        self, *, run_handle: _FirstLettersDiscoveryRunHandle, reason: str,
    ) -> dict[str, Any]:
        """Permanently close an ambiguous RUNNING job without re-execution."""

        allowed = {
            "PROVIDER_RESPONSE_AMBIGUOUS_AFTER_RUNNING",
            "COMPLETION_READBACK_AMBIGUOUS_AFTER_RUNNING",
            "ACTIVE_CLAIM_HEARTBEAT_FAILED",
            "COMPLETION_FAILED_AFTER_RUNNING",
            "START_RESPONSE_AMBIGUOUS_AFTER_RUNNING",
        }
        if reason not in allowed:
            raise ValueError("discovery incomplete reason is not a closed terminal")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._first_letters_discovery_lifecycle_claim(
                connection, run_handle=run_handle, expected_state="RUNNING",
                require_live_lease=False,
            )
            connection.execute(
                """UPDATE first_letters_discovery_evidence_runs
                      SET state='CONTROL_INCOMPLETE',incomplete_at=?,
                          incomplete_reason=? WHERE run_id=?""",
                (now, reason, run_handle.run_id),
            )
            connection.execute(
                """UPDATE first_letters_discovery_executor_claims
                      SET state='CONTROL_INCOMPLETE',incomplete_at=?,
                          incomplete_reason=? WHERE run_id=?""",
                (now, reason, run_handle.run_id),
            )
        return self.read_first_letters_discovery_evidence_run_status(
            run_handle.run_id
        )

    def reconcile_expired_first_letters_discovery_evidence_run(
        self, *, run_id: str,
    ) -> dict[str, Any]:
        """Close a cryptographically intact expired RUNNING owner as lost."""

        from .seed_probe import (
            _validated_first_letters_discovery_executor_claim,
        )

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("discovery run ID is invalid")
        now = utc_now()
        reason = "WORKER_LOST_AFTER_RUNNING"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM first_letters_discovery_evidence_runs "
                "WHERE run_id=?",
                (run_id,),
            ).fetchone()
            claim_row = connection.execute(
                "SELECT * FROM first_letters_discovery_executor_claims "
                "WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if (row is None or claim_row is None
                    or row["state"] != "RUNNING"
                    or claim_row["state"] != "RUNNING"
                    or row["lease_expires_at"] > now
                    or claim_row["lease_expires_at"] > now
                    or row["lease_expires_at"] !=
                        claim_row["lease_expires_at"]
                    or row["worker_id"] != claim_row["worker_id"]
                    or not row["started_at"] or not row["last_heartbeat_at"]
                    or not claim_row["started_at"]
                    or not claim_row["last_heartbeat_at"]
                    or row["completed_at"] or claim_row["completed_at"]
                    or row["incomplete_at"] or claim_row["incomplete_at"]
                    or row["incomplete_reason"]
                    or claim_row["incomplete_reason"]):
                raise ValueError(
                    "discovery reconciliation requires one expired RUNNING owner"
                )
            authority = _load(row["run_authority_json"])
            provider_request = _load(row["provider_request_json"])
            if (authority.get("run_authority_sha256") !=
                    row["run_authority_sha256"]
                    or content_sha256({
                        key: value for key, value in authority.items()
                        if key != "run_authority_sha256"
                    }) != row["run_authority_sha256"]):
                raise ValueError("discovery persisted run authority drift")
            registration = (
                self._persisted_discovery_executor_registration_from_connection(
                    connection, worker_id=row["worker_id"],
                )
            )
            claim = _validated_first_letters_discovery_executor_claim(
                authority.get("executor_claim"), run_authority=authority,
                registration=registration, provider_request=provider_request,
            )
            if (_load(claim_row["claim_json"]) != claim
                    or claim_row["claim_sha256"] != claim["claim_sha256"]
                    or claim_row["lease_expires_at"] !=
                        claim["lease_expires_at"]):
                raise ValueError("discovery expired executor claim is invalid")
            if connection.execute(
                "SELECT 1 FROM first_letters_discovery_evidence_sets "
                "WHERE run_id=?",
                (run_id,),
            ).fetchone() is not None:
                raise ValueError("expired RUNNING run already has evidence")
            connection.execute(
                """UPDATE first_letters_discovery_evidence_runs
                      SET state='CONTROL_INCOMPLETE',incomplete_at=?,
                          incomplete_reason=?
                    WHERE run_id=? AND state='RUNNING'""",
                (now, reason, run_id),
            )
            connection.execute(
                """UPDATE first_letters_discovery_executor_claims
                      SET state='CONTROL_INCOMPLETE',incomplete_at=?,
                          incomplete_reason=?
                    WHERE run_id=? AND state='RUNNING'""",
                (now, reason, run_id),
            )
        return self.read_first_letters_discovery_evidence_run_status(run_id)

    def complete_first_letters_discovery_evidence_run(
        self, *, run_handle: _FirstLettersDiscoveryRunHandle,
        provider_response_bytes: bytes,
        failpoint: str | None = None,
    ) -> dict[str, Any]:
        """Persist one immutable evidence set from a live discovery claim."""

        from .seed_probe import (
            _measure_first_letters_discovery_with_executor,
            _produce_first_letters_discovery_evidence_set,
            _validated_first_letters_discovery_executor_claim,
        )

        if type(run_handle) is not _FirstLettersDiscoveryRunHandle:
            raise ValueError("discovery completion requires the sealed run handle")
        if not isinstance(provider_response_bytes, bytes):
            raise ValueError("provider response must be exact bytes")
        if failpoint not in {None, "evidence.after_commit_before_response"}:
            raise ValueError("unknown discovery evidence failpoint")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM first_letters_discovery_evidence_runs WHERE run_id=?",
                (run_handle.run_id,),
            ).fetchone()
            if (row is None or row["state"] != "RUNNING"
                    or row["worker_id"] != run_handle.worker_id
                    or row["cell_id"] != run_handle.cell_id
                    or row["run_token_sha256"] != hashlib.sha256(
                        run_handle.run_token.encode("utf-8")
                    ).hexdigest()
                    or row["lease_expires_at"] <= utc_now()):
                raise ValueError(
                    "discovery run claim is stale, wrong, incomplete, or not RUNNING"
                )
            reservation_row = connection.execute(
                "SELECT reservation_json FROM first_letters_discovery_compute_reservations WHERE reservation_id=?",
                (row["reservation_id"],),
            ).fetchone()
            if reservation_row is None:
                raise ValueError("discovery run reservation disappeared")
            run_authority = _load(row["run_authority_json"])
            provider_request = _load(row["provider_request_json"])
            if (row["run_authority_sha256"] !=
                    run_authority.get("run_authority_sha256")
                    or row["run_authority_sha256"] != content_sha256({
                        key: value for key, value in run_authority.items()
                        if key != "run_authority_sha256"
                    })):
                raise ValueError("discovery persisted run authority drift")
            registration = self._discovery_executor_registration_from_connection(
                connection
            )
            claim_row = connection.execute(
                "SELECT * FROM first_letters_discovery_executor_claims "
                "WHERE run_id=?",
                (run_handle.run_id,),
            ).fetchone()
            claim = _validated_first_letters_discovery_executor_claim(
                run_authority.get("executor_claim"),
                run_authority=run_authority, registration=registration,
                provider_request=provider_request,
            )
            if (claim_row is None or claim_row["state"] != "RUNNING"
                    or _load(claim_row["claim_json"]) != claim
                    or claim_row["claim_sha256"] != claim["claim_sha256"]
                    or claim_row["worker_id"] != row["worker_id"]
                    or claim_row["executor_id"] != claim["executor_id"]
                    or claim_row["executor_sha256"] !=
                        claim["executor_sha256"]
                    or claim_row["capability"] != claim["capability"]
                    or claim_row["execution_lease_token_sha256"] !=
                        claim["execution_lease_token_sha256"]
                    or claim_row["lease_expires_at"] !=
                        claim["lease_expires_at"]
                    or claim_row["lease_expires_at"] !=
                        row["lease_expires_at"]
                    or claim_row["lease_expires_at"] <= utc_now()):
                raise ValueError("DISCOVERY_EXECUTOR_CLAIM_STALE")
            source_row = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (row["source_snapshot_id"],),
            ).fetchone()
            if source_row is None:
                raise ValueError("discovery run source disappeared")
            measurements = _measure_first_letters_discovery_with_executor(
                executor=self._first_letters_discovery_executor,
                run_authority=run_authority,
                provider_request=provider_request,
                provider_response_bytes=provider_response_bytes,
                source_snapshot=self._snapshot(source_row),
            )
            registered = _produce_first_letters_discovery_evidence_set(
                run_authority=run_authority,
                profile_bytes=bytes(row["profile_bytes"]),
                provider_request=provider_request,
                provider_response_bytes=provider_response_bytes,
                measurements=measurements,
                reservation=_load(reservation_row["reservation_json"]),
            )
            evidence_json = copy.deepcopy({
                key: value for key, value in registered.items()
                if key not in {"profile_bytes", "retained_files"}
            })
            evidence_json["inputs"]["provider_response"].pop("response_bytes")
            evidence_sha = content_sha256(evidence_json)
            connection.execute(
                """INSERT INTO first_letters_discovery_evidence_sets
                   (evidence_set_id,run_id,evidence_json,evidence_set_sha256,created_at)
                   VALUES(?,?,?,?,?)""",
                (
                    registered["evidence_set_id"], run_handle.run_id,
                    _dump(evidence_json), evidence_sha, utc_now(),
                ),
            )
            for file_order, retained in enumerate(registered["retained_files"]):
                payload = retained["bytes"]
                connection.execute(
                    """INSERT INTO first_letters_discovery_evidence_files
                       (evidence_set_id,file_order,relative_path,role,payload,
                        byte_count,sha256) VALUES(?,?,?,?,?,?,?)""",
                    (
                        registered["evidence_set_id"], file_order,
                        retained["relative_path"], retained["role"], payload,
                        len(payload), hashlib.sha256(payload).hexdigest(),
                    ),
                )
            connection.execute(
                """UPDATE first_letters_discovery_evidence_runs
                      SET state='COMPLETED',completed_at=? WHERE run_id=?""",
                (utc_now(), run_handle.run_id),
            )
            connection.execute(
                """UPDATE first_letters_discovery_executor_claims
                      SET state='COMPLETED',completed_at=? WHERE run_id=?""",
                (utc_now(), run_handle.run_id),
            )
        if failpoint == "evidence.after_commit_before_response":
            raise RuntimeError(
                "discovery evidence response lost; READBACK_BY_RUN_ID_REQUIRED"
            )
        return self.read_first_letters_discovery_evidence_set(
            registered["evidence_set_id"]
        )

    @staticmethod
    def _read_first_letters_discovery_evidence_set_from_connection(
        connection: sqlite3.Connection, evidence_set_id: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            """SELECT e.*,r.profile_bytes,r.profile_file_sha256,
                      r.run_authority_json,r.run_authority_sha256,
                      c.reservation_json
                 FROM first_letters_discovery_evidence_sets e
                 JOIN first_letters_discovery_evidence_runs r ON r.run_id=e.run_id
                 JOIN first_letters_discovery_compute_reservations c
                   ON c.reservation_id=r.reservation_id
                WHERE e.evidence_set_id=? AND r.state='COMPLETED'""",
            (evidence_set_id,),
        ).fetchone()
        if row is None:
            raise KeyError(evidence_set_id)
        evidence = _load(row["evidence_json"])
        if (evidence.get("evidence_set_id") != evidence_set_id
                or content_sha256(evidence) != row["evidence_set_sha256"]
                or hashlib.sha256(bytes(row["profile_bytes"])).hexdigest() !=
                    row["profile_file_sha256"]
                or content_sha256({
                    key: value
                    for key, value in _load(row["run_authority_json"]).items()
                    if key != "run_authority_sha256"
                }) != row["run_authority_sha256"]):
            raise ValueError("registered discovery evidence-set integrity drift")
        files = connection.execute(
            """SELECT relative_path,role,payload,byte_count,sha256
                 FROM first_letters_discovery_evidence_files
                WHERE evidence_set_id=? ORDER BY file_order""",
            (evidence_set_id,),
        ).fetchall()
        retained_files = []
        response_bytes = None
        for file_row in files:
            payload = bytes(file_row["payload"])
            if (len(payload) != file_row["byte_count"]
                    or hashlib.sha256(payload).hexdigest() != file_row["sha256"]):
                raise ValueError("registered discovery evidence file drift")
            retained_files.append({
                "relative_path": file_row["relative_path"],
                "role": file_row["role"], "bytes": payload,
            })
            if file_row["role"] == "CANDIDATE_PROVIDER_RESPONSE":
                if response_bytes is not None:
                    raise ValueError("registered provider response is ambiguous")
                response_bytes = payload
        if response_bytes is None:
            raise ValueError("registered provider response is missing")
        evidence["inputs"]["provider_response"]["response_bytes"] = response_bytes
        return {
            "evidence_set_id": evidence_set_id,
            "execution_authority": evidence["execution_authority"],
            "profile_bytes": bytes(row["profile_bytes"]),
            "inputs": evidence["inputs"],
            "candidate_outcomes": evidence["candidate_outcomes"],
            "retained_files": retained_files,
            "reservation": _load(row["reservation_json"]),
            "selection": evidence["selection"],
        }

    def read_first_letters_discovery_evidence_set(
        self, evidence_set_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            return self._read_first_letters_discovery_evidence_set_from_connection(
                connection, evidence_set_id
            )

    def read_first_letters_discovery_evidence_run_status(
        self, run_id: str,
    ) -> dict[str, Any]:
        """Read the closed producer lifecycle without exposing either token."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("discovery run ID is invalid")
        with self.connect() as connection:
            row = connection.execute(
                """SELECT run_id,worker_id,cell_id,state,lease_expires_at,
                          started_at,last_heartbeat_at,completed_at,
                          incomplete_at,incomplete_reason
                     FROM first_letters_discovery_evidence_runs
                    WHERE run_id=?""",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            evidence = connection.execute(
                "SELECT evidence_set_id FROM first_letters_discovery_evidence_sets "
                "WHERE run_id=?",
                (run_id,),
            ).fetchall()
            claims = connection.execute(
                """SELECT state,lease_expires_at,started_at,last_heartbeat_at,
                          completed_at,incomplete_at,incomplete_reason
                     FROM first_letters_discovery_executor_claims
                    WHERE run_id=?""",
                (run_id,),
            ).fetchall()
        if len(evidence) > 1:
            raise ValueError("discovery run evidence readback is ambiguous")
        if len(claims) != 1:
            raise ValueError("discovery run claim lifecycle is ambiguous")
        claim = claims[0]
        evidence_set_id = evidence[0]["evidence_set_id"] if evidence else None
        if ((row["state"] == "COMPLETED") != (evidence_set_id is not None)
                or row["state"] == "CONTROL_INCOMPLETE"
                and not row["incomplete_reason"]
                or claim["state"] != row["state"]
                or claim["lease_expires_at"] != row["lease_expires_at"]
                or bool(claim["started_at"]) != bool(row["started_at"])
                or bool(claim["last_heartbeat_at"]) !=
                    bool(row["last_heartbeat_at"])
                or bool(claim["completed_at"]) != bool(row["completed_at"])
                or bool(claim["incomplete_at"]) != bool(row["incomplete_at"])
                or claim["incomplete_reason"] != row["incomplete_reason"]):
            raise ValueError("discovery run lifecycle readback is inconsistent")
        return {
            "schema": "campaignx.first_letters_discovery_run_status.v1",
            "run_id": row["run_id"], "worker_id": row["worker_id"],
            "cell_id": row["cell_id"], "state": row["state"],
            "lease_expires_at": row["lease_expires_at"],
            "started_at": row["started_at"],
            "last_heartbeat_at": row["last_heartbeat_at"],
            "completed_at": row["completed_at"],
            "incomplete_at": row["incomplete_at"],
            "incomplete_reason": row["incomplete_reason"],
            "evidence_set_id": evidence_set_id,
        }

    def read_first_letters_discovery_evidence_run(
        self, run_id: str,
    ) -> dict[str, Any]:
        """Recover one committed evidence set without rerunning its producer."""

        with self.connect() as connection:
            rows = connection.execute(
                """SELECT evidence_set_id
                     FROM first_letters_discovery_evidence_sets
                    WHERE run_id=?""",
                (run_id,),
            ).fetchall()
            if len(rows) != 1:
                raise KeyError(run_id)
            return self._read_first_letters_discovery_evidence_set_from_connection(
                connection, rows[0]["evidence_set_id"]
            )

    def build_first_letters_discovery_artifact_and_receipt(
        self, evidence_set_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build only after a concrete SQLite evidence-registry readback."""

        from .seed_probe import (
            _build_first_letters_discovery_artifact_and_receipt_from_evidence_set,
        )

        with self.connect() as connection:
            registered = self._read_first_letters_discovery_evidence_set_from_connection(
                connection, evidence_set_id
            )
        return _build_first_letters_discovery_artifact_and_receipt_from_evidence_set(
            registered, evidence_set_id=evidence_set_id,
        )

    def resolve_discovery_promotion_evidence(
        self, evidence_set_id: str,
    ) -> dict[str, Any]:
        """Resolve a promotion candidate only from one registered evidence ID."""

        from .seed_probe import (
            _resolve_discovery_promotion_evidence_from_evidence_set,
        )

        with self.connect() as connection:
            registered = self._read_first_letters_discovery_evidence_set_from_connection(
                connection, evidence_set_id
            )
            return _resolve_discovery_promotion_evidence_from_evidence_set(
                registered, evidence_set_id=evidence_set_id,
            )

    def discovery_compute_rows(self, mission_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            requests = [row["request_id"] for row in connection.execute(
                """SELECT request_id FROM first_letters_discovery_compute_reservations
                    WHERE mission_id=? ORDER BY created_at,reservation_id""",
                (mission_id,),
            ).fetchall()]
        return [self.read_discovery_compute_request(mission_id, request_id) for request_id in requests]

    def validate_discovery_compute_reservation(
        self, reservation_id: str, reservation_sha256: str, *,
        mission_id: str, work_kind: str, work_authority_sha256: str,
        ordered_item_ids: list[str],
    ) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT r.reservation_json,w.work_json
                     FROM first_letters_discovery_compute_reservations r
                     JOIN first_letters_discovery_work_bindings w
                       ON w.reservation_id=r.reservation_id
                    WHERE r.reservation_id=?""",
                (reservation_id,),
            ).fetchall()
        if len(rows) != 1:
            raise ValueError("discovery compute reservation is missing or incomplete")
        reservation = _load(rows[0]["reservation_json"])
        work = _load(rows[0]["work_json"])
        if (reservation.get("reservation_sha256") != reservation_sha256
                or reservation.get("mission_id") != mission_id
                or reservation.get("work_kind") != work_kind
                or reservation.get("work_authority_sha256") !=
                    work_authority_sha256
                or reservation.get("ordered_item_ids") != ordered_item_ids
                or work.get("reservation_sha256") != reservation_sha256
                or work.get("ordered_item_ids") != ordered_item_ids):
            raise ValueError("discovery compute reservation authority mismatch")
        expected = content_sha256({
            key: row for key, row in reservation.items()
            if key not in {"reservation_sha256", "created_at"}
        })
        if expected != reservation_sha256:
            raise ValueError("discovery compute reservation hash is invalid")
        return reservation

    def record_discovery_compute_outcome(
        self, mission_id: str, request_id: str, outcome: str,
    ) -> None:
        if outcome not in {"CANCELLED", "FAILED", "ABSTAINED"}:
            raise ValueError("unsupported discovery compute outcome")
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO first_letters_discovery_compute_outcomes
                   (mission_id,request_id,outcome,created_at) VALUES(?,?,?,?)
                   ON CONFLICT(mission_id,request_id,outcome) DO NOTHING""",
                (mission_id, request_id, outcome, utc_now()),
            )

    @staticmethod
    def _run_promotion_failpoint(callback, name: str) -> None:
        if callback is not None:
            callback(name)

    def _derive_discovery_promotion_admission_from_connection(
        self, connection: sqlite3.Connection, *, evidence_set_id: str,
        task9_gate: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve every promotion input from registered rows in this txn."""

        from .campaign_decision import authorize_promotion_child
        from .seed_probe import (
            _build_first_letters_discovery_artifact_and_receipt_from_evidence_set,
            load_first_letters_normal_growth_lock,
        )

        registered = self._read_first_letters_discovery_evidence_set_from_connection(
            connection, evidence_set_id
        )
        artifact, receipt = (
            _build_first_letters_discovery_artifact_and_receipt_from_evidence_set(
                registered, evidence_set_id=evidence_set_id,
            )
        )
        if artifact["selection_outcome"] != "DISCOVERY_WINNER_RETAINED":
            raise ValueError("registered discovery evidence has no winner")
        selected = [
            row for row in artifact["candidates"]
            if row["candidate_id"] == artifact["selected_candidate_id"]
        ]
        if len(selected) != 1 or artifact["parent_attempt_id"] is None:
            raise ValueError("registered discovery promotion lineage is incomplete")
        parent = connection.execute(
            "SELECT * FROM tasks WHERE task_id=? AND mission_id=?",
            (artifact["parent_task_id"], artifact["mission_id"]),
        ).fetchone()
        attempt = connection.execute(
            "SELECT * FROM attempts WHERE attempt_id=? AND task_id=?",
            (artifact["parent_attempt_id"], artifact["parent_task_id"]),
        ).fetchone()
        source_row = connection.execute(
            "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
            (artifact["source_snapshot_id"],),
        ).fetchone()
        if parent is None or attempt is None or source_row is None:
            raise ValueError("registered discovery promotion authority is missing")
        payload = _load(parent["payload_json"])
        source = self._snapshot(source_row)
        promotion_scope = (
            source.get("first_letters_discovery_authority") or {}
        ).get("promotion_authority") or {}
        budget_sha = payload.get("campaign_budget_admission_sha256")
        budget_rows = connection.execute(
            """SELECT admission_json FROM campaign_budget_admissions
                WHERE mission_id=? AND sample_id=? AND admission_sha256=?""",
            (artifact["mission_id"], artifact["sample_id"], budget_sha),
        ).fetchall()
        if len(budget_rows) != 1:
            raise ValueError("registered discovery budget authority is missing")
        budget = _load(budget_rows[0]["admission_json"])
        parent_authority = {
            "task_id": parent["task_id"],
            "attempt_id": attempt["attempt_id"],
            "mission_id": parent["mission_id"],
            "sample_id": payload.get("sample_id"),
            "source_snapshot_id": parent["source_snapshot_id"],
            "grid_version": parent["grid_version"],
            "cell_id": parent["cell_id"],
            "policy_version": parent["policy_version"],
            "selection_rank": payload.get("selection_rank"),
            "campaign_budget_admission_sha256": budget_sha,
            "p0_artifact_id": payload.get("p0_artifact_id"),
            "p0_artifact_sha256": payload.get("p0_artifact_sha256"),
            "catalog_snapshot_sha256": parent["catalog_snapshot_sha256"],
        }
        normal_lock = load_first_letters_normal_growth_lock(
            source_snapshot_id=artifact["source_snapshot_id"],
            coordinate=selected[0]["promotion_coordinate_ct_l0_xyz"],
            coordinate_sha256=selected[0]["promotion_coordinate_sha256"],
            deployed_revision=task9_gate["deployed_revision"],
            retry_budget=2,
        )
        admission = authorize_promotion_child(
            parent_task=parent_authority,
            registered_budget_admission=budget,
            active_policy_chain=copy.deepcopy(
                promotion_scope.get("active_policy_chain")
            ),
            benchmark_authorization_v2=copy.deepcopy(
                promotion_scope.get("benchmark_authorization_v2")
            ),
            discovery_receipt=receipt,
            selected_candidate=selected[0],
            normal_growth_lock=normal_lock,
            task9_gate=task9_gate,
        )
        if (admission["scientific_opportunity_id"] !=
                artifact["scientific_opportunity_id"]
                or admission["p0_artifact_id"] !=
                    artifact["accepted_p0_artifact_id"]
                or admission["p0_artifact_sha256"] !=
                    artifact["accepted_p0_artifact_sha256"]):
            raise ValueError(
                "registered discovery evidence differs from promotion authority"
            )
        return admission

    def begin_discovery_promotion(
        self, *, request_id: str, evidence_set_id: str, task9_gate: Any,
        promotion_failpoint=None,
    ) -> dict[str, Any]:
        """Atomically create one fresh ordinary child and terminalize its parent."""

        from .campaign_decision import (  # noqa: PLC0415
            build_promotion_child_task,
            validate_promotion_child_task,
        )
        connection = self.connect()
        committed = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            admission = self._derive_discovery_promotion_admission_from_connection(
                connection, evidence_set_id=evidence_set_id,
                task9_gate=task9_gate,
            )
            mission_id = admission["mission_id"]
            resolver = self._task9_discovery_gate_resolver
            authoritative_gate = (
                resolver(mission_id) if resolver is not None else None
            )
            if authoritative_gate != task9_gate:
                raise ValueError(
                    "TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED"
                )
            from .campaign_decision import (  # noqa: PLC0415
                _validated_task9_discovery_gate,
            )
            _validated_task9_discovery_gate(
                authoritative_gate, mission_id=mission_id
            )
            child = build_promotion_child_task(admission)
            request_core = {
                "mission_id": mission_id, "request_id": request_id,
                "evidence_set_id": evidence_set_id,
                "admission": admission, "task9_gate": task9_gate,
            }
            request_sha = content_sha256(request_core)
            existing = connection.execute(
                """SELECT request_sha256 FROM first_letters_discovery_promotions
                    WHERE mission_id=? AND request_id=?""",
                (mission_id, request_id),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_sha:
                    raise ValueError("PROMOTION_AUTHORITY_CONFLICT")
                connection.commit()
                committed = True
                return self.read_discovery_promotion(mission_id, request_id)
            parent = connection.execute(
                "SELECT * FROM tasks WHERE task_id=? AND mission_id=?",
                (admission["parent_task_id"], mission_id),
            ).fetchone()
            if (parent is None or parent["state"] not in ACTIVE_STATES
                    or parent["active_attempt_id"] != admission["parent_attempt_id"]):
                raise ValueError("promotion parent task/attempt is not active authority")
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id=? AND task_id=?",
                (admission["parent_attempt_id"], admission["parent_task_id"]),
            ).fetchone()
            if attempt is None:
                raise ValueError("promotion parent attempt is missing")
            source_row = connection.execute(
                "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                (admission["source_snapshot_id"],),
            ).fetchone()
            if source_row is None:
                raise ValueError("promotion authoritative source is missing")
            validate_promotion_child_task(
                child, admission=admission,
                registered_budget_admission=admission[
                    "registered_budget_admission"],
                authoritative_source_snapshot=self._snapshot(source_row),
            )
            promotion_id = stable_id("first-letters-discovery-promotion", {
                "mission_id": mission_id,
                "scientific_opportunity_id": admission[
                    "scientific_opportunity_id"],
                "admission_sha256": admission["admission_sha256"],
            })
            authority_core = {
                "schema":
                    "campaignx.first_letters_discovery_promotion_authority.v1",
                "promotion_id": promotion_id,
                "mission_id": mission_id,
                "request_id": request_id,
                "scientific_opportunity_id": admission[
                    "scientific_opportunity_id"],
                "parent_task_id": admission["parent_task_id"],
                "child_task_id": admission["child_task_id"],
                "admission": copy.deepcopy(admission),
                "admission_sha256": admission["admission_sha256"],
                "task9_gate_sha256": task9_gate["gate_sha256"],
                "terminal_state": "CHILD_CREATED_PARENT_TERMINAL",
                "allow_unvalidated": False,
            }
            authority = {
                **authority_core,
                "authority_sha256": content_sha256(authority_core),
            }
            self._run_promotion_failpoint(
                promotion_failpoint, "promotion.before_authority_insert"
            )
            now = utc_now()
            connection.execute(
                """INSERT INTO first_letters_discovery_promotions
                   (promotion_id,mission_id,request_id,scientific_opportunity_id,
                    parent_task_id,child_task_id,admission_sha256,authority_json,
                    authority_sha256,request_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (promotion_id, mission_id, request_id,
                 admission["scientific_opportunity_id"],
                 admission["parent_task_id"], admission["child_task_id"],
                 admission["admission_sha256"], _dump(authority),
                 authority["authority_sha256"], request_sha, now),
            )
            self._run_promotion_failpoint(
                promotion_failpoint,
                "promotion.after_authority_insert_before_child_insert",
            )
            requirements = normalize_resource_requirements(
                child["resource_requirements"]
            )
            payload = {
                **copy.deepcopy(child),
                "promotion_id": promotion_id,
                "promotion_authority_sha256": authority["authority_sha256"],
                "resource_requirements": requirements,
            }
            connection.execute(
                """INSERT INTO tasks
                   (task_id,mission_id,source_snapshot_id,cell_id,grid_version,
                    policy_version,bounds_xyz_json,center_xyz_json,priority,
                    parameter_envelope_json,catalog_snapshot_sha256,payload_json,
                    state,gpu_required,minimum_vram_gb,seed_probe_required,
                    created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (child["task_id"], mission_id, child["source_snapshot_id"],
                 child["cell_id"], child["grid_version"],
                 child["policy_version"], _dump(child["bounds_xyz"]),
                 _dump(child["center_xyz"]), float(child["priority"]),
                 _dump(child["parameter_envelope"]),
                 child["catalog_snapshot_sha256"], _dump(payload), "PENDING",
                 int(requirements["gpu_required"]),
                 requirements["minimum_vram_gb"],
                 int(requirements["seed_probe_required"]), now, now),
            )
            self.event(
                connection, "PROMOTION_CHILD_CREATED",
                {"promotion_id": promotion_id,
                 "promotion_authority_sha256": authority["authority_sha256"]},
                child["task_id"],
            )
            self._run_promotion_failpoint(
                promotion_failpoint,
                "promotion.after_child_insert_before_parent_terminal",
            )
            terminal = {
                "promotion_id": promotion_id,
                "promotion_authority_sha256": authority["authority_sha256"],
                "child_task_id": child["task_id"],
                "scientific_opportunity_id": admission[
                    "scientific_opportunity_id"],
            }
            updated_attempt = connection.execute(
                """UPDATE attempts SET state='DISCOVERY_PROMOTED',result_json=?,
                   updated_at=? WHERE attempt_id=? AND task_id=?""",
                (_dump(terminal), now, admission["parent_attempt_id"],
                 admission["parent_task_id"]),
            ).rowcount
            updated_task = connection.execute(
                """UPDATE tasks SET state='DISCOVERY_PROMOTED',updated_at=?
                   WHERE task_id=? AND active_attempt_id=?""",
                (now, admission["parent_task_id"],
                 admission["parent_attempt_id"]),
            ).rowcount
            if updated_attempt != 1 or updated_task != 1:
                raise RuntimeError("promotion parent terminal transition conflicted")
            self.event(
                connection, "DISCOVERY_PROMOTED", terminal,
                admission["parent_task_id"], admission["parent_attempt_id"],
            )
            self._run_promotion_failpoint(
                promotion_failpoint,
                "promotion.after_parent_terminal_before_commit",
            )
            self._run_promotion_failpoint(
                promotion_failpoint, "promotion.before_commit"
            )
            connection.commit()
            committed = True
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        self._run_promotion_failpoint(
            promotion_failpoint, "promotion.commit_outcome_unknown"
        )
        self._run_promotion_failpoint(
            promotion_failpoint, "promotion.after_commit_before_response"
        )
        return self.read_discovery_promotion(mission_id, request_id)

    def read_discovery_promotion(
        self, mission_id: str, request_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM first_letters_discovery_promotions
                    WHERE mission_id=? AND request_id=?""",
                (mission_id, request_id),
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            child = connection.execute(
                "SELECT state,payload_json FROM tasks WHERE task_id=?",
                (row["child_task_id"],),
            ).fetchone()
            parent = connection.execute(
                "SELECT state,active_attempt_id FROM tasks WHERE task_id=?",
                (row["parent_task_id"],),
            ).fetchone()
        if child is None or parent is None:
            raise ValueError("CONTROL_INCOMPLETE_PROMOTION_READBACK")
        child_value = _load(child["payload_json"])
        child_value["state"] = child["state"]
        return {
            "authority": _load(row["authority_json"]),
            "child": child_value,
            "parent": {
                "task_id": row["parent_task_id"],
                "attempt_id": parent["active_attempt_id"],
                "state": parent["state"],
            },
        }

    def append_discovery_promotion_attempt_binding(
        self, *, promotion_id: str, attempt_number: int, attempt_id: str,
        claim_event_sha256: str, predecessor_attempt_id: str | None,
        retry_reason: str | None,
    ) -> dict[str, Any]:
        if (not isinstance(attempt_number, int) or isinstance(attempt_number, bool)
                or attempt_number < 1):
            raise ValueError("promotion attempt number is invalid")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            promotion = connection.execute(
                "SELECT * FROM first_letters_discovery_promotions WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            if promotion is None:
                raise KeyError(promotion_id)
            authority = _load(promotion["authority_json"])
            normal = authority["admission"]["normal_growth_lock"]
            existing = connection.execute(
                """SELECT binding_json FROM
                   first_letters_discovery_promotion_attempt_bindings
                   WHERE promotion_id=? AND attempt_number=?""",
                (promotion_id, attempt_number),
            ).fetchone()
            prior = connection.execute(
                """SELECT attempt_number,attempt_id,binding_json FROM
                   first_letters_discovery_promotion_attempt_bindings
                   WHERE promotion_id=? ORDER BY attempt_number DESC LIMIT 1""",
                (promotion_id,),
            ).fetchone()
            if attempt_number == 1:
                if predecessor_attempt_id is not None or retry_reason is not None:
                    raise ValueError("first promotion attempt cannot be a retry")
            else:
                allowed_retry_reasons = {
                    "WORKER_FAILURE", "LEASE_EXHAUSTION",
                    "PUBLICATION_FAILURE", "SOURCE_FAILURE",
                }
                if (prior is None or prior["attempt_number"] != attempt_number - 1
                        or predecessor_attempt_id != prior["attempt_id"]
                        or retry_reason not in allowed_retry_reasons
                        or attempt_number > 1 + normal["retry_budget"]):
                    raise ValueError("promotion retry is not authorized")
            core = {
                "schema":
                    "campaignx.first_letters_discovery_promotion_attempt_binding.v1",
                "promotion_id": promotion_id,
                "promotion_authority_sha256": promotion["authority_sha256"],
                "child_task_id": promotion["child_task_id"],
                "attempt_number": attempt_number,
                "attempt_id": attempt_id,
                "claim_event_sha256": claim_event_sha256,
                "normal_full_grow_profile_id": normal[
                    "normal_full_grow_profile_id"],
                "normal_full_grow_profile_sha256": normal[
                    "normal_full_grow_profile_sha256"],
                "growth_parameter_envelope_sha256": normal[
                    "growth_parameter_envelope_sha256"],
                "deployed_revision": normal["deployed_revision"],
                "predecessor_attempt_id": predecessor_attempt_id,
                "retry_reason": retry_reason,
                "scientific_denominator_delta": 0,
                "allow_unvalidated": False,
            }
            binding = {**core, "binding_sha256": content_sha256(core)}
            if existing is not None:
                prior_value = _load(existing["binding_json"])
                if prior_value != binding:
                    raise ValueError("PROMOTION_ATTEMPT_BINDING_CONFLICT")
                return prior_value
            connection.execute(
                """INSERT INTO first_letters_discovery_promotion_attempt_bindings
                   (promotion_id,attempt_number,attempt_id,binding_json,
                    binding_sha256,created_at) VALUES(?,?,?,?,?,?)""",
                (promotion_id, attempt_number, attempt_id, _dump(binding),
                 binding["binding_sha256"], utc_now()),
            )
        return binding

    def register_snapshot(self, payload: dict[str, Any]) -> str:
        identity = {
            key: payload.get(key)
            for key in (
                "sample_id",
                "ct_uri",
                "ct_sha256",
                "m7_uri",
                "m7_sha256",
                "shape_xyz",
                "voxel_size_um",
                "coordinate_frame",
                "source_content_lock",
            )
        }
        source_id = str(payload.get("source_snapshot_id") or stable_id("source", identity))
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO source_snapshots(source_snapshot_id,sample_id,ct_uri,ct_sha256,m7_uri,m7_sha256,shape_xyz_json,voxel_size_um,coordinate_frame,payload_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_snapshot_id) DO NOTHING""",
                (source_id, payload["sample_id"], payload["ct_uri"], payload.get("ct_sha256"), payload["m7_uri"], payload.get("m7_sha256"), _dump(payload["shape_xyz"]), float(payload["voxel_size_um"]), payload.get("coordinate_frame", "ct_l0_xyz"), _dump({**payload, "source_snapshot_id": source_id}), now),
            )
        return source_id

    def snapshots(self, samples: set[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM source_snapshots"
        args: list[Any] = []
        if samples:
            placeholders = ",".join("?" for _ in samples)
            query += f" WHERE sample_id IN ({placeholders})"
            args.extend(sorted(samples))
        query += " ORDER BY sample_id, source_snapshot_id"
        with self.connect() as connection:
            return [self._snapshot(row) for row in connection.execute(query, args)]

    def register_campaign_budget_admission(
        self, admission: dict[str, Any], *,
        resume_authorization: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create-once the signed CLI admission consumed by task transactions."""
        from .campaign_decision import (  # noqa: PLC0415
            campaign_budget_task_matches_admission,
            derive_campaign_active_policy_chain,
            load_campaign_policy_profile,
            validate_campaign_resume_authorization,
        )

        digest = admission.get("admission_sha256")
        if (admission.get("schema") !=
                "campaignx.first_letters_task_budget_admission.v1"
                or digest != content_sha256({
                    key: value for key, value in admission.items()
                    if key != "admission_sha256"
                })):
            raise ValueError("campaign budget admission hash or schema is invalid")
        encoded = _dump(admission)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            registered = connection.execute(
                """SELECT admission_json FROM campaign_budget_admissions
                    WHERE mission_id=? AND sample_id=? AND receipt_sha256=?""",
                (admission["mission_id"], admission["sample_id"],
                 admission["receipt_sha256"]),
            ).fetchone()
            mission_tasks = connection.execute(
                "SELECT payload_json FROM tasks WHERE mission_id=?",
                (admission["mission_id"],),
            ).fetchall()
            registered_admissions = [
                _load(row["admission_json"])
                for row in connection.execute(
                    """SELECT admission_json
                         FROM campaign_budget_admissions
                        WHERE mission_id=?
                        ORDER BY sample_id,receipt_sha256""",
                    (admission["mission_id"],),
                ).fetchall()
            ]
            registered_decisions = [
                _load(row["receipt_json"])
                for row in connection.execute(
                    """SELECT receipt_json FROM campaign_decisions
                        WHERE mission_id=? ORDER BY created_at,receipt_sha256""",
                    (admission["mission_id"],),
                ).fetchall()
            ]
            registered_authorizations = [
                _load(row["authorization_json"])
                for row in connection.execute(
                    """SELECT authorization_json
                         FROM campaign_resume_authorizations
                        WHERE mission_id=? ORDER BY created_at,authorization_sha256""",
                    (admission["mission_id"],),
                ).fetchall()
            ]
            if registered is not None and (
                _load(registered["admission_json"]) != admission
            ):
                raise ValueError(
                    "campaign budget admission registry already contains "
                    "different authority")
            same_sample = [
                authority for authority in registered_admissions
                if authority.get("sample_id") == admission["sample_id"]
            ]
            if registered is None:
                chain = derive_campaign_active_policy_chain(
                    registered_admissions, registered_decisions,
                    registered_authorizations,
                    mission_id=admission["mission_id"],
                )
                new_policy = (admission.get("execution_bindings") or {}).get(
                    "policy_version")
                if chain is None:
                    if resume_authorization is not None:
                        raise ValueError(
                            "campaign resume authorization has no active policy")
                else:
                    active_policy = chain["active_policy_version"]
                    blocking = chain["active_blocking_decision"]
                    if (blocking is not None
                            and blocking.get("decision") == "CONTROL_INCOMPLETE"):
                        raise ValueError(
                            "active policy evidence is incomplete and cannot be resumed")
                    if blocking is None:
                        if new_policy != active_policy:
                            raise ValueError(
                                "campaign successor policy requires an active pause")
                        if resume_authorization is not None:
                            raise ValueError(
                                "campaign resume authorization has no active pause")
                        if same_sample:
                            raise ValueError(
                                "controlled mission/sample/policy is already "
                                "bound to another task budget receipt")
                    else:
                        if (new_policy == active_policy
                                or not isinstance(resume_authorization, dict)):
                            raise ValueError(
                                "active mission pause requires a new policy and "
                                "exact campaign resume authorization")
                        prior = next((
                            authority for authority in registered_admissions
                            if authority.get("admission_sha256") ==
                                resume_authorization.get("prior_admission_sha256")
                            and (authority.get("execution_bindings") or {}).get(
                                "policy_version") == active_policy
                        ), None)
                        if (prior is None
                                or resume_authorization.get(
                                    "prior_decision_receipt_sha256") !=
                                    blocking.get("receipt_sha256")):
                            raise ValueError(
                                "campaign resume authorization is not bound to "
                                "the active policy pause")
                        attestation = connection.execute(
                            """SELECT authorization_json
                                 FROM campaign_resume_principal_attestations
                                WHERE authorization_sha256=? AND mission_id=?""",
                            (
                                resume_authorization.get("authorization_sha256"),
                                admission["mission_id"],
                            ),
                        ).fetchone()
                        if (attestation is None
                                or _load(attestation["authorization_json"]) !=
                                    resume_authorization):
                            raise ValueError(
                                "campaign resume authorization has no trusted "
                                "principal attestation")
                        authoritative_attempts, authoritative_admissions = (
                            self._campaign_decision_inputs(
                                connection,
                                mission_id=admission["mission_id"],
                                policy_version=active_policy,
                            )
                        )
                        validated_resume = validate_campaign_resume_authorization(
                            resume_authorization,
                            prior_admission=prior,
                            new_admission=admission,
                            prior_decision=blocking,
                            policy=load_campaign_policy_profile(),
                            authoritative_attempts=authoritative_attempts,
                            registered_admissions=authoritative_admissions,
                            trusted_authorization_sha256s={
                                resume_authorization["authorization_sha256"]},
                        )
                        connection.execute(
                            """INSERT OR IGNORE INTO campaign_resume_authorizations
                               (authorization_sha256,mission_id,sample_id,
                                prior_policy_version,new_policy_version,
                                new_admission_sha256,authorization_json,created_at)
                               VALUES(?,?,?,?,?,?,?,?)""",
                            (
                                validated_resume["authorization_sha256"],
                                validated_resume["mission_id"],
                                validated_resume["new_sample_id"],
                                validated_resume["prior_policy_version"],
                                validated_resume["new_policy_version"],
                                admission["admission_sha256"],
                                _dump(validated_resume), utc_now(),
                            ),
                        )
                        persisted_resume = connection.execute(
                            """SELECT authorization_json
                                 FROM campaign_resume_authorizations
                                WHERE mission_id=? AND sample_id=?
                                  AND new_admission_sha256=?""",
                            (
                                admission["mission_id"], admission["sample_id"],
                                admission["admission_sha256"],
                            ),
                        ).fetchone()
                        if (persisted_resume is None
                                or _load(persisted_resume["authorization_json"])
                                != validated_resume):
                            raise ValueError(
                                "campaign resume authorization registry already "
                                "contains different evidence")
            elif resume_authorization is not None:
                persisted_resume = connection.execute(
                    """SELECT authorization_json
                         FROM campaign_resume_authorizations
                        WHERE mission_id=? AND sample_id=?
                          AND new_admission_sha256=?""",
                    (
                        admission["mission_id"], admission["sample_id"],
                        admission["admission_sha256"],
                    ),
                ).fetchone()
                if (registered is None or persisted_resume is None
                        or _load(persisted_resume["authorization_json"])
                        != resume_authorization):
                    raise ValueError(
                        "campaign resume authorization is only valid for an "
                        "exact paused mission/sample replacement or its "
                        "idempotent replay")
            if mission_tasks and (
                not registered_admissions
                or any(not any(campaign_budget_task_matches_admission(
                    _load(row["payload_json"]), authority)
                    for authority in registered_admissions)
                    for row in mission_tasks)
            ):
                raise ValueError(
                    "pre-existing mission tasks prevent controlled admission")
            connection.execute(
                """INSERT OR IGNORE INTO campaign_budget_admissions
                   (mission_id,sample_id,receipt_sha256,admission_json,
                    admission_sha256,created_at) VALUES(?,?,?,?,?,?)""",
                (admission["mission_id"], admission["sample_id"],
                 admission["receipt_sha256"], encoded, digest, utc_now()),
            )
            row = connection.execute(
                """SELECT admission_json FROM campaign_budget_admissions
                    WHERE mission_id=? AND sample_id=? AND receipt_sha256=?""",
                (admission["mission_id"], admission["sample_id"],
                 admission["receipt_sha256"]),
            ).fetchone()
            if row is None or _load(row["admission_json"]) != admission:
                raise ValueError(
                    "campaign budget admission registry already contains different authority")
            connection.commit()
        return admission

    def _campaign_decision_inputs(
        self, connection: sqlite3.Connection, *, mission_id: str,
        policy_version: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Read the rows that govern a decision inside the caller's lock."""
        attempt_rows = connection.execute(
            """SELECT t.task_id,t.mission_id,t.policy_version,t.state AS task_state,
                      t.payload_json,t.updated_at AS task_updated_at,
                      a.attempt_id,a.attempt_number,a.state AS attempt_state,
                      a.result_json,a.updated_at AS attempt_updated_at
                 FROM tasks t
                 LEFT JOIN attempts a ON a.task_id=t.task_id
                WHERE t.mission_id=? AND t.policy_version=?""",
            (mission_id, policy_version),
        ).fetchall()
        attempts = []
        for row in attempt_rows:
            payload = _load(row["payload_json"]) or {}
            attempts.append({
                "task_id": row["task_id"],
                "attempt_id": row["attempt_id"],
                "attempt_number": row["attempt_number"],
                "mission_id": row["mission_id"],
                "sample_id": payload.get("sample_id"),
                "policy_version": row["policy_version"],
                "cell_id": payload.get("cell_id"),
                "campaign_budget": payload.get("campaign_budget"),
                "state": row["attempt_state"] or row["task_state"],
                "result": _load(row["result_json"]),
                "terminal_at_utc": (
                    row["attempt_updated_at"] or row["task_updated_at"]),
            })
        admissions = []
        for row in connection.execute(
            """SELECT admission_json,created_at
                 FROM campaign_budget_admissions
                WHERE mission_id=? ORDER BY created_at,sample_id,receipt_sha256""",
            (mission_id,),
        ).fetchall():
            admission = _load(row["admission_json"])
            admission["registered_at_utc"] = row["created_at"]
            admissions.append(admission)
        return attempts, admissions

    def _refresh_campaign_decisions(
        self, connection: sqlite3.Connection, *, mission_id: str,
        policy_version: str,
    ) -> list[dict[str, Any]]:
        from .campaign_decision import (  # noqa: PLC0415
            derive_campaign_decision_receipts,
            load_campaign_policy_profile,
        )

        attempts, admissions = self._campaign_decision_inputs(
            connection,
            mission_id=mission_id,
            policy_version=policy_version,
        )
        derived = derive_campaign_decision_receipts(
            attempts,
            admissions,
            load_campaign_policy_profile(),
            mission_id=mission_id,
            policy_version=policy_version,
        )
        for receipt in derived:
            connection.execute(
                """INSERT OR IGNORE INTO campaign_decisions
                   (receipt_sha256,mission_id,policy_version,evaluation_kind,
                    evaluation_index,decision,receipt_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (receipt["receipt_sha256"], mission_id, policy_version,
                 receipt["evaluation_kind"], receipt["evaluation_index"],
                 receipt["decision"], _dump(receipt), utc_now()),
            )
            persisted = connection.execute(
                """SELECT receipt_sha256 FROM campaign_decisions
                    WHERE mission_id=? AND policy_version=?
                      AND evaluation_kind=? AND evaluation_index=?""",
                (mission_id, policy_version, receipt["evaluation_kind"],
                 receipt["evaluation_index"]),
            ).fetchone()
            if (persisted is None
                    or persisted["receipt_sha256"] != receipt["receipt_sha256"]):
                raise ValueError(
                    "campaign decision evaluation already has different evidence")
        return [
            _load(row["receipt_json"])
            for row in connection.execute(
                """SELECT receipt_json FROM campaign_decisions
                    WHERE mission_id=? AND policy_version=?
                    ORDER BY created_at,receipt_sha256""",
                (mission_id, policy_version),
            ).fetchall()
        ]

    def campaign_decisions(
        self, *, mission_id: str, policy_version: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """SELECT receipt_json FROM campaign_decisions
                    WHERE mission_id=?"""
        parameters: list[Any] = [mission_id]
        if policy_version is not None:
            query += " AND policy_version=?"
            parameters.append(policy_version)
        query += " ORDER BY created_at,receipt_sha256"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_load(row["receipt_json"]) for row in rows]

    def campaign_active_decision(
        self, *, mission_id: str,
    ) -> dict[str, Any] | None:
        from .campaign_decision import (  # noqa: PLC0415
            derive_campaign_active_decision,
            load_campaign_policy_profile,
        )

        with self.connect() as connection:
            admissions = []
            for row in connection.execute(
                """SELECT admission_json,created_at
                     FROM campaign_budget_admissions
                    WHERE mission_id=? ORDER BY created_at,sample_id,receipt_sha256""",
                (mission_id,),
            ).fetchall():
                admission = _load(row["admission_json"])
                admission["registered_at_utc"] = row["created_at"]
                admissions.append(admission)
            decisions = [
                _load(row["receipt_json"])
                for row in connection.execute(
                    """SELECT receipt_json FROM campaign_decisions
                        WHERE mission_id=? ORDER BY created_at,receipt_sha256""",
                    (mission_id,),
                ).fetchall()
            ]
            authorizations = [
                _load(row["authorization_json"])
                for row in connection.execute(
                    """SELECT authorization_json
                         FROM campaign_resume_authorizations
                        WHERE mission_id=? ORDER BY created_at,authorization_sha256""",
                    (mission_id,),
                ).fetchall()
            ]
            attempts = []
            for row in connection.execute(
                """SELECT t.task_id,t.mission_id,t.policy_version,
                          t.state AS task_state,t.payload_json,
                          t.updated_at AS task_updated_at,
                          a.attempt_id,a.attempt_number,
                          a.state AS attempt_state,a.result_json,
                          a.updated_at AS attempt_updated_at
                     FROM tasks t
                     LEFT JOIN attempts a ON a.task_id=t.task_id
                    WHERE t.mission_id=?""",
                (mission_id,),
            ).fetchall():
                payload = _load(row["payload_json"]) or {}
                attempts.append({
                    "task_id": row["task_id"],
                    "attempt_id": row["attempt_id"],
                    "attempt_number": row["attempt_number"],
                    "mission_id": row["mission_id"],
                    "sample_id": payload.get("sample_id"),
                    "policy_version": row["policy_version"],
                    "campaign_budget": payload.get("campaign_budget"),
                    "state": row["attempt_state"] or row["task_state"],
                    "result": _load(row["result_json"]),
                    "terminal_at_utc": (
                        row["attempt_updated_at"] or row["task_updated_at"]),
                })
        return derive_campaign_active_decision(
            attempts, admissions, decisions, authorizations,
            load_campaign_policy_profile(), mission_id=mission_id,
        )

    @staticmethod
    def _snapshot(row: sqlite3.Row) -> dict[str, Any]:
        value = _load(row["payload_json"])
        value.update({"source_snapshot_id": row["source_snapshot_id"], "shape_xyz": _load(row["shape_xyz_json"])})
        return value

    def _write_routing_receipt(self, connection: sqlite3.Connection,
                               surface: dict[str, Any], now: str) -> dict[str, Any]:
        """Classify a surface as it is created, inside the same transaction.

        Not in a downstream worker: a guard there arrives after the row that
        lets the work start already exists, and PHerc0268 shows what that costs
        -- a two-square-millimetre surface reached the ink screen and its EMPTY
        result files beside an EMPTY over five square centimetres.
        """
        from . import surface_routing

        # An unmeasured surface cannot be routed, and refusing the import here
        # would be the router deciding something outside its question. A direct
        # catalogue import creates no QC job and starts no work; the boundaries
        # that do -- finalize, the QC enqueue, and the promotion inside
        # certification -- go through _require_routing_receipt below and fail
        # closed instead.
        if surface.get("area_cm2") is None:
            return None

        receipt = surface_routing.receipt_for_surface(surface)
        connection.execute(
            """INSERT INTO surface_routing_receipts(surface_id,route,measured_area_cm2,
               minimum_area_cm2,policy_version,profile_id,receipt_sha256,receipt_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (receipt["surface_id"], receipt["route"], receipt["measured_area_cm2"],
             receipt["minimum_area_cm2"], receipt["policy_version"],
             receipt["profile_id"], receipt["receipt_sha256"], _dump(receipt), now),
        )
        return receipt

    @staticmethod
    def _stored_routing_receipt(connection: sqlite3.Connection,
                                surface_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT receipt_json FROM surface_routing_receipts WHERE surface_id=?",
            (surface_id,),
        ).fetchone()
        return json.loads(row["receipt_json"]) if row is not None else None

    def _require_routing_receipt(self, connection: sqlite3.Connection,
                                 surface: dict[str, Any],
                                 now: str) -> dict[str, Any]:
        """The exact, valid receipt this surface must have before work starts.

        Resolved, written, and read back inside the caller's transaction, then
        compared: the persisted document has to equal the one just built, and
        its digest has to verify. A receipt that is merely present proves the
        row was touched; a receipt that reads back byte-identical and verifies
        proves the decision on record is the decision that was made.

        When a receipt already exists it is re-decided rather than trusted --
        the stored route must still be the route this surface's measured area
        produces under the current policy. The area behind a receipt can move
        (the QC backfill path replaces it while a surface is unvalidated) and
        the receipt cannot, so drift is the one way a valid signature could end
        up describing an area the surface no longer has.
        """
        from . import surface_routing

        surface_id = str(surface["surface_id"])
        policy = surface_routing.load_policy()
        stored = self._stored_routing_receipt(connection, surface_id)
        if stored is None:
            built = self._write_routing_receipt(connection, surface, now)
            if built is None:
                raise RuntimeError(
                    f"{surface_id} has no measured area, so it cannot be "
                    "routed and nothing may start work on it")
            persisted = self._stored_routing_receipt(connection, surface_id)
            if persisted != built or not surface_routing.verify_receipt(persisted):
                raise RuntimeError(
                    f"{surface_id} routing receipt did not persist exactly as "
                    "decided")
            return persisted
        if not surface_routing.agrees_with_measurement(
            stored, surface.get("area_cm2"), policy=policy,
        ):
            raise RuntimeError(
                f"{surface_id} routing receipt no longer agrees with its "
                "measured area or the current routing policy")
        return stored

    def first_letters_discovery_reconciliation_states(
        self, mission_id: str,
    ) -> list[str]:
        """A mission's reconciliation states, in the order they were written.

        `created_at` comes from utc_now(), which truncates to whole seconds, so
        two reconciliations in one second tie. rowid breaks that tie by
        insertion, which is the thing being asked about; the digest that used to
        break it has no relationship to when a row was written.
        """
        with self.connect() as connection:
            return [
                str(row["state"]) for row in connection.execute(
                    """SELECT state
                         FROM first_letters_discovery_history_reconciliations_v19
                        WHERE mission_id=?
                        ORDER BY created_at,rowid""",
                    (mission_id,),
                )
            ]

    def routing_receipt(self, surface_id: str) -> dict[str, Any] | None:
        """The stored routing decision, or None if the surface predates routing."""
        with self.connect() as connection:
            return self._stored_routing_receipt(connection, surface_id)

    def backfill_routing_receipts(self, *, apply: bool = False) -> dict[str, Any]:
        """Route the surfaces that existed before the routing did.

        A control plane in service before Task 8 holds surfaces with no receipt,
        and every gate built for Task 8 fails closed on a missing one. Deploying
        without this stops pending QC jobs and refuses every flattening -- it
        does not corrupt anything, it simply declines to work.

        This decides nothing new. It runs the same frozen router over rows that
        already exist and writes the receipt each one earns, which is why it
        cannot be a data fix-up written by hand: a second implementation of the
        decision is a second decision.

        Two refusals, both deliberate:

        * A surface with no measured area is reported, never guessed. Inventing a
          measurement during the repair is the same failure the repair is for.
        * A surface that already has a receipt is skipped. The receipt is
          immutable and records what the surface was when it first existed;
          re-deciding it later would let a changed area rewrite history.

        `apply` defaults to False so the default answer is a census.
        """
        from . import surface_routing

        policy = surface_routing.load_policy()
        summary: dict[str, Any] = {
            "schema": "campaignx.small_surface_routing_backfill.v1",
            "policy_version": policy["policy_version"],
            "profile_id": policy["profile_id"],
            "minimum_area_cm2": float(policy["minimum_area_cm2"]),
            "applied": bool(apply),
            "considered": 0, "routed": 0, "would_route": 0,
            "already_routed": 0, "unroutable": 0,
            "by_route": {}, "unroutable_surface_ids": [],
        }
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """SELECT s.surface_id, s.source_snapshot_id, s.sample_id,
                              s.artifact_sha256, s.area_cm2, s.bbox_xyz_json,
                              s.sample_points_json, s.geometry_qc_state,
                              r.surface_id AS routed
                         FROM surfaces s
                         LEFT JOIN surface_routing_receipts r
                           ON r.surface_id = s.surface_id
                        ORDER BY s.surface_id"""
                ).fetchall()

                for row in rows:
                    summary["considered"] += 1
                    if row["routed"] is not None:
                        summary["already_routed"] += 1
                        continue

                    surface = {
                        "surface_id": row["surface_id"],
                        "source_snapshot_id": row["source_snapshot_id"],
                        "sample_id": row["sample_id"],
                        "artifact_sha256": row["artifact_sha256"],
                        "area_cm2": row["area_cm2"],
                        "bbox_xyz": json.loads(row["bbox_xyz_json"] or "null"),
                        "sample_points": json.loads(
                            row["sample_points_json"] or "null"),
                        "geometry_qc_state": row["geometry_qc_state"],
                    }
                    try:
                        decision, _ = surface_routing.route(
                            surface["area_cm2"], policy=policy)
                    except ValueError:
                        # No usable measurement. Named, not guessed.
                        summary["unroutable"] += 1
                        summary["unroutable_surface_ids"].append(row["surface_id"])
                        continue

                    summary["by_route"][decision] = (
                        summary["by_route"].get(decision, 0) + 1)
                    if apply:
                        self._write_routing_receipt(connection, surface, now)
                        summary["routed"] += 1
                    else:
                        summary["would_route"] += 1

                connection.commit() if apply else connection.rollback()
            except BaseException:
                connection.rollback()
                raise
        return summary

    def _surface_policy_version(self, connection: sqlite3.Connection,
                               surface_id: str) -> str | None:
        """The policy version a surface was produced under, where there is one.

        Grown surfaces carry their task; imported ones may carry the version in
        their payload and often carry nothing, because task identity is what
        versions a grow and an import has no task.
        """
        row = connection.execute(
            "SELECT payload_json FROM surfaces WHERE surface_id=?",
            (surface_id,),
        ).fetchone()
        if row is None:
            return None
        payload = _load(row["payload_json"]) or {}
        version = payload.get("policy_version")
        if isinstance(version, str) and version.strip():
            return version
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return None
        task = connection.execute(
            "SELECT policy_version FROM tasks WHERE task_id=?", (task_id,),
        ).fetchone()
        return task["policy_version"] if task is not None else None

    def _resolve_expansion_authority(
        self, connection: sqlite3.Connection, *, successor_surface_id: str,
        source: dict[str, Any], asserted: Any = None,
    ) -> dict[str, Any] | None:
        """Re-resolve one expansion against the catalogue, under the write lock.

        Every caller is already inside ``BEGIN IMMEDIATE``, which is SQLite's
        write lock and the reason this is a re-resolution rather than a second
        opinion: the predecessor's row and its routing receipt cannot move
        between this read and the successor's write.

        ``asserted`` is what the caller claims the authority is -- stamped onto
        a resume task when it was created, or supplied on an import. It is
        compared, never trusted. A caller may assert; it may not decide.
        """
        from . import surface_expansion  # noqa: PLC0415

        shape = surface_expansion.resume_shape(source)
        if shape is None:
            if asserted is not None:
                raise RuntimeError(
                    "an expansion authority was asserted for a surface that "
                    "continues nothing")
            return None
        predecessor_id = shape["expands_surface_id"]
        predecessor = connection.execute(
            "SELECT surface_id FROM surfaces WHERE surface_id=?",
            (predecessor_id,),
        ).fetchone()
        if predecessor is None:
            raise RuntimeError(
                f"expansion names an unknown surface: {predecessor_id}")
        receipt = self._stored_routing_receipt(connection, predecessor_id)
        from . import surface_routing  # noqa: PLC0415

        if receipt is None or not surface_routing.verify_receipt(receipt):
            # An expansion claims something about the surface it continues. A
            # surface with no valid routing decision has nothing to claim about.
            raise RuntimeError(
                f"expansion names {predecessor_id}, which has no valid routing "
                "decision to continue")
        authority = surface_expansion.build_authority(
            expands_surface_id=predecessor_id,
            successor_surface_id=str(successor_surface_id),
            predecessor_route=receipt["route"],
            predecessor_receipt_sha256=receipt["receipt_sha256"],
            prior_policy_version=self._surface_policy_version(
                connection, predecessor_id),
            new_policy_version=shape["new_policy_version"],
            resume_from=shape["resume_from"],
        )
        if asserted is not None and asserted != authority:
            raise RuntimeError(
                "the asserted expansion authority differs from the one the "
                "catalogue resolves")
        return authority

    def _persist_expansion_authority(self, connection: sqlite3.Connection,
                                     authority: dict[str, Any],
                                     now: str) -> dict[str, Any]:
        """Write it, read it back, and require the two to be the same document."""
        from . import surface_expansion  # noqa: PLC0415

        connection.execute(
            """INSERT INTO surface_expansion_authorities(successor_surface_id,
               expands_surface_id,predecessor_route,predecessor_receipt_sha256,
               prior_policy_version,new_policy_version,authority_sha256,
               authority_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
            (authority["successor_surface_id"], authority["expands_surface_id"],
             authority["predecessor_route"],
             authority["predecessor_receipt_sha256"],
             authority["prior_policy_version"], authority["new_policy_version"],
             authority["authority_sha256"], _dump(authority), now),
        )
        persisted = self._stored_expansion_authority(
            connection, authority["successor_surface_id"])
        if (persisted != authority
                or not surface_expansion.verify_authority(persisted)):
            raise RuntimeError(
                "the expansion authority did not persist exactly as resolved")
        return persisted

    @staticmethod
    def _stored_expansion_authority(connection: sqlite3.Connection,
                                    successor_surface_id: str,
                                    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT authority_json FROM surface_expansion_authorities "
            "WHERE successor_surface_id=?", (successor_surface_id,),
        ).fetchone()
        return json.loads(row["authority_json"]) if row is not None else None

    def expansion_authority(self, successor_surface_id: str,
                            ) -> dict[str, Any] | None:
        """The stored permission this surface was created under, if any."""
        with self.connect() as connection:
            return self._stored_expansion_authority(
                connection, successor_surface_id)

    def resolve_expansion_authority(self, *, successor_surface_id: str,
                                    source: dict[str, Any],
                                    asserted: Any = None,
                                    ) -> dict[str, Any] | None:
        """The public read-only resolution, on the transactional resolver.

        A caller can ask what the catalogue would authorize before committing to
        it. It delegates rather than restating the rules, so there is one place
        where an expansion is decided and no second, weaker one.
        """
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                return self._resolve_expansion_authority(
                    connection, successor_surface_id=successor_surface_id,
                    source=source, asserted=asserted)
            finally:
                connection.rollback()

    def import_surface(self, payload: dict[str, Any]) -> str:
        surface_id = str(payload.get("surface_id") or stable_id("surface-import", payload))
        source_id = str(payload["source_snapshot_id"])
        from .canonical_lineage import (  # noqa: PLC0415
            refuse_asserted_lineage, require_canonical_lineage,
        )

        refuse_asserted_lineage(payload)
        controlled = payload.get("controlled_first_letters") is True
        require_canonical_lineage(
            boundary="DIRECT_SURFACE_IMPORT",
            controlled_mission=controlled,
            authoritative_lineage=payload.get("authoritative_lineage"),
            allow_unvalidated=payload.get("allow_unvalidated"),
        )
        now = utc_now()
        with self.connect() as connection:
            # Explicitly transactional. `connect` opens SQLite with
            # isolation_level=None -- autocommit -- so without this the surface
            # INSERT commits on its own and a router that then raised would
            # leave a committed surface with no routing decision beside it:
            # exactly the state every gate downstream has to treat as unsafe,
            # created by the code that exists to prevent it.
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz_json,sample_points_json,area_cm2,state,physical_qc_state,payload_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(surface_id) DO NOTHING""",
                    (surface_id, source_id, payload["sample_id"], payload.get("owner", "imported"), payload.get("artifact_sha256"), payload.get("artifact_uri"), _dump(payload["bbox_xyz"]), _dump(payload.get("sample_points")) if payload.get("sample_points") is not None else None, payload.get("area_cm2"), payload.get("state", "IMPORTED"), payload.get("physical_qc_state", "UNVALIDATED"), _dump({**payload, "surface_id": surface_id}), now),
                )
                # The insert is ON CONFLICT DO NOTHING, so re-importing the same
                # surface must not attempt a second receipt: the decision is made
                # once, at the moment the surface first exists.
                # Resolved on every arrival, even a replay: a payload asking a
                # surface to expand itself is refused rather than quietly
                # ignored because the surface already exists. Persisted only
                # once, like the routing decision beside it.
                authority = self._resolve_expansion_authority(
                    connection, successor_surface_id=surface_id,
                    source=payload,
                    asserted=payload.get("expansion_authority"))
                # ON CONFLICT DO NOTHING answers a replay of *different* bytes
                # exactly as it answers a replay of identical ones: the id comes
                # back and the caller's area, digest and URI are dropped without
                # a word. A bootstrap replaying the wrong surface under a known
                # id then believes it stored what it sent. Identity is compared
                # against what is actually there, and a disagreement is refused
                # rather than discarded.
                stored = connection.execute(
                    """SELECT source_snapshot_id,sample_id,artifact_sha256,
                              artifact_uri,area_cm2
                         FROM surfaces WHERE surface_id=?""",
                    (surface_id,),
                ).fetchone()
                differing = [
                    name for name, incoming in (
                        ("source_snapshot_id", source_id),
                        ("sample_id", payload["sample_id"]),
                        ("artifact_sha256", payload.get("artifact_sha256")),
                        ("artifact_uri", payload.get("artifact_uri")),
                        ("area_cm2", payload.get("area_cm2")),
                    ) if stored[name] != incoming
                ]
                if differing:
                    raise RuntimeError(
                        f"surface {surface_id} already exists and differs from "
                        f"this import on {', '.join(differing)}; refusing rather "
                        "than discarding the difference"
                    )
                existing = connection.execute(
                    "SELECT 1 FROM surface_routing_receipts WHERE surface_id=?",
                    (surface_id,),
                ).fetchone()
                if existing is None:
                    self._write_routing_receipt(
                        connection, {**payload, "surface_id": surface_id,
                                     "source_snapshot_id": source_id}, now)
                if (authority is not None
                        and self._stored_expansion_authority(
                            connection, surface_id) is None):
                    self._persist_expansion_authority(
                        connection, authority, now)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return surface_id

    def resolve_canonical_surface_lineage(
        self, *, surface_id: str, mission_id: str,
    ) -> dict[str, Any]:
        """Resolve a surface from registered rows for shared boundary guards."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM surfaces WHERE surface_id=?", (surface_id,),
            ).fetchone()
        if row is None:
            return {}
        payload = _load(row["payload_json"]) or {}
        retained = payload.get("authoritative_lineage")
        if isinstance(retained, dict):
            value = copy.deepcopy(retained)
            value["surface_id"] = surface_id
            value["mission_id"] = mission_id
            return value
        external_admission = payload.get("canonical_external_admission")
        external_sha = (
            content_sha256(external_admission)
            if isinstance(external_admission, dict) else None
        )
        namespace = payload.get("namespace") or "CANONICAL_SURFACE"
        return {
            "schema": "campaignx.authoritative_surface_lineage.v1",
            "mission_id": mission_id,
            "surface_id": surface_id,
            "namespace": namespace,
            "artifact_identity": payload.get("artifact_set_id") or
                f"surface:{surface_id}",
            "artifact_sha256": row["artifact_sha256"],
            "artifact_uri": row["artifact_uri"],
            "source_snapshot_id": row["source_snapshot_id"],
            "source_binding_sha256": payload.get("source_binding_sha256") or
                content_sha256({
                    "source_snapshot_id": row["source_snapshot_id"],
                    "sample_id": row["sample_id"],
                }),
            "promotion_lineage_sha256": payload.get(
                "promotion_authority_sha256"),
            "promotion_lineage_kind": payload.get(
                "promotion_lineage_kind"),
            "route_sha256": payload.get("route_sha256"),
            "surface_state": row["state"],
            "canonical": namespace != "NONCANONICAL_DISCOVERY",
            "external": payload.get("owner") == "imported",
            "external_admission_sha256": external_sha,
            "ambiguous": False,
            "hash_conflict": False,
        }

    @staticmethod
    def _require_surface_payload_lineage(
        payload: dict[str, Any], *, boundary: str,
    ) -> None:
        from .canonical_lineage import require_canonical_lineage  # noqa: PLC0415
        controlled = payload.get("controlled_first_letters") is True
        require_canonical_lineage(
            boundary=boundary,
            controlled_mission=controlled,
            authoritative_lineage=payload.get("authoritative_lineage"),
            allow_unvalidated=payload.get("allow_unvalidated"),
        )

    def enqueue_imported_surface_qc(
        self,
        payload: dict[str, Any],
        *,
        profile_id: str,
        job_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reconcile one preserved surface and enqueue exactly one QC job.

        Historical catalogues predate the fleet artifact contract and may use
        a digest over file hashes rather than the artifact-set digest consumed
        by the QC adapter.  This transaction may replace that legacy digest
        only while the surface is still unvalidated and has no job for this
        profile.  Once a job exists, every scientific input is immutable.
        """

        surface_id = str(payload["surface_id"])
        self._require_surface_payload_lineage(
            payload, boundary="PHYSICAL_QC_DIRECT_ENQUEUE"
        )
        source_id = str(payload["source_snapshot_id"])
        sample_id = str(payload["sample_id"])
        if is_fixture_surface(payload):
            raise ValueError("fixture-only surfaces cannot enter scientific QC")
        artifact_sha256 = payload.get("artifact_sha256")
        artifact_uri = payload.get("artifact_uri")
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise ValueError("QC backfill surface requires an artifact SHA-256")
        if not isinstance(artifact_uri, str) or not artifact_uri.strip():
            raise ValueError("QC backfill surface requires an artifact URI")
        qc_id = stable_id(
            "qc-job", {"surface_id": surface_id, "profile_id": profile_id}
        )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                source = connection.execute(
                    "SELECT sample_id FROM source_snapshots WHERE source_snapshot_id=?",
                    (source_id,),
                ).fetchone()
                if source is None or source["sample_id"] != sample_id:
                    raise RuntimeError("QC backfill source snapshot is missing or mismatched")
                existing = connection.execute(
                    "SELECT * FROM surfaces WHERE surface_id=?", (surface_id,)
                ).fetchone()
                job = connection.execute(
                    "SELECT * FROM qc_jobs WHERE surface_id=? AND profile_id=?",
                    (surface_id, profile_id),
                ).fetchone()
                if job is not None:
                    if existing is None:
                        raise RuntimeError("QC job exists without its surface")
                    if (
                        existing["source_snapshot_id"] != source_id
                        or existing["sample_id"] != sample_id
                        or existing["artifact_sha256"] != artifact_sha256
                        or existing["artifact_uri"] != artifact_uri
                    ):
                        raise RuntimeError(
                            "refusing to mutate a surface after its QC job exists"
                        )
                    connection.commit()
                    return {
                        "status": "ALREADY_ENQUEUED",
                        "surface_id": surface_id,
                        "qc_job_id": job["qc_job_id"],
                        "qc_state": job["state"],
                    }
                value = {**payload, "surface_id": surface_id}
                if existing is None:
                    connection.execute(
                        """INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz_json,sample_points_json,area_cm2,state,physical_qc_state,payload_json,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,'QC_PENDING','UNVALIDATED',?,?)""",
                        (
                            surface_id,
                            source_id,
                            sample_id,
                            payload.get("owner", "campaign-x"),
                            artifact_sha256,
                            artifact_uri,
                            _dump(payload["bbox_xyz"]),
                            _dump(payload.get("sample_points"))
                            if payload.get("sample_points") is not None
                            else None,
                            payload.get("area_cm2"),
                            _dump(value),
                            now,
                        ),
                    )
                    reconciliation = "INSERTED"
                else:
                    if (
                        existing["source_snapshot_id"] != source_id
                        or existing["sample_id"] != sample_id
                    ):
                        raise RuntimeError("existing surface identity conflicts with backfill")
                    if existing["physical_qc_state"] != "UNVALIDATED":
                        raise RuntimeError(
                            "refusing to replace artifact metadata after physical QC"
                        )
                    connection.execute(
                        """UPDATE surfaces SET owner=?,artifact_sha256=?,artifact_uri=?,
                           bbox_xyz_json=?,sample_points_json=?,area_cm2=?,state='QC_PENDING',
                           payload_json=? WHERE surface_id=?""",
                        (
                            payload.get("owner", existing["owner"]),
                            artifact_sha256,
                            artifact_uri,
                            _dump(payload["bbox_xyz"]),
                            _dump(payload.get("sample_points"))
                            if payload.get("sample_points") is not None
                            else None,
                            payload.get("area_cm2"),
                            _dump(value),
                            surface_id,
                        ),
                    )
                    reconciliation = "RECONCILED_UNVALIDATED"
                # As in the PostgreSQL store: this is the imported-surface path,
                # imports carry no geometry verdict, and a hardcoded PENDING let
                # every one of them straight through the gate.
                # As there: a verdict the surface already carries outranks the
                # default, or certifying before enqueuing strands the job.
                geometry_state = (
                    payload.get("geometry_qc_state")
                    or (existing["geometry_qc_state"]
                        if existing is not None else None)
                    or DEFAULT_GEOMETRY_QC_STATE
                )
                job_state = qc_job_state_for(geometry_state)
                # And the size gate, which geometry cannot see: PHerc0268 was
                # GEOMETRY_CERTIFIED and two square millimetres. A surface under
                # the floor never becomes claimable whatever geometry says.
                #
                # Required, not consulted. This used to read the route only when
                # a receipt happened to exist, so a surface this method inserted
                # itself -- which wrote no receipt at all -- passed the gate by
                # not being subject to it. And the question is asked positively:
                # exactly a verified STANDARD route admits a claimable job, so a
                # forged receipt fails the same way a missing one does.
                from . import surface_routing  # noqa: PLC0415

                routing_receipt = self._require_routing_receipt(
                    connection, {**value, "surface_id": surface_id,
                                 "geometry_qc_state": geometry_state}, now)
                if not surface_routing.enters_standard_qc(routing_receipt):
                    job_state = QC_SMALL_SURFACE_DIAGNOSTIC
                connection.execute(
                    """INSERT INTO qc_jobs(qc_job_id,surface_id,profile_id,state,payload_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        qc_id,
                        surface_id,
                        profile_id,
                        job_state,
                        _dump(job_payload or {}),
                        now,
                        now,
                    ),
                )
                self.event(
                    connection,
                    "QC_BACKFILL_ENQUEUED",
                    {
                        "qc_job_id": qc_id,
                        "surface_id": surface_id,
                        "profile_id": profile_id,
                        "reconciliation": reconciliation,
                    },
                )
                connection.commit()
                return {
                    "status": "ENQUEUED",
                    "surface_id": surface_id,
                    "qc_job_id": qc_id,
                    "qc_state": job_state,
                    "reconciliation": reconciliation,
                }
            except BaseException:
                connection.rollback()
                raise

    def surfaces_for_snapshot(self, source_snapshot_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM surfaces WHERE source_snapshot_id=? ORDER BY surface_id", (source_snapshot_id,)).fetchall()
        result = []
        for row in rows:
            value = _load(row["payload_json"])
            value.update({
                "surface_id": row["surface_id"],
                "artifact_sha256": row["artifact_sha256"],
                "artifact_uri": row["artifact_uri"],
                "bbox_xyz": _load(row["bbox_xyz_json"]),
                "sample_points": _load(row["sample_points_json"]),
                "state": row["state"],
                "physical_qc_state": row["physical_qc_state"],
                "geometry_qc_state": row["geometry_qc_state"],
            })
            result.append(value)
        return result

    def create_tasks(self, tasks: Iterable[dict[str, Any]]) -> tuple[int, int]:
        tasks = list(tasks)
        inserted = 0
        seen = 0
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                from .campaign_decision import (  # noqa: PLC0415
                    validate_campaign_budget_task_batch,
                )
                budgeted = next((
                    task for task in tasks
                    if isinstance(task.get("campaign_budget"), dict)
                ), None)
                existing_budget_payloads: list[dict[str, Any]] = []
                missions = {str(task.get("mission_id") or "unfiled")
                            for task in tasks}
                controlled_mission_registered = False
                for mission in missions:
                    rows = connection.execute(
                        "SELECT payload_json FROM tasks WHERE mission_id=?",
                        (mission,),
                    ).fetchall()
                    existing_budget_payloads.extend(
                        _load(row["payload_json"]) for row in rows
                        if isinstance(_load(row["payload_json"]).get(
                            "campaign_budget"), dict)
                    )
                    if connection.execute(
                        """SELECT 1 FROM campaign_budget_admissions
                            WHERE mission_id=? LIMIT 1""",
                        (mission,),
                    ).fetchone() is not None:
                        controlled_mission_registered = True
                if budgeted is None and (
                    existing_budget_payloads or controlled_mission_registered
                ):
                    raise ValueError(
                        "campaign budget envelope cannot be omitted after a "
                        "controlled mission has been admitted")
                registered_admission = authoritative_snapshot = None
                if budgeted is not None:
                    envelope = budgeted["campaign_budget"]
                    registry = connection.execute(
                        """SELECT admission_json FROM campaign_budget_admissions
                            WHERE mission_id=? AND sample_id=? AND receipt_sha256=?""",
                        (envelope.get("mission_id"), envelope.get("sample_id"),
                         envelope.get("receipt_sha256")),
                    ).fetchone()
                    if registry is None:
                        raise ValueError(
                            "campaign budget admission is not registered")
                    registered_admission = _load(registry["admission_json"])
                    source_id = (registered_admission.get("execution_bindings") or {}).get(
                        "source_snapshot_id")
                    source = connection.execute(
                        "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                        (source_id,),
                    ).fetchone()
                    if source is None:
                        raise ValueError(
                            "campaign budget authoritative source snapshot is unavailable")
                    authoritative_snapshot = self._snapshot(source)
                validate_campaign_budget_task_batch(
                    tasks, existing_budget_payloads,
                    registered_admission=registered_admission,
                    authoritative_snapshot=authoritative_snapshot)
                if registered_admission is not None:
                    execution = registered_admission["execution_bindings"]
                    decisions = self._refresh_campaign_decisions(
                        connection,
                        mission_id=registered_admission["mission_id"],
                        policy_version=execution["policy_version"],
                    )
                    blocking = next((
                        receipt for receipt in decisions
                        if receipt.get("decision") in {
                            "PAUSE_CANDIDATE_STARVATION",
                            "CONTROL_INCOMPLETE",
                        }
                    ), None)
                    if blocking is not None:
                        # The immutable gate receipt is evidence in its own
                        # right, including when this call is the first reader
                        # to observe a terminal block.  Publish it while still
                        # holding the same write boundary, then reject without
                        # touching any existing task.
                        connection.commit()
                        raise ValueError(
                            "campaign decision blocks new P1 task creation: "
                            f"{blocking['decision']} "
                            f"({blocking['receipt_sha256']})")
                for task in tasks:
                    seen += 1
                    normalized_probe = None
                    normalized_benchmark = None
                    if task.get("seed_probe") is not None:
                        from .seed_probe import (
                            normalize_seed_probe_policy,
                            seed_probe_resume_envelope,
                        )

                        normalized_probe = normalize_seed_probe_policy(
                            task["seed_probe"]
                        )
                        if normalized_probe["mode"] == "select":
                            seed_probe_resume_envelope(
                                task["parameter_envelope"],
                                normalized_probe,
                            )
                    from .seed_probe import (
                        validate_seed_probe_benchmark_execution_task,
                    )

                    normalized_benchmark = (
                        validate_seed_probe_benchmark_execution_task(task)
                    )
                    mission_id = str(task.get("mission_id") or "unfiled")
                    identity = {
                        "mission_id": mission_id,
                        **{
                            key: task[key]
                            for key in (
                                "source_snapshot_id", "grid_version",
                                "cell_id", "policy_version",
                            )
                        },
                    }
                    task_id = str(task.get("task_id") or stable_id("task", identity))
                    requirements = normalize_resource_requirements(task.get("resource_requirements"))
                    if requirements["seed_probe_required"] != (
                        normalized_probe is not None
                    ):
                        raise ValueError(
                            "seed_probe policy and seed_probe_required must agree"
                        )
                    # A resume task carries the authority it was created under,
                    # resolved here against the same locked catalogue
                    # finalization will re-resolve it against. The successor
                    # surface does not exist yet, so the stamp names the task;
                    # finalization rebuilds it for the real surface id and
                    # compares every field that does not depend on it.
                    stamped_expansion = self._resolve_expansion_authority(
                        connection,
                        successor_surface_id=f"task:{task_id}",
                        source=task,
                    )
                    value = {
                        **task,
                        **(
                            {"seed_probe": normalized_probe}
                            if normalized_probe is not None
                            else {}
                        ),
                        **(
                            {"benchmark_execution": normalized_benchmark}
                            if normalized_benchmark is not None
                            else {}
                        ),
                        **(
                            {"expansion_authority": stamped_expansion}
                            if stamped_expansion is not None
                            else {}
                        ),
                        "task_id": task_id,
                        "resource_requirements": requirements,
                    }
                    cursor = connection.execute(
                        """INSERT OR IGNORE INTO tasks(task_id,mission_id,source_snapshot_id,cell_id,grid_version,policy_version,bounds_xyz_json,center_xyz_json,priority,parameter_envelope_json,catalog_snapshot_sha256,payload_json,state,gpu_required,minimum_vram_gb,seed_probe_required,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (task_id, mission_id, task["source_snapshot_id"], task["cell_id"], task["grid_version"], task["policy_version"], _dump(task["bounds_xyz"]), _dump(task["center_xyz"]), float(task["priority"]), _dump(task["parameter_envelope"]), task["catalog_snapshot_sha256"], _dump(value), "PENDING", int(requirements["gpu_required"]), requirements["minimum_vram_gb"], int(requirements["seed_probe_required"]), now, now),
                    )
                    inserted += cursor.rowcount
                    if cursor.rowcount:
                        self.event(connection, "TASK_CREATED", {"priority": task["priority"]}, task_id=task_id)
                    else:
                        existing = connection.execute(
                            """SELECT payload_json FROM tasks
                                WHERE mission_id=? AND source_snapshot_id=? AND grid_version=?
                                  AND cell_id=? AND policy_version=?""",
                            (
                                mission_id,
                                task["source_snapshot_id"],
                                task["grid_version"],
                                task["cell_id"],
                                task["policy_version"],
                            ),
                        ).fetchone()
                        existing_value = (
                            _load(existing["payload_json"])
                            if existing is not None
                            else {}
                        )
                        existing_candidate_authority = {
                            "candidate_rank": int(existing_value.get(
                                "candidate_rank", 1
                            )),
                            "reconsider_covered": bool(existing_value.get(
                                "reconsider_covered", False
                            )),
                        }
                        incoming_candidate_authority = {
                            "candidate_rank": int(value.get(
                                "candidate_rank", 1
                            )),
                            "reconsider_covered": bool(value.get(
                                "reconsider_covered", False
                            )),
                        }
                        if (
                            existing_candidate_authority
                            != incoming_candidate_authority
                        ):
                            raise ValueError(
                                "task candidate authority differs from "
                                "existing task"
                            )
                        if existing_value.get("seed_probe") != value.get(
                            "seed_probe"
                        ):
                            raise ValueError(
                                "task identity already exists with a different "
                                "seed_probe policy; use a new policy_version"
                            )
                        if existing_value.get(
                            "benchmark_execution"
                        ) != value.get("benchmark_execution"):
                            raise ValueError(
                                "task identity already exists with a different "
                                "benchmark_execution contract; use a new "
                                "policy_version"
                            )
                        if existing_value.get(
                            "parameter_envelope"
                        ) != value.get("parameter_envelope"):
                            raise ValueError(
                                "task identity already exists with a different "
                                "parameter_envelope; use a new policy_version"
                            )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return inserted, seen

    def _expire_leases(self, connection: sqlite3.Connection) -> None:
        now = utc_now()
        rows = connection.execute(
            f"SELECT task_id,active_attempt_id FROM tasks WHERE state IN ({','.join('?' for _ in ACTIVE_STATES)}) AND lease_expires_at IS NOT NULL AND lease_expires_at<=?",
            (*ACTIVE_STATES, now),
        ).fetchall()
        for row in rows:
            if row["active_attempt_id"]:
                connection.execute(
                    """UPDATE attempts
                          SET state='LEASE_EXPIRED',result_json=?,updated_at=?
                        WHERE attempt_id=?""",
                    (_dump({
                        "status": "LEASE_EXPIRED",
                        "failure_class": "LEASE_EXHAUSTION",
                        "ink_used": False,
                    }), now, row["active_attempt_id"]),
                )
            connection.execute("UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,active_attempt_id=NULL,updated_at=? WHERE task_id=?", (now, row["task_id"]))
            self.event(connection, "LEASE_EXPIRED", {}, task_id=row["task_id"], attempt_id=row["active_attempt_id"])

    def claim(
        self,
        worker_id: str,
        lease_seconds: int,
        task_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if lease_seconds < 30:
            raise ValueError("lease_seconds must be at least 30")
        worker_capabilities = normalize_worker_capabilities(capabilities)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._expire_leases(connection)
                connection.execute(
                    """INSERT INTO worker_capabilities(worker_id,capabilities_json,updated_at)
                       VALUES(?,?,?) ON CONFLICT(worker_id) DO UPDATE SET
                       capabilities_json=excluded.capabilities_json,updated_at=excluded.updated_at""",
                    (worker_id, _dump(worker_capabilities), utc_now()),
                )
                compatible = (
                    "AND (gpu_required=0 OR ?=1) AND minimum_vram_gb<=? "
                    "AND (seed_probe_required=0 OR ?=1) "
                    "AND ((? IS NULL AND "
                    "json_extract(payload_json,"
                    "'$.benchmark_execution.benchmark_spec_sha256') IS NULL) "
                    "OR json_extract(payload_json,"
                    "'$.benchmark_execution.benchmark_spec_sha256')=?)"
                )
                capability_args = (
                    int(worker_capabilities["cuda_available"]),
                    worker_capabilities["gpu_vram_gb"],
                    int(worker_capabilities["seed_probe_v1"]),
                    worker_capabilities["benchmark_spec_sha256"],
                    worker_capabilities["benchmark_spec_sha256"],
                )
                if task_id is None:
                    row = connection.execute(
                        f"SELECT * FROM tasks WHERE state='PENDING' AND (retry_after IS NULL OR retry_after<=?) {compatible} ORDER BY priority DESC,task_id LIMIT 1",
                        (utc_now(), *capability_args),
                    ).fetchone()
                else:
                    row = connection.execute(
                        f"SELECT * FROM tasks WHERE task_id=? AND state='PENDING' AND (retry_after IS NULL OR retry_after<=?) {compatible}",
                        (task_id, utc_now(), *capability_args),
                    ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                task_id = row["task_id"]
                attempt_number = int(connection.execute("SELECT COALESCE(MAX(attempt_number),0)+1 FROM attempts WHERE task_id=?", (task_id,)).fetchone()[0])
                attempt_id = stable_id("attempt", {"task_id": task_id, "attempt_number": attempt_number})
                token = secrets.token_urlsafe(32)
                now = utc_now()
                connection.execute(
                    "INSERT INTO attempts(attempt_id,task_id,attempt_number,worker_id,lease_token,state,created_at,updated_at) VALUES(?,?,?,?,?,'CLAIMED',?,?)",
                    (attempt_id, task_id, attempt_number, worker_id, token, now, now),
                )
                connection.execute(
                    "UPDATE tasks SET state='CLAIMED',worker_id=?,lease_token=?,lease_expires_at=?,retry_after=NULL,active_attempt_id=?,updated_at=? WHERE task_id=?",
                    (worker_id, token, _deadline(lease_seconds), attempt_id, now, task_id),
                )
                self.event(connection, "TASK_CLAIMED", {"worker_id": worker_id, "attempt_number": attempt_number, "capabilities": worker_capabilities}, task_id, attempt_id)
                connection.commit()
                return self.task_packet(task_id, attempt_id=attempt_id, lease_token=token)
            except BaseException:
                connection.rollback()
                raise

    def pending_tasks(self, limit: int) -> list[dict[str, Any]]:
        """Read pending task packets without leasing or mutating them."""
        if limit < 1:
            raise ValueError("survey limit must be at least one")
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT task_id FROM tasks WHERE state='PENDING' AND (retry_after IS NULL OR retry_after<=?) ORDER BY priority DESC,task_id LIMIT ?",
                (utc_now(), limit),
            ).fetchall()
        return [self.task_packet(str(row["task_id"])) for row in rows]

    def task_packet(self, task_id: str, attempt_id: str | None = None, lease_token: str | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            source = connection.execute("SELECT * FROM source_snapshots WHERE source_snapshot_id=?", (row["source_snapshot_id"],)).fetchone()
            if source is None:
                raise RuntimeError("task source snapshot disappeared")
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id or row["active_attempt_id"],)).fetchone() if (attempt_id or row["active_attempt_id"]) else None
        payload = _load(row["payload_json"])
        payload.update({
            "schema": "campaignx.segmentation_task.v1",
            "task_id": row["task_id"],
            "state": row["state"],
            "bounds_xyz": _load(row["bounds_xyz_json"]),
            "center_xyz": _load(row["center_xyz_json"]),
            "parameter_envelope": _load(row["parameter_envelope_json"]),
            "resource_requirements": normalize_resource_requirements({
                "gpu_required": bool(row["gpu_required"]),
                "minimum_vram_gb": float(row["minimum_vram_gb"]),
                "seed_probe_required": bool(row["seed_probe_required"]),
            }),
            "source": self._snapshot(source),
        })
        if attempt is not None:
            payload["attempt_id"] = attempt["attempt_id"]
            payload["attempt_number"] = attempt["attempt_number"]
            payload["worker_id"] = attempt["worker_id"]
            payload["lease_token"] = lease_token or attempt["lease_token"]
        return payload

    def regional_attempt_history(
        self, task: dict[str, Any], *, limit: int = 12
    ) -> dict[str, Any]:
        """Return bounded, geometry-only failures overlapping this task cell."""

        attempt_states = sorted(
            state
            for state in ACTIONABLE_SEGMENTATION_HISTORY_STATES
            if not state.startswith("GEOMETRY_REJECTED_")
        )
        geometry_states = sorted(
            state
            for state in ACTIONABLE_SEGMENTATION_HISTORY_STATES
            if state.startswith("GEOMETRY_REJECTED_")
        )
        placeholders = ",".join("?" for _ in attempt_states)
        geometry_placeholders = ",".join("?" for _ in geometry_states)
        low, high = task["bounds_xyz"]
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT a.attempt_id,a.task_id,a.state,a.locked_plan_json,
                           a.result_json,a.updated_at,
                           t.cell_id,t.policy_version,t.bounds_xyz_json
                    FROM attempts a JOIN tasks t ON t.task_id=a.task_id
                    WHERE t.source_snapshot_id=? AND a.attempt_id<>?
                      AND (a.state IN ({placeholders})
                           OR (a.state='QC_PENDING'
                               AND json_extract(a.result_json,'$.geometry_qc_state')
                                   IN ({geometry_placeholders})))
                      AND CAST(json_extract(t.bounds_xyz_json,'$[0][0]') AS REAL)<=?
                      AND CAST(json_extract(t.bounds_xyz_json,'$[1][0]') AS REAL)>=?
                      AND CAST(json_extract(t.bounds_xyz_json,'$[0][1]') AS REAL)<=?
                      AND CAST(json_extract(t.bounds_xyz_json,'$[1][1]') AS REAL)>=?
                      AND CAST(json_extract(t.bounds_xyz_json,'$[0][2]') AS REAL)<=?
                      AND CAST(json_extract(t.bounds_xyz_json,'$[1][2]') AS REAL)>=?
                    ORDER BY a.updated_at DESC,a.attempt_id
                    LIMIT ?""",
                (
                    task["source"]["source_snapshot_id"],
                    task.get("attempt_id", ""),
                    *attempt_states,
                    *geometry_states,
                    float(high[0]),
                    float(low[0]),
                    float(high[1]),
                    float(low[1]),
                    float(high[2]),
                    float(low[2]),
                    limit,
                ),
            ).fetchall()
        normalized = [
            {
                "attempt_id": row["attempt_id"],
                "task_id": row["task_id"],
                "state": row["state"],
                "locked_plan": _load(row["locked_plan_json"])
                if row["locked_plan_json"]
                else None,
                "result": (
                    _load(row["result_json"]) if row["result_json"] else None
                ),
                "updated_at": row["updated_at"],
                "cell_id": row["cell_id"],
                "policy_version": row["policy_version"],
                "bounds_xyz": _load(row["bounds_xyz_json"]),
            }
            for row in rows
        ]
        return build_regional_attempt_history(task, normalized, limit=limit)

    @staticmethod
    def _assert_probe_parent_owner(
        connection: sqlite3.Connection,
        task_id: str,
        attempt_id: str,
        lease_token: str,
    ) -> sqlite3.Row:
        owner = connection.execute(
            """SELECT active_attempt_id,lease_token,source_snapshot_id,payload_json
                 FROM tasks WHERE task_id=?""",
            (task_id,),
        ).fetchone()
        if (
            owner is None
            or owner["active_attempt_id"] != attempt_id
            or owner["lease_token"] != lease_token
        ):
            raise RuntimeError("probe operation belongs to a stale parent lease")
        return owner

    @staticmethod
    def _assert_probe_trial_owner(
        connection: sqlite3.Connection,
        task_id: str,
        probe_trial_id: str,
        probe_attempt_id: str,
        lease_token: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """SELECT t.*
                 FROM probe_trials t
                 JOIN probe_runs r ON r.probe_run_id=t.probe_run_id
                 JOIN probe_attempts a
                   ON a.probe_attempt_id=t.active_probe_attempt_id
                  AND a.probe_trial_id=t.probe_trial_id
                WHERE t.probe_trial_id=? AND r.task_id=?
                  AND a.probe_attempt_id=?""",
            (probe_trial_id, task_id, probe_attempt_id),
        ).fetchone()
        if (
            row is None
            or row["active_probe_attempt_id"] != probe_attempt_id
            or row["lease_token"] != lease_token
        ):
            raise RuntimeError("probe operation belongs to a stale trial lease")
        return row

    @staticmethod
    def _assert_probe_run_task(
        connection: sqlite3.Connection,
        task_id: str,
        probe_run_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """SELECT * FROM probe_runs
                WHERE probe_run_id=? AND task_id=?""",
            (probe_run_id, task_id),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "probe run does not belong to the leased parent task"
            )
        return row

    @staticmethod
    def _expire_probe_leases(
        connection: sqlite3.Connection, probe_run_id: str
    ) -> None:
        now = utc_now()
        rows = connection.execute(
            """SELECT probe_trial_id,active_probe_attempt_id
                 FROM probe_trials
                WHERE probe_run_id=?
                  AND state IN ('CLAIMED','RUNNING','UPLOADED','EVALUATING')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?""",
            (probe_run_id, now),
        ).fetchall()
        for row in rows:
            if row["active_probe_attempt_id"]:
                connection.execute(
                    """UPDATE probe_attempts
                          SET state='LEASE_EXPIRED',updated_at=?
                        WHERE probe_attempt_id=?""",
                    (now, row["active_probe_attempt_id"]),
                )
            connection.execute(
                """UPDATE probe_trials
                      SET state='PENDING',worker_id=NULL,lease_token=NULL,
                          lease_expires_at=NULL,active_probe_attempt_id=NULL,
                          updated_at=?
                    WHERE probe_trial_id=?""",
                (now, row["probe_trial_id"]),
            )

    def prepare_probe_run(
        self,
        task: dict[str, Any],
        candidates: list[dict[str, Any]],
        policy: dict[str, Any],
        executor_fingerprint: dict[str, Any],
    ) -> dict[str, Any]:
        """Create or recover the logical run shared by replacement attempts."""

        from .seed_probe import normalize_seed_probe_policy

        normalized = normalize_seed_probe_policy(policy)
        if not candidates:
            raise ValueError("probe candidate set cannot be empty")
        ranks = [int(row["candidate_rank"]) for row in candidates]
        if ranks != list(range(1, len(candidates) + 1)):
            raise ValueError("probe candidate ranks must be contiguous from one")
        if len(candidates) > int(normalized["top_k"]):
            raise ValueError("probe candidate set exceeds frozen top_k")
        candidate_set_sha = content_sha256(candidates)
        policy_sha = content_sha256(normalized)
        executor_sha = content_sha256(executor_fingerprint)
        identity = {
            "task_id": task["task_id"],
            "candidate_set_sha256": candidate_set_sha,
            "policy_sha256": policy_sha,
            "executor_fingerprint_sha256": executor_sha,
        }
        probe_run_id = stable_id("probe-run", identity)
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = self._assert_probe_parent_owner(
                    connection,
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                )
                caller_source_id = str(
                    (task.get("source") or {}).get("source_snapshot_id") or ""
                )
                authoritative_source_id = str(owner["source_snapshot_id"])
                if caller_source_id != authoritative_source_id:
                    raise ValueError(
                        "probe task source does not match the authoritative "
                        "parent source snapshot"
                    )
                authoritative_task = _load(owner["payload_json"])
                authoritative_policy = normalize_seed_probe_policy(
                    authoritative_task.get("seed_probe")
                )
                caller_task_policy = normalize_seed_probe_policy(
                    task.get("seed_probe")
                )
                if (
                    normalized != authoritative_policy
                    or caller_task_policy != authoritative_policy
                ):
                    raise ValueError(
                        "probe policy does not match the authoritative parent task"
                    )
                existing = connection.execute(
                    "SELECT * FROM probe_runs WHERE task_id=?",
                    (task["task_id"],),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO probe_runs
                           (probe_run_id,task_id,created_by_attempt_id,
                            source_snapshot_id,candidate_set_json,
                            candidate_set_sha256,policy_id,policy_json,
                            policy_sha256,executor_fingerprint_json,
                            executor_fingerprint_sha256,state,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,'PENDING',?,?)""",
                        (
                            probe_run_id,
                            task["task_id"],
                            task["attempt_id"],
                            authoritative_source_id,
                            _dump(candidates),
                            candidate_set_sha,
                            normalized["policy_id"],
                            _dump(normalized),
                            policy_sha,
                            _dump(executor_fingerprint),
                            executor_sha,
                            now,
                            now,
                        ),
                    )
                    requirements = normalize_resource_requirements(
                        {
                            "gpu_required": bool(
                                normalized["probe_parameters"]["use_cuda"]
                            ),
                            "minimum_vram_gb": (
                                float(
                                    task.get("resource_requirements", {}).get(
                                        "minimum_vram_gb", 0.0
                                    )
                                )
                                if normalized["probe_parameters"]["use_cuda"]
                                else 0.0
                            ),
                        }
                    )
                    for candidate in candidates:
                        probe_trial_id = stable_id(
                            "probe-trial",
                            {
                                "probe_run_id": probe_run_id,
                                "candidate_id": candidate["candidate_id"],
                                "candidate_rank": candidate["candidate_rank"],
                            },
                        )
                        connection.execute(
                            """INSERT INTO probe_trials
                               (probe_trial_id,probe_run_id,candidate_id,
                                candidate_rank,candidate_json,state,gpu_required,
                                minimum_vram_gb,created_at,updated_at)
                               VALUES(?,?,?,?,?,'PENDING',?,?,?,?)""",
                            (
                                probe_trial_id,
                                probe_run_id,
                                candidate["candidate_id"],
                                int(candidate["candidate_rank"]),
                                _dump(candidate),
                                int(requirements["gpu_required"]),
                                requirements["minimum_vram_gb"],
                                now,
                                now,
                            ),
                        )
                    self.event(
                        connection,
                        "PROBE_RUN_CREATED",
                        {
                            "probe_run_id": probe_run_id,
                            "candidate_set_sha256": candidate_set_sha,
                            "policy_sha256": policy_sha,
                            "executor_fingerprint_sha256": executor_sha,
                        },
                        task["task_id"],
                        task["attempt_id"],
                    )
                else:
                    exact = (
                        existing["probe_run_id"] == probe_run_id
                        and existing["task_id"] == task["task_id"]
                        and existing["source_snapshot_id"]
                        == authoritative_source_id
                        and existing["candidate_set_sha256"] == candidate_set_sha
                        and existing["policy_sha256"] == policy_sha
                        and existing["executor_fingerprint_sha256"] == executor_sha
                    )
                    if not exact:
                        raise RuntimeError(
                            "parent task already has a different immutable probe run"
                        )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return self.probe_run(probe_run_id)

    def probe_run(self, probe_run_id: str) -> dict[str, Any]:
        """Return a complete, read-only run snapshot for recovery or decision."""

        with self.connect() as connection:
            run = connection.execute(
                "SELECT * FROM probe_runs WHERE probe_run_id=?",
                (probe_run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(probe_run_id)
            rows = connection.execute(
                """SELECT t.*,a.probe_artifact_set_id,a.manifest_json,
                          a.manifest_sha256,a.artifact_uri,a.state AS artifact_state,
                          e.evaluation_id,e.result_json AS evaluation_json,
                          e.result_sha256 AS evaluation_sha256,e.verdict
                     FROM probe_trials t
                     LEFT JOIN probe_artifact_sets a
                       ON a.probe_trial_id=t.probe_trial_id
                     LEFT JOIN probe_evaluations e
                       ON e.probe_trial_id=t.probe_trial_id
                    WHERE t.probe_run_id=?
                    ORDER BY t.candidate_rank""",
                (probe_run_id,),
            ).fetchall()
            decision_row = connection.execute(
                "SELECT * FROM probe_decisions WHERE probe_run_id=?",
                (probe_run_id,),
            ).fetchone()
        trials = []
        for row in rows:
            trials.append(
                {
                    "probe_run_id": probe_run_id,
                    "probe_trial_id": row["probe_trial_id"],
                    "candidate_id": row["candidate_id"],
                    "candidate_rank": row["candidate_rank"],
                    "candidate": _load(row["candidate_json"]),
                    "state": row["state"],
                    "locked_plan": _load(row["locked_plan_json"]),
                    "result": _load(row["result_json"]),
                    "probe_artifact_set_id": row["probe_artifact_set_id"],
                    "manifest": (
                        _load(row["manifest_json"])
                        if row["manifest_json"] is not None
                        else None
                    ),
                    "manifest_sha256": row["manifest_sha256"],
                    "artifact_uri": row["artifact_uri"],
                    "artifact_state": row["artifact_state"],
                    "evaluation": (
                        _load(row["evaluation_json"])
                        if row["evaluation_json"] is not None
                        else None
                    ),
                    "evaluation_sha256": row["evaluation_sha256"],
                    "updated_at": row["updated_at"],
                }
            )
        return {
            "schema": "campaignx.seed_probe_run.v1",
            "probe_run_id": run["probe_run_id"],
            "task_id": run["task_id"],
            "source_snapshot_id": run["source_snapshot_id"],
            "state": run["state"],
            "candidate_set": _load(run["candidate_set_json"]),
            "candidate_set_sha256": run["candidate_set_sha256"],
            "policy": _load(run["policy_json"]),
            "policy_sha256": run["policy_sha256"],
            "executor_fingerprint": _load(run["executor_fingerprint_json"]),
            "executor_fingerprint_sha256": run[
                "executor_fingerprint_sha256"
            ],
            "trials": trials,
            "decision": (
                _load(decision_row["receipt_json"])
                if decision_row is not None
                else None
            ),
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
        }

    def claim_probe_trial(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_run_id: str,
        worker_id: str,
        lease_seconds: int,
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if lease_seconds < 30:
            raise ValueError("probe lease_seconds must be at least 30")
        worker_capabilities = normalize_worker_capabilities(capabilities)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection, task_id, parent_attempt_id, parent_lease_token
                )
                run = self._assert_probe_run_task(
                    connection, task_id, probe_run_id
                )
                self._expire_probe_leases(connection, probe_run_id)
                maximum_attempts = int(
                    _load(run["policy_json"])["maximum_attempts_per_candidate"]
                )
                pending = connection.execute(
                    """SELECT t.*,
                              (SELECT COUNT(*) FROM probe_attempts a
                                WHERE a.probe_trial_id=t.probe_trial_id) AS attempts
                         FROM probe_trials t
                        WHERE t.probe_run_id=? AND t.state='PENDING'
                          AND (t.retry_after IS NULL OR t.retry_after<=?)
                          AND (t.gpu_required=0 OR ?=1)
                          AND t.minimum_vram_gb<=?
                        ORDER BY t.candidate_rank
                        LIMIT 1""",
                    (
                        probe_run_id,
                        utc_now(),
                        int(worker_capabilities["cuda_available"]),
                        worker_capabilities["gpu_vram_gb"],
                    ),
                ).fetchone()
                while pending is not None and int(pending["attempts"]) >= maximum_attempts:
                    now = utc_now()
                    connection.execute(
                        """UPDATE probe_trials
                              SET state='FAILED',result_json=?,updated_at=?
                            WHERE probe_trial_id=?""",
                        (
                            _dump(
                                {
                                    "status": "PROBE_ATTEMPTS_EXHAUSTED",
                                    "maximum_attempts": maximum_attempts,
                                    "ink_used": False,
                                }
                            ),
                            now,
                            pending["probe_trial_id"],
                        ),
                    )
                    pending = connection.execute(
                        """SELECT t.*,
                                  (SELECT COUNT(*) FROM probe_attempts a
                                    WHERE a.probe_trial_id=t.probe_trial_id) AS attempts
                             FROM probe_trials t
                            WHERE t.probe_run_id=? AND t.state='PENDING'
                              AND (t.retry_after IS NULL OR t.retry_after<=?)
                              AND (t.gpu_required=0 OR ?=1)
                              AND t.minimum_vram_gb<=?
                            ORDER BY t.candidate_rank LIMIT 1""",
                        (
                            probe_run_id,
                            utc_now(),
                            int(worker_capabilities["cuda_available"]),
                            worker_capabilities["gpu_vram_gb"],
                        ),
                    ).fetchone()
                if pending is None:
                    connection.commit()
                    return None
                attempt_number = int(pending["attempts"]) + 1
                probe_attempt_id = stable_id(
                    "probe-attempt",
                    {
                        "probe_trial_id": pending["probe_trial_id"],
                        "attempt_number": attempt_number,
                    },
                )
                token = secrets.token_urlsafe(32)
                now = utc_now()
                connection.execute(
                    """INSERT INTO probe_attempts
                       (probe_attempt_id,probe_trial_id,attempt_number,worker_id,
                        state,created_at,updated_at)
                       VALUES(?,?,?,?, 'CLAIMED',?,?)""",
                    (
                        probe_attempt_id,
                        pending["probe_trial_id"],
                        attempt_number,
                        worker_id,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """UPDATE probe_trials
                          SET state='CLAIMED',worker_id=?,lease_token=?,
                              lease_expires_at=?,retry_after=NULL,
                              active_probe_attempt_id=?,updated_at=?
                        WHERE probe_trial_id=?""",
                    (
                        worker_id,
                        token,
                        _deadline(lease_seconds),
                        probe_attempt_id,
                        now,
                        pending["probe_trial_id"],
                    ),
                )
                connection.execute(
                    "UPDATE probe_runs SET state='PROBING',updated_at=? WHERE probe_run_id=?",
                    (now, probe_run_id),
                )
                self.event(
                    connection,
                    "PROBE_TRIAL_CLAIMED",
                    {
                        "probe_run_id": probe_run_id,
                        "probe_trial_id": pending["probe_trial_id"],
                        "probe_attempt_id": probe_attempt_id,
                        "attempt_number": attempt_number,
                    },
                    task_id,
                    parent_attempt_id,
                )
                connection.commit()
                return {
                    "probe_run_id": probe_run_id,
                    "probe_trial_id": pending["probe_trial_id"],
                    "probe_attempt_id": probe_attempt_id,
                    "attempt_number": attempt_number,
                    "candidate_rank": pending["candidate_rank"],
                    "candidate": _load(pending["candidate_json"]),
                    "lease_token": token,
                }
            except BaseException:
                connection.rollback()
                raise

    def transition_probe_trial(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_trial_id: str,
        probe_attempt_id: str,
        probe_lease_token: str,
        state: str,
        *,
        locked_plan: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if state != "RUNNING":
            raise ValueError(
                "the only direct probe transition is CLAIMED to RUNNING"
            )
        if locked_plan is None:
            raise ValueError("starting a probe trial requires its locked plan")
        if result is not None:
            raise ValueError("a RUNNING probe transition cannot carry a result")
        parent_task = self.task_packet(
            task_id,
            attempt_id=parent_attempt_id,
            lease_token=parent_lease_token,
        )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    connection,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                run = self._assert_probe_run_task(
                    connection, task_id, trial["probe_run_id"]
                )
                from .seed_probe import validate_probe_locked_plan

                authoritative_plan = validate_probe_locked_plan(
                    task=parent_task,
                    trial={
                        "probe_run_id": trial["probe_run_id"],
                        "probe_trial_id": probe_trial_id,
                        "candidate": _load(trial["candidate_json"]),
                    },
                    policy=_load(run["policy_json"]),
                    locked_plan=locked_plan,
                )
                locked_plan_sha = content_sha256(authoritative_plan)
                if (
                    trial["state"] not in {"CLAIMED", "RUNNING"}
                    or (
                        trial["locked_plan_sha256"] is not None
                        and trial["locked_plan_sha256"] != locked_plan_sha
                    )
                ):
                    raise RuntimeError(
                        "probe trial cannot enter RUNNING from its current state"
                    )
                attempt = connection.execute(
                    "SELECT state FROM probe_attempts WHERE probe_attempt_id=?",
                    (probe_attempt_id,),
                ).fetchone()
                if attempt is None or attempt["state"] not in {"CLAIMED", "RUNNING"}:
                    raise RuntimeError(
                        "probe attempt cannot enter RUNNING from its current state"
                    )
                connection.execute(
                    """UPDATE probe_trials
                          SET state='RUNNING',locked_plan_json=?,
                              locked_plan_sha256=?,updated_at=?
                        WHERE probe_trial_id=?""",
                    (
                        _dump(authoritative_plan),
                        locked_plan_sha,
                        now,
                        probe_trial_id,
                    ),
                )
                connection.execute(
                    """UPDATE probe_attempts SET state='RUNNING',updated_at=?
                        WHERE probe_attempt_id=?""",
                    (now, probe_attempt_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def reserve_probe_artifact(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_trial_id: str,
        probe_attempt_id: str,
        probe_lease_token: str,
        manifest: dict[str, Any],
    ) -> str:
        expected_manifest_keys = {
            "schema",
            "probe_run_id",
            "probe_trial_id",
            "locked_plan_sha256",
            "files",
            "artifact_sha256",
            "noncanonical",
            "ink_used",
        }
        files = manifest.get("files")
        required_files = {
            "x.tif",
            "y.tif",
            "z.tif",
            "generations.tif",
            "meta.json",
        }
        valid_files = (
            isinstance(files, dict)
            and set(files) == required_files
            and all(
                isinstance(entry, dict)
                and set(entry) == {"sha256", "size_bytes"}
                and isinstance(entry["size_bytes"], int)
                and not isinstance(entry["size_bytes"], bool)
                and entry["size_bytes"] > 0
                and isinstance(entry["sha256"], str)
                and len(entry["sha256"]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in entry["sha256"]
                )
                for entry in files.values()
            )
        )
        if (
            manifest.get("schema")
            != "campaignx.seed_probe_artifact_set.v1"
            or manifest.get("noncanonical") is not True
            or manifest.get("ink_used") is not False
            or set(manifest) != expected_manifest_keys
            or manifest.get("probe_trial_id") != probe_trial_id
            or not valid_files
            or content_sha256(files) != manifest.get(
                "artifact_sha256"
            )
        ):
            raise ValueError("probe artifact manifest is not noncanonical v1")
        manifest_sha = content_sha256(manifest)
        probe_artifact_set_id = stable_id(
            "probe-artifact-set",
            {
                "probe_trial_id": probe_trial_id,
                "manifest_sha256": manifest_sha,
            },
        )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    connection,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                existing = connection.execute(
                    "SELECT * FROM probe_artifact_sets WHERE probe_trial_id=?",
                    (probe_trial_id,),
                ).fetchone()
                if (
                    manifest.get("probe_run_id") != trial["probe_run_id"]
                ):
                    raise ValueError(
                        "probe artifact manifest belongs to a different run"
                    )
                if (
                    trial["locked_plan_sha256"] is None
                    or manifest.get("locked_plan_sha256")
                    != trial["locked_plan_sha256"]
                ):
                    raise ValueError(
                        "probe artifact manifest is not bound to the locked plan"
                    )
                run = self._assert_probe_run_task(
                    connection, task_id, trial["probe_run_id"]
                )
                from .seed_probe import expected_probe_artifact_uri

                locked_plan = _load(trial["locked_plan_json"]) or {}
                expected_artifact_uri = expected_probe_artifact_uri(
                    namespace_identity=(
                        _load(run["executor_fingerprint_json"]) or {}
                    ).get("probe_namespace", {}),
                    sample_id=str(locked_plan.get("sample_id") or ""),
                    probe_run_id=trial["probe_run_id"],
                    probe_trial_id=probe_trial_id,
                    artifact_sha256=str(manifest["artifact_sha256"]),
                )
                if existing is not None:
                    if existing["manifest_sha256"] != manifest_sha:
                        raise RuntimeError(
                            "the same logical probe trial produced different bytes"
                        )
                    if trial["state"] not in {"RUNNING", "UPLOADED"}:
                        raise RuntimeError(
                            "probe artifact replay is illegal from the current state"
                        )
                    if existing["artifact_uri"] not in {
                        None,
                        expected_artifact_uri,
                    }:
                        raise RuntimeError(
                            "reserved probe artifact has a different publication URI"
                        )
                    connection.execute(
                        """UPDATE probe_artifact_sets SET artifact_uri=?
                            WHERE probe_artifact_set_id=?""",
                        (
                            expected_artifact_uri,
                            existing["probe_artifact_set_id"],
                        ),
                    )
                    connection.execute(
                        "UPDATE probe_trials SET state='UPLOADED',updated_at=? "
                        "WHERE probe_trial_id=?",
                        (now, probe_trial_id),
                    )
                    connection.execute(
                        "UPDATE probe_attempts SET state='UPLOADED',updated_at=? "
                        "WHERE probe_attempt_id=?",
                        (now, probe_attempt_id),
                    )
                    connection.commit()
                    return str(existing["probe_artifact_set_id"])
                if trial["state"] != "RUNNING":
                    raise RuntimeError(
                        "a probe artifact may only be reserved from RUNNING"
                    )
                connection.execute(
                    """INSERT INTO probe_artifact_sets
                       (probe_artifact_set_id,probe_trial_id,probe_attempt_id,
                        manifest_json,manifest_sha256,artifact_uri,state,created_at)
                       VALUES(?,?,?,?,?,?,'RESERVED',?)""",
                    (
                        probe_artifact_set_id,
                        probe_trial_id,
                        probe_attempt_id,
                        _dump(manifest),
                        manifest_sha,
                        expected_artifact_uri,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE probe_trials SET state='UPLOADED',updated_at=? WHERE probe_trial_id=?",
                    (now, probe_trial_id),
                )
                connection.execute(
                    "UPDATE probe_attempts SET state='UPLOADED',updated_at=? WHERE probe_attempt_id=?",
                    (now, probe_attempt_id),
                )
                connection.commit()
                return probe_artifact_set_id
            except BaseException:
                connection.rollback()
                raise

    def complete_probe_trial(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_trial_id: str,
        probe_attempt_id: str,
        probe_lease_token: str,
        probe_artifact_set_id: str,
        artifact_uri: str,
        growth_receipt: dict[str, Any],
        evaluation: dict[str, Any],
        geometry: dict[str, Any],
    ) -> dict[str, Any]:
        verdict = str(evaluation.get("verdict"))
        state_for = {
            "ELIGIBLE": "SUCCEEDED",
            "REJECTED": "REJECTED",
            "UNMEASURED": "UNMEASURED",
        }
        if verdict not in state_for:
            raise ValueError(f"unsupported probe evaluation verdict: {verdict}")
        if evaluation.get("ink_used") is not False:
            raise ValueError("probe evaluation must be ink-blind")
        profile_sha256 = str(evaluation.get("profile_sha256") or "")
        if len(profile_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in profile_sha256
        ):
            raise ValueError(
                "probe evaluation profile_sha256 must be lowercase SHA-256"
            )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    connection,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                if trial["state"] != "UPLOADED":
                    raise RuntimeError(
                        "probe completion requires an uploaded reserved artifact"
                    )
                artifact = connection.execute(
                    """SELECT * FROM probe_artifact_sets
                        WHERE probe_artifact_set_id=? AND probe_trial_id=?""",
                    (
                        probe_artifact_set_id,
                        probe_trial_id,
                    ),
                ).fetchone()
                if artifact is None:
                    raise RuntimeError("probe artifact was not reserved by this trial")
                if artifact["state"] != "RESERVED":
                    raise RuntimeError(
                        "probe completion requires a RESERVED artifact"
                    )
                if artifact["artifact_uri"] != artifact_uri:
                    raise ValueError(
                        "published probe URI differs from the reserved namespace URI"
                    )
                run = connection.execute(
                    """SELECT policy_json,executor_fingerprint_json
                         FROM probe_runs WHERE probe_run_id=?""",
                    (trial["probe_run_id"],),
                ).fetchone()
                from .seed_probe import validate_probe_completion_evidence

                verdict = validate_probe_completion_evidence(
                    evaluation=evaluation,
                    geometry=geometry,
                    growth_receipt=growth_receipt,
                    artifact_uri=artifact_uri,
                    artifact_manifest=_load(artifact["manifest_json"]),
                    task_id=task_id,
                    probe_run_id=trial["probe_run_id"],
                    probe_trial_id=probe_trial_id,
                    probe_artifact_set_id=probe_artifact_set_id,
                    locked_plan_sha256=str(trial["locked_plan_sha256"] or ""),
                    sample_id=str(
                        (_load(trial["locked_plan_json"]) or {}).get(
                            "sample_id", ""
                        )
                    ),
                    namespace_identity=(
                        _load(run["executor_fingerprint_json"]) or {}
                    ).get("probe_namespace", {}),
                    policy=_load(run["policy_json"]),
                )
                evaluation_sha = content_sha256(evaluation)
                evaluation_id = stable_id(
                    "probe-evaluation",
                    {
                        "probe_artifact_set_id": probe_artifact_set_id,
                        "profile_id": evaluation["profile_id"],
                        "result_sha256": evaluation_sha,
                    },
                )
                existing = connection.execute(
                    "SELECT * FROM probe_evaluations WHERE probe_trial_id=?",
                    (probe_trial_id,),
                ).fetchone()
                if existing is not None and existing["result_sha256"] != evaluation_sha:
                    raise RuntimeError(
                        "the same probe artifact received different evaluations"
                    )
                if existing is None:
                    profile_sha = profile_sha256
                    connection.execute(
                        """INSERT INTO probe_evaluations
                           (evaluation_id,probe_trial_id,probe_artifact_set_id,
                            profile_id,profile_sha256,verdict,result_json,
                            result_sha256,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            evaluation_id,
                            probe_trial_id,
                            probe_artifact_set_id,
                            evaluation["profile_id"],
                            profile_sha,
                            verdict,
                            _dump(evaluation),
                            evaluation_sha,
                            now,
                        ),
                    )
                result = {
                    "schema": "campaignx.seed_probe_trial_result.v1",
                    "probe_artifact_set_id": probe_artifact_set_id,
                    "artifact_uri": artifact_uri,
                    "growth_receipt": growth_receipt,
                    "evaluation": evaluation,
                    "geometry_certification": geometry,
                    "ink_used": False,
                }
                trial_state = state_for[verdict]
                connection.execute(
                    """UPDATE probe_artifact_sets
                          SET artifact_uri=?,state='AVAILABLE'
                        WHERE probe_artifact_set_id=?""",
                    (artifact_uri, probe_artifact_set_id),
                )
                connection.execute(
                    """UPDATE probe_attempts
                          SET state='COMPLETED',growth_receipt_json=?,
                              result_json=?,updated_at=?
                        WHERE probe_attempt_id=?""",
                    (
                        _dump(growth_receipt),
                        _dump(result),
                        now,
                        probe_attempt_id,
                    ),
                )
                connection.execute(
                    """UPDATE probe_trials
                          SET state=?,result_json=?,worker_id=NULL,
                              lease_token=NULL,lease_expires_at=NULL,
                              active_probe_attempt_id=NULL,updated_at=?
                        WHERE probe_trial_id=?""",
                    (trial_state, _dump(result), now, probe_trial_id),
                )
                unfinished = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM probe_trials
                            WHERE probe_run_id=? AND state NOT IN
                                  ('SUCCEEDED','REJECTED','UNMEASURED','FAILED')""",
                        (trial["probe_run_id"],),
                    ).fetchone()[0]
                )
                if unfinished == 0:
                    connection.execute(
                        """UPDATE probe_runs SET state='READY_TO_DECIDE',
                                  updated_at=? WHERE probe_run_id=?""",
                        (now, trial["probe_run_id"]),
                    )
                self.event(
                    connection,
                    "PROBE_TRIAL_COMPLETED",
                    {
                        "probe_run_id": trial["probe_run_id"],
                        "probe_trial_id": probe_trial_id,
                        "probe_attempt_id": probe_attempt_id,
                        "verdict": verdict,
                        "geometry_qc_state": evaluation.get(
                            "geometry_qc_state"
                        ),
                    },
                    task_id,
                    parent_attempt_id,
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def fail_probe_trial(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_trial_id: str,
        probe_attempt_id: str,
        probe_lease_token: str,
        result: dict[str, Any],
        *,
        retryable: bool,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    connection,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                run = connection.execute(
                    "SELECT policy_json FROM probe_runs WHERE probe_run_id=?",
                    (trial["probe_run_id"],),
                ).fetchone()
                attempts = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM probe_attempts WHERE probe_trial_id=?",
                        (probe_trial_id,),
                    ).fetchone()[0]
                )
                maximum = int(
                    _load(run["policy_json"])["maximum_attempts_per_candidate"]
                )
                will_retry = bool(retryable and attempts < maximum)
                attempt_state = (
                    "RETRY_ON_LARGER_GPU"
                    if result.get("status") == "RETRY_ON_LARGER_GPU"
                    else "GROW_FAILED"
                )
                connection.execute(
                    """UPDATE probe_attempts SET state=?,result_json=?,updated_at=?
                        WHERE probe_attempt_id=?""",
                    (attempt_state, _dump(result), now, probe_attempt_id),
                )
                connection.execute(
                    """UPDATE probe_trials
                          SET state=?,result_json=?,worker_id=NULL,
                              lease_token=NULL,lease_expires_at=NULL,
                              active_probe_attempt_id=NULL,updated_at=?
                        WHERE probe_trial_id=?""",
                    (
                        "PENDING" if will_retry else "FAILED",
                        _dump(result),
                        now,
                        probe_trial_id,
                    ),
                )
                self.event(
                    connection,
                    "PROBE_TRIAL_RETRY"
                    if will_retry
                    else "PROBE_TRIAL_FAILED",
                    {
                        "probe_run_id": trial["probe_run_id"],
                        "probe_trial_id": probe_trial_id,
                        "probe_attempt_id": probe_attempt_id,
                        "result": result,
                    },
                    task_id,
                    parent_attempt_id,
                )
                connection.commit()
                return {"retry": will_retry, "attempts": attempts, "maximum": maximum}
            except BaseException:
                connection.rollback()
                raise

    def record_probe_decision(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_run_id: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            decision.get("schema") != "campaignx.seed_probe_decision.v1"
            or decision.get("probe_run_id") != probe_run_id
            or decision.get("action")
            not in {"CONTINUE_WINNER", "HUMAN_REVIEW", "REJECT_ALL"}
            or decision.get("ink_used") is not False
        ):
            raise ValueError("invalid seed probe decision")
        # The ledger, rather than the worker process, is the final authority on
        # which terminal evidence implies which action.  Without this
        # comparison a caller holding the parent lease could name any trial as
        # the winner while supplying internally well-formed JSON.
        from .seed_probe import decide_probe_run

        expected_decision = decide_probe_run(self.probe_run(probe_run_id))
        if content_sha256(decision) != content_sha256(expected_decision):
            raise RuntimeError(
                "probe decision does not match the persisted terminal evidence"
            )
        now = utc_now()
        receipt_sha = content_sha256(decision)
        decision_id = stable_id(
            "probe-decision",
            {
                "probe_run_id": probe_run_id,
                "receipt_sha256": receipt_sha,
            },
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection, task_id, parent_attempt_id, parent_lease_token
                )
                run_row = self._assert_probe_run_task(
                    connection, task_id, probe_run_id
                )
                existing = connection.execute(
                    "SELECT * FROM probe_decisions WHERE probe_run_id=?",
                    (probe_run_id,),
                ).fetchone()
                if existing is not None:
                    if existing["receipt_sha256"] != receipt_sha:
                        raise RuntimeError(
                            "probe run already has a different immutable decision"
                        )
                    connection.commit()
                    return _load(existing["receipt_json"])
                trials = connection.execute(
                    "SELECT probe_trial_id,state FROM probe_trials WHERE probe_run_id=?",
                    (probe_run_id,),
                ).fetchall()
                if not trials or any(
                    row["state"]
                    not in {"SUCCEEDED", "REJECTED", "UNMEASURED", "FAILED"}
                    for row in trials
                ):
                    raise RuntimeError(
                        "probe decision requires every trial to be terminal"
                    )
                winner = decision.get("winner_trial_id")
                if winner is not None and winner not in {
                    row["probe_trial_id"] for row in trials
                }:
                    raise RuntimeError("probe winner does not belong to this run")
                if (decision["action"] == "CONTINUE_WINNER") != (
                    winner is not None
                ):
                    raise RuntimeError(
                        "only CONTINUE_WINNER may name exactly one winner"
                    )
                connection.execute(
                    """INSERT INTO probe_decisions
                       (decision_id,probe_run_id,policy_id,policy_sha256,
                        evidence_set_sha256,action,winner_trial_id,receipt_json,
                        receipt_sha256,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        decision_id,
                        probe_run_id,
                        decision["policy_id"],
                        decision["policy_sha256"],
                        decision["evidence_set_sha256"],
                        decision["action"],
                        winner,
                        _dump(decision),
                        receipt_sha,
                        now,
                    ),
                )
                state_for = {
                    "CONTINUE_WINNER": "CONTINUATION_QUEUED",
                    "HUMAN_REVIEW": "HUMAN_REVIEW",
                    "REJECT_ALL": "REJECTED",
                }
                run_state = (
                    "SHADOW_COMPLETE"
                    if _load(run_row["policy_json"])["mode"] == "shadow"
                    else state_for[decision["action"]]
                )
                mode = _load(run_row["policy_json"])["mode"]
                connection.execute(
                    "UPDATE probe_runs SET state=?,updated_at=? WHERE probe_run_id=?",
                    (run_state, now, probe_run_id),
                )
                retain_30_days = _deadline(30 * 24 * 60 * 60)
                if mode == "shadow":
                    connection.execute(
                        """UPDATE probe_artifact_sets
                              SET state='BENCHMARK_RETAINED',
                                  retain_until=NULL
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id FROM probe_trials
                                   WHERE probe_run_id=?)""",
                        (probe_run_id,),
                    )
                elif decision["action"] == "HUMAN_REVIEW":
                    connection.execute(
                        """UPDATE probe_artifact_sets
                              SET state='REVIEW_RETAINED',retain_until=NULL
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id FROM probe_trials
                                   WHERE probe_run_id=?)""",
                        (probe_run_id,),
                    )
                else:
                    connection.execute(
                        """UPDATE probe_artifact_sets
                              SET state='LOSER_RETAINED',retain_until=?
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id FROM probe_trials
                                   WHERE probe_run_id=?)
                              AND (? IS NULL OR probe_trial_id<>?)""",
                        (
                            retain_30_days,
                            probe_run_id,
                            winner,
                            winner,
                        ),
                    )
                    if winner is not None:
                        connection.execute(
                            """UPDATE probe_artifact_sets
                                  SET state='WINNER_RETAINED',
                                      retain_until=NULL
                                WHERE probe_trial_id=?""",
                            (winner,),
                        )
                self.event(
                    connection,
                    f"PROBE_DECISION_{decision['action']}",
                    {
                        "probe_run_id": probe_run_id,
                        "decision_id": decision_id,
                        "winner_trial_id": winner,
                        "receipt_sha256": receipt_sha,
                    },
                    task_id,
                    parent_attempt_id,
                )
                connection.commit()
                return decision
            except BaseException:
                connection.rollback()
                raise

    def begin_probe_promotion(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_run_id: str,
        locked_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Durably link one selected probe to the ordinary full-grow boundary."""

        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_probe_parent_owner(
                    connection,
                    task_id,
                    parent_attempt_id,
                    parent_lease_token,
                )
                self._assert_probe_run_task(
                    connection, task_id, probe_run_id
                )
                row = connection.execute(
                    """SELECT d.decision_id,d.receipt_sha256,d.winner_trial_id,
                              d.action,r.task_id,r.policy_json,
                              r.source_snapshot_id,
                              t.candidate_json,
                              a.probe_artifact_set_id,a.manifest_sha256,
                              a.artifact_uri
                         FROM probe_decisions d
                         JOIN probe_runs r ON r.probe_run_id=d.probe_run_id
                         JOIN probe_trials t
                           ON t.probe_trial_id=d.winner_trial_id
                         JOIN probe_artifact_sets a
                           ON a.probe_trial_id=d.winner_trial_id
                        WHERE d.probe_run_id=? AND r.task_id=?""",
                    (probe_run_id, task_id),
                ).fetchone()
                if (
                    row is None
                    or row["task_id"] != task_id
                    or row["action"] != "CONTINUE_WINNER"
                    or _load(row["policy_json"])["mode"] != "select"
                ):
                    raise RuntimeError(
                        "only a persisted select-mode winner can be promoted"
                    )
                from .seed_probe import build_probe_continuation_contract

                continuation_contract = build_probe_continuation_contract(
                    task_id=task_id,
                    continuation_attempt_id=parent_attempt_id,
                    probe_run_id=probe_run_id,
                    source_snapshot_id=row["source_snapshot_id"],
                    decision_id=row["decision_id"],
                    decision_sha256=row["receipt_sha256"],
                    winner_trial_id=row["winner_trial_id"],
                    winner_probe_artifact_set_id=row[
                        "probe_artifact_set_id"
                    ],
                    winner_manifest_sha256=row["manifest_sha256"],
                    winner_artifact_uri=row["artifact_uri"],
                    winner_candidate=_load(row["candidate_json"]),
                    locked_plan=locked_plan,
                )
                continuation_contract_sha = content_sha256(
                    continuation_contract
                )
                continuation_locked_plan_sha = content_sha256(locked_plan)
                receipt = {
                    "schema": "campaignx.seed_probe_promotion.v1",
                    "probe_run_id": probe_run_id,
                    "decision_id": row["decision_id"],
                    "decision_sha256": row["receipt_sha256"],
                    "winner_trial_id": row["winner_trial_id"],
                    "winner_probe_artifact_set_id": row[
                        "probe_artifact_set_id"
                    ],
                    # V1 continues under the already leased ordinary task.  It
                    # still crosses the catalogue boundary only in finalize().
                    "continuation_task_id": task_id,
                    "continuation_attempt_id": parent_attempt_id,
                    "continuation_contract": continuation_contract,
                    "continuation_contract_sha256": (
                        continuation_contract_sha
                    ),
                    "continuation_locked_plan_sha256": (
                        continuation_locked_plan_sha
                    ),
                    "state": "CONTINUING",
                    "ink_used": False,
                    "non_claim": (
                        "a retained probe winner is not a canonical surface"
                    ),
                }
                receipt_sha = content_sha256(receipt)
                promotion_id = stable_id(
                    "probe-promotion",
                    {
                        "decision_id": row["decision_id"],
                        "continuation_task_id": task_id,
                    },
                )
                existing = connection.execute(
                    "SELECT * FROM probe_promotions WHERE decision_id=?",
                    (row["decision_id"],),
                ).fetchone()
                if existing is not None:
                    immutable_match = (
                        existing["winner_trial_id"] == row["winner_trial_id"]
                        and existing["winner_probe_artifact_set_id"]
                        == row["probe_artifact_set_id"]
                        and existing["continuation_task_id"] == task_id
                        and existing["continuation_contract_sha256"]
                        == continuation_contract_sha
                    )
                    if not immutable_match:
                        raise RuntimeError(
                            "probe winner already has a conflicting promotion"
                        )
                    if existing["state"] == "PROMOTED":
                        connection.commit()
                        return _load(existing["receipt_json"])
                    if existing["state"] != "CONTINUING":
                        raise RuntimeError(
                            "probe promotion is not open for continuation"
                        )
                    connection.execute(
                        """UPDATE probe_promotions
                              SET continuation_attempt_id=?,
                                  continuation_locked_plan_sha256=?,
                                  receipt_json=?,receipt_sha256=?,updated_at=?
                            WHERE promotion_id=?""",
                        (
                            parent_attempt_id,
                            continuation_locked_plan_sha,
                            _dump(receipt),
                            receipt_sha,
                            now,
                            existing["promotion_id"],
                        ),
                    )
                    connection.commit()
                    return receipt
                connection.execute(
                    """INSERT INTO probe_promotions
                       (promotion_id,decision_id,winner_trial_id,
                        winner_probe_artifact_set_id,continuation_task_id,
                        continuation_attempt_id,
                        continuation_contract_sha256,
                        continuation_locked_plan_sha256,
                        state,receipt_json,receipt_sha256,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,'CONTINUING',?,?,?,?)""",
                    (
                        promotion_id,
                        row["decision_id"],
                        row["winner_trial_id"],
                        row["probe_artifact_set_id"],
                        task_id,
                        parent_attempt_id,
                        continuation_contract_sha,
                        continuation_locked_plan_sha,
                        _dump(receipt),
                        receipt_sha,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """UPDATE probe_artifact_sets SET state='WINNER_RETAINED'
                        WHERE probe_artifact_set_id=?""",
                    (row["probe_artifact_set_id"],),
                )
                connection.execute(
                    "UPDATE probe_runs SET state='CONTINUING',updated_at=? "
                    "WHERE probe_run_id=?",
                    (now, probe_run_id),
                )
                self.event(
                    connection,
                    "PROBE_PROMOTION_CONTINUING",
                    {
                        "probe_run_id": probe_run_id,
                            "promotion_id": promotion_id,
                            "winner_trial_id": row["winner_trial_id"],
                            "continuation_locked_plan_sha256": (
                                continuation_locked_plan_sha
                            ),
                    },
                    task_id,
                    parent_attempt_id,
                )
                connection.commit()
                return receipt
            except BaseException:
                connection.rollback()
                raise

    def mark_probe_continuation_review(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_run_id: str,
        receipt: dict[str, Any],
    ) -> None:
        """Atomically stop an unreadable/corrupt selected winner for review."""

        if (
            receipt.get("status") != "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE"
            or receipt.get("failure_class") != "SOURCE_FAILURE"
            or receipt.get("probe_run_id") != probe_run_id
            or receipt.get("ink_used") is not False
        ):
            raise ValueError(
                "probe continuation review receipt is incomplete"
            )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                scope = connection.execute(
                    """SELECT mission_id,policy_version
                         FROM tasks WHERE task_id=?""",
                    (task_id,),
                ).fetchone()
                if scope is None:
                    raise RuntimeError("attempt no longer owns the task")
                self._assert_probe_parent_owner(
                    connection,
                    task_id,
                    parent_attempt_id,
                    parent_lease_token,
                )
                run = self._assert_probe_run_task(
                    connection, task_id, probe_run_id
                )
                decision = connection.execute(
                    """SELECT action,winner_trial_id
                         FROM probe_decisions WHERE probe_run_id=?""",
                    (probe_run_id,),
                ).fetchone()
                promotion = connection.execute(
                    """SELECT 1 FROM probe_promotions p
                         JOIN probe_decisions d
                           ON d.decision_id=p.decision_id
                        WHERE d.probe_run_id=?""",
                    (probe_run_id,),
                ).fetchone()
                if (
                    run["state"] != "CONTINUATION_QUEUED"
                    or decision is None
                    or decision["action"] != "CONTINUE_WINNER"
                    or promotion is not None
                ):
                    raise RuntimeError(
                        "only an unpromoted queued winner can enter review"
                    )
                connection.execute(
                    """UPDATE attempts SET state='BLOCKED_PROBE_ARTIFACT_UNAVAILABLE',
                              result_json=?,updated_at=?
                        WHERE attempt_id=?""",
                    (_dump(receipt), now, parent_attempt_id),
                )
                connection.execute(
                    """UPDATE tasks
                          SET state='BLOCKED_PROBE_ARTIFACT_UNAVAILABLE',worker_id=NULL,
                              lease_token=NULL,lease_expires_at=NULL,
                              updated_at=?
                        WHERE task_id=?""",
                    (now, task_id),
                )
                connection.execute(
                    """UPDATE probe_runs SET state='REVIEW_PENDING',
                              updated_at=? WHERE probe_run_id=?""",
                    (now, probe_run_id),
                )
                connection.execute(
                    """UPDATE probe_artifact_sets
                          SET state='REVIEW_RETAINED',retain_until=NULL
                        WHERE probe_trial_id IN (
                              SELECT probe_trial_id FROM probe_trials
                               WHERE probe_run_id=?)""",
                    (probe_run_id,),
                )
                controlled = any(
                    (_load(row["admission_json"]).get(
                        "execution_bindings") or {}).get("policy_version")
                    == scope["policy_version"]
                    for row in connection.execute(
                        """SELECT admission_json
                             FROM campaign_budget_admissions
                            WHERE mission_id=?""",
                        (scope["mission_id"],),
                    ).fetchall()
                )
                if controlled:
                    self._refresh_campaign_decisions(
                        connection,
                        mission_id=scope["mission_id"],
                        policy_version=scope["policy_version"],
                    )
                self.event(
                    connection,
                    "PROBE_CONTINUATION_REVIEW_REQUIRED",
                    {
                        "probe_run_id": probe_run_id,
                        "winner_trial_id": decision["winner_trial_id"],
                        "reason": receipt.get("reason"),
                    },
                    task_id,
                    parent_attempt_id,
                )
                self.event(
                    connection,
                    "STATE_BLOCKED_PROBE_ARTIFACT_UNAVAILABLE",
                    receipt,
                    task_id,
                    parent_attempt_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def probe_status(self) -> dict[str, Any]:
        with self.connect() as connection:
            runs = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM probe_runs GROUP BY state"
                )
            }
            trials = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM probe_trials GROUP BY state"
                )
            }
            decisions = {
                row["action"]: int(row["count"])
                for row in connection.execute(
                    "SELECT action,COUNT(*) AS count FROM probe_decisions GROUP BY action"
                )
            }
            promotions = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM probe_promotions "
                    "GROUP BY state"
                )
            }
        return {
            "schema": "campaignx.seed_probe_status.v1",
            "runs": dict(sorted(runs.items())),
            "trials": dict(sorted(trials.items())),
            "decisions": dict(sorted(decisions.items())),
            "promotions": dict(sorted(promotions.items())),
            "noncanonical": True,
        }

    def heartbeat(self, task_id: str, attempt_id: str, lease_token: str, lease_seconds: int) -> None:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE tasks SET lease_expires_at=?,updated_at=? WHERE task_id=? AND active_attempt_id=? AND lease_token=? AND state IN ({','.join('?' for _ in ACTIVE_STATES)})",
                (_deadline(lease_seconds), now, task_id, attempt_id, lease_token, *ACTIVE_STATES),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("lease heartbeat rejected")
            connection.execute("UPDATE attempts SET updated_at=? WHERE attempt_id=? AND lease_token=?", (now, attempt_id, lease_token))

    def transition(self, task_id: str, attempt_id: str, lease_token: str, state: str, *, proposal: dict[str, Any] | None = None, locked_plan: dict[str, Any] | None = None, result: dict[str, Any] | None = None) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT state,active_attempt_id,lease_token FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                if row is None or row["active_attempt_id"] != attempt_id or row["lease_token"] != lease_token:
                    raise RuntimeError("attempt no longer owns the task")
                fields: dict[str, Any] = {"state": state, "updated_at": now}
                if proposal is not None:
                    fields.update(proposal_json=_dump(proposal), proposal_sha256=content_sha256(proposal))
                if locked_plan is not None:
                    fields.update(locked_plan_json=_dump(locked_plan), locked_plan_sha256=content_sha256(locked_plan))
                if result is not None:
                    fields["result_json"] = _dump(result)
                assignments = ",".join(f"{key}=?" for key in fields)
                connection.execute(f"UPDATE attempts SET {assignments} WHERE attempt_id=?", (*fields.values(), attempt_id))
                connection.execute("UPDATE tasks SET state=?,updated_at=? WHERE task_id=?", (state, now, task_id))
                self.event(connection, f"STATE_{state}", result or {}, task_id, attempt_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def add_artifact_set(self, task_id: str, attempt_id: str, lease_token: str, manifest: dict[str, Any], staging_uri: str) -> str:
        artifact_id = stable_id("artifact-set", {"attempt_id": attempt_id, "manifest": manifest})
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute("SELECT active_attempt_id,lease_token FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                if owner is None or owner["active_attempt_id"] != attempt_id or owner["lease_token"] != lease_token:
                    raise RuntimeError("artifact upload belongs to a stale lease")
                connection.execute(
                    "INSERT INTO artifact_sets(artifact_set_id,attempt_id,manifest_json,manifest_sha256,staging_uri,state,created_at) VALUES(?,?,?,?,?,'UPLOADED',?)",
                    (artifact_id, attempt_id, _dump(manifest), content_sha256(manifest), staging_uri, utc_now()),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        self.transition(task_id, attempt_id, lease_token, "UPLOADED", result={"artifact_set_id": artifact_id})
        return artifact_id

    def finalize(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        surface: dict[str, Any],
        artifact_set_id: str,
        qc_profile_id: str,
    ) -> dict[str, Any]:
        if not qc_profile_id or "@" not in qc_profile_id:
            raise ValueError("finalization requires a versioned semantic QC profile ID")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                context = connection.execute(
                    """SELECT t.state AS task_state,t.source_snapshot_id,
                              t.mission_id,t.policy_version,
                              t.payload_json,t.seed_probe_required,
                              t.active_attempt_id,t.lease_token,
                              a.task_id AS attempt_task_id,
                              a.state AS attempt_state,
                              a.locked_plan_json,a.locked_plan_sha256,
                              a.result_json,
                              s.attempt_id AS artifact_attempt_id,
                              s.state AS artifact_state,
                              s.manifest_json,s.manifest_sha256
                         FROM tasks t
                         JOIN attempts a ON a.attempt_id=?
                         JOIN artifact_sets s ON s.artifact_set_id=?
                        WHERE t.task_id=?""",
                    (attempt_id, artifact_set_id, task_id),
                ).fetchone()
                if (
                    context is None
                    or context["attempt_task_id"] != task_id
                ):
                    raise RuntimeError(
                        "finalization task, attempt, or artifact set does not exist"
                    )
                task_payload = _load(context["payload_json"]) or {}
                replay = (
                    context["task_state"] in FINALIZATION_OUTCOME_STATES
                    and context["attempt_state"] == context["task_state"]
                )
                validate_finalization_evidence(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    task_state=context["task_state"],
                    attempt_state=context["attempt_state"],
                    task_source_snapshot_id=context["source_snapshot_id"],
                    task_sample_id=str(task_payload.get("sample_id") or ""),
                    attempt_locked_plan_sha256=context["locked_plan_sha256"],
                    artifact_attempt_id=context["artifact_attempt_id"],
                    artifact_state=context["artifact_state"],
                    artifact_manifest=_load(context["manifest_json"]),
                    artifact_manifest_sha256=context["manifest_sha256"],
                    surface=surface,
                    replay=replay,
                )
                locked_plan = _load(context["locked_plan_json"]) or {}
                probe_authority = connection.execute(
                    """SELECT r.probe_run_id,r.state AS run_state,
                              r.policy_json,d.decision_id,d.action,
                              d.winner_trial_id,d.receipt_json,d.receipt_sha256
                         FROM probe_runs r
                         LEFT JOIN probe_decisions d
                           ON d.probe_run_id=r.probe_run_id
                        WHERE r.task_id=?""",
                    (task_id,),
                ).fetchall()
                if len(probe_authority) > 1:
                    raise RuntimeError(
                        "task has more than one authoritative probe run"
                    )
                authority_row = (
                    probe_authority[0] if probe_authority else None
                )
                probe_run = (
                    {
                        "probe_run_id": authority_row["probe_run_id"],
                        "state": authority_row["run_state"],
                        "policy": _load(authority_row["policy_json"]),
                    }
                    if authority_row is not None
                    else None
                )
                probe_decision = (
                    {
                        "decision_id": authority_row["decision_id"],
                        "action": authority_row["action"],
                        "winner_trial_id": authority_row[
                            "winner_trial_id"
                        ],
                        "receipt": _load(authority_row["receipt_json"]),
                        "receipt_sha256": authority_row["receipt_sha256"],
                    }
                    if authority_row is not None
                    and authority_row["decision_id"] is not None
                    else None
                )
                promotion = connection.execute(
                    """SELECT p.*,d.probe_run_id
                         FROM probe_promotions p
                         JOIN probe_decisions d
                           ON d.decision_id=p.decision_id
                        WHERE p.continuation_task_id=?""",
                    (task_id,),
                ).fetchone()
                from .seed_probe import (
                    validate_probe_finalization_authority,
                )

                validate_probe_finalization_authority(
                    task_payload=task_payload,
                    seed_probe_required=bool(
                        context["seed_probe_required"]
                    ),
                    probe_run=probe_run,
                    probe_decision=probe_decision,
                    promotion=(
                        dict(promotion) if promotion is not None else None
                    ),
                    locked_plan=locked_plan,
                    replay=replay,
                )
                if promotion is not None:
                    expected_promotion_state = (
                        "PROMOTED" if replay else "CONTINUING"
                    )
                    if (
                        promotion["state"] != expected_promotion_state
                        or promotion["continuation_attempt_id"] != attempt_id
                        or promotion["continuation_locked_plan_sha256"]
                        != context["locked_plan_sha256"]
                    ):
                        raise RuntimeError(
                            "probe promotion is not bound to this final locked plan"
                        )
                    promotion_receipt = (
                        _load(promotion["receipt_json"]) or {}
                    )
                    continuation_contract = promotion_receipt.get(
                        "continuation_contract"
                    )
                    if (
                        content_sha256(promotion_receipt)
                        != promotion["receipt_sha256"]
                        or not isinstance(continuation_contract, dict)
                        or content_sha256(continuation_contract)
                        != promotion["continuation_contract_sha256"]
                        or promotion_receipt.get(
                            "continuation_contract_sha256"
                        )
                        != promotion["continuation_contract_sha256"]
                        or promotion_receipt.get(
                            "continuation_locked_plan_sha256"
                        )
                        != context["locked_plan_sha256"]
                        or promotion_receipt.get("continuation_attempt_id")
                        != attempt_id
                    ):
                        raise RuntimeError(
                            "probe promotion continuation receipt is corrupt"
                        )
                    from .seed_probe import (
                        validate_probe_continuation_contract,
                    )

                    validate_probe_continuation_contract(
                        continuation_contract,
                        locked_plan,
                        task_id=task_id,
                        attempt_id=attempt_id,
                    )
                if replay:
                    prior = _load(context["result_json"]) or {}
                    if (
                        prior.get("artifact_set_id") != artifact_set_id
                        or prior.get("surface_id") != surface.get("surface_id")
                    ):
                        raise RuntimeError(
                            "finalization replay differs from the terminal result"
                        )
                    connection.commit()
                    return {
                        "status": context["task_state"],
                        "duplicate_of": prior.get("duplicate_of"),
                        "duplicate_diagnostics": prior.get(
                            "duplicate_diagnostics", {}
                        ),
                        "geometry_qc_state": prior.get("geometry_qc_state"),
                        "geometry_blocked_qc": bool(
                            prior.get("geometry_blocked_qc", False)
                        ),
                        "probe_promotion_id": (
                            promotion["promotion_id"]
                            if promotion is not None
                            else None
                        ),
                    }
                if (
                    context["active_attempt_id"] != attempt_id
                    or context["lease_token"] != lease_token
                ):
                    raise RuntimeError("finalization belongs to a stale lease")
                task6_controlled = (
                    task_payload.get("schema") ==
                        "campaignx.first_letters_promotion_child_task.v1"
                    or task_payload.get("namespace") ==
                        "NONCANONICAL_DISCOVERY"
                    or isinstance(task_payload.get("normal_growth_lock"), dict)
                )
                from .canonical_lineage import require_canonical_lineage  # noqa: PLC0415
                finalization_lineage = {
                    "schema": "campaignx.authoritative_surface_lineage.v1",
                    "mission_id": context["mission_id"],
                    "surface_id": surface.get("surface_id"),
                    "namespace": task_payload.get("namespace") or
                        surface.get("namespace") or "CANONICAL_SURFACE",
                    "artifact_identity": f"artifact_sets:{artifact_set_id}",
                    "artifact_sha256": surface.get("artifact_sha256"),
                    "artifact_uri": surface.get("artifact_uri"),
                    "source_snapshot_id": context["source_snapshot_id"],
                    "source_binding_sha256": content_sha256({
                        "source_snapshot_id": context["source_snapshot_id"],
                        "task_id": task_id, "attempt_id": attempt_id,
                    }),
                    "promotion_lineage_sha256": task_payload.get(
                        "promotion_authority_sha256"),
                    "promotion_lineage_kind": (
                        "FRESH_ORDINARY_CHILD"
                        if task_payload.get("schema") ==
                            "campaignx.first_letters_promotion_child_task.v1"
                        else "DISCOVERY_PARENT"
                        if task_payload.get("namespace") ==
                            "NONCANONICAL_DISCOVERY" else None
                    ),
                    "route_sha256": None,
                    "surface_state": "QC_PENDING",
                    "canonical": task_payload.get("namespace") !=
                        "NONCANONICAL_DISCOVERY",
                    "external": False,
                    "external_admission_sha256": None,
                    "ambiguous": False,
                    "hash_conflict": False,
                }
                require_canonical_lineage(
                    boundary="P1_FINALIZATION_INSERT",
                    controlled_mission=task6_controlled,
                    authoritative_lineage=finalization_lineage,
                    allow_unvalidated=(
                        task_payload.get("allow_unvalidated")
                        if task6_controlled else False
                    ),
                )
                if task6_controlled:
                    surface = {
                        **surface,
                        "mission_id": context["mission_id"],
                        "controlled_first_letters": True,
                        "authoritative_lineage": finalization_lineage,
                        "allow_unvalidated": False,
                    }
                known = []
                for row in connection.execute(
                    "SELECT surface_id,artifact_sha256,sample_points_json FROM surfaces WHERE source_snapshot_id=? ORDER BY surface_id",
                    (surface["source_snapshot_id"],),
                ):
                    known.append({
                        "surface_id": row["surface_id"],
                        "artifact_sha256": row["artifact_sha256"],
                        "sample_points": _load(row["sample_points_json"]),
                    })
                duplicate_of, duplicate_diagnostics = find_duplicate_in_surfaces(
                    known,
                    surface["artifact_sha256"],
                    surface.get("sample_points") or [],
                )
                fixture_only = is_fixture_surface(surface)
                state = (
                    "DUPLICATE_SURFACE"
                    if duplicate_of
                    else "FIXTURE_ONLY"
                    if fixture_only
                    else "QC_PENDING"
                )
                surface_id = str(surface["surface_id"])
                geometry_state = str(
                    surface.get("geometry_qc_state") or DEFAULT_GEOMETRY_QC_STATE
                )
                if geometry_state not in GEOMETRY_QC_STATES:
                    raise ValueError(f"unsupported geometry QC state: {geometry_state}")
                geometry_rejected = is_geometry_rejected(geometry_state)
                # Re-resolved here rather than read off the task: this is the
                # transaction that creates the successor, and it is the only
                # point at which the predecessor's row and routing receipt are
                # held still. The stamp from task creation is compared, not
                # trusted -- the catalogue could have moved since.
                from . import surface_expansion  # noqa: PLC0415

                expansion_authority = self._resolve_expansion_authority(
                    connection, successor_surface_id=surface_id,
                    source=task_payload)
                stamped_expansion = task_payload.get("expansion_authority")
                if expansion_authority is not None:
                    if stamped_expansion is not None and not (
                        surface_expansion.agrees_with_stamp(
                            expansion_authority, stamped_expansion)
                    ):
                        raise RuntimeError(
                            "the expansion this task was queued under is not "
                            "the one the catalogue resolves now")
                elif stamped_expansion is not None:
                    raise RuntimeError(
                        "the task carries an expansion authority but continues "
                        "no surface")
                if not duplicate_of:
                    # Decide the route before the row exists, and refuse here if
                    # it cannot be decided. The persisted receipt follows the
                    # surface insert only because it references it; both land in
                    # this transaction, so no committed surface row has ever
                    # existed without its routing decision beside it.
                    from . import surface_routing  # noqa: PLC0415

                    if expansion_authority is not None:
                        surface = {**surface, "resumes_surface":
                                   expansion_authority["expands_surface_id"]}
                    routing_decision = surface_routing.receipt_for_surface(
                        {**surface, "surface_id": surface_id,
                         "geometry_qc_state": geometry_state})
                    if not surface_routing.verify_receipt(routing_decision):
                        raise RuntimeError(
                            "surface routing decision does not verify")
                    surface_state = "FIXTURE_ONLY" if fixture_only else "QC_PENDING"
                    physical_state = (
                        "NOT_APPLICABLE_FIXTURE" if fixture_only else "UNVALIDATED"
                    )
                    connection.execute(
                        """INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz_json,sample_points_json,area_cm2,state,physical_qc_state,geometry_qc_state,payload_json,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (surface_id, surface["source_snapshot_id"], surface["sample_id"], surface.get("owner", "campaign-x"), surface["artifact_sha256"], surface["artifact_uri"], _dump(surface["bbox_xyz"]), _dump(surface.get("sample_points")) if surface.get("sample_points") is not None else None, surface.get("area_cm2"), surface_state, physical_state, geometry_state, _dump(surface), now),
                    )
                    routing_receipt = self._require_routing_receipt(
                        connection, {**surface, "surface_id": surface_id,
                                     "geometry_qc_state": geometry_state}, now)
                    if routing_receipt != routing_decision:
                        raise RuntimeError(
                            "the persisted routing receipt is not the decision "
                            "this finalization made")
                    if expansion_authority is not None:
                        self._persist_expansion_authority(
                            connection, expansion_authority, now)
                    if not fixture_only:
                        qc_id = stable_id(
                            "qc-job",
                            {"surface_id": surface_id, "profile_id": qc_profile_id},
                        )
                        # The geometry gate sits between finalization and the
                        # model.  A surface with a hard geometric defect keeps a
                        # durable, auditable job row, but it is created FAILED so
                        # claim_qc -- which only takes PENDING -- can never hand
                        # it to the ink model.
                        #
                        # The size gate sits beside it and geometry cannot see
                        # it: PHerc0268 was GEOMETRY_CERTIFIED at two square
                        # millimetres. Exactly a verified STANDARD route earns a
                        # claimable job.
                        job_state = qc_job_state_for(geometry_state)
                        if not surface_routing.enters_standard_qc(
                            routing_receipt
                        ):
                            job_state = QC_SMALL_SURFACE_DIAGNOSTIC
                        connection.execute(
                            "INSERT INTO qc_jobs(qc_job_id,surface_id,profile_id,state,payload_json,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                            (
                                qc_id,
                                surface_id,
                                qc_profile_id,
                                job_state,
                                _dump({
                                    "artifact_set_id": artifact_set_id,
                                    "surface_artifact_sha256": surface["artifact_sha256"],
                                    "source_attempt_id": surface.get("attempt_id"),
                                    "created_geometry_certified": (
                                        geometry_state == "GEOMETRY_CERTIFIED"),
                                }),
                                _dump(
                                    {
                                        "schema": "campaignx.segment_qc_geometry_block.v1",
                                        "geometry_qc_state": geometry_state,
                                        "geometry_certification": surface.get(
                                            "geometry_certification"
                                        ),
                                        "no_scientific_conclusion": True,
                                    }
                                )
                                if geometry_rejected
                                else None,
                                now,
                                now,
                            ),
                        )
                        if geometry_rejected:
                            self.event(
                                connection,
                                "GEOMETRY_REJECTED_BEFORE_MODEL",
                                {
                                    "surface_id": surface_id,
                                    "qc_job_id": qc_id,
                                    "geometry_qc_state": geometry_state,
                                },
                                task_id,
                                attempt_id,
                            )
                artifact_update = connection.execute(
                    """UPDATE artifact_sets SET state=?
                        WHERE artifact_set_id=? AND attempt_id=?
                          AND state='UPLOADED'""",
                    (
                        "DUPLICATE" if duplicate_of else "PROMOTED",
                        artifact_set_id,
                        attempt_id,
                    ),
                )
                if artifact_update.rowcount != 1:
                    raise RuntimeError(
                        "artifact set changed during finalization"
                    )
                result = {
                    "surface_id": surface_id,
                    "artifact_set_id": artifact_set_id,
                    "duplicate_of": duplicate_of,
                    "duplicate_diagnostics": duplicate_diagnostics,
                    "geometry_qc_state": geometry_state,
                    "geometry_blocked_qc": bool(
                        geometry_rejected and not duplicate_of
                    ),
                }
                if state == "FIXTURE_ONLY":
                    result["failure_class"] = "FIXTURE_ONLY"
                promotion_id = None
                if promotion is not None:
                    promotion_id = promotion["promotion_id"]
                    canonical_surface_id = duplicate_of or surface_id
                    canonical_artifact_set_id = (
                        None if duplicate_of else artifact_set_id
                    )
                    promotion_receipt = {
                        **_load(promotion["receipt_json"]),
                        "state": "PROMOTED",
                        **(
                            {
                                "canonical_artifact_set_id": (
                                    canonical_artifact_set_id
                                )
                            }
                            if canonical_artifact_set_id is not None
                            else {}
                        ),
                        "surface_id": canonical_surface_id,
                        "final_status": state,
                    }
                    connection.execute(
                        """UPDATE probe_promotions
                              SET canonical_artifact_set_id=?,surface_id=?,
                                  state='PROMOTED',receipt_json=?,
                                  receipt_sha256=?,updated_at=?
                            WHERE promotion_id=?""",
                        (
                            canonical_artifact_set_id,
                            canonical_surface_id,
                            _dump(promotion_receipt),
                            content_sha256(promotion_receipt),
                            now,
                            promotion_id,
                        ),
                    )
                    connection.execute(
                        "UPDATE probe_runs SET state='PROMOTED',updated_at=? "
                        "WHERE probe_run_id=?",
                        (now, promotion["probe_run_id"]),
                    )
                    connection.execute(
                        """UPDATE probe_artifact_sets
                              SET state='PROMOTED_RETAINED',retain_until=?
                            WHERE probe_artifact_set_id=?""",
                        (
                            _deadline(30 * 24 * 60 * 60),
                            promotion["winner_probe_artifact_set_id"],
                        ),
                    )
                    self.event(
                        connection,
                        "PROBE_PROMOTED",
                        {
                            "probe_run_id": promotion["probe_run_id"],
                            "promotion_id": promotion_id,
                            "canonical_artifact_set_id": (
                                canonical_artifact_set_id
                            ),
                            "surface_id": canonical_surface_id,
                            "final_status": state,
                        },
                        task_id,
                        attempt_id,
                    )
                connection.execute("UPDATE attempts SET state=?,result_json=?,updated_at=? WHERE attempt_id=?", (state, _dump(result), now, attempt_id))
                connection.execute("UPDATE tasks SET state=?,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,updated_at=? WHERE task_id=?", (state, now, task_id))
                controlled = any(
                    (_load(row["admission_json"]).get(
                        "execution_bindings") or {}).get("policy_version")
                    == context["policy_version"]
                    for row in connection.execute(
                        """SELECT admission_json
                             FROM campaign_budget_admissions
                            WHERE mission_id=?""",
                        (context["mission_id"],),
                    ).fetchall()
                )
                if controlled:
                    self._refresh_campaign_decisions(
                        connection,
                        mission_id=context["mission_id"],
                        policy_version=context["policy_version"],
                    )
                self.event(connection, f"STATE_{state}", result, task_id, attempt_id)
                connection.commit()
                return {
                    "status": state,
                    "duplicate_of": duplicate_of,
                    "duplicate_diagnostics": duplicate_diagnostics,
                    "geometry_qc_state": geometry_state,
                    "geometry_blocked_qc": bool(geometry_rejected and not duplicate_of),
                    "probe_promotion_id": promotion_id,
                }
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise RuntimeError(f"surface finalization violated catalogue uniqueness: {error}") from error
            except BaseException:
                connection.rollback()
                raise

    # A surface belongs to a mission if one of its tasks grew it or it was
    # uploaded into it. The third way -- derived by one of the mission's ink
    # jobs -- cannot be asked here: `ink_jobs` and `surface_derivations` live in
    # the ink store, and a derived surface therefore cannot exist in this
    # mirror either. Consumes two copies of the mission id.
    MISSION_SURFACE_PREDICATE = """(
      EXISTS (
        SELECT 1 FROM artifact_sets art
        JOIN attempts a ON a.attempt_id=art.attempt_id
        JOIN tasks t ON t.task_id=a.task_id
        WHERE json_extract(art.manifest_json,'$.artifact_sha256')
              = surfaces.artifact_sha256
          AND t.mission_id = ?)
      OR json_extract(surfaces.payload_json,'$.mission_id') = ?
    )"""

    def surfaces_without_geometry_verdict(
        self, limit: int = 25, sample_id: str | None = None,
        surface_id: str | None = None, mission_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """The sqlite mirror of the same question, so tests need no server."""
        query = """SELECT surface_id, sample_id, artifact_uri, artifact_sha256,
                          state, geometry_qc_state, payload_json
                   FROM surfaces
                   WHERE (geometry_qc_state IS NULL
                          OR geometry_qc_state = 'GEOMETRY_UNMEASURED')
                     AND artifact_uri IS NOT NULL"""
        arguments: list[Any] = []
        if sample_id is not None:
            query += " AND sample_id=?"
            arguments.append(sample_id)
        if surface_id is not None:
            query += " AND surface_id=?"
            arguments.append(surface_id)
        if mission_id is not None:
            query += " AND " + self.MISSION_SURFACE_PREDICATE
            arguments.extend([mission_id] * 2)
        query += " ORDER BY created_at, surface_id LIMIT ?"
        arguments.append(int(limit))
        with self.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                query, arguments
            ).fetchall()]
        result = []
        for row in rows:
            payload = _load(row.pop("payload_json")) or {}
            self._require_surface_payload_lineage(
                payload, boundary="P2_QUEUE_ADMISSION"
            )
            result.append(row)
        return result

    def surface_artifact(
        self, surface_id: str, *, boundary: str = "P2_EXECUTION_RESOLUTION",
    ) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT surface_id,source_snapshot_id,sample_id,artifact_uri,"
                "artifact_sha256,payload_json "
                "FROM surfaces WHERE surface_id=?", (surface_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"unknown surface: {surface_id}")
        payload = _load(row["payload_json"]) or {}
        self._require_surface_payload_lineage(payload, boundary=boundary)
        return {"surface_id": row["surface_id"],
                "source_snapshot_id": row["source_snapshot_id"],
                "sample_id": row["sample_id"], "artifact_uri": row["artifact_uri"],
                "artifact_sha256": row["artifact_sha256"],
                "payload": payload}

    def record_geometry_certification(
        self,
        surface_id: str,
        geometry_state: str,
        receipt: dict[str, Any] | None = None,
        *,
        requested_by_job_id: str,
        profile_id: str,
        profile_sha256: str,
    ) -> dict[str, Any]:
        """Record a geometry verdict on the axis orthogonal to physical QC.

        Re-certifying an already imported surface uses this path: it never
        touches ``physical_qc_state``, and a rejection fails any QC job that has
        not already been claimed so the surface cannot reach the ink model.
        """

        if geometry_state not in GEOMETRY_QC_STATES:
            raise ValueError(f"unsupported geometry QC state: {geometry_state}")
        if not requested_by_job_id or not profile_id or len(profile_sha256) != 64:
            raise ValueError("geometry certification requires job/profile/hash lineage")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT surface_id,physical_qc_state,artifact_sha256,area_cm2,"
                    "source_snapshot_id,sample_id,bbox_xyz_json,sample_points_json,"
                    "payload_json FROM surfaces WHERE surface_id=?",
                    (surface_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"unknown surface: {surface_id}")
                result_sha256 = content_sha256(receipt or {})
                lineage = {
                    "geometry_certified_by_job_id": requested_by_job_id,
                    "surface_id": surface_id,
                    "surface_artifact_sha256": row["artifact_sha256"],
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256,
                    "result_sha256": result_sha256,
                    "result": receipt or {},
                }
                surface_payload = _load(row["payload_json"]) or {}
                surface_payload["geometry_certification_lineage"] = lineage
                connection.execute(
                    "UPDATE surfaces SET geometry_qc_state=?,payload_json=? WHERE surface_id=?",
                    (geometry_state, _dump(surface_payload), surface_id),
                )
                blocked = 0
                promotion_links: list[dict[str, str]] = []
                # The verdict promotes what was waiting on it. Without this,
                # holding a job back for unmeasured geometry would strand the
                # surface rather than gate it.
                if str(geometry_state) == "GEOMETRY_CERTIFIED":
                    waiting = connection.execute(
                        f"SELECT qc_job_id,payload_json FROM qc_jobs WHERE surface_id=? "
                        f"AND state='{QC_WAITING_GEOMETRY}' ORDER BY qc_job_id",
                        (surface_id,),
                    ).fetchall()
                    # A geometry verdict is not a size verdict and cannot
                    # overrule one. This is the exact place PHerc0268's
                    # certificate turned into a claimable job, so the routing
                    # decision is required here too -- before the row becomes
                    # PENDING, not in whatever reads it afterwards.
                    routing_receipt = (
                        self._require_routing_receipt(
                            connection,
                            {"surface_id": surface_id,
                             "area_cm2": row["area_cm2"],
                             "source_snapshot_id": row["source_snapshot_id"],
                             "sample_id": row["sample_id"],
                             "artifact_sha256": row["artifact_sha256"],
                             "bbox_xyz": _load(row["bbox_xyz_json"]),
                             "sample_points": _load(row["sample_points_json"]),
                             "geometry_qc_state": geometry_state}, now)
                        if waiting else None
                    )
                    from . import surface_routing  # noqa: PLC0415

                    releasable = (
                        routing_receipt is not None
                        and surface_routing.enters_standard_qc(routing_receipt)
                    )
                    for qc_row in waiting:
                        promotion_event_id = stable_id("geometry-qc-promotion", {
                            "geometry_job_id": requested_by_job_id,
                            "qc_job_id": qc_row["qc_job_id"],
                            "result_sha256": result_sha256,
                        })
                        qc_payload = _load(qc_row["payload_json"]) or {}
                        qc_payload.update({
                            "surface_artifact_sha256": row["artifact_sha256"],
                            "unblocked_by_job_id": requested_by_job_id,
                            "promotion_event_id": promotion_event_id,
                        })
                        if not releasable:
                            # Below the floor: the job stays durable, auditable,
                            # and unclaimable. Nothing failed and no verdict is
                            # coming; the surface is too small for the standard
                            # path and that is a fact about its size alone.
                            qc_payload["small_surface_routing_sha256"] = (
                                routing_receipt["receipt_sha256"])
                            connection.execute(
                                "UPDATE qc_jobs SET state=?,payload_json=?,"
                                "updated_at=? WHERE qc_job_id=?",
                                (QC_SMALL_SURFACE_DIAGNOSTIC, _dump(qc_payload),
                                 now, qc_row["qc_job_id"]),
                            )
                            continue
                        connection.execute(
                            "UPDATE qc_jobs SET state='PENDING',payload_json=?,updated_at=? "
                            "WHERE qc_job_id=?",
                            (_dump(qc_payload), now, qc_row["qc_job_id"]),
                        )
                        promotion_links.append({"qc_job_id": qc_row["qc_job_id"],
                                                "promotion_event_id": promotion_event_id})
                if is_geometry_rejected(geometry_state):
                    cursor = connection.execute(
                        f"""UPDATE qc_jobs SET state='FAILED',result_json=?,worker_id=NULL,
                           lease_token=NULL,lease_expires_at=NULL,retry_after=NULL,updated_at=?
                           WHERE surface_id=? AND state IN ('PENDING','{QC_WAITING_GEOMETRY}')""",
                        (
                            _dump(
                                {
                                    "schema": "campaignx.segment_qc_geometry_block.v1",
                                    "geometry_qc_state": geometry_state,
                                    "geometry_certification": receipt,
                                    "blocked_by_job_id": requested_by_job_id,
                                    "no_scientific_conclusion": True,
                                }
                            ),
                            now,
                            surface_id,
                        ),
                    )
                    blocked = int(cursor.rowcount)
                self.event(
                    connection,
                    "GEOMETRY_CERTIFICATION_RECORDED",
                    {
                        "surface_id": surface_id,
                        "geometry_qc_state": geometry_state,
                        "blocked_qc_jobs": blocked,
                        **lineage,
                        "qc_promotion_links": promotion_links,
                    },
                )
                connection.commit()
                return {
                    "surface_id": surface_id,
                    "geometry_qc_state": geometry_state,
                    "physical_qc_state": row["physical_qc_state"],
                    "blocked_qc_jobs": blocked,
                    **lineage,
                    "qc_promotion_links": promotion_links,
                    "promotion_event_id": (
                        promotion_links[0]["promotion_event_id"] if len(promotion_links) == 1 else None),
                }
            except BaseException:
                connection.rollback()
                raise

    def mark_terminal(self, task_id: str, attempt_id: str, lease_token: str, state: str, result: dict[str, Any]) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state}")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            try:
                owner = connection.execute(
                    "SELECT active_attempt_id,lease_token,mission_id,policy_version FROM tasks "
                    "WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if (
                    owner is None
                    or owner["active_attempt_id"] != attempt_id
                    or owner["lease_token"] != lease_token
                ):
                    raise RuntimeError("attempt no longer owns the task")
                connection.execute(
                    """UPDATE attempts SET state=?,result_json=?,updated_at=?
                        WHERE attempt_id=?""",
                    (state, _dump(result), now, attempt_id),
                )
                connection.execute(
                    """UPDATE tasks
                          SET state=?,worker_id=NULL,lease_token=NULL,
                              lease_expires_at=NULL,updated_at=?
                        WHERE task_id=?""",
                    (state, now, task_id),
                )
                promotion = connection.execute(
                    """SELECT p.*,d.probe_run_id
                         FROM probe_promotions p
                         JOIN probe_decisions d ON d.decision_id=p.decision_id
                        WHERE p.continuation_task_id=?
                          AND p.state='CONTINUING'""",
                    (task_id,),
                ).fetchone()
                if promotion is not None:
                    failure_receipt = {
                        **_load(promotion["receipt_json"]),
                        "state": "FAILED",
                        "final_status": state,
                    }
                    connection.execute(
                        """UPDATE probe_promotions
                              SET state='FAILED',receipt_json=?,
                                  receipt_sha256=?,updated_at=?
                            WHERE promotion_id=?""",
                        (
                            _dump(failure_receipt),
                            content_sha256(failure_receipt),
                            now,
                            promotion["promotion_id"],
                        ),
                    )
                    connection.execute(
                        "UPDATE probe_runs SET state='PROMOTION_FAILED',"
                        "updated_at=? WHERE probe_run_id=?",
                        (now, promotion["probe_run_id"]),
                    )
                    connection.execute(
                        """UPDATE probe_artifact_sets
                              SET state='REVIEW_RETAINED',retain_until=NULL
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id FROM probe_trials
                                   WHERE probe_run_id=?)""",
                        (promotion["probe_run_id"],),
                    )
                    self.event(
                        connection,
                        "PROBE_PROMOTION_FAILED",
                        {
                            "probe_run_id": promotion["probe_run_id"],
                            "promotion_id": promotion["promotion_id"],
                            "final_status": state,
                        },
                        task_id,
                        attempt_id,
                    )
                else:
                    queued = connection.execute(
                        """SELECT r.probe_run_id,r.policy_json,
                                  d.winner_trial_id
                             FROM probe_runs r
                             JOIN probe_decisions d
                               ON d.probe_run_id=r.probe_run_id
                            WHERE r.task_id=?
                              AND r.state='CONTINUATION_QUEUED'
                              AND d.action='CONTINUE_WINNER'""",
                        (task_id,),
                    ).fetchone()
                    if (
                        queued is not None
                        and (_load(queued["policy_json"]) or {}).get("mode")
                        == "select"
                    ):
                        connection.execute(
                            """UPDATE probe_runs
                                  SET state='CONTINUATION_FAILED',updated_at=?
                                WHERE probe_run_id=?""",
                            (now, queued["probe_run_id"]),
                        )
                        connection.execute(
                            """UPDATE probe_artifact_sets
                                  SET state='REVIEW_RETAINED',
                                      retain_until=NULL
                                WHERE probe_trial_id IN (
                                      SELECT probe_trial_id
                                        FROM probe_trials
                                       WHERE probe_run_id=?)""",
                            (queued["probe_run_id"],),
                        )
                        self.event(
                            connection,
                            "PROBE_CONTINUATION_FAILED",
                            {
                                "probe_run_id": queued["probe_run_id"],
                                "winner_trial_id": queued[
                                    "winner_trial_id"
                                ],
                                "final_status": state,
                                "promotion_started": False,
                            },
                            task_id,
                            attempt_id,
                        )
                controlled = any(
                    (_load(row["admission_json"]).get(
                        "execution_bindings") or {}).get("policy_version")
                    == owner["policy_version"]
                    for row in connection.execute(
                        """SELECT admission_json
                             FROM campaign_budget_admissions
                            WHERE mission_id=?""",
                        (owner["mission_id"],),
                    ).fetchall()
                )
                if controlled:
                    self._refresh_campaign_decisions(
                        connection,
                        mission_id=owner["mission_id"],
                        policy_version=owner["policy_version"],
                    )
                self.event(
                    connection,
                    f"STATE_{state}",
                    result,
                    task_id,
                    attempt_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def probe_artifacts_due_for_gc(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return only expired bytes whose run reached a safe terminal state."""

        if limit < 1 or limit > 1000:
            raise ValueError("probe GC limit must be between 1 and 1000")
        now = utc_now()
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT a.probe_artifact_set_id,a.artifact_uri,
                          a.manifest_json,a.state,a.retain_until,
                          t.probe_run_id,r.task_id
                     FROM probe_artifact_sets a
                     JOIN probe_trials t
                       ON t.probe_trial_id=a.probe_trial_id
                     JOIN probe_runs r ON r.probe_run_id=t.probe_run_id
                    WHERE a.deleted_at IS NULL
                      AND a.artifact_uri IS NOT NULL
                      AND a.retain_until IS NOT NULL
                      AND a.retain_until<=?
                      AND a.state IN
                          ('LOSER_RETAINED','PROMOTED_RETAINED')
                      AND r.state IN ('PROMOTED','REJECTED')
                      AND NOT EXISTS (
                          SELECT 1 FROM probe_promotions p
                          JOIN probe_decisions d
                            ON d.decision_id=p.decision_id
                         WHERE d.probe_run_id=r.probe_run_id
                           AND p.state='CONTINUING')
                    ORDER BY a.retain_until,a.probe_artifact_set_id
                    LIMIT ?""",
                (now, limit),
            ).fetchall()
        return [
            {
                "probe_artifact_set_id": row["probe_artifact_set_id"],
                "artifact_uri": row["artifact_uri"],
                "manifest": _load(row["manifest_json"]),
                "state": row["state"],
                "retain_until": row["retain_until"],
                "probe_run_id": row["probe_run_id"],
                "task_id": row["task_id"],
            }
            for row in rows
        ]

    def mark_probe_artifact_expired(
        self, probe_artifact_set_id: str, artifact_uri: str
    ) -> None:
        """Close the digest ledger after exact external bytes were deleted."""

        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """SELECT a.*,t.probe_run_id,r.task_id,r.state AS run_state
                         FROM probe_artifact_sets a
                         JOIN probe_trials t
                           ON t.probe_trial_id=a.probe_trial_id
                         JOIN probe_runs r ON r.probe_run_id=t.probe_run_id
                        WHERE a.probe_artifact_set_id=?""",
                    (probe_artifact_set_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(probe_artifact_set_id)
                if row["deleted_at"] is not None:
                    connection.commit()
                    return
                if (
                    row["artifact_uri"] != artifact_uri
                    or row["state"]
                    not in {"LOSER_RETAINED", "PROMOTED_RETAINED"}
                    or row["retain_until"] is None
                    or row["retain_until"] > now
                    or row["run_state"] not in {"PROMOTED", "REJECTED"}
                ):
                    raise RuntimeError(
                        "probe artifact is not eligible for irreversible expiry"
                    )
                connection.execute(
                    """UPDATE probe_artifact_sets
                          SET state='EXPIRED',deleted_at=?
                        WHERE probe_artifact_set_id=?""",
                    (now, probe_artifact_set_id),
                )
                self.event(
                    connection,
                    "PROBE_ARTIFACT_EXPIRED",
                    {
                        "probe_run_id": row["probe_run_id"],
                        "probe_artifact_set_id": probe_artifact_set_id,
                        "artifact_uri": artifact_uri,
                    },
                    row["task_id"],
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    # -- the candidate preflight queue --------------------------------------

    def enqueue_candidate_preflight(self, request: dict[str, Any]) -> dict[str, Any]:
        """Queue one preflight, idempotently on what was asked.

        The digest is the identity: the panel client refuses to retry a mutation
        whose answer it could not read and tells the caller to read state
        instead, which is only safe if enqueuing the same request twice is the
        same job rather than two.
        """
        for field in ("mission_id", "sample_id", "source_snapshot_id"):
            if not str(request.get(field) or "").strip():
                raise ValueError(f"a preflight request needs {field}")
        digest = content_sha256(request)
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # A FAILED job is a record of an attempt, not the answer to the
                # next ask. Excluding it here is what lets a frozen request be
                # measured again after a transient outage.
                existing = connection.execute(
                    """SELECT preflight_job_id,state FROM preflight_jobs
                        WHERE mission_id=? AND sample_id=? AND request_sha256=?
                          AND state<>'FAILED'""",
                    (request["mission_id"], request["sample_id"], digest),
                ).fetchone()
                attempt_ordinal = connection.execute(
                    """SELECT count(*) AS n FROM preflight_jobs
                        WHERE mission_id=? AND sample_id=? AND request_sha256=?""",
                    (request["mission_id"], request["sample_id"], digest),
                ).fetchone()["n"]
                job_id = stable_id("preflight-job", {
                    "mission_id": request["mission_id"],
                    "sample_id": request["sample_id"],
                    "request_sha256": digest,
                    "attempt_ordinal": attempt_ordinal,
                })
                if existing is not None:
                    connection.commit()
                    return {"preflight_job_id": existing["preflight_job_id"],
                            "state": existing["state"], "created": False}
                connection.execute(
                    """INSERT INTO preflight_jobs(preflight_job_id,mission_id,sample_id,
                       source_snapshot_id,state,request_json,request_sha256,
                       created_at,updated_at)
                       VALUES(?,?,?,?,'PENDING',?,?,?,?)""",
                    (job_id, request["mission_id"], request["sample_id"],
                     request["source_snapshot_id"], _dump(request), digest, now, now),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {"preflight_job_id": job_id, "state": "PENDING", "created": True}

    def claim_preflight(self, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        """Take one pending preflight, expiring abandoned leases first."""
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        token = secrets.token_urlsafe(32)
        expires = _deadline(lease_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """UPDATE preflight_jobs SET state='PENDING',worker_id=NULL,
                       lease_token=NULL,lease_expires_at=NULL,updated_at=?
                       WHERE state='CLAIMED' AND lease_expires_at IS NOT NULL
                         AND lease_expires_at<=?""",
                    (now, now),
                )
                row = connection.execute(
                    """SELECT * FROM preflight_jobs WHERE state='PENDING'
                         AND (retry_after IS NULL OR retry_after<=?)
                        ORDER BY created_at,rowid LIMIT 1""",
                    (now,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                connection.execute(
                    """UPDATE preflight_jobs SET state='CLAIMED',worker_id=?,
                       lease_token=?,lease_expires_at=?,attempts=attempts+1,updated_at=?
                       WHERE preflight_job_id=?""",
                    (worker_id, token, expires, now, row["preflight_job_id"]),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {
            "preflight_job_id": row["preflight_job_id"],
            "mission_id": row["mission_id"],
            "sample_id": row["sample_id"],
            "source_snapshot_id": row["source_snapshot_id"],
            "request": json.loads(row["request_json"]),
            "lease_token": token,
            "lease_expires_at": expires,
        }

    def _preflight_owner(self, connection, preflight_job_id: str,
                         lease_token: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT * FROM preflight_jobs WHERE preflight_job_id=?",
            (preflight_job_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("no such preflight job")
        if row["state"] != "CLAIMED" or row["lease_token"] != lease_token:
            raise RuntimeError("this preflight job is not held by that lease")
        return row

    def heartbeat_preflight(self, preflight_job_id: str, lease_token: str,
                            lease_seconds: int) -> dict[str, Any]:
        """Extend a lease, and answer whether it is still this worker's.

        The only contract method that takes the token, so it is the only one
        that can answer "is this still mine" -- which is why the worker calls it
        before its I/O and not only from a background thread.
        """
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        expires = _deadline(lease_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._preflight_owner(connection, preflight_job_id, lease_token)
                connection.execute(
                    """UPDATE preflight_jobs SET lease_expires_at=?,updated_at=?
                        WHERE preflight_job_id=?""",
                    (expires, now, preflight_job_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {"preflight_job_id": preflight_job_id, "lease_expires_at": expires}

    def finalize_preflight(self, preflight_job_id: str, lease_token: str,
                           receipt: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._preflight_owner(connection, preflight_job_id, lease_token)
                connection.execute(
                    """UPDATE preflight_jobs SET state='COMPLETED',receipt_json=?,
                       lease_token=NULL,lease_expires_at=NULL,updated_at=?
                        WHERE preflight_job_id=?""",
                    (_dump(receipt), now, preflight_job_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {"preflight_job_id": preflight_job_id, "state": "COMPLETED"}

    def fail_preflight(self, preflight_job_id: str, lease_token: str,
                       reason_code: str, detail: str | None = None) -> dict[str, Any]:
        """Terminal, with the reason and the sentence behind it.

        `detail` is the worker's redacted account. Without it an operator reading
        FAILED has to go to a worker's stdout, and a reason that lives only there
        is not evidence.
        """
        if not str(reason_code or "").strip():
            raise ValueError("a failed preflight needs a reason code")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._preflight_owner(connection, preflight_job_id, lease_token)
                connection.execute(
                    """UPDATE preflight_jobs SET state='FAILED',reason_code=?,detail=?,
                       lease_token=NULL,lease_expires_at=NULL,updated_at=?
                        WHERE preflight_job_id=?""",
                    (reason_code, detail, now, preflight_job_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {"preflight_job_id": preflight_job_id, "state": "FAILED",
                "reason_code": reason_code}

    def requeue_preflight_source_unavailable(
        self,
        preflight_job_id: str,
        lease_token: str,
        receipt: dict[str, Any],
        *,
        retry_delay_seconds: int,
        maximum_requeues: int,
    ) -> dict[str, Any]:
        """Send a preflight back to the queue after a source outage.

        The worker already classifies a source failure as recoverable and then
        had only `fail_preflight` to call, so one dropped connection ended a
        measurement that had been running for over an hour. This is the same
        shape the segmentation lane already uses for the same situation:
        bounded by `maximum_requeues`, delayed by `retry_after`, and terminal
        once the budget is spent -- because a source that is genuinely gone has
        to surface as gone rather than hide behind an endless retry.
        """
        if retry_delay_seconds < 0:
            raise ValueError("retry delay must be non-negative")
        if maximum_requeues < 0:
            raise ValueError("maximum_requeues must be non-negative")
        detail = _safe_outage_detail(receipt)
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._preflight_owner(connection, preflight_job_id, lease_token)
                spent = int(row["requeues"] or 0) >= maximum_requeues
                if spent:
                    # The budget is gone. Terminal, with the outage as the
                    # reason, so the run reports what actually stopped it.
                    connection.execute(
                        """UPDATE preflight_jobs SET state='FAILED',
                           reason_code='PREFLIGHT_SOURCE_UNAVAILABLE',detail=?,
                           receipt_json=?,worker_id=NULL,lease_token=NULL,
                           lease_expires_at=NULL,updated_at=?
                            WHERE preflight_job_id=?""",
                        (detail, _dump(receipt), now, preflight_job_id),
                    )
                    self.event(
                        connection,
                        "PREFLIGHT_SOURCE_UNAVAILABLE_EXHAUSTED",
                        {
                            "preflight_job_id": preflight_job_id,
                            "sample_id": row["sample_id"],
                            "requeues": int(row["requeues"] or 0),
                            "detail": detail,
                        },
                    )
                    connection.commit()
                    return {
                        "status": "PREFLIGHT_SOURCE_UNAVAILABLE",
                        "preflight_job_id": preflight_job_id,
                        "state": "FAILED",
                        "reason_code": "PREFLIGHT_SOURCE_UNAVAILABLE",
                        "requeues": int(row["requeues"] or 0),
                    }
                retry_after = _deadline(retry_delay_seconds)
                connection.execute(
                    """UPDATE preflight_jobs SET state='PENDING',detail=?,
                       receipt_json=?,worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,retry_after=?,requeues=requeues+1,
                       updated_at=? WHERE preflight_job_id=?""",
                    (detail, _dump(receipt), retry_after, now, preflight_job_id),
                )
                self.event(
                    connection,
                    "PREFLIGHT_REQUEUED_SOURCE_UNAVAILABLE",
                    {
                        "preflight_job_id": preflight_job_id,
                        "sample_id": row["sample_id"],
                        "retry_after": retry_after,
                        "requeues": int(row["requeues"] or 0) + 1,
                        "detail": detail,
                    },
                )
                connection.commit()
                return {
                    "status": "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE",
                    "preflight_job_id": preflight_job_id,
                    "state": "PENDING",
                    "retry_after": retry_after,
                    "requeues": int(row["requeues"] or 0) + 1,
                }
            except BaseException:
                connection.rollback()
                raise

    def preflight_job(self, preflight_job_id: str) -> dict[str, Any] | None:
        """One job, for whoever is polling it."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM preflight_jobs WHERE preflight_job_id=?",
                (preflight_job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "preflight_job_id": row["preflight_job_id"],
            "mission_id": row["mission_id"],
            "sample_id": row["sample_id"],
            "source_snapshot_id": row["source_snapshot_id"],
            "state": row["state"],
            "request": json.loads(row["request_json"]),
            "receipt": json.loads(row["receipt_json"]) if row["receipt_json"] else None,
            "reason_code": row["reason_code"],
            "detail": row["detail"],
            "attempts": int(row["attempts"] or 0),
            # An operator polling this has to be able to see that the job has
            # already been through an outage, not only its latest state.
            "requeues": int(row["requeues"] or 0),
            "retry_after": row["retry_after"],
        }

    def defer_qc_jobs(self, sample_id: str, *, until: str | None,
                      reason: str, by: str) -> dict[str, Any]:
        """Hold one sample's pending QC so the GPUs take something else first.

        Not a reordering and not a new state: `claim_qc` already skips a job
        whose ``retry_after`` is in the future, and the job stays PENDING with
        its history, so this is the queue's own lever. Only PENDING rows are
        touched -- a worker holding a lease finishes what it started.

        Bounded and attributed on purpose. An unbounded hold is a delete with
        better manners, and which hour a GPU spends on whose scroll is a
        decision somebody has to be able to read afterwards.
        """
        if not str(reason or "").strip():
            raise ValueError("deferring QC needs a reason")
        if not str(until or "").strip():
            raise ValueError("deferring QC needs a time it ends")
        deadline = _instant(str(until))
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """UPDATE qc_jobs SET retry_after=?,updated_at=?
                        WHERE state='PENDING' AND surface_id IN (
                          SELECT surface_id FROM surfaces WHERE sample_id=?)""",
                    (deadline, now, sample_id))
                record = {"sample_id": sample_id, "deferred": cursor.rowcount,
                          "until": deadline, "reason": str(reason).strip(), "by": by}
                connection.execute(
                    """INSERT INTO events(event_type,payload_json,created_at)
                       VALUES('qc.deferred',?,?)""",
                    (_dump(record), now))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return record

    def release_qc_jobs(self, sample_id: str, *, by: str) -> dict[str, Any]:
        """Take a deferred sample back up. The jobs never left the queue."""
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """UPDATE qc_jobs SET retry_after=NULL,updated_at=?
                        WHERE state='PENDING' AND retry_after IS NOT NULL
                          AND surface_id IN (
                            SELECT surface_id FROM surfaces WHERE sample_id=?)""",
                    (now, sample_id))
                record = {"sample_id": sample_id, "released": cursor.rowcount, "by": by}
                connection.execute(
                    """INSERT INTO events(event_type,payload_json,created_at)
                       VALUES('qc.released',?,?)""",
                    (_dump(record), now))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return record

    def requeue_blocked_qc_jobs(self, sample_id: str, *, fixed: str,
                                by: str) -> dict[str, Any]:
        """Put a sample's configuration-blocked QC back in the queue, once.

        The other half of `block_qc_configuration`. Blocking is terminal because
        a job that fails the same way every time must not be reclaimed; but the
        thing it waits for -- a profile pin, a checkpoint -- does get fixed, and
        until this existed the only way back was an UPDATE typed into psql.

        Nothing calls this but a person: `claim_qc` still takes only PENDING and
        no timer or sweep moves a blocked row, so the fleet cannot resume
        spinning on its own. What changed is that somebody can say so, once,
        having stated what they fixed -- a terminal state that gets undone
        without a reason on the record is not terminal, it is unreliable.

        The stale receipt goes with the state. A job's reported `error` and
        `last_status` are composed from `result`, so keeping the blocked
        attempt's receipt would show a corrected deployment still quoting the
        hash it no longer pins, which is how a fixed control plane reads as
        unfixed. The receipt is not lost: QC_BLOCKED_CONFIGURATION holds it, and
        this requeue is recorded beside it.
        """
        if not str(fixed or "").strip():
            raise ValueError("requeueing blocked QC needs what was fixed")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Read the ids inside the same transaction that clears them, so
                # the record names the jobs this call moved rather than whatever
                # was blocked a moment earlier.
                requeued = sorted(row["qc_job_id"] for row in connection.execute(
                    """SELECT qc_job_id FROM qc_jobs
                        WHERE state='BLOCKED_CONFIGURATION' AND surface_id IN (
                          SELECT surface_id FROM surfaces WHERE sample_id=?)""",
                    (sample_id,)).fetchall())
                connection.execute(
                    """UPDATE qc_jobs SET state='PENDING',result_json=NULL,
                       worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,
                       retry_after=NULL,updated_at=?
                        WHERE state='BLOCKED_CONFIGURATION' AND surface_id IN (
                          SELECT surface_id FROM surfaces WHERE sample_id=?)""",
                    (now, sample_id))
                record = {"sample_id": sample_id, "requeued": len(requeued),
                          "qc_job_ids": requeued, "fixed": str(fixed).strip(),
                          "by": by}
                connection.execute(
                    """INSERT INTO events(event_type,payload_json,created_at)
                       VALUES('qc.requeued_after_fix',?,?)""",
                    (_dump(record), now))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return record

    def claim_qc(self, worker_id: str, lease_seconds: int, profile_id: str | None = None) -> dict[str, Any] | None:
        """Atomically claim one pending surface QC job."""
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        token = secrets.token_urlsafe(32)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """UPDATE qc_jobs SET state='PENDING',worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,updated_at=?
                       WHERE state='CLAIMED' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?""",
                    (now, now),
                )
                query = "SELECT * FROM qc_jobs WHERE state='PENDING' AND (retry_after IS NULL OR retry_after<=?)"
                arguments: list[Any] = [now]
                if profile_id is not None:
                    query += " AND profile_id=?"
                    arguments.append(profile_id)
                query += " ORDER BY created_at,qc_job_id LIMIT 1"
                row = connection.execute(query, arguments).fetchone()
                if row is None:
                    connection.commit()
                    return None
                surface = connection.execute(
                    "SELECT * FROM surfaces WHERE surface_id=?",
                    (row["surface_id"],),
                ).fetchone()
                if surface is None:
                    raise RuntimeError("QC job has no authoritative surface")
                surface_value = _load(surface["payload_json"]) or {}
                self._require_surface_payload_lineage(
                    surface_value, boundary="PHYSICAL_QC_CLAIM_RESOLUTION"
                )
                source = connection.execute(
                    "SELECT * FROM source_snapshots WHERE source_snapshot_id=?",
                    (surface["source_snapshot_id"],),
                ).fetchone()
                cursor = connection.execute(
                    """UPDATE qc_jobs SET state='CLAIMED',worker_id=?,lease_token=?,lease_expires_at=?,
                       retry_after=NULL,updated_at=? WHERE qc_job_id=? AND state='PENDING'""",
                    (worker_id, token, _deadline(lease_seconds), now, row["qc_job_id"]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("QC claim lost its atomic update")
                self.event(connection, "QC_CLAIMED", {"qc_job_id": row["qc_job_id"], "surface_id": row["surface_id"], "worker_id": worker_id})
                connection.commit()
                surface_value.update({
                    "surface_id": surface["surface_id"],
                    "artifact_uri": surface["artifact_uri"],
                    "artifact_sha256": surface["artifact_sha256"],
                    "bbox_xyz": _load(surface["bbox_xyz_json"]),
                    "state": surface["state"],
                })
                return {
                    "qc_job_id": row["qc_job_id"],
                    "surface_id": row["surface_id"],
                    "profile_id": row["profile_id"],
                    "payload": _load(row["payload_json"]),
                    "worker_id": worker_id,
                    "lease_token": token,
                    "surface": surface_value,
                    "source": self._snapshot(source),
                }
            except BaseException:
                connection.rollback()
                raise

    def heartbeat_qc(self, qc_job_id: str, lease_token: str, lease_seconds: int) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE qc_jobs SET lease_expires_at=?,updated_at=?
                   WHERE qc_job_id=? AND state='CLAIMED' AND lease_token=?
                     AND lease_expires_at IS NOT NULL AND lease_expires_at>?""",
                (_deadline(lease_seconds), now, qc_job_id, lease_token, now),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("QC lease heartbeat rejected")

    def finalize_qc(self, qc_job_id: str, lease_token: str, outcome: str, result: dict[str, Any]) -> dict[str, Any]:
        if outcome not in QC_OUTCOME_STATES:
            raise ValueError(f"unsupported QC outcome: {outcome}")
        now = utc_now()
        surface_state, physical_state = QC_OUTCOME_STATES[outcome]
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                job = connection.execute("SELECT * FROM qc_jobs WHERE qc_job_id=?", (qc_job_id,)).fetchone()
                if (
                    job is None
                    or job["state"] != "CLAIMED"
                    or job["lease_token"] != lease_token
                    or job["lease_expires_at"] is None
                    or job["lease_expires_at"] <= now
                ):
                    raise RuntimeError("QC finalization belongs to a stale lease")
                validate_qc_result_contract(job["surface_id"], outcome, result)
                stored_result = {**result, "result_sha256": content_sha256(result)}
                connection.execute(
                    """UPDATE qc_jobs SET state='COMPLETED',result_json=?,worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,retry_after=NULL,updated_at=? WHERE qc_job_id=?""",
                    (_dump(stored_result), now, qc_job_id),
                )
                connection.execute(
                    "UPDATE surfaces SET state=?,physical_qc_state=? WHERE surface_id=?",
                    (surface_state, physical_state, job["surface_id"]),
                )
                self.event(connection, "QC_COMPLETED", {"qc_job_id": qc_job_id, "surface_id": job["surface_id"], "outcome": outcome, "evidence_manifest_sha256": result["evidence_manifest_sha256"]})
                connection.commit()
                return {"status": "COMPLETED", "qc_job_id": qc_job_id, "surface_id": job["surface_id"], "outcome": outcome, "surface_state": surface_state, "physical_qc_state": physical_state}
            except BaseException:
                connection.rollback()
                raise

    def block_qc_configuration(
        self,
        qc_job_id: str,
        lease_token: str,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        """Stop a QC job that cannot succeed until somebody changes something.

        Terminal, and deliberately not a scientific verdict. The surface is left
        exactly as it was: a wrong profile hash or a missing checkpoint says
        nothing about the papyrus, and writing a physical_qc_state here would
        turn an operator's mistake into a measurement.

        claim_qc takes only PENDING, so BLOCKED_CONFIGURATION is not picked up
        again. That is the whole point -- the previous behaviour requeued it and
        the fleet spun on it for two days.
        """
        from framework.contracts import qc_diagnostics

        safe_receipt = qc_diagnostics.receipt_with_safe_error(
            receipt, "the configuration error had no safe detail"
        )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                job = connection.execute(
                    "SELECT * FROM qc_jobs WHERE qc_job_id=?", (qc_job_id,)
                ).fetchone()
                if (
                    job is None
                    or job["state"] != "CLAIMED"
                    or job["lease_token"] != lease_token
                    or job["lease_expires_at"] is None
                    or job["lease_expires_at"] <= now
                ):
                    raise RuntimeError("QC block belongs to a stale lease")
                connection.execute(
                    """UPDATE qc_jobs SET state='BLOCKED_CONFIGURATION',result_json=?,
                       worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,
                       retry_after=NULL,updated_at=? WHERE qc_job_id=?""",
                    (_dump(safe_receipt), now, qc_job_id),
                )
                self.event(
                    connection,
                    "QC_BLOCKED_CONFIGURATION",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "error": safe_receipt["error"],
                    },
                )
                connection.commit()
                return {
                    "status": "BLOCKED_CONFIGURATION",
                    "qc_job_id": qc_job_id,
                    "surface_id": job["surface_id"],
                    "error": safe_receipt["error"],
                }
            except BaseException:
                connection.rollback()
                raise

    def requeue_qc_unavailable(
        self,
        qc_job_id: str,
        lease_token: str,
        receipt: dict[str, Any],
        *,
        retry_delay_seconds: int,
    ) -> dict[str, Any]:
        """Release a QC lease after an operational outage without losing the job."""
        if retry_delay_seconds < 0:
            raise ValueError("retry delay must be non-negative")
        from framework.contracts import qc_diagnostics

        safe_receipt = qc_diagnostics.receipt_with_safe_error(
            receipt, "RuntimeError: retryable QC failure had no safe detail"
        )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                job = connection.execute(
                    "SELECT * FROM qc_jobs WHERE qc_job_id=?", (qc_job_id,)
                ).fetchone()
                if (
                    job is None
                    or job["state"] != "CLAIMED"
                    or job["lease_token"] != lease_token
                    or job["lease_expires_at"] is None
                    or job["lease_expires_at"] <= now
                ):
                    raise RuntimeError("QC requeue belongs to a stale lease")
                retry_after = _deadline(retry_delay_seconds)
                connection.execute(
                    """UPDATE qc_jobs SET state='PENDING',result_json=?,worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,retry_after=?,updated_at=? WHERE qc_job_id=?""",
                    (_dump(safe_receipt), retry_after, now, qc_job_id),
                )
                self.event(
                    connection,
                    "QC_REQUEUED_UNAVAILABLE",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "retry_after": retry_after,
                        "error": safe_receipt["error"],
                    },
                )
                connection.commit()
                return {
                    "status": "RETRYABLE_QC_UNAVAILABLE",
                    "qc_job_id": qc_job_id,
                    "surface_id": job["surface_id"],
                    "retry_after": retry_after,
                }
            except BaseException:
                connection.rollback()
                raise

    def _requeue_operational_unavailable(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        retry_delay_seconds: int,
        kind: str,
        maximum_requeues: int,
    ) -> bool:
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        if maximum_requeues < 0:
            raise ValueError("maximum_requeues must be non-negative")
        if kind not in {"PROBE_ARTIFACT", "FINALIZATION"}:
            raise ValueError("unsupported operational retry kind")
        now = utc_now()
        retry_after = _deadline(retry_delay_seconds)
        attempt_state = f"RETRYABLE_{kind}_UNAVAILABLE"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute(
                    """SELECT active_attempt_id,lease_token
                         FROM tasks WHERE task_id=?""",
                    (task_id,),
                ).fetchone()
                if (
                    owner is None
                    or owner["active_attempt_id"] != attempt_id
                    or owner["lease_token"] != lease_token
                ):
                    raise RuntimeError("attempt no longer owns the task")
                prior_requeues = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM attempts
                            WHERE task_id=? AND state=?""",
                        (task_id, attempt_state),
                    ).fetchone()[0]
                )
                if prior_requeues >= maximum_requeues:
                    connection.commit()
                    return False
                connection.execute(
                    """UPDATE attempts SET state=?,result_json=?,updated_at=?
                        WHERE attempt_id=?""",
                    (attempt_state, _dump(result), now, attempt_id),
                )
                connection.execute(
                    """UPDATE tasks
                          SET state='PENDING',worker_id=NULL,lease_token=NULL,
                              lease_expires_at=NULL,retry_after=?,
                              active_attempt_id=NULL,updated_at=?
                        WHERE task_id=?""",
                    (retry_after, now, task_id),
                )
                self.event(
                    connection,
                    f"STATE_{attempt_state}",
                    {**result, "retry_after": retry_after},
                    task_id,
                    attempt_id,
                )
                connection.commit()
                return True
            except BaseException:
                connection.rollback()
                raise

    def requeue_probe_artifact_unavailable(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        retry_delay_seconds: int,
        maximum_requeues: int = 5,
    ) -> bool:
        return self._requeue_operational_unavailable(
            task_id,
            attempt_id,
            lease_token,
            result,
            retry_delay_seconds=retry_delay_seconds,
            kind="PROBE_ARTIFACT",
            maximum_requeues=maximum_requeues,
        )

    def requeue_finalization_unavailable(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        retry_delay_seconds: int,
        maximum_requeues: int = 2,
    ) -> bool:
        return self._requeue_operational_unavailable(
            task_id,
            attempt_id,
            lease_token,
            result,
            retry_delay_seconds=retry_delay_seconds,
            kind="FINALIZATION",
            maximum_requeues=maximum_requeues,
        )

    def requeue_provider_unavailable(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        retry_delay_seconds: int,
    ) -> None:
        """Preserve a failed provider attempt, then release only its task.

        The retry time is task-local.  A watch worker can therefore keep
        processing other cells instead of hammering a temporarily saturated
        provider or permanently discarding a valid geometry opportunity.
        """
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        now = utc_now()
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute(
                    "SELECT active_attempt_id,lease_token FROM tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if owner is None or owner["active_attempt_id"] != attempt_id or owner["lease_token"] != lease_token:
                    raise RuntimeError("attempt no longer owns the task")
                connection.execute(
                    "UPDATE attempts SET state='RETRYABLE_PROVIDER_UNAVAILABLE',result_json=?,updated_at=? WHERE attempt_id=?",
                    (_dump(result), now, attempt_id),
                )
                connection.execute(
                    "UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,retry_after=?,active_attempt_id=NULL,updated_at=? WHERE task_id=?",
                    (retry_after, now, task_id),
                )
                self.event(
                    connection,
                    "STATE_RETRYABLE_PROVIDER_UNAVAILABLE",
                    {**result, "retry_after": retry_after},
                    task_id,
                    attempt_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def requeue_for_larger_gpu(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        minimum_vram_gb: float,
    ) -> None:
        """Preserve a GPU OOM and make the task ineligible to that worker.

        This is not a geometry failure.  The attempt remains auditable while
        the task returns to PENDING with a monotonically stronger VRAM floor.
        """

        if minimum_vram_gb <= 0:
            raise ValueError("larger-GPU retry requires a positive VRAM floor")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute(
                    "SELECT active_attempt_id,lease_token,minimum_vram_gb FROM tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if owner is None or owner["active_attempt_id"] != attempt_id or owner["lease_token"] != lease_token:
                    raise RuntimeError("attempt no longer owns the task")
                required = max(float(owner["minimum_vram_gb"]), float(minimum_vram_gb))
                receipt = {**result, "minimum_vram_gb": required}
                connection.execute(
                    "UPDATE attempts SET state='RETRY_ON_LARGER_GPU',result_json=?,updated_at=? WHERE attempt_id=?",
                    (_dump(receipt), now, attempt_id),
                )
                connection.execute(
                    """UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,retry_after=NULL,active_attempt_id=NULL,
                       gpu_required=1,minimum_vram_gb=?,updated_at=? WHERE task_id=?""",
                    (required, now, task_id),
                )
                self.event(
                    connection,
                    "STATE_RETRY_ON_LARGER_GPU",
                    receipt,
                    task_id,
                    attempt_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def requeue_source_unavailable(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        retry_delay_seconds: int,
    ) -> None:
        """Preserve a transient source outage and release only its task.

        A 5xx response from VC3D/MCP precedes candidate discovery.  Treating
        it as a terminal geometry outcome would silently reduce the search
        area whenever the source service is momentarily unavailable.
        """
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        now = utc_now()
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute(
                    "SELECT active_attempt_id,lease_token FROM tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if owner is None or owner["active_attempt_id"] != attempt_id or owner["lease_token"] != lease_token:
                    raise RuntimeError("attempt no longer owns the task")
                connection.execute(
                    "UPDATE attempts SET state='RETRYABLE_SOURCE_UNAVAILABLE',result_json=?,updated_at=? WHERE attempt_id=?",
                    (_dump(result), now, attempt_id),
                )
                connection.execute(
                    "UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,retry_after=?,active_attempt_id=NULL,updated_at=? WHERE task_id=?",
                    (retry_after, now, task_id),
                )
                self.event(
                    connection,
                    "STATE_RETRYABLE_SOURCE_UNAVAILABLE",
                    {**result, "retry_after": retry_after},
                    task_id,
                    attempt_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def recover_terminal_provider_outage(
        self,
        task_id: str,
        attempt_id: str,
        *,
        retry_delay_seconds: int,
    ) -> dict[str, Any]:
        """Recover one historically misclassified OpenCode outage.

        Early workers classified an upstream model outage as
        ``POLICY_REJECTED``. This narrowly scoped recovery refuses every other
        terminal result and verifies its retained error string before returning
        the same task to PENDING. The original attempt stays in the audit trail
        under a distinct recovered state rather than being deleted or replaced.
        """
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        now = utc_now()
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT state,active_attempt_id FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                attempt = connection.execute(
                    "SELECT state,result_json FROM attempts WHERE attempt_id=? AND task_id=?", (attempt_id, task_id)
                ).fetchone()
                if task is None or attempt is None:
                    raise RuntimeError("task or attempt does not exist")
                if task["state"] != "POLICY_REJECTED" or attempt["state"] != "POLICY_REJECTED":
                    raise RuntimeError("only a terminal POLICY_REJECTED attempt may be recovered")
                prior = _load(attempt["result_json"]) if attempt["result_json"] else {}
                error = str(prior.get("error", ""))
                if "PlannerProviderUnavailable" not in error:
                    raise RuntimeError("terminal attempt is not a recorded transient planner outage")
                recovered = {
                    **prior,
                    "status": "RECOVERED_PROVIDER_UNAVAILABLE",
                    "recovered_at_utc": now,
                    "retry_after": retry_after,
                    "recovery_reason": "historical worker classified PlannerProviderUnavailable as POLICY_REJECTED",
                }
                connection.execute(
                    "UPDATE attempts SET state='RECOVERED_PROVIDER_UNAVAILABLE',result_json=?,updated_at=? WHERE attempt_id=?",
                    (_dump(recovered), now, attempt_id),
                )
                connection.execute(
                    "UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,retry_after=?,active_attempt_id=NULL,updated_at=? WHERE task_id=?",
                    (retry_after, now, task_id),
                )
                self.event(connection, "STATE_RECOVERED_PROVIDER_UNAVAILABLE", recovered, task_id, attempt_id)
                connection.commit()
                return recovered
            except BaseException:
                connection.rollback()
                raise

    def recover_terminal_mcp_auth_outage(
        self,
        task_id: str,
        attempt_id: str,
        *,
        retry_delay_seconds: int,
    ) -> dict[str, Any]:
        """Recover one terminal loopback-MCP token mismatch, and nothing else.

        A stale MCP listener using another runtime token can return HTTP 401
        before candidate discovery begins. The original attempt remains in the
        audit trail; only this exact operational receipt may return to PENDING.
        """
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        now = utc_now()
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT state FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                attempt = connection.execute(
                    "SELECT state,result_json FROM attempts WHERE attempt_id=? AND task_id=?", (attempt_id, task_id)
                ).fetchone()
                if task is None or attempt is None:
                    raise RuntimeError("task or attempt does not exist")
                if task["state"] != "BLOCKED_SOURCE_UNAVAILABLE" or attempt["state"] != "BLOCKED_SOURCE_UNAVAILABLE":
                    raise RuntimeError("only a terminal BLOCKED_SOURCE_UNAVAILABLE attempt may be recovered")
                prior = _load(attempt["result_json"]) if attempt["result_json"] else {}
                if prior.get("status") != "BLOCKED_SOURCE_UNAVAILABLE" or prior.get("error") != "HTTPError: HTTP Error 401: Unauthorized":
                    raise RuntimeError("terminal attempt is not the exact loopback MCP authentication outage")
                recovered = {
                    **prior,
                    "status": "RECOVERED_MCP_AUTH_OUTAGE",
                    "recovered_at_utc": now,
                    "retry_after": retry_after,
                    "recovery_reason": "duplicate loopback MCP listener used a different runtime token",
                }
                connection.execute(
                    "UPDATE attempts SET state='RECOVERED_MCP_AUTH_OUTAGE',result_json=?,updated_at=? WHERE attempt_id=?",
                    (_dump(recovered), now, attempt_id),
                )
                connection.execute(
                    "UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,retry_after=?,active_attempt_id=NULL,updated_at=? WHERE task_id=?",
                    (retry_after, now, task_id),
                )
                self.event(connection, "STATE_RECOVERED_MCP_AUTH_OUTAGE", recovered, task_id, attempt_id)
                connection.commit()
                return recovered
            except BaseException:
                connection.rollback()
                raise

    def recover_terminal_finalizer_dependency(
        self,
        task_id: str,
        attempt_id: str,
        *,
        retry_delay_seconds: int,
    ) -> dict[str, Any]:
        """Recover only a grow lost to a missing pinned finalizer module."""
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        now = utc_now()
        retry_after = _deadline(retry_delay_seconds)
        allowed_errors = {
            "ModuleNotFoundError: No module named 'numpy'",
            "ModuleNotFoundError: No module named 'tifffile'",
        }
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT state FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                attempt = connection.execute(
                    "SELECT state,result_json FROM attempts WHERE attempt_id=? AND task_id=?", (attempt_id, task_id)
                ).fetchone()
                if task is None or attempt is None:
                    raise RuntimeError("task or attempt does not exist")
                if task["state"] != "FINALIZATION_FAILED" or attempt["state"] != "FINALIZATION_FAILED":
                    raise RuntimeError("only a terminal FINALIZATION_FAILED attempt may be recovered")
                prior = _load(attempt["result_json"]) if attempt["result_json"] else {}
                if prior.get("status") != "FINALIZATION_FAILED" or prior.get("error") not in allowed_errors:
                    raise RuntimeError("terminal attempt is not an exact missing finalizer dependency")
                promotion = connection.execute(
                    """SELECT p.*,d.probe_run_id
                         FROM probe_promotions p
                         JOIN probe_decisions d
                           ON d.decision_id=p.decision_id
                        WHERE p.continuation_task_id=?""",
                    (task_id,),
                ).fetchone()
                if promotion is not None:
                    failed_receipt = _load(promotion["receipt_json"]) or {}
                    if (
                        promotion["state"] != "FAILED"
                        or failed_receipt.get("state") != "FAILED"
                        or failed_receipt.get("final_status")
                        != "FINALIZATION_FAILED"
                    ):
                        raise RuntimeError(
                            "probe promotion is not the matching failed finalization"
                        )
                    reopened_receipt = {
                        key: value
                        for key, value in failed_receipt.items()
                        if key != "final_status"
                    }
                    reopened_receipt["state"] = "CONTINUING"
                    connection.execute(
                        """UPDATE probe_promotions
                              SET state='CONTINUING',receipt_json=?,
                                  receipt_sha256=?,updated_at=?
                            WHERE promotion_id=?""",
                        (
                            _dump(reopened_receipt),
                            content_sha256(reopened_receipt),
                            now,
                            promotion["promotion_id"],
                        ),
                    )
                    connection.execute(
                        """UPDATE probe_runs SET state='CONTINUING',
                                  updated_at=? WHERE probe_run_id=?""",
                        (now, promotion["probe_run_id"]),
                    )
                    connection.execute(
                        """UPDATE probe_artifact_sets
                              SET state='WINNER_RETAINED',retain_until=NULL
                            WHERE probe_artifact_set_id=?""",
                        (promotion["winner_probe_artifact_set_id"],),
                    )
                    self.event(
                        connection,
                        "PROBE_PROMOTION_RECOVERED",
                        {
                            "probe_run_id": promotion["probe_run_id"],
                            "promotion_id": promotion["promotion_id"],
                            "recovered_from_attempt_id": attempt_id,
                        },
                        task_id,
                        attempt_id,
                    )
                recovered = {
                    **prior,
                    "status": "RECOVERED_FINALIZER_DEPENDENCY",
                    "recovered_at_utc": now,
                    "retry_after": retry_after,
                    "recovery_reason": "pinned TIFXYZ finalizer module was absent from the worker runtime",
                }
                connection.execute(
                    "UPDATE attempts SET state='RECOVERED_FINALIZER_DEPENDENCY',result_json=?,updated_at=? WHERE attempt_id=?",
                    (_dump(recovered), now, attempt_id),
                )
                connection.execute(
                    "UPDATE tasks SET state='PENDING',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,retry_after=?,active_attempt_id=NULL,updated_at=? WHERE task_id=?",
                    (retry_after, now, task_id),
                )
                self.event(connection, "STATE_RECOVERED_FINALIZER_DEPENDENCY", recovered, task_id, attempt_id)
                connection.commit()
                return recovered
            except BaseException:
                connection.rollback()
                raise

    def insert_human_review(self, event: dict[str, Any]) -> dict[str, Any]:
        """Insert one immutable exact-job review, or return its idempotent twin.

        The three refusals below are the reason a direct call to this method is
        not a way around the server resolver. The lock has to re-derive from the
        event itself, and the route is re-read here from this store's own
        immutable receipt table rather than believed from the event -- which is
        the part a caller holding only a well-formed dictionary cannot supply.
        """
        from .review_lineage import require_reviewable_event  # noqa: PLC0415

        required = (
            "review_event_id", "p7_job_id", "intent", "mission_id", "sample_id",
            "surface_id", "verdict_sha256", "card_sha256", "config_sha256",
            "vetting_packet_sha256", "by", "event_sha256", "at",
        )
        if any(not event.get(key) for key in required):
            raise ValueError("human review event is incomplete")
        canonical = {key: value for key, value in event.items()
                     if key != "event_sha256"}
        if event["event_sha256"] != content_sha256(canonical):
            raise ValueError("human review event hash is not canonical")
        require_reviewable_event(
            event, self.routing_receipt(str(event["surface_id"])))
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT OR IGNORE INTO human_review_events(
                           review_event_id,p7_job_id,intent,mission_id,sample_id,
                           surface_id,verdict_sha256,card_sha256,config_sha256,
                           vetting_packet_sha256,author,event_json,event_sha256,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        event["review_event_id"], event["p7_job_id"], event["intent"],
                        event["mission_id"], event["sample_id"], event["surface_id"],
                        event["verdict_sha256"], event["card_sha256"],
                        event["config_sha256"], event["vetting_packet_sha256"],
                        event["by"], _dump(event), event["event_sha256"], event["at"],
                    ),
                )
                row = connection.execute(
                    """SELECT event_json FROM human_review_events
                        WHERE p7_job_id=? AND intent=?""",
                    (event["p7_job_id"], event["intent"]),
                ).fetchone()
                connection.commit()
                return _load(row["event_json"])
            except BaseException:
                connection.rollback()
                raise

    def human_reviews(self, p7_job_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                _load(row["event_json"])
                for row in connection.execute(
                    """SELECT event_json FROM human_review_events
                        WHERE p7_job_id=? ORDER BY created_at,review_event_id""",
                    (p7_job_id,),
                )
            ]

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            tasks = {row[0]: row[1] for row in connection.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state ORDER BY state")}
            attempts = {row[0]: row[1] for row in connection.execute("SELECT state,COUNT(*) FROM attempts GROUP BY state ORDER BY state")}
            qc_states = {row[0]: row[1] for row in connection.execute("SELECT state,COUNT(*) FROM qc_jobs GROUP BY state ORDER BY state")}
            workers = {
                row["worker_id"]: _load(row["capabilities_json"])
                for row in connection.execute(
                    "SELECT worker_id,capabilities_json FROM worker_capabilities ORDER BY worker_id"
                )
            }
            return {
                "schema": "campaignx.segment_fleet_status.v1",
                "database": str(self.path),
                "source_snapshots": connection.execute("SELECT COUNT(*) FROM source_snapshots").fetchone()[0],
                "surfaces": connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0],
                "tasks": tasks,
                "attempts": attempts,
                "seed_probes": self.probe_status(),
                "qc_jobs": connection.execute("SELECT COUNT(*) FROM qc_jobs").fetchone()[0],
                "qc_job_states": qc_states,
                "workers": workers,
            }
    def register_campaign_resume_principal_attestation(
        self, authorization: dict[str, Any], *,
        authenticated_principal: str,
    ) -> dict[str, Any]:
        """Create-once trust anchor written only by the authenticated panel."""
        digest = authorization.get("authorization_sha256")
        context = authorization.get("authentication_context")
        principal = authenticated_principal
        if (not isinstance(digest, str)
                or digest != content_sha256({
                    key: value for key, value in authorization.items()
                    if key != "authorization_sha256"
                })
                or not isinstance(principal, str) or not principal
                or not isinstance(context, dict)
                or set(context) != {
                    "mechanism", "principal", "session_fingerprint_sha256",
                    "request_method", "request_path",
                }
                or context.get("mechanism") !=
                    "HELENA_AUTHENTICATED_PANEL_SESSION"
                or context.get("principal") != principal
                or not isinstance(context.get("session_fingerprint_sha256"), str)
                or len(context["session_fingerprint_sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in context["session_fingerprint_sha256"])
                or context.get("request_method") != "POST"
                or context.get("request_path") != "/api/segmentation/runs"
                or authorization.get("authorized_by") != principal
                or not isinstance(authorization.get("mission_id"), str)):
            raise ValueError(
                "campaign resume principal attestation is not panel-authenticated")
        encoded = _dump(authorization)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO campaign_resume_principal_attestations
                   (authorization_sha256,mission_id,principal,
                    authorization_json,created_at) VALUES(?,?,?,?,?)""",
                (digest, authorization["mission_id"], principal, encoded, utc_now()),
            )
            row = connection.execute(
                """SELECT authorization_json
                     FROM campaign_resume_principal_attestations
                    WHERE authorization_sha256=?""",
                (digest,),
            ).fetchone()
            if row is None or _load(row["authorization_json"]) != authorization:
                connection.rollback()
                raise ValueError(
                    "campaign resume principal attestation already differs")
            connection.commit()
        return copy.deepcopy(authorization)
