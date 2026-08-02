from __future__ import annotations

import json
import secrets
import sqlite3
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
    "PROBE_REVIEW_PENDING",
    "PROBE_REJECTED_ALL",
)

QC_OUTCOME_STATES = {
    "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS": ("QC_CT_INSUFFICIENT", "CT_INSUFFICIENT"),
    "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL": ("QC_SCREENED", "CT_SUPPORTED"),
    "CT_SUPPORTED_RETAINED_FOR_REVIEW": ("QC_REVIEW_PENDING", "CT_SUPPORTED_REVIEW"),
}

# The geometry verdict is a second, orthogonal axis.  It deliberately does NOT
# live inside QC_OUTCOME_STATES: those three outcomes are the ink/CT axis that
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

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT,
  attempt_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _deadline(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class FleetStore:
    """Single-host reference control plane.

    SQLite uses BEGIN IMMEDIATE for atomic claims and is suitable for tests or
    several local processes. Distributed workers must use the PostgreSQL
    migration and service before V2 deployment.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SQLITE_SCHEMA)
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

    def event(self, connection: sqlite3.Connection, event_type: str, payload: Any, task_id: str | None = None, attempt_id: str | None = None) -> None:
        connection.execute(
            "INSERT INTO events(task_id,attempt_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
            (task_id, attempt_id, event_type, _dump(payload), utc_now()),
        )

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

    @staticmethod
    def _snapshot(row: sqlite3.Row) -> dict[str, Any]:
        value = _load(row["payload_json"])
        value.update({"source_snapshot_id": row["source_snapshot_id"], "shape_xyz": _load(row["shape_xyz_json"])})
        return value

    def import_surface(self, payload: dict[str, Any]) -> str:
        surface_id = str(payload.get("surface_id") or stable_id("surface-import", payload))
        source_id = str(payload["source_snapshot_id"])
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz_json,sample_points_json,area_cm2,state,physical_qc_state,payload_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(surface_id) DO NOTHING""",
                (surface_id, source_id, payload["sample_id"], payload.get("owner", "imported"), payload.get("artifact_sha256"), payload.get("artifact_uri"), _dump(payload["bbox_xyz"]), _dump(payload.get("sample_points")) if payload.get("sample_points") is not None else None, payload.get("area_cm2"), payload.get("state", "IMPORTED"), payload.get("physical_qc_state", "UNVALIDATED"), _dump({**payload, "surface_id": surface_id}), now),
            )
        return surface_id

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
                geometry_state = (payload.get("geometry_qc_state")
                                  or DEFAULT_GEOMETRY_QC_STATE)
                connection.execute(
                    """INSERT INTO qc_jobs(qc_job_id,surface_id,profile_id,state,payload_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        qc_id,
                        surface_id,
                        profile_id,
                        qc_job_state_for(geometry_state),
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
                    "qc_state": "PENDING",
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
        inserted = 0
        seen = 0
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
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
                connection.execute("UPDATE attempts SET state='LEASE_EXPIRED',updated_at=? WHERE attempt_id=?", (now, row["active_attempt_id"]))
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
            receipt.get("status") != "PROBE_REVIEW_PENDING"
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
                    """UPDATE attempts SET state='PROBE_REVIEW_PENDING',
                              result_json=?,updated_at=?
                        WHERE attempt_id=?""",
                    (_dump(receipt), now, parent_attempt_id),
                )
                connection.execute(
                    """UPDATE tasks
                          SET state='PROBE_REVIEW_PENDING',worker_id=NULL,
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
                    "STATE_PROBE_REVIEW_PENDING",
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
                if not duplicate_of:
                    surface_state = "FIXTURE_ONLY" if fixture_only else "QC_PENDING"
                    physical_state = (
                        "NOT_APPLICABLE_FIXTURE" if fixture_only else "UNVALIDATED"
                    )
                    connection.execute(
                        """INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz_json,sample_points_json,area_cm2,state,physical_qc_state,geometry_qc_state,payload_json,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (surface_id, surface["source_snapshot_id"], surface["sample_id"], surface.get("owner", "campaign-x"), surface["artifact_sha256"], surface["artifact_uri"], _dump(surface["bbox_xyz"]), _dump(surface.get("sample_points")) if surface.get("sample_points") is not None else None, surface.get("area_cm2"), surface_state, physical_state, geometry_state, _dump(surface), now),
                    )
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
                        job_state = qc_job_state_for(geometry_state)
                        connection.execute(
                            "INSERT INTO qc_jobs(qc_job_id,surface_id,profile_id,state,payload_json,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                            (
                                qc_id,
                                surface_id,
                                qc_profile_id,
                                job_state,
                                _dump({"artifact_set_id": artifact_set_id}),
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

    def surfaces_without_geometry_verdict(
        self, limit: int = 25, sample_id: str | None = None
    ) -> list[dict[str, Any]]:
        """The sqlite mirror of the same question, so tests need no server."""
        query = """SELECT surface_id, sample_id, artifact_uri, artifact_sha256,
                          state, geometry_qc_state
                   FROM segment_surfaces
                   WHERE (geometry_qc_state IS NULL
                          OR geometry_qc_state = 'GEOMETRY_UNMEASURED')
                     AND artifact_uri IS NOT NULL"""
        arguments: list[Any] = []
        if sample_id is not None:
            query += " AND sample_id=?"
            arguments.append(sample_id)
        query += " ORDER BY created_at, surface_id LIMIT ?"
        arguments.append(int(limit))
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, arguments).fetchall()]

    def record_geometry_certification(
        self,
        surface_id: str,
        geometry_state: str,
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a geometry verdict on the axis orthogonal to physical QC.

        Re-certifying an already imported surface uses this path: it never
        touches ``physical_qc_state``, and a rejection fails any QC job that has
        not already been claimed so the surface cannot reach the ink model.
        """

        if geometry_state not in GEOMETRY_QC_STATES:
            raise ValueError(f"unsupported geometry QC state: {geometry_state}")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT surface_id,physical_qc_state FROM surfaces WHERE surface_id=?",
                    (surface_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"unknown surface: {surface_id}")
                connection.execute(
                    "UPDATE surfaces SET geometry_qc_state=? WHERE surface_id=?",
                    (geometry_state, surface_id),
                )
                blocked = 0
                # The verdict promotes what was waiting on it. Without this,
                # holding a job back for unmeasured geometry would strand the
                # surface rather than gate it.
                if str(geometry_state) == "GEOMETRY_CERTIFIED":
                    connection.execute(
                        f"""UPDATE qc_jobs SET state='PENDING',updated_at=?
                            WHERE surface_id=? AND state='{QC_WAITING_GEOMETRY}'""",
                        (now, surface_id),
                    )
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
                    },
                )
                connection.commit()
                return {
                    "surface_id": surface_id,
                    "geometry_qc_state": geometry_state,
                    "physical_qc_state": row["physical_qc_state"],
                    "blocked_qc_jobs": blocked,
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
                    "SELECT active_attempt_id,lease_token FROM tasks "
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
                cursor = connection.execute(
                    """UPDATE qc_jobs SET state='CLAIMED',worker_id=?,lease_token=?,lease_expires_at=?,
                       retry_after=NULL,updated_at=? WHERE qc_job_id=? AND state='PENDING'""",
                    (worker_id, token, _deadline(lease_seconds), now, row["qc_job_id"]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("QC claim lost its atomic update")
                surface = connection.execute("SELECT * FROM surfaces WHERE surface_id=?", (row["surface_id"],)).fetchone()
                source = connection.execute("SELECT * FROM source_snapshots WHERE source_snapshot_id=?", (surface["source_snapshot_id"],)).fetchone()
                self.event(connection, "QC_CLAIMED", {"qc_job_id": row["qc_job_id"], "surface_id": row["surface_id"], "worker_id": worker_id})
                connection.commit()
                surface_value = _load(surface["payload_json"])
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
                connection.execute(
                    """UPDATE qc_jobs SET state='COMPLETED',result_json=?,worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,retry_after=NULL,updated_at=? WHERE qc_job_id=?""",
                    (_dump(result), now, qc_job_id),
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
                    (_dump(receipt), now, qc_job_id),
                )
                self.event(
                    connection,
                    "QC_BLOCKED_CONFIGURATION",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "error": str(receipt.get("error", "")),
                    },
                )
                connection.commit()
                return {
                    "status": "BLOCKED_CONFIGURATION",
                    "qc_job_id": qc_job_id,
                    "surface_id": job["surface_id"],
                    "error": str(receipt.get("error", "")),
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
                    (_dump(receipt), retry_after, now, qc_job_id),
                )
                self.event(
                    connection,
                    "QC_REQUEUED_UNAVAILABLE",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "retry_after": retry_after,
                        "error": str(receipt.get("error", "")),
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
