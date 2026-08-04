from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterable

from .common import content_sha256, is_fixture_surface, stable_id, utc_now
from .dedup import find_duplicate_in_surfaces
from .planner_history import (
    ACTIONABLE_SEGMENTATION_HISTORY_STATES,
    build_regional_attempt_history,
)
from .store import (
    ACTIVE_STATES,
    ADMISSIBLE_PHYSICAL_QC_STATES,
    DEFAULT_GEOMETRY_QC_STATE,
    FINALIZATION_OUTCOME_STATES,
    GEOMETRY_QC_STATES,
    QC_OUTCOME_STATES,
    QC_WAITING_GEOMETRY,
    TERMINAL_STATES,
    is_geometry_rejected,
    qc_job_state_for,
    normalize_resource_requirements,
    normalize_worker_capabilities,
    validate_finalization_evidence,
    validate_qc_result_contract,
)


def _grid_step(centers: Any) -> list[float] | None:
    """The spacing of a grid, from the cell centres its tasks carry.

    Not from `bounds_xyz`: that is the candidate discovery region around the
    centre -- 256 voxels across where the grid steps by 1024 -- and a cell count
    derived from it is wrong by a factor of 64. Not from the grid version's name
    either, which is a label somebody typed.

    The smallest positive gap between neighbouring centres on an axis is the
    step. It needs two cells that differ on that axis; with fewer, this says it
    does not know.
    """
    if not centers:
        return None
    try:
        points = [[float(centre[axis]) for axis in ("x", "y", "z")]
                  for centre in centers if isinstance(centre, dict)]
    except Exception:  # noqa: BLE001 -- an older task may carry no centre
        return None
    steps: list[float] = []
    for axis in range(3):
        values = sorted({point[axis] for point in points})
        gaps = [b - a for a, b in zip(values, values[1:], strict=False) if b > a]
        if not gaps:
            return None
        steps.append(min(gaps))
    return steps


def _cells_in_volume(shape_xyz: Any, edges: list[float] | None) -> int | None:
    if not shape_xyz or not edges:
        return None
    try:
        spans = [float(value) for value in shape_xyz]
    except Exception:  # noqa: BLE001
        return None
    if len(spans) != len(edges):
        return None
    total = 1.0
    for span, edge in zip(spans, edges, strict=True):
        total *= max(span / edge, 1.0)
    return int(round(total))


def _deadline(seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _json_utc_timestamp(value: Any) -> str:
    """Normalize a PostgreSQL timestamptz before it enters a JSON receipt."""

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise RuntimeError(
            "PostgreSQL returned a retain_until value without a timezone"
        )
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PostgresFleetStore:
    """Authoritative multi-host control plane.

    Every task claim uses ``FOR UPDATE SKIP LOCKED`` and a hashed lease token.
    The raw database URL is deliberately never exposed through ``identity`` or
    status receipts.
    """

    def __init__(self, database_url: str, *, identity: str = "postgresql://redacted"):
        self.database_url = database_url
        self.identity = identity

    def connect(self):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError("PostgreSQL mode requires psycopg2 or psycopg2-binary") from error
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def initialize(self) -> None:
        migration = Path(__file__).with_name("migrations") / "001_postgresql.sql"
        sql = migration.read_text(encoding="utf-8")
        # Read the sentinel out of the file instead of hard-coding it here.
        #
        # It was the literal 2, and the file later grew a geometry column
        # without that literal moving. Every database that had already recorded
        # version 2 then skipped the whole file forever, so the column was never
        # added -- silently, because the fast path exists precisely to say
        # nothing when there is nothing to do. Deriving it means changing the
        # schema and forgetting the sentinel is no longer possible.
        target = max(
            (int(match) for match in re.findall(
                r"INSERT\s+INTO\s+segment_schema_migrations\s*\([^)]*\)\s*"
                r"VALUES\s*\(\s*(\d+)", sql, re.IGNORECASE)),
            default=0)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                # Worker CLIs call initialize before every task. Replaying DDL
                # concurrently can deadlock with another worker that has already
                # moved on to claiming a row: ALTER TABLE queues an
                # AccessExclusiveLock while claim needs RowExclusiveLock. Fast
                # path an already initialized database and serialize the first
                # installation so multi-GPU workers never race schema DDL.
                cursor.execute(
                    """SELECT EXISTS (
                           SELECT 1
                           FROM pg_catalog.pg_class c
                           JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
                           WHERE n.nspname=current_schema()
                             AND c.relname='segment_schema_migrations'
                       ) AS present"""
                )
                if cursor.fetchone()["present"]:
                    cursor.execute(
                        "SELECT 1 FROM segment_schema_migrations WHERE version=%s", (target,)
                    )
                    if cursor.fetchone() is not None:
                        return
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (0x43414D505849,))
                cursor.execute(
                    "SELECT to_regclass('segment_schema_migrations') AS relation"
                )
                if cursor.fetchone()["relation"] is not None:
                    cursor.execute(
                        "SELECT 1 FROM segment_schema_migrations WHERE version=%s", (target,)
                    )
                    if cursor.fetchone() is not None:
                        return
                cursor.execute(sql)

    @staticmethod
    def event(cursor: Any, event_type: str, payload: Any, task_id: str | None = None, attempt_id: str | None = None) -> None:
        cursor.execute(
            "INSERT INTO segment_events(task_id,attempt_id,event_type,payload) VALUES(%s,%s,%s,%s::jsonb)",
            (task_id, attempt_id, event_type, json.dumps(payload, sort_keys=True, separators=(",", ":"))),
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
        value = {**payload, "source_snapshot_id": source_id}
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO segment_source_snapshots
                       (source_snapshot_id,sample_id,ct_uri,ct_sha256,m7_uri,m7_sha256,shape_xyz,voxel_size_um,coordinate_frame,payload)
                       VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                       ON CONFLICT(source_snapshot_id) DO NOTHING""",
                    (
                        source_id,
                        payload["sample_id"],
                        payload["ct_uri"],
                        payload.get("ct_sha256"),
                        payload["m7_uri"],
                        payload.get("m7_sha256"),
                        json.dumps(payload["shape_xyz"]),
                        float(payload["voxel_size_um"]),
                        payload.get("coordinate_frame", "ct_l0_xyz"),
                        json.dumps(value, sort_keys=True, separators=(",", ":")),
                    ),
                )
        return source_id

    def snapshots(self, samples: set[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM segment_source_snapshots"
        args: list[Any] = []
        if samples:
            query += " WHERE sample_id = ANY(%s)"
            args.append(sorted(samples))
        query += " ORDER BY sample_id,source_snapshot_id"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, args)
                return [self._snapshot(row) for row in cursor.fetchall()]

    @staticmethod
    def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
        value = dict(row["payload"])
        value.update({"source_snapshot_id": row["source_snapshot_id"], "shape_xyz": row["shape_xyz"]})
        return value

    def import_surface(self, payload: dict[str, Any]) -> str:
        surface_id = str(payload.get("surface_id") or stable_id("surface-import", payload))
        value = {**payload, "surface_id": surface_id}
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO segment_surfaces
                       (surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz,sample_points,area_cm2,state,physical_qc_state,payload)
                       VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(surface_id) DO NOTHING""",
                    (
                        surface_id,
                        payload["source_snapshot_id"],
                        payload["sample_id"],
                        payload.get("owner", "imported"),
                        payload.get("artifact_sha256"),
                        payload.get("artifact_uri"),
                        json.dumps(payload["bbox_xyz"]),
                        json.dumps(payload.get("sample_points")) if payload.get("sample_points") is not None else None,
                        payload.get("area_cm2"),
                        payload.get("state", "IMPORTED"),
                        payload.get("physical_qc_state", "UNVALIDATED"),
                        json.dumps(value, sort_keys=True, separators=(",", ":")),
                    ),
                )
        return surface_id

    def enqueue_imported_surface_qc(
        self,
        payload: dict[str, Any],
        *,
        profile_id: str,
        job_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically reconcile an unvalidated import and enqueue one QC job."""

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
        value = {**payload, "surface_id": surface_id}
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT sample_id FROM segment_source_snapshots WHERE source_snapshot_id=%s",
                    (source_id,),
                )
                source = cursor.fetchone()
                if source is None or source["sample_id"] != sample_id:
                    raise RuntimeError("QC backfill source snapshot is missing or mismatched")
                cursor.execute(
                    "SELECT * FROM segment_surfaces WHERE surface_id=%s FOR UPDATE",
                    (surface_id,),
                )
                existing = cursor.fetchone()
                cursor.execute(
                    "SELECT * FROM segment_qc_jobs WHERE surface_id=%s AND profile_id=%s FOR UPDATE",
                    (surface_id, profile_id),
                )
                job = cursor.fetchone()
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
                    return {
                        "status": "ALREADY_ENQUEUED",
                        "surface_id": surface_id,
                        "qc_job_id": job["qc_job_id"],
                        "qc_state": job["state"],
                    }
                if existing is None:
                    cursor.execute(
                        """INSERT INTO segment_surfaces
                           (surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz,sample_points,area_cm2,state,physical_qc_state,payload)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,'QC_PENDING','UNVALIDATED',%s::jsonb)""",
                        (
                            surface_id,
                            source_id,
                            sample_id,
                            payload.get("owner", "campaign-x"),
                            artifact_sha256,
                            artifact_uri,
                            json.dumps(payload["bbox_xyz"]),
                            json.dumps(payload.get("sample_points"))
                            if payload.get("sample_points") is not None
                            else None,
                            payload.get("area_cm2"),
                            json.dumps(value, sort_keys=True, separators=(",", ":")),
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
                    cursor.execute(
                        """UPDATE segment_surfaces SET owner=%s,artifact_sha256=%s,
                           artifact_uri=%s,bbox_xyz=%s::jsonb,sample_points=%s::jsonb,
                           area_cm2=%s,state='QC_PENDING',payload=%s::jsonb
                           WHERE surface_id=%s""",
                        (
                            payload.get("owner", existing["owner"]),
                            artifact_sha256,
                            artifact_uri,
                            json.dumps(payload["bbox_xyz"]),
                            json.dumps(payload.get("sample_points"))
                            if payload.get("sample_points") is not None
                            else None,
                            payload.get("area_cm2"),
                            json.dumps(value, sort_keys=True, separators=(",", ":")),
                            surface_id,
                        ),
                    )
                    reconciliation = "RECONCILED_UNVALIDATED"
                # The same gate finalization uses, and it was missing here: this
                # path is how an imported surface gets a QC job, imports are
                # geometrically unmeasured, and a hardcoded PENDING made every one
                # of them immediately claimable by the ink model. The gate held on
                # the path that measures geometry and leaked on the path that never
                # does, which is the wrong way round.
                #
                # A verdict the surface already carries outranks the default, or
                # certifying a surface and then enqueuing its job strands the job:
                # an import's payload never names a geometry state, and
                # record_geometry_certification -- the thing that promotes a job
                # out of WAITING_GEOMETRY -- has already run by then, so nothing
                # is left to promote it.
                geometry_state = (
                    payload.get("geometry_qc_state")
                    or (existing.get("geometry_qc_state")
                        if existing is not None else None)
                    or DEFAULT_GEOMETRY_QC_STATE
                )
                job_state = qc_job_state_for(geometry_state)
                cursor.execute(
                    """INSERT INTO segment_qc_jobs
                       (qc_job_id,surface_id,profile_id,state,payload,updated_at)
                       VALUES(%s,%s,%s,%s,%s::jsonb,now())""",
                    (
                        qc_id,
                        surface_id,
                        profile_id,
                        job_state,
                        json.dumps(job_payload or {}, sort_keys=True, separators=(",", ":")),
                    ),
                )
                self.event(
                    cursor,
                    "QC_BACKFILL_ENQUEUED",
                    {
                        "qc_job_id": qc_id,
                        "surface_id": surface_id,
                        "profile_id": profile_id,
                        "reconciliation": reconciliation,
                    },
                )
                return {
                    "status": "ENQUEUED",
                    "surface_id": surface_id,
                    "qc_job_id": qc_id,
                    # What was written, not what used to be assumed: this said
                    # PENDING unconditionally, so a caller reading the answer was
                    # told the job was claimable while the row said otherwise.
                    "qc_state": job_state,
                    "reconciliation": reconciliation,
                }

    def surfaces_for_snapshot(self, source_snapshot_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segment_surfaces WHERE source_snapshot_id=%s ORDER BY surface_id",
                    (source_snapshot_id,),
                )
                rows = cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row["payload"])
            value.update(
                {
                    "surface_id": row["surface_id"],
                    "artifact_sha256": row["artifact_sha256"],
                    "artifact_uri": row["artifact_uri"],
                    "bbox_xyz": row["bbox_xyz"],
                    "sample_points": row["sample_points"],
                    "state": row["state"],
                    # Both verdicts, because they are the two things a caller
                    # has to know about a surface and neither is `state`: this
                    # returned the payload plus `state` alone, so a reader of
                    # this method could not tell a certified sheet from one the
                    # gate rejected.
                    "physical_qc_state": row["physical_qc_state"],
                    "geometry_qc_state": row.get("geometry_qc_state"),
                }
            )
            result.append(value)
        return result

    def create_tasks(self, tasks: Iterable[dict[str, Any]]) -> tuple[int, int]:
        inserted = 0
        seen = 0
        with self.connect() as connection:
            with connection.cursor() as cursor:
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
                        **{key: task[key] for key in (
                            "source_snapshot_id", "grid_version", "cell_id",
                            "policy_version")},
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
                    cursor.execute(
                        """INSERT INTO segment_tasks
                           (task_id,mission_id,source_snapshot_id,cell_id,grid_version,policy_version,bounds_xyz,center_xyz,priority,parameter_envelope,catalog_snapshot_sha256,payload,state,gpu_required,minimum_vram_gb,seed_probe_required,created_by)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s,%s::jsonb,'PENDING',%s,%s,%s,%s)
                           ON CONFLICT(mission_id,source_snapshot_id,grid_version,cell_id,policy_version) DO NOTHING""",
                        (
                            task_id,
                            mission_id,
                            task["source_snapshot_id"],
                            task["cell_id"],
                            task["grid_version"],
                            task["policy_version"],
                            json.dumps(task["bounds_xyz"]),
                            json.dumps(task["center_xyz"]),
                            float(task["priority"]),
                            json.dumps(task["parameter_envelope"], sort_keys=True, separators=(",", ":")),
                            task["catalog_snapshot_sha256"],
                            json.dumps(value, sort_keys=True, separators=(",", ":")),
                            requirements["gpu_required"],
                            requirements["minimum_vram_gb"],
                            requirements["seed_probe_required"],
                            # The task's own payload carries it when the caller
                            # said who they were; the default keeps a backlog
                            # honest rather than crediting it to the next asker.
                            str(task.get("created_by") or "unattributed"),
                        ),
                    )
                    if cursor.rowcount:
                        inserted += 1
                        self.event(cursor, "TASK_CREATED", {"priority": task["priority"]}, task_id=task_id)
                    else:
                        cursor.execute(
                            """SELECT payload FROM segment_tasks
                                WHERE mission_id=%s AND source_snapshot_id=%s AND grid_version=%s
                                  AND cell_id=%s AND policy_version=%s""",
                            (
                                mission_id,
                                task["source_snapshot_id"],
                                task["grid_version"],
                                task["cell_id"],
                                task["policy_version"],
                            ),
                        )
                        existing = cursor.fetchone()
                        existing_value = (
                            dict(existing["payload"]) if existing is not None else {}
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
        return inserted, seen

    def _expire_leases(self, cursor: Any) -> None:
        cursor.execute(
            """SELECT task_id,active_attempt_id FROM segment_tasks
               WHERE state=ANY(%s) AND lease_expires_at IS NOT NULL AND lease_expires_at<=now()
               FOR UPDATE SKIP LOCKED""",
            (list(ACTIVE_STATES),),
        )
        for row in cursor.fetchall():
            if row["active_attempt_id"]:
                cursor.execute(
                    "UPDATE segment_attempts SET state='LEASE_EXPIRED',updated_at=now() WHERE attempt_id=%s",
                    (row["active_attempt_id"],),
                )
            cursor.execute(
                """UPDATE segment_tasks SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                   lease_expires_at=NULL,active_attempt_id=NULL,updated_at=now() WHERE task_id=%s""",
                (row["task_id"],),
            )
            self.event(cursor, "LEASE_EXPIRED", {}, row["task_id"], row["active_attempt_id"])

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
        token = secrets.token_urlsafe(32)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._expire_leases(cursor)
                cursor.execute(
                    """INSERT INTO segment_worker_capabilities(worker_id,capabilities,updated_at)
                       VALUES(%s,%s::jsonb,now()) ON CONFLICT(worker_id) DO UPDATE SET
                       capabilities=excluded.capabilities,updated_at=excluded.updated_at""",
                    (worker_id, json.dumps(worker_capabilities, sort_keys=True, separators=(",", ":"))),
                )
                if task_id is None:
                    cursor.execute(
                        """SELECT * FROM segment_tasks
                           WHERE state='PENDING' AND (retry_after IS NULL OR retry_after<=now())
                           AND (gpu_required=false OR %s=true) AND minimum_vram_gb<=%s
                           AND (seed_probe_required=false OR %s=true)
                           AND ((%s IS NULL AND
                                 payload->'benchmark_execution'->>'benchmark_spec_sha256'
                                 IS NULL)
                                OR payload->'benchmark_execution'->>'benchmark_spec_sha256'=%s)
                           ORDER BY priority DESC,task_id FOR UPDATE SKIP LOCKED LIMIT 1""",
                        (
                            worker_capabilities["cuda_available"],
                            worker_capabilities["gpu_vram_gb"],
                            worker_capabilities["seed_probe_v1"],
                            worker_capabilities["benchmark_spec_sha256"],
                            worker_capabilities["benchmark_spec_sha256"],
                        ),
                    )
                else:
                    cursor.execute(
                        """SELECT * FROM segment_tasks
                           WHERE task_id=%s AND state='PENDING' AND (retry_after IS NULL OR retry_after<=now())
                           AND (gpu_required=false OR %s=true) AND minimum_vram_gb<=%s
                           AND (seed_probe_required=false OR %s=true)
                           AND ((%s IS NULL AND
                                 payload->'benchmark_execution'->>'benchmark_spec_sha256'
                                 IS NULL)
                                OR payload->'benchmark_execution'->>'benchmark_spec_sha256'=%s)
                           FOR UPDATE SKIP LOCKED""",
                        (
                            task_id,
                            worker_capabilities["cuda_available"],
                            worker_capabilities["gpu_vram_gb"],
                            worker_capabilities["seed_probe_v1"],
                            worker_capabilities["benchmark_spec_sha256"],
                            worker_capabilities["benchmark_spec_sha256"],
                        ),
                    )
                row = cursor.fetchone()
                if row is None:
                    return None
                task_id = row["task_id"]
                cursor.execute(
                    "SELECT COALESCE(MAX(attempt_number),0)+1 AS value FROM segment_attempts WHERE task_id=%s",
                    (task_id,),
                )
                attempt_number = int(cursor.fetchone()["value"])
                attempt_id = stable_id("attempt", {"task_id": task_id, "attempt_number": attempt_number})
                cursor.execute(
                    """INSERT INTO segment_attempts(attempt_id,task_id,attempt_number,worker_id,state)
                       VALUES(%s,%s,%s,%s,'CLAIMED')""",
                    (attempt_id, task_id, attempt_number, worker_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state='CLAIMED',worker_id=%s,lease_token_hash=%s,
                       lease_expires_at=%s,retry_after=NULL,active_attempt_id=%s,updated_at=now()
                       WHERE task_id=%s""",
                    (worker_id, _token_hash(token), _deadline(lease_seconds), attempt_id, task_id),
                )
                self.event(cursor, "TASK_CLAIMED", {"worker_id": worker_id, "attempt_number": attempt_number, "capabilities": worker_capabilities}, task_id, attempt_id)
        return self.task_packet(task_id, attempt_id=attempt_id, lease_token=token)

    def pending_tasks(self, limit: int) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("survey limit must be at least one")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT task_id FROM segment_tasks WHERE state='PENDING'
                       AND (retry_after IS NULL OR retry_after<=now())
                       ORDER BY priority DESC,task_id LIMIT %s""",
                    (limit,),
                )
                ids = [row["task_id"] for row in cursor.fetchall()]
        return [self.task_packet(task_id) for task_id in ids]

    def task_packet(self, task_id: str, attempt_id: str | None = None, lease_token: str | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM segment_tasks WHERE task_id=%s", (task_id,))
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(task_id)
                cursor.execute("SELECT * FROM segment_source_snapshots WHERE source_snapshot_id=%s", (row["source_snapshot_id"],))
                source = cursor.fetchone()
                if source is None:
                    raise RuntimeError("task source snapshot disappeared")
                selected_attempt = attempt_id or row["active_attempt_id"]
                attempt = None
                if selected_attempt:
                    cursor.execute("SELECT * FROM segment_attempts WHERE attempt_id=%s", (selected_attempt,))
                    attempt = cursor.fetchone()
        payload = dict(row["payload"])
        payload.update(
            {
                "schema": "campaignx.segmentation_task.v1",
                "task_id": row["task_id"],
                "state": row["state"],
                "bounds_xyz": row["bounds_xyz"],
                "center_xyz": row["center_xyz"],
                "parameter_envelope": row["parameter_envelope"],
                "resource_requirements": normalize_resource_requirements({
                    "gpu_required": bool(row["gpu_required"]),
                    "minimum_vram_gb": float(row["minimum_vram_gb"]),
                    "seed_probe_required": bool(row["seed_probe_required"]),
                }),
                "source": self._snapshot(source),
            }
        )
        if attempt is not None:
            payload.update(
                {
                    "attempt_id": attempt["attempt_id"],
                    "attempt_number": attempt["attempt_number"],
                    "worker_id": attempt["worker_id"],
                }
            )
            if lease_token is not None:
                payload["lease_token"] = lease_token
        return payload

    def regional_attempt_history(
        self, task: dict[str, Any], *, limit: int = 12
    ) -> dict[str, Any]:
        """Return the same bounded planner history contract as SQLite."""

        low, high = task["bounds_xyz"]
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT a.attempt_id,a.task_id,a.state,a.locked_plan,
                              a.result,a.updated_at,
                              t.cell_id,t.policy_version,t.bounds_xyz
                       FROM segment_attempts a
                       JOIN segment_tasks t ON t.task_id=a.task_id
                       WHERE t.source_snapshot_id=%s AND a.attempt_id<>%s
                         AND (a.state=ANY(%s)
                              OR (a.state='QC_PENDING'
                                  AND a.result->>'geometry_qc_state'=ANY(%s)))
                         AND (t.bounds_xyz->0->>0)::double precision<=%s
                         AND (t.bounds_xyz->1->>0)::double precision>=%s
                         AND (t.bounds_xyz->0->>1)::double precision<=%s
                         AND (t.bounds_xyz->1->>1)::double precision>=%s
                         AND (t.bounds_xyz->0->>2)::double precision<=%s
                         AND (t.bounds_xyz->1->>2)::double precision>=%s
                       ORDER BY a.updated_at DESC,a.attempt_id
                       LIMIT %s""",
                    (
                        task["source"]["source_snapshot_id"],
                        task.get("attempt_id", ""),
                        attempt_states,
                        geometry_states,
                        float(high[0]),
                        float(low[0]),
                        float(high[1]),
                        float(low[1]),
                        float(high[2]),
                        float(low[2]),
                        limit,
                    ),
                )
                rows = [dict(row) for row in cursor.fetchall()]
        return build_regional_attempt_history(task, rows, limit=limit)

    @staticmethod
    def _assert_owner(cursor: Any, task_id: str, attempt_id: str, lease_token: str) -> dict[str, Any]:
        cursor.execute("SELECT * FROM segment_tasks WHERE task_id=%s FOR UPDATE", (task_id,))
        row = cursor.fetchone()
        if row is None or row["active_attempt_id"] != attempt_id or row["lease_token_hash"] != _token_hash(lease_token):
            raise RuntimeError("attempt no longer owns the task")
        return row

    @staticmethod
    def _assert_probe_trial_owner(
        cursor: Any,
        task_id: str,
        probe_trial_id: str,
        probe_attempt_id: str,
        lease_token: str,
    ) -> dict[str, Any]:
        cursor.execute(
            """SELECT t.*
                 FROM segment_probe_trials t
                 JOIN segment_probe_runs r
                   ON r.probe_run_id=t.probe_run_id
                 JOIN segment_probe_attempts a
                   ON a.probe_attempt_id=t.active_probe_attempt_id
                  AND a.probe_trial_id=t.probe_trial_id
                WHERE t.probe_trial_id=%s AND r.task_id=%s
                  AND a.probe_attempt_id=%s
                FOR UPDATE OF t""",
            (probe_trial_id, task_id, probe_attempt_id),
        )
        row = cursor.fetchone()
        if (
            row is None
            or row["active_probe_attempt_id"] != probe_attempt_id
            or row["lease_token_hash"] != _token_hash(lease_token)
        ):
            raise RuntimeError("probe operation belongs to a stale trial lease")
        return row

    @staticmethod
    def _assert_probe_run_task(
        cursor: Any,
        task_id: str,
        probe_run_id: str,
    ) -> dict[str, Any]:
        cursor.execute(
            """SELECT * FROM segment_probe_runs
                WHERE probe_run_id=%s AND task_id=%s
                FOR UPDATE""",
            (probe_run_id, task_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                "probe run does not belong to the leased parent task"
            )
        return row

    @staticmethod
    def _expire_probe_leases(cursor: Any, probe_run_id: str) -> None:
        cursor.execute(
            """SELECT probe_trial_id,active_probe_attempt_id
                 FROM segment_probe_trials
                WHERE probe_run_id=%s
                  AND state=ANY(%s)
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=now()
                FOR UPDATE SKIP LOCKED""",
            (
                probe_run_id,
                ["CLAIMED", "RUNNING", "UPLOADED", "EVALUATING"],
            ),
        )
        for row in cursor.fetchall():
            if row["active_probe_attempt_id"]:
                cursor.execute(
                    """UPDATE segment_probe_attempts
                          SET state='LEASE_EXPIRED',updated_at=now()
                        WHERE probe_attempt_id=%s""",
                    (row["active_probe_attempt_id"],),
                )
            cursor.execute(
                """UPDATE segment_probe_trials
                      SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                          lease_expires_at=NULL,active_probe_attempt_id=NULL,
                          updated_at=now()
                    WHERE probe_trial_id=%s""",
                (row["probe_trial_id"],),
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                owner = self._assert_owner(
                    cursor,
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
                authoritative_task = dict(owner["payload"])
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
                cursor.execute(
                    """SELECT * FROM segment_probe_runs
                        WHERE task_id=%s FOR UPDATE""",
                    (task["task_id"],),
                )
                existing = cursor.fetchone()
                inserted = existing is None
                if inserted:
                    cursor.execute(
                        """INSERT INTO segment_probe_runs
                           (probe_run_id,task_id,created_by_attempt_id,
                            source_snapshot_id,candidate_set,
                            candidate_set_sha256,policy_id,policy,
                            policy_sha256,executor_fingerprint,
                            executor_fingerprint_sha256,state)
                           VALUES(%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,
                                  %s::jsonb,%s,'PENDING')""",
                        (
                            probe_run_id,
                            task["task_id"],
                            task["attempt_id"],
                            authoritative_source_id,
                            json.dumps(
                                candidates,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            candidate_set_sha,
                            normalized["policy_id"],
                            json.dumps(
                                normalized,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            policy_sha,
                            json.dumps(
                                executor_fingerprint,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            executor_sha,
                        ),
                    )
                else:
                    exact = (
                        existing["probe_run_id"] == probe_run_id
                        and existing["task_id"] == task["task_id"]
                        and existing["source_snapshot_id"]
                        == authoritative_source_id
                        and existing["candidate_set_sha256"]
                        == candidate_set_sha
                        and existing["policy_sha256"] == policy_sha
                        and existing["executor_fingerprint_sha256"]
                        == executor_sha
                    )
                    if not exact:
                        raise RuntimeError(
                            "parent task already has a different immutable probe run"
                        )
                if inserted:
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
                        cursor.execute(
                            """INSERT INTO segment_probe_trials
                               (probe_trial_id,probe_run_id,candidate_id,
                                candidate_rank,candidate,state,gpu_required,
                                minimum_vram_gb)
                               VALUES(%s,%s,%s,%s,%s::jsonb,'PENDING',%s,%s)""",
                            (
                                probe_trial_id,
                                probe_run_id,
                                candidate["candidate_id"],
                                int(candidate["candidate_rank"]),
                                json.dumps(
                                    candidate,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                requirements["gpu_required"],
                                requirements["minimum_vram_gb"],
                            ),
                        )
                    self.event(
                        cursor,
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
        return self.probe_run(probe_run_id)

    def probe_run(self, probe_run_id: str) -> dict[str, Any]:
        """Return a complete, read-only run snapshot for recovery or decision."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segment_probe_runs WHERE probe_run_id=%s",
                    (probe_run_id,),
                )
                run = cursor.fetchone()
                if run is None:
                    raise KeyError(probe_run_id)
                cursor.execute(
                    """SELECT t.*,a.probe_artifact_set_id,a.manifest,
                              a.manifest_sha256,a.artifact_uri,
                              a.state AS artifact_state,
                              e.evaluation_id,e.result AS evaluation,
                              e.result_sha256 AS evaluation_sha256,e.verdict
                         FROM segment_probe_trials t
                         LEFT JOIN segment_probe_artifact_sets a
                           ON a.probe_trial_id=t.probe_trial_id
                         LEFT JOIN segment_probe_evaluations e
                           ON e.probe_trial_id=t.probe_trial_id
                        WHERE t.probe_run_id=%s
                        ORDER BY t.candidate_rank""",
                    (probe_run_id,),
                )
                rows = cursor.fetchall()
                cursor.execute(
                    "SELECT * FROM segment_probe_decisions WHERE probe_run_id=%s",
                    (probe_run_id,),
                )
                decision_row = cursor.fetchone()
        trials = [
            {
                "probe_run_id": probe_run_id,
                "probe_trial_id": row["probe_trial_id"],
                "candidate_id": row["candidate_id"],
                "candidate_rank": row["candidate_rank"],
                "candidate": row["candidate"],
                "state": row["state"],
                "locked_plan": row["locked_plan"],
                "result": row["result"],
                "probe_artifact_set_id": row["probe_artifact_set_id"],
                "manifest": row["manifest"],
                "manifest_sha256": row["manifest_sha256"],
                "artifact_uri": row["artifact_uri"],
                "artifact_state": row["artifact_state"],
                "evaluation": row["evaluation"],
                "evaluation_sha256": row["evaluation_sha256"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
        return {
            "schema": "campaignx.seed_probe_run.v1",
            "probe_run_id": run["probe_run_id"],
            "task_id": run["task_id"],
            "source_snapshot_id": run["source_snapshot_id"],
            "state": run["state"],
            "candidate_set": run["candidate_set"],
            "candidate_set_sha256": run["candidate_set_sha256"],
            "policy": run["policy"],
            "policy_sha256": run["policy_sha256"],
            "executor_fingerprint": run["executor_fingerprint"],
            "executor_fingerprint_sha256": run[
                "executor_fingerprint_sha256"
            ],
            "trials": trials,
            "decision": (
                decision_row["receipt"] if decision_row is not None else None
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
        token = secrets.token_urlsafe(32)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor, task_id, parent_attempt_id, parent_lease_token
                )
                run = self._assert_probe_run_task(
                    cursor, task_id, probe_run_id
                )
                self._expire_probe_leases(cursor, probe_run_id)
                maximum_attempts = int(
                    run["policy"]["maximum_attempts_per_candidate"]
                )
                while True:
                    cursor.execute(
                        """SELECT t.*,
                                  (SELECT COUNT(*)
                                     FROM segment_probe_attempts a
                                    WHERE a.probe_trial_id=t.probe_trial_id)
                                      AS attempts
                             FROM segment_probe_trials t
                            WHERE t.probe_run_id=%s AND t.state='PENDING'
                              AND (t.retry_after IS NULL
                                   OR t.retry_after<=now())
                              AND (t.gpu_required=false OR %s=true)
                              AND t.minimum_vram_gb<=%s
                            ORDER BY t.candidate_rank
                            LIMIT 1 FOR UPDATE SKIP LOCKED""",
                        (
                            probe_run_id,
                            worker_capabilities["cuda_available"],
                            worker_capabilities["gpu_vram_gb"],
                        ),
                    )
                    pending = cursor.fetchone()
                    if pending is None:
                        return None
                    if int(pending["attempts"]) < maximum_attempts:
                        break
                    cursor.execute(
                        """UPDATE segment_probe_trials
                              SET state='FAILED',result=%s::jsonb,updated_at=now()
                            WHERE probe_trial_id=%s""",
                        (
                            json.dumps(
                                {
                                    "status": "PROBE_ATTEMPTS_EXHAUSTED",
                                    "maximum_attempts": maximum_attempts,
                                    "ink_used": False,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            pending["probe_trial_id"],
                        ),
                    )
                attempt_number = int(pending["attempts"]) + 1
                probe_attempt_id = stable_id(
                    "probe-attempt",
                    {
                        "probe_trial_id": pending["probe_trial_id"],
                        "attempt_number": attempt_number,
                    },
                )
                cursor.execute(
                    """INSERT INTO segment_probe_attempts
                       (probe_attempt_id,probe_trial_id,attempt_number,worker_id,
                        state)
                       VALUES(%s,%s,%s,%s,'CLAIMED')""",
                    (
                        probe_attempt_id,
                        pending["probe_trial_id"],
                        attempt_number,
                        worker_id,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_probe_trials
                          SET state='CLAIMED',worker_id=%s,lease_token_hash=%s,
                              lease_expires_at=%s,retry_after=NULL,
                              active_probe_attempt_id=%s,updated_at=now()
                        WHERE probe_trial_id=%s""",
                    (
                        worker_id,
                        _token_hash(token),
                        _deadline(lease_seconds),
                        probe_attempt_id,
                        pending["probe_trial_id"],
                    ),
                )
                cursor.execute(
                    """UPDATE segment_probe_runs SET state='PROBING',
                              updated_at=now() WHERE probe_run_id=%s""",
                    (probe_run_id,),
                )
                self.event(
                    cursor,
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
                return {
                    "probe_run_id": probe_run_id,
                    "probe_trial_id": pending["probe_trial_id"],
                    "probe_attempt_id": probe_attempt_id,
                    "attempt_number": attempt_number,
                    "candidate_rank": pending["candidate_rank"],
                    "candidate": pending["candidate"],
                    "lease_token": token,
                }

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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    cursor,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                run = self._assert_probe_run_task(
                    cursor, task_id, trial["probe_run_id"]
                )
                from .seed_probe import validate_probe_locked_plan

                authoritative_plan = validate_probe_locked_plan(
                    task=parent_task,
                    trial={
                        "probe_run_id": trial["probe_run_id"],
                        "probe_trial_id": probe_trial_id,
                        "candidate": dict(trial["candidate"]),
                    },
                    policy=dict(run["policy"]),
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
                cursor.execute(
                    """SELECT state FROM segment_probe_attempts
                        WHERE probe_attempt_id=%s FOR UPDATE""",
                    (probe_attempt_id,),
                )
                attempt = cursor.fetchone()
                if attempt is None or attempt["state"] not in {"CLAIMED", "RUNNING"}:
                    raise RuntimeError(
                        "probe attempt cannot enter RUNNING from its current state"
                    )
                cursor.execute(
                    """UPDATE segment_probe_trials
                          SET state='RUNNING',locked_plan=%s::jsonb,
                              locked_plan_sha256=%s,updated_at=now()
                        WHERE probe_trial_id=%s""",
                    (
                        json.dumps(
                            authoritative_plan,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        locked_plan_sha,
                        probe_trial_id,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_probe_attempts
                          SET state='RUNNING',updated_at=now()
                        WHERE probe_attempt_id=%s""",
                    (probe_attempt_id,),
                )

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
        expected_files = {
            "x.tif",
            "y.tif",
            "z.tif",
            "generations.tif",
            "meta.json",
        }
        files = manifest.get("files")
        valid_files = (
            isinstance(files, dict)
            and set(files) == expected_files
            and all(
                isinstance(entry, dict)
                and set(entry) == {"sha256", "size_bytes"}
                and isinstance(entry["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None
                and isinstance(entry["size_bytes"], int)
                and not isinstance(entry["size_bytes"], bool)
                and entry["size_bytes"] > 0
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
            or content_sha256(files) != manifest.get("artifact_sha256")
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    cursor,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                if manifest.get("probe_run_id") != trial["probe_run_id"]:
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
                    cursor, task_id, trial["probe_run_id"]
                )
                from .seed_probe import expected_probe_artifact_uri

                locked_plan = dict(trial["locked_plan"] or {})
                expected_artifact_uri = expected_probe_artifact_uri(
                    namespace_identity=dict(
                        (dict(run["executor_fingerprint"] or {})).get(
                            "probe_namespace", {}
                        )
                    ),
                    sample_id=str(locked_plan.get("sample_id") or ""),
                    probe_run_id=trial["probe_run_id"],
                    probe_trial_id=probe_trial_id,
                    artifact_sha256=str(manifest["artifact_sha256"]),
                )
                cursor.execute(
                    """SELECT * FROM segment_probe_artifact_sets
                        WHERE probe_trial_id=%s FOR UPDATE""",
                    (probe_trial_id,),
                )
                existing = cursor.fetchone()
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
                    cursor.execute(
                        """UPDATE segment_probe_artifact_sets
                              SET artifact_uri=%s
                            WHERE probe_artifact_set_id=%s""",
                        (
                            expected_artifact_uri,
                            existing["probe_artifact_set_id"],
                        ),
                    )
                    cursor.execute(
                        """UPDATE segment_probe_trials SET state='UPLOADED',
                                  updated_at=now() WHERE probe_trial_id=%s""",
                        (probe_trial_id,),
                    )
                    cursor.execute(
                        """UPDATE segment_probe_attempts SET state='UPLOADED',
                                  updated_at=now() WHERE probe_attempt_id=%s""",
                        (probe_attempt_id,),
                    )
                    return str(existing["probe_artifact_set_id"])
                if trial["state"] != "RUNNING":
                    raise RuntimeError(
                        "a probe artifact may only be reserved from RUNNING"
                    )
                cursor.execute(
                    """INSERT INTO segment_probe_artifact_sets
                       (probe_artifact_set_id,probe_trial_id,probe_attempt_id,
                        manifest,manifest_sha256,artifact_uri,state)
                       VALUES(%s,%s,%s,%s::jsonb,%s,%s,'RESERVED')""",
                    (
                        probe_artifact_set_id,
                        probe_trial_id,
                        probe_attempt_id,
                        json.dumps(
                            manifest, sort_keys=True, separators=(",", ":")
                        ),
                        manifest_sha,
                        expected_artifact_uri,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_probe_trials SET state='UPLOADED',
                              updated_at=now() WHERE probe_trial_id=%s""",
                    (probe_trial_id,),
                )
                cursor.execute(
                    """UPDATE segment_probe_attempts SET state='UPLOADED',
                              updated_at=now() WHERE probe_attempt_id=%s""",
                    (probe_attempt_id,),
                )
                return probe_artifact_set_id

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
        profile_sha = evaluation.get("profile_sha256")
        if (
            not isinstance(profile_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", profile_sha) is None
        ):
            raise ValueError(
                "probe evaluation profile_sha256 must be 64 lowercase hex characters"
            )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    cursor,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                if trial["state"] != "UPLOADED":
                    raise RuntimeError(
                        "probe completion requires an uploaded reserved artifact"
                    )
                cursor.execute(
                    """SELECT * FROM segment_probe_artifact_sets
                        WHERE probe_artifact_set_id=%s AND probe_trial_id=%s
                        FOR UPDATE""",
                    (
                        probe_artifact_set_id,
                        probe_trial_id,
                    ),
                )
                artifact = cursor.fetchone()
                if artifact is None:
                    raise RuntimeError(
                        "probe artifact was not reserved by this trial"
                    )
                if artifact["state"] != "RESERVED":
                    raise RuntimeError(
                        "probe completion requires a RESERVED artifact"
                    )
                if artifact["artifact_uri"] != artifact_uri:
                    raise ValueError(
                        "published probe URI differs from the reserved namespace URI"
                    )
                cursor.execute(
                    """SELECT policy,executor_fingerprint FROM segment_probe_runs
                        WHERE probe_run_id=%s FOR UPDATE""",
                    (trial["probe_run_id"],),
                )
                run = cursor.fetchone()
                if run is None:
                    raise RuntimeError("probe trial belongs to a missing run")
                from .seed_probe import validate_probe_completion_evidence

                verdict = validate_probe_completion_evidence(
                    evaluation=evaluation,
                    geometry=geometry,
                    growth_receipt=growth_receipt,
                    artifact_uri=artifact_uri,
                    artifact_manifest=dict(artifact["manifest"]),
                    task_id=task_id,
                    probe_run_id=trial["probe_run_id"],
                    probe_trial_id=probe_trial_id,
                    probe_artifact_set_id=probe_artifact_set_id,
                    locked_plan_sha256=str(
                        trial["locked_plan_sha256"] or ""
                    ),
                    sample_id=str(
                        (dict(trial["locked_plan"] or {})).get("sample_id", "")
                    ),
                    namespace_identity=dict(
                        (dict(run["executor_fingerprint"] or {})).get(
                            "probe_namespace", {}
                        )
                    ),
                    policy=dict(run["policy"]),
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
                cursor.execute(
                    """SELECT * FROM segment_probe_evaluations
                        WHERE probe_trial_id=%s FOR UPDATE""",
                    (probe_trial_id,),
                )
                existing = cursor.fetchone()
                if (
                    existing is not None
                    and existing["result_sha256"] != evaluation_sha
                ):
                    raise RuntimeError(
                        "the same probe artifact received different evaluations"
                    )
                if existing is None:
                    cursor.execute(
                        """INSERT INTO segment_probe_evaluations
                           (evaluation_id,probe_trial_id,probe_artifact_set_id,
                            profile_id,profile_sha256,verdict,result,
                            result_sha256)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s)""",
                        (
                            evaluation_id,
                            probe_trial_id,
                            probe_artifact_set_id,
                            evaluation["profile_id"],
                            profile_sha,
                            verdict,
                            json.dumps(
                                evaluation,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            evaluation_sha,
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
                result_json = json.dumps(
                    result, sort_keys=True, separators=(",", ":")
                )
                cursor.execute(
                    """UPDATE segment_probe_artifact_sets
                          SET artifact_uri=%s,state='AVAILABLE'
                        WHERE probe_artifact_set_id=%s""",
                    (artifact_uri, probe_artifact_set_id),
                )
                cursor.execute(
                    """UPDATE segment_probe_attempts
                          SET state='COMPLETED',growth_receipt=%s::jsonb,
                              result=%s::jsonb,updated_at=now()
                        WHERE probe_attempt_id=%s""",
                    (
                        json.dumps(
                            growth_receipt,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        result_json,
                        probe_attempt_id,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_probe_trials
                          SET state=%s,result=%s::jsonb,worker_id=NULL,
                              lease_token_hash=NULL,lease_expires_at=NULL,
                              active_probe_attempt_id=NULL,updated_at=now()
                        WHERE probe_trial_id=%s""",
                    (state_for[verdict], result_json, probe_trial_id),
                )
                cursor.execute(
                    """SELECT COUNT(*) AS count FROM segment_probe_trials
                        WHERE probe_run_id=%s AND state<>ALL(%s)""",
                    (
                        trial["probe_run_id"],
                        ["SUCCEEDED", "REJECTED", "UNMEASURED", "FAILED"],
                    ),
                )
                if int(cursor.fetchone()["count"]) == 0:
                    cursor.execute(
                        """UPDATE segment_probe_runs
                              SET state='READY_TO_DECIDE',updated_at=now()
                            WHERE probe_run_id=%s""",
                        (trial["probe_run_id"],),
                    )
                self.event(
                    cursor,
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
                return result

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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor, task_id, parent_attempt_id, parent_lease_token
                )
                trial = self._assert_probe_trial_owner(
                    cursor,
                    task_id,
                    probe_trial_id,
                    probe_attempt_id,
                    probe_lease_token,
                )
                cursor.execute(
                    """SELECT policy FROM segment_probe_runs
                        WHERE probe_run_id=%s FOR UPDATE""",
                    (trial["probe_run_id"],),
                )
                run = cursor.fetchone()
                if run is None:
                    raise RuntimeError("probe run disappeared")
                cursor.execute(
                    """SELECT COUNT(*) AS count FROM segment_probe_attempts
                        WHERE probe_trial_id=%s""",
                    (probe_trial_id,),
                )
                attempts = int(cursor.fetchone()["count"])
                maximum = int(
                    run["policy"]["maximum_attempts_per_candidate"]
                )
                will_retry = bool(retryable and attempts < maximum)
                attempt_state = (
                    "RETRY_ON_LARGER_GPU"
                    if result.get("status") == "RETRY_ON_LARGER_GPU"
                    else "GROW_FAILED"
                )
                result_json = json.dumps(
                    result, sort_keys=True, separators=(",", ":")
                )
                cursor.execute(
                    """UPDATE segment_probe_attempts
                          SET state=%s,result=%s::jsonb,updated_at=now()
                        WHERE probe_attempt_id=%s""",
                    (attempt_state, result_json, probe_attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_probe_trials
                          SET state=%s,result=%s::jsonb,worker_id=NULL,
                              lease_token_hash=NULL,lease_expires_at=NULL,
                              active_probe_attempt_id=NULL,updated_at=now()
                        WHERE probe_trial_id=%s""",
                    (
                        "PENDING" if will_retry else "FAILED",
                        result_json,
                        probe_trial_id,
                    ),
                )
                self.event(
                    cursor,
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
                return {
                    "retry": will_retry,
                    "attempts": attempts,
                    "maximum": maximum,
                }

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
        receipt_sha = content_sha256(decision)
        from .seed_probe import decide_probe_run

        canonical_decision = decide_probe_run(self.probe_run(probe_run_id))
        if content_sha256(canonical_decision) != receipt_sha:
            raise RuntimeError(
                "seed probe decision does not match canonical run evidence"
            )
        decision_id = stable_id(
            "probe-decision",
            {
                "probe_run_id": probe_run_id,
                "receipt_sha256": receipt_sha,
            },
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor, task_id, parent_attempt_id, parent_lease_token
                )
                run_row = self._assert_probe_run_task(
                    cursor, task_id, probe_run_id
                )
                cursor.execute(
                    """SELECT * FROM segment_probe_decisions
                        WHERE probe_run_id=%s FOR UPDATE""",
                    (probe_run_id,),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["receipt_sha256"] != receipt_sha:
                        raise RuntimeError(
                            "probe run already has a different immutable decision"
                        )
                    return dict(existing["receipt"])
                cursor.execute(
                    """SELECT probe_trial_id,state
                         FROM segment_probe_trials
                        WHERE probe_run_id=%s
                        ORDER BY candidate_rank FOR UPDATE""",
                    (probe_run_id,),
                )
                trials = cursor.fetchall()
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
                cursor.execute(
                    """INSERT INTO segment_probe_decisions
                       (decision_id,probe_run_id,policy_id,policy_sha256,
                        evidence_set_sha256,action,winner_trial_id,receipt,
                        receipt_sha256)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)""",
                    (
                        decision_id,
                        probe_run_id,
                        decision["policy_id"],
                        decision["policy_sha256"],
                        decision["evidence_set_sha256"],
                        decision["action"],
                        winner,
                        json.dumps(
                            decision, sort_keys=True, separators=(",", ":")
                        ),
                        receipt_sha,
                    ),
                )
                state_for = {
                    "CONTINUE_WINNER": "CONTINUATION_QUEUED",
                    "HUMAN_REVIEW": "HUMAN_REVIEW",
                    "REJECT_ALL": "REJECTED",
                }
                run_state = (
                    "SHADOW_COMPLETE"
                    if run_row["policy"]["mode"] == "shadow"
                    else state_for[decision["action"]]
                )
                cursor.execute(
                    """UPDATE segment_probe_runs SET state=%s,updated_at=now()
                        WHERE probe_run_id=%s""",
                    (run_state, probe_run_id),
                )
                mode = run_row["policy"]["mode"]
                if mode == "shadow":
                    cursor.execute(
                        """UPDATE segment_probe_artifact_sets
                              SET state='BENCHMARK_RETAINED',
                                  retain_until=NULL
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id
                                    FROM segment_probe_trials
                                   WHERE probe_run_id=%s)""",
                        (probe_run_id,),
                    )
                elif decision["action"] == "HUMAN_REVIEW":
                    cursor.execute(
                        """UPDATE segment_probe_artifact_sets
                              SET state='REVIEW_RETAINED',retain_until=NULL
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id
                                    FROM segment_probe_trials
                                   WHERE probe_run_id=%s)""",
                        (probe_run_id,),
                    )
                else:
                    cursor.execute(
                        """UPDATE segment_probe_artifact_sets
                              SET state='LOSER_RETAINED',
                                  retain_until=now() + interval '30 days'
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id
                                   FROM segment_probe_trials
                                   WHERE probe_run_id=%s)
                              AND (%s::text IS NULL OR probe_trial_id<>%s)""",
                        (probe_run_id, winner, winner),
                    )
                    if winner is not None:
                        cursor.execute(
                            """UPDATE segment_probe_artifact_sets
                                  SET state='WINNER_RETAINED',
                                      retain_until=NULL
                                WHERE probe_trial_id=%s""",
                            (winner,),
                        )
                self.event(
                    cursor,
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
                return decision

    def begin_probe_promotion(
        self,
        task_id: str,
        parent_attempt_id: str,
        parent_lease_token: str,
        probe_run_id: str,
        locked_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Durably link one selected probe to the ordinary full-grow boundary."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor,
                    task_id,
                    parent_attempt_id,
                    parent_lease_token,
                )
                self._assert_probe_run_task(
                    cursor, task_id, probe_run_id
                )
                cursor.execute(
                    """SELECT d.decision_id,d.receipt_sha256,d.winner_trial_id,
                              d.action,r.task_id,r.policy,
                              r.source_snapshot_id,t.candidate,
                              a.probe_artifact_set_id,a.manifest_sha256,
                              a.artifact_uri
                         FROM segment_probe_decisions d
                         JOIN segment_probe_runs r
                           ON r.probe_run_id=d.probe_run_id
                         JOIN segment_probe_trials t
                           ON t.probe_trial_id=d.winner_trial_id
                         JOIN segment_probe_artifact_sets a
                           ON a.probe_trial_id=d.winner_trial_id
                        WHERE d.probe_run_id=%s AND r.task_id=%s
                        FOR UPDATE OF d,r,t,a""",
                    (probe_run_id, task_id),
                )
                row = cursor.fetchone()
                if (
                    row is None
                    or row["task_id"] != task_id
                    or row["action"] != "CONTINUE_WINNER"
                    or row["policy"]["mode"] != "select"
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
                    winner_candidate=dict(row["candidate"]),
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
                cursor.execute(
                    """SELECT * FROM segment_probe_promotions
                        WHERE decision_id=%s FOR UPDATE""",
                    (row["decision_id"],),
                )
                existing = cursor.fetchone()
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
                        return dict(existing["receipt"])
                    if existing["state"] != "CONTINUING":
                        raise RuntimeError(
                            "probe promotion is not open for continuation"
                        )
                    cursor.execute(
                        """UPDATE segment_probe_promotions
                              SET continuation_attempt_id=%s,
                                  continuation_locked_plan_sha256=%s,
                                  receipt=%s::jsonb,receipt_sha256=%s,
                                  updated_at=now()
                            WHERE promotion_id=%s""",
                        (
                            parent_attempt_id,
                            continuation_locked_plan_sha,
                            json.dumps(
                                receipt,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            receipt_sha,
                            existing["promotion_id"],
                        ),
                    )
                    return receipt
                cursor.execute(
                    """INSERT INTO segment_probe_promotions
                       (promotion_id,decision_id,winner_trial_id,
                        winner_probe_artifact_set_id,continuation_task_id,
                        continuation_attempt_id,
                        continuation_contract_sha256,
                        continuation_locked_plan_sha256,
                        state,receipt,receipt_sha256)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,
                              'CONTINUING',%s::jsonb,%s)""",
                    (
                        promotion_id,
                        row["decision_id"],
                        row["winner_trial_id"],
                        row["probe_artifact_set_id"],
                        task_id,
                        parent_attempt_id,
                        continuation_contract_sha,
                        continuation_locked_plan_sha,
                        json.dumps(
                            receipt, sort_keys=True, separators=(",", ":")
                        ),
                        receipt_sha,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_probe_artifact_sets
                          SET state='WINNER_RETAINED'
                        WHERE probe_artifact_set_id=%s""",
                    (row["probe_artifact_set_id"],),
                )
                cursor.execute(
                    """UPDATE segment_probe_runs SET state='CONTINUING',
                              updated_at=now() WHERE probe_run_id=%s""",
                    (probe_run_id,),
                )
                self.event(
                    cursor,
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
                return receipt

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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(
                    cursor,
                    task_id,
                    parent_attempt_id,
                    parent_lease_token,
                )
                run = self._assert_probe_run_task(
                    cursor, task_id, probe_run_id
                )
                cursor.execute(
                    """SELECT action,winner_trial_id
                         FROM segment_probe_decisions
                        WHERE probe_run_id=%s""",
                    (probe_run_id,),
                )
                decision = cursor.fetchone()
                cursor.execute(
                    """SELECT 1 FROM segment_probe_promotions p
                         JOIN segment_probe_decisions d
                           ON d.decision_id=p.decision_id
                        WHERE d.probe_run_id=%s""",
                    (probe_run_id,),
                )
                promotion = cursor.fetchone()
                if (
                    run["state"] != "CONTINUATION_QUEUED"
                    or decision is None
                    or decision["action"] != "CONTINUE_WINNER"
                    or promotion is not None
                ):
                    raise RuntimeError(
                        "only an unpromoted queued winner can enter review"
                    )
                cursor.execute(
                    """UPDATE segment_attempts
                          SET state='PROBE_REVIEW_PENDING',
                              result=%s::jsonb,updated_at=now()
                        WHERE attempt_id=%s""",
                    (
                        json.dumps(
                            receipt,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        parent_attempt_id,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_tasks
                          SET state='PROBE_REVIEW_PENDING',worker_id=NULL,
                              lease_token_hash=NULL,lease_expires_at=NULL,
                              updated_at=now()
                        WHERE task_id=%s""",
                    (task_id,),
                )
                cursor.execute(
                    """UPDATE segment_probe_runs SET state='REVIEW_PENDING',
                              updated_at=now() WHERE probe_run_id=%s""",
                    (probe_run_id,),
                )
                cursor.execute(
                    """UPDATE segment_probe_artifact_sets
                          SET state='REVIEW_RETAINED',retain_until=NULL
                        WHERE probe_trial_id IN (
                              SELECT probe_trial_id
                                FROM segment_probe_trials
                               WHERE probe_run_id=%s)""",
                    (probe_run_id,),
                )
                self.event(
                    cursor,
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
                    cursor,
                    "STATE_PROBE_REVIEW_PENDING",
                    receipt,
                    task_id,
                    parent_attempt_id,
                )

    def probe_status(self) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT state,COUNT(*) AS count
                         FROM segment_probe_runs GROUP BY state"""
                )
                runs = {
                    row["state"]: int(row["count"]) for row in cursor.fetchall()
                }
                cursor.execute(
                    """SELECT state,COUNT(*) AS count
                         FROM segment_probe_trials GROUP BY state"""
                )
                trials = {
                    row["state"]: int(row["count"]) for row in cursor.fetchall()
                }
                cursor.execute(
                    """SELECT action,COUNT(*) AS count
                         FROM segment_probe_decisions GROUP BY action"""
                )
                decisions = {
                    row["action"]: int(row["count"])
                    for row in cursor.fetchall()
                }
                cursor.execute(
                    """SELECT state,COUNT(*) AS count
                         FROM segment_probe_promotions GROUP BY state"""
                )
                promotions = {
                    row["state"]: int(row["count"])
                    for row in cursor.fetchall()
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_tasks SET lease_expires_at=%s,updated_at=now()
                       WHERE task_id=%s AND active_attempt_id=%s AND lease_token_hash=%s AND state=ANY(%s)""",
                    (_deadline(lease_seconds), task_id, attempt_id, _token_hash(lease_token), list(ACTIVE_STATES)),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("lease heartbeat rejected")
                cursor.execute("UPDATE segment_attempts SET updated_at=now() WHERE attempt_id=%s", (attempt_id,))

    def transition(self, task_id: str, attempt_id: str, lease_token: str, state: str, *, proposal: dict[str, Any] | None = None, locked_plan: dict[str, Any] | None = None, result: dict[str, Any] | None = None) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(cursor, task_id, attempt_id, lease_token)
                assignments = ["state=%s", "updated_at=now()"]
                values: list[Any] = [state]
                if proposal is not None:
                    assignments.extend(["proposal=%s::jsonb", "proposal_sha256=%s"])
                    values.extend([json.dumps(proposal, sort_keys=True, separators=(",", ":")), content_sha256(proposal)])
                if locked_plan is not None:
                    assignments.extend(["locked_plan=%s::jsonb", "locked_plan_sha256=%s"])
                    values.extend([json.dumps(locked_plan, sort_keys=True, separators=(",", ":")), content_sha256(locked_plan)])
                if result is not None:
                    assignments.append("result=%s::jsonb")
                    values.append(json.dumps(result, sort_keys=True, separators=(",", ":")))
                values.append(attempt_id)
                cursor.execute(f"UPDATE segment_attempts SET {','.join(assignments)} WHERE attempt_id=%s", values)
                cursor.execute("UPDATE segment_tasks SET state=%s,updated_at=now() WHERE task_id=%s", (state, task_id))
                self.event(cursor, f"STATE_{state}", result or {}, task_id, attempt_id)

    def add_artifact_set(self, task_id: str, attempt_id: str, lease_token: str, manifest: dict[str, Any], staging_uri: str) -> str:
        artifact_id = stable_id("artifact-set", {"attempt_id": attempt_id, "manifest": manifest})
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(cursor, task_id, attempt_id, lease_token)
                cursor.execute(
                    """INSERT INTO segment_artifact_sets
                       (artifact_set_id,attempt_id,manifest,manifest_sha256,staging_uri,state)
                       VALUES(%s,%s,%s::jsonb,%s,%s,'UPLOADED')""",
                    (
                        artifact_id,
                        attempt_id,
                        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                        content_sha256(manifest),
                        staging_uri,
                    ),
                )
                cursor.execute("UPDATE segment_attempts SET state='UPLOADED',result=%s::jsonb,updated_at=now() WHERE attempt_id=%s", (json.dumps({"artifact_set_id": artifact_id}), attempt_id))
                cursor.execute("UPDATE segment_tasks SET state='UPLOADED',updated_at=now() WHERE task_id=%s", (task_id,))
                self.event(cursor, "STATE_UPLOADED", {"artifact_set_id": artifact_id}, task_id, attempt_id)
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
        try:
            with self.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """SELECT t.state AS task_state,t.source_snapshot_id,
                                  t.payload,t.seed_probe_required,
                                  t.active_attempt_id,
                                  a.task_id AS attempt_task_id,
                                  a.state AS attempt_state,
                                  a.locked_plan,a.locked_plan_sha256,a.result,
                                  s.attempt_id AS artifact_attempt_id,
                                  s.state AS artifact_state,
                                  s.manifest,s.manifest_sha256
                             FROM segment_tasks t
                             JOIN segment_attempts a ON a.attempt_id=%s
                             JOIN segment_artifact_sets s
                               ON s.artifact_set_id=%s
                            WHERE t.task_id=%s
                            FOR UPDATE OF t,a,s""",
                        (attempt_id, artifact_set_id, task_id),
                    )
                    context = cursor.fetchone()
                    if (
                        context is None
                        or context["attempt_task_id"] != task_id
                    ):
                        raise RuntimeError(
                            "finalization task, attempt, or artifact set does not exist"
                        )
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
                        task_sample_id=str(
                            (dict(context["payload"] or {})).get("sample_id")
                            or ""
                        ),
                        attempt_locked_plan_sha256=context[
                            "locked_plan_sha256"
                        ],
                        artifact_attempt_id=context["artifact_attempt_id"],
                        artifact_state=context["artifact_state"],
                        artifact_manifest=dict(context["manifest"]),
                        artifact_manifest_sha256=context["manifest_sha256"],
                        surface=surface,
                        replay=replay,
                    )
                    locked_plan = dict(context["locked_plan"] or {})
                    cursor.execute(
                        """SELECT r.probe_run_id,r.state AS run_state,
                                  r.policy,d.decision_id,d.action,
                                  d.winner_trial_id,d.receipt,d.receipt_sha256
                             FROM segment_probe_runs r
                             LEFT JOIN segment_probe_decisions d
                               ON d.probe_run_id=r.probe_run_id
                            WHERE r.task_id=%s
                            FOR UPDATE OF r""",
                        (task_id,),
                    )
                    probe_authority = cursor.fetchall()
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
                            "policy": dict(authority_row["policy"] or {}),
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
                            "receipt": dict(
                                authority_row["receipt"] or {}
                            ),
                            "receipt_sha256": authority_row[
                                "receipt_sha256"
                            ],
                        }
                        if authority_row is not None
                        and authority_row["decision_id"] is not None
                        else None
                    )
                    cursor.execute(
                        """SELECT p.*,d.probe_run_id
                             FROM segment_probe_promotions p
                             JOIN segment_probe_decisions d
                               ON d.decision_id=p.decision_id
                            WHERE p.continuation_task_id=%s
                            FOR UPDATE OF p""",
                        (task_id,),
                    )
                    promotion = cursor.fetchone()
                    from .seed_probe import (
                        validate_probe_finalization_authority,
                    )

                    validate_probe_finalization_authority(
                        task_payload=dict(context["payload"] or {}),
                        seed_probe_required=bool(
                            context["seed_probe_required"]
                        ),
                        probe_run=probe_run,
                        probe_decision=probe_decision,
                        promotion=(
                            dict(promotion)
                            if promotion is not None
                            else None
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
                            or promotion[
                                "continuation_locked_plan_sha256"
                            ]
                            != context["locked_plan_sha256"]
                        ):
                            raise RuntimeError(
                                "probe promotion is not bound to this final "
                                "locked plan"
                            )
                        promotion_receipt = dict(
                            promotion["receipt"] or {}
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
                            or promotion_receipt.get(
                                "continuation_attempt_id"
                            )
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
                        prior = dict(context["result"] or {})
                        if (
                            prior.get("artifact_set_id") != artifact_set_id
                            or prior.get("surface_id")
                            != surface.get("surface_id")
                        ):
                            raise RuntimeError(
                                "finalization replay differs from the terminal result"
                            )
                        return {
                            "status": context["task_state"],
                            "duplicate_of": prior.get("duplicate_of"),
                            "duplicate_diagnostics": prior.get(
                                "duplicate_diagnostics", {}
                            ),
                            "geometry_qc_state": prior.get(
                                "geometry_qc_state"
                            ),
                            "geometry_blocked_qc": bool(
                                prior.get("geometry_blocked_qc", False)
                            ),
                            "probe_promotion_id": (
                                promotion["promotion_id"]
                                if promotion is not None
                                else None
                            ),
                        }
                    self._assert_owner(cursor, task_id, attempt_id, lease_token)
                    # Serialize finalizers for one immutable source snapshot.
                    cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (surface["source_snapshot_id"],))
                    cursor.execute(
                        "SELECT surface_id,artifact_sha256,sample_points FROM segment_surfaces WHERE source_snapshot_id=%s ORDER BY surface_id",
                        (surface["source_snapshot_id"],),
                    )
                    duplicate_of, duplicate_diagnostics = find_duplicate_in_surfaces(
                        cursor.fetchall(),
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
                        raise ValueError(
                            f"unsupported geometry QC state: {geometry_state}"
                        )
                    geometry_rejected = is_geometry_rejected(geometry_state)
                    if not duplicate_of:
                        surface_state = "FIXTURE_ONLY" if fixture_only else "QC_PENDING"
                        physical_state = (
                            "NOT_APPLICABLE_FIXTURE"
                            if fixture_only
                            else "UNVALIDATED"
                        )
                        cursor.execute(
                            """INSERT INTO segment_surfaces
                               (surface_id,source_snapshot_id,sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz,sample_points,area_cm2,state,physical_qc_state,geometry_qc_state,payload)
                               VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s::jsonb)""",
                            (
                                surface_id,
                                surface["source_snapshot_id"],
                                surface["sample_id"],
                                surface.get("owner", "campaign-x"),
                                surface["artifact_sha256"],
                                surface["artifact_uri"],
                                json.dumps(surface["bbox_xyz"]),
                                json.dumps(surface.get("sample_points")) if surface.get("sample_points") is not None else None,
                                surface.get("area_cm2"),
                                surface_state,
                                physical_state,
                                geometry_state,
                                json.dumps(surface, sort_keys=True, separators=(",", ":")),
                            ),
                        )
                        if not fixture_only:
                            qc_id = stable_id(
                                "qc-job",
                                {"surface_id": surface_id, "profile_id": qc_profile_id},
                            )
                            # Three states, not two. A hard geometric defect
                            # keeps an auditable FAILED row; a certified surface
                            # is claimable; an unmeasured one waits. claim_qc
                            # takes only PENDING, so "unmeasured" used to mean
                            # claimable -- the ink/CT adapter could spend model
                            # time on a surface the geometry gate had not passed.
                            cursor.execute(
                                """INSERT INTO segment_qc_jobs(qc_job_id,surface_id,profile_id,state,payload,result,updated_at)
                                   VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb,now())""",
                                (
                                    qc_id,
                                    surface_id,
                                    qc_profile_id,
                                    qc_job_state_for(geometry_state),
                                    json.dumps({"artifact_set_id": artifact_set_id}),
                                    json.dumps(
                                        {
                                            "schema": "campaignx.segment_qc_geometry_block.v1",
                                            "geometry_qc_state": geometry_state,
                                            "geometry_certification": surface.get(
                                                "geometry_certification"
                                            ),
                                            "no_scientific_conclusion": True,
                                        },
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                    if geometry_rejected
                                    else None,
                                ),
                            )
                            if geometry_rejected:
                                self.event(
                                    cursor,
                                    "GEOMETRY_REJECTED_BEFORE_MODEL",
                                    {
                                        "surface_id": surface_id,
                                        "qc_job_id": qc_id,
                                        "geometry_qc_state": geometry_state,
                                    },
                                    task_id,
                                    attempt_id,
                                )
                    cursor.execute(
                        """UPDATE segment_artifact_sets SET state=%s
                            WHERE artifact_set_id=%s AND attempt_id=%s
                              AND state='UPLOADED'""",
                        (
                            "DUPLICATE" if duplicate_of else "PROMOTED",
                            artifact_set_id,
                            attempt_id,
                        ),
                    )
                    if cursor.rowcount != 1:
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
                            **dict(promotion["receipt"]),
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
                        cursor.execute(
                            """UPDATE segment_probe_promotions
                                  SET canonical_artifact_set_id=%s,surface_id=%s,
                                      state='PROMOTED',receipt=%s::jsonb,
                                      receipt_sha256=%s,updated_at=now()
                                WHERE promotion_id=%s""",
                            (
                                canonical_artifact_set_id,
                                canonical_surface_id,
                                json.dumps(
                                    promotion_receipt,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                content_sha256(promotion_receipt),
                                promotion_id,
                            ),
                        )
                        cursor.execute(
                            """UPDATE segment_probe_runs SET state='PROMOTED',
                                      updated_at=now() WHERE probe_run_id=%s""",
                            (promotion["probe_run_id"],),
                        )
                        cursor.execute(
                            """UPDATE segment_probe_artifact_sets
                                  SET state='PROMOTED_RETAINED',
                                      retain_until=now() + interval '30 days'
                                WHERE probe_artifact_set_id=%s""",
                            (promotion["winner_probe_artifact_set_id"],),
                        )
                        self.event(
                            cursor,
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
                    cursor.execute(
                        "UPDATE segment_attempts SET state=%s,result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                        (state, json.dumps(result, sort_keys=True, separators=(",", ":")), attempt_id),
                    )
                    cursor.execute(
                        """UPDATE segment_tasks SET state=%s,worker_id=NULL,lease_token_hash=NULL,
                           lease_expires_at=NULL,updated_at=now() WHERE task_id=%s""",
                        (state, task_id),
                    )
                    self.event(cursor, f"STATE_{state}", result, task_id, attempt_id)
                    return {
                        "status": state,
                        "duplicate_of": duplicate_of,
                        "duplicate_diagnostics": duplicate_diagnostics,
                        "geometry_qc_state": geometry_state,
                        "geometry_blocked_qc": bool(
                            geometry_rejected and not duplicate_of
                        ),
                        "probe_promotion_id": promotion_id,
                    }
        except Exception as error:
            try:
                import psycopg2
            except ImportError:  # pragma: no cover
                raise
            if isinstance(error, psycopg2.IntegrityError):
                raise RuntimeError(f"surface finalization violated catalogue uniqueness: {error}") from error
            raise

    def claim_qc(self, worker_id: str, lease_seconds: int, profile_id: str | None = None) -> dict[str, Any] | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        token = secrets.token_urlsafe(32)
        token_hash = _token_hash(token)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,updated_at=now()
                       WHERE state='CLAIMED' AND lease_expires_at IS NOT NULL AND lease_expires_at<=now()"""
                )
                query = """SELECT * FROM segment_qc_jobs
                           WHERE state='PENDING' AND (retry_after IS NULL OR retry_after<=now())"""
                arguments: list[Any] = []
                if profile_id is not None:
                    query += " AND profile_id=%s"
                    arguments.append(profile_id)
                query += " ORDER BY created_at,qc_job_id FOR UPDATE SKIP LOCKED LIMIT 1"
                cursor.execute(query, arguments)
                job = cursor.fetchone()
                if job is None:
                    return None
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='CLAIMED',worker_id=%s,lease_token_hash=%s,
                       lease_expires_at=%s,retry_after=NULL,updated_at=now() WHERE qc_job_id=%s""",
                    (worker_id, token_hash, _deadline(lease_seconds), job["qc_job_id"]),
                )
                cursor.execute("SELECT * FROM segment_surfaces WHERE surface_id=%s", (job["surface_id"],))
                surface = cursor.fetchone()
                cursor.execute("SELECT * FROM segment_source_snapshots WHERE source_snapshot_id=%s", (surface["source_snapshot_id"],))
                source = cursor.fetchone()
                self.event(cursor, "QC_CLAIMED", {"qc_job_id": job["qc_job_id"], "surface_id": job["surface_id"], "worker_id": worker_id})
                surface_value = dict(surface["payload"])
                surface_value.update({
                    "surface_id": surface["surface_id"],
                    "artifact_uri": surface["artifact_uri"],
                    "artifact_sha256": surface["artifact_sha256"],
                    "bbox_xyz": surface["bbox_xyz"],
                    "state": surface["state"],
                })
                return {
                    "qc_job_id": job["qc_job_id"],
                    "surface_id": job["surface_id"],
                    "profile_id": job["profile_id"],
                    "payload": dict(job["payload"]),
                    "worker_id": worker_id,
                    "lease_token": token,
                    "surface": surface_value,
                    "source": self._snapshot(source),
                }

    def heartbeat_qc(self, qc_job_id: str, lease_token: str, lease_seconds: int) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_qc_jobs SET lease_expires_at=%s,updated_at=now()
                       WHERE qc_job_id=%s AND state='CLAIMED' AND lease_token_hash=%s
                         AND lease_expires_at IS NOT NULL AND lease_expires_at>now()""",
                    (_deadline(lease_seconds), qc_job_id, _token_hash(lease_token)),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("QC lease heartbeat rejected")

    def finalize_qc(self, qc_job_id: str, lease_token: str, outcome: str, result: dict[str, Any]) -> dict[str, Any]:
        if outcome not in QC_OUTCOME_STATES:
            raise ValueError(f"unsupported QC outcome: {outcome}")
        surface_state, physical_state = QC_OUTCOME_STATES[outcome]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM segment_qc_jobs WHERE qc_job_id=%s FOR UPDATE", (qc_job_id,))
                job = cursor.fetchone()
                if (
                    job is None
                    or job["state"] != "CLAIMED"
                    or job["lease_token_hash"] != _token_hash(lease_token)
                    or job["lease_expires_at"] is None
                    or job["lease_expires_at"] <= datetime.now(timezone.utc)
                ):
                    raise RuntimeError("QC finalization belongs to a stale lease")
                validate_qc_result_contract(job["surface_id"], outcome, result)
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='COMPLETED',result=%s::jsonb,worker_id=NULL,
                       lease_token_hash=NULL,lease_expires_at=NULL,retry_after=NULL,updated_at=now()
                       WHERE qc_job_id=%s""",
                    (json.dumps(result, sort_keys=True, separators=(",", ":")), qc_job_id),
                )
                cursor.execute(
                    "UPDATE segment_surfaces SET state=%s,physical_qc_state=%s WHERE surface_id=%s",
                    (surface_state, physical_state, job["surface_id"]),
                )
                self.event(cursor, "QC_COMPLETED", {"qc_job_id": qc_job_id, "surface_id": job["surface_id"], "outcome": outcome, "evidence_manifest_sha256": result["evidence_manifest_sha256"]})
                return {"status": "COMPLETED", "qc_job_id": qc_job_id, "surface_id": job["surface_id"], "outcome": outcome, "surface_state": surface_state, "physical_qc_state": physical_state}

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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segment_qc_jobs WHERE qc_job_id=%s FOR UPDATE",
                    (qc_job_id,),
                )
                job = cursor.fetchone()
                if (
                    job is None
                    or job["state"] != "CLAIMED"
                    or job["lease_token_hash"] != _token_hash(lease_token)
                    or job["lease_expires_at"] is None
                    or job["lease_expires_at"] <= datetime.now(timezone.utc)
                ):
                    raise RuntimeError("QC block belongs to a stale lease")
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='BLOCKED_CONFIGURATION',
                       result=%s::jsonb,worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,retry_after=NULL,updated_at=now()
                       WHERE qc_job_id=%s""",
                    (json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                     qc_job_id),
                )
                self.event(
                    cursor,
                    "QC_BLOCKED_CONFIGURATION",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "error": str(receipt.get("error", "")),
                    },
                )
                return {
                    "status": "BLOCKED_CONFIGURATION",
                    "qc_job_id": qc_job_id,
                    "surface_id": job["surface_id"],
                    "error": str(receipt.get("error", "")),
                }

    def requeue_qc_unavailable(
        self,
        qc_job_id: str,
        lease_token: str,
        receipt: dict[str, Any],
        *,
        retry_delay_seconds: int,
    ) -> dict[str, Any]:
        if retry_delay_seconds < 0:
            raise ValueError("retry delay must be non-negative")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segment_qc_jobs WHERE qc_job_id=%s FOR UPDATE",
                    (qc_job_id,),
                )
                job = cursor.fetchone()
                if (
                    job is None
                    or job["state"] != "CLAIMED"
                    or job["lease_token_hash"] != _token_hash(lease_token)
                    or job["lease_expires_at"] is None
                    or job["lease_expires_at"] <= datetime.now(timezone.utc)
                ):
                    raise RuntimeError("QC requeue belongs to a stale lease")
                retry_after = _deadline(retry_delay_seconds)
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='PENDING',result=%s::jsonb,worker_id=NULL,
                       lease_token_hash=NULL,lease_expires_at=NULL,retry_after=%s,updated_at=now()
                       WHERE qc_job_id=%s""",
                    (
                        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                        retry_after,
                        qc_job_id,
                    ),
                )
                self.event(
                    cursor,
                    "QC_REQUEUED_UNAVAILABLE",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "retry_after": retry_after.isoformat(),
                        "error": str(receipt.get("error", "")),
                    },
                )
                return {
                    "status": "RETRYABLE_QC_UNAVAILABLE",
                    "qc_job_id": qc_job_id,
                    "surface_id": job["surface_id"],
                    "retry_after": retry_after.isoformat(),
                }

    # Names a worker may be given. An allowlist rather than free text: this is
    # read straight into a process environment, and a name nobody vetted is a
    # way to set PATH or LD_PRELOAD from a web form.
    FLEET_SECRET_NAMES = (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_ENDPOINT_URL",
    )

    def set_secret(self, name: str, value: str, updated_by: str) -> dict[str, Any]:
        if name not in self.FLEET_SECRET_NAMES:
            raise ValueError(
                f"{name} is not a credential a worker reads; "
                f"allowed: {', '.join(self.FLEET_SECRET_NAMES)}")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO fleet_secrets(name,value,updated_by)
                       VALUES(%s,%s,%s)
                       ON CONFLICT(name) DO UPDATE
                         SET value=EXCLUDED.value, updated_by=EXCLUDED.updated_by,
                             updated_at=now()""",
                    (name, value, updated_by))
                # The name and who set it, never the value -- an event log that
                # quotes a credential is a credential in a second place.
                self.event(cursor, "FLEET_SECRET_SET",
                           {"name": name, "updated_by": updated_by})
        return {"name": name, "updated_by": updated_by}

    def forget_secret(self, name: str) -> bool:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM fleet_secrets WHERE name=%s", (name,))
                removed = cursor.rowcount == 1
                if removed:
                    self.event(cursor, "FLEET_SECRET_FORGOTTEN", {"name": name})
        return removed

    def secret_status(self) -> list[dict[str, Any]]:
        """Which credentials are set, and never what they are."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT name, updated_at, updated_by, length(value) "
                    "FROM fleet_secrets ORDER BY name")
                held = {r["name"]: r for r in cursor.fetchall()}
        return [{"name": name,
                 "set": name in held,
                 "characters": held[name]["length"] if name in held else 0,
                 "updated_at": (held[name]["updated_at"].isoformat()
                                if name in held else None),
                 "updated_by": held[name]["updated_by"] if name in held else None}
                for name in self.FLEET_SECRET_NAMES]

    def secrets(self) -> dict[str, str]:
        """The credentials themselves, for a worker to put in its environment."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT name, value FROM fleet_secrets")
                return {r["name"]: r["value"] for r in cursor.fetchall()
                        if r["name"] in self.FLEET_SECRET_NAMES}

    def surfaces_awaiting_flattening(
        self, profile_id: str, limit: int = 10, sample_id: str | None = None,
        require_physical_qc: bool = True,
        surface_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Admissible surfaces this profile has not flattened yet.

        Certified and not merely finished: P3 consumes a certified surface, and
        flattening an uncertified one produces a flat sheet of whatever the
        surface was -- including one that crossed a lamina -- with the seam
        smoothed out of view.

        Certified and CT-supported, not certified alone. Geometry says the shape
        is a plausible lamina; the physical axis says there is papyrus there.
        Gating on geometry alone let ten surfaces whose CT support was never
        measured through to the detector. `require_physical_qc=False` is the
        deliberate override, for comparing against what the old gate admitted.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = """SELECT s.surface_id, s.sample_id, s.artifact_uri,
                                  s.artifact_sha256, s.geometry_qc_state,
                                  s.physical_qc_state, n.voxel_size_um
                           FROM segment_surfaces s
                           JOIN segment_source_snapshots n
                             ON n.source_snapshot_id = s.source_snapshot_id
                           WHERE s.geometry_qc_state = 'GEOMETRY_CERTIFIED'
                             AND s.artifact_uri IS NOT NULL
                             AND NOT EXISTS (
                               SELECT 1 FROM surface_flattenings f
                               WHERE f.surface_id = s.surface_id
                                 AND f.profile_id = %s
                                 AND f.state = 'FLATTENED')"""
                arguments: list[Any] = [profile_id]
                if require_physical_qc:
                    query += " AND s.physical_qc_state = ANY(%s)"
                    arguments.append(list(ADMISSIBLE_PHYSICAL_QC_STATES))
                if sample_id is not None:
                    query += " AND s.sample_id=%s"
                    arguments.append(sample_id)
                if surface_id is not None:
                    query += " AND s.surface_id=%s"
                    arguments.append(surface_id)
                # Never-attempted first, the same rule P2's backlog already
                # uses. Ordered by created_at alone, a surface whose flattening
                # fails every time sits at the head of the queue forever -- the
                # two whose artifacts were published to a worker's local disk did
                # exactly that, and with --limit 5 they were 40% of every run.
                # Still retried, because a failure can be the network.
                query += (" ORDER BY EXISTS (SELECT 1 FROM surface_flattenings f2"
                          "                   WHERE f2.surface_id = s.surface_id"
                          "                     AND f2.profile_id = %s),"
                          " s.created_at, s.surface_id LIMIT %s")
                arguments.append(profile_id)
                arguments.append(int(limit))
                cursor.execute(query, arguments)
                return [dict(row) for row in cursor.fetchall()]

    def record_flattening(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a flattening, or leave the one already there alone."""
        flattening_id = stable_id(
            "flattening",
            {"surface_id": payload["surface_id"], "profile_id": payload["profile_id"]},
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO surface_flattenings
                       (flattening_id,surface_id,profile_id,state,artifact_uri,
                        artifact_sha256,area_ratio,payload)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       -- A success supersedes an earlier failure, and a success
                       -- never overwrites a success. DO NOTHING let one failed
                       -- attempt hold the identity forever: the retry flattened
                       -- the surface, and the row still said FLATTENING_FAILED.
                       ON CONFLICT(surface_id,profile_id) DO UPDATE
                         SET state=EXCLUDED.state,
                             artifact_uri=EXCLUDED.artifact_uri,
                             artifact_sha256=EXCLUDED.artifact_sha256,
                             area_ratio=EXCLUDED.area_ratio,
                             payload=EXCLUDED.payload,
                             created_at=now()
                         WHERE surface_flattenings.state <> 'FLATTENED'""",
                    (
                        flattening_id,
                        payload["surface_id"],
                        payload["profile_id"],
                        payload.get("state", "FLATTENED"),
                        payload.get("artifact_uri"),
                        payload.get("artifact_sha256"),
                        payload.get("area_ratio"),
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    ),
                )
                inserted = int(cursor.rowcount) == 1
                self.event(
                    cursor,
                    "SURFACE_FLATTENED" if payload.get("state") == "FLATTENED"
                    else "SURFACE_FLATTENING_FAILED",
                    {"surface_id": payload["surface_id"],
                     "profile_id": payload["profile_id"],
                     "state": payload.get("state"),
                     "inserted": inserted},
                )
        return {"flattening_id": flattening_id, "inserted": inserted,
                "state": payload.get("state")}

    def flattenings(self, limit: int = 100, sample_id: str | None = None
                    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = """SELECT f.flattening_id, f.surface_id, f.profile_id, f.state,
                                  f.artifact_uri, f.area_ratio, f.created_at,
                                  s.sample_id
                           FROM surface_flattenings f
                           JOIN segment_surfaces s ON s.surface_id = f.surface_id"""
                arguments: list[Any] = []
                if sample_id is not None:
                    query += " WHERE s.sample_id=%s"
                    arguments.append(sample_id)
                query += " ORDER BY f.created_at DESC LIMIT %s"
                arguments.append(int(limit))
                cursor.execute(query, arguments)
                return [dict(row) for row in cursor.fetchall()]

    def coverage(self, sample_id: str | None = None,
                 mission_id: str | None = None) -> dict[str, Any]:
        """How much of each scroll has been looked at, and with what result.

        The framework is named for exploration and had no way to answer this.
        Coverage existed only as a ranking input -- generate_tasks_for_snapshot
        scores a candidate cell by its distance from the bounding boxes of known
        surfaces -- and was never reported, so "are we making progress" was
        answered with a surface count, which grows whether the fleet is finding
        new ground or re-treading old.

        Counted per grid version, because a grid version is a coverage universe:
        cells under two different grids are not the same cells and adding them
        is meaningless.

        Non-claims
        ----------
        * A cell with a surface is a cell where something grew, not a cell that
          is understood. It says the fleet has been there.
        * Cell counts are not area. Two surfaces in one cell count once here and
          twice in the area total, which is why both are reported.
        * `cells_in_volume` is what the grid would divide this volume into, not
          what anyone intends to attempt. Most of a scroll is not lamina.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                conditions = []
                arguments = []
                if sample_id:
                    conditions.append("n.sample_id = %s")
                    arguments.append(sample_id)
                if mission_id:
                    conditions.append("t.mission_id = %s")
                    arguments.append(mission_id)
                where = ("WHERE " + " AND ".join(conditions)
                         if conditions else "")
                cursor.execute(
                    f"""SELECT n.sample_id, t.grid_version,
                               count(DISTINCT t.cell_id) AS cells_attempted,
                               count(DISTINCT t.cell_id) FILTER (
                                   WHERE t.state = 'NO_SEED') AS cells_no_seed,
                               count(DISTINCT t.cell_id) FILTER (
                                   WHERE t.state = 'QC_PENDING') AS cells_with_surface,
                               count(*) AS tasks,
                               jsonb_agg(DISTINCT t.payload -> 'center_xyz') AS centers
                          FROM segment_tasks t
                          JOIN segment_source_snapshots n
                            ON n.source_snapshot_id = t.source_snapshot_id
                          {where}
                         GROUP BY 1, 2 ORDER BY 1, 2""", arguments)
                # Named, not positional: this cursor yields mappings, and
                # row[0] on one is a KeyError rather than the first column.
                grids = [{"sample_id": row["sample_id"],
                          "grid_version": row["grid_version"],
                          "cells_attempted": int(row["cells_attempted"]),
                          "cells_no_seed": int(row["cells_no_seed"]),
                          "cells_with_surface": int(row["cells_with_surface"]),
                          "tasks": int(row["tasks"]),
                          "grid_step_xyz": _grid_step(row["centers"])}
                         for row in cursor.fetchall()]
                volume_conditions = []
                volume_arguments: list[Any] = []
                if sample_id:
                    volume_conditions.append("n.sample_id = %s")
                    volume_arguments.append(sample_id)
                surface_cte = ""
                surface_relation = "segment_surfaces"
                if mission_id:
                    surface_cte = """WITH mission_surfaces AS (
                        SELECT DISTINCT s.* FROM segment_surfaces s
                        JOIN segment_artifact_sets art
                          ON art.manifest->>'artifact_sha256'=s.artifact_sha256
                        JOIN segment_attempts a ON a.attempt_id=art.attempt_id
                        JOIN segment_tasks t ON t.task_id=a.task_id
                        WHERE t.mission_id=%s
                    ) """
                    surface_relation = "mission_surfaces"
                    volume_arguments.insert(0, mission_id)
                volume_where = ("WHERE " + " AND ".join(volume_conditions)
                                if volume_conditions else "")
                cursor.execute(
                    surface_cte + f"""SELECT n.sample_id, n.shape_xyz, n.voxel_size_um,
                               coalesce(sum(s.area_cm2), 0) AS area_cm2,
                               count(s.surface_id) AS surfaces
                          FROM segment_source_snapshots n
                          LEFT JOIN {surface_relation} s ON s.source_snapshot_id
                                                        = n.source_snapshot_id
                          {volume_where}
                         GROUP BY 1, 2, 3 ORDER BY 1""", volume_arguments)
                volumes = [{"sample_id": row["sample_id"],
                            "shape_xyz": row["shape_xyz"],
                            "voxel_size_um": (float(row["voxel_size_um"])
                                              if row["voxel_size_um"] else None),
                            "surface_area_cm2": round(float(row["area_cm2"]), 2),
                            "surfaces": int(row["surfaces"])}
                           for row in cursor.fetchall()]
        # How many cells of this size the volume holds, from the cell bounds a
        # task carries and the shape the snapshot froze. Derived rather than
        # parsed out of the grid version's name, which is a label.
        shapes = {volume["sample_id"]: volume["shape_xyz"] for volume in volumes}
        for grid in grids:
            total = _cells_in_volume(shapes.get(grid["sample_id"]),
                                     grid["grid_step_xyz"])
            grid["cells_in_volume"] = total
            grid["fraction_attempted"] = (
                round(grid["cells_attempted"] / total, 4) if total else None)
        return {"schema": "campaignx.segment_coverage.v1",
                "grids": grids, "volumes": volumes,
                "non_claims": [
                    "a cell with a surface is a cell the fleet has been to, not "
                    "one that is understood",
                    "surface area double-counts overlap below the deduplication "
                    "threshold, so it is an upper bound on ground covered",
                    "cells under different grid versions are different cells and "
                    "are not added together",
                    "a cell attempted twice under different policies appears in "
                    "more than one outcome, so the outcomes do not sum to the "
                    "cells attempted",
                    "cells_in_volume is what this grid divides the volume into, "
                    "not what anyone intends to attempt: most of a scroll is not "
                    "lamina",
                ]}

    def no_seed_cells(self, *, sample_id: str | None = None,
                      causes: Sequence[str] | None = None,
                      limit: int = 50) -> list[dict[str, Any]]:
        """Cells the planner looked at and found nothing in, with why.

        NO_SEED is the exploration budget's largest single outcome -- 169 of 241
        tasks here -- and the worker records a real diagnosis for each: how many
        candidates the provider offered, and which screen removed them. Nothing
        ever read it back. A cell where the provider offered nothing and a cell
        where eight candidates were all rejected on clearance are different
        problems, and both looked like "no seed" from outside.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = """SELECT t.task_id, t.cell_id, t.grid_version,
                                  t.policy_version, t.source_snapshot_id,
                                  n.sample_id, t.payload, a.result
                             FROM segment_tasks t
                             JOIN segment_source_snapshots n
                               ON n.source_snapshot_id = t.source_snapshot_id
                             LEFT JOIN segment_attempts a
                               ON a.attempt_id = t.active_attempt_id
                            WHERE t.state = 'NO_SEED'"""
                arguments: list[Any] = []
                if sample_id:
                    query += " AND n.sample_id = %s"
                    arguments.append(sample_id)
                if causes:
                    # The worker's own cause vocabulary, matched inside the
                    # attempt's result rather than re-derived here.
                    query += """ AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(
                            coalesce(a.result -> 'primary_causes', '[]'::jsonb)) c
                         WHERE c = ANY(%s))"""
                    arguments.append(list(causes))
                query += " ORDER BY t.created_at, t.task_id LIMIT %s"
                arguments.append(int(limit))
                cursor.execute(query, arguments)
                # Named, like every other read here: this cursor yields
                # mappings and row[0] on one is a KeyError, not the first column.
                return [{"task_id": row["task_id"], "cell_id": row["cell_id"],
                         "grid_version": row["grid_version"],
                         "policy_version": row["policy_version"],
                         "source_snapshot_id": row["source_snapshot_id"],
                         "sample_id": row["sample_id"],
                         "payload": row["payload"],
                         "causes": list((row["result"] or {}).get("primary_causes") or []),
                         "raw_candidate_count": (row["result"] or {}).get(
                             "raw_candidate_count")}
                        for row in cursor.fetchall()]

    def surfaces_without_geometry_verdict(
        self, limit: int = 25, sample_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Surfaces P2 has never given a verdict to.

        GEOMETRY_UNMEASURED and a NULL column mean the same thing here -- one is
        a surface the gate could not measure, the other predates the column --
        and both are surfaces with no verdict, which is what P2 owes them.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = """SELECT surface_id, sample_id, artifact_uri, artifact_sha256,
                                  state, geometry_qc_state
                           FROM segment_surfaces
                           WHERE (geometry_qc_state IS NULL
                                  OR geometry_qc_state = 'GEOMETRY_UNMEASURED')
                             AND artifact_uri IS NOT NULL"""
                arguments: list[Any] = []
                if sample_id is not None:
                    query += " AND sample_id=%s"
                    arguments.append(sample_id)
                # Never-attempted first, so a permanently missing artifact
                # cannot starve the backlog by being retried ahead of surfaces
                # nobody has looked at -- while still being retried, because a
                # fetch failure can be the network rather than the corpus.
                query += (" ORDER BY (payload ? 'geometry_certification'),"
                          " created_at, surface_id LIMIT %s")
                arguments.append(int(limit))
                cursor.execute(query, arguments)
                return [dict(row) for row in cursor.fetchall()]

    def surfaces_on_local_artifacts(self, limit: int = 100,
                                    sample_id: str | None = None) -> list[dict[str, Any]]:
        """Surfaces published to a path instead of to object storage.

        Thirty per cent of this corpus. They were written before the rule that a
        worker's disk is not a place to publish, and they exist on exactly one
        machine: any other host reads their artifact_uri and finds nothing, which
        is how P3 came to fail on three of them with FileNotFoundError.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = ("SELECT surface_id, sample_id, artifact_uri, artifact_sha256 "
                         "FROM segment_surfaces "
                         "WHERE artifact_uri IS NOT NULL "
                         "  AND artifact_uri NOT LIKE 's3://%%' "
                         "  AND artifact_uri NOT LIKE 'http://%%' "
                         "  AND artifact_uri NOT LIKE 'https://%%'")
                arguments: list[Any] = []
                if sample_id:
                    query += " AND sample_id = %s"
                    arguments.append(sample_id)
                query += " ORDER BY created_at, surface_id LIMIT %s"
                arguments.append(int(limit))
                cursor.execute(query, arguments)
                return [dict(row) for row in cursor.fetchall()]

    def repoint_surface_artifact(self, surface_id: str, artifact_uri: str,
                                 artifact_sha256: str) -> dict[str, Any]:
        """Point a surface at where it now lives, keeping what it is.

        The digest is written too, and it is the same digest: republishing moves
        bytes, it does not change them. A different one here would mean the copy
        is not the artifact the verdicts were given to.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE segment_surfaces SET artifact_uri=%s, artifact_sha256=%s "
                    "WHERE surface_id=%s RETURNING surface_id, artifact_uri",
                    (artifact_uri, artifact_sha256, surface_id))
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(surface_id)
                self.event(cursor, "SURFACE_REPUBLISHED",
                           {"surface_id": surface_id, "artifact_uri": artifact_uri})
                return dict(row)

    def record_geometry_certification(
        self,
        surface_id: str,
        geometry_state: str,
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a geometry verdict without touching the ink/CT axis."""

        if geometry_state not in GEOMETRY_QC_STATES:
            raise ValueError(f"unsupported geometry QC state: {geometry_state}")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT surface_id,physical_qc_state FROM segment_surfaces WHERE surface_id=%s FOR UPDATE",
                    (surface_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(f"unknown surface: {surface_id}")
                # The receipt rides with the verdict. Without it "unmeasured"
                # is a word with no reason attached, and the difference between
                # a grid too coarse to measure and an artifact that could not be
                # fetched is only visible in whoever ran it's terminal.
                cursor.execute(
                    """UPDATE segment_surfaces
                       SET geometry_qc_state=%s,
                           payload = payload || %s::jsonb
                       WHERE surface_id=%s""",
                    (geometry_state,
                     json.dumps({"geometry_certification": receipt} if receipt else {},
                                sort_keys=True, separators=(",", ":")),
                     surface_id),
                )
                blocked = 0
                # A verdict arriving is what makes waiting different from
                # stranded: a job held back for unmeasured geometry becomes
                # claimable here, and only here.
                promoted = 0
                if str(geometry_state) == "GEOMETRY_CERTIFIED":
                    cursor.execute(
                        f"""UPDATE segment_qc_jobs SET state='PENDING',updated_at=now()
                            WHERE surface_id=%s AND state='{QC_WAITING_GEOMETRY}'""",
                        (surface_id,),
                    )
                    promoted = cursor.rowcount or 0
                if is_geometry_rejected(geometry_state):
                    cursor.execute(
                        f"""UPDATE segment_qc_jobs SET state='FAILED',result=%s::jsonb,worker_id=NULL,
                           lease_token_hash=NULL,lease_expires_at=NULL,retry_after=NULL,updated_at=now()
                           WHERE surface_id=%s AND state IN ('PENDING','{QC_WAITING_GEOMETRY}')""",
                        (
                            json.dumps(
                                {
                                    "schema": "campaignx.segment_qc_geometry_block.v1",
                                    "geometry_qc_state": geometry_state,
                                    "geometry_certification": receipt,
                                    "no_scientific_conclusion": True,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            surface_id,
                        ),
                    )
                    blocked = int(cursor.rowcount)
                self.event(
                    cursor,
                    "GEOMETRY_CERTIFICATION_RECORDED",
                    {
                        "surface_id": surface_id,
                        "geometry_qc_state": geometry_state,
                        "blocked_qc_jobs": blocked,
                    },
                )
                return {
                    "surface_id": surface_id,
                    "geometry_qc_state": geometry_state,
                    "physical_qc_state": row["physical_qc_state"],
                    "blocked_qc_jobs": blocked,
                }

    def mark_terminal(self, task_id: str, attempt_id: str, lease_token: str, state: str, result: dict[str, Any]) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state}")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(cursor, task_id, attempt_id, lease_token)
                cursor.execute(
                    "UPDATE segment_attempts SET state=%s,result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                    (state, json.dumps(result, sort_keys=True, separators=(",", ":")), attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state=%s,worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,updated_at=now() WHERE task_id=%s""",
                    (state, task_id),
                )
                cursor.execute(
                    """SELECT p.*,d.probe_run_id
                         FROM segment_probe_promotions p
                         JOIN segment_probe_decisions d
                           ON d.decision_id=p.decision_id
                        WHERE p.continuation_task_id=%s
                          AND p.state='CONTINUING'
                        FOR UPDATE OF p""",
                    (task_id,),
                )
                promotion = cursor.fetchone()
                if promotion is not None:
                    failure_receipt = {
                        **dict(promotion["receipt"]),
                        "state": "FAILED",
                        "final_status": state,
                    }
                    cursor.execute(
                        """UPDATE segment_probe_promotions
                              SET state='FAILED',receipt=%s::jsonb,
                                  receipt_sha256=%s,updated_at=now()
                            WHERE promotion_id=%s""",
                        (
                            json.dumps(
                                failure_receipt,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            content_sha256(failure_receipt),
                            promotion["promotion_id"],
                        ),
                    )
                    cursor.execute(
                        """UPDATE segment_probe_runs
                              SET state='PROMOTION_FAILED',updated_at=now()
                            WHERE probe_run_id=%s""",
                        (promotion["probe_run_id"],),
                    )
                    cursor.execute(
                        """UPDATE segment_probe_artifact_sets
                              SET state='REVIEW_RETAINED',retain_until=NULL
                            WHERE probe_trial_id IN (
                                  SELECT probe_trial_id
                                    FROM segment_probe_trials
                                   WHERE probe_run_id=%s)""",
                        (promotion["probe_run_id"],),
                    )
                    self.event(
                        cursor,
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
                    cursor.execute(
                        """SELECT r.probe_run_id,d.winner_trial_id
                             FROM segment_probe_runs r
                             JOIN segment_probe_decisions d
                               ON d.probe_run_id=r.probe_run_id
                            WHERE r.task_id=%s
                              AND r.state='CONTINUATION_QUEUED'
                              AND r.policy->>'mode'='select'
                              AND d.action='CONTINUE_WINNER'
                            FOR UPDATE OF r""",
                        (task_id,),
                    )
                    queued = cursor.fetchone()
                    if queued is not None:
                        cursor.execute(
                            """UPDATE segment_probe_runs
                                  SET state='CONTINUATION_FAILED',
                                      updated_at=now()
                                WHERE probe_run_id=%s""",
                            (queued["probe_run_id"],),
                        )
                        cursor.execute(
                            """UPDATE segment_probe_artifact_sets
                                  SET state='REVIEW_RETAINED',
                                      retain_until=NULL
                                WHERE probe_trial_id IN (
                                      SELECT probe_trial_id
                                        FROM segment_probe_trials
                                       WHERE probe_run_id=%s)""",
                            (queued["probe_run_id"],),
                        )
                        self.event(
                            cursor,
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
                self.event(cursor, f"STATE_{state}", result, task_id, attempt_id)

    def probe_artifacts_due_for_gc(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return only expired bytes whose run reached a safe terminal state."""

        if limit < 1 or limit > 1000:
            raise ValueError("probe GC limit must be between 1 and 1000")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT a.probe_artifact_set_id,a.artifact_uri,
                              a.manifest,a.state,a.retain_until,
                              t.probe_run_id,r.task_id
                         FROM segment_probe_artifact_sets a
                         JOIN segment_probe_trials t
                           ON t.probe_trial_id=a.probe_trial_id
                         JOIN segment_probe_runs r
                           ON r.probe_run_id=t.probe_run_id
                        WHERE a.deleted_at IS NULL
                          AND a.artifact_uri IS NOT NULL
                          AND a.retain_until IS NOT NULL
                          AND a.retain_until<=now()
                          AND a.state IN
                              ('LOSER_RETAINED','PROMOTED_RETAINED')
                          AND r.state IN ('PROMOTED','REJECTED')
                          AND NOT EXISTS (
                              SELECT 1
                                FROM segment_probe_promotions p
                                JOIN segment_probe_decisions d
                                  ON d.decision_id=p.decision_id
                               WHERE d.probe_run_id=r.probe_run_id
                                 AND p.state='CONTINUING')
                        ORDER BY a.retain_until,a.probe_artifact_set_id
                        LIMIT %s""",
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "probe_artifact_set_id": row["probe_artifact_set_id"],
                "artifact_uri": row["artifact_uri"],
                "manifest": row["manifest"],
                "state": row["state"],
                "retain_until": _json_utc_timestamp(row["retain_until"]),
                "probe_run_id": row["probe_run_id"],
                "task_id": row["task_id"],
            }
            for row in rows
        ]

    def mark_probe_artifact_expired(
        self, probe_artifact_set_id: str, artifact_uri: str
    ) -> None:
        """Close the digest ledger after exact external bytes were deleted."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT a.*,t.probe_run_id,r.task_id,
                              r.state AS run_state
                         FROM segment_probe_artifact_sets a
                         JOIN segment_probe_trials t
                           ON t.probe_trial_id=a.probe_trial_id
                         JOIN segment_probe_runs r
                           ON r.probe_run_id=t.probe_run_id
                        WHERE a.probe_artifact_set_id=%s
                        FOR UPDATE OF a""",
                    (probe_artifact_set_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(probe_artifact_set_id)
                if row["deleted_at"] is not None:
                    return
                if (
                    row["artifact_uri"] != artifact_uri
                    or row["state"]
                    not in {"LOSER_RETAINED", "PROMOTED_RETAINED"}
                    or row["retain_until"] is None
                    or row["retain_until"] > datetime.now(timezone.utc)
                    or row["run_state"] not in {"PROMOTED", "REJECTED"}
                ):
                    raise RuntimeError(
                        "probe artifact is not eligible for irreversible expiry"
                    )
                cursor.execute(
                    """UPDATE segment_probe_artifact_sets
                          SET state='EXPIRED',deleted_at=now()
                        WHERE probe_artifact_set_id=%s""",
                    (probe_artifact_set_id,),
                )
                self.event(
                    cursor,
                    "PROBE_ARTIFACT_EXPIRED",
                    {
                        "probe_run_id": row["probe_run_id"],
                        "probe_artifact_set_id": probe_artifact_set_id,
                        "artifact_uri": artifact_uri,
                    },
                    row["task_id"],
                )

    def _requeue_unavailable(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        retry_delay_seconds: int,
        kind: str,
        maximum_requeues: int | None = None,
    ) -> bool:
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        if maximum_requeues is not None and maximum_requeues < 0:
            raise ValueError("maximum_requeues must be non-negative")
        attempt_state = f"RETRYABLE_{kind}_UNAVAILABLE"
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._assert_owner(cursor, task_id, attempt_id, lease_token)
                if maximum_requeues is not None:
                    cursor.execute(
                        """SELECT COUNT(*) AS count
                             FROM segment_attempts
                            WHERE task_id=%s AND state=%s""",
                        (task_id, attempt_state),
                    )
                    if int(cursor.fetchone()["count"]) >= maximum_requeues:
                        return False
                cursor.execute(
                    "UPDATE segment_attempts SET state=%s,result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                    (attempt_state, json.dumps(result, sort_keys=True, separators=(",", ":")), attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,retry_after=%s,active_attempt_id=NULL,updated_at=now() WHERE task_id=%s""",
                    (retry_after, task_id),
                )
                self.event(cursor, f"STATE_{attempt_state}", {**result, "retry_after": retry_after.isoformat()}, task_id, attempt_id)
                return True

    def requeue_provider_unavailable(self, task_id: str, attempt_id: str, lease_token: str, result: dict[str, Any], *, retry_delay_seconds: int) -> None:
        self._requeue_unavailable(task_id, attempt_id, lease_token, result, retry_delay_seconds=retry_delay_seconds, kind="PROVIDER")

    def requeue_source_unavailable(self, task_id: str, attempt_id: str, lease_token: str, result: dict[str, Any], *, retry_delay_seconds: int) -> None:
        self._requeue_unavailable(task_id, attempt_id, lease_token, result, retry_delay_seconds=retry_delay_seconds, kind="SOURCE")

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
        return self._requeue_unavailable(
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
        return self._requeue_unavailable(
            task_id,
            attempt_id,
            lease_token,
            result,
            retry_delay_seconds=retry_delay_seconds,
            kind="FINALIZATION",
            maximum_requeues=maximum_requeues,
        )

    def requeue_for_larger_gpu(
        self,
        task_id: str,
        attempt_id: str,
        lease_token: str,
        result: dict[str, Any],
        *,
        minimum_vram_gb: float,
    ) -> None:
        if minimum_vram_gb <= 0:
            raise ValueError("larger-GPU retry requires a positive VRAM floor")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                task = self._assert_owner(cursor, task_id, attempt_id, lease_token)
                required = max(float(task["minimum_vram_gb"]), float(minimum_vram_gb))
                receipt = {**result, "minimum_vram_gb": required}
                cursor.execute(
                    "UPDATE segment_attempts SET state='RETRY_ON_LARGER_GPU',result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                    (json.dumps(receipt, sort_keys=True, separators=(",", ":")), attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,retry_after=NULL,active_attempt_id=NULL,
                       gpu_required=true,minimum_vram_gb=%s,updated_at=now() WHERE task_id=%s""",
                    (required, task_id),
                )
                self.event(cursor, "STATE_RETRY_ON_LARGER_GPU", receipt, task_id, attempt_id)

    def recover_terminal_provider_outage(self, task_id: str, attempt_id: str, *, retry_delay_seconds: int) -> dict[str, Any]:
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM segment_tasks WHERE task_id=%s FOR UPDATE", (task_id,))
                task = cursor.fetchone()
                cursor.execute("SELECT * FROM segment_attempts WHERE attempt_id=%s AND task_id=%s", (attempt_id, task_id))
                attempt = cursor.fetchone()
                if task is None or attempt is None:
                    raise RuntimeError("task or attempt does not exist")
                if task["state"] != "POLICY_REJECTED" or attempt["state"] != "POLICY_REJECTED":
                    raise RuntimeError("only a terminal POLICY_REJECTED attempt may be recovered")
                prior = dict(attempt["result"] or {})
                if "PlannerProviderUnavailable" not in str(prior.get("error", "")):
                    raise RuntimeError("terminal attempt is not a recorded transient planner outage")
                recovered = {
                    **prior,
                    "status": "RECOVERED_PROVIDER_UNAVAILABLE",
                    "recovered_at_utc": utc_now(),
                    "retry_after": retry_after.isoformat(),
                    "recovery_reason": "historical worker classified PlannerProviderUnavailable as POLICY_REJECTED",
                }
                cursor.execute(
                    "UPDATE segment_attempts SET state='RECOVERED_PROVIDER_UNAVAILABLE',result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                    (json.dumps(recovered, sort_keys=True, separators=(",", ":")), attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,retry_after=%s,active_attempt_id=NULL,updated_at=now() WHERE task_id=%s""",
                    (retry_after, task_id),
                )
                self.event(cursor, "STATE_RECOVERED_PROVIDER_UNAVAILABLE", recovered, task_id, attempt_id)
                return recovered

    def recover_terminal_mcp_auth_outage(self, task_id: str, attempt_id: str, *, retry_delay_seconds: int) -> dict[str, Any]:
        """Recover only the exact terminal 401 produced by a stale loopback MCP."""
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        retry_after = _deadline(retry_delay_seconds)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM segment_tasks WHERE task_id=%s FOR UPDATE", (task_id,))
                task = cursor.fetchone()
                cursor.execute("SELECT * FROM segment_attempts WHERE attempt_id=%s AND task_id=%s", (attempt_id, task_id))
                attempt = cursor.fetchone()
                if task is None or attempt is None:
                    raise RuntimeError("task or attempt does not exist")
                if task["state"] != "BLOCKED_SOURCE_UNAVAILABLE" or attempt["state"] != "BLOCKED_SOURCE_UNAVAILABLE":
                    raise RuntimeError("only a terminal BLOCKED_SOURCE_UNAVAILABLE attempt may be recovered")
                prior = dict(attempt["result"] or {})
                if prior.get("status") != "BLOCKED_SOURCE_UNAVAILABLE" or prior.get("error") != "HTTPError: HTTP Error 401: Unauthorized":
                    raise RuntimeError("terminal attempt is not the exact loopback MCP authentication outage")
                recovered = {
                    **prior,
                    "status": "RECOVERED_MCP_AUTH_OUTAGE",
                    "recovered_at_utc": utc_now(),
                    "retry_after": retry_after.isoformat(),
                    "recovery_reason": "duplicate loopback MCP listener used a different runtime token",
                }
                cursor.execute(
                    "UPDATE segment_attempts SET state='RECOVERED_MCP_AUTH_OUTAGE',result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                    (json.dumps(recovered, sort_keys=True, separators=(",", ":")), attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,retry_after=%s,active_attempt_id=NULL,updated_at=now() WHERE task_id=%s""",
                    (retry_after, task_id),
                )
                self.event(cursor, "STATE_RECOVERED_MCP_AUTH_OUTAGE", recovered, task_id, attempt_id)
                return recovered

    def recover_terminal_finalizer_dependency(self, task_id: str, attempt_id: str, *, retry_delay_seconds: int) -> dict[str, Any]:
        """Recover only a grow lost to a missing pinned finalizer module."""
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        retry_after = _deadline(retry_delay_seconds)
        allowed_errors = {
            "ModuleNotFoundError: No module named 'numpy'",
            "ModuleNotFoundError: No module named 'tifffile'",
        }
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM segment_tasks WHERE task_id=%s FOR UPDATE", (task_id,))
                task = cursor.fetchone()
                cursor.execute("SELECT * FROM segment_attempts WHERE attempt_id=%s AND task_id=%s", (attempt_id, task_id))
                attempt = cursor.fetchone()
                if task is None or attempt is None:
                    raise RuntimeError("task or attempt does not exist")
                if task["state"] != "FINALIZATION_FAILED" or attempt["state"] != "FINALIZATION_FAILED":
                    raise RuntimeError("only a terminal FINALIZATION_FAILED attempt may be recovered")
                prior = dict(attempt["result"] or {})
                if prior.get("status") != "FINALIZATION_FAILED" or prior.get("error") not in allowed_errors:
                    raise RuntimeError("terminal attempt is not an exact missing finalizer dependency")
                cursor.execute(
                    """SELECT p.*,d.probe_run_id
                         FROM segment_probe_promotions p
                         JOIN segment_probe_decisions d
                           ON d.decision_id=p.decision_id
                        WHERE p.continuation_task_id=%s
                        FOR UPDATE OF p""",
                    (task_id,),
                )
                promotion = cursor.fetchone()
                if promotion is not None:
                    failed_receipt = dict(promotion["receipt"] or {})
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
                    cursor.execute(
                        """UPDATE segment_probe_promotions
                              SET state='CONTINUING',receipt=%s::jsonb,
                                  receipt_sha256=%s,updated_at=now()
                            WHERE promotion_id=%s""",
                        (
                            json.dumps(
                                reopened_receipt,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            content_sha256(reopened_receipt),
                            promotion["promotion_id"],
                        ),
                    )
                    cursor.execute(
                        """UPDATE segment_probe_runs SET state='CONTINUING',
                                  updated_at=now() WHERE probe_run_id=%s""",
                        (promotion["probe_run_id"],),
                    )
                    cursor.execute(
                        """UPDATE segment_probe_artifact_sets
                              SET state='WINNER_RETAINED',retain_until=NULL
                            WHERE probe_artifact_set_id=%s""",
                        (promotion["winner_probe_artifact_set_id"],),
                    )
                    self.event(
                        cursor,
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
                    "recovered_at_utc": utc_now(),
                    "retry_after": retry_after.isoformat(),
                    "recovery_reason": "pinned TIFXYZ finalizer module was absent from the worker runtime",
                }
                cursor.execute(
                    "UPDATE segment_attempts SET state='RECOVERED_FINALIZER_DEPENDENCY',result=%s::jsonb,updated_at=now() WHERE attempt_id=%s",
                    (json.dumps(recovered, sort_keys=True, separators=(",", ":")), attempt_id),
                )
                cursor.execute(
                    """UPDATE segment_tasks SET state='PENDING',worker_id=NULL,lease_token_hash=NULL,
                       lease_expires_at=NULL,retry_after=%s,active_attempt_id=NULL,updated_at=now() WHERE task_id=%s""",
                    (retry_after, task_id),
                )
                self.event(cursor, "STATE_RECOVERED_FINALIZER_DEPENDENCY", recovered, task_id, attempt_id)
                return recovered

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT state,COUNT(*) AS count FROM segment_tasks GROUP BY state ORDER BY state")
                tasks = {row["state"]: row["count"] for row in cursor.fetchall()}
                cursor.execute("SELECT state,COUNT(*) AS count FROM segment_attempts GROUP BY state ORDER BY state")
                attempts = {row["state"]: row["count"] for row in cursor.fetchall()}
                cursor.execute("SELECT COUNT(*) AS count FROM segment_source_snapshots")
                snapshots = cursor.fetchone()["count"]
                cursor.execute("SELECT COUNT(*) AS count FROM segment_surfaces")
                surfaces = cursor.fetchone()["count"]
                cursor.execute("SELECT COUNT(*) AS count FROM segment_qc_jobs")
                qc_jobs = cursor.fetchone()["count"]
                cursor.execute("SELECT state,COUNT(*) AS count FROM segment_qc_jobs GROUP BY state ORDER BY state")
                qc_job_states = {row["state"]: row["count"] for row in cursor.fetchall()}
                cursor.execute("SELECT worker_id,capabilities FROM segment_worker_capabilities ORDER BY worker_id")
                workers = {row["worker_id"]: dict(row["capabilities"]) for row in cursor.fetchall()}
        return {
            "schema": "campaignx.segment_fleet_status.v1",
            "database": self.identity,
            "source_snapshots": snapshots,
            "surfaces": surfaces,
            "tasks": tasks,
            "attempts": attempts,
            "qc_jobs": qc_jobs,
            "qc_job_states": qc_job_states,
            "workers": workers,
            "seed_probes": self.probe_status(),
        }
