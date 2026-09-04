from __future__ import annotations

import copy
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
    QC_SMALL_SURFACE_DIAGNOSTIC,
    QC_WAITING_GEOMETRY,
    TERMINAL_STATES,
    FleetStore,
    _FirstLettersDiscoveryRunHandle,
    _instant,
    _safe_outage_detail,
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


class _PostgresIncompleteHistory(RuntimeError):
    """Internal control flow after durable incomplete history was written."""


class _CommitIncompleteHistory:
    """Commit an incomplete cut, suppress its sentinel, then raise outside."""

    def __init__(self, connection: Any):
        self.connection = connection
        self.incomplete = False

    def __enter__(self):
        return self.connection.__enter__()

    def __exit__(self, error_type, error, traceback):
        if error_type is _PostgresIncompleteHistory:
            self.incomplete = True
            return bool(self.connection.__exit__(None, None, None)) or True
        return self.connection.__exit__(error_type, error, traceback)


class PostgresFleetStore:
    """Authoritative multi-host control plane.

    Every task claim uses ``FOR UPDATE SKIP LOCKED`` and a hashed lease token.
    The raw database URL is deliberately never exposed through ``identity`` or
    status receipts.
    """

    def __init__(
        self, database_url: str, *, identity: str = "postgresql://redacted",
        task9_discovery_gate_resolver=None,
        first_letters_discovery_executor=None,
        first_letters_discovery_worker_id: str | None = None,
        first_letters_discovery_executor_id: str | None = None,
        first_letters_discovery_executor_registration: dict[str, Any] | None = None,
        first_letters_discovery_profile_resolver=None,
        first_letters_experimental_arm_resolver=None,
    ):
        self.database_url = database_url
        self.identity = identity
        self._task9_discovery_gate_resolver = task9_discovery_gate_resolver
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
        self._first_letters_discovery_profile_resolver = (
            first_letters_discovery_profile_resolver
        )
        self._first_letters_experimental_arm_resolver = (
            first_letters_experimental_arm_resolver
        )

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
                schema_ready = False
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
                        cursor.execute(
                            """SELECT EXISTS (
                                 SELECT 1 FROM pg_constraint
                                  WHERE conrelid =
                                    'segment_first_letters_discovery_historical_imports_v19'::regclass
                                    AND contype='u'
                                    AND pg_get_constraintdef(oid)=
                                      'UNIQUE (reservation_id)'
                               ) AS legacy_import_cardinality"""
                        )
                        schema_ready = not cursor.fetchone()[
                            "legacy_import_cardinality"
                        ]
                if not schema_ready:
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)", (0x43414D505849,)
                    )
                    # Scoped to current_schema(), like the fast path above and
                    # unlike to_regclass, which searches the whole path. An
                    # unqualified CREATE TABLE lands in current_schema(), so
                    # that is the only schema whose ledger answers "is there
                    # anything to do". Searching the path answers for whichever
                    # schema happens to be reachable, and answers "ready" for a
                    # schema holding no table at all -- after which every later
                    # statement silently reads and writes the neighbour's rows.
                    cursor.execute(
                        """SELECT EXISTS (
                               SELECT 1
                               FROM pg_catalog.pg_class c
                               JOIN pg_catalog.pg_namespace n
                                 ON n.oid=c.relnamespace
                               WHERE n.nspname=current_schema()
                                 AND c.relname='segment_schema_migrations'
                           ) AS present"""
                    )
                    if cursor.fetchone()["present"]:
                        cursor.execute(
                            "SELECT 1 FROM segment_schema_migrations "
                            "WHERE version=%s", (target,)
                        )
                        if cursor.fetchone() is not None:
                            cursor.execute(
                                """SELECT EXISTS (
                                     SELECT 1 FROM pg_constraint
                                      WHERE conrelid =
                                        'segment_first_letters_discovery_historical_imports_v19'::regclass
                                        AND contype='u'
                                        AND pg_get_constraintdef(oid)=
                                          'UNIQUE (reservation_id)'
                                   ) AS legacy_import_cardinality"""
                            )
                            schema_ready = not cursor.fetchone()[
                                "legacy_import_cardinality"
                            ]
                    if not schema_ready:
                        cursor.execute(sql)
        if self._first_letters_discovery_executor_registration is not None:
            self.register_first_letters_discovery_executor(
                self._first_letters_discovery_executor_registration
            )

    @staticmethod
    def event(cursor: Any, event_type: str, payload: Any, task_id: str | None = None, attempt_id: str | None = None) -> None:
        cursor.execute(
            "INSERT INTO segment_events(task_id,attempt_id,event_type,payload) VALUES(%s,%s,%s,%s::jsonb)",
            (task_id, attempt_id, event_type, json.dumps(payload, sort_keys=True, separators=(",", ":"))),
        )

    @staticmethod
    def _validated_first_letters_discovery_executor_registration(
        value: Any,
    ) -> dict[str, Any]:
        return FleetStore._validated_first_letters_discovery_executor_registration(
            value
        )

    def register_first_letters_discovery_executor(
        self, registration: dict[str, Any],
    ) -> dict[str, Any]:
        value = self._validated_first_letters_discovery_executor_registration(
            registration
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_executor_registry
                       (worker_id,executor_id,executor_sha256,capabilities,
                        registration,registration_sha256,enabled,created_at)
                       VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,now())
                       ON CONFLICT(worker_id) DO NOTHING""",
                    (
                        value["worker_id"], value["executor_id"],
                        value["executor_sha256"], json.dumps(
                            value["capabilities"], sort_keys=True,
                            separators=(",", ":"),
                        ),
                        json.dumps(value, sort_keys=True, separators=(",", ":")),
                        value["registration_sha256"], value["enabled"],
                    ),
                )
                cursor.execute(
                    "SELECT registration_sha256 FROM "
                    "segment_first_letters_discovery_executor_registry "
                    "WHERE worker_id=%s FOR UPDATE",
                    (value["worker_id"],),
                )
                existing = cursor.fetchone()
                if (existing is None or existing["registration_sha256"] !=
                        value["registration_sha256"]):
                    raise ValueError(
                        "DISCOVERY_EXECUTOR_REGISTRATION_CONFLICT"
                    )
        return value

    def _persisted_discovery_executor_registration_from_cursor(
        self, cursor: Any, *, worker_id: str,
    ) -> dict[str, Any]:
        from .discovery_executor import DISCOVERY_EXECUTOR_CAPABILITY

        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        cursor.execute(
            "SELECT * FROM segment_first_letters_discovery_executor_registry "
            "WHERE worker_id=%s FOR SHARE",
            (worker_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        registration = self._validated_first_letters_discovery_executor_registration(
            copy.deepcopy(row["registration"])
        )
        if (row["registration_sha256"] !=
                registration["registration_sha256"]
                or row["executor_id"] != registration["executor_id"]
                or row["executor_sha256"] !=
                    registration["executor_sha256"]
                or list(row["capabilities"]) != registration["capabilities"]
                or row["enabled"] is not registration["enabled"]
                or registration["worker_id"] != worker_id
                or registration["enabled"] is not True):
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        if DISCOVERY_EXECUTOR_CAPABILITY not in registration["capabilities"]:
            raise ValueError("DISCOVERY_EXECUTOR_CAPABILITY_REQUIRED")
        return registration

    def _discovery_executor_registration_from_cursor(
        self, cursor: Any,
    ) -> dict[str, Any]:
        from .discovery_executor import runtime_discovery_executor_sha256

        worker_id = self._first_letters_discovery_worker_id
        executor_id = self._first_letters_discovery_executor_id
        executor = self._first_letters_discovery_executor
        if (not isinstance(worker_id, str) or not worker_id
                or not isinstance(executor_id, str) or not executor_id
                or executor is None):
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        registration = self._persisted_discovery_executor_registration_from_cursor(
            cursor, worker_id=worker_id,
        )
        if registration["executor_id"] != executor_id:
            raise ValueError("REGISTERED_DISCOVERY_EXECUTOR_REQUIRED")
        if (runtime_discovery_executor_sha256(executor) !=
                registration["executor_sha256"]):
            raise ValueError("DISCOVERY_EXECUTOR_CODE_HASH_MISMATCH")
        return registration

    @staticmethod
    def _validated_discovery_compute_cap(value: Any) -> dict[str, Any]:
        return FleetStore._validated_discovery_compute_cap(value)

    @staticmethod
    def _validated_discovery_work_authority(
        value: Any, *, mission_id: str, work_kind: str,
        work_authority_id: str, work_authority_sha256: str,
        ordered_item_ids: list[str], cap_authority_id: str,
        cap_authority_sha256: str,
    ) -> dict[str, Any]:
        return FleetStore._validated_discovery_work_authority(
            value, mission_id=mission_id, work_kind=work_kind,
            work_authority_id=work_authority_id,
            work_authority_sha256=work_authority_sha256,
            ordered_item_ids=ordered_item_ids,
            cap_authority_id=cap_authority_id,
            cap_authority_sha256=cap_authority_sha256,
        )

    def register_discovery_compute_cap(
        self, authority: dict[str, Any],
    ) -> dict[str, Any]:
        value = self._validated_discovery_compute_cap(authority)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{value['mission_id']}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-compute-mission:{value['mission_id']}",),
                )
                cursor.execute(
                    """SELECT authority
                         FROM segment_first_letters_discovery_compute_caps
                        WHERE mission_id=%s FOR UPDATE""",
                    (value["mission_id"],),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    prior = copy.deepcopy(existing["authority"])
                    if prior != value:
                        raise ValueError(
                            "MISSION_COMPUTE_CAP_AUTHORITY_CONFLICT"
                        )
                    return prior
                cursor.execute(
                    """INSERT INTO segment_first_letters_discovery_compute_caps
                       (mission_id,cap_authority_id,authority_sha256,cap_units,
                        authority,created_at)
                       VALUES(%s,%s,%s,%s,%s::jsonb,%s)""",
                    (value["mission_id"], value["cap_authority_id"],
                     value["authority_sha256"],
                     value["mission_compute_cap_units"],
                     json.dumps(value, sort_keys=True, separators=(",", ":")),
                     utc_now()),
                )
        return value

    def discovery_compute_cap(self, mission_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT authority
                         FROM segment_first_letters_discovery_compute_caps
                        WHERE mission_id=%s""",
                    (mission_id,),
                )
                row = cursor.fetchone()
        return copy.deepcopy(row["authority"]) if row is not None else None

    def discovery_compute_total(self, mission_id: str) -> int:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT COALESCE(SUM(reserved_units),0) AS total
                         FROM segment_first_letters_discovery_compute_reservations
                        WHERE mission_id=%s""",
                    (mission_id,),
                )
                row = cursor.fetchone()
        return int(row["total"])

    def _block_discovery_compute_ledger(
        self, mission_id: str, evidence: Any,
    ) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_id}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-compute-mission:{mission_id}",),
                )
                cursor.execute(
                    """INSERT INTO segment_first_letters_discovery_compute_blocks
                       (mission_id,reason,evidence,created_at)
                       VALUES(%s,%s,%s::jsonb,%s)
                       ON CONFLICT(mission_id) DO NOTHING""",
                    (mission_id, "CONTROL_INCOMPLETE_COMPUTE_LEDGER",
                     json.dumps(evidence, sort_keys=True, separators=(",", ":")),
                     utc_now()),
                )

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
            "compute.commit_outcome_unknown",
            "compute.after_commit_before_response",
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
                raise ValueError(
                    "TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED"
                )
            authoritative_gate = resolver(mission_id)
            if (not isinstance(authoritative_gate, dict)
                    or authoritative_gate != task9_gate
                    or authoritative_gate.get("schema") !=
                        "campaignx.first_letters_task9_discovery_gate.v1"
                    or authoritative_gate.get("allow_unvalidated") is not False):
                raise ValueError(
                    "TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED"
                )
        authority = self._validated_discovery_work_authority(
            work_authority, mission_id=mission_id, work_kind=work_kind,
            work_authority_id=work_authority_id,
            work_authority_sha256=work_authority_sha256,
            ordered_item_ids=ordered_item_ids,
            cap_authority_id=cap_authority_id,
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
        should_read = False
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_id}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-compute-mission:{mission_id}",),
                )
                cursor.execute(
                    """SELECT reason
                         FROM segment_first_letters_discovery_compute_blocks
                        WHERE mission_id=%s FOR SHARE""",
                    (mission_id,),
                )
                if cursor.fetchone() is not None:
                    raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
                cursor.execute(
                    """SELECT *
                         FROM segment_first_letters_discovery_compute_caps
                        WHERE mission_id=%s FOR UPDATE""",
                    (mission_id,),
                )
                cap_row = cursor.fetchone()
                if (cap_row is None
                        or cap_row["cap_authority_id"] != cap_authority_id
                        or cap_row["authority_sha256"] !=
                            cap_authority_sha256):
                    raise ValueError(
                        "mission discovery compute cap authority mismatch"
                    )
                cursor.execute(
                    """SELECT request_sha256
                         FROM segment_first_letters_discovery_compute_reservations
                        WHERE mission_id=%s AND request_id=%s FOR UPDATE""",
                    (mission_id, request_id),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["request_sha256"] != request_sha:
                        raise ValueError(
                            "DISCOVERY_COMPUTE_RESERVATION_CONFLICT"
                        )
                    should_read = True
                else:
                    cursor.execute(
                        """SELECT COALESCE(SUM(reserved_units),0) AS total
                             FROM segment_first_letters_discovery_compute_reservations
                            WHERE mission_id=%s""",
                        (mission_id,),
                    )
                    used = int(cursor.fetchone()["total"])
                    cap_units = int(cap_row["cap_units"])
                    available_items = max(0, (cap_units - used) // 24)
                    if reservation_mode == "EXACT":
                        if len(ordered_item_ids) > available_items:
                            raise ValueError(
                                "mission discovery compute cap exhausted"
                            )
                        selected = list(ordered_item_ids)
                    else:
                        selected = list(ordered_item_ids[
                            :min(len(ordered_item_ids), available_items)
                        ])
                        if not selected:
                            return None
                    reserved = len(selected) * 24
                    reservation_core = {
                        "schema":
                            "campaignx.first_letters_discovery_compute_reservation.v1",
                        "reservation_id": stable_id(
                            "first-letters-discovery-reservation", request_core,
                        ),
                        "mission_id": mission_id, "request_id": request_id,
                        "work_kind": work_kind,
                        "work_authority_id": work_authority_id,
                        "work_authority_sha256": work_authority_sha256,
                        "ordered_item_ids": selected,
                        "ordered_item_ids_sha256": content_sha256(selected),
                        "item_count": len(selected),
                        "compute_unit": "probe_generation_units",
                        "top_k": 2, "probe_generations": 12,
                        "maximum_attempts_per_candidate": 1,
                        "units_per_item": 24, "reserved_units": reserved,
                        "cap_authority_id": cap_authority_id,
                        "cap_authority_sha256": cap_authority_sha256,
                        "reserved_before_units": used,
                        "reserved_after_units": used + reserved,
                        "source": source, "allow_unvalidated": False,
                    }
                    reservation = {
                        **reservation_core,
                        "reservation_sha256": content_sha256(reservation_core),
                        "created_at": utc_now(),
                    }
                    if failpoint == "compute.before_reservation_insert":
                        raise RuntimeError(failpoint)
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_compute_reservations
                           (reservation_id,mission_id,request_id,work_kind,
                            work_authority_id,work_authority_sha256,
                            ordered_item_ids_sha256,item_count,units_per_item,
                            reserved_units,reserved_before_units,
                            reserved_after_units,source,reservation,
                            reservation_sha256,request_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                  %s::jsonb,%s,%s,%s)""",
                        (reservation["reservation_id"], mission_id, request_id,
                         work_kind, work_authority_id, work_authority_sha256,
                         reservation["ordered_item_ids_sha256"], len(selected),
                         24, reserved, used, used + reserved, source,
                         json.dumps(reservation, sort_keys=True,
                                    separators=(",", ":")),
                         reservation["reservation_sha256"], request_sha,
                         reservation["created_at"]),
                    )
                    if failpoint == (
                        "compute.after_reservation_insert_before_work_insert"
                    ):
                        raise RuntimeError(failpoint)
                    dispatch_kind = {
                        "BASELINE_ARM": "BASELINE_DISPATCH",
                        "ALTERNATIVE_SOURCE_ARM":
                            "ALTERNATIVE_SOURCE_DISPATCH",
                        "ADAPTIVE_CHILD": "ADAPTIVE_CHILDREN",
                    }[work_kind]
                    work_core = {
                        "schema":
                            "campaignx.first_letters_discovery_work_binding.v1",
                        "reservation_id": reservation["reservation_id"],
                        "reservation_sha256":
                            reservation["reservation_sha256"],
                        "mission_id": mission_id, "request_id": request_id,
                        "work_kind": work_kind,
                        "dispatch_kind": dispatch_kind,
                        "work_authority": authority,
                        "ordered_item_ids": selected,
                        "allow_unvalidated": False,
                    }
                    work = {
                        **work_core, "work_sha256": content_sha256(work_core),
                    }
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_work_bindings
                           (reservation_id,mission_id,request_id,work_kind,
                            dispatch_kind,work,work_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s)""",
                        (reservation["reservation_id"], mission_id, request_id,
                         work_kind, dispatch_kind,
                         json.dumps(work, sort_keys=True,
                                    separators=(",", ":")),
                         work["work_sha256"], utc_now()),
                    )
                    if failpoint == "compute.after_work_insert_before_commit":
                        raise RuntimeError(failpoint)
                    if failpoint == "compute.before_commit":
                        raise RuntimeError(failpoint)
                    should_read = True
        if failpoint == "compute.commit_outcome_unknown":
            raise RuntimeError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        if not should_read:  # pragma: no cover - defensive transaction guard
            raise RuntimeError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        return self.read_discovery_compute_request(mission_id, request_id)

    def read_discovery_compute_request(
        self, mission_id: str, request_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT r.reservation,w.work
                         FROM segment_first_letters_discovery_compute_reservations r
                         JOIN segment_first_letters_discovery_work_bindings w
                           ON w.reservation_id=r.reservation_id
                        WHERE r.mission_id=%s AND r.request_id=%s""",
                    (mission_id, request_id),
                )
                rows = cursor.fetchall()
        if len(rows) != 1:
            raise ValueError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        return {
            "reservation": copy.deepcopy(rows[0]["reservation"]),
            "work": copy.deepcopy(rows[0]["work"]),
        }

    def _first_letters_empty_history_tx(
        self, cursor: Any, *, mission_id: str,
    ) -> dict[str, Any]:
        """Persist the exact empty cutover or fail closed on legacy rows."""

        cursor.execute(
            """SELECT r.probe_run_id FROM segment_probe_runs r
                 JOIN segment_tasks t ON t.task_id=r.task_id
                WHERE t.mission_id=%s ORDER BY r.probe_run_id FOR SHARE OF r,t""",
            (mission_id,),
        )
        legacy_ids = [str(row["probe_run_id"]) for row in cursor.fetchall()]
        cursor.execute(
            """SELECT r.reservation_id,
                      w.reservation_id AS work_reservation_id,r.*
                 FROM segment_first_letters_discovery_compute_reservations r
                 LEFT JOIN segment_first_letters_discovery_work_bindings w
                   ON w.reservation_id=r.reservation_id
                 LEFT JOIN
                   segment_first_letters_discovery_native_adapters_v19 a
                   ON a.reservation_id=r.reservation_id
                WHERE r.mission_id=%s AND a.reservation_id IS NULL
                  AND r.source!='IMPORTED_HISTORICAL_EXACT'
                  AND r.work_kind IN
                    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')
                ORDER BY r.reservation_id FOR SHARE OF r""",
            (mission_id,),
        )
        retained_root_rows = cursor.fetchall()
        cursor.execute(
            """SELECT w.*
                 FROM segment_first_letters_discovery_work_bindings w
                 LEFT JOIN
                   segment_first_letters_discovery_compute_reservations r
                   ON r.reservation_id=w.reservation_id
                WHERE w.mission_id=%s AND r.reservation_id IS NULL
                  AND w.work_kind IN
                    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')
                ORDER BY w.reservation_id FOR SHARE OF w""",
            (mission_id,),
        )
        orphan_work_rows = cursor.fetchall()
        cursor.execute(
            """SELECT r.*,w.work,w.work_sha256,
                      w.mission_id AS w_mission_id,
                      w.request_id AS w_request_id,
                      w.work_kind AS w_work_kind,
                      w.dispatch_kind
                 FROM segment_first_letters_discovery_compute_reservations r
                 JOIN segment_first_letters_discovery_work_bindings w
                   ON w.reservation_id=r.reservation_id
                 LEFT JOIN
                   segment_first_letters_discovery_native_adapters_v19 a
                   ON a.reservation_id=r.reservation_id
                WHERE r.mission_id=%s AND a.reservation_id IS NULL
                  AND r.source!='IMPORTED_HISTORICAL_EXACT'
                  AND r.work_kind IN
                    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')
                ORDER BY r.reservation_id FOR SHARE OF r,w""",
            (mission_id,),
        )
        retained_rows = cursor.fetchall()
        retained_graphs: list[dict[str, Any]] = []
        retained_complete = True
        for root in retained_root_rows:
            if root["work_reservation_id"] is not None:
                continue
            retained_complete = False
            retained_graphs.append({
                "graph_kind": "V16_ORPHAN_RESERVATION",
                "reservation": FleetStore._history_row_projection(root),
                "logical_units": 0,
            })
        for root in orphan_work_rows:
            retained_complete = False
            retained_graphs.append({
                "graph_kind": "V16_ORPHAN_WORK",
                "work": FleetStore._history_row_projection(root),
                "logical_units": 0,
            })
        from .seed_probe import (
            _derive_first_letters_discovery_run_authority,
            load_first_letters_discovery_profile_bytes,
            validate_experimental_arm_admission,
        )
        for retained_row in retained_rows:
            reservation = copy.deepcopy(retained_row["reservation"])
            work = copy.deepcopy(retained_row["work"])
            reservation_complete = (
                isinstance(reservation, dict)
                and isinstance(work, dict)
                and retained_row["source"] == "RESERVED_BEFORE_EXECUTION"
                and reservation.get("source") == "RESERVED_BEFORE_EXECUTION"
                and reservation.get("reservation_id")
                    == retained_row["reservation_id"]
                and reservation.get("mission_id") == mission_id
                and reservation.get("request_id")
                    == retained_row["request_id"]
                and reservation.get("work_kind")
                    == retained_row["work_kind"]
                and reservation.get("reservation_sha256")
                    == retained_row["reservation_sha256"]
                and reservation.get("reservation_sha256")
                    == content_sha256({
                        key: value for key, value in reservation.items()
                        if key not in {"reservation_sha256", "created_at"}
                    })
                and work.get("reservation_id")
                    == retained_row["reservation_id"]
                and work.get("reservation_sha256")
                    == reservation.get("reservation_sha256")
                and work.get("mission_id") == mission_id
                and work.get("request_id") == retained_row["request_id"]
                and work.get("work_kind") == retained_row["work_kind"]
                and work.get("work_sha256") == retained_row["work_sha256"]
                and work.get("work_sha256") == content_sha256({
                    key: value for key, value in work.items()
                    if key != "work_sha256"
                })
                and work.get("ordered_item_ids")
                    == reservation.get("ordered_item_ids")
                and retained_row["w_mission_id"] == mission_id
                and retained_row["w_request_id"]
                    == retained_row["request_id"]
                and retained_row["w_work_kind"]
                    == retained_row["work_kind"]
                and retained_row["dispatch_kind"]
                    == work.get("dispatch_kind")
            )
            cursor.execute(
                """SELECT * FROM
                   segment_first_letters_discovery_evidence_runs
                   WHERE reservation_id=%s ORDER BY cell_id,run_id FOR SHARE""",
                (retained_row["reservation_id"],),
            )
            runs = cursor.fetchall()
            reservation_complete = reservation_complete and (
                len(runs) == reservation.get("item_count")
                and [str(row["cell_id"]) for row in runs]
                    == sorted(reservation.get("ordered_item_ids") or [])
            )
            if not runs:
                retained_complete = False
                retained_graphs.append({
                    "graph_kind": "V16_DISCOVERY_EVIDENCE",
                    "reservation": FleetStore._history_row_projection(
                        retained_row
                    ),
                    "run": None, "executor_claims": [],
                    "evidence_sets": [], "evidence_files": [],
                    "retained_row_ids": {
                        "reservation_id": retained_row["reservation_id"],
                        "run_id": None,
                    },
                    "logical_execution_id": None,
                    "producer_kind": (
                        "BASELINE_RECONCILIATION"
                        if retained_row["work_kind"] == "BASELINE_ARM"
                        else "EXPERIMENTAL_ARM_ADMISSION"
                    ),
                    "source_snapshot_sha256": None,
                    "profile_file_sha256": None,
                    "item_id": None, "logical_units": 0,
                })
                continue
            for run in runs:
                cursor.execute(
                    """SELECT * FROM
                       segment_first_letters_discovery_executor_claims
                       WHERE run_id=%s ORDER BY claim_id FOR SHARE""",
                    (run["run_id"],),
                )
                claims = cursor.fetchall()
                cursor.execute(
                    """SELECT * FROM
                       segment_first_letters_discovery_evidence_sets
                       WHERE run_id=%s ORDER BY evidence_set_id FOR SHARE""",
                    (run["run_id"],),
                )
                evidence_sets = cursor.fetchall()
                files: list[Any] = []
                for evidence_set in evidence_sets:
                    cursor.execute(
                        """SELECT * FROM
                           segment_first_letters_discovery_evidence_files
                           WHERE evidence_set_id=%s
                           ORDER BY file_order,relative_path FOR SHARE""",
                        (evidence_set["evidence_set_id"],),
                    )
                    files.extend(cursor.fetchall())
                profile_payload = bytes(run["profile_bytes"])
                run_authority = copy.deepcopy(run["run_authority"])
                claim = (
                    copy.deepcopy(claims[0]["claim"])
                    if len(claims) == 1 else {}
                )
                evidence = (
                    copy.deepcopy(evidence_sets[0]["evidence"])
                    if len(evidence_sets) == 1 else {}
                )
                expected_paths = [
                    f"probes/{run['run_id']}/provider-request.json",
                    f"probes/{run['run_id']}/provider-response.json",
                    f"probes/{run['run_id']}/selection-policy-receipt.json",
                ]
                files_complete = (
                    [str(row["role"]) for row in files] == [
                        "CANDIDATE_PROVIDER_REQUEST",
                        "CANDIDATE_PROVIDER_RESPONSE",
                        "DISCOVERY_SELECTION_POLICY_RECEIPT",
                    ]
                    and [str(row["relative_path"]) for row in files]
                        == expected_paths
                    and [int(row["file_order"]) for row in files]
                        == [0, 1, 2]
                    and all(
                        int(row["byte_count"]) == len(bytes(row["payload"]))
                        and row["sha256"] == hashlib.sha256(
                            bytes(row["payload"])
                        ).hexdigest()
                        for row in files
                    )
                )
                provider_request = copy.deepcopy(run["provider_request"])
                authority = work.get("work_authority") or {}
                bindings = [
                    binding for binding in authority.get(
                        "ordered_item_bindings", []
                    ) if binding.get("item_id") == run["cell_id"]
                ]
                cursor.execute(
                    """SELECT * FROM segment_source_snapshots
                        WHERE source_snapshot_id=%s FOR SHARE""",
                    (run["source_snapshot_id"],),
                )
                source_row = cursor.fetchone()
                source = self._snapshot(source_row) if source_row is not None else {}
                binding = bindings[0] if len(bindings) == 1 else {}
                task_row = None
                attempt_row = None
                if len(bindings) == 1 and binding.get("parent_task_id"):
                    cursor.execute(
                        "SELECT * FROM segment_tasks WHERE task_id=%s FOR SHARE",
                        (binding["parent_task_id"],),
                    )
                    task_row = cursor.fetchone()
                if len(bindings) == 1 and binding.get("parent_attempt_id"):
                    cursor.execute(
                        "SELECT * FROM segment_attempts WHERE attempt_id=%s FOR SHARE",
                        (binding["parent_attempt_id"],),
                    )
                    attempt_row = cursor.fetchone()
                try:
                    profile = load_first_letters_discovery_profile_bytes(
                        profile_payload
                    )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    profile = {}
                try:
                    current_profile = (
                        self._first_letters_discovery_profile_resolver(
                            mission_id, run["source_snapshot_id"]
                        )
                        if self._first_letters_discovery_profile_resolver is not None
                        else None
                    )
                except Exception:
                    current_profile = None
                profile_complete = (
                    current_profile == profile_payload
                    and profile.get("source_snapshot_id")
                        == run["source_snapshot_id"]
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
                    and profile.get("m7_resolution")
                        == source.get("m7_resolution")
                    and profile.get("m7_level") == source.get("m7_level")
                    and profile.get("m7_threshold")
                        == source.get("m7_threshold")
                    and profile.get("m7_transform_sha256")
                        == source.get("m7_transform_sha256")
                    and profile.get("canonical_ordered_cell_set_sha256")
                        == content_sha256(reservation.get("ordered_item_ids"))
                    and profile.get("mission_compute_cap_authority_id")
                        == reservation.get("cap_authority_id")
                    and profile.get("mission_compute_cap_authority_sha256")
                        == reservation.get("cap_authority_sha256")
                )
                if retained_row["work_kind"] == "ALTERNATIVE_SOURCE_ARM":
                    arm_id = profile.get("experimental_arm_admission_id")
                    try:
                        arm = (
                            validate_experimental_arm_admission(
                                self._first_letters_experimental_arm_resolver(
                                    arm_id
                                )
                            )
                            if self._first_letters_experimental_arm_resolver
                                is not None and isinstance(arm_id, str)
                            else None
                        )
                    except Exception:
                        arm = None
                    profile_complete = profile_complete and (
                        isinstance(arm, dict)
                        and arm.get("admission_sha256")
                            == profile.get("experimental_arm_admission_sha256")
                        and arm.get("source_snapshot_id")
                            == run["source_snapshot_id"]
                        and arm.get("source_snapshot_sha256")
                            == source.get("source_snapshot_sha256")
                    )
                task_payload = (
                    copy.deepcopy(task_row["payload"])
                    if task_row is not None else {}
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
                                == run["source_snapshot_id"]
                            and task_row["cell_id"] == run["cell_id"]
                            and task_payload.get("scientific_opportunity_id")
                                == binding.get("scientific_opportunity_id")
                            and task_payload.get(
                                "accepted_p0_artifact_id",
                                task_payload.get("p0_artifact_id"),
                            ) == binding.get("accepted_p0_artifact_id")
                            and task_payload.get(
                                "accepted_p0_artifact_sha256",
                                task_payload.get("p0_artifact_sha256"),
                            ) == binding.get("accepted_p0_artifact_sha256")
                            and (
                                task_payload.get("candidate_discovery") or {}
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
                    current_derived = (
                        _derive_first_letters_discovery_run_authority(
                            profile_bytes=profile_payload,
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
                    and run["state"] == "COMPLETED"
                    and run["reservation_id"]
                        == retained_row["reservation_id"]
                    and run["mission_id"] == mission_id
                    and run["request_id"] == retained_row["request_id"]
                    and run["cell_id"] in reservation.get(
                        "ordered_item_ids", []
                    )
                    and run["cell_id"] == run_authority.get("cell_id")
                    and run["source_snapshot_id"]
                        == run_authority.get("source_snapshot_id")
                    and run["profile_file_sha256"]
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
                    and claims[0]["run_id"] == run["run_id"]
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
                    and evidence_sets[0]["run_id"] == run["run_id"]
                    and content_sha256(evidence)
                        == evidence_sets[0]["evidence_set_sha256"]
                    and run_authority.get("run_id") == run["run_id"]
                    and run_authority.get("run_authority_sha256")
                        == run["run_authority_sha256"]
                    and run_authority.get("run_authority_sha256")
                        == content_sha256({
                            key: value for key, value in run_authority.items()
                            if key != "run_authority_sha256"
                        })
                    and hashlib.sha256(profile_payload).hexdigest()
                        == run["profile_file_sha256"]
                    and files_complete
                )
                retained_complete = retained_complete and run_complete
                retained_graphs.append({
                    "graph_kind": "V16_DISCOVERY_EVIDENCE",
                    "reservation": FleetStore._history_row_projection(
                        retained_row
                    ),
                    "run": FleetStore._history_row_projection(run),
                    "executor_claims": [
                        FleetStore._history_row_projection(row)
                        for row in claims
                    ],
                    "evidence_sets": [
                        FleetStore._history_row_projection(row)
                        for row in evidence_sets
                    ],
                    "evidence_files": [
                        FleetStore._history_row_projection(row)
                        for row in files
                    ],
                    "source_snapshot": (
                        FleetStore._history_row_projection(source_row)
                        if source_row is not None else None
                    ),
                    "parent_task": (
                        FleetStore._history_row_projection(task_row)
                        if task_row is not None else None
                    ),
                    "parent_attempt": (
                        FleetStore._history_row_projection(attempt_row)
                        if attempt_row is not None else None
                    ),
                    "retained_row_ids": {
                        "reservation_id": retained_row["reservation_id"],
                        "run_id": run["run_id"],
                        "claim_ids": [row["claim_id"] for row in claims],
                        "evidence_set_ids": [
                            row["evidence_set_id"] for row in evidence_sets
                        ],
                        "evidence_file_paths": [
                            row["relative_path"] for row in files
                        ],
                    },
                    "logical_execution_id": run["run_id"],
                    "producer_kind": (
                        "BASELINE_RECONCILIATION"
                        if retained_row["work_kind"] == "BASELINE_ARM"
                        else "EXPERIMENTAL_ARM_ADMISSION"
                    ),
                    "source_snapshot_sha256": source.get(
                        "source_snapshot_sha256"
                    ),
                    "profile_file_sha256": run["profile_file_sha256"],
                    "item_id": run["cell_id"],
                    "logical_units": 24 if run_complete else 0,
                })
        manifest = {
            "schema":
                "campaignx.first_letters_discovery_history_manifest.v1",
            "mission_id": mission_id,
            "legacy_probe_run_ids": legacy_ids,
            "legacy_v16_reservation_ids": [
                str(row["reservation_id"]) for row in retained_rows
            ],
            "retained_execution_graphs": retained_graphs,
            "allow_unvalidated": False,
        }
        manifest_sha = content_sha256(manifest)
        cursor.execute(
            """SELECT reconciliation,manifest_sha256,state
                 FROM segment_first_letters_discovery_history_reconciliations_v19
                WHERE mission_id=%s
                ORDER BY created_at,reconciliation_seq LIMIT 1 FOR UPDATE""",
            (mission_id,),
        )
        prior = cursor.fetchone()
        complete = not legacy_ids and retained_complete
        if (
            prior is not None and prior["state"] == "COMPLETE"
            and prior["manifest_sha256"] != manifest_sha
        ):
            complete = False
        if (
            prior is not None and prior["state"] == "COMPLETE"
            and prior["manifest_sha256"] == manifest_sha
        ):
            cursor.execute(
                """SELECT * FROM
                   segment_first_letters_discovery_historical_imports_v19
                   WHERE mission_id=%s
                   ORDER BY logical_execution_id,import_id FOR SHARE""",
                (mission_id,),
            )
            materialized_imports = cursor.fetchall()
            imports_by_execution = {
                str(row["logical_execution_id"]): row
                for row in materialized_imports
            }
            materialization_complete = (
                len(materialized_imports) == len(retained_graphs)
                and len(imports_by_execution) == len(materialized_imports)
            )
            for graph in retained_graphs:
                row = imports_by_execution.get(
                    str(graph["logical_execution_id"])
                )
                projection_sha = content_sha256(graph)
                imported = (
                    copy.deepcopy(row["import_binding"])
                    if row is not None else None
                )
                import_core = {
                    key: value for key, value in imported.items()
                    if key not in {"import_sha256", "created_at"}
                } if isinstance(imported, dict) else {}
                materialization_complete = materialization_complete and (
                    row is not None
                    and isinstance(imported, dict)
                    and row["reservation_id"]
                        == graph["retained_row_ids"]["reservation_id"]
                    and imported.get("reservation_id")
                        == row["reservation_id"]
                    and row["logical_execution_id"]
                        == graph["logical_execution_id"]
                    and imported.get("logical_execution_id")
                        == graph["logical_execution_id"]
                    and row["producer_kind"] == graph["producer_kind"]
                    and imported.get("producer_kind")
                        == graph["producer_kind"]
                    and row["source_snapshot_sha256"]
                        == graph["source_snapshot_sha256"]
                    and imported.get("source_snapshot_sha256")
                        == graph["source_snapshot_sha256"]
                    and row["profile_file_sha256"]
                        == graph["profile_file_sha256"]
                    and imported.get("profile_file_sha256")
                        == graph["profile_file_sha256"]
                    and row["item_id"] == graph["item_id"]
                    and imported.get("item_id") == graph["item_id"]
                    and row["fixed_units"] == 24
                    and imported.get("fixed_units") == 24
                    and copy.deepcopy(row["retained_row_ids"])
                        == graph["retained_row_ids"]
                    and imported.get("retained_row_ids")
                        == graph["retained_row_ids"]
                    and row["retained_projection_sha256"] == projection_sha
                    and imported.get("retained_projection_sha256")
                        == projection_sha
                    and row["history_manifest_sha256"] == manifest_sha
                    and imported.get("history_manifest_sha256")
                        == manifest_sha
                    and row["import_sha256"]
                        == imported.get("import_sha256")
                    and row["import_sha256"] == content_sha256(import_core)
                )
            if materialization_complete:
                return copy.deepcopy(prior["reconciliation"])
            complete = False
        state = "COMPLETE" if complete else "CONTROL_INCOMPLETE"
        reason = None if complete else "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
        watermark = {
            "mission_id": mission_id,
            "legacy_probe_run_ids": legacy_ids,
            "legacy_v16_reservation_ids": manifest[
                "legacy_v16_reservation_ids"
            ],
            "retained_graph_count": len(retained_graphs),
            "retained_projection_sha256s": sorted(
                content_sha256(graph) for graph in retained_graphs
            ),
        }
        reconciliation_id = stable_id(
            "first-letters-discovery-history-reconciliation",
            {"mission_id": mission_id, "manifest_sha256": manifest_sha,
             "state": state},
        )
        core = {
            "schema":
                "campaignx.first_letters_discovery_history_reconciliation.v1",
            "reconciliation_id": reconciliation_id,
            "mission_id": mission_id, "state": state,
            "watermark": watermark,
            "watermark_sha256": content_sha256(watermark),
            "manifest": manifest, "manifest_sha256": manifest_sha,
            "fixed_units": (
                sum(int(graph["logical_units"]) for graph in retained_graphs)
                if complete else 0
            ), "reason": reason,
            "allow_unvalidated": False,
        }
        reconciliation = {
            **core, "reconciliation_sha256": content_sha256(core),
            "created_at": utc_now(),
        }
        cursor.execute(
            """INSERT INTO
               segment_first_letters_discovery_history_reconciliations_v19
               (reconciliation_id,mission_id,state,watermark_sha256,manifest,
                manifest_sha256,fixed_units,reason,reconciliation,
                reconciliation_sha256,created_at)
               VALUES(%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s,%s)
               ON CONFLICT(mission_id,manifest_sha256,state) DO NOTHING""",
            (
                reconciliation_id, mission_id, state,
                reconciliation["watermark_sha256"],
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                manifest_sha, reconciliation["fixed_units"], reason,
                json.dumps(reconciliation, sort_keys=True,
                           separators=(",", ":")),
                reconciliation["reconciliation_sha256"],
                reconciliation["created_at"],
            ),
        )
        if not complete:
            cursor.execute(
                """INSERT INTO segment_first_letters_discovery_compute_blocks
                   (mission_id,reason,evidence,created_at)
                   VALUES(%s,%s,%s::jsonb,%s)
                   ON CONFLICT(mission_id) DO NOTHING""",
                (
                    mission_id, reason,
                    json.dumps(reconciliation, sort_keys=True,
                               separators=(",", ":")),
                    utc_now(),
                ),
            )
        else:
            for graph in retained_graphs:
                projection_sha = content_sha256(graph)
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
                    "schema": "campaignx.first_letters_discovery_"
                        "historical_import.v1",
                    "import_id": import_id,
                    "reservation_id": graph["retained_row_ids"][
                        "reservation_id"
                    ],
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
                imported = {
                    **import_core,
                    "import_sha256": content_sha256(import_core),
                    "created_at": reconciliation["created_at"],
                }
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_historical_imports_v19
                       (import_id,reservation_id,mission_id,
                        logical_execution_id,producer_kind,
                        source_snapshot_sha256,profile_file_sha256,item_id,
                        fixed_units,retained_row_ids,
                        retained_projection_sha256,history_manifest_sha256,
                        import_binding,import_sha256,created_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,24,%s::jsonb,%s,%s,
                              %s::jsonb,%s,%s)
                       ON CONFLICT(mission_id,logical_execution_id) DO NOTHING""",
                    (
                        import_id, imported["reservation_id"], mission_id,
                        imported["logical_execution_id"],
                        imported["producer_kind"],
                        imported["source_snapshot_sha256"],
                        imported["profile_file_sha256"], imported["item_id"],
                        json.dumps(imported["retained_row_ids"], sort_keys=True,
                                   separators=(",", ":")),
                        projection_sha, manifest_sha,
                        json.dumps(imported, sort_keys=True,
                                   separators=(",", ":")),
                        imported["import_sha256"], imported["created_at"],
                    ),
                )
        return reconciliation

    def reconcile_first_letters_discovery_history(
        self, *, mission_id: str,
    ) -> dict[str, Any]:
        """Persist a PostgreSQL history cut while holding its mission locks."""

        if not isinstance(mission_id, str) or not mission_id:
            raise ValueError("discovery history mission ID is invalid")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_id}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-compute-mission:{mission_id}",),
                )
                reconciliation = self._first_letters_empty_history_tx(
                    cursor, mission_id=mission_id,
                )
        return reconciliation

    def _first_letters_baseline_reconciliation_tx(
        self, cursor: Any, *, request_id: str,
        budget_admission_sha256: str, history: dict[str, Any],
        profile_bytes: bytes, profile_source_snapshot_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from .discovery_bridge import (
            validate_first_letters_baseline_reconciliation,
        )
        from .seed_probe import load_first_letters_discovery_profile_bytes

        cursor.execute(
            """SELECT admission FROM segment_campaign_budget_admissions
                WHERE admission_sha256=%s FOR SHARE""",
            (budget_admission_sha256,),
        )
        rows = cursor.fetchall()
        if len(rows) != 1:
            raise ValueError("baseline budget admission is missing or ambiguous")
        admission = copy.deepcopy(rows[0]["admission"])
        if (
            admission.get("schema")
                != "campaignx.first_letters_task_budget_admission.v1"
            or admission.get("admission_sha256") != budget_admission_sha256
            or budget_admission_sha256 != content_sha256({
                key: value for key, value in admission.items()
                if key != "admission_sha256"
            })
        ):
            raise ValueError("baseline budget admission hash is invalid")
        execution = admission.get("execution_bindings") or {}
        cursor.execute(
            """SELECT * FROM segment_source_snapshots
                WHERE source_snapshot_id=%s FOR SHARE""",
            (execution.get("source_snapshot_id"),),
        )
        source_row = cursor.fetchone()
        if source_row is None:
            raise ValueError("baseline source snapshot is missing")
        source = self._snapshot(source_row)
        profile_source = source
        if profile_source_snapshot_id is not None:
            cursor.execute(
                """SELECT * FROM segment_source_snapshots
                    WHERE source_snapshot_id=%s FOR SHARE""",
                (profile_source_snapshot_id,),
            )
            profile_source_row = cursor.fetchone()
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
        cursor.execute(
            """SELECT *
                 FROM segment_first_letters_discovery_compute_caps
                WHERE mission_id=%s FOR UPDATE""",
            (admission["mission_id"],),
        )
        cap_row = cursor.fetchone()
        if cap_row is None:
            raise ValueError("baseline compute cap is missing")
        cap = self._validated_discovery_compute_cap(cap_row["authority"])
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
            cursor.execute(
                """SELECT * FROM segment_tasks
                    WHERE mission_id=%s AND cell_id=%s FOR SHARE""",
                (admission["mission_id"], item_id),
            )
            tasks = cursor.fetchall()
            if len(tasks) != 1:
                raise ValueError("baseline task cohort is missing or ambiguous")
            task = tasks[0]
            payload = copy.deepcopy(task["payload"] or {})
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
                region = {
                    "minimum": task["bounds_xyz"][0],
                    "maximum": task["bounds_xyz"][1],
                }
            opportunity = payload.get("scientific_opportunity_id")
            p0_id = payload.get(
                "accepted_p0_artifact_id", payload.get("p0_artifact_id")
            )
            p0_sha = payload.get(
                "accepted_p0_artifact_sha256", payload.get("p0_artifact_sha256")
            )
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
                "item_id": item_id, "selection_rank": rank,
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
            "mission_id": admission["mission_id"], "request_id": request_id,
            "sample_id": admission["sample_id"],
            "budget_admission_sha256": budget_admission_sha256,
            "source_snapshot_id": source["source_snapshot_id"],
            "source_snapshot_sha256": source["source_snapshot_sha256"],
            "source_content_lock_sha256": source["source_content_lock_sha256"],
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
            "profile_scientific_core_sha256": profile["scientific_core_sha256"],
            "policy_sha256": cap["policy_chain_sha256"],
            "deployed_revision": cap["deployed_revision"],
            "history_manifest_sha256": history["manifest_sha256"],
            "mode": "shadow", "namespace": "NONCANONICAL_DISCOVERY",
            "canonical_admission": "PROHIBITED", "top_k": 2,
            "probe_generations": 12, "maximum_attempts_per_candidate": 1,
            "units_per_item": 24, "allow_unvalidated": False,
        }
        return validate_first_letters_baseline_reconciliation({
            **core, "reconciliation_sha256": content_sha256(core),
        }), source

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
        resolver = self._first_letters_discovery_profile_resolver
        if resolver is None:
            raise ValueError("DISCOVERY_SERVER_PROFILE_AUTHORITY_REQUIRED")
        result_identity = None
        history_transaction = _CommitIncompleteHistory(self.connect())
        with history_transaction as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT admission FROM segment_campaign_budget_admissions
                        WHERE admission_sha256=%s""",
                    (budget_admission_sha256,),
                )
                rows = cursor.fetchall()
                if len(rows) != 1:
                    raise ValueError("baseline budget admission is missing")
                admission = copy.deepcopy(rows[0]["admission"])
                source_id = (admission.get("execution_bindings") or {}).get(
                    "source_snapshot_id"
                )
                arm = None
                if arm_id is not None:
                    arm_resolver = self._first_letters_experimental_arm_resolver
                    arm = arm_resolver(arm_id) if arm_resolver else None
                    if not isinstance(arm, dict):
                        raise ValueError("DISCOVERY_SERVER_ARM_AUTHORITY_REQUIRED")
                profile_source_id = (
                    arm.get("source_snapshot_id") if arm is not None
                    else source_id
                )
                profile_bytes = resolver(
                    admission["mission_id"], profile_source_id
                )
                if not isinstance(profile_bytes, bytes):
                    raise ValueError("DISCOVERY_SERVER_PROFILE_AUTHORITY_REQUIRED")
                mission_id = admission["mission_id"]
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_id}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-compute-mission:{mission_id}",),
                )
                history = self._first_letters_empty_history_tx(
                    cursor, mission_id=mission_id,
                )
                if history["state"] != "COMPLETE":
                    raise _PostgresIncompleteHistory
                reconciliation, baseline_source = (
                    self._first_letters_baseline_reconciliation_tx(
                    cursor, request_id=request_id,
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
                    cursor.execute(
                        """SELECT * FROM segment_source_snapshots
                            WHERE source_snapshot_id=%s FOR SHARE""",
                        (arm.get("source_snapshot_id"),),
                    )
                    arm_source = cursor.fetchone()
                    if arm_source is None:
                        raise ValueError("experimental arm source is missing")
                    adapter = adapt_first_letters_alternative_shadow(
                        reconciliation, arm, self._snapshot(arm_source),
                    )
                generic = adapter["generic_work_authority"]
                request_sha = content_sha256(adapter)
                cursor.execute(
                    """SELECT request_sha256
                         FROM segment_first_letters_discovery_compute_reservations
                        WHERE mission_id=%s AND request_id=%s FOR UPDATE""",
                    (mission_id, request_id),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["request_sha256"] != request_sha:
                        raise ValueError("DISCOVERY_COMPUTE_RESERVATION_CONFLICT")
                    result_identity = (mission_id, request_id)
                else:
                    cursor.execute(
                        """SELECT *
                             FROM segment_first_letters_discovery_compute_caps
                            WHERE mission_id=%s FOR UPDATE""",
                        (mission_id,),
                    )
                    cap = cursor.fetchone()
                    cursor.execute(
                        """SELECT COALESCE(SUM(reserved_units),0) AS total
                             FROM segment_first_letters_discovery_compute_reservations
                            WHERE mission_id=%s""",
                        (mission_id,),
                    )
                    used = int(cursor.fetchone()["total"])
                    items = generic["ordered_item_ids"]
                    reserved = len(items) * 24
                    if cap is None or used + reserved > int(cap["cap_units"]):
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
                        "ordered_item_ids": items,
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
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_compute_reservations
                           (reservation_id,mission_id,request_id,work_kind,
                            work_authority_id,work_authority_sha256,
                            ordered_item_ids_sha256,item_count,units_per_item,
                            reserved_units,reserved_before_units,
                            reserved_after_units,source,reservation,
                            reservation_sha256,request_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,24,%s,%s,%s,
                                  'RESERVED_BEFORE_EXECUTION',%s::jsonb,%s,%s,%s)""",
                        (
                            reservation["reservation_id"], mission_id,
                            request_id, adapter["work_kind"],
                            generic["work_authority_id"],
                            generic["work_authority_sha256"],
                            reservation["ordered_item_ids_sha256"], len(items),
                            reserved, used, used + reserved,
                            json.dumps(reservation, sort_keys=True,
                                       separators=(",", ":")),
                            reservation["reservation_sha256"], request_sha,
                            reservation["created_at"],
                        ),
                    )
                    work_core = {
                        "schema":
                            "campaignx.first_letters_discovery_work_binding.v1",
                        "reservation_id": reservation["reservation_id"],
                        "reservation_sha256": reservation["reservation_sha256"],
                        "mission_id": mission_id, "request_id": request_id,
                        "work_kind": adapter["work_kind"],
                        "dispatch_kind": {
                            "BASELINE_ARM": "BASELINE_DISPATCH",
                            "ALTERNATIVE_SOURCE_ARM":
                                "ALTERNATIVE_SOURCE_DISPATCH",
                        }[adapter["work_kind"]],
                        "work_authority": generic,
                        "ordered_item_ids": items,
                        "allow_unvalidated": False,
                    }
                    work = {**work_core, "work_sha256": content_sha256(work_core)}
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_work_bindings
                           (reservation_id,mission_id,request_id,work_kind,
                            dispatch_kind,work,work_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s)""",
                        (
                            reservation["reservation_id"], mission_id,
                            request_id, adapter["work_kind"],
                            work["dispatch_kind"],
                            json.dumps(work, sort_keys=True,
                                       separators=(",", ":")),
                            work["work_sha256"], utc_now(),
                        ),
                    )
                    if failpoint == "bridge.after_reservation_before_adapter":
                        raise RuntimeError(failpoint)
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_native_adapters_v19
                           (reservation_id,mission_id,request_id,work_kind,
                            producer_kind,native_schema,native_authority,
                            native_authority_sha256,generic_work_authority,
                            generic_work_authority_sha256,profile_bytes,adapter,
                            adapter_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,
                                  %s,%s,%s::jsonb,%s,%s)""",
                        (
                            reservation["reservation_id"], mission_id,
                            request_id, adapter["work_kind"],
                            adapter["producer_kind"], adapter["native_schema"],
                            json.dumps(adapter["native_authority"], sort_keys=True,
                                       separators=(",", ":")),
                            adapter["native_authority_sha256"],
                            json.dumps(generic, sort_keys=True,
                                       separators=(",", ":")),
                            adapter["generic_work_authority_sha256"],
                            profile_bytes,
                            json.dumps(adapter, sort_keys=True,
                                       separators=(",", ":")),
                            adapter["adapter_sha256"], utc_now(),
                        ),
                    )
                    if failpoint == "bridge.after_adapter_before_dispatch":
                        raise RuntimeError(failpoint)
                    dispatch = build_first_letters_discovery_dispatch(
                        reservation, adapter,
                    )
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_dispatches_v19
                           (dispatch_id,reservation_id,mission_id,request_id,
                            work_kind,adapter_sha256,profile_file_sha256,
                            source_snapshot_sha256,ordered_item_ids_sha256,
                            item_count,dispatch,dispatch_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)""",
                        (
                            dispatch["dispatch_id"], reservation["reservation_id"],
                            mission_id, request_id, adapter["work_kind"],
                            adapter["adapter_sha256"],
                            dispatch["profile_file_sha256"],
                            dispatch["source_snapshot_sha256"],
                            dispatch["ordered_item_ids_sha256"],
                            dispatch["item_count"],
                            json.dumps(dispatch, sort_keys=True,
                                       separators=(",", ":")),
                            dispatch["dispatch_sha256"], utc_now(),
                        ),
                    )
                    if failpoint == "bridge.after_dispatch_before_jobs":
                        raise RuntimeError(failpoint)
                    for job in build_first_letters_discovery_jobs(
                        dispatch, adapter,
                    ):
                        cursor.execute(
                            """INSERT INTO
                               segment_first_letters_discovery_jobs_v19
                               (job_id,dispatch_id,reservation_id,item_order,
                                item_id,work_item_binding_sha256,
                                profile_file_sha256,source_snapshot_sha256,
                                job,job_sha256,created_at)
                               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)""",
                            (
                                job["job_id"], dispatch["dispatch_id"],
                                reservation["reservation_id"], job["item_order"],
                                job["item_id"], job["work_item_binding_sha256"],
                                job["profile_file_sha256"],
                                job["source_snapshot_sha256"],
                                json.dumps(job, sort_keys=True,
                                           separators=(",", ":")),
                                job["job_sha256"], utc_now(),
                            ),
                        )
                        if failpoint == "bridge.after_each_job":
                            raise RuntimeError(failpoint)
                    if failpoint in {
                        "bridge.after_jobs_before_commit", "bridge.before_commit",
                    }:
                        raise RuntimeError(failpoint)
                    result_identity = (mission_id, request_id)
        if history_transaction.incomplete:
            raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
        if failpoint == "bridge.commit_outcome_unknown":
            raise RuntimeError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        if result_identity is None:
            raise RuntimeError("CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK")
        return self.read_first_letters_discovery_request(*result_identity)

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
        self, cursor: Any, *, adapter: dict[str, Any],
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
        current_reconciliation, baseline_source = (
            self._first_letters_baseline_reconciliation_tx(
                cursor,
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
            cursor.execute(
                "SELECT * FROM segment_source_snapshots "
                "WHERE source_snapshot_id=%s FOR SHARE",
                (profile_source_id,),
            )
            source_row = cursor.fetchone()
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
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT r.reservation,r.reservation_sha256,
                              w.work,w.work_sha256,
                              a.adapter,a.profile_bytes,
                              a.reservation_id AS adapter_reservation_id,
                              a.mission_id AS adapter_mission_id,
                              a.request_id AS adapter_request_id,
                              a.work_kind AS adapter_work_kind,
                              a.producer_kind AS adapter_producer_kind,
                              a.native_schema AS adapter_native_schema,
                              a.native_authority AS adapter_native_authority,
                              a.native_authority_sha256
                                AS adapter_native_authority_sha256,
                              a.generic_work_authority
                                AS adapter_generic_work_authority,
                              a.generic_work_authority_sha256
                                AS adapter_generic_work_authority_sha256,
                              a.adapter_sha256,
                              d.dispatch,d.dispatch_id,
                              d.reservation_id AS dispatch_reservation_id,
                              d.mission_id AS dispatch_mission_id,
                              d.request_id AS dispatch_request_id,
                              d.work_kind AS dispatch_work_kind,
                              d.adapter_sha256 AS dispatch_adapter_sha256,
                              d.profile_file_sha256
                                AS dispatch_profile_file_sha256,
                              d.source_snapshot_sha256
                                AS dispatch_source_snapshot_sha256,
                              d.ordered_item_ids_sha256
                                AS dispatch_ordered_item_ids_sha256,
                              d.item_count AS dispatch_item_count,
                              d.dispatch_sha256
                         FROM segment_first_letters_discovery_compute_reservations r
                         JOIN segment_first_letters_discovery_work_bindings w
                           ON w.reservation_id=r.reservation_id
                         JOIN segment_first_letters_discovery_native_adapters_v19 a
                           ON a.reservation_id=r.reservation_id
                         JOIN segment_first_letters_discovery_dispatches_v19 d
                           ON d.reservation_id=r.reservation_id
                        WHERE r.mission_id=%s AND r.request_id=%s""",
                    (mission_id, request_id),
                )
                rows = cursor.fetchall()
                if len(rows) != 1:
                    raise ValueError("CONTROL_INCOMPLETE_DISCOVERY_DISPATCH")
                row = rows[0]
                reservation = copy.deepcopy(row["reservation"])
                work = copy.deepcopy(row["work"])
                adapter = validate_first_letters_discovery_native_adapter(
                    row["adapter"]
                )
                dispatch = copy.deepcopy(row["dispatch"])
                cursor.execute(
                    """SELECT * FROM segment_first_letters_discovery_jobs_v19
                        WHERE reservation_id=%s ORDER BY item_order,job_id""",
                    (reservation["reservation_id"],),
                )
                job_rows = cursor.fetchall()
                jobs = [copy.deepcopy(value["job"]) for value in job_rows]
        expected_dispatch = build_first_letters_discovery_dispatch(
            reservation, adapter,
        )
        expected_jobs = build_first_letters_discovery_jobs(
            expected_dispatch, adapter,
        )
        scalar_jobs_valid = all(
            persisted["job_id"] == job["job_id"]
            and persisted["dispatch_id"] == job["dispatch_id"]
            and persisted["reservation_id"] == job["reservation_id"]
            and persisted["item_order"] == job["item_order"]
            and persisted["item_id"] == job["item_id"]
            and persisted["work_item_binding_sha256"]
                == job["work_item_binding_sha256"]
            and persisted["profile_file_sha256"]
                == job["profile_file_sha256"]
            and persisted["source_snapshot_sha256"]
                == job["source_snapshot_sha256"]
            and persisted["job_sha256"] == job["job_sha256"]
            for persisted, job in zip(job_rows, jobs, strict=True)
        ) if len(job_rows) == len(jobs) else False
        if (
            dispatch != expected_dispatch or jobs != expected_jobs
            or work.get("reservation_sha256")
                != reservation.get("reservation_sha256")
            or work.get("work_authority")
                != adapter["generic_work_authority"]
            or row["reservation_sha256"]
                != reservation.get("reservation_sha256")
            or row["work_sha256"] != work.get("work_sha256")
            or row["adapter_reservation_id"]
                != reservation["reservation_id"]
            or row["adapter_mission_id"] != reservation["mission_id"]
            or row["adapter_request_id"] != reservation["request_id"]
            or row["adapter_work_kind"] != adapter["work_kind"]
            or row["adapter_producer_kind"] != adapter["producer_kind"]
            or row["adapter_native_schema"] != adapter["native_schema"]
            or copy.deepcopy(row["adapter_native_authority"])
                != adapter["native_authority"]
            or row["adapter_native_authority_sha256"]
                != adapter["native_authority_sha256"]
            or copy.deepcopy(row["adapter_generic_work_authority"])
                != adapter["generic_work_authority"]
            or row["adapter_generic_work_authority_sha256"]
                != adapter["generic_work_authority_sha256"]
            or row["adapter_sha256"] != adapter["adapter_sha256"]
            or hashlib.sha256(bytes(row["profile_bytes"])).hexdigest()
                != adapter["profile_file_sha256"]
            or row["dispatch_id"] != dispatch["dispatch_id"]
            or row["dispatch_reservation_id"]
                != reservation["reservation_id"]
            or row["dispatch_mission_id"] != reservation["mission_id"]
            or row["dispatch_request_id"] != reservation["request_id"]
            or row["dispatch_work_kind"] != adapter["work_kind"]
            or row["dispatch_adapter_sha256"] != adapter["adapter_sha256"]
            or row["dispatch_profile_file_sha256"]
                != dispatch["profile_file_sha256"]
            or row["dispatch_source_snapshot_sha256"]
                != dispatch["source_snapshot_sha256"]
            or row["dispatch_ordered_item_ids_sha256"]
                != dispatch["ordered_item_ids_sha256"]
            or row["dispatch_item_count"] != dispatch["item_count"]
            or row["dispatch_sha256"] != dispatch["dispatch_sha256"]
            or not scalar_jobs_valid
        ):
            raise ValueError("CONTROL_INCOMPLETE_DISCOVERY_DISPATCH")
        try:
            with self.connect() as connection:
                with connection.cursor() as cursor:
                    current_adapter = (
                        self._current_first_letters_discovery_adapter_tx(
                            cursor, adapter=adapter,
                            persisted_profile_bytes=bytes(row["profile_bytes"]),
                        )
                    )
            if current_adapter != adapter:
                raise ValueError("current discovery adapter authority differs")
        except Exception as error:
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
        """Claim the same noncanonical producer seam as SQLite."""

        from .seed_probe import (
            _accept_first_letters_discovery_executor_claim,
            _bind_first_letters_discovery_executor_claim,
            _derive_first_letters_discovery_run_authority,
        )

        if (not isinstance(item_id, str) or not item_id
                or isinstance(lease_seconds, bool)
                or not isinstance(lease_seconds, int) or lease_seconds < 30
                or not isinstance(profile_bytes, bytes)):
            raise ValueError("discovery producer claim arguments are invalid")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-evidence-reservation:{reservation_id}",),
                )
                cursor.execute(
                    """SELECT 1
                         FROM segment_first_letters_discovery_native_adapters_v19
                        WHERE reservation_id=%s""",
                    (reservation_id,),
                )
                controlled = cursor.fetchone() is not None
                if controlled and _job_id is None:
                    raise ValueError(
                        "controlled discovery reservations require a job ID claim"
                    )
                if _job_id is not None:
                    cursor.execute(
                        """SELECT j.job,j.job_sha256,j.item_id,j.dispatch_id,
                                  d.dispatch_sha256,a.profile_bytes
                             FROM segment_first_letters_discovery_jobs_v19 j
                             JOIN segment_first_letters_discovery_dispatches_v19 d
                               ON d.dispatch_id=j.dispatch_id
                             JOIN segment_first_letters_discovery_native_adapters_v19 a
                               ON a.reservation_id=j.reservation_id
                            WHERE j.job_id=%s AND j.reservation_id=%s
                            FOR UPDATE OF j,d,a""",
                        (_job_id, reservation_id),
                    )
                    graph = cursor.fetchone()
                    if graph is None:
                        raise ValueError("discovery job graph is missing")
                    persisted_job = copy.deepcopy(graph["job"])
                    if (
                        persisted_job.get("job_id") != _job_id
                        or persisted_job.get("job_sha256") != graph["job_sha256"]
                        or persisted_job.get("job_sha256") != content_sha256({
                            key: value for key, value in persisted_job.items()
                            if key != "job_sha256"
                        })
                        or persisted_job.get("item_id") != item_id
                        or graph["item_id"] != item_id
                        or persisted_job.get("dispatch_id")
                            != graph["dispatch_id"]
                        or persisted_job.get("dispatch_sha256")
                            != graph["dispatch_sha256"]
                        or persisted_job.get("profile_file_sha256")
                            != hashlib.sha256(profile_bytes).hexdigest()
                        or bytes(graph["profile_bytes"]) != profile_bytes
                    ):
                        raise ValueError("discovery job graph differs before claim")
                cursor.execute(
                    """SELECT r.reservation,r.reservation_sha256,
                              w.work,w.work_sha256
                         FROM segment_first_letters_discovery_compute_reservations r
                         JOIN segment_first_letters_discovery_work_bindings w
                           ON w.reservation_id=r.reservation_id
                        WHERE r.reservation_id=%s FOR UPDATE OF r,w""",
                    (reservation_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("discovery reservation/work authority is missing")
                reservation = copy.deepcopy(row["reservation"])
                work = copy.deepcopy(row["work"])
                if (reservation.get("reservation_sha256") !=
                        row["reservation_sha256"]
                        or reservation.get("reservation_sha256") !=
                            content_sha256({
                                key: value for key, value in reservation.items()
                                if key not in {"reservation_sha256", "created_at"}
                            })
                        or work.get("work_sha256") != row["work_sha256"]
                        or work.get("work_sha256") != content_sha256({
                            key: value for key, value in work.items()
                            if key != "work_sha256"
                        })):
                    raise ValueError(
                        "discovery persisted reservation/work authority drift"
                    )
                authority = work.get("work_authority") or {}
                matches = [
                    binding for binding in authority.get(
                        "ordered_item_bindings", []
                    ) if binding.get("item_id") == item_id
                ]
                if len(matches) != 1:
                    raise ValueError("discovery work item authority is missing")
                binding = matches[0]
                cursor.execute(
                    """SELECT * FROM segment_source_snapshots
                        WHERE source_snapshot_id=%s FOR SHARE""",
                    (binding["source_snapshot_id"],),
                )
                source_row = cursor.fetchone()
                if source_row is None:
                    raise ValueError("discovery work-item source is unavailable")
                source = self._snapshot(source_row)
                parent_task_id = binding["parent_task_id"]
                parent_attempt_id = binding["parent_attempt_id"]
                if parent_task_id is not None:
                    cursor.execute(
                        """SELECT mission_id,source_snapshot_id,cell_id,
                                  grid_version,payload
                             FROM segment_tasks WHERE task_id=%s FOR SHARE""",
                        (parent_task_id,),
                    )
                    parent = cursor.fetchone()
                    if (parent is None
                            or parent["mission_id"] != reservation["mission_id"]
                            or parent["source_snapshot_id"] !=
                                source["source_snapshot_id"]
                            or parent["cell_id"] != item_id):
                        raise ValueError(
                            "optional discovery parent lineage is unregistered"
                        )
                    parent_payload = copy.deepcopy(parent["payload"])
                    expected_p0_id = parent_payload.get(
                        "accepted_p0_artifact_id",
                        parent_payload.get("p0_artifact_id"),
                    )
                    expected_p0_sha = parent_payload.get(
                        "accepted_p0_artifact_sha256",
                        parent_payload.get("p0_artifact_sha256"),
                    )
                    if (binding["sample_id"] !=
                            parent_payload.get("sample_id")
                            or binding["grid_version"] !=
                                parent["grid_version"]
                            or binding["scientific_opportunity_id"] !=
                                parent_payload.get("scientific_opportunity_id")
                            or binding["accepted_p0_artifact_id"] !=
                                expected_p0_id
                            or binding["accepted_p0_artifact_sha256"] !=
                                expected_p0_sha):
                        raise ValueError(
                            "discovery opportunity/P0 authority differs from "
                            "persisted parent task"
                        )
                if parent_attempt_id is not None:
                    cursor.execute(
                        "SELECT task_id FROM segment_attempts WHERE attempt_id=%s FOR SHARE",
                        (parent_attempt_id,),
                    )
                    attempt = cursor.fetchone()
                    if attempt is None or attempt["task_id"] != parent_task_id:
                        raise ValueError(
                            "optional discovery parent attempt is unregistered"
                        )
                derived = _derive_first_letters_discovery_run_authority(
                    profile_bytes=profile_bytes, reservation=reservation,
                    work=work, binding=binding, source=source,
                )
                run_id = derived["run_authority"]["run_id"]
                cursor.execute(
                    "SELECT 1 FROM "
                    "segment_first_letters_discovery_evidence_runs "
                    "WHERE run_id=%s",
                    (run_id,),
                )
                if cursor.fetchone() is not None:
                    raise ValueError("discovery evidence run already claimed")
                registration = (
                    self._discovery_executor_registration_from_cursor(cursor)
                )
                lease_deadline = _deadline(lease_seconds)
                derived = _bind_first_letters_discovery_executor_claim(
                    derived=derived,
                    registration=registration,
                    lease_expires_at=_json_utc_timestamp(lease_deadline),
                )
                executor_claim_token = derived.pop("_executor_claim_token")
                run_authority = derived["run_authority"]
                worker_id = run_authority["worker_id"]
                run_token = secrets.token_urlsafe(32)
                cursor.execute(
                    """INSERT INTO segment_first_letters_discovery_evidence_runs
                       (run_id,reservation_id,mission_id,request_id,parent_task_id,
                        parent_attempt_id,worker_id,cell_id,source_snapshot_id,
                        run_token_sha256,lease_expires_at,profile_bytes,
                        profile_file_sha256,provider_request,run_authority,
                        run_authority_sha256,state,created_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                              %s::jsonb,%s::jsonb,%s,'CLAIMED',%s)""",
                    (
                        run_id, reservation_id, reservation["mission_id"],
                        reservation["request_id"], parent_task_id,
                        parent_attempt_id, worker_id, item_id,
                        source["source_snapshot_id"],
                        hashlib.sha256(run_token.encode("utf-8")).hexdigest(),
                        lease_deadline, profile_bytes,
                        derived["profile_file_sha256"],
                        json.dumps(derived["provider_request"], sort_keys=True,
                                   separators=(",", ":")),
                        json.dumps(run_authority, sort_keys=True,
                                   separators=(",", ":")),
                        run_authority["run_authority_sha256"], utc_now(),
                    ),
                )
                claim = run_authority["executor_claim"]
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_executor_claims
                       (claim_id,run_id,worker_id,executor_id,executor_sha256,
                        capability,claim_attempt_number,
                        execution_lease_token_sha256,lease_expires_at,claim,
                        claim_sha256,state,created_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,
                              'CLAIMED',now())""",
                    (
                        claim["claim_id"], run_id, claim["worker_id"],
                        claim["executor_id"], claim["executor_sha256"],
                        claim["capability"], claim["claim_attempt_number"],
                        claim["execution_lease_token_sha256"], lease_deadline,
                        json.dumps(
                            claim, sort_keys=True, separators=(",", ":")
                        ), claim["claim_sha256"],
                    ),
                )
                _accept_first_letters_discovery_executor_claim(
                    executor=self._first_letters_discovery_executor,
                    run_id=run_id, claim_token=executor_claim_token,
                )
        return _FirstLettersDiscoveryRunHandle(
            run_id=run_id, run_token=run_token, worker_id=worker_id,
            cell_id=item_id,
            provider_request=copy.deepcopy(derived["provider_request"]),
        )

    def claim_first_letters_discovery_job(
        self, *, job_id: str, lease_seconds: int,
    ):
        from .discovery_bridge import build_first_letters_discovery_job_claim

        if not isinstance(job_id, str) or not job_id:
            raise ValueError("discovery job ID is invalid")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT j.reservation_id,j.item_id,a.profile_bytes,
                              r.mission_id,r.request_id
                         FROM segment_first_letters_discovery_jobs_v19 j
                         JOIN segment_first_letters_discovery_native_adapters_v19 a
                           ON a.reservation_id=j.reservation_id
                         JOIN segment_first_letters_discovery_compute_reservations r
                           ON r.reservation_id=j.reservation_id
                        WHERE j.job_id=%s""",
                    (job_id,),
                )
                row = cursor.fetchone()
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
        from .discovery_bridge import (
            validate_first_letters_discovery_job_claim,
        )

        claim = validate_first_letters_discovery_job_claim(claim)
        identity = None
        history_transaction = _CommitIncompleteHistory(self.connect())
        with history_transaction as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT j.job_sha256,j.item_id,d.dispatch_sha256,
                              a.adapter,a.adapter_sha256,a.profile_bytes,
                              r.reservation_sha256,r.mission_id,r.request_id
                         FROM segment_first_letters_discovery_jobs_v19 j
                         JOIN segment_first_letters_discovery_dispatches_v19 d
                           ON d.dispatch_id=j.dispatch_id
                         JOIN segment_first_letters_discovery_native_adapters_v19 a
                           ON a.reservation_id=j.reservation_id
                         JOIN segment_first_letters_discovery_compute_reservations r
                           ON r.reservation_id=j.reservation_id
                        WHERE j.job_id=%s FOR SHARE OF j,d,a,r""",
                    (claim.job_id,),
                )
                row = cursor.fetchone()
                if row is None or (
                    row["job_sha256"] != claim.job_sha256
                    or row["item_id"] != claim.item_id
                    or row["dispatch_sha256"] != claim.dispatch_sha256
                    or row["adapter_sha256"] != claim.adapter_sha256
                    or row["reservation_sha256"] != claim.reservation_sha256
                    or hashlib.sha256(bytes(row["profile_bytes"])).hexdigest()
                        != claim.profile_file_sha256
                ):
                    raise ValueError("discovery claim graph differs")
                mission_id = row["mission_id"]
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_id}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-compute-mission:{mission_id}",),
                )
                cursor.execute(
                    """SELECT reason
                         FROM segment_first_letters_discovery_compute_blocks
                        WHERE mission_id=%s FOR SHARE""",
                    (mission_id,),
                )
                if cursor.fetchone() is not None:
                    raise _PostgresIncompleteHistory
                adapter = copy.deepcopy(row["adapter"])
                native = adapter.get("native_authority") or {}
                reconciliation = (
                    native if adapter.get("producer_kind")
                        == "BASELINE_RECONCILIATION"
                    else native.get("baseline_reconciliation") or {}
                )
                history = self._first_letters_empty_history_tx(
                    cursor, mission_id=mission_id,
                )
                if (
                    history["state"] != "COMPLETE"
                    or history["manifest_sha256"]
                        != reconciliation.get("history_manifest_sha256")
                ):
                    if history["state"] == "COMPLETE":
                        evidence = {
                            "schema": "campaignx.first_letters_discovery_"
                                "history_claim_mismatch.v1",
                            "mission_id": mission_id,
                            "current_history_manifest_sha256": history[
                                "manifest_sha256"
                            ],
                            "claimed_history_manifest_sha256": (
                                reconciliation.get("history_manifest_sha256")
                            ),
                            "allow_unvalidated": False,
                        }
                        cursor.execute(
                            """INSERT INTO
                               segment_first_letters_discovery_compute_blocks
                               (mission_id,reason,evidence,created_at)
                               VALUES(%s,%s,%s::jsonb,%s)
                               ON CONFLICT(mission_id) DO NOTHING""",
                            (
                                mission_id,
                                "CONTROL_INCOMPLETE_COMPUTE_LEDGER",
                                json.dumps(
                                    evidence, sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                utc_now(),
                            ),
                        )
                    raise _PostgresIncompleteHistory
                self._first_letters_discovery_lifecycle_claim(
                    cursor, run_handle=claim._run_handle,
                    expected_state="CLAIMED", require_live_lease=True,
                )
                identity = (mission_id, row["request_id"])
        if history_transaction.incomplete:
            raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
        if identity is None:  # pragma: no cover - defensive transaction guard
            raise ValueError("CONTROL_INCOMPLETE_COMPUTE_LEDGER")
        branch = self.read_first_letters_discovery_request(*identity)
        if [job["job_id"] for job in branch["jobs"]].count(claim.job_id) != 1:
            raise ValueError("discovery claim dispatch graph differs")

    def _first_letters_discovery_lifecycle_claim(
        self, cursor: Any, *, run_handle: _FirstLettersDiscoveryRunHandle,
        expected_state: str, require_live_lease: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        from .seed_probe import (
            _validated_first_letters_discovery_executor_claim,
        )

        if type(run_handle) is not _FirstLettersDiscoveryRunHandle:
            raise ValueError("discovery lifecycle requires the sealed run handle")
        cursor.execute(
            """SELECT * FROM segment_first_letters_discovery_evidence_runs
                WHERE run_id=%s FOR UPDATE""",
            (run_handle.run_id,),
        )
        row = cursor.fetchone()
        now = datetime.now(timezone.utc)
        if (row is None or row["state"] != expected_state
                or row["worker_id"] != run_handle.worker_id
                or row["cell_id"] != run_handle.cell_id
                or row["run_token_sha256"] != hashlib.sha256(
                    run_handle.run_token.encode("utf-8")
                ).hexdigest()
                or require_live_lease and row["lease_expires_at"] <= now):
            raise ValueError(
                f"discovery claim owner/token must hold a live {expected_state} lease"
            )
        authority = copy.deepcopy(row["run_authority"])
        provider_request = copy.deepcopy(row["provider_request"])
        if (authority.get("run_authority_sha256") !=
                row["run_authority_sha256"]
                or content_sha256({
                    key: value for key, value in authority.items()
                    if key != "run_authority_sha256"
                }) != row["run_authority_sha256"]):
            raise ValueError("discovery persisted run authority drift")
        registration = self._discovery_executor_registration_from_cursor(cursor)
        claim = _validated_first_letters_discovery_executor_claim(
            authority.get("executor_claim"), run_authority=authority,
            registration=registration, provider_request=provider_request,
        )
        cursor.execute(
            """SELECT * FROM segment_first_letters_discovery_executor_claims
                WHERE run_id=%s FOR UPDATE""",
            (run_handle.run_id,),
        )
        claim_row = cursor.fetchone()
        if (claim_row is None or claim_row["state"] != expected_state
                or copy.deepcopy(claim_row["claim"]) != claim
                or claim_row["claim_sha256"] != claim["claim_sha256"]
                or claim_row["worker_id"] != row["worker_id"]
                or claim_row["lease_expires_at"] != row["lease_expires_at"]
                or _json_utc_timestamp(claim_row["lease_expires_at"]) !=
                    claim["lease_expires_at"]
                or require_live_lease and claim_row["lease_expires_at"] <= now):
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

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-evidence-run:{run_handle.run_id}",),
                )
                self._first_letters_discovery_lifecycle_claim(
                    cursor, run_handle=run_handle, expected_state="CLAIMED",
                    require_live_lease=True,
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_evidence_runs
                          SET state='RUNNING',started_at=now(),
                              last_heartbeat_at=now() WHERE run_id=%s""",
                    (run_handle.run_id,),
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_executor_claims
                          SET state='RUNNING',started_at=now(),
                              last_heartbeat_at=now() WHERE run_id=%s""",
                    (run_handle.run_id,),
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

        lease_deadline = _deadline(lease_seconds)
        lease_expires_at = _json_utc_timestamp(lease_deadline)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-evidence-run:{run_handle.run_id}",),
                )
                _, _, authority, claim = (
                    self._first_letters_discovery_lifecycle_claim(
                        cursor, run_handle=run_handle,
                        expected_state="RUNNING", require_live_lease=True,
                    )
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
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_evidence_runs
                          SET lease_expires_at=%s,run_authority=%s::jsonb,
                              run_authority_sha256=%s,last_heartbeat_at=now()
                        WHERE run_id=%s AND state='RUNNING'""",
                    (
                        lease_deadline,
                        json.dumps(renewed_authority, sort_keys=True,
                                   separators=(",", ":")),
                        renewed_authority["run_authority_sha256"],
                        run_handle.run_id,
                    ),
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_executor_claims
                          SET lease_expires_at=%s,claim=%s::jsonb,
                              claim_sha256=%s,last_heartbeat_at=now()
                        WHERE run_id=%s AND state='RUNNING'""",
                    (
                        lease_deadline,
                        json.dumps(renewed_claim, sort_keys=True,
                                   separators=(",", ":")),
                        renewed_claim["claim_sha256"], run_handle.run_id,
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-evidence-run:{run_handle.run_id}",),
                )
                self._first_letters_discovery_lifecycle_claim(
                    cursor, run_handle=run_handle, expected_state="RUNNING",
                    require_live_lease=False,
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_evidence_runs
                          SET state='CONTROL_INCOMPLETE',incomplete_at=now(),
                              incomplete_reason=%s WHERE run_id=%s""",
                    (reason, run_handle.run_id),
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_executor_claims
                          SET state='CONTROL_INCOMPLETE',incomplete_at=now(),
                              incomplete_reason=%s WHERE run_id=%s""",
                    (reason, run_handle.run_id),
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
        reason = "WORKER_LOST_AFTER_RUNNING"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-evidence-run:{run_id}",),
                )
                cursor.execute(
                    """SELECT *
                         FROM segment_first_letters_discovery_evidence_runs
                        WHERE run_id=%s FOR UPDATE""",
                    (run_id,),
                )
                row = cursor.fetchone()
                cursor.execute(
                    """SELECT *
                         FROM segment_first_letters_discovery_executor_claims
                        WHERE run_id=%s FOR UPDATE""",
                    (run_id,),
                )
                claim_row = cursor.fetchone()
                now = datetime.now(timezone.utc)
                if (row is None or claim_row is None
                        or row["state"] != "RUNNING"
                        or claim_row["state"] != "RUNNING"
                        or row["lease_expires_at"] > now
                        or claim_row["lease_expires_at"] > now
                        or row["lease_expires_at"] !=
                            claim_row["lease_expires_at"]
                        or row["worker_id"] != claim_row["worker_id"]
                        or not row["started_at"]
                        or not row["last_heartbeat_at"]
                        or not claim_row["started_at"]
                        or not claim_row["last_heartbeat_at"]
                        or row["completed_at"] or claim_row["completed_at"]
                        or row["incomplete_at"] or claim_row["incomplete_at"]
                        or row["incomplete_reason"]
                        or claim_row["incomplete_reason"]):
                    raise ValueError(
                        "discovery reconciliation requires one expired RUNNING "
                        "owner"
                    )
                authority = copy.deepcopy(row["run_authority"])
                provider_request = copy.deepcopy(row["provider_request"])
                if (authority.get("run_authority_sha256") !=
                        row["run_authority_sha256"]
                        or content_sha256({
                            key: value for key, value in authority.items()
                            if key != "run_authority_sha256"
                        }) != row["run_authority_sha256"]):
                    raise ValueError("discovery persisted run authority drift")
                registration = (
                    self._persisted_discovery_executor_registration_from_cursor(
                        cursor, worker_id=row["worker_id"],
                    )
                )
                claim = _validated_first_letters_discovery_executor_claim(
                    authority.get("executor_claim"), run_authority=authority,
                    registration=registration,
                    provider_request=provider_request,
                )
                if (copy.deepcopy(claim_row["claim"]) != claim
                        or claim_row["claim_sha256"] != claim["claim_sha256"]
                        or _json_utc_timestamp(
                            claim_row["lease_expires_at"]
                        ) != claim["lease_expires_at"]):
                    raise ValueError(
                        "discovery expired executor claim is invalid"
                    )
                cursor.execute(
                    """SELECT 1
                         FROM segment_first_letters_discovery_evidence_sets
                        WHERE run_id=%s""",
                    (run_id,),
                )
                if cursor.fetchone() is not None:
                    raise ValueError("expired RUNNING run already has evidence")
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_evidence_runs
                          SET state='CONTROL_INCOMPLETE',incomplete_at=now(),
                              incomplete_reason=%s
                        WHERE run_id=%s AND state='RUNNING'""",
                    (reason, run_id),
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_executor_claims
                          SET state='CONTROL_INCOMPLETE',incomplete_at=now(),
                              incomplete_reason=%s
                        WHERE run_id=%s AND state='RUNNING'""",
                    (reason, run_id),
                )
        return self.read_first_letters_discovery_evidence_run_status(run_id)

    def complete_first_letters_discovery_evidence_run(
        self, *, run_handle: _FirstLettersDiscoveryRunHandle,
        provider_response_bytes: bytes,
        failpoint: str | None = None,
    ) -> dict[str, Any]:
        from .seed_probe import (
            _measure_first_letters_discovery_with_executor,
            _produce_first_letters_discovery_evidence_set,
            _validated_first_letters_discovery_executor_claim,
        )

        if (type(run_handle) is not _FirstLettersDiscoveryRunHandle
                or not isinstance(provider_response_bytes, bytes)):
            raise ValueError("discovery completion requires exact sealed inputs")
        if failpoint not in {None, "evidence.after_commit_before_response"}:
            raise ValueError("unknown discovery evidence failpoint")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"discovery-evidence-run:{run_handle.run_id}",),
                )
                cursor.execute(
                    """SELECT * FROM segment_first_letters_discovery_evidence_runs
                        WHERE run_id=%s FOR UPDATE""",
                    (run_handle.run_id,),
                )
                row = cursor.fetchone()
                if (row is None or row["state"] != "RUNNING"
                        or row["worker_id"] != run_handle.worker_id
                        or row["cell_id"] != run_handle.cell_id
                        or row["run_token_sha256"] != hashlib.sha256(
                            run_handle.run_token.encode("utf-8")
                        ).hexdigest()
                        or row["lease_expires_at"] <= datetime.now(timezone.utc)):
                    raise ValueError(
                        "discovery run claim is stale, wrong, incomplete, or not RUNNING"
                    )
                cursor.execute(
                    """SELECT reservation FROM
                       segment_first_letters_discovery_compute_reservations
                       WHERE reservation_id=%s FOR SHARE""",
                    (row["reservation_id"],),
                )
                reservation_row = cursor.fetchone()
                if reservation_row is None:
                    raise ValueError("discovery run reservation disappeared")
                run_authority = copy.deepcopy(row["run_authority"])
                provider_request = copy.deepcopy(row["provider_request"])
                if (row["run_authority_sha256"] !=
                        run_authority.get("run_authority_sha256")
                        or row["run_authority_sha256"] != content_sha256({
                            key: value for key, value in run_authority.items()
                            if key != "run_authority_sha256"
                        })):
                    raise ValueError("discovery persisted run authority drift")
                registration = (
                    self._discovery_executor_registration_from_cursor(cursor)
                )
                cursor.execute(
                    """SELECT * FROM
                       segment_first_letters_discovery_executor_claims
                       WHERE run_id=%s FOR UPDATE""",
                    (run_handle.run_id,),
                )
                claim_row = cursor.fetchone()
                claim = _validated_first_letters_discovery_executor_claim(
                    run_authority.get("executor_claim"),
                    run_authority=run_authority, registration=registration,
                    provider_request=provider_request,
                )
                if (claim_row is None or claim_row["state"] != "RUNNING"
                        or copy.deepcopy(claim_row["claim"]) != claim
                        or claim_row["claim_sha256"] != claim["claim_sha256"]
                        or claim_row["worker_id"] != row["worker_id"]
                        or claim_row["executor_id"] != claim["executor_id"]
                        or claim_row["executor_sha256"] !=
                            claim["executor_sha256"]
                        or claim_row["capability"] != claim["capability"]
                        or claim_row["execution_lease_token_sha256"] !=
                            claim["execution_lease_token_sha256"]
                        or _json_utc_timestamp(
                            claim_row["lease_expires_at"]
                        ) != claim["lease_expires_at"]
                        or claim_row["lease_expires_at"] !=
                            row["lease_expires_at"]
                        or claim_row["lease_expires_at"] <=
                            datetime.now(timezone.utc)):
                    raise ValueError("DISCOVERY_EXECUTOR_CLAIM_STALE")
                cursor.execute(
                    """SELECT * FROM segment_source_snapshots
                        WHERE source_snapshot_id=%s FOR SHARE""",
                    (row["source_snapshot_id"],),
                )
                source_row = cursor.fetchone()
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
                    reservation=copy.deepcopy(reservation_row["reservation"]),
                )
                evidence = copy.deepcopy({
                    key: value for key, value in registered.items()
                    if key not in {"profile_bytes", "retained_files"}
                })
                evidence["inputs"]["provider_response"].pop("response_bytes")
                evidence_sha = content_sha256(evidence)
                cursor.execute(
                    """INSERT INTO segment_first_letters_discovery_evidence_sets
                       (evidence_set_id,run_id,evidence,evidence_set_sha256,created_at)
                       VALUES(%s,%s,%s::jsonb,%s,%s)""",
                    (
                        registered["evidence_set_id"], run_handle.run_id,
                        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
                        evidence_sha, utc_now(),
                    ),
                )
                for index, retained in enumerate(registered["retained_files"]):
                    payload = retained["bytes"]
                    cursor.execute(
                        """INSERT INTO segment_first_letters_discovery_evidence_files
                           (evidence_set_id,file_order,relative_path,role,payload,
                            byte_count,sha256) VALUES(%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            registered["evidence_set_id"], index,
                            retained["relative_path"], retained["role"], payload,
                            len(payload), hashlib.sha256(payload).hexdigest(),
                        ),
                    )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_evidence_runs
                          SET state='COMPLETED',completed_at=%s WHERE run_id=%s""",
                    (utc_now(), run_handle.run_id),
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_executor_claims
                          SET state='COMPLETED',completed_at=now()
                        WHERE run_id=%s""",
                    (run_handle.run_id,),
                )
        if failpoint == "evidence.after_commit_before_response":
            raise RuntimeError(
                "discovery evidence response lost; READBACK_BY_RUN_ID_REQUIRED"
            )
        return self.read_first_letters_discovery_evidence_set(
            registered["evidence_set_id"]
        )

    @staticmethod
    def _read_first_letters_discovery_evidence_set_from_cursor(
        cursor: Any, evidence_set_id: str,
    ) -> dict[str, Any]:
        cursor.execute(
            """SELECT e.*,r.profile_bytes,r.profile_file_sha256,
                      r.run_authority,r.run_authority_sha256,c.reservation
                 FROM segment_first_letters_discovery_evidence_sets e
                 JOIN segment_first_letters_discovery_evidence_runs r
                   ON r.run_id=e.run_id
                 JOIN segment_first_letters_discovery_compute_reservations c
                   ON c.reservation_id=r.reservation_id
                WHERE e.evidence_set_id=%s AND r.state='COMPLETED'""",
            (evidence_set_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(evidence_set_id)
        evidence = copy.deepcopy(row["evidence"])
        run_authority = copy.deepcopy(row["run_authority"])
        profile_bytes = bytes(row["profile_bytes"])
        if (evidence.get("evidence_set_id") != evidence_set_id
                or content_sha256(evidence) != row["evidence_set_sha256"]
                or hashlib.sha256(profile_bytes).hexdigest() !=
                    row["profile_file_sha256"]
                or content_sha256({
                    key: value for key, value in run_authority.items()
                    if key != "run_authority_sha256"
                }) != row["run_authority_sha256"]):
            raise ValueError("registered discovery evidence-set integrity drift")
        cursor.execute(
            """SELECT relative_path,role,payload,byte_count,sha256
                 FROM segment_first_letters_discovery_evidence_files
                WHERE evidence_set_id=%s ORDER BY file_order""",
            (evidence_set_id,),
        )
        retained_files = []
        response_bytes = None
        for file_row in cursor.fetchall():
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
            "profile_bytes": profile_bytes, "inputs": evidence["inputs"],
            "candidate_outcomes": evidence["candidate_outcomes"],
            "retained_files": retained_files,
            "reservation": copy.deepcopy(row["reservation"]),
            "selection": evidence["selection"],
        }

    def read_first_letters_discovery_evidence_set(
        self, evidence_set_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                return self._read_first_letters_discovery_evidence_set_from_cursor(
                    cursor, evidence_set_id
                )

    def read_first_letters_discovery_evidence_run_status(
        self, run_id: str,
    ) -> dict[str, Any]:
        """Read the closed producer lifecycle without exposing either token."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("discovery run ID is invalid")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT run_id,worker_id,cell_id,state,lease_expires_at,
                              started_at,last_heartbeat_at,completed_at,
                              incomplete_at,incomplete_reason
                         FROM segment_first_letters_discovery_evidence_runs
                        WHERE run_id=%s""",
                    (run_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(run_id)
                cursor.execute(
                    """SELECT evidence_set_id
                         FROM segment_first_letters_discovery_evidence_sets
                        WHERE run_id=%s""",
                    (run_id,),
                )
                evidence = cursor.fetchall()
                cursor.execute(
                    """SELECT state,lease_expires_at,started_at,
                              last_heartbeat_at,completed_at,incomplete_at,
                              incomplete_reason
                         FROM segment_first_letters_discovery_executor_claims
                        WHERE run_id=%s""",
                    (run_id,),
                )
                claims = cursor.fetchall()
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

        def timestamp(value):
            return _json_utc_timestamp(value) if value is not None else None

        return {
            "schema": "campaignx.first_letters_discovery_run_status.v1",
            "run_id": row["run_id"], "worker_id": row["worker_id"],
            "cell_id": row["cell_id"], "state": row["state"],
            "lease_expires_at": timestamp(row["lease_expires_at"]),
            "started_at": timestamp(row["started_at"]),
            "last_heartbeat_at": timestamp(row["last_heartbeat_at"]),
            "completed_at": timestamp(row["completed_at"]),
            "incomplete_at": timestamp(row["incomplete_at"]),
            "incomplete_reason": row["incomplete_reason"],
            "evidence_set_id": evidence_set_id,
        }

    def read_first_letters_discovery_evidence_run(
        self, run_id: str,
    ) -> dict[str, Any]:
        """Recover one committed evidence set without rerunning its producer."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT evidence_set_id
                         FROM segment_first_letters_discovery_evidence_sets
                        WHERE run_id=%s""",
                    (run_id,),
                )
                rows = cursor.fetchall()
                if len(rows) != 1:
                    raise KeyError(run_id)
                return self._read_first_letters_discovery_evidence_set_from_cursor(
                    cursor, rows[0]["evidence_set_id"]
                )

    def build_first_letters_discovery_artifact_and_receipt(
        self, evidence_set_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from .seed_probe import (
            _build_first_letters_discovery_artifact_and_receipt_from_evidence_set,
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                registered = self._read_first_letters_discovery_evidence_set_from_cursor(
                    cursor, evidence_set_id
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
            with connection.cursor() as cursor:
                registered = self._read_first_letters_discovery_evidence_set_from_cursor(
                    cursor, evidence_set_id
                )
                return _resolve_discovery_promotion_evidence_from_evidence_set(
                    registered, evidence_set_id=evidence_set_id,
                )

    def discovery_compute_rows(self, mission_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT request_id
                         FROM segment_first_letters_discovery_compute_reservations
                        WHERE mission_id=%s
                        ORDER BY created_at,reservation_id""",
                    (mission_id,),
                )
                request_ids = [row["request_id"] for row in cursor.fetchall()]
        return [
            self.read_discovery_compute_request(mission_id, request_id)
            for request_id in request_ids
        ]

    def validate_discovery_compute_reservation(
        self, reservation_id: str, reservation_sha256: str, *,
        mission_id: str, work_kind: str, work_authority_sha256: str,
        ordered_item_ids: list[str],
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT r.reservation,w.work
                         FROM segment_first_letters_discovery_compute_reservations r
                         JOIN segment_first_letters_discovery_work_bindings w
                           ON w.reservation_id=r.reservation_id
                        WHERE r.reservation_id=%s""",
                    (reservation_id,),
                )
                rows = cursor.fetchall()
        if len(rows) != 1:
            raise ValueError(
                "discovery compute reservation is missing or incomplete"
            )
        reservation = copy.deepcopy(rows[0]["reservation"])
        work = copy.deepcopy(rows[0]["work"])
        if (reservation.get("reservation_sha256") != reservation_sha256
                or reservation.get("mission_id") != mission_id
                or reservation.get("work_kind") != work_kind
                or reservation.get("work_authority_sha256") !=
                    work_authority_sha256
                or reservation.get("ordered_item_ids") != ordered_item_ids
                or work.get("reservation_sha256") != reservation_sha256
                or work.get("ordered_item_ids") != ordered_item_ids):
            raise ValueError(
                "discovery compute reservation authority mismatch"
            )
        expected = content_sha256({
            key: value for key, value in reservation.items()
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
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_compute_outcomes
                       (mission_id,request_id,outcome,created_at)
                       VALUES(%s,%s,%s,%s)
                       ON CONFLICT(mission_id,request_id,outcome) DO NOTHING""",
                    (mission_id, request_id, outcome, utc_now()),
                )

    @staticmethod
    def _run_promotion_failpoint(callback, name: str) -> None:
        if callback is not None:
            callback(name)

    def _derive_discovery_promotion_admission_from_cursor(
        self, cursor: Any, *, evidence_set_id: str,
        task9_gate: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve every promotion input from registered rows in this txn."""

        from .campaign_decision import authorize_promotion_child
        from .seed_probe import (
            _build_first_letters_discovery_artifact_and_receipt_from_evidence_set,
            load_first_letters_normal_growth_lock,
        )

        registered = self._read_first_letters_discovery_evidence_set_from_cursor(
            cursor, evidence_set_id
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
        cursor.execute(
            """SELECT * FROM segment_tasks
                WHERE task_id=%s AND mission_id=%s FOR UPDATE""",
            (artifact["parent_task_id"], artifact["mission_id"]),
        )
        parent = cursor.fetchone()
        cursor.execute(
            """SELECT * FROM segment_attempts
                WHERE attempt_id=%s AND task_id=%s FOR UPDATE""",
            (artifact["parent_attempt_id"], artifact["parent_task_id"]),
        )
        attempt = cursor.fetchone()
        cursor.execute(
            """SELECT * FROM segment_source_snapshots
                WHERE source_snapshot_id=%s FOR SHARE""",
            (artifact["source_snapshot_id"],),
        )
        source_row = cursor.fetchone()
        if parent is None or attempt is None or source_row is None:
            raise ValueError("registered discovery promotion authority is missing")
        payload = copy.deepcopy(parent["payload"])
        source = self._snapshot(source_row)
        promotion_scope = (
            source.get("first_letters_discovery_authority") or {}
        ).get("promotion_authority") or {}
        budget_sha = payload.get("campaign_budget_admission_sha256")
        cursor.execute(
            """SELECT admission FROM segment_campaign_budget_admissions
                WHERE mission_id=%s AND sample_id=%s
                  AND admission_sha256=%s FOR SHARE""",
            (artifact["mission_id"], artifact["sample_id"], budget_sha),
        )
        budget_rows = cursor.fetchall()
        if len(budget_rows) != 1:
            raise ValueError("registered discovery budget authority is missing")
        budget = copy.deepcopy(budget_rows[0]["admission"])
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
        """Create authority + fresh child + terminal parent atomically."""

        from .campaign_decision import (  # noqa: PLC0415
            _validated_task9_discovery_gate,
            build_promotion_child_task,
            validate_promotion_child_task,
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT r.mission_id
                         FROM segment_first_letters_discovery_evidence_sets e
                         JOIN segment_first_letters_discovery_evidence_runs r
                           ON r.run_id=e.run_id
                        WHERE e.evidence_set_id=%s AND r.state='COMPLETED'""",
                    (evidence_set_id,),
                )
                evidence_row = cursor.fetchone()
                if evidence_row is None:
                    raise KeyError(evidence_set_id)
                mission_id = evidence_row["mission_id"]
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_id}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"promotion-mission:{mission_id}",),
                )
                admission = self._derive_discovery_promotion_admission_from_cursor(
                    cursor, evidence_set_id=evidence_set_id,
                    task9_gate=task9_gate,
                )
                resolver = self._task9_discovery_gate_resolver
                authoritative_gate = (
                    resolver(mission_id) if resolver is not None else None
                )
                if authoritative_gate != task9_gate:
                    raise ValueError(
                        "TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED"
                    )
                _validated_task9_discovery_gate(
                    authoritative_gate, mission_id=mission_id,
                )
                child = build_promotion_child_task(admission)
                request_core = {
                    "mission_id": mission_id, "request_id": request_id,
                    "evidence_set_id": evidence_set_id,
                    "admission": admission, "task9_gate": task9_gate,
                }
                request_sha = content_sha256(request_core)
                cursor.execute(
                    """SELECT request_sha256
                         FROM segment_first_letters_discovery_promotions
                        WHERE mission_id=%s AND request_id=%s FOR UPDATE""",
                    (mission_id, request_id),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["request_sha256"] != request_sha:
                        raise ValueError("PROMOTION_AUTHORITY_CONFLICT")
                else:
                    cursor.execute(
                        """SELECT * FROM segment_tasks
                            WHERE task_id=%s AND mission_id=%s FOR UPDATE""",
                        (admission["parent_task_id"], mission_id),
                    )
                    parent = cursor.fetchone()
                    if (parent is None or parent["state"] not in ACTIVE_STATES
                            or parent["active_attempt_id"] !=
                                admission["parent_attempt_id"]):
                        raise ValueError(
                            "promotion parent task/attempt is not active authority"
                        )
                    cursor.execute(
                        """SELECT * FROM segment_attempts
                            WHERE attempt_id=%s AND task_id=%s FOR UPDATE""",
                        (admission["parent_attempt_id"],
                         admission["parent_task_id"]),
                    )
                    if cursor.fetchone() is None:
                        raise ValueError("promotion parent attempt is missing")
                    cursor.execute(
                        """SELECT * FROM segment_source_snapshots
                            WHERE source_snapshot_id=%s FOR SHARE""",
                        (admission["source_snapshot_id"],),
                    )
                    source_row = cursor.fetchone()
                    if source_row is None:
                        raise ValueError(
                            "promotion authoritative source is missing"
                        )
                    validate_promotion_child_task(
                        child, admission=admission,
                        registered_budget_admission=admission[
                            "registered_budget_admission"],
                        authoritative_source_snapshot=self._snapshot(source_row),
                    )
                    promotion_id = stable_id(
                        "first-letters-discovery-promotion", {
                            "mission_id": mission_id,
                            "scientific_opportunity_id": admission[
                                "scientific_opportunity_id"],
                            "admission_sha256": admission["admission_sha256"],
                        },
                    )
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
                        promotion_failpoint,
                        "promotion.before_authority_insert",
                    )
                    now = utc_now()
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_promotions
                           (promotion_id,mission_id,request_id,
                            scientific_opportunity_id,parent_task_id,
                            child_task_id,admission_sha256,authority,
                            authority_sha256,request_sha256,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)""",
                        (promotion_id, mission_id, request_id,
                         admission["scientific_opportunity_id"],
                         admission["parent_task_id"], admission["child_task_id"],
                         admission["admission_sha256"],
                         json.dumps(authority, sort_keys=True,
                                    separators=(",", ":")),
                         authority["authority_sha256"], request_sha, now),
                    )
                    self._run_promotion_failpoint(
                        promotion_failpoint,
                        "promotion.after_authority_insert_before_child_insert",
                    )
                    requirements = normalize_resource_requirements(
                        child["resource_requirements"],
                    )
                    payload = {
                        **copy.deepcopy(child),
                        "promotion_id": promotion_id,
                        "promotion_authority_sha256":
                            authority["authority_sha256"],
                        "resource_requirements": requirements,
                    }
                    cursor.execute(
                        """INSERT INTO segment_tasks
                           (task_id,mission_id,source_snapshot_id,cell_id,
                            grid_version,policy_version,bounds_xyz,center_xyz,
                            priority,parameter_envelope,catalog_snapshot_sha256,
                            payload,state,gpu_required,minimum_vram_gb,
                            seed_probe_required,created_by,created_at,updated_at)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,
                                  %s::jsonb,%s,%s::jsonb,'PENDING',%s,%s,%s,%s,
                                  %s,%s)""",
                        (child["task_id"], mission_id,
                         child["source_snapshot_id"], child["cell_id"],
                         child["grid_version"], child["policy_version"],
                         json.dumps(child["bounds_xyz"], sort_keys=True,
                                    separators=(",", ":")),
                         json.dumps(child["center_xyz"], sort_keys=True,
                                    separators=(",", ":")),
                         float(child["priority"]),
                         json.dumps(child["parameter_envelope"], sort_keys=True,
                                    separators=(",", ":")),
                         child["catalog_snapshot_sha256"],
                         json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")),
                         requirements["gpu_required"],
                         requirements["minimum_vram_gb"],
                         requirements["seed_probe_required"],
                         str(child.get("created_by") or "unattributed"), now,
                         now),
                    )
                    self.event(
                        cursor, "PROMOTION_CHILD_CREATED", {
                            "promotion_id": promotion_id,
                            "promotion_authority_sha256":
                                authority["authority_sha256"],
                        }, child["task_id"],
                    )
                    self._run_promotion_failpoint(
                        promotion_failpoint,
                        "promotion.after_child_insert_before_parent_terminal",
                    )
                    terminal = {
                        "promotion_id": promotion_id,
                        "promotion_authority_sha256":
                            authority["authority_sha256"],
                        "child_task_id": child["task_id"],
                        "scientific_opportunity_id": admission[
                            "scientific_opportunity_id"],
                    }
                    cursor.execute(
                        """UPDATE segment_attempts
                              SET state='DISCOVERY_PROMOTED',result=%s::jsonb,
                                  updated_at=%s
                            WHERE attempt_id=%s AND task_id=%s""",
                        (json.dumps(terminal, sort_keys=True,
                                    separators=(",", ":")), now,
                         admission["parent_attempt_id"],
                         admission["parent_task_id"]),
                    )
                    updated_attempt = cursor.rowcount
                    cursor.execute(
                        """UPDATE segment_tasks
                              SET state='DISCOVERY_PROMOTED',updated_at=%s
                            WHERE task_id=%s AND active_attempt_id=%s""",
                        (now, admission["parent_task_id"],
                         admission["parent_attempt_id"]),
                    )
                    if updated_attempt != 1 or cursor.rowcount != 1:
                        raise RuntimeError(
                            "promotion parent terminal transition conflicted"
                        )
                    self.event(
                        cursor, "DISCOVERY_PROMOTED", terminal,
                        admission["parent_task_id"],
                        admission["parent_attempt_id"],
                    )
                    self._run_promotion_failpoint(
                        promotion_failpoint,
                        "promotion.after_parent_terminal_before_commit",
                    )
                    self._run_promotion_failpoint(
                        promotion_failpoint, "promotion.before_commit",
                    )
        # Both post-commit boundaries deliberately force the caller to perform
        # the three-fact readback before deciding whether a retry is safe.
        self._run_promotion_failpoint(
            promotion_failpoint, "promotion.commit_outcome_unknown",
        )
        self._run_promotion_failpoint(
            promotion_failpoint, "promotion.after_commit_before_response",
        )
        return self.read_discovery_promotion(mission_id, request_id)

    def read_discovery_promotion(
        self, mission_id: str, request_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT *
                         FROM segment_first_letters_discovery_promotions
                        WHERE mission_id=%s AND request_id=%s""",
                    (mission_id, request_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(request_id)
                cursor.execute(
                    "SELECT state,payload FROM segment_tasks WHERE task_id=%s",
                    (row["child_task_id"],),
                )
                child = cursor.fetchone()
                cursor.execute(
                    """SELECT state,active_attempt_id
                         FROM segment_tasks WHERE task_id=%s""",
                    (row["parent_task_id"],),
                )
                parent = cursor.fetchone()
                cursor.execute(
                    """SELECT state,task_id,result
                         FROM segment_attempts WHERE attempt_id=%s""",
                    ((row["authority"] or {}).get("admission", {}).get(
                        "parent_attempt_id"),),
                )
                parent_attempt = cursor.fetchone()
        if (child is None or parent is None or parent_attempt is None
                or parent["state"] != "DISCOVERY_PROMOTED"
                or parent_attempt["state"] != "DISCOVERY_PROMOTED"
                or parent_attempt["task_id"] != row["parent_task_id"]
                or parent["active_attempt_id"] != (row["authority"] or {}).get(
                    "admission", {}).get("parent_attempt_id")):
            raise ValueError("CONTROL_INCOMPLETE_PROMOTION_READBACK")
        child_value = copy.deepcopy(child["payload"])
        child_value["state"] = child["state"]
        return {
            "authority": copy.deepcopy(row["authority"]),
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
        if (not isinstance(attempt_number, int)
                or isinstance(attempt_number, bool) or attempt_number < 1):
            raise ValueError("promotion attempt number is invalid")
        binding: dict[str, Any] | None = None
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT mission_id
                         FROM segment_first_letters_discovery_promotions
                        WHERE promotion_id=%s""",
                    (promotion_id,),
                )
                mission_row = cursor.fetchone()
                if mission_row is None:
                    raise KeyError(promotion_id)
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{mission_row['mission_id']}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"promotion-mission:{mission_row['mission_id']}",),
                )
                cursor.execute(
                    """SELECT *
                         FROM segment_first_letters_discovery_promotions
                        WHERE promotion_id=%s FOR UPDATE""",
                    (promotion_id,),
                )
                promotion = cursor.fetchone()
                if promotion is None:  # pragma: no cover - lock serialization
                    raise KeyError(promotion_id)
                authority = copy.deepcopy(promotion["authority"])
                normal = authority["admission"]["normal_growth_lock"]
                cursor.execute(
                    """SELECT binding
                         FROM segment_first_letters_discovery_promotion_attempt_bindings
                        WHERE promotion_id=%s AND attempt_number=%s""",
                    (promotion_id, attempt_number),
                )
                existing = cursor.fetchone()
                cursor.execute(
                    """SELECT attempt_number,attempt_id,binding
                         FROM segment_first_letters_discovery_promotion_attempt_bindings
                        WHERE promotion_id=%s
                        ORDER BY attempt_number DESC LIMIT 1""",
                    (promotion_id,),
                )
                prior = cursor.fetchone()
                if attempt_number == 1:
                    if (predecessor_attempt_id is not None
                            or retry_reason is not None):
                        raise ValueError(
                            "first promotion attempt cannot be a retry"
                        )
                else:
                    allowed_retry_reasons = {
                        "WORKER_FAILURE", "LEASE_EXHAUSTION",
                        "PUBLICATION_FAILURE", "SOURCE_FAILURE",
                    }
                    if (prior is None
                            or prior["attempt_number"] != attempt_number - 1
                            or predecessor_attempt_id != prior["attempt_id"]
                            or retry_reason not in allowed_retry_reasons
                            or attempt_number > 1 + normal["retry_budget"]):
                        raise ValueError("promotion retry is not authorized")
                cursor.execute(
                    """SELECT task_id,attempt_number
                         FROM segment_attempts WHERE attempt_id=%s FOR SHARE""",
                    (attempt_id,),
                )
                attempt = cursor.fetchone()
                if (attempt is None
                        or attempt["task_id"] != promotion["child_task_id"]
                        or attempt["attempt_number"] != attempt_number):
                    raise ValueError(
                        "promotion attempt is not bound to the fresh child"
                    )
                core = {
                    "schema":
                        "campaignx.first_letters_discovery_promotion_attempt_binding.v1",
                    "promotion_id": promotion_id,
                    "promotion_authority_sha256":
                        promotion["authority_sha256"],
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
                binding = {
                    **core, "binding_sha256": content_sha256(core),
                }
                if existing is not None:
                    prior_value = copy.deepcopy(existing["binding"])
                    if prior_value != binding:
                        raise ValueError(
                            "PROMOTION_ATTEMPT_BINDING_CONFLICT"
                        )
                    binding = prior_value
                else:
                    cursor.execute(
                        """INSERT INTO
                           segment_first_letters_discovery_promotion_attempt_bindings
                           (promotion_id,attempt_number,attempt_id,binding,
                            binding_sha256,created_at)
                           VALUES(%s,%s,%s,%s::jsonb,%s,%s)""",
                        (promotion_id, attempt_number, attempt_id,
                         json.dumps(binding, sort_keys=True,
                                    separators=(",", ":")),
                         binding["binding_sha256"], utc_now()),
                    )
        if binding is None:  # pragma: no cover - defensive transaction guard
            raise RuntimeError("CONTROL_INCOMPLETE_PROMOTION_READBACK")
        return binding

    def register_snapshot(self, payload: dict[str, Any]) -> str:
        """Record where a scroll's CT lives, and its m7 prediction when it has one.

        sample_id and ct_uri are the only two this ever required; a scroll with
        a published m7 gets the rest along with it, from bootstrap_sources or
        the control cohort, and a scroll with none -- a community mesh and
        surface volume P4 and P5 read directly, nothing upstream of them -- gets
        a source with m7_uri absent rather than no way to be named at all.
        """
        if not payload.get("sample_id") or not payload.get("ct_uri"):
            raise ValueError("register_snapshot requires sample_id and ct_uri")
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
        shape_xyz = payload.get("shape_xyz")
        voxel_size_um = payload.get("voxel_size_um")
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
                        payload.get("m7_uri"),
                        payload.get("m7_sha256"),
                        json.dumps(shape_xyz) if shape_xyz is not None else None,
                        float(voxel_size_um) if voxel_size_um is not None else None,
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{admission['mission_id']}",),
                )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    ("campaign-budget-admission:"
                     f"{admission['mission_id']}:{admission['sample_id']}:"
                     f"{admission['receipt_sha256']}",),
                )
                cursor.execute(
                    """SELECT admission FROM segment_campaign_budget_admissions
                        WHERE mission_id=%s AND sample_id=%s
                          AND receipt_sha256=%s FOR UPDATE""",
                    (admission["mission_id"], admission["sample_id"],
                     admission["receipt_sha256"]),
                )
                registered = cursor.fetchone()
                cursor.execute(
                    """SELECT payload FROM segment_tasks
                        WHERE mission_id=%s FOR UPDATE""",
                    (admission["mission_id"],),
                )
                mission_tasks = cursor.fetchall()
                cursor.execute(
                    """SELECT admission
                         FROM segment_campaign_budget_admissions
                        WHERE mission_id=%s
                        ORDER BY sample_id,receipt_sha256
                        FOR SHARE""",
                    (admission["mission_id"],),
                )
                registered_admissions = [
                    dict(row["admission"]) for row in cursor.fetchall()
                ]
                cursor.execute(
                    """SELECT receipt FROM segment_campaign_decisions
                        WHERE mission_id=%s ORDER BY created_at,receipt_sha256""",
                    (admission["mission_id"],),
                )
                registered_decisions = [
                    dict(row["receipt"]) for row in cursor.fetchall()
                ]
                cursor.execute(
                    """SELECT "authorization"
                         FROM segment_campaign_resume_authorizations
                        WHERE mission_id=%s
                        ORDER BY created_at,authorization_sha256""",
                    (admission["mission_id"],),
                )
                registered_authorizations = [
                    dict(row["authorization"]) for row in cursor.fetchall()
                ]
                if registered is not None and (
                    dict(registered["admission"]) != admission
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
                            cursor.execute(
                                """SELECT "authorization"
                                     FROM segment_campaign_resume_principal_attestations
                                    WHERE authorization_sha256=%s
                                      AND mission_id=%s FOR SHARE""",
                                (
                                    resume_authorization.get(
                                        "authorization_sha256"),
                                    admission["mission_id"],
                                ),
                            )
                            attestation = cursor.fetchone()
                            if (attestation is None
                                    or dict(attestation["authorization"]) !=
                                        resume_authorization):
                                raise ValueError(
                                    "campaign resume authorization has no trusted "
                                    "principal attestation")
                            authoritative_attempts, authoritative_admissions = (
                                self._campaign_decision_inputs(
                                    cursor,
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
                                    resume_authorization[
                                        "authorization_sha256"]},
                            )
                            cursor.execute(
                                """INSERT INTO segment_campaign_resume_authorizations
                                   (authorization_sha256,mission_id,sample_id,
                                    prior_policy_version,new_policy_version,
                                    new_admission_sha256,"authorization")
                                   VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)
                                   ON CONFLICT DO NOTHING""",
                                (
                                    validated_resume["authorization_sha256"],
                                    validated_resume["mission_id"],
                                    validated_resume["new_sample_id"],
                                    validated_resume["prior_policy_version"],
                                    validated_resume["new_policy_version"],
                                    admission["admission_sha256"],
                                    json.dumps(
                                        validated_resume, sort_keys=True,
                                        separators=(",", ":")),
                                ),
                            )
                            cursor.execute(
                                """SELECT "authorization"
                                     FROM segment_campaign_resume_authorizations
                                    WHERE mission_id=%s AND sample_id=%s
                                      AND new_admission_sha256=%s""",
                                (
                                    admission["mission_id"], admission["sample_id"],
                                    admission["admission_sha256"],
                                ),
                            )
                            persisted_resume = cursor.fetchone()
                            if (persisted_resume is None
                                    or dict(persisted_resume["authorization"])
                                    != validated_resume):
                                raise ValueError(
                                    "campaign resume authorization registry already "
                                    "contains different evidence")
                elif resume_authorization is not None:
                    cursor.execute(
                        """SELECT "authorization"
                             FROM segment_campaign_resume_authorizations
                            WHERE mission_id=%s AND sample_id=%s
                              AND new_admission_sha256=%s FOR SHARE""",
                        (
                            admission["mission_id"], admission["sample_id"],
                            admission["admission_sha256"],
                        ),
                    )
                    persisted_resume = cursor.fetchone()
                    if (registered is None or persisted_resume is None
                            or dict(persisted_resume["authorization"])
                            != resume_authorization):
                        raise ValueError(
                            "campaign resume authorization is only valid for "
                            "an exact paused mission/sample replacement or "
                            "its idempotent replay")
                if mission_tasks and (
                    not registered_admissions
                    or any(not any(campaign_budget_task_matches_admission(
                        dict(row["payload"]), authority)
                        for authority in registered_admissions)
                        for row in mission_tasks)
                ):
                    raise ValueError(
                        "pre-existing mission tasks prevent controlled admission")
                cursor.execute(
                    """INSERT INTO segment_campaign_budget_admissions
                       (mission_id,sample_id,receipt_sha256,admission,
                        admission_sha256) VALUES(%s,%s,%s,%s::jsonb,%s)
                       ON CONFLICT(mission_id,sample_id,receipt_sha256)
                       DO NOTHING""",
                    (admission["mission_id"], admission["sample_id"],
                     admission["receipt_sha256"],
                     json.dumps(admission, sort_keys=True, separators=(",", ":")),
                     digest),
                )
                cursor.execute(
                    """SELECT admission FROM segment_campaign_budget_admissions
                        WHERE mission_id=%s AND sample_id=%s AND receipt_sha256=%s""",
                    (admission["mission_id"], admission["sample_id"],
                     admission["receipt_sha256"]),
                )
                row = cursor.fetchone()
                if row is None or dict(row["admission"]) != admission:
                    raise ValueError(
                        "campaign budget admission registry already contains different authority")
        return admission

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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"campaign-budget-mission:{authorization['mission_id']}",),
                )
                cursor.execute(
                    """INSERT INTO segment_campaign_resume_principal_attestations
                       (authorization_sha256,mission_id,principal,"authorization")
                       VALUES(%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING""",
                    (
                        digest, authorization["mission_id"], principal,
                        json.dumps(
                            authorization, sort_keys=True, separators=(",", ":")),
                    ),
                )
                cursor.execute(
                    """SELECT "authorization"
                         FROM segment_campaign_resume_principal_attestations
                        WHERE authorization_sha256=%s""",
                    (digest,),
                )
                row = cursor.fetchone()
                if row is None or dict(row["authorization"]) != authorization:
                    raise ValueError(
                        "campaign resume principal attestation already differs")
        return dict(authorization)

    def _campaign_decision_inputs(
        self, cursor: Any, *, mission_id: str, policy_version: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Read the rows that govern a decision inside the caller's lock."""
        cursor.execute(
            """SELECT t.task_id,t.mission_id,t.policy_version,
                      t.state AS task_state,t.payload,
                      t.updated_at AS task_updated_at,
                      a.attempt_id,a.attempt_number,
                      a.state AS attempt_state,a.result,
                      a.updated_at AS attempt_updated_at
                 FROM segment_tasks t
                 LEFT JOIN segment_attempts a ON a.task_id=t.task_id
                WHERE t.mission_id=%s AND t.policy_version=%s""",
            (mission_id, policy_version),
        )
        attempts = []
        for row in cursor.fetchall():
            payload = dict(row["payload"] or {})
            terminal_at = row["attempt_updated_at"] or row["task_updated_at"]
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
                "result": (
                    dict(row["result"]) if isinstance(row["result"], dict)
                    else row["result"]
                ),
                "terminal_at_utc": (
                    terminal_at.isoformat()
                    if hasattr(terminal_at, "isoformat")
                    else str(terminal_at or "")
                ),
            })
        cursor.execute(
            """SELECT admission,created_at
                 FROM segment_campaign_budget_admissions
                WHERE mission_id=%s
                ORDER BY created_at,sample_id,receipt_sha256""",
            (mission_id,),
        )
        admissions = []
        for row in cursor.fetchall():
            admission = dict(row["admission"])
            registered_at = row["created_at"]
            admission["registered_at_utc"] = (
                registered_at.isoformat()
                if hasattr(registered_at, "isoformat")
                else str(registered_at or "")
            )
            admissions.append(admission)
        return attempts, admissions

    def _refresh_campaign_decisions(
        self, cursor: Any, *, mission_id: str, policy_version: str,
    ) -> list[dict[str, Any]]:
        """Derive and create-once campaign decisions under the mission lock."""
        from .campaign_decision import (  # noqa: PLC0415
            derive_campaign_decision_receipts,
            load_campaign_policy_profile,
        )

        attempts, admissions = self._campaign_decision_inputs(
            cursor,
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
            cursor.execute(
                """INSERT INTO segment_campaign_decisions
                   (receipt_sha256,mission_id,policy_version,evaluation_kind,
                    evaluation_index,decision,receipt)
                   VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT DO NOTHING""",
                (
                    receipt["receipt_sha256"], mission_id, policy_version,
                    receipt["evaluation_kind"], receipt["evaluation_index"],
                    receipt["decision"], json.dumps(
                        receipt, sort_keys=True, separators=(",", ":")),
                ),
            )
            cursor.execute(
                """SELECT receipt_sha256
                     FROM segment_campaign_decisions
                    WHERE mission_id=%s AND policy_version=%s
                      AND evaluation_kind=%s AND evaluation_index=%s""",
                (
                    mission_id, policy_version, receipt["evaluation_kind"],
                    receipt["evaluation_index"],
                ),
            )
            persisted = cursor.fetchone()
            if (persisted is None
                    or persisted["receipt_sha256"] != receipt["receipt_sha256"]):
                raise ValueError(
                    "campaign decision evaluation already has different evidence")
        cursor.execute(
            """SELECT receipt
                 FROM segment_campaign_decisions
                WHERE mission_id=%s AND policy_version=%s
                ORDER BY created_at,receipt_sha256""",
            (mission_id, policy_version),
        )
        return [dict(row["receipt"]) for row in cursor.fetchall()]

    def campaign_decisions(
        self, *, mission_id: str, policy_version: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = """SELECT receipt
                             FROM segment_campaign_decisions
                            WHERE mission_id=%s"""
                parameters: list[Any] = [mission_id]
                if policy_version is not None:
                    query += " AND policy_version=%s"
                    parameters.append(policy_version)
                query += " ORDER BY created_at,receipt_sha256"
                cursor.execute(query, parameters)
                return [dict(row["receipt"]) for row in cursor.fetchall()]

    def campaign_active_decision(
        self, *, mission_id: str,
    ) -> dict[str, Any] | None:
        from .campaign_decision import (  # noqa: PLC0415
            derive_campaign_active_decision,
            load_campaign_policy_profile,
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT admission,created_at
                         FROM segment_campaign_budget_admissions
                        WHERE mission_id=%s
                        ORDER BY created_at,sample_id,receipt_sha256""",
                    (mission_id,),
                )
                admissions = []
                for row in cursor.fetchall():
                    admission = dict(row["admission"])
                    created = row["created_at"]
                    admission["registered_at_utc"] = (
                        created.isoformat() if hasattr(created, "isoformat")
                        else str(created or ""))
                    admissions.append(admission)
                cursor.execute(
                    """SELECT receipt FROM segment_campaign_decisions
                        WHERE mission_id=%s ORDER BY created_at,receipt_sha256""",
                    (mission_id,),
                )
                decisions = [dict(row["receipt"]) for row in cursor.fetchall()]
                cursor.execute(
                    """SELECT "authorization"
                         FROM segment_campaign_resume_authorizations
                        WHERE mission_id=%s
                        ORDER BY created_at,authorization_sha256""",
                    (mission_id,),
                )
                authorizations = [
                    dict(row["authorization"]) for row in cursor.fetchall()
                ]
                cursor.execute(
                    """SELECT t.task_id,t.mission_id,t.policy_version,
                              t.state AS task_state,t.payload,
                              t.updated_at AS task_updated_at,
                              a.attempt_id,a.attempt_number,
                              a.state AS attempt_state,a.result,
                              a.updated_at AS attempt_updated_at
                         FROM segment_tasks t
                         LEFT JOIN segment_attempts a ON a.task_id=t.task_id
                        WHERE t.mission_id=%s""",
                    (mission_id,),
                )
                attempts = []
                for row in cursor.fetchall():
                    payload = dict(row["payload"] or {})
                    terminal_at = (
                        row["attempt_updated_at"] or row["task_updated_at"])
                    attempts.append({
                        "task_id": row["task_id"],
                        "attempt_id": row["attempt_id"],
                        "attempt_number": row["attempt_number"],
                        "mission_id": row["mission_id"],
                        "sample_id": payload.get("sample_id"),
                        "policy_version": row["policy_version"],
                        "campaign_budget": payload.get("campaign_budget"),
                        "state": row["attempt_state"] or row["task_state"],
                        "result": (
                            dict(row["result"])
                            if isinstance(row["result"], dict) else row["result"]),
                        "terminal_at_utc": (
                            terminal_at.isoformat()
                            if hasattr(terminal_at, "isoformat")
                            else str(terminal_at or "")),
                    })
        return derive_campaign_active_decision(
            attempts, admissions, decisions, authorizations,
            load_campaign_policy_profile(), mission_id=mission_id,
        )

    @staticmethod
    def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
        value = dict(row["payload"])
        value.update({"source_snapshot_id": row["source_snapshot_id"], "shape_xyz": row["shape_xyz"]})
        return value

    def _write_routing_receipt(self, cursor: Any,
                               surface: dict[str, Any]) -> dict[str, Any] | None:
        """Classify a surface as it is created, inside the same transaction.

        The SQLite twin of this ran from the day the router landed and this one
        did not exist at all, which meant the size gate held in the tests and
        not in the deployment. Same pure router, same receipt bytes, same
        transaction boundary.
        """
        from . import surface_routing  # noqa: PLC0415

        # An unmeasured surface cannot be routed, and refusing a direct
        # catalogue import would be the router deciding something outside its
        # question -- an import creates no QC job and starts no work. The
        # boundaries that do go through _require_routing_receipt and fail closed.
        if surface.get("area_cm2") is None:
            return None

        receipt = surface_routing.receipt_for_surface(surface)
        cursor.execute(
            """INSERT INTO segment_surface_routing_receipts
               (surface_id,route,measured_area_cm2,minimum_area_cm2,
                policy_version,profile_id,receipt_sha256,receipt)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
            (receipt["surface_id"], receipt["route"],
             receipt["measured_area_cm2"], receipt["minimum_area_cm2"],
             receipt["policy_version"], receipt["profile_id"],
             receipt["receipt_sha256"],
             json.dumps(receipt, sort_keys=True, separators=(",", ":"))),
        )
        return receipt

    @staticmethod
    def _stored_routing_receipt(cursor: Any,
                                surface_id: str) -> dict[str, Any] | None:
        """The stored decision, read on the cursor the caller already holds.

        Reading inside the caller's transaction is the point: a route fetched on
        a separate connection is a fact about some other moment, and a gate is
        only a gate when it and the write it guards see the same database.

        The only implementation. There were two, defined the same week, and the
        public method in front of one of them was unreachable -- Python keeps
        the last definition of a name and discards the first in silence.
        """
        cursor.execute(
            "SELECT receipt FROM segment_surface_routing_receipts "
            "WHERE surface_id=%s",
            (surface_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        # A real cursor hands back a mapping; a scripted one in a test hands
        # back a tuple, and jsonb arrives decoded but a text column does not.
        stored = row.get("receipt") if hasattr(row, "get") else row[0]
        if isinstance(stored, (str, bytes)):
            try:
                stored = json.loads(stored)
            except ValueError:
                return None
        return copy.deepcopy(stored) if isinstance(stored, dict) else None

    def _require_routing_receipt(self, cursor: Any,
                                 surface: dict[str, Any]) -> dict[str, Any]:
        """The exact, valid receipt this surface must have before work starts.

        The PostgreSQL twin of ``FleetStore._require_routing_receipt``, with the
        same contract: resolve, write, read back, and require the persisted
        document to equal the one just decided and to verify; and where a
        receipt already exists, re-decide rather than trust it, because the area
        behind a receipt can move and the receipt cannot.
        """
        from . import surface_routing  # noqa: PLC0415

        surface_id = str(surface["surface_id"])
        policy = surface_routing.load_policy()
        stored = self._stored_routing_receipt(cursor, surface_id)
        if stored is None:
            built = self._write_routing_receipt(cursor, surface)
            if built is None:
                raise RuntimeError(
                    f"{surface_id} has no measured area, so it cannot be "
                    "routed and nothing may start work on it")
            persisted = self._stored_routing_receipt(cursor, surface_id)
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

    @staticmethod
    def _surface_policy_version(cursor: Any, surface_id: str) -> str | None:
        """The policy version a surface was produced under, where there is one."""
        cursor.execute(
            "SELECT payload FROM segment_surfaces WHERE surface_id=%s",
            (surface_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        payload = dict(row["payload"] or {})
        version = payload.get("policy_version")
        if isinstance(version, str) and version.strip():
            return version
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return None
        cursor.execute(
            "SELECT policy_version FROM segment_tasks WHERE task_id=%s",
            (task_id,),
        )
        task = cursor.fetchone()
        return task["policy_version"] if task is not None else None

    def _resolve_expansion_authority(
        self, cursor: Any, *, successor_surface_id: str,
        source: dict[str, Any], asserted: Any = None,
    ) -> dict[str, Any] | None:
        """Re-resolve one expansion against the catalogue, with the row locked.

        The PostgreSQL twin of ``FleetStore._resolve_expansion_authority``. The
        predecessor is taken ``FOR UPDATE`` so its row cannot move between this
        read and the successor's write; its routing receipt needs no lock,
        because a routing receipt cannot be changed at all.

        ``asserted`` is compared, never trusted. A caller may assert; it may not
        decide.
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
        cursor.execute(
            "SELECT surface_id FROM segment_surfaces WHERE surface_id=%s "
            "FOR UPDATE",
            (predecessor_id,),
        )
        if cursor.fetchone() is None:
            raise RuntimeError(
                f"expansion names an unknown surface: {predecessor_id}")
        receipt = self._stored_routing_receipt(cursor, predecessor_id)
        from . import surface_routing  # noqa: PLC0415

        if receipt is None or not surface_routing.verify_receipt(receipt):
            raise RuntimeError(
                f"expansion names {predecessor_id}, which has no valid routing "
                "decision to continue")
        authority = surface_expansion.build_authority(
            expands_surface_id=predecessor_id,
            successor_surface_id=str(successor_surface_id),
            predecessor_route=receipt["route"],
            predecessor_receipt_sha256=receipt["receipt_sha256"],
            prior_policy_version=self._surface_policy_version(
                cursor, predecessor_id),
            new_policy_version=shape["new_policy_version"],
            resume_from=shape["resume_from"],
        )
        if asserted is not None and asserted != authority:
            raise RuntimeError(
                "the asserted expansion authority differs from the one the "
                "catalogue resolves")
        return authority

    def _persist_expansion_authority(self, cursor: Any,
                                     authority: dict[str, Any],
                                     ) -> dict[str, Any]:
        """Write it, read it back, and require the two to be the same document."""
        from . import surface_expansion  # noqa: PLC0415

        cursor.execute(
            """INSERT INTO segment_surface_expansion_authorities
               (successor_surface_id,expands_surface_id,predecessor_route,
                predecessor_receipt_sha256,prior_policy_version,
                new_policy_version,authority_sha256,authority)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
            (authority["successor_surface_id"], authority["expands_surface_id"],
             authority["predecessor_route"],
             authority["predecessor_receipt_sha256"],
             authority["prior_policy_version"], authority["new_policy_version"],
             authority["authority_sha256"],
             json.dumps(authority, sort_keys=True, separators=(",", ":"))),
        )
        persisted = self._stored_expansion_authority(
            cursor, authority["successor_surface_id"])
        if (persisted != authority
                or not surface_expansion.verify_authority(persisted)):
            raise RuntimeError(
                "the expansion authority did not persist exactly as resolved")
        return persisted

    @staticmethod
    def _stored_expansion_authority(cursor: Any, successor_surface_id: str,
                                    ) -> dict[str, Any] | None:
        cursor.execute(
            "SELECT authority FROM segment_surface_expansion_authorities "
            "WHERE successor_surface_id=%s", (successor_surface_id,),
        )
        row = cursor.fetchone()
        return copy.deepcopy(row["authority"]) if row is not None else None

    def expansion_authority(self, successor_surface_id: str,
                            ) -> dict[str, Any] | None:
        """The stored permission this surface was created under, if any."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                return self._stored_expansion_authority(
                    cursor, successor_surface_id)

    def resolve_expansion_authority(self, *, successor_surface_id: str,
                                    source: dict[str, Any],
                                    asserted: Any = None,
                                    ) -> dict[str, Any] | None:
        """The public read-only resolution, on the transactional resolver.

        It delegates rather than restating the rules. A wrapper that re-derived
        the answer here would be a second, weaker set of rules -- the one nobody
        runs the parity test against -- and would answer from a catalogue that
        could move between its read and the caller's write.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                try:
                    return self._resolve_expansion_authority(
                        cursor, successor_surface_id=successor_surface_id,
                        source=source, asserted=asserted)
                finally:
                    connection.rollback()

    def import_surface(self, payload: dict[str, Any]) -> str:
        surface_id = str(payload.get("surface_id") or stable_id("surface-import", payload))
        from .canonical_lineage import (  # noqa: PLC0415
            refuse_asserted_lineage, require_canonical_lineage,
        )

        refuse_asserted_lineage(payload)
        require_canonical_lineage(
            boundary="DIRECT_SURFACE_IMPORT",
            controlled_mission=payload.get("controlled_first_letters") is True,
            authoritative_lineage=payload.get("authoritative_lineage"),
            allow_unvalidated=payload.get("allow_unvalidated"),
        )
        value = {**payload, "surface_id": surface_id}
        with self.connect() as connection:
            with connection.cursor() as cursor:
                # Serialize importers of one surface id.
                #
                # `ON CONFLICT(surface_id)` names one arbiter, and the table
                # also carries UNIQUE(source_snapshot_id, artifact_sha256). Two
                # concurrent imports of the same surface therefore race: the
                # loser's speculative insert can trip the second index, which no
                # conflict clause covers, and it raises where the SQLite path --
                # serialized by BEGIN IMMEDIATE -- returns quietly. A recovery
                # bootstrap re-importing in parallel is exactly that shape.
                #
                # Not a bare `ON CONFLICT DO NOTHING`: that would also swallow a
                # *different* surface id colliding on the artifact digest, which
                # is a real conflict and has to keep raising.
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                    (f"import_surface:{surface_id}",),
                )
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
                # Resolved on every arrival, even a replay: a payload asking a
                # surface to expand itself is refused rather than quietly
                # ignored because the surface already exists. Persisted only
                # once, like the routing decision beside it.
                authority = self._resolve_expansion_authority(
                    cursor, successor_surface_id=surface_id, source=payload,
                    asserted=payload.get("expansion_authority"))
                # After the expansion authority, which is the order the
                # SQLite twin uses and the order the errors depend on: an
                # expansion of itself has to be refused as that, and a
                # difference report reaching the caller first answers a
                # question nobody asked.
                #
                # ON CONFLICT DO NOTHING answers a replay of *different* bytes
                # exactly as it answers a replay of identical ones: the id comes
                # back and the caller's area, digest and URI are dropped without
                # a word. A bootstrap replaying the wrong surface under a known
                # id then believes it stored what it sent.
                #
                # The SQLite twin has compared identity against what is actually
                # there since it learned that; this one never did, and this is
                # the control plane the fleet runs on.
                cursor.execute(
                    """SELECT source_snapshot_id,sample_id,artifact_sha256,
                              artifact_uri,area_cm2
                         FROM segment_surfaces WHERE surface_id=%s""",
                    (surface_id,),
                )
                stored = cursor.fetchone()
                # By name: `connect` builds this cursor with psycopg2's
                # RealDictCursor, so a row is a mapping here exactly as it is a
                # Row in the SQLite twin.
                differing = [] if stored is None else [
                    name for name, incoming in (
                        ("source_snapshot_id", payload["source_snapshot_id"]),
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
                # The insert is ON CONFLICT DO NOTHING, so re-importing the same
                # surface must not attempt a second receipt: the decision is
                # made once, at the moment the surface first exists.
                if self._stored_routing_receipt(cursor, surface_id) is None:
                    self._write_routing_receipt(
                        cursor, {**payload, "surface_id": surface_id})
                if (authority is not None
                        and self._stored_expansion_authority(
                            cursor, surface_id) is None):
                    self._persist_expansion_authority(cursor, authority)
        return surface_id

    @staticmethod
    def _require_surface_payload_lineage(
        payload: dict[str, Any], *, boundary: str,
    ) -> None:
        from .canonical_lineage import require_canonical_lineage  # noqa: PLC0415
        require_canonical_lineage(
            boundary=boundary,
            controlled_mission=payload.get("controlled_first_letters") is True,
            authoritative_lineage=payload.get("authoritative_lineage"),
            allow_unvalidated=payload.get("allow_unvalidated"),
        )

    def resolve_canonical_surface_lineage(
        self, *, surface_id: str, mission_id: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segment_surfaces WHERE surface_id=%s",
                    (surface_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return {}
        payload = dict(row["payload"] or {})
        retained = payload.get("authoritative_lineage")
        if isinstance(retained, dict):
            return {**copy.deepcopy(retained), "surface_id": surface_id,
                    "mission_id": mission_id}
        external_admission = payload.get("canonical_external_admission")
        return {
            "schema": "campaignx.authoritative_surface_lineage.v1",
            "mission_id": mission_id, "surface_id": surface_id,
            "namespace": payload.get("namespace") or "CANONICAL_SURFACE",
            "artifact_identity": payload.get("artifact_set_id") or
                f"surface:{surface_id}",
            "artifact_sha256": row["artifact_sha256"],
            "artifact_uri": row["artifact_uri"],
            "source_snapshot_id": row["source_snapshot_id"],
            "source_binding_sha256": payload.get("source_binding_sha256") or
                content_sha256({"source_snapshot_id": row["source_snapshot_id"],
                                "sample_id": row["sample_id"]}),
            "promotion_lineage_sha256": payload.get(
                "promotion_authority_sha256"),
            "promotion_lineage_kind": payload.get("promotion_lineage_kind"),
            "route_sha256": payload.get("route_sha256"),
            "surface_state": row["state"],
            "canonical": payload.get("namespace") !=
                "NONCANONICAL_DISCOVERY",
            "external": payload.get("owner") == "imported",
            "external_admission_sha256": (
                content_sha256(external_admission)
                if isinstance(external_admission, dict) else None
            ),
            "ambiguous": False, "hash_conflict": False,
        }

    def enqueue_imported_surface_qc(
        self,
        payload: dict[str, Any],
        *,
        profile_id: str,
        job_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically reconcile an unvalidated import and enqueue one QC job."""

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
                # And the size gate, which geometry cannot see: PHerc0268 was
                # GEOMETRY_CERTIFIED at two square millimetres and reached the
                # ink screen. Required, not consulted, and asked positively --
                # exactly a verified STANDARD route admits a claimable job, so a
                # forged receipt fails the same way a missing one does.
                from . import surface_routing  # noqa: PLC0415

                routing_receipt = self._require_routing_receipt(
                    cursor, {**value, "surface_id": surface_id,
                             "geometry_qc_state": geometry_state})
                if not surface_routing.enters_standard_qc(routing_receipt):
                    job_state = QC_SMALL_SURFACE_DIAGNOSTIC
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
        tasks = list(tasks)
        inserted = 0
        seen = 0
        with self.connect() as connection:
            with connection.cursor() as cursor:
                from .campaign_decision import (  # noqa: PLC0415
                    validate_campaign_budget_task_batch,
                )
                budgeted = next((
                    task for task in tasks
                    if isinstance(task.get("campaign_budget"), dict)
                ), None)
                existing_budget_payloads: list[dict[str, Any]] = []
                missions = sorted({str(task.get("mission_id") or "unfiled")
                                   for task in tasks})
                controlled_mission_registered = False
                if missions:
                    # Serialize the first controlled writer as well as later
                    # writers.  A receipt-scoped lock cannot stop two distinct
                    # receipts racing while the mission has no rows yet.
                    for mission in missions:
                        cursor.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (f"campaign-budget-mission:{mission}",),
                        )
                    cursor.execute(
                        """SELECT payload FROM segment_tasks
                            WHERE mission_id=ANY(%s)
                              AND payload ? 'campaign_budget'
                            FOR UPDATE""",
                        (missions,),
                    )
                    existing_budget_payloads = [
                        dict(row["payload"]) for row in cursor.fetchall()
                    ]
                    cursor.execute(
                        """SELECT 1 AS admission_exists
                            FROM segment_campaign_budget_admissions
                            WHERE mission_id=ANY(%s) LIMIT 1""",
                        (missions,),
                    )
                    controlled_mission_registered = cursor.fetchone() is not None
                if budgeted is None and (
                    existing_budget_payloads or controlled_mission_registered
                ):
                    raise ValueError(
                        "campaign budget envelope cannot be omitted after a "
                        "controlled mission has been admitted")
                registered_admission = authoritative_snapshot = None
                if budgeted is not None:
                    envelope = budgeted["campaign_budget"]
                    cursor.execute(
                        """SELECT admission FROM segment_campaign_budget_admissions
                            WHERE mission_id=%s AND sample_id=%s
                              AND receipt_sha256=%s FOR SHARE""",
                        (envelope.get("mission_id"), envelope.get("sample_id"),
                         envelope.get("receipt_sha256")),
                    )
                    registry = cursor.fetchone()
                    if registry is None:
                        raise ValueError(
                            "campaign budget admission is not registered")
                    registered_admission = dict(registry["admission"])
                    source_id = (registered_admission.get("execution_bindings") or {}).get(
                        "source_snapshot_id")
                    cursor.execute(
                        """SELECT * FROM segment_source_snapshots
                            WHERE source_snapshot_id=%s FOR SHARE""", (source_id,))
                    source = cursor.fetchone()
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
                        cursor,
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
                    # A resume task carries the authority it was created under,
                    # resolved against the same locked catalogue finalization
                    # will re-resolve it against. The successor surface does not
                    # exist yet, so the stamp names the task; finalization
                    # rebuilds it for the real surface id and compares every
                    # field that does not depend on it.
                    stamped_expansion = self._resolve_expansion_authority(
                        cursor, successor_surface_id=f"task:{task_id}",
                        source=task)
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
                    """UPDATE segment_attempts
                          SET state='LEASE_EXPIRED',result=%s::jsonb,
                              updated_at=now()
                        WHERE attempt_id=%s""",
                    (
                        json.dumps({
                            "status": "LEASE_EXPIRED",
                            "failure_class": "LEASE_EXHAUSTION",
                            "ink_used": False,
                        }, sort_keys=True, separators=(",", ":")),
                        row["active_attempt_id"],
                    ),
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
            receipt.get("status") != "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE"
            or receipt.get("failure_class") != "SOURCE_FAILURE"
            or receipt.get("probe_run_id") != probe_run_id
            or receipt.get("ink_used") is not False
        ):
            raise ValueError(
                "probe continuation review receipt is incomplete"
            )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT t.mission_id,t.policy_version,
                              EXISTS(
                                SELECT 1
                                  FROM segment_campaign_budget_admissions a
                                 WHERE a.mission_id=t.mission_id
                                   AND a.admission->'execution_bindings'
                                       ->>'policy_version'=t.policy_version
                              ) AS controlled
                         FROM segment_tasks t WHERE t.task_id=%s""",
                    (task_id,),
                )
                scope = cursor.fetchone()
                if scope is None:
                    raise RuntimeError("attempt no longer owns the task")
                if scope["controlled"]:
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"campaign-budget-mission:{scope['mission_id']}",),
                    )
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
                          SET state='BLOCKED_PROBE_ARTIFACT_UNAVAILABLE',
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
                          SET state='BLOCKED_PROBE_ARTIFACT_UNAVAILABLE',worker_id=NULL,
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
                if scope["controlled"]:
                    self._refresh_campaign_decisions(
                        cursor,
                        mission_id=scope["mission_id"],
                        policy_version=scope["policy_version"],
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
                    "STATE_BLOCKED_PROBE_ARTIFACT_UNAVAILABLE",
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
                        """SELECT t.mission_id,t.policy_version,
                                  EXISTS(
                                    SELECT 1
                                      FROM segment_campaign_budget_admissions a
                                     WHERE a.mission_id=t.mission_id
                                       AND a.admission->'execution_bindings'
                                           ->>'policy_version'=t.policy_version
                                  ) AS controlled
                             FROM segment_tasks t WHERE t.task_id=%s""",
                        (task_id,),
                    )
                    scope = cursor.fetchone()
                    if scope is None:
                        raise RuntimeError(
                            "finalization task, attempt, or artifact set does not exist")
                    if scope["controlled"]:
                        cursor.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (f"campaign-budget-mission:{scope['mission_id']}",),
                        )
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
                    task_payload = dict(context["payload"] or {})
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
                        "mission_id": scope["mission_id"],
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
                        "route_sha256": None, "surface_state": "QC_PENDING",
                        "canonical": task_payload.get("namespace") !=
                            "NONCANONICAL_DISCOVERY",
                        "external": False, "external_admission_sha256": None,
                        "ambiguous": False, "hash_conflict": False,
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
                            **surface, "mission_id": scope["mission_id"],
                            "controlled_first_letters": True,
                            "authoritative_lineage": finalization_lineage,
                            "allow_unvalidated": False,
                        }
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
                    # Re-resolved here rather than read off the task: this is
                    # the transaction that creates the successor, and the only
                    # point at which the predecessor's row is held still. The
                    # stamp from task creation is compared, not trusted.
                    from . import surface_expansion  # noqa: PLC0415

                    expansion_authority = self._resolve_expansion_authority(
                        cursor, successor_surface_id=surface_id,
                        source=task_payload)
                    stamped_expansion = task_payload.get("expansion_authority")
                    if expansion_authority is not None:
                        if stamped_expansion is not None and not (
                            surface_expansion.agrees_with_stamp(
                                expansion_authority, stamped_expansion)
                        ):
                            raise RuntimeError(
                                "the expansion this task was queued under is "
                                "not the one the catalogue resolves now")
                    elif stamped_expansion is not None:
                        raise RuntimeError(
                            "the task carries an expansion authority but "
                            "continues no surface")
                    if not duplicate_of:
                        # Decide the route before the row exists, and refuse
                        # here if it cannot be decided. The persisted receipt
                        # follows the surface insert only because it references
                        # it; both land in this transaction, so no committed
                        # surface row has ever existed without its decision.
                        from . import surface_routing  # noqa: PLC0415

                        if expansion_authority is not None:
                            surface = {
                                **surface, "resumes_surface":
                                    expansion_authority["expands_surface_id"]}
                        routing_decision = surface_routing.receipt_for_surface(
                            {**surface, "surface_id": surface_id,
                             "geometry_qc_state": geometry_state})
                        if not surface_routing.verify_receipt(routing_decision):
                            raise RuntimeError(
                                "surface routing decision does not verify")
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
                        routing_receipt = self._require_routing_receipt(
                            cursor, {**surface, "surface_id": surface_id,
                                     "geometry_qc_state": geometry_state})
                        if routing_receipt != routing_decision:
                            raise RuntimeError(
                                "the persisted routing receipt is not the "
                                "decision this finalization made")
                        if expansion_authority is not None:
                            self._persist_expansion_authority(
                                cursor, expansion_authority)
                        if not fixture_only:
                            qc_id = stable_id(
                                "qc-job",
                                {"surface_id": surface_id, "profile_id": qc_profile_id},
                            )
                            # The size gate beside the geometry gate: exactly a
                            # verified STANDARD route earns a claimable job.
                            qc_state = qc_job_state_for(geometry_state)
                            if not surface_routing.enters_standard_qc(
                                routing_receipt
                            ):
                                qc_state = QC_SMALL_SURFACE_DIAGNOSTIC
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
                                    qc_state,
                                    json.dumps({
                                        "artifact_set_id": artifact_set_id,
                                        "surface_artifact_sha256": surface["artifact_sha256"],
                                        "source_attempt_id": surface.get("attempt_id"),
                                        "created_geometry_certified": (
                                            geometry_state == "GEOMETRY_CERTIFIED"),
                                    }),
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
                    if scope["controlled"]:
                        self._refresh_campaign_decisions(
                            cursor,
                            mission_id=scope["mission_id"],
                            policy_version=scope["policy_version"],
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

    # -- the candidate preflight queue --------------------------------------

    def enqueue_candidate_preflight(self, request: dict[str, Any]) -> dict[str, Any]:
        """Queue one preflight, idempotently on what was asked.

        The PostgreSQL twin, and the one the deployment runs. The digest is the
        identity so that a mutation whose answer could not be read is resolved by
        reading state rather than by enqueuing a second job.
        """
        for field in ("mission_id", "sample_id", "source_snapshot_id"):
            if not str(request.get(field) or "").strip():
                raise ValueError(f"a preflight request needs {field}")
        digest = content_sha256(request)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                # A FAILED job is a record of an attempt, not the answer to the
                # next ask. Excluding it here is what lets a frozen request be
                # measured again after a transient outage.
                cursor.execute(
                    """SELECT preflight_job_id,state FROM segment_preflight_jobs
                        WHERE mission_id=%s AND sample_id=%s AND request_sha256=%s
                          AND state<>'FAILED'
                        FOR UPDATE""",
                    (request["mission_id"], request["sample_id"], digest),
                )
                row = cursor.fetchone()
                if row is not None:
                    return {"preflight_job_id": row["preflight_job_id"],
                            "state": row["state"], "created": False}
                cursor.execute(
                    """SELECT count(*) AS n FROM segment_preflight_jobs
                        WHERE mission_id=%s AND sample_id=%s AND request_sha256=%s""",
                    (request["mission_id"], request["sample_id"], digest),
                )
                job_id = stable_id("preflight-job", {
                    "mission_id": request["mission_id"],
                    "sample_id": request["sample_id"],
                    "request_sha256": digest,
                    "attempt_ordinal": cursor.fetchone()["n"],
                })
                cursor.execute(
                    """INSERT INTO segment_preflight_jobs(preflight_job_id,mission_id,
                       sample_id,source_snapshot_id,state,request,request_sha256)
                       VALUES(%s,%s,%s,%s,'PENDING',%s::jsonb,%s)""",
                    (job_id, request["mission_id"], request["sample_id"],
                     request["source_snapshot_id"],
                     json.dumps(request, sort_keys=True, separators=(",", ":")),
                     digest),
                )
        return {"preflight_job_id": job_id, "state": "PENDING", "created": True}

    def claim_preflight(self, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        token = secrets.token_urlsafe(32)
        deadline = _deadline(lease_seconds)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_preflight_jobs SET state='PENDING',worker_id=NULL,
                       lease_token=NULL,lease_expires_at=NULL,updated_at=now()
                        WHERE state='CLAIMED' AND lease_expires_at IS NOT NULL
                          AND lease_expires_at<=now()"""
                )
                cursor.execute(
                    """SELECT * FROM segment_preflight_jobs WHERE state='PENDING'
                         AND (retry_after IS NULL OR retry_after<=now())
                        ORDER BY created_at,preflight_job_id
                        FOR UPDATE SKIP LOCKED LIMIT 1"""
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                cursor.execute(
                    """UPDATE segment_preflight_jobs SET state='CLAIMED',worker_id=%s,
                       lease_token=%s,lease_expires_at=%s,attempts=attempts+1,
                       updated_at=now() WHERE preflight_job_id=%s""",
                    (worker_id, token, deadline, row["preflight_job_id"]),
                )
        return {
            "preflight_job_id": row["preflight_job_id"],
            "mission_id": row["mission_id"],
            "sample_id": row["sample_id"],
            "source_snapshot_id": row["source_snapshot_id"],
            "request": dict(row["request"] or {}),
            "lease_token": token,
            "lease_expires_at": deadline.isoformat(),
        }

    @staticmethod
    def _preflight_owner(cursor: Any, preflight_job_id: str, lease_token: str) -> Any:
        cursor.execute(
            "SELECT * FROM segment_preflight_jobs WHERE preflight_job_id=%s FOR UPDATE",
            (preflight_job_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("no such preflight job")
        if row["state"] != "CLAIMED" or row["lease_token"] != lease_token:
            raise RuntimeError("this preflight job is not held by that lease")
        return row

    def heartbeat_preflight(self, preflight_job_id: str, lease_token: str,
                            lease_seconds: int) -> dict[str, Any]:
        """Extend a lease, and answer whether it is still this worker's."""
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        deadline = _deadline(lease_seconds)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._preflight_owner(cursor, preflight_job_id, lease_token)
                cursor.execute(
                    """UPDATE segment_preflight_jobs SET lease_expires_at=%s,
                       updated_at=now() WHERE preflight_job_id=%s""",
                    (deadline, preflight_job_id),
                )
        return {"preflight_job_id": preflight_job_id,
                "lease_expires_at": deadline.isoformat()}

    def finalize_preflight(self, preflight_job_id: str, lease_token: str,
                           receipt: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._preflight_owner(cursor, preflight_job_id, lease_token)
                cursor.execute(
                    """UPDATE segment_preflight_jobs SET state='COMPLETED',
                       receipt=%s::jsonb,lease_token=NULL,lease_expires_at=NULL,
                       updated_at=now() WHERE preflight_job_id=%s""",
                    (json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                     preflight_job_id),
                )
        return {"preflight_job_id": preflight_job_id, "state": "COMPLETED"}

    def fail_preflight(self, preflight_job_id: str, lease_token: str,
                       reason_code: str, detail: str | None = None) -> dict[str, Any]:
        """Terminal, with the reason and the sentence behind it."""
        if not str(reason_code or "").strip():
            raise ValueError("a failed preflight needs a reason code")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._preflight_owner(cursor, preflight_job_id, lease_token)
                cursor.execute(
                    """UPDATE segment_preflight_jobs SET state='FAILED',reason_code=%s,
                       detail=%s,lease_token=NULL,lease_expires_at=NULL,
                       updated_at=now() WHERE preflight_job_id=%s""",
                    (reason_code, detail, preflight_job_id),
                )
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
        """The SQLite twin's behaviour, on the store the deployment runs."""
        if retry_delay_seconds < 0:
            raise ValueError("retry delay must be non-negative")
        if maximum_requeues < 0:
            raise ValueError("maximum_requeues must be non-negative")
        detail = _safe_outage_detail(receipt)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                row = self._preflight_owner(cursor, preflight_job_id, lease_token)
                requeues = int(row["requeues"] or 0)
                if requeues >= maximum_requeues:
                    cursor.execute(
                        """UPDATE segment_preflight_jobs SET state='FAILED',
                           reason_code='PREFLIGHT_SOURCE_UNAVAILABLE',detail=%s,
                           receipt=%s::jsonb,worker_id=NULL,lease_token=NULL,
                           lease_expires_at=NULL,updated_at=now()
                            WHERE preflight_job_id=%s""",
                        (detail, json.dumps(receipt, sort_keys=True,
                                            separators=(",", ":")),
                         preflight_job_id),
                    )
                    self.event(
                        cursor,
                        "PREFLIGHT_SOURCE_UNAVAILABLE_EXHAUSTED",
                        {
                            "preflight_job_id": preflight_job_id,
                            "sample_id": row["sample_id"],
                            "requeues": requeues,
                            "detail": detail,
                        },
                    )
                    return {
                        "status": "PREFLIGHT_SOURCE_UNAVAILABLE",
                        "preflight_job_id": preflight_job_id,
                        "state": "FAILED",
                        "reason_code": "PREFLIGHT_SOURCE_UNAVAILABLE",
                        "requeues": requeues,
                    }
                retry_after = _deadline(retry_delay_seconds)
                cursor.execute(
                    """UPDATE segment_preflight_jobs SET state='PENDING',detail=%s,
                       receipt=%s::jsonb,worker_id=NULL,lease_token=NULL,
                       lease_expires_at=NULL,retry_after=%s,requeues=requeues+1,
                       updated_at=now() WHERE preflight_job_id=%s""",
                    (detail, json.dumps(receipt, sort_keys=True,
                                        separators=(",", ":")),
                     retry_after, preflight_job_id),
                )
                self.event(
                    cursor,
                    "PREFLIGHT_REQUEUED_SOURCE_UNAVAILABLE",
                    {
                        "preflight_job_id": preflight_job_id,
                        "sample_id": row["sample_id"],
                        "retry_after": retry_after.isoformat(),
                        "requeues": requeues + 1,
                        "detail": detail,
                    },
                )
                return {
                    "status": "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE",
                    "preflight_job_id": preflight_job_id,
                    "state": "PENDING",
                    "retry_after": retry_after.isoformat(),
                    "requeues": requeues + 1,
                }

    def preflight_job(self, preflight_job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segment_preflight_jobs WHERE preflight_job_id=%s",
                    (preflight_job_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "preflight_job_id": row["preflight_job_id"],
            "mission_id": row["mission_id"],
            "sample_id": row["sample_id"],
            "source_snapshot_id": row["source_snapshot_id"],
            "state": row["state"],
            "request": dict(row["request"] or {}),
            "receipt": dict(row["receipt"]) if row["receipt"] else None,
            "reason_code": row["reason_code"],
            "detail": row["detail"],
            "attempts": int(row["attempts"] or 0),
            "requeues": int(row["requeues"] or 0),
            "retry_after": (row["retry_after"].isoformat()
                            if row["retry_after"] is not None else None),
        }

    def defer_qc_jobs(self, sample_id: str, *, until: str | None,
                      reason: str, by: str) -> dict[str, Any]:
        """The PostgreSQL twin, and the one the deployment runs.

        `claim_qc` already skips a job whose ``retry_after`` is in the future, so
        this holds a sample without reordering anything, inventing a state, or
        moving a row. Only PENDING rows: a worker holding a lease finishes.
        """
        if not str(reason or "").strip():
            raise ValueError("deferring QC needs a reason")
        if not str(until or "").strip():
            raise ValueError("deferring QC needs a time it ends")
        deadline = _instant(str(until))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_qc_jobs SET retry_after=%s,updated_at=now()
                        WHERE state='PENDING' AND surface_id IN (
                          SELECT surface_id FROM segment_surfaces WHERE sample_id=%s)""",
                    (deadline, sample_id))
                record = {"sample_id": sample_id, "deferred": cursor.rowcount,
                          "until": deadline, "reason": str(reason).strip(), "by": by}
                self.event(cursor, "qc.deferred", record)
        return record

    def release_qc_jobs(self, sample_id: str, *, by: str) -> dict[str, Any]:
        """Take a deferred sample back up. The jobs never left the queue."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_qc_jobs SET retry_after=NULL,updated_at=now()
                        WHERE state='PENDING' AND retry_after IS NOT NULL
                          AND surface_id IN (
                            SELECT surface_id FROM segment_surfaces WHERE sample_id=%s)""",
                    (sample_id,))
                record = {"sample_id": sample_id, "released": cursor.rowcount, "by": by}
                self.event(cursor, "qc.released", record)
        return record

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
                    "SELECT * FROM segment_surfaces WHERE surface_id=%s",
                    (job["surface_id"],),
                )
                surface = cursor.fetchone()
                if surface is None:
                    raise RuntimeError("QC job has no authoritative surface")
                surface_value = dict(surface["payload"] or {})
                self._require_surface_payload_lineage(
                    surface_value, boundary="PHYSICAL_QC_CLAIM_RESOLUTION"
                )
                cursor.execute(
                    "SELECT * FROM segment_source_snapshots WHERE source_snapshot_id=%s",
                    (surface["source_snapshot_id"],),
                )
                source = cursor.fetchone()
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='CLAIMED',worker_id=%s,lease_token_hash=%s,
                       lease_expires_at=%s,retry_after=NULL,updated_at=now() WHERE qc_job_id=%s""",
                    (worker_id, token_hash, _deadline(lease_seconds), job["qc_job_id"]),
                )
                self.event(cursor, "QC_CLAIMED", {"qc_job_id": job["qc_job_id"], "surface_id": job["surface_id"], "worker_id": worker_id})
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
                stored_result = {**result, "result_sha256": content_sha256(result)}
                cursor.execute(
                    """UPDATE segment_qc_jobs SET state='COMPLETED',result=%s::jsonb,worker_id=NULL,
                       lease_token_hash=NULL,lease_expires_at=NULL,retry_after=NULL,updated_at=now()
                       WHERE qc_job_id=%s""",
                    (json.dumps(stored_result, sort_keys=True, separators=(",", ":")), qc_job_id),
                )
                cursor.execute(
                    "UPDATE segment_surfaces SET state=%s,physical_qc_state=%s WHERE surface_id=%s",
                    (surface_state, physical_state, job["surface_id"]),
                )
                self.event(cursor, "QC_COMPLETED", {"qc_job_id": qc_job_id, "surface_id": job["surface_id"], "outcome": outcome, "evidence_manifest_sha256": result["evidence_manifest_sha256"]})
                return {"status": "COMPLETED", "qc_job_id": qc_job_id, "surface_id": job["surface_id"], "outcome": outcome, "surface_state": surface_state, "physical_qc_state": physical_state}

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

        The stale receipt goes with the state, for the same reason as in the
        SQLite store: a corrected deployment still quoting the hash it no longer
        pins reads as unfixed. QC_BLOCKED_CONFIGURATION keeps it.

        This existed only on the SQLite store, and the panel calls it through
        whichever store is configured -- so on every Postgres deployment, which
        is every real one, the requeue route raised AttributeError and answered
        500. Both stores are asserted to offer it.
        """
        if not str(fixed or "").strip():
            raise ValueError("requeueing blocked QC needs what was fixed")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                # Selected and locked in the same transaction that clears them,
                # so the record names the jobs this call moved rather than
                # whatever was blocked a moment earlier.
                cursor.execute(
                    """SELECT qc_job_id FROM segment_qc_jobs
                        WHERE state='BLOCKED_CONFIGURATION' AND surface_id IN (
                          SELECT surface_id FROM segment_surfaces
                           WHERE sample_id=%s)
                        ORDER BY qc_job_id FOR UPDATE""",
                    (sample_id,),
                )
                requeued = [row["qc_job_id"] for row in cursor.fetchall()]
                if requeued:
                    cursor.execute(
                        """UPDATE segment_qc_jobs SET state='PENDING',result=NULL,
                           worker_id=NULL,lease_token_hash=NULL,
                           lease_expires_at=NULL,retry_after=NULL,updated_at=now()
                            WHERE qc_job_id = ANY(%s)""",
                        (requeued,),
                    )
                record = {"sample_id": sample_id, "requeued": len(requeued),
                          "qc_job_ids": requeued, "fixed": str(fixed).strip(),
                          "by": by}
                self.event(cursor, "qc.requeued_after_fix", record)
                return record

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
                    (json.dumps(safe_receipt, sort_keys=True, separators=(",", ":")),
                     qc_job_id),
                )
                self.event(
                    cursor,
                    "QC_BLOCKED_CONFIGURATION",
                    {
                        "qc_job_id": qc_job_id,
                        "surface_id": job["surface_id"],
                        "error": safe_receipt["error"],
                    },
                )
                return {
                    "status": "BLOCKED_CONFIGURATION",
                    "qc_job_id": qc_job_id,
                    "surface_id": job["surface_id"],
                    "error": safe_receipt["error"],
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
        from framework.contracts import qc_diagnostics

        safe_receipt = qc_diagnostics.receipt_with_safe_error(
            receipt, "RuntimeError: retryable QC failure had no safe detail"
        )
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
                        json.dumps(safe_receipt, sort_keys=True, separators=(",", ":")),
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
                        "error": safe_receipt["error"],
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
        surface_id: str | None = None, mission_id: str | None = None,
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
                                  s.physical_qc_state, n.voxel_size_um, s.payload
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
                if mission_id is not None:
                    query += " AND " + self.MISSION_SURFACE_PREDICATE.format(alias="s.")
                    arguments.extend([mission_id] * 3)
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
                rows = [dict(row) for row in cursor.fetchall()]
        from .canonical_lineage import require_canonical_lineage  # noqa: PLC0415
        for row in rows:
            payload = dict(row.pop("payload") or {})
            controlled = payload.get("controlled_first_letters") is True
            require_canonical_lineage(
                boundary="P3_QUEUE_ADMISSION",
                controlled_mission=controlled,
                authoritative_lineage=payload.get("authoritative_lineage"),
                allow_unvalidated=(
                    not require_physical_qc if controlled
                    else payload.get("allow_unvalidated")
                ),
            )
        return rows

    def backfill_routing_receipts(self, *, apply: bool = False) -> dict[str, Any]:
        """Route the surfaces that existed before the routing did.

        The PostgreSQL twin of ``FleetStore.backfill_routing_receipts``, and the
        one that matters: the deployment runs PostgreSQL. Same frozen router,
        same two refusals -- a surface with no measured area is reported rather
        than guessed, and a surface that already has a receipt is left alone.

        One transaction for the whole pass, so a failure halfway leaves the
        control plane exactly as it was rather than half-routed.
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT s.surface_id, s.source_snapshot_id, s.sample_id,
                              s.artifact_sha256, s.area_cm2, s.bbox_xyz,
                              s.sample_points, s.geometry_qc_state,
                              r.surface_id AS routed
                         FROM segment_surfaces s
                         LEFT JOIN segment_surface_routing_receipts r
                           ON r.surface_id = s.surface_id
                        ORDER BY s.surface_id"""
                )
                for row in cursor.fetchall():
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
                        "bbox_xyz": row["bbox_xyz"],
                        "sample_points": row["sample_points"],
                        "geometry_qc_state": row["geometry_qc_state"],
                    }
                    try:
                        decision, _ = surface_routing.route(
                            surface["area_cm2"], policy=policy)
                    except ValueError:
                        summary["unroutable"] += 1
                        summary["unroutable_surface_ids"].append(row["surface_id"])
                        continue

                    summary["by_route"][decision] = (
                        summary["by_route"].get(decision, 0) + 1)
                    if apply:
                        self._write_routing_receipt(cursor, surface)
                        summary["routed"] += 1
                    else:
                        summary["would_route"] += 1

                if not apply:
                    connection.rollback()
        return summary

    def routing_receipt(self, surface_id: str) -> dict[str, Any] | None:
        """The stored routing decision, or None if the surface predates routing.

        Parity with ``FleetStore.routing_receipt`` is the document and not the
        column: PostgreSQL keeps it in ``receipt`` jsonb where SQLite keeps it in
        ``receipt_json`` text, and both hand back exactly what
        ``surface_routing.build_receipt`` wrote, so ``verify_receipt`` and the
        ``enters_*`` helpers answer the same whichever control plane was asked.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                return self._stored_routing_receipt(cursor, surface_id)

    def _require_standard_route(self, cursor: Any, surface_id: str) -> None:
        """Refuse a surface the router did not put on the standard path.

        PHerc0268 was 0.0198 cm2, GEOMETRY_CERTIFIED, and reached physical QC
        because the size question was asked in exactly one place. Every stage
        after the router asks it again for itself; this is P3 asking.

        The question goes to ``enters_canonical_downstream`` rather than to the
        ``route`` column, because that helper also verifies the receipt digest.
        A receipt whose bytes were edited is not a routing decision, and it fails
        here exactly as a missing one does -- which is what makes storing the
        receipt worth more than storing a boolean.
        """
        from . import surface_routing  # noqa: PLC0415

        receipt = self._stored_routing_receipt(cursor, surface_id)
        if surface_routing.enters_canonical_downstream(receipt or {}):
            return
        if receipt is None:
            observed = "no routing receipt"
        elif not surface_routing.verify_receipt(receipt):
            observed = "a routing receipt whose digest does not verify"
        else:
            observed = f"the route {receipt.get('route')}"
        raise RuntimeError(
            f"refusing to flatten {surface_id}: P3 is a canonical downstream "
            f"stage and admits only surfaces routed "
            f"{surface_routing.STANDARD}, and this surface has {observed}"
        )

    def record_flattening(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a flattening, or leave the one already there alone."""
        receipt_sha256 = payload.get("receipt_sha256")
        unhashed = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if (not payload.get("requested_by_job_id")
                or not payload.get("source_artifact_sha256")
                or not payload.get("profile_file_sha256")
                or receipt_sha256 != content_sha256(unhashed)):
            raise ValueError("flattening receipt lacks immutable current-job lineage")
        flattening_id = stable_id(
            "flattening",
            {"surface_id": payload["surface_id"], "profile_id": payload["profile_id"]},
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT payload FROM segment_surfaces WHERE surface_id=%s",
                    (payload["surface_id"],),
                )
                source_surface = cursor.fetchone()
                if source_surface is None:
                    raise RuntimeError("flattening source surface is missing")
                self._require_surface_payload_lineage(
                    dict(source_surface["payload"] or {}),
                    boundary="P3_EXECUTION_RESOLUTION",
                )
                self._require_standard_route(
                    cursor, str(payload["surface_id"]),
                )
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
                cursor.execute(
                    "SELECT payload FROM surface_flattenings WHERE surface_id=%s AND profile_id=%s",
                    (payload["surface_id"], payload["profile_id"]),
                )
                stored_payload = cursor.fetchone()["payload"]
                self.event(
                    cursor,
                    "SURFACE_FLATTENED" if payload.get("state") == "FLATTENED"
                    else "SURFACE_FLATTENING_FAILED",
                    {"surface_id": payload["surface_id"],
                     "profile_id": payload["profile_id"],
                     "state": payload.get("state"),
                     "inserted": inserted},
                )
        return {**stored_payload, "flattening_id": flattening_id,
                "artifact_id": flattening_id, "inserted": inserted}

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

    # How a surface belongs to a mission, for the two backlogs that hand work
    # to a worker.
    #
    # P2 and P3 are queued from inside a mission and worked the whole control
    # plane: `--sample` and `--limit` were the only filters, so a flatten run
    # started in one mission consumed a surface from another -- or from the
    # pre-mission `unfiled` history -- and published a sheet that is invisible
    # in the mission that asked for it. That is the rule the rest of the
    # platform holds to: nothing exists outside a mission.
    #
    # Three ways in, matching the panel's own predicate: grown by a task of this
    # mission, derived by one of its ink jobs, or uploaded into it. Consumes
    # three copies of the mission id.
    MISSION_SURFACE_PREDICATE = """(
      EXISTS (
        SELECT 1 FROM segment_artifact_sets art
        JOIN segment_attempts a ON a.attempt_id=art.attempt_id
        JOIN segment_tasks t ON t.task_id=a.task_id
        WHERE art.manifest->>'artifact_sha256'={alias}artifact_sha256
          AND t.mission_id = %s)
      OR EXISTS (
        SELECT 1 FROM surface_derivations d
        JOIN ink_jobs j ON j.job_id=d.job_id
        WHERE d.child_surface_id={alias}surface_id
          AND j.mission_id = %s)
      OR {alias}payload ->> 'mission_id' = %s
    )"""

    def surfaces_without_geometry_verdict(
        self, limit: int = 25, sample_id: str | None = None,
        surface_id: str | None = None, mission_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Surfaces P2 has never given a verdict to.

        GEOMETRY_UNMEASURED and a NULL column mean the same thing here -- one is
        a surface the gate could not measure, the other predates the column --
        and both are surfaces with no verdict, which is what P2 owes them.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                query = """SELECT surface_id, sample_id, artifact_uri, artifact_sha256,
                                  state, geometry_qc_state, payload
                           FROM segment_surfaces
                           WHERE (geometry_qc_state IS NULL
                                  OR geometry_qc_state = 'GEOMETRY_UNMEASURED')
                             AND artifact_uri IS NOT NULL"""
                arguments: list[Any] = []
                if sample_id is not None:
                    query += " AND sample_id=%s"
                    arguments.append(sample_id)
                if surface_id is not None:
                    query += " AND surface_id=%s"
                    arguments.append(surface_id)
                if mission_id is not None:
                    query += " AND " + self.MISSION_SURFACE_PREDICATE.format(alias="")
                    arguments.extend([mission_id] * 3)
                # Never-attempted first, so a permanently missing artifact
                # cannot starve the backlog by being retried ahead of surfaces
                # nobody has looked at -- while still being retried, because a
                # fetch failure can be the network rather than the corpus.
                query += (" ORDER BY (payload ? 'geometry_certification'),"
                          " created_at, surface_id LIMIT %s")
                arguments.append(int(limit))
                cursor.execute(query, arguments)
                rows = [dict(row) for row in cursor.fetchall()]
        result = []
        for row in rows:
            payload = dict(row.pop("payload") or {})
            self._require_surface_payload_lineage(
                payload, boundary="P2_QUEUE_ADMISSION"
            )
            result.append(row)
        return result

    def surface_artifact(
        self, surface_id: str, *, boundary: str = "P2_EXECUTION_RESOLUTION",
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT surface_id,source_snapshot_id,sample_id,artifact_uri,"
                    "artifact_sha256,payload "
                    "FROM segment_surfaces WHERE surface_id=%s", (surface_id,))
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"unknown surface: {surface_id}")
        value = dict(row)
        self._require_surface_payload_lineage(
            dict(value.get("payload") or {}), boundary=boundary
        )
        return value

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

    def record_lamina_assessment(
        self,
        surface_id: str,
        lamina_state: str,
        receipt: dict[str, Any] | None = None,
        *,
        requested_by_job_id: str,
        profile_id: str,
        profile_sha256: str,
    ) -> dict[str, Any]:
        """Record the lamina verdict without touching the other two axes.

        Orthogonal on purpose, like geometry beside CT support: a surface can be
        GEOMETRY_CERTIFIED, CT_SUPPORTED and LAMINA_FUSED at once, and that
        combination is the one worth seeing -- a mesh that is a plausible sheet,
        over material that is there, where the CT resolves two laminae welded
        together. Rendering it produces a slab.

        The receipt rides with the verdict for the same reason it does there:
        without the three numbers and the bands they were read against, a word
        like FUSED is a judgement with no measurement attached.
        """
        from .store import LAMINA_QC_STATES  # noqa: PLC0415

        if lamina_state not in LAMINA_QC_STATES:
            raise ValueError(f"unsupported lamina QC state: {lamina_state}")
        if not requested_by_job_id or not profile_id or len(profile_sha256) != 64:
            raise ValueError("a lamina verdict requires job/profile/hash lineage")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT surface_id,artifact_sha256 FROM segment_surfaces "
                    "WHERE surface_id=%s FOR UPDATE", (surface_id,))
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(f"unknown surface: {surface_id}")
                lineage = {
                    "lamina_assessed_by_job_id": requested_by_job_id,
                    "surface_id": surface_id,
                    "surface_artifact_sha256": row["artifact_sha256"],
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256,
                    "result_sha256": content_sha256(receipt or {}),
                    "result": receipt or {},
                }
                cursor.execute(
                    """UPDATE segment_surfaces
                       SET lamina_qc_state=%s,
                           payload = payload || %s::jsonb
                       WHERE surface_id=%s""",
                    (lamina_state,
                     json.dumps({"lamina_assessment": receipt,
                                 "lamina_assessment_lineage": lineage},
                                sort_keys=True, separators=(",", ":")),
                     surface_id))
                connection.commit()
        return {"surface_id": surface_id, "lamina_qc_state": lamina_state,
                "lineage": lineage}

    def record_seed_agreement(
        self,
        surface_id: str,
        seed_state: str,
        receipt: dict[str, Any] | None = None,
        *,
        requested_by_job_id: str,
        paired_with_surface_id: str | None = None,
    ) -> dict[str, Any]:
        """Record how far a second run of the same fit landed from this one.

        The fifth judgement, and the only one that is not about this surface:
        the other four say where it came from, whether the scan supports it,
        whether the mesh is a plausible sheet and whether the CT resolves one
        lamina under it. This says whether the fit that produced it is
        reproducible, which is a property of the method and not of the artifact.

        Kept apart from the other four so it can contradict them, because on
        this corpus it does: the most tangled band of PHerc0826 w015 -- 830
        fold-back intersections and real self-contact -- had the best agreement
        between seeds of the three bands in that patch. A low number there is
        not evidence of a good surface, and a schema that let this stand in for
        a geometry verdict would be lying.

        The receipt rides with the state for the reason the lamina verdict's
        does: without the decomposition, the percentiles and what the
        normalisation divided by, a number like "12 um" is a judgement with no
        measurement attached -- and this particular one collides with a figure
        the campaign published, so the receipt has to say which it is.
        """
        from .store import SEED_AGREEMENT_STATES  # noqa: PLC0415

        if seed_state not in SEED_AGREEMENT_STATES:
            raise ValueError(f"unsupported seed agreement state: {seed_state}")
        if not requested_by_job_id:
            raise ValueError("a seed agreement verdict requires its job lineage")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT surface_id,artifact_sha256 FROM segment_surfaces "
                    "WHERE surface_id=%s FOR UPDATE", (surface_id,))
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(f"unknown surface: {surface_id}")
                lineage = {
                    "seed_agreement_assessed_by_job_id": requested_by_job_id,
                    "surface_id": surface_id,
                    "surface_artifact_sha256": row["artifact_sha256"],
                    # Which surface it was compared against, because a pair that
                    # cannot be named is a number nobody can reproduce.
                    "paired_with_surface_id": paired_with_surface_id,
                    "result_sha256": content_sha256(receipt or {}),
                    "result": receipt or {},
                }
                cursor.execute(
                    """UPDATE segment_surfaces
                       SET seed_agreement_state=%s,
                           payload = payload || %s::jsonb
                       WHERE surface_id=%s""",
                    (seed_state,
                     json.dumps({"seed_agreement": receipt,
                                 "seed_agreement_lineage": lineage},
                                sort_keys=True, separators=(",", ":")),
                     surface_id))
                connection.commit()
        return {"surface_id": surface_id, "seed_agreement_state": seed_state,
                "lineage": lineage}

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
        """Record a geometry verdict without touching the ink/CT axis."""

        if geometry_state not in GEOMETRY_QC_STATES:
            raise ValueError(f"unsupported geometry QC state: {geometry_state}")
        if not requested_by_job_id or not profile_id or len(profile_sha256) != 64:
            raise ValueError("geometry certification requires job/profile/hash lineage")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT surface_id,physical_qc_state,artifact_sha256,"
                    "area_cm2,source_snapshot_id,sample_id,bbox_xyz,"
                    "sample_points,payload "
                    "FROM segment_surfaces WHERE surface_id=%s FOR UPDATE",
                    (surface_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(f"unknown surface: {surface_id}")
                # The receipt rides with the verdict. Without it "unmeasured"
                # is a word with no reason attached, and the difference between
                # a grid too coarse to measure and an artifact that could not be
                # fetched is only visible in whoever ran it's terminal.
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
                cursor.execute(
                    """UPDATE segment_surfaces
                       SET geometry_qc_state=%s,
                           payload = payload || %s::jsonb
                       WHERE surface_id=%s""",
                    (geometry_state,
                     json.dumps({"geometry_certification": receipt,
                                 "geometry_certification_lineage": lineage},
                                sort_keys=True, separators=(",", ":")),
                     surface_id),
                )
                blocked = 0
                # A verdict arriving is what makes waiting different from
                # stranded: a job held back for unmeasured geometry becomes
                # claimable here, and only here.
                promoted = 0
                promotion_links: list[dict[str, str]] = []
                if str(geometry_state) == "GEOMETRY_CERTIFIED":
                    cursor.execute(
                        f"SELECT qc_job_id,payload FROM segment_qc_jobs WHERE surface_id=%s "
                        f"AND state='{QC_WAITING_GEOMETRY}' ORDER BY qc_job_id FOR UPDATE",
                        (surface_id,),
                    )
                    waiting = cursor.fetchall()
                    # A geometry verdict is not a size verdict and cannot
                    # overrule one. This is the exact place PHerc0268's
                    # certificate turned into a claimable job, so the routing
                    # decision is required here too -- before the row becomes
                    # PENDING, not in whatever reads it afterwards.
                    from . import surface_routing  # noqa: PLC0415

                    routing_receipt = (
                        self._require_routing_receipt(
                            cursor,
                            {"surface_id": surface_id,
                             "area_cm2": row["area_cm2"],
                             "source_snapshot_id": row["source_snapshot_id"],
                             "sample_id": row["sample_id"],
                             "artifact_sha256": row["artifact_sha256"],
                             "bbox_xyz": row["bbox_xyz"],
                             "sample_points": row["sample_points"],
                             "geometry_qc_state": geometry_state})
                        if waiting else None
                    )
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
                        if not releasable:
                            # Below the floor: the job stays durable, auditable,
                            # and unclaimable. Nothing failed and no verdict is
                            # coming; the surface is too small for the standard
                            # path and that is a fact about its size alone.
                            cursor.execute(
                                """UPDATE segment_qc_jobs SET state=%s,
                                   payload=payload || %s::jsonb,updated_at=now()
                                   WHERE qc_job_id=%s""",
                                (QC_SMALL_SURFACE_DIAGNOSTIC, json.dumps({
                                    "surface_artifact_sha256":
                                        row["artifact_sha256"],
                                    "unblocked_by_job_id": requested_by_job_id,
                                    "promotion_event_id": promotion_event_id,
                                    "small_surface_routing_sha256":
                                        routing_receipt["receipt_sha256"],
                                }, sort_keys=True, separators=(",", ":")),
                                 qc_row["qc_job_id"]),
                            )
                            continue
                        cursor.execute(
                            """UPDATE segment_qc_jobs SET state='PENDING',
                               payload=payload || %s::jsonb,updated_at=now()
                               WHERE qc_job_id=%s""",
                            (json.dumps({
                                "surface_artifact_sha256": row["artifact_sha256"],
                                "unblocked_by_job_id": requested_by_job_id,
                                "promotion_event_id": promotion_event_id,
                            }, sort_keys=True, separators=(",", ":")),
                             qc_row["qc_job_id"]),
                        )
                        promotion_links.append({"qc_job_id": qc_row["qc_job_id"],
                                                "promotion_event_id": promotion_event_id})
                    promoted = len(promotion_links)
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
                                    "blocked_by_job_id": requested_by_job_id,
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
                        **lineage,
                        "qc_promotion_links": promotion_links,
                    },
                )
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

    def mark_terminal(self, task_id: str, attempt_id: str, lease_token: str, state: str, result: dict[str, Any]) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state}")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT t.mission_id,t.policy_version,
                              EXISTS(
                                SELECT 1
                                  FROM segment_campaign_budget_admissions a
                                 WHERE a.mission_id=t.mission_id
                                   AND a.admission->'execution_bindings'
                                       ->>'policy_version'=t.policy_version
                              ) AS controlled
                         FROM segment_tasks t WHERE t.task_id=%s""",
                    (task_id,),
                )
                scope = cursor.fetchone()
                if scope is None:
                    raise RuntimeError("attempt no longer owns the task")
                if scope["controlled"]:
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"campaign-budget-mission:{scope['mission_id']}",),
                    )
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
                if scope["controlled"]:
                    self._refresh_campaign_decisions(
                        cursor,
                        mission_id=scope["mission_id"],
                        policy_version=scope["policy_version"],
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

    def insert_human_review(self, event: dict[str, Any]) -> dict[str, Any]:
        """Atomically insert an immutable review or return the existing intent.

        Same three refusals as the SQLite control plane, for the same reason:
        the intent comes from a frozen enum, the lineage lock has to re-derive,
        and the route is re-read from this store's own routing receipt instead
        of being believed from the event.
        """
        from .review_lineage import (  # noqa: PLC0415
            require_review_intent, require_standard_route,
            verify_review_lineage_lock,
        )

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
        require_review_intent(event.get("intent"))
        verify_review_lineage_lock(event)
        # A control plane too old to answer "which route did this surface take"
        # is refused, not waved through: the whole failure being prevented is a
        # surface entering canonical downstream with nobody having classified it.
        reader = getattr(self, "routing_receipt", None)
        stored_route = require_standard_route(
            reader(str(event["surface_id"])) if callable(reader) else None)
        if (stored_route.get("surface_id") != event.get("surface_id")
                or stored_route.get("receipt_sha256")
                != event.get("routing_receipt_sha256")):
            raise ValueError(
                "the stored route is not the route this review was locked to")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO human_review_events(
                           review_event_id,p7_job_id,intent,mission_id,sample_id,
                           surface_id,verdict_sha256,card_sha256,config_sha256,
                           vetting_packet_sha256,author,event,event_sha256,created_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
                       ON CONFLICT(p7_job_id,intent) DO NOTHING""",
                    (
                        event["review_event_id"], event["p7_job_id"], event["intent"],
                        event["mission_id"], event["sample_id"], event["surface_id"],
                        event["verdict_sha256"], event["card_sha256"],
                        event["config_sha256"], event["vetting_packet_sha256"],
                        event["by"], json.dumps(event, sort_keys=True, separators=(",", ":")),
                        event["event_sha256"], event["at"],
                    ),
                )
                cursor.execute(
                    """SELECT event FROM human_review_events
                        WHERE p7_job_id=%s AND intent=%s""",
                    (event["p7_job_id"], event["intent"]),
                )
                row = cursor.fetchone()
                return dict(row["event"])

    def human_reviews(self, p7_job_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT event FROM human_review_events
                        WHERE p7_job_id=%s ORDER BY created_at,review_event_id""",
                    (p7_job_id,),
                )
                return [dict(row["event"]) for row in cursor.fetchall()]

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
