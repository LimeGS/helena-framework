from __future__ import annotations

import copy
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .store import FleetStore


POSTGRES_ENV_PREFIX = "postgres-env://"
DISCOVERY_PROFILE_PATH_ENV = "HELENA_FIRST_LETTERS_DISCOVERY_PROFILE_PATH"
DISCOVERY_ARMS_PATH_ENV = "HELENA_FIRST_LETTERS_EXPERIMENTAL_ARMS_PATH"


def _discovery_authority_resolvers() -> dict[str, Any]:
    """Load configured authority once; callers receive defensive copies."""

    profile_path = os.environ.get(DISCOVERY_PROFILE_PATH_ENV)
    arms_path = os.environ.get(DISCOVERY_ARMS_PATH_ENV)
    result: dict[str, Any] = {}
    if profile_path:
        profile_bytes = Path(profile_path).read_bytes()
        result["first_letters_discovery_profile_resolver"] = (
            lambda mission_id, source_snapshot_id: bytes(profile_bytes)
        )
    if arms_path:
        arms = json.loads(Path(arms_path).read_text(encoding="utf-8"))
        if not isinstance(arms, dict):
            raise ValueError("configured experimental arms must be an object")
        result["first_letters_experimental_arm_resolver"] = (
            lambda arm_id: copy.deepcopy(arms.get(arm_id))
        )
    return result


def _database_url(spec: str) -> str:
    if spec.startswith(POSTGRES_ENV_PREFIX):
        variable = spec.removeprefix(POSTGRES_ENV_PREFIX)
        if not variable or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" for character in variable):
            raise ValueError("postgres-env database spec must contain one environment-variable name")
        value = os.environ.get(variable)
        if not value:
            raise RuntimeError(f"database environment variable is not set: {variable}")
        if not value.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(f"database environment variable is not a PostgreSQL URL: {variable}")
        return value
    return spec


def open_fleet_store(spec: str | Path) -> Any:
    """Open a local SQLite or central PostgreSQL control plane.

    ``postgres-env://NAME`` is the preferred production form because it keeps
    credentials out of process listings, shell history, logs and receipts.
    """

    raw = str(spec)
    resolved = _database_url(raw)
    from .discovery_executor import (
        PRODUCTION_DISCOVERY_EXECUTOR_ID,
        ProductionFirstLettersDiscoveryExecutor,
        production_discovery_executor_registration,
        production_discovery_worker_id,
    )

    executor = ProductionFirstLettersDiscoveryExecutor()
    # The executor is part of who this worker is. Without it the id holds still
    # across a code change while the registration does not, and the registry --
    # immutable per worker_id, with no update path -- refuses the deploy.
    worker_id = production_discovery_worker_id(executor=executor)
    discovery = {
        "first_letters_discovery_executor": executor,
        "first_letters_discovery_worker_id": worker_id,
        "first_letters_discovery_executor_id":
            PRODUCTION_DISCOVERY_EXECUTOR_ID,
        "first_letters_discovery_executor_registration":
            production_discovery_executor_registration(
                executor, worker_id=worker_id,
            ),
        **_discovery_authority_resolvers(),
    }
    if resolved.startswith(("postgresql://", "postgres://")):
        from .postgres_store import PostgresFleetStore

        return PostgresFleetStore(
            resolved,
            identity=(raw if raw.startswith(POSTGRES_ENV_PREFIX)
                      else "postgresql://redacted"),
            **discovery,
        )
    return FleetStore(Path(raw), **discovery)


class ReadOnlySqliteFleetView(FleetStore):
    """The two preflight reads on an existing SQLite control plane.

    URI ``mode=ro`` is the enforcement boundary.  ``query_only`` is a second
    guard and, unlike FleetStore.connect(), this does not request WAL mode.
    """

    def connect(self) -> sqlite3.Connection:
        uri = self.path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def initialize(self) -> None:  # pragma: no cover - fail-closed tripwire
        raise RuntimeError("a read-only fleet view cannot initialize or migrate schema")

    def verify_read_only_schema(self) -> None:
        required = {
            "source_snapshots": {"source_snapshot_id", "sample_id", "payload_json",
                                 "shape_xyz_json"},
            "surfaces": {"surface_id", "source_snapshot_id", "payload_json",
                         "bbox_xyz_json", "sample_points_json", "geometry_qc_state"},
        }
        with self.connect() as connection:
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if not required.keys() <= tables:
                raise RuntimeError("segmentation schema is not initialized")
            for table, columns in required.items():
                actual = {str(row[1]) for row in connection.execute(
                    f"PRAGMA table_info({table})")}
                if not columns <= actual:
                    raise RuntimeError(f"segmentation schema is incomplete: {table}")


def open_fleet_store_read_only(spec: str | Path) -> Any:
    """Open a verified existing control plane with database-enforced read-only I/O."""
    raw = str(spec)
    resolved = _database_url(raw)
    if resolved.startswith(("postgresql://", "postgres://")):
        from .postgres_store import PostgresFleetStore

        class ReadOnlyPostgresFleetView(PostgresFleetStore):
            def connect(self):
                connection = super().connect()
                connection.set_session(readonly=True, autocommit=True)
                return connection

            def initialize(self) -> None:  # pragma: no cover - fail-closed tripwire
                raise RuntimeError("a read-only fleet view cannot initialize or migrate schema")

            def verify_read_only_schema(self) -> None:
                required = {
                    "segment_source_snapshots": {
                        "source_snapshot_id", "sample_id", "payload", "shape_xyz"},
                    "segment_surfaces": {
                        "surface_id", "source_snapshot_id", "payload", "bbox_xyz",
                        "sample_points", "geometry_qc_state"},
                }
                with self.connect() as connection:
                    with connection.cursor() as cursor:
                        for table, columns in required.items():
                            cursor.execute(
                                """SELECT column_name FROM information_schema.columns
                                   WHERE table_schema=current_schema() AND table_name=%s""",
                                (table,),
                            )
                            actual = {str(row["column_name"]) for row in cursor.fetchall()}
                            if not columns <= actual:
                                raise RuntimeError(
                                    f"segmentation schema is incomplete: {table}")

        view = ReadOnlyPostgresFleetView(
            resolved,
            identity=(raw if raw.startswith(POSTGRES_ENV_PREFIX)
                      else "postgresql://redacted"),
        )
    else:
        path = Path(resolved)
        if not path.is_file():
            raise RuntimeError("segmentation store is not initialized")
        view = ReadOnlySqliteFleetView(path)
    view.verify_read_only_schema()
    return view


def store_identity(store: Any) -> str:
    return str(getattr(store, "identity", getattr(store, "path", "unknown")))
