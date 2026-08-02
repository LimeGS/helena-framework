from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .store import FleetStore


POSTGRES_ENV_PREFIX = "postgres-env://"


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
    if resolved.startswith(("postgresql://", "postgres://")):
        from .postgres_store import PostgresFleetStore

        return PostgresFleetStore(resolved, identity=raw if raw.startswith(POSTGRES_ENV_PREFIX) else "postgresql://redacted")
    return FleetStore(Path(raw))


def store_identity(store: Any) -> str:
    return str(getattr(store, "identity", getattr(store, "path", "unknown")))
