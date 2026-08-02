"""The schema can change without anyone remembering to bump a number.

`initialize()` returns early once a sentinel version is recorded. That sentinel
used to be the literal 2 in the source, and the migration file later grew

    ALTER TABLE segment_surfaces ADD COLUMN IF NOT EXISTS geometry_qc_state text

without the literal moving. Every database that had already recorded version 2
skipped the file forever, so the column was never added -- while the finalizer's
INSERT names it. The failure was scheduled for the first surface to finish
after that change, in the one code path that only runs when segmentation
succeeds, which is where a crash costs the most and is seen the least.

These are static: they read the two files and need no database.
"""

from __future__ import annotations

import re
from pathlib import Path

FLEET = Path(__file__).resolve().parents[1] / "framework/stages/01-segmentation/fleet"
SQL = FLEET / "migrations/001_postgresql.sql"
STORE = FLEET / "postgres_store.py"

VERSION_ROW = re.compile(
    r"INSERT\s+INTO\s+segment_schema_migrations\s*\([^)]*\)\s*VALUES\s*\(\s*(\d+)",
    re.IGNORECASE)


def declared_versions() -> list[int]:
    return [int(m) for m in VERSION_ROW.findall(SQL.read_text(encoding="utf-8"))]


def test_the_sentinel_is_read_from_the_migration_not_hardcoded():
    """A literal here is the bug. It cannot follow a file it does not read."""
    source = STORE.read_text(encoding="utf-8")
    hardcoded = re.findall(r"segment_schema_migrations\s+WHERE\s+version\s*=\s*(\d+)",
                           source, re.IGNORECASE)
    assert not hardcoded, (
        f"initialize() compares against the literal version {hardcoded} instead of "
        "the highest version the migration declares, so adding DDL to the file "
        "without also editing this literal silently skips it")
    assert "target = max(" in source, "the sentinel is no longer derived from the file"


def test_every_declared_version_is_consecutive_from_one():
    versions = declared_versions()
    assert versions, "the migration records no version at all"
    assert sorted(versions) == list(range(1, len(versions) + 1)), (
        f"versions {sorted(versions)} skip or repeat; the sentinel is the max, so a "
        "gap means a database can record the top one without the rest having run")


def test_the_migration_declares_every_column_the_store_writes():
    """An INSERT naming a column the schema never creates fails at runtime only.

    Statically: every column listed in an INSERT into a segment_ table must
    appear somewhere in the migration that builds that table.
    """
    schema = SQL.read_text(encoding="utf-8")
    source = STORE.read_text(encoding="utf-8")
    missing: list[str] = []
    for table, columns in re.findall(
            r"INSERT\s+INTO\s+(segment_\w+)\s*\(([^)]*)\)", source, re.IGNORECASE):
        for column in (c.strip() for c in columns.split(",")):
            # Bare identifiers only: skip anything with an expression in it.
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", column):
                continue
            if not re.search(rf"\b{re.escape(column)}\b", schema):
                missing.append(f"{table}.{column}")
    assert not missing, f"written by the store, never created by the migration: {missing}"
