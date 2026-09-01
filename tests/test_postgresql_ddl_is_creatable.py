"""The PostgreSQL schema has to be creatable by PostgreSQL.

`authorization` is a reserved word there. Two tables declared a column with that
name unquoted, so `CREATE TABLE` raised a syntax error, so `initialize()` never
finished, so *no* table existed and every PostgreSQL-backed test errored at its
fixture. The deployment runs PostgreSQL; the tests ran SQLite and never noticed.

Two checks, because they fail at different times. The first parses the DDL and
needs no database, so it holds in CI where there is none. The second creates the
schema on a real server, which is the only thing that proves the first is asking
the right question.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

DDL = ROOT / "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql"

# PostgreSQL's reserved keywords: those that cannot appear as a bare column name.
# Kept short and exact rather than exhaustive -- an unquoted identifier from this
# set is a syntax error, and a name outside it is not made safe by being absent.
RESERVED = {
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc",
    "asymmetric", "authorization", "binary", "both", "case", "cast", "check",
    "collate", "collation", "column", "concurrently", "constraint", "create",
    "cross", "current_catalog", "current_date", "current_role", "current_schema",
    "current_time", "current_timestamp", "current_user", "default", "deferrable",
    "desc", "distinct", "do", "else", "end", "except", "false", "fetch", "for",
    "foreign", "freeze", "from", "full", "grant", "group", "having", "ilike",
    "in", "initially", "inner", "intersect", "into", "is", "isnull", "join",
    "lateral", "leading", "left", "like", "limit", "localtime", "localtimestamp",
    "natural", "not", "notnull", "null", "offset", "on", "only", "or", "order",
    "outer", "overlaps", "placing", "primary", "references", "returning",
    "right", "select", "session_user", "similar", "some", "symmetric", "table",
    "tablesample", "then", "to", "trailing", "true", "union", "unique", "user",
    "using", "variadic", "verbose", "when", "where", "window", "with",
}

# Split each CREATE TABLE body on the commas that separate its items, ignoring
# commas inside parentheses. Line-oriented matching cannot do this: `REFERENCES
# other(id) ON DELETE CASCADE` and a CHECK expression both continue onto lines
# that look exactly like a column declaration, which is how a first attempt at
# this test reported fifty-eight columns that do not exist.
_TABLE = re.compile(
    r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+[a-z_][a-z0-9_]*\s*\((.*?)\n\s*\)\s*;",
    re.IGNORECASE | re.DOTALL)
_CONSTRAINT = re.compile(
    r"^(PRIMARY|UNIQUE|FOREIGN|CHECK|CONSTRAINT|EXCLUDE|LIKE)\b", re.IGNORECASE)


def _without_comments(body: str) -> str:
    """Drop `-- …` comments before splitting.

    A comma inside a comment splits the body as if it were a column boundary,
    and the text after it becomes an imaginary column. This check reported one
    named `and`, from a comment reading "…stdout, and a reason…". Skipping items
    that *start* with `--` is not enough: the fragment after the comma does not.
    """
    return "\n".join(line.split("--", 1)[0] for line in body.splitlines())


def _split_items(body: str) -> list[str]:
    items, depth, current = [], 0, []
    for character in _without_comments(body):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            items.append("".join(current))
            current = []
            continue
        current.append(character)
    items.append("".join(current))
    return items


def _declared_columns() -> list[tuple[int, str]]:
    text = DDL.read_text()
    found = []
    for table in _TABLE.finditer(text):
        line_of_body = text[: table.start(1)].count("\n") + 1
        for item in _split_items(table.group(1)):
            stripped = item.strip()
            if not stripped or stripped.startswith("--") or _CONSTRAINT.match(stripped):
                continue
            name = stripped.split()[0]
            found.append((line_of_body + item.count("\n"), name))
    return found


def test_the_parser_sees_the_columns_it_is_meant_to_check() -> None:
    """A check that silently matches nothing passes forever."""
    columns = _declared_columns()
    assert len(columns) > 100, f"only {len(columns)} columns parsed out of the DDL"
    assert any(name == "mission_id" for _, name in columns)


def test_no_column_is_named_with_a_reserved_word() -> None:
    offenders = [(number, name) for number, name in _declared_columns()
                 if name.lower() in RESERVED]
    assert not offenders, (
        "PostgreSQL will not parse these column declarations: "
        + ", ".join(f"{DDL.name}:{n} {name!r}" for n, name in offenders)
    )


DSN = os.environ.get("HELENA_TEST_DSN")


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_the_schema_creates_on_a_real_server() -> None:
    """The only check that proves the one above is asking the right question."""
    from fleet.postgres_store import PostgresFleetStore

    store = PostgresFleetStore(DSN)
    store.initialize()

    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) AS n FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE 'segment_%'"
            )
            assert cursor.fetchone()["n"] > 20, (
                "initialize() returned but the schema is not there"
            )
            # The column this file exists for, by name, on the server.
            cursor.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name='segment_campaign_resume_authorizations' "
                "AND column_name='authorization'"
            )
            row = cursor.fetchone()
            assert row is not None and row["data_type"] == "jsonb"


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_initialize_installs_into_the_schema_it_creates_tables_in() -> None:
    """`initialize()` must ask whether *this* schema is ready, not any schema.

    An unqualified CREATE TABLE lands in `current_schema()`, so that is the
    only schema whose migration ledger answers "is there anything to do".  A
    search-path-wide lookup answers a different question, and answers "ready"
    for a schema that has no table in it at all -- after which every statement
    silently reads and writes the neighbour's rows.
    """
    import uuid

    from fleet.postgres_store import PostgresFleetStore

    PostgresFleetStore(DSN).initialize()
    schema = f"searchpath_{uuid.uuid4().hex}"

    class _ScopedStore(PostgresFleetStore):
        def connect(self):
            connection = super().connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}, public")
            return connection

    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    try:
        _ScopedStore(DSN).initialize()
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) AS n FROM information_schema.tables "
                    "WHERE table_schema=%s AND table_name LIKE 'segment\\_%%'",
                    (schema,),
                )
                assert cursor.fetchone()["n"] > 20, (
                    "initialize() read another schema's migration ledger and "
                    "left this one empty"
                )
    finally:
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA {schema} CASCADE")
