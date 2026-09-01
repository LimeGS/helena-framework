"""Every name the PostgreSQL store borrows from the SQLite one has to exist.

`PostgresFleetStore` reuses `FleetStore`'s pure validators rather than keeping a
second copy of each contract, which is right: two copies of a validator is two
validators, and the one that drifts is the one nobody is looking at. The cost is
that the delegation is only checked when the delegating method runs, and most of
these never do -- forty of this class's method bodies do not execute once in the
whole suite.

One of them delegated to `FleetStore._validate_historical_discovery_manifest`,
which exists nowhere in the repository. Import succeeded, every linter passed,
the suite was green, and the call would have raised `AttributeError` the first
time a real deployment reached it.

So the delegations are checked without running them: read the class, resolve
every borrowed name, and fail on the ones that resolve to nothing. This is the
cheap half of the parity question and it holds with no database at all.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

SOURCE = STAGE / "fleet/postgres_store.py"
_TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
_CLASS = next(node for node in _TREE.body
              if isinstance(node, ast.ClassDef) and node.name == "PostgresFleetStore")


def _attribute_reads(owner: str, tree: ast.AST) -> set[tuple[str, int]]:
    return {
        (node.attr, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
        and isinstance(node.ctx, ast.Load)
    }


def test_every_borrowed_fleetstore_name_resolves():
    missing = sorted(
        f"{SOURCE.name}:{line}: FleetStore.{name}"
        for name, line in _attribute_reads("FleetStore", _TREE)
        if not hasattr(FleetStore, name)
    )
    assert not missing, (
        "the PostgreSQL store borrows names the SQLite store does not have: "
        + "; ".join(missing)
    )


def test_every_self_reference_resolves():
    """The same question asked of the store about itself.

    A method reaching for `self.something` that is neither a method nor an
    attribute this class ever assigns fails the same way, for the same reason,
    on the same never-executed paths.
    """
    assigned = {
        node.attr
        for node in ast.walk(_CLASS)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
    }
    missing = sorted(
        f"{SOURCE.name}:{line}: self.{name}"
        for name, line in _attribute_reads("self", _CLASS)
        if name not in assigned and not hasattr(PostgresFleetStore, name)
    )
    assert not missing, (
        "the PostgreSQL store calls into itself for names it does not have: "
        + "; ".join(missing)
    )
