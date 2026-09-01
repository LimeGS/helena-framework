"""A method defined twice in one class is one method and one dead body.

Python keeps the last definition and discards the first without a word. Nothing
warns, every test exercises the survivor, and the discarded one reads in review
as covered code. `PostgresFleetStore.routing_receipt` was defined twice by two
agents working the same week -- both correct, both querying the same table, one
of them unreachable.

The cost is not the duplication. It is that a later change to the dead copy
passes review, passes its own reading, and does nothing.

Checked by parsing rather than by importing, so a module that cannot be imported
in this environment is still covered.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    ROOT / "framework/stages/01-segmentation/fleet/store.py",
    ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py",
    ROOT / "framework/stages/01-segmentation/fleet/surface_routing.py",
    ROOT / "framework/stages/01-segmentation/fleet/surface_expansion.py",
    ROOT / "framework/stages/03-ink/fleet/job_store.py",
    ROOT / "framework/stages/03-ink/fleet/ink_worker.py",
    ROOT / "panel/app.py",
]


def _duplicates(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        seen: dict[str, int] = {}
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # A @property and its @x.setter share a name legitimately, and so do
            # @typing.overload stubs. Neither shadows anything.
            decorators = {
                ast.unparse(d).split("(")[0] for d in member.decorator_list}
            if any(d.endswith((".setter", ".getter", ".deleter")) or
                   d in {"overload", "typing.overload"} for d in decorators):
                continue
            if member.name in seen:
                found.append(
                    f"{path.name}: {node.name}.{member.name} defined at line "
                    f"{seen[member.name]} and again at line {member.lineno}; "
                    "the first is unreachable")
            seen[member.name] = member.lineno
    return found


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_method_is_defined_twice_in_one_class(path: Path) -> None:
    assert path.exists(), f"{path} moved; this check is now pointing at nothing"
    assert not _duplicates(path), "\n".join(_duplicates(path))


def test_the_check_detects_a_shadowed_method(tmp_path: Path) -> None:
    """Proven against the shape it is meant to catch, not assumed."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "class Store:\n"
        "    def read(self): return 1\n"
        "    def read(self): return 2\n",
        encoding="utf-8")
    assert _duplicates(sample), "the parser missed a plainly shadowed method"

    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        "class Store:\n"
        "    @property\n"
        "    def size(self): return 1\n"
        "    @size.setter\n"
        "    def size(self, value): pass\n",
        encoding="utf-8")
    assert not _duplicates(innocent), "a property setter was called a duplicate"
