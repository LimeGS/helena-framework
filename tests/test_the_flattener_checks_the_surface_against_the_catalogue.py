"""P3 hands the reader the inventory the catalogue recorded.

The adapter can only prefer our inventory over the source's manifest if someone
passes it one. The flattening queue already selects `s.payload`, so nothing new
has to be fetched -- the list just has to survive the last few lines between the
query and the call.

Absent for surfaces this fleet grew, which arrive with a manifest from an
artifact store we run. Present for imports, which is the case that needs it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.cli import locked_inventory  # noqa: E402

FILES = [{"path": "x.tif", "sha256": "a" * 64, "size_bytes": None}]


def test_an_imported_surface_carries_its_inventory() -> None:
    assert locked_inventory({"payload": {"artifacts": FILES}}) == FILES


def test_a_grown_surface_has_none() -> None:
    assert locked_inventory({"payload": {"source_catalog": "fleet"}}) is None
    assert locked_inventory({"payload": None}) is None
    assert locked_inventory({}) is None


def test_an_empty_or_malformed_inventory_is_not_an_inventory() -> None:
    """An empty list must not read as 'nothing to check' further down: the
    adapter refuses an inventory that omits required files, and it can only do
    that if it is handed one at all."""
    assert locked_inventory({"payload": {"artifacts": []}}) is None
    assert locked_inventory({"payload": {"artifacts": "x.tif"}}) is None


def test_the_flattener_passes_it_to_the_reader() -> None:
    source = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    body = source[source.index("def command_flatten"):]
    body = body[:body.index("\ndef ")]

    assert "materialize_surface(" in body
    call = body[body.index("materialize_surface("):]
    call = call[:call.index(")\n") + 1]
    assert "expected_files=" in call, (
        "P3 reads the surface without the catalogue's inventory, so the reader "
        "falls back to whatever manifest the source offered")
