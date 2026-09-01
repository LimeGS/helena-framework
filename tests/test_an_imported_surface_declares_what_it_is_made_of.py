"""A surface whose bytes live on somebody else's host declares its contents.

An import records a URI and one digest over the artifact set. That is enough
when the bytes sit in our own bucket: we control what is there. It is not enough
when they sit on a public host, because the only thing that could turn that
digest back into per-file expectations is a manifest fetched from the same place
as the bytes, which vouches for itself.

So the inventory is recorded at the door, from whatever source the importer
chose to trust -- for the control that is the file list the frozen manifest
already locks -- and the reader checks fetched bytes against it. The host
supplies bytes and nothing else.

Required only for http(s) URIs. An `s3://` import lands in a bucket this fleet
runs, and demanding an inventory there would be ceremony rather than evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from panel.app import imported_surface_inventory  # noqa: E402

PUBLIC = "https://dl.example.invalid/community/w025.tifxyz"
GOOD = [
    {"path": "x.tif", "sha256": "a" * 64, "size_bytes": None},
    {"path": "meta.json", "sha256": "b" * 64, "size_bytes": 415},
]


def test_a_public_surface_without_an_inventory_is_refused() -> None:
    with pytest.raises(HTTPException) as refusal:
        imported_surface_inventory({}, PUBLIC)

    assert refusal.value.status_code == 400
    assert "artifacts" in str(refusal.value.detail)


def test_a_public_surface_with_an_inventory_keeps_it() -> None:
    kept = imported_surface_inventory({"artifacts": GOOD}, PUBLIC)

    assert kept == GOOD


def test_an_entry_without_a_usable_digest_is_refused() -> None:
    for broken in ({"path": "x.tif"},
                   {"path": "x.tif", "sha256": "nope"},
                   {"sha256": "a" * 64}):
        with pytest.raises(HTTPException):
            imported_surface_inventory({"artifacts": [broken]}, PUBLIC)


def test_a_declared_size_must_be_a_number() -> None:
    with pytest.raises(HTTPException):
        imported_surface_inventory(
            {"artifacts": [{"path": "x.tif", "sha256": "a" * 64,
                            "size_bytes": "biggish"}]}, PUBLIC)


def test_a_bucket_we_run_needs_no_inventory() -> None:
    assert imported_surface_inventory({}, "s3://our-bucket/surfaces/w1") is None
    assert imported_surface_inventory({}, "/mnt/bulk/surfaces/w1") is None


def test_an_inventory_is_kept_wherever_it_is_offered() -> None:
    """Not required for S3, but recorded if the importer took the trouble."""
    assert imported_surface_inventory(
        {"artifacts": GOOD}, "s3://our-bucket/surfaces/w1") == GOOD


def test_the_inventory_reaches_the_stored_payload() -> None:
    """The route persists it, or nothing downstream can read it back."""
    source = (ROOT / "panel/app.py").read_text()
    handler = source[source.index('@app.post("/api/segmentation/import")'):]
    handler = handler[:handler.index("\n@app.")]

    assert "imported_surface_inventory" in handler, (
        "the import route does not validate the inventory")
    assert '"artifacts": artifacts' in handler, (
        "the inventory is validated and then dropped: the payload never carries it")


def test_an_inventory_can_be_recorded_for_a_surface_already_imported() -> None:
    """Surfaces imported before this existed have no inventory, and the reader
    cannot check what it fetches for them.

    `DO NOTHING` left no way to supply one short of direct SQL, which the lane
    forbids for real-data mutations. So the conflict path fills the inventory in
    when it is absent -- and only then. It never replaces one already recorded
    and never touches another field: adding what was missing, not rewriting what
    was measured.
    """
    source = (ROOT / "panel/app.py").read_text()
    handler = source[source.index('@app.post("/api/segmentation/import")'):]
    handler = handler[:handler.index("\n@app.")]
    conflict = handler[handler.index("ON CONFLICT"):]
    conflict = conflict[:conflict.index('"""')]

    assert "DO UPDATE" in conflict, "an import can never supply a missing inventory"
    assert "artifacts" in conflict
    assert "IS DISTINCT FROM 'array'" in conflict, (
        "the conflict path can overwrite an inventory that was already recorded")
