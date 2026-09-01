"""What a fetched surface is checked against must be ours, not the host's.

`_verify_surface` compares the manifest's declared `artifact_sha256` to the
catalogue digest and then checks every file against that same manifest. Both
halves come from wherever the surface came from. For our own S3 bucket that is
merely circular; pointed at a public host it is a hole you can drive through --
serve a manifest declaring the locked digest, list the hashes of forged bytes
in it, and both checks pass.

It also does not work. A community host does not serve `ARTIFACT_SET.json`;
that name is a Helena convention. Requiring third parties to publish our
private manifest format is the kind of obstacle that does not survive contact
with 600 scrolls, none of whose segmentations will have one.

So the inventory travels with the catalogue entry instead: recorded once when
the surface is imported, from a source we chose to trust, and authoritative
from then on. The host supplies bytes and nothing else. Where an inventory is
present it also overrides any manifest the source offered, which closes the
same hole on the S3 and local branches rather than only the new one.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE))

from tests.test_surface_qc_adapter import load_adapter, make_surface  # noqa: E402

BASE = "https://dl.example.invalid/community/w025.tifxyz"


def inventory(root: Path, names) -> list[dict]:
    """The shape the control manifest already locks a reference in: a list of
    path/sha256, with `size_bytes` optional because the locked reference for
    w025 does not carry one."""
    return [{"path": name,
             "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
             "size_bytes": None}
            for name in names]


def wire_for(adapter, root: Path) -> tuple[dict[str, bytes], list[dict]]:
    source, _digest = make_surface(root)
    names = adapter.REQUIRED_SURFACE_FILES
    served = {f"{BASE}/{name}": (source / name).read_bytes() for name in names}
    return served, inventory(source, names)


def serve(adapter, monkeypatch, wire: dict[str, bytes]) -> list[str]:
    asked: list[str] = []

    def fake_get(url: str, **kwargs) -> bytes:
        asked.append(url)
        if url not in wire:
            raise FileNotFoundError(f"404: {url}")
        return wire[url]

    monkeypatch.setattr(adapter, "_http_get", fake_get)
    return asked


def test_the_host_is_never_asked_for_a_manifest(tmp_path, monkeypatch) -> None:
    """The whole point: a public host has no ARTIFACT_SET.json to give."""
    adapter = load_adapter()
    wire, files = wire_for(adapter, tmp_path / "source")
    asked = serve(adapter, monkeypatch, wire)

    adapter.materialize_surface(
        BASE, "d" * 64, tmp_path / "materialized", expected_files=files)

    assert not any(url.endswith("ARTIFACT_SET.json") for url in asked)
    assert set(asked) == set(wire)


def test_the_fetched_bytes_are_the_locked_bytes(tmp_path, monkeypatch) -> None:
    adapter = load_adapter()
    wire, files = wire_for(adapter, tmp_path / "source")
    serve(adapter, monkeypatch, wire)
    destination = tmp_path / "materialized"

    manifest = adapter.materialize_surface(
        BASE, "d" * 64, destination, expected_files=files)

    assert manifest["artifact_sha256"] == "d" * 64
    for name in adapter.REQUIRED_SURFACE_FILES:
        assert (destination / name).read_bytes() == wire[f"{BASE}/{name}"]


def test_a_forged_file_is_caught_by_our_digest(tmp_path, monkeypatch) -> None:
    """Same length, different bytes -- the substitution a size check misses."""
    adapter = load_adapter()
    wire, files = wire_for(adapter, tmp_path / "source")
    wire[f"{BASE}/x.tif"] = b"forged"
    serve(adapter, monkeypatch, wire)

    with pytest.raises(RuntimeError, match="x.tif"):
        adapter.materialize_surface(
            BASE, "d" * 64, tmp_path / "materialized", expected_files=files)


def test_an_inventory_that_omits_a_required_file_is_refused(
    tmp_path, monkeypatch
) -> None:
    """A short inventory would silently narrow what gets verified."""
    adapter = load_adapter()
    wire, files = wire_for(adapter, tmp_path / "source")
    serve(adapter, monkeypatch, wire)
    short = [entry for entry in files if entry["path"] != "z.tif"]

    with pytest.raises(ValueError, match="z.tif"):
        adapter.materialize_surface(
            BASE, "d" * 64, tmp_path / "materialized", expected_files=short)


def test_a_declared_size_is_checked_and_a_missing_one_is_not(
    tmp_path, monkeypatch
) -> None:
    adapter = load_adapter()
    wire, files = wire_for(adapter, tmp_path / "source")
    serve(adapter, monkeypatch, wire)
    for entry in files:
        entry["size_bytes"] = 999_999

    with pytest.raises(RuntimeError, match="size"):
        adapter.materialize_surface(
            BASE, "d" * 64, tmp_path / "materialized", expected_files=files)


def test_our_inventory_overrides_a_manifest_the_source_supplied(
    tmp_path,
) -> None:
    """The hole, closed on the branch that always had it.

    This artifact set is internally consistent and self-asserts the locked
    digest: its manifest lists the hashes of its own forged bytes. Every check
    `_verify_surface` performs passes. Only an inventory from outside the source
    catches it.
    """
    adapter = load_adapter()
    honest, _ = make_surface(tmp_path / "honest")
    files = inventory(honest, adapter.REQUIRED_SURFACE_FILES)

    forged, _ = make_surface(tmp_path / "forged")
    (forged / "x.tif").write_bytes(b"forged")
    rebuilt = {
        name: {"sha256": hashlib.sha256((forged / name).read_bytes()).hexdigest(),
               "size_bytes": (forged / name).stat().st_size}
        for name in adapter.REQUIRED_SURFACE_FILES}
    (forged / "ARTIFACT_SET.json").write_text(
        json.dumps({"schema": "campaignx.segmentation_artifact_set.v1",
                    "files": rebuilt, "artifact_sha256": "d" * 64},
                   sort_keys=True) + "\n", encoding="utf-8")

    # Without the inventory the forgery is accepted -- stated so the test fails
    # loudly if that ever stops being true rather than silently proving nothing.
    accepted = adapter.materialize_surface(
        str(forged), "d" * 64, tmp_path / "unguarded")
    assert accepted["artifact_sha256"] == "d" * 64

    with pytest.raises(RuntimeError, match="x.tif"):
        adapter.materialize_surface(
            str(forged), "d" * 64, tmp_path / "guarded", expected_files=files)


def test_an_s3_mirror_is_checked_like_the_bucket_it_stands_in_for(
    tmp_path, monkeypatch
) -> None:
    """The mirror branch returned early, so it skipped the inventory entirely.

    A convenience copy that is trusted more than the thing it copies is the
    weakest link, and it is the branch most likely to be pointed at a directory
    somebody else can write to.
    """
    adapter = load_adapter()
    honest, _ = make_surface(tmp_path / "honest")
    files = inventory(honest, adapter.REQUIRED_SURFACE_FILES)

    root = tmp_path / "mirror"
    mirrored = root / "bucket" / "surfaces/w1"
    mirrored.parent.mkdir(parents=True)
    forged, _ = make_surface(mirrored)
    (forged / "x.tif").write_bytes(b"forged")
    rebuilt = {
        name: {"sha256": hashlib.sha256((forged / name).read_bytes()).hexdigest(),
               "size_bytes": (forged / name).stat().st_size}
        for name in adapter.REQUIRED_SURFACE_FILES}
    (forged / "ARTIFACT_SET.json").write_text(
        json.dumps({"schema": "campaignx.segmentation_artifact_set.v1",
                    "files": rebuilt, "artifact_sha256": "d" * 64},
                   sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv("HELENA_QC_SURFACE_MIRROR_ROOT", str(root))

    with pytest.raises(RuntimeError, match="x.tif"):
        adapter.materialize_surface(
            "s3://bucket/surfaces/w1", "d" * 64, tmp_path / "materialized",
            expected_files=files)


def test_without_an_inventory_a_published_set_still_carries_its_manifest(
    tmp_path, monkeypatch
) -> None:
    """Helena publishes artifact sets with an ARTIFACT_SET.json over the panel,
    and those keep working: absent an inventory the manifest is still read."""
    adapter = load_adapter()
    source, digest = make_surface(tmp_path / "source")
    names = (*adapter.REQUIRED_SURFACE_FILES, "ARTIFACT_SET.json")
    wire = {f"{BASE}/{name}": (source / name).read_bytes() for name in names}
    asked = serve(adapter, monkeypatch, wire)

    manifest = adapter.materialize_surface(BASE, digest, tmp_path / "materialized")

    assert manifest["artifact_sha256"] == digest
    assert f"{BASE}/ARTIFACT_SET.json" in asked
