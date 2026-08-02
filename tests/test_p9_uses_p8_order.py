"""P9 reads the measured radial order produced by its named P8 job."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "framework/stages/03-ink/fleet"
sys.path.insert(0, str(FLEET))

import ink_worker  # noqa: E402


def plates_module():
    path = ROOT / "framework/vendored/pherc0139-column-atlas-gh/scripts/make_plates.py"
    spec = importlib.util.spec_from_file_location("helena_make_plates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plate_order_is_derived_from_radius_not_a_literal(tmp_path):
    radial = tmp_path / "wrap_radial.json"
    radial.write_text(json.dumps({"segments": {
        "surface-w023_body": {"r_mean": 10.0},
        "surface-w059_body": {"r_mean": 30.0},
        "surface-title_control": {"r_mean": 2.0},
    }}))

    assert plates_module().reading_from_radial(radial) == ["w059", "w023", "title"]


def test_duplicate_wraps_are_refused_instead_of_silently_overwritten(tmp_path):
    radial = tmp_path / "wrap_radial.json"
    radial.write_text(json.dumps({"segments": {
        "surface-w059_a": {"r_mean": 30.0},
        "surface-w059_b": {"r_mean": 29.0},
    }}))

    with pytest.raises(RuntimeError, match="duplicate wrap"):
        plates_module().reading_from_radial(radial)


class Store:
    def __init__(self, upstream):
        self.upstream = upstream

    def job(self, job_id):
        return self.upstream


def test_worker_resolves_the_successful_p8_order_file(tmp_path):
    order = tmp_path / "wrap_radial.json"
    order.write_text(json.dumps({"segments": {"surface-w059_x": {"r_mean": 30}}}))
    upstream = {"job_id": "p8-order", "phase": "P8", "state": "succeeded",
                "parameters": {"lane": "column-atlas", "out_path": str(order)},
                "result": {}}

    resolved = ink_worker.resolve_wrap_order(
        Store(upstream), {"sample_id": "PHerc0139",
                          "parameters": {"ordering_of": "p8-order"}})

    assert resolved == str(order)


def test_worker_refuses_a_merge_job_as_a_plate_order(tmp_path):
    upstream = {"job_id": "p8-merge", "phase": "P8", "state": "succeeded",
                "parameters": {"lane": "vc3d-tifxyz-merge"}, "result": {}}

    with pytest.raises(RuntimeError, match="published no radial order"):
        ink_worker.resolve_wrap_order(
            Store(upstream), {"sample_id": "PHerc0139",
                              "parameters": {"ordering_of": "p8-merge"}})


def test_plate_builder_refuses_a_measured_wrap_without_an_official_map(
    tmp_path, monkeypatch,
):
    radial = tmp_path / "wrap_radial.json"
    radial.write_text(json.dumps({"segments": {
        "surface-w059_body": {"r_mean": 30.0},
    }}))
    module = plates_module()

    def fake_ls(prefix, *, delimiter="/"):
        if prefix.endswith("/segments/"):
            return (["PHerc0139/segments/surface-w059_body/"], [])
        return ([], [])

    monkeypatch.setattr(module, "ls", fake_ls)
    monkeypatch.setattr(sys, "argv", [
        "make_plates.py", "--order", str(radial),
        "--out", str(tmp_path / "plates"),
        "--work", str(tmp_path / "maps"),
    ])

    with pytest.raises(RuntimeError, match="no official ds8 map"):
        module.main()


def test_plate_builder_writes_a_content_addressed_manifest(tmp_path, monkeypatch):
    radial = tmp_path / "wrap_radial.json"
    radial.write_text(json.dumps({"segments": {
        "surface-w059_body": {"r_mean": 30.0},
    }}))
    module = plates_module()

    def fake_ls(prefix, *, delimiter="/"):
        if prefix.endswith("/segments/"):
            return (["PHerc0139/segments/surface-w059_body/"], [])
        return ([], [("PHerc0139/maps/w059-ds8.jpg", 123)])

    def fake_download(_source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (20, 10), color=127).save(destination)

    output = tmp_path / "plates"
    monkeypatch.setattr(module, "ls", fake_ls)
    monkeypatch.setattr(module, "download", fake_download)
    monkeypatch.setattr(sys, "argv", [
        "make_plates.py", "--order", str(radial),
        "--out", str(output), "--work", str(tmp_path / "maps"),
    ])

    module.main()

    manifest = json.loads((output / "PLATE_MANIFEST.json").read_text())
    plate = output / "01_w059.png"
    assert manifest["schema"] == "campaignx.p9_plate_set.v1"
    assert manifest["status"] == "PASS"
    assert manifest["plate_count"] == 1
    assert manifest["ordering_sha256"] == hashlib.sha256(radial.read_bytes()).hexdigest()
    assert manifest["plates"][0]["sha256"] == hashlib.sha256(plate.read_bytes()).hexdigest()


def test_worker_verifies_the_complete_hashed_plate_inventory(tmp_path):
    order = tmp_path / "wrap_radial.json"
    order.write_text(json.dumps({"segments": {
        "surface-w059_body": {"r_mean": 30.0},
    }}))
    plates = tmp_path / "plates"
    plates.mkdir()
    plate = plates / "01_w059.png"
    Image.new("L", (20, 10), color=127).save(plate)
    digest = hashlib.sha256(plate.read_bytes()).hexdigest()
    manifest = {
        "schema": "campaignx.p9_plate_set.v1",
        "status": "PASS",
        "sample_id": "PHerc0139",
        "ordering_sha256": hashlib.sha256(order.read_bytes()).hexdigest(),
        "plate_count": 1,
        "plates": [{
            "file": plate.name,
            "wrap": "w059",
            "sha256": digest,
            "bytes": plate.stat().st_size,
            "width": 20,
            "height": 10,
        }],
    }
    (plates / "PLATE_MANIFEST.json").write_text(json.dumps(manifest))
    job = {"sample_id": "PHerc0139", "parameters": {
        "out_dir": str(plates), "order_path": str(order),
    }}

    verified = ink_worker.verify_plate_set(job)

    assert verified["plate_count"] == 1
    assert verified["bytes"] == plate.stat().st_size
    assert verified["plates"][0]["sha256"] == digest


def test_worker_refuses_a_manifest_whose_plate_is_missing(tmp_path):
    order = tmp_path / "wrap_radial.json"
    order.write_text(json.dumps({"segments": {
        "surface-w059_body": {"r_mean": 30.0},
    }}))
    plates = tmp_path / "plates"
    plates.mkdir()
    (plates / "PLATE_MANIFEST.json").write_text(json.dumps({
        "schema": "campaignx.p9_plate_set.v1",
        "status": "PASS",
        "sample_id": "PHerc0139",
        "ordering_sha256": hashlib.sha256(order.read_bytes()).hexdigest(),
        "plate_count": 1,
        "plates": [{
            "file": "01_w059.png", "wrap": "w059", "sha256": "0" * 64,
            "bytes": 1, "width": 1, "height": 1,
        }],
    }))

    with pytest.raises(RuntimeError, match="missing plate"):
        ink_worker.verify_plate_set({
            "sample_id": "PHerc0139",
            "parameters": {"out_dir": str(plates), "order_path": str(order)},
        })
