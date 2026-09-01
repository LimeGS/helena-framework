"""Bringing a surface in from a browser, rather than from a worker.

Two ways into the catalogue already existed and neither serves a person with a
folder on their laptop. `/api/segmentation/import` registers a URI and a digest
-- it never sees the bytes, so somebody must already have published them
somewhere and hashed them by hand. `PUT /api/artifacts/{key}` does take bytes,
but it is a worker-facing blob sink: it takes a gzipped tar a browser cannot
make, checks no file set, computes no digest, and records nothing.

So this is the third door, and the properties below are what make it a door
rather than a hole. A caller reaching it is a signed-in person, not our own
worker with a machine token, which changes what may be trusted: not the
filenames, not the upload id, not the size, and not that the bytes parse.

The identity of what lands must be computed exactly as the finalizer computes
it for a surface this fleet grew. If the two disagree, the same bytes have two
names, and every duplicate check downstream is comparing the wrong things.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

SURFACE = {"x.tif": b"xxxx", "y.tif": b"yyyy", "z.tif": b"zzzz",
           "meta.json": json.dumps({"width": 4, "height": 4}).encode()}


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.ARTIFACTS = tmp_path / "artifacts"
    module.ARTIFACTS.mkdir()
    module.SURFACE_UPLOADS = tmp_path / "uploads"
    module.AUTH_ROOT = tmp_path / "auth"
    return module


@pytest.fixture
def anonymous(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


@pytest.fixture
def client(app_module, anonymous):
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    response = anonymous.post("/api/session",
                              json={"username": "tester", "password": "a-long-enough-one"})
    assert response.status_code == 200, response.text
    return anonymous


def open_upload(client) -> str:
    response = client.post("/api/segmentation/uploads")
    assert response.status_code == 201, response.text
    return response.json()["upload_id"]


def send(client, upload_id: str, name: str, body: bytes):
    return client.put(f"/api/segmentation/uploads/{upload_id}/{name}", content=body)


# -- the door opens ------------------------------------------------------


def test_an_upload_is_opened_named_and_filled(client) -> None:
    upload_id = open_upload(client)
    for name, body in SURFACE.items():
        assert send(client, upload_id, name, body).status_code == 201

    state = client.get(f"/api/segmentation/uploads/{upload_id}").json()
    assert set(state["received"]) == set(SURFACE)
    assert state["complete"] is True


# A traversal is normalised out of the URL before routing, so it arrives at
# some other route or at none -- 405 and 404 rather than 400. All three are the
# same outcome and the one that matters: nothing was written. Accepting only
# 400 would make this test assert which layer refused rather than that one did.
REFUSED = (400, 404, 405)


def test_an_upload_id_is_minted_here_not_chosen_by_the_caller(client) -> None:
    """A caller who picks the id picks a directory name on our disk."""
    assert send(client, "../../etc", "x.tif", b"xxxx").status_code in REFUSED
    assert send(client, "not-a-real-upload", "x.tif", b"xxxx").status_code == 404


def test_a_filename_outside_the_surface_is_refused(client, app_module) -> None:
    upload_id = open_upload(client)
    for name in ("../escape.tif", "x.tif/../../y", "notes.txt", "x.TIF"):
        assert send(client, upload_id, name, b"nope").status_code in REFUSED, name

    # The refusal has to mean nothing landed, not merely that a status was
    # returned: a sanitised name is still a written file under some other name.
    assert not list(app_module.SURFACE_UPLOADS.rglob("*escape*"))
    assert not list(app_module.SURFACE_UPLOADS.rglob("*notes*"))
    assert client.get(f"/api/segmentation/uploads/{upload_id}").json()["received"] == {}


def test_a_file_over_the_cap_is_refused_while_it_arrives(client, app_module) -> None:
    """Refused during the stream, not after a disk has filled."""
    upload_id = open_upload(client)
    oversize = b"\0" * (app_module.MAX_SURFACE_FILE_BYTES + 1)
    assert send(client, upload_id, "x.tif", oversize).status_code == 413


def test_an_empty_body_is_not_a_file(client) -> None:
    upload_id = open_upload(client)
    assert send(client, upload_id, "x.tif", b"").status_code == 400


# -- what commit is allowed to do ----------------------------------------


def commit(client, upload_id, **overrides):
    body = {"sample_id": "PHerc1447", "owner": "tester"}
    body.update(overrides)
    return client.post(f"/api/segmentation/uploads/{upload_id}/commit", json=body)


def test_half_a_surface_is_not_committed(client) -> None:
    """x, y, z and meta or nothing: a surface missing a channel reads as one
    until something tries to use it."""
    upload_id = open_upload(client)
    send(client, upload_id, "x.tif", SURFACE["x.tif"])
    send(client, upload_id, "meta.json", SURFACE["meta.json"])

    refused = commit(client, upload_id)
    assert refused.status_code == 400
    assert "y.tif" in refused.text and "z.tif" in refused.text


def test_a_meta_that_is_not_json_is_refused(client) -> None:
    upload_id = open_upload(client)
    for name, body in SURFACE.items():
        send(client, upload_id, name, b"<html>404</html>" if name == "meta.json" else body)

    refused = commit(client, upload_id)
    assert refused.status_code == 400
    assert "meta.json" in refused.text


def test_the_bounds_are_measured_here_not_typed_by_the_uploader(client) -> None:
    """/api/segmentation/import takes a bbox from its caller because it never
    sees the bytes. These bytes are on our disk, so a typed bounding box would
    be a restated measurement -- and the restatement is what would be recorded.

    The measurement is the finalizer's, not a min and a max: VC3D writes -1 for
    an invalid coordinate, and a bbox that counts those means something
    different from every other bbox in the table.
    """
    import inspect

    import panel.app as module

    fields = module.UploadCommitRequest.model_fields
    assert "bbox_xyz" not in fields and "area_cm2" not in fields, (
        "commit still accepts bounds from its caller")

    body = inspect.getsource(module.api_surface_upload_commit)
    assert "measure_uploaded_surface" in body
    assert "inspect_tifxyz" in inspect.getsource(module.measure_uploaded_surface), (
        "the bounds are measured by a second implementation of the finalizer's rule")


def test_a_directory_that_is_not_a_tifxyz_never_reaches_the_artifact_volume(
        client, app_module, monkeypatch) -> None:
    """Once a directory is on the artifact volume nothing that looks at it can
    tell it from a published surface, so it must fail before it moves."""
    monkeypatch.setattr(app_module, "DSN", "postgresql://unused")
    monkeypatch.setattr(app_module, "sample_voxel_size_um", lambda sample: 7.91)

    upload_id = open_upload(client)
    for name, body in SURFACE.items():
        send(client, upload_id, name, body)      # "xxxx" is not a TIFF

    refused = commit(client, upload_id)
    assert refused.status_code == 400, refused.text
    assert "TIFXYZ" in refused.text
    assert not list((app_module.ARTIFACTS / "surfaces").rglob("*.tifxyz")), (
        "a directory that does not read as a surface reached the artifact volume")


def test_a_real_tifxyz_is_measured_the_way_the_finalizer_measures_it(
        tmp_path, app_module) -> None:
    """The same bytes, the same bounds. A surface uploaded here and the same
    surface grown here must be comparable, and the bbox is what the spatial
    duplicate index compares."""
    import numpy as np
    import tifffile

    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.finalizer import inspect_tifxyz

    directory = tmp_path / "surface"
    directory.mkdir()
    grid = np.arange(16, dtype=np.float32).reshape(4, 4)
    for axis, offset in (("x", 100.0), ("y", 200.0), ("z", 300.0)):
        plane = grid + offset
        plane[0, 0] = -1.0                       # the invalid-coordinate sentinel
        tifffile.imwrite(directory / f"{axis}.tif", plane)
    (directory / "meta.json").write_text("{}")

    measured = app_module.measure_uploaded_surface(directory, 7.91)
    expected = inspect_tifxyz(directory, 7.91)

    assert measured["bbox_xyz"] == expected["bbox_xyz"]
    assert measured["area_cm2"] == expected["area_cm2"]
    # And the sentinel is excluded, or the whole reuse was pointless.
    assert measured["bbox_xyz"][0][0] > 0


# -- identity ------------------------------------------------------------


def test_the_digest_is_the_one_the_finalizer_would_have_computed(tmp_path, app_module) -> None:
    """The same bytes must get the same name whether they were grown here or
    carried in. Two derivations would mean two identities for one surface, and
    every duplicate check downstream compares digests."""
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import artifact_manifest, content_sha256

    directory = tmp_path / "surface"
    directory.mkdir()
    for name, body in SURFACE.items():
        (directory / name).write_bytes(body)

    expected = content_sha256(artifact_manifest(
        directory, ("x.tif", "y.tif", "z.tif", "meta.json")))

    assert app_module.uploaded_surface_identity(directory)["artifact_sha256"] == expected


def test_an_uploaded_surface_carries_its_own_inventory(tmp_path, app_module) -> None:
    """Its bytes are in a store this fleet runs, so the per-file digests are
    measured here rather than taken from a stranger's manifest."""
    directory = tmp_path / "surface"
    directory.mkdir()
    for name, body in SURFACE.items():
        (directory / name).write_bytes(body)

    identity = app_module.uploaded_surface_identity(directory)
    inventory = {entry["path"]: entry for entry in identity["artifacts"]}

    assert set(inventory) == set(SURFACE)
    assert inventory["x.tif"]["size_bytes"] == len(SURFACE["x.tif"])
    assert len(inventory["x.tif"]["sha256"]) == 64


def test_an_upload_is_never_grown_here(client) -> None:
    """Origin is what the whole catalogue splits on. A surface this fleet did
    not grow must not be counted as its output, however it arrived."""
    source = (ROOT / "panel/app.py").read_text()
    handler = source[source.index('/api/segmentation/uploads/{upload_id}/commit'):]
    handler = handler[:handler.index("\n@app.")]
    said, _, done = handler.partition('"""')      # past the opening quotes
    body = done.partition('"""')[2]               # and past the closing ones
    assert "IMPORTED, never GROWN_HERE" in said + done, "the intent is not stated"

    assert "GROWN_HERE" not in body, (
        "the commit path names GROWN_HERE outside its docstring")
    assert "ImportRequest(" in body, (
        "commit does not go through the import path, so what importing means "
        "now has a second definition here")


def test_an_upload_cannot_be_opened_where_the_bytes_would_have_nowhere_to_go(
        anonymous, app_module, monkeypatch, tmp_path) -> None:
    """The artifact volume is a mount. A deployment missing it should say so.

    Without this the mkdir raises, FastAPI turns that into a 500 with a
    plain-text body, and the person reading it learns that something went wrong
    but not that a volume is not mounted -- on the one endpoint whose whole job
    is to put bytes on that volume.
    """
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    anonymous.post("/api/session",
                   json={"username": "tester", "password": "a-long-enough-one"})
    unwritable = tmp_path / "not-mounted"
    unwritable.write_text("this is a file, so nothing can be made under it")
    monkeypatch.setattr(app_module, "SURFACE_UPLOADS", unwritable / "_uploads")

    refused = anonymous.post("/api/segmentation/uploads")

    assert refused.status_code == 503, refused.text
    assert refused.headers["content-type"].startswith("application/json")
    assert "artifact" in refused.text.lower()


def test_a_forgotten_upload_is_swept_rather_than_kept_forever(
        client, app_module) -> None:
    """Not every upload is abandoned on purpose.

    A browser closed mid-transfer leaves whole files on the volume that is this
    deployment's copy of record, and nothing will ever finish them. The tidy
    exit in the uploader covers the failures it sees; this covers the ones it
    does not, and it runs where a new upload is opened so there is no timer to
    forget to start.
    """
    stale = open_upload(client)
    send(client, stale, "x.tif", SURFACE["x.tif"])
    fresh = open_upload(client)

    old = time.time() - app_module.UPLOAD_SWEEP_SECONDS - 60
    os.utime(app_module.SURFACE_UPLOADS / stale, (old, old))

    open_upload(client)                                   # the sweep runs here

    assert not (app_module.SURFACE_UPLOADS / stale).exists(), "the stale upload survived"
    assert (app_module.SURFACE_UPLOADS / fresh).exists(), (
        "the sweep took an upload somebody is still filling")


def test_an_abandoned_upload_leaves_nothing_behind(client, app_module) -> None:
    upload_id = open_upload(client)
    send(client, upload_id, "x.tif", SURFACE["x.tif"])

    assert client.delete(f"/api/segmentation/uploads/{upload_id}").status_code == 200
    assert client.get(f"/api/segmentation/uploads/{upload_id}").status_code == 404
    assert not list(app_module.SURFACE_UPLOADS.glob(f"{upload_id}*")), (
        "the bytes of an abandoned upload are still on disk")


def test_a_real_surface_goes_all_the_way_in(client, app_module, tmp_path,
                                            monkeypatch) -> None:
    """Open, send, commit -- with bytes that are actually a TIFXYZ.

    The tests above each hold one piece of this still. This is the piece none of
    them cover: that the pieces compose. Everything api_import is handed here is
    derived from the bytes that arrived, so a step that quietly dropped one --
    the inventory, the measured bounds, the move onto the artifact volume --
    would leave every unit test above green.

    api_import itself is stood in for rather than stubbed out at the database:
    what it does with a surface is its own tests' subject, and what matters here
    is what it is given.
    """
    import numpy as np
    import tifffile

    handed: dict = {}

    def record(request, http):
        handed["request"] = request
        return app_module.JSONResponse({"inserted": 1}, status_code=201)

    monkeypatch.setattr(app_module, "api_import", record)
    monkeypatch.setattr(app_module, "DSN", "postgresql://unused")
    monkeypatch.setattr(app_module, "sample_voxel_size_um", lambda sample: 7.91)

    source = tmp_path / "w025.tifxyz"
    source.mkdir()
    # A sheet, not a line. x and y have to vary along different axes of the
    # grid or every triangle is degenerate and the measured area is a truthful
    # zero -- which would say nothing about whether the area was measured.
    rows, columns = np.meshgrid(np.arange(6.0), np.arange(6.0), indexing="ij")
    planes = {"x": 1000.0 + 3.0 * columns,
              "y": 2000.0 + 3.0 * rows,
              "z": 3000.0 + 0.5 * rows}
    for axis, plane in planes.items():
        plane = plane.astype(np.float32)
        plane[0, 0] = -1.0                       # the invalid-coordinate sentinel
        tifffile.imwrite(source / f"{axis}.tif", plane)
    (source / "meta.json").write_text(json.dumps({"width": 6, "height": 6}))

    upload_id = open_upload(client)
    for path in sorted(source.iterdir()):
        assert send(client, upload_id, path.name, path.read_bytes()).status_code == 201

    assert commit(client, upload_id).status_code == 201

    offered = handed["request"].surfaces[0]
    assert handed["request"].source_catalog == "panel-upload"

    # Measured, not declared: the sentinel is out and the bounds are the real
    # extent of the sheet.
    low, high = offered["bbox_xyz"]
    assert low[0] == 1000.0 and high[0] == 1015.0
    assert offered["area_cm2"] > 0

    # The inventory is this panel's own measurement of what it received.
    inventory = {entry["path"]: entry for entry in offered["artifacts"]}
    assert set(inventory) == {"x.tif", "y.tif", "z.tif", "meta.json"}
    assert inventory["meta.json"]["size_bytes"] == len(
        json.dumps({"width": 6, "height": 6}))

    # And the bytes are on the artifact volume, at a key derived from what they
    # are rather than from which upload happened to carry them.
    landed = Path(offered["artifact_uri"])
    assert landed.is_dir() and landed.name.startswith(offered["artifact_sha256"][:32])
    assert sorted(p.name for p in landed.iterdir()) == [
        "ARTIFACT_SET.json", "meta.json", "x.tif", "y.tif", "z.tif"]

    # The manifest a worker opens this surface through. Without it the fleet's
    # fetcher raises before measuring anything, and P2 comes back
    # GEOMETRY_UNMEASURED / ARTIFACT_UNAVAILABLE for a surface that is complete
    # and hashed and simply cannot say so.
    manifest = json.loads((landed / "ARTIFACT_SET.json").read_text())
    assert manifest["schema"] == "campaignx.segmentation_artifact_set.v1"
    assert manifest["artifact_sha256"] == offered["artifact_sha256"]
    assert set(manifest["files"]) == {"x.tif", "y.tif", "z.tif", "meta.json"}
    for name, entry in manifest["files"].items():
        assert entry["size_bytes"] == (landed / name).stat().st_size
        assert entry["sha256"] == inventory[name]["sha256"]
    # The manifest is not part of what it describes: the digest this surface is
    # named by is over the files, and hashing the manifest into itself is not a
    # thing that can be recomputed.
    assert "ARTIFACT_SET.json" not in manifest["files"]

    # The staging directory is gone, not copied.
    assert not (app_module.SURFACE_UPLOADS / upload_id).exists()
