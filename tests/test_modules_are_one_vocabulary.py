"""What each phase can be done with, in one shape and one switch.

The platform grew three extension mechanisms and each is right for what it does:
a lane is a program, a profile is a model with its physical scale, a seeder
chooses a point to grow from. What was missing was a single answer to "what can
this phase run, and is it on".

These hold the properties that make the answer trustworthy: every phase reports,
the switch actually stops work rather than only hiding it, and a phase cannot be
switched off entirely and then present itself as ready.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    module.MODULES_PATH = tmp_path / "modules.json"
    monkeypatch.setattr(module, "DSN", "")
    return module


@pytest.fixture
def client(app_module):
    from fastapi.testclient import TestClient
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    session = TestClient(app_module.app)
    assert session.post("/api/session", json={"username": "tester",
                                              "password": "a-long-enough-one"}).status_code == 200
    return session


# --------------------------------------------------------------------------
# One vocabulary
# --------------------------------------------------------------------------

def test_every_phase_answers(client):
    served = client.get("/api/modules").json()
    assert [p["phase"] for p in served["phases"]] == [f"P{n}" for n in range(10)]


def test_the_three_mechanisms_are_named_rather_than_blurred(client, app_module):
    """Reporting a seeder and a lane as the same word would tell somebody
    integrating that one contract exists. Three do, and they are different."""
    served = client.get("/api/modules").json()
    kinds = {m["kind"] for p in served["phases"] for m in p["modules"]}
    assert {"lane", "profile", "backend", "seeder"} <= kinds
    for kind in kinds:
        assert served["kinds"].get(kind), f"{kind} is reported without an explanation"


def test_segmentation_offers_its_three_backends(client):
    """The example the community will look at first."""
    modules = client.get("/api/modules?phase=P1").json()["phases"][0]["modules"]
    backends = {m["id"] for m in modules if m["kind"] == "backend"}
    assert {"vc3d", "scrollfiesta", "thaumato"} <= backends
    assert {m["id"] for m in modules if m["kind"] == "seeder"}


# --------------------------------------------------------------------------
# The switch means something
# --------------------------------------------------------------------------

def test_switching_off_survives_a_restart(client, app_module):
    assert client.post("/api/modules/P1/scrollfiesta",
                       json={"enabled": False}).status_code == 200
    assert app_module.module_disabled("P1", "scrollfiesta")
    assert json.loads(app_module.MODULES_PATH.read_text())["disabled"] == \
        ["P1:scrollfiesta"]


def test_a_switched_off_module_cannot_be_queued(client, app_module):
    """A switch that only hides a module from a form is decoration: the API is
    the interface, and the panel is one client of it."""
    client.post("/api/modules/P4/chunk-gather", json={"enabled": False})
    # A real mission with this scroll selected. Queueing needs one now, and the
    # switch is only reached once the scope checks out -- so without this the
    # test would pass on the mission refusal and never touch the switch.
    client.post("/api/missions", json={
        "mission_id": "modules", "name": "Modules", "scrolls": ["PHerc0826"]})
    refused = client.post("/api/jobs", json={
        "sample_id": "PHerc0826", "phase": "P4", "mission_id": "modules",
        "parameters": {"lane": "chunk-gather"}})
    assert refused.status_code == 409
    assert "switched off" in json.dumps(refused.json())


def test_a_phase_cannot_be_switched_off_entirely(client):
    """It would then report itself ready to run, with nothing to run it."""
    assert client.post("/api/modules/P8/column-atlas",
                       json={"enabled": False}).status_code == 200
    assert client.post("/api/modules/P8/mesh-relations",
                       json={"enabled": False}).status_code == 200
    last = client.post("/api/modules/P8/vc3d-tifxyz-merge",
                       json={"enabled": False})
    assert last.status_code == 409
    assert "last module" in json.dumps(last.json())


def test_an_unknown_module_is_a_404_naming_the_real_ones(client):
    missing = client.post("/api/modules/P4/not-a-lane", json={"enabled": False})
    assert missing.status_code == 404
    assert "vc-render-tifxyz" in json.dumps(missing.json())


# --------------------------------------------------------------------------
# Adding one from Hugging Face
# --------------------------------------------------------------------------

@pytest.fixture
def published(app_module, monkeypatch):
    """One repository, as Hugging Face describes it.

    Registration reads the listing now, so these tests would otherwise reach the
    network and invent repository names at it. Reading it is the point: the
    profile records the checkpoint digest, which the endpoint used to leave null
    while its own docstring said the worker verifies it.
    """
    monkeypatch.setattr(app_module, "hugging_face_model", lambda repo, revision="main", **k: {
        "sha": "8a0f2c1e6b4d",
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "model.safetensors",
             "lfs": {"sha256": "ab" * 32, "size": 151_853_128}},
        ],
    })


def test_a_detector_can_be_added_by_naming_its_repository(client, app_module, published,
                                                          tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PROFILE_DIR", tmp_path / "profiles")
    created = client.post("/api/modules/P5/huggingface", json={
        "repo_id": "someone/a-detector",
        "adapter": "framework/stages/03-ink/scripts/run_ink_timesformer.py",
        "training_pixel_um": 7.91, "frames": 26})
    assert created.status_code == 201, created.text

    written = json.loads((tmp_path / "profiles").glob("*.json").__next__().read_text())
    assert written["source"] == {
        "kind": "huggingface", "repo_id": "someone/a-detector",
        # The commit the branch pointed at, not the branch. A profile is frozen
        # and a branch moves, so recording "main" recorded nothing.
        "revision": "8a0f2c1e6b4d", "requested_revision": "main",
        "file": "model.safetensors"}
    assert written["checkpoint_sha256"] == "ab" * 32, (
        "the profile was written with no digest, so the worker has nothing to "
        "verify the weights against and the docstring's promise is empty"
    )
    assert written["input_contract"]["training_pixel_um"] == 7.91
    # The scale is the thing a reader needs and the platform cannot infer.
    assert written["input_contract"]["physical_resampling_required"] is True


def test_a_model_added_without_limits_says_so(client, app_module, published,
                                              tmp_path, monkeypatch):
    """Silence would be read as "validated". The profile says the opposite,
    because that is what is true of something registered from a form."""
    monkeypatch.setattr(app_module, "PROFILE_DIR", tmp_path / "profiles")
    client.post("/api/modules/P5/huggingface", json={
        "repo_id": "someone/another",
        "adapter": "framework/stages/03-ink/scripts/run_ink.py",
        "training_pixel_um": 2.4, "frames": 62})
    written = json.loads((tmp_path / "profiles").glob("*.json").__next__().read_text())
    assert "validated against a known positive" in written["known_limits"]


def test_an_adapter_nobody_wrote_is_refused_with_the_list(client, app_module, published,
                                                          tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PROFILE_DIR", tmp_path / "profiles")
    refused = client.post("/api/modules/P5/huggingface", json={
        "repo_id": "someone/exotic", "adapter": "run_my_own_thing.py",
        "training_pixel_um": 7.9, "frames": 26})
    assert refused.status_code == 400
    body = refused.json()["detail"]
    assert len(body["adapters"]) >= 4
    assert "command-line contract" in body["why"]


def test_the_repository_must_be_owner_and_name(client, app_module, published,
                                               tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PROFILE_DIR", tmp_path / "profiles")
    refused = client.post("/api/modules/P5/huggingface", json={
        "repo_id": "justaname",
        "adapter": "framework/stages/03-ink/scripts/run_ink.py",
        "training_pixel_um": 7.9, "frames": 26})
    assert refused.status_code == 400


def test_a_repository_that_does_not_exist_never_becomes_a_profile(
        client, app_module, tmp_path, monkeypatch):
    """A typo used to write a profile that failed at run time, on a GPU, later.

    The listing is read at registration, so it fails here instead -- and the
    same read is what supplies the digest.
    """
    monkeypatch.setattr(app_module, "PROFILE_DIR", tmp_path / "profiles")

    def gone(repo, revision="main", **k):
        raise OSError("nothing there")

    monkeypatch.setattr(app_module, "hugging_face_model", gone)
    refused = client.post("/api/modules/P5/huggingface", json={
        "repo_id": "someone/typo",
        "adapter": "framework/stages/03-ink/scripts/run_ink.py",
        "training_pixel_um": 7.9, "frames": 26})
    assert refused.status_code == 502
    assert not list((tmp_path / "profiles").glob("*.json")), (
        "a profile was written for a repository nobody can read"
    )


def test_a_file_the_repository_does_not_have_is_refused_with_what_it_does(
        client, app_module, published, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PROFILE_DIR", tmp_path / "profiles")
    refused = client.post("/api/modules/P5/huggingface", json={
        "repo_id": "someone/a-detector", "checkpoint_file": "weights.safetensors",
        "adapter": "framework/stages/03-ink/scripts/run_ink.py",
        "training_pixel_um": 7.9, "frames": 26})
    assert refused.status_code == 404
    assert "model.safetensors" in refused.json()["detail"]["files"]
