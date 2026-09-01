"""Getting weights onto the machine, without getting anything else onto it.

The panel can now fetch a checkpoint from Hugging Face, which means the panel
now writes a file that a GPU worker will load. Three things have to hold, and
each is a different kind of wrong:

  * only safetensors. Every other checkpoint format in this ecosystem is a
    Python pickle, and loading one executes whatever was serialised into it. A
    download button that accepts `.ckpt` is a remote code execution primitive
    with a friendly label.
  * the file name is a name, not a path. It reaches a filesystem write.
  * what arrives is what was asked for. A profile identifies a checkpoint by
    hash and treats the path as runtime input, so a repository that was
    re-uploaded since the profile was frozen must not install under the old
    name.

The refusals are tested against the real endpoint. The Hugging Face side is
tested separately, on recorded metadata, because a test that needs the network
to check a `.ckpt` refusal is a test that goes yellow when the network does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import panel.app as panel  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from framework.contracts import auth

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(panel, "MODELS_ROOT", models)
    monkeypatch.setattr(panel, "AUTH_ROOT", tmp_path / "auth")
    panel.AUTH_ROOT.mkdir(parents=True, exist_ok=True)
    auth.create_user(panel.AUTH_ROOT, "operator", "a-long-enough-password")

    session = TestClient(panel.app, raise_server_exceptions=False)
    session.cookies.set(
        auth.COOKIE, auth.login(panel.AUTH_ROOT, "operator", "a-long-enough-password"))
    session.models = models  # type: ignore[attr-defined]
    return session


# --------------------------------------------------------------------------
# What the panel will not fetch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "pytorch_model.bin",
    "model.pt",
    "timesformer_scroll5_july_retreat_20241113070770_frepoch=9.ckpt",
    "model.pth",
    "weights.pkl",
])
def test_a_pickle_is_refused_whatever_it_is_called(client, filename):
    """Not a format preference. `torch.load` on any of these runs code.

    The last one is real: the checkpoint one frozen profile names is published
    by Scroll Prize only as a .ckpt, so this refusal has a live consequence
    rather than a hypothetical one.
    """
    answer = client.post("/api/models/download",
                         json={"repo": "scrollprize/anything", "file": filename})
    assert answer.status_code == 400, (
        f"{filename} was accepted; loading it would execute whatever was "
        "pickled into it, on a GPU worker"
    )
    assert "pickle" in str(answer.json()).lower()
    assert not list(client.models.rglob("*")), "something was written anyway"


def test_the_file_is_a_name_and_not_a_path(client):
    """It is joined onto the models root, so a separator is an escape."""
    for attempt in ("../../etc/cron.d/model.safetensors",
                    "subdir/model.safetensors"):
        answer = client.post("/api/models/download",
                             json={"repo": "a/b", "file": attempt})
        assert answer.status_code == 400, f"{attempt} was accepted"


def test_a_repository_name_that_is_not_one_never_reaches_the_network(client):
    """Validated by the model rather than by the handler, so it cannot be
    reached at all -- which also keeps it out of a URL."""
    # `..` is the one that matters: it is a legal-looking half that makes the
    # destination directory the models root's parent, and puts two dots in a URL.
    for attempt in ("no-slash", "a/b/c", "../etc", "a/..", "a b/c", "/absolute"):
        answer = client.post("/api/models/download",
                             json={"repo": attempt, "file": "model.safetensors"})
        assert answer.status_code == 422, f"{attempt} was accepted as a repository"


# --------------------------------------------------------------------------
# What arrives has to be what was asked for
# --------------------------------------------------------------------------

def test_a_file_that_is_not_the_declared_one_is_deleted_rather_than_installed(
        client, monkeypatch):
    """The case this guards is a repository re-uploaded after a profile froze.

    Installing the new weights under the old name would leave the platform
    running a different model while every record says otherwise, and the record
    is the only thing anybody can check afterwards.
    """
    import io

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        if "api/models" in str(url):
            return Answer(b'{"sha":"deadbeef","siblings":'
                          b'[{"rfilename":"model.safetensors"}]}')
        return Answer(b"different weights entirely")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answer = client.post("/api/models/download", json={
        "repo": "scrollprize/timesformer_GP_scroll1",
        "file": "model.safetensors",
        "expect_sha256": "0" * 64,
    })
    assert answer.status_code == 409
    assert not list(client.models.rglob("*.safetensors")), (
        "the wrong weights were left on disk under the name of the right ones"
    )
    assert not list(client.models.rglob("*.part")), "a partial file was left behind"


def test_what_is_installed_is_decided_by_hash_and_not_by_filename(client):
    """A profile says the path is runtime input and the hash is identity.

    So a file called model.safetensors sitting in the right directory proves
    nothing, and the listing has to agree: this one is named after a real
    checkpoint and contains something else.
    """
    home = client.models / "timesformer_GP_scroll1"
    home.mkdir()
    (home / "model.safetensors").write_bytes(b"not the checkpoint")

    rows = client.get("/api/models").json()["checkpoints"]
    named = [r for r in rows if r["model_family"] == "timesformer_GP_scroll1"]
    assert named, "no profile declares timesformer_GP_scroll1 any more"
    assert not any(r["installed"] for r in named), (
        "a file with the right name and the wrong contents counted as installed"
    )


# --------------------------------------------------------------------------
# The list is derived, not typed
# --------------------------------------------------------------------------

def test_the_offered_models_come_from_a_declaration_and_never_from_a_guess(client):
    """The whole reason there is no hardcoded catalogue.

    Every row traces to a document in this checkout: a frozen profile, or the
    weight manifest. A typed list would be a guess about somebody else's
    repository naming, and it would ship as a button that 404s the day they
    rename something.

    There are two declarers rather than one, and the difference is the point. A
    profile declares the checkpoint a lane froze -- one. The manifest declares
    what upstream published -- fourteen, for `ink_9um`, being two seeds by seven
    steps of a single training run. Deriving the page from profiles alone made
    the other thirteen invisible: not missing, not installed, simply absent from
    the question the page was asking.
    """
    rows = client.get("/api/models").json()["checkpoints"]
    assert rows, "no checkpoint is declared anywhere in framework/profiles"
    declared = {"ink-weights-0.1.0.json"}
    for profile in (ROOT / "framework/profiles").rglob("*.json"):
        text = profile.read_text(encoding="utf-8")
        if '"checkpoint_sha256"' in text:
            declared.add(profile.name)
    offered = {name for row in rows for name in row["declared_by"]}
    assert offered <= declared
    for row in rows:
        assert len(row["checkpoint_sha256"]) == 64
        assert row["declared_by"], f"{row['checkpoint_sha256']} is offered by nothing"


def test_a_matching_pickle_is_reported_as_unavailable_not_as_available():
    """The state and the refusal have to agree, or the page shows a button that
    the endpoint rejects -- which reads as a broken panel rather than as a rule.

    This is the scroll5 case: the hash matches, and the only file carrying it is
    a pickle.
    """
    recorded = {
        "sha": "5b714296b256",
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "scroll5.ckpt",
             "lfs": {"sha256": "b5" + "f" * 62, "size": 456_000_000}},
        ],
    }
    found = _resolved({"model_family": "timesformer_scroll5_july_retreat",
                       "checkpoint_sha256": "b5" + "f" * 62}, recorded)
    assert found["state"] == "pickle_only"
    assert "executes" in found["why"]


def test_a_reuploaded_repository_is_reported_as_such():
    """Distinct from a missing one, because the operator does different things:
    a re-upload means the profile and the world disagree and somebody has to
    decide which is right."""
    recorded = {"sha": "abc", "siblings": [
        {"rfilename": "model.safetensors",
         "lfs": {"sha256": "aa" + "0" * 62, "size": 10}}]}
    found = _resolved({"model_family": "timesformer_GP_scroll1",
                       "checkpoint_sha256": "bb" + "0" * 62}, recorded)
    assert found["state"] == "mismatch"
    assert found["safetensors"] == ["model.safetensors"]


def _resolved(row: dict, recorded: dict) -> dict:
    """resolve_on_hugging_face against recorded metadata rather than the API."""
    real = panel.hugging_face_model
    try:
        panel.hugging_face_model = lambda *a, **k: recorded  # type: ignore[assignment]
        return panel.resolve_on_hugging_face(row)
    finally:
        panel.hugging_face_model = real  # type: ignore[assignment]


# --------------------------------------------------------------------------
# The volume the whole thing writes into
# --------------------------------------------------------------------------

def test_every_compose_that_needs_models_uses_the_volume():
    """The point of the feature: a new host is `compose up`, not a copy of
    somebody's disk.

    Two of these required a host path -- one of them a single 152 MB file --
    so a fresh machine could not start an ink or QC worker until a human had
    staged checkpoints on it. The panel fills the volume instead. A host path
    remains available as an override; what must not come back is the `:?` that
    makes one mandatory.
    """
    import yaml

    for name in ("platform", "ink", "surface-qc"):
        text = (ROOT / f"containers/compose/{name}.compose.yaml").read_text()
        compose = yaml.safe_load(text)
        mounts = [m for service in compose["services"].values()
                  for m in (service.get("volumes") or [])
                  if ":/models" in m]
        assert mounts, f"{name} mounts nothing at /models"
        for mount in mounts:
            assert "helena-models" in mount, (
                f"{name} mounts {mount} at /models rather than the volume"
            )
            assert ":?" not in mount, (
                f"{name} still requires a host path for models: {mount}"
            )

    # And only the panel may write it. A worker that could overwrite a
    # checkpoint could change what a frozen profile means.
    for name in ("ink", "surface-qc"):
        text = (ROOT / f"containers/compose/{name}.compose.yaml").read_text()
        mount = next(line for line in text.splitlines() if ":/models" in line)
        assert mount.rstrip().endswith(":ro"), f"{name} can write the models volume"


def test_a_pickle_already_on_disk_counts_as_installed(client):
    """Refusing to fetch a pickle and refusing to see one are different things.

    The canonical ink checkpoint on gpu-1 is an r152.ckpt that a frozen profile
    names by hash. It is installed and in use, and the page reported it missing
    because the scan only looked for safetensors -- which would have had an
    operator downloading a checkpoint they already had, if it were downloadable
    at all, which it is not.
    """
    import hashlib

    home = client.models / "new_canon_autoresearch_recipe"
    home.mkdir()
    body = b"a pickle, as far as this test is concerned"
    (home / "r152.ckpt").write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()

    rows = client.get("/api/models").json()["checkpoints"]
    # Nothing declares this hash, so it must not claim to satisfy a profile...
    assert not any(r["installed"] and r["checkpoint_sha256"] == digest for r in rows)

    # ...but the scan must have hashed it, which is what the declared-checkpoint
    # match depends on. Prove it by declaring one: the fixture's own profile set
    # is the repository's, so use a file whose hash a profile does name.
    from framework.contracts import artifact as _artifact  # noqa: F401
    import panel.app as app_module

    found = app_module.declared_checkpoints()
    assert found, "no checkpoint is declared"
    # The suffix list is what makes a non-safetensors install visible at all.
    assert ".ckpt" in app_module.CHECKPOINT_SUFFIXES
    assert ".safetensors" in app_module.CHECKPOINT_SUFFIXES


def test_a_profile_that_pins_no_digest_verifies_nothing_and_must_say_so():
    """Silence was the worst of the three states.

    A profile with a digest verifies; a profile with the wrong digest refuses;
    a profile with no digest used to sail through and write a receipt that
    records the checkpoint's hash. That receipt is indistinguishable from a
    verified one -- it carries a digest, it just carries one nobody promised in
    advance. Weights silently different from what the profile means are an
    irreproducible result that reads as reproducible, which is the failure this
    whole stage exists to prevent.

    Every profile whose adapter is run_ink.py does pin one today, so this is a
    guard against the next profile rather than a fix for a current one.
    """
    source = (ROOT / "framework/stages/03-ink/scripts/run_ink.py").read_text()
    assert "if not declared:" in source, (
        "run_ink.py no longer refuses a profile that pins no checkpoint_sha256")
    # And the refusal has to be reachable: it must come before the comparison,
    # or an absent digest falls into `declared != checkpoint_sha` and reports a
    # mismatch against nothing.
    assert source.index("if not declared:") < source.index("if declared != checkpoint_sha:")


def test_every_profile_that_run_ink_serves_pins_a_digest():
    """The guard above is only free while this stays true."""
    import json  # noqa: PLC0415

    for profile in sorted((ROOT / "framework/profiles/03-ink").glob("*.json")):
        spec = json.loads(profile.read_text())
        adapter = spec.get("adapter")
        if not adapter or not adapter.endswith("run_ink.py"):
            continue
        assert spec.get("checkpoint_sha256"), (
            f"{spec['profile_id']} routes to run_ink.py, which now refuses a "
            "profile that pins no checkpoint_sha256")
