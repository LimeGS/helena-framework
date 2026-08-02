"""The control plane, in object storage, and the versions a host is asked to run.

Everything this platform produces already outlives the machine that made it: a
surface, a sheet, a layer stack and a probability map are published with a
digest. The record of which of them is certified, which cell was attempted and
what every verdict was lived in one database on one host with no copy anywhere,
and on 2026-07-28 that host filled its disk and PostgreSQL died mid-recovery.

The second half of this file is the other way a deployment loses its footing:
every compose file pointed at `:latest`, so two hosts could run different bytes
under one name -- which is what the registry was introduced to prevent.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "containers/images"))

import backup_to_s3 as backup  # noqa: E402

COMPOSE = sorted((ROOT / "containers/compose").glob("*.compose.yaml"))
VERSION = (ROOT / "VERSION").read_text().strip()


# --------------------------------------------------------------------------
# The backup
# --------------------------------------------------------------------------

def test_a_dump_is_named_so_the_most_recent_one_is_findable():
    """Lexicographic order is time order, which is what makes "the latest
    backup" answerable with a list and no metadata."""
    early = backup.utc_stamp(datetime(2026, 7, 28, 9, 5, 1, tzinfo=timezone.utc))
    late = backup.utc_stamp(datetime(2026, 7, 28, 23, 5, 1, tzinfo=timezone.utc))
    assert early == "20260728T090501Z"
    assert early < late


def test_an_empty_dump_is_not_uploaded_as_a_backup(monkeypatch, tmp_path):
    """A dump written and never opened is a file. pg_restore --list is the
    strongest check available without a second database to restore into, and a
    dump that lists nothing means the upload would be a lie."""
    empty = tmp_path / "empty.dump"
    empty.write_bytes(b"")

    def listing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="; no objects\n", stderr="")

    monkeypatch.setattr(backup.subprocess, "run", listing)
    with pytest.raises(RuntimeError) as refused:
        backup.verify_dump(empty)
    assert "empty" in str(refused.value)


def test_a_dump_that_lists_objects_reports_how_many(monkeypatch, tmp_path):
    def listing(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout="; comment\n1; 2345 TABLE segment_surfaces\n"
                            "2; 2345 TABLE segment_tasks\n", stderr="")

    monkeypatch.setattr(backup.subprocess, "run", listing)
    assert backup.verify_dump(tmp_path / "x.dump") == 2


def test_the_panel_state_travels_too(tmp_path):
    """Losing it loses no artefact. It loses everybody's ability to sign in,
    which this project has already done once, with an rsync --delete."""
    state = tmp_path / "panel-state"
    (state / "auth").mkdir(parents=True)
    (state / "auth" / "USERS.json").write_text("[]")
    archive = backup.archive_directory(state, tmp_path / "state.tgz")
    import tarfile

    with tarfile.open(archive) as opened:
        assert "panel-state/auth/USERS.json" in opened.getnames()


def test_a_round_uploads_the_dump_the_state_and_a_receipt(monkeypatch, tmp_path):
    """The receipt goes up as well, so "what was in that backup" is answerable
    without downloading the backup."""
    uploaded: list[tuple[str, str]] = []
    monkeypatch.setattr(backup, "dump_database",
                        lambda dsn, destination: (destination.write_bytes(b"x"),
                                                  destination)[1])
    monkeypatch.setattr(backup, "verify_dump", lambda path: 42)
    monkeypatch.setattr(backup, "upload",
                        lambda path, target: uploaded.append((path.name, target)))
    state = tmp_path / "state"
    (state / "auth").mkdir(parents=True)
    # Missions and the P0 selections they froze. The compose mounted this
    # read-only for a while and the backup read two things and never it: a
    # mount is not a backup, and the receipt only listed what had been written,
    # so the gap could not be seen from the log either.
    runs = tmp_path / "runs"
    (runs / "mission-a").mkdir(parents=True)
    (runs / "mission-a" / "SELECTION.json").write_text("{}")

    receipt = backup.backup_once(dsn="postgresql://x", prefix="s3://bucket/helena/",
                                 state_dir=state, runs_dir=runs, workspace=tmp_path)
    kinds = [artefact["kind"] for artefact in receipt["artefacts"]]
    assert kinds == ["postgres", "panel_state", "runs"]
    targets = [target for _, target in uploaded]
    assert any(t.startswith("s3://bucket/helena/postgres/") for t in targets)
    assert any(t.startswith("s3://bucket/helena/panel-state/") for t in targets)
    assert any(t.startswith("s3://bucket/helena/runs/") for t in targets), (
        "the runs root was mounted and not uploaded"
    )
    assert any(t.startswith("s3://bucket/helena/receipts/") for t in targets)
    # Every artefact carries its own digest, because a backup whose integrity is
    # assumed is the same as one nobody opened.
    assert all(len(a["sha256"]) == 64 for a in receipt["artefacts"])


def test_the_platform_compose_runs_the_backup():
    import yaml

    compose = yaml.safe_load((ROOT / "containers/compose/platform.compose.yaml").read_text())
    service = compose["services"]["backup"]
    assert "HELENA_BACKUP_S3" in str(service["environment"]), \
        "a backup with no destination is a container that does nothing"
    assert any(str(volume).endswith(":ro") for volume in service["volumes"]), \
        "it reads the panel's accounts and has no business writing them"


# --------------------------------------------------------------------------
# The versions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", COMPOSE, ids=lambda p: p.name)
def test_no_service_floats_on_latest(path):
    """Two hosts running different bytes under one name is what the registry was
    introduced to prevent, and every compose file undid it in its default."""
    floating = [line.strip() for line in path.read_text().splitlines()
                if "image:" in line and ":latest" in line]
    assert not floating, f"{path.name} defaults to a floating tag: {floating}"


@pytest.mark.parametrize("path", COMPOSE, ids=lambda p: p.name)
def test_the_default_version_is_the_one_in_the_version_file(path):
    """A release that bumps VERSION and leaves a compose default behind deploys
    the previous release from a file that says otherwise.

    HELENA_RUNTIME_VERSION is deliberately not this: the frozen runtimes carry
    their own versions, which mean "these exact bytes" and must not move when
    the framework does.
    """
    for default in re.findall(r"HELENA_VERSION:-([0-9]+\.[0-9]+\.[0-9]+)", path.read_text()):
        assert default == VERSION, (
            f"{path.name} falls back to {default} and VERSION says {VERSION}")


def test_the_version_is_semantic():
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", VERSION), VERSION


def test_the_build_writes_down_the_digest_each_version_resolved_to():
    """A tag says which release a host is asked to run; a digest says which
    bytes it got, and only the second survives somebody rebuilding a tag."""
    script = (ROOT / "containers/build-images.sh").read_text()
    assert "images.lock.json" in script
    assert "RepoDigests" in script
    assert 'cat "$context/VERSION"' in script, "the tag must come from VERSION"


def test_a_deployment_with_no_runs_root_still_gets_its_database(monkeypatch, tmp_path):
    """Absent is skipped, not fatal. A deployment that keeps missions elsewhere,
    or has not made one yet, must still be backed up."""
    monkeypatch.setattr(backup, "dump_database",
                        lambda dsn, destination: (destination.write_bytes(b"x"),
                                                  destination)[1])
    monkeypatch.setattr(backup, "verify_dump", lambda path: 1)
    monkeypatch.setattr(backup, "upload", lambda path, target: None)

    receipt = backup.backup_once(dsn="postgresql://x", prefix="s3://b/h/",
                                 state_dir=None, runs_dir=tmp_path / "nope",
                                 workspace=tmp_path)
    assert [a["kind"] for a in receipt["artefacts"]] == ["postgres"]


def test_the_compose_tells_the_backup_where_the_runs_are() -> None:
    """The code reads HELENA_BACKUP_RUNS. Mounting the directory without naming
    it is the state this bug was already in."""
    import yaml

    compose = yaml.safe_load(
        (ROOT / "containers/compose/platform.compose.yaml").read_text())
    environment = compose["services"]["backup"]["environment"]
    assert "HELENA_BACKUP_RUNS" in environment, (
        "the runs root is mounted and the backup is not told where it is"
    )
    mounted = [str(v) for v in compose["services"]["backup"]["volumes"]]
    assert any(":ro" in v and "RUNS" in v for v in mounted), (
        "the runs root is named and not mounted, or mounted writable"
    )
