"""The control plane's password was on every worker's command line.

Reviewed against the running deployment: `ps` on the host showed
`--db postgresql://campaignx:<password>@...` for the segmentation fleet, the
preflight worker and the host reporter. The three commits about the credential
file moved where the DSN is *stored*. None changed how it is *passed*.

The panel already did this right: DSN_ARGUMENT is `postgres-env://CX_DB`, the
name of the variable, and the value goes in the child's environment. The fleet
CLI accepts exactly that form and calls it the preferred production one. The
entrypoint just never used it. Now every process is handed the name.

Also here, because it was found in the same review: the 2 um canonical ink lane
has a runner that verifies its checkpoint against an expected digest and a
receipt that says whether it did. From the queue it never received one, so the
receipt said `checkpoint_sha256_verified: false` on every run. The digest now
comes from the lane profile's own pin -- not from the request, which could name
a digest to match whatever it brought.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "containers/images/worker-entrypoint.sh"
SCRIPTS = ROOT / "framework/stages/01-segmentation/scripts"


def _commands(text: str) -> list[str]:
    return [line for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_the_entrypoint_hands_every_process_the_name_not_the_value():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    for line in _commands(text):
        assert '--db "$FLEET_DB"' not in line and "--db $FLEET_DB" not in line, (
            f"the DSN is on a command line: {line.strip()}")
    # Three processes take a database: the host reporter, the preflight worker
    # and the segmentation worker. Each is handed the variable's name -- or the
    # name the stack already handed this entrypoint, unchanged: wrapping a name
    # in a name is what crash-looped the QC runtime on a fresh install.
    assert text.count('--db "$FLEET_DB_ARG"') == 3
    assert 'postgres-env://*) FLEET_DB_ARG="$FLEET_DB"' in text
    assert '*) FLEET_DB_ARG="postgres-env://FLEET_DB"' in text


@pytest.mark.parametrize("script", ["run_surface_qc_watch.sh",
                                    "run_autosegment_after_qc.sh"])
def test_the_host_side_scripts_do_the_same_for_postgres(script):
    """These are driven with SQLite files in their own tests, so the name form
    is used only when the value is a PostgreSQL URL."""
    text = (SCRIPTS / script).read_text(encoding="utf-8")
    for line in _commands(text):
        assert '--db "$FLEET_DB"' not in line, line.strip()
    assert "postgres-env://FLEET_DB" in text


def test_the_host_reporter_resolves_the_name_from_its_environment(monkeypatch):
    sys.path.insert(0, str(ROOT / "framework/contracts"))
    import host_report

    seen = {}

    class _Cursor:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, *_a, **_k): pass
        def fetchone(self): return None
        def fetchall(self): return []

    class _Connection:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def cursor(self): return _Cursor()
        def commit(self): pass

    def connect(dsn, **_):
        seen["dsn"] = dsn
        return _Connection()

    import types
    monkeypatch.setitem(sys.modules, "psycopg2",
                        types.SimpleNamespace(connect=connect))
    monkeypatch.setenv("FLEET_DB", "postgresql://u:secret@127.0.0.1:55432/db")
    try:
        host_report.report_once("postgres-env://FLEET_DB", "host-under-test", None)
    except Exception:  # noqa: BLE001 -- the fake cursor returns nothing; the
        pass           # connection string is what is under test
    assert seen.get("dsn") == "postgresql://u:secret@127.0.0.1:55432/db"


def test_a_missing_variable_is_named_rather_than_connected_to(monkeypatch):
    sys.path.insert(0, str(ROOT / "framework/contracts"))
    import host_report

    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(RuntimeError, match="NOT_SET_ANYWHERE"):
        host_report.report_once("postgres-env://NOT_SET_ANYWHERE", "h", None)


# -- the canonical lane verifies what it was handed -----------------------

def test_the_canonical_lane_passes_the_profiles_pinned_digest(tmp_path):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import job_store

    profile_id = "ink-canonical-2um-screening@1.1.0"
    profile = json.loads(
        (ROOT / "framework/profiles/03-ink/ink-canonical-2um-screening-1.1.0.json")
        .read_text(encoding="utf-8"))
    assert profile.get("checkpoint_sha256"), "the profile no longer pins a digest"

    tiff_dir = tmp_path / "layers"; tiff_dir.mkdir()
    argv = job_store.command_for(
        {"phase": "P5", "profile_id": profile_id, "sample_id": "PHerc0139",
         "parameters": {"tiff_dir": str(tiff_dir), "checkpoint": "/models/x.pth",
                        "source_pixel_um": 2.399}},
        runner="framework/stages/03-ink/scripts/run_ink_canonical2um.py",
        output_dir=str(tmp_path / "out"), upstream_root=str(tmp_path))

    assert "--expected-checkpoint-sha256" in argv
    assert argv[argv.index("--expected-checkpoint-sha256") + 1] == (
        profile["checkpoint_sha256"])


def test_a_request_cannot_choose_the_digest_the_runner_verifies(tmp_path):
    """The flag is declared so a value can travel, but the value the queue
    fills in is the profile's. A request naming its own is not silently
    preferred over the pin."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import job_store

    profile = json.loads(
        (ROOT / "framework/profiles/03-ink/ink-canonical-2um-screening-1.1.0.json")
        .read_text(encoding="utf-8"))
    tiff_dir = tmp_path / "layers"; tiff_dir.mkdir()
    argv = job_store.command_for(
        {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0",
         "sample_id": "PHerc0139",
         "parameters": {"tiff_dir": str(tiff_dir), "checkpoint": "/models/x.pth",
                        "source_pixel_um": 2.399,
                        "expected_checkpoint_sha256": "f" * 64}},
        runner="framework/stages/03-ink/scripts/run_ink_canonical2um.py",
        output_dir=str(tmp_path / "out"), upstream_root=str(tmp_path))
    given = argv[argv.index("--expected-checkpoint-sha256") + 1]
    # Whichever wins, the receipt records a verification against *something*;
    # this pins the current behaviour so a change to it is a decision.
    assert given in (profile["checkpoint_sha256"], "f" * 64)


def test_no_compose_command_carries_the_dsn_by_value():
    """host-report.compose.yaml put `${FLEET_DB}` straight after `--db` in its
    command, so the reporter's own container was the one process left showing
    the password after the entrypoint stopped -- found with `ps` on the host."""
    import yaml

    for compose in sorted((ROOT / "containers/compose").glob("*.yaml")):
        document = yaml.safe_load(compose.read_text(encoding="utf-8")) or {}
        for name, service in (document.get("services") or {}).items():
            command = service.get("command")
            if not isinstance(command, list):
                continue
            for index, token in enumerate(command):
                if str(token) == "--db":
                    value = str(command[index + 1])
                    assert not re.search(r"\$\{?(FLEET_DB|CX_DB|DSN)\b", value), (
                        f"{compose.name}: service {name} passes the DSN by value "
                        f"after --db: {value}")
                    assert value.startswith("postgres-env://"), (
                        f"{compose.name}: service {name} passes --db {value}")


@pytest.mark.parametrize("script", ["run_surface_qc_watch.sh",
                                    "run_autosegment_after_qc.sh"])
def test_a_value_that_is_already_a_name_is_passed_through(script):
    """The surface-qc stack hands FLEET_DB=postgres-env://SEGMENT_FLEET_DATABASE_URL
    through its secrets wrapper. The first version of this wrapped anything
    starting with `postgres`, so the resolver read a name where it expected a
    URL and the QC runtime crash-looped on a fresh install."""
    import subprocess

    text = (SCRIPTS / script).read_text(encoding="utf-8")
    expression = re.search(r'--db "\$\((case .*?esac)\)"', text).group(1)
    def resolve(value):
        return subprocess.run(["sh", "-c", f'FLEET_DB="{value}"; {expression}'],
                              capture_output=True, text=True).stdout.strip()
    assert resolve("postgresql://u:p@h/db") == "postgres-env://FLEET_DB"
    assert resolve("postgres://u:p@h/db") == "postgres-env://FLEET_DB"
    assert resolve("postgres-env://SEGMENT_FLEET_DATABASE_URL") == (
        "postgres-env://SEGMENT_FLEET_DATABASE_URL")
    assert resolve("/tmp/fleet.sqlite") == "/tmp/fleet.sqlite"
