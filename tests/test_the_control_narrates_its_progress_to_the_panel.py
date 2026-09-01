"""A persistent channel for what a control run is doing, separate from evidence.

The receipt is content-addressed, write-once, and re-derived by the server
before it is trusted -- exactly right for the thing a campaign's readiness gate
reads. Progress is none of that: it is a mutable narration of a run in flight,
nobody gates on it, and a corrupted or half-written line in it should never be
able to make evidence/ unreadable. So it lives beside evidence/, in its own
directory, and nothing that reads evidence/ needs to know it exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/harness"))

pytest.importorskip("fastapi")

SCHEMA = "campaignx.first_letters_control_progress_event.v1"


def _event(**changes) -> dict:
    base = {
        "schema": SCHEMA,
        "run_id": "run-a",
        "mission_id": "dev-control",
        "event": "boundary_started",
        "boundary": "P0",
        "at_utc": "2026-08-19T12:00:00+00:00",
    }
    base.update(changes)
    return base


def test_posting_an_event_appends_it_to_the_run_s_own_file(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    stamped = panel_app._append_first_letters_control_progress_event(
        "dev-control", _event())

    assert stamped["event"] == "boundary_started"
    assert "received_at_utc" in stamped
    written = (tmp_path / "control-progress" / "run-a.jsonl").read_text()
    assert json.loads(written.strip())["boundary"] == "P0"


def test_progress_never_lands_inside_evidence(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    panel_app._append_first_letters_control_progress_event("dev-control", _event())

    progress_dir = tmp_path / "control-progress"
    evidence_dir = tmp_path / "evidence"
    assert progress_dir.is_dir()
    assert not str(progress_dir).startswith(str(evidence_dir) + "/")
    assert not str(evidence_dir).startswith(str(progress_dir) + "/")


def test_two_events_in_the_same_run_append_in_order(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    panel_app._append_first_letters_control_progress_event(
        "dev-control", _event(event="run_started"))
    panel_app._append_first_letters_control_progress_event(
        "dev-control", _event(event="boundary_started", boundary="P1"))

    events = panel_app._read_first_letters_control_progress(
        "dev-control", run_id="run-a")
    assert [row["event"] for row in events] == ["run_started", "boundary_started"]
    assert events[1]["boundary"] == "P1"


def test_reading_without_a_run_id_returns_the_latest_run(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    # Genuinely different started_at_utc values -- a tie here would leave
    # "latest" decided by directory order, which is exactly the bug
    # test_a_same_second_tie_breaks_on_run_id_not_on_read_order covers.
    panel_app._append_first_letters_control_progress_event(
        "dev-control", _event(run_id="run-old", event="run_finished",
                              at_utc="2026-08-19T11:00:00+00:00"))
    panel_app._append_first_letters_control_progress_event(
        "dev-control", _event(run_id="run-new", event="run_started",
                              at_utc="2026-08-19T12:00:00+00:00"))

    latest = panel_app._latest_first_letters_control_progress("dev-control")
    assert latest["run_id"] == "run-new"
    assert latest["events"][0]["event"] == "run_started"
    run_ids = {row["run_id"] for row in latest["runs"]}
    assert run_ids == {"run-old", "run-new"}


def test_a_same_second_tie_breaks_on_run_id_not_on_read_order():
    """Two runs that started in the same second (started_at_utc's own
    resolution) must sort the same way regardless of which one the
    filesystem happens to list first -- `glob()` makes no ordering promise,
    so a tiebreak that silently falls back to Python's stable sort over
    glob's own enumeration order picks up whatever order the filesystem
    happens to hand back that day.
    """
    import panel.app as panel_app

    same_second = "2026-08-19T12:00:00+00:00"
    tied = {
        "20260819T120000Z-aaaaaaaa": [{"event": "run_started", "at_utc": same_second}],
        "20260819T120000Z-bbbbbbbb": [{"event": "run_started", "at_utc": same_second}],
    }
    summaries = panel_app._control_progress_summaries(tied)
    assert [row["run_id"] for row in summaries] == [
        "20260819T120000Z-bbbbbbbb", "20260819T120000Z-aaaaaaaa",
    ], "a same-second tie must break deterministically, not on dict/glob order"


def test_a_named_run_id_parses_each_file_exactly_once(monkeypatch, tmp_path):
    """`runs` and `events` used to come from two independent filesystem reads
    of a directory a runner can be actively appending to -- taken far enough
    apart to disagree about what the latest event was, and for the named
    run's own file, redundant regardless of timing: the same bytes parsed
    twice to answer one request. One read, shared between both parts of the
    response, removes both the race and the duplicate work.
    """
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    panel_app._append_first_letters_control_progress_event("dev-control", _event())

    calls = []
    real_read = panel_app._read_control_progress_file

    def counting_read(path):
        calls.append(path)
        return real_read(path)

    monkeypatch.setattr(panel_app, "_read_control_progress_file", counting_read)
    monkeypatch.setattr(panel_app, "REQUIRE_LOGIN", False)
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(panel_app.app)

    response = client.get(
        "/api/missions/dev-control/first-letters-control/progress",
        params={"run_id": "run-a"})

    assert response.status_code == 200
    assert len(calls) == 1, (
        f"run-a.jsonl was parsed {len(calls)} times to answer one request -- "
        "'runs' and 'events' must come from the same read")


def test_a_malformed_schema_is_refused_and_writes_nothing(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    with pytest.raises(panel_app.HTTPException) as refused:
        panel_app._append_first_letters_control_progress_event(
            "dev-control", _event(schema="not-the-right-schema"))
    assert refused.value.status_code == 400
    assert not (tmp_path / "control-progress").exists()


def test_an_event_naming_a_different_mission_is_refused(monkeypatch, tmp_path):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    with pytest.raises(panel_app.HTTPException) as refused:
        panel_app._append_first_letters_control_progress_event(
            "dev-control", _event(mission_id="some-other-mission"))
    assert refused.value.status_code == 409


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "", "x" * 200, "..", "."])
def test_a_run_id_that_cannot_be_a_bare_filename_is_refused(monkeypatch, tmp_path, run_id):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    with pytest.raises(panel_app.HTTPException) as refused:
        panel_app._append_first_letters_control_progress_event(
            "dev-control", _event(run_id=run_id))
    assert refused.value.status_code == 400
    written = list(tmp_path.rglob("*.jsonl"))
    assert written == [], f"a bad run_id still wrote {written}"


def test_the_routes_are_the_same_functions_end_to_end(monkeypatch, tmp_path):
    """One round trip through HTTP, to catch a route wired to the wrong
    function or a response shape the handler silently changed -- the internal
    functions above are exercised directly for everything else."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "mission_directory", lambda _m: tmp_path)
    monkeypatch.setattr(panel_app, "REQUIRE_LOGIN", False)
    client = fastapi_testclient.TestClient(panel_app.app)

    posted = client.post(
        "/api/missions/dev-control/first-letters-control/progress",
        json={"event": _event(event="run_started")})
    assert posted.status_code == 201
    assert posted.json()["event"] == "run_started"

    posted_second = client.post(
        "/api/missions/dev-control/first-letters-control/progress",
        json={"event": _event(event="boundary_started", boundary="P1")})
    assert posted_second.status_code == 201

    fetched = client.get(
        "/api/missions/dev-control/first-letters-control/progress")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["run_id"] == "run-a"
    assert [row["event"] for row in body["events"]] == [
        "run_started", "boundary_started"]
    assert body["runs"][0]["run_id"] == "run-a"
    assert body["runs"][0]["current_boundary"] == "P1"

    by_run_id = client.get(
        "/api/missions/dev-control/first-letters-control/progress",
        params={"run_id": "run-a"})
    assert by_run_id.status_code == 200
    assert len(by_run_id.json()["events"]) == 2
