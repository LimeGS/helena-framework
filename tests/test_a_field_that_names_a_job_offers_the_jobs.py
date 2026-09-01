"""A parameter that names another job offers the jobs it can name.

`screening_of` is a P5 job id and `ordering_of` a P8 one: the worker fetches
that job's output rather than being handed a path. The form drew both as free
text, so queueing a screen meant finding an opaque id on another page, typing it
back, and learning from the queue's refusal whether it was even the right kind
of job. Manual QA hit exactly that: a P7 queued against a P5 job that existed,
had run and was ALIVE, and was refused because a control mission wants the P5
job bound to its control -- a fact no part of the form carried.

The candidates are the succeeded jobs of the producing phase in this mission,
which is the set the worker can actually fetch from. Free text stays legal: a
job id from outside the mission is still an answer somebody may have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

pytest.importorskip("fastapi")


def test_the_producing_phase_is_declared_beside_the_parameters() -> None:
    from job_store import JOB_INPUT_PARAMETERS, PHASE_PARAMETERS

    # Declared where the parameters are, so a new job-naming field cannot be
    # added in one place and forgotten in the other.
    assert JOB_INPUT_PARAMETERS["screening_of"] == "P5"
    assert JOB_INPUT_PARAMETERS["ordering_of"] == "P8"
    for name, producer in JOB_INPUT_PARAMETERS.items():
        assert any(name in fields for fields in PHASE_PARAMETERS.values()), name
        assert producer in PHASE_PARAMETERS, producer


def test_the_schema_says_which_phase_a_field_names() -> None:
    from job_store import phase_parameter_schema

    fields = {f["name"]: f for f in phase_parameter_schema("P7")["fields"]}
    assert fields["screening_of"]["names_a_job_from"] == "P5"
    # Everything else says nothing, rather than saying something empty.
    assert fields["bbox"]["names_a_job_from"] is None


HASHED = {
    "artifact_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
}


class _Store:
    def __init__(self, rows=None) -> None:
        self.asked: list[dict] = []
        self.rows = rows

    def jobs(self, **query):
        self.asked.append(query)
        if self.rows is not None:
            return self.rows
        return [{"job_id": "p5-0c1c5934eaf442", "sample_id": "PHerc0139",
                 "profile_id": "ink-9um-hybrid-3d2d@1.0.0",
                 "finished_at": "2026-08-27T18:02:00Z",
                 "result": {"probability_map": HASHED}}]


def test_the_parameters_endpoint_offers_this_missions_jobs(monkeypatch) -> None:
    import panel.app as app

    store = _Store()
    monkeypatch.setattr(app, "job_store", lambda: store)
    body = app.api_phase_parameters(
        "P7", mission="qa-web-p0-p9-20260827", sample="PHerc0139")
    import json

    payload = json.loads(bytes(body.body))
    field = next(f for f in payload["fields"] if f["name"] == "screening_of")
    assert field["choices"] == [{
        "value": "p5-0c1c5934eaf442",
        "note": "PHerc0139 · ink-9um-hybrid-3d2d@1.0.0 · 2026-08-27T18:02",
    }]
    # Succeeded jobs of the producing phase, in this mission, for this scroll.
    assert store.asked == [{
        "states": ("succeeded",), "mission_id": "qa-web-p0-p9-20260827",
        "phase": "P5", "sample_id": "PHerc0139", "limit": 50,
    }]


def test_without_a_mission_the_form_asks_for_nothing(monkeypatch) -> None:
    """No mission, no candidates -- and no query either.

    Every job belongs to a mission, so a picker outside one would either be
    empty or offer the whole fleet's jobs as if they were available here.
    """
    import panel.app as app

    store = _Store()
    monkeypatch.setattr(app, "job_store", lambda: store)
    import json

    payload = json.loads(bytes(app.api_phase_parameters("P7").body))
    field = next(f for f in payload["fields"] if f["name"] == "screening_of")
    assert "choices" not in field
    assert store.asked == []


def test_a_store_that_cannot_answer_leaves_the_field_alone(monkeypatch) -> None:
    """An unreachable queue costs the picker, not the form.

    The rest of the parameters are static and the field still accepts a typed
    id, so a database blip must not turn the launcher into an error page.
    """
    import panel.app as app

    class _Broken:
        def jobs(self, **_query):
            raise RuntimeError("no route to host")

    monkeypatch.setattr(app, "job_store", _Broken)
    import json

    payload = json.loads(bytes(
        app.api_phase_parameters("P7", mission="m", sample="PHerc0139").body))
    field = next(f for f in payload["fields"] if f["name"] == "screening_of")
    assert "choices" not in field
    assert payload["available"] is True


def test_a_p5_the_screen_would_refuse_is_not_offered(monkeypatch) -> None:
    """P7 takes the map by hash, so a P5 without one is not a candidate.

    Offering it hands somebody the single answer the queue will not take:
    "P7 requires an exact hash-bound P5 probability map", after the form said
    this was one of the jobs to choose from.
    """
    import panel.app as app
    import json

    store = _Store(rows=[
        {"job_id": "p5-old", "sample_id": "PHerc826", "result": {}},
        {"job_id": "p5-hashed", "sample_id": "PHerc826",
         "result": {"probability_map": HASHED}},
        {"job_id": "p5-half", "sample_id": "PHerc826",
         "result": {"probability_map": {"artifact_sha256": "c" * 64}}},
    ])
    monkeypatch.setattr(app, "job_store", lambda: store)
    payload = json.loads(bytes(
        app.api_phase_parameters("P7", mission="golden-run", sample="PHerc826").body))
    field = next(f for f in payload["fields"] if f["name"] == "screening_of")
    assert [c["value"] for c in field["choices"]] == ["p5-hashed"]
