"""Which planner runs, and whether the CT is asked at all.

Two defaults that were wrong in the same direction: the choice a person made
when queueing a run did not reach the machine that grew it, and the screen that
checks the raw scan agrees with the prediction was off unless someone remembered
a flag. Both made the fleet look like it was doing something it was not -- a run
labelled "Panel of LLM experts" grew with the deterministic planner, and 142 of
172 queued tasks trusted the prediction without the scan.

The planner name rides in the task payload rather than a new column, because the
payload is jsonb and the claim already spreads it over the task dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.cli import build_parser  # noqa: E402
from fleet.planner import DeterministicPlanner, PlannerProviderUnavailable  # noqa: E402
from fleet.worker import SegmentWorker  # noqa: E402


def bootstrap_args(*extra: str):
    return build_parser(ROOT).parse_args(
        ["bootstrap", "--db", "x", "--eligible", "e.json", "--catalog", "c.json",
         *extra])


def test_the_ct_is_asked_unless_somebody_says_not_to():
    assert bootstrap_args().ct_material_support_gate is True
    assert bootstrap_args("--no-ct-material-support-gate").ct_material_support_gate is False


def test_turning_the_gate_on_takes_no_value():
    """What broke it from the panel: a store_true flag handed a value.

    argparse read the value as a positional and refused the whole command, so
    the one option that could not be set from the form was the one that decides
    whether the scan gets a say.
    """
    with pytest.raises(SystemExit):
        bootstrap_args("--ct-material-support-gate", "on")


def _bare(planner, factory):
    instance = object.__new__(SegmentWorker)
    instance.planner = planner
    instance.planner_factory = factory
    return instance


def test_the_task_names_the_planner_and_gets_it():
    host_default = DeterministicPlanner(contract_version="v1")
    asked_for = DeterministicPlanner(contract_version="v2")
    seen = {}

    def factory(name, model=None):
        seen.update(name=name, model=model)
        return asked_for

    resolved = _bare(host_default, factory).planner_for(
        {"planner": "deterministic-v2", "planner_model": "anthropic/claude-opus-5"})
    assert resolved is asked_for
    # The model chosen when the run was queued reaches the planner that runs, or
    # picking one in the form sets a field nothing reads.
    assert seen == {"name": "deterministic-v2", "model": "anthropic/claude-opus-5"}


def test_a_task_that_names_nothing_gets_the_host_default():
    host_default = DeterministicPlanner(contract_version="v1")
    def factory(name, model=None):
        raise AssertionError("nothing was named; the factory must not be called")
    assert _bare(host_default, factory).planner_for({}) is host_default


def test_a_host_that_cannot_build_it_requeues_rather_than_consuming_the_task():
    """No API key is an outage on this host, not a verdict on the task.

    POLICY_REJECTED is terminal: a worker started without OPENROUTER_API_KEY
    would burn every task asking for a model planner. PlannerProviderUnavailable
    puts it back with a retry delay for a host that is equipped.
    """
    def factory(name, model=None):
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    with pytest.raises(PlannerProviderUnavailable) as failure:
        _bare(DeterministicPlanner(), factory).planner_for({"planner": "fusion-v2"})
    assert "fusion-v2" in str(failure.value)


class NoSurfaces:
    def surfaces_for_snapshot(self, _identifier):
        return []


def test_the_name_is_written_on_the_task_the_queue_stores():
    """No migration: the claim spreads payload over the task dict already."""
    from fleet.generator import generate_manual_tasks

    task = generate_manual_tasks(
        NoSurfaces(),
        {"source_snapshot_id": "src-1", "sample_id": "PHerc826",
         "shape_xyz": [8000, 8000, 20000],
         "m7_uri": "https://example.invalid/m7.zarr"},
        [{"x": 4000, "y": 4000, "z": 10000}],
        catalog_snapshot_sha256="deadbeef", grid_step=2048, query_radius=64,
        volume_edge_margin=64, grid_version="ct-l0-manual-v1",
        policy_version="ink-blind-v1", submitted_by="limegs",
        planner="fusion-v2")[0]
    assert task["planner"] == "fusion-v2"


def test_a_run_that_inserted_nothing_is_not_reported_as_started():
    """172 tasks, 0 with a planner: the run said 201 and queued nothing.

    Every cell of PHerc826 already had a task, so ON CONFLICT DO NOTHING ate
    the insert and the panel reported success anyway.
    """
    sys.path.insert(0, str(ROOT / "panel"))
    from app import bootstrap_counts

    assert bootstrap_counts('{"queued": {"PHerc826": {"generated": 12, "inserted": 0}}}') == (12, 0)
    assert bootstrap_counts('{"queued": {"PHerc826": {"generated": 12, "inserted": 12}}}') == (12, 12)
    # An older fleet that prints no "queued" key must not read as a success.
    assert bootstrap_counts('{"receipt": "x.json"}') == (0, 0)
    assert bootstrap_counts("not json") == (0, 0)


def test_the_bucket_spelling_of_a_scroll_reaches_the_catalogs_spelling(monkeypatch):
    """PHerc0826 in the bucket, PHerc826 in the catalog, one scroll.

    The catalog name is hashed into every frozen plan, so renaming it fails
    closeout -- it was tried. Translating here is what is left, and without it
    New Run answers a Python traceback for that scroll.
    """
    sys.path.insert(0, str(ROOT / "panel"))
    import json as json_module

    import app

    catalog = Path(pytest.importorskip("tempfile").mkdtemp()) / "eligible.json"
    # The catalog's real shape: {"entries": [...]}, read the way the fleet's
    # own bootstrap_sources reads it.
    catalog.write_text(json_module.dumps(
        {"entries": [{"sample_id": "PHerc826"}, {"sample_id": "PHerc1667"}]}))
    monkeypatch.setattr(app, "CATALOG", catalog)

    assert app.catalog_sample_id("PHerc0826") == "PHerc826"
    assert app.catalog_sample_id("PHerc826") == "PHerc826"
    assert app.catalog_sample_id("pherc-0826") == "PHerc826"
    # A number with no padding to strip must not be mangled into another scroll.
    assert app.catalog_sample_id("PHerc1667") == "PHerc1667"
    with pytest.raises(Exception) as refused:
        app.catalog_sample_id("PHerc9999")
    assert "PHerc826" in str(refused.value)


def test_a_mission_scopes_by_the_name_the_control_plane_stores(monkeypatch):
    """A mission lists PHerc0826; the fleet stores PHerc826.

    Compared raw, every mission-scoped page read zero while the control plane
    held forty-two surfaces -- which looks like an empty campaign and is a
    spelling. Reading is also not queueing: a mission may name a scroll the
    frozen cohort does not carry, and a page that refuses because of one is a
    page nobody can open.
    """
    sys.path.insert(0, str(ROOT / "panel"))
    import json as json_module
    import tempfile

    import app

    catalog = Path(tempfile.mkdtemp()) / "eligible.json"
    catalog.write_text(json_module.dumps({"entries": [{"sample_id": "PHerc826"}]}))
    monkeypatch.setattr(app, "CATALOG", catalog)

    assert app.catalog_sample_id("PHerc0826", strict=False) == "PHerc826"
    # Unknown to the catalog: kept as written rather than refused.
    assert app.catalog_sample_id("PHerc0846A", strict=False) == "PHerc0846A"
    # Queueing still refuses, because there the name has to resolve to a volume.
    with pytest.raises(Exception):
        app.catalog_sample_id("PHerc0846A")


def test_every_scroll_filter_translates_the_name(monkeypatch):
    """The guard for the class, not for the case.

    Everything reaching the panel names a scroll the way the bucket does --
    PHerc0826 -- and every row names it the way the frozen catalog does --
    PHerc826. I fixed this at the queue boundary, then in mission scoping, then
    in subject scoping, each time believing it was the last place. This asserts
    that every read path which turns a scroll into a filter goes through one
    translation, so the next one is a failing test rather than a page of zeros.
    """
    sys.path.insert(0, str(ROOT / "panel"))
    import inspect

    import app

    # The functions that turn a scroll into a filter. api_segmentation_runs
    # delegates, so segmentation_runs is where the translation belongs.
    translated = {"segmentation_state", "subject_surfaces", "phase_state"}
    for name in sorted(translated):
        source = inspect.getsource(getattr(app, name))
        assert "stored_scroll(" in source, (
            f"{name} filters by scroll without translating the name; a request "
            "spelling it the bucket's way will match no row")
    assert "stored_scroll(" in inspect.getsource(app.read_scope)
    for name in ("api_segments", "segmentation_runs", "api_no_seed"):
        assert "read_scope(" in inspect.getsource(getattr(app, name))
    assert app.stored_scroll(None) is None
