"""The endpoints today's pages were built on, through the app itself.

Everything these serve was written this week, and each of them was wrong once in
a way that looked like an empty campaign rather than a broken read: the coverage
query returned KeyError and the page said "no cells attempted", the replan query
did the same and the panel reported it as a policy refusal, and the parameter
schema is the thing standing between a flag existing in the queue and anybody
being able to set it.

Driven through TestClient with no control plane behind it, so these run
anywhere. What they hold is the shape of each answer and the behaviour when the
database is absent -- which is the state a fresh deployment is in, and the one
where a page that lies is most expensive.
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
    # No control plane: every one of these has to answer without one.
    monkeypatch.setattr(module, "DSN", "")
    return module


@pytest.fixture
def anonymous(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


@pytest.fixture
def client(app_module, anonymous):
    """Signed in through the real login, because the gate is what stands between
    the network and a panel that queues GPU work."""
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    response = anonymous.post("/api/session",
                              json={"username": "tester", "password": "a-long-enough-one"})
    assert response.status_code == 200, response.text
    return anonymous


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

NEW_ROUTES = ["/api/phases/P4/parameters", "/api/coverage", "/api/ink/lanes"]


@pytest.mark.parametrize("route", NEW_ROUTES)
def test_a_new_route_is_closed_without_a_session(anonymous, route):
    """Deny by default: a list of what is protected goes stale the moment
    somebody adds a route, and the one they forget is the one that queues work."""
    assert anonymous.get(route).status_code == 401


def test_queueing_a_replan_is_closed_too(anonymous):
    response = anonymous.post("/api/segmentation/replan",
                              json={"grid_version": "g", "policy_version": "p"})
    assert response.status_code == 401


# --------------------------------------------------------------------------
# The parameter schema, which is what the form draws itself from
# --------------------------------------------------------------------------

def test_the_schema_offers_every_parameter_the_queue_accepts(client):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import PHASE_PARAMETERS

    for phase, accepted in PHASE_PARAMETERS.items():
        body = client.get(f"/api/phases/{phase}/parameters").json()
        assert body["available"], f"{phase}: {body.get('reason')}"
        assert {field["name"] for field in body["fields"]} == set(accepted)


def test_the_flag_that_took_the_control_from_0_09_to_0_885_is_offered(client):
    fields = {f["name"]: f for f in
              client.get("/api/phases/P4/parameters").json()["fields"]}
    assert fields["flip_normals"]["type"] == "boolean"
    assert not fields["flip_normals"]["filled_by_deployment"]
    # And what the deployment decides is not asked of a person.
    assert fields["artifact_store"]["filled_by_deployment"]


def test_the_pairs_that_must_be_exactly_one_travel_with_the_schema(client):
    """Neither is required and exactly one must be there, which "required"
    cannot express -- so the browser must not have to know it separately."""
    p5 = client.get("/api/phases/P5/parameters").json()
    assert {"tiff_dir", "layer_stack"} == set(p5["exactly_one_of"][0]["names"])


def test_a_phase_with_nothing_to_queue_says_so(client):
    body = client.get("/api/phases/P0/parameters").json()
    assert body["available"] is False
    assert "takes no queued parameters" in body["reason"]


def test_an_unknown_phase_does_not_pretend(client):
    body = client.get("/api/phases/P99/parameters").json()
    assert body["available"] is False


# --------------------------------------------------------------------------
# The reads, with no control plane behind them
# --------------------------------------------------------------------------

def test_coverage_without_a_control_plane_says_why(client):
    """It answered "no cells attempted" when the query was broken, which is
    indistinguishable from an unexplored scroll -- the exact silence this phase
    exists to break."""
    body = client.get("/api/coverage").json()
    assert body["available"] is False
    assert body["reason"]


def test_the_lane_inventory_survives_without_a_control_plane(client):
    """It reads the profile directory and the method registry, neither of which
    is the database, so a fresh deployment can still see what it could run."""
    body = client.get("/api/ink/lanes").json()
    assert body["available"] is True
    assert body["lanes"], "no ink lanes are described at all"
    assert body["routable"] >= 1
    assert all(lane["reason"] for lane in body["lanes"] if not lane["routable"])


def test_flattening_without_a_control_plane_says_why(client):
    body = client.get("/api/flattening").json()
    assert body["available"] is False


def test_seed_probe_status_is_rollout_safe_without_its_control_plane(client):
    body = client.get("/api/segmentation/probes").json()
    assert body["available"] is False
    assert body["reason"]
    assert body["counts"] == {
        "runs": 0, "trials": 0, "decisions": 0, "promotions": 0,
    }


def test_seed_probe_launcher_contract_is_served_with_its_bounds(client):
    probe = client.get("/api/segmentation/options").json()["probe"]
    assert [mode["id"] for mode in probe["modes"]] == ["off", "shadow", "select"]
    assert probe["default_mode"] == "off"
    # 20, not 3. Three made the probe a tie-break between m7's best candidates,
    # and the first shadow run measured 8 of 8 ELIGIBLE at that depth -- a 0%
    # rejection rate against a 34% break-even. Whether m7's ordering holds at
    # rank 20 is the open question, and the cap is what made it unaskable.
    assert probe["top_k"] == {"minimum": 1, "maximum": 20, "default": 2}
    assert probe["generations"] == {"minimum": 10, "maximum": 20, "default": 12}
    assert probe["select_readiness"]["available"] is False
    assert probe["select_readiness"]["benchmark_approved"] is False
    assert probe["select_readiness"]["review_owner_declared"] is False
    assert "benchmark receipt" in probe["select_readiness"]["reason"]
    assert "verified immutable" in probe["select_readiness"]["reason"]
    assert "review-resolution owner" in probe["select_readiness"]["reason"]
    assert "not proof" in probe["caveat"]


def test_seed_probe_select_is_scoped_to_receipt_samples(
    app_module, tmp_path, monkeypatch
):
    version = "version-1234"

    def locked_entry(sample_id: str) -> dict:
        ct_uri = f"s3://test/{sample_id}/ct?versionId={version}"
        m7_uri = f"s3://test/{sample_id}/m7?versionId={version}"
        return {
            "sample_id": sample_id,
            "ct_uri": ct_uri,
            "ct_sha256": "a" * 64,
            "surface_prediction_uri": m7_uri,
            "surface_prediction_sha256": "b" * 64,
            "source_content_lock": {
                "schema": "campaignx.source_content_lock.v1",
                "status": "VERIFIED_IMMUTABLE",
                "verification_method": "immutable-uri-manifest-sha256-v1",
                "verified_at_utc": "2026-07-30T00:00:00Z",
                "ct_uri": ct_uri,
                "ct_sha256": "a" * 64,
                "ct_version_id": version,
                "m7_uri": m7_uri,
                "m7_sha256": "b" * 64,
                "m7_version_id": version,
            },
        }

    catalog = tmp_path / "eligible.json"
    catalog.write_text(
        json.dumps(
            {"entries": [locked_entry("PHerc125"), locked_entry("PHerc191")]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "CATALOG", catalog)
    monkeypatch.setattr(
        app_module, "SEED_PROBE_BENCHMARK_RECEIPT", tmp_path / "decision.json"
    )
    monkeypatch.setattr(
        app_module, "SEED_PROBE_REVIEW_OWNER", "segmentation-review-test"
    )
    monkeypatch.setenv("HELENA_ENABLE_SEED_PROBE_SELECT", "1")

    stage_root = ROOT / "framework/stages/01-segmentation"
    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))
    from fleet import seed_probe

    monkeypatch.setattr(
        seed_probe,
        "load_seed_probe_benchmark_receipt",
        lambda _path: {
            "benchmark_id": "scope-test-v1",
            "decision_receipt_sha256": "c" * 64,
            "paired_cell_count": 60,
            "scroll_count": 1,
            "authorized_sample_ids": ["PHerc125"],
        },
    )

    authorized = app_module.seed_probe_select_readiness("PHerc0125")
    assert authorized["available"] is True
    assert authorized["benchmark_scope_allows"] is True
    assert authorized["authorized_sample_count"] == 1

    unauthorized = app_module.seed_probe_select_readiness("PHerc0191")
    assert unauthorized["available"] is False
    assert unauthorized["benchmark_approved"] is True
    assert unauthorized["source_locked"] is True
    assert unauthorized["benchmark_scope_allows"] is False
    assert "outside the approved benchmark scope" in unauthorized["reason"]


def test_seed_probe_select_refuses_a_lane_it_would_steer(client):
    response = client.post("/api/segmentation/runs", json={
        "sample_id": "PHerc0826",
        "planner": "opencode-v2",
        "seed_probe_mode": "select",
    })
    assert response.status_code == 400
    assert "cost-aware-v2 or deterministic-v2" in response.json()["detail"]


def test_seed_probe_shadow_does_not_restrict_the_planner_lane(client):
    response = client.post("/api/segmentation/runs", json={
        "sample_id": "PHerc0826",
        "planner": "opencode-v2",
        "seed_probe_mode": "shadow",
    })
    # It reached the ordinary deployment precondition, so the lane/probe pair
    # was accepted. There is deliberately no control plane in this fixture.
    assert response.status_code == 409
    assert "control plane" in response.json()["detail"]


def test_a_replan_needs_a_control_plane_rather_than_failing_late(client):
    response = client.post("/api/segmentation/replan",
                           json={"grid_version": "replan-x", "policy_version": "replan-x"})
    assert response.status_code == 409
    assert "control plane" in response.json()["detail"]


def test_a_replan_refuses_a_version_it_cannot_use(client):
    """The versions are the whole safety of this: task identity is (snapshot,
    grid, cell, policy), so a replan under the same policy inserts nothing and
    reports success."""
    response = client.post("/api/segmentation/replan",
                           json={"grid_version": "", "policy_version": "p"})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# The documentation the panel serves instead of a directory of markdown
# --------------------------------------------------------------------------

def test_every_queueable_field_explains_itself(app_module):
    """The user guide documents each phase from the same schema the form draws
    itself from, so a field with no note is a field the guide cannot explain.
    Thirty-four of them had none: the guide would have printed "no note
    recorded" thirty-four times."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import job_store

    missing = [(phase, field["name"])
               for phase in job_store.PHASE_PARAMETERS
               for field in job_store.phase_parameter_schema(phase)["fields"]
               if not field["note"] and not field["filled_by_deployment"]]
    assert missing == []


def test_the_documentation_is_served_rather_than_shipped_as_files(app_module):
    """It moved into the panel: a user guide per phase and a developer
    reference, both reading the committed contracts rather than restating
    them. What is left under docs/ is hash-bound evidence, not prose."""
    stale = sorted(p.name for p in (ROOT / "docs").glob("*.md"))
    assert stale == [], f"markdown left behind: {stale}"

    guide = (ROOT / "panel/web/src/routes/UserGuide.tsx").read_text()
    # Built from the contract and the live schema, not from a copy of either.
    assert "/api/phases" in guide and "/parameters" in guide
    assert "details" in guide, "the per-phase modules are collapsible"
    # Segmentation is planned rather than parameterised, so its controls are the
    # seeders and the eighteen options, and both are served too.
    assert "/api/segmentation/options" in guide
    assert '"/api/segmentation"' in guide

    developer = (ROOT / "panel/web/src/routes/DeveloperReference.tsx").read_text()
    for topic in ("Adding a model or a tool", "Extending each phase",
                  "Doing a phase a different way",
                  "Adding a phase that does not exist yet",
                  "Running a worker of your own"):
        assert topic in developer, topic


def test_the_guide_is_readable_before_a_mission_exists(anonymous):
    """It explains what a mission is. Putting it behind the screen that demands
    one is a manual locked inside the machine it explains."""
    app_tsx = (ROOT / "panel/web/src/App.tsx").read_text()
    gated = app_tsx[app_tsx.index("function Gated()"):]
    assert "/documentation" not in gated


def test_two_screens_do_not_share_a_layout_class(app_module):
    """The sign-in screen and the mission picker were both `.gate`. They want
    opposite layouts -- one centres itself in the viewport, the other is a card
    list at the top of the page -- and the later rule won, so the picker
    rendered as a card floating in the middle of a 100vh box."""
    css = (ROOT / "panel/web/src/styles.css").read_text()
    mission = (ROOT / "panel/web/src/mission.tsx").read_text()
    login = (ROOT / "panel/web/src/Login.tsx").read_text()

    assert 'className="gate"' not in mission
    assert 'className="missiongate"' in mission
    assert 'className="gate"' in login

    # One declaration each, and only the sign-in one takes the viewport.
    assert css.count("\n.gate {") == 1
    assert css.count("\n.missiongate {") == 1
    assert "100vh" not in css[css.index("\n.missiongate {"):][:200]


def test_configuration_does_not_need_a_mission(app_module):
    """Settings, hosts, accounts and the audit log belong to the deployment, not
    to a campaign. Behind the mission gate, Configuration was a menu entry that
    did nothing when clicked."""
    app_tsx = (ROOT / "panel/web/src/App.tsx").read_text()
    ungated, gated = app_tsx.split("function Gated()")
    assert "<Configuration />" not in gated
    assert '<Route path="/configuration"' in ungated


def test_the_secrets_card_survives_having_no_control_plane(app_module):
    """It dereferenced the response before checking for one. Between retries a
    failed query is neither loading nor errored and its data is undefined, so a
    deployment with no control plane -- a fresh install -- got a Configuration
    page that rendered nothing at all.

    And the 409 is an answer, not a hiccup: retrying it three times only delayed
    showing somebody the reason."""
    source = (ROOT / "panel/web/src/components/FleetSecrets.tsx").read_text()
    assert "queryGate" in source
    assert "data!" not in source
    assert "retry: false" in source


def test_every_phase_is_actually_guided(app_module):
    """A module that only restates the contract is not a guide. Each phase must
    say what you are doing, what to have ready, the decisions in order, what to
    look at afterwards, and how it goes wrong quietly."""
    content = (ROOT / "panel/web/src/routes/guide-content.ts").read_text()
    for phase in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"):
        assert f"\n  {phase}: {{" in content, f"{phase} has no guide"
    for section in ("purpose:", "before:", "steps:", "reading:", "traps:"):
        assert content.count(section) >= 10, f"{section} missing from some phase"


def test_the_guide_ships_the_pictures_it_embeds(app_module):
    """A figure whose file is missing is a broken image and a caption explaining
    it. Every src named in the guide must exist under public/, which is what
    Vite copies into the build."""
    import re

    guide = (ROOT / "panel/web/src/routes/UserGuide.tsx").read_text()
    content = (ROOT / "panel/web/src/routes/guide-content.ts").read_text()
    named = set(re.findall(r'src=\{?"([a-z-]+)"', guide + content))
    named |= set(re.findall(r'src: "([a-z-]+)"', content))
    assert named, "the guide embeds no figures at all"

    # Under src/assets and imported, not under a public directory: the panel
    # serves static files from /assets only, so a picture anywhere else falls
    # through to the single-page fallback and renders broken.
    shipped = {p.stem for p in (ROOT / "panel/web/src/assets/guide").glob("*.png")}
    assert named <= shipped, f"missing: {sorted(named - shipped)}"
    for figure in sorted(named):
        assert f'assets/guide/{figure}.png"' in guide, f"{figure} is not imported"

    # And every caption teaches: a figure with no words under it is decoration.
    assert guide.count("caption=") + content.count("caption:") >= len(named)


def test_a_link_carries_the_scope_it_was_read_in(app_module):
    """A phase page is a different page in a different mission, and its tabs are
    different views. Without either in the address, a shared link lands the
    reader wherever their browser last was."""
    mission = (ROOT / "panel/web/src/mission.tsx").read_text()
    phase = (ROOT / "panel/web/src/routes/Phase.tsx").read_text()
    assert 'get("mission")' in mission
    assert 'get("tab")' in phase


def test_the_documentation_opens_folded_shut(app_module):
    """A page that opens as several screens of prose is a page people scroll
    past. Every section starts closed, so the first thing you see is an index of
    what is there."""
    import re

    for name in ("UserGuide", "DeveloperReference"):
        source = (ROOT / f"panel/web/src/routes/{name}.tsx").read_text()
        cards = re.findall(r"<Card [^>]*?>", source, re.S)
        # The one card that stays open is the list of phase modules, and every
        # module inside it is itself a closed <details>.
        open_cards = [c for c in cards if "collapsed" not in c]
        assert open_cards == [], f"{name}: {open_cards}"

    guide = (ROOT / "panel/web/src/routes/UserGuide.tsx").read_text()
    assert "<details className=\"guide-phase\">" in guide, "modules must start closed"
    assert "guide-phase\" open" not in guide


def test_the_prose_has_a_readable_measure(app_module):
    """96ch of small text is a line the eye loses its place in. Prose is capped;
    tables and figures keep the full width because they are scanned."""
    css = (ROOT / "panel/web/src/styles.css").read_text()
    block = css[css.index("---- documentation: a column you can actually read ----"):]
    assert "max-width:74ch" in block
    assert ".guide-prose p" in block and ".guide-phase p" in block


def test_the_openapi_document_is_not_swallowed_by_the_spa() -> None:
    """The API reference reads it, so where it lives is load-bearing.

    FastAPI defaults openapi_url to /openapi.json, and the SPA serves a
    catch-all at /{full_path}. The catch-all won: /openapi.json answered 200
    with index.html, which is the worst shape of wrong for a document a client
    parses as JSON -- no error, just HTML where the paths should be.
    """
    source = (ROOT / "panel/app.py").read_text()
    assert 'openapi_url="/api/openapi.json"' in source, (
        "openapi_url is not under /api/, so the SPA catch-all answers it with HTML"
    )
    page = (ROOT / "panel/web/src/routes/ApiReference.tsx").read_text()
    assert '"/api/openapi.json"' in page, "the API reference reads a different URL"
    assert '"/openapi.json"' not in page.replace('"/api/openapi.json"', ""), (
        "the page still fetches the path the SPA swallows"
    )
