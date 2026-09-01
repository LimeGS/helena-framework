"""The control plane answers "may this campaign queue?" with its evidence.

Task 9 joins four already-implemented gates -- the positive control, the
candidate preflight, the derived task budget, and the candidate-starvation
pause -- into one read-only readiness answer and one P1 admission authority.

Nothing here weakens acceptance.  A readiness answer is a report about
evidence; it never becomes evidence, and there is no field in it an operator
can set to make a missing control, a stale preflight, an unjustified budget or
an active pause stop blocking.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
for extra in (STAGE, ROOT / "scripts/harness", ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fleet import campaign_decision  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from framework.contracts import mission as mission_contract  # noqa: E402
from run_first_letters_positive_control import (  # noqa: E402
    BOUNDARIES,
    canonical_sha256,
    evaluate_survival_matrix,
)
from test_first_letters_campaign_decision import (  # noqa: E402
    compute_cap,
    eligible_cell_ids,
    preflight,
)
from test_first_letters_campaign_stop import (  # noqa: E402
    admission,
    admitted_attempt,
    attempt,
    budget_admission,
    budget_task,
    derive,
    no_seed_result,
    resume_authorization,
    resumed_sample_admission,
    store_source,
)


CONTROL_PATH = (
    ROOT / "framework/profiles/01-segmentation/"
    "first-letters-control-policy-1.1.0.json"
)
POLICY_PATH = (
    ROOT / "framework/profiles/01-segmentation/"
    "first-letters-campaign-decision-policy-1.2.0.json"
)
REVISION = "4" * 40
MISSION = "first-letters"
SCROLL = "PHerc0358"
STORED_SCROLL = "PHerc358"


def control_manifest() -> dict:
    return json.loads(CONTROL_PATH.read_text(encoding="utf-8"))


def campaign_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def control_runtime(document: dict | None = None) -> dict:
    """What ``verified_control_runtime`` reports on a matching deployment."""
    document = document or control_manifest()
    locks = []
    for row in document["profile_locks"]:
        profile = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
        locks.append({
            **copy.deepcopy(row),
            "declared_sha256_semantics": "RAW_FILE_BYTES_SHA256",
            "actual_sha256": row["sha256"],
            "actual_file_sha256": row["sha256"],
            "actual_canonical_document_sha256": canonical_sha256(profile),
            "actual_profile_id": row["profile_id"],
            "verified": True,
        })
    return {
        "deployed_revision": REVISION,
        "control_profile_id": document["profile_id"],
        "control_profile_sha256": canonical_sha256(document),
        "source_locks_sha256": canonical_sha256(document["source_locks"]),
        "profile_locks": locks,
        "profile_locks_verified": True,
        "models": [
            {**copy.deepcopy(row), "installed": True,
             "installed_at": "2026-08-02T00:00:00Z"}
            for row in document.get("model_locks", [])
        ],
    }


def control_receipt(*, revision: str = REVISION, mission_id: str = MISSION,
                    failing_at: str | None = None,
                    incomplete_at: str | None = None) -> dict:
    """A stage-survival matrix, evaluated by the runner's own pure rule."""
    document = control_manifest()
    runtime = control_runtime(document)
    rows = []
    for boundary in BOUNDARIES:
        state, reason = "PASS", "FIXTURE_BOUNDARY_PASS"
        if boundary == failing_at:
            state, reason = "FAILED", "FIXTURE_BOUNDARY_CONTRADICTED"
        elif boundary == incomplete_at:
            state, reason = "INCOMPLETE", "FIXTURE_BOUNDARY_NOT_TERMINAL"
        rows.append({
            "boundary": boundary, "input_artifacts": [], "profile_ids": [],
            "parameters": {}, "counts": {}, "terminal_state": state,
            "reason_code": reason, "elapsed_seconds": 0.0,
            "resource_identity": {}, "output_hashes": {},
            "non_claim": "One boundary of the frozen method, and nothing more.",
        })
    return evaluate_survival_matrix({
        "schema": "campaignx.first_letters_stage_survival.v1",
        "control_id": document["control_cohort"]["control_id"],
        "mission_id": mission_id,
        "bindings": {
            "deployed_revision": revision,
            "control_profile_id": document["profile_id"],
            "control_profile_sha256": runtime["control_profile_sha256"],
            "source_locks_sha256": runtime["source_locks_sha256"],
            "profile_locks": list(document["profile_locks"]),
            "models": list(runtime["models"]),
            "p0_artifact": {"artifact_id": "p0-control", "sha256": "5" * 64},
        },
        "allow_unvalidated": False,
        "ink_blind_discovery": True,
        "control_pass_is_independent_validation": False,
        "automatic_letter_acceptance": False,
        "stages": rows,
    })


class DecisionStore:
    """A read-only fleet view carrying real starvation evidence."""

    def __init__(self, attempts=(), admissions=(), authorizations=()):
        self.attempts = [copy.deepcopy(row) for row in attempts]
        self.admissions = [copy.deepcopy(row) for row in admissions]
        self.authorizations = [copy.deepcopy(row) for row in authorizations]
        self.decisions = (
            derive(self.attempts, self.admissions) if self.attempts else [])

    def campaign_decisions(self, *, mission_id, policy_version=None):
        return [row for row in self.decisions
                if row["mission_id"] == mission_id
                and (policy_version is None
                     or row.get("policy_version") == policy_version)]

    def campaign_active_decision(self, *, mission_id):
        return campaign_decision.derive_campaign_active_decision(
            self.attempts, self.admissions, self.decisions,
            self.authorizations, campaign_policy(), mission_id=mission_id)


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    """One controlled mission with every gate wired to an explicit seam."""
    pytest.importorskip("fastapi")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("CX_RUNS", str(runs))
    import panel.app as app

    monkeypatch.setattr(app, "RUNS", runs)
    monkeypatch.setattr(app, "DSN", "")
    manifest = mission_contract.create(
        runs, mission_id=MISSION, name="first letters", scrolls=[SCROLL],
        created_by="alice", campaign_kind="FIRST_LETTERS_DISCOVERY",
        campaign_policy_id="first-letters-campaign-decision-policy@1.2.0",
        campaign_policy_sha256=content_sha256(campaign_policy()),
        deployed_revision=REVISION)
    monkeypatch.setattr(app, "_deployed_revision", lambda: REVISION)
    monkeypatch.setattr(app, "verified_control_runtime", control_runtime)
    monkeypatch.setattr(app, "fleet_store_read_only", DecisionStore)
    state = SimpleState(app=app, runs=runs, manifest=manifest,
                        directory=runs / MISSION)
    monkeypatch.setattr(app, "_latest_candidate_preflight_evidence",
                        lambda _mission, _sample: state.preflight)
    return state


class SimpleState:
    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.preflight = None

    def publish_control(self, receipt: dict, *, by: str = "campaign-owner") -> dict:
        return self.app._publish_first_letters_control(
            MISSION, receipt, published_by=by)

    def publish_budget(self, *, evidence: dict, cap: int = 100,
                       manual: int | None = None,
                       manual_reason: str | None = None) -> dict:
        population = campaign_decision.summarize_eligible_population(
            eligible_cell_ids(
                evidence["funnel"]["geometrically_eligible_cells"]
                if evidence["measurement_kind"] == "CENSUS"
                else evidence["funnel"]["geometrically_eligible_cells_estimate"]),
            order_seed_sha256=campaign_decision.probability_prefix_order_seed(
                evidence, campaign_policy(), compute_cap(cap)),
            prefix_limit=cap,
        )
        private = campaign_decision.derive_task_budget(
            evidence, campaign_policy(), compute_cap(cap),
            manual_task_count=manual, manual_lower_reason=manual_reason,
            eligible_population=population)
        public = campaign_decision.sanitize_task_budget_receipt(private)
        root = self.directory / "evidence" / "task-budgets" / SCROLL
        campaign_decision.persist_task_budget_receipt_pair(
            root / f"{private['receipt_sha256']}.private.json",
            root / f"{private['receipt_sha256']}.sanitized.json",
            private, public)
        return private

    def readiness(self) -> dict:
        return self.app.first_letters_readiness(MISSION)


def scroll_of(readiness: dict, sample: str = STORED_SCROLL) -> dict:
    return next(row for row in readiness["scrolls"]
                if row["sample_id"] == sample)


def codes(rows) -> list[str]:
    return [row["code"] for row in rows]


# ---------------------------------------------------------------------------
# The deployed revision and its positive control
# ---------------------------------------------------------------------------

def test_readiness_reports_a_missing_control_as_an_explicit_blocker(controlled):
    readiness = controlled.readiness()
    assert readiness["schema"] == "campaignx.first_letters_readiness.v1"
    assert readiness["controlled"] is True
    assert readiness["deployed_revision"] == REVISION
    assert readiness["control"]["evidence_status"] == "MISSING"
    assert readiness["control"]["control_state"] is None
    assert "CONTROL_EVIDENCE_MISSING" in codes(readiness["blockers"])
    assert readiness["queue_admitted"] is False


def test_readiness_reports_a_current_passing_control(controlled):
    published = controlled.publish_control(control_receipt())
    readiness = controlled.readiness()
    control = readiness["control"]
    assert control["evidence_status"] == "CURRENT"
    assert control["control_state"] == "CONTROL_PASS"
    assert control["content_sha256"] == published["content_sha256"]
    assert control["bound_deployed_revision"] == REVISION
    assert control["first_nonpassing_boundary"] is None
    assert "CONTROL_EVIDENCE_MISSING" not in codes(readiness["blockers"])
    assert "CONTROL_NOT_PASSING" not in codes(readiness["blockers"])


def test_a_control_bound_to_another_revision_is_stale_not_current(controlled):
    controlled.publish_control(control_receipt(revision="9" * 40))
    readiness = controlled.readiness()
    assert readiness["control"]["evidence_status"] == "STALE"
    assert readiness["control"]["control_state"] == "CONTROL_PASS"
    assert "CONTROL_EVIDENCE_STALE" in codes(readiness["blockers"])
    assert readiness["queue_admitted"] is False


def test_a_failing_control_blocks_even_on_the_exact_deployed_revision(controlled):
    controlled.publish_control(control_receipt(failing_at="P5"))
    readiness = controlled.readiness()
    assert readiness["control"]["evidence_status"] == "CURRENT"
    assert readiness["control"]["control_state"] == "CONTROL_FAILED"
    assert readiness["control"]["first_nonpassing_boundary"] == "P5"
    assert "CONTROL_NOT_PASSING" in codes(readiness["blockers"])


def test_an_incomplete_control_blocks_and_names_its_boundary(controlled):
    controlled.publish_control(control_receipt(incomplete_at="QC"))
    readiness = controlled.readiness()
    assert readiness["control"]["control_state"] == "CONTROL_INCOMPLETE"
    assert readiness["control"]["first_nonpassing_boundary"] == "QC"
    assert "CONTROL_NOT_PASSING" in codes(readiness["blockers"])


def test_publishing_a_control_the_server_cannot_reproduce_is_refused(controlled):
    forged = control_receipt(failing_at="P5")
    forged["control_state"] = "CONTROL_PASS"
    with pytest.raises(Exception, match="stage-survival"):
        controlled.publish_control(forged)
    assert controlled.readiness()["control"]["evidence_status"] == "MISSING"


def test_a_control_claiming_unvalidated_acceptance_is_refused(controlled):
    receipt = control_receipt()
    receipt["allow_unvalidated"] = True
    receipt["content_sha256"] = canonical_sha256(
        {k: v for k, v in receipt.items() if k != "content_sha256"})
    with pytest.raises(Exception, match="allow_unvalidated"):
        controlled.publish_control(receipt)


def test_a_control_published_into_another_mission_is_refused(controlled):
    with pytest.raises(Exception, match="mission"):
        controlled.publish_control(control_receipt(mission_id="somewhere-else"))


def test_the_runner_publishes_its_matrix_where_the_gate_reads_it(controlled):
    """The receipt used to be a local file the control plane never saw."""
    import run_first_letters_positive_control as runner

    receipt = control_receipt()
    posted: list[tuple[str, str, dict]] = []

    class Recorder:
        def call(self, method, path, body=None):
            posted.append((method, path, body))
            return controlled.publish_control(body["receipt"])

    runner.publish_positive_control(Recorder(), receipt)
    assert posted[0][0] == "POST"
    assert posted[0][1] == f"/api/missions/{MISSION}/first-letters-control"
    readiness = controlled.readiness()
    assert readiness["control"]["content_sha256"] == receipt["content_sha256"]
    assert readiness["control"]["control_state"] == "CONTROL_PASS"


def test_a_failing_control_is_published_rather_than_withheld(controlled):
    """"The control failed" is evidence the gate must see."""
    import run_first_letters_positive_control as runner

    class Recorder:
        def call(self, _method, _path, body=None):
            return controlled.publish_control(body["receipt"])

    runner.publish_positive_control(Recorder(), control_receipt(failing_at="P2"))
    control = controlled.readiness()["control"]
    assert control["control_state"] == "CONTROL_FAILED"
    assert control["evidence_status"] == "CURRENT"


def test_republishing_the_same_control_is_idempotent(controlled):
    receipt = control_receipt()
    assert controlled.publish_control(receipt) == controlled.publish_control(
        copy.deepcopy(receipt), by="somebody-else")
    root = controlled.directory / "evidence" / "first-letters-control"
    assert sorted(path.name for path in root.iterdir()) == sorted(
        ["attestations", f"{receipt['content_sha256']}.json"])
    attestation = json.loads(
        (root / "attestations" / f"{receipt['content_sha256']}.json"
         ).read_text(encoding="utf-8"))
    # The first attestation stands. A second publisher does not get to inherit
    # or overwrite who vouched for the run.
    assert attestation["published_by"] == "campaign-owner"


def test_readiness_names_who_vouched_for_the_stage_outcomes(controlled):
    """The server verifies the bindings; a person ran the nine boundaries."""
    controlled.publish_control(control_receipt(), by="campaign-owner")
    control = controlled.readiness()["control"]
    assert control["stage_outcomes_attested_by"] == "campaign-owner"
    assert control["published_at_utc"]
    assert control["control_pass_is_independent_validation"] is False


# ---------------------------------------------------------------------------
# Per-scroll preflight
# ---------------------------------------------------------------------------

def test_readiness_reports_a_missing_preflight_per_scroll(controlled):
    controlled.publish_control(control_receipt())
    row = scroll_of(controlled.readiness())
    assert row["preflight"]["evidence_status"] == "MISSING"
    assert "PREFLIGHT_MISSING" in codes(row["blockers"])
    assert row["queue_admitted"] is False


def test_readiness_reports_a_census_preflight_and_its_measurement(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=2)
    row = scroll_of(controlled.readiness())
    assert row["preflight"]["evidence_status"] == "CURRENT"
    assert row["preflight"]["measurement_kind"] == "CENSUS"
    assert row["preflight"]["private_receipt_sha256"] == "a" * 64
    assert "PREFLIGHT_MISSING" not in codes(row["blockers"])


def test_readiness_reports_an_estimated_preflight_as_an_estimate(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(
        population=100, usable_cells=2, sampled_cells=25,
        measurement_kind="ESTIMATE")
    row = scroll_of(controlled.readiness())
    assert row["preflight"]["evidence_status"] == "CURRENT"
    assert row["preflight"]["measurement_kind"] == "ESTIMATE"


def test_a_stale_preflight_blocks_the_scroll(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(
        population=10, usable_cells=2, evidence_status="STALE")
    row = scroll_of(controlled.readiness())
    assert row["preflight"]["evidence_status"] == "STALE"
    assert "PREFLIGHT_NOT_CURRENT" in codes(row["blockers"])
    assert row["queue_admitted"] is False


# ---------------------------------------------------------------------------
# Per-scroll task budget
# ---------------------------------------------------------------------------

def test_a_current_preflight_without_a_budget_blocks_the_scroll(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=2)
    row = scroll_of(controlled.readiness())
    assert row["budget"]["evidence_status"] == "MISSING"
    assert "BUDGET_MISSING" in codes(row["blockers"])


def test_a_justified_budget_completes_the_scroll_gate(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=2)
    budget = controlled.publish_budget(evidence=controlled.preflight)
    row = scroll_of(controlled.readiness())
    assert row["budget"]["evidence_status"] == "CURRENT"
    assert row["budget"]["decision"] == "CONTINUE"
    assert row["budget"]["planned_task_count"] == budget["planned_task_count"]
    assert row["budget"]["receipt_sha256"] == budget["receipt_sha256"]
    assert row["budget"]["binds_current_preflight"] is True
    assert row["blockers"] == []
    assert row["queue_admitted"] is True
    assert controlled.readiness()["queue_admitted"] is True


def test_a_budget_clipped_by_the_frozen_compute_cap_says_so(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=100, usable_cells=1)
    budget = controlled.publish_budget(evidence=controlled.preflight, cap=10)
    row = scroll_of(controlled.readiness())
    assert budget["requested_task_count"] == 95
    assert row["budget"]["planned_task_count"] == 10
    assert row["budget"]["requested_task_count"] == 95
    assert row["budget"]["clipped_by_compute_cap"] is True
    assert row["budget"]["target_detection_probability_met"] is False
    assert "BUDGET_CLIPPED_BELOW_TARGET_DETECTION_PROBABILITY" in codes(
        row["advisories"])
    assert row["queue_admitted"] is True


def test_a_budget_bound_to_an_older_preflight_does_not_authorize_queueing(
    controlled,
):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=2)
    controlled.publish_budget(evidence=controlled.preflight)
    moved = preflight(population=12, usable_cells=3)
    moved["private_receipt_sha256"] = "b" * 64
    controlled.preflight = moved
    row = scroll_of(controlled.readiness())
    assert row["budget"]["binds_current_preflight"] is False
    assert "BUDGET_DOES_NOT_BIND_CURRENT_PREFLIGHT" in codes(row["blockers"])
    assert row["queue_admitted"] is False


def test_a_census_with_no_usable_cells_refuses_the_current_source(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=0)
    controlled.publish_budget(evidence=controlled.preflight)
    row = scroll_of(controlled.readiness())
    assert row["budget"]["decision"] == "DO_NOT_QUEUE_CURRENT_SOURCE"
    assert "BUDGET_DOES_NOT_AUTHORIZE_QUEUEING" in codes(row["blockers"])
    assert "CHANGE_CANDIDATE_SOURCE_OR_POLICY" in row["allowed_actions"]
    assert row["queue_admitted"] is False


# ---------------------------------------------------------------------------
# The candidate-starvation pause
# ---------------------------------------------------------------------------

def ready(controlled) -> None:
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=2)
    controlled.publish_budget(evidence=controlled.preflight)


def starvation(no_m7: int, *, count: int = 8, platform: int = 0):
    """``count`` scientific-terminal attempts, ``platform`` failures among them.

    The platform failures sit inside the block rather than after it: a worker
    that dies partway through a wave is the case that must not quietly shrink
    or fill the denominator, and one that dies after the block already closed
    would never have tested that.
    """
    authority = admission(STORED_SCROLL, 0, count=count + platform)
    plan = (["empty"] * no_m7 + ["platform"] * platform
            + ["found"] * (count - no_m7))
    rows = []
    for rank, kind in enumerate(plan):
        if kind == "platform":
            row = attempt(rank, state="GROW_FAILED", raw_m7=None,
                          failure_class="WORKER_FAILURE",
                          sample_id=STORED_SCROLL)
            row["cell_id"] = authority["prefix_cell_ids"][rank]
            row["campaign_budget"] = {
                key: value for key, value in authority.items()
                if key != "registered_at_utc"}
            row["campaign_budget"]["selection_rank"] = rank
        else:
            row = admitted_attempt(rank, authority, rank,
                                   raw_m7=(0 if kind == "empty" else 3))
        rows.append(row)
    return rows, [authority]


def test_six_of_eight_raw_m7_empty_attempts_continue(controlled, monkeypatch):
    ready(controlled)
    rows, admissions = starvation(6)
    monkeypatch.setattr(controlled.app, "fleet_store_read_only",
                        lambda: DecisionStore(rows, admissions))
    readiness = controlled.readiness()
    assert readiness["pause"]["active"] is False
    assert readiness["pause"]["decision"] == "CONTINUE"
    evaluated = readiness["pause"]["latest_evaluation"]
    assert evaluated["decision"] == "CONTINUE"
    assert evaluated["no_m7_numerator"] == 6
    assert evaluated["scientific_terminal_denominator"] == 8
    assert "CAMPAIGN_PAUSED" not in codes(readiness["blockers"])
    assert readiness["queue_admitted"] is True


def test_seven_of_eight_raw_m7_empty_attempts_pause_the_campaign(controlled,
                                                                 monkeypatch):
    ready(controlled)
    rows, admissions = starvation(7)
    monkeypatch.setattr(controlled.app, "fleet_store_read_only",
                        lambda: DecisionStore(rows, admissions))
    readiness = controlled.readiness()
    assert readiness["pause"]["active"] is True
    assert readiness["pause"]["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert readiness["pause"]["no_m7_numerator"] == 7
    assert len(readiness["pause"]["trigger_attempt_ids"]) == 7
    assert "CAMPAIGN_PAUSED" in codes(readiness["blockers"])
    assert readiness["queue_admitted"] is False
    assert "CREATE_MATERIALLY_CHANGED_VERSIONED_STRATEGY" in (
        readiness["pause"]["allowed_next_actions"])


def test_an_unreadable_pause_gate_blocks_rather_than_admits(controlled,
                                                            monkeypatch):
    """An unreachable control plane must not read as "not paused"."""
    ready(controlled)

    def unreachable():
        raise controlled.app.HTTPException(409, "CX_DB is not set")

    monkeypatch.setattr(controlled.app, "fleet_store_read_only", unreachable)
    readiness = controlled.readiness()
    assert readiness["pause"]["available"] is False
    assert "PAUSE_STATE_UNREADABLE" in codes(readiness["blockers"])
    assert readiness["queue_admitted"] is False
    with pytest.raises(Exception, match="PAUSE_STATE_UNREADABLE"):
        controlled.app._require_first_letters_queue_readiness(MISSION, SCROLL)


def test_a_store_that_cannot_derive_a_campaign_decision_blocks(controlled,
                                                               monkeypatch):
    ready(controlled)
    monkeypatch.setattr(controlled.app, "fleet_store_read_only", object)
    readiness = controlled.readiness()
    assert "PAUSE_STATE_UNREADABLE" in codes(readiness["blockers"])
    assert readiness["queue_admitted"] is False


def test_platform_failures_never_fill_the_scientific_denominator(controlled,
                                                                 monkeypatch):
    ready(controlled)
    rows, admissions = starvation(6, platform=2)
    monkeypatch.setattr(controlled.app, "fleet_store_read_only",
                        lambda: DecisionStore(rows, admissions))
    readiness = controlled.readiness()
    evaluated = readiness["pause"]["latest_evaluation"]
    assert evaluated["scientific_terminal_denominator"] == 8
    assert evaluated["no_m7_numerator"] == 6
    assert evaluated["excluded_attempt_count"] == 2
    assert [row["reason"] for row in evaluated["excluded_attempts"]] == [
        "WORKER_FAILURE", "WORKER_FAILURE"]
    assert evaluated["decision"] == "CONTINUE"
    assert readiness["pause"]["active"] is False
    assert readiness["queue_admitted"] is True


def test_a_new_versioned_strategy_lifts_the_pause_on_the_real_store(
    controlled, monkeypatch, tmp_path,
):
    """Not by clearing anything: by adding an authorized successor policy."""
    from fleet.store import FleetStore

    ready(controlled)
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    authority = budget_admission(tmp_path / "campaign-mission", source_id)
    store.register_campaign_budget_admission(authority)
    store.create_tasks(campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, authority)
         for cell_id in authority["prefix_cell_ids"]], authority))
    for index in range(8):
        claim = store.claim(f"worker-{index}", 60)
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 2 if index == 7 else 0))
    monkeypatch.setattr(controlled.app, "fleet_store_read_only", lambda: store)

    paused = controlled.readiness()
    assert paused["pause"]["active"] is True
    assert paused["pause"]["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert "CAMPAIGN_PAUSED" in codes(paused["blockers"])
    assert paused["queue_admitted"] is False
    with pytest.raises(Exception, match="PAUSE_CANDIDATE_STARVATION"):
        controlled.app._require_first_letters_queue_readiness(MISSION, SCROLL)

    decision = store.campaign_decisions(
        mission_id="first-letters",
        policy_version=authority["execution_bindings"]["policy_version"])[-1]
    resumed, _tasks = resumed_sample_admission(store, authority)
    authorization = resume_authorization(authority, resumed, decision)
    store.register_campaign_resume_principal_attestation(
        authorization, authenticated_principal="campaign-owner")
    store.register_campaign_budget_admission(
        resumed, resume_authorization=authorization)

    resumed_readiness = controlled.readiness()
    assert resumed_readiness["pause"]["active"] is False
    assert resumed_readiness["pause"]["policy_version"] == "search-v2"
    assert resumed_readiness["pause"]["policy_chain"] == [
        authority["execution_bindings"]["policy_version"], "search-v2"]
    assert "CAMPAIGN_PAUSED" not in codes(resumed_readiness["blockers"])
    assert controlled.app._require_first_letters_queue_readiness(
        MISSION, SCROLL)["queue_admitted"] is True


# ---------------------------------------------------------------------------
# The P1 admission authority
# ---------------------------------------------------------------------------

def test_p1_creation_is_refused_without_a_current_passing_control(controlled):
    controlled.preflight = preflight(population=10, usable_cells=2)
    controlled.publish_budget(evidence=controlled.preflight)
    with pytest.raises(Exception, match="control"):
        controlled.app._require_first_letters_queue_readiness(MISSION, SCROLL)


def test_p1_creation_is_refused_without_a_current_preflight(controlled):
    controlled.publish_control(control_receipt())
    with pytest.raises(Exception, match="preflight"):
        controlled.app._require_first_letters_queue_readiness(MISSION, SCROLL)


def test_p1_creation_is_refused_without_a_justified_budget(controlled):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=2)
    with pytest.raises(Exception, match="budget"):
        controlled.app._require_first_letters_queue_readiness(MISSION, SCROLL)


def test_p1_creation_is_refused_while_the_campaign_is_paused(controlled,
                                                             monkeypatch):
    ready(controlled)
    rows, admissions = starvation(7)
    monkeypatch.setattr(controlled.app, "fleet_store_read_only",
                        lambda: DecisionStore(rows, admissions))
    with pytest.raises(Exception, match="pause"):
        controlled.app._require_first_letters_queue_readiness(MISSION, SCROLL)


def test_p1_creation_is_admitted_when_all_three_gates_hold(controlled):
    ready(controlled)
    admitted = controlled.app._require_first_letters_queue_readiness(
        MISSION, SCROLL)
    assert admitted["queue_admitted"] is True
    assert admitted["scroll"]["sample_id"] == STORED_SCROLL
    assert admitted["scroll"]["budget"]["decision"] == "CONTINUE"


def queue_request(app, **overrides):
    return app.SegmentationRunRequest(
        sample_id=SCROLL, mission_id=MISSION, backend="vc3d", **overrides)


def test_the_queue_route_refuses_a_controlled_wave_before_touching_anything(
    controlled, monkeypatch,
):
    """The refusal costs no database read, no file write and no subprocess."""
    app = controlled.app
    monkeypatch.setattr(app, "require_write_sample",
                        lambda *_args: STORED_SCROLL)
    forbidden: list[str] = []
    monkeypatch.setattr(app.subprocess, "run",
                        lambda *_a, **_k: forbidden.append("subprocess"))
    monkeypatch.setattr(app, "fleet_store",
                        lambda: forbidden.append("writable store"))
    monkeypatch.setattr(app, "module_disabled",
                        lambda *_a: forbidden.append("module gate") or False)
    before = sorted(p.name for p in controlled.directory.rglob("*"))
    with pytest.raises(Exception) as refusal:
        app.api_queue_segmentation(queue_request(app), http=None)
    assert "CONTROL_EVIDENCE_MISSING" in json.dumps(refusal.value.detail)
    assert forbidden == []
    assert sorted(p.name for p in controlled.directory.rglob("*")) == before


def test_the_queue_route_admits_a_controlled_wave_once_every_gate_holds(
    controlled, monkeypatch,
):
    ready(controlled)
    app = controlled.app
    monkeypatch.setattr(app, "require_write_sample",
                        lambda *_args: STORED_SCROLL)
    reached = []
    monkeypatch.setattr(app, "module_disabled",
                        lambda *_a: reached.append("module") or False)
    with pytest.raises(Exception) as refusal:
        app.api_queue_segmentation(queue_request(app), http=None)
    # Past the campaign gate and into the ordinary preconditions after it.
    assert reached == ["module", "module"]
    assert "CONTROL_EVIDENCE" not in json.dumps(refusal.value.detail)
    assert "CAMPAIGN_PAUSED" not in json.dumps(refusal.value.detail)


def test_an_uncontrolled_mission_keeps_its_existing_p1_workflow(tmp_path,
                                                                monkeypatch):
    pytest.importorskip("fastapi")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("CX_RUNS", str(runs))
    import panel.app as app
    monkeypatch.setattr(app, "RUNS", runs)
    mission_contract.create(runs, mission_id="ordinary", name="ordinary",
                            scrolls=[SCROLL], created_by="alice")
    assert app._require_first_letters_queue_readiness("ordinary", SCROLL) is None
    assert app._require_first_letters_queue_readiness(None, SCROLL) is None


# ---------------------------------------------------------------------------
# What the answer must never say or offer
# ---------------------------------------------------------------------------

def test_readiness_offers_no_acceptance_gate_bypass(controlled):
    ready(controlled)
    readiness = controlled.readiness()
    offered = json.dumps(readiness).lower()
    for forbidden in ("allow_unvalidated", "bypass", "override",
                      "force_queue", "skip_gate", "accept_anyway"):
        assert forbidden not in offered
    assert "allow_unvalidated" not in readiness


def test_readiness_never_infers_absence_of_ink_from_a_bounded_negative(
    controlled,
):
    controlled.publish_control(control_receipt())
    controlled.preflight = preflight(population=10, usable_cells=0)
    controlled.publish_budget(evidence=controlled.preflight)
    readiness = controlled.readiness()
    said = json.dumps(readiness).lower()
    # Affirmative absence, not the policy's explicit refusal to claim it. The
    # frozen small-surface non-claims legitimately contain the phrase "no ink"
    # inside "is not a finding of no ink", which is the opposite of a claim.
    for forbidden in ("contains no ink", "has no ink", "there is no ink",
                      "ink absent", "absence of ink", "no ink was found",
                      "holds no text", "contains no letters"):
        assert forbidden not in said
    assert any("not evidence" in claim.lower()
               for claim in readiness["non_claims"])
    assert all(
        "not a finding of no ink" in claim.lower() or "no ink" not in claim.lower()
        for claim in readiness["small_surfaces"]["explicit_non_claims"])


def test_readiness_is_content_addressed_and_stable(controlled):
    ready(controlled)
    first = controlled.readiness()
    second = controlled.readiness()
    assert first["readiness_sha256"] == second["readiness_sha256"]
    assert first["readiness_sha256"] == content_sha256({
        key: value for key, value in first.items()
        if key not in {"generated_at_utc", "readiness_sha256"}})


def test_small_surface_diagnostics_are_reported_without_being_required(
    controlled,
):
    ready(controlled)
    diagnostics = controlled.readiness()["small_surfaces"]
    assert diagnostics["minimum_area_cm2"] == 0.1
    assert diagnostics["profile_id"] == "small-surface-routing@1.0.0"
    assert diagnostics["is_absence_evidence"] is False
    assert "queue_admitted" not in diagnostics


def test_a_tiny_surface_is_isolated_and_a_standard_one_advances(controlled,
                                                                monkeypatch):
    """The area floor decides the route, and says nothing about content."""
    ready(controlled)
    app = controlled.app
    monkeypatch.setattr(app, "DSN", "postgresql://fixture.invalid/fleet")
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})
    monkeypatch.setitem(sys.modules, "psycopg", FakeSurfaceRows([
        ("surface-tiny", STORED_SCROLL, 0.004),
        ("surface-canonical", STORED_SCROLL, 1.25),
    ]))
    diagnostics = controlled.readiness()["small_surfaces"]
    assert diagnostics["surfaces_available"] is True
    assert diagnostics["diagnostic_count"] == 1
    assert diagnostics["standard_count"] == 1
    tiny = next(row for row in diagnostics["surfaces"]
                if row["surface_id"] == "surface-tiny")
    canonical = next(row for row in diagnostics["surfaces"]
                     if row["surface_id"] == "surface-canonical")
    assert tiny["route"] == "SMALL_SURFACE_DIAGNOSTIC"
    assert tiny["is_absence_evidence"] is False
    assert tiny["ink_claim"] == "NONE_MADE"
    assert "no claim either way about what is written on it" in tiny["why"]
    assert canonical["route"] == "STANDARD_QC_PENDING"
    assert diagnostics["promotion_in_place"] == "PROHIBITED"


def test_the_readiness_answer_survives_small_surface_routing_being_absent(
    controlled, monkeypatch,
):
    """Task 8 owns that module; readiness reports its absence, never guesses."""
    ready(controlled)
    import fleet.surface_routing as routing

    monkeypatch.setattr(routing, "load_policy", _raise_missing_policy)
    readiness = controlled.readiness()
    assert readiness["small_surfaces"]["available"] is False
    assert "unavailable" in readiness["small_surfaces"]["reason"]
    assert readiness["small_surfaces"]["is_absence_evidence"] is False
    assert readiness["queue_admitted"] is True


def _raise_missing_policy(*_args, **_kwargs):
    raise OSError("the routing policy is not deployed here")


class FakeSurfaceRows:
    """A psycopg stand-in returning one fixed surface query result."""

    def __init__(self, rows):
        self.rows = list(rows)

    def connect(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def cursor(self):
        return self

    def execute(self, *_args):
        return self

    def fetchall(self):
        return list(self.rows)


def test_the_readiness_route_is_reachable_and_read_only(controlled):
    from fastapi.testclient import TestClient
    from framework.contracts import auth as auth_contract

    ready(controlled)
    app = controlled.app
    app.AUTH_ROOT = controlled.runs.parent / "auth"
    app.AUDIT_ROOT = controlled.runs.parent / "audit"
    auth_contract.create_user(app.AUTH_ROOT, "tester", "a-long-enough-one")
    client = TestClient(app.app)
    assert client.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one"}).status_code == 200
    response = client.get(f"/api/missions/{MISSION}/first-letters-readiness")
    assert response.status_code == 200, response.text
    assert response.json()["readiness_sha256"] == (
        controlled.readiness()["readiness_sha256"])
    assert client.post(
        f"/api/missions/{MISSION}/first-letters-readiness").status_code == 405


# ---------------------------------------------------------------------------
# The transactional wave orchestrator
# ---------------------------------------------------------------------------

def campaign_module():
    import run_first_letters_campaign  # noqa: PLC0415

    return run_first_letters_campaign


class FakePanel:
    """The HTTP boundary, and nothing inside the orchestrator, is faked."""

    def __init__(self, *, budget: int = 3, waves: int = 4,
                 pause_after: int | None = None,
                 packet_after: int | None = None,
                 ambiguous_on: int | None = None,
                 blockers: list[dict] | None = None):
        self.budget = budget
        self.waves = waves
        self.pause_after = pause_after
        self.packet_after = packet_after
        self.ambiguous_on = ambiguous_on
        self.blockers = blockers or []
        self.posts: list[dict] = []
        self.active: list[str] = []
        self.packets: list[dict] = []
        self.attempts = 0
        self.wave = 0
        self.readiness_reads = 0

    # -- the two reads --------------------------------------------------
    def _readiness(self) -> dict:
        self.readiness_reads += 1
        if self.active:
            # One task settles per observation, so a caller that does not wait
            # sees a queue that is still moving.
            self.active.pop(0)
            self.attempts += 1
        paused = self.pause_after is not None and self.wave >= self.pause_after
        answer = {
            "schema": "campaignx.first_letters_readiness.v1",
            "mission_id": MISSION, "controlled": True,
            "deployed_revision": REVISION,
            "blockers": list(self.blockers),
            "queue_admitted": not self.blockers and not paused,
            "pause": {"available": True, "active": paused,
                      "decision": ("PAUSE_CANDIDATE_STARVATION" if paused
                                   else "CONTINUE"),
                      "no_m7_numerator": 7 if paused else 1,
                      "scientific_terminal_denominator": 8 if paused else 1,
                      "trigger_attempt_ids": [], "allowed_next_actions": []},
            "queue": {"available": True, "active_task_ids": list(self.active),
                      "attempt_count": self.attempts,
                      "task_count": self.attempts + len(self.active)},
            "scrolls": [{
                "sample_id": STORED_SCROLL, "requested_sample_id": SCROLL,
                "queue_admitted": not self.blockers and not paused,
                "blockers": [],
                "budget": {"receipt_sha256": "c" * 64, "decision": "CONTINUE",
                           "planned_task_count": self.budget,
                           "requested_task_count": self.budget},
            }],
        }
        answer["readiness_sha256"] = content_sha256(answer)
        return answer

    def _enqueue(self, body: dict) -> dict:
        self.wave += 1
        if self.ambiguous_on == self.wave:
            from panel_client import AmbiguousMutationError  # noqa: PLC0415
            raise AmbiguousMutationError(
                "POST", "/api/segmentation/runs", "HTTP 504: gateway timeout")
        self.posts.append(copy.deepcopy(body))
        count = int(body["max_tasks"])
        self.active = [f"wave{self.wave}-task{index}" for index in range(count)]
        if self.packet_after is not None and self.wave >= self.packet_after:
            self.packets.append({
                "artifact_id": f"packet-{self.wave}", "kind": "vetting-packet",
                "phase": "P7", "content_sha256": f"{self.wave:064x}"})
        return {"inserted": count, "generated": count}

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        if method == "GET" and "first-letters-readiness" in path:
            return self._readiness()
        if method == "GET" and "/artifacts" in path:
            return {"artifacts": list(self.packets)}
        if method == "POST" and path == "/api/segmentation/runs":
            return self._enqueue(body or {})
        raise AssertionError(f"the orchestrator called {method} {path}")


def run(panel, **overrides):
    arguments = {"mission_id": MISSION, "sample_id": SCROLL,
                 "maximum_waves": 3, "poll_seconds": 0, "wait_minutes": 5,
                 "sleep": lambda _seconds: None}
    arguments.update(overrides)
    return campaign_module().run_campaign(panel, **arguments)


def test_a_wave_enqueues_exactly_the_budget_and_never_more(controlled):
    panel = FakePanel(budget=3)
    ledger = run(panel, maximum_waves=2)
    assert [post["max_tasks"] for post in panel.posts] == [3, 3]
    assert all(post["mission_id"] == MISSION for post in panel.posts)
    assert all(post["task_budget_receipt_sha256"] == "c" * 64
               for post in panel.posts)
    assert [wave["enqueued_task_count"] for wave in ledger["waves"]] == [3, 3]
    assert all(wave["enqueued_task_count"] <= wave["budgeted_task_count"]
               for wave in ledger["waves"])
    assert ledger["stopped_because"] == "WAVE_LIMIT_REACHED"


def test_a_wave_waits_for_every_task_to_reach_terminal_before_the_next(
    controlled,
):
    panel = FakePanel(budget=4)
    ledger = run(panel, maximum_waves=2)
    for wave in ledger["waves"]:
        assert wave["reached_terminal"] is True
        assert wave["active_task_ids_at_close"] == []
        assert wave["closing_attempt_count"] > wave["baseline_attempt_count"]
    # Four tasks per wave, one settling per observation: the orchestrator
    # cannot have posted the second wave before draining the first.
    assert panel.readiness_reads >= 8


def test_an_ambiguous_post_freezes_the_orchestrator_without_retrying(controlled):
    panel = FakePanel(budget=2, ambiguous_on=1)
    module = campaign_module()
    with pytest.raises(module.CampaignFrozen) as frozen:
        run(panel)
    assert panel.posts == []
    assert frozen.value.readback["mission_id"] == MISSION
    assert frozen.value.wave_index == 1
    assert "must not be retried" in str(frozen.value)


def test_an_ambiguous_post_on_a_later_wave_keeps_the_earlier_ledger(controlled):
    panel = FakePanel(budget=2, ambiguous_on=2)
    module = campaign_module()
    with pytest.raises(module.CampaignFrozen) as frozen:
        run(panel)
    assert len(panel.posts) == 1
    waves = frozen.value.ledger["waves"]
    assert [wave["outcome"] for wave in waves] == [
        "TERMINAL", "AMBIGUOUS_NOT_RETRIED"]
    assert waves[1]["enqueued_task_count"] is None
    assert frozen.value.ledger["stopped_because"] == "FROZEN_AMBIGUOUS_WAVE_POST"


def test_the_orchestrator_stops_when_the_pause_activates(controlled):
    panel = FakePanel(budget=2, pause_after=1)
    ledger = run(panel)
    assert len(panel.posts) == 1
    assert ledger["stopped_because"] == "PAUSE_CANDIDATE_STARVATION"
    assert ledger["waves"][0]["pause_after_wave"]["active"] is True


def test_the_orchestrator_stops_on_the_first_human_review_packet(controlled):
    panel = FakePanel(budget=2, packet_after=1)
    ledger = run(panel)
    assert len(panel.posts) == 1
    assert ledger["stopped_because"] == "HUMAN_REVIEW_PACKET_ROUTED"
    assert ledger["review_packet_sha256s"] == [f"{1:064x}"]
    assert any("blinded human" in claim for claim in ledger["non_claims"])


def test_the_orchestrator_refuses_to_open_a_blocked_campaign(controlled):
    panel = FakePanel(blockers=[{"code": "CONTROL_EVIDENCE_STALE",
                                 "scope": MISSION, "detail": "revision moved"}])
    ledger = run(panel)
    assert panel.posts == []
    assert ledger["stopped_because"] == "READINESS_BLOCKED"
    assert codes(ledger["blockers"]) == ["CONTROL_EVIDENCE_STALE"]


def test_the_ledger_is_content_bound_and_claims_nothing(controlled):
    panel = FakePanel(budget=1)
    ledger = run(panel, maximum_waves=1)
    assert ledger["schema"] == "campaignx.first_letters_campaign_ledger.v1"
    assert ledger["content_sha256"] == content_sha256(
        {key: value for key, value in ledger.items()
         if key != "content_sha256"})
    said = json.dumps(ledger).lower()
    for forbidden in ("allow_unvalidated", "bypass", "override",
                      "contains no ink", "absence of ink"):
        assert forbidden not in said
