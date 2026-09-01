"""The contract must describe the code, not a plan for it.

Every bug in this file's history was the same bug: a JSON declaration that the
running code contradicted, believed because it was written down. P8 declared a
`segments_dir` its runner has no flag for, so every queued job died at argparse.
P4 declared its PPM comes from P3; nothing in P3 makes one. P1 declared it
consumes the CT volume; it is handed the m7 surface prediction.

None of those are subtle. They survived because nothing compared the two.

This compares them. It is deliberately mechanical -- flags, argument names,
paths -- because the failure is mechanical: prose about a phase can be
aspirational and still be useful, but a required parameter that no flag accepts
is a job that cannot run.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "framework" / "stages" / "03-ink" / "fleet"))
from job_store import (  # noqa: E402
    EXACTLY_ONE_OF,
    DEFAULT_P4_LANE, P4_LANES, PHASE_LANES, PHASE_PARAMETERS, PHASE_REQUIRED,
    PHASE_RUNNERS, command_for,
)


def default_lane(phase: str) -> str | None:
    """The lane a job gets when it names none: the first registered."""
    return next(iter(PHASE_LANES.get(phase) or {}), None)


def profile_for(phase: str, lane: str | None = None) -> str:
    """A profile id the phase's lane will actually accept.

    A lane may pin the profiles it runs -- P1's fitter and P8's TIFXYZ merge
    both do -- and the queue refuses anything else. A contract test that sent a
    made-up id would be testing the refusal, not the contract.
    """
    spec = (PHASE_LANES.get(phase) or {}).get(lane or default_lane(phase) or "", {})
    return next(iter(spec.get("profiles") or ()), "p")


def sample_parameters(phase: str, lane: str | None = None) -> dict:
    """Plausible values for every parameter a phase declares.

    P4 has two renderers with different flags, so a lane has to be named or the
    builder is being asked to satisfy both at once.
    """
    values = {name: (f"value-of-{name}" if kind is not int else 7)
              for name, kind in PHASE_PARAMETERS[phase].items()}
    for name, kind in PHASE_PARAMETERS[phase].items():
        if kind is float:
            values[name] = 1.5
    # Names that must not appear together cannot all be set at once: setting
    # every parameter a phase declares would be asking the builder to satisfy a
    # contract that forbids it. The first of each group is kept -- for P5 that
    # is `tiff_dir`, which every ink lane reads; the alternatives belong to
    # particular lanes and have their own tests.
    for rule in EXACTLY_ONE_OF.get(phase, ()):
        if rule.get("lane") and rule["lane"] != (lane or DEFAULT_P4_LANE):
            continue
        for spare in rule["names"][1:]:
            values.pop(spare, None)
    # A lane name is not free text. Any phase that accepts one is given a lane
    # that exists, because `f"value-of-lane"` is refused by the queue before it
    # can build a command -- which reports the wrong failure for this test.
    if "lane" in values:
        values["lane"] = lane or default_lane(phase) or values["lane"]
    if phase == "P4":
        values["lane"] = lane or DEFAULT_P4_LANE
    elif phase == "P8":
        # P8 now has several fixed implementations.  This generic phase-level
        # contract exercises the historical default; lane-specific contracts
        # have their own tests and must never use a made-up lane name.
        values["lane"] = lane or "column-atlas"
    return values

PIPELINE = json.loads((ROOT / "framework/contracts/pipeline_phases.json").read_text())
PHASES = {entry["id"]: entry for entry in PIPELINE["phases"]}


def argparse_flags(path: Path) -> set[str]:
    """Every long flag a script's argparse accepts, read from its source.

    Parsed rather than executed: these scripts import torch, numpy and a bucket
    client at module scope, and a test that needs a GPU to check a flag name is
    a test nobody runs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not (isinstance(target, ast.Attribute) and target.attr == "add_argument"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                flags.add(str(argument.value))
    return flags


@pytest.mark.parametrize("phase", sorted(PHASE_RUNNERS))
def test_every_flag_the_queue_sends_exists_in_the_runner(phase):
    """The one that broke P8 in production.

    A required parameter the runner has no flag for is not a mismatch to
    reconcile later -- it is a job that dies at argparse after being claimed,
    burning an attempt.
    """
    runner = ROOT / PHASE_RUNNERS[phase]
    if not runner.exists():
        pytest.skip(f"{phase} runner is not vendored here: {PHASE_RUNNERS[phase]}")
    accepted = argparse_flags(runner)
    if not accepted:
        pytest.skip(f"{runner.name} does not use argparse")

    # Only the lanes that run a script in this repository can be checked against
    # an argparse; the default P4 lane is a binary from the VC3D runtime.
    lanes = [l for l, spec in P4_LANES.items() if spec.get("script")] if phase == "P4" else [None]
    for lane in lanes:
        lane_spec = P4_LANES.get(lane or "", {})
        sample_id = next(iter(lane_spec.get("sample_ids", ())), "S")
        job = {"job_id": "j", "sample_id": sample_id,
               "profile_id": profile_for(phase, lane), "phase": phase,
               "parameters": sample_parameters(phase, lane)}
        argv = command_for(job, runner=str(runner), output_dir="/tmp/out",
                           upstream_root=WORKER_UPSTREAM_ROOT)
        sent = {token for token in argv if token.startswith("--")}
        unknown = sorted(sent - accepted)
        assert not unknown, (
            f"{phase}{f' lane {lane}' if lane else ''} sends flags {runner.name} "
            f"does not accept: {unknown}. It accepts {sorted(accepted)}"
        )


# What a worker carrying the vendored architecture would report. Supplied here
# because two P5 lanes take it from the host rather than from the job, and the
# flag they build from it still has to exist in their argparse.
WORKER_UPSTREAM_ROOT = "/opt/villa/ink-detection"


@pytest.mark.parametrize("phase", sorted(PHASE_REQUIRED))
def test_required_parameters_all_reach_the_command(phase):
    """A required parameter that the builder ignores is a lie in the schema:
    the caller is made to supply something nothing uses."""
    runner = ROOT / PHASE_RUNNERS[phase]
    parameters = sample_parameters(phase)
    sample_id = parameters["sample"] if phase in {"P2", "P3"} else "S"
    job = {"job_id": "j", "sample_id": sample_id, "profile_id": profile_for(phase),
           "phase": phase, "parameters": parameters}
    argv = [str(token) for token in
            command_for(job, runner=str(runner), output_dir="/tmp/out",
                        upstream_root=WORKER_UPSTREAM_ROOT)]

    required = list(PHASE_REQUIRED[phase])
    if phase == "P4":
        # P4 requires per lane, because the two renderers take different things.
        required += list(P4_LANES[DEFAULT_P4_LANE]["required"])
    for name in required:
        expected = parameters[name]
        assert any(str(expected) in token for token in argv), (
            f"{phase} requires {name!r} and the command never carries it: {argv}"
        )


@pytest.mark.parametrize("phase_id", sorted(PHASES))
def test_a_phase_names_only_phases_that_exist_as_producers(phase_id):
    prerequisites = PHASES[phase_id].get("prerequisites") or {}
    for producer in prerequisites.get("produced_by") or []:
        assert producer in PHASES, f"{phase_id} names {producer}, which is not a phase"
        assert producer != phase_id, f"{phase_id} lists itself as its own producer"


@pytest.mark.parametrize("phase_id", sorted(PHASES))
def test_lives_in_points_at_something_real(phase_id):
    """A path in the contract that does not exist is how P2 came to name a file
    that holds only a loader for the gate, which lives in another stage."""
    for relative in PHASES[phase_id].get("lives_in") or []:
        assert (ROOT / relative).exists(), (
            f"{phase_id} lives_in {relative}, which is not in the repository"
        )


@pytest.mark.parametrize("phase_id", sorted(PHASES))
def test_a_phase_claiming_a_runner_has_one(phase_id):
    """`how_to_run` that describes a command nobody can issue reads as a phase
    you have not got round to using, rather than one that cannot be used."""
    entry = PHASES[phase_id]
    runnable = entry.get("runnable_here")
    assert runnable is not None, (
        f"{phase_id} does not say whether it can be run from this repository. "
        "Add runnable_here: true or false, with how_to_run matching."
    )
    if runnable:
        assert phase_id in PHASE_RUNNERS or entry.get("runner"), (
            f"{phase_id} claims to be runnable here but no runner is registered"
        )


def test_the_queueable_phases_are_exactly_those_with_runners():
    queueable = set(PHASE_PARAMETERS) & set(PHASE_REQUIRED) & set(PHASE_RUNNERS)
    assert set(PHASE_PARAMETERS) == queueable, (
        "a phase has parameters but no runner or no required set: "
        f"{sorted(set(PHASE_PARAMETERS) ^ queueable)}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# The behaviour the contract claims, actually enforced.
# --------------------------------------------------------------------------

def queue_p7(tmp_path, receipt: dict | None):
    """Try to enqueue a P7 screen for a map with this receipt beside it."""
    import json as json_module

    from job_store import JobRejected, validate_parameters

    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "probability.npy").write_bytes(b"not really an array")
    if receipt is not None:
        (run / "INK_PROFILE_RECEIPT.json").write_text(json_module.dumps(receipt))
    return validate_parameters, JobRejected, {
        "map_path": str(run / "probability.npy"),
        "bbox": "0,0,10,10",
        "px_um": 8.0,
    }


@pytest.mark.parametrize("verdict", ["DEGENERATE", "EMPTY"])
def test_p7_refuses_a_map_the_lane_could_not_read(tmp_path, verdict):
    """Screening finds shapes in noise perfectly well, which is the danger.

    The contract said "a probability map that passed P6" from the day it was
    written and nothing checked it.
    """
    validate, rejected, parameters = queue_p7(tmp_path, {"liveness": {"verdict": verdict}})
    with pytest.raises(rejected, match=verdict):
        validate(parameters, phase="P7")


def test_p7_accepts_a_live_map(tmp_path):
    validate, _rejected, parameters = queue_p7(tmp_path, {"liveness": {"verdict": "ALIVE"}})
    assert validate(parameters, phase="P7")["px_um"] == 8.0


def test_p7_refuses_a_map_with_no_receipt(tmp_path):
    """A map with no provenance has not passed anything."""
    validate, rejected, parameters = queue_p7(tmp_path, None)
    with pytest.raises(rejected, match="no INK_PROFILE_RECEIPT"):
        validate(parameters, phase="P7")


def test_p7_refuses_a_receipt_that_predates_the_check(tmp_path):
    """Absent is not the same as passing: those 95 runs never recorded one."""
    validate, rejected, parameters = queue_p7(tmp_path, {"statistics": {"p90": 0.5}})
    with pytest.raises(rejected, match="no liveness verdict"):
        validate(parameters, phase="P7")
