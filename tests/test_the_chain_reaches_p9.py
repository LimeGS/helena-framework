"""Every phase the queue can run, and the chains between them.

The pipeline is nine phases and the queue could express four of them as work
somebody could ask for from a browser. The rest were reachable only as a path
typed into a form: P7 took `map_path`, a file on whichever machine held it, and
P9 was in the contract with a runner named and no way to queue it at all.

What is held here is the shape of the chain -- that each phase can name its
predecessor's output rather than a path, and that naming it wrong is refused
rather than run on whatever the path happens to hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import (  # noqa: E402
    PHASE_PARAMETERS, PHASE_RUNNERS, JobRejected, command_for,
    phase_parameter_schema, validate_parameters,
)

QUEUEABLE = ("P2", "P3", "P4", "P5", "P7", "P8", "P9")


@pytest.mark.parametrize("phase", QUEUEABLE)
def test_every_queueable_phase_has_a_runner_that_is_here(phase):
    """A phase the panel offers and the worker cannot run is a button that
    fails after a GPU has been reserved."""
    if phase in ("P2", "P3"):
        pytest.skip("P2 and P3 are subcommands of the fleet CLI, checked elsewhere")
    runner = ROOT / PHASE_RUNNERS[phase]
    assert runner.is_file(), f"{phase} names {runner} and it is not here"


@pytest.mark.parametrize("phase", QUEUEABLE)
def test_every_queueable_phase_describes_its_parameters(phase):
    schema = phase_parameter_schema(phase)
    assert schema["fields"], f"{phase} offers no fields, so no form can draw it"
    assert set(PHASE_PARAMETERS[phase]) == {f["name"] for f in schema["fields"]}


# --------------------------------------------------------------------------
# P5 -> P7: the adjudication names the screening
# --------------------------------------------------------------------------

def test_a_screen_may_name_the_screening_it_is_about():
    clean = validate_parameters(
        {"screening_of": "p5-01facc430f694c", "bbox": "0,0,64,64", "px_um": 2.399}, "P7")
    assert clean["screening_of"] == "p5-01facc430f694c"
    assert "map_path" not in clean


def test_naming_both_a_path_and_a_screening_is_refused(tmp_path):
    with pytest.raises(JobRejected) as refused:
        validate_parameters({"map_path": str(tmp_path / "p.npy"),
                             "screening_of": "p5-1", "bbox": "0,0,1,1",
                             "px_um": 2.4}, "P7")
    assert "exactly one" in str(refused.value)


def test_an_unfetched_map_never_reaches_the_screen():
    job = {"phase": "P7", "sample_id": "PHerc0139",
           "parameters": {"screening_of": "p5-1", "bbox": "0,0,1,1", "px_um": 2.4}}
    with pytest.raises(JobRejected) as refused:
        command_for(job, runner="unused", output_dir="/runs/p7-1")
    assert "did not run" in str(refused.value)


def test_a_dead_map_is_refused_at_the_queue(tmp_path):
    """P6's verdict is a field of the P5 receipt, and the contract's own note is
    that nothing downstream checked it: screening finds shapes in noise
    perfectly well."""
    import json

    map_path = tmp_path / "probability.npy"
    map_path.write_bytes(b"not really an array")
    (tmp_path / "INK_PROFILE_RECEIPT.json").write_text(
        json.dumps({"liveness": {"verdict": "DEGENERATE"}}))
    with pytest.raises(JobRejected) as refused:
        validate_parameters({"map_path": str(map_path), "bbox": "0,0,1,1",
                             "px_um": 2.4}, "P7")
    assert "not ALIVE" in str(refused.value)


# --------------------------------------------------------------------------
# P9: the plates
# --------------------------------------------------------------------------

def test_p9_is_something_the_queue_can_run():
    """It was in the contract with a runner named and no parameters, so the one
    phase that produces something a person reads could not be asked for."""
    clean = validate_parameters({"scroll": "PHerc0139", "out_dir": "/runs/plates",
                                 "order_path": "/runs/wrap_radial.json"}, "P9")
    argv = command_for({"phase": "P9", "sample_id": "PHerc0139", "parameters": clean},
                       runner=str(ROOT / PHASE_RUNNERS["P9"]), output_dir="/runs/p9-1")
    assert argv[argv.index("--scroll") + 1] == "PHerc0139"
    assert argv[argv.index("--out") + 1] == "/runs/plates"
    assert argv[argv.index("--order") + 1] == "/runs/wrap_radial.json"
    # It fetches the official maps itself, so it is given somewhere to stage them.
    assert argv[argv.index("--work") + 1].startswith("/runs/p9-1")


def test_p9_needs_somewhere_to_write():
    with pytest.raises(JobRejected):
        validate_parameters({"scroll": "PHerc0139",
                             "order_path": "/runs/wrap_radial.json"}, "P9")


def test_p9_requires_one_measured_order_source():
    with pytest.raises(JobRejected):
        validate_parameters({"scroll": "PHerc0139", "out_dir": "/runs/plates"}, "P9")
    clean = validate_parameters({"scroll": "PHerc0139", "out_dir": "/runs/plates",
                                 "ordering_of": "p8-order"}, "P9")
    assert clean["ordering_of"] == "p8-order"
    with pytest.raises(JobRejected):
        command_for({"phase": "P9", "sample_id": "PHerc0139", "parameters": clean},
                    runner="unused", output_dir="/runs/p9")


def test_the_runner_each_phase_names_takes_the_flags_it_is_given():
    """Every P8 job died at argparse because the builder passed --segments and
    the runner takes --scroll. The same check, for every phase the queue runs."""
    cases = {
        "P8": {"scroll": "PHerc0139", "out_path": "/runs/wrap.json"},
        "P9": {"scroll": "PHerc0139", "out_dir": "/runs/plates",
               "order_path": "/runs/wrap_radial.json"},
        "P7": {"map_path": "/dev/null", "bbox": "0,0,1,1", "px_um": 2.4},
    }
    for phase, parameters in cases.items():
        runner = ROOT / PHASE_RUNNERS[phase]
        argv = command_for({"phase": phase, "sample_id": "PHerc0139",
                            "parameters": parameters},
                           runner=str(runner), output_dir="/runs/x")
        source = runner.read_text()
        for token in argv:
            if token.startswith("--"):
                assert f'"{token}"' in source, f"{runner.name} takes no {token}"


def test_the_screen_is_given_a_file_to_write_and_not_a_directory():
    """vet_map's --out is "path to write the verdict JSON". Handed a directory
    it wrote the verdict *as* that directory, and the card that followed died on
    "Not a directory" -- a P7 job that reached the runner and produced nothing."""
    argv = command_for({"phase": "P7", "sample_id": "PHerc0139",
                        "parameters": {"map_path": "/m.npy", "bbox": "0,0,1,1",
                                       "px_um": 2.4}},
                       runner="unused", output_dir="/runs/p7-1")
    assert argv[argv.index("--out") + 1].endswith(".json")
    assert argv[argv.index("--card") + 1].endswith("VETTING_CARD.md")
    assert argv[argv.index("--out") + 1] != argv[argv.index("--card") + 1]
