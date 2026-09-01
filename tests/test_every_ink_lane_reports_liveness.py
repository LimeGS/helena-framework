"""Every lane says whether its map carries a decision, or the job fails.

`assess_liveness` was called by one adapter out of four. The other three --
timesformer, canonical2um, 3d-dino -- produced probability maps that nothing
examined, and `ink_worker` stored the resulting `None` beside a job it called
"succeeded".

Proven on gpu-1 on 2026-07-31, not inferred: a real P5 job through
timesformer-gp-scroll1-screening@1.1.0 finished `succeeded` with
`result.liveness = None`. The end-to-end suite caught it on the first run of the
heavy stage.

What makes it worth a gate rather than a note: a map that is one value
everywhere is what a wrong depth window or a back-to-front slab produces, and it
exits zero. The checkpoint hash matches, the profile is satisfied, the receipt is
well formed. Only the shape of the distribution gives it away, and for three of
four lanes nothing was looking at the shape.

There was a second half, quieter than the first. The worker read
`INK_PROFILE_RECEIPT.json` -- run_ink.py's filename -- so even once the other
adapters computed a verdict, the worker would have gone on storing null: the
file it opened was never written by those lanes.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework/stages/03-ink/scripts"
ADAPTERS = sorted(SCRIPTS.glob("run_ink*.py"))

sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))


def _calls(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == name
    )


def test_every_lane_on_disk_is_checked() -> None:
    """A glob that matched nothing would make every assertion below vacuous.

    Five since the 9 um lane joined: the count is asserted rather than the
    glob trusted, so a lane added without liveness is a failure here and not a
    silently shorter parametrisation."""
    assert len(ADAPTERS) == 5, [p.name for p in ADAPTERS]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda p: p.name)
def test_the_lane_assesses_its_own_map(adapter: Path) -> None:
    assert _calls(adapter, "assess_liveness") >= 1, (
        f"{adapter.name} produces a probability map and never asks whether it "
        "carries a decision. A dead head answers every input with the same "
        "number and exits zero."
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda p: p.name)
def test_the_lane_refuses_a_map_that_carries_no_decision(adapter: Path) -> None:
    """Assessing and recording is not a gate. Exit 3 is the gate: the worker
    reads it as a refusal rather than a crash."""
    assert _calls(adapter, "refuse_if_not_alive") >= 1, (
        f"{adapter.name} records a verdict and acts on none"
    )
    source = adapter.read_text(encoding="utf-8")
    assert "--on-degenerate" in source, (
        f"{adapter.name} cannot be told to continue past a degenerate map, so a "
        "diagnostic run has no way through"
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda p: p.name)
def test_the_verdict_reaches_the_receipt(adapter: Path) -> None:
    """The worker reads the receipt, not the adapter's stdout."""
    source = adapter.read_text(encoding="utf-8")
    assert '"liveness"' in source, f"{adapter.name} keeps the verdict to itself"


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------

def test_the_worker_looks_for_the_receipt_the_lane_actually_writes() -> None:
    """Three adapters, three filenames, and the worker knew one of them."""
    import job_store

    named = {spec.get("receipt") for spec in job_store.INK_ADAPTERS.values()}
    named.discard(None)
    assert len(named) >= 3, f"the registry records only {named}"

    for adapter, spec in job_store.INK_ADAPTERS.items():
        receipt = spec.get("receipt")
        if not receipt:
            continue
        source = (ROOT / adapter).read_text(encoding="utf-8")
        assert receipt in source, (
            f"the registry says {adapter} writes {receipt} and it does not"
        )

    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    assert 'output / "INK_PROFILE_RECEIPT.json"' not in worker, (
        "the worker hardcodes one lane's receipt name again"
    )
    assert "receipt_names(job)" in worker


def test_a_screening_with_no_verdict_is_not_a_success() -> None:
    """The quiet half of the bug.

    A missing verdict read exactly like a checked-and-alive one: both left the
    job `succeeded`. Nothing downstream could tell "this lane was examined" from
    "this lane has never been examined", which is why three adapters went
    unnoticed for as long as they did.
    """
    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    block = worker[worker.index('if completed.returncode == 3:'):]
    block = block[: block.index("\n        if state ==", 1) + 400]
    assert 'job.get("phase") == "P5"' in block and 'not result["liveness"]' in block, (
        "a P5 job with no liveness verdict is still reported as succeeded"
    )
    assert 'state = "failed"' in block


def test_the_adjudicator_still_refuses_what_the_screening_could_not_read() -> None:
    """P7's gate depends on the verdict being there to read. It was already
    right; it was reading a field nobody filled in."""
    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    body = worker[worker.index("def resolve_screened_map"):]
    body = body[: body.index("\ndef ", 1)]
    assert 'verdict != "ALIVE"' in body


# --------------------------------------------------------------------------
# The shared refusal
# --------------------------------------------------------------------------

def test_the_refusal_keeps_the_evidence_and_returns_three(tmp_path) -> None:
    import io

    sys.path.insert(0, str(ROOT))
    from framework.contracts.lane_liveness import refuse_if_not_alive

    report = {"verdict": "DEGENERATE", "reason": "std 0.0001 < 0.02",
              "interpretation": "the output head is untrained"}
    stream = io.StringIO()
    code = refuse_if_not_alive(report, lane="a-lane", output=tmp_path,
                               on_degenerate="fail", stream=stream)
    assert code == 3, "the worker reads exit 3 as a refusal rather than a crash"
    marker = tmp_path / "LANE_NOT_USABLE"
    assert marker.is_file(), "the refusal left nothing to diagnose from"
    assert "std 0.0001" in marker.read_text()
    assert "a-lane" in stream.getvalue()

    # warn is the way through for a diagnostic run, and it still leaves the mark.
    (tmp_path / "LANE_NOT_USABLE").unlink()
    assert refuse_if_not_alive(report, lane="a-lane", output=tmp_path,
                               on_degenerate="warn", stream=io.StringIO()) == 0
    assert (tmp_path / "LANE_NOT_USABLE").is_file()


def test_an_alive_map_passes_untouched(tmp_path) -> None:
    sys.path.insert(0, str(ROOT))
    from framework.contracts.lane_liveness import refuse_if_not_alive

    assert refuse_if_not_alive({"verdict": "ALIVE"}, lane="x", output=tmp_path) == 0
    assert not (tmp_path / "LANE_NOT_USABLE").exists()
