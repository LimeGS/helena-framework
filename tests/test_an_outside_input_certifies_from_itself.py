"""A job may bring its input from outside, and certify from there.

Two controls use the same scroll. The campaign control is the chained one --
P0 through P5, every step produced here and naming the one before it, so a map
can be walked back to the scan. The public control asks a different question:
can this platform drive the recommended tooling on data anybody can download?
Its input is a surface volume in the open bucket, which is exactly what makes
it reproducible by a stranger, and which nothing here produced.

Queueing the second one hit the first one's gate. The selected P0 bytes *are*
the campaign control's source, so a P5 over them was read as part of the chain
and asked which P4 render produced its input. The honest answer is none: it
came from the internet, and the platform had no way to hear that.

So an input that came from outside by construction -- `surface_volume` fetched,
or `tiff_dir` carried in -- runs without the campaign binding. Not by skipping
the check: by not being bound at all. A job with no control binding cannot be
read back as the control, because the binding is what says it is one, and it is
the absence of that stamp rather than a promise in a comment that keeps the two
apart.

`layer_stack` is unchanged. Naming one of this platform's own P4 renders is a
claim about the chain, and the chain is then checked exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")


def enqueue_source() -> str:
    import inspect

    from panel.app import api_enqueue

    return inspect.getsource(api_enqueue)


def test_an_outside_input_drops_the_binding_rather_than_skipping_the_check() -> None:
    """Skipping the check would leave the job stamped as the control while
    unbound to it, which is the one outcome worse than refusing."""
    body = enqueue_source()
    p5 = body[body.index('if request.phase == "P5":'):]
    p5 = p5[:p5.index('if request.phase == "P7"')]

    assert "control_binding = None" in p5, (
        "an outside input does not clear the binding")
    dropped = p5.index("control_binding = None")
    checked = p5.index("control P5 requires the exact bound P4 job")
    assert dropped < checked, (
        "the binding is cleared after the refusal it is meant to make moot")


def test_naming_one_of_our_own_renders_still_demands_the_exact_job() -> None:
    body = enqueue_source()

    assert "control P5 requires the exact bound P4 job" in body
    assert "layer_stack" in body


def test_the_worker_records_where_the_chain_begins() -> None:
    """The stamp is absent; that is structural. What a reader also needs is the
    positive statement, on the job, of what was supplied and that nothing before
    it is claimed."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from ink_worker import supplied_input_note

    for named, value in (("tiff_dir", "/layers"),
                         ("surface_volume", "https://open-data/vol.zarr")):
        note = supplied_input_note({"phase": "P5", "parameters": {named: value}})
        assert note is not None, f"{named} is not recorded as supplied"
        assert note["path"] == value
        assert "was not produced by" in note["non_claim"]


def test_a_render_this_platform_made_is_not_called_supplied() -> None:
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from ink_worker import supplied_input_note

    assert supplied_input_note(
        {"phase": "P5", "parameters": {"layer_stack": "p4-abc", "tiff_dir": "/l"}}) is None


def test_releasing_the_binding_does_not_blind_the_enqueue_proof() -> None:
    """The proof re-derives the binding just before insert and refuses if it
    moved: that is what catches a selected P0 changing under a request in
    flight, and it must keep working.

    Releasing the binding for an outside input made the two disagree by
    construction -- the fresh derivation still finds the real binding, and the
    released value is None -- so every such job was refused as if the selection
    had changed. The comparison is against what was derived at the check, not
    against what the release left behind. The guard still sees a P0 that moved;
    it no longer mistakes a deliberate release for one.
    """
    body = enqueue_source()

    assert "control_binding_at_check" in body, (
        "the proof has nothing to compare against but the released value")
    proof = body[body.index("refreshed_binding = "):]
    proof = proof[:proof.index("enqueue proof") + 20]
    assert "refreshed_binding != control_binding_at_check" in proof, (
        "the proof still compares against the released binding")


def test_the_proof_still_refuses_a_p0_that_actually_moved() -> None:
    body = enqueue_source()

    assert "selected P0 control binding changed during enqueue proof" in body
