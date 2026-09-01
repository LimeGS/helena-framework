"""A P5 job that names weights nobody installed is refused when it is queued.

A P5 job names a checkpoint by path. Until now the first thing that noticed an
absent one was the runner: a worker claimed the job, spent a lease, opened the
layer stack, and then reported whatever the loader says about a file that is
not there -- an hour of GPU time to learn something the panel could have
answered in a millisecond, and a failure that reads like a model problem rather
than an install problem.

There are twenty-seven checkpoints across ten profiles, four of them installed.
The other twenty-three are exactly this failure waiting for whoever queues one.

The refusal is deliberately narrow. On a deployment where the panel and the
workers do not share a model root, absence here proves nothing about the
worker, so the check declines to guess: no root, no refusal. That is the
difference between a check that helps and one that a split deployment has to
disable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="panel")
def _panel():
    sys.path.insert(0, str(ROOT / "panel"))
    import app  # noqa: PLC0415
    return app


def test_the_named_checkpoint_is_absent_and_the_queue_says_so(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "MODELS_ROOT", tmp_path)
    with pytest.raises(panel.HTTPException) as refusal:
        panel.refuse_absent_checkpoint(str(tmp_path / "ink_9um/step-075000.pth"))
    # The message has to name the file: "checkpoint not found" sends the reader
    # to the profile, and the profile is not what is wrong.
    assert "step-075000.pth" in refusal.value.detail
    assert refusal.value.status_code == 409


def test_the_checkpoint_is_there_and_nothing_is_refused(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "MODELS_ROOT", tmp_path)
    weights = tmp_path / "canonical" / "r152.ckpt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not really a checkpoint")
    panel.refuse_absent_checkpoint(str(weights))


def test_a_panel_that_cannot_see_the_models_does_not_guess(panel, tmp_path, monkeypatch):
    """The split deployment: the workers have the weights, this panel does not.

    Refusing here would make the check something an operator has to switch off,
    and a check that gets switched off protects nobody.
    """
    monkeypatch.setattr(panel, "MODELS_ROOT", tmp_path / "no-such-root")
    panel.refuse_absent_checkpoint(str(tmp_path / "no-such-root/ink_9um/step-075000.pth"))


def test_a_checkpoint_outside_this_root_is_not_this_panels_business(panel, tmp_path, monkeypatch):
    """A path the panel is not authoritative about is left alone.

    The models root is the only place this panel installs to, so it is the only
    place its absence means anything.
    """
    monkeypatch.setattr(panel, "MODELS_ROOT", tmp_path)
    panel.refuse_absent_checkpoint("/somewhere/else/entirely/model.pth")


def test_no_checkpoint_named_is_not_this_checks_problem(panel, tmp_path, monkeypatch):
    """Lanes that take no checkpoint exist; a required-parameter check owns that."""
    monkeypatch.setattr(panel, "MODELS_ROOT", tmp_path)
    panel.refuse_absent_checkpoint(None)
    panel.refuse_absent_checkpoint("")
    panel.refuse_absent_checkpoint("   ")


def test_the_check_runs_inside_the_p5_branch_and_before_the_row_is_written():
    """Placement is the whole value: refusing after the job exists is not a
    refusal, it is a failed job with a nicer message.

    Asserted against the source the way the other P5 ordering tests here do,
    because reaching this line through the API means assembling a mission, a
    surface and a lineage -- which would test those instead.
    """
    body = (ROOT / "panel/app.py").read_text(encoding="utf-8")
    branch = body[body.index('if request.phase == "P5":'):]
    assert "refuse_absent_checkpoint(parameters.get(\"checkpoint\"))" in branch

    called = branch.index("refuse_absent_checkpoint(")
    # Anything that persists the job has to come after. `checkpoint` is a
    # required P5 parameter, so by this point it has already been established
    # to be present -- what is in question is whether it names a real file.
    for writes in ("enqueue(", "INSERT INTO", "store.add"):
        at = branch.find(writes)
        if at != -1:
            assert called < at, f"the refusal runs after {writes!r}"


def test_a_profile_that_needs_an_architecture_is_refused_without_one(panel):
    """`model_config_required: true` was in the profile and read by nothing.

    Five P5 jobs went to the queue without the parameter, claimed a worker,
    loaded the checkpoint and died on

        size mismatch for backbone.layers.0.0.fn.to_qkv.weight: copying a param
        with shape torch.Size([1536, 512]) ... current model is [1152, 512]

    about a hundred tensors at a time. That is a 6-head model meeting 8-head
    weights: the runner has always accepted --model-config and the queue has
    always been able to name one, so nothing was missing except the check that
    the profile asking for it gets it.
    """
    wanting = [p for p in panel.ink_profiles()
               if (p.get("input_contract") or {}).get("model_config_required")]
    assert wanting, (
        "no profile declares model_config_required any more; if that is "
        "deliberate this check has nothing left to guard")

    for profile in wanting:
        with pytest.raises(panel.HTTPException) as refusal:
            panel.refuse_missing_model_config(profile, None)
        assert refusal.value.status_code == 409
        assert "model_config" in refusal.value.detail
        assert "model-configs" in refusal.value.detail, (
            "the refusal has to say where the file lives; knowing one is "
            "missing is not knowing what to do")

        # Given one, it gets out of the way.
        panel.refuse_missing_model_config(
            profile, "framework/registries/model-configs/whatever.json")


def test_a_profile_with_a_frozen_architecture_is_not_asked_for_one(panel):
    """The GP lane's checkpoint *is* the default, and demanding a config for it
    would refuse the one profile that has always worked."""
    for profile in panel.ink_profiles():
        if (profile.get("input_contract") or {}).get("model_config_required"):
            continue
        panel.refuse_missing_model_config(profile, None)
