"""The short chain: a control anybody can reproduce from public inputs.

Six boundaries, each verifiable without a credential:

    PUBLIC_SOURCE  the surface volume is reachable and is what it declares
    SCALE          it is at the model's scale, or pooled with a receipt
    CHECKPOINT     the model file's digest is the declared one
    INK            inference completed and wrote a map
    LIVENESS       the map carries a decision, rather than one value everywhere
    HUMAN_REVIEW   it is routed for review, and claims nothing on its own

What this control proves is narrow and is written into the receipt: that this
platform can drive the recommended tooling end to end on data anybody can
obtain. It is not a reading, not an ink claim, and not a statement that the
nine-boundary campaign control passes -- that one is a different receipt with
a different schema, and this cannot be published as it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

from run_public_ink_control import (  # noqa: E402
    PUBLIC_BOUNDARIES, run_public_ink_control,
)


class FakeSource:
    """Stands in for the open-data bucket: answers metadata, no credentials."""

    def __init__(self, *, voxel_um=9.362, reachable=True):
        self.voxel_um, self.reachable = voxel_um, reachable
        self.asked = []

    def read_metadata(self, uri):
        self.asked.append(uri)
        if not self.reachable:
            raise RuntimeError("HTTP 403 fetching .zattrs")
        return {"axes": ["z", "y", "x"], "voxel_size_um": self.voxel_um,
                "canvas_size": [8120, 6120]}


def checkpoint(tmp_path: Path, body: bytes = b"weights") -> tuple[Path, str]:
    path = tmp_path / "step-075000.pth"
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def inference_that_writes(probability):
    def run(**kwargs):
        np.save(Path(kwargs["output"]) / "probability.npy", probability)
        return {"argv": ["python", "-m", "koine_machines.inference.infer"]}
    return run


ALIVE = np.clip(np.linspace(0.25, 0.95, 64 * 64).reshape(64, 64), 0, 1)
FLAT = np.full((64, 64), 0.25)


def _run(tmp_path, **overrides):
    path, digest = checkpoint(tmp_path)
    kwargs = dict(
        surface_volume="https://vesuvius-challenge-open-data.s3.amazonaws.com/x.zarr",
        checkpoint=path, expected_checkpoint_sha256=digest,
        output=tmp_path / "out", source=FakeSource(),
        inference=inference_that_writes(ALIVE))
    kwargs.update(overrides)
    (tmp_path / "out").mkdir(exist_ok=True)
    return run_public_ink_control(**kwargs)


# -- the happy path --------------------------------------------------------

def test_a_public_run_passes_every_boundary(tmp_path):
    receipt = _run(tmp_path)
    assert receipt["control_state"] == "CONTROL_PASS"
    assert [r["boundary"] for r in receipt["stages"]] == list(PUBLIC_BOUNDARIES)
    assert receipt["content_sha256"]


def test_the_receipt_says_what_it_does_not_establish(tmp_path):
    receipt = _run(tmp_path)
    claims = " ".join(receipt["non_claims"]).lower()
    assert "reading" in claims
    assert "ink" in claims


def test_the_receipt_names_the_public_inputs_it_used(tmp_path):
    """The whole point is that a reader can obtain them."""
    receipt = _run(tmp_path)
    source = next(r for r in receipt["stages"] if r["boundary"] == "PUBLIC_SOURCE")
    assert "open-data" in json.dumps(source["resource_identity"])


# -- each boundary refusing ------------------------------------------------

def test_an_unreachable_source_stops_at_the_first_boundary(tmp_path):
    receipt = _run(tmp_path, source=FakeSource(reachable=False))
    assert receipt["first_nonpassing_boundary"] == "PUBLIC_SOURCE"
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"


def test_a_volume_off_the_model_scale_stops_at_scale(tmp_path):
    """7.9 um is neither the model's scale nor the one the pooling recipe is
    written for, and inventing a factor is what the recipe refuses."""
    receipt = _run(tmp_path, source=FakeSource(voxel_um=7.9))
    assert receipt["first_nonpassing_boundary"] == "SCALE"


def test_a_checkpoint_that_is_not_the_declared_one_stops_at_checkpoint(tmp_path):
    receipt = _run(tmp_path, expected_checkpoint_sha256="0" * 64)
    assert receipt["first_nonpassing_boundary"] == "CHECKPOINT"
    row = next(r for r in receipt["stages"] if r["boundary"] == "CHECKPOINT")
    assert row["reason_code"] == "CHECKPOINT_DIGEST_MISMATCH"


def test_inference_that_writes_nothing_stops_at_ink(tmp_path):
    receipt = _run(tmp_path, inference=lambda **k: {"argv": []})
    assert receipt["first_nonpassing_boundary"] == "INK"


def test_a_map_with_no_decision_stops_at_liveness(tmp_path):
    """One value everywhere is what a wrong depth window produces, and it
    exits zero."""
    receipt = _run(tmp_path, inference=inference_that_writes(FLAT))
    assert receipt["first_nonpassing_boundary"] == "LIVENESS"
    row = next(r for r in receipt["stages"] if r["boundary"] == "LIVENESS")
    assert "DEGENERATE" in row["reason_code"] or "DEGENERATE" in str(row.get("detail", ""))


# -- what it refuses to become --------------------------------------------

def test_it_cannot_be_published_as_the_campaign_control(tmp_path):
    from run_first_letters_positive_control import SCHEMA, evaluate_survival_matrix

    receipt = _run(tmp_path)
    assert receipt["schema"] != SCHEMA
    with pytest.raises(ValueError):
        evaluate_survival_matrix({**receipt, "schema": SCHEMA})


def test_the_control_finds_the_repository_from_its_own_location(tmp_path):
    """Found by running it for real: `parents[1]` of
    <root>/scripts/harness/run_public_ink_control.py is <root>/scripts, not
    <root>, so `import framework...` failed with ModuleNotFoundError the
    moment it ran outside a test.

    The tests could not see it. Every one of them is imported by a test module
    that has already put ROOT on sys.path, so the module's own path setup was
    never the thing being exercised. This asserts the arithmetic itself.
    """
    import run_public_ink_control as module

    root = Path(module.__file__).resolve().parents[2]
    assert (root / "framework/contracts/lane_liveness.py").is_file(), (
        f"the control resolves its repository root to {root}, which does not "
        "contain the framework it imports")
    assert (root / "framework/stages/03-ink/scripts"
            / "prepare_9um_isotropic_input.py").is_file()


def test_the_control_runs_as_a_subprocess_with_a_bare_environment(tmp_path):
    """The real check: no test module has arranged sys.path for it."""
    import subprocess
    import run_public_ink_control as module

    completed = subprocess.run(
        [sys.executable, str(Path(module.__file__).resolve()), "--help"],
        capture_output=True, text=True, timeout=120,
        cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"})
    assert completed.returncode == 0, (
        f"the control cannot start on its own: {completed.stderr[-600:]}")
