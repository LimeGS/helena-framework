"""The six-boundary control, driven through Helena rather than beside it.

The control already passes: it drives the recommended tooling end to end on
public data. What it did not do is exercise Helena. Its INK boundary shelled out
to the lane adapter in a local subprocess, so a green receipt said the *tooling*
works and said nothing about the queue, the worker, the routing or the
publication -- which is the half a reviewer is being asked to trust.

So the same six boundaries, with the ink step queued through the panel's API and
nothing else: enqueue, wait, and fetch what the job published. No path on a
worker's disk is read, because a control that reaches into the machine it is
testing is not testing the interface anybody else would use.

The receipt says which way it ran. A run through the queue and a run beside it
are different claims and must not be readable as the same one.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "harness"))

from run_public_ink_control import queued_inference  # noqa: E402

ARTIFACT = "/artifacts/ink-maps-v1/surfaces/PHerc0332/ink-maps/p5-abc"


def a_published_map(shape=(4, 4)) -> bytes:
    """What the panel hands back for an artifact key: the directory, gzipped."""
    buffer = io.BytesIO()
    payload = io.BytesIO()
    np.save(payload, np.full(shape, 0.25, dtype=np.float32))
    body = payload.getvalue()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("./probability.npy")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


class FakePanel:
    """Enough panel to drive the queue, and a record of what was asked."""

    def __init__(self, *, state="succeeded", result=None, artifact=None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self._state = state
        self._result = result if result is not None else {
            "probability_map": {"artifact_uri": ARTIFACT},
            "liveness": {"verdict": "ALIVE"},
        }
        self._artifact = artifact if artifact is not None else a_published_map()

    def call(self, method, path, body=None, **_):
        self.calls.append((method, path, body))
        if path == "/api/jobs" and method == "POST":
            return {"job_id": "p5-abc"}
        raise AssertionError(f"unexpected call {method} {path}")

    def wait_for_job(self, job_id, **_):
        return {"job_id": job_id, "state": self._state, "result": self._result}

    def fetch(self, path):
        self.calls.append(("GET", path, None))
        return self._artifact


def run(panel, output: Path, **overrides):
    body = {"mission_id": "control", "sample_id": "PHerc0332",
            "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0",
            "parameters": {"tiff_dir": "/layers", "checkpoint": "/models/c.pth",
                           "source_pixel_um": 9.362,
                           "artifact_store": "/artifacts/ink-maps-v1"}}
    body.update(overrides)
    inference = queued_inference(panel, **body)
    return inference(surface_volume="ignored", checkpoint=Path("/models/c.pth"),
                     output=output)


def test_the_map_arrives_through_the_api_not_off_a_worker_disk(tmp_path) -> None:
    panel = FakePanel()

    execution = run(panel, tmp_path)

    assert (tmp_path / "probability.npy").is_file()
    fetched = [path for method, path, _ in panel.calls if method == "GET"]
    assert fetched == ["/api/artifacts/ink-maps-v1/surfaces/PHerc0332/ink-maps/p5-abc"]
    assert execution["job_id"] == "p5-abc"


def test_the_job_is_queued_inside_a_mission(tmp_path) -> None:
    """Work does not exist outside one, and a control that queued outside would
    not be exercising the platform as anybody else must use it."""
    panel = FakePanel()

    run(panel, tmp_path)

    posted = next(body for method, path, body in panel.calls
                  if method == "POST" and path == "/api/jobs")
    assert posted["mission_id"] == "control"


def test_a_failed_job_is_raised_rather_than_reported_as_a_map(tmp_path) -> None:
    panel = FakePanel(state="failed",
                      result={"error": "the lane died", "exit_code": 1})

    with pytest.raises(RuntimeError) as failure:
        run(panel, tmp_path)

    assert "the lane died" in str(failure.value)
    assert not (tmp_path / "probability.npy").exists()


def test_a_job_that_published_nothing_is_not_treated_as_success(tmp_path) -> None:
    """An exit code is not evidence that the output exists -- the same rule the
    control already applies to the local path."""
    panel = FakePanel(result={"liveness": {"verdict": "ALIVE"}})

    with pytest.raises(RuntimeError) as failure:
        run(panel, tmp_path)

    assert "published" in str(failure.value).lower()


def test_an_archive_that_reaches_outside_the_output_is_refused(tmp_path) -> None:
    """The bytes arrive over HTTP from a host this control is testing, which is
    exactly the assumption not to make about a tar."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escaped.npy")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    panel = FakePanel(artifact=buffer.getvalue())

    with pytest.raises(RuntimeError) as refused:
        run(panel, tmp_path)

    assert "unsafe" in str(refused.value).lower()
    assert not (tmp_path.parent / "escaped.npy").exists()


def test_the_receipt_says_it_went_through_the_queue(tmp_path) -> None:
    """A run through the queue and a run beside it are different claims."""
    panel = FakePanel()

    execution = run(panel, tmp_path)

    assert execution["through"] == "helena-queue"
    assert execution["artifact_uri"] == ARTIFACT


def test_the_queued_job_names_the_volume_the_control_verified(tmp_path) -> None:
    """One input, verified once and used once.

    The boundaries above check that the public volume is reachable and at the
    model's scale. Handing the worker some other path afterwards would put a
    second input in the receipt and leave which one the map came from a matter
    of trust -- which is the whole thing a control exists to remove.
    """
    panel = FakePanel()

    run(panel, tmp_path,
        parameters={"surface_volume": "s3://open-data/vol.zarr",
                    "checkpoint": "/models/c.pth",
                    "artifact_store": "/artifacts/ink-maps-v1"})

    posted = next(body for method, path, body in panel.calls
                  if method == "POST" and path == "/api/jobs")
    assert posted["parameters"]["surface_volume"] == "s3://open-data/vol.zarr"
    assert "tiff_dir" not in posted["parameters"]


# -- what the receipt says about how the ink was made ------------------------


def a_control(tmp_path, inference):
    """The six boundaries, with everything before INK stubbed to pass."""
    import numpy as np

    from run_public_ink_control import run_public_ink_control

    checkpoint = tmp_path / "step.pth"
    checkpoint.write_bytes(b"weights")
    digest = __import__("hashlib").sha256(b"weights").hexdigest()

    class Source:
        def read_metadata(self, uri):
            return {"axes": ["z", "y", "x"], "voxel_size_um": 9.362,
                    "canvas_size": [64, 64]}

    def writing_inference(**kwargs):
        np.save(Path(kwargs["output"]) / "probability.npy",
                np.linspace(0, 1, 4096, dtype=np.float32).reshape(64, 64))
        return inference

    return run_public_ink_control(
        surface_volume="https://open-data/vol.zarr", checkpoint=checkpoint,
        expected_checkpoint_sha256=digest, output=tmp_path / "out",
        source=Source(), inference=writing_inference)


def ink_row(receipt):
    return next(row for row in receipt["stages"] if row["boundary"] == "INK")


def test_a_queued_run_is_identified_by_its_job_not_by_a_blank(tmp_path) -> None:
    """It read `argv`, which the queue has no equivalent of, so the receipt of
    the run that most needed identifying said `command: null`."""
    receipt = a_control(tmp_path, {"through": "helena-queue", "job_id": "p5-abc",
                                   "artifact_uri": "/artifacts/x",
                                   "artifact_sha256": "a" * 64})

    identity = ink_row(receipt)["resource_identity"]
    assert identity["through"] == "helena-queue"
    assert identity["job_id"] == "p5-abc"
    assert identity["artifact_uri"] == "/artifacts/x"


def test_a_local_run_still_carries_its_argv_and_says_it_was_local(tmp_path) -> None:
    """Both halves answer the same question -- how was this made -- so both say
    which way they were run. A receipt where only one path names itself invites
    the reader to assume the other."""
    receipt = a_control(tmp_path, {"through": "local-subprocess",
                                   "argv": ["python", "run_ink_9um.py"]})

    identity = ink_row(receipt)["resource_identity"]
    assert identity["through"] == "local-subprocess"
    assert identity["argv"] == ["python", "run_ink_9um.py"]


def test_the_default_inference_names_itself(tmp_path) -> None:
    import inspect

    from run_public_ink_control import _default_inference

    assert '"through": "local-subprocess"' in inspect.getsource(_default_inference)


# --- the CHECKPOINT boundary, asked of the platform ------------------------
#
# It used to hash a file this process could see, which required mounting the
# models volume into the control -- reaching into the machine under test, which
# is the thing the queued path exists to stop doing. Hashing a local file also
# proves the wrong thing: that *this* process found the right bytes somewhere,
# not that the deployment about to run the job has them.

from run_public_ink_control import run_public_ink_control  # noqa: E402

SHA = "e" * 64


def _boundary(receipt, name):
    return next(s for s in receipt["stages"] if s["boundary"] == name)


def _control(installed, tmp_path):
    """The two boundaries before CHECKPOINT are not what these tests are about,
    so the run is allowed to stop there; CHECKPOINT is reached either way."""
    return run_public_ink_control(
        surface_volume="https://example.invalid/v.zarr",
        checkpoint=tmp_path / "absent.pth",
        expected_checkpoint_sha256=SHA,
        output=tmp_path, installed=installed,
        source=_AlwaysPublic(), inference=lambda **_: {"through": "test"})


class _AlwaysPublic:
    """The scale the model wants, so PUBLIC_SOURCE and SCALE pass and CHECKPOINT
    is the boundary under test."""

    def read_metadata(self, _uri):
        return {"voxel_size_um": 9.362, "axes": "zyx",
                "canvas_size": [1, 1, 1]}


def test_a_declared_and_installed_checkpoint_passes_without_a_local_file(tmp_path):
    """The file is deliberately absent: nothing hashes it any more."""
    row = {"checkpoint_sha256": SHA, "installed": True,
           "declared_by": ["ink-9um-hybrid-3d2d@1.0.0"],
           "expected_path": "ink_9um/hybrid_3d2d-seed42/step-075000.pth"}
    stage = _boundary(_control(lambda _s: row, tmp_path), "CHECKPOINT")
    assert stage["terminal_state"] == "PASS", stage
    assert stage["resource_identity"]["established_by"] == "helena-api", (
        "the receipt does not say the platform established this, so a reader "
        "cannot tell it from a run that hashed a file beside itself")


def test_a_hash_no_profile_declares_is_refused(tmp_path):
    """Correct bytes are not enough: a deployment cannot be asked to use a
    checkpoint none of its frozen profiles names."""
    stage = _boundary(_control(lambda _s: None, tmp_path), "CHECKPOINT")
    assert stage["terminal_state"] == "FAILED"
    assert stage["reason_code"] == "CHECKPOINT_NOT_DECLARED"


def test_declared_but_absent_is_incomplete_rather_than_failed(tmp_path):
    """Nothing is wrong with the checkpoint; it is not there yet. Those are
    different answers and the receipt keeps them apart."""
    row = {"checkpoint_sha256": SHA, "installed": False, "declared_by": ["x"]}
    stage = _boundary(_control(lambda _s: row, tmp_path), "CHECKPOINT")
    assert stage["terminal_state"] == "INCOMPLETE"
    assert stage["reason_code"] == "CHECKPOINT_NOT_INSTALLED"


def test_without_a_platform_it_still_hashes_the_file(tmp_path):
    """The local path is what `--panel` is not, and it has to keep working: it
    is how the control runs against tooling with no deployment at all."""
    import hashlib
    f = tmp_path / "ckpt.pth"
    f.write_bytes(b"weights")
    receipt = run_public_ink_control(
        surface_volume="https://example.invalid/v.zarr", checkpoint=f,
        expected_checkpoint_sha256=hashlib.sha256(b"weights").hexdigest(),
        output=tmp_path, source=_AlwaysPublic(),
        inference=lambda **_: {"through": "test"})
    stage = _boundary(receipt, "CHECKPOINT")
    assert stage["terminal_state"] == "PASS"
    assert stage["resource_identity"]["established_by"] == "local-file"
