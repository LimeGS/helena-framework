"""Both devices, always -- and a receipt that says which one ran.

Every runner that takes a device defaulted to `cuda`, unconditionally. That is
a fine default on the two hosts with cards and a crash on any host without one,
so "supports both" was true of the flag and false of the program.

`auto` is now the default and means the card if this host has one, the CPU if it
does not. An explicit `cuda` stays a requirement rather than a preference: the
seeded grow's own CUDA path was measured by upstream at 40x *slower* than its
CPU one, so swapping devices silently is not a detail, and a run that asked for
a card and did not get one is refused before it loads anything.

The receipt records the device that ran, never the word the caller wrote.
`auto` in a receipt says nothing about where the work happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from framework.contracts import host_probe  # noqa: E402
from job_store import JobRejected, validate_parameters  # noqa: E402

RUNNERS = [
    "framework/stages/01-segmentation/scripts/scan_large_surface_windows.py",
    "framework/stages/03-ink/scripts/run_ink.py",
    "framework/stages/03-ink/scripts/run_ink_3d_dino.py",
    "framework/stages/03-ink/scripts/run_ink_canonical2um.py",
    "framework/stages/03-ink/scripts/run_ink_timesformer.py",
    "framework/stages/06-discovery/scripts/run_pherc1667_iteration0.py",
]


@pytest.fixture
def card(monkeypatch):
    def present(available: bool, detail: str = "1 device(s)"):
        monkeypatch.setattr(host_probe, "_cuda_available",
                            lambda: (available, detail))
    return present


# -- resolving --------------------------------------------------------------

def test_auto_takes_the_card_when_there_is_one(card):
    card(True)
    assert host_probe.resolve_device("auto")["device"] == "cuda"
    assert host_probe.resolve_device(None)["device"] == "cuda"


def test_auto_falls_back_to_the_cpu_when_there_is_not(card):
    card(False, "torch reports no CUDA device")
    outcome = host_probe.resolve_device("auto")

    assert outcome["device"] == "cpu"
    assert "no card" in outcome["reason"]


def test_an_explicit_card_is_a_requirement_not_a_preference(card):
    """The failure this prevents: a run queued for a card quietly finishing on
    the CPU some hours later, at a cost nobody agreed to."""
    card(False, "torch reports no CUDA device")

    with pytest.raises(RuntimeError, match="no usable card"):
        host_probe.resolve_device("cuda:0")


def test_an_explicit_cpu_is_honoured_on_a_host_that_has_a_card(card):
    card(True)
    assert host_probe.resolve_device("cpu")["device"] == "cpu"


def test_a_device_that_is_neither_is_refused_by_name(card):
    card(True)
    with pytest.raises(ValueError, match="auto"):
        host_probe.resolve_device("gpu")


def test_the_resolution_carries_what_a_receipt_needs(card):
    card(False, "torch is not installed here")
    outcome = host_probe.resolve_device("auto")

    assert set(outcome) == {"device", "requested", "cuda_available", "reason"}
    assert outcome["requested"] == "auto" and outcome["device"] == "cpu"


def test_asking_for_a_card_never_imports_torch_twice_or_raises(monkeypatch):
    """A broken driver is 'no card', not a traceback out of a probe."""
    monkeypatch.setattr(host_probe, "_cuda_available",
                        lambda: (False, "asking torch for a device raised OSError"))
    assert host_probe.resolve_device("auto")["device"] == "cpu"


# -- the runners ------------------------------------------------------------

@pytest.mark.parametrize("relative", RUNNERS)
def test_no_runner_still_assumes_a_card(relative):
    source = (ROOT / relative).read_text(encoding="utf-8")

    assert 'default="cuda"' not in source, (
        f"{relative} defaults to a card again; on a host without one it crashes "
        "instead of running on the CPU")
    assert 'default="auto"' in source
    assert "resolve_device(args.device)" in source, (
        f"{relative} takes a device and never resolves it, so 'auto' reaches "
        "torch as the literal string")


@pytest.mark.parametrize("relative", RUNNERS)
def test_a_runner_resolves_the_moment_it_has_the_arguments(relative):
    """Cheap refusals first: a host without a card has to find out at parse
    time, not after a checkpoint is on the device. Checked as adjacency to
    `parse_args`, which is a fact about the source; where the load happens is
    not, because the helpers are defined above main."""
    source = (ROOT / relative).read_text(encoding="utf-8")
    parsed = max(source.rfind("parse_args()\n"), 0)
    resolved = source.index("resolve_device(args.device)")

    assert resolved > parsed
    between = source[parsed:resolved]
    assert between.count("\n") <= 8, (
        f"{relative} resolves the device {between.count(chr(10))} lines after "
        "parsing; something runs in between")


# -- the queue --------------------------------------------------------------

P5 = {"checkpoint": "/models/m/model.ckpt", "tiff_dir": "/layers",
      "source_pixel_um": 9.362}


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:1"])
def test_the_queue_accepts_every_form(device):
    assert validate_parameters({**P5, "device": device}, "P5")["device"] == device


def test_the_queue_still_refuses_a_device_that_is_not_one():
    with pytest.raises(JobRejected, match="auto"):
        validate_parameters({**P5, "device": "gpu"}, "P5")


def test_a_job_that_names_no_device_carries_none_and_the_worker_decides():
    """Absent is the common case and must stay meaningful: the runner's own
    default is `auto`, so an unset parameter is 'whatever claims this'."""
    assert "device" not in validate_parameters(dict(P5), "P5")
