"""The gate that would have caught resnet50_7.9um on its first run.

Fixtures are the real measured shapes: the degenerate case reproduces the
distribution that checkpoint actually produced on known ink (everything piled
against 0.5), the alive case reproduces the canonical 2 um lane's spread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.contracts.lane_liveness import assess_liveness, require_alive


def degenerate_map(rng: np.random.Generator) -> np.ndarray:
    """resnet50_7.9um on PHerc0139: p50 0.5091, p99 0.5232, max 0.5403."""
    return np.clip(rng.normal(0.509, 0.006, (512, 512)), 0.4, 0.6)


def alive_map(rng: np.random.Generator) -> np.ndarray:
    """canonical 2 um on the same window: p50 0.129, p90 0.196, p99 0.903."""
    body = rng.beta(2.0, 9.0, (512, 512))
    hot = rng.random((512, 512)) < 0.02
    return np.clip(np.where(hot, rng.beta(9.0, 1.5, (512, 512)), body), 1e-4, 1.0)


def test_degenerate_map_is_rejected():
    report = assess_liveness(degenerate_map(np.random.default_rng(0)))
    assert report["verdict"] == "DEGENERATE"
    assert "0.5" in report["reason"] or "std" in report["reason"]


def test_alive_map_passes():
    assert assess_liveness(alive_map(np.random.default_rng(1)))["verdict"] == "ALIVE"


def test_empty_map_is_its_own_verdict():
    """No valid pixels is a different failure from a dead head; don't conflate them."""
    assert assess_liveness(np.zeros((64, 64)))["verdict"] == "EMPTY"


def test_require_alive_fails_closed_and_names_the_lane():
    with pytest.raises(RuntimeError, match="resnet50-7.9um"):
        require_alive(degenerate_map(np.random.default_rng(2)), lane="resnet50-7.9um")


def test_a_constant_map_never_passes():
    """The limiting case: one value everywhere carries no decision at all."""
    for value in (0.01, 0.5, 0.99):
        report = assess_liveness(np.full((256, 256), value))
        assert report["verdict"] == "DEGENERATE", f"constant {value} passed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_a_disqualified_method_is_refused_before_it_runs():
    """The registry disqualification must block routing, not just document it."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_ink", root / "framework" / "stages" / "03-ink" / "scripts" / "run_ink.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(RuntimeError, match="DISQUALIFIED"):
        module.check_routable("resnet50-7.9um-scroll1-frags@1.0.0")
    # a method with no disqualification passes silently
    module.check_routable("ink-canonical-2um@1.0.0")


def test_the_disqualified_profile_cannot_be_read():
    """Reading its profile is the routing path; it must fail there too."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_ink2", root / "framework" / "stages" / "03-ink" / "scripts" / "run_ink.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = root / "framework" / "profiles" / "03-ink" / "resnet50-7.9um-scroll1-frags-screening-1.0.0.json"
    with pytest.raises(RuntimeError, match="DISQUALIFIED"):
        module.read_profile(profile)
