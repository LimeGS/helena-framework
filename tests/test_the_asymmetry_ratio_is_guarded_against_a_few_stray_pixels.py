"""p50 does not separate a real stack from the same stack with its layers
shuffled; a rigorous measurement found that and the panel used to show p50 as
though it did. This is the metric that replaces it as the discriminator, and
the failure it has to survive on its own: a threshold with a handful of bright
pixels on one side producing a ratio that reads exactly like a strong
detection.

Reference numbers, from the team's own measurement on PHerc0139 w043 (the
community's confirmed positive control) against the same stack with its layers
shuffled, and are not reproduced exactly here -- these are synthetic arrays
with pixel counts chosen to make the arithmetic checkable by hand, not a
re-run of the real inference:

    ratio @         0.5    0.6    0.7
    real order      2.53   3.36   5.21
    shuffled        0.67   0.56   0.46
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.contracts.lane_liveness import (
    MIN_OVER_THRESHOLD_PIXELS, forward_reverse_asymmetry,
)


def test_the_ratio_is_counted_pixels_over_counted_pixels():
    """Nothing smoothed, nothing else folded in: forward-over-threshold count
    divided by reverse-over-threshold count, at each of the three thresholds
    independently."""
    forward = np.zeros((10, 100))   # 1000 px
    reverse = np.zeros((10, 100))
    # >= 0.5: 600 forward, 300 reverse
    forward.flat[:600] = 0.55
    reverse.flat[:300] = 0.55
    # >= 0.6 (denser band, still >= 0.5): 450 forward, 300 reverse
    forward.flat[:450] = 0.65
    reverse.flat[:300] = 0.65
    # >= 0.7 (densest band): 300 forward, 300 reverse
    forward.flat[:300] = 0.75
    reverse.flat[:300] = 0.75

    report = forward_reverse_asymmetry(forward, reverse)

    assert report["thresholds"]["0.5"]["forward_over_px"] == 600
    assert report["thresholds"]["0.5"]["reverse_over_px"] == 300
    assert report["thresholds"]["0.5"]["ratio"] == pytest.approx(2.0)
    assert report["thresholds"]["0.6"]["ratio"] == pytest.approx(1.5)
    assert report["thresholds"]["0.7"]["ratio"] == pytest.approx(1.0)


def test_the_guard_hides_a_ratio_built_from_a_handful_of_pixels():
    """500 forward pixels over 0.7 against 2 reverse pixels is a ratio of 250
    that reads like the strongest possible detection and is two stray bright
    pixels. The guard has to report absence and say why, not a fabricated 250,
    0 or 1."""
    forward = np.zeros((50, 50))   # 2500 px
    forward.flat[:500] = 0.9
    reverse = np.zeros((50, 50))
    reverse.flat[:2] = 0.9

    report = forward_reverse_asymmetry(forward, reverse)
    entry = report["thresholds"]["0.7"]

    assert entry["forward_over_px"] == 500
    assert entry["reverse_over_px"] == 2
    assert entry["ratio"] is None
    assert str(MIN_OVER_THRESHOLD_PIXELS) in entry["reason"]
    assert "reverse" in entry["reason"]


def test_the_guard_applies_to_either_side_alone():
    """The floor is per side, not on the ratio: too few pixels on the
    forward side is refused exactly like too few on the reverse side."""
    forward = np.zeros((50, 50))
    forward.flat[:5] = 0.9
    reverse = np.zeros((50, 50))
    reverse.flat[:400] = 0.9

    entry = forward_reverse_asymmetry(forward, reverse)["thresholds"]["0.7"]
    assert entry["ratio"] is None
    assert "forward" in entry["reason"]


def test_the_boundary_pixel_count_is_not_guarded():
    """Exactly the floor, on both sides, is a measurement -- one pixel short
    is not. The guard is `< 300`, not `<= 300`."""
    forward = np.zeros((50, 50))
    forward.flat[:MIN_OVER_THRESHOLD_PIXELS] = 0.9
    reverse = np.zeros((50, 50))
    reverse.flat[:MIN_OVER_THRESHOLD_PIXELS] = 0.9

    entry = forward_reverse_asymmetry(forward, reverse)["thresholds"]["0.7"]
    assert entry["ratio"] == pytest.approx(1.0)

    reverse_short = np.zeros((50, 50))
    reverse_short.flat[:MIN_OVER_THRESHOLD_PIXELS - 1] = 0.9
    entry_short = forward_reverse_asymmetry(forward, reverse_short)["thresholds"]["0.7"]
    assert entry_short["ratio"] is None


def test_sustained_needs_both_0_6_and_0_7_above_1_5():
    """Their own operative reading, reported rather than enforced: real ink
    is asymmetry that holds at both thresholds, not a spike at one."""
    forward = np.zeros((20, 100))   # 2000 px
    reverse = np.zeros((20, 100))
    forward.flat[:1000] = 0.9       # over every threshold
    reverse.flat[:400] = 0.9        # 400 over every threshold too: ratio 2.5 throughout

    report = forward_reverse_asymmetry(forward, reverse)
    assert report["thresholds"]["0.6"]["ratio"] == pytest.approx(2.5)
    assert report["thresholds"]["0.7"]["ratio"] == pytest.approx(2.5)
    assert report["sustained_above_1_5"] is True


def test_sustained_is_false_when_a_threshold_is_guarded_away():
    """A ratio that cannot be measured at 0.6 or 0.7 cannot be sustained
    there -- absence must not read as a pass."""
    forward = np.zeros((20, 100))
    reverse = np.zeros((20, 100))
    forward.flat[:1000] = 0.55      # only clears 0.5
    reverse.flat[:400] = 0.55

    report = forward_reverse_asymmetry(forward, reverse)
    assert report["thresholds"]["0.6"]["ratio"] is None
    assert report["sustained_above_1_5"] is False


def test_sustained_is_false_at_exactly_1_5():
    """The bar is `> 1.5`, not `>= 1.5` -- their own number, taken literally."""
    forward = np.zeros((20, 100))
    reverse = np.zeros((20, 100))
    forward.flat[:600] = 0.9
    reverse.flat[:400] = 0.9   # ratio exactly 1.5 at every threshold

    report = forward_reverse_asymmetry(forward, reverse)
    assert report["thresholds"]["0.7"]["ratio"] == pytest.approx(1.5)
    assert report["sustained_above_1_5"] is False


def test_mismatched_shapes_are_refused():
    """Nothing here compares one pixel against another unless the two arrays
    are the same shape; run_ink_9um.py only calls this once it has checked
    that itself, and this is the backstop if it or a future caller does not."""
    with pytest.raises(ValueError, match="shape"):
        forward_reverse_asymmetry(np.zeros((10, 10)), np.zeros((10, 11)))
