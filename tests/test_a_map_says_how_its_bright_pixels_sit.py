"""Liveness could not tell salt-and-pepper noise from strokes.

Its three gates all read the distribution of values -- spread, standard
deviation, how much piles up at 0.5 -- and a distribution cannot distinguish a
map whose bright pixels are individually isolated from one whose bright pixels
form lines. Both have the same histogram.

That is not hypothetical here. Measured on PHerc826, the 9 um lane's brightest
1% formed 332 connected components with a median size of one pixel; on the
public positive control, 10,957 components, also median one. The TimeSformer
lane on the same surface formed 18 components with a median of 579 pixels. All
of them passed every liveness gate with room to spare, and five surfaces were
compared against each other on a p99 that came out of isolated hot pixels.

So the receipt now carries how the bright pixels sit. It is reported, not
enforced: what counts as too fragmented depends on the render's scale, and a
threshold fitted to two lanes on one scroll is the kind of post-hoc number this
platform refuses elsewhere. Measuring it is what makes calibrating it possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.contracts.lane_liveness import assess_liveness  # noqa: E402


def maps():
    """Two maps with the same background and the same bright fraction.

    Only where the bright pixels are differs, which is exactly the difference
    the value gates cannot see.
    """
    rng = np.random.default_rng(0)
    base = np.clip(rng.normal(0.30, 0.05, (400, 400)), 0, 1)

    speckle = base.copy()
    speckle.ravel()[rng.choice(400 * 400, 1600, replace=False)] = 0.9

    strokes = base.copy()
    for row in range(60, 340, 40):
        strokes[row:row + 6, 60:260] = 0.9
    return speckle, strokes


def test_speckle_and_strokes_are_told_apart():
    speckle, strokes = maps()
    a = assess_liveness(speckle)["metrics"]
    b = assess_liveness(strokes)["metrics"]

    assert a["top1_median_component_px"] == 1
    assert b["top1_median_component_px"] > 100
    assert a["top1_components"] > 100 * b["top1_components"]
    assert a["top1_share_in_components_over_10px"] < 0.05
    assert b["top1_share_in_components_over_10px"] > 0.95


def test_the_value_gates_pass_both_which_is_the_whole_point():
    """If the existing gates could separate these, none of this would be needed."""
    speckle, strokes = maps()
    assert assess_liveness(speckle)["verdict"] == "ALIVE"
    assert assess_liveness(strokes)["verdict"] == "ALIVE"


def test_the_verdict_is_unchanged_by_the_new_measurement():
    """Reported, not enforced. A map that was ALIVE stays ALIVE, and a
    degenerate one still fails on the reason it always failed on."""
    flat = np.full((400, 400), 0.5)
    verdict = assess_liveness(flat)
    assert verdict["verdict"] == "DEGENERATE"
    assert "p99-p50" in verdict["reason"]


def test_a_valid_mask_is_honoured():
    """The measurement must not count the region outside the render."""
    rng = np.random.default_rng(1)
    array = np.clip(rng.normal(0.3, 0.05, (200, 200)), 0, 1)
    array[:100] = 0.0
    valid = np.zeros((200, 200), bool)
    valid[100:] = True
    metrics = assess_liveness(array, valid=valid)["metrics"]
    assert metrics["valid_pixels"] == 100 * 200


def test_too_small_a_map_says_so_rather_than_reporting_a_number():
    tiny = np.linspace(0, 1, 400).reshape(20, 20)
    metrics = assess_liveness(tiny)["metrics"]
    assert "unmeasured" in str(metrics.get("spatial_character", ""))


def test_the_9um_lane_publishes_the_reverse_map_it_was_paying_for():
    """`direction: both` runs inference twice because which side faces the ink
    is a measurement. The runner read ink.tif, published it, and left
    ink_reverse.tif on the disk unread -- with nothing in the receipt saying
    which side the published map came from."""
    source = (ROOT / "framework/stages/03-ink/scripts/run_ink_9um.py").read_text()
    assert 'REVERSE_MAP_NAME = "ink_reverse.tif"' in source
    assert 'np.save(directory / "probability_reverse.npy"' in source
    assert '"pearson_r_with_forward"' in source, (
        "publishing both is not enough: a reader needs to know how far apart "
        "they are to know whether the choice of side mattered")
    assert '"published_map"' in source and '"direction"' in source


def test_the_9um_receipt_states_its_quantisation():
    """255 steps across a range that occupies about 0.22 to 0.81. A reader
    comparing this lane against a float lane is comparing resolutions too."""
    source = (ROOT / "framework/stages/03-ink/scripts/run_ink_9um.py").read_text()
    assert '"value_quantisation"' in source
    assert '"distinct_levels"' in source
