"""What the 2 um ink lane refuses, and the arithmetic it refuses on.

This is the adapter the canonical lane runs in production, and it was the least
covered of the ink scripts: a third of it, against the 9 um one's tests written
alongside it. What went untested is not the inference -- that needs a card and a
model -- but the checks standing in front of it, and those are the whole reason
a wrong run does not become a scientific record.

Each of these fails silently if it regresses. A disqualified method that routes
anyway produces a map nobody can tell from a good one. A depth window read past
the end of the stack, or a tile grid that misses the last strip, produces a map
that is wrong in a way no exit code reports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "framework/stages/03-ink/scripts/run_ink.py"

pytest.importorskip("torch", reason="the adapter imports torch at module scope")
pytest.importorskip("PIL", reason="the adapter reads layers with Pillow")


def load():
    spec = importlib.util.spec_from_file_location("run_ink_under_test", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lane = load()


# -- the registry gate -------------------------------------------------------


def test_a_disqualified_method_is_refused_before_a_card_is_claimed(monkeypatch) -> None:
    """The liveness gate catches a dead lane after an hour of GPU time. This
    catches the ones already known dead, before any."""
    monkeypatch.setattr(lane, "registry_status", lambda m: {
        "method_id": m, "validation_status": "INTEGRATED_THEN_DISQUALIFIED",
        "recommended_policy": "do not route",
    })

    with pytest.raises(RuntimeError) as refused:
        lane.check_routable("ink-something@1.0.0")

    assert "DISQUALIFIED" in str(refused.value)
    assert "do not route" in str(refused.value)


def test_a_method_the_registry_does_not_mention_is_allowed(monkeypatch) -> None:
    """Absent is not disqualified. Refusing the unknown would stop every method
    the registry has not caught up with, which is how a gate becomes a blocker
    somebody removes."""
    monkeypatch.setattr(lane, "registry_status", lambda m: None)

    lane.check_routable("ink-brand-new@0.1.0")


def test_a_method_recorded_as_working_is_allowed(monkeypatch) -> None:
    monkeypatch.setattr(lane, "registry_status", lambda m: {
        "method_id": m, "validation_status": "RUNNABLE_PRIMARY"})

    lane.check_routable("ink-canonical-2um@1.0.0")


# -- the profile it is handed ------------------------------------------------


def a_profile(tmp_path: Path, **overrides) -> Path:
    body = {
        "schema": "campaignx.ink_lane_profile.v1",
        "profile_id": "ink-canonical-2um-screening@1.0.0",
        "method_id": "ink-canonical-2um@1.0.0",
        # model_type is checked before the tile is, so it has to be one this
        # runner dispatches or every case below fails on the wrong refusal.
        "input_contract": {"tile_size_y_x": [64, 64], "frames": 26,
                           "training_pixel_um": 7.91,
                           "model_type": "resnet3d-152"},
        "checkpoint_sha256": "a" * 64,
    }
    body.update(overrides)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(body))
    return path


def test_a_document_that_is_not_a_lane_profile_is_refused(tmp_path) -> None:
    path = a_profile(tmp_path, schema="campaignx.something_else.v1")

    with pytest.raises(RuntimeError) as refused:
        lane.read_profile(path)

    assert "not an ink lane profile" in str(refused.value)


def test_a_tile_that_is_not_square_is_refused(tmp_path) -> None:
    """The window is applied as an outer product of one 1-D taper, so a
    non-square tile would weight the two axes differently without saying so."""
    path = a_profile(tmp_path, input_contract={
        "tile_size_y_x": [64, 32], "frames": 26, "training_pixel_um": 7.91,
        "model_type": "resnet3d-152"})

    with pytest.raises(RuntimeError) as refused:
        lane.read_profile(path)

    assert "square" in str(refused.value)


# -- the depth window --------------------------------------------------------


def test_the_depth_window_is_centred_and_evenly_spaced() -> None:
    """Taking consecutive slices from a 2.399 um stack for a model trained at
    7.9 um hands it a slab 3.3x too thin, and the model answers with a
    near-constant map. The spacing is what prevents that."""
    positions = lane.depth_positions(centre=10.0, frames=5, step=3.3)

    assert len(positions) == 5
    assert positions[2] == pytest.approx(10.0)
    assert np.allclose(np.diff(positions), 3.3)


def test_an_even_frame_count_straddles_the_centre() -> None:
    positions = lane.depth_positions(centre=10.0, frames=4, step=1.0)

    assert positions.mean() == pytest.approx(10.0)
    assert 10.0 not in positions


def test_a_window_that_runs_off_the_stack_is_refused(tmp_path) -> None:
    """Reading past the end silently would sample the same edge slice several
    times and report a map computed from a depth that was never there."""
    files = [tmp_path / f"{n:02d}.tif" for n in range(5)]

    with pytest.raises(RuntimeError) as refused:
        lane.read_interpolated_depth(files, np.array([-1.0, 0.0, 1.0]))
    assert "outside the 5 available layers" in str(refused.value)

    with pytest.raises(RuntimeError):
        lane.read_interpolated_depth(files, np.array([3.0, 4.0, 5.0]))


# -- the tiling --------------------------------------------------------------


def test_the_grid_always_reaches_the_last_strip() -> None:
    """Without the final origin the right and bottom edges go unscreened, and
    the map is blank there rather than negative -- a difference nothing
    downstream can see."""
    xs = lane.grid(length=200, tile=64, stride=50)

    assert xs[0] == 0
    assert xs[-1] == 200 - 64
    assert all(x + 64 <= 200 for x in xs)


def test_the_grid_does_not_repeat_an_origin() -> None:
    """The last strip is added to a set, so an exact fit must not tile twice --
    the overlap accumulator would weight that strip double."""
    xs = lane.grid(length=192, tile=64, stride=64)

    assert xs == sorted(set(xs))
    assert xs == [0, 64, 128]


def test_an_axis_smaller_than_one_tile_is_refused() -> None:
    with pytest.raises(RuntimeError) as refused:
        lane.grid(length=32, tile=64, stride=32)

    assert "smaller than tile" in str(refused.value)


# -- the blend window --------------------------------------------------------


def test_the_window_peaks_at_one_and_never_reaches_zero() -> None:
    """A taper that touched zero at the border would leave seams unweighted,
    and the accumulator divides by the summed weight."""
    w = lane.hann2d(64)

    assert w.shape == (64, 64)
    assert w.max() == pytest.approx(1.0)
    assert w.min() > 0.0


def test_the_window_is_symmetric_in_both_axes() -> None:
    w = lane.hann2d(32)

    assert np.allclose(w, w.T)
    assert np.allclose(w, w[::-1, :])
