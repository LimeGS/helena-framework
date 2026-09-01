"""A structure score over an upsampled render can be measuring the render.

The First Letters campaign measured the cost: their row-periodicity score read
0.74 where there is real ink and up to 0.85 where there is none, and in both
cases the dominant period was 20 px -- the mesh cell spacing. A Greek line sits
about 5 mm apart, which is roughly 530 px at 9.362 um. A score that cannot tell
those apart agrees with itself on blank papyrus.

The alarm marks; it never overrules the card. What it adds is the one fact a
reader needs before believing a structure number: whether the strongest periodic
thing in the window is the writing or the grid it was rendered on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))

np = pytest.importorskip("numpy")

from grid_alarm import dominant_period_px, grid_alarm  # noqa: E402

PX_UM = 9.362          # the campaign's own render scale
LINE_UM = 5000.0       # a line of Greek, about 5 mm


def striped(period_px: int, *, size: int = 256, axis: int = 0):
    """A window whose only structure is one repetition, along one axis."""
    index = np.arange(size)
    wave = 0.5 + 0.4 * np.sin(2 * np.pi * index / period_px)
    return (np.tile(wave[:, None], (1, size)) if axis == 0
            else np.tile(wave[None, :], (size, 1)))


def test_the_dominant_period_is_the_one_that_is_there() -> None:
    measured = dominant_period_px(striped(20))

    assert measured["period_px"] == pytest.approx(20.0, rel=0.1)


def test_a_flat_window_offers_no_period() -> None:
    assert dominant_period_px(np.full((64, 64), 0.5))["period_px"] is None


def test_the_grid_period_is_named_as_the_grid() -> None:
    """20 px against a 20 px mesh cell: the exact case they measured."""
    outcome = grid_alarm(striped(20), px_um=PX_UM, render_cell_px=20.0,
                         line_spacing_um=LINE_UM)

    assert outcome["alarm"] is True
    assert "reading the grid" in outcome["reason"]


def test_without_a_declared_cell_the_scale_still_gives_it_away() -> None:
    """A job that does not carry the render's cell spacing is the common case,
    and 187 um is three orders off a line of script whatever produced it."""
    outcome = grid_alarm(striped(20), px_um=PX_UM, line_spacing_um=LINE_UM)

    assert outcome["alarm"] is True
    assert "grid scale, not text scale" in outcome["reason"]


def test_text_scale_repetition_is_not_an_alarm() -> None:
    """Lines 5 mm apart at 9.362 um is about 530 px."""
    outcome = grid_alarm(striped(534, size=2048), px_um=PX_UM,
                         render_cell_px=20.0, line_spacing_um=LINE_UM)

    assert outcome["alarm"] is False
    assert "text scale" in outcome["reason"]


def test_a_period_near_the_cell_alarms_without_being_exactly_it() -> None:
    """Sampling and interpolation move the peak a little; 15% off a 20 px cell
    is still the cell."""
    outcome = grid_alarm(striped(23), px_um=PX_UM, render_cell_px=20.0,
                         line_spacing_um=LINE_UM, tolerance=0.2)

    assert outcome["alarm"] is True
    assert "within" in outcome["reason"]


def test_it_reports_what_it_measured_in_both_units() -> None:
    """A reader arguing with this needs the number, not the verdict."""
    outcome = grid_alarm(striped(20), px_um=PX_UM, render_cell_px=20.0)

    assert outcome["period_px"] == pytest.approx(20.0, rel=0.1)
    assert outcome["period_um"] == pytest.approx(20.0 * PX_UM, rel=0.1)
    assert outcome["schema"] == "campaignx.structure_grid_alarm.v1"


def test_a_window_too_small_to_measure_says_so_rather_than_passing() -> None:
    outcome = grid_alarm(np.zeros((4, 4)), px_um=PX_UM)

    assert outcome["alarm"] is False
    assert "too small" in outcome["reason"]
