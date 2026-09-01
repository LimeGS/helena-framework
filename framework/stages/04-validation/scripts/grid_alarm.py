"""Is a structure metric measuring the writing, or the render's own grid?

Any structure metric computed over an upsampled render carries a spectral peak
at the upsampling factor. The First Letters campaign measured what that costs:
their row-periodicity score read 0.74 where there is real ink and up to 0.85
where there is none, and in both cases the dominant period was 20 px -- exactly
the mesh cell spacing. The metric was measuring the grid.

A Greek line sits about 5 mm apart, which is roughly 530 px at 9.362 um. Three
orders of magnitude of difference between "text" and "the mesh", and a score
that cannot tell them apart is a score that agrees with itself on blank papyrus.

So this measures the dominant period of the screened window and says which of
the two it is nearer. It marks; it does not overrule the card. A screen whose
structure sits at grid scale is not necessarily wrong -- it is a claim nobody
should read without knowing that the strongest periodic thing in the window is
the render.
"""

from __future__ import annotations

from typing import Any


def dominant_period_px(window) -> dict[str, Any]:
    """The strongest periodic spacing in a window, along each axis.

    The mean profile along rows and along columns, detrended, through a real
    FFT: a grid shows up as one sharp peak in both, a line of text as a broader
    one along the axis across the lines. Periods longer than half the window
    cannot be distinguished from a trend and are not offered as an answer.
    """
    import numpy as np

    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or min(values.shape) < 8:
        return {"rows": None, "columns": None, "period_px": None}

    def axis_period(profile) -> float | None:
        profile = np.asarray(profile, dtype=float)
        finite = profile[np.isfinite(profile)]
        if finite.size < 8 or float(finite.std()) == 0.0:
            return None
        centred = profile - float(np.nanmean(profile))
        centred = np.nan_to_num(centred)
        spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size)))
        if spectrum.size < 3:
            return None
        # Bin 0 is the mean and bin 1 is a period as long as the window: both
        # are trends, not repetition.
        index = int(np.argmax(spectrum[2:])) + 2
        return float(centred.size) / float(index)

    rows = axis_period(np.nanmean(values, axis=1))      # across rows: vertical
    columns = axis_period(np.nanmean(values, axis=0))   # across columns
    candidates = [value for value in (rows, columns) if value is not None]
    return {"rows": rows, "columns": columns,
            "period_px": min(candidates) if candidates else None}


def grid_alarm(
    window,
    *,
    px_um: float,
    render_cell_px: float | None = None,
    line_spacing_um: float = 5000.0,
    tolerance: float = 0.2,
) -> dict[str, Any]:
    """Whether this window's strongest repetition is the render's own grid.

    `render_cell_px` is the render's cell or upsampling spacing, when the job
    carries one: a dominant period within `tolerance` of it is the grid, said
    plainly. Without it the fallback is scale -- a dominant period far below the
    line spacing a script actually has is structure at grid scale, whatever
    produced it.

    Marks, never overrules. The verdict stays the card's.
    """
    measurement = dominant_period_px(window)
    period = measurement["period_px"]
    outcome: dict[str, Any] = {
        "schema": "campaignx.structure_grid_alarm.v1",
        **measurement,
        "px_um": float(px_um),
        "render_cell_px": float(render_cell_px) if render_cell_px else None,
        "line_spacing_um": float(line_spacing_um),
        "alarm": False,
        "reason": None,
    }
    if period is None:
        outcome["reason"] = ("no periodic structure to measure in this window: "
                             "too small, or flat")
        return outcome

    period_um = period * float(px_um)
    outcome["period_um"] = period_um
    outcome["line_spacing_px"] = float(line_spacing_um) / float(px_um)

    if render_cell_px:
        distance = abs(period - float(render_cell_px)) / float(render_cell_px)
        if distance <= float(tolerance):
            outcome.update({
                "alarm": True,
                "reason": (f"the strongest repetition is {period:.1f} px, within "
                           f"{distance:.0%} of the render's own {float(render_cell_px):.1f} px "
                           "cell: this metric is reading the grid")})
            return outcome

    if period_um < float(line_spacing_um) / 4.0:
        outcome.update({
            "alarm": True,
            "reason": (f"the strongest repetition is {period_um:.0f} um, far below "
                       f"the {float(line_spacing_um):.0f} um a line of script sits "
                       "at: structure at grid scale, not text scale")})
        return outcome

    outcome["reason"] = (f"the strongest repetition is {period_um:.0f} um, which is "
                         "text scale rather than grid scale")
    return outcome
