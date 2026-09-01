"""Does the CT resolve a single lamina under this surface?

P2 certifies mesh integrity -- seams, self-intersection, folds -- and says so in
its own non-claims: *not a claim that the segmentation followed the correct
lamina*. The vetting card's contrast bimodality is Otsu variance over an ink
map, against blob pareidolia. Neither looks at the CT density profile along the
normal, and that is the question which decides whether rendering is worth 29
minutes: are there two air/papyrus interfaces, one sheet's thickness apart, on
this surface, in this volume?

This is the measurement, as three numbers per surface:

  bimodality        h(interface level) / min(height of the two modes). A real
                    interface sits in a valley between the air mode and the
                    papyrus mode, so the ratio falls below 1.
  median thickness  the half-max crossing pair, in microns. A papyrus sheet is
                    about 35 um. Three to five times that means the pair of
                    interfaces spanned neighbouring laminae, and a render there
                    produces a slab that is not a sheet.
  clean fraction    columns with no missing chunk anywhere along them. Sampling
                    is per column and a column with a hole is not evidence.

Thickness is the one that discriminates. Measured over nine wraps of PHerc0826,
bimodality passed on all of them and thickness separated cleanly: 35-45 um for
the wraps that are one sheet, 120-170 um for the fused ones, with nothing in
between. The bands live in a frozen profile rather than here, because they are a
calibration and this is the arithmetic.

What this does NOT claim, and the campaign that measured it says so first: the
gate is geometric, not about content. In their run a wrap passed comfortably --
32-36 um across three bands -- and the patch turned out to hold no ink at all.
It filters where one *can* look, never where there is something to see.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# What a column can be, before it is aggregated into a verdict.
COLUMN_CLEAN = "CLEAN"
COLUMN_HOLED = "HOLED"           # a missing chunk anywhere along it
COLUMN_NO_CROSSING = "NO_CROSSING"  # no interface pair in it at all
COLUMN_SATURATED = "SATURATED"   # above half maximum all the way to both ends:
                                 # the pair is wider than the window, which is
                                 # evidence of a slab rather than an absence


def interface_level(values: Sequence[float], *, bins: int = 64) -> float | None:
    """Where air stops and papyrus starts, in intensity, over many columns.

    Taken from the pooled samples rather than from one column, because a column
    that lies entirely inside the material has no two levels to be halfway
    between -- and that column is the fused case, the one this gate exists to
    catch. The level is the valley between the two modes; with no valley there
    is nothing to separate and the answer is None.
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < bins or bins < 4:
        return None
    low, high = min(finite), max(finite)
    if high <= low:
        return None
    width = (high - low) / bins
    counts = [0] * bins
    for value in finite:
        counts[min(bins - 1, int((value - low) / width))] += 1
    peaks = sorted(range(bins), key=lambda index: counts[index], reverse=True)
    first = peaks[0]
    second = next((index for index in peaks[1:] if abs(index - first) > 1), None)
    if second is None:
        return None
    lower, upper = sorted((first, second))
    # The middle of the valley, not its first bin. Between two well separated
    # materials the valley is wide and flat -- often empty for many bins -- and
    # taking the first minimum puts the level hard against the air mode, which
    # measures the outer edge of every interface ramp and reads a sheet several
    # microns thicker than it is.
    floor_count = min(counts[lower:upper + 1])
    flat = [index for index in range(lower, upper + 1)
            if counts[index] == floor_count]
    valley = flat[len(flat) // 2]
    return low + (valley + 0.5) * width


def half_maximum_crossings(profile: Sequence[float], *,
                           level: float | None = None) -> tuple[float, float] | None:
    """Where a profile crosses the interface level, on the way up and down.

    Interpolated between samples rather than snapped to one: at 9.362 um a
    sheet is four samples thick, and rounding each crossing to a sample is a
    9 um quantisation on a 35 um measurement -- a quarter of the quantity.

    `level` is the crossing threshold. Given one -- from `interface_level` over
    every column of the surface -- the measurement is against where air stops
    for this volume. Without one it falls back to half of the column's own
    height, which is only meaningful when the column contains both materials.

    Returns None when the profile does not cross on both sides of its peak.
    """
    values = [float(value) for value in profile]
    if len(values) < 3:
        return None
    low, high = min(values), max(values)
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    if level is None:
        if high <= low:
            return None
        half = low + (high - low) / 2.0
    else:
        half = float(level)
    peak = values.index(high)

    def outward(step: int) -> float | None:
        # From the peak, out to the first sample below half. The crossing is
        # between that sample and the one before it, which is why this walks
        # outward rather than scanning from the edge: a profile can dip below
        # half more than once, and the pair that bounds the peak is the one
        # this measurement is about.
        index = peak
        while 0 <= index + step < len(values):
            following = index + step
            if values[following] < half:
                span = values[index] - values[following]
                if span == 0:
                    return float(index)
                return index + step * ((values[index] - half) / span)
            index = following
        return None      # never falls below half on this side: no interface

    rising = outward(-1)
    falling = outward(1)
    if rising is None and falling is None:
        # Above half at both ends. Either the window is entirely inside the
        # material -- which is what a fused pair looks like through too short a
        # window -- or the profile has no structure at all. `saturated` above
        # separates the two.
        return None
    if rising is None or falling is None:
        return None
    return (min(rising, falling), max(rising, falling))


def is_saturated(profile: Sequence[float], *, level: float | None = None) -> bool:
    """Whether the whole window sits in the material.

    A column that is above the interface level from end to end did not fail to
    be measured: the pair of interfaces it belongs to is wider than the window
    that was read. Reporting that as "no measurement" is how the most obviously
    fused surface -- the one a column cannot get out of -- passes for a surface
    nobody could measure.

    Needs the level from the surface as a whole. A column of constant material
    has no two levels of its own to be halfway between, which is exactly why
    this is the case that was getting lost.
    """
    values = [float(value) for value in profile]
    if len(values) < 3 or level is None:
        return False
    return values[0] >= float(level) and values[-1] >= float(level)


def column_material_depth(
    profile: Sequence[float], *, sample_step_um: float,
    missing: Sequence[bool] | None = None,
    level: float | None = None,
) -> dict[str, Any]:
    """One column's thickness in microns, and whether it is evidence at all.

    `missing` marks samples the volume could not answer for -- a chunk absent
    from the store, not a zero. A column with a hole is dropped rather than
    measured: the hole reads as air, which is exactly what a thickness
    measurement is looking for.
    """
    if missing is not None and any(missing):
        return {"state": COLUMN_HOLED, "thickness_um": None}
    crossings = half_maximum_crossings(profile, level=level)
    if crossings is None:
        if is_saturated(profile, level=level):
            return {"state": COLUMN_SATURATED, "thickness_um": None,
                    "window_um": float(len(profile) * float(sample_step_um))}
        return {"state": COLUMN_NO_CROSSING, "thickness_um": None}
    lower, upper = crossings
    return {"state": COLUMN_CLEAN,
            "thickness_um": float((upper - lower) * float(sample_step_um)),
            "crossings": [float(lower), float(upper)]}


def histogram_bimodality(values: Sequence[float], *, bins: int = 64) -> float | None:
    """The valley between the two modes, over the smaller of them.

    Below 1 the profile has two populations with a dip between them, which is
    what air and papyrus look like. At or above 1 there is no valley to speak
    of: one material, or noise.

    None when there is no second mode at all -- a flat or single-peaked
    histogram is not a bimodality of some large value, it is an absence of the
    measurement.
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < bins or bins < 4:
        return None
    low, high = min(finite), max(finite)
    if high <= low:
        return None
    width = (high - low) / bins
    counts = [0] * bins
    for value in finite:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1

    # The two tallest peaks that are not neighbours, and the lowest count
    # between them. Neighbouring bins are one mode seen twice.
    peaks = sorted(range(bins), key=lambda index: counts[index], reverse=True)
    first = peaks[0]
    second = next((index for index in peaks[1:] if abs(index - first) > 1), None)
    if second is None:
        return None
    lower, upper = sorted((first, second))
    valley = min(counts[lower:upper + 1])
    floor = min(counts[first], counts[second])
    if floor == 0:
        return None
    return float(valley) / float(floor)


def assess_lamina(
    columns: Sequence[dict[str, Any]],
    *,
    profile: dict[str, Any],
    bimodality: float | None = None,
    interface_level: float | None = None,
    level_was_measured: bool = True,
) -> dict[str, Any]:
    """The three numbers, and the verdict the frozen bands give them.

    `columns` are `column_material_depth` results. `profile` carries the bands;
    it is passed in rather than read here so a verdict can name the calibration
    it was given and a test can state its own.
    """
    thicknesses = sorted(column["thickness_um"] for column in columns
                         if column.get("thickness_um") is not None)
    clean = len(thicknesses)
    total = len(columns)
    clean_fraction = (clean / total) if total else 0.0
    median = _median(thicknesses)

    bands = profile["thickness_um"]
    minimum_clean = float(profile["minimum_clean_fraction"])
    ceiling = profile.get("bimodality_ceiling")

    measurement = {
        "schema": "campaignx.lamina_gate.v1",
        "profile_id": profile["profile_id"],
        "columns_sampled": total,
        "columns_clean": clean,
        "clean_fraction": clean_fraction,
        "median_thickness_um": median,
        "bimodality": bimodality,
        "interface_level": interface_level,
        "sheet_thickness_um": [float(bands["sheet_low"]), float(bands["sheet_high"])],
    }

    # Order matters, and it is the order of what the measurement can support.
    # Too few clean columns is not a thin sheet or a fused one: it is a surface
    # this volume could not answer for, and calling that a verdict would put a
    # judgement on a hole.
    # No level at all means the columns held one population: every sample the
    # same material, or noise with no valley in it. The columns are readable and
    # the two interfaces are not there to read, which is a different answer from
    # "too few columns" -- and saying the latter would file a featureless volume
    # under a hole in the sampling.
    if total and not level_was_measured:
        return {**measurement, "state": "LAMINA_UNRESOLVED",
                "reason": ("no air/papyrus valley anywhere on this surface: the "
                           "sampled window holds one population, which is one "
                           "material or noise")}

    # A window full of material is a slab measured through too short a window,
    # not a column nobody could read. Answered before the clean-fraction test,
    # which would otherwise call the clearest fused surface unmeasurable.
    saturated = [column for column in columns
                 if column.get("state") == COLUMN_SATURATED]
    if total and len(saturated) > total / 2:
        window = next((column.get("window_um") for column in saturated
                       if column.get("window_um")), None)
        return {**measurement, "state": "LAMINA_FUSED",
                "saturated_fraction": len(saturated) / total,
                "reason": ("most columns never leave the material inside the "
                           + (f"{window:.0f} um " if window else "")
                           + "window: the interface pair is wider than what was "
                             "read, which is a slab")}
    if total == 0 or clean_fraction < minimum_clean or median is None:
        return {**measurement, "state": "LAMINA_INSUFFICIENT_COLUMNS",
                "reason": (f"{clean} of {total} columns carried an interface pair; "
                           f"the profile wants {minimum_clean:.0%}")}
    if ceiling is not None and bimodality is not None and bimodality >= float(ceiling):
        return {**measurement, "state": "LAMINA_UNRESOLVED",
                "reason": (f"no valley between the modes: bimodality {bimodality:.3f} "
                           f"is at or above {float(ceiling):.3f}")}
    if median > float(bands["sheet_high"]):
        return {**measurement, "state": "LAMINA_FUSED",
                "reason": (f"median thickness {median:.1f} um is above "
                           f"{float(bands['sheet_high']):.1f}: the interface pair "
                           "spans more than one lamina")}
    if median < float(bands["sheet_low"]):
        return {**measurement, "state": "LAMINA_TOO_THIN",
                "reason": (f"median thickness {median:.1f} um is below "
                           f"{float(bands['sheet_low']):.1f}: thinner than a sheet, "
                           "so the pair is unlikely to be two interfaces")}
    return {**measurement, "state": "LAMINA_SINGLE_SHEET",
            "reason": (f"median thickness {median:.1f} um sits in "
                       f"{float(bands['sheet_low']):.1f}-{float(bands['sheet_high']):.1f}")}


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)
