#!/usr/bin/env python3
"""Producers for the Stage 02 flattening gates that had none.

Three quantities were measured somewhere in the pipeline but never consumed:

* the renderer's ``Rasterized N / M points (X%)`` line, which reached at worst
  73.9% and was treated exactly like a 96.5% render;
* the exploratory window's ``valid_fraction``, which reached 0.7855 while the
  window was still exported for ink screening with ``-1.0`` sentinels in place
  of the missing 21.5%;
* ``target_winding_index``, stored in the quality-map NPZ with zero readers, so
  a four-neighbour connected ROI could straddle a sheet jump.

Each function here is a fail-closed producer for one of those gates.  None of
them relaxes an existing check: they exist so that checks the profile already
declares can actually fire.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


#: ``Rasterized 11647 / 15750 points (73.9492%)``
RASTERIZED_LINE = re.compile(
    r"Rasterized\s+(?P<rasterized>\d+)\s*/\s*(?P<output>\d+)\s+points"
    r"(?:\s*\(\s*(?P<percent>[0-9]*\.?[0-9]+)\s*%\s*\))?"
)

#: Derived from the observed fleet distribution, not chosen a priori.
#:
#: The 0.95 originally specified in the 2026-07-24 pipeline audit FIX-12 was
#: not evidence-based.  Measured over all 166 archived render logs the fleet
#: sits at median 0.9429 (p05 0.9349, max 0.9660); only 22 of 166 clear 0.95,
#: so that gate would have ordered a re-render of 86% of the corpus.
#:
#: More decisively, raster fraction has no demonstrated relationship to screen
#: outcome: the U0 PHerc0139 VC3D positive control rasterized at 0.9635 — top
#: 5% of the whole fleet — and still produced 1 glyph candidate against a
#: minimum of 10, while official surfaces at the same physical location produce
#: 15.  Whatever suppresses that control, it is not rasterization coverage.
#:
#: This floor therefore isolates the single genuine outlier (0.7395, a 26% loss
#: that does become zero fill) and passes 165 of 166.  It is a fail-closed
#: guard against gross degradation, NOT a quality threshold.  Raising it
#: requires a control that shows raster fraction predicts screen outcome.
MINIMUM_VALID_RASTER_FRACTION = 0.90

#: Unchanged from FIX-12.  This one is evidence-based: the sentinel `-1.0` fill
#: outside a requested window is the documented COVERAGE_BOUNDARY_RENDER_ARTIFACT
#: confounder, and the five failing receipts (0.7855-0.8864) are the surfaces
#: that produced it.
MINIMUM_WINDOW_VALID_FRACTION = 0.95
SENTINEL_AUDIT_RADIUS_QUADS = 1
MAXIMUM_ROI_WINDING_STEP = 1


class FlatteningGateError(RuntimeError):
    """Raised when a Stage 02 artifact must not cross into inference."""


def parse_rasterized_fractions(text: str) -> list[dict[str, float | int]]:
    """Extract every ``Rasterized N / M points`` observation from a log."""

    observations: list[dict[str, float | int]] = []
    for match in RASTERIZED_LINE.finditer(text):
        rasterized = int(match.group("rasterized"))
        output = int(match.group("output"))
        if output <= 0:
            raise FlatteningGateError(
                f"render log reports {output} output points; a render with no "
                "output points cannot be screened"
            )
        if rasterized > output:
            raise FlatteningGateError(
                f"render log reports {rasterized} rasterized of {output} output "
                "points, which is not a fraction"
            )
        fraction = rasterized / output
        percent = match.group("percent")
        if percent is not None and not math.isclose(
            float(percent) / 100.0, fraction, rel_tol=0.0, abs_tol=1e-4
        ):
            raise FlatteningGateError(
                f"render log percentage {percent}% disagrees with "
                f"{rasterized}/{output} = {fraction * 100:.4f}%"
            )
        observations.append(
            {
                "rasterized_points": rasterized,
                "output_points": output,
                "valid_raster_fraction": fraction,
            }
        )
    return observations


def evaluate_raster_gate(
    text: str,
    *,
    minimum_valid_raster_fraction: float = MINIMUM_VALID_RASTER_FRACTION,
) -> dict[str, Any]:
    """Measure and gate a render log's rasterized-point coverage.

    The unrasterized remainder becomes zero fill in the rendered stack, which
    is the documented ink false-positive confuser, so a log that never reports
    the line at all is a failure rather than a pass: an unmeasured render is
    not a good render.  When a log carries several observations the worst one
    decides.
    """

    if not 0.0 < minimum_valid_raster_fraction <= 1.0:
        raise FlatteningGateError(
            "minimum_valid_raster_fraction must lie in (0, 1]"
        )
    observations = parse_rasterized_fractions(text)
    if not observations:
        raise FlatteningGateError(
            "render log carries no 'Rasterized N / M points' line; the "
            "rasterized fraction is UNMEASURED and cannot be gated"
        )
    worst = min(observations, key=lambda row: row["valid_raster_fraction"])
    result = {
        "gate": "P2_RENDER_VALID_RASTER_FRACTION",
        "observation_count": len(observations),
        "minimum_valid_raster_fraction": float(minimum_valid_raster_fraction),
        "observations": observations,
        **worst,
    }
    result["passed"] = bool(
        worst["valid_raster_fraction"] >= minimum_valid_raster_fraction
    )
    if not result["passed"]:
        raise FlatteningGateError(
            "render rasterized only "
            f"{worst['rasterized_points']}/{worst['output_points']} points "
            f"({worst['valid_raster_fraction'] * 100:.4f}%), below the "
            f"{minimum_valid_raster_fraction * 100:.2f}% gate; the remainder "
            "becomes zero fill and is the documented ink confuser"
        )
    return result


def dilate_quads(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation of a boolean quad mask by ``radius`` cells."""

    mask = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise FlatteningGateError("audit radius must be non-negative")
    result = mask.copy()
    for _ in range(int(radius)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(result)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                grown |= padded[
                    di : di + result.shape[0], dj : dj + result.shape[1]
                ]
        result = grown
    return result


def evaluate_window_validity(
    window: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_valid_fraction: float = MINIMUM_WINDOW_VALID_FRACTION,
    sentinel_audit_radius_quads: int = SENTINEL_AUDIT_RADIUS_QUADS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gate a requested window and return the validity-intersected mask.

    ``window`` is the purely rectangular request.  The exported mask is the
    intersection with ``valid``, so a vertex outside the surface never reaches
    the TIFXYZ as a ``-1.0`` sentinel.  Two conditions must hold before an
    export is allowed:

    1. the requested rectangle is at least ``minimum_valid_fraction`` valid, so
       a window that is mostly hole is refused rather than silently shrunk; and
    2. no exported quad sits within ``sentinel_audit_radius_quads`` of an
       invalid quad, because a rendered value one cell from the sentinel is
       already contaminated by the interpolation that produced it.
    """

    window = np.asarray(window, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if window.shape != valid.shape:
        raise FlatteningGateError(
            f"window shape {window.shape} does not match validity map {valid.shape}"
        )
    requested = int(window.sum())
    if requested == 0:
        raise FlatteningGateError("requested window is empty")
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise FlatteningGateError("minimum_valid_fraction must lie in (0, 1]")

    exported = window & valid
    valid_fraction = float(exported.sum() / requested)
    guard_band = dilate_quads(~valid, int(sentinel_audit_radius_quads)) & valid
    near_sentinel = int((exported & guard_band).sum())
    report = {
        "gate": "P2_WINDOW_SENTINEL_FREE_VALID_FRACTION",
        "requested_quad_count": requested,
        "valid_quad_count": int(exported.sum()),
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction": float(minimum_valid_fraction),
        "sentinel_audit_radius_quads": int(sentinel_audit_radius_quads),
        "quads_within_audit_radius_of_sentinel": near_sentinel,
        "mask_intersected_with_valid": True,
    }
    report["passed"] = bool(
        valid_fraction >= minimum_valid_fraction and near_sentinel == 0
    )
    if valid_fraction < minimum_valid_fraction:
        raise FlatteningGateError(
            f"requested window is only {valid_fraction:.4f} valid, below the "
            f"{minimum_valid_fraction:.4f} export gate; the invalid remainder "
            "would be written as the -1.0 TIFXYZ sentinel"
        )
    if near_sentinel:
        raise FlatteningGateError(
            f"{near_sentinel} exported quads sit within "
            f"{sentinel_audit_radius_quads} cell of the -1.0 sentinel; "
            "move the window away from the surface border"
        )
    return exported, report


def label_winding_aware_components(
    mask: np.ndarray,
    winding: np.ndarray | None,
    *,
    maximum_winding_step: int = MAXIMUM_ROI_WINDING_STEP,
) -> tuple[np.ndarray, int]:
    """Label four-connected components, cutting them at sheet jumps.

    Two quads are joined only when both are in ``mask``, are four-neighbours,
    and their ``target_winding_index`` differs by strictly less than
    ``maximum_winding_step``.  A winding discontinuity is a wrap of the spiral,
    so a component that crossed it would be one contiguous region in the map
    and two different physical sheets in the scroll.

    ``winding`` may be ``None`` only when the caller has proved no winding
    field exists; in that case labelling degrades to plain four-connectivity
    and the caller is responsible for recording that the cut was UNMEASURED.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise FlatteningGateError("component mask must be two-dimensional")
    if maximum_winding_step <= 0:
        raise FlatteningGateError("maximum_winding_step must be positive")

    if winding is not None:
        winding = np.asarray(winding, dtype=np.float64)
        if winding.shape != mask.shape:
            raise FlatteningGateError(
                f"winding shape {winding.shape} does not match mask {mask.shape}"
            )

    flat = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    left_nodes: list[np.ndarray] = []
    right_nodes: list[np.ndarray] = []
    for axis in (0, 1):
        if mask.shape[axis] < 2:
            continue
        head = [slice(None), slice(None)]
        tail = [slice(None), slice(None)]
        head[axis] = slice(0, mask.shape[axis] - 1)
        tail[axis] = slice(1, mask.shape[axis])
        both = mask[tuple(head)] & mask[tuple(tail)]
        if winding is not None:
            step = np.abs(winding[tuple(tail)] - winding[tuple(head)])
            joinable = both & np.isfinite(step) & (step < maximum_winding_step)
        else:
            joinable = both
        left_nodes.append(flat[tuple(head)][joinable])
        right_nodes.append(flat[tuple(tail)][joinable])

    selected = np.flatnonzero(mask.reshape(-1))
    if selected.size == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0
    position = np.full(mask.size, -1, dtype=np.int64)
    position[selected] = np.arange(selected.size, dtype=np.int64)
    rows = position[np.concatenate(left_nodes)] if left_nodes else np.empty(0, np.int64)
    columns = (
        position[np.concatenate(right_nodes)] if right_nodes else np.empty(0, np.int64)
    )
    graph = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)),
        shape=(selected.size, selected.size),
    ).tocsr()
    count, membership = csgraph.connected_components(
        graph, directed=False, return_labels=True
    )

    # Renumber so component ids follow the raster order of their first cell:
    # ``ndimage.label`` does the same, and downstream receipts compare ids.
    order = np.full(count, -1, dtype=np.int64)
    next_id = 0
    for value in membership.tolist():
        if order[value] < 0:
            order[value] = next_id
            next_id += 1
    labels = np.zeros(mask.size, dtype=np.int32)
    labels[selected] = (order[membership] + 1).astype(np.int32)
    return labels.reshape(mask.shape), int(count)


def winding_step_report(
    mask: np.ndarray,
    winding: np.ndarray | None,
    *,
    maximum_winding_step: int = MAXIMUM_ROI_WINDING_STEP,
) -> dict[str, Any]:
    """Summarize the winding discontinuities inside one selected region."""

    mask = np.asarray(mask, dtype=bool)
    if winding is None:
        return {
            "gate": "P2_ROI_WINDING_CONTINUITY",
            "status": "UNMEASURED",
            "maximum_winding_step": int(maximum_winding_step),
            "target_winding_index_available": False,
        }
    winding = np.asarray(winding, dtype=np.float64)
    crossings = 0
    largest = 0.0
    for axis in (0, 1):
        if mask.shape[axis] < 2:
            continue
        head = [slice(None), slice(None)]
        tail = [slice(None), slice(None)]
        head[axis] = slice(0, mask.shape[axis] - 1)
        tail[axis] = slice(1, mask.shape[axis])
        both = mask[tuple(head)] & mask[tuple(tail)]
        step = np.abs(winding[tuple(tail)] - winding[tuple(head)])[both]
        step = step[np.isfinite(step)]
        if step.size:
            largest = max(largest, float(step.max()))
            crossings += int(np.count_nonzero(step >= maximum_winding_step))
    return {
        "gate": "P2_ROI_WINDING_CONTINUITY",
        "status": "PASS" if crossings == 0 else "FAIL",
        "maximum_winding_step": int(maximum_winding_step),
        "target_winding_index_available": True,
        "adjacent_pair_crossing_count": crossings,
        "largest_adjacent_winding_step": largest,
    }


def component_summaries(
    labels: np.ndarray, cell_area_cm2: float
) -> list[dict[str, Any]]:
    """Deterministic per-component geometry, largest area first."""

    labels = np.asarray(labels)
    summaries: list[dict[str, Any]] = []
    for component_id in range(1, int(labels.max()) + 1 if labels.size else 1):
        ii, jj = np.where(labels == component_id)
        if not len(ii):
            continue
        summaries.append(
            {
                "component_id": component_id,
                "quad_count": int(len(ii)),
                "area_cm2": float(len(ii) * cell_area_cm2),
                "bbox_quad_ij": [
                    int(ii.min()),
                    int(jj.min()),
                    int(ii.max() + 1),
                    int(jj.max() + 1),
                ],
            }
        )
    summaries.sort(key=lambda item: (-item["area_cm2"], item["component_id"]))
    return summaries
