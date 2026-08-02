"""The seed service has to read the region the queue actually writes.

The fleet encodes a probe box as centre and radius. This module was rebuilt from
a prose contract after the original service was lost, and prose does not say
whether a box is corners or a centre -- so it expected x_min/x_max, and every
task that reached it died on `KeyError: 'x_min'`.

The failure surfaced as BLOCKED_SOURCE_UNAVAILABLE, which reads as the bucket
being down, and it went unnoticed for days because nothing had ever been queued
through this path. This is the shape, pinned, taken from a real task packet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MCP = Path(__file__).resolve().parents[1] / "framework/stages/01-segmentation/mcp"
sys.path.insert(0, str(MCP))

from seed_candidates import SeedSearchError, region_bounds  # noqa: E402

# Copied from segment_tasks.payload->candidate_discovery->region on the live
# control plane, not invented.
FLEET_REGION = {"center": {"x": 64, "y": 64, "z": 16448},
                "radius": {"x": 64, "y": 64, "z": 64}}


def test_the_shape_the_fleet_sends_is_understood():
    assert region_bounds(FLEET_REGION) == {"x": (0, 128), "y": (0, 128),
                                           "z": (16384, 16512)}


def test_corners_still_work_for_a_hand_built_call():
    bounds = region_bounds({"x_min": 10, "x_max": 20, "y_min": 0, "y_max": 8,
                            "z_min": 5, "z_max": 9})
    assert bounds == {"x": (10, 20), "y": (0, 8), "z": (5, 9)}


def test_a_centre_near_the_edge_clamps_instead_of_going_negative():
    """A negative index does not fail on a zarr array -- it wraps.

    Which means an unclamped low bound returns geometry from the far side of the
    volume and calls it a candidate here.
    """
    bounds = region_bounds({"center": {"x": 8, "y": 8, "z": 8},
                            "radius": {"x": 64, "y": 64, "z": 64}})
    assert all(low == 0 for low, _ in bounds.values())


def test_an_unrecognised_region_says_what_it_saw():
    with pytest.raises(SeedSearchError) as failure:
        region_bounds({"bbox": [0, 0, 0, 1, 1, 1]})
    assert "bbox" in str(failure.value)
