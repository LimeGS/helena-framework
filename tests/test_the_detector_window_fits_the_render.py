"""The detector window against the stack P4 actually rendered.

A lane profile's depth centres are written for the stack depth its author had:
the GP Scroll1 lane says 25, 32 and 39, which are positions in a 62-layer
surface volume. Pointed at the 33-slice render this pipeline produces they fall
off the end, and the failure arrives as "depth positions extend beyond the
source stack" -- after the job was claimed, the stack fetched from object
storage and the model loaded onto a GPU.

The arithmetic is the adapter's own: a model that wants 26 frames at 7.91 um
covers 26 * 7.91 / 9.362 = 22 slices of a 9.362 um render.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import depth_centers_that_fit, depth_window  # noqa: E402

GP = {"frames": 26, "training_slice_um": 7.91}


def test_the_profiles_own_centres_do_not_fit_a_33_slice_render():
    fits = depth_centers_that_fit(33, GP["frames"], [25, 32, 39],
                                  source_slice_um=9.362,
                                  training_slice_um=GP["training_slice_um"])
    assert fits == []


def test_the_middle_of_that_render_does():
    fits = depth_centers_that_fit(33, GP["frames"], [16],
                                  source_slice_um=9.362,
                                  training_slice_um=GP["training_slice_um"])
    assert fits == [16.0]


def test_the_same_centres_fit_the_62_layer_stack_they_were_written_for():
    fits = depth_centers_that_fit(62, GP["frames"], [25, 32, 39],
                                  source_slice_um=9.362,
                                  training_slice_um=GP["training_slice_um"])
    assert fits == [25.0, 32.0, 39.0]


def test_the_window_is_measured_in_source_slices():
    """The two pitches are what make this non-obvious: 26 frames is 26 slices
    only when the render is at the training pitch."""
    low, high = depth_window(16, 26, source_slice_um=9.362, training_slice_um=7.91)
    assert round(high - low, 1) == 21.1
    same = depth_window(16, 26, source_slice_um=7.91, training_slice_um=7.91)
    assert round(same[1] - same[0], 1) == 25.0


def test_a_stack_too_shallow_for_any_centre_fits_nowhere():
    """Rendering ten slices and asking for a 22-slice window is not a centring
    problem, and centring it silently would hand the model padding."""
    assert depth_centers_that_fit(10, 26, [5, 4, 6], source_slice_um=9.362,
                                  training_slice_um=7.91) == []


def test_a_render_at_the_training_pitch_needs_the_full_frame_count():
    """The canonical 2 um lane runs native -- 62 frames at 2.399 um on a 2.399 um
    render -- so its window is exactly as deep as the stack and centres on it."""
    assert depth_centers_that_fit(62, 62, [30.5], source_slice_um=2.399,
                                  training_slice_um=2.399) == [30.5]
    assert depth_centers_that_fit(61, 62, [30.0], source_slice_um=2.399,
                                  training_slice_um=2.399) == []
