"""FIX-05 — the review render must not erase the top decile of the signal.

The delivered renders used a hardcoded display floor of 0.20 while the p90 of
the mean map was 0.192-0.279, so everything at or below the 90th percentile
collapsed to byte 0: a real candidate sitting at p89 was indistinguishable from
background.  These tests pin the property that actually failed -- monotonic
discriminability of the top decile -- rather than the amount of black in the
frame, which a median floor necessarily leaves at about half.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))

from analyze_ink_stability import (  # noqa: E402
    DISPLAY_PROFILE_DATA_DERIVED,
    LEGACY_DISPLAY_LOWER,
    LEGACY_DISPLAY_UPPER,
    assert_top_decile_visible,
    build_display_policy,
    contrast_image,
    heat_overlay,
    layer_bounds,
    probability_display_bounds,
    probability_image,
    render_probability_layer,
)

MINIMUM_GREY_SEPARATION_P90_P99 = 32


# Percentiles of the PHerc1447 mean map as recorded by its own
# INK_STABILITY_ANALYSIS.json.  Reproducing the real shape matters: a synthetic
# with a brighter tail would let the legacy render look acceptable.
DELIVERED_QUANTILES = (0.0, 0.50, 0.90, 0.99, 0.995, 1.0)
DELIVERED_VALUES = (0.02, 0.1486409, 0.1917724, 0.3465169, 0.4076741, 0.87)


def synthetic_map_with_p90_of_0_2(seed: int = 20260724) -> np.ndarray:
    """Build a map with the delivered distribution shape (p90 just under 0.2)."""

    rng = np.random.default_rng(seed)
    uniform = rng.uniform(0.0, 1.0, size=256 * 256)
    values = np.interp(uniform, DELIVERED_QUANTILES, DELIVERED_VALUES)
    return values.reshape(256, 256).astype(np.float32)


def grey_at(array: np.ndarray, valid: np.ndarray, image: np.ndarray, percentile: float) -> float:
    """Read the rendered grey level at a percentile of the underlying data."""

    target = float(np.percentile(array[valid], percentile))
    index = int(np.argmin(np.abs(array[valid] - target)))
    return float(image[valid][index])


@pytest.fixture()
def synthetic() -> tuple[np.ndarray, np.ndarray]:
    array = synthetic_map_with_p90_of_0_2()
    valid = np.ones_like(array, dtype=bool)
    return array, valid


def test_synthetic_map_really_has_p90_at_0_2(synthetic) -> None:
    array, valid = synthetic
    assert float(np.percentile(array[valid], 90)) == pytest.approx(0.20, abs=0.01)


def test_new_render_is_not_mostly_black_and_the_old_one_is(synthetic) -> None:
    array, valid = synthetic
    bounds = layer_bounds(array, valid)

    new_image = render_probability_layer(array, valid, bounds)
    old_image = probability_image(
        array, valid, lower=LEGACY_DISPLAY_LOWER, upper=LEGACY_DISPLAY_UPPER
    )

    old_zero = float((old_image == 0).mean())
    new_zero = float((new_image == 0).mean())

    # The old render is overwhelmingly black: the floor sits at the p90.
    assert old_zero > 0.85
    # The new render is not mostly black.  A median floor puts about half the
    # valid pixels at zero by construction, so this is a bound, not a gate.
    assert new_zero <= 0.55
    assert new_zero < old_zero - 0.30


def test_top_decile_stays_discriminable_after_the_fix(synthetic) -> None:
    """PRIMARY gate: grey(p99) - grey(p90) >= 32 of 255."""

    array, valid = synthetic
    bounds = layer_bounds(array, valid)
    new_image = render_probability_layer(array, valid, bounds)

    separation = grey_at(array, valid, new_image, 99) - grey_at(
        array, valid, new_image, 90
    )
    assert separation >= MINIMUM_GREY_SEPARATION_P90_P99
    assert bounds["grey_separation_p90_p99"] >= MINIMUM_GREY_SEPARATION_P90_P99


def test_legacy_render_flattens_everything_up_to_p90_into_background(
    synthetic,
) -> None:
    """The regression itself: a candidate at p90 was byte-identical to nothing.

    Note the p90->p99 separation alone does not expose this -- inside the top
    decile the legacy ramp still separates.  What it destroyed is the boundary:
    nine tenths of the map, including candidate pixels just under p90, rendered
    as exactly 0.  That is what `display_lower < p90` now forbids.
    """

    array, valid = synthetic
    old_image = probability_image(
        array, valid, lower=LEGACY_DISPLAY_LOWER, upper=LEGACY_DISPLAY_UPPER
    )
    new_image = render_probability_layer(array, valid, layer_bounds(array, valid))

    assert grey_at(array, valid, old_image, 90) == 0
    assert grey_at(array, valid, old_image, 75) == 0
    assert grey_at(array, valid, old_image, 50) == 0

    assert grey_at(array, valid, new_image, 90) > 0
    assert grey_at(array, valid, new_image, 75) > 0


def test_bounds_are_data_derived_p50_p995(synthetic) -> None:
    array, valid = synthetic
    lower, upper = probability_display_bounds(array[valid])

    # probability_display_bounds promotes to float64 before the percentile, so
    # compare against the same promotion rather than the float32 result.
    reference = array[valid].astype(np.float64)
    assert lower == pytest.approx(float(np.percentile(reference, 50.0)))
    assert upper == pytest.approx(float(np.percentile(reference, 99.5)))


def test_invariant_floor_below_p90_and_ceiling_at_or_above_p99(synthetic) -> None:
    array, valid = synthetic
    bounds = layer_bounds(array, valid)

    p90 = float(np.percentile(array[valid], 90))
    p99 = float(np.percentile(array[valid], 99))
    assert bounds["display_lower"] < p90
    assert bounds["display_upper"] >= p99
    assert bounds["display_profile"] == DISPLAY_PROFILE_DATA_DERIVED


def test_legacy_floor_is_rejected_by_the_invariant(synthetic) -> None:
    """The exact regression: a floor at or above p90 must fail closed."""

    array, valid = synthetic
    with pytest.raises(RuntimeError, match="not below p90"):
        assert_top_decile_visible(
            array[valid], LEGACY_DISPLAY_LOWER, LEGACY_DISPLAY_UPPER
        )


def test_ceiling_below_p99_is_rejected(synthetic) -> None:
    array, valid = synthetic
    lower = float(np.percentile(array[valid], 10))
    with pytest.raises(RuntimeError, match="below p99"):
        assert_top_decile_visible(array[valid], lower, lower + 1e-3)


def test_bounds_are_stamped_for_every_rendered_layer(synthetic) -> None:
    array, valid = synthetic
    policy = build_display_policy(
        mean_map=array,
        robust_map=array * 0.9,
        depth_maps=[array, array * 0.95],
        valid=valid,
    )

    for name in (
        "mean_probability",
        "robust_minimum",
        "replica_disagreement",
        "depth_probability",
    ):
        record = policy[name]
        assert "display_lower" in record and "display_upper" in record
        assert record["display_upper"] > record["display_lower"]
        assert "display_profile" in record


def test_persistent_overlay_now_differs_materially_from_bare_ct(synthetic) -> None:
    """The overlay used to differ from the naked CT in under 4 % of pixels."""

    array, valid = synthetic
    rng = np.random.default_rng(7)
    ct = rng.integers(30, 220, size=array.shape, dtype=np.uint8)
    bare = np.repeat(contrast_image(ct)[..., None], 3, axis=2).astype(np.int16)

    new_overlay = np.asarray(
        heat_overlay(ct, array, valid, layer_bounds(array, valid))
    ).astype(np.int16)
    old_overlay = np.asarray(
        heat_overlay(
            ct,
            array,
            valid,
            {
                "display_lower": LEGACY_DISPLAY_LOWER,
                "display_upper": LEGACY_DISPLAY_UPPER,
            },
        )
    ).astype(np.int16)

    def changed_fraction(overlay: np.ndarray) -> float:
        return float((np.abs(overlay - bare).max(axis=2) > 8).mean())

    assert changed_fraction(new_overlay) > 0.10
    assert changed_fraction(new_overlay) > changed_fraction(old_overlay)


def test_legacy_signature_still_reproduces_historical_renders(synthetic) -> None:
    """The frozen transfer function stays reachable for reproduction."""

    array, valid = synthetic
    reproduced = probability_image(array, valid)
    explicit = probability_image(array, valid, lower=0.20, upper=0.70)

    assert np.array_equal(reproduced, explicit)


def test_crops_share_the_full_frame_scale(synthetic) -> None:
    """A crop must not be rescaled against itself, or panels stop comparing."""

    array, valid = synthetic
    bounds = layer_bounds(array, valid)
    full = render_probability_layer(array, valid, bounds)

    crop = array[32:96, 32:96]
    crop_valid = valid[32:96, 32:96]
    crop_image = render_probability_layer(crop, crop_valid, bounds)

    assert np.array_equal(crop_image, full[32:96, 32:96])


def test_empty_valid_mask_does_not_raise(synthetic) -> None:
    array, _ = synthetic
    empty = np.zeros_like(array, dtype=bool)

    lower, upper = probability_display_bounds(array[empty])
    assert upper > lower
