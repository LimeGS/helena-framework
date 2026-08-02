"""T4a + FIX-09 + FIX-10.4 — the ink preprocessing chain in physical units.

Three residual defects of ``framework/stages/03-ink/scripts/run_ink_timesformer.py``
are pinned here:

T4a (double uint8 quantization)
    ``interpolate_depth`` rounded every interpolated depth plane back onto 256
    levels, and ``resize_stack`` rounded the bilinear result onto 256 levels
    again.  At 9.362 um the resize is a 1.18x upscale toward the 7.91 um
    training grid, so both operators exist precisely to create intermediate
    grey values -- and both roundings destroyed them, on a signal whose useful
    contrast is a handful of levels.  The model never wanted uint8: ``infer_map``
    feeds it ``clamp(0,200)/255`` as float32.  The float32 chain is a different
    numeric result, so it ships as a new lane profile (@1.1.0) and leaves the
    frozen @1.0.0 path reproducible.

FIX-09 (hardcoded 7.91)
    No file in this lane may restate the training scale; it is declared by the
    ink lane profile and resolved through ``resolve_training_pixel_um``.

FIX-10.4 (physical invariance)
    Nothing tested ``interpolate_depth`` / ``resize_stack`` / the physical
    rescale.  A scale error there does not crash; it silently displaces every
    screening coordinate between an 8.64 um and a 9.362 um scan.  The same
    synthetic object rendered at both voxel sizes must land on the same
    *physical* peak, and declaring the wrong source scale must move it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "framework/stages/03-ink/scripts/run_ink_timesformer.py"
PROFILES = ROOT / "framework/profiles/03-ink"
FROZEN_PROFILE = PROFILES / "timesformer-gp-scroll1-screening-1.0.0.json"
FLOAT32_PROFILE = PROFILES / "timesformer-gp-scroll1-screening-1.1.0.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INK = _load("helena_ink_precision_runner", RUNNER)

TRAINING_PIXEL_UM = 7.91
TRAINING_SLICE_UM = 7.91
FRAMES = 26
VOXEL_SIZES = (8.64, 9.362)

# One synthetic object, described only in micrometres.
PEAK_DEPTH_UM = 340.0
PEAK_Y_UM = 430.0
PEAK_X_UM = 520.0
SIGMA_DEPTH_UM = 25.0
SIGMA_PLANE_UM = 45.0
DEPTH_EXTENT_UM = 620.0
PLANE_EXTENT_UM = 900.0


def render_volume(voxel_um: float) -> np.ndarray:
    """Render the same physical object onto a grid of the given voxel size.

    Native uint8, because ``load_tiff_stack`` refuses anything else; the point
    of T4a is the *re*-quantization the preprocessing chain adds on top.
    """

    depth = int(round(DEPTH_EXTENT_UM / voxel_um))
    side = int(round(PLANE_EXTENT_UM / voxel_um))
    z = (np.arange(depth) + 0.5) * voxel_um
    y = (np.arange(side) + 0.5) * voxel_um
    x = (np.arange(side) + 0.5) * voxel_um
    depth_profile = np.exp(-0.5 * ((z - PEAK_DEPTH_UM) / SIGMA_DEPTH_UM) ** 2)
    row_profile = np.exp(-0.5 * ((y - PEAK_Y_UM) / SIGMA_PLANE_UM) ** 2)
    column_profile = np.exp(-0.5 * ((x - PEAK_X_UM) / SIGMA_PLANE_UM) ** 2)
    volume = 255.0 * (
        depth_profile[:, None, None]
        * row_profile[None, :, None]
        * column_profile[None, None, :]
    )
    return np.clip(np.rint(volume), 0, 255).astype(np.uint8)


def normalize(
    volume: np.ndarray,
    *,
    declared_pixel_um: float,
    declared_slice_um: float,
    dtype,
) -> np.ndarray:
    """Run exactly the runner's preprocessing chain on one depth centre."""

    center = PEAK_DEPTH_UM / declared_slice_um - 0.5
    positions = INK.depth_positions(
        center,
        FRAMES,
        source_slice_um=declared_slice_um,
        training_slice_um=TRAINING_SLICE_UM,
    )
    frames = INK.interpolate_depth(volume, positions, dtype=dtype)
    target = round(volume.shape[1] * declared_pixel_um / TRAINING_PIXEL_UM)
    return INK.resize_stack(
        frames,
        target_height=target,
        target_width=target,
        dtype=dtype,
    )


def centroid(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    assert values.sum() > 0
    return float((values * (np.arange(values.size) + 0.5)).sum() / values.sum())


def reported_peak_um(
    volume: np.ndarray,
    stack: np.ndarray,
    *,
    declared_pixel_um: float,
) -> tuple[float, float, float]:
    """Where a downstream consumer would say the peak is, in micrometres.

    The depth axis is read off the training slice pitch the frames were sampled
    at; the plane axes are read off the output grid pitch implied by the
    *declared* source scale.  Both are what the receipt says the maps mean.
    """

    values = stack.astype(np.float64)
    plane = values.sum(axis=0)
    plane_pitch_um = declared_pixel_um * volume.shape[1] / stack.shape[1]
    depth_index = centroid(values.sum(axis=(1, 2))) - 0.5
    depth_um = PEAK_DEPTH_UM + (depth_index - (FRAMES - 1) / 2.0) * TRAINING_SLICE_UM
    return (
        depth_um,
        centroid(plane.sum(axis=1)) * plane_pitch_um,
        centroid(plane.sum(axis=0)) * plane_pitch_um,
    )


# --------------------------------------------------------------------------
# T4a -- how many distinct levels survive preprocessing
# --------------------------------------------------------------------------


def low_contrast_stack(seed: int = 11) -> np.ndarray:
    """A stack whose useful contrast is a handful of grey levels."""

    rng = np.random.default_rng(seed)
    volume = render_volume(9.362).astype(np.float64)
    faint = 120.0 + 9.0 * (volume / 255.0) + rng.normal(0.0, 0.8, volume.shape)
    return np.clip(np.rint(faint), 0, 255).astype(np.uint8)


def surviving_levels(dtype) -> tuple[int, int, int]:
    source = low_contrast_stack()
    positions = INK.depth_positions(
        PEAK_DEPTH_UM / 9.362 - 0.5,
        FRAMES,
        source_slice_um=9.362,
        training_slice_um=TRAINING_SLICE_UM,
    )
    frames = INK.interpolate_depth(source, positions, dtype=dtype)
    target = round(source.shape[1] * 9.362 / TRAINING_PIXEL_UM)
    resized = INK.resize_stack(
        frames, target_height=target, target_width=target, dtype=dtype
    )
    return (
        int(np.unique(source).size),
        int(np.unique(frames).size),
        int(np.unique(resized).size),
    )


def test_the_frozen_uint8_chain_collapses_onto_the_source_levels() -> None:
    source, after_depth, after_resize = surviving_levels(np.uint8)

    assert source < 32
    assert after_depth <= source
    assert after_resize <= source


def test_float32_preserves_the_levels_interpolation_creates() -> None:
    _, uint8_depth, uint8_resize = surviving_levels(np.uint8)
    _, float_depth, float_resize = surviving_levels(np.float32)

    assert float_depth > 50 * uint8_depth
    assert float_resize > 1000 * uint8_resize


def test_the_model_input_is_float_so_no_quantization_is_required() -> None:
    """``infer_map`` normalizes to float32; nothing downstream wants uint8."""

    source = RUNNER.read_text(encoding="utf-8")

    # The divisor is the clip value, per the upstream contract in
    # ink-detection/optimized_inference/inference.py. This assertion used to
    # pin ``.div_(255.0)``, which froze the defect in place as the expected
    # behaviour: the model saw 78.4% of its trained contrast range for 95
    # screening runs and no test objected.
    assert "tensor.clamp_(0, max_clip_value).div_(float(max_clip_value))" in source
    assert ".div_(255" not in source
    assert INK.PREPROCESS_UINT8_QUANTIZATION_STEPS == {"uint8": 2, "float32": 0}


def test_float32_resize_does_not_round_or_leave_the_physical_range() -> None:
    source = render_volume(9.362)
    positions = INK.depth_positions(
        PEAK_DEPTH_UM / 9.362 - 0.5,
        FRAMES,
        source_slice_um=9.362,
        training_slice_um=TRAINING_SLICE_UM,
    )
    frames = INK.interpolate_depth(source, positions, dtype=np.float32)
    target = round(source.shape[1] * 9.362 / TRAINING_PIXEL_UM)
    resized = INK.resize_stack(
        frames, target_height=target, target_width=target, dtype=np.float32
    )

    assert resized.dtype == np.float32
    assert resized.min() >= 0.0
    assert resized.max() <= 255.0
    assert not np.array_equal(resized, np.rint(resized))


def test_the_default_preprocessing_dtype_is_still_the_frozen_uint8_path() -> None:
    """Every existing importer keeps bit-for-bit behaviour."""

    source = render_volume(9.362)
    positions = INK.depth_positions(
        PEAK_DEPTH_UM / 9.362 - 0.5,
        FRAMES,
        source_slice_um=9.362,
        training_slice_um=TRAINING_SLICE_UM,
    )
    frames = INK.interpolate_depth(source, positions)
    target = round(source.shape[1] * 9.362 / TRAINING_PIXEL_UM)
    resized = INK.resize_stack(frames, target_height=target, target_width=target)

    assert frames.dtype == np.uint8
    assert resized.dtype == np.uint8
    assert np.array_equal(
        frames, INK.interpolate_depth(source, positions, dtype=np.uint8)
    )


def test_an_unsupported_preprocessing_dtype_fails_closed() -> None:
    source = render_volume(9.362)
    positions = INK.depth_positions(
        PEAK_DEPTH_UM / 9.362 - 0.5,
        FRAMES,
        source_slice_um=9.362,
        training_slice_um=TRAINING_SLICE_UM,
    )

    with pytest.raises(ValueError, match="unsupported preprocessing dtype"):
        INK.interpolate_depth(source, positions, dtype=np.uint16)


# --------------------------------------------------------------------------
# FIX-10.4 -- physical invariance across voxel sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_the_peak_lands_on_the_same_physical_point_at_8_64_and_9_362(dtype) -> None:
    """The whole point of the physical normalization, in micrometres."""

    reported = {}
    for voxel_um in VOXEL_SIZES:
        volume = render_volume(voxel_um)
        stack = normalize(
            volume,
            declared_pixel_um=voxel_um,
            declared_slice_um=voxel_um,
            dtype=dtype,
        )
        reported[voxel_um] = reported_peak_um(
            volume, stack, declared_pixel_um=voxel_um
        )

    truth = (PEAK_DEPTH_UM, PEAK_Y_UM, PEAK_X_UM)
    for voxel_um, measured in reported.items():
        for axis, (found, expected) in enumerate(zip(measured, truth)):
            assert abs(found - expected) < 1.0, (voxel_um, axis, found, expected)

    for axis, (a, b) in enumerate(zip(reported[8.64], reported[9.362])):
        assert abs(a - b) < 0.5, (axis, a, b)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_the_normalized_grids_share_one_physical_pitch(dtype) -> None:
    """Both voxel sizes must resample onto the same physical output pitch."""

    pitches = []
    for voxel_um in VOXEL_SIZES:
        volume = render_volume(voxel_um)
        stack = normalize(
            volume,
            declared_pixel_um=voxel_um,
            declared_slice_um=voxel_um,
            dtype=dtype,
        )
        pitches.append(voxel_um * volume.shape[1] / stack.shape[1])

    for pitch in pitches:
        assert abs(pitch - TRAINING_PIXEL_UM) < 0.05
    assert abs(pitches[0] - pitches[1]) < 0.05


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_declaring_9_362_on_an_8_64_volume_moves_the_peak(dtype) -> None:
    """The regression FIX-09 prevents: a wrong source scale must be detectable.

    Four eligible volumes (PHerc268, PHerc800, PHerc1218, PHerc1447) are 8.64 um
    scans, and the legacy CLI default was 9.362.  Silently accepting that default
    displaces the screening peak by tens of micrometres -- more than a whole
    letter stroke -- without any error.
    """

    volume = render_volume(8.64)
    honest = reported_peak_um(
        volume,
        normalize(
            volume, declared_pixel_um=8.64, declared_slice_um=8.64, dtype=dtype
        ),
        declared_pixel_um=8.64,
    )
    mislabelled = reported_peak_um(
        volume,
        normalize(
            volume, declared_pixel_um=9.362, declared_slice_um=9.362, dtype=dtype
        ),
        declared_pixel_um=9.362,
    )

    for axis, (a, b) in enumerate(zip(honest, mislabelled)):
        assert abs(a - b) > 15.0, (axis, a, b)


def test_depth_positions_are_spaced_by_the_training_pitch_in_micrometres() -> None:
    """The depth sampling is physical, not index based."""

    for voxel_um in VOXEL_SIZES:
        positions = INK.depth_positions(
            PEAK_DEPTH_UM / voxel_um - 0.5,
            FRAMES,
            source_slice_um=voxel_um,
            training_slice_um=TRAINING_SLICE_UM,
        )
        spacing_um = np.diff(positions) * voxel_um
        assert np.allclose(spacing_um, TRAINING_SLICE_UM)
        span_um = (positions[-1] - positions[0]) * voxel_um
        assert abs(span_um - (FRAMES - 1) * TRAINING_SLICE_UM) < 1e-9


# --------------------------------------------------------------------------
# FIX-09 -- the training scale is declared, never restated
# --------------------------------------------------------------------------

OWNED_SOURCES = (
    "framework/stages/03-ink/scripts/run_ink_timesformer.py",
    "framework/stages/01-segmentation/scripts/run_coverage_surface_v2.py",
    "framework/stages/06-discovery/scripts/rank_expanded_candidate_windows.py",
    "framework/stages/06-discovery/scripts/rank_expanded_candidate_windows_v2.py",
    "framework/stages/06-discovery/scripts/run_expanded_robust_windows.py",
    "framework/stages/06-discovery/scripts/run_exploratory_target_screen.py",
)


@pytest.mark.parametrize("relative", OWNED_SOURCES)
def test_no_ink_lane_source_restates_the_training_scale(relative: str) -> None:
    assert "7.91" not in (ROOT / relative).read_text(encoding="utf-8")


def test_the_runner_reuses_one_shared_scale_contract() -> None:
    """Imported, not re-implemented -- and from a dependency-free module.

    The contract first lived in the Stage 04 analysis module, so the runner
    imported it from there and inherited that module's scipy dependency. The
    project's pinned ink image carries only numpy, Pillow, timesformer-pytorch,
    einops and safetensors, so the runner stopped importing inside the very
    container built to run it. The contract now lives in framework/contracts,
    which needs nothing beyond the standard library.
    """

    from framework.contracts import physical_scale

    assert INK.resolve_training_pixel_um is physical_scale.resolve_training_pixel_um
    assert INK.PIXEL_UM_TOLERANCE is physical_scale.PIXEL_UM_TOLERANCE
    assert INK.DEFAULT_INK_PROFILE == physical_scale.DEFAULT_INK_PROFILE

    sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))
    import analyze_ink_stability as stability

    assert stability.PIXEL_UM_TOLERANCE == physical_scale.PIXEL_UM_TOLERANCE

    contract = (ROOT / "framework/contracts/physical_scale.py").read_text(
        encoding="utf-8"
    )
    for heavy in ("import scipy", "from scipy", "import numpy", "import torch"):
        assert heavy not in contract, (
            f"the shared scale contract must stay importable inside the pinned "
            f"ink image; it now pulls in {heavy!r}"
        )


def test_the_runner_cli_carries_no_scale_default() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '"--training-pixel-um", type=float, default=None' in source
    assert '"--training-slice-um", type=float, default=None' in source


def test_the_training_slice_resolves_from_the_profile() -> None:
    declared = json.loads(FROZEN_PROFILE.read_text(encoding="utf-8"))
    expected = float(declared["input_contract"]["training_slice_um"])

    assert (
        INK.resolve_training_slice_um(profile_path=FROZEN_PROFILE, requested=None)
        == expected
    )
    assert (
        INK.resolve_training_slice_um(
            profile_path=FROZEN_PROFILE,
            requested=expected + INK.PIXEL_UM_TOLERANCE / 2,
        )
        == expected
    )


def test_a_training_slice_disagreement_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="disagrees with the ink lane profile"):
        INK.resolve_training_slice_um(profile_path=FROZEN_PROFILE, requested=8.64)


def test_a_profile_without_a_training_slice_fails_closed(tmp_path: Path) -> None:
    bogus = tmp_path / "profile.json"
    bogus.write_text(
        json.dumps({"profile_id": "x@1.0.0", "input_contract": {}}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="declares no training_slice_um"):
        INK.resolve_training_slice_um(profile_path=bogus, requested=None)


# --------------------------------------------------------------------------
# T4a -- the numeric path is bound to a profile identity
# --------------------------------------------------------------------------


def test_the_frozen_profile_keeps_the_double_quantized_path() -> None:
    frozen = json.loads(FROZEN_PROFILE.read_text(encoding="utf-8"))

    assert frozen["profile_id"] == "timesformer-gp-scroll1-screening@1.0.0"
    assert "preprocess_precision" not in frozen["input_contract"]
    assert (
        INK.resolve_preprocess_precision(profile_path=FROZEN_PROFILE, requested=None)
        == "uint8"
    )


def test_the_new_profile_declares_the_single_quantization_path() -> None:
    new = json.loads(FLOAT32_PROFILE.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PROFILE.read_text(encoding="utf-8"))

    assert new["profile_id"] == "timesformer-gp-scroll1-screening@1.1.0"
    assert new["profile_id"] != frozen["profile_id"]
    # Same checkpoint, same method, same physical contract: only the numeric
    # path differs, which is exactly why it needs its own identity.
    assert new["method_id"] == frozen["method_id"]
    assert new["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    for key in ("frames", "tile_size_y_x", "training_pixel_um", "training_slice_um"):
        assert new["input_contract"][key] == frozen["input_contract"][key]
    assert new["input_contract"]["preprocess_precision"] == "float32"
    assert new["input_contract"]["preprocess_uint8_quantization_steps"] == 0
    assert (
        INK.resolve_preprocess_precision(profile_path=FLOAT32_PROFILE, requested=None)
        == "float32"
    )


def test_a_precision_assertion_that_contradicts_the_profile_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="disagrees with the ink lane profile"):
        INK.resolve_preprocess_precision(
            profile_path=FROZEN_PROFILE, requested="float32"
        )
    with pytest.raises(RuntimeError, match="disagrees with the ink lane profile"):
        INK.resolve_preprocess_precision(
            profile_path=FLOAT32_PROFILE, requested="uint8"
        )


def test_an_unknown_declared_precision_fails_closed(tmp_path: Path) -> None:
    bogus = tmp_path / "profile.json"
    bogus.write_text(
        json.dumps(
            {
                "profile_id": "x@1.0.0",
                "input_contract": {"preprocess_precision": "int4"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unknown preprocess_precision"):
        INK.resolve_preprocess_precision(profile_path=bogus, requested=None)


def test_the_two_profiles_do_not_produce_the_same_maps() -> None:
    """A new identity is required because the numbers genuinely differ."""

    source = low_contrast_stack()
    positions = INK.depth_positions(
        PEAK_DEPTH_UM / 9.362 - 0.5,
        FRAMES,
        source_slice_um=9.362,
        training_slice_um=TRAINING_SLICE_UM,
    )
    target = round(source.shape[1] * 9.362 / TRAINING_PIXEL_UM)
    frozen = INK.resize_stack(
        INK.interpolate_depth(source, positions, dtype=np.uint8),
        target_height=target,
        target_width=target,
        dtype=np.uint8,
    ).astype(np.float64)
    fresh = INK.resize_stack(
        INK.interpolate_depth(source, positions, dtype=np.float32),
        target_height=target,
        target_width=target,
        dtype=np.float32,
    ).astype(np.float64)

    assert not np.array_equal(frozen, fresh)
    # ... but only by rounding: the frozen path is the new path rounded.
    assert np.abs(frozen - fresh).max() <= 1.0
