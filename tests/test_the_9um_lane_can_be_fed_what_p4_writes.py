"""P4 writes numbered TIFFs; the 9 um lane reads an OME-Zarr. This is the join.

Registering ink-9um-hybrid-3d2d@1.0.0 made a lane that nothing could route:
its runner streams a surface volume from Zarr at ~9 um isotropic, and every
P4 output in this platform is a numbered uint8 TIFF stack at 2.399 um. The
lane was declared unroutable with exactly that reason.

The model card gives the conversion and this implements it, rather than
inventing one:

    For 2.4 um surface volumes, first pool them to the ~9.6 um isotropic
    representation the models were trained on (XY pyramid level 2, 4x z
    mean-pooling)

XY pyramid level 2 is a factor of four, 4x in z is a factor of four, and
2.399 * 4 = 9.596 um -- which is the isotropy the models want and the reason
a native 9.362 um render needs none of this.

A stack that is already near the model scale passes through untouched, and
one that is neither 2.4 nor ~9.6 is refused -- unless a caller explicitly asks
for `resample_from_um`, an opt-in XY-only resample to the model's scale that
this file also covers below. Nothing here invents that resample on its own;
absent the parameter, the refusal is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

from prepare_9um_isotropic_input import (  # noqa: E402
    MODEL_SCALE_UM,
    IncompatibleSourceScale,
    plan_pooling,
    plan_resample,
    prepare,
)


def _stack(directory: Path, *, slices: int, size: int, seed: int = 0) -> Path:
    """A numbered uint8 TIFF stack, the shape P4 writes."""
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(slices):
        plane = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
        tifffile.imwrite(directory / f"{index:02d}.tif", plane)
    return directory


# -- the plan --------------------------------------------------------------

def test_a_2_4_um_stack_is_pooled_by_four_in_every_axis():
    """The model card's own recipe: XY pyramid level 2, 4x z mean-pooling."""
    plan = plan_pooling(2.399)
    assert plan.xy_factor == 4
    assert plan.z_factor == 4
    assert 9.5 <= plan.output_voxel_um <= 9.7


def test_a_native_9_um_stack_is_passed_through_untouched():
    """A native 9.362 um render is already at the model's scale. Resampling it
    to 9.6 would create no new spatial information."""
    plan = plan_pooling(9.362)
    assert plan.xy_factor == 1
    assert plan.z_factor == 1
    assert plan.output_voxel_um == pytest.approx(9.362)


def test_a_scale_the_recipe_does_not_cover_is_refused():
    """7.9 um is neither the 2.4 the pooling recipe is written for nor near
    the model scale. Inventing a factor for it would be inventing a recipe."""
    with pytest.raises(IncompatibleSourceScale, match="7.9"):
        plan_pooling(7.9)


def test_the_model_scale_is_the_one_the_card_states():
    assert 9.5 <= MODEL_SCALE_UM <= 9.7


# -- the conversion --------------------------------------------------------

def test_the_output_is_a_readable_zarr_at_the_model_scale(tmp_path):
    zarr = pytest.importorskip("zarr")
    source = _stack(tmp_path / "layers", slices=16, size=32)

    receipt = prepare(source, tmp_path / "out.zarr", source_voxel_um=2.399)

    group = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    array = group["0"]
    # 16 slices pooled 4x -> 4; 32 px pooled 4x -> 8.
    assert array.shape == (4, 8, 8)
    assert array.dtype == np.uint8
    assert receipt["output_voxel_um"] == pytest.approx(9.596, abs=0.01)


def test_pooling_is_a_mean_and_not_a_subsample(tmp_path):
    """Subsampling keeps one voxel in sixty-four and throws the rest away;
    the models were trained on the mean."""
    import tifffile

    source = tmp_path / "layers"
    source.mkdir()
    for index in range(4):
        plane = np.full((4, 4), 0 if index < 2 else 200, dtype=np.uint8)
        tifffile.imwrite(source / f"{index:02d}.tif", plane)

    prepare(source, tmp_path / "out.zarr", source_voxel_um=2.399)

    zarr = pytest.importorskip("zarr")
    value = np.asarray(zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["0"])
    assert value.shape == (1, 1, 1)
    assert value[0, 0, 0] == 100, "a mean of 0 and 200 is 100, not 0 or 200"


def test_the_receipt_binds_the_input_it_pooled(tmp_path):
    """A surface volume with no lineage is a stack somebody has to trust."""
    source = _stack(tmp_path / "layers", slices=8, size=16)

    receipt = prepare(source, tmp_path / "out.zarr", source_voxel_um=2.399)

    assert receipt["schema"] == "campaignx.ink_9um_isotropic_input_receipt.v1"
    assert receipt["source_voxel_um"] == 2.399
    assert receipt["xy_factor"] == 4 and receipt["z_factor"] == 4
    assert receipt["source_slices"] == 8
    assert len(receipt["source_sha256"]) == 64
    written = json.loads(
        (tmp_path / "out.zarr" / "INK_9UM_INPUT_RECEIPT.json").read_text())
    assert written["source_sha256"] == receipt["source_sha256"]


def test_a_stack_too_thin_to_pool_is_refused(tmp_path):
    """Three slices pooled 4x in z is zero slices, and a zero-depth volume is
    not something to hand a model."""
    source = _stack(tmp_path / "layers", slices=3, size=16)
    with pytest.raises(IncompatibleSourceScale, match="slices"):
        prepare(source, tmp_path / "out.zarr", source_voxel_um=2.399)


def test_it_refuses_to_overwrite(tmp_path):
    source = _stack(tmp_path / "layers", slices=8, size=16)
    (tmp_path / "out.zarr").mkdir()
    with pytest.raises(RuntimeError, match="refusing"):
        prepare(source, tmp_path / "out.zarr", source_voxel_um=2.399)


def test_it_writes_through_whichever_zarr_api_is_installed(tmp_path, monkeypatch):
    """Found by running this on the image that will actually run it.

    The 9 um image takes zarr from ink-detection's frozen lock -- 2.18.7 --
    and `create_array` is a zarr 3 name. This module's tests run against the
    repo's zarr 3.x, so a v3-only call passes here and raises
    AttributeError on the only host that matters.
    """
    import prepare_9um_isotropic_input as module

    source = _stack(tmp_path / "layers", slices=8, size=16)
    real_open = module.__dict__.get("zarr")

    seen = {}

    class ZarrTwoGroup:
        """A zarr 2 group: create_dataset, and no create_array at all."""
        def __init__(self):
            self.arrays, self.attrs = {}, {}
        def create_dataset(self, name, *, shape, dtype, chunks):
            seen["api"] = "create_dataset"
            self.arrays[name] = np.zeros(shape, dtype=dtype)
        def __getitem__(self, name):
            return self.arrays[name]

    group = ZarrTwoGroup()
    monkeypatch.setitem(
        sys.modules, "zarr",
        type("ZarrTwo", (), {"open_group": staticmethod(lambda *a, **k: group)}))

    module.prepare(source, tmp_path / "out.zarr", source_voxel_um=2.399)

    assert seen.get("api") == "create_dataset", (
        "a zarr 2 group has no create_array and this must not reach for one")


# -- resample_from_um, opt-in -----------------------------------------------
#
# 4 of the 13 eligible scrolls (0268, 0800, 1218, 1447) were scanned at 8.64
# um/116 keV: neither the 2.4 um the pooling recipe is written for nor within
# tolerance of the model's own ~9.362-9.6 um, so plan_pooling refuses them
# outright. resample_from_um is a caller's explicit declaration -- "this
# stack is actually at this scale, resample it to the target" -- reached only
# when a job names it; absent, the refusal above is unchanged.


def test_plan_resample_computes_the_xy_zoom_from_source_over_target():
    """Fewer, larger voxels covering the same physical span: the array
    shrinks by the same fraction the voxel grows by."""
    plan = plan_resample(8.640, 9.362)
    assert plan.xy_zoom == pytest.approx(8.640 / 9.362)
    assert plan.output_voxel_um == 9.362


def test_plan_resample_refuses_a_non_positive_source():
    with pytest.raises(IncompatibleSourceScale, match="resample_from_um"):
        plan_resample(0, 9.362)


def test_plan_resample_refuses_a_non_positive_target():
    with pytest.raises(IncompatibleSourceScale, match="target"):
        plan_resample(8.640, 0)


def test_without_resample_from_um_the_refusal_is_unchanged(tmp_path):
    """Point 2 of the spec this implements: absent the parameter, prepare
    refuses 8.64 um exactly as it always has."""
    source = _stack(tmp_path / "layers", slices=8, size=16)
    with pytest.raises(IncompatibleSourceScale):
        prepare(source, tmp_path / "out.zarr", source_voxel_um=8.640)


def test_prepare_resamples_when_asked_instead_of_refusing(tmp_path):
    zarr = pytest.importorskip("zarr")
    source = _stack(tmp_path / "layers", slices=8, size=32)

    receipt = prepare(source, tmp_path / "out.zarr", source_voxel_um=9.362,
                      resample_from_um=8.640)

    group = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    array = group["0"]
    assert array.shape[0] == 8, "resampling is XY-only; z is left alone"
    assert array.shape[1] < 32 and array.shape[2] < 32, (
        "8.64 -> 9.362 um is fewer, larger voxels; the plane must shrink")
    assert receipt["output_voxel_um"] == pytest.approx(9.362)


def test_the_resample_receipt_carries_the_method_not_a_pooling_factor(tmp_path):
    """Point 3 of the spec: the receipt records the applied factor and the
    interpolation method -- and, since this path never pooled anything,
    carries no xy_factor/z_factor to be mistaken for one that did."""
    source = _stack(tmp_path / "layers", slices=8, size=16)

    receipt = prepare(source, tmp_path / "out.zarr", source_voxel_um=9.362,
                      resample_from_um=8.640)

    assert receipt["resample_from_um"] == 8.640
    assert receipt["xy_zoom_factor"] == pytest.approx(8.640 / 9.362)
    assert receipt["interpolation"] == (
        "linear, no pre-filter (scipy.ndimage.zoom order=1)")
    assert "xy_factor" not in receipt and "z_factor" not in receipt
    assert receipt["isotropic"] is False, (
        "this path only ever touches xy and must not claim z isotropy")
    assert any("4%" in claim for claim in receipt["non_claims"]), (
        "the measured cost belongs in the receipt, not just the docstring"
    )
