#!/usr/bin/env python3
"""Pool a P4 layer stack into the ~9.6 um isotropic OME-Zarr the 9 um lane reads.

P4 writes a numbered uint8 TIFF stack. `ink-9um-hybrid-3d2d@1.0.0` streams a
surface volume from Zarr at ~9 um isotropic. That mismatch is the whole reason
the lane was registered unroutable, and this is the join.

The recipe is upstream's, from the ink_9um model card, not one invented here:

    For 2.4 um surface volumes, first pool them to the ~9.6 um isotropic
    representation the models were trained on (XY pyramid level 2, 4x z
    mean-pooling)

XY pyramid level 2 is a factor of four and 4x in z is a factor of four, so
2.399 um becomes 9.596 um in every axis -- which is why a native 9.362 um
render needs none of this and passes through untouched.

A source that is neither near 2.4 nor near the model scale is refused rather
than silently stretched -- unless a caller explicitly says otherwise. Four of
the thirteen eligible scrolls were scanned at 8.64 um/116 keV, which is
neither: `resample_from_um` is that caller's declaration, an opt-in XY-only
linear resample to the model's scale (see `plan_resample`), never invented on
this module's own initiative. Silence still means the refusal above; the
registry says the same thing about not inventing an upsample when routing to
the 3D-DINO lane's model scale.

Non-claims
----------
* Pooling is not a measurement. It changes the sampling of a stack that P4
  already produced and says nothing about whether that stack found the lamina.
* A surface volume this writes is an input to a screening lane, not evidence
  of ink.
* A resample is not free: linear XY interpolation from a caller-declared
  scale costs measured correlation against published letters (~4% at
  8.640->9.362 um) -- see `plan_resample`'s docstring for the figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# What the models were trained at: public 2.399 um volumes pooled to ~9.6 um
# isotropic, plus native 9.362 um segments. Both sit inside the tolerance below.
MODEL_SCALE_UM = 9.6
# Wide enough to hold 9.362 and 9.596 -- the two scales the training corpus
# actually contains -- and narrow enough that 7.9 is not quietly accepted.
MODEL_SCALE_TOLERANCE_UM = 0.5
# The scale the card's pooling recipe is written for.
POOLABLE_SOURCE_UM = 2.4
POOLABLE_TOLERANCE_UM = 0.15
POOL_FACTOR = 4

RECEIPT_NAME = "INK_9UM_INPUT_RECEIPT.json"


class IncompatibleSourceScale(RuntimeError):
    """The source is not something this recipe knows how to reach 9 um from."""


# What `main` exits with for the refusal above, so a caller running this as a
# subprocess can turn it back into that exception rather than into "exit 1".
REFUSED_SCALE_EXIT = 3


@dataclass(frozen=True)
class PoolingPlan:
    xy_factor: int
    z_factor: int
    output_voxel_um: float


@dataclass(frozen=True)
class ResamplePlan:
    xy_zoom: float
    output_voxel_um: float


def plan_pooling(source_voxel_um: float) -> PoolingPlan:
    """How to get from this scale to the model's, or a refusal."""
    scale = float(source_voxel_um)
    if abs(scale - MODEL_SCALE_UM) <= MODEL_SCALE_TOLERANCE_UM:
        # Already there. Resampling 9.362 to 9.6 would invent detail.
        return PoolingPlan(1, 1, scale)
    if abs(scale - POOLABLE_SOURCE_UM) <= POOLABLE_TOLERANCE_UM:
        return PoolingPlan(POOL_FACTOR, POOL_FACTOR, scale * POOL_FACTOR)
    raise IncompatibleSourceScale(
        f"{scale} um is neither within {POOLABLE_TOLERANCE_UM} of the "
        f"{POOLABLE_SOURCE_UM} um the pooling recipe is written for nor within "
        f"{MODEL_SCALE_TOLERANCE_UM} of the model's {MODEL_SCALE_UM} um. "
        "Reaching the model scale from here would mean inventing a factor the "
        "model card does not give. Pass --resample-from-um if this stack's "
        "actual scale is something else and you want it resampled there "
        "explicitly -- see plan_resample.")


def plan_resample(resample_from_um: float, target_voxel_um: float) -> ResamplePlan:
    """A caller-declared scale, resampled to the target -- not upstream's own
    recipe, and only reached when a job names both scales explicitly.

    Unlike plan_pooling, this is XY-only linear zoom, not integer-factor
    pooling: the two scales this exists for (8.640/9.362, the 116 keV scan
    family against this lane's native target) are not related by a whole
    number, so there is no way to express the step as "one voxel in N" the
    way the 2.4 um pooling recipe can. Z is left alone -- the slice count is
    a property of how P4 rendered the stack, not of this in-plane mismatch.

    Measured against the public control (PHerc0139 w043), round-tripped
    9.362 -> 8.640 -> 9.362 (worse than the real case, which is one pass, not
    two) against the community's published letters: correlation 0.359 native
    vs. 0.338 round-tripped, top-1% enrichment over random 19.89x vs. 19.13x
    -- a ~4% cost, against a shuffled-layer floor of 0.026 / 1.28x. Verified
    end to end on PHerc0800 z13472_w080, rendered at 8.64 and resampled to
    9.362: a valid ALIVE map where the unresampled input was refused outright.
    """
    source = float(resample_from_um)
    target = float(target_voxel_um)
    if not source > 0:
        raise IncompatibleSourceScale(
            f"resample_from_um must be positive, got {resample_from_um!r}")
    if not target > 0:
        raise IncompatibleSourceScale(
            f"the target voxel size must be positive, got {target_voxel_um!r}")
    # The array shrinks by the same fraction the physical voxel grows by:
    # fewer, larger voxels covering the same span.
    return ResamplePlan(xy_zoom=source / target, output_voxel_um=target)


def _ordered_slices(directory: Path) -> list[Path]:
    slices = sorted(
        (path for path in Path(directory).iterdir()
         if path.suffix.lower() in (".tif", ".tiff") and path.stem.isdigit()),
        key=lambda path: int(path.stem))
    if not slices:
        raise IncompatibleSourceScale(f"no numbered TIFF slices in {directory}")
    return slices


def _pool(volume: np.ndarray, plan: PoolingPlan) -> np.ndarray:
    """Mean-pool, which is what the models were trained on.

    Subsampling keeps one voxel in sixty-four and throws the rest away; the
    difference shows up as noise the model was never shown.
    """
    if plan.z_factor == 1 and plan.xy_factor == 1:
        return volume
    depth, height, width = volume.shape
    depth -= depth % plan.z_factor
    height -= height % plan.xy_factor
    width -= width % plan.xy_factor
    trimmed = volume[:depth, :height, :width]
    reshaped = trimmed.reshape(
        depth // plan.z_factor, plan.z_factor,
        height // plan.xy_factor, plan.xy_factor,
        width // plan.xy_factor, plan.xy_factor)
    # float32 for the mean, back to the source dtype for the model: the input
    # contract is a uint8 surface volume.
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32).round().astype(volume.dtype)


def _resample_xy(volume: np.ndarray, zoom: float) -> np.ndarray:
    """Linear interpolation in Y and X only, no pre-filter -- exactly the
    method the correlation-against-published-letters figure in
    plan_resample's docstring was measured with. scipy.ndimage.zoom's own
    Gaussian pre-filter is for downsampling by a large factor to fight
    aliasing; this platform's own measurement is the number that says
    whether skipping it is fine here, not a general argument either way."""
    from scipy import ndimage  # noqa: PLC0415 - optional until this path runs

    resized = ndimage.zoom(volume.astype(np.float32), (1.0, zoom, zoom),
                           order=1, prefilter=False)
    return resized.round().astype(volume.dtype)


def _stack_digest(slices: list[Path]) -> str:
    """One digest over the stack, in slice order, so the receipt names the
    bytes that were pooled rather than the directory they were in."""
    digest = hashlib.sha256()
    for path in slices:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare(source: Path, destination: Path, *, source_voxel_um: float,
            chunk: int = 128, resample_from_um: float | None = None) -> dict[str, Any]:
    """Pool -- or, if resample_from_um names one, resample -- one P4 layer
    stack into an OME-Zarr the 9 um lane can stream.

    resample_from_um is a caller's explicit declaration, not a measurement
    this reaches for on its own: without it, an out-of-tolerance scale is
    still refused exactly as before. See plan_resample for the method and
    what it costs.
    """
    import zarr  # noqa: PLC0415 - optional until this lane is used

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite: {destination}")

    resample_plan = (plan_resample(resample_from_um, source_voxel_um)
                     if resample_from_um is not None else None)
    plan = None if resample_plan is not None else plan_pooling(source_voxel_um)
    slices = _ordered_slices(source)
    if plan is not None and len(slices) < plan.z_factor:
        raise IncompatibleSourceScale(
            f"{len(slices)} slices cannot be pooled {plan.z_factor}x in z; a "
            "zero-depth volume is not something to hand a model")

    volume = np.stack([tifffile.imread(path) for path in slices])
    pooled = (_resample_xy(volume, resample_plan.xy_zoom) if resample_plan is not None
              else _pool(volume, plan))
    output_voxel_um = (resample_plan.output_voxel_um if resample_plan is not None
                       else plan.output_voxel_um)

    destination.mkdir(parents=True)
    group = zarr.open_group(str(destination), mode="w")
    chunks = tuple(min(chunk, size) for size in pooled.shape)
    # zarr 2 and zarr 3 name this differently, and both are in play: this
    # module's own tests run against whatever the repo has, while the image
    # that actually runs the lane takes zarr from ink-detection's frozen lock
    # -- 2.18.7 there against 3.x here. Writing for one of them is a module
    # that passes its tests and raises AttributeError on the only host that
    # matters.
    if hasattr(group, "create_array"):
        group.create_array("0", shape=pooled.shape, dtype=pooled.dtype,
                           chunks=chunks)
    else:
        group.create_dataset("0", shape=pooled.shape, dtype=pooled.dtype,
                             chunks=chunks)
    group["0"][:] = pooled
    # Enough OME metadata to say what the axes and the scale are. The lane
    # streams array "0"; this is so a reader is not left inferring microns.
    group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": name, "type": "space", "unit": "micrometer"}
                 for name in ("z", "y", "x")],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [output_voxel_um] * 3}],
        }],
    }]

    if resample_plan is not None:
        recipe = ("caller-declared resample: XY-only linear zoom from "
                  f"{float(resample_from_um)} um, no pre-filter -- see "
                  "prepare_9um_isotropic_input.plan_resample")
        method_fields: dict[str, Any] = {
            "resample_from_um": float(resample_from_um),
            "xy_zoom_factor": resample_plan.xy_zoom,
            "interpolation": "linear, no pre-filter (scipy.ndimage.zoom order=1)",
        }
    else:
        recipe = "ink_9um model card: XY pyramid level 2, 4x z mean-pooling"
        method_fields = {"xy_factor": plan.xy_factor, "z_factor": plan.z_factor}

    receipt = {
        "schema": "campaignx.ink_9um_isotropic_input_receipt.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "recipe": recipe,
        "source": str(source),
        "source_sha256": _stack_digest(slices),
        "source_slices": len(slices),
        "source_shape_zyx": [int(n) for n in volume.shape],
        "source_voxel_um": float(source_voxel_um),
        **method_fields,
        "output_shape_zyx": [int(n) for n in pooled.shape],
        "output_voxel_um": output_voxel_um,
        # The pooling path pools z along with xy, by construction isotropic.
        # The resample path only ever touches xy -- z is whatever the
        # source's own slice spacing already was, unclaimed here because
        # this module has no source_slice_um to check it against.
        "isotropic": resample_plan is None,
        "non_claims": [
            "pooling is not a measurement and says nothing about whether the "
            "stack it pooled found the lamina",
            "a surface volume is an input to a screening lane, not evidence of ink",
        ] + (["resampling from a caller-declared scale costs measured "
              "correlation against published letters (~4% at 8.640->9.362); "
              "see plan_resample's docstring for the figures, not asserted here"]
             if resample_plan is not None else []),
    }
    (destination / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=Path, required=True,
                    help="the numbered TIFF stack P4 wrote")
    ap.add_argument("--output", type=Path, required=True,
                    help="OME-Zarr to write")
    ap.add_argument("--source-voxel-um", type=float, required=True)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--resample-from-um", type=float, default=None,
                    help="the stack's actual scale, when it is neither near "
                         "2.4 nor near the model's own -- resamples XY "
                         "linearly to --source-voxel-um instead of refusing. "
                         "Only reached when named explicitly; see plan_resample.")
    args = ap.parse_args()
    try:
        receipt = prepare(args.layers, args.output,
                          source_voxel_um=args.source_voxel_um, chunk=args.chunk,
                          resample_from_um=args.resample_from_um)
    except IncompatibleSourceScale as refused:
        # Its own exit code, because the lane runs this as a subprocess and the
        # caller has to tell "this scale cannot be reached" -- a refusal with a
        # sentence somebody can act on -- from "the pooling broke".
        print(str(refused), file=sys.stderr)
        return REFUSED_SCALE_EXIT
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
