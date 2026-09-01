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

What this deliberately will not do is resample an arbitrary scale to the model's.
Upsampling creates no new spatial information; the registry already says exactly
that about routing 8.64/9.362 um targets to the 3D-DINO lane's model scale. A
source that is neither near 2.4 nor near the model scale is refused rather than
stretched.

Non-claims
----------
* Pooling is not a measurement. It changes the sampling of a stack that P4
  already produced and says nothing about whether that stack found the lamina.
* A surface volume this writes is an input to a screening lane, not evidence
  of ink.
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
        "model card does not give.")


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


def _stack_digest(slices: list[Path]) -> str:
    """One digest over the stack, in slice order, so the receipt names the
    bytes that were pooled rather than the directory they were in."""
    digest = hashlib.sha256()
    for path in slices:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare(source: Path, destination: Path, *, source_voxel_um: float,
            chunk: int = 128) -> dict[str, Any]:
    """Pool one P4 layer stack into an OME-Zarr the 9 um lane can stream."""
    import zarr  # noqa: PLC0415 - optional until this lane is used

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite: {destination}")

    plan = plan_pooling(source_voxel_um)
    slices = _ordered_slices(source)
    if len(slices) < plan.z_factor:
        raise IncompatibleSourceScale(
            f"{len(slices)} slices cannot be pooled {plan.z_factor}x in z; a "
            "zero-depth volume is not something to hand a model")

    volume = np.stack([tifffile.imread(path) for path in slices])
    pooled = _pool(volume, plan)

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
                {"type": "scale", "scale": [plan.output_voxel_um] * 3}],
        }],
    }]

    receipt = {
        "schema": "campaignx.ink_9um_isotropic_input_receipt.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "recipe": "ink_9um model card: XY pyramid level 2, 4x z mean-pooling",
        "source": str(source),
        "source_sha256": _stack_digest(slices),
        "source_slices": len(slices),
        "source_shape_zyx": [int(n) for n in volume.shape],
        "source_voxel_um": float(source_voxel_um),
        "xy_factor": plan.xy_factor,
        "z_factor": plan.z_factor,
        "output_shape_zyx": [int(n) for n in pooled.shape],
        "output_voxel_um": plan.output_voxel_um,
        "isotropic": True,
        "non_claims": [
            "pooling is not a measurement and says nothing about whether the "
            "stack it pooled found the lamina",
            "a surface volume is an input to a screening lane, not evidence of ink",
        ],
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
    args = ap.parse_args()
    try:
        receipt = prepare(args.layers, args.output,
                          source_voxel_um=args.source_voxel_um, chunk=args.chunk)
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
