#!/usr/bin/env python3
"""How far two runs of the same fit land from each other.

The spiral fit publishes no uncertainty and the paper describing it reports no
run-to-run variability, so a fitted surface is a number with no error bar. The
only one available is to run the fit twice changing nothing but
`random_seed` -- which is one of upstream's own config keys, so it costs an
override rather than a source rewrite -- and measure how far the two surfaces
separate.

Four decisions about how to report it, each of which was got wrong once.

**Averaged, not summed.** The two Chamfer conventions differ by a factor of two.
The campaign carried the inflated number for half its life: 258 um reported
where the averaged value was 121.

**Decomposed.** Sliding 75 um *along* a sheet does not take you off it; moving
35 um *through* it does. A single total says "121 um against a 35 um sheet" and
reads as failure, when the component that answers the campaign's question --
at what depth is the ink sampled -- is 12.

Both components are reported, and neither is optional. The normal is the
headline because it decides whether a render samples the right lamina. The
lateral has to stay visible because it is the error bar on a different claim
entirely: stitching segments in P8 and asserting in P9 that these letters sit
at this position on the sheet. Seventy-five microns of seed-dependent lateral
slide is harmless for sampling ink and central for reconstructing a page.

**Normalized by lamina thickness, not by the pitch between windings.** The
campaign divided by the 371 um winding pitch and reported "0.33 laminae" when in
depth it was a third of one *sheet*. Those are different statements about
different failures.

**Per z band.** On PHerc0826 the agreement runs from 94 to 157 um by winding,
and a single mean hides which turns are worth trusting.

Non-claims
----------
* This measures **reproducibility, not correctness**. Two runs can converge
  beautifully on a wrong surface when the failure comes from the data rather
  than the optimization -- measured on this corpus: in the rows-160-250 band of
  PHerc0826 w015, the band with 830 fold-back intersections and real
  self-contact, the two seeds agree *better* than in either neighbouring band
  (90.9 um against 93.1 and 95.5). A low number there is not evidence of a good
  surface. This belongs beside the geometry and lamina verdicts where they can
  contradict it, never collapsed into them.
* The +-51 um from the campaign's depth-tolerance probe is **not a threshold**.
  It is the offset at which the ink score fell 43%: a degradation curve, not a
  cut. Nothing here turns it into a pass mark.
* The centre is not the distribution. Everything this campaign measured had an
  excellent median and a poor tail -- interface localisation is 1.4 um at the
  median and 128 um at p90 -- so the normal component is reported with its p90
  beside its median, and a reader who takes only the median has been warned.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA = "campaignx.seed_agreement.v1"

MEASURED = "SEED_AGREEMENT_MEASURED"
UNPAIRED = "SEED_UNPAIRED"
UNMEASURED = "SEED_AGREEMENT_UNMEASURED"
# The state that exists because this is the one metric whose failure disguises
# itself as its best possible result. See `_refuse_identical`.
NOT_A_PAIR = "SEED_OVERRIDE_DID_NOT_TAKE"

# Below this the pair is treated as suspect rather than excellent. Two
# independent stochastic optimizations do not land a tenth of a voxel apart in
# the median; the campaign's own pairs sit at 9-17 voxels.
SUSPICIOUSLY_CLOSE_VOXELS = 0.1

# The campaign's own sheet reference, from the lamina gate's calibration. Used
# when a surface carries no measured thickness of its own.
SHEET_REFERENCE_UM = 35.5
# Reported beside the normal component, never as a gate. See the non-claims.
INK_DEPTH_TOLERANCE_UM = 51.0


class AgreementUnmeasurable(RuntimeError):
    """The pair cannot be compared, said rather than answered with a number."""


def load_grid(directory: Path):
    """A TIFXYZ as its coordinate grid and a validity mask.

    The same finite-and-non-negative policy the finalizer and the geometry gate
    already use, so three measurements of one surface cannot disagree about
    which of its cells exist.
    """
    import numpy as np
    import tifffile

    directory = Path(directory)
    missing = [name for name in ("x.tif", "y.tif", "z.tif")
               if not (directory / name).is_file()]
    if missing:
        raise AgreementUnmeasurable(f"{directory} is missing {missing}")
    planes = [np.asarray(tifffile.imread(directory / f"{axis}.tif"), dtype=np.float64)
              for axis in "xyz"]
    if any(plane.shape != planes[0].shape for plane in planes[1:]):
        raise AgreementUnmeasurable(
            f"TIFXYZ shapes differ: {[plane.shape for plane in planes]}")
    valid = np.logical_and.reduce([np.isfinite(p) & (p >= 0.0) for p in planes])
    return np.stack(planes, axis=-1), valid


def cell_normals(points, valid):
    """A unit normal per grid cell, from central differences.

    A cell on the border or beside a hole has none and is not decomposed: a
    displacement split against a made-up direction is two numbers that mean
    nothing rather than one that means something.
    """
    import numpy as np

    du = np.full_like(points, np.nan)
    dv = np.full_like(points, np.nan)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dv[1:-1, :] = points[2:, :] - points[:-2, :]
    usable = np.zeros(valid.shape, dtype=bool)
    usable[1:-1, 1:-1] = (valid[1:-1, 2:] & valid[1:-1, :-2]
                          & valid[2:, 1:-1] & valid[:-2, 1:-1]
                          & valid[1:-1, 1:-1])
    normal = np.cross(du, dv)
    length = np.linalg.norm(normal, axis=-1)
    usable &= np.isfinite(length) & (length > 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        normal = normal / length[..., None]
    return normal, usable


def _identical(points_a, valid_a, points_b, valid_b) -> bool:
    """Whether the two surfaces are the same bytes.

    Definitive rather than thresholded: two runs of a stochastic optimization do
    not agree exactly, so exact agreement is not a very good result, it is
    evidence that only one run happened. The analogue in the campaign's own
    script is refusing when both templates resolve to one directory -- that was
    the easy way to obtain a very convincing zero.
    """
    import numpy as np

    if points_a.shape != points_b.shape:
        return False
    return bool(np.array_equal(valid_a, valid_b)
                and np.array_equal(points_a[valid_a], points_b[valid_b]))


def _one_way(points_a, valid_a, normals_a, usable_a, cloud_b, sample: int, seed: int):
    """Nearest-neighbour displacement from A's cells to B's surface, split.

    Returns the total distance, and the normal and lateral magnitudes for the
    subset of cells that have a normal to split against.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    take = valid_a
    coordinates = points_a[take]
    if len(coordinates) < 16:
        raise AgreementUnmeasurable(
            f"only {len(coordinates)} valid cells to compare; a Chamfer over a "
            "handful of points is not a measurement")
    rng = np.random.default_rng(seed)
    if len(coordinates) > sample:
        chosen = rng.choice(len(coordinates), sample, replace=False)
    else:
        chosen = np.arange(len(coordinates))
    query = coordinates[chosen]
    distance, _ = cKDTree(cloud_b).query(query, workers=-1)

    # The same cells again, this time only where a normal exists.
    split = usable_a & valid_a
    split_points = points_a[split]
    split_normals = normals_a[split]
    if len(split_points) > sample:
        picked = rng.choice(len(split_points), sample, replace=False)
        split_points = split_points[picked]
        split_normals = split_normals[picked]
    if len(split_points) == 0:
        return distance, None, None, query
    _, index = cKDTree(cloud_b).query(split_points, workers=-1)
    displacement = cloud_b[index] - split_points
    along_normal = np.abs(np.einsum("ij,ij->i", displacement, split_normals))
    total = np.linalg.norm(displacement, axis=1)
    # Lateral is what is left of the displacement once the normal part is
    # removed, not the total: they are the two legs of a right triangle.
    lateral = np.sqrt(np.maximum(total ** 2 - along_normal ** 2, 0.0))
    return distance, along_normal, lateral, query


def measure(
    directory_a: Path,
    directory_b: Path,
    *,
    voxel_um: float,
    lamina_thickness_um: float | None = None,
    z_bands: int = 3,
    sample: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two fits of the same target that differ only in their seed."""
    import numpy as np

    directory_a, directory_b = Path(directory_a), Path(directory_b)
    if directory_a.resolve() == directory_b.resolve():
        raise AgreementUnmeasurable(
            "both directories are the same surface. A Chamfer of zero measured "
            "this way is very convincing and means nothing, which is why the "
            "pair is named rather than globbed.")
    if voxel_um <= 0:
        raise AgreementUnmeasurable(
            "a comparison in microns needs the volume's voxel size")

    points_a, valid_a = load_grid(directory_a)
    points_b, valid_b = load_grid(directory_b)
    identical = _identical(points_a, valid_a, points_b, valid_b)
    if identical:
        raise AgreementUnmeasurable(
            "these two surfaces are bit-identical, so the seed override did not "
            "take and this is not a pair. Every other measurement in this "
            "framework fails downward; this one fails upward -- an override that "
            "reached nothing produces a Chamfer of zero, which reads as perfect "
            "reproducibility. Check that the seed key exists in the pinned "
            "commit's default_config: it is `random_seed` at 05dcf034 and "
            "`optimizer_random_seed` later.")
    normals_a, usable_a = cell_normals(points_a, valid_a)
    normals_b, usable_b = cell_normals(points_b, valid_b)
    cloud_a, cloud_b = points_a[valid_a], points_b[valid_b]
    if len(cloud_a) < 16 or len(cloud_b) < 16:
        raise AgreementUnmeasurable("one of the two surfaces has almost no cells")

    forward, normal_f, lateral_f, query_f = _one_way(
        points_a, valid_a, normals_a, usable_a, cloud_b, sample, seed)
    backward, normal_b_, lateral_b_, query_b = _one_way(
        points_b, valid_b, normals_b, usable_b, cloud_a, sample, seed)

    # Averaged, not summed. The two conventions differ by exactly this factor,
    # and the campaign reported 258 um where the averaged value was 121.
    chamfer_voxels = 0.5 * (float(forward.mean()) + float(backward.mean()))

    def as_um(values):
        return np.asarray(values, dtype=float) * float(voxel_um)

    normals = np.concatenate([v for v in (normal_f, normal_b_) if v is not None]) \
        if any(v is not None for v in (normal_f, normal_b_)) else None
    laterals = np.concatenate([v for v in (lateral_f, lateral_b_) if v is not None]) \
        if any(v is not None for v in (lateral_f, lateral_b_)) else None

    if chamfer_voxels < SUSPICIOUSLY_CLOSE_VOXELS:
        raise AgreementUnmeasurable(
            f"the two surfaces agree to {chamfer_voxels:.4f} voxels, which is "
            "closer than two independent stochastic fits land -- the campaign's "
            "own pairs sit at 9 to 17. Read as a seed that did not reach the "
            "optimizer rather than as reproducibility: this is the one metric "
            "here whose failure looks like its best result, so it refuses "
            "instead of passing.")

    sheet = float(lamina_thickness_um or SHEET_REFERENCE_UM)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "state": MEASURED,
        "surfaces": [str(directory_a), str(directory_b)],
        "voxel_um": float(voxel_um),
        "lamina_thickness_um": sheet,
        "lamina_thickness_measured": lamina_thickness_um is not None,
        "cells_compared": int(len(query_f) + len(query_b)),
    }

    if normals is None or laterals is None:
        # A surface with no interior cell has no normal anywhere: the total is
        # all there is, and it is the one number that must not stand alone.
        result.update({
            "state": UNMEASURED,
            "reason": "no cell of either surface has a normal to decompose "
                      "against, so only the total exists and the total alone is "
                      "the misleading number this exists to avoid",
            "total_um": chamfer_voxels * float(voxel_um),
        })
        return result

    normal_um, lateral_um = as_um(normals), as_um(laterals)
    result.update({
        # The headline: it answers the question the campaign asks of geometry.
        "normal_um": {
            "median": float(np.median(normal_um)),
            "p90": float(np.percentile(normal_um, 90)),
            "p99": float(np.percentile(normal_um, 99)),
            # Named for what it is divided by, because the old number and this
            # one collide: the campaign reported "0.33 laminae" for the *total*
            # over the *winding pitch* (121 / 371 = 0.326), and this is the
            # *normal* over the *sheet thickness* (12 / 35.5 = 0.338). Both
            # round to 0.34, the numerator and the denominator and the meaning
            # are all different, and a reader comparing against an old report
            # would conclude nothing had changed.
            "normal_in_sheet_thicknesses": float(np.median(normal_um) / sheet),
        },
        # Visible always: this is the error bar on P8 and P9's claim about
        # where on the sheet a letter sits.
        "lateral_um": {
            "median": float(np.median(lateral_um)),
            "p90": float(np.percentile(lateral_um, 90)),
        },
        # Reported, never alone.
        "total_um": {
            "chamfer_voxels": chamfer_voxels,
            "chamfer_um": chamfer_voxels * float(voxel_um),
            "convention": "symmetric mean of both directions, averaged not "
                          "summed; the summed convention doubles it",
        },
        "by_z_band": _bands(points_a, valid_a, normals_a, usable_a, cloud_b,
                            voxel_um, z_bands, sample, seed, sheet),
        "ink_depth_tolerance_um": INK_DEPTH_TOLERANCE_UM,
        "normalisation": {
            "divided_by": "lamina thickness",
            "value_um": sheet,
            "not": "the winding pitch",
            "collision": (
                "0.33 laminae was reported through this campaign and is a "
                "different number: the total over the 371 um winding pitch, "
                "121 / 371 = 0.326. This is the normal component over the sheet "
                "thickness, 12 / 35.5 = 0.338. They round alike and share no "
                "term, so the field is named for its divisor and the bare word "
                "laminae is not used."),
        },
        "non_claims": [
            "this is reproducibility, not correctness: two runs converge on the "
            "same wrong surface when the failure is in the data rather than the "
            "optimization, measured on this corpus where the most tangled band "
            "of a patch had the best seed agreement in it",
            "the +-51 um depth tolerance is a degradation curve -- the offset at "
            "which the ink score fell 43% -- and not a threshold to pass",
            "the median is not the distribution; the p90 is reported beside it "
            "because everything measured in this campaign had a good median and "
            "a poor tail",
        ],
    })
    return result


def _bands(points_a, valid_a, normals_a, usable_a, cloud_b, voxel_um,
           count, sample, seed, sheet):
    """The same measurement per slab of z.

    A single mean over a whole winding hides which turns are worth trusting: on
    PHerc0826 the agreement ran from 94 to 157 um across one surface.
    """
    import numpy as np

    zs = points_a[..., 2]
    live = valid_a & usable_a
    if not live.any() or count < 1:
        return []
    low, high = float(zs[live].min()), float(zs[live].max())
    if not math.isfinite(low) or high <= low:
        return []
    edges = np.linspace(low, high, count + 1)
    out = []
    for index in range(count):
        lo, hi = edges[index], edges[index + 1]
        band = live & (zs >= lo) & (zs <= hi if index == count - 1 else zs < hi)
        if int(band.sum()) < 16:
            out.append({"z": [lo, hi], "cells": int(band.sum()),
                        "reason": "too few cells in this band to measure"})
            continue
        try:
            _, normal, lateral, _ = _one_way(
                points_a, band, normals_a, usable_a, cloud_b, sample, seed)
        except AgreementUnmeasurable as refusal:
            out.append({"z": [lo, hi], "cells": int(band.sum()),
                        "reason": str(refusal)})
            continue
        if normal is None:
            out.append({"z": [lo, hi], "cells": int(band.sum()),
                        "reason": "no normal in this band"})
            continue
        normal_um = np.asarray(normal) * float(voxel_um)
        lateral_um = np.asarray(lateral) * float(voxel_um)
        out.append({
            "z": [lo, hi], "cells": int(band.sum()),
            "normal_um": float(np.median(normal_um)),
            "normal_p90_um": float(np.percentile(normal_um, 90)),
            "lateral_um": float(np.median(lateral_um)),
            "normal_in_sheet_thicknesses": float(np.median(normal_um) / sheet),
        })
    return out


def unpaired(directory: Path, why: str = "only one seed was run") -> dict[str, Any]:
    """The verdict for a surface with no second run.

    A separate state rather than a missing field or a zero: a surface nobody
    measured twice has no error bar, and saying so is different from saying the
    error bar is small.
    """
    return {
        "schema": SCHEMA,
        "state": UNPAIRED,
        "surfaces": [str(directory)],
        "reason": why,
        "defensible": False,
        "why_not": (
            "a surface with one seed has no error bar, which is a different "
            "thing from having a large one. A large one can be defended with "
            "its number beside it; an absent one cannot be defended at all, so "
            "this is a state and not a missing field."),
        "non_claims": ["an unpaired surface is not a reproducible one; it is one "
                       "whose reproducibility was never asked"],
    }


def headline(result: dict[str, Any]) -> str:
    """One line, with the normal first and the total never on its own.

    The total is the actively misleading number: 121 um against a 35 um sheet
    reads as failure, and the component that answers the question is 12.
    """
    if result["state"] == UNPAIRED:
        return "SIN PAREJA -- one seed, so no error bar"
    if result["state"] != MEASURED:
        return f"{result['state']} -- {result.get('reason', '')}"
    normal, lateral = result["normal_um"], result["lateral_um"]
    return (
        f"normal {normal['median']:.0f} um (p90 {normal['p90']:.0f}, "
        f"{normal['normal_in_sheet_thicknesses']:.2f} sheet thicknesses) · "
        f"lateral {lateral['median']:.0f} um · "
        f"total {result['total_um']['chamfer_um']:.0f} um · "
        "reproducibility, not correctness"
    )


def cell(result: dict[str, Any]) -> str:
    """What a surfaces table shows: the state and one number.

    The four judgements beside this one are scannable, and a cell carrying four
    numbers per row stops the table being something you can sweep with your
    eyes. The decomposition, the percentiles, the normalisation and the
    non-claims live in the surface's own detail, where there is room to say what
    each was measured against.
    """
    if result["state"] == UNPAIRED:
        return "SIN PAREJA"
    if result["state"] != MEASURED:
        return result["state"]
    return f"{result['normal_um']['median']:.0f} um normal"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True,
                        help="one seed's surface; named, never globbed -- a glob "
                             "over a directory holding both can pick the same "
                             "one twice and answer zero very convincingly")
    parser.add_argument("--b", type=Path, help="the other seed's surface")
    parser.add_argument("--voxel-um", type=float, required=True)
    parser.add_argument("--lamina-thickness-um", type=float,
                        help="this surface's own measured sheet thickness; the "
                             f"campaign's {SHEET_REFERENCE_UM} um reference is "
                             "used when it is not given")
    parser.add_argument("--z-bands", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.b is None:
            result = unpaired(args.a)
        else:
            result = measure(args.a, args.b, voxel_um=args.voxel_um,
                             lamina_thickness_um=args.lamina_thickness_um,
                             z_bands=args.z_bands)
    except AgreementUnmeasurable as refusal:
        result = {"schema": SCHEMA, "state": UNMEASURED,
                  "surfaces": [str(args.a), str(args.b) if args.b else None],
                  "reason": str(refusal)}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(headline(result))
    if not args.output:
        print(payload)
    return 0 if result["state"] in (MEASURED, UNPAIRED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
