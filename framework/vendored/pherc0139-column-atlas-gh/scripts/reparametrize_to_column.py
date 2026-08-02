#!/usr/bin/env python3
"""Re-parametrize an arbitrary mesh region into a 'column' image: one axis is
physical arc-length (circumference direction) at roughly fixed radius, the
other is axial position along the scroll -- the same convention the official
wNNN segments already have built in.

STATUS (2026-07-13): validated pieces + one confirmed, unsolved limitation.
Run `--self-test` to see the full finding. In short: isolating "one physical
wrap" via a narrow RADIUS THRESHOLD does not work, even on the official,
known-good w043 segment -- verified directly on its native mesh grid (no
remapping involved): a 2.0mm-wide band only densely fills 35% of its own
bounding box, a 1.0mm band 19%, a 0.6mm band 12.5%. Fill gets WORSE as the
band narrows, the opposite of a resolution/sampling artifact -- it means the
papyrus genuinely undulates in radius by MORE than 2mm within a single
official wrap (real physical rippling of carbonized papyrus, not a bug).
A radius threshold was never a sound way to separate wraps; multi-wrap
grown patches (see data/index_s5_0139/vc_grow_pilot_20260713/) just made the
consequence visible first. This file is retained as a negative-result
diagnostic, not a proposed production solution. A winding-angle prototype now
lives in separate_wraps_by_winding.py, but it remains experimental: it has
passed one official single-wrap control and synthetic controls, not a real
mixed-wrap mesh. Do not use either script as evidence of end-to-end multi-wrap
recovery.

WHAT IS VALIDATED AND REUSABLE:
  - load_axis/perpendicular_frame/point_coords: correct scroll-relative
    (r, theta, along) for any point cloud, reusing the same axis/origin as
    the rest of this project (wrap_radial.json).
  - Native-grid connected-component isolation: separates physically
    distinct surface crossings using TRUE mesh adjacency (which original
    tifxyz row/col a point came from), not 3D distance -- 3D-distance-only
    methods wrongly merge disconnected folds/crossings that happen to sit
    at a similar radius.
  - griddata-based scattered-point-to-grid interpolation with a
    nearest-point distance cutoff (does not invent data across real gaps).
These are genuinely useful for future segmentation work on any wrap/scroll;
the radius-band assumption built on top of them is what does not hold.

METHOD as implemented (radius-band cylindrical projection): for a single
mesh point, define
  r     = perpendicular distance to the scroll axis (already used everywhere
          in this project, see wrap_order.py)
  theta = angle around the axis, in a FIXED reference frame (e0, e1)
  along = position along the axis
then grid (theta*r_mean, along) within one narrow radius band. This produces
a geometrically sane, correctly-labeled image (circumference/axial extent
match physical expectations), but with real, resolution-independent gaps
from the undulation described above -- treat any output as a partial,
patchy view of that radius band, not a complete column.

Usage:
    python scripts/reparametrize_to_column.py --self-test
        # validates against the OFFICIAL w043 segment (known-good, already
        # column-shaped) before trusting the method on new/exploratory mesh.

    python scripts/reparametrize_to_column.py --mesh-dir <tifxyz dir> \
        --texture <scalar-valued PNG on the SAME (H,W) grid as the mesh> \
        --band-mm 4.0 --out <output dir>
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vesuvius_data import ls, download, read_tifxyz  # noqa: E402

ATLAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VX_UM = 2.399
PX_MM = 0.05  # target output resolution; matches the tifxyz scale [0.05,0.05]
              # already used by the official plates (see make_plates.py)
MAX_NEIGHBOR_MM = 0.15  # nearest-neighbor cutoff: beyond this, leave the
                        # output pixel invalid rather than inventing a value


def load_axis():
    """The one scroll axis/origin used everywhere in this project (2.399um
    frame, PCA over all 38 official meshes, Spearman 0.9993 -- see
    wrap_order.py). Reused as-is: refitting from a single small patch would
    be underdetermined (not enough angular coverage to constrain an axis).
    """
    wr = json.load(open(f"{ATLAS}/data/wrap_radial.json"))
    return np.array(wr["axis"]), np.array(wr["origin"])


def perpendicular_frame(axis):
    """A fixed (e0, e1) basis spanning the plane perpendicular to axis, so
    theta means the same physical direction for every band/segment we run
    this on (not re-derived per call)."""
    world = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, world)) > 0.9:  # near-parallel, pick a different one
        world = np.array([1.0, 0.0, 0.0])
    e0 = np.cross(axis, world)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(axis, e0)
    e1 /= np.linalg.norm(e1)
    return e0, e1


def point_coords(pts_mm, origin_mm, axis, e0, e1):
    """pts_mm, origin_mm: (N,3)/(3,) already in mm. Returns r, theta, along
    (mm, radians, mm)."""
    v = pts_mm - origin_mm
    along = v @ axis
    perp = v - np.outer(along, axis)
    r = np.linalg.norm(perp, axis=1)
    theta = np.arctan2(perp @ e1, perp @ e0)
    return r, theta, along


def reparametrize_band(pts_mm, values, origin_mm, axis, e0, e1, r_lo, r_hi, px_mm=PX_MM,
                        native_idx=None, grid_shape=None):
    """pts_mm: (N,3) mesh points already in mm, in the SAME frame as origin.
    values: (N,) scalar to resample (texture intensity, ink score, ...).
    native_idx: (N,2) (row,col) of each point in the mesh's OWN tifxyz grid,
      grid_shape: that grid's (H,W) -- if given, restrict the band to its
      LARGEST connected component in the mesh's native grid before
      unwrapping. This matters: a narrow radius band is NOT guaranteed to
      hit the papyrus surface only once -- real carbonized scrolls fold and
      crumple locally, so the same band can be crossed multiple times at
      very different angular positions within one axial slice (verified
      empirically on the official w043 segment: within one native mesh row,
      r cycles 9.4mm->11.3mm->9.7mm->11.4mm... several times as theta
      advances). Naively gridding ALL of those crossings together aliases
      them into one image. Native-grid adjacency (true 3D neighbors, unlike
      3D distance) lets us keep only the one coherent crossing.
    Returns (image uint8 or float, manifest dict) for the band [r_lo, r_hi).
    """
    r, theta, along = point_coords(pts_mm, origin_mm, axis, e0, e1)
    band = (r >= r_lo) & (r < r_hi)
    if band.sum() < 50:
        return None, {"n_px": int(band.sum()), "skipped": "too few points"}

    n_components, largest_frac = 1, 1.0
    if native_idx is not None and grid_shape is not None:
        band_grid = np.zeros(grid_shape, dtype=bool)
        band_grid[native_idx[band, 0], native_idx[band, 1]] = True
        labels, n_components = ndimage.label(band_grid, structure=np.ones((3, 3)))
        if n_components > 1:
            sizes = ndimage.sum(band_grid, labels, index=range(1, n_components + 1))
            biggest = 1 + int(np.argmax(sizes))
            largest_frac = float(sizes.max() / sizes.sum())
            keep = labels[native_idx[:, 0], native_idx[:, 1]] == biggest
            band = band & keep

    r_b, theta_b, along_b, v_b = r[band], theta[band], along[band], values[band]
    r_mean = float(r_b.mean())

    # unwrap theta so a band that happens to straddle the +-pi seam doesn't
    # split in two; safe because within one band (one turn) theta should
    # span well under 2*pi of real angular travel
    theta_u = np.unwrap(np.sort(theta_b))  # just to detect a seam-crossing case
    if theta_u.max() - theta_b.min() > 1.5 * np.pi:
        # points genuinely straddle the seam -- unwrap properly relative to
        # the circular mean direction instead of a naive sort
        ref = np.angle(np.mean(np.exp(1j * theta_b)))
        theta_b = np.angle(np.exp(1j * (theta_b - ref))) + ref

    arc_b = r_mean * theta_b  # mm, physical arc-length at this band's radius

    arc_lo, arc_hi = arc_b.min(), arc_b.max()
    along_lo, along_hi = along_b.min(), along_b.max()
    W = max(1, int(np.ceil((arc_hi - arc_lo) / px_mm)))
    H = max(1, int(np.ceil((along_hi - along_lo) / px_mm)))
    if W * H > 30_000_000:  # sanity cap, avoid a runaway allocation on bad input
        return None, {"n_px": int(band.sum()), "skipped": f"grid too large {H}x{W}"}

    # Linear (Delaunay) interpolation, not nearest-in-a-quantized-bin: the
    # theta/arc-length remap is not measure-preserving (using one r_mean per
    # band distorts local point spacing), so matching source points to exact
    # output-grid bins under-fills by simple Poisson statistics even though
    # the real data is dense enough -- interpolating between the actual
    # (continuous) point positions fills that correctly. A separate
    # nearest-point distance check (on the true positions, not quantized)
    # keeps us from inventing data across REAL gaps/holes in the mesh.
    yy, xx = np.mgrid[0:H, 0:W]
    query_arc = arc_lo + (xx.ravel() + 0.5) * px_mm
    query_along = along_lo + (yy.ravel() + 0.5) * px_mm
    query = np.stack([query_arc, query_along], axis=1)
    src = np.stack([arc_b, along_b], axis=1)

    tree = cKDTree(src)
    dist, _ = tree.query(query, k=1)
    ok = dist <= MAX_NEIGHBOR_MM

    interp = griddata(src, v_b, query, method="linear")
    interp = np.where(np.isnan(interp), 0.0, interp)
    out = np.where(ok, interp, 0.0).astype(np.float32)
    img = out.reshape(H, W)

    manifest = {
        "n_px_source": int(band.sum()), "r_band_mm": [round(r_lo, 2), round(r_hi, 2)],
        "r_mean_mm": round(r_mean, 3), "arc_range_mm": [round(float(arc_lo), 2), round(float(arc_hi), 2)],
        "along_range_mm": [round(float(along_lo), 2), round(float(along_hi), 2)],
        "output_shape_hw": [H, W], "px_mm": px_mm, "max_neighbor_mm": MAX_NEIGHBOR_MM,
        "valid_frac": round(float(ok.mean()), 3),
        "n_connected_components_in_raw_band": n_components,
        "largest_component_frac_of_band": round(largest_frac, 3),
    }
    return img, manifest


def _self_test():
    """Validate the method against the OFFICIAL w043 segment: known-good,
    already effectively one wrap. Re-parametrizing it should reproduce a
    sane, correctly-proportioned column (circumference in the expected
    46-83mm range for this scroll's radii, no garbling) -- if this fails,
    do not trust the method on the exploratory grown segments."""
    print("=== self-test: reparametrizing the OFFICIAL w043 segment ===")
    work = "/tmp/reparam_selftest_w043"
    os.makedirs(work, exist_ok=True)
    subs, _ = ls("PHerc0139/segments/", delimiter="/")
    seg = next(s for s in subs if "-w043_" in s)
    msubs, _ = ls(f"{seg}mesh/", delimiter="/")
    tdir = next(s for s in msubs if "tifxyz" in s)
    local = os.path.join(work, "mesh")
    os.makedirs(local, exist_ok=True)
    for fn in ["meta.json", "x.tif", "y.tif", "z.tif"]:
        download(tdir + fn, os.path.join(local, fn))
    d = read_tifxyz(local)
    coords, valid = d["coords"], d["valid"]
    native_idx = np.argwhere(valid)
    pts_vx = coords[valid]  # 2.399um voxels, native frame -- no transform needed
    pts_mm = pts_vx * VX_UM / 1000

    axis, origin_vx = load_axis()
    origin_mm = origin_vx * VX_UM / 1000

    # official ds8 ink map, same (H,W) grid as valid -- use as the scalar to
    # resample, so the self-test output is directly comparable by eye to the
    # existing atlas plate for w043
    _, keys = ls(f"{seg}ink-detection/downsampled/", delimiter="/")
    ds8 = [k for k, _ in keys if k.endswith("-ds8.jpg")][0]
    ink_path = download(ds8, os.path.join(work, "ink.jpg"))
    ink_im = Image.open(ink_path).convert("L")
    if ink_im.size != (valid.shape[1], valid.shape[0]):
        # the official ink map and the raw tifxyz (u,v) grid are not always
        # pixel-aligned (different flattening resolutions) -- resize onto
        # the mesh's own grid, since that's the coordinate system r/theta/
        # along are computed in.
        print(f"resizing ink map {ink_im.size} -> mesh grid {(valid.shape[1], valid.shape[0])}")
        ink_im = ink_im.resize((valid.shape[1], valid.shape[0]), Image.BILINEAR)
    ink = np.array(ink_im, dtype=np.float32)
    values = ink[valid]

    e0, e1 = perpendicular_frame(axis)
    r_known = 10.43  # w043's established r_mean, wrap_radial.json
    img, manifest = reparametrize_band(pts_mm, values, origin_mm, axis, e0, e1,
                                        r_known - 1.0, r_known + 1.0,
                                        native_idx=native_idx, grid_shape=valid.shape)
    assert img is not None, f"self-test FAILED to produce an image: {manifest}"
    circumference_mm = manifest["arc_range_mm"][1] - manifest["arc_range_mm"][0]
    axial_mm = manifest["along_range_mm"][1] - manifest["along_range_mm"][0]
    print(json.dumps(manifest, indent=1))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f"{work}/w043_reparam_selftest.png")
    print(f"wrote {work}/w043_reparam_selftest.png")

    # PASS/FAIL checks against known facts about w043 (not vibes):
    checks = {
        "circumference in plausible 2*pi*r range (30-90mm)": 30 <= circumference_mm <= 90,
        "axial extent is a meaningful fraction of the known ~52mm band (>5mm)": axial_mm > 5,
        "output is not blank (std > 5)": float(np.std(img[img > 0])) > 5 if (img > 0).any() else False,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n  valid_frac achieved: {manifest['valid_frac']:.3f}  "
          f"(informational -- see KNOWN LIMITATION below, this is NOT a pass/fail bar)")

    if not all(checks.values()):
        raise SystemExit("SELF-TEST FAILED on the basic sanity checks -- do not trust this method")

    print("""
SELF-TEST result: the geometry/orientation is sane (plausible circumference,
axial extent, non-blank), BUT there is a KNOWN, VERIFIED LIMITATION:

  A narrow radius band does NOT densely fill its own bounding box even on
  the OFFICIAL, known-good w043 segment -- verified directly on the mesh's
  NATIVE (u,v) grid, with NO remapping involved:
      band width 2.00mm -> native fill 35.4%
      band width 1.00mm -> native fill 19.0%
      band width 0.60mm -> native fill 12.5%
  Fill rate gets WORSE as the band narrows -- the opposite of what a
  resolution/sampling problem would show. This means the papyrus surface
  genuinely UNDULATES IN RADIUS by more than 2mm within what is officially
  catalogued as a single wrap -- real physical rippling, not a computation
  bug. A radius threshold is therefore not a valid way to isolate "one
  wrap" from a mesh, for EITHER exploratory grown patches or official
  segments -- it was never a sound assumption, only the multi-wrap grown
  segments happened to make the consequence visible first.

  What IS validated and reusable here: load_axis/perpendicular_frame/
  point_coords (correct scroll-relative r/theta/along for any point cloud),
  the native-grid connected-component technique (correctly separates
  physically distinct crossings using true mesh adjacency, not 3D distance
  -- this part is genuinely useful for other tasks), and the griddata-based
  interpolation utility.

  What is NOT solved here: isolating "one physical wrap" from a mesh that
  mixes several. Radius thresholding is falsified by this control. The
  winding-angle prototype in separate_wraps_by_winding.py is the next
  experimental hypothesis, not a validated replacement; real mixed-wrap
  evaluation remains unfinished engineering work, not a parameter to tune.
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mesh-dir")
    ap.add_argument("--texture")
    ap.add_argument("--frame", choices=["native_2399um", "registered_2399um_from_9362um"],
                     default="native_2399um",
                     help="native_2399um: mesh coords are already 2.399um voxels (official segments). "
                          "registered_2399um_from_9362um: mesh coords are 9.362um and need the "
                          "official transform (grown/registered segments from the VC3D pilot).")
    ap.add_argument("--band-mm", type=float, default=4.0)
    ap.add_argument("--out", default="reparam_out")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.mesh_dir or not args.texture:
        ap.error("--mesh-dir and --texture are required unless --self-test")

    d = read_tifxyz(args.mesh_dir)
    coords, valid = d["coords"], d["valid"]
    native_idx = np.argwhere(valid)
    tex = np.array(Image.open(args.texture).convert("L"), dtype=np.float32)
    assert tex.shape == valid.shape, f"texture {tex.shape} vs mesh grid {valid.shape}"

    pts = coords[valid]
    if args.frame == "registered_2399um_from_9362um":
        t = json.load(open("/tmp/vol_transform.json_20260102.json"))
        M = np.array(t["transformation_matrix"])
        R, T = M[:, :3], M[:, 3]
        pts = (pts - T) @ np.linalg.inv(R).T
    pts_mm = pts * VX_UM / 1000
    values = tex[valid]

    axis, origin_vx = load_axis()
    origin_mm = origin_vx * VX_UM / 1000
    e0, e1 = perpendicular_frame(axis)

    r_all, _, _ = point_coords(pts_mm, origin_mm, axis, e0, e1)
    r_lo, r_hi = np.percentile(r_all, [1, 99])
    n_bands = max(1, int(np.ceil((r_hi - r_lo) / args.band_mm)))

    os.makedirs(args.out, exist_ok=True)
    manifests = []
    for b in range(n_bands):
        b_lo, b_hi = r_lo + b * args.band_mm, r_lo + (b + 1) * args.band_mm
        img, manifest = reparametrize_band(pts_mm, values, origin_mm, axis, e0, e1, b_lo, b_hi,
                                            native_idx=native_idx, grid_shape=valid.shape)
        manifest["band_index"] = b
        if img is not None:
            fname = f"band{b:02d}_{b_lo:.1f}-{b_hi:.1f}mm.png"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f"{args.out}/{fname}")
            manifest["file"] = fname
            print(f"band {b}: {manifest['output_shape_hw']} valid_frac={manifest['valid_frac']}")
        manifests.append(manifest)

    json.dump({"band_mm": args.band_mm, "px_mm": PX_MM, "max_neighbor_mm": MAX_NEIGHBOR_MM,
               "bands": manifests}, open(f"{args.out}/reparam_manifest.json", "w"), indent=1)
    print(f"wrote {args.out}/reparam_manifest.json")


if __name__ == "__main__":
    main()
