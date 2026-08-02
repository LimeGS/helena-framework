"""Generic tifxyz -> band npz converter.

A "band" (stb.band.load_band's format) is just a rectangular row/col crop
of a tifxyz mesh's {x,y,z}.tif coordinate grids plus the derived validity
mask, saved as {xyz: (rows, cols, 3) float32, valid: (rows, cols) bool,
row0: int64 scalar recording the crop's row offset in the source mesh}.
This module has no dependency on the rest of stb/ or on reference_src/ --
it only needs the three coordinate TIFFs vc_render-style tifxyz export
produces (any scroll, any segment), so the same tool builds a band for a
newly-scouted segment (see docs/PHERC1667_SCOUT.md) as well as re-deriving
PHerc0332's fixture from its source mesh if that were ever needed.

Invalid cells in these TIFFs are marked with the sentinel -1.0 in all three
channels (verified against PHerc1667 mesh exports: every -1 in x coincides
exactly with -1 in y and in z, and no other negative values occur).

Reads x.tif/y.tif (and z.tif unless --no-z) with a 3-tier fallback --
cv2 (fastest, handles any TIFF compression via libtiff) -> tifffile (needs
the optional `imagecodecs` package for compressed TIFFs) -> Pillow (pure
Python, reliably decodes LZW-compressed float32 TIFFs without extra
dependencies) -- so it works in environments missing any one of those
packages.
"""
import argparse
from pathlib import Path

import numpy as np

INVALID = -1.0


def _read_tif(path):
    path = str(path)
    try:
        import cv2

        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            return arr
    except ModuleNotFoundError:
        pass
    try:
        import tifffile

        return tifffile.imread(path)
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path))


def read_tifxyz(dirpath, load_z=True):
    """Read a tifxyz mesh directory into (xyz, valid).

    dirpath must contain x.tif and y.tif (z.tif too unless load_z=False);
    all must be equal-shape 2D arrays. valid is True where every loaded
    channel differs from the -1 sentinel.
    """
    dirpath = Path(dirpath)
    names = ("x", "y", "z") if load_z else ("x", "y")
    arrays = []
    for name in names:
        path = dirpath / f"{name}.tif"
        arr = _read_tif(path)
        if arr is None:
            raise FileNotFoundError(f"could not read coordinate TIFF: {path}")
        arrays.append(np.asarray(arr, dtype=np.float32))
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1 or arrays[0].ndim != 2:
        raise ValueError(f"tifxyz coordinates in {dirpath} must be equal-size 2D arrays, got {shapes}")

    valid = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        valid &= a != INVALID
    if load_z:
        x, y, z = arrays
    else:
        x, y = arrays
        z = np.zeros_like(x)
    xyz = np.stack([x, y, z], axis=-1)
    return xyz, valid


def build_band(tifxyz_dir, row_start, row_end, out_path,
               col_start=0, col_end=None, load_z=True):
    """Crop tifxyz_dir's mesh to rows [row_start, row_end) and columns
    [col_start, col_end) and write {xyz, valid, row0} to out_path.

    row0 records row_start (the crop's offset in the source mesh), the
    same provenance field fixtures/band_r1145_200_xyz.npz carries. Returns
    (xyz_crop, valid_crop) for callers that want to inspect the result
    without re-reading the file. load_z=False skips reading z.tif (z comes
    back as all-zero) for a cheaper scout when only x/y (theta, coverage)
    are needed; the result is not a valid band for anything that uses real
    z (height), only for that kind of geometry-only survey.
    """
    xyz, valid = read_tifxyz(tifxyz_dir, load_z=load_z)
    H, W = valid.shape
    row_end = H if row_end is None else row_end
    col_end = W if col_end is None else col_end
    if not (0 <= row_start < row_end <= H):
        raise ValueError(f"row range [{row_start},{row_end}) out of mesh bounds (0,{H})")
    if not (0 <= col_start < col_end <= W):
        raise ValueError(f"col range [{col_start},{col_end}) out of mesh bounds (0,{W})")

    xyz_c = np.ascontiguousarray(xyz[row_start:row_end, col_start:col_end])
    valid_c = np.ascontiguousarray(valid[row_start:row_end, col_start:col_end])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, xyz=xyz_c, valid=valid_c, row0=np.int64(row_start))
    return xyz_c, valid_c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tifxyz_dir", help="directory containing x.tif, y.tif, z.tif")
    ap.add_argument("out_path", help="destination .npz path")
    ap.add_argument("--row-start", type=int, default=0)
    ap.add_argument("--row-end", type=int, default=None, help="default: full height")
    ap.add_argument("--col-start", type=int, default=0)
    ap.add_argument("--col-end", type=int, default=None, help="default: full width")
    ap.add_argument("--no-z", action="store_true",
                     help="skip z.tif (band's z channel is filled with 0.0; "
                          "only useful for a theta/coverage-only scout, not a real band)")
    args = ap.parse_args()

    xyz_c, valid_c = build_band(args.tifxyz_dir, args.row_start, args.row_end,
                                args.out_path, args.col_start, args.col_end,
                                load_z=not args.no_z)
    print(f"wrote {args.out_path}: xyz{xyz_c.shape} valid_frac={valid_c.mean():.4f}")


if __name__ == "__main__":
    main()
