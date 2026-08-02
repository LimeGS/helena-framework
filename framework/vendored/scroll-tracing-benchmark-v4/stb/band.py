"""Band loading and rotation-center fitting.

A "band" is the (rows, cols, 3) xyz mesh + (rows, cols) valid mask that
reference_src/v2_pipeline.py's load_band produces (fixtures/*.npz also
carry an optional row0 provenance scalar recording the original sheet's
row offset; it is not used by any algorithm here).
"""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def load_band(path):
    """Load {xyz, valid[, row0]} from an .npz band file."""
    with np.load(Path(path)) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        row0 = int(data["row0"]) if "row0" in data.files else None
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("band must contain xyz=(rows, cols, 3) and matching valid")
    return xyz, valid, row0


def fit_center(xyz, valid, row=100):
    """Least-squares circle fit to one band row's valid (x, y) points,
    returning (cx, cy).

    Two-stage fit, the standard circle-fit recipe (Coope 1993): the Kasa
    algebraic solution (closed-form linear least squares on
    x^2+y^2+Dx+Ey+F=0) seeds a geometric refinement (Gauss-Newton on the
    actual radial residual r_i - mean(r)).
    """
    mask = valid[row]
    if mask.sum() < 3:
        raise ValueError(f"row {row} has fewer than 3 valid points to fit a circle")
    x = xyz[row, mask, 0]
    y = xyz[row, mask, 1]

    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x ** 2 + y ** 2)
    (D, E, _F), *_ = np.linalg.lstsq(A, b, rcond=None)
    guess = np.array([-D / 2.0, -E / 2.0])

    def radial_residual(center):
        r = np.hypot(x - center[0], y - center[1])
        return r - r.mean()

    result = least_squares(radial_residual, guess)
    return float(result.x[0]), float(result.x[1])
