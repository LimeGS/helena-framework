"""Windowed reference construction (port of
reference_src/v2_pipeline.reference_at).

The only window-dependent quantity is u_center, the unwrapped angle at the
window's middle column on cfg.ref_row; the winding-class assignment,
KD-trees and STEP-sampled seed grid are otherwise identical for every
window. cfg supplies the rotation center, CLASSES and STEP (module globals
in reference_src) so the same function builds a reference for any scroll.
"""
import numpy as np
from scipy.spatial import cKDTree

from .core import Reference


def reference_at(xyz, valid, col_start, cfg):
    """v1/v2 reference construction with the window at
    [col_start, col_start + cfg.window)."""
    c0, c1 = int(col_start), int(col_start) + cfg.window
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("xyz must be (rows, cols, 3) and valid must match")
    if not 0 <= cfg.ref_row < xyz.shape[0]:
        raise ValueError(f"ref_row {cfg.ref_row} outside band height {xyz.shape[0]}")
    if valid[cfg.ref_row].sum() < 2:
        raise ValueError("ref_row must contain at least two valid points")
    if not (0 <= c0 and c1 <= xyz.shape[1]):
        raise ValueError(f"window [{c0},{c1}) out of band")
    seed = xyz[:, c0:c1, :]

    cx, cy = cfg.center
    ref_row = cfg.ref_row
    theta = np.where(valid, np.arctan2(xyz[..., 1] - cy, xyz[..., 0] - cx), np.nan)
    unwrapped = np.full_like(theta, np.nan)
    cols_ref = np.where(valid[ref_row])[0]
    unwrapped[ref_row] = np.interp(
        np.arange(xyz.shape[1]), cols_ref, np.unwrap(theta[ref_row, cols_ref])
    )
    for direction in (+1, -1):
        prev = unwrapped[ref_row].copy()
        row = ref_row + direction
        while 0 <= row <= xyz.shape[0] - 1:
            row_unwrapped = theta[row] + 2 * np.pi * np.round(
                (prev - theta[row]) / (2 * np.pi)
            )
            carry = ~np.isfinite(row_unwrapped)
            row_unwrapped[carry] = prev[carry]
            unwrapped[row] = row_unwrapped
            prev = row_unwrapped
            row += direction

    u_valid = np.where(valid, unwrapped, np.nan)
    u_center = u_valid[ref_row, (c0 + c1) // 2]
    winding = (u_valid - u_center) / (2 * np.pi)
    cls = np.where(np.isfinite(winding), np.rint(winding), 99).astype(np.int64)

    trees, rows_of, cols_of, pts_of = {}, {}, {}, {}
    rr_all, cc_all = np.where(valid)
    for n in cfg.classes:
        mask = cls[rr_all, cc_all] == n
        if mask.sum() < 100:
            continue
        pts = xyz[rr_all[mask], cc_all[mask]]
        trees[n] = cKDTree(pts)
        rows_of[n] = rr_all[mask]
        cols_of[n] = cc_all[mask]
        pts_of[n] = pts

    rows_s = np.arange(0, seed.shape[0], cfg.step)
    cols_s = np.arange(0, seed.shape[1], cfg.step)
    rr, cc = np.meshgrid(rows_s, cols_s, indexing="ij")
    seed_cls = cls[rr.ravel(), cc.ravel() + c0]

    return Reference(xyz=xyz, valid=valid, row0=0, seed=seed, cls=cls,
                      trees=trees, rows_of=rows_of, cols_of=cols_of,
                      pts_of=pts_of, rr=rr, cc=cc, seed_cls=seed_cls)


def validate_window_center(xyz, valid, col_start, cfg):
    """Strict V4 preflight without changing legacy selection semantics."""
    c0, c1 = int(col_start), int(col_start) + cfg.window
    if not (0 <= c0 and c1 <= xyz.shape[1]):
        raise ValueError(f"window [{c0},{c1}) out of band")
    center_col = (c0 + c1) // 2
    if not valid[cfg.ref_row, center_col]:
        raise ValueError("window center is invalid on ref_row")
    return True
