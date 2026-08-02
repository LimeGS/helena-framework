"""Shared local-reference scoring utilities for the compressed-wrap benchmark."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    import cv2

    def read_tif(path):
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    def resize_nearest(img, scale):
        return cv2.resize(
            img,
            (img.shape[1] * scale, img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    def write_png(path, img):
        return cv2.imwrite(str(path), img)

except ModuleNotFoundError:
    import tifffile
    from PIL import Image

    def read_tif(path):
        return tifffile.imread(path)

    def resize_nearest(img, scale):
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    def write_png(path, img):
        # color images arrive in cv2's BGR channel order (the palette below is
        # specified BGR); PIL expects RGB, so swap or the failure-map colors
        # render red/blue inverted relative to the bundled figures
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]
        Image.fromarray(img).save(path)
        return True


VOX_UM = 7.91
C0, C1 = 12750, 12950
STEP = 2
CLASSES = list(range(-3, 4))
CX, CY = 1809.26609333, 1732.01937758


@dataclass
class Reference:
    xyz: np.ndarray
    valid: np.ndarray
    row0: int
    seed: np.ndarray
    cls: np.ndarray
    trees: dict
    rows_of: dict
    cols_of: dict
    pts_of: dict
    rr: np.ndarray
    cc: np.ndarray
    seed_cls: np.ndarray


def load_grid(dirpath):
    dirpath = Path(dirpath)
    arrays = []
    for name in ("x", "y", "z"):
        path = dirpath / f"{name}.tif"
        array = read_tif(path)
        if array is None:
            raise FileNotFoundError(f"could not read coordinate TIFF: {path}")
        arrays.append(np.asarray(array, dtype=np.float64))
    x, y, z = arrays
    if x.shape != y.shape or x.shape != z.shape or x.ndim != 2:
        raise ValueError(f"tifxyz coordinates in {dirpath} must be equal-size 2D arrays")
    return np.stack([x, y, z], axis=-1)


def load_reference(npz_path):
    with np.load(npz_path) as data:
        required = {"xyz", "valid", "row0"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"reference NPZ is missing arrays: {', '.join(missing)}")
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        row0 = int(data["row0"])
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("reference must contain xyz=(rows, cols, 3) and matching valid")
    if xyz.shape[0] != 200 or xyz.shape[1] < C1:
        raise ValueError("this benchmark requires 200 rows and columns through C1=12950")
    if not np.isfinite(xyz[valid]).all():
        raise ValueError("valid reference coordinates must be finite")
    seed = xyz[:, C0:C1, :]

    theta = np.where(valid, np.arctan2(xyz[..., 1] - CY, xyz[..., 0] - CX), np.nan)
    unwrapped = np.full_like(theta, np.nan)
    ref_row = 100
    cols_ref = np.where(valid[ref_row])[0]
    if len(cols_ref) < 2:
        raise ValueError("reference row 100 has insufficient valid points for unwrap")
    unwrapped[ref_row] = np.interp(
        np.arange(xyz.shape[1]),
        cols_ref,
        np.unwrap(theta[ref_row, cols_ref]),
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
    u_center = u_valid[100, (C0 + C1) // 2]
    winding = (u_valid - u_center) / (2 * np.pi)
    cls = np.where(np.isfinite(winding), np.rint(winding), 99).astype(np.int64)

    trees, rows_of, cols_of, pts_of = {}, {}, {}, {}
    rr_all, cc_all = np.where(valid)
    for n in CLASSES:
        mask = cls[rr_all, cc_all] == n
        if mask.sum() < 100:
            continue
        pts = xyz[rr_all[mask], cc_all[mask]]
        trees[n] = cKDTree(pts)
        rows_of[n] = rr_all[mask]
        cols_of[n] = cc_all[mask]
        pts_of[n] = pts

    rows_s = np.arange(0, seed.shape[0], STEP)
    cols_s = np.arange(0, seed.shape[1], STEP)
    rr, cc = np.meshgrid(rows_s, cols_s, indexing="ij")
    seed_cls = cls[rr.ravel(), cc.ravel() + C0]

    return Reference(
        xyz=xyz,
        valid=valid,
        row0=row0,
        seed=seed,
        cls=cls,
        trees=trees,
        rows_of=rows_of,
        cols_of=cols_of,
        pts_of=pts_of,
        rr=rr,
        cc=cc,
        seed_cls=seed_cls,
    )


def gaps_for(ref, expected_class):
    if expected_class not in ref.trees:
        raise ValueError(f"reference has no populated winding class {expected_class}")
    seed_points = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    dist, idx = ref.trees[expected_class].query(seed_points, workers=-1)
    edge = np.isin(
        ref.rows_of[expected_class][idx], (0, ref.xyz.shape[0] - 1)
    ) | np.isin(ref.cols_of[expected_class][idx], (0, ref.xyz.shape[1] - 1))
    return dist, idx, edge


def score_prediction(ref, pred_grid, expected_class, threshold_kind="gap_fraction", threshold_value=0.5):
    pred_grid = np.asarray(pred_grid, dtype=np.float64)
    if pred_grid.shape != ref.seed.shape:
        raise ValueError(
            f"prediction shape {pred_grid.shape} does not match seed {ref.seed.shape}"
        )
    if not np.isfinite(pred_grid).all():
        raise ValueError("prediction grid contains non-finite coordinates")
    if float(threshold_value) <= 0:
        raise ValueError("threshold value must be positive")
    pred_points = pred_grid[ref.rr.ravel(), ref.cc.ravel()]
    gap, gap_idx, gap_edge = gaps_for(ref, expected_class)
    d_exp, idx_exp = ref.trees[expected_class].query(pred_points, workers=-1)
    exp_edge = np.isin(
        ref.rows_of[expected_class][idx_exp], (0, ref.xyz.shape[0] - 1)
    ) | np.isin(
        ref.cols_of[expected_class][idx_exp], (0, ref.xyz.shape[1] - 1)
    )

    dists = np.full((pred_points.shape[0], len(CLASSES)), np.inf)
    for j, n in enumerate(CLASSES):
        if n in ref.trees:
            dists[:, j], _ = ref.trees[n].query(pred_points, workers=-1)
    nearest_cls = np.asarray(CLASSES)[np.argmin(dists, axis=1)]

    if threshold_kind == "gap_fraction":
        threshold = threshold_value * gap
    elif threshold_kind == "fixed_vox":
        threshold = np.full_like(gap, float(threshold_value))
    else:
        raise ValueError(f"unknown threshold kind: {threshold_kind}")

    excluded = exp_edge | gap_edge | (ref.seed_cls != 0)
    ok = ~excluded
    wrong_wrap = ok & (nearest_cls != expected_class)
    distance_miss = ok & (nearest_cls == expected_class) & (d_exp >= threshold)
    correct = ok & (nearest_cls == expected_class) & (d_exp < threshold)

    status = np.full(pred_points.shape[0], 3, dtype=np.int8)
    status[wrong_wrap] = 1
    status[distance_miss] = 2
    status[correct] = 0

    return {
        "status": status,
        "ok": ok,
        "excluded": excluded,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct": correct,
        "gap": gap,
        "gap_idx": gap_idx,
        "d_exp": d_exp,
        "idx_exp": idx_exp,
        "nearest_cls": nearest_cls,
        "wrap_index_error": nearest_cls - expected_class,
        "threshold": threshold,
        "ratio": d_exp / np.maximum(gap, 1e-9),
        "pred_points": pred_points,
        "expected_class": expected_class,
    }


def summarize_score(result):
    included = int(result["ok"].sum())
    excluded = int(result["excluded"].sum())
    correct = int(result["correct"].sum())
    wrong_wrap = int(result["wrong_wrap"].sum())
    distance_miss = int(result["distance_miss"].sum())
    wrong_hop = wrong_wrap + distance_miss
    def pct(value):
        return 100.0 * value / included if included else float("nan")

    return {
        "included": included,
        "excluded": excluded,
        "correct": correct,
        "wrong_hop": wrong_hop,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct_pct": pct(correct),
        "wrong_hop_pct": pct(wrong_hop),
        "wrong_wrap_pct": pct(wrong_wrap),
        "distance_miss_pct": pct(distance_miss),
    }


def wrap_error_histogram(result):
    """Return signed nearest-class errors for scorable predictions."""
    values, counts = np.unique(
        result["wrap_index_error"][result["ok"]], return_counts=True
    )
    return {int(value): int(count) for value, count in zip(values, counts)}


def render_status(status, shape):
    palette = {
        0: (60, 180, 60),   # hop-correct
        1: (40, 40, 220),   # wrong adjacent-wrap identity
        2: (0, 165, 255),   # correct wrap but too far
        3: (90, 90, 90),    # excluded
    }
    status = status.reshape(shape)
    img = np.zeros((*status.shape, 3), dtype=np.uint8)
    for key, color in palette.items():
        img[status == key] = color
    return resize_nearest(img, 4)
