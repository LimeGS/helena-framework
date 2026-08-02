"""tifxyz -> Prediction adapter (the v2 GPU-tracer path).

Wraps a front/back tifxyz directory pair -- x.tif/y.tif/z.tif, exactly the
format the v2 GPU model wrote to `out_v2/` and reference_src/v2_score.py
scored directly via `benchmark_core.load_grid` -- into stb.contract's
pipeline-agnostic Prediction, so the same stb.core scoring machinery can
score it without knowing anything about tifxyz.

`load_tifxyz` is the missing port of reference_src/benchmark_core.py's
load_grid. It was deliberately not placed in stb/core.py (Agent A's
port): reading a directory of TIFFs is an I/O/format concern that belongs
next to the pipeline that emits that format, not in the scoring core.
"""
from pathlib import Path

import numpy as np

try:
    import cv2

    def _read_tif(path):
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

except ModuleNotFoundError:
    import tifffile

    def _read_tif(path):
        return tifffile.imread(path)


def load_tifxyz(dirpath):
    """Load {x,y,z}.tif from `dirpath` into one (rows, cols, 3) float64
    array (port of reference_src/benchmark_core.load_grid)."""
    dirpath = Path(dirpath)
    arrays = []
    for name in ("x", "y", "z"):
        path = dirpath / f"{name}.tif"
        array = _read_tif(path)
        if array is None:
            raise FileNotFoundError(f"could not read coordinate TIFF: {path}")
        arrays.append(np.asarray(array, dtype=np.float64))
    x, y, z = arrays
    if x.shape != y.shape or x.shape != z.shape or x.ndim != 2:
        raise ValueError(f"tifxyz coordinates in {dirpath} must be equal-size 2D arrays")
    return np.stack([x, y, z], axis=-1)


def _seed_grid_index(grid_shape, step):
    """The STEP-sampled (row, col) index pairs stb.reference.reference_at
    uses to build ref.rr/ref.cc, rederived from the grid's own shape so
    this module never needs a stb.core.Reference passed in. Valid because
    a tifxyz prediction grid for a window is always (band_rows,
    cfg.window, 3) -- the same shape reference_at slices ref.seed to --
    so the identical arange/meshgrid recipe lines up index-for-index."""
    rows_s = np.arange(0, grid_shape[0], step)
    cols_s = np.arange(0, grid_shape[1], step)
    rr, cc = np.meshgrid(rows_s, cols_s, indexing="ij")
    return rr.ravel(), cc.ravel()


def predict(task, dirpath, cfg):
    """WindowTask -> Prediction: `dirpath`'s tifxyz grid, indexed at the
    same STEP-sampled seed cells as task.seed_points (see
    _seed_grid_index). The v2 tracer's grids are dense over the whole
    window (every cell has a coordinate, per its meta.json
    n_stored_valid == area), so this adapter never abstains; NaN rows can
    only appear if the tracer's own TIFF had a hole.
    """
    grid = load_tifxyz(dirpath)
    expected_shape = getattr(task, "grid_shape", None)
    if expected_shape is not None and tuple(grid.shape[:2]) != tuple(expected_shape):
        raise ValueError(
            f"tifxyz grid at {dirpath} has shape {grid.shape[:2]}, "
            f"task requires exact grid shape {tuple(expected_shape)}"
        )
    if getattr(task, "sample_rows", None) is not None:
        rr = np.asarray(task.sample_rows, dtype=np.int64)
        cc = np.asarray(task.sample_cols, dtype=np.int64)
    else:
        rr, cc = _seed_grid_index(grid.shape, cfg.step)
    pred = grid[rr, cc]
    if pred.shape != task.seed_points.shape:
        raise ValueError(
            f"tifxyz grid at {dirpath} yields {pred.shape} predictions, "
            f"task expects {task.seed_points.shape}"
        )
    return pred


def prediction_dir(out_root, start, name):
    """The v2 run's directory-naming convention: out_root/seed_v2_s{start:05d}_{name}
    (name is "front" or "back"), as written by reference_src/v2_score.py
    and read by fixtures/out_v2/."""
    return Path(out_root) / f"seed_v2_s{int(start):05d}_{name}"
