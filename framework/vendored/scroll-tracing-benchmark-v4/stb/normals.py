"""Surface normals and column curvature over a full band (straight port of
reference_src/v2_pipeline.band_normals + kappa_per_column: purely
geometric, no per-scroll parameters)."""
import numpy as np


def band_normals(xyz, valid):
    """Unit normals by central differences (local_axes convention:
    normal = cross(col_tangent, row_tangent))."""
    H, W = valid.shape
    c_plus = np.clip(np.arange(W) + 1, 0, W - 1)
    c_minus = np.clip(np.arange(W) - 1, 0, W - 1)
    r_plus = np.clip(np.arange(H) + 1, 0, H - 1)
    r_minus = np.clip(np.arange(H) - 1, 0, H - 1)
    t_c = xyz[:, c_plus] - xyz[:, c_minus]
    t_r = xyz[r_plus] - xyz[r_minus]
    n = np.cross(t_c, t_r)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        n = n / np.maximum(norm, 1e-9)
    ok = valid & valid[:, c_plus] & valid[:, c_minus] & valid[r_plus] & valid[r_minus] \
        & (norm[..., 0] > 1e-9)
    n[~ok] = np.nan
    return n, ok


def kappa_per_column(normals, n_ok, lag=50):
    """kappa(c) = median over rows of arccos(n(r,c+lag).n(r,c-lag))."""
    H, W = n_ok.shape
    kappa = np.full(W, np.nan)
    for c in range(lag, W - lag):
        both = n_ok[:, c - lag] & n_ok[:, c + lag]
        if both.sum() < 50:
            continue
        dots = np.einsum("ij,ij->i", normals[both, c - lag], normals[both, c + lag])
        kappa[c] = float(np.median(np.arccos(np.clip(dots, -1.0, 1.0))))
    return kappa
