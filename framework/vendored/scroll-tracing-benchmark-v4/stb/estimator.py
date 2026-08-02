"""CT-profile pitch estimators (port of reference_src/v2_pipeline's
sample_profiles, spacing_from_profile, grid_cells, estimator_p2 and
estimator_p1).

`volume` is any object supporting `volume[z0:z1, y0:y1, x0:x1]` -> ndarray
(a zarr array in production; tests inject a plain numpy array, or any
other indexable stand-in, so no network/zarr access is required offline).

TUNED_SIGMA/TUNED_PROM are the frozen tuning result (reference_src/
v2_pipeline.py's TUNED_SIGMA/TUNED_PROM, provenance
docs/evidence/v2_tuning_20260711.log): grid search over sigma/prominence
minimizing median relative error against the KD-tree front gap on the
tuning zone, subject to a >=60% coverage floor.

PROFILE_HALFWIDTH_VOX (default 40, PHerc0332's frozen value) is the
half-range of the CT profile in vox, previously hardcoded as the module
constant PROFILE_T (+/-40 vox, 1-vox steps). docs/PHERC1667_INSTANCE.md
section (c) fixes the *physical* half-range across scrolls at 316 um
(= 40 vox * PHerc0332's 7.91 um/vox) and derives each scroll's own
half-width in vox as round(316 / vox_um) -- see
halfwidth_vox_from_physical below. sample_profiles/estimator_p2 accept an
explicit profile_halfwidth_vox so every existing call site that doesn't
pass one keeps PHerc0332's exact bit-for-bit behavior
(profile_t(40) == PROFILE_T).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

TUNED_SIGMA = 0.5
TUNED_PROM = 0.15

PROFILE_HALFWIDTH_VOX = 40  # PHerc0332 frozen default; see docstring above
PROFILE_T = np.arange(-40.0, 40.0 + 1e-9, 1.0)  # +/-40 vox, 1-vox steps == profile_t(40)

PHYSICAL_HALFWIDTH_UM = 316.0  # 40 vox * 7.91 um/vox (PHerc0332), rounded;
# the scroll-independent physical constant docs/PHERC1667_INSTANCE.md (c)
# fixes: the estimator should look the same physical distance into the
# sheet stack regardless of which scroll's voxel size it samples at.


def profile_t(halfwidth_vox):
    """+/-halfwidth_vox (1-vox steps) profile-sampling grid.
    profile_t(40) is bit-identical to the frozen module constant
    PROFILE_T (PHerc0332's default)."""
    hw = float(halfwidth_vox)
    return np.arange(-hw, hw + 1e-9, 1.0)


def halfwidth_vox_from_physical(vox_um, physical_um=PHYSICAL_HALFWIDTH_UM):
    """profile_halfwidth_vox for a scroll of voxel size vox_um, holding the
    physical CT-profile half-range fixed at `physical_um` (default 316 um,
    PHerc0332's own +/-40 vox @ 7.91um/vox). round(316/7.91) == 40 (exact
    match to the historical hardcoded default); round(316/2.399) == 132
    (PHerc1667, docs/PHERC1667_INSTANCE.md (c))."""
    return int(round(float(physical_um) / float(vox_um)))


def sample_profiles(volume, seeds, normals, profile_halfwidth_vox=PROFILE_HALFWIDTH_VOX,
                    out_of_bounds="abstain", return_valid=False):
    """Raw CT profiles (n_cells, 2*profile_halfwidth_vox+1) along each seed
    normal."""
    t = profile_t(profile_halfwidth_vox)
    if out_of_bounds not in {"abstain", "legacy_nearest"}:
        raise ValueError("out_of_bounds must be abstain or legacy_nearest")
    seeds = np.asarray(seeds, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if seeds.shape != normals.shape or seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("seeds and normals must both be (n, 3)")
    pts = seeds[:, None, :] + t[None, :, None] * normals[:, None, :]
    # volume is indexed Z,Y,X while points are X,Y,Z.
    z_size, y_size, x_size = (int(v) for v in volume.shape[:3])
    complete = (
        (pts[..., 0] >= 0) & (pts[..., 0] <= x_size - 1) &
        (pts[..., 1] >= 0) & (pts[..., 1] <= y_size - 1) &
        (pts[..., 2] >= 0) & (pts[..., 2] <= z_size - 1)
    ).all(axis=1)
    use = np.ones(len(seeds), dtype=bool) if out_of_bounds == "legacy_nearest" else complete
    raw = np.full((len(seeds), len(t)), np.nan, dtype=np.float32)
    if use.any():
        flat = pts[use].reshape(-1, 3)
        z0, y0, x0 = [max(int(np.floor(flat[:, i].min())) - 2, 0) for i in (2, 1, 0)]
        z1 = min(int(np.ceil(flat[:, 2].max())) + 3, z_size)
        y1 = min(int(np.ceil(flat[:, 1].max())) + 3, y_size)
        x1 = min(int(np.ceil(flat[:, 0].max())) + 3, x_size)
        block = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32)
        if block.size:
            coords = np.vstack([flat[:, 2] - z0, flat[:, 1] - y0, flat[:, 0] - x0])
            raw[use] = map_coordinates(
                block, coords, order=1, mode="nearest"
            ).reshape(int(use.sum()), -1)
    return (raw, complete) if return_valid else raw


def spacing_from_profile(raw, sigma, prom_frac):
    """Smooth, find_peaks with prominence >= prom_frac*(p95-p5), spacing =
    median consecutive peak distance; abstain (<2 peaks) -> nan.

    The multiplier is the profile's sampling step, always 1.0 vox
    regardless of profile_halfwidth_vox (only the *extent* of
    profile_t/PROFILE_T scales with the scroll; the step between
    consecutive samples never does), so it is a literal here rather than
    derived from PROFILE_T (which a caller may have sampled at a
    different half-width than the module default)."""
    prof = gaussian_filter1d(raw.astype(np.float64), sigma=sigma)
    lo, hi = np.percentile(prof, [5, 95])
    peaks, _ = find_peaks(prof, prominence=prom_frac * max(hi - lo, 1e-9))
    if len(peaks) < 2:
        return np.nan
    return float(np.median(np.diff(peaks)) * 1.0)


def _axis_grid(length, step, n):
    if length <= 0 or step <= 0 or n <= 0:
        raise ValueError("length, step and n must be positive")
    margin = max(step, int(round(0.05 * length)))
    lo = min(margin, length - 1)
    hi = max(lo, length - margin)
    return np.linspace(lo, hi, n).astype(int) // step * step


def grid_cells(col_start, step, n=9, height=200, window=200):
    """n x n grid derived from actual band height and configured window.

    Defaults reproduce the frozen 10..190 PHerc0332 grid bit-for-bit.
    `col_start` remains in the signature for API compatibility; returned
    columns are window-local and callers add it when indexing the full band.
    """
    rows = _axis_grid(int(height), int(step), int(n))
    cols = _axis_grid(int(window), int(step), int(n))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel()


def estimator_p2(volume, xyz, normals_band, col_start, cfg, sigma=TUNED_SIGMA,
                  prom_frac=TUNED_PROM, cached_raw=None,
                  profile_halfwidth_vox=PROFILE_HALFWIDTH_VOX):
    """Window-scalar pitch P2 + per-cell spacings on the 9x9 grid.

    profile_halfwidth_vox (default PROFILE_HALFWIDTH_VOX=40, PHerc0332's
    frozen value) is forwarded to sample_profiles; see module docstring
    and docs/PHERC1667_INSTANCE.md (c) for how a different scroll picks
    its own value (halfwidth_vox_from_physical(cfg.vox_um))."""
    rr, cc = grid_cells(
        col_start, cfg.step, height=xyz.shape[0], window=cfg.window
    )
    seeds = xyz[rr, cc + col_start]
    norms = normals_band[rr, cc + col_start]
    good = np.isfinite(norms).all(axis=1) & np.isfinite(seeds).all(axis=1)
    raw = cached_raw
    if raw is None:
        n_samples = len(profile_t(profile_halfwidth_vox))
        raw = np.full((len(rr), n_samples), np.nan, dtype=np.float32)
        sampled, complete = sample_profiles(
            volume, seeds[good], norms[good], profile_halfwidth_vox,
            out_of_bounds="abstain", return_valid=True,
        )
        raw[good] = sampled
        good_indices = np.where(good)[0]
        good[good_indices[~complete]] = False
    spac = np.array([
        spacing_from_profile(raw[i], sigma, prom_frac) if good[i] else np.nan
        for i in range(len(rr))
    ])
    valid = np.isfinite(spac)
    p2 = float(np.median(spac[valid])) if valid.mean() >= 0.30 else np.nan
    return {"p2": p2, "cell_valid_frac": float(valid.mean()),
            "spacings": spac, "rr": rr, "cc": cc, "raw": raw}


def estimator_p1(spacings, rr, cc, p2):
    """Per-cell P1: median over valid spacings of the 5x5 neighbor cells on
    the 9x9 grid; abstain (<5 valid) -> fallback to P2 (fraction reported)."""
    n = int(np.sqrt(len(rr)))
    S = spacings.reshape(n, n)
    P1 = np.full((n, n), np.nan)
    fallback = 0
    for i in range(n):
        for j in range(n):
            block = S[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
            vals = block[np.isfinite(block)]
            if len(vals) >= 5:
                P1[i, j] = np.median(vals)
            else:
                P1[i, j] = p2
                fallback += 1
    return P1.ravel(), fallback / (n * n)
