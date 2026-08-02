"""Benchmark v2 pipeline: window-parameterized reference, pitch estimators,
curvature, coverage, validity gates and deterministic window selection.

Implements docs/BENCHMARK_V2_SPEC.md sections 1-4. Everything here reuses
benchmark_core's semantics exactly; `reference_at(band, col_start=12750)`
must reproduce v1's reference bit-for-bit (regression-checked by
`python v2_pipeline.py selftest`, which re-scores the bundled native
predictions and asserts the published 63.17/56.28 control).

The only v1 quantity that depends on the window position is the winding
center `u_center` (v1: unwrapped theta at row 100, column (C0+C1)//2).
Everything else (unwrap, classes, KD-trees, STEP-2 sampling, exclusion
rules, gap-fraction threshold) is shared machinery.
"""
import argparse
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

import benchmark_core as bc

BAND_DEFAULT = "band_r1145_200_xyz.npz"
WIN = 200                     # spec 3: 200 rows x 200 columns
TUNE_LO, TUNE_HI = 2000, 4000  # spec 2: tuning zone (column start range)
EXCL = [(1500, 4500), (12550, 13150)]  # spec 2: eligibility exclusions
STRIDE = 100                  # spec 3: candidate col-start grid
PROFILE_T = np.arange(-40.0, 40.0 + 1e-9, 1.0)  # spec 4: +/-40 vox, 1-vox steps


def load_band(path=BAND_DEFAULT):
    with np.load(path) as data:
        return (np.asarray(data["xyz"], dtype=np.float64),
                np.asarray(data["valid"], dtype=bool))


def reference_at(xyz, valid, col_start):
    """v1 reference construction with the window at [col_start, col_start+WIN)."""
    c0, c1 = int(col_start), int(col_start) + WIN
    if not (0 <= c0 and c1 <= xyz.shape[1]):
        raise ValueError(f"window [{c0},{c1}) out of band")
    seed = xyz[:, c0:c1, :]

    theta = np.where(valid, np.arctan2(xyz[..., 1] - bc.CY, xyz[..., 0] - bc.CX), np.nan)
    unwrapped = np.full_like(theta, np.nan)
    ref_row = 100
    cols_ref = np.where(valid[ref_row])[0]
    unwrapped[ref_row] = np.interp(
        np.arange(xyz.shape[1]), cols_ref, np.unwrap(theta[ref_row, cols_ref])
    )
    for direction in (+1, -1):
        prev = unwrapped[ref_row].copy()
        row = ref_row + direction
        while 0 <= row <= xyz.shape[0] - 1:
            row_unwrapped = theta[row] + 2 * np.pi * np.round((prev - theta[row]) / (2 * np.pi))
            carry = ~np.isfinite(row_unwrapped)
            row_unwrapped[carry] = prev[carry]
            unwrapped[row] = row_unwrapped
            prev = row_unwrapped
            row += direction

    u_valid = np.where(valid, unwrapped, np.nan)
    u_center = u_valid[100, (c0 + c1) // 2]
    winding = (u_valid - u_center) / (2 * np.pi)
    cls = np.where(np.isfinite(winding), np.rint(winding), 99).astype(np.int64)

    trees, rows_of, cols_of, pts_of = {}, {}, {}, {}
    rr_all, cc_all = np.where(valid)
    for n in bc.CLASSES:
        mask = cls[rr_all, cc_all] == n
        if mask.sum() < 100:
            continue
        pts = xyz[rr_all[mask], cc_all[mask]]
        trees[n] = bc.cKDTree(pts) if hasattr(bc, "cKDTree") else __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(pts)
        rows_of[n] = rr_all[mask]
        cols_of[n] = cc_all[mask]
        pts_of[n] = pts

    rows_s = np.arange(0, seed.shape[0], bc.STEP)
    cols_s = np.arange(0, seed.shape[1], bc.STEP)
    rr, cc = np.meshgrid(rows_s, cols_s, indexing="ij")
    seed_cls = cls[rr.ravel(), cc.ravel() + c0]

    return bc.Reference(xyz=xyz, valid=valid, row0=0, seed=seed, cls=cls,
                        trees=trees, rows_of=rows_of, cols_of=cols_of,
                        pts_of=pts_of, rr=rr, cc=cc, seed_cls=seed_cls)


def band_normals(xyz, valid):
    """Unit normals over the full band by central differences (v1's
    local_axes convention: normal = cross(col_tangent, row_tangent))."""
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
    """spec 3: kappa(c) = median over rows of arccos(n(r,c+50).n(r,c-50))."""
    H, W = n_ok.shape
    kappa = np.full(W, np.nan)
    for c in range(lag, W - lag):
        both = n_ok[:, c - lag] & n_ok[:, c + lag]
        if both.sum() < 50:
            continue
        dots = np.einsum("ij,ij->i", normals[both, c - lag], normals[both, c + lag])
        kappa[c] = float(np.median(np.arccos(np.clip(dots, -1.0, 1.0))))
    return kappa


def eligible_starts(width):
    """Column starts on the stride grid whose whole window is eligible."""
    starts = []
    for s in range(0, width - WIN + 1, STRIDE):
        window = (s, s + WIN)
        if any(not (window[1] <= lo or window[0] >= hi) for lo, hi in EXCL):
            continue
        starts.append(s)
    return starts


def coverage_and_gates_ab(ref):
    """Prediction-free coverage + gates (a) self-test and (b) wrong-side.

    coverage: fraction of sampled cells scorable for BOTH directions under
    v1 exclusion rules, using the prediction-independent parts
    (gap_edge and seed_cls != 0; exp_edge is prediction-dependent and, for
    on-target predictions, coincides with gap_edge).
    gate a: distance from each scorable seed point to its own class-0 tree
    (median must be 0.000, p90 <= 0.5).
    gate b: the oracle target grid (each cell moved to its nearest
    expected-wrap point) scored against the OPPOSITE class must be
    <= 5% correct, in both directions.
    """
    out = {}
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    scorable = {}
    for e in (+1, -1):
        if e not in ref.trees:
            return {"coverage": 0.0, "error": f"class {e} unpopulated"}
        gap, idx, gap_edge = bc.gaps_for(ref, e)
        scorable[e] = ~(gap_edge | (ref.seed_cls != 0))
        out[f"gap_median_{'front' if e==1 else 'back'}"] = (
            float(np.median(gap[scorable[e]])) if scorable[e].any() else float("nan"))
    both = scorable[+1] & scorable[-1]
    out["coverage"] = float(both.mean())
    if not both.any():
        out["gate_a_pass"] = out["gate_b_pass"] = False
        return out

    d0, _ = ref.trees[0].query(seed_pts[both], workers=-1)
    out["selftest_median"] = float(np.median(d0))
    out["selftest_p90"] = float(np.percentile(d0, 90))
    out["gate_a_pass"] = bool(out["selftest_median"] == 0.0 and out["selftest_p90"] <= 0.5)

    wrong = {}
    for e in (+1, -1):
        gap, idx, _ = bc.gaps_for(ref, e)
        oracle = seed_pts.copy()
        oracle[:] = ref.pts_of[e][idx]
        grid = ref.seed.copy()
        grid[ref.rr.ravel(), ref.cc.ravel()] = oracle
        res = bc.score_prediction(ref, grid, -e)
        s = bc.summarize_score(res)
        wrong[e] = s["correct_pct"]
    out["wrongside_front_correct_pct"] = float(wrong[+1])
    out["wrongside_back_correct_pct"] = float(wrong[-1])
    out["gate_b_pass"] = bool(wrong[+1] <= 5.0 and wrong[-1] <= 5.0)
    return out


# ---------------------------------------------------------------- estimator --

def open_zarr():
    import zarr
    root = zarr.open(
        "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
        "volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr", mode="r")
    try:
        return root["0"]
    except Exception:
        return root


def sample_profiles(volume, seeds, normals):
    """Raw CT profiles (n_cells, len(PROFILE_T)) along each seed normal."""
    pts = seeds[:, None, :] + PROFILE_T[None, :, None] * normals[:, None, :]
    flat = pts.reshape(-1, 3)
    z0, y0, x0 = [max(int(np.floor(flat[:, i].min())) - 2, 0) for i in (2, 1, 0)]
    z1, y1, x1 = [int(np.ceil(flat[:, i].max())) + 3 for i in (2, 1, 0)]
    block = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    coords = np.vstack([flat[:, 2] - z0, flat[:, 1] - y0, flat[:, 0] - x0])
    return map_coordinates(block, coords, order=1, mode="nearest").reshape(len(seeds), -1)


def spacing_from_profile(raw, sigma, prom_frac):
    """spec 4: smooth, find_peaks with prominence >= prom_frac*(p95-p5),
    spacing = median consecutive peak distance; abstain (<2 peaks) -> nan."""
    prof = gaussian_filter1d(raw.astype(np.float64), sigma=sigma)
    lo, hi = np.percentile(prof, [5, 95])
    peaks, _ = find_peaks(prof, prominence=prom_frac * max(hi - lo, 1e-9))
    if len(peaks) < 2:
        return np.nan
    return float(np.median(np.diff(peaks)) * (PROFILE_T[1] - PROFILE_T[0]))


def grid_cells(col_start, n=9):
    """spec 4: 9x9 cell grid across the window (on the STEP-2 sample grid)."""
    rows = np.linspace(10, 190, n).astype(int) // bc.STEP * bc.STEP
    cols = np.linspace(10, 190, n).astype(int) // bc.STEP * bc.STEP
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel()


def estimator_p2(volume, xyz, normals_band, col_start, sigma, prom_frac,
                 cached_raw=None):
    """Window-scalar pitch P2 + per-cell spacings on the 9x9 grid."""
    rr, cc = grid_cells(col_start)
    seeds = xyz[rr, cc + col_start]
    norms = normals_band[rr, cc + col_start]
    good = np.isfinite(norms).all(axis=1) & np.isfinite(seeds).all(axis=1)
    raw = cached_raw
    if raw is None:
        raw = np.full((len(rr), len(PROFILE_T)), np.nan, dtype=np.float32)
        raw[good] = sample_profiles(volume, seeds[good], norms[good])
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
            block = S[max(i-2, 0):i+3, max(j-2, 0):j+3]
            vals = block[np.isfinite(block)]
            if len(vals) >= 5:
                P1[i, j] = np.median(vals)
            else:
                P1[i, j] = p2
                fallback += 1
    return P1.ravel(), fallback / (n * n)


# -------------------------------------------------------------------- CLIs --

def cmd_selftest():
    xyz, valid = load_band()
    ref = reference_at(xyz, valid, bc.C0)
    for name, e, want_c, want_wh in (("front", +1, 36.83, 63.17), ("back", -1, 43.72, 56.28)):
        grid = bc.load_grid(f"gate_3090/out_native/seed_segment_{name}")
        s = bc.summarize_score(bc.score_prediction(ref, grid, e))
        print(f"[selftest {name}] correct {s['correct_pct']:.2f}% wrong-hop {s['wrong_hop_pct']:.2f}%"
              f"  (v1 published: {want_c:.2f}/{want_wh:.2f})")
        assert abs(s["wrong_hop_pct"] - want_wh) < 0.005, "REGRESSION vs v1"
    print("reference_at(C0) reproduces the v1 published control exactly: OK")


def cmd_tune():
    xyz, valid = load_band()
    normals, _ = band_normals(xyz, valid)
    volume = open_zarr()
    starts = [2000, 2400, 2800, 3200, 3600]
    sigmas = [1.0, 1.5, 2.0, 3.0, 4.0]
    proms = [0.10, 0.15, 0.20, 0.30]
    cache, gaps = {}, {}
    for s in starts:
        ref = reference_at(xyz, valid, s)
        est = estimator_p2(volume, xyz, normals, s, 2.0, 0.15)
        cache[s] = est
        gap_f, _, edge = bc.gaps_for(ref, +1)
        gmap = np.full(ref.rr.shape, np.nan).ravel()
        okm = ~(edge | (ref.seed_cls != 0))
        gmap[okm] = gap_f[okm]
        gmap = gmap.reshape(ref.rr.shape)
        idx = (est["rr"] // bc.STEP, est["cc"] // bc.STEP)
        gaps[s] = gmap[idx]
    print("objective: median relative |spacing - KD front gap| over tuning cells"
          " with finite gap; coverage = fraction of cells with a spacing")
    best = None
    for sg in sigmas:
        for pf in proms:
            errs, cov_n, tot = [], 0, 0
            for s in starts:
                est = cache[s]
                spac = np.array([
                    spacing_from_profile(est["raw"][i], sg, pf)
                    for i in range(len(est["rr"]))
                ])
                g = gaps[s]
                m = np.isfinite(spac) & np.isfinite(g)
                errs += list(np.abs(spac[m] - g[m]) / g[m])
                cov_n += int(np.isfinite(spac).sum())
                tot += len(spac)
            med = float(np.median(errs)) if errs else np.inf
            cov = cov_n / tot
            score = med + max(0.0, 0.60 - cov)   # coverage floor 60% soft penalty
            print(f"  sigma={sg:.1f} prom={pf:.2f}: med_rel_err={med:.3f} coverage={cov:.2f} score={score:.3f}")
            if best is None or score < best[0]:
                best = (score, sg, pf, med, cov)
    print(f"TUNED: sigma={best[1]} prominence_frac={best[2]} "
          f"(med_rel_err={best[3]:.3f}, coverage={best[4]:.2f})")




# -------------------------------------------------------------- selection --

TUNED_SIGMA = 0.5        # frozen by docs/evidence/v2_tuning_20260711.log
TUNED_PROM = 0.15

def cmd_select():
    import time
    xyz, valid = load_band()
    normals, n_ok = band_normals(xyz, valid)
    kappa = kappa_per_column(normals, n_ok)
    starts = eligible_starts(valid.shape[1])
    print(f"candidatas elegibles (stride {STRIDE}, ventana {WIN}): {len(starts)}")

    rows = []
    t0 = time.time()
    for i, s in enumerate(starts):
        ref = reference_at(xyz, valid, s)
        g = coverage_and_gates_ab(ref) if (+1 in ref.trees and -1 in ref.trees and 0 in ref.trees) else {"coverage": 0.0, "error": "missing class"}
        kw = kappa[s:s + WIN]
        g["kappa"] = float(np.nanmedian(kw))
        g["start"] = s
        rows.append(g)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(starts)}  ({time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in rows if r["coverage"] >= 0.40 and r.get("gate_a_pass") and r.get("gate_b_pass") and np.isfinite(r["kappa"])]
    print(f"pasan coverage>=0.40 + gates a/b: {len(ok)}")
    ok_by_kappa = sorted(ok, key=lambda r: r["kappa"])
    med = float(np.median([r["kappa"] for r in ok]))

    def try_pick(cands, picked):
        for r in cands:
            if all(abs(r["start"] - p["start"]) >= 300 for p in picked):
                return r
        return None

    picked = []
    strata = [("low", ok_by_kappa), ("low", ok_by_kappa),
              ("median", sorted(ok, key=lambda r: abs(r["kappa"] - med))),
              ("median", sorted(ok, key=lambda r: abs(r["kappa"] - med))),
              ("high", ok_by_kappa[::-1]), ("high", ok_by_kappa[::-1])]
    for name, order in strata:
        r = try_pick([c for c in order if c not in picked], picked)
        if r is None:
            print(f"  WARN: no candidate for stratum {name}")
            continue
        r["stratum"] = name
        picked.append(r)

    out = {
        "spec": "docs/BENCHMARK_V2_SPEC.md v2.0.0",
        "date": "2026-07-11",
        "tuned": {"sigma": TUNED_SIGMA, "prominence_frac": TUNED_PROM},
        "kappa_median_eligible": med,
        "n_candidates": len(starts), "n_eligible": len(ok),
        "windows": [{k: v for k, v in r.items() if k != "error"} for r in picked],
        "all_candidates": [{"start": r["start"], "kappa": r.get("kappa"),
                            "coverage": r.get("coverage"),
                            "gate_a": r.get("gate_a_pass"), "gate_b": r.get("gate_b_pass")}
                           for r in rows],
    }
    with open("docs/evidence/windows_v2.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nSELECCION:")
    for r in picked:
        print(f"  {r['stratum']:6s} start={r['start']:5d} kappa={r['kappa']:.4f} cov={r['coverage']:.2f} "
              f"gapF={r['gap_median_front']:.1f} gapB={r['gap_median_back']:.1f}")
    print("escrito docs/evidence/windows_v2.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "tune", "select"])
    args = ap.parse_args()
    {"selftest": cmd_selftest, "tune": cmd_tune, "select": cmd_select}[args.cmd]()
