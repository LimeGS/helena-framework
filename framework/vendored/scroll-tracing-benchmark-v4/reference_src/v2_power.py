"""Benchmark v2 synthetic power pre-check (spec section 6, pre-freeze).

Question the spec registers: at the curvature of the top stratum, does
arm C (seed normal + true window-scalar pitch) degrade below 85% correct?

Honest simulation, parameters measured from the band's TUNING ZONE only:
- grid spacing 1.0 vox (measured 1.002), surface noise sigma = 0.045 vox
  (RMS normal-component residual vs sigma=3 column smoothing),
- per-cell gap field g with the tuning zone's empirical dispersion
  (lognormal fit to gap/median, spatially smoothed to the measured
  column autocorrelation),
- sheets: concentric arcs of curvature kappa (radians per 100 columns,
  same definition as the selection metric), seed sheet + one sheet at
  local gap g on each side,
- arm C: hop = P2 * unit normal estimated from the NOISY seed grid by the
  same central differences the pipeline uses; P2 = median simulated gap.
- correct: nearest sheet is the expected one AND radial miss < 0.5 * g.

Geometric honesty note, stated up front: for exactly concentric sheets the
normal direction is exact regardless of kappa, so curvature enters only
through noisy-normal estimation; the dominant failure source is the gap
dispersion against a scalar pitch. The check therefore measures the
combination the spec's arm C actually faces, and a flat C(kappa) curve is
a legitimate (pre-registered) outcome that triggers the spec's downgrade
clause rather than a silent amendment.

Run: python v2_power.py --kappas 0.02 0.1 0.3 0.6 1.0 --seed 0
"""
import argparse
import json

import numpy as np

import v2_pipeline as v2


def measure_gap_field_params():
    """Gap dispersion + column autocorrelation length from the tuning zone."""
    xyz, valid = v2.load_band()
    ratios, lags = [], []
    for s in (2000, 2400, 2800, 3200, 3600):
        ref = v2.reference_at(xyz, valid, s)
        import benchmark_core as bc
        gap, _, edge = bc.gaps_for(ref, +1)
        okm = ~(edge | (ref.seed_cls != 0))
        g = np.where(okm, gap, np.nan).reshape(ref.rr.shape)
        med = np.nanmedian(g)
        ratios += list((g[np.isfinite(g)] / med).ravel())
        row = g[g.shape[0] // 2]
        row = row[np.isfinite(row)]
        if len(row) > 40:
            r = row - row.mean()
            ac = np.correlate(r, r, "full")[len(r) - 1:] / (np.arange(len(r), 0, -1) * r.var() + 1e-9)
            below = np.where(ac < 0.5)[0]
            lags.append(int(below[0]) if len(below) else 20)
    ratios = np.asarray(ratios)
    return {
        "log_sigma": float(np.std(np.log(ratios))),
        "corr_cells": float(np.median(lags)),   # in STEP-2 cells
    }


def simulate_arm_c(kappa, gapp, rng, n_rows=100, n_cols=100, pitch=10.0,
                   noise=0.045, cell=2.0):
    """One window at curvature kappa; returns arm-C correct fraction."""
    # gap field: lognormal around pitch, smoothed to the measured corr length
    from scipy.ndimage import gaussian_filter
    logg = rng.normal(0.0, gapp["log_sigma"], (n_rows, n_cols))
    logg = gaussian_filter(logg, sigma=gapp["corr_cells"] / 2.0, mode="nearest")
    logg *= gapp["log_sigma"] / max(logg.std(), 1e-9)
    g_front = pitch * np.exp(logg)

    # concentric geometry: kappa rad per 100 COLUMNS of the band grid
    # (1 vox/col); cells here are STEP-2 samples -> cell=2 vox per cell col.
    dtheta = kappa / 100.0 * cell
    R = max(cell / max(dtheta, 1e-9), 5.0 * pitch)
    theta = (np.arange(n_cols) - n_cols / 2) * dtheta
    zz = (np.arange(n_rows) - n_rows / 2) * cell

    # seed sheet points (radial noise), estimated normals via central diffs
    rad = R + rng.normal(0.0, noise, (n_rows, n_cols))
    X = rad[None is None and slice(None)] * 0  # placeholder to keep flake quiet
    x = rad * np.cos(theta)[None, :]
    y = rad * np.sin(theta)[None, :]
    z = np.tile(zz[:, None], (1, n_cols))
    P = np.stack([x, y, z], axis=-1)

    cp = np.clip(np.arange(n_cols) + 1, 0, n_cols - 1)
    cm = np.clip(np.arange(n_cols) - 1, 0, n_cols - 1)
    rp = np.clip(np.arange(n_rows) + 1, 0, n_rows - 1)
    rm = np.clip(np.arange(n_rows) - 1, 0, n_rows - 1)
    n_est = np.cross(P[:, cp] - P[:, cm], P[rp] - P[rm])
    n_est /= np.maximum(np.linalg.norm(n_est, axis=-1, keepdims=True), 1e-9)
    # orient outward (radial +)
    r_hat = np.stack([np.cos(theta)[None, :] * np.ones((n_rows, 1)),
                      np.sin(theta)[None, :] * np.ones((n_rows, 1)),
                      np.zeros((n_rows, n_cols))], axis=-1)
    sign = np.sign(np.einsum("ijk,ijk->ij", n_est, r_hat))
    n_est *= sign[..., None]

    p2 = float(np.median(g_front))
    pred = P + n_est * p2
    # radial coordinate of prediction; sheets at R + g_front (front, +1) and
    # R - g_back (use same dispersion field for back, independent draw)
    pr = np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
    d_front = np.abs(pr - (R + g_front))
    g_back = pitch * np.exp(gaussian_filter(
        rng.normal(0.0, gapp["log_sigma"], (n_rows, n_cols)),
        sigma=gapp["corr_cells"] / 2.0, mode="nearest") * 1.0)
    d_self = np.abs(pr - R)
    d_back = np.abs(pr - (R - g_back))
    nearest = np.argmin(np.stack([d_self, d_front, d_back]), axis=0)
    correct = (nearest == 1) & (d_front < 0.5 * g_front)
    return float(correct.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kappas", nargs="+", type=float,
                    default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gapp = measure_gap_field_params()
    print(f"gap field (tuning zone): log-sigma={gapp['log_sigma']:.3f} "
          f"corr={gapp['corr_cells']:.0f} cells")
    rng = np.random.default_rng(args.seed)
    results = {}
    for k in args.kappas:
        vals = [simulate_arm_c(k, gapp, rng) for _ in range(args.reps)]
        results[k] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"kappa={k:5.2f} rad/100col: arm C correct = "
              f"{100*results[k][0]:.1f}% +/- {100*results[k][1]:.1f}")
    with open("docs/evidence/v2_power_20260711.json", "w") as f:
        json.dump({"gap_field": gapp,
                   "arm_c_correct_by_kappa": {str(k): v for k, v in results.items()},
                   "reps": args.reps, "seed": args.seed}, f, indent=1)
    print("escrito docs/evidence/v2_power_20260711.json")


if __name__ == "__main__":
    main()
