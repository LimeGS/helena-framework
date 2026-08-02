"""PHerc1667 instance, step 3 of docs/PHERC1667_INSTANCE.md: verify (and,
if needed, retune) the CT-profile pitch estimator's sigma/prominence in
the reserved tuning zone (col-starts [199, 1999], see (a)), then write
docs/evidence/v1667_tuning.log with every number the protocol calls for.

Needs network (the PHerc1667 zarr volume, see configs/pherc1667.json's
volume_url); NOT part of the offline test suite (STB_NETWORK-gated
elsewhere in this repo's convention).

Method (mirrors reference_src/v2_pipeline.py's cmd_tune, restricted to
the 1667 tuning zone and using stb's port instead of reference_src):
1. Sample raw CT profiles once per valid tuning-zone window (skip windows
   missing a winding class -- an edge effect, logged, not silently
   dropped) at profile_halfwidth_vox = estimator.halfwidth_vox_from_physical(cfg.vox_um).
   Raw profiles don't depend on sigma/prominence, so they are cached and
   reused for every (sigma, prom) combination tried below.
2. Verify: pool every 9x9-grid-cell's |spacing - kd_front_gap| / kd_front_gap
   (both finite) across every valid tuning-zone window at sigma=0.5,
   prom=0.15 (PHerc0332's frozen values); take the median.
3. Decision: adopt those values verbatim if the pooled median <= 0.35;
   otherwise retune on the same grid v2_pipeline.cmd_tune used
   (sigma in {1.0,1.5,2.0,3.0,4.0}, prom_frac in {0.10,0.15,0.20,0.30}),
   scoring each combo by med_rel_err + max(0, 0.60 - coverage), picking
   the minimum.

Run: python scripts/tune_1667.py
"""
import sys
import json
import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stb import band as stb_band  # noqa: E402
from stb import config as stb_config  # noqa: E402
from stb import core  # noqa: E402
from stb import estimator as estimator_mod  # noqa: E402
from stb import normals as stb_normals  # noqa: E402
from stb import reference as reference_mod  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
EVIDENCE = REPO_ROOT / "docs" / "evidence"

TUNE_LO_HI_WINDOW = 200  # cfg.window, restated for the zone-start-range calc below
VERIFY_THRESHOLD = 0.35
SIGMA_GRID = [1.0, 1.5, 2.0, 3.0, 4.0]
PROM_GRID = [0.10, 0.15, 0.20, 0.30]


def open_volume(url):
    import fsspec
    import zarr

    root = zarr.open(fsspec.get_mapper(url), mode="r")
    try:
        return root["0"]
    except Exception:
        return root


def tuning_zone_starts(cfg, W):
    tune_lo = round(0.05 * W)
    tune_hi = tune_lo + 2000
    starts = []
    s = 0
    while s + cfg.window <= W:
        if tune_lo <= s and s + cfg.window <= tune_hi:
            starts.append(s)
        s += cfg.stride
    return tune_lo, tune_hi, starts


def per_cell_front_gap(ref, rr_grid, cc_grid, cfg):
    """KD-tree front (class +1) gap at each 9x9 grid-cell location,
    restricted exactly as stb.gates.coverage_and_gates_ab's scorable mask
    does (gap_edge cleared, seed_cls == 0 required) -- mirrors
    reference_src/v2_pipeline.cmd_tune's gmap/idx construction."""
    gap, _idx, gap_edge = core.gaps_for(ref, +1)
    scorable = ~(gap_edge | (ref.seed_cls != 0))
    gmap = np.full(ref.rr.shape, np.nan).ravel()
    gmap[scorable] = gap[scorable]
    gmap = gmap.reshape(ref.rr.shape)
    ir = rr_grid // cfg.step
    ic = cc_grid // cfg.step
    return gmap[ir, ic]


def main():
    cfg = stb_config.load_config(CONFIGS / "pherc1667.json")
    xyz, valid, _row0 = stb_band.load_band(cfg.band_path)
    cfg = stb_config.resolve(cfg, xyz, valid)
    W = valid.shape[1]

    tune_lo, tune_hi, starts = tuning_zone_starts(cfg, W)
    halfwidth = estimator_mod.halfwidth_vox_from_physical(cfg.vox_um)

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("PHerc1667 estimator verification/tuning log (docs/PHERC1667_INSTANCE.md step (b))")
    log(f"band: {cfg.band_path} shape={xyz.shape} fitted_center=({cfg.center[0]:.4f}, {cfg.center[1]:.4f})")
    log(f"tuning zone: tune_lo={tune_lo} tune_hi={tune_hi} (rule: round(0.05*W), W={W}; +2000)")
    log(f"candidate tuning-zone starts (stride={cfg.stride}, window={cfg.window}): {starts}")
    log(f"profile_halfwidth_vox = round(316/{cfg.vox_um}) = {halfwidth} "
        f"(docs/PHERC1667_INSTANCE.md (c))")
    log("")

    normals_band, n_ok = stb_normals.band_normals(xyz, valid)
    volume = open_volume(cfg.volume_url)

    valid_starts, skipped = [], []
    cache = {}   # start -> estimator_p2 dict (cached raw profiles)
    gaps = {}    # start -> per-cell front gap array (9x9, matched to est rr/cc)

    for s in starts:
        ref = reference_mod.reference_at(xyz, valid, s, cfg)
        if not all(k in ref.trees for k in (+1, -1, 0)):
            skipped.append(s)
            continue
        est = estimator_mod.estimator_p2(
            volume, xyz, normals_band, s, cfg,
            sigma=estimator_mod.TUNED_SIGMA, prom_frac=estimator_mod.TUNED_PROM,
            profile_halfwidth_vox=halfwidth,
        )
        cache[s] = est
        gaps[s] = per_cell_front_gap(ref, est["rr"], est["cc"], cfg)
        valid_starts.append(s)

    log(f"valid tuning-zone windows (all 3 winding classes populated): {valid_starts}")
    if skipped:
        log(f"skipped (missing winding class -- edge effect near the tuning zone's own "
            f"boundary, not used in any statistic below): {skipped}")
    log("")

    def pooled_median_and_coverage(sigma, prom_frac):
        errs, cov_n, tot = [], 0, 0
        for s in valid_starts:
            est = cache[s]
            spac = np.array([
                estimator_mod.spacing_from_profile(est["raw"][i], sigma, prom_frac)
                for i in range(len(est["rr"]))
            ])
            g = gaps[s]
            m = np.isfinite(spac) & np.isfinite(g) & (g > 0)
            errs += list(np.abs(spac[m] - g[m]) / g[m])
            cov_n += int(np.isfinite(spac).sum())
            tot += len(spac)
        med = float(np.median(errs)) if errs else float("inf")
        cov = cov_n / tot if tot else 0.0
        return med, cov, len(errs)

    med0, cov0, n0 = pooled_median_and_coverage(estimator_mod.TUNED_SIGMA, estimator_mod.TUNED_PROM)
    log(f"VERIFY at PHerc0332's frozen sigma={estimator_mod.TUNED_SIGMA} "
        f"prom={estimator_mod.TUNED_PROM}:")
    log(f"  pooled median relative error = {med0:.4f}  (n_cells={n0}, coverage={cov0:.2f})")
    log(f"  decision threshold: {VERIFY_THRESHOLD}")

    if med0 <= VERIFY_THRESHOLD:
        decision = {"sigma": estimator_mod.TUNED_SIGMA, "prominence_frac": estimator_mod.TUNED_PROM}
        log(f"  DECISION: ADOPT PHerc0332's values verbatim "
            f"({med0:.4f} <= {VERIFY_THRESHOLD}) -- no retune.")
    else:
        log(f"  {med0:.4f} > {VERIFY_THRESHOLD}: RETUNE on the tuning zone, same grid as "
            f"reference_src/v2_pipeline.cmd_tune (sigma in {SIGMA_GRID}, prom_frac in {PROM_GRID}).")
        log("")
        log("  grid search (score = median_relative_error + max(0, 0.60 - coverage)):")
        best = None
        for sg in SIGMA_GRID:
            for pf in PROM_GRID:
                med, cov, n = pooled_median_and_coverage(sg, pf)
                score = med + max(0.0, 0.60 - cov)
                log(f"    sigma={sg:.1f} prom={pf:.2f}: med_rel_err={med:.4f} "
                    f"coverage={cov:.2f} n_cells={n} score={score:.4f}")
                if best is None or score < best[0]:
                    best = (score, sg, pf, med, cov)
        decision = {"sigma": best[1], "prominence_frac": best[2]}
        log(f"  DECISION: RETUNE -> sigma={best[1]} prominence_frac={best[2]} "
            f"(med_rel_err={best[3]:.4f}, coverage={best[4]:.2f}, score={best[0]:.4f})")

    log("")
    log(f"FINAL: {decision}")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cell_arrays = {}
    for s in valid_starts:
        est = cache[s]
        final_spacings = np.array([
            estimator_mod.spacing_from_profile(
                est["raw"][i], decision["sigma"], decision["prominence_frac"]
            ) for i in range(len(est["rr"]))
        ])
        cell_arrays[f"s{s}_raw"] = est["raw"]
        cell_arrays[f"s{s}_spacings"] = final_spacings
        cell_arrays[f"s{s}_rr"] = est["rr"]
        cell_arrays[f"s{s}_cc"] = est["cc"]
        cell_arrays[f"s{s}_kd_front_gap"] = gaps[s]
    cells_path = EVIDENCE / "v1667_tuning_cells.npz"
    np.savez_compressed(cells_path, **cell_arrays)
    (EVIDENCE / "v1667_tuning.log").write_text("\n".join(lines) + "\n")
    tuning_json = {
        "schema": "stb-v4-pherc1667-tuning-v1",
        "date": datetime.date.today().isoformat(),
        "band_sha256": "45250db6c5e08e515acdd392bab660c26d8c9f3e54a80ef8d5bc27d80d6da63e",
        "tune_lo": tune_lo,
        "tune_hi": tune_hi,
        "profile_halfwidth_vox": halfwidth,
        "decision": decision,
        "verification": {"median_relative_error": med0, "coverage": cov0, "cells": n0},
        "cell_fixture": cells_path.name,
    }
    (EVIDENCE / "v1667_tuning.json").write_text(json.dumps(tuning_json, indent=2) + "\n")
    print(f"\nwrote {EVIDENCE / 'v1667_tuning.log'}")
    print(f"wrote {EVIDENCE / 'v1667_tuning.json'}")
    print(f"wrote {cells_path}")
    return decision


if __name__ == "__main__":
    main()
