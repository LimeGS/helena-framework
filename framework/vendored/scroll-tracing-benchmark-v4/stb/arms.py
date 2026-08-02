"""Benchmark v2 scoring: arms A-D, saturation classification, E1/G1 summary.

Port of reference_src/v2_score.py's score_window + the tail of main()
(the E1/G1 aggregation), WITHOUT the CLI wrapper (argparse, the zarr
open, the normal-orientation self-check against seed_normals.npy -- all
one-time setup that belongs to a runner script, not this library).

Frozen denominator, per window and direction: the intersection of the
scorable masks (`ok`) of every arm evaluated there, so all arms are
compared on exactly the same cells; its size is reported as
`frozen_included`.

Arm sign convention (spec 5, carried from v1): front (+1) hops by
-normal, back (-1) hops by +normal -- `sign = -1.0 if direction == +1
else 1.0`.

Unlike reference_src/v2_score.py, train_um/vox_um both come from `cfg`
(ScrollConfig) instead of the hardcoded module constant `FACTOR = 4.8 /
7.91`, so arm A's factor is scroll-config-driven (see stb/config.py's
`train_um` field and BLOCKERS.md's Agent-B note on it).
"""
import numpy as np

from . import core
from . import estimator as estimator_mod
from . import reference as reference_mod
from adapters import tifxyz_tracer


def normal_displacement_physical(seed_points, normals, direction, distance_um, cfg):
    """V4 normal baseline with a scroll-independent physical displacement."""
    if direction not in (+1, -1):
        raise ValueError("direction must be +1 or -1")
    if distance_um <= 0 or cfg.vox_um <= 0:
        raise ValueError("distance_um and cfg.vox_um must be positive")
    sign = -1.0 if direction == +1 else 1.0
    return np.asarray(seed_points) + np.asarray(normals) * (sign * distance_um / cfg.vox_um)


def factor_arm_a(cfg):
    """Arm A's uniform scale factor: (GPU-tracer training voxel size) /
    (this scroll's scan voxel size), both in um/voxel -- both read off
    cfg (train_um, vox_um). reference_src/v2_score.py: FACTOR = 4.8 /
    7.91, PHerc0332-specific; cfg.train_um/cfg.vox_um generalizes it."""
    return float(cfg.train_um) / float(cfg.vox_um)


def summarize_on(res, mask):
    """correct / wrong-hop / wrong-wrap percentages of a
    core.score_prediction result, restricted to `mask` (the frozen
    cross-arm denominator). Port of v2_score.summarize_on."""
    inc = int(mask.sum())
    c = int((res["correct"] & mask).sum())
    ww = int((res["wrong_wrap"] & mask).sum())
    dm = int((res["distance_miss"] & mask).sum())
    assert c + ww + dm == inc, "mask must be inside every arm's ok"
    pct = lambda v: 100.0 * v / inc if inc else float("nan")
    return {"included": inc, "correct_pct": pct(c), "wrong_hop_pct": pct(ww + dm),
            "wrong_wrap_pct": pct(ww)}


def p1_per_cell(ref, spacings, grid_rr, grid_cc, p2):
    """Project the 9x9 per-cell P1 estimate (stb.estimator.estimator_p1,
    sampled on the coarse `grid_rr`/`grid_cc` cell centers) onto every
    scoring cell of `ref`'s STEP-sampled seed grid: nearest row match and
    nearest column match independently (both grids are separable
    row x col lattices), then index. Port of v2_score.p1_per_cell."""
    p1_grid, fallback_frac = estimator_mod.estimator_p1(spacings, grid_rr, grid_cc, p2)
    n = int(np.sqrt(len(grid_rr)))
    P1 = p1_grid.reshape(n, n)
    grid_r = grid_rr.reshape(n, n)[:, 0]
    grid_c = grid_cc.reshape(n, n)[0, :]
    cells_r = ref.rr.ravel()
    cells_c = ref.cc.ravel()
    ir = np.argmin(np.abs(cells_r[:, None] - grid_r[None, :]), axis=1)
    ic = np.argmin(np.abs(cells_c[:, None] - grid_c[None, :]), axis=1)
    return P1[ir, ic], fallback_frac


def score_window(xyz, valid, normals_band, volume, start, out_root, cfg,
                  rng_seed=0, p2_estimate=None):
    """Score one window's front/back tifxyz predictions (loaded from
    `out_root`, v2's `seed_v2_s{start:05d}_{front,back}` layout) on arms
    A/B_p2/C_p2/B_p1/C_p1/D_9vox + the saturation null, against the
    ScrollConfig-parameterized reference at `start`.

    p2_estimate, when given, REPLACES the estimator_mod.estimator_p2(volume,
    ...) call -- a dict with at least {"p2": float|nan,
    "cell_valid_frac": float} and, optionally, {"spacings", "rr", "cc"}
    (estimator_p2's per-cell arrays). This is how offline callers
    reproduce a pinned run without volume/network access: inject the
    *recorded* p2 (fixtures/v2_scores_20260711.json carries p2 and
    p2_cell_valid_frac per window). `volume` may be None whenever
    p2_estimate makes the real CT sampling unnecessary.

    Recorded fixtures never carry the raw per-cell "spacings" the real
    run sampled from CT (only the reduced p2/cell_valid_frac scalars), so
    when p2_estimate lacks "spacings"/"rr"/"cc" the B_p1/C_p1 arms -- which
    need a per-cell P1 -- are skipped rather than fabricated (p1_fallback_frac
    is reported as None in that case). PLAN_V3.md's pinned test (b) only
    checks A/B_p2/C_p2/D_9vox/null_perm for exactly this reason.
    """
    ref = reference_mod.reference_at(xyz, valid, start, cfg)
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    n_cells = normals_band[ref.rr.ravel(), ref.cc.ravel() + start]

    if p2_estimate is None:
        est = estimator_mod.estimator_p2(volume, xyz, normals_band, start, cfg)
    else:
        est = p2_estimate
    p2 = est["p2"]
    have_p1_inputs = all(k in est for k in ("spacings", "rr", "cc"))
    p1_cells, p1_fallback = None, None
    if np.isfinite(p2) and have_p1_inputs:
        p1_cells, p1_fallback = p1_per_cell(ref, est["spacings"], est["rr"], est["cc"], p2)

    out = {"start": start, "p2": None if not np.isfinite(p2) else float(p2),
           "p2_cell_valid_frac": est["cell_valid_frac"],
           "p1_fallback_frac": p1_fallback, "directions": {}}

    factor = factor_arm_a(cfg)
    for name, e in (("front", +1), ("back", -1)):
        pred = tifxyz_tracer.load_tifxyz(tifxyz_tracer.prediction_dir(out_root, start, name))
        d = pred[ref.rr.ravel(), ref.cc.ravel()] - seed_pts
        unit = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        sign = -1.0 if e == +1 else 1.0
        nn = np.where(np.isfinite(n_cells), n_cells, 0.0)

        def as_grid(pts):
            g = ref.seed.copy()
            g[ref.rr.ravel(), ref.cc.ravel()] = pts
            return g

        arms = {"A": seed_pts + d * factor}
        if np.isfinite(p2):
            arms["B_p2"] = seed_pts + unit * p2
            arms["C_p2"] = seed_pts + nn * (sign * p2)
            if p1_cells is not None:
                arms["B_p1"] = seed_pts + unit * p1_cells[:, None]
                arms["C_p1"] = seed_pts + nn * (sign * p1_cells[:, None])
        arms["D_9vox"] = seed_pts + nn * (sign * 9.0)

        results = {k: core.score_prediction(ref, as_grid(v), e, cfg) for k, v in arms.items()}
        frozen = np.logical_and.reduce([r["ok"] for r in results.values()])
        dirout = {"frozen_included": int(frozen.sum())}
        for k, r in results.items():
            dirout[k] = summarize_on(r, frozen)

        # saturation null: arm-B magnitudes, directions permuted within window
        if np.isfinite(p2):
            rng = np.random.default_rng(rng_seed)
            mag = np.linalg.norm(arms["B_p2"] - seed_pts, axis=-1, keepdims=True)
            perm = unit[rng.permutation(len(unit))]
            null = core.score_prediction(ref, as_grid(seed_pts + perm * mag), e, cfg)
            dirout["null_perm"] = summarize_on(null, frozen & null["ok"])
        out["directions"][name] = dirout
    return out


def summarize_run(per_window):
    """E1/G1 (spec 6): window-direction units; DISCRIMINATING iff the
    saturation null scores <= 85% correct. Port of v2_score.main's
    post-loop summary block."""
    contrasts, signs, saturated = [], [], []
    for r in per_window:
        for name in ("front", "back"):
            d = r["directions"][name]
            if "null_perm" not in d or "B_p2" not in d:
                continue
            unit_id = f"s{r['start']}:{name}"
            if d["null_perm"]["correct_pct"] > 85.0:
                saturated.append(unit_id)
                continue
            contrasts.append(d["B_p2"]["correct_pct"] - d["C_p2"]["correct_pct"])
            signs.append(d["B_p2"]["correct_pct"] > d["C_p2"]["correct_pct"])
    e1 = float(np.mean(contrasts)) if contrasts else float("nan")
    g1 = bool(np.isfinite(e1) and e1 >= 5.0 and sum(signs) >= 4)
    return {
        "E1_mean_B_minus_C_pp": e1,
        "discriminating_units": len(contrasts),
        "saturated_units": saturated,
        "B_gt_C_count": int(sum(signs)),
        "G1_directional_skill": g1,
        "verdict": ("evidence of directional skill" if g1 else
                    "no evidence of directional skill beyond a coherent "
                    "local normal in this band"),
    }
