"""Benchmark v2 scoring: arms A-D, saturation classification, E1/G1.

Implements docs/BENCHMARK_V2_SPEC.md sections 5-6 on the frozen window set.

Frozen denominator, per window and direction: the intersection of the
scorable masks (`ok`) of every arm evaluated there, so all arms are
compared on exactly the same cells; its size is reported. Estimator-arm
abstentions (P2 invalid) drop the whole window from estimator arms only,
reported in the coverage table.

Arm sign convention (spec 5, carried from v1): front = -normal,
back = +normal, in the band_normals orientation, whose agreement with
v1's seed_normals.npy is asserted at startup (mean dot > 0.99).

Run: python v2_score.py --windows docs/evidence/windows_v2.json \
    --out-root out_v2 [--include-v1-window]
"""
import argparse
import json

import numpy as np

import benchmark_core as bc
import v2_pipeline as v2

FACTOR = 4.8 / 7.91


def summarize_on(res, mask):
    inc = int(mask.sum())
    c = int((res["correct"] & mask).sum())
    ww = int((res["wrong_wrap"] & mask).sum())
    dm = int((res["distance_miss"] & mask).sum())
    assert c + ww + dm == inc, "mask must be inside every arm's ok"
    pct = lambda v: 100.0 * v / inc if inc else float("nan")
    return {"included": inc, "correct_pct": pct(c), "wrong_hop_pct": pct(ww + dm),
            "wrong_wrap_pct": pct(ww)}


def p1_per_cell(ref, est, p2):
    p1_grid, fb = v2.estimator_p1(est["spacings"], est["rr"], est["cc"], p2)
    n = int(np.sqrt(len(est["rr"])))
    P1 = p1_grid.reshape(n, n)
    grid_r = est["rr"].reshape(n, n)[:, 0]
    grid_c = est["cc"].reshape(n, n)[0, :]
    cells_r = ref.rr.ravel()
    cells_c = ref.cc.ravel()
    ir = np.argmin(np.abs(cells_r[:, None] - grid_r[None, :]), axis=1)
    ic = np.argmin(np.abs(cells_c[:, None] - grid_c[None, :]), axis=1)
    return P1[ir, ic], fb


def score_window(xyz, valid, normals_band, volume, start, out_root, rng_seed=0):
    ref = v2.reference_at(xyz, valid, start)
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    n_cells = normals_band[ref.rr.ravel(), ref.cc.ravel() + start]
    est = v2.estimator_p2(volume, xyz, normals_band, start,
                          v2.TUNED_SIGMA, v2.TUNED_PROM)
    p2 = est["p2"]
    p1_cells, p1_fallback = (p1_per_cell(ref, est, p2) if np.isfinite(p2)
                             else (None, None))

    out = {"start": start, "p2": None if not np.isfinite(p2) else float(p2),
           "p2_cell_valid_frac": est["cell_valid_frac"],
           "p1_fallback_frac": p1_fallback, "directions": {}}

    for name, e in (("front", +1), ("back", -1)):
        pred = bc.load_grid(f"{out_root}/seed_v2_s{start:05d}_{name}")
        d = pred[ref.rr.ravel(), ref.cc.ravel()] - seed_pts
        unit = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        sign = -1.0 if e == +1 else +1.0
        nn = np.where(np.isfinite(n_cells), n_cells, 0.0)

        def as_grid(pts):
            g = ref.seed.copy()
            g[ref.rr.ravel(), ref.cc.ravel()] = pts
            return g

        arms = {"A": seed_pts + d * FACTOR}
        if np.isfinite(p2):
            arms["B_p2"] = seed_pts + unit * p2
            arms["C_p2"] = seed_pts + nn * (sign * p2)
            arms["B_p1"] = seed_pts + unit * p1_cells[:, None]
            arms["C_p1"] = seed_pts + nn * (sign * p1_cells[:, None])
        arms["D_9vox"] = seed_pts + nn * (sign * 9.0)

        results = {k: bc.score_prediction(ref, as_grid(v), e) for k, v in arms.items()}
        frozen = np.logical_and.reduce([r["ok"] for r in results.values()])
        dirout = {"frozen_included": int(frozen.sum())}
        for k, r in results.items():
            dirout[k] = summarize_on(r, frozen)

        # saturation null: arm-B magnitudes, directions permuted within window
        if np.isfinite(p2):
            rng = np.random.default_rng(rng_seed)
            mag = np.linalg.norm(arms["B_p2"] - seed_pts, axis=-1, keepdims=True)
            perm = unit[rng.permutation(len(unit))]
            null = bc.score_prediction(ref, as_grid(seed_pts + perm * mag), e)
            dirout["null_perm"] = summarize_on(null, frozen & null["ok"])
        out["directions"][name] = dirout
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="docs/evidence/windows_v2.json")
    ap.add_argument("--out-root", default="out_v2")
    ap.add_argument("--out-json", default="docs/evidence/v2_scores_20260711.json")
    args = ap.parse_args()

    xyz, valid = v2.load_band()
    normals_band, _ = v2.band_normals(xyz, valid)
    # sign-convention assertion vs v1
    v1n = np.load("seed_normals.npy")
    mine = normals_band[:, bc.C0:bc.C1][::1]
    ok = np.isfinite(mine).all(-1) & np.isfinite(v1n).all(-1)
    dot = float(np.mean(np.einsum("ijk,ijk->ij", mine, v1n)[ok]))
    print(f"orientacion normales vs v1: mean dot = {dot:.4f}")
    assert dot > 0.99, "normal orientation mismatch vs v1 convention"

    volume = v2.open_zarr()
    sel = json.load(open(args.windows))
    per_window = []
    for w in sel["windows"]:
        r = score_window(xyz, valid, normals_band, volume, int(w["start"]),
                         args.out_root)
        r["stratum"] = w.get("stratum")
        r["kappa"] = w.get("kappa")
        per_window.append(r)
        for name in ("front", "back"):
            d = r["directions"][name]
            line = f"  s{r['start']:05d} {r['stratum']:6s} k={r['kappa']:.3f} {name}: "
            line += " ".join(f"{k}={d[k]['correct_pct']:.1f}%"
                             for k in ("A", "B_p2", "C_p2", "D_9vox") if k in d)
            if "null_perm" in d:
                line += f" null={d['null_perm']['correct_pct']:.1f}%"
            print(line, flush=True)

    # E1/G1 (spec 6): window-direction units; DISCRIMINATING iff null <= 85%
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
    summary = {
        "E1_mean_B_minus_C_pp": e1,
        "discriminating_units": len(contrasts),
        "saturated_units": saturated,
        "B_gt_C_count": int(sum(signs)),
        "G1_directional_skill": g1,
        "verdict": ("evidence of directional skill" if g1 else
                    "no evidence of directional skill beyond a coherent "
                    "local normal in this band"),
    }
    print(json.dumps(summary, indent=1))
    json.dump({"tuned": {"sigma": v2.TUNED_SIGMA, "prom": v2.TUNED_PROM},
               "factor_armA": FACTOR, "windows": per_window,
               "summary": summary},
              open(args.out_json, "w"), indent=1)
    print(f"escrito {args.out_json}")


if __name__ == "__main__":
    main()
