"""PHerc1667 scouting: theta-span survey report + gates a/b demo on the
chosen band (see docs/PHERC1667_SCOUT.md for the narrative).

Self-contained: uses reference_src/{benchmark_core,v2_pipeline}.py as a
library (added to sys.path below) exactly as shipped -- reference_src/ is
never modified. Those modules hardcode PHerc0332's rotation center as
module attributes (benchmark_core.CX/CY), so this script monkeypatches
those two attributes to PHerc1667's band-fitted center before calling
v2_pipeline.reference_at; everything else (STEP, CLASSES, WIN, the gates
a/b math in coverage_and_gates_ab, gaps_for, score_prediction) is used
unmodified and is scroll-agnostic already. This does not import stb/
(besides the band npz format, which is plain numpy) to stay independent
of stb/'s parallel, in-progress build.

Run: python scout_1667.py data/pherc1667/band_w013_r1400_1600_xyz.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "reference_src"))

import benchmark_core as bc  # noqa: E402
import v2_pipeline as v2  # noqa: E402


def load_band_npz(path):
    with np.load(path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        row0 = int(data["row0"]) if "row0" in data.files else None
    return xyz, valid, row0


def fit_center_kasa_geometric(xyz, valid, row):
    """Same two-stage recipe as stb.band.fit_center (Kasa seed + geometric
    refinement), reimplemented here so this script has no stb/ dependency."""
    from scipy.optimize import least_squares

    mask = valid[row]
    x, y = xyz[row, mask, 0], xyz[row, mask, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x ** 2 + y ** 2)
    (D, E, _F), *_ = np.linalg.lstsq(A, b, rcond=None)
    guess = np.array([-D / 2.0, -E / 2.0])

    def radial_residual(center):
        r = np.hypot(x - center[0], y - center[1])
        return r - r.mean()

    result = least_squares(radial_residual, guess)
    return float(result.x[0]), float(result.x[1])


def theta_span_report(xyz, valid, cx, cy):
    theta = np.arctan2(xyz[..., 1] - cy, xyz[..., 0] - cx)
    spans = []
    for r in range(valid.shape[0]):
        cols = np.where(valid[r])[0]
        if len(cols) < 50:
            continue
        th = np.unwrap(theta[r, cols])
        spans.append(th.max() - th.min())
    spans = np.array(spans)
    return {
        "n_rows": int(len(spans)),
        "span_min_rad": float(spans.min()),
        "span_median_rad": float(np.median(spans)),
        "span_max_rad": float(spans.max()),
        "rev_min": float(spans.min() / (2 * np.pi)),
        "rev_median": float(np.median(spans) / (2 * np.pi)),
        "rev_max": float(spans.max() / (2 * np.pi)),
    }


def gates_table(xyz, valid, cx, cy, stride=200, win=200):
    """coverage_and_gates_ab (reference_src, unmodified) for every
    non-overlapping stride-window across the band, with CX/CY monkeypatched
    to the band's own fitted center. Also reports median curvature (kappa,
    v2_pipeline.kappa_per_column -- geometric, center-independent) per
    window so passing windows can be stratified low/median/high like
    PHerc0332's cmd_select."""
    bc.CX, bc.CY = cx, cy
    v2.WIN = win  # module-level WIN used by reference_at's window slicing
    normals, n_ok = v2.band_normals(xyz, valid)
    kappa = v2.kappa_per_column(normals, n_ok)
    W = valid.shape[1]
    rows = []
    for s in range(0, W - win + 1, stride):
        ref = v2.reference_at(xyz, valid, s)
        if not all(k in ref.trees for k in (+1, -1, 0)):
            rows.append({"start": s, "coverage": 0.0, "error": "missing class"})
            continue
        g = v2.coverage_and_gates_ab(ref)
        g["start"] = s
        g["kappa"] = float(np.nanmedian(kappa[s:s + win]))
        rows.append(g)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("band_path")
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--out-json", default=None, help="optional path to dump the full report as JSON")
    args = ap.parse_args()

    xyz, valid, row0 = load_band_npz(args.band_path)
    print(f"band: {args.band_path}  shape={xyz.shape}  valid_frac={valid.mean():.4f}  row0={row0}")

    cx, cy = fit_center_kasa_geometric(xyz, valid, row=100)
    print(f"fitted center (row=100 of band, Kasa+geometric): ({cx:.2f}, {cy:.2f})")

    span = theta_span_report(xyz, valid, cx, cy)
    print(f"theta span over {span['n_rows']} rows: "
          f"min={span['span_min_rad']:.3f} median={span['span_median_rad']:.3f} "
          f"max={span['span_max_rad']:.3f} rad  "
          f"({span['rev_min']:.3f} / {span['rev_median']:.3f} / {span['rev_max']:.3f} revolutions)")
    print(f"qualifies (>= 2 rev, 4*pi = {4*np.pi:.3f} rad)? {span['span_min_rad'] >= 4*np.pi}")

    print(f"\ngates a/b + coverage, stride={args.stride} win={args.win} windows:")
    rows = gates_table(xyz, valid, cx, cy, stride=args.stride, win=args.win)
    header = (f"{'start':>6} {'coverage':>9} {'gate_a':>7} {'gate_b':>7} {'kappa':>8} "
              f"{'gapF':>8} {'gapB':>8} {'wrongF%':>8} {'wrongB%':>8}")
    print(header)
    n_pass = 0
    for r in rows:
        if r.get("error"):
            print(f"{r['start']:6d} {'--':>9} {'--':>7} {'--':>7}  {r['error']}")
            continue
        ok = r.get("gate_a_pass") and r.get("gate_b_pass") and r["coverage"] >= 0.40
        n_pass += int(ok)
        print(f"{r['start']:6d} {r['coverage']:9.4f} {str(r.get('gate_a_pass')):>7} "
              f"{str(r.get('gate_b_pass')):>7} {r.get('kappa', float('nan')):8.4f} "
              f"{r.get('gap_median_front', float('nan')):8.2f} "
              f"{r.get('gap_median_back', float('nan')):8.2f} "
              f"{r.get('wrongside_front_correct_pct', float('nan')):8.2f} "
              f"{r.get('wrongside_back_correct_pct', float('nan')):8.2f}")
    print(f"\n{n_pass}/{len(rows)} windows pass coverage>=0.40 + gate_a + gate_b")

    if args.out_json:
        import json
        report = {
            "band_path": args.band_path, "shape": list(xyz.shape), "row0": row0,
            "valid_frac": float(valid.mean()), "fitted_center": {"cx": cx, "cy": cy},
            "theta_span": span, "stride": args.stride, "win": args.win,
            "windows": rows, "n_pass": n_pass, "n_windows": len(rows),
        }
        def _default(o):
            if isinstance(o, (np.bool_, np.integer)):
                return o.item()
            if isinstance(o, np.floating):
                return float(o)
            raise TypeError(f"not JSON serializable: {type(o)}")

        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=1, default=_default)
        print(f"wrote {args.out_json}")
    return rows, span, (cx, cy)


if __name__ == "__main__":
    main()
