#!/usr/bin/env python3
"""Experimental winding-angle separation of a mesh into candidate "wraps".

This is a research prototype, not a validated multi-wrap recovery method. Its
versioned self-test covers one official single-wrap control plus synthetic
multi-turn and invalid-geometry controls; it has not been evaluated against a
real mixed-wrap mesh with frozen ground truth.

HYPOTHESIS: WINDING ANGLE, NOT RADIUS (2026-07-13 experiment):
  A scroll is ONE continuous spiral sheet. "One wrap" is not a physically
  distinct object with a boundary -- it is a 2*pi-angular-length section of the
  continuous sheet. Official wNNN segments are just consecutive ~2*pi sections.
  So "separate one wrap from a multi-wrap mesh" = "cut the mesh into
  2*pi-angular-length pieces along the cumulative winding-angle coordinate."
  There is NO 3D discontinuity to find at a wrap boundary (which is exactly why
  the grow pilot's Gate-C step/coherence check finds none -- the sheet IS
  continuous across the boundary), and RADIUS cannot separate wraps because on
  PHerc0139 the papyrus undulates in radius by ~6mm peak-to-peak within a single
  wrap while the wrap PITCH is only ~0.17mm/turn -- a ~50x confound (measured
  directly on the official w043 segment's native grid; see KNOWHOW.md sec 7-8).
  Winding angle may be less sensitive to radial undulation because it advances
  by ~2*pi per physical turn; this is the hypothesis tested by this prototype.

KEY STRUCTURAL FACT (measured): VC3D's native tifxyz (u,v) grid is ALREADY a
flattened parametrization -- for a coherent wrap, one grid axis is ~monotonic in
winding angle and the other is ~monotonic in axial position (w043: corr(row,
along)=0.999, and unwrapped winding is a clean monotonic gradient across the
grid). So a correctly-isolated single wrap needs NO cylindrical remap at all --
you just CUT the native grid at each 2*pi of unwrapped winding. Each piece is a
contiguous sub-grid that is already a column.

THE GATE (this is the load-bearing part). Winding angle is only meaningful when
the mesh genuinely winds AROUND the axis. Two failure modes make it garbage, and
both occur in real data:
  (a) the axis PIERCES the sheet (perpendicular distance -> 0 at an interior
      point) -> radius shows a bullseye and theta a 2*pi VORTEX around the
      puncture; the "winding" you measure is the spurious vortex, not scroll
      turns. This is exactly what the ESTABLISHED GLOBAL axis does to meshes
      grown OUTSIDE the segmented band (their local structure axis is tilted
      55-90 deg from the extrapolated global axis, and it stabs through them).
  (b) the sheet is warped/saddle-shaped -> no straight cylinder axis fits, so
      "winding about the axis" is ill-defined.
  Gate metrics (thresholds calibrated on the official w043 control and the
  synthetic invalid control):
    minr_ratio = p1(r)/median(r)  must be > 0.30   (axis stays outside the sheet)
    resid/R    = rms(r-R)/R       must be < 0.30   (sheet is cylindrical here)
  If the gate FAILS, this returns a clear refusal. Passing the gate does not
  establish correct wrap identity; that requires a real mixed-wrap benchmark.

Usage:
  python scripts/separate_wraps_by_winding.py --self-test
  python scripts/separate_wraps_by_winding.py --mesh-dir <tifxyz> \
      [--frame native_2399um|registered_2399um_from_9362um] [--axis global|local]
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import least_squares
from skimage.restoration import unwrap_phase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vesuvius_data import ls, download, read_tifxyz  # noqa: E402

ATLAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VX_UM = 2.399
GATE_MINR_RATIO = 0.30   # axis must stay outside the sheet (no puncture/vortex)
GATE_RESID_OVER_R = 0.30  # sheet must be cylindrical about the axis
SINGLE_WRAP_MAX_TURNS = 1.35  # <= this spans one wrap (+boundary sliver); no split


def load_axis():
    wr = json.load(open(f"{ATLAS}/data/wrap_radial.json"))
    return np.array(wr["axis"], float), np.array(wr["origin"], float)


def _perp_basis(a):
    e0 = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(e0) < 1e-6:
        e0 = np.cross(a, [1.0, 0.0, 0.0])
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(a, e0)
    return e0, e1


def cyl_about(pts, p0, axis):
    """(r, winding_theta_wrapped) of pts about the line (p0, axis)."""
    a = axis / np.linalg.norm(axis)
    v = pts - p0
    perp = v - np.outer(v @ a, a)
    r = np.linalg.norm(perp, axis=1)
    e0, e1 = _perp_basis(a)
    theta = np.arctan2(perp @ e1, perp @ e0)
    return r, theta


def fit_local_cylinder(pts, axis0):
    """Free least-squares cylinder fit (axis position + direction + radius),
    seeded from axis0. Returns (p0, axis, R, resid_rms)."""
    sub = pts if len(pts) <= 40000 else pts[
        np.random.default_rng(0).choice(len(pts), 40000, replace=False)]
    c = sub.mean(0)
    r0, _ = cyl_about(sub, c, axis0)
    def resid(x):
        r, _ = cyl_about(sub, x[:3], x[3:6])
        return r - x[6]
    sol = least_squares(resid, np.r_[c, axis0, np.median(r0)],
                        method="lm", max_nfev=4000)
    p0 = sol.x[:3]
    a = sol.x[3:6] / np.linalg.norm(sol.x[3:6])
    R = abs(sol.x[6])
    r, _ = cyl_about(sub, p0, a)
    return p0, a, R, float(np.sqrt(np.mean((r - R) ** 2)))


def winding_field(pts, native_idx, grid_shape, p0, axis):
    """Unwrapped cumulative winding angle on the mesh's native (row,col) grid.
    Returns (Wg [H,W] with nan off-mesh, r_all, mask)."""
    H, W = grid_shape
    r, theta = cyl_about(pts, p0, axis)
    Tg = np.full((H, W), np.nan)
    Tg[native_idx[:, 0], native_idx[:, 1]] = theta
    mask = ~np.isnan(Tg)
    Tu = unwrap_phase(np.ma.array(np.nan_to_num(Tg), mask=~mask))
    Wg = np.full((H, W), np.nan)
    Wg[mask] = np.asarray(Tu[mask])
    return Wg, r, mask


def windability(r, R, Wg, mask, resid_rms):
    """Gate: does the mesh wind coherently around this axis?"""
    minr_ratio = float(np.percentile(r, 1) / np.median(r))
    resid_over_R = float(resid_rms / R) if R > 0 else np.inf
    # winding monotonicity along the more-circumferential grid axis
    def mono(ax):
        n = Wg.shape[ax]
        vals = []
        for ln in range(0, n, max(1, n // 60)):
            col = Wg[ln, :] if ax == 0 else Wg[:, ln]
            c = col[~np.isnan(col)]
            if c.size < 20:
                continue
            tv = np.abs(np.diff(c)).sum()
            if tv > 0:
                vals.append(abs(c[-1] - c[0]) / tv)
        return float(np.mean(vals)) if vals else 0.0
    winding_mono = max(mono(0), mono(1))
    reasons = []
    if minr_ratio <= GATE_MINR_RATIO:
        reasons.append(f"axis pierces sheet (minr_ratio {minr_ratio:.2f} <= {GATE_MINR_RATIO})")
    if resid_over_R >= GATE_RESID_OVER_R:
        reasons.append(f"not cylindrical about axis (resid/R {resid_over_R:.2f} >= {GATE_RESID_OVER_R})")
    return {"windable": not reasons, "reasons": reasons,
            "minr_ratio": round(minr_ratio, 3),
            "resid_over_R": round(resid_over_R, 3),
            "winding_mono": round(winding_mono, 3)}


def separate_wraps(Wg, mask):
    """Cut the native grid into 2*pi cumulative-winding bands. Returns list of
    per-wrap boolean masks (contiguous sub-grids) + per-wrap info."""
    wv = Wg[mask]
    span_turns = (wv.max() - wv.min()) / (2 * np.pi)
    if span_turns <= SINGLE_WRAP_MAX_TURNS:
        return [mask.copy()], span_turns, "single wrap (<=1 turn +sliver); no split"
    lo = wv.min()
    n_bands = int(np.ceil(span_turns - 1e-9))
    wraps = []
    for b in range(n_bands):
        band = mask & (Wg >= lo + b * 2 * np.pi) & (Wg < lo + (b + 1) * 2 * np.pi)
        if band.sum() > 0:
            wraps.append(band)
    return wraps, span_turns, f"{len(wraps)} wraps by 2*pi winding slicing"


def analyze(pts_mm, native_idx, grid_shape, axis_mode):
    axis0, origin_vx = load_axis()
    origin_mm = origin_vx * VX_UM / 1000
    if axis_mode == "local":
        p0, axis, R, resid = fit_local_cylinder(pts_mm, axis0)
    else:
        p0, axis = origin_mm, axis0
        r, _ = cyl_about(pts_mm, p0, axis)
        R = float(np.median(r))
        resid = float(np.sqrt(np.mean((r - R) ** 2)))
    Wg, r, mask = winding_field(pts_mm, native_idx, grid_shape, p0, axis)
    gate = windability(r, R, Wg, mask, resid)
    out = {"axis_mode": axis_mode, "R_mm": round(R, 2),
           "local_axis_dot_global": round(abs(float(axis @ axis0)), 3),
           "gate": gate,
           "winding_turns": round(float((Wg[mask].max() - Wg[mask].min()) / (2 * np.pi)), 3),
           "r_min_mm": round(float(r.min()), 2), "r_med_mm": round(float(np.median(r)), 2),
           "r_max_mm": round(float(r.max()), 2)}
    if gate["windable"]:
        wraps, span, note = separate_wraps(Wg, mask)
        out["n_wraps"] = len(wraps)
        out["separation_note"] = note
        out["per_wrap"] = [{"n_px": int(w.sum()),
                            "winding_turns": round(float((Wg[w].max() - Wg[w].min()) / (2 * np.pi)), 3),
                            "r_range_mm": [round(float(r[w[native_idx[:, 0], native_idx[:, 1]]].min()), 2),
                                           round(float(r[w[native_idx[:, 0], native_idx[:, 1]]].max()), 2)]}
                           for w in wraps]
    else:
        out["n_wraps"] = None
        out["separation_note"] = "REFUSED: " + "; ".join(gate["reasons"])
    return out


# --------------------------- self test ---------------------------------------

def _load_w043():
    work = "/tmp/reparam_selftest_w043/mesh"
    if not os.path.exists(os.path.join(work, "z.tif")):
        subs, _ = ls("PHerc0139/segments/", delimiter="/")
        seg = next(s for s in subs if "-w043_" in s)
        msubs, _ = ls(f"{seg}mesh/", delimiter="/")
        tdir = next(s for s in msubs if "tifxyz" in s)
        os.makedirs(work, exist_ok=True)
        for fn in ["meta.json", "x.tif", "y.tif", "z.tif"]:
            download(tdir + fn, os.path.join(work, fn))
    d = read_tifxyz(work)
    coords, valid = d["coords"], d["valid"]
    return coords[valid] * VX_UM / 1000, np.argwhere(valid), valid.shape


def _synthetic_spiral(n_turns=2.6, R0=10.0, pitch=0.17, undulation=2.0,
                      H=200, Wc=520, axial_mm=25.0):
    """Clean multi-turn spiral on a native grid: rows=axial, cols=winding.
    pitch (0.17mm/turn) << undulation (2mm) -- exactly the regime where radius
    bands FAIL but winding slicing must succeed."""
    axis = np.array([0.0, 0.0, 1.0])
    u = np.linspace(0, n_turns * 2 * np.pi, Wc)      # winding angle
    v = np.linspace(0, axial_mm, H)                  # axial
    UU, VV = np.meshgrid(u, v)
    r = R0 + pitch * UU / (2 * np.pi) + undulation * np.sin(3.0 * UU)
    x = r * np.cos(UU); y = r * np.sin(UU); z = VV
    pts = np.stack([x, y, z], -1).reshape(-1, 3)
    native_idx = np.array([(i, j) for i in range(H) for j in range(Wc)])
    return pts, native_idx, (H, Wc), axis


def _synthetic_pierced(H=200, Wc=200):
    """Flat sheet the axis stabs through the middle (bullseye) -- models the
    grown-patch failure; the gate MUST refuse this."""
    xs = np.linspace(-15, 15, Wc); ys = np.linspace(-12, 12, H)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = 0.3 * np.sin(XX / 5)  # gentle warp, ~flat
    pts = np.stack([XX, YY, ZZ], -1).reshape(-1, 3)
    native_idx = np.array([(i, j) for i in range(H) for j in range(Wc)])
    return pts, native_idx, (H, Wc)


def _self_test():
    print("=== self-test 1: OFFICIAL w043 (real single wrap) must PASS gate, ~1 wrap ===")
    pts, nidx, shape = _load_w043()
    res = analyze(pts, nidx, shape, "global")
    print(json.dumps(res, indent=1))
    assert res["gate"]["windable"], "w043 must pass the windability gate"
    assert res["n_wraps"] == 1, f"w043 should be ONE wrap, got {res['n_wraps']}"
    assert 0.9 <= res["winding_turns"] <= 1.35, res["winding_turns"]
    print("  [PASS] w043 recognized as one coherent wrap\n")

    print("=== self-test 2: synthetic 2.6-turn spiral (pitch 0.17mm << undulation 2mm) ===")
    print("  (radius bands cannot separate this; winding slicing must give 3 wraps)")
    pts, nidx, shape, ax = _synthetic_spiral()
    axis0, origin = load_axis()  # override axis for the synthetic (z-axis)
    # analyze with an explicit z-axis at origin 0 (bypass load_axis for synthetic)
    p0 = np.zeros(3)
    Wg, r, mask = winding_field(pts, nidx, shape, p0, ax)
    resid = float(np.sqrt(np.mean((r - np.median(r)) ** 2)))
    gate = windability(r, np.median(r), Wg, mask, resid)
    wraps, span, note = separate_wraps(Wg, mask)
    print(f"  gate={gate}")
    print(f"  winding span={span:.2f} turns -> {note}")
    for i, w in enumerate(wraps):
        rr = r[w[nidx[:, 0], nidx[:, 1]]]
        tt = (Wg[w].max() - Wg[w].min()) / (2 * np.pi)
        contig = _is_contiguous(w)
        print(f"    wrap {i}: n_px={w.sum()} winding={tt:.2f}t r={rr.min():.1f}-{rr.max():.1f}mm contiguous={contig}")
    assert gate["windable"], "clean spiral must pass gate"
    assert len(wraps) == 3, f"2.6-turn spiral should slice into 3 wraps, got {len(wraps)}"
    # each full wrap ~2pi and radius ranges OVERLAP across wraps (undulation>>pitch)
    rr0 = r[wraps[0][nidx[:, 0], nidx[:, 1]]]; rr1 = r[wraps[1][nidx[:, 0], nidx[:, 1]]]
    assert rr0.max() > rr1.min(), "radius ranges must OVERLAP (proves radius can't separate)"
    print("  [PASS] spiral sliced into 3 wraps by winding; radius ranges overlap "
          "(confirming winding, not radius, is the separating coordinate)\n")

    print("=== self-test 3: synthetic axis-pierced flat sheet -- gate MUST refuse ===")
    pts, nidx, shape = _synthetic_pierced()
    p0 = np.zeros(3); ax = np.array([0.0, 0.0, 1.0])
    Wg, r, mask = winding_field(pts, nidx, shape, p0, ax)
    resid = float(np.sqrt(np.mean((r - np.median(r)) ** 2)))
    gate = windability(r, np.median(r), Wg, mask, resid)
    print(f"  gate={gate}")
    assert not gate["windable"], "pierced flat sheet MUST fail the gate"
    print("  [PASS] gate correctly refuses an axis-pierced sheet "
          "(the exact failure mode of the grown patches vs the global axis)\n")
    print("ALL SELF-TESTS PASSED")


def _is_contiguous(m):
    from scipy import ndimage
    _, n = ndimage.label(m, structure=np.ones((3, 3)))
    return n == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mesh-dir")
    ap.add_argument("--frame", choices=["native_2399um", "registered_2399um_from_9362um"],
                    default="native_2399um")
    ap.add_argument("--axis", choices=["global", "local"], default="global")
    args = ap.parse_args()
    if args.self_test:
        _self_test(); return
    if not args.mesh_dir:
        ap.error("--mesh-dir required unless --self-test")
    d = read_tifxyz(args.mesh_dir)
    coords, valid = d["coords"], d["valid"]
    pts = coords[valid]
    if args.frame == "registered_2399um_from_9362um":
        t = json.load(open("/tmp/vol_transform.json_20260102.json"))
        M = np.array(t["transformation_matrix"]); R, T = M[:, :3], M[:, 3]
        pts = (pts - T) @ np.linalg.inv(R).T
    res = analyze(pts * VX_UM / 1000, np.argwhere(valid), valid.shape, args.axis)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
