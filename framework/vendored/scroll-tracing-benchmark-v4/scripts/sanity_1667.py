"""Port-fidelity sanity check (PHerc1667 instance, step 2 of
docs/PHERC1667_INSTANCE.md): run stb's ported gates a/b + coverage over
the real band at the same stride=200/window=200 Agent C used for the
16/19 demo in docs/PHERC1667_SCOUT.md, and diff row-by-row against
`data/pherc1667/gates_report_stride200.json` (Agent C's independent
reference_src-based run). This is a regression check on stb's port, not
a new scientific claim -- same band, same math, two code paths.

Offline except for loading the already-built band npz (no zarr/network
access; gates a/b are prediction-free geometry+KD-tree checks).

Run: python scripts/sanity_1667.py
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data" / "pherc1667"

import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))

from stb import band as stb_band  # noqa: E402
from stb import config as stb_config  # noqa: E402
from stb import gates as gates_mod  # noqa: E402
from stb import normals as stb_normals  # noqa: E402
from stb import reference as reference_mod  # noqa: E402

STRIDE, WIN = 200, 200


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    band_path = DATA / "band_w013_r1400_1600_xyz.npz"
    digest = sha256(band_path)
    print(f"SHA-256({band_path.relative_to(REPO_ROOT)}) = {digest}")

    cfg = stb_config.load_config(CONFIGS / "pherc1667.json")
    xyz, valid, row0 = stb_band.load_band(cfg.band_path)
    cfg = stb_config.resolve(cfg, xyz, valid)
    print(f"band shape={xyz.shape} valid_frac={valid.mean():.4f} row0={row0} "
          f"fitted_center=({cfg.center[0]:.4f}, {cfg.center[1]:.4f})")

    normals_band, n_ok = stb_normals.band_normals(xyz, valid)
    kappa = stb_normals.kappa_per_column(normals_band, n_ok)

    W = valid.shape[1]
    rows = []
    for s in range(0, W - WIN + 1, STRIDE):
        ref = reference_mod.reference_at(xyz, valid, s, cfg)
        if not all(k in ref.trees for k in (+1, -1, 0)):
            rows.append({"start": s, "coverage": 0.0, "error": "missing class"})
            continue
        g = gates_mod.coverage_and_gates_ab(ref, cfg)
        g["start"] = s
        g["kappa"] = float(np.nanmedian(kappa[s:s + WIN]))
        rows.append(g)

    with open(DATA / "gates_report_stride200.json") as f:
        agent_c = json.load(f)
    want_rows = {r["start"]: r for r in agent_c["windows"]}

    n_pass = sum(
        1 for r in rows
        if not r.get("error") and r["coverage"] >= 0.40
        and r.get("gate_a_pass") and r.get("gate_b_pass")
    )
    print(f"\nstb port: {n_pass}/{len(rows)} windows pass coverage>=0.40 + gate_a + gate_b")
    print(f"Agent C : {agent_c['n_pass']}/{agent_c['n_windows']} (from gates_report_stride200.json)")

    mismatches = []
    for r in rows:
        s = r["start"]
        w = want_rows.get(s)
        if w is None:
            mismatches.append(f"start={s}: no matching row in Agent C's report")
            continue
        if bool(r.get("error")) != bool(w.get("error")):
            mismatches.append(f"start={s}: error flag differs (got {r.get('error')!r}, want {w.get('error')!r})")
            continue
        if r.get("error"):
            continue
        for key, tol in (("coverage", 1e-3), ("gap_median_front", 1.0), ("gap_median_back", 1.0),
                         ("selftest_median", 1e-6), ("selftest_p90", 1e-6), ("kappa", 1e-3)):
            gv, wv = r.get(key), w.get(key)
            if gv is None or wv is None:
                continue
            if abs(float(gv) - float(wv)) > tol:
                mismatches.append(f"start={s} {key}: got {gv} want {wv} (tol {tol})")
        for key in ("gate_a_pass", "gate_b_pass"):
            if r.get(key) != w.get(key):
                mismatches.append(f"start={s} {key}: got {r.get(key)} want {w.get(key)}")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES vs Agent C's report:")
        for m in mismatches[:30]:
            print(f"  {m}")
        print("\nSTOP: see PLAN_V3.md's non-negotiable rule -- write BLOCKERS.md.")
    else:
        print("\nOK: stb port reproduces Agent C's stride-200 gates report row-by-row "
              "(coverage/gap medians within tolerance, gate_a/gate_b flags identical, "
              "kappa within tolerance).")
    return rows, mismatches


if __name__ == "__main__":
    main()
