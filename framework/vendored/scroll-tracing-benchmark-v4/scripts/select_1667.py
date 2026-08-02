"""PHerc1667 instance, step 4 of docs/PHERC1667_INSTANCE.md: full window
selection over the evaluation zone (col-starts >= 2699, disjoint from the
[199,1999] tuning zone scripts/tune_1667.py used), producing
docs/evidence/windows_1667.json in exactly fixtures/windows_v2.json's
schema (spec, date, tuned, kappa_median_eligible, n_candidates,
n_eligible, windows, all_candidates, windows_initial_by_rule,
gate_c_attempt_log).

Needs network (gate c's CT spacing per candidate tried); NOT part of the
offline test suite. TUNED_SIGMA_1667/TUNED_PROM_1667 below are the
decision scripts/tune_1667.py wrote to docs/evidence/v1667_tuning.log
(frozen here the same way reference_src/v2_pipeline.py freezes
TUNED_SIGMA/TUNED_PROM with a provenance comment).

Run: python scripts/select_1667.py
"""
import dataclasses
import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stb import band as stb_band  # noqa: E402
from stb import config as stb_config  # noqa: E402
from stb import estimator as estimator_mod  # noqa: E402
from stb import gates as gates_mod  # noqa: E402
from stb import normals as stb_normals  # noqa: E402
from stb import reference as reference_mod  # noqa: E402
from stb import selection as stb_selection  # noqa: E402
from tune_1667 import open_volume  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
EVIDENCE = REPO_ROOT / "docs" / "evidence"

def load_frozen_tuning():
    path = EVIDENCE / "v1667_tuning.json"
    if not path.exists():
        raise RuntimeError(
            "PHerc1667 tuning is not frozen. Run scripts/tune_1667.py first; "
            "selection is intentionally blocked rather than using placeholder values."
        )
    record = json.loads(path.read_text())
    decision = record.get("decision", {})
    sigma = decision.get("sigma")
    prominence = decision.get("prominence_frac")
    if sigma is None or prominence is None:
        raise RuntimeError(f"invalid frozen tuning record: {path}")
    return float(sigma), float(prominence), record


def main():
    tuned_sigma, tuned_prom, tuning_record = load_frozen_tuning()
    cfg = stb_config.load_config(CONFIGS / "pherc1667.json")
    xyz, valid, _row0 = stb_band.load_band(cfg.band_path)
    cfg = stb_config.resolve(cfg, xyz, valid)
    W = valid.shape[1]

    tune_lo = round(0.05 * W)
    tune_hi = tune_lo + 2000
    exclusion = (tune_lo - 500, tune_hi + 500)
    cfg = dataclasses.replace(cfg, exclusions=(exclusion,))
    print(f"band shape={xyz.shape} fitted_center=({cfg.center[0]:.4f}, {cfg.center[1]:.4f})")
    print(f"exclusion (tuning-zone buffer): {exclusion}")

    normals_band, n_ok = stb_normals.band_normals(xyz, valid)
    kappa = stb_normals.kappa_per_column(normals_band, n_ok)

    rows = stb_selection.scan_candidates(xyz, valid, cfg, kappa)
    all_candidates = stb_selection.all_candidates_rows(rows)
    n_eligible = len(stb_selection._ok_rows(rows))
    print(f"n_candidates(eligible starts, post-exclusion)={len(rows)} n_eligible(pass cov+gates)={n_eligible}")

    picked, kappa_median = stb_selection.stratified_pick(rows)
    print(f"kappa_median_eligible={kappa_median}")
    print(f"initial stratified pick: {[(p['stratum'], p['start']) for p in picked]}")

    halfwidth = estimator_mod.halfwidth_vox_from_physical(cfg.vox_um)
    volume = open_volume(cfg.volume_url)
    print(f"profile_halfwidth_vox={halfwidth} sigma={tuned_sigma} prom={tuned_prom}")

    gate_c_cache = {}
    gate_c_cells = {}

    def gate_c_inputs(start):
        if start in gate_c_cache:
            return gate_c_cache[start]
        ref = reference_mod.reference_at(xyz, valid, start, cfg)
        g = gates_mod.coverage_and_gates_ab(ref, cfg)
        kd_gap = g["gap_median_front"]
        est = estimator_mod.estimator_p2(
            volume, xyz, normals_band, start, cfg,
            sigma=tuned_sigma, prom_frac=tuned_prom,
            profile_halfwidth_vox=halfwidth,
        )
        result = (est["p2"], kd_gap)
        gate_c_cache[start] = result
        gate_c_cells[start] = est
        print(f"  gate_c_inputs(start={start}): p2_ct={est['p2']!r} kd_gap_median={kd_gap!r}")
        return result

    # windows_initial_by_rule: the raw stratified pick with gate c EVALUATED
    # (informational) but NOT yet replaced -- same shape as
    # fixtures/windows_v2.json's own field.
    windows_initial_by_rule = []
    for slot in picked:
        ct, kd = gate_c_inputs(slot["start"])
        gc = gates_mod.gate_c(ct, kd)
        windows_initial_by_rule.append(dict(slot, **gc))

    accepted, attempt_log = stb_selection.apply_gate_c(picked, rows, gate_c_inputs)
    print(f"\nfinal accepted windows: {[(w['stratum'], w['start'], w['gate_c_pass']) for w in accepted]}")

    strata_seen = {p["stratum"] for p in picked}
    strata_accepted = {w["stratum"] for w in accepted}
    for s in ("low", "median", "high"):
        n_slots = sum(1 for p in picked if p["stratum"] == s)
        n_ok = sum(1 for w in accepted if w["stratum"] == s)
        print(f"stratum {s}: {n_ok}/{n_slots} slots filled after gate c")

    out = {
        "spec": "docs/PHERC1667_INSTANCE.md (d)",
        "date": datetime.date.today().isoformat(),
        "tuned": {"sigma": tuned_sigma, "prominence_frac": tuned_prom},
        "tuning_record": tuning_record,
        "kappa_median_eligible": kappa_median,
        "n_candidates": len(rows),
        "n_eligible": n_eligible,
        "windows": accepted,
        "all_candidates": all_candidates,
        "windows_initial_by_rule": windows_initial_by_rule,
        "gate_c_attempt_log": attempt_log,
    }

    def _default(o):
        import numpy as np

        if isinstance(o, (np.bool_, np.integer)):
            return o.item()
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"not JSON serializable: {type(o)}")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cell_arrays = {}
    for start, est in gate_c_cells.items():
        for key in ("raw", "spacings", "rr", "cc"):
            cell_arrays[f"s{start}_{key}"] = est[key]
    cells_path = EVIDENCE / "windows_1667_cells.npz"
    np.savez_compressed(cells_path, **cell_arrays)
    out["cell_fixture"] = cells_path.name
    out_path = EVIDENCE / "windows_1667.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=_default)
    print(f"\nwrote {out_path}")
    print(f"wrote {cells_path}")
    return out


if __name__ == "__main__":
    main()
