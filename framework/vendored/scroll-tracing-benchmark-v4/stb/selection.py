"""Column-start eligibility, per-window gates and stratified window
selection (port of reference_src/v2_pipeline.eligible_starts + cmd_select).

cmd_select's single function is split into composable pieces here:
  eligible_starts    the stride grid minus cfg.exclusions
  scan_candidates    per-start coverage + gates a/b + median kappa
                     (offline: no CT/zarr access)
  all_candidates_rows the (start, kappa, coverage, gate_a, gate_b) view
                     pinned by fixtures/windows_v2.json's `all_candidates`
  stratified_pick    cmd_select's low/median/high kappa strata, >=300 apart
  apply_gate_c       the gate-c replacement loop, with gate-c's CT/KD
                     inputs injected so it too can run offline

stratified_pick + apply_gate_c reconstruct the window-selection step that
produced windows_v2.json's `windows` + `gate_c_attempt_log` (cmd_select, as
given in reference_src, predates gate c entirely -- it has no CT access).
That reconstruction is NOT pinned by a regression test (see BLOCKERS.md);
it is provided because the architecture calls for it and downstream
scrolls need a selection path, not because its exact attempt-log shape is
verified against the published run.
"""
import numpy as np

from . import gates
from . import reference as reference_mod


def eligible_starts(width, cfg):
    """Column starts on the cfg.stride grid whose whole window clears
    cfg.exclusions."""
    starts = []
    for s in range(0, width - cfg.window + 1, cfg.stride):
        window = (s, s + cfg.window)
        if any(not (window[1] <= lo or window[0] >= hi) for lo, hi in cfg.exclusions):
            continue
        starts.append(s)
    return starts


def scan_candidates(xyz, valid, cfg, kappa):
    """Coverage + gates a/b + median kappa for every eligible start."""
    rows = []
    for s in eligible_starts(valid.shape[1], cfg):
        ref = reference_mod.reference_at(xyz, valid, s, cfg)
        g = (gates.coverage_and_gates_ab(ref, cfg)
             if all(k in ref.trees for k in (+1, -1, 0))
             else {"coverage": 0.0, "error": "missing class"})
        window_kappa = kappa[s:s + cfg.window]
        g["kappa"] = (
            float(np.nanmedian(window_kappa))
            if np.isfinite(window_kappa).any() else float("nan")
        )
        g["start"] = s
        rows.append(g)
    return rows


def all_candidates_rows(rows):
    """The (start, kappa, coverage, gate_a, gate_b) view pinned by
    fixtures/windows_v2.json's `all_candidates`."""
    return [{"start": r["start"], "kappa": r.get("kappa"),
             "coverage": r.get("coverage"),
             "gate_a": r.get("gate_a_pass"), "gate_b": r.get("gate_b_pass")}
            for r in rows]


def _ok_rows(rows):
    return [r for r in rows if r["coverage"] >= 0.40 and r.get("gate_a_pass")
            and r.get("gate_b_pass") and np.isfinite(r["kappa"])]


def stratified_pick(rows, min_sep=300):
    """2 low-kappa + 2 median-kappa + 2 high-kappa windows, >=min_sep
    columns apart, each passing coverage>=0.40 + gates a/b."""
    ok = _ok_rows(rows)
    if not ok:
        return [], float("nan")
    ok_by_kappa = sorted(ok, key=lambda r: r["kappa"])
    med = float(np.median([r["kappa"] for r in ok]))

    def try_pick(cands, picked):
        for r in cands:
            if all(abs(r["start"] - p["start"]) >= min_sep for p in picked):
                return r
        return None

    picked = []
    strata = [("low", ok_by_kappa), ("low", ok_by_kappa),
              ("median", sorted(ok, key=lambda r: abs(r["kappa"] - med))),
              ("median", sorted(ok, key=lambda r: abs(r["kappa"] - med))),
              ("high", ok_by_kappa[::-1]), ("high", ok_by_kappa[::-1])]
    for name, order in strata:
        r = try_pick([c for c in order if c not in picked], picked)
        if r is not None:
            picked.append(dict(r, stratum=name))
    return picked, med


def apply_gate_c(picked, rows, gate_c_inputs, min_sep=300, max_attempts=8):
    """Replace each stratified pick that fails gate c with the next
    same-stratum candidate (by that stratum's own kappa ordering), >=min_sep
    from every already-accepted window; give up on a slot (dropping it)
    after max_attempts failures. Returns (accepted_windows, attempt_log).

    gate_c_inputs(start) -> (ct_spacing, kd_gap) is injected so this runs
    offline against recorded CT/KD values; no zarr access happens here.
    """
    ok = _ok_rows(rows)
    ok_by_kappa = sorted(ok, key=lambda r: r["kappa"])
    med = float(np.median([r["kappa"] for r in ok])) if ok else float("nan")
    orders = {
        "low": ok_by_kappa,
        "median": sorted(ok, key=lambda r: abs(r["kappa"] - med)),
        "high": ok_by_kappa[::-1],
    }

    accepted, log = [], []
    for slot in picked:
        stratum = slot["stratum"]
        taken = {a["start"] for a in accepted}
        candidates = [c for c in orders[stratum] if c["start"] not in taken]
        chosen = None
        attempts = 0
        for cand in candidates:
            if attempts >= max_attempts:
                break
            if not all(abs(cand["start"] - a["start"]) >= min_sep for a in accepted):
                continue
            attempts += 1
            ct_spacing, kd_gap = gate_c_inputs(cand["start"])
            gc = gates.gate_c(ct_spacing, kd_gap)
            log.append({"stratum": stratum, "start": cand["start"],
                        "gate_c_ratio": round(gc["gate_c_ratio"], 3),
                        "pass": gc["gate_c_pass"]})
            if gc["gate_c_pass"]:
                chosen = dict(cand, stratum=stratum, **gc)
                break
        if chosen is None:
            log.append({"stratum": stratum, "start": None, "pass": False,
                        "note": f"stratum exhausted (first {max_attempts} candidates fail gate c)"})
        else:
            accepted.append(chosen)
    return accepted, log
