# Scroll Tracing Benchmark v3 — build plan (executable spec for subagents)

Goal: turn the v1/v2 benchmark machinery (in `reference_src/`, read-only
references) into a **multi-scroll, multi-pipeline evaluation kit**: a small
python package (`stb/`), per-scroll configs, qualified strip packs, a
pipeline-agnostic candidate contract with adapters, and regression tests
that pin every port to the published numbers.

NON-NEGOTIABLE RULES for every agent:
- Work ONLY inside this repo. Never modify `reference_src/` or `fixtures/`
  or anything outside this directory. Never push. Never create remotes.
- Commit locally with clear messages as you complete each numbered item.
- Acceptance = the listed pytest tests pass (`python -m pytest tests/ -x -q`).
  Do not weaken a test to make it pass; if a port cannot reproduce a pinned
  number exactly, STOP and write the discrepancy to `BLOCKERS.md`.
- Offline by default: tests must not need network. Anything requiring the
  public zarr goes behind `STB_NETWORK=1` and is excluded from acceptance.
- Style: match reference_src (numpy, plain functions, tight docstrings that
  state constraints, no dead code, no placeholder TODOs).

## Architecture

```
stb/core.py        Reference dataclass + score_prediction + summarize
                   (port of reference_src/benchmark_core.py, but VOX_UM,
                   CLASSES, STEP, threshold come from a ScrollConfig)
stb/config.py      ScrollConfig dataclass + load from configs/*.json:
                   {scroll_id, volume_url, vox_um, center: [cx,cy] | "fit",
                    band: {segment, path}, classes, step}
stb/band.py        load_band(path) + fit_center(xyz, valid): circle fit on
                   the row-100 valid points (least squares); must reproduce
                   the hardcoded PHerc0332 center within 2 vox (test).
stb/reference.py   reference_at(xyz, valid, col_start, cfg) — port of
                   reference_src/v2_pipeline.reference_at with cfg center.
stb/normals.py     band_normals + kappa_per_column (straight port).
stb/gates.py       coverage_and_gates_ab (port), gate_c(ct_spacing, kd_gap)
                   ratio check in [0.60, 1.60]; every gate returns a dict
                   with pass flags + raw numbers (same keys as v2).
stb/estimator.py   profiles/spacing/P2/P1 (port; tuned defaults sigma=0.5,
                   prom=0.15 exposed as constants with provenance comment);
                   sample_profiles takes an injectable `volume` object so
                   tests can pass synthetic arrays (no network).
stb/selection.py   eligible_starts(cfg-driven exclusions) + stratified
                   selection with >=300 separation + documented replacement
                   chain (port of cmd_select + the gate-c replacement loop;
                   gate-c values injectable for offline tests).
stb/arms.py        arms A (x train_um/vox_um), B, C, D + permutation null
                   (rng seed param, default 0) + E1/G1 summary — port of
                   reference_src/v2_score.py score_window/main WITHOUT the
                   CLI wrapper; frozen denominator = intersection of arms'
                   ok masks (identical semantics).
stb/contract.py    THE pipeline-agnostic interface:
                     WindowTask: {scroll_id, col_start, seed_points (n,3),
                       normals (n,3), direction (+1|-1), gap_hint_median}
                     Prediction: array (n,3) float, NaN row = abstain.
                   score_candidate(ref, task, prediction, cfg) -> summary
                   (abstains excluded from the denominator and counted).
stb/strips.py      export_strip(ref, window meta, gates, cfg) -> .npz with
                   {xyz window + classes -1/0/+1 point sets local to the
                   window neighborhood, pitch, gates dict, provenance};
                   load_strip + qualify_strip(strip) re-running gates a/b.
adapters/tifxyz_tracer.py   wraps a front/back tifxyz dir pair into
                   Predictions for the contract (the v2 path).
adapters/mesh_adapter.py    trimesh-based: for each seed cell, ray-cast
                   along +/- normal; first hit with distance in
                   (0.25*gap_hint, 2.5*gap_hint) -> prediction; else NaN
                   (abstain). Pure geometry, no learning.
configs/pherc0332.json      our band (fixtures/band_r1145_200_xyz.npz),
                   vox_um 7.91, center "fit" (must match hardcode), volume
                   URL as in reference_src.
```

## Pinned regression numbers (acceptance tests)

- test_regression_332.py, pinned on the v2 published run:
  (a) selection: stb.selection over the band must produce candidate list
      whose (start, kappa, coverage, gate_a, gate_b) rows match
      fixtures/windows_v2.json `all_candidates` (kappa/coverage to 1e-6).
  (b) arms: scoring fixtures/out_v2 seed_v2_s11000/{front,back} etc. with
      stb.arms must reproduce fixtures/v2_scores_20260711.json per-window
      correct_pct for arms A, B_p2, C_p2, D_9vox and null_perm to 0.01 pp,
      injecting p2 values RECORDED in that json (so no network).
  (c) center fit: |fit - (1809.26609333, 1732.01937758)| < 2 vox.
  (d) E1 summary from (b) equals -55.2038 +/- 0.01 with the same
      discriminating-unit classification.
- test_contract.py: tifxyz adapter re-scores s11000 front IDENTICALLY to
  direct arms path; mesh_adapter on a synthetic oracle mesh built from the
  band's own class +1 points near s11000 scores >= 95% correct front;
  abstain accounting exercised (a mesh clipped to half the window must
  yield ~half abstains, and the denominator shrinks accordingly).
- test_strips.py: export the 4 v2 windows + the v1 window (12750) as
  strips; load them back; qualify_strip passes gates a/b for all 5; a
  deliberately label-shuffled strip FAILS gate b (regression guard).
- test_estimator.py: synthetic profile with known 10-vox peak spacing
  returns 10.0 +/- 0.2 at tuned params; <2 peaks -> NaN abstain; P1
  fallback fraction computed on a crafted 9x9 grid.

## Task split

AGENT A (port core): stb/config.py, band.py, core.py, reference.py,
normals.py, estimator.py, gates.py, selection.py + configs/pherc0332.json +
tests (a)(c) + test_estimator. Read reference_src for exact semantics.
Deliverable: those tests green offline, committed.

AGENT B (scoring + contract + strips, AFTER A): stb/arms.py, contract.py,
strips.py, adapters/* + tests (b)(d), test_contract, test_strips +
strips/pherc0332_pack/ exported (5 strips + PACK.md table of gates).
Deliverable: full pytest green, committed.

AGENT C (parallel with B): PHerc1667 instance scouting. Using the public
bucket (network OK for exploration, NOT for tests): find segments of
PHerc1667 with >= 2 winding revolutions (compute theta span from their
tifxyz x/y at coarse stride around a fitted center; the volume URL and
vox_um from the bucket layout/meta). Deliverable: configs/pherc1667.json
(band left as a recipe: which segment + row range + how to build),
docs/PHERC1667_SCOUT.md with the evidence (segments examined, theta spans,
chosen candidate, µm audit), and if a band builds cleanly under 30 min,
fixtures-free band build script `stb/build_band.py` (generic: tifxyz ->
band npz) demonstrated on it with gates a/b + coverage table for stride-200
windows. NO benchmark claims, NO GPU. If nothing qualifies, that IS the
deliverable (documented).

FINAL (Fable, not agents): review, docs/README/SPEC by the lead, packaging
commit.
