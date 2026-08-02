# Scroll Tracing Benchmark (v4)

A leakage-aware, multi-pipeline evaluation kit for surface tracing in
compressed Herculaneum scroll regions. This packages the machinery behind
the neural-tracing-audit v1/v2 benchmarks (63.2% -> 5.9% unit-fix finding;
the v2 "no directional skill beyond a coherent local normal" result) into
a reusable library, per-scroll configs, qualified reference strips, and a
pipeline-agnostic scoring contract.

## What is in the box

- `stb/` — the library: windowed multi-wrap reference construction,
  prediction-free validity gates (self-test, wrong-side, CT cross-check),
  CT pitch estimators, curvature-stratified window selection, arms A-D
  scoring with saturation classification (permutation nulls), and the
  candidate contract (NaN row = abstain, abstains reported and excluded
  from the denominator).
- `adapters/` — `tifxyz_tracer.py` (displacement-tracer outputs) and
  `mesh_adapter.py` (any mesh, via normal ray-casting; oracle-mesh
  self-test scores 100%).
- `strips/pherc0332_pack/` — 5 qualified strips (the v1 window + the 4
  frozen v2 windows), each re-qualifiable from the .npz alone; a
  label-shuffled strip fails the gates (the suite guards its own guards).
- `configs/` — `pherc0332.json` (the validated instance) and
  `pherc1667.json` (scouted second instance: segment w013, 5.6 winding
  revolutions, 16/19 stride-200 windows pass gates; band recipe +
  `stb/build_band.py`; see `docs/PHERC1667_SCOUT.md`).
- `tests/` — every port is pinned to the published v2 numbers: the full
  217-candidate selection reproduces `windows_v2.json` at 1e-6 (11-min
  slow test), the arms reproduce the published per-window scores (34/40
  at exactly 0.000000 pp; one documented 0.0108 pp gap is a strict xfail
  whose root cause is recorded in BLOCKERS.md), and
  E1 = -55.2 pp reproduces within 0.002 pp.
- V4 is additive: fixed-denominator yields, autoregressive same-surface
  traces, leakage guards, top-K proposals, risk-coverage, cluster bootstrap,
  physical-scale helpers and safer CT boundary handling. See `docs/V4_SPEC.md`.

## Quick start

```bash
python -m pip install -e '.[all]'
python fetch_fixtures.py            # one-time: rebuilds the 48 MB band bit-exactly
                                    # from public data (SHA-256 verified); the
                                    # frozen GPU predictions ship committed
python -m pytest tests/ -q          # add -m \"not slow\" to skip the 11-min pin
```

To score your own tracer on the 332 pack: produce predictions for each
strip's seed cells (see `stb/contract.py`; NaN = abstain), then
`score_candidate`. Meshes: see `adapters/mesh_adapter.py`.

V4 promotion uses `coverage_pct`, fixed-denominator yields and
`coverage_gate`; retained-set accuracy alone is not a valid headline.
Same-surface tracing uses `stb.sequential`, not the adjacent-hop direction.

## Provenance and honesty rules

Built by porting, never rewriting: `reference_src/` contains the exact
v1/v2 sources this library must agree with, and the tests enforce the
agreement numerically. Known gaps live in `BLOCKERS.md`, not in weakened
tolerances. The v2 findings themselves (and their caveats) are in the
neural-tracing-audit repo: https://github.com/LimeGS/neural-tracing-audit

## Limits

A local geometric reference, not manual ground truth: it only grades where the
source segment recorded the neighboring wrap, and the CT cross-check gate
exists precisely because high-curvature terrain fails it (8/8 candidates
in the 332 band) — windows that fail gates are refused, not graded.

PHerc1667 remains a candidate instance until CT tuning, Gate C and its freeze
block are committed. V4 exposes that incomplete state instead of fabricating
network-derived evidence.

## License

MIT (code). Scroll data referenced is Vesuvius Challenge CC BY-NC 4.0.
