# BLOCKERS

## Agent A: `fit_center` does not reproduce the hardcoded PHerc0332 center within 2 vox

**Status:** RESOLVED by the lead (2026-07-11): the 2-vox geometric pin was a plan error (a band row is a spiral, not a circle). Replaced by a FUNCTIONAL pin — test_fit_center_is_functionally_adequate_for_classing — which PASSES: a fitted center 189 vox from the historical constant still yields identical-quality gates and coverage within 0.05 at window 11000. Configs may carry literal centers when known; fit_center is the validated primitive for new scrolls.

**Pinned claim (PLAN_V3.md, Architecture / stb/band.py):**
> `fit_center(xyz, valid)`: circle fit on the row-100 valid points (least
> squares); must reproduce the hardcoded PHerc0332 center within 2 vox (test).

**Test:** `tests/test_regression_332.py::test_fit_center_reproduces_hardcoded_332_center`
asserts `|fit_center(band, row=100) - (1809.26609333, 1732.01937758)| < 2`
vox. It currently fails:

```
fit_center=(1620.54588, 1728.66488) vs hardcoded (1809.26609333, 1732.01937758); dist=188.750 vox
```

**What was tried** (all on `fixtures/band_r1145_200_xyz.npz`, row 100, which
has a single contiguous run of 25371 valid columns out of 25706 — no stray
outlier islands to blame):

1. **Kasa algebraic circle fit** (closed-form linear least squares on
   `x^2+y^2+Dx+Ey+F=0`) on the full row: center `(1659.4, 1740.5)`,
   dist ≈ 150.
2. **Geometric (nonlinear) circle fit** (Gauss-Newton on the radial
   residual, Kasa-seeded) on the full row: center `(1620.5, 1728.7)`,
   dist ≈ 189. This is what `stb.band.fit_center` implements (the
   standard "circle fit via least squares" recipe).
3. Both (1) and (2) repeated on: the original v1 seed window
   (cols 12750:12950, 200 pts) → dist ≈ 1529-1554; cols 0:12950 →
   dist ≈ 191-229; the tuning zone cols 2000:4000 → dist ≈ 1494-1615;
   cols 0:2000 → dist ≈ 52-340; a sliding 200-col window scanned across
   the whole band at stride 200 (136 windows) → best dist ≈ 80, most
   windows much worse (hundreds to thousands of vox).
4. **Per-column local circumcenter** (3-point circle through columns
   `c-lag, c, c+lag`, lag in {1,2,5,10,20,50,100}), median/mean over all
   columns: best (lag=1, median) dist ≈ 103; mean is far worse (the
   3-point estimator is not robust — a handful of near-collinear triples
   blow up).
5. **Archimedean spiral fit** `r(theta) = a + b*theta_unwrapped` with
   `(cx, cy, a, b)` jointly optimized (Levenberg-Marquardt): dist ≈ 188.
6. **Iterative self-consistent "class 0" bootstrap**: guess a center,
   unwrap row 100 the same way `reference_at` does, take only the points
   within ±0.5 winding of column `(C0+C1)//2 = 12850` (i.e. `cls == 0`),
   refit a circle to just those, repeat to a fixed point. Converges in 2
   iterations to a stable `(1651.6, 1737.4)`, dist ≈ 158 — a genuine
   fixed point, not just non-convergence, and still nowhere near 2 vox.

**Why:** with the *hardcoded* center, the radial distance of row-100 points
ranges from ~715 to ~1576 vox and is spread fairly continuously across
that whole range (checked via histogram) — i.e. across the band's full
25706-column width, row 100 traces something close to an Archimedean
spiral through ~3 revolutions (checked by unwrapping with the hardcoded
center: unwrapped range spans ~3.03 x 2π), not a circle. Even segmenting
into individual revolutions *using the true center* and re-fitting a
circle to just one revolution's points still lands ~140-160 vox off,
so the bias isn't "wrong arc, right circle" — a circle genuinely is not
the right local model for this real, deformed surface at the precision
this test demands, in any windowing tried.

**Hypothesis:** the hardcoded `CX, CY` most likely comes from a fit against
a different, unavailable dataset (e.g. a full scroll segmentation / CT
cross-section, or the original narrower v1 band before it was extended to
this wide v2 scan band), or from a fundamentally different method than a
row-wise circle fit. Nothing in `reference_src/` (`benchmark_core.py`,
`v2_pipeline.py`, `v2_score.py`, `v2_power.py`) computes it — it is only
ever consumed as a hardcoded constant — and git history (2 commits) has no
earlier version that derives it.

**What this does *not* affect:** `configs/pherc0332.json` sets `"center"`
to the literal hardcoded pair `[1809.26609333, 1732.01937758]`, **not**
the string `"fit"` (a deliberate deviation from PLAN_V3.md's illustrative
`center: "fit" (must match hardcode)`, documented here and in the config's
neighborhood in code review). `stb.config.resolve()` is a no-op when
`center` is already literal, so `reference_at`/`selection` use the exact
same `CX, CY` as `reference_src`, and
`test_all_candidates_match_windows_v2` (part a) is unaffected — confirmed
by an exact match against `fixtures/windows_v2.json` on a spot-check
(start=11000: gap_median_front/back, coverage, gate_a/b, kappa all matched
to full float64 precision before the full 217-candidate run was kicked
off).

**`stb.band.fit_center` is still implemented and shipped** as the
standard, defensible geometric circle fit (Kasa-seeded Gauss-Newton) — it
is the right general-purpose primitive for scrolls that have no hardcoded
center to fall back on (e.g. Agent C's PHerc1667 scouting), and is
directly reused by `stb.config.resolve()` for any config with
`"center": "fit"`. It is simply not the derivation of PHerc0332's specific
constant.

**Recommendation for the lead:** either (a) confirm/track down the actual
original derivation of `CX, CY` so `fit_center` can be corrected to match
it, or (b) accept that PHerc0332's config should carry the literal center
permanently (as done here) and treat `fit_center` as a general fallback
whose accuracy is validated per-scroll rather than pinned to this
particular historical constant.

---

## Agent A: `stb.selection`'s gate-c replacement chain is a documented reconstruction, not a verified port

Not a test failure (no test pins it for Agent A), flagging for visibility.
**Update: manually walked against the fixture and it reproduces the
published outcome exactly** (see below) — downgraded from "unverified" to
"outcome-verified, one logging cosmetic unverified."

`fixtures/windows_v2.json` contains `windows_initial_by_rule`, `windows`
(post gate-c) and `gate_c_attempt_log` fields that no script in
`reference_src/` computes — `v2_pipeline.py`'s `cmd_select` (as given)
predates gate c and produces only the equivalent of
`windows_initial_by_rule` (confirmed: its 6-slot low/low/median/median/
high/high stratified pick, ignoring gate c entirely, matches that field).
Whatever script layered gate c + the replacement loop on top and wrote the
final `windows`/`gate_c_attempt_log` is not among the 4 files in
`reference_src/`.

`stb.selection.apply_gate_c` reconstructs a version (try up to 8
candidates per stratum slot in that stratum's own kappa ordering, `>=300`
columns from every already-accepted window, log every attempt, drop the
slot if all attempts fail). Manually driving `stb.selection.stratified_pick`
+ `apply_gate_c` against `fixtures/windows_v2.json`'s own `all_candidates`
rows and per-candidate gate_c ratios (recovered from `windows` /
`windows_initial_by_rule` / `gate_c_attempt_log`) reproduces:
 - `kappa_median_eligible`: exact match (0.18622995807754805)
 - the initial 6-slot stratified pick: exact match to
   `windows_initial_by_rule` (11000/11300 low, 10600/8200 median,
   8900/10200 high)
 - the final accepted windows: exact match to `windows`
   (11000, 11300, 10600, 13400 — `low`/`low`/`median`/`median` — with
   `high` fully exhausted, matching the published run exactly)

The **only** unreproduced detail is cosmetic: the fixture's second
`high`-stratum draw logs only one failed attempt (`start=8900`) before
declaring the stratum exhausted, whereas a straightforward re-run of the
same deterministic ordering (what `apply_gate_c` does) retries all 8
candidates again, as the first `high` draw did — same final answer
(nothing accepted for `high` either way), just a more verbose log for
that second draw. This suggests the original implementation carried some
cross-draw memory of prior failures within a stratum name that isn't
otherwise observable from the fixture, and reproducing that exact logging
quirk isn't achievable from the available evidence. Since the outcome
(the only thing downstream code could depend on) matches exactly, this is
left as a documented cosmetic gap rather than pursued further.

---

## Agent B: `stb/config.py` extended with `train_um` (not a bug fix -- authorized, additive)

Not a blocker. Recorded here per the house rule that touching another
agent's module gets a note, even when it isn't a bug.

`stb.arms`'s arm A needs `train_um / vox_um` (v2_score.py hardcoded
`FACTOR = 4.8 / 7.91`; PLAN_V3.md's Agent-B task text: "Los factores:
train_um=4.8 y vox_um vienen de la config, no hardcodeados"). `vox_um`
was already a `ScrollConfig` field (Agent A); `train_um` was not. Added
`train_um: float = 4.8` to `ScrollConfig` (default = the PHerc0332 value,
so every existing config/test is unaffected) and a matching optional
`"train_um"` key to `load_config`, and set it explicitly in
`configs/pherc0332.json`. This was cleared live with the build
coordinator (who offered the same two options PLAN_V3.md's ambiguity
allowed: put it in config, or make it an arms.py parameter with a
documented default) before editing; picked "in config" since it's the
one place every other scroll/train-checkpoint constant already lives,
and it keeps `stb.arms.score_window`'s signature free of one more
scroll-specific magic number. No other field, semantics, or test in
`stb/config.py` changed.

---

## Agent B: `stb.arms`'s frozen denominator can't be reproduced bit-exactly offline for one (window, direction, arm) unit

**Status:** not a bug in the port; a genuine, investigated, unavoidable
offline-data gap. Documented per PLAN_V3.md's rule, test left pinned at
the specified 0.01pp tolerance and marked `xfail(strict=True)` rather
than weakened (`tests/test_arms.py::test_known_gap_s13400_front_null_perm_frozen_denominator`).

**Mechanism.** `stb.arms.score_window`'s `frozen` denominator (per
window/direction) is the intersection of every scored arm's `ok` mask
(`reference_src/v2_score.py`'s own docstring: "the intersection of the
scorable masks of every arm evaluated there"). The published run scored
six arms whenever `p2` was finite: A, B_p2, C_p2, B_p1, C_p1, D_9vox.
B_p1/C_p1 need a per-cell P1 pitch estimate
(`stb.arms.p1_per_cell`/`stb.estimator.estimator_p1`), which is derived
from the *raw per-cell CT profile spacings* the real run sampled from the
zarr volume (`estimator_p2`'s `"spacings"` array, 81 values per window).
`fixtures/v2_scores_20260711.json` records only the two reduced scalars
(`p2`, `p2_cell_valid_frac`) per window, never the raw per-cell
`spacings` -- so B_p1/C_p1 cannot be reconstructed from any offline
fixture, and PLAN_V3.md's own test spec (b) only requires reproducing
arms A/B_p2/C_p2/D_9vox/null_perm for exactly this reason (B_p1/C_p1 are
never checked).

`stb.arms.score_window(..., p2_estimate=...)` therefore computes `frozen`
from only the four reconstructable arms (plus null_perm's own additional
mask, which does not affect `frozen` itself). Since `ok`'s only
arm-dependent ingredient is `exp_edge` (whether the arm's *own* predicted
point's nearest neighbor in the expected-class tree happens to sit on the
band's physical row/col edge), dropping two arms from the AND-reduction
can only ever make the reconstructed `frozen` a superset (or equal) of
the true one -- never a subset.

**Measured impact** (`python -m pytest tests/test_arms.py -v`, all 4
windows x 2 directions = 8 units, 5 arms each = 40 checks):
- 6 of 8 units: `frozen_included` matches the fixture exactly, so all 5
  arms match to full float64 precision (diff 0.000000).
- 1 unit (s11300 front): `frozen_included` differs by 1 cell (got 5307,
  fixture 5306); all 5 arms still land within the 0.01pp tolerance
  (largest diff 0.0073pp).
- 1 unit (s13400 front): `frozen_included` differs by 2 cells (got 4843,
  fixture 4841); A/B_p2/C_p2/D_9vox stay within tolerance (largest diff
  0.0048pp), but `null_perm` -- whose own denominator is
  `frozen & null["ok"]`, compounding the frozen-set error -- lands at
  25.913621% vs the published 25.924387%, diff 0.010766pp, marginally
  over the 0.01pp bar.

**Why not closeable offline:** any attempt to approximate B_p1/C_p1 (e.g.
substituting `p2` for the unknown per-cell `p1_cells`, which is exactly
what `p1_fallback_frac == 0.0` in every published window tells us did
*not* happen in the real run) makes B_p1 identical to B_p2 and C_p1
identical to C_p2, a no-op that changes nothing about `frozen` --
confirmed by construction, not just by testing. The true per-cell
spacings are not recoverable from any recorded aggregate.

**Recommendation for the lead:** either accept this single, thoroughly
quantified 0.0108pp overshoot as inherent to offline reproduction (the
`xfail(strict=True)` will loudly flag if it should ever start passing,
e.g. if a future fixture set records the raw per-cell spacings), or add
`spacings`/`rr`/`cc` for the 9x9 grid to a future fixture export so
B_p1/C_p1 -- and hence bit-exact `frozen` -- become reproducible offline.
