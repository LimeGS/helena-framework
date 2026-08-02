# helena-framework

A small, dependency-free framework for running a long, unattended batch of
work ("seeds") on rented compute, under hard financial ceilings, supervised
by a local watchdog that doesn't trust the remote loop's own self-reports.

Extracted from the PHerc0139 (Herculaneum scroll) VC3D segmentation-expansion
campaign in the `vesuvius-challenge` project, where it drove a bounded batch
of geometry-growing jobs on a rented vast.ai instance. Nothing here is
specific to that domain — the pluggable piece is a `grow_fn(seed_id,
seed_coords, prediction_uri, log, budget, fail_mode=None) -> (payload, gb, usd)`
callable; swap that out and this runs any batch of independent, retryable
units of work with a cost per unit.

## Why this exists

Running unattended jobs on rented-by-the-hour, billed-by-bandwidth compute
has two failure modes worth designing against up front:

1. **A silent runaway.** No one is watching in real time, so the loop itself
   must refuse to exceed its own budget *before* it spends, not after —
   and an external, independent watchdog must be able to kill the instance
   even if the loop's own accounting has a bug.
2. **A silent stall.** A systemic problem (expired credentials, a dead
   dependency, a network outage) shouldn't burn through the whole seed list
   failing one by one — the loop should recognize the pattern and stop
   itself, escalating to a cheap model for a second opinion only when that
   happens (not on every iteration).

## Pieces

| File | Role |
|---|---|
| `heartbeat.py` | Atomic (temp+rename) JSON heartbeat writer/reader, fixed schema |
| `camplog.py` | Structured JSON-lines logging with size-based rotation |
| `executor_loop.py` | The engine: retry+backoff, proactive budget gating, systemic-failure detection, crash handling, and a concurrent (`ThreadPoolExecutor`) runner alongside the sequential one |
| `openrouter_helper.py` | Escalation-only LLM call (OpenRouter), full prompt/response audit trail, safe no-op with no API key |
| `watchdog.sh` / `qa_watchdog.sh` | External, local, dependency-free bash watchdog that polls the remote heartbeat over SSH and kills+destroys the instance on any red flag — plus its own mock-mode QA harness |

## Design principles

- **Proactive over reactive.** The loop checks its own budget *before* an
  expensive operation, not after. The external watchdog is the backup for
  when the loop's own check has a bug, not the primary defense — a watchdog
  polling every 60-90s cannot catch a burst that blows a ceiling in seconds.
- **Fail-safe defaults.** If the watchdog can't verify state (heartbeat
  unreachable, an SSH/API check fails), it assumes the worst and kills,
  rather than assuming fine and continuing.
- **Bounded retry, not infinite hope.** Each unit of work gets a small,
  fixed number of retries with exponential backoff. A *pattern* of several
  different units failing in a row is treated as a systemic signal (stop and
  escalate), not as several instances of bad luck.
- **Concurrency without a shared-state race.** `run_campaign_concurrent`
  dispatches work via a thread pool; all manifest/heartbeat/counter state is
  touched only from the single thread draining `as_completed()`, so it needs
  no locking. The one piece of state genuinely touched from worker threads —
  the budget — gets its own atomic `reserve()`/`reconcile()` pair specifically
  to close the check-then-commit race that naive concurrent callers would hit.
- **Escalation is a circuit breaker, not a co-pilot.** The LLM call fires
  only after N consecutive distinct units have failed — a deterministic,
  already-validated pipeline shouldn't re-decide its own control flow via a
  model on every iteration. Every call (or skipped call) is logged in full.
- **No silent schema drift.** Every result record has the same fixed set of
  fields regardless of outcome, specifically so a later batch-merge step can
  flat-iterate a manifest without a per-record shape check.

## Self-tests

Every module is self-testing, zero cost, no network/credentials required:

```
python3 heartbeat.py
python3 camplog.py
python3 openrouter_helper.py
python3 executor_loop.py --qa-self-test
```

`executor_loop.py --qa-self-test` covers both the sequential and concurrent
runners: happy path, flaky-retry recovery, hard failure, systemic-failure
detection + escalation, proactive budget-ceiling enforcement (including the
atomic `reserve()`/`reconcile()` race check under real concurrent hammering),
crash isolation, and full output-schema completeness — 27 checks, all
passing as of the last run in this repo.

## Using it elsewhere

1. Write your own `grow_fn` (or reuse the name `grow_and_gate` if you like —
   nothing requires it) matching the signature above.
2. Call `run_campaign_concurrent(seeds, out_dir, usd_ceiling, gb_ceiling,
   hb_path, log_path, grow_fn=your_fn, max_workers=N)` (or `run_campaign` for
   the simpler sequential version).
3. Point `watchdog.sh` at the remote heartbeat path over SSH, set your own
   `USD_CEILING`/`GB_CEILING`, run it locally alongside the remote loop.
4. The escalation path is optional — with no `OPENROUTER_API_KEY` set it
   safely no-ops (logs a skip, returns `None`), so you can leave it wired in
   without a key if you don't want it.

## What's intentionally NOT in this repo

Anything domain-specific (what a "seed" is, how to grow/validate one, where
the data lives) stays in the calling project. This repo is the scheduling/
supervision engine only.
