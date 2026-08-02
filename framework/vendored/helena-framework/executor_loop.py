#!/usr/bin/env python3
"""Main segmentation-campaign executor loop. Runs ON the rented instance
(under tmux/nohup so it survives SSH disconnects -- see PLAN doc).

Deterministic control flow (seed -> grow -> gate -> save -> next); an LLM
(OpenRouter) is called ONLY on escalation (a seed type failing repeatedly
in a way plain retry didn't fix), not per-iteration -- see openrouter_helper.py.
This is deliberately NOT "ask an LLM what to do every step": the mesh-grow-
and-gate pipeline is already fully deterministic, validated code from
today's pilots: re-deciding it via a free small model every iteration would
be strictly worse (slower, riskier, burns free-tier rate limit) with no
upside.

OUTPUT SCHEMA (fixed, so a later batch-merge step can just iterate the
manifest without re-deriving anything):

  <OUT>/manifest.json                 -- list of all attempted seed results
  <OUT>/results/<seed_id>.json        -- one fixed-schema record per seed:
    {
      "seed_id": str, "timestamp": iso8601, "status": one of
        ["success","gate_c_fail","windability_fail","job_failed","skipped_budget"],
      "seed_coords": {"x":int,"y":int,"z":int,"space":str},
      "prediction_uri": str, "job_id": str or null,
      "area_cm2": float or null, "bbox_mm": [float,float,float] or null,
      "gate_c": {...} or null, "windability_gate": {...} or null,
      "mesh_path": str or null (relative to <OUT>/meshes/),
      "cost_usd_estimate": float, "gb_fetched_estimate": float,
      "elapsed_s": float, "attempt": int, "error": str or null,
    }
  <OUT>/meshes/<seed_id>/             -- tifxyz mesh output, only for status=="success"

Every field above is present in every record (never a variably-shaped
dict) specifically so batch processing later is a flat iteration, not a
per-record schema check.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heartbeat  # noqa: E402
from camplog import CampaignLog  # noqa: E402
from openrouter_helper import escalate  # noqa: E402

MAX_RETRIES_PER_SEED = 3
BACKOFF_BASE_S = 5
CONSECUTIVE_DISTINCT_FAILURE_LIMIT = 5
RESULT_STATUSES = {"success", "gate_c_fail", "windability_fail", "job_failed", "skipped_budget"}


def new_result(seed_id, seed_coords, prediction_uri, attempt):
    return {
        "seed_id": seed_id, "timestamp": _now(), "status": None,
        "seed_coords": seed_coords, "prediction_uri": prediction_uri,
        "job_id": None, "area_cm2": None, "bbox_mm": None,
        "gate_c": None, "windability_gate": None, "mesh_path": None,
        "cost_usd_estimate": 0.0, "gb_fetched_estimate": 0.0,
        "elapsed_s": 0.0, "attempt": attempt, "error": None,
    }


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def check_disk_headroom(min_free_gb=5.0):
    """Refuse to keep going if the local (remote-instance) disk is nearly
    full -- the exact failure mode that hit the local Mac earlier today."""
    st = os.statvfs(".")
    free_gb = (st.f_bavail * st.f_frsize) / 1e9
    return free_gb >= min_free_gb, free_gb


def estimate_fetch_gb(n_chunks, bytes_per_chunk=128 ** 3):
    return n_chunks * bytes_per_chunk / 1e9


class Budget:
    """Proactive, in-loop ceiling enforcement -- the PRIMARY defense against
    a bandwidth blowup (see PLAN doc: an external watchdog polling every N
    seconds cannot catch a burst that exceeds the ceiling in seconds, which
    is exactly what happened in today's real $3.08 incident)."""

    def __init__(self, usd_ceiling, gb_ceiling):
        self.usd_ceiling = usd_ceiling
        self.gb_ceiling = gb_ceiling
        self.spent_usd = 0.0
        self.fetched_gb = 0.0
        self._lock = threading.Lock()

    def would_exceed(self, extra_usd=0.0, extra_gb=0.0):
        return (self.spent_usd + extra_usd > self.usd_ceiling or
                self.fetched_gb + extra_gb > self.gb_ceiling)

    def commit(self, usd=0.0, gb=0.0):
        self.spent_usd += usd
        self.fetched_gb += gb

    def reserve(self, extra_usd=0.0, extra_gb=0.0):
        """Thread-safe atomic check-and-reserve. would_exceed()+commit() has
        a TOCTOU race under concurrent callers (N workers can all pass the
        check before any of them commits, overshooting the ceiling by up to
        N-1 seeds' worth). reserve() closes that by doing the check and the
        increment under one lock. Returns True (and reserves) if it fits,
        False (no reservation made) otherwise. Only the concurrent runner
        uses this -- the sequential run_campaign() has no race with a single
        thread, so it keeps using would_exceed()/commit() unchanged."""
        with self._lock:
            if (self.spent_usd + extra_usd > self.usd_ceiling or
                    self.fetched_gb + extra_gb > self.gb_ceiling):
                return False
            self.spent_usd += extra_usd
            self.fetched_gb += extra_gb
            return True

    def reconcile(self, reserved_usd, reserved_gb, actual_usd, actual_gb):
        """Replace a prior reserve()'s (conservative estimate) with the real
        measured usage, once grow_fn returns actual numbers."""
        with self._lock:
            self.spent_usd += actual_usd - reserved_usd
            self.fetched_gb += actual_gb - reserved_gb


def grow_and_gate_MOCK(seed_id, seed_coords, prediction_uri, log, budget, fail_mode=None):
    """Stand-in for the real VC3D MCP call chain (vc_generate_surface ->
    vc_get_job poll -> Gate C coherence check -> windability gate), so the
    loop's CONTROL FLOW (retry/backoff, failure counting, budget gating,
    crash handling) can be exhaustively tested without a live instance --
    exactly the same mock-first QA discipline used for watchdog.sh.
    The real implementation swaps this function out for actual MCP tool
    calls (contract already reverse-engineered in KNOWHOW.md section 2);
    everything else in this file is unchanged.
    fail_mode: None (succeed), "job_failed", "gate_c_fail", "windability_fail",
    "budget" (simulate a fetch that would exceed budget), "flaky" (fails
    attempt 1-2, succeeds on 3 -- tests retry/backoff actually recovers).
    """
    est_gb = estimate_fetch_gb(n_chunks=500)  # placeholder realistic-ish size
    if fail_mode == "budget":
        return {"status": "skipped_budget", "error": "would exceed GB ceiling"}, est_gb, 0.01
    if fail_mode == "job_failed":
        return {"status": "job_failed", "error": "vc_get_job: state=failed (mock)"}, 0.0, 0.0
    if fail_mode == "gate_c_fail":
        return {"status": "gate_c_fail", "gate_c": {"pass": False}, "error": "coherence gate failed (mock)"}, est_gb, 0.01
    if fail_mode == "windability_fail":
        return {"status": "windability_fail", "windability_gate": {"windable": False}, "error": "axis pierces sheet (mock)"}, est_gb, 0.01
    return {
        "status": "success", "job_id": f"mock-job-{seed_id}",
        "area_cm2": 12.3, "bbox_mm": [30.0, 30.0, 5.0],
        "gate_c": {"pass": True}, "windability_gate": {"windable": True},
        "mesh_path": f"{seed_id}/",
    }, est_gb, 0.02


def run_seed(seed_id, seed_coords, prediction_uri, out_dir, log, budget, grow_fn, force_fail_mode=None,
             atomic_budget=False):
    """One seed, with bounded retry + backoff. Returns the final result dict.
    atomic_budget=True uses Budget.reserve()/reconcile() (thread-safe, no
    TOCTOU race) instead of would_exceed()/commit() -- set by the concurrent
    runner, where multiple seeds can hit this check at the same instant."""
    last_err = None
    for attempt in range(1, MAX_RETRIES_PER_SEED + 1):
        t0 = time.time()
        result = new_result(seed_id, seed_coords, prediction_uri, attempt)
        ok, free_gb = check_disk_headroom()
        if not ok:
            result["status"] = "job_failed"
            result["error"] = f"insufficient disk headroom ({free_gb:.1f}GB free)"
            log.error("disk headroom check failed", seed=seed_id, free_gb=free_gb)
            return result

        est_gb = estimate_fetch_gb(n_chunks=500)
        if atomic_budget:
            reserved = budget.reserve(extra_gb=est_gb)
            if not reserved:
                result["status"] = "skipped_budget"
                result["error"] = "proactive estimate would exceed GB ceiling"
                log.warn("proactive budget check refused fetch", seed=seed_id,
                         estimated_gb=est_gb, already_fetched_gb=budget.fetched_gb)
                return result
        elif budget.would_exceed(extra_gb=est_gb):
            result["status"] = "skipped_budget"
            result["error"] = "proactive estimate would exceed GB ceiling"
            log.warn("proactive budget check refused fetch", seed=seed_id,
                     estimated_gb=est_gb, already_fetched_gb=budget.fetched_gb)
            return result

        try:
            payload, gb, usd = grow_fn(seed_id, seed_coords, prediction_uri, log, budget,
                                        fail_mode=force_fail_mode)
            if atomic_budget:
                budget.reconcile(reserved_usd=0.0, reserved_gb=est_gb, actual_usd=usd, actual_gb=gb)
            else:
                budget.commit(usd=usd, gb=gb)
            result.update(payload)
            result["gb_fetched_estimate"] = gb
            result["cost_usd_estimate"] = usd
            result["elapsed_s"] = round(time.time() - t0, 2)
            assert result["status"] in RESULT_STATUSES, f"bad status {result['status']!r}"

            if result["status"] == "success":
                log.info("seed succeeded", seed=seed_id, attempt=attempt, area_cm2=result["area_cm2"])
                return result
            if result["status"] == "skipped_budget":
                log.warn("seed skipped, over budget", seed=seed_id)
                return result
            # gate_c_fail / windability_fail / job_failed: worth a retry
            last_err = result["error"]
            log.warn("seed attempt failed, will retry" if attempt < MAX_RETRIES_PER_SEED else "seed exhausted retries",
                      seed=seed_id, attempt=attempt, status=result["status"], error=last_err)
        except Exception as e:  # noqa: BLE001 -- a grow_fn bug must not crash the whole loop
            last_err = f"{type(e).__name__}: {e}"
            result["status"] = "job_failed"
            result["error"] = last_err
            log.error("unhandled exception in grow_fn", seed=seed_id, attempt=attempt, error=last_err)
            if atomic_budget:
                # grow_fn raised before returning real usage -- release the
                # reservation (else it leaks forever and can wrongly trip
                # the ceiling with zero real spend behind it).
                budget.reconcile(reserved_usd=0.0, reserved_gb=est_gb, actual_usd=0.0, actual_gb=0.0)

        if attempt < MAX_RETRIES_PER_SEED:
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))

    # Preserve the LAST attempt's actual status (already sitting in `result`
    # from the final iteration's payload merge, or "job_failed" if that last
    # attempt raised) -- do NOT collapse everything to "job_failed" here.
    # A seed that failed 3 times on a legitimate, deterministic gate
    # rejection (gate_c_fail/windability_fail) must still be REPORTED as
    # that, not mislabeled as an infrastructure failure -- this exact
    # mislabeling is what made real windability rejections in a production
    # run look like "job_failed" in the manifest, and (before the
    # accompanying circuit-breaker fix) made a legitimate cluster of
    # non-windable seeds masquerade as a systemic infrastructure problem.
    result["error"] = f"exhausted {MAX_RETRIES_PER_SEED} attempts, last error: {last_err}"
    return result


def run_campaign(seeds, out_dir, usd_ceiling, gb_ceiling, hb_path, log_path,
                  grow_fn=grow_and_gate_MOCK, force_fail_modes=None):
    """seeds: list of (seed_id, seed_coords, prediction_uri).
    force_fail_modes: optional {seed_id: fail_mode} for QA -- simulates
    specific failure classes without touching real infra."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    force_fail_modes = force_fail_modes or {}
    log = CampaignLog(log_path)
    # dedicated log for the full audit trail of every LLM call (prompt +
    # response + latency + success flag), separate from the operational
    # log so it can be reviewed on its own -- see openrouter_helper.py
    llm_log = CampaignLog(os.path.join(out_dir, "llm_calls.log"))
    budget = Budget(usd_ceiling, gb_ceiling)
    manifest = []
    consecutive_distinct_failures = 0
    seeds_succeeded = seeds_failed = 0

    def hb_write(status, current_seed=None, last_error=None):
        heartbeat.write(
            hb_path, status=status, current_seed=current_seed,
            seeds_attempted=len(manifest), seeds_succeeded=seeds_succeeded,
            seeds_failed=seeds_failed,
            consecutive_distinct_seed_failures=consecutive_distinct_failures,
            spent_usd_estimate=round(budget.spent_usd, 4),
            fetched_gb_estimate=round(budget.fetched_gb, 4),
            last_error=last_error,
            gate_c_pass_count=sum(1 for r in manifest if (r.get("gate_c") or {}).get("pass")),
            windability_gate_pass_count=sum(1 for r in manifest if (r.get("windability_gate") or {}).get("windable")),
        )

    log.info("campaign starting", n_seeds=len(seeds), usd_ceiling=usd_ceiling, gb_ceiling=gb_ceiling)
    hb_write("running")

    try:
        for seed_id, seed_coords, prediction_uri in seeds:
            hb_write("running", current_seed=seed_id)
            result = run_seed(seed_id, seed_coords, prediction_uri, out_dir, log, budget,
                               grow_fn, force_fail_mode=force_fail_modes.get(seed_id))
            with open(os.path.join(out_dir, "results", f"{seed_id}.json"), "w") as f:
                json.dump(result, f, indent=1)
            manifest.append(result)

            if result["status"] == "success":
                seeds_succeeded += 1
                consecutive_distinct_failures = 0
            elif result["status"] == "skipped_budget":
                log.warn("budget ceiling reached, pausing campaign", seed=seed_id)
                hb_write("paused_budget", current_seed=seed_id, last_error=result["error"])
                with open(os.path.join(out_dir, "manifest.json"), "w") as f:
                    json.dump(manifest, f, indent=1)
                return manifest
            elif result["status"] in ("gate_c_fail", "windability_fail"):
                # A legitimate, deterministic gate REJECTION -- the pipeline
                # correctly identified non-conforming geometry. This is an
                # expected, valid outcome (proof the gates are working), not
                # a sign of systemic breakage, so it must NOT count toward
                # (and in fact resets, like a success) the systemic-failure
                # circuit breaker. Only genuine infrastructure/job failures
                # (job_failed) indicate something is actually broken.
                seeds_failed += 1
                consecutive_distinct_failures = 0
            else:
                seeds_failed += 1
                consecutive_distinct_failures += 1
                if consecutive_distinct_failures >= CONSECUTIVE_DISTINCT_FAILURE_LIMIT:
                    log.error("systemic failure: too many consecutive distinct seeds failed",
                              count=consecutive_distinct_failures)
                    recent_errors = [r["error"] for r in manifest[-CONSECUTIVE_DISTINCT_FAILURE_LIMIT:] if r.get("error")]
                    suggestion = escalate(
                        context=f"{consecutive_distinct_failures} consecutive distinct seeds failed, "
                                f"last seed={seed_id}",
                        recent_errors=recent_errors, log=llm_log,
                    )
                    log.info("escalation result (full prompt/response in llm_calls.log)",
                             suggestion=suggestion)
                    hb_write("systemic_failure", current_seed=seed_id, last_error=result["error"])
                    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
                        json.dump(manifest, f, indent=1)
                    return manifest

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        log.info("campaign done", seeds_succeeded=seeds_succeeded, seeds_failed=seeds_failed)
        hb_write("done")
        return manifest

    except Exception as e:  # noqa: BLE001 -- top-level crash handler (PLAN doc requirement)
        tb = traceback.format_exc()
        log.error("CRASH", error=str(e), traceback=tb)
        hb_write("crashed", last_error=f"{type(e).__name__}: {e}")
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        raise


def run_campaign_concurrent(seeds, out_dir, usd_ceiling, gb_ceiling, hb_path, log_path,
                             grow_fn=grow_and_gate_MOCK, force_fail_modes=None, max_workers=6):
    """Concurrent variant of run_campaign: dispatches up to max_workers seeds
    at once via a thread pool. grow_and_gate is I/O-bound (MCP HTTP calls +
    polling, with the actual grow/gate compute happening server-side and in
    short local numpy calls) -- threads are the right primitive here, the
    GIL is not the bottleneck.

    All shared mutable state (manifest, counters, heartbeat, stop decision)
    is touched ONLY from the MAIN thread, in the as_completed() loop below --
    so none of that needs its own lock. The one exception is Budget, which
    IS touched from worker threads (inside run_seed's atomic_budget path),
    so it has its own internal lock (see reserve()/reconcile()).

    Systemic-failure / budget-ceiling stop is cooperative, not preemptive: a
    shared stop_event is checked at the START of each worker, before any
    real grow work, so no NEW seed starts real work once triggered. Seeds
    already mid-flight when the trigger fires finish naturally (there is no
    safe way to cancel a thread mid network-call) -- given each seed's real
    cost here is a few cents / <1MB, this is an accepted, bounded tradeoff,
    not a gap in the ceiling: it can only ever be exceeded by work already
    reserved+in-flight at the moment of the trigger, never by new work.

    Unlike run_campaign(), every submitted seed ends up in the manifest
    (even ones that got cooperatively skipped post-trigger, as a
    status=skipped_budget stub) -- there is no way to un-submit an already-
    queued future, so "stopped early" is expressed via stop reason + status,
    not via a short manifest.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    force_fail_modes = force_fail_modes or {}
    log = CampaignLog(log_path)
    llm_log = CampaignLog(os.path.join(out_dir, "llm_calls.log"))
    budget = Budget(usd_ceiling, gb_ceiling)
    manifest = []
    consecutive_distinct_failures = 0
    seeds_succeeded = seeds_failed = 0
    stop_event = threading.Event()
    stop_reason = {"status": None, "seed": None, "error": None}

    def hb_write(status, current_seed=None, last_error=None):
        heartbeat.write(
            hb_path, status=status, current_seed=current_seed,
            seeds_attempted=len(manifest), seeds_succeeded=seeds_succeeded,
            seeds_failed=seeds_failed,
            consecutive_distinct_seed_failures=consecutive_distinct_failures,
            spent_usd_estimate=round(budget.spent_usd, 4),
            fetched_gb_estimate=round(budget.fetched_gb, 4),
            last_error=last_error,
            gate_c_pass_count=sum(1 for r in manifest if (r.get("gate_c") or {}).get("pass")),
            windability_gate_pass_count=sum(1 for r in manifest if (r.get("windability_gate") or {}).get("windable")),
        )

    def worker(seed_id, seed_coords, prediction_uri):
        if stop_event.is_set():
            r = new_result(seed_id, seed_coords, prediction_uri, attempt=0)
            r["status"] = "skipped_budget"
            r["error"] = "campaign already stopped (systemic failure or budget ceiling)"
            return r
        return run_seed(seed_id, seed_coords, prediction_uri, out_dir, log, budget,
                         grow_fn, force_fail_mode=force_fail_modes.get(seed_id),
                         atomic_budget=True)

    log.info("campaign starting (concurrent)", n_seeds=len(seeds), usd_ceiling=usd_ceiling,
             gb_ceiling=gb_ceiling, max_workers=max_workers)
    hb_write("running")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, sid, coords, uri): sid for sid, coords, uri in seeds}
            for fut in as_completed(futures):
                seed_id = futures[fut]
                result = fut.result()  # worker/run_seed never raises (per-attempt catch-all)
                with open(os.path.join(out_dir, "results", f"{seed_id}.json"), "w") as f:
                    json.dump(result, f, indent=1)
                manifest.append(result)

                if result["status"] == "success":
                    seeds_succeeded += 1
                    consecutive_distinct_failures = 0
                elif result["status"] == "skipped_budget":
                    seeds_failed += 1
                    if stop_reason["status"] is None:
                        stop_reason.update(status="paused_budget", seed=seed_id, error=result["error"])
                        stop_event.set()
                elif result["status"] in ("gate_c_fail", "windability_fail"):
                    # legitimate deterministic gate rejection, not a systemic
                    # signal -- see the matching comment in run_campaign()
                    seeds_failed += 1
                    consecutive_distinct_failures = 0
                else:
                    seeds_failed += 1
                    consecutive_distinct_failures += 1
                    if consecutive_distinct_failures >= CONSECUTIVE_DISTINCT_FAILURE_LIMIT and stop_reason["status"] is None:
                        log.error("systemic failure: too many consecutive distinct seeds failed (concurrent)",
                                  count=consecutive_distinct_failures)
                        recent_errors = [r["error"] for r in manifest[-CONSECUTIVE_DISTINCT_FAILURE_LIMIT:] if r.get("error")]
                        suggestion = escalate(
                            context=f"{consecutive_distinct_failures} consecutive distinct seeds failed "
                                    f"(concurrent run), last seed={seed_id}",
                            recent_errors=recent_errors, log=llm_log,
                        )
                        log.info("escalation result (full prompt/response in llm_calls.log)", suggestion=suggestion)
                        stop_reason.update(status="systemic_failure", seed=seed_id, error=result["error"])
                        stop_event.set()

                hb_write(stop_reason["status"] or "running", current_seed=seed_id,
                         last_error=stop_reason["error"])

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)

        if stop_reason["status"] is not None:
            log.warn("campaign stopped early", reason=stop_reason["status"])
            hb_write(stop_reason["status"], current_seed=stop_reason["seed"], last_error=stop_reason["error"])
        else:
            log.info("campaign done", seeds_succeeded=seeds_succeeded, seeds_failed=seeds_failed)
            hb_write("done")
        return manifest

    except Exception as e:  # noqa: BLE001 -- top-level crash handler (PLAN doc requirement)
        tb = traceback.format_exc()
        log.error("CRASH", error=str(e), traceback=tb)
        hb_write("crashed", last_error=f"{type(e).__name__}: {e}")
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-self-test", action="store_true")
    args = ap.parse_args()
    if not args.qa_self_test:
        ap.error("only --qa-self-test is wired up in this file; the real "
                 "deployment entrypoint swaps grow_and_gate_MOCK for the "
                 "real VC3D MCP client and is built once vast.ai credit "
                 "is available to test it end-to-end against a live server")

    import tempfile
    d = tempfile.mkdtemp()
    print(f"=== executor_loop.py QA self-test (all mock, zero cost) in {d} ===")
    passed = failed = 0

    def check(name, cond):
        global passed, failed
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        passed += cond
        failed += (not cond)

    # 1) happy path: all succeed
    seeds = [(f"s{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(3)]
    m = run_campaign(seeds, f"{d}/run1", usd_ceiling=10, gb_ceiling=100,
                      hb_path=f"{d}/run1_hb.json", log_path=f"{d}/run1.log")
    check("happy path: 3/3 succeed", all(r["status"] == "success" for r in m))
    hb, _ = heartbeat.read(f"{d}/run1_hb.json")
    check("happy path: heartbeat status done", hb["status"] == "done")
    check("happy path: manifest.json written", os.path.exists(f"{d}/run1/manifest.json"))

    # 2) flaky retry: gate_c_fail on attempt 1-2, success on 3 (tests backoff actually recovers)
    calls = {"n": 0}
    def flaky_grow(seed_id, coords, uri, log, budget, fail_mode=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"status": "gate_c_fail", "gate_c": {"pass": False}, "error": "transient (mock)"}, 0.01, 0.001
        return grow_and_gate_MOCK(seed_id, coords, uri, log, budget, fail_mode=None)
    import time as _t
    orig_sleep = _t.sleep; _t.sleep = lambda s: None  # don't actually wait through backoff in QA
    m = run_campaign([("flaky", {"x": 1, "y": 1, "z": 1, "space": "ct_l0_xyz"}, "uri://mock")],
                      f"{d}/run2", usd_ceiling=10, gb_ceiling=100,
                      hb_path=f"{d}/run2_hb.json", log_path=f"{d}/run2.log", grow_fn=flaky_grow)
    check("flaky retry: recovers by attempt 3", m[0]["status"] == "success" and m[0]["attempt"] == 3)

    # 3) hard failure: exhausts all retries
    m = run_campaign([("bad", {"x": 1, "y": 1, "z": 1, "space": "ct_l0_xyz"}, "uri://mock")],
                      f"{d}/run3", usd_ceiling=10, gb_ceiling=100,
                      hb_path=f"{d}/run3_hb.json", log_path=f"{d}/run3.log",
                      force_fail_modes={"bad": "job_failed"})
    check("hard failure: status job_failed after exhausting retries",
          m[0]["status"] == "job_failed" and m[0]["attempt"] == MAX_RETRIES_PER_SEED)

    # 4) systemic failure: 5 consecutive distinct seeds fail -> loop stops itself
    #    AND escalates to the LLM helper, which must log the full call
    #    (prompt+response) even with no real key configured (QA has none set)
    os.environ.pop("OPENROUTER_API_KEY", None)  # deterministic: force the no-key logged path
    seeds5 = [(f"bad{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(7)]
    fails = {sid: "job_failed" for sid, _, _ in seeds5[:5]}
    m = run_campaign(seeds5, f"{d}/run4", usd_ceiling=10, gb_ceiling=100,
                      hb_path=f"{d}/run4_hb.json", log_path=f"{d}/run4.log", force_fail_modes=fails)
    check("systemic failure: stops after 5 consecutive distinct failures (not all 7 attempted)", len(m) == 5)
    hb, _ = heartbeat.read(f"{d}/run4_hb.json")
    check("systemic failure: heartbeat status systemic_failure", hb["status"] == "systemic_failure")
    llm_lines = [json.loads(l) for l in open(f"{d}/run4/llm_calls.log")]
    check("systemic failure: escalation call was made and logged", len(llm_lines) == 1)
    check("systemic failure: logged call has kind=llm_call and the failure context in the prompt",
          llm_lines and llm_lines[0]["kind"] == "llm_call" and "5 consecutive" in llm_lines[0]["prompt"])
    campaign_log_lines = [json.loads(l) for l in open(f"{d}/run4.log")]
    check("systemic failure: campaign.log has a pointer to the escalation result",
          any("escalation result" in r.get("msg", "") for r in campaign_log_lines))

    # 4b) legitimate gate rejections (gate_c_fail/windability_fail) must NOT
    #     trip the systemic-failure breaker, even many in a row -- these are
    #     expected, deterministic pipeline outcomes (proof the gates work),
    #     not a systemic-breakage signal. Only job_failed should trip it
    #     (verified in test 4 above). This is the fix for the real incident
    #     where a legitimate cluster of non-windable adjacent wraps stopped
    #     the campaign after only 5-6 seeds, leaving 31/42 never attempted.
    seeds10 = [(f"rej{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(10)]
    fails_rej = {sid: "windability_fail" for sid, _, _ in seeds10}
    m = run_campaign(seeds10, f"{d}/run4b", usd_ceiling=10, gb_ceiling=100,
                      hb_path=f"{d}/run4b_hb.json", log_path=f"{d}/run4b.log", force_fail_modes=fails_rej)
    check("gate rejections: all 10 seeds attempted, breaker never trips on windability_fail",
          len(m) == 10 and all(r["status"] == "windability_fail" for r in m))
    hb, _ = heartbeat.read(f"{d}/run4b_hb.json")
    check("gate rejections: heartbeat status done (not systemic_failure)", hb["status"] == "done")

    # 5) budget ceiling: proactive in-loop check refuses BEFORE exceeding, pauses campaign
    seeds_budget = [(f"b{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(5)]
    tiny_gb_ceiling = estimate_fetch_gb(500) * 1.5  # room for ~1 successful seed, not more
    m = run_campaign(seeds_budget, f"{d}/run5", usd_ceiling=1000, gb_ceiling=tiny_gb_ceiling,
                      hb_path=f"{d}/run5_hb.json", log_path=f"{d}/run5.log")
    check("budget ceiling: campaign stops itself before exceeding GB ceiling",
          len(m) < len(seeds_budget) and m[-1]["status"] == "skipped_budget")
    hb, _ = heartbeat.read(f"{d}/run5_hb.json")
    check("budget ceiling: heartbeat status paused_budget", hb["status"] == "paused_budget")
    check("budget ceiling: fetched_gb_estimate never exceeds ceiling",
          hb["fetched_gb_estimate"] <= tiny_gb_ceiling)

    # 6) crash handling: grow_fn raises -> caught per-attempt, doesn't kill the whole campaign
    def crashy_grow(seed_id, coords, uri, log, budget, fail_mode=None):
        raise RuntimeError("simulated MCP client crash")
    m = run_campaign([("crashy", {"x": 1, "y": 1, "z": 1, "space": "ct_l0_xyz"}, "uri://mock")],
                      f"{d}/run6", usd_ceiling=10, gb_ceiling=100,
                      hb_path=f"{d}/run6_hb.json", log_path=f"{d}/run6.log", grow_fn=crashy_grow)
    check("per-seed exception: caught, seed marked job_failed, campaign continues",
          m[0]["status"] == "job_failed" and "simulated MCP client crash" in m[0]["error"])

    # 7) CONCURRENT happy path: N seeds all succeed, AND it's actually
    #    parallel (wall-clock << N * per-seed latency), not sequential
    #    dressed up as concurrent.
    N = 12
    SLEEP_S = 0.3
    def slow_grow(seed_id, coords, uri, log, budget, fail_mode=None):
        _t.sleep(SLEEP_S)  # simulate real MCP round-trip latency
        return grow_and_gate_MOCK(seed_id, coords, uri, log, budget, fail_mode=None)
    _t.sleep = orig_sleep  # this test needs REAL sleep to prove real concurrency
    seedsN = [(f"c{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(N)]
    t0 = time.time()
    m7 = run_campaign_concurrent(seedsN, f"{d}/run7", usd_ceiling=1000, gb_ceiling=1000,
                                  hb_path=f"{d}/run7_hb.json", log_path=f"{d}/run7.log",
                                  grow_fn=slow_grow, max_workers=6)
    elapsed = time.time() - t0
    sequential_would_take = N * SLEEP_S
    check("concurrent happy path: all N succeed", len(m7) == N and all(r["status"] == "success" for r in m7))
    check("concurrent happy path: no lost/duplicated seeds",
          {r["seed_id"] for r in m7} == {f"c{i}" for i in range(N)})
    check(f"concurrent happy path: actually parallel (took {elapsed:.2f}s vs {sequential_would_take:.2f}s sequential)",
          elapsed < sequential_would_take * 0.7)
    hb, _ = heartbeat.read(f"{d}/run7_hb.json")
    check("concurrent happy path: heartbeat status done", hb["status"] == "done")
    _t.sleep = lambda s: None  # back to instant for the remaining QA

    # 8) concurrent budget ceiling: atomic reserve() stops new dispatches,
    #    never overshoots the ceiling (this is the race would_exceed()+
    #    commit() would have under concurrency -- reserve()/reconcile()
    #    closes it, checked here directly on the Budget object too)
    b = Budget(usd_ceiling=1000, gb_ceiling=estimate_fetch_gb(500) * 1.5)
    results_par = [None] * 8
    def hammer(i):
        results_par[i] = b.reserve(extra_gb=estimate_fetch_gb(500))
    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for th in threads: th.start()
    for th in threads: th.join()
    check("Budget.reserve: exactly 1 of 8 concurrent reservations admitted (ceiling fits ~1)",
          sum(1 for r in results_par if r) == 1)
    check("Budget.reserve: fetched_gb never exceeds ceiling under concurrent hammering",
          b.fetched_gb <= b.gb_ceiling)

    seeds_budget2 = [(f"cb{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(6)]
    tiny_gb_ceiling2 = estimate_fetch_gb(500) * 1.5
    m8 = run_campaign_concurrent(seeds_budget2, f"{d}/run8", usd_ceiling=1000, gb_ceiling=tiny_gb_ceiling2,
                                  hb_path=f"{d}/run8_hb.json", log_path=f"{d}/run8.log", max_workers=6)
    hb, _ = heartbeat.read(f"{d}/run8_hb.json")
    check("concurrent budget ceiling: heartbeat status paused_budget", hb["status"] == "paused_budget")
    check("concurrent budget ceiling: fetched_gb_estimate never exceeds ceiling",
          hb["fetched_gb_estimate"] <= tiny_gb_ceiling2)
    check("concurrent budget ceiling: every submitted seed still lands in the manifest",
          len(m8) == len(seeds_budget2))

    # 9) concurrent systemic failure: still triggers, still escalates exactly once
    os.environ.pop("OPENROUTER_API_KEY", None)
    seeds9 = [(f"csf{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(9)]
    fails9 = {sid: "job_failed" for sid, _, _ in seeds9}  # ALL fail -> guarantees the trigger fires
    m9 = run_campaign_concurrent(seeds9, f"{d}/run9", usd_ceiling=1000, gb_ceiling=1000,
                                  hb_path=f"{d}/run9_hb.json", log_path=f"{d}/run9.log",
                                  force_fail_modes=fails9, max_workers=3)
    hb, _ = heartbeat.read(f"{d}/run9_hb.json")
    check("concurrent systemic failure: heartbeat status systemic_failure", hb["status"] == "systemic_failure")
    llm_lines9 = [json.loads(l) for l in open(f"{d}/run9/llm_calls.log")]
    check("concurrent systemic failure: escalation called exactly once (no duplicate trigger race)",
          len(llm_lines9) == 1)
    check("concurrent systemic failure: every submitted seed still lands in the manifest",
          len(m9) == len(seeds9))

    # 9b) concurrent: gate rejections must not trip the breaker either
    seeds9b = [(f"rejc{i}", {"x": i, "y": i, "z": i, "space": "ct_l0_xyz"}, "uri://mock") for i in range(10)]
    fails9b = {sid: "windability_fail" for sid, _, _ in seeds9b}
    m9b = run_campaign_concurrent(seeds9b, f"{d}/run9b", usd_ceiling=1000, gb_ceiling=1000,
                                   hb_path=f"{d}/run9b_hb.json", log_path=f"{d}/run9b.log",
                                   force_fail_modes=fails9b, max_workers=3)
    hb, _ = heartbeat.read(f"{d}/run9b_hb.json")
    check("concurrent gate rejections: all 10 attempted, no systemic_failure trip",
          len(m9b) == 10 and hb["status"] == "done"
          and all(r["status"] == "windability_fail" for r in m9b))

    # 10) output schema: every result has every field, no variably-shaped records
    all_records = m
    run_names = [1, 2, 3, "4b", 4, 5, 6, 7, 8, 9, "9b"]
    for r in [rr for run in run_names for rr in
              json.load(open(f"{d}/run{run}/manifest.json"))]:
        missing = {"seed_id", "timestamp", "status", "seed_coords", "prediction_uri",
                   "job_id", "area_cm2", "bbox_mm", "gate_c", "windability_gate",
                   "mesh_path", "cost_usd_estimate", "gb_fetched_estimate",
                   "elapsed_s", "attempt", "error"} - set(r.keys())
        if missing:
            check(f"schema completeness for {r.get('seed_id')}", False)
            break
    else:
        check(f"output schema: every record has every field across all {len(run_names)} runs", True)

    _t.sleep = orig_sleep
    print(f"\n=== {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)
