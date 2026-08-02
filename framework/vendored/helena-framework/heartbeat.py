#!/usr/bin/env python3
"""Atomic heartbeat writer/reader for the segmentation campaign loop.

Schema is FIXED (see PLAN_segmentation_campaign.md) so any consumer --
the local watchdog, Claude re-checking in, a future batch-merge step --
can parse it cheaply without reading logs. Written atomically (temp file +
os.rename) so a reader never sees a partial write.
"""
import json
import os
import time

SCHEMA_FIELDS = {
    "timestamp", "status", "current_seed", "seeds_attempted",
    "seeds_succeeded", "seeds_failed", "consecutive_distinct_seed_failures",
    "spent_usd_estimate", "fetched_gb_estimate", "last_error",
    "gate_c_pass_count", "windability_gate_pass_count",
}
VALID_STATUSES = {"running", "paused_budget", "systemic_failure", "crashed", "done"}


def write(path, **fields):
    """Write the heartbeat atomically. Missing fields default to sane values
    on first write; on later writes callers should pass the full running
    state (the executor loop owns a single in-memory state dict and calls
    this each iteration)."""
    state = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running", "current_seed": None,
        "seeds_attempted": 0, "seeds_succeeded": 0, "seeds_failed": 0,
        "consecutive_distinct_seed_failures": 0,
        "spent_usd_estimate": 0.0, "fetched_gb_estimate": 0.0,
        "last_error": None, "gate_c_pass_count": 0,
        "windability_gate_pass_count": 0,
    }
    state.update(fields)
    assert state["status"] in VALID_STATUSES, f"bad status {state['status']!r}"
    unknown = set(fields) - SCHEMA_FIELDS
    assert not unknown, f"unknown heartbeat field(s): {unknown}"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.rename(tmp, path)  # atomic on the same filesystem
    return state


def read(path, max_age_s=None):
    """Returns (state_dict, age_seconds) or (None, None) if missing/corrupt.
    If max_age_s is given and the heartbeat is older, still returns it but
    the caller should treat that as 'stale' per its own policy."""
    try:
        with open(path) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    ts = time.strptime(state["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
    age = time.time() - time.mktime(ts) + time.timezone
    return state, age


if __name__ == "__main__":
    # self-test: write, read back, verify atomicity assumption and schema
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "heartbeat.json")
    s = write(p, status="running", current_seed="w999_test", seeds_attempted=3)
    r, age = read(p)
    assert r == s, "read-back mismatch"
    assert age < 2, f"age should be ~0 right after write, got {age}"
    assert set(r.keys()) == SCHEMA_FIELDS, "schema drift"
    try:
        write(p, status="not_a_real_status")
        raise SystemExit("FAIL: bad status was not rejected")
    except AssertionError:
        pass
    try:
        write(p, status="running", made_up_field=1)
        raise SystemExit("FAIL: unknown field was not rejected")
    except AssertionError:
        pass
    missing, missing_age = read(os.path.join(tempfile.mkdtemp(), "nope.json"))
    assert missing is None and missing_age is None
    print("heartbeat.py self-test PASSED")
