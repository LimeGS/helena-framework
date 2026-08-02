#!/usr/bin/env bash
# Exhaustive QA of watchdog.sh in MOCK_MODE -- zero cost, no real instance.
# Exercises every kill path + the one clean-exit path from the plan doc.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WD="$HERE/watchdog.sh"
TMP="$(mktemp -d)"
PASS=0; FAIL=0

hb() {  # hb status spent gb  -> writes a mock heartbeat with given values
  python3 -c "
import json,time
json.dump({
  'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime($4)),
  'status': '$1', 'current_seed': 'w_test', 'seeds_attempted': 1,
  'seeds_succeeded': 0, 'seeds_failed': 0,
  'consecutive_distinct_seed_failures': 0,
  'spent_usd_estimate': $2, 'fetched_gb_estimate': $3,
  'last_error': None, 'gate_c_pass_count': 0, 'windability_gate_pass_count': 0,
}, open('$TMP/hb.json', 'w'))
"
}

run_case() {
  # run_case name expected_marker extra_env...
  name="$1"; expected="$2"; shift 2
  rm -f "$TMP/wd.log"
  out=$(env MOCK_MODE=1 MOCK_HEARTBEAT_FILE="$TMP/hb.json" \
            MOCK_SPEND_USD_FILE="$TMP/spend_usd" MOCK_SPEND_GB_FILE="$TMP/spend_gb" \
            LOG_FILE="$TMP/wd.log" POLL_INTERVAL_S=1 STALE_AFTER_S=5 \
            FAIL_STREAK_MAX=2 INVOICE_EVERY_N=2 \
            USD_CEILING=2.0 GB_CEILING=1.0 \
            "$@" timeout 10 "$WD" 2>&1)
  if echo "$out" | grep -q "$expected"; then
    echo "  [PASS] $name"; PASS=$((PASS+1))
  else
    echo "  [FAIL] $name -- expected '$expected', got: $out"; FAIL=$((FAIL+1))
  fi
}

echo "=== QA: watchdog.sh (mock mode, zero cost) ==="

echo "1) healthy running heartbeat -> watchdog should NOT kill within the window"
hb running 0.5 0.2 "$(date -u +%s)"
out=$(env MOCK_MODE=1 MOCK_HEARTBEAT_FILE="$TMP/hb.json" LOG_FILE="$TMP/wd.log" \
  POLL_INTERVAL_S=1 STALE_AFTER_S=5 USD_CEILING=2.0 GB_CEILING=1.0 \
  timeout 4 "$WD" 2>&1)
if echo "$out" | grep -qi "KILLED"; then
  echo "  [FAIL] healthy heartbeat was killed unexpectedly: $out"; FAIL=$((FAIL+1))
else
  echo "  [PASS] healthy heartbeat: watchdog stayed alive, no kill"; PASS=$((PASS+1))
fi

echo "2) stale heartbeat (old timestamp) -> must kill"
hb running 0.5 0.2 "$(( $(date -u +%s) - 9999 ))"
run_case "stale heartbeat" "stale"

echo "3) self-reported spend over USD ceiling -> must kill"
hb running 5.0 0.2 "$(date -u +%s)"
run_case "over USD ceiling" "exceeds ceiling"

echo "4) self-reported GB over ceiling -> must kill"
hb running 0.5 50.0 "$(date -u +%s)"
run_case "over GB ceiling" "exceeds ceiling"

echo "5) loop reports crashed -> must kill"
hb crashed 0.5 0.2 "$(date -u +%s)"
run_case "crashed status" "reported status=crashed"

echo "6) loop reports systemic_failure -> must kill"
hb systemic_failure 0.5 0.2 "$(date -u +%s)"
run_case "systemic_failure status" "reported status=systemic_failure"

echo "7) loop reports done -> clean exit, no kill/destroy"
hb done 0.5 0.2 "$(date -u +%s)"
out=$(env MOCK_MODE=1 MOCK_HEARTBEAT_FILE="$TMP/hb.json" LOG_FILE="$TMP/wd.log" \
  POLL_INTERVAL_S=1 STALE_AFTER_S=5 USD_CEILING=2.0 GB_CEILING=1.0 \
  timeout 4 "$WD" 2>&1)
if echo "$out" | grep -q "^DONE$" && ! echo "$out" | grep -qi "KILLED"; then
  echo "  [PASS] done status: clean exit, no kill"; PASS=$((PASS+1))
else
  echo "  [FAIL] done status: expected clean DONE exit, got: $out"; FAIL=$((FAIL+1))
fi

echo "8) heartbeat file missing entirely -> fail-safe kill after FAIL_STREAK_MAX"
rm -f "$TMP/hb.json"
run_case "missing heartbeat (fail-safe)" "cannot verify heartbeat"

echo "9) systemic_failure -> copy_back_outputs runs BEFORE destroy (the fix for"
echo "   tonight's data-loss race: watchdog itself preserves outputs, doesn't"
echo "   just race an operator's manual copy-back attempt)"
hb systemic_failure 0.5 0.2 "$(date -u +%s)"
rm -f "$TMP/copyback_marker"
env MOCK_MODE=1 MOCK_HEARTBEAT_FILE="$TMP/hb.json" LOG_FILE="$TMP/wd.log" \
    MOCK_COPYBACK_MARKER="$TMP/copyback_marker" \
    POLL_INTERVAL_S=1 STALE_AFTER_S=5 USD_CEILING=2.0 GB_CEILING=1.0 \
    timeout 4 "$WD" > /dev/null 2>&1
if [ -f "$TMP/copyback_marker" ]; then
  echo "  [PASS] copy-back invoked before destroy on systemic_failure kill"; PASS=$((PASS+1))
else
  echo "  [FAIL] copy-back was NOT invoked before destroy"; FAIL=$((FAIL+1))
fi

echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
