#!/usr/bin/env bash
# Deterministic (no LLM) local watchdog for the segmentation campaign loop.
# Runs on the LOCAL machine, never on the rented instance -- if it ran there
# it would die along with anything that takes the instance down, defeating
# its purpose. Polls the remote heartbeat via a single short-timeout SSH
# command, and independently checks REAL vast.ai spend on a slower cadence.
# Kills the loop + destroys the instance on any ceiling breach, a stale/
# missing heartbeat, or repeated inability to verify state (fail-safe:
# assume the worst, don't assume fine).
#
# Config via env vars (all required unless MOCK_MODE=1):
#   INSTANCE_ID       vast.ai instance id (for `vastai destroy instance`)
#   SSH_HOST SSH_PORT SSH_KEY   how to reach the instance
#   USD_CEILING       hard $ limit for this increment
#   GB_CEILING        hard GB limit for this increment
#   POLL_INTERVAL_S   seconds between heartbeat polls (default 75)
#   INVOICE_EVERY_N   check real vast.ai invoice every Nth poll (default 5)
#   STALE_AFTER_S     heartbeat older than this = stuck (default 600)
#   FAIL_STREAK_MAX   consecutive check failures before fail-safe kill (default 3)
#   LOCAL_COPYBACK_DIR  if set, scp manifest/results/logs/meshes from
#                     ~/campaign on the instance to this local dir BEFORE
#                     destroying it on any kill path (crashed/systemic_failure/
#                     stale/over-ceiling) -- closes the race where the
#                     watchdog's own kill could otherwise beat an operator's
#                     manual copy-back and lose the run's data for good.
#                     Bounded (60s + 90s timeouts) so a huge/hung copy can't
#                     block the kill indefinitely; a failed copy is logged,
#                     never blocks the destroy.
#   MOCK_MODE=1       read heartbeat/spend from local files instead of
#                     SSH/vastai -- for local QA, no real instance needed:
#     MOCK_HEARTBEAT_FILE, MOCK_SPEND_USD_FILE, MOCK_SPEND_GB_FILE
#   LOG_FILE          where the watchdog logs its own decisions (JSON lines)

set -uo pipefail

POLL_INTERVAL_S="${POLL_INTERVAL_S:-75}"
INVOICE_EVERY_N="${INVOICE_EVERY_N:-5}"
STALE_AFTER_S="${STALE_AFTER_S:-600}"
FAIL_STREAK_MAX="${FAIL_STREAK_MAX:-3}"
LOG_FILE="${LOG_FILE:-./watchdog.log}"
MOCK_MODE="${MOCK_MODE:-0}"

wlog() {  # wlog LEVEL msg
  printf '{"ts":"%s","level":"%s","msg":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$LOG_FILE"
}

fetch_heartbeat_json() {
  if [ "$MOCK_MODE" = "1" ]; then
    cat "$MOCK_HEARTBEAT_FILE" 2>/dev/null
  else
    timeout 15 ssh -i "$SSH_KEY" -p "$SSH_PORT" \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 -o BatchMode=yes \
      "root@$SSH_HOST" 'cat ~/campaign/heartbeat.json' 2>/dev/null
  fi
}

fetch_real_spend_usd() {
  if [ "$MOCK_MODE" = "1" ]; then
    cat "$MOCK_SPEND_USD_FILE" 2>/dev/null
  else
    vastai show invoices-v1 --raw 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    rows=d if isinstance(d,list) else d.get("results",[])
    total=sum(float(r.get("amount",0)) for r in rows if str(r.get("instance_id"))=="'"$INSTANCE_ID"'")
    print(round(total,4))
except Exception:
    print("")'
  fi
}

copy_back_outputs() {
  if [ "$MOCK_MODE" = "1" ]; then
    [ -n "${MOCK_COPYBACK_MARKER:-}" ] && : > "$MOCK_COPYBACK_MARKER"
    return
  fi
  if [ -z "${LOCAL_COPYBACK_DIR:-}" ]; then
    wlog "WARN" "no LOCAL_COPYBACK_DIR set, skipping copy-back before destroy (DATA WILL BE LOST)"
    return
  fi
  mkdir -p "$LOCAL_COPYBACK_DIR"
  wlog "INFO" "copying back outputs to $LOCAL_COPYBACK_DIR before destroy"
  timeout 60 scp -i "$SSH_KEY" -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r \
    "root@$SSH_HOST:~/campaign/manifest.json" "root@$SSH_HOST:~/campaign/results" \
    "root@$SSH_HOST:~/campaign/campaign.log" "root@$SSH_HOST:~/campaign/setup.log" \
    "root@$SSH_HOST:~/campaign/heartbeat.json" "root@$SSH_HOST:~/campaign/llm_calls.log" \
    "$LOCAL_COPYBACK_DIR/" >> "$LOG_FILE" 2>&1
  small_rc=$?
  # meshes can be large -- its own bounded timeout so it can't block the kill
  # sequence indefinitely; a partial/failed mesh copy still leaves the small
  # files (manifest, results, logs) already copied above.
  timeout 90 scp -i "$SSH_KEY" -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r \
    "root@$SSH_HOST:~/campaign/meshes" "$LOCAL_COPYBACK_DIR/" >> "$LOG_FILE" 2>&1
  mesh_rc=$?
  wlog "INFO" "copy-back attempt done (small_files_rc=$small_rc meshes_rc=$mesh_rc), see $LOCAL_COPYBACK_DIR"
}

kill_and_destroy() {
  reason="$1"
  wlog "ERROR" "KILLING: $reason"
  if [ "$MOCK_MODE" != "1" ]; then
    copy_back_outputs
    vastai destroy instance "$INSTANCE_ID" -y >> "$LOG_FILE" 2>&1
    sleep 5
    remaining=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null)
    wlog "INFO" "post-destroy instance count: ${remaining:-unknown}"
  else
    copy_back_outputs
  fi
  echo "KILLED: $reason"
  exit 1
}

wlog "INFO" "watchdog starting, mock_mode=$MOCK_MODE usd_ceiling=${USD_CEILING:-} gb_ceiling=${GB_CEILING:-}"

fail_streak=0
poll_n=0
while true; do
  poll_n=$((poll_n + 1))
  hb_json=$(fetch_heartbeat_json)

  if [ -z "$hb_json" ]; then
    fail_streak=$((fail_streak + 1))
    wlog "WARN" "heartbeat fetch failed (streak=$fail_streak)"
    if [ "$fail_streak" -ge "$FAIL_STREAK_MAX" ]; then
      kill_and_destroy "cannot verify heartbeat after $FAIL_STREAK_MAX attempts (fail-safe)"
    fi
    sleep "$POLL_INTERVAL_S"
    continue
  fi
  fail_streak=0

  # parse fields we need without a heavy dependency
  read -r hb_ts hb_status hb_spent hb_gb <<EOF
$(python3 -c '
import json,sys
d=json.loads(sys.argv[1])
print(d.get("timestamp",""), d.get("status",""), d.get("spent_usd_estimate",0), d.get("fetched_gb_estimate",0))
' "$hb_json")
EOF

  now_epoch=$(date -u +%s)
  hb_epoch=$(python3 -c "
import time,sys
try:
    print(int(time.mktime(time.strptime(sys.argv[1], '%Y-%m-%dT%H:%M:%SZ')) - time.timezone))
except Exception:
    print(0)
" "$hb_ts")
  age=$((now_epoch - hb_epoch))

  if [ "$age" -gt "$STALE_AFTER_S" ]; then
    kill_and_destroy "heartbeat stale (${age}s > ${STALE_AFTER_S}s), last status=$hb_status"
  fi

  if [ "$hb_status" = "crashed" ] || [ "$hb_status" = "systemic_failure" ]; then
    kill_and_destroy "loop reported status=$hb_status"
  fi
  if [ "$hb_status" = "done" ]; then
    wlog "INFO" "loop reported done, watchdog exiting cleanly (no kill needed)"
    echo "DONE"
    exit 0
  fi

  # proactive ceilings from the loop's OWN self-reported estimate (fast,
  # no invoice lag -- this is the primary check, see PLAN doc)
  over_usd=$(python3 -c "print(1 if float('$hb_spent' or 0) > float('${USD_CEILING:-1e18}') else 0)")
  over_gb=$(python3 -c "print(1 if float('$hb_gb' or 0) > float('${GB_CEILING:-1e18}') else 0)")
  if [ "$over_usd" = "1" ]; then
    kill_and_destroy "self-reported spend \$$hb_spent exceeds ceiling \$${USD_CEILING:-}"
  fi
  if [ "$over_gb" = "1" ]; then
    kill_and_destroy "self-reported download ${hb_gb}GB exceeds ceiling ${GB_CEILING:-}GB"
  fi

  # slower, independent real-spend audit (invoice data may lag -- see PLAN doc,
  # step 2 explicitly measures this lag before it's trusted as authoritative)
  if [ $((poll_n % INVOICE_EVERY_N)) -eq 0 ]; then
    real_usd=$(fetch_real_spend_usd)
    if [ -n "$real_usd" ]; then
      over_real=$(python3 -c "print(1 if float('$real_usd' or 0) > float('${USD_CEILING:-1e18}') else 0)")
      wlog "INFO" "invoice audit: real_spend=\$${real_usd} self_reported=\$${hb_spent}"
      if [ "$over_real" = "1" ]; then
        kill_and_destroy "REAL invoice spend \$${real_usd} exceeds ceiling \$${USD_CEILING:-} (self-report said \$${hb_spent})"
      fi
    else
      wlog "WARN" "invoice audit fetch failed, relying on self-reported figure this cycle"
    fi
  fi

  wlog "INFO" "poll ok: status=$hb_status spent=\$${hb_spent} gb=${hb_gb} age=${age}s"
  sleep "$POLL_INTERVAL_S"
done
