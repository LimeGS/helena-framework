#!/bin/sh
# Wait for the authoritative surface-QC queue to drain, then dedicate the
# declared local GPU slots to bounded Stage 01 segmentation jobs.
#
# This controller is intentionally suitable for a detached byobu/tmux session:
# PostgreSQL and S3 remain authoritative, every child handles at most one task,
# and disconnecting the terminal does not affect leases or completed artifacts.

set -eu

script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
repo_root="${HELENA_REPO_ROOT:-$(CDPATH='' cd -- "$script_dir/../../../.." && pwd)}"
python_bin="${PYTHON_BIN:-python3}"
fleet_cli="$script_dir/helena_segment_search_fleet.py"
postgres_wrapper="$script_dir/run_with_postgres_control_plane.sh"
bounded_worker="$script_dir/run_segment_fleet_once.sh"
gpu_supervisor="$script_dir/run_gpu_tier_supervisor.py"

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FLEET_DB:?FLEET_DB is required}"
: "${SEGMENT_FLEET_POSTGRES_ENV_FILE:?SEGMENT_FLEET_POSTGRES_ENV_FILE is required}"
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
: "${VC3D_GROW_BINARY:?VC3D_GROW_BINARY is required}"
: "${VC_MCP_SERVER_BINARY:?VC_MCP_SERVER_BINARY is required}"
: "${VC_MCP_TOKEN_FILE:?VC_MCP_TOKEN_FILE is required}"
: "${VC_MCP_GROW_EXECUTABLE:?VC_MCP_GROW_EXECUTABLE is required}"
: "${VC_MCP_WORK_ROOT:?VC_MCP_WORK_ROOT is required}"
: "${VC_MCP_RUNTIME_DIR:?VC_MCP_RUNTIME_DIR is required}"
: "${SURFACE_QC_PROFILE_ID:?SURFACE_QC_PROFILE_ID is required}"

case "$FLEET_DB" in
  postgres-env://SEGMENT_FLEET_DATABASE_URL) ;;
  *)
    printf '%s\n' \
      "run_autosegment_after_qc requires the authoritative PostgreSQL control plane" >&2
    exit 2
    ;;
esac

gpu_slots="${AUTO_SEGMENT_GPU_SLOTS:-0,1}"
minimum_vram_gib="${AUTO_SEGMENT_MINIMUM_VRAM_GIB:-5.5}"
minimum_free_gib="${AUTO_SEGMENT_MINIMUM_FREE_GIB:-20}"
qc_poll_seconds="${AUTO_SEGMENT_QC_POLL_SECONDS:-30}"
idle_seconds="${AUTO_SEGMENT_IDLE_SECONDS:-30}"
receipt_history="${AUTO_SEGMENT_RECEIPT_HISTORY:-256}"
worker_prefix="${AUTO_SEGMENT_WORKER_PREFIX:-$(hostname)-byobu-autosegment}"
control_root="${AUTO_SEGMENT_CONTROL_ROOT:-$RUN_ROOT/control}"
receipt="${AUTO_SEGMENT_RECEIPT:-$control_root/gpu-supervisor.json}"
drain_file="${AUTO_SEGMENT_DRAIN_FILE:-$control_root/gpu-supervisor.drain}"
status_file="$control_root/authoritative-status.json"

mkdir -p "$RUN_ROOT/attempts" "$control_root"
export HELENA_REPO_ROOT="$repo_root"

for required in \
  "$fleet_cli" "$postgres_wrapper" "$bounded_worker" "$gpu_supervisor" \
  "$SEGMENT_FLEET_POSTGRES_ENV_FILE" "$VC3D_GROW_BINARY" \
  "$VC_MCP_SERVER_BINARY" "$VC_MCP_TOKEN_FILE"
do
  if [ ! -r "$required" ]; then
    printf '%s\n' "required autosegmentation input is not readable: $required" >&2
    exit 2
  fi
done

qc_active_count() {
  "$postgres_wrapper" \
    "$python_bin" "$fleet_cli" status --db "$(case "$FLEET_DB" in postgres*) echo postgres-env://FLEET_DB;; *) echo "$FLEET_DB";; esac)" >"$status_file"
  "$python_bin" -c '
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    status = json.load(handle)
states = status.get("qc_job_states", {})
print(int(states.get("PENDING", 0)) + int(states.get("CLAIMED", 0)))
' "$status_file"
}

pending_segmentation_count() {
  "$python_bin" -c '
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    status = json.load(handle)
print(int(status.get("tasks", {}).get("PENDING", 0)))
' "$status_file"
}

printf '%s\n' \
  "Helena Framework detached autosegmentation is waiting for surface QC to drain."
while :; do
  active="$(qc_active_count)"
  case "$active" in
    ''|*[!0-9]*)
      printf '%s\n' "authoritative QC active count is invalid: $active" >&2
      exit 2
      ;;
  esac
  [ "$active" -eq 0 ] && break
  printf '%s\n' "surface QC still active: $active job(s)"
  sleep "$qc_poll_seconds"
done

# The queue can reach zero shortly before a container exits and releases its
# CUDA context. Wait for all compute processes rather than racing the final
# TimeSformer cleanup. This host is dedicated to Helena Framework.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | grep -Eq '[0-9]'; do
  printf '%s\n' "surface QC is drained; waiting for CUDA contexts to exit"
  sleep "$qc_poll_seconds"
done

pending="$(pending_segmentation_count)"
printf '%s\n' \
  "surface QC drained; starting two-slot segmentation with $pending queued task(s)"

# A stale drain marker is not removed automatically. Reusing one must fail
# closed, because it may represent an explicit operator request.
if [ -e "$drain_file" ]; then
  printf '%s\n' "autosegmentation drain marker already exists: $drain_file" >&2
  exit 2
fi

exec "$python_bin" "$gpu_supervisor" run \
  --role always-on \
  --gpu-slots "$gpu_slots" \
  --minimum-vram-gib "$minimum_vram_gib" \
  --work-root "$RUN_ROOT" \
  --minimum-free-disk-gib "$minimum_free_gib" \
  --worker-prefix "$worker_prefix" \
  --receipt "$receipt" \
  --receipt-history "$receipt_history" \
  --drain-file "$drain_file" \
  --idle-seconds "$idle_seconds" \
  -- \
  "$postgres_wrapper" "$bounded_worker"
