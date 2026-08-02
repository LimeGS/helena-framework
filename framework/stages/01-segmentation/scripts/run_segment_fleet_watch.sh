#!/bin/sh
# Long-lived, restart-safe single-job supervisor for the distributed
# segmentation fleet.  Several hosts may run this exact script against the
# same PostgreSQL control plane: the atomic lease prevents duplicate claims.
# SQLite remains supported only when all workers are on this one host.
#
# It intentionally runs one task per child process rather than one unbounded
# Python watch process, so completed receipts cannot accumulate in memory and
# a transient OpenCode/MCP failure is isolated to one task.

set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
stage_dir="$(CDPATH= cd -- "$script_dir/.." && pwd)"
repo_root="${HELENA_REPO_ROOT:-$(CDPATH= cd -- "$stage_dir/../../.." && pwd)}"

: "${RUN_ROOT:?RUN_ROOT is required (the shared campaign run directory)}"
: "${VC3D_GROW_BINARY:?VC3D_GROW_BINARY is required}"
: "${VC_MCP_SERVER_BINARY:?VC_MCP_SERVER_BINARY is required}"
: "${VC_MCP_TOKEN_FILE:?VC_MCP_TOKEN_FILE is required}"
: "${VC_MCP_GROW_EXECUTABLE:?VC_MCP_GROW_EXECUTABLE is required}"
: "${VC_MCP_WORK_ROOT:?VC_MCP_WORK_ROOT is required}"
: "${VC_MCP_RUNTIME_DIR:?VC_MCP_RUNTIME_DIR is required}"
: "${SURFACE_QC_PROFILE_ID:?SURFACE_QC_PROFILE_ID is required (versioned semantic profile)}"

fleet_db="${FLEET_DB:-$RUN_ROOT/control/fleet.sqlite}"
artifact_root="${ARTIFACT_ROOT:-$repo_root/workspace/surfaces/campaign-x}"
worker_id="${WORKER_ID:-$(hostname)-$$}"
python_bin="${PYTHON_BIN:-python3}"
opencode_bin="${OPENCODE_BIN:-opencode}"
opencode_model="${OPENCODE_MODEL:-}"
fleet_planner="${FLEET_PLANNER:-cost-aware-v2}"
poll_seconds="${FLEET_IDLE_SECONDS:-30}"
grow_timeout="${FLEET_GROW_TIMEOUT_SECONDS:-1800}"
lease_seconds="${FLEET_LEASE_SECONDS:-900}"

case "$fleet_db" in
  postgres-env://*) ;;
  *)
    if [ ! -f "$fleet_db" ]; then
      echo "fleet database does not exist: $fleet_db" >&2
      exit 2
    fi
    ;;
esac
if [ ! -x "$VC3D_GROW_BINARY" ]; then
  echo "VC3D_GROW_BINARY is not executable: $VC3D_GROW_BINARY" >&2
  exit 2
fi
if [ "$fleet_planner" != "cost-aware-v2" ] && [ "$fleet_planner" != "fusion-v2" ] && [ "$fleet_planner" != "opencode-v2" ] && [ "$fleet_planner" != "opencode" ] && [ "$fleet_planner" != "deterministic-v2" ] && [ "$fleet_planner" != "deterministic" ]; then
  echo "FLEET_PLANNER must be cost-aware-v2, fusion-v2, deterministic-v2, deterministic, opencode-v2, or opencode" >&2
  exit 2
fi
if { [ "$fleet_planner" = "opencode" ] || [ "$fleet_planner" = "opencode-v2" ]; } && ! command -v "$opencode_bin" >/dev/null 2>&1; then
  echo "OpenCode executable is unavailable: $opencode_bin" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/attempts" "$RUN_ROOT/control"
case "$artifact_root" in
  s3://*) ;;
  *) mkdir -p "$artifact_root" ;;
esac
printf '%s\n' "Helena Framework fleet supervisor starting: worker=$worker_id db=$fleet_db"

while :; do
  set -- "$python_bin" \
    "$script_dir/helena_segment_search_fleet.py" worker run \
    --db "$fleet_db" \
    --worker-id "$worker_id" \
    --run-root "$RUN_ROOT/attempts" \
    --artifact-root "$artifact_root" \
    --qc-profile-id "$SURFACE_QC_PROFILE_ID" \
    --repo-root "$repo_root" \
    --seed-provider mcp \
    --planner "$fleet_planner" \
    --opencode "$opencode_bin" \
    --model "$opencode_model" \
    --vc3d-binary "$VC3D_GROW_BINARY" \
    --grow-timeout "$grow_timeout" \
    --lease-seconds "$lease_seconds" \
    --max-jobs 1
  case "${HELENA_SEED_PROBE_SUPPORT:-0}" in
    1|true|TRUE|yes|YES)
      set -- "$@" --seed-probe-support
      ;;
    0|false|FALSE|no|NO|"")
      ;;
    *)
      printf '%s\n' "HELENA_SEED_PROBE_SUPPORT must be 0 or 1" >&2
      exit 2
      ;;
  esac
  "$script_dir/with_vc3d_mcp.sh" "$@"
  rc=$?
  # 0 means an eligible task completed or no task was ready; 2 preserves a
  # per-task terminal receipt.  Both are normal supervisor outcomes.
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
    printf '%s\n' "fleet child exited unexpectedly (rc=$rc); retrying after $poll_seconds seconds" >&2
  fi
  sleep "$poll_seconds"
done
