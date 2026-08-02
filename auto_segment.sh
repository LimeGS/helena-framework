#!/bin/sh
# Campaign X autonomous segmentation entrypoint.
#
# Quick start on a prepared worker:
#   ./auto_segment.sh
#
# Optional:
#   RUN_ROOT=/path/to/run WORKER_SLOT=1 ./auto_segment.sh
#   FLEET_DB=postgres-env://SEGMENT_FLEET_DATABASE_URL \
#   ARTIFACT_ROOT=s3://bucket/prefix RUN_ROOT=/srv/helena/runs/current \
#   SEGMENT_FLEET_POSTGRES_ENV_FILE=/secure/postgres.env ./auto_segment.sh
#   ./auto_segment.sh status
#   ./auto_segment.sh foreground
#
# The script never stores credentials. V2 queues may use the direct OpenRouter
# Fusion planner when an API key is injected; otherwise the matching frozen
# deterministic planner keeps the geometry queue moving.

set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
stage_scripts="$repo_root/framework/stages/01-segmentation/scripts"
fleet_cli="$stage_scripts/helena_segment_search_fleet.py"
supervisor="$stage_scripts/run_segment_fleet_watch.sh"
action="${1:-start}"

fleet_cli() {
  if [ "${FLEET_DB:-}" = "postgres-env://SEGMENT_FLEET_DATABASE_URL" ]; then
    : "${SEGMENT_FLEET_POSTGRES_ENV_FILE:?required for PostgreSQL mode}"
    "$stage_scripts/run_with_postgres_control_plane.sh" \
      "${PYTHON_BIN:-python3}" "$fleet_cli" "$@"
  else
    "${PYTHON_BIN:-python3}" "$fleet_cli" "$@"
  fi
}

find_active_run() {
  latest_db=""
  for candidate in "$repo_root"/workspace/campaigns/*/runs/*/control/fleet.sqlite; do
    [ -f "$candidate" ] || continue
    candidate_status="$(python3 "$fleet_cli" status --db "$candidate" 2>/dev/null || true)"
    printf '%s\n' "$candidate_status" | grep -Eq '"PENDING": [1-9][0-9]*' || continue
    if [ -z "$latest_db" ] || [ "$candidate" -nt "$latest_db" ]; then
      latest_db="$candidate"
    fi
  done
  [ -n "$latest_db" ] || return 1
  dirname "$(dirname "$latest_db")"
}

resolve_executable() {
  explicit="$1"
  shift
  if [ -n "$explicit" ] && [ -x "$explicit" ]; then
    printf '%s\n' "$explicit"
    return 0
  fi
  for candidate in "$@"; do
    [ -x "$candidate" ] || continue
    printf '%s\n' "$candidate"
    return 0
  done
  return 1
}

run_root="${RUN_ROOT:-}"
if [ -z "$run_root" ]; then
  run_root="$(find_active_run)" || {
    echo "No fleet.sqlite was found. Set RUN_ROOT to a bootstrapped segmentation run." >&2
    exit 2
  }
fi
case "$run_root" in
  /*) ;;
  *) run_root="$repo_root/$run_root" ;;
esac

fleet_db="${FLEET_DB:-$run_root/control/fleet.sqlite}"
case "$fleet_db" in
  postgres-env://*) ;;
  *) [ -f "$fleet_db" ] || { echo "Fleet database is missing: $fleet_db" >&2; exit 2; } ;;
esac
[ -x "$supervisor" ] || { echo "Supervisor is missing or not executable: $supervisor" >&2; exit 2; }
[ -f "$fleet_cli" ] || { echo "Fleet CLI is missing: $fleet_cli" >&2; exit 2; }

worker_slot="${WORKER_SLOT:-0}"
worker_id="${WORKER_ID:-$(hostname)-gpu${worker_slot}}"
pid_file="$run_root/control/auto-segment-${worker_slot}.pid"
log_file="$run_root/control/auto-segment-${worker_slot}.log"

if [ "$action" = "status" ]; then
  FLEET_DB="$fleet_db" fleet_cli status --db "$fleet_db"
  if [ -s "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf 'supervisor: RUNNING pid=%s worker=%s\n' "$(cat "$pid_file")" "$worker_id"
  else
    printf 'supervisor: NOT_RUNNING worker=%s\n' "$worker_id"
  fi
  exit 0
fi

if [ "$action" != "start" ] && [ "$action" != "foreground" ]; then
  echo "usage: ./auto_segment.sh [start|status|foreground]" >&2
  exit 2
fi

queue_status="$(FLEET_DB="$fleet_db" fleet_cli status --db "$fleet_db")"
if ! printf '%s\n' "$queue_status" | grep -Eq '"PENDING": [1-9][0-9]*'; then
  echo "This queue has no pending tasks; bootstrap or select another RUN_ROOT." >&2
  exit 3
fi

vc3d_grow="$(resolve_executable "${VC3D_GROW_BINARY:-}" \
  /workspace/villa-phase3/build-phase3-gcc13/bin/vc_grow_seg_from_seed \
  /workspace/villa/build/bin/vc_grow_seg_from_seed)" || {
  echo "VC3D grow executable was not found; set VC3D_GROW_BINARY." >&2
  exit 2
}
vc_mcp_server="$(resolve_executable "${VC_MCP_SERVER_BINARY:-}" \
  /workspace/campaign-x-vc3d-mcp-linux-build/bin/vc_mcp_server)" || {
  echo "VC3D MCP server was not found; set VC_MCP_SERVER_BINARY." >&2
  exit 2
}

vc_mcp_token_file="${VC_MCP_TOKEN_FILE:-/root/.local/state/campaign-x-vc3d/token}"
[ -r "$vc_mcp_token_file" ] || {
  echo "MCP token file is not readable; set VC_MCP_TOKEN_FILE." >&2
  exit 2
}

opencode_bin="${OPENCODE_BIN:-}"
if [ -z "$opencode_bin" ]; then
  opencode_bin="$(command -v opencode 2>/dev/null || true)"
fi
if [ -z "$opencode_bin" ] && [ -x /root/.opencode/bin/opencode ]; then
  opencode_bin=/root/.opencode/bin/opencode
fi

planner="${FLEET_PLANNER:-auto}"
if [ "$planner" = "auto" ]; then
  planner_contract="${FLEET_PLANNER_CONTRACT_VERSION:-v1}"
  if [ "$planner_contract" = "v2" ]; then
    planner=deterministic-v2
    if [ -n "${OPENROUTER_API_KEY:-}" ]; then
      planner=cost-aware-v2
    fi
  elif [ "$planner_contract" = "v1" ]; then
    planner=deterministic
  else
    echo "FLEET_PLANNER_CONTRACT_VERSION must be v1 or v2" >&2
    exit 2
  fi
fi
if { [ "$planner" = "opencode" ] || [ "$planner" = "opencode-v2" ]; } && { [ -z "$opencode_bin" ] || [ ! -x "$opencode_bin" ]; }; then
  echo "FLEET_PLANNER=$planner but OpenCode is unavailable; set OPENCODE_BIN." >&2
  exit 2
fi

export HELENA_REPO_ROOT="$repo_root"
export RUN_ROOT="$run_root"
export FLEET_DB="$fleet_db"
export WORKER_ID="$worker_id"
export FLEET_PLANNER="$planner"
export OPENCODE_BIN="${opencode_bin:-opencode}"
export OPENCODE_MODEL="${OPENCODE_MODEL:-}"
export VC3D_GROW_BINARY="$vc3d_grow"
export VC_MCP_SERVER_BINARY="$vc_mcp_server"
export VC_MCP_TOKEN_FILE="$vc_mcp_token_file"
export VC_MCP_GROW_EXECUTABLE="${VC_MCP_GROW_EXECUTABLE:-$vc3d_grow}"
export VC_MCP_WORK_ROOT="${VC_MCP_WORK_ROOT:-/root/.local/state/campaign-x-vc3d/jobs}"
export VC_MCP_RUNTIME_DIR="${VC_MCP_RUNTIME_DIR:-/root/.local/state/campaign-x-vc3d}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$repo_root/workspace/surfaces/campaign-x}"
export SURFACE_QC_PROFILE_ID="${SURFACE_QC_PROFILE_ID:-surface-qc-gp-scroll1-ct-fiber-v3@1.0.0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$worker_slot}"

# Fail before claiming a task if this worker cannot inspect and finalize the
# TIFXYZ it is about to grow.  These imports happen late in the Python worker,
# so omitting this preflight used to waste a complete VC3D grow before exposing
# a missing runtime dependency.
required_python_modules="numpy tifffile"
case "$ARTIFACT_ROOT" in
  s3://*) required_python_modules="$required_python_modules boto3" ;;
esac
case "$FLEET_DB" in
  postgres-env://*) required_python_modules="$required_python_modules psycopg2" ;;
esac
"$PYTHON_BIN" -c '
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing worker Python modules: " + ", ".join(missing))
' $required_python_modules

printf 'Campaign X auto-segment: run=%s worker=%s planner=%s\n' "$run_root" "$worker_id" "$planner"

if [ "${AUTO_SEGMENT_DRY_RUN:-0}" = "1" ]; then
  printf 'dry-run: supervisor=%s db=%s grow=%s mcp=%s\n' "$supervisor" "$fleet_db" "$vc3d_grow" "$vc_mcp_server"
  exit 0
fi

run_supervisor() {
  if [ "$fleet_db" = "postgres-env://SEGMENT_FLEET_DATABASE_URL" ]; then
    exec "$stage_scripts/run_with_postgres_control_plane.sh" "$supervisor"
  fi
  exec "$supervisor"
}

if [ "$action" = "foreground" ]; then
  run_supervisor
fi

if [ -s "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  printf 'Already running: pid=%s log=%s\n' "$(cat "$pid_file")" "$log_file"
  exit 0
fi

mkdir -p "$run_root/control"
if [ "$fleet_db" = "postgres-env://SEGMENT_FLEET_DATABASE_URL" ]; then
  nohup "$stage_scripts/run_with_postgres_control_plane.sh" "$supervisor" > "$log_file" 2>&1 </dev/null &
else
  nohup "$supervisor" > "$log_file" 2>&1 </dev/null &
fi
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" > "$pid_file"
printf 'Started: pid=%s log=%s\n' "$supervisor_pid" "$log_file"
