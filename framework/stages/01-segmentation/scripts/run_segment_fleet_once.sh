#!/bin/sh
# Execute at most one Stage 01 segmentation job and exit.  This bounded form
# lets the GPU-tier supervisor drain without interrupting a live VC3D grow.

set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
stage_dir="$(CDPATH= cd -- "$script_dir/.." && pwd)"
repo_root="${HELENA_REPO_ROOT:-$(CDPATH= cd -- "$stage_dir/../../.." && pwd)}"

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${VC3D_GROW_BINARY:?VC3D_GROW_BINARY is required}"
: "${VC_MCP_SERVER_BINARY:?VC_MCP_SERVER_BINARY is required}"
: "${VC_MCP_TOKEN_FILE:?VC_MCP_TOKEN_FILE is required}"
: "${VC_MCP_GROW_EXECUTABLE:?VC_MCP_GROW_EXECUTABLE is required}"
: "${VC_MCP_WORK_ROOT:?VC_MCP_WORK_ROOT is required}"
: "${VC_MCP_RUNTIME_DIR:?VC_MCP_RUNTIME_DIR is required}"
: "${SURFACE_QC_PROFILE_ID:?SURFACE_QC_PROFILE_ID is required}"
: "${HELENA_GPU_MODEL:?HELENA_GPU_MODEL is required}"
: "${HELENA_GPU_VRAM_GB:?HELENA_GPU_VRAM_GB is required}"
: "${HELENA_CUDA_DEVICE_INDEX:?HELENA_CUDA_DEVICE_INDEX is required}"

fleet_db="${FLEET_DB:-$RUN_ROOT/control/fleet.sqlite}"
artifact_root="${ARTIFACT_ROOT:-$repo_root/workspace/surfaces/campaign-x}"
worker_id="${WORKER_ID:-$(hostname)-bounded-$$}"
python_bin="${PYTHON_BIN:-python3}"
opencode_bin="${OPENCODE_BIN:-opencode}"
opencode_model="${OPENCODE_MODEL:-}"
fleet_planner="${FLEET_PLANNER:-deterministic}"
grow_timeout="${FLEET_GROW_TIMEOUT_SECONDS:-1800}"
lease_seconds="${FLEET_LEASE_SECONDS:-900}"
minimum_free_gib="${FLEET_MINIMUM_FREE_GIB:-2}"

case "$fleet_db" in
  postgres-env://SEGMENT_FLEET_DATABASE_URL)
    if [ -z "${SEGMENT_FLEET_DATABASE_URL:-}" ]; then
      printf '%s\n' "PostgreSQL URL is not injected; invoke via run_with_postgres_control_plane.sh" >&2
      exit 2
    fi
    ;;
  postgres-env://*)
    printf '%s\n' "unsupported PostgreSQL indirection: $fleet_db" >&2
    exit 2
    ;;
  *)
    [ -f "$fleet_db" ] || { printf '%s\n' "fleet database does not exist: $fleet_db" >&2; exit 2; }
    ;;
esac

mkdir -p "$RUN_ROOT/attempts" "$RUN_ROOT/control"
case "$artifact_root" in s3://*) ;; *) mkdir -p "$artifact_root" ;; esac

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
  --minimum-free-gib "$minimum_free_gib" \
  --cuda-available \
  --gpu-model "$HELENA_GPU_MODEL" \
  --gpu-vram-gb "$HELENA_GPU_VRAM_GB" \
  --cuda-device-index "$HELENA_CUDA_DEVICE_INDEX" \
  --terminal-outcomes-exit-zero \
  --max-jobs 1

# A migration/drain may need to run a frozen, explicit subset without letting
# this worker claim unrelated work from the shared PostgreSQL queue.
if [ -n "${FLEET_TASK_ID:-}" ]; then
  set -- "$@" --task-id "$FLEET_TASK_ID"
fi
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

exec "$script_dir/with_vc3d_mcp.sh" "$@"
