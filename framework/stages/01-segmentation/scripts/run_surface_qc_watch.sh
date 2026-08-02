#!/bin/sh
# Long-lived, restart-safe single-job supervisor for the automatic surface-QC
# queue. Atomic QC leases make multiple hosts safe, while one child per loop
# bounds memory and isolates an unavailable scientific dependency to one job.

set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
stage_dir="$(CDPATH= cd -- "$script_dir/.." && pwd)"
repo_root="${HELENA_REPO_ROOT:-$(CDPATH= cd -- "$stage_dir/../../.." && pwd)}"

: "${FLEET_DB:?FLEET_DB is required}"
: "${QC_RUN_ROOT:?QC_RUN_ROOT is required}"
: "${SURFACE_QC_EXECUTABLE:?SURFACE_QC_EXECUTABLE is required}"
: "${HELENA_QC_RENDERER:?HELENA_QC_RENDERER is required}"
: "${HELENA_QC_CHECKPOINT:?HELENA_QC_CHECKPOINT is required}"
: "${HELENA_QC_PROFILE:?HELENA_QC_PROFILE is required}"
: "${HELENA_QC_PROFILE_SHA256:?HELENA_QC_PROFILE_SHA256 is required}"
: "${SURFACE_QC_PROFILE_ID:?SURFACE_QC_PROFILE_ID is required}"
: "${HELENA_QC_EVIDENCE_ROOT:?HELENA_QC_EVIDENCE_ROOT is required}"
: "${HELENA_QC_CODE_COMMIT:?HELENA_QC_CODE_COMMIT is required}"

python_bin="${PYTHON_BIN:-python3}"
worker_id="${QC_WORKER_ID:-$(hostname)-surface-qc-$$}"
poll_seconds="${QC_IDLE_SECONDS:-20}"
timeout_seconds="${QC_TIMEOUT_SECONDS:-7200}"
lease_seconds="${QC_LEASE_SECONDS:-1800}"
retry_delay_seconds="${QC_RETRY_DELAY_SECONDS:-300}"

case "$FLEET_DB" in
  postgres-env://*) ;;
  *)
    if [ ! -f "$FLEET_DB" ]; then
      echo "fleet database does not exist: $FLEET_DB" >&2
      exit 2
    fi
    ;;
esac
if [ ! -f "$SURFACE_QC_EXECUTABLE" ]; then
  echo "surface QC executable does not exist: $SURFACE_QC_EXECUTABLE" >&2
  exit 2
fi
if [ ! -x "$HELENA_QC_RENDERER" ]; then
  echo "QC renderer is not executable: $HELENA_QC_RENDERER" >&2
  exit 2
fi
if [ ! -f "$HELENA_QC_CHECKPOINT" ]; then
  echo "QC checkpoint does not exist: $HELENA_QC_CHECKPOINT" >&2
  exit 2
fi
if [ ! -f "$HELENA_QC_PROFILE" ]; then
  echo "QC profile does not exist: $HELENA_QC_PROFILE" >&2
  exit 2
fi

mkdir -p "$QC_RUN_ROOT"
printf '%s\n' "Helena Framework surface-QC supervisor starting: worker=$worker_id db=$FLEET_DB"

while :; do
  "$python_bin" "$script_dir/helena_segment_search_fleet.py" qc run \
    --db "$FLEET_DB" \
    --worker-id "$worker_id" \
    --run-root "$QC_RUN_ROOT" \
    --qc-executable "$SURFACE_QC_EXECUTABLE" \
    --qc-timeout "$timeout_seconds" \
    --lease-seconds "$lease_seconds" \
    --retry-delay-seconds "$retry_delay_seconds" \
    --profile-id "$SURFACE_QC_PROFILE_ID" \
    --max-jobs 1
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
    printf '%s\n' "surface-QC child exited unexpectedly (rc=$rc); retrying after $poll_seconds seconds" >&2
  fi
  sleep "$poll_seconds"
done
