#!/bin/sh
# Sequential surface-QC supervisor for several immutable fleet databases.
#
# Each database keeps its own transactional QC leases and evidence bindings.
# This wrapper only schedules one `qc run --max-jobs 1` child at a time, so a
# single GPU can drain historical or target-specific queues without launching
# concurrent TimeSformer jobs.  The database list contains no credentials and
# is reread on every cycle, allowing an operator to add another queue without
# restarting a long scientific job.

set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
stage_dir="$(CDPATH= cd -- "$script_dir/.." && pwd)"
repo_root="${HELENA_REPO_ROOT:-$(CDPATH= cd -- "$stage_dir/../../.." && pwd)}"

: "${FLEET_DB_LIST_FILE:?FLEET_DB_LIST_FILE is required}"
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
worker_prefix="${QC_WORKER_ID_PREFIX:-$(hostname)-surface-qc-federated}"
poll_seconds="${QC_IDLE_SECONDS:-20}"
timeout_seconds="${QC_TIMEOUT_SECONDS:-7200}"
lease_seconds="${QC_LEASE_SECONDS:-1800}"
retry_delay_seconds="${QC_RETRY_DELAY_SECONDS:-300}"
max_cycles="${QC_FEDERATED_MAX_CYCLES:-0}"

case "$max_cycles" in
  ''|*[!0-9]*)
    echo "QC_FEDERATED_MAX_CYCLES must be a non-negative integer" >&2
    exit 2
    ;;
esac

if [ ! -r "$FLEET_DB_LIST_FILE" ]; then
  echo "fleet database list is not readable: $FLEET_DB_LIST_FILE" >&2
  exit 2
fi
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
printf '%s\n' "Helena Framework federated surface-QC supervisor starting: list=$FLEET_DB_LIST_FILE"

cycle=0
while :; do
  cycle=$((cycle + 1))
  database_count=0
  while IFS= read -r fleet_db || [ -n "$fleet_db" ]; do
    case "$fleet_db" in
      ''|'#'*) continue ;;
      postgres-env://*) ;;
      *)
        if [ ! -f "$fleet_db" ]; then
          printf '%s\n' "fleet database does not exist: $fleet_db" >&2
          continue
        fi
        ;;
    esac
    database_count=$((database_count + 1))
    database_id="$(printf '%s' "$fleet_db" | cksum | awk '{print $1}')"
    "$python_bin" "$script_dir/helena_segment_search_fleet.py" qc run \
      --db "$fleet_db" \
      --worker-id "$worker_prefix-$database_id" \
      --run-root "$QC_RUN_ROOT/$database_id" \
      --qc-executable "$SURFACE_QC_EXECUTABLE" \
      --qc-timeout "$timeout_seconds" \
      --lease-seconds "$lease_seconds" \
      --retry-delay-seconds "$retry_delay_seconds" \
      --profile-id "$SURFACE_QC_PROFILE_ID" \
      --max-jobs 1
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
      printf '%s\n' "surface-QC child exited unexpectedly for $fleet_db (rc=$rc)" >&2
    fi
  done < "$FLEET_DB_LIST_FILE"

  if [ "$database_count" -eq 0 ]; then
    printf '%s\n' "fleet database list contains no usable entries" >&2
  fi
  if [ "$max_cycles" -gt 0 ] && [ "$cycle" -ge "$max_cycles" ]; then
    printf '%s\n' "Helena Framework federated surface-QC supervisor completed $cycle cycle(s)"
    exit 0
  fi
  sleep "$poll_seconds"
done
