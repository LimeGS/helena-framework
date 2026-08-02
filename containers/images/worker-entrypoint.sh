#!/bin/sh
# Start the seed service, then claim work against it.
#
# Both in one container because they are one unit of failure: a worker whose
# seed service died claims tasks and fails them, burning an attempt each time.
# When the service stops, so does this, and systemd restarts the pair.
#
# The token is minted here and never written anywhere durable. It authorises
# loopback inside this container and nothing else, so it has no reason to
# outlive the process.
set -eu

: "${FLEET_DB:?FLEET_DB is required: the control plane connection string}"
: "${WORKER_ID:=$(hostname)-$$}"
: "${RUN_ROOT:=/artifacts/attempts}"
: "${ARTIFACT_ROOT:=/artifacts/surfaces}"
: "${QC_PROFILE_ID:?QC_PROFILE_ID is required: a versioned QC profile}"
: "${FLEET_PLANNER:=deterministic}"
: "${MCP_PORT:=18080}"

# Where a surface goes when there is no object storage.
#
# S3 is optional. The default is that the panel host keeps everything in a
# volume of its own -- helena-artifacts, mounted at /artifacts on the panel --
# and a worker on that same host publishes into it rather than to a bucket.
#
# The worker refuses a local --artifact-root on principle, and it is right to:
# a surface written to a worker's own scratch carries that path downstream, so
# QC on another host requeues it forever and it dies with the machine. But when
# the directory IS the panel's volume, the surface is exactly where every other
# phase will look for it. That is the "single-machine run" the refusal names.
#
# Nothing in the container can tell those two cases apart -- both are a local
# path -- so the deployment says which it is. Set only where the artifact mount
# really is the panel's storage; see containers/deploy-platform.sh.
artifact_local_ok=""
case "$ARTIFACT_ROOT" in
  s3://*|http://*|https://*) : ;;
  *)
    if [ -n "${HELENA_ARTIFACTS_ON_PANEL:-}" ]; then
      artifact_local_ok="--allow-local-artifacts"
      echo "artifacts go to the panel volume at $ARTIFACT_ROOT (no object storage configured)" >&2
    fi ;;
esac

# OpenBLAS chooses its kernel from CPUID at load time. On a generic hypervisor
# CPU model -- Proxmox's x86-64-v3, qemu64, kvm64 -- the brand is "QEMU Virtual
# CPU" and the guess lands on a kernel using instructions the model does not
# provide: SIGILL inside libopenblas, minutes into a grow, on a host where the
# binary's own --help ran fine and every feature probe passed.
#
# Naming the kernel removes the guess. Haswell is the x86-64-v3 baseline (AVX2
# and FMA, which is what the rest of this stack needs anyway); Nehalem is SSE4.2
# and runs on a v2 host too. Measured on a v3 QEMU model: unset exits 132,
# either of these exits 0.
#
# This is what makes the image portable rather than the operator's hypervisor
# configuration. Overridable, because a host with a real CPU may do better by
# letting OpenBLAS recognise it.
if [ -z "${OPENBLAS_CORETYPE:-}" ]; then
  if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then
    OPENBLAS_CORETYPE=Haswell
  else
    OPENBLAS_CORETYPE=Nehalem
  fi
  export OPENBLAS_CORETYPE
fi
echo "OpenBLAS kernel: $OPENBLAS_CORETYPE"

MCP_DIR=/workspace/campaign-x/framework/stages/01-segmentation/mcp
TOKEN_FILE=$(mktemp)
head -c 48 /dev/urandom | base64 | tr -d '=+/' > "$TOKEN_FILE"

mkdir -p "$RUN_ROOT" "$ARTIFACT_ROOT"

PYTHONPATH="$MCP_DIR" /opt/venv/bin/python "$MCP_DIR/server.py" \
  --token-file "$TOKEN_FILE" \
  --host 127.0.0.1 --port "$MCP_PORT" \
  ${VOLUME_ROOT:+--volume-root "$VOLUME_ROOT"} &
MCP_PID=$!

# If the service cannot start there is no point claiming anything: without it
# every task ends NO_SEED for a reason that has nothing to do with the scroll.
for _ in $(seq 1 20); do
  if ! kill -0 "$MCP_PID" 2>/dev/null; then
    echo "the seed service exited before it was ready" >&2
    exit 3
  fi
  if /opt/venv/bin/python -c "
import socket,sys
try:
    socket.create_connection(('127.0.0.1', $MCP_PORT), timeout=1).close()
except OSError:
    sys.exit(1)
" 2>/dev/null; then
    break
  fi
  sleep 1
done

# Host inventory, on a timer and beside the worker rather than inside it: a
# worker between tasks is still a host whose free memory is worth knowing, and
# the claim path runs in a locked transaction that should not also be writing to
# the host table. Failures here are logged and ignored -- a machine that cannot
# report its cores must still be able to segment.
/opt/venv/bin/python /workspace/campaign-x/framework/contracts/host_report.py \
  --db "$FLEET_DB" \
  --host-id "${HELENA_HOST_ID:-$(hostname)}" \
  --disk "$ARTIFACT_ROOT" \
  --every "${HELENA_HOST_REPORT_SECONDS:-60}" &
REPORT_PID=$!

trap 'kill "$MCP_PID" "$REPORT_PID" 2>/dev/null || true; rm -f "$TOKEN_FILE"' EXIT INT TERM

export VC_MCP_URL="http://127.0.0.1:${MCP_PORT}/mcp"
export VC_MCP_AUTH_TOKEN="$(cat "$TOKEN_FILE")"
# The token is in the environment of the worker and nowhere on disk that
# outlives it; the file goes when this exits.
rm -f "$TOKEN_FILE"
trap 'kill "$MCP_PID" "$REPORT_PID" 2>/dev/null || true' EXIT INT TERM

# Through the stage's own entry point, not cli.py directly: the CLI uses
# relative imports and only works as part of its package, which this resolves.
exec /opt/venv/bin/python \
  /workspace/campaign-x/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py \
  worker run \
  --db "$FLEET_DB" \
  --worker-id "$WORKER_ID" \
  --run-root "$RUN_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --qc-profile-id "$QC_PROFILE_ID" \
  --repo-root /workspace/campaign-x \
  --seed-provider mcp \
  --planner "$FLEET_PLANNER" \
  --vc3d-binary /opt/campaignx/vc3d/bin/vc_grow_seg_from_seed \
  ${artifact_local_ok} \
  "$@"
