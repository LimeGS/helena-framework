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
# Either the URL itself, or already the name of the variable holding it: the
# surface-qc stack hands its workers `postgres-env://SEGMENT_FLEET_DATABASE_URL`
# through a secrets wrapper, and wrapping that again names a variable whose
# value is a name. Only a URL gets the indirection.
case "$FLEET_DB" in
  postgres-env://*) FLEET_DB_ARG="$FLEET_DB" ;;
  *) FLEET_DB_ARG="postgres-env://FLEET_DB" ;;
esac
# Passed to every process below by name -- `postgres-env://FLEET_DB` -- and never
# by value. The value is postgresql://user:password@host/db, and `--db "$FLEET_DB"`
# put it in argv, where any user on the host reads it with `ps`. The fleet CLI
# and host_report resolve the name from their own environment, which is this
# one. The panel already passed its DSN this way; the workers now do too.
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

# Only the ones that are directories. ARTIFACT_ROOT is an s3:// URL on this
# fleet, and `mkdir -p s3://bucket/surfaces` asks for a directory called `s3:`
# at the container root -- which root created, silently and pointlessly, on
# every start. As a non-root user it fails instead, and the worker never got
# past this line.
for root in "$RUN_ROOT" "$ARTIFACT_ROOT"; do
  case "$root" in
    /*) mkdir -p "$root" || { echo "cannot create $root" >&2; exit 3; } ;;
    "") ;;
    *) : ;;   # a URL: whatever it names, it is not this filesystem's to create
  esac
done

# Nobody may already be there. This container runs with the host's network, so
# `something accepts a connection on $MCP_PORT` was satisfied by any process on
# the machine -- and the readiness loop below then declared the seed service up
# and sent it every seed request the worker made. A port that is occupied
# before the service starts is a host problem, and it is one to say out loud
# rather than to route seed traffic into.
if /opt/venv/bin/python -c "
import socket,sys
try:
    socket.create_connection(('127.0.0.1', $MCP_PORT), timeout=1).close()
except OSError:
    sys.exit(1)
" 2>/dev/null; then
  echo "something is already listening on 127.0.0.1:$MCP_PORT." >&2
  echo "This worker shares the host's network, so that is not necessarily" >&2
  echo "another Helena worker. Refusing rather than sending seed requests to" >&2
  echo "whatever it is. Free the port, or set MCP_PORT." >&2
  exit 3
fi

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
  # Answering with our token, not merely accepting a connection. The service
  # already authenticates every request, so asking it something is the cheapest
  # available proof that the thing on the port is the child we started and not
  # a stranger that took it in the second between the check above and now.
  if /opt/venv/bin/python -c "
import json,sys,urllib.request
token = open('$TOKEN_FILE').read().strip()
request = urllib.request.Request(
    'http://127.0.0.1:$MCP_PORT/healthz',
    headers={'Authorization': 'Bearer ' + token})
try:
    with urllib.request.urlopen(request, timeout=1) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
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
  --db "$FLEET_DB_ARG" \
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

# Which worker this container is. One image and one entrypoint, because every
# kind that talks to M7 needs what the lines above did: a seed service of its
# own, a token minted for it and kept out of the filesystem, and the refusal to
# claim anything when the service did not start. A second kind that overrode the
# entrypoint to get its own command would skip all of it.
: "${HELENA_WORKER_KIND:=segment}"

# Through the stage's own entry point, not cli.py directly: the CLI uses
# relative imports and only works as part of its package, which this resolves.
FLEET_ENTRY=/workspace/campaign-x/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py

if [ "$HELENA_WORKER_KIND" = "preflight" ]; then
  # Candidate preflights: read-only, ink-blind, and the reason this dispatch
  # exists. It measures through the same seed service the segmentation worker
  # uses, which is why it belongs behind this entrypoint rather than in the
  # panel, where the measurement had neither service nor token.
  exec /opt/venv/bin/python "$FLEET_ENTRY" \
    preflight-worker run \
    --db "$FLEET_DB_ARG" \
    --worker-id "$WORKER_ID" \
    "$@"
fi

exec /opt/venv/bin/python \
  "$FLEET_ENTRY" \
  worker run \
  --db "$FLEET_DB_ARG" \
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
