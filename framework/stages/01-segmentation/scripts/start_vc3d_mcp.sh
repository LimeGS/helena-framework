#!/bin/sh
# Start the local authenticated VC3D MCP service without persisting its token.
#
# This is an operational helper, not a scientific stage. It deliberately does
# not choose a seed, alter a queue, touch a CT volume, or change VC3D options.
# Its only output is a runtime PID/log/public receipt under VC_MCP_RUNTIME_DIR.

set -eu

require_value() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "required environment variable is empty: $name" >&2
    exit 2
  fi
}

require_value VC_MCP_SERVER_BINARY
require_value VC_MCP_TOKEN_FILE
require_value VC_MCP_GROW_EXECUTABLE
require_value VC_MCP_WORK_ROOT
require_value VC_MCP_RUNTIME_DIR

if [ ! -x "$VC_MCP_SERVER_BINARY" ]; then
  echo "VC_MCP_SERVER_BINARY is not executable" >&2
  exit 2
fi
if [ ! -x "$VC_MCP_GROW_EXECUTABLE" ]; then
  echo "VC_MCP_GROW_EXECUTABLE is not executable" >&2
  exit 2
fi
if [ ! -r "$VC_MCP_TOKEN_FILE" ]; then
  echo "VC_MCP_TOKEN_FILE is not readable" >&2
  exit 2
fi

mkdir -p "$VC_MCP_RUNTIME_DIR" "$VC_MCP_WORK_ROOT"
pid_file="$VC_MCP_RUNTIME_DIR/server.pid"
log_file="$VC_MCP_RUNTIME_DIR/server.log"
receipt_file="$VC_MCP_RUNTIME_DIR/MCP_READY.json"
start_lock="$VC_MCP_RUNTIME_DIR/start.lock"

write_receipt() {
  receipt_pid="$1"
  umask 077
  cat > "$receipt_file" <<EOF
{"schema":"campaignx.vc3d_mcp_runtime.v1","status":"READY","pid":$receipt_pid,"endpoint":"http://${VC_MCP_HOST:-127.0.0.1}:${VC_MCP_PORT:-18080}/mcp","transport":"${VC_MCP_TRANSPORT:-streamable-http}","max_seed_candidate_chunks":${VC_MCP_MAX_SEED_CANDIDATE_CHUNKS:-27}}
EOF
}

# Two GPU slots may reach this wrapper at the same instant.  Starting the MCP
# used to have a check-then-launch race: both processes could observe no PID
# and attempt to bind the same authenticated endpoint.  An atomic directory
# lock lets one process start it while every peer waits for the same public
# readiness condition.  No token is written into the lock or receipt.
if ! mkdir "$start_lock" 2>/dev/null; then
  for _ in $(seq 1 30); do
    if [ -f "$pid_file" ] && \
       kill -0 "$(cat "$pid_file")" 2>/dev/null && \
       grep -q "listening" "$log_file" 2>/dev/null; then
      server_pid="$(cat "$pid_file")"
      write_receipt "$server_pid"
      printf '%s\n' "VC3D MCP became ready: pid=$server_pid endpoint=http://${VC_MCP_HOST:-127.0.0.1}:${VC_MCP_PORT:-18080}/mcp"
      exit 0
    fi
    sleep 1
  done
  echo "VC3D MCP start lock remained active without a ready server" >&2
  exit 3
fi
trap 'rmdir "$start_lock" 2>/dev/null || true' EXIT HUP INT TERM

if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  server_pid="$(cat "$pid_file")"
  if grep -q "listening" "$log_file" 2>/dev/null; then
    write_receipt "$server_pid"
    printf '%s\n' "VC3D MCP already running: pid=$server_pid endpoint=http://${VC_MCP_HOST:-127.0.0.1}:${VC_MCP_PORT:-18080}/mcp"
    exit 0
  fi
  echo "VC3D MCP PID exists but is not ready; stop it explicitly before restarting" >&2
  exit 3
fi

# Different experiments may use different runtime directories while sharing
# the same loopback endpoint. The transport permits more than one process to
# bind with SO_REUSEPORT; requests can then alternate between servers carrying
# different bearer tokens. Detect any pre-existing listener before launching
# a second process, even when its PID file lives in another runtime directory.
if command -v curl >/dev/null 2>&1 && \
   curl -sS -o /dev/null --connect-timeout 1 \
     "http://${VC_MCP_HOST:-127.0.0.1}:${VC_MCP_PORT:-18080}/mcp"; then
  echo "VC3D MCP endpoint is already occupied by an unmanaged process; refusing a second listener" >&2
  exit 4
fi

# The secret is read only into this process environment. It is neither echoed
# nor written into the PID file, log, receipt, Git, an image, or object store.
VC_MCP_AUTH_TOKEN="$(tr -d '\r\n' < "$VC_MCP_TOKEN_FILE")"
if [ "${#VC_MCP_AUTH_TOKEN}" -lt 32 ]; then
  echo "VC_MCP_TOKEN_FILE does not contain a valid server token" >&2
  exit 2
fi

export VC_MCP_AUTH_TOKEN
export VC_MCP_GROW_EXECUTABLE
export VC_MCP_WORK_ROOT
export VC_MCP_TRANSPORT="${VC_MCP_TRANSPORT:-streamable-http}"
export VC_MCP_HOST="${VC_MCP_HOST:-127.0.0.1}"
export VC_MCP_PORT="${VC_MCP_PORT:-18080}"
# The server hard-caps this at 27. This default is the chunk-safe replacement
# for the historic eight-chunk broad-probe failure; callers may choose a lower
# cap, but no worker silently falls back to eight.
export VC_MCP_MAX_SEED_CANDIDATE_CHUNKS="${VC_MCP_MAX_SEED_CANDIDATE_CHUNKS:-27}"

nohup "$VC_MCP_SERVER_BINARY" >>"$log_file" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$pid_file"

for _ in $(seq 1 20); do
  if kill -0 "$server_pid" 2>/dev/null && grep -q "listening" "$log_file"; then
    write_receipt "$server_pid"
    printf '%s\n' "VC3D MCP running: pid=$server_pid endpoint=http://$VC_MCP_HOST:$VC_MCP_PORT/mcp"
    exit 0
  fi
  sleep 1
done

echo "VC3D MCP did not become ready; inspect $log_file" >&2
exit 1
