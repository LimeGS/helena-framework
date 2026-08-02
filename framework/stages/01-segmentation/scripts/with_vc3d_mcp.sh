#!/bin/sh
# Start/reuse loopback VC3D MCP and run one worker command with its secret only
# in the child environment. Nothing here persists the token.
set -eu

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$script_dir/start_vc3d_mcp.sh"

: "${VC_MCP_TOKEN_FILE:?VC_MCP_TOKEN_FILE is required}"
: "${VC_MCP_HOST:=127.0.0.1}"
: "${VC_MCP_PORT:=18080}"
[ "$#" -gt 0 ] || { echo "usage: with_vc3d_mcp.sh command [arguments...]" >&2; exit 2; }

# The token is deliberately neither echoed nor written to disk.  It is exposed
# only to the child command that speaks to loopback MCP.
export VC_MCP_AUTH_TOKEN="$(tr -d '\r\n' < "$VC_MCP_TOKEN_FILE")"
export VC_MCP_URL="http://$VC_MCP_HOST:$VC_MCP_PORT/mcp"
exec "$@"
