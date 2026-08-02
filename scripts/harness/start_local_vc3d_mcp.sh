#!/usr/bin/env bash
# Start the installed VC3D MCP server safely on loopback only.
set -euo pipefail

install_root="$HOME/.local/share/campaign-x/villa-mcp/volume-cartographer"
server="$install_root/build-macos/bin/vc_mcp_server"
grow="$install_root/build-macos/bin/vc_grow_seg_from_seed"
analysis_python="$HOME/.local/share/campaign-x/villa-mcp/analysis-venv/bin/python"
adapter_root="$install_root/apps/VC3D/mcp"
runtime="$HOME/.local/state/campaign-x-vc3d"
token_file="$runtime/token"

[[ -x "$server" ]] || { echo "VC3D MCP server is not built: $server" >&2; exit 1; }
[[ -x "$grow" ]] || { echo "VC3D growth worker is not built: $grow" >&2; exit 1; }
[[ -x "$analysis_python" ]] || { echo "VC3D analysis environment is not installed: $analysis_python" >&2; exit 1; }

mkdir -p "$runtime/jobs"
chmod 700 "$runtime" "$runtime/jobs"
if [[ ! -s "$token_file" ]]; then
  umask 077
  openssl rand -hex 32 > "$token_file"
fi
chmod 600 "$token_file"

export VC_MCP_TRANSPORT=streamable-http
export VC_MCP_HOST=127.0.0.1
export VC_MCP_PORT=18080
export VC_MCP_AUTH_TOKEN="$(<"$token_file")"
export VC_MCP_WORK_ROOT="$runtime/jobs"
export VC_MCP_GROW_EXECUTABLE="$grow"
export VC_MCP_ANALYSIS_PYTHON="$analysis_python"
export VC_MCP_VOLUME_STAGER="$adapter_root/volume_stager.py"
export VC_MCP_SURFACE_BUNDLE_ADAPTER="$adapter_root/surface_bundle_adapter.py"
export VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER="$adapter_root/structural_evidence_adapter.py"
export VC_MCP_EVIDENCE_FUSION_ADAPTER="$adapter_root/evidence_fusion_adapter.py"
export VC_MCP_REVIEW_ADAPTER="$adapter_root/review_adapter.py"
exec "$server"
