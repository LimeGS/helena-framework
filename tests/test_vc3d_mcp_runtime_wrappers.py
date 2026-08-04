from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework/stages/01-segmentation/scripts"


def test_mcp_runtime_wrappers_are_shell_valid_and_secret_ephemeral() -> None:
    for name in ("start_vc3d_mcp.sh", "with_vc3d_mcp.sh"):
        path = SCRIPTS / name
        subprocess.run(["sh", "-n", str(path)], check=True)
    wrapper = (SCRIPTS / "with_vc3d_mcp.sh").read_text(encoding="utf-8")
    assert "VC_MCP_AUTH_TOKEN" in wrapper
    assert "exec \"$@\"" in wrapper
    assert "MCP_READY.json" not in wrapper


def test_mcp_start_defaults_to_chunk_safe_cap_and_public_receipt() -> None:
    text = (SCRIPTS / "start_vc3d_mcp.sh").read_text(encoding="utf-8")
    assert 'VC_MCP_MAX_SEED_CANDIDATE_CHUNKS="${VC_MCP_MAX_SEED_CANDIDATE_CHUNKS:-27}"' in text
    assert 'receipt_file="$VC_MCP_RUNTIME_DIR/MCP_READY.json"' in text
    assert "SO_REUSEPORT" in text
    assert "endpoint is already occupied by an unmanaged process" in text
    assert "--connect-timeout 1" in text
    receipt = text.split('cat > "$receipt_file" <<EOF', 1)[1].split('EOF', 1)[0]
    assert "VC_MCP_AUTH_TOKEN" not in receipt


def test_mcp_start_is_serialized_for_parallel_gpu_slots() -> None:
    text = (SCRIPTS / "start_vc3d_mcp.sh").read_text(encoding="utf-8")
    assert 'start_lock="$VC_MCP_RUNTIME_DIR/start.lock"' in text
    assert 'if ! mkdir "$start_lock"' in text
    assert 'grep -q "listening" "$log_file"' in text
    assert 'rmdir "$start_lock"' in text


def test_fleet_watch_supervisor_uses_one_child_task_and_shared_lease_db() -> None:
    path = SCRIPTS / "run_segment_fleet_watch.sh"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert 'fleet_db="${FLEET_DB:-$RUN_ROOT/control/fleet.sqlite}"' in text
    assert "--max-jobs 1" in text
    assert '"$script_dir/with_vc3d_mcp.sh"' in text
    assert 'fleet_planner="${FLEET_PLANNER:-cost-aware-v2}"' in text
    assert '--planner "$fleet_planner"' in text
    assert '[ "$fleet_planner" = "opencode-v2" ]' in text
    assert "poolside/laguna" not in text
    assert "VC_MCP_AUTH_TOKEN=" not in text


def test_surface_qc_supervisor_is_single_job_secretless_and_restart_safe() -> None:
    path = SCRIPTS / "run_surface_qc_watch.sh"
    subprocess.run(["sh", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    assert ': "${FLEET_DB:?FLEET_DB is required}"' in text
    assert ': "${SURFACE_QC_EXECUTABLE:?SURFACE_QC_EXECUTABLE is required}"' in text
    assert '"$script_dir/helena_segment_search_fleet.py" qc run' in text
    assert "--max-jobs 1" in text
    assert "AWS_ACCESS_KEY_ID=" not in text
    assert "AWS_SECRET_ACCESS_KEY=" not in text
    assert "sleep \"$poll_seconds\"" in text


def test_surface_qc_pause_waits_for_claim_completion_before_stopping_container() -> None:
    path = SCRIPTS / "pause_surface_qc_after_active.sh"
    subprocess.run(["sh", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    assert "state='CLAIMED'" in text
    assert 'worker_id=\'$worker_id\'' in text
    assert 'sudo docker stop "$container"' in text
    assert "QC_PAUSE_PAIRS" in text
    assert "AWS_ACCESS_KEY_ID=" not in text
    assert "AWS_SECRET_ACCESS_KEY=" not in text


def test_root_auto_segment_entrypoint_is_secretless_and_auto_selects_planner() -> None:
    path = ROOT / "auto_segment.sh"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "find_active_run" in text
    assert "grep -Eq '\"PENDING\": [1-9][0-9]*'" in text
    assert 'planner="${FLEET_PLANNER:-auto}"' in text
    assert "OPENROUTER_API_KEY:-" in text
    assert "planner=cost-aware-v2" in text
    assert "planner=deterministic-v2" in text
    assert "poolside/laguna" not in text
    assert 'nohup "$supervisor"' in text
    assert "VC_MCP_AUTH_TOKEN=" not in text
    assert "sk-or-" not in text


def test_detached_autosegment_waits_for_qc_and_uses_bounded_gpu_workers() -> None:
    controller = (
        ROOT
        / "framework"
        / "stages"
        / "01-segmentation"
        / "scripts"
        / "run_autosegment_after_qc.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "framework"
        / "stages"
        / "01-segmentation"
        / "scripts"
        / "start_autosegment_byobu.sh"
    ).read_text(encoding="utf-8")

    assert 'states.get("PENDING", 0)' in controller
    assert 'states.get("CLAIMED", 0)' in controller
    assert "run_segment_fleet_once.sh" in controller
    assert "run_gpu_tier_supervisor.py" in controller
    assert "--role always-on" in controller
    assert "nvidia-smi --query-compute-apps=pid" in controller
    assert "drain marker already exists" in controller
    assert 'receipt_history="${AUTO_SEGMENT_RECEIPT_HISTORY:-256}"' in controller
    assert '--receipt-history "$receipt_history"' in controller
    assert '--receipt-history "$control_root/receipt-history"' not in controller
    assert "vastai" not in controller
    assert "aws_access_key" not in controller.lower()

    assert "byobu-tmux new-session -d" in launcher
    assert "run_autosegment_after_qc.sh" in launcher
    assert "OPENROUTER_API_KEY" not in launcher
    assert "AWS_SECRET_ACCESS_KEY" not in launcher


def test_container_grow_wrapper_preserves_atomic_output_contract() -> None:
    path = SCRIPTS / "run_vc3d_grow_container.sh"
    subprocess.run(["sh", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    # The registry prefix is derived rather than hardcoded: unset means a bare
    # local tag, which is what an unconfigured host can actually resolve.
    # Hardcoding one broke the worker build -- it looked for the Villa base in a
    # registry that does not exist while the image sat on the host under the
    # real one.
    assert "HELENA_VC3D_IMAGE:-${HELENA_REGISTRY" in text
    assert "helena-vc3d:0.3.2" in text
    assert "HELENA_WORKER_DATA_ROOT:-/srv/helena" in text
    assert "refusing an unmounted absolute path" in text
    assert '-v "$data_root:$data_root"' in text
    assert '/opt/campaignx/vc3d/bin/vc_grow_seg_from_seed "$@"' in text
    assert "--target-dir /output" not in text
