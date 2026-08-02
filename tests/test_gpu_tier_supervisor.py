from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/01-segmentation/scripts/run_gpu_tier_supervisor.py"
)
ONCE = ROOT / "framework/stages/01-segmentation/scripts/run_segment_fleet_once.sh"
SPEC = importlib.util.spec_from_file_location("gpu_tier_supervisor", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def fake_nvidia_smi(tmp_path: Path, *, memory_mib: int = 6144) -> Path:
    return executable(
        tmp_path / "nvidia-smi",
        "#!/bin/sh\n"
        f"printf '%s\\n' '0, NVIDIA GeForce GTX 1660, {memory_mib}, GPU-a' "
        f"'1, NVIDIA GeForce GTX 1660, {memory_mib}, GPU-b'\n",
    )


def test_gpu_inventory_parser_and_slot_validation() -> None:
    rows = MODULE.parse_gpu_inventory(
        "0, NVIDIA GeForce GTX 1660, 6144, GPU-a\n"
        "1, NVIDIA GeForce RTX 3090, 24576, GPU-b\n"
    )
    assert [row.index for row in rows] == ["0", "1"]
    assert rows[1].memory_total_mib == 24576
    assert MODULE.parse_slots("0,1") == ["0", "1"]
    with pytest.raises(ValueError, match="unique"):
        MODULE.parse_slots("0,0")


def test_preflight_is_fail_closed_for_vram_and_disk(tmp_path: Path) -> None:
    nvidia_smi = fake_nvidia_smi(tmp_path)
    result = MODULE.preflight(
        slots=["0", "1"],
        minimum_vram_gib=5.5,
        work_root=tmp_path / "work",
        minimum_free_disk_gib=0.001,
        nvidia_smi=str(nvidia_smi),
    )
    assert result["status"] == "PASSED"
    assert len(result["selected_gpus"]) == 2

    with pytest.raises(RuntimeError, match="VRAM preflight failed"):
        MODULE.preflight(
            slots=["0"],
            minimum_vram_gib=8,
            work_root=tmp_path / "work",
            minimum_free_disk_gib=0,
            nvidia_smi=str(nvidia_smi),
        )
    with pytest.raises(RuntimeError, match="disk preflight failed"):
        MODULE.preflight(
            slots=["0"],
            minimum_vram_gib=5,
            work_root=tmp_path / "work",
            minimum_free_disk_gib=10**9,
            nvidia_smi=str(nvidia_smi),
        )


def test_run_writes_a_terminal_receipt_when_preflight_fails(tmp_path: Path) -> None:
    nvidia_smi = fake_nvidia_smi(tmp_path)
    receipt = tmp_path / "failed-receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--role",
            "always-on",
            "--gpu-slots",
            "0",
            "--minimum-vram-gib",
            "12",
            "--work-root",
            str(tmp_path / "work"),
            "--minimum-free-disk-gib",
            "0",
            "--nvidia-smi",
            str(nvidia_smi),
            "--receipt",
            str(receipt),
            "--drain-file",
            str(tmp_path / "drain"),
            "--",
            "/bin/true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "PREFLIGHT_FAILED"
    assert value["requested_gpu_slots"] == ["0"]
    assert value["lifecycle_policy"]["cloud_instance_actions"] == "PROHIBITED"


def test_two_slots_are_isolated_and_emit_a_bounded_receipt(tmp_path: Path) -> None:
    nvidia_smi = fake_nvidia_smi(tmp_path)
    capture = tmp_path / "capture"
    capture.mkdir()
    child = executable(
        tmp_path / "child.py",
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib\n"
        "slot = os.environ['WORKER_SLOT']\n"
        "pathlib.Path(os.environ['CAPTURE_ROOT'], slot + '.json').write_text(json.dumps({\n"
        "  'cuda': os.environ['CUDA_VISIBLE_DEVICES'],\n"
        "  'worker_id': os.environ['WORKER_ID'],\n"
        "  'role': os.environ['HELENA_GPU_TIER_ROLE'],\n"
        "  'gpu_model': os.environ['HELENA_GPU_MODEL'],\n"
        "  'gpu_vram_gb': os.environ['HELENA_GPU_VRAM_GB'],\n"
        "  'device_index': os.environ['HELENA_CUDA_DEVICE_INDEX'],\n"
        "}))\n",
    )
    receipt = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--role",
            "always-on",
            "--gpu-slots",
            "0,1",
            "--minimum-vram-gib",
            "5.5",
            "--work-root",
            str(tmp_path / "work"),
            "--minimum-free-disk-gib",
            "0",
            "--nvidia-smi",
            str(nvidia_smi),
            "--receipt",
            str(receipt),
            "--drain-file",
            str(tmp_path / "drain"),
            "--maximum-cycles-per-slot",
            "1",
            "--poll-seconds",
            "0.01",
            "--idle-seconds",
            "0.01",
            "--",
            str(child),
        ],
        check=False,
        env={**os.environ, "CAPTURE_ROOT": str(capture)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["schema"] == MODULE.SCHEMA
    assert value["status"] == "DRAINED"
    assert value["drain_reason"] == "MAXIMUM_CYCLES_REACHED"
    assert value["aggregate"] == {
        "children_active": 0,
        "children_failed": 0,
        "children_started": 2,
        "children_succeeded": 2,
    }
    assert "CAPTURE_ROOT" not in json.dumps(value)
    assert set(value["cycles_by_gpu"]) == {"0", "1"}
    for slot in ("0", "1"):
        captured = json.loads((capture / f"{slot}.json").read_text(encoding="utf-8"))
        assert captured["cuda"] == slot
        assert captured["role"] == "always-on"
        assert captured["gpu_model"] == "NVIDIA GeForce GTX 1660"
        assert captured["gpu_vram_gb"] == "6.000"
        assert captured["device_index"] == slot
        assert captured["worker_id"].endswith(f"-always-on-gpu{slot}")


def test_drain_waits_for_the_active_child_instead_of_killing_it(tmp_path: Path) -> None:
    nvidia_smi = fake_nvidia_smi(tmp_path, memory_mib=24576)
    finished = tmp_path / "child-finished"
    child = executable(
        tmp_path / "slow-child.py",
        "#!/usr/bin/env python3\n"
        "import os, pathlib, time\n"
        "time.sleep(0.35)\n"
        "pathlib.Path(os.environ['FINISHED_MARKER']).write_text('complete\\n')\n",
    )
    receipt = tmp_path / "receipt.json"
    drain = tmp_path / "drain"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--role",
            "burst",
            "--gpu-slots",
            "0",
            "--minimum-vram-gib",
            "20",
            "--work-root",
            str(tmp_path / "work"),
            "--minimum-free-disk-gib",
            "0",
            "--nvidia-smi",
            str(nvidia_smi),
            "--receipt",
            str(receipt),
            "--drain-file",
            str(drain),
            "--poll-seconds",
            "0.01",
            "--idle-seconds",
            "0.01",
            "--",
            str(child),
        ],
        env={**os.environ, "FINISHED_MARKER": str(finished)},
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        if receipt.is_file():
            current = json.loads(receipt.read_text(encoding="utf-8"))
            if current["aggregate"]["children_active"] == 1:
                break
        time.sleep(0.01)
    else:
        process.kill()
        raise AssertionError("child never became active")

    assert MODULE.main(["drain", "--drain-file", str(drain)]) == 0
    assert process.wait(timeout=5) == 0
    assert finished.read_text(encoding="utf-8") == "complete\n"
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "DRAINED"
    assert value["drain_reason"] == "DRAIN_FILE"


def test_time_bounded_burst_finishes_its_active_job(tmp_path: Path) -> None:
    nvidia_smi = fake_nvidia_smi(tmp_path, memory_mib=24576)
    finished = tmp_path / "timed-child-finished"
    child = executable(
        tmp_path / "timed-child.py",
        "#!/usr/bin/env python3\n"
        "import os, pathlib, time\n"
        "time.sleep(0.15)\n"
        "pathlib.Path(os.environ['FINISHED_MARKER']).write_text('complete\\n')\n",
    )
    receipt = tmp_path / "timed-receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--role",
            "burst",
            "--gpu-slots",
            "0",
            "--minimum-vram-gib",
            "20",
            "--work-root",
            str(tmp_path / "work"),
            "--minimum-free-disk-gib",
            "0",
            "--nvidia-smi",
            str(nvidia_smi),
            "--receipt",
            str(receipt),
            "--drain-file",
            str(tmp_path / "drain"),
            "--maximum-runtime-seconds",
            "0.03",
            "--poll-seconds",
            "0.01",
            "--idle-seconds",
            "0.01",
            "--",
            str(child),
        ],
        check=False,
        env={**os.environ, "FINISHED_MARKER": str(finished)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert finished.read_text(encoding="utf-8") == "complete\n"
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "DRAINED"
    assert value["drain_reason"] == "MAXIMUM_RUNTIME_REACHED"
    assert value["aggregate"]["children_succeeded"] == 1


def test_bounded_worker_and_supervisor_never_control_cloud_lifecycle() -> None:
    combined = SCRIPT.read_text(encoding="utf-8") + ONCE.read_text(encoding="utf-8")
    assert "--max-jobs 1" in combined
    assert "--cuda-available" in combined
    assert "--gpu-model" in combined
    assert "--gpu-vram-gb" in combined
    assert "--cuda-device-index" in combined
    assert "--terminal-outcomes-exit-zero" in combined
    assert "while :" not in ONCE.read_text(encoding="utf-8")
    for forbidden in ("vastai destroy", "vastai stop", "shutdown -h", "poweroff"):
        assert forbidden not in combined.lower()
