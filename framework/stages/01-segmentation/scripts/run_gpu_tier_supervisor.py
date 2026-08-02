#!/usr/bin/env python3
"""Resource-aware process supervisor for Helena Framework GPU workers.

This is deliberately an operational wrapper around a *one-job* worker command.
It never edits scientific parameters, never controls a cloud instance and never
interrupts an in-flight child during a normal drain.  The durable queue and S3
remain authoritative; this process only assigns one local GPU to each child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "campaignx.gpu_tier_supervisor_receipt.v1"
SECRET_MARKERS = ("TOKEN", "PASSWORD", "SECRET", "CREDENTIAL", "API_KEY")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Gpu:
    index: str
    name: str
    memory_total_mib: int
    uuid: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "uuid": self.uuid,
        }


def parse_gpu_inventory(text: str) -> list[Gpu]:
    result: list[Gpu] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {line!r}")
        try:
            memory = int(fields[2])
        except ValueError as error:
            raise RuntimeError(f"invalid GPU memory value in row: {line!r}") from error
        result.append(Gpu(fields[0], fields[1], memory, fields[3]))
    if not result:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return result


def query_gpu_inventory(executable: str) -> list[Gpu]:
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_inventory(completed.stdout)


def parse_slots(value: str) -> list[str]:
    slots = [part.strip() for part in value.split(",") if part.strip()]
    if not slots:
        raise ValueError("at least one GPU slot is required")
    if len(slots) != len(set(slots)):
        raise ValueError("GPU slots must be unique")
    return slots


def command_identity(command: Sequence[str]) -> dict[str, str]:
    if not command:
        raise ValueError("a one-job worker command is required after --")
    encoded = b"\0".join(part.encode("utf-8") for part in command)
    return {
        "executable": Path(command[0]).name,
        "argv_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def preflight(
    *,
    slots: Sequence[str],
    minimum_vram_gib: float,
    work_root: Path,
    minimum_free_disk_gib: float,
    nvidia_smi: str,
) -> dict[str, Any]:
    inventory = query_gpu_inventory(nvidia_smi)
    by_index = {gpu.index: gpu for gpu in inventory}
    missing = [slot for slot in slots if slot not in by_index]
    if missing:
        raise RuntimeError(f"requested GPU slots are absent: {', '.join(missing)}")
    selected = [by_index[slot] for slot in slots]
    minimum_mib = int(minimum_vram_gib * 1024)
    undersized = [gpu for gpu in selected if gpu.memory_total_mib < minimum_mib]
    if undersized:
        detail = ", ".join(
            f"{gpu.index}:{gpu.memory_total_mib}MiB" for gpu in undersized
        )
        raise RuntimeError(
            f"GPU VRAM preflight failed; minimum is {minimum_mib}MiB: {detail}"
        )

    work_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(work_root)
    free_gib = disk.free / (1024**3)
    if free_gib < minimum_free_disk_gib:
        raise RuntimeError(
            f"disk preflight failed: {free_gib:.3f} GiB free, "
            f"{minimum_free_disk_gib:.3f} GiB required"
        )
    return {
        "status": "PASSED",
        "selected_gpus": [gpu.public_dict() for gpu in selected],
        "minimum_vram_gib": minimum_vram_gib,
        "work_root": str(work_root.resolve()),
        "free_disk_gib": round(free_gib, 3),
        "minimum_free_disk_gib": minimum_free_disk_gib,
    }


def sanitized_worker_environment(
    base: dict[str, str],
    slot: str,
    worker_id: str,
    role: str,
    gpu: Gpu,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": slot,
            "WORKER_SLOT": slot,
            "WORKER_ID": worker_id,
            "HELENA_GPU_TIER_ROLE": role,
            "HELENA_GPU_MODEL": gpu.name,
            "HELENA_GPU_VRAM_GB": f"{gpu.memory_total_mib / 1024:.3f}",
            "HELENA_CUDA_DEVICE_INDEX": slot,
        }
    )
    return environment


def public_environment(environment: dict[str, str]) -> dict[str, str]:
    allowed = {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "WORKER_SLOT",
        "WORKER_ID",
        "HELENA_GPU_TIER_ROLE",
        "HELENA_GPU_MODEL",
        "HELENA_GPU_VRAM_GB",
        "HELENA_CUDA_DEVICE_INDEX",
    }
    result = {key: environment[key] for key in sorted(allowed) if key in environment}
    if any(marker in key.upper() for key in result for marker in SECRET_MARKERS):
        raise AssertionError("public worker environment unexpectedly contains a secret name")
    return result


def base_receipt(args: argparse.Namespace, check: dict[str, Any], command: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "host": socket.gethostname(),
        "role": args.role,
        "worker_prefix": args.worker_prefix,
        "preflight": check,
        "command": command_identity(command),
        "drain_file": str(args.drain_file.resolve()),
        "lifecycle_policy": {
            "normal_drain_interrupts_active_child": False,
            "cloud_instance_actions": "PROHIBITED",
            "scientific_parameters_mutated": False,
        },
        "aggregate": {
            "children_started": 0,
            "children_succeeded": 0,
            "children_failed": 0,
            "children_active": 0,
        },
        "recent_children": [],
    }


def append_history(receipt: dict[str, Any], row: dict[str, Any], limit: int) -> None:
    history = receipt["recent_children"]
    history.append(row)
    if len(history) > limit:
        del history[: len(history) - limit]


def run_supervisor(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    command_identity(command)
    slots = parse_slots(args.gpu_slots)
    try:
        check = preflight(
            slots=slots,
            minimum_vram_gib=args.minimum_vram_gib,
            work_root=args.work_root,
            minimum_free_disk_gib=args.minimum_free_disk_gib,
            nvidia_smi=args.nvidia_smi,
        )
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        timestamp = utc_now()
        write_json_atomic(
            args.receipt,
            {
                "schema": SCHEMA,
                "status": "PREFLIGHT_FAILED",
                "started_at_utc": timestamp,
                "completed_at_utc": timestamp,
                "host": socket.gethostname(),
                "role": args.role,
                "worker_prefix": args.worker_prefix,
                "requested_gpu_slots": slots,
                "command": command_identity(command),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "lifecycle_policy": {
                    "normal_drain_interrupts_active_child": False,
                    "cloud_instance_actions": "PROHIBITED",
                    "scientific_parameters_mutated": False,
                },
            },
        )
        raise
    receipt = base_receipt(args, check, command)
    receipt["maximum_runtime_seconds"] = args.maximum_runtime_seconds or None
    write_json_atomic(args.receipt, receipt)
    if args.dry_run:
        receipt.update(status="READY", completed_at_utc=utc_now(), updated_at_utc=utc_now())
        write_json_atomic(args.receipt, receipt)
        return 0

    draining = False
    drain_reason = ""
    started_monotonic = time.monotonic()
    active: dict[str, tuple[subprocess.Popen[bytes], dict[str, Any]]] = {}
    cycles = {slot: 0 for slot in slots}
    consecutive_failures = {slot: 0 for slot in slots}
    selected_gpus = {
        str(value["index"]): Gpu(
            index=str(value["index"]),
            name=str(value["name"]),
            memory_total_mib=int(value["memory_total_mib"]),
            uuid=str(value["uuid"]),
        )
        for value in check["selected_gpus"]
    }

    def request_drain(signum: int, _frame: object) -> None:
        nonlocal draining, drain_reason
        draining = True
        drain_reason = f"SIGNAL_{signum}"

    signal.signal(signal.SIGTERM, request_drain)
    signal.signal(signal.SIGINT, request_drain)

    while True:
        if (
            args.maximum_runtime_seconds
            and time.monotonic() - started_monotonic >= args.maximum_runtime_seconds
            and not draining
        ):
            draining = True
            drain_reason = "MAXIMUM_RUNTIME_REACHED"
        if args.drain_file.exists() and not draining:
            draining = True
            drain_reason = "DRAIN_FILE"

        for slot in list(active):
            child, row = active[slot]
            returncode = child.poll()
            if returncode is None:
                continue
            row.update(completed_at_utc=utc_now(), returncode=returncode)
            append_history(receipt, row, args.receipt_history)
            del active[slot]
            if returncode == 0:
                receipt["aggregate"]["children_succeeded"] += 1
                consecutive_failures[slot] = 0
            else:
                receipt["aggregate"]["children_failed"] += 1
                consecutive_failures[slot] += 1
                if consecutive_failures[slot] >= args.maximum_consecutive_failures:
                    draining = True
                    drain_reason = f"CHILD_FAILURE_LIMIT_GPU_{slot}"

        if args.maximum_cycles_per_slot and all(
            cycles[slot] >= args.maximum_cycles_per_slot for slot in slots
        ):
            draining = True
            drain_reason = drain_reason or "MAXIMUM_CYCLES_REACHED"

        if not draining:
            for slot in slots:
                if slot in active:
                    continue
                if args.maximum_cycles_per_slot and cycles[slot] >= args.maximum_cycles_per_slot:
                    continue
                worker_id = f"{args.worker_prefix}-{args.role}-gpu{slot}"
                environment = sanitized_worker_environment(
                    os.environ,
                    slot,
                    worker_id,
                    args.role,
                    selected_gpus[slot],
                )
                started_at = utc_now()
                child = subprocess.Popen(command, env=environment)
                active[slot] = (
                    child,
                    {
                        "slot": slot,
                        "worker_id": worker_id,
                        "pid": child.pid,
                        "started_at_utc": started_at,
                        "environment": public_environment(environment),
                    },
                )
                cycles[slot] += 1
                receipt["aggregate"]["children_started"] += 1

        receipt["aggregate"]["children_active"] = len(active)
        receipt["updated_at_utc"] = utc_now()
        receipt["drain_requested"] = draining
        receipt["drain_reason"] = drain_reason or None
        receipt["cycles_by_gpu"] = dict(cycles)
        write_json_atomic(args.receipt, receipt)

        if draining and not active:
            break
        time.sleep(args.poll_seconds if active else args.idle_seconds)

    failed = drain_reason.startswith("CHILD_FAILURE_LIMIT")
    receipt.update(
        status="FAILED" if failed else "DRAINED",
        completed_at_utc=utc_now(),
        updated_at_utc=utc_now(),
    )
    receipt["aggregate"]["children_active"] = 0
    write_json_atomic(args.receipt, receipt)
    return 2 if failed else 0


def command_preflight(args: argparse.Namespace) -> int:
    result = preflight(
        slots=parse_slots(args.gpu_slots),
        minimum_vram_gib=args.minimum_vram_gib,
        work_root=args.work_root,
        minimum_free_disk_gib=args.minimum_free_disk_gib,
        nvidia_smi=args.nvidia_smi,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_drain(args: argparse.Namespace) -> int:
    args.drain_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.drain_file.with_name(f".{args.drain_file.name}.tmp")
    temporary.write_text(f"drain requested at {utc_now()}\n", encoding="utf-8")
    temporary.replace(args.drain_file)
    print(json.dumps({"status": "DRAIN_REQUESTED", "drain_file": str(args.drain_file)}))
    return 0


def command_status(args: argparse.Namespace) -> int:
    if not args.receipt.is_file():
        raise RuntimeError(f"supervisor receipt does not exist: {args.receipt}")
    value = json.loads(args.receipt.read_text(encoding="utf-8"))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def add_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu-slots", required=True, help="Physical nvidia-smi indices, e.g. 0,1")
    parser.add_argument("--minimum-vram-gib", type=float, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=2.5)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    add_preflight_arguments(preflight_parser)
    preflight_parser.set_defaults(function=command_preflight)

    run_parser = subparsers.add_parser("run")
    add_preflight_arguments(run_parser)
    run_parser.add_argument("--role", choices=("always-on", "burst"), required=True)
    run_parser.add_argument("--worker-prefix", default=socket.gethostname())
    run_parser.add_argument("--receipt", type=Path, required=True)
    run_parser.add_argument("--drain-file", type=Path, required=True)
    run_parser.add_argument("--poll-seconds", type=float, default=2.0)
    run_parser.add_argument("--idle-seconds", type=float, default=30.0)
    run_parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    run_parser.add_argument("--maximum-cycles-per-slot", type=int, default=0)
    run_parser.add_argument(
        "--maximum-runtime-seconds",
        type=float,
        default=0,
        help="Stop claiming new jobs after this duration; zero means unbounded.",
    )
    run_parser.add_argument("--receipt-history", type=int, default=256)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(function=run_supervisor)

    drain_parser = subparsers.add_parser("drain")
    drain_parser.add_argument("--drain-file", type=Path, required=True)
    drain_parser.set_defaults(function=command_drain)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--receipt", type=Path, required=True)
    status_parser.set_defaults(function=command_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "maximum_consecutive_failures", 1) < 1:
        raise ValueError("maximum-consecutive-failures must be at least one")
    if getattr(args, "receipt_history", 1) < 1:
        raise ValueError("receipt-history must be at least one")
    for name in (
        "minimum_vram_gib",
        "minimum_free_disk_gib",
        "poll_seconds",
        "idle_seconds",
        "maximum_runtime_seconds",
    ):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must not be negative")
    return int(args.function(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"gpu tier supervisor: {error}", file=sys.stderr)
        raise SystemExit(2)
