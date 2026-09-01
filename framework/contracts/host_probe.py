"""What a machine can currently offer, measured on the machine.

Lifted out of the ink worker so a segmentation host can report the same facts.
It was the only implementation and it lived inside a module that imports the ink
job store at import time, which needs psycopg and a package also called `fleet`
-- so a segmentation worker could neither import it nor load it by path, and the
Hosts table showed dashes for every column on any host that was not the one the
panel happens to run on.

Standard library only, and no knowledge of any stage. That is what makes it
importable from both.
"""

from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def host_state(runs_root: Path | None = None) -> dict:
    """What this host can currently offer. Recorded so the panel can show it."""
    state: dict = {"hostname": socket.gethostname(),
                   "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        # nvidia-smi honours CUDA_VISIBLE_DEVICES, so a worker pinned to one card
        # reports one card and the panel shows a host with half its hardware. The
        # probe describes the machine, not this process's slice of it, so the
        # variable is dropped for the call.
        environment = {k: v for k, v in os.environ.items()
                       if k not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, env=environment)
        if out.returncode == 0:
            state["gpus"] = [
                {"index": int(p[0]), "name": p[1], "used_mb": int(p[2]),
                 "total_mb": int(p[3]), "util_pct": int(p[4]), "uuid": p[5]}
                for p in (line.split(", ") for line in out.stdout.strip().splitlines())
                if len(p) == 6
            ]
    except (OSError, subprocess.SubprocessError, ValueError):
        state["gpus"] = []
    # The runs root, not "/". On this fleet they are different disks and the
    # difference is the whole point: / had 4 GB free while the volume the work
    # lands on had 103. A free-space figure for a filesystem nothing is written
    # to answers a question nobody asked.
    measured = Path(runs_root) if runs_root else Path("/")
    while not measured.exists() and measured != measured.parent:
        measured = measured.parent
    try:
        usage = os.statvfs(measured)
        state["disk_free_gb"] = round(usage.f_bavail * usage.f_frsize / 1e9, 1)
        state["disk_path"] = str(measured)
    except OSError:
        pass
    state.update(cpu_and_memory())
    state["images"] = local_images()
    return state


def local_images() -> dict[str, str]:
    """Which framework images this host holds, fingerprinted by layer chain.

    Reported by the host rather than fetched by the panel. Giving the control
    plane SSH into every machine so it can look would be a larger privilege
    than anything else it has, and the host already tells us its GPUs, cores
    and disk this way.

    The fingerprint is the RootFS layer chain because `docker inspect .Id` is
    not comparable between hosts: the classic image store reports the config
    digest and the containerd store reports the OCI manifest digest, so the
    same image reads as two.
    """
    found: dict[str, str] = {}
    try:
        listing = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}"],
            capture_output=True, text=True, timeout=15)
        if listing.returncode != 0:
            return found
        for name in sorted({n for n in listing.stdout.split() if n.startswith("helena-")}):
            inspected = subprocess.run(
                ["docker", "inspect", name, "--format",
                 "{{range .RootFS.Layers}}{{slice . 7 19}}+{{end}}"],
                capture_output=True, text=True, timeout=15)
            if inspected.returncode == 0 and inspected.stdout.strip():
                found[name] = inspected.stdout.strip().rstrip("+")
    except (OSError, subprocess.SubprocessError):
        return found
    return found


def cpu_and_memory() -> dict:
    """Cores and RAM, as the scheduler would have to see them.

    Two deliberate choices. Cores come from the affinity mask rather than
    os.cpu_count(), because a worker confined to two cores on a machine with
    thirty-two will not go faster for the other thirty existing. And free memory
    is MemAvailable, not MemFree: MemFree excludes reclaimable page cache, so it
    reads as almost nothing on any host that has been up a while and would make
    every box look like it is about to die.
    """
    out: dict = {}
    try:
        out["cores"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        out["cores"] = os.cpu_count() or 0
    total = os.cpu_count() or 0
    if total and total != out["cores"]:
        out["cores_total"] = total
    try:
        meminfo = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                meminfo[key] = rest.strip()
        for key, name in (("MemTotal", "ram_total_gb"), ("MemAvailable", "ram_free_gb")):
            if key in meminfo:
                out[name] = round(int(meminfo[key].split()[0]) * 1024 / 1e9, 1)
    except (OSError, ValueError, IndexError):
        pass
    return out



def resolve_device(requested: str | None = None) -> dict:
    """Which compute device this run uses, and how that was decided.

    Every runner in this platform that takes a device defaulted to ``cuda``,
    unconditionally. That is a fine default on the two hosts that have cards and
    a crash on any host that does not, so "supports both" was true of the flag
    and false of the program.

    Three answers, and the difference between them is the point:

    ``auto`` -- or nothing at all -- is "the card if this host has one, the CPU
    if it does not". It is what a job should carry when nobody cared, and it is
    now the default.

    An explicit ``cuda`` is a requirement, not a preference. A run that asked
    for a card and did not get one is refused here rather than finishing on the
    CPU some hours later at a cost nobody agreed to -- upstream measured the
    seeded grow's own CUDA path at 40x *slower* than its CPU one, so "the other
    device" is not a detail that can be swapped in silently.

    An explicit ``cpu`` is honoured on any host, including one with a card.

    The return value is what a receipt should record: the device actually used,
    not the word the caller wrote. `auto` in a receipt says nothing about where
    the work ran.
    """

    asked = (requested or "auto").strip().lower()
    available, detail = _cuda_available()
    if asked in ("", "auto"):
        return {
            "device": "cuda" if available else "cpu",
            "requested": "auto",
            "cuda_available": available,
            "reason": ("this host offers a card" if available
                       else f"this host offers no card: {detail}"),
        }
    if asked == "cpu":
        return {"device": "cpu", "requested": "cpu", "cuda_available": available,
                "reason": "the run asked for the CPU"}
    if not asked.startswith("cuda"):
        raise ValueError(
            f"device must be 'auto', 'cpu' or start with 'cuda'; got {requested!r}")
    if not available:
        raise RuntimeError(
            f"this run asked for {asked} and this host has no usable card: "
            f"{detail}. Queue it with device 'auto' to let the worker decide, "
            "or 'cpu' to say plainly that the CPU is acceptable here.")
    return {"device": asked, "requested": asked, "cuda_available": True,
            "reason": "the run asked for a card and this host has one"}


def _cuda_available() -> tuple[bool, str]:
    """Whether torch can actually reach a card, said without importing torch
    when it is absent.

    Asked of torch rather than of nvidia-smi: a driver on the host says nothing
    about whether this container was given a device, and every consumer of this
    answer is a torch program.
    """

    try:
        import torch  # noqa: PLC0415
    except ImportError as absent:
        return False, f"torch is not installed here ({absent})"
    try:
        if not torch.cuda.is_available():
            return False, "torch reports no CUDA device"
        if torch.cuda.device_count() < 1:
            return False, "torch reports zero CUDA devices"
    except Exception as failure:  # noqa: BLE001 - a broken driver is "no card"
        return False, f"asking torch for a device raised {type(failure).__name__}: {failure}"
    return True, f"{torch.cuda.device_count()} device(s)"
