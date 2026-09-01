#!/usr/bin/env python3
"""Report what this machine offers to the control plane, on a timer.

    host_report.py --db <dsn> [--host-id <id>] [--disk /artifacts] [--every 60]

A worker host reported only its admission capabilities -- whether it has a usable
GPU -- because that is what the fleet filters claims on. Cores, RAM and disk were
never reported by anything, so the Hosts table showed dashes for every machine
except the one the panel runs on, where it could just look. A host that was
working looked the same as a host nobody had set up.

This is a separate process rather than a hook in the worker, for two reasons.
Host inventory is a property of the machine, not of the job loop -- a worker
between tasks is still a host worth knowing the free memory of. And a claim runs
inside `FOR UPDATE SKIP LOCKED`; adding a write to another table inside that
transaction buys a lock interaction on the hot path in exchange for a display.

It writes ink_hosts.last_state, which is the host inventory table the panel
already reads, and not the admission contract -- that one stays frozen and
narrow because task routing depends on its shape.

psycopg2 rather than psycopg: it is what the worker image installs, and adding a
second driver to move one row would be a dependency for nothing.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from framework.contracts.host_probe import host_state  # noqa: E402


def report_once(dsn: str, host_id: str, disk: str | None) -> str:
    """One reading, written to the host's row. Returns what happened.

    A missing row is normal rather than an error: a machine can run a worker
    before anyone registers it in the panel, and the reading is simply not
    wanted yet. Saying so and continuing beats exiting and being restarted
    forever by systemd.
    """
    import psycopg2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stages/03-ink/fleet"))
    from job_store import merge_gpu_observations  # noqa: PLC0415

    state = host_state(Path(disk) if disk else None)
    with psycopg2.connect(dsn, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            # Read before writing, and merge the cards rather than replacing
            # them. This process does not necessarily run where the cards are:
            # on gpu-1 it runs in the segmentation container, which has no
            # nvidia-smi, while the ink worker beside it can see the GPU and
            # records it correctly. Writing wholesale erased that card thirteen
            # seconds after every one of its heartbeats, and the Hosts page
            # showed "not reported" for a machine with two of them.
            #
            # The ink worker's record_host_state has merged for exactly this
            # reason since the same bug was found there; this second writer
            # never got it. A probe that sees no card now contributes none
            # rather than denying what another saw, and a card that is really
            # gone still ages out on the merge's own hour-long clock.
            cursor.execute("SELECT last_state FROM ink_hosts WHERE host_id=%s",
                           (host_id,))
            row = cursor.fetchone()
            merged = dict(state)
            merged["gpus"] = merge_gpu_observations(
                (row[0] if row else None) or {}, state)
            cursor.execute(
                "UPDATE ink_hosts SET last_seen_at=now(), last_state=%s::jsonb "
                "WHERE host_id=%s",
                (json.dumps(merged, sort_keys=True), host_id))
            if cursor.rowcount == 0:
                # Where, not just what. This ran every minute on a fresh
                # install, saying a row was missing and leaving no way to find
                # out how one is made -- the reader has the panel open and no
                # reason to guess that Configuration holds a Hosts tab.
                return (f"no host row named {host_id!r}; add it under "
                        f"Configuration -> Hosts in the panel to see this")
    return (f"{state.get('cores', '?')} cores, "
            f"{state.get('ram_free_gb', '?')}/{state.get('ram_total_gb', '?')} GB RAM, "
            f"{state.get('disk_free_gb', '?')} GB free on "
            f"{state.get('disk_path', '?')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="the control plane DSN")
    parser.add_argument("--host-id", default=None,
                        help="the row in ink_hosts to update; the hostname by default, "
                             "because that is what a host is registered as")
    parser.add_argument("--disk", default=None,
                        help="the volume whose free space matters -- where artifacts "
                             "land, not /, which on these hosts is a different disk")
    parser.add_argument("--every", type=int, default=60,
                        help="seconds between readings; 0 reports once and exits")
    arguments = parser.parse_args()
    host_id = arguments.host_id or socket.gethostname()

    while True:
        try:
            print(f"{host_id}: {report_once(arguments.db, host_id, arguments.disk)}",
                  flush=True)
        except Exception as failure:  # noqa: BLE001
            # A control plane that is restarting must not take the worker's
            # container down with it: this process shares a container with the
            # thing that does the actual work.
            print(f"{host_id}: could not report -- "
                  f"{type(failure).__name__}: {failure}", file=sys.stderr, flush=True)
        if arguments.every <= 0:
            return 0
        time.sleep(arguments.every)


if __name__ == "__main__":
    raise SystemExit(main())
