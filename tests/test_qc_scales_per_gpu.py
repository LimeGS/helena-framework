"""One QC instance per GPU, from one compose file.

A two-card host measured one surface at a time, because the unit named device 0
and there was no way to say "and also device 1" without copying the file. An
eight-card rig was not expressible at all.

Concurrency was never the obstacle -- claim_qc takes one job under FOR UPDATE
SKIP LOCKED with an atomic lease, and two workers ran side by side during the
July migration. The deployment description was.

What this checks is what a per-instance file gets wrong quietly: two instances
sharing one mutable thing. Each is a real incident waiting: a shared container
name means the second `up` replaces the first, a shared run root means two
workers write the same staging paths, and one worker id makes the record unable
to say which card measured a surface.
"""

from __future__ import annotations

import re
from pathlib import Path

COMPOSE = (Path(__file__).resolve().parents[1]
           / "containers/compose/surface-qc.compose.yaml")
TEXT = COMPOSE.read_text(encoding="utf-8")
ENV_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "containers/compose/surface-qc.env.example"
).read_text(encoding="utf-8")

# Everything that must differ between instances is derived from this one value,
# so an operator sets one variable and a project name.
DEVICE = "${HELENA_QC_DEVICE:-0}"


def test_the_gpu_is_chosen_by_variable_and_not_by_all():
    """`gpus: all` on a two-card box means both services think they own both,
    and 6 GiB will not hold two TimeSformer passes."""
    assert 'device_ids: ["${HELENA_QC_DEVICE:-0}"]' in TEXT
    assert "gpus: all" not in TEXT


def test_the_container_name_is_per_instance():
    name = next(l for l in TEXT.splitlines() if "container_name:" in l)
    assert DEVICE in name, name


def test_the_run_root_is_per_instance():
    mount = next(l for l in TEXT.splitlines() if "qc-runtime" in l)
    assert DEVICE in mount, mount


def test_the_worker_id_is_per_instance():
    worker = next(l for l in TEXT.splitlines() if "QC_WORKER_ID" in l)
    assert DEVICE in worker, worker


def test_the_file_says_how_to_run_more_than_one():
    """A file that scales only if you already know the trick does not scale.
    The project name is what lets two of these coexist."""
    assert re.search(r"-p helena-qc-\d", TEXT)
    assert "eight-card" in TEXT or "eight" in TEXT


def test_it_survives_a_reboot_without_a_supervisor():
    """Nothing outside docker: no systemd unit means the restart policy is the
    only thing that brings this back, so it has to be set."""
    assert "restart: unless-stopped" in TEXT


def test_a_direct_compose_restart_keeps_the_host_provenance_fallback():
    """The documented per-card command does not run deploy-platform.sh.

    A deployment supplies HELENA_COMMIT_FULL and must win, while a direct
    operator restart must still be able to use the full SHA in surface-qc.env.
    """
    assert (
        'HELENA_QC_CODE_COMMIT: "${HELENA_COMMIT_FULL:-${HELENA_QC_CODE_COMMIT:?set in surface-qc.env}}"'
        in TEXT
    )


def test_qc_can_read_surfaces_published_by_the_p8_reconstruction_lane():
    """P8 publishes below ``/artifacts/reconstruction-v1``.  Mounting only
    ``/artifacts/surfaces`` leaves its automatic QC job permanently unable to
    materialize the child it was created to inspect."""
    mount = next(
        line for line in TEXT.splitlines()
        if ":/artifacts/reconstruction-v1:ro" in line
    )
    assert "HELENA_QC_RECONSTRUCTIONS" in mount
    assert "HELENA_QC_RECONSTRUCTIONS=" in ENV_EXAMPLE
