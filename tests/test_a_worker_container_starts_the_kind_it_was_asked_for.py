"""One image, one entrypoint, more than one kind of worker.

The entrypoint does something no caller can do for itself: it starts the MCP
seed service, mints a token for it, exports the pair into the worker's
environment, deletes the token from disk, and refuses to claim anything if the
service did not come up. Every worker that talks to M7 needs all of that.

The candidate preflight needs it too -- that is why it moved out of the panel,
which had none of it. But the entrypoint's exec was fixed to the segmentation
worker, so the only way to run a different kind was to override the entrypoint,
which skips exactly the setup that made moving worthwhile.

Two things this pins that are easy to get wrong:

* the readiness gate applies to every kind. A worker whose seed service died
  claims work and fails it, burning an attempt each time, and for the preflight
  that would be a source-unavailable receipt about a service rather than a
  scroll.
* each container mints its own service on its own port. With host networking two
  of them on one machine collide, and only one holds the port -- which is how a
  previous run got 401s from five workers that all thought they had a service.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (ROOT / "containers/images/worker-entrypoint.sh").read_text()
COMPOSE = yaml.safe_load(
    (ROOT / "containers/compose/segment.compose.yaml").read_text())


def test_the_entrypoint_dispatches_on_a_worker_kind() -> None:
    assert "HELENA_WORKER_KIND" in ENTRYPOINT, (
        "the entrypoint runs one hard-coded worker, so a second kind can only "
        "be started by overriding it -- which skips the seed service it exists "
        "to start"
    )


def test_every_kind_gets_the_service_before_it_runs() -> None:
    """The exports must precede every exec, not just the original one."""
    exports = ENTRYPOINT.index('export VC_MCP_URL=')
    for match in re.finditer(r"^exec ", ENTRYPOINT, re.MULTILINE):
        assert match.start() > exports, (
            "a worker is exec'd before VC_MCP_URL is exported, so it starts "
            "without the service the entrypoint just began"
        )


def test_the_readiness_gate_is_not_skipped_by_the_new_kind() -> None:
    """The wait for the service comes before any dispatch on kind."""
    gate = ENTRYPOINT.index("the seed service exited before it was ready")
    dispatch = ENTRYPOINT.index("HELENA_WORKER_KIND")
    assert gate < dispatch, (
        "the kind is chosen before the service is known to be up, so one kind "
        "can claim work against a service that never started"
    )


def test_the_preflight_kind_enters_through_the_package():
    """cli.py uses relative imports and only works inside its package.

    The entrypoint says so itself for the segmentation worker. A second entry
    that ignored it would fail at import, in a container, at deploy time.
    """
    preflight = ENTRYPOINT[ENTRYPOINT.index("HELENA_WORKER_KIND"):]
    assert "helena_segment_search_fleet.py" in preflight, (
        "the preflight worker is started some other way than the stage's own "
        "entry point"
    )
    assert "python3 -m fleet.cli" not in ENTRYPOINT
    assert "fleet/cli.py" not in ENTRYPOINT


def test_the_preflight_service_exists_and_runs_the_preflight_kind() -> None:
    services = COMPOSE["services"]
    assert "preflight" in services, "no candidate-preflight worker is declared"
    preflight = services["preflight"]
    assert preflight.get("network_mode") == "host", (
        "the seed service binds loopback; without host networking the worker "
        "cannot reach the one its own container started"
    )
    assert "HELENA_WORKER_KIND" in (preflight.get("environment") or {})
    assert not preflight.get("entrypoint"), (
        "overriding the entrypoint skips the seed service, which is the reason "
        "this work moved off the panel in the first place"
    )


def test_two_workers_on_one_host_do_not_share_a_port() -> None:
    """Each container mints its own service; host networking makes the port
    a shared resource, and only one holder wins."""
    services = COMPOSE["services"]
    # Only the containers that run the entrypoint mint a service. fleet-runner
    # overrides it to start the ink worker directly, so it never binds the port
    # -- a first version of this check counted it and reported a collision that
    # cannot happen.
    minting = {
        name: str((service.get("environment") or {}).get("MCP_PORT", "18080"))
        for name, service in services.items()
        if service.get("network_mode") == "host" and not service.get("entrypoint")
    }
    assert len(minting) >= 2, "this check has nothing to compare"
    assert len(set(minting.values())) == len(minting), (
        f"two containers start a seed service on the same port: {minting}"
    )


def test_the_preflight_worker_is_told_it_may_claim_forever() -> None:
    """--max-jobs 0 is unlimited (`args.max_jobs or None`); a default of 1 would
    make the service exit after one measurement and restart forever."""
    command = COMPOSE["services"]["preflight"].get("command") or ""
    assert "--max-jobs 0" in command
    assert "--watch" in command
