"""provision-host.sh promised this in its header and then broke it.

Its own opening paragraph said "nothing is installed outside the image", and
forty lines later it scp'd two systemd units into /etc/systemd/system and
enabled them. One of the two, helena-worker-cpp.service, was not in the tree at
all any more -- so the section had been failing on a missing file, which is how
long it had been since anyone ran it.

A per-host unit is a second way to deploy: its own failure mode, its own logs,
its own upgrade path, and nothing that reports it to the panel. Both units are
compose services now -- the worker was already segment.compose.yaml, and the
forward to the control plane is control-tunnel.compose.yaml.

Docker's own service is not in scope here. The host holds Docker, the compose
files and the env files; enabling dockerd is what makes containers possible
rather than something running beside them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROVISION = ROOT / "containers/provision-host.sh"
COMPOSE = ROOT / "containers/compose"


def _provision() -> str:
    if not PROVISION.exists():
        pytest.skip("this checkout does not ship provision-host.sh")
    return PROVISION.read_text(encoding="utf-8")


def test_no_unit_file_is_shipped():
    assert not (ROOT / "containers/systemd").exists(), (
        "containers/systemd is back; a unit here is a second way to deploy")


def _commands(script: str) -> list[str]:
    """The lines that run, without the comments that explain the history.

    The header says what the units used to do, on purpose: a removal with no
    account of what was removed is how the same thing comes back.
    """
    return [line for line in script.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_provisioning_installs_no_unit():
    for line in _commands(_provision()):
        for forbidden in ("/etc/systemd/system", "daemon-reload", ".service"):
            assert forbidden not in line, (
                f"provision-host.sh still installs a unit: {line.strip()}")


def test_the_only_systemctl_left_is_the_container_runtime():
    """Docker is the exception and the only one: it is what runs the rest."""
    script = _provision()
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "systemctl" not in stripped:
            continue
        assert "docker" in stripped, f"systemctl on something else: {stripped}"


def test_provisioning_brings_both_halves_up_with_compose():
    script = _provision()
    assert "docker compose" in script, (
        "nothing is started with compose, so the units were removed and not "
        "replaced")
    for stack in ("segment.compose.yaml", "control-tunnel.compose.yaml"):
        assert stack in script, f"{stack} is never deployed to the host"


def test_the_tunnel_is_a_compose_service():
    tunnel = COMPOSE / "control-tunnel.compose.yaml"
    assert tunnel.is_file(), "the unit was deleted with nothing to replace it"

    yaml = pytest.importorskip("yaml")
    document = yaml.safe_load(tunnel.read_text(encoding="utf-8"))
    service = document["services"]["tunnel"]
    # Host networking, because the whole point is binding the host's loopback:
    # that is the address every worker's FLEET_DB names.
    assert service["network_mode"] == "host"
    # Restart, because the unit it replaces used Restart=always and the forward
    # is meant to be permanent.
    assert service["restart"] == "unless-stopped"
    # The key is mounted read-only. It opens a forward; it is not the host's.
    key = [mount for mount in service["volumes"] if "/keys/" in mount]
    assert key and key[0].endswith(":ro"), (
        f"the forward's key is not mounted read-only: {service['volumes']}")


def test_every_image_the_provisioner_streams_is_one_the_build_makes():
    """It streams what `docker save` can find, so the build has to make it."""
    script = _provision()
    build = (ROOT / "containers/build-images.sh").read_text(encoding="utf-8")
    for image in re.findall(r"helena-[a-z-]+(?=:\$)", script):
        # build-images.sh composes its tags as ${prefix}helena-$name, so the
        # full name never appears in it -- the suffix is what it lists.
        assert image.removeprefix("helena-") in build, (
            f"provision-host.sh streams {image}, which build-images.sh never "
            "builds, so `docker save` has nothing to send")
