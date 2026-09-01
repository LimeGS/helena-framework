"""Adding a host has to put the worker on it, not just record that it exists.

Registering wrote a database row and stopped. The host then showed up in the
table looking ready and claimed nothing, and the symptom was a queue that did
not move -- which reads as "no work available" rather than "this machine was
never set up". These pin the two things that made it silent.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "containers" / "provision-host.sh"

sys.path.insert(0, str(ROOT))


def test_provisioning_is_the_default_for_a_new_host():
    """Opt-out, not opt-in. The default is what a forgotten flag chooses."""
    from panel.app import HostRequest
    assert HostRequest(host_id="h", ssh_target="user@host").provision is True
    assert HostRequest(host_id="h", ssh_target="user@host",
                       provision=False).provision is False


def test_the_script_exists_and_is_executable():
    assert SCRIPT.exists(), "the endpoint spawns this by path"
    assert SCRIPT.stat().st_mode & 0o111, f"{SCRIPT} is not executable"


def test_the_script_refuses_without_a_target():
    """No target must fail loudly rather than provision something ambient."""
    done = subprocess.run(["/bin/sh", str(SCRIPT)], capture_output=True, text=True,
                          timeout=30)
    assert done.returncode != 0
    assert "usage" in (done.stderr + done.stdout).lower()


def test_the_script_is_valid_shell():
    done = subprocess.run(["/bin/sh", "-n", str(SCRIPT)], capture_output=True,
                          text=True, timeout=30)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("needle, why", [
    ("command -v docker", "must install Docker only when it is missing"),
    ("helena-worker-cpp", "must install the worker unit"),
    ("helena-control-tunnel", "must install the tunnel the worker reaches the "
                                 "control plane through"),
    ("is-active", "must report what the units ended up doing rather than assume it"),
])
def test_the_script_does_the_four_things_registration_promised(needle, why):
    assert needle in SCRIPT.read_text(encoding="utf-8"), why


def test_nothing_is_built_on_the_host():
    """The image is the deliverable; a host that compiles is not reproducible.

    This host previously had VC3D and ink models built directly on it, which is
    the failure the image exists to prevent.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    for builder in (r"\bcmake\b", r"\bmake -j", r"\bpip install\b", r"\bgit clone\b"):
        assert not re.search(builder, text), (
            f"{builder} in the provisioner: the host must run the image, not build it")
