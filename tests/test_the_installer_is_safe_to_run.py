"""The one-line installer, and what it must not do.

It exists for the failures that are illegible from inside Docker: a full disk
surfaces as apt-get exiting 100 about /var/cache/apt, which reads as a broken
base image; a compose v1 shim surfaces as a YAML error about a valid key. Those
are checked before anything is downloaded or built.

The dangerous one is subtler. platform.compose.yaml names its project `helena`,
so running this on a host that already has a deployment does not start a second
stack -- it recreates the first with this script's defaults. Testing the
installer against a live machine replaced its panel with a locally built 0.10.0
on another port, and the real one stopped answering until it was put back.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "install.sh").read_text()
COMPOSE = (ROOT / "containers/compose/platform.compose.yaml").read_text()


def test_it_refuses_to_run_over_an_existing_deployment() -> None:
    """The failure that has actually happened."""
    assert "docker compose ls" in INSTALL, (
        "nothing checks for a stack already on this host, so the installer will "
        "recreate it instead of refusing"
    )
    project = re.search(r"^name:\s*(\S+)", COMPOSE, re.MULTILINE)
    assert project, "the compose file no longer names its project"
    assert f'"Name":"{project.group(1)}"' in INSTALL, (
        f"the installer checks for a project other than {project.group(1)!r}, "
        "which is the one it would collide with"
    )
    # And it has to stop, not warn.
    guard = INSTALL[INSTALL.index("docker compose ls"):]
    assert "die " in guard[:400], "an existing deployment only produces a warning"


def test_it_checks_before_it_builds() -> None:
    """Every check has to happen before the clone and the build, or it is a
    check that reports after the damage."""
    for probe in ("command -v git", "docker info", "docker compose version",
                  "df -Pk", "docker compose ls"):
        assert probe in INSTALL, f"the installer does not check {probe!r}"
        assert INSTALL.index(probe) < INSTALL.index("git clone"), (
            f"{probe!r} runs after the clone"
        )


def test_it_waits_for_the_panel_to_answer() -> None:
    """`up -d` returns when containers start, not when the panel serves.
    Reporting success there is how somebody opens a browser to a connection
    reset and concludes the install failed."""
    assert "waiting for the panel" in INSTALL
    assert "127.0.0.1:$PORT" in INSTALL
    # And says what to look at when it does not come up, rather than exiting mute.
    tail = INSTALL[INSTALL.index("waiting for the panel"):]
    assert "logs" in tail[:900], "a panel that never answers leaves no diagnosis"


def test_it_does_not_hide_what_it_runs() -> None:
    """A script people are asked to pipe into a shell has to be readable, and
    has to say so."""
    head = INSTALL[:2000]
    assert "less install.sh" in head or "read it first" in head, (
        "the header does not offer the inspect-then-run path"
    )
    assert "curl" in head and "sh" in head
    # Nothing fetched and executed beyond the repository itself.
    assert not re.search(r"curl[^\n|]*\|\s*(ba)?sh", INSTALL[INSTALL.index("set -eu"):]), (
        "the installer pipes something else into a shell"
    )


def test_the_installer_and_the_readme_agree_on_where_this_lives() -> None:
    """Two copies of a URL is two chances to be wrong, and the one people paste
    is whichever they saw first."""
    readme = (ROOT / "README.md").read_text()
    urls = set(re.findall(r"https://(?:raw\.githubusercontent|github)\.com/([\w.-]+/[\w.-]+)",
                          INSTALL + readme))
    urls = {u.removesuffix(".git") for u in urls}
    assert len(urls) == 1, f"the installer and the README name different repositories: {urls}"
    assert "OWNER/REPO" not in (INSTALL + readme), "a placeholder URL is still in place"


def test_it_installs_from_a_branch_that_is_meant_to_be_installed() -> None:
    """The default ref is what a stranger gets.

    Pointing it at a branch that lags behind is how somebody follows the README
    and ends up with a version nobody is testing -- the install succeeds, so
    nothing says otherwise.
    """
    ref = re.search(r'REF="\$\{HELENA_REF:-([\w.-]+)\}"', INSTALL)
    assert ref, "the installer does not pin a default branch"
    readme = (ROOT / "README.md").read_text()
    assert f"/{ref.group(1)}/install.sh" in readme, (
        f"the README fetches install.sh from a different branch than the {ref.group(1)!r} "
        "the script then clones"
    )
