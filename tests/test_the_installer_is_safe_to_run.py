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
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "install.sh").read_text()
# The installer runs docker through `$D`, which is HELENA_DOCKER or plain
# `docker`: the docker group is root-equivalent, so a good number of people run
# `sudo docker` and never join it. Match on the command, not on the spelling --
# these checks are about which questions get asked before anything is built.
DOCKER = "$D"
COMPOSE = (ROOT / "containers/compose/platform.compose.yaml").read_text()


def test_it_refuses_to_run_over_an_existing_deployment() -> None:
    """The failure that has actually happened."""
    assert f"{DOCKER} compose ls" in INSTALL, (
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
    guard = INSTALL[INSTALL.index(f"{DOCKER} compose ls"):]
    assert "die " in guard[:400], "an existing deployment only produces a warning"


def test_it_checks_before_it_builds() -> None:
    """Every check has to happen before the clone and the build, or it is a
    check that reports after the damage."""
    for probe in ("command -v git", f"{DOCKER} info", f"{DOCKER} compose version",
                  "df -Pk", f"{DOCKER} compose ls"):
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
    # Nothing fetched and executed beyond the repository itself. Comments
    # stripped: the installer's own published invocation is `curl ... | sh` and
    # explaining that in a comment is not the installer doing it.
    body = "\n".join(line.split("#", 1)[0]
                     for line in INSTALL[INSTALL.index("set -eu"):].splitlines())
    assert not re.search(r"curl[^\n|]*\|\s*(ba)?sh", body), (
        "the installer pipes something else into a shell"
    )


# Repositories that are deliberately somebody else's. The worker build takes
# volume-cartographer source as a build context, so the README has to say where
# that comes from -- naming upstream is the point, not a second address for
# Helena.
UPSTREAM_REPOSITORIES = {"ScrollPrize/villa"}


def test_the_installer_and_the_readme_agree_on_where_this_lives() -> None:
    """Two copies of a URL is two chances to be wrong, and the one people paste
    is whichever they saw first."""
    readme = (ROOT / "README.md").read_text()
    urls = set(re.findall(r"https://(?:raw\.githubusercontent|github)\.com/([\w.-]+/[\w.-]+)",
                          INSTALL + readme))
    urls = {u.removesuffix(".git") for u in urls} - UPSTREAM_REPOSITORIES
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


def test_it_can_be_pointed_at_a_docker_the_user_can_actually_reach() -> None:
    """Not joining the docker group is a choice, not a misconfiguration.

    Membership grants the daemon, and the daemon can mount the host filesystem,
    so it is root-equivalent -- plenty of people deliberately stay out of it and
    run `sudo docker`. deploy-platform.sh has always read HELENA_DOCKER for
    exactly that; the installer called `docker` directly and told those people
    their daemon was down, which is the one diagnosis that sends them looking in
    the wrong place entirely.

    Measured on a host whose user is not in the group: the install proceeded
    with HELENA_DOCKER='sudo -n docker' and stopped without it.
    """
    assert 'D="${HELENA_DOCKER:-docker}"' in INSTALL, (
        "the installer no longer takes HELENA_DOCKER, so a host that runs "
        "`sudo docker` cannot install at all")

    # `command -v "$D"` cannot work for a two-word value, and that is the value
    # this exists to support.
    assert 'command -v "${D%% *}"' in INSTALL, (
        "the presence check would look for a command named `sudo docker`")

    # Every invocation, including the ones printed as instructions: an
    # instruction that does not work for the reader it is printed to is worse
    # than no instruction.
    body = INSTALL[INSTALL.index('D="${HELENA_DOCKER'):]
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        assert not re.search(r"(?<![\w$-])docker\s+(compose|info|volume|run|system)\b", code), (
            f"this still calls docker directly, so HELENA_DOCKER will not "
            f"cover it: {line.strip()!r}")


def test_the_busy_port_check_uses_a_tool_the_host_will_have() -> None:
    """A guard that skips itself on the machines that need it is not a guard.

    The check was `command -v lsof && lsof ...`. Measured on Ubuntu: iproute2,
    which provides `ss`, is Priority: important and is on effectively every
    install; lsof is Priority: standard and is absent from minimal and cloud
    images -- which is what somebody installs this on. So the check quietly did
    nothing exactly there, and the busy port surfaced where the comment above it
    says it must not: after the build, as a compose bind error.

    Verified on a clean host: 8800 in use dies before anything is built, a free
    port passes on to the next guard, and a base image with neither tool warns
    rather than reporting a port it never looked at.
    """
    # Comments stripped: the prose above this check discusses `command -v lsof`
    # by name, and matching that read the explanation as the implementation --
    # which reported lsof first in a file where it is the fallback.
    code = "\n".join(line.split("#", 1)[0] for line in INSTALL.splitlines())

    assert "command -v ss" in code, (
        "the port check no longer tries ss, so it is skipped on any host "
        "without lsof -- which is most of them")
    assert code.index("command -v ss") < code.index("command -v lsof"), (
        "lsof is tried first, so the more widely present tool is the fallback "
        "rather than the default")

    # And an unchecked port has to be distinguishable from a free one.
    tail = code[code.index("command -v ss"):]
    assert "warn " in tail[:1200], (
        "with neither tool present the check passes silently, which tells the "
        "reader the port is free when nothing looked")

    for probe in ("command -v ss", "command -v lsof"):
        assert code.index(probe) < code.index("git clone"), (
            f"{probe!r} runs after the clone, so the check reports after the "
            "work it exists to save")


def test_it_asks_what_to_install_and_can_be_told() -> None:
    """One command used to install something that could not run a phase.

    True, and said at the end, and still the wrong thing to hand somebody who
    typed one line: a panel that queues work and nothing that runs it. It asks
    now -- panel, panel plus CPU workers, panel plus GPU workers -- and takes
    --panel/--cpu/--gpu or HELENA_INSTALL when nobody is there to answer.
    """
    for flag in ("--panel", "--cpu", "--gpu"):
        assert flag in INSTALL, f"{flag} is gone; a non-interactive install cannot choose"
    assert "HELENA_INSTALL" in INSTALL

    # The workers come from deploy-platform.sh, not from a second copy of what
    # the platform is. Two sets of compose invocations is how they drift.
    assert "deploy-platform.sh" in INSTALL, (
        "the installer brings workers up some other way, which duplicates what "
        "deploy-platform.sh already decides")


def test_the_prompt_reads_the_terminal_and_not_stdin() -> None:
    """The published way to run this is `curl -fsSL ... | sh`.

    stdin is the script there. A `read` without /dev/tty eats the rest of the
    installer and the shell runs whatever is left of it.
    """
    assert "read -r reply < /dev/tty" in INSTALL, (
        "the prompt reads stdin, which under `curl | sh` is the installer")


def test_no_terminal_is_detected_by_opening_it() -> None:
    """`[ -r /dev/tty ]` answers yes where there is no controlling terminal.

    It asks access(2) about the device node, which exists either way. Measured:
    in a process with no controlling terminal the test passes and the open
    fails -- so the installer would have tried to prompt, the read would have
    failed, and the default would have installed CPU workers on a machine where
    nobody was asked. An hour of compiling, unprompted.
    """
    # Comments stripped, for the third time in this file's history and the same
    # reason each time: the paragraph above the check names the wrong form in
    # order to say why it is wrong, and matching prose reads an explanation as
    # an implementation.
    code = "\n".join(line.split("#", 1)[0] for line in INSTALL.splitlines())
    assert "(exec < /dev/tty)" in code, (
        "the no-terminal check does not open the terminal in a subshell, so "
        "either it cannot tell an absent terminal from a present one, or the "
        "failed open kills the installer")
    assert "[ -r /dev/tty ]" not in code
    assert "{ : < /dev/tty" not in code, (
        "`:` is a special built-in and a redirection error on one is fatal to "
        "a non-interactive shell: under dash this exits the installer instead "
        "of falling back")

    # And the check must behave: this process has no controlling terminal.
    # And the form must survive a shell with no terminal rather than dying in
    # it. Under dash the `{ : ...; }` version printed nothing at all.
    probe = subprocess.run(
        ["sh", "-c", 'if ! (exec < /dev/tty) 2>/dev/null; then echo none; else echo some; fi'],
        capture_output=True, text=True, check=True)
    assert probe.stdout.strip() in ("none", "some"), (
        f"the check produced {probe.stdout!r}: the shell died on the redirect")


def test_the_helpers_exist_before_the_arguments_are_parsed() -> None:
    """`install.sh --bogus` called die() from above where die() is defined.

    Under `set -eu` that is `die: not found` and an exit status nobody reads as
    "unknown argument". The order is the whole of the fix.
    """
    code = "\n".join(line.split("#", 1)[0] for line in INSTALL.splitlines())
    assert code.index("die()") < code.index("--panel)"), (
        "the argument parser runs before die() exists, so a bad argument "
        "reports a missing command instead of the argument")
