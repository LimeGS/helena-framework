"""A deploy has to leave every service on what it just built.

Three services once sat six commits behind through several releases, and every
deploy in between exited zero. Nothing was broken -- `all` simply did not mean
all, and nothing ever looked afterwards.

Wiring alone does not fix that. A service missing from a stack, a compose that
decided nothing had changed, a container that started and died a second later:
all three look exactly like success from outside. So the deploy ends by
checking, and these tests make sure it keeps checking everything.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = (ROOT / "containers/deploy-platform.sh").read_text()

# Read out of the deploy rather than restated beside it. A hand-kept list is
# the same bug this file exists for, one level up: the 9 um slot ran six months
# on the image it was born with because the deploy had never been told about it,
# and a list here that nobody updates would let the next such slot through the
# same way. Every compose file the deploy brings up is checked, and adding one
# enrols it automatically.
COMPOSE_NAMES = sorted(set(re.findall(r"\$compose/([A-Za-z0-9._-]+)\.compose\.yaml",
                                     DEPLOY)))
COMPOSE = {
    name: yaml.safe_load((ROOT / f"containers/compose/{name}.compose.yaml").read_text())
    for name in COMPOSE_NAMES
}


def test_the_deploy_brings_up_every_stack_this_file_knows_to_check() -> None:
    """The list is derived, so this guards the derivation rather than the list:
    a deploy that stopped naming its compose files by path would silently check
    nothing at all."""
    assert {"platform", "segment", "ink", "surface-qc", "spiral"} <= set(COMPOSE_NAMES), (
        f"the deploy names only {COMPOSE_NAMES}; a stack it stopped bringing up "
        "is a stack nothing below checks"
    )


def _image_variables() -> set[str]:
    """Every ${VAR} a compose file uses to name a service's image."""
    found = set()
    for document in COMPOSE.values():
        for service in (document.get("services") or {}).values():
            image = service.get("image", "")
            match = re.match(r"\$\{([A-Z_]+)", str(image))
            if match:
                found.add(match.group(1))
    return found


def test_every_image_a_compose_file_names_is_set_by_the_deploy() -> None:
    """A variable nobody sets falls back to a default in the compose file --
    `helena-worker-cpp:0.10.0`, a version tag that stopped moving long ago. The
    service then runs whatever that means on this host, forever, while the
    deploy reports the new commit."""
    for variable in _image_variables():
        assert variable in DEPLOY, (
            f"{variable} names a service's image and the deploy never sets it, "
            "so that service keeps whatever it is running now"
        )


def test_the_deploy_refreshes_the_images_it_does_not_build() -> None:
    """postgres is a pinned third-party tag, and `compose up` uses whatever copy
    is on the host. Once pulled it stayed forever, so a patch release of
    Postgres would never have reached either deployment."""
    assert "postgres_image=" in DEPLOY
    pull_section = DEPLOY[: DEPLOY.index("set_image platform.env")]
    assert "$postgres_image" in pull_section and "pull" in pull_section, (
        "postgres is never pulled, so the host keeps its first copy indefinitely"
    )


def test_the_deploy_checks_what_it_left_running() -> None:
    """The whole point. Bringing a stack up is not evidence it came up."""
    assert "expect()" in DEPLOY, "the deploy does not verify anything afterwards"
    checked = set(re.findall(r'expect "?(helena-[a-z0-9-]+|\$\{?\w+)"?', DEPLOY))
    for container in ("helena-postgres", "helena-panel", "helena-segment",
                      "helena-fleet-runner", "helena-host-report",
                      "helena-backup", "helena-ink-0"):
        assert container in checked, f"{container} is never verified after the deploy"
    # The QC workers are one container per card, so they are checked in a loop.
    assert "helena-gpu-runtime-$device" in DEPLOY


def test_qc_receipts_use_the_commit_being_deployed() -> None:
    """The QC env file is long-lived host configuration, not provenance.

    If ``HELENA_QC_CODE_COMMIT`` comes only from that file, every new image can
    keep signing evidence with the commit from the day the host was installed.
    The deploy must export the resolved full SHA and compose must override the
    host env value with it.
    """
    assert 'export HELENA_COMMIT_FULL="$commit_full"' in DEPLOY
    qc_environment = COMPOSE["surface-qc"]["services"]["surface-qc"]["environment"]
    assert qc_environment["HELENA_QC_CODE_COMMIT"] == \
        "${HELENA_COMMIT_FULL:-${HELENA_QC_CODE_COMMIT:?set in surface-qc.env}}"


def test_a_service_on_the_wrong_image_fails_the_deploy() -> None:
    """Reporting drift is not enough; the pipeline has to go red.

    A deploy that prints STALE and exits zero is a deploy that lies, which is
    the failure this replaces rather than repeats.
    """
    tail = DEPLOY[DEPLOY.index("expect()"):]
    assert 'if [ "$drift" != "0" ]' in tail
    assert "exit 7" in tail, "drift is detected and the deploy still succeeds"


def test_the_check_waits_before_looking() -> None:
    """A container that dies on startup is running for a second or two first."""
    tail = DEPLOY[DEPLOY.index("expect()"):]
    assert "sleep" in tail[: tail.index("drift=0")], (
        "the check runs immediately, so a container that dies on startup passes"
    )


def test_an_absent_backup_does_not_fail_the_check() -> None:
    """A host with no backup destination is a supported configuration.

    Written as an `if`: under `set -e` an AND-list whose first test fails is a
    failed command, so `[ -n "$backup_profile" ] && expect ...` exits the script
    on exactly the hosts this was written to allow.
    """
    assert 'if [ -n "$backup_profile" ]; then' in DEPLOY, (
        "the backup check is not guarded by an if, so a host without a backup "
        "destination fails its own deploy"
    )


def test_the_check_compares_bytes_and_not_only_a_name() -> None:
    """A tag is a pointer, and something can re-point it.

    Caught by it happening: the same commit was built twice, once per branch,
    and the second build moved the tag under a container that kept running the
    first. The container's image *string* was right the whole time. Comparing
    names would have called that up to date.
    """
    tail = DEPLOY[DEPLOY.index("expect()"):]
    assert "{{.Image}}" in tail and "{{.Id}}" in tail, (
        "the check compares image names only, so a container running older "
        "bytes under a re-pointed tag passes"
    )


def test_an_image_is_built_once_per_commit() -> None:
    """Docker builds are not reproducible by default.

    Both branches build the same commit on the same runner, so a second build
    produced different bytes and the second push re-pointed the tag. One commit
    then named two different images, which makes "this tag is those exact
    bytes" -- the premise the whole deploy rests on -- false.
    """
    ci = (ROOT / ".gitlab-ci.yml").read_text()
    build = ci[ci.index("build the panel image:"):]
    build = build[: build.index("\n# ---")]
    assert "already published" in build.lower(), (
        "the build job rebuilds a commit that is already in the registry"
    )
    # The check has to come before the build, or it has already happened.
    #
    # Anchored to the command and not the words "docker build", which also
    # appear in a comment further up -- matching prose put the boundary in the
    # wrong place and failed a job that was correct.
    assert build.index("manifests/$CI_COMMIT_SHA") < build.index("- docker build -f")


def test_a_deploy_works_without_the_private_registry() -> None:
    """The README tells people to run this script.

    HELENA_REGISTRY defaults to a VIP inside one particular network, and the
    pull that follows is the first thing the script does. For anybody cloning
    this from GitHub that is an immediate failure on the documented command --
    so a pull that cannot happen falls back to building from the checkout.

    Still a fallback, not the default: an image built per host is bytes only
    that host has, which is the thing publishing exists to avoid.
    """
    fallback = DEPLOY[DEPLOY.index("panel_image="):]
    fallback = fallback[: fallback.index("set_image platform.env")]
    assert '$D pull -q "$panel_image"' in fallback, (
        "the internal registry is tried first, and its absence must not skip "
        "straight to a build without the pull ever having been attempted"
    )
    assert 'if [ "$panel_pulled" = false ]' in fallback, (
        "a pull that never succeeded -- not just one that failed once -- must "
        "still reach the build fallback, so the documented command cannot "
        "work outside the network HELENA_REGISTRY points into"
    )
    assert "Containerfile.panel" in fallback, "there is no fallback build"
    assert "exit 5" in fallback, "a panel that can be neither pulled nor built passes"
