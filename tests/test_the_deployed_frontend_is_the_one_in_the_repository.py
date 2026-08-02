"""What reaches a host has to include what the image builds from.

The panel image grew a node stage so a clean checkout could build it. At that
moment the frontend's build input stopped being `panel/web/dist`, produced on
whoever's laptop ran the deploy, and became `panel/web` on the host.

sync-to-host.sh was still sending only dist. So the image built from a source
tree that had not been updated in weeks, and gpu-1 came up with a panel missing
its user guide, its API reference and its developer reference. Nothing looked
wrong: the build succeeded, the image label carried the right commit -- because
the Python half really was current -- and only the pages were old.

That is the failure worth pinning. A deploy that silently ships stale bytes is
worse than one that fails, because the second kind gets fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = (ROOT / "containers/sync-to-host.sh").read_text()
DEPLOY = (ROOT / "containers/deploy-to-host.sh").read_text()
CI = (ROOT / ".gitlab-ci.yml").read_text()
CONTAINERFILE = (ROOT / "containers/images/Containerfile.panel").read_text()


def _copied_into_the_web_stage() -> list[str]:
    """The paths the frontend build stage reads, from the Containerfile."""
    stage = CONTAINERFILE[CONTAINERFILE.index("AS web"):]
    stage = stage[: stage.index("\nFROM ")]
    return [match.group(1) for match in
            re.finditer(r"^COPY\s+(?!--from)(\S+)", stage, re.MULTILINE)]


def test_everything_the_image_builds_from_is_sent_to_the_host() -> None:
    """The build context is the host's tree, so a path the image reads and the
    sync does not send is a path that is silently stale."""
    for path in _copied_into_the_web_stage():
        # A directory sent whole covers everything under it: framework/contracts
        # arrives because the script sends framework. So any prefix counts.
        parts = path.split("/")
        prefixes = ["/".join(parts[:n]) for n in range(len(parts), 0, -1)]
        assert any(prefix in SYNC for prefix in prefixes), (
            f"the panel image builds from {path} and sync-to-host.sh sends no "
            f"prefix of it ({prefixes}). The image will build from whatever that "
            "host happens to have, which is how a deploy ships a frontend from "
            "weeks earlier while reporting the right commit"
        )


def test_the_frontend_source_is_sent_and_not_only_its_build_output() -> None:
    """The specific regression, named.

    dist may still be sent -- it is harmless -- but the source must be, because
    the source is what the image compiles.
    """
    assert '"$here/panel/web/"' in SYNC, (
        "sync-to-host.sh does not send panel/web; the image's node stage will "
        "compile the host's stale copy"
    )
    assert "--exclude 'node_modules'" in SYNC, (
        "node_modules would be copied over the wire on every deploy, and npm ci "
        "recreates it from the lock file anyway"
    )


def test_the_sync_proves_a_frontend_file_arrived() -> None:
    """The script verifies hashes rather than trusting rsync, and that list has
    to reach the half that broke.

    It checked panel/app.py and panel/requirements.txt -- both of which were
    arriving correctly the whole time the frontend was not.
    """
    verified = re.search(r"for file in ([^\n;]+(?:\\\n[^\n;]+)?); do\n"
                         r"\s*mine=", SYNC)
    assert verified, "the hash check's file list could not be read"
    checked = verified.group(1).replace("\\\n", " ")
    assert "panel/web/" in checked, (
        "nothing under panel/web is hash-checked, so a frontend that did not "
        "arrive still exits zero"
    )


# --------------------------------------------------------------------------
# The stronger guarantee: the host does not build at all
# --------------------------------------------------------------------------

def test_the_panel_is_deployed_as_an_image_rather_than_built_on_the_target() -> None:
    """A host that compiles its own copy can compile the wrong one.

    Every check above guards the sync that feeds a build on the target. This is
    the reason those guards should become unnecessary: the image is built once,
    named by commit, pushed, and pulled. The bytes are identical everywhere or
    the tag is different -- there is no third outcome, which is what the stale
    tree produced.
    """
    panel = DEPLOY[DEPLOY.index("  panel|all)"):]
    panel = panel[: panel.index("\n    ;;")]
    assert "pull panel" in panel, "the host is not asked to pull a published image"
    assert "--no-build" in panel, "compose may still rebuild on the host"
    assert "build-panel.sh" not in panel, (
        "the deploy builds the panel. Nothing outside the pipeline should: "
        "building on the target uses whatever tree is there, and building on "
        "the operator's machine needs a Docker daemon they may not have"
    )
    # And it must refuse a tag the pipeline never published, rather than
    # deploying something that happens to be cached on the host.
    assert "/v2/helena/helena-panel/manifests/" in panel, (
        "the deploy does not check the image exists in the registry"
    )


def test_ci_publishes_the_image_it_tested() -> None:
    """Pushing an image nobody opened is how an unexamined build becomes the
    thing every host trusts."""
    build = CI[CI.index("build the panel image:"):]
    build = build[: build.index("\n# ---")]
    assert "docker push $HELENA_REGISTRY/helena-panel:$CI_COMMIT_SHA" in build
    # The push is last: the import check and the bundle check come first.
    assert build.index("import panel.app") < build.index("docker push")
    assert build.index("UserGuide") < build.index("docker push")


def test_the_deployed_tag_names_a_commit() -> None:
    """`latest` is how two hosts run different bytes under one name."""
    for text, what in ((CI, "the pipeline"), (DEPLOY, "the deploy script")):
        assert "helena-panel:latest" not in text, (
            f"{what} deploys a floating tag"
        )


def test_every_job_names_the_image_the_build_produced() -> None:
    """One line kept the old name after the registry prefix was introduced.

    The build tagged $HELENA_REGISTRY/helena-panel:$SHA and a later check in the
    same job still ran `helena-panel:$SHA`, which no longer existed. The job
    failed at that line, the pipeline stopped before deploy, and the registry
    stayed empty -- a rename that missed one occurrence out of five.
    """
    import re

    referenced = set(re.findall(r"docker (?:run --rm|push|image inspect)\s+(\S*helena-panel:\S+)", CI))
    assert referenced, "no job references the panel image any more"
    for name in referenced:
        assert name.startswith("$HELENA_REGISTRY/"), (
            f"{name} is not the name the build job tags, so this step runs an "
            "image that was never produced"
        )


def test_the_deploy_asks_for_the_tag_the_pipeline_publishes() -> None:
    """CI tags with $CI_COMMIT_SHA, the full hash.

    The deploy script derived every name from `git rev-parse --short`, so it
    looked up a tag the pipeline never writes and refused every image it had
    genuinely built. The refusal was correct and the question was wrong.
    """
    assert 'helena-panel:$commit_full' in DEPLOY, (
        "the deploy names the panel image by short hash; CI publishes the full one"
    )
    assert "manifests/$commit_full" in DEPLOY, (
        "the existence check asks the registry about a tag CI never publishes"
    )
    assert "$CI_COMMIT_SHA" in CI and "$CI_COMMIT_SHORT_SHA" not in CI.split(
        "docker push")[0].split("-t $HELENA_REGISTRY")[-1], (
        "the pipeline stopped tagging with the full hash"
    )


def test_an_image_is_only_built_where_one_is_wanted() -> None:
    """Publishing a tag per branch push is minutes of build time and a registry
    entry for work nobody deploys.

    The rules were lost when the job was rewritten for the registry -- the kind
    of thing that costs nothing visible and is never noticed.
    """
    import yaml

    build = yaml.safe_load(CI)["build the panel image"]
    conditions = " ".join(r.get("if", "") for r in build["rules"])
    assert 'staging' in conditions and "merge_request_event" in conditions
    assert build["rules"], "the build job runs on every branch again"


def test_a_job_that_needs_the_host_is_pinned_to_it() -> None:
    """A tag is what makes placement deterministic.

    Untagged, these landed on whichever runner answered first -- sometimes an
    instance runner with no Docker socket, where they refuse -- so a push either
    deployed or did not, depending on nothing.

    It matters more now than it did. With two deployments, a deploy job on the
    wrong runner does not merely fail: it deploys development over the GPU host,
    or staging over the CPU one.
    """
    import yaml

    parsed = yaml.safe_load(CI)
    for name, host in (
        ("deploy to swisspost-1", "swisspost-1"),
        ("smoke on swisspost-1", "swisspost-1"),
        ("deploy to gpu-1", "gpu-1"),
        ("smoke on gpu-1", "gpu-1"),
        ("heavy", "gpu-1"),
        ("build the panel image", "swisspost-1"),
    ):
        assert parsed[name].get("tags") == [host], (
            f"{name} is not pinned to {host}"
        )
    # unit tests is pinned too, but for a different reason: its image is built
    # on that runner's daemon and pushed nowhere, so it exists on one machine.
    assert parsed["unit tests"].get("tags") == ["swisspost-1"]
    # frontend needs only a public image and stays unpinned, or it queues behind
    # a single machine for no reason.
    assert not parsed["frontend"].get("tags"), (
        "frontend needs only a container and should run anywhere"
    )


def test_each_branch_deploys_to_exactly_one_host() -> None:
    """development is swisspost-1 and staging is gpu-1, and neither is both.

    A job whose rules let it run on both branches would deploy one branch's code
    to the other's machine on whichever pipeline ran second.
    """
    import yaml

    parsed = yaml.safe_load(CI)
    expected = {
        "deploy to swisspost-1": "development",
        "smoke on swisspost-1": "development",
        "deploy to gpu-1": "staging",
        "smoke on gpu-1": "staging",
    }
    for name, branch in expected.items():
        conditions = [r.get("if", "") for r in parsed[name]["rules"]]
        assert conditions == [f'$CI_COMMIT_BRANCH == "{branch}"'], (
            f"{name} should run on {branch} and only there, not {conditions}"
        )


def test_every_job_that_touches_a_deployment_proves_it_is_on_that_host() -> None:
    """Nothing pins these but a tag, and a tag is a request.

    The check reads the name from the Docker daemon, which is the host rather
    than the container. gpu-1 runs a panel and holds a socket too, so it
    satisfies every weaker check.
    """
    import yaml

    parsed = yaml.safe_load(CI)
    for name in (".deploy", ".smoke", "heavy"):
        job = parsed[name]
        steps = " ".join(job.get("before_script", []) + job.get("script", []))
        assert "docker info" in steps and "HELENA_DEPLOY_HOST" in steps, (
            f"{name} touches a deployment without proving which host it is on"
        )
    # And each concrete job has to say which host it means, or the guard above
    # compares against an empty string and passes anywhere.
    for name in ("deploy to swisspost-1", "deploy to gpu-1",
                 "smoke on swisspost-1", "smoke on gpu-1", "heavy"):
        assert parsed[name]["variables"].get("HELENA_DEPLOY_HOST"), (
            f"{name} does not name the host it is allowed to touch"
        )


def test_neither_verification_stage_can_pass_by_skipping() -> None:
    """smoke had no credentials and all 44 tests opted out, green.

    heavy is the same shape with more at stake: two tests that skip when their
    upstream artefacts are missing, on a stage that costs half an hour of two
    GPUs. A scheduled run that skipped both would report success having proved
    nothing about the science.
    """
    import yaml

    parsed = yaml.safe_load(CI)
    for name in (".smoke", "heavy"):
        assert parsed[name]["variables"].get("HELENA_E2E_NO_SKIP") == "1", (
            f"{name} can report success without asserting anything"
        )


def test_the_deploy_is_one_script_called_with_a_profile() -> None:
    """Two deployments, one definition of what a deployment is.

    The moment the GPU host and the CPU host are described by two different
    scripts is the moment they start drifting, which is exactly how ink, backup
    and host-report ended up six commits behind.
    """
    import yaml

    parsed = yaml.safe_load(CI)
    steps = " ".join(parsed[".deploy"]["script"])
    assert "deploy-platform.sh" in steps, (
        "the deploy does not call the script that holds the service list"
    )
    assert "$HELENA_PROFILE" in steps, "the profile is not passed"
    for name, profile in (("deploy to swisspost-1", "nogpu"),
                          ("deploy to gpu-1", "gpu")):
        assert parsed[name]["variables"]["HELENA_PROFILE"] == profile


def test_the_deploy_script_covers_every_compose_project() -> None:
    """`all` used to mean three of six.

    deploy-to-host.sh had branches for panel, segment and qc, so ink, backup and
    host-report were never deployed by anything automatic -- and the job exited
    zero, which is the part that let it go unnoticed through several releases.
    """
    script = (ROOT / "containers/deploy-platform.sh").read_text()
    for project in ("helena-segment", "helena-host-report", "helena-ink-0",
                    "helena-qc-", "helena-backup"):
        assert project in script, f"{project} is not deployed by this script"
    # platform.compose.yaml carries postgres, init, panel and backup.
    assert "platform.compose.yaml" in script


def test_a_machine_with_no_card_is_not_asked_to_reserve_one() -> None:
    """A `gpus:` reservation is not conditional: on a host with no NVIDIA driver
    the container refuses to start. So the nogpu profile has to stop before the
    stacks that reserve one, rather than starting them and failing."""
    script = (ROOT / "containers/deploy-platform.sh").read_text()
    stop = script.index('if [ "$profile" = nogpu ]')
    after = script[stop:]
    assert "exit 0" in after[:400], "the nogpu profile does not stop"
    # The GPU stacks must all come after that exit.
    for gpu_only in ("ink.compose.yaml", "surface-qc.compose.yaml"):
        assert gpu_only not in script[:stop], (
            f"{gpu_only} reserves a card and is started before the nogpu exit"
        )
    # host-report is the exception: it runs on both, and the overlay is what
    # adds the reservation.
    assert "host-report.gpu.compose.yaml" in script


def test_the_runner_can_read_the_host_configuration_it_deploys_with() -> None:
    """The compose files read /etc/helena, which is deliberately not in the
    repository. Without the mount the deploy job cannot bring up anything but
    the panel, and would say only that a variable was missing."""
    runner = (ROOT / "containers/compose/runner.compose.yaml").read_text()
    assert "/etc/helena:/etc/helena" in runner, (
        "the runner cannot read the env files the deploy needs"
    )
    # And the part that actually matters, written down because it cost a failed
    # deploy: this compose mount is the manager's, and the manager is not what
    # runs the job. Job containers take their mounts from config.toml.
    assert "config.toml" in runner, (
        "nothing says the job containers need this in config.toml too, which is "
        "where it has to be for a deploy job to see it"
    )
    assert "/etc/helena:/etc/helena:ro" not in runner, (
        "a deploy records the image it installed by writing the tag back; "
        "read-only would leave the file disagreeing with the running container"
    )


def test_the_suite_does_not_reinstall_its_dependencies_every_push() -> None:
    """44% of a pipeline was apt-get and pip install.

    124 seconds of the unit job's 284, fetching the same wheels every time. The
    image that holds them is rebuilt every pipeline and costs nothing when the
    requirements have not changed, because only those files are COPYied before
    the install -- measured at 0 seconds unchanged.

    GitLab's pip cache was measured too and is worse: 38 seconds saved for a
    2.7 GB cache the runner has to move.
    """
    import yaml

    parsed = yaml.safe_load(CI)
    assert "prepare" in parsed["stages"]
    prepare = parsed["build the ci image"]
    assert prepare["stage"] == "prepare", "the image is built after it is needed"

    unit = parsed["unit tests"]
    assert unit["image"] == "helena-ci:$CI_COMMIT_SHA", (
        "the unit job does not use the prepared image"
    )
    steps = " ".join(unit.get("before_script", []) + unit.get("script", []))
    assert "pip install" not in steps, "the job installs dependencies again"
    assert "apt-get" not in steps

    # Built locally and never pushed: the job that uses it runs on that same
    # daemon, and a registry round-trip for three gigabytes undoes the saving.
    assert "docker push" not in " ".join(prepare["script"])
    assert unit.get("tags") == ["swisspost-1"], (
        "the image exists only on that runner's daemon, so the job must run there"
    )


def test_the_ci_image_installs_only_the_requirements_before_the_copy() -> None:
    """The layer cache is the whole mechanism. COPYing the repository before the
    install would rebuild it on every commit, which is what this avoids."""
    dockerfile = (ROOT / "containers/images/Containerfile.ci").read_text()
    # The RUN, not the first mention: the header comment says "pip install" and
    # matching that put the boundary above every COPY, so this test passed
    # nothing and reported it as a failure.
    install = dockerfile.index("RUN pip install")
    copied = re.findall(r"^COPY\s+(\S+)", dockerfile[:install], re.MULTILINE)
    assert copied, "nothing is copied before the install"
    for path in copied:
        assert path.endswith("requirements.txt"), (
            f"{path} is copied before the dependency install, so any change to "
            "it rebuilds the whole install"
        )


def test_the_ci_image_does_not_accumulate_forever() -> None:
    """One 2.5 GB image per pipeline, kept for good, is a host that fills up.

    It did. Twenty-three of them plus their build cache took swisspost-1 to zero
    bytes free, and every pipeline after that failed inside apt-get with a
    message about /var/cache/apt rather than about disk -- so the cause looked
    like a broken base image for as long as nobody ran df.

    The prepare stage was written knowing it would leave an image behind and
    without saying what removes it. This is that.
    """
    import yaml

    prepare = yaml.safe_load(CI)["build the ci image"]
    steps = " ".join(prepare["script"])
    assert "docker rmi" in steps, (
        "nothing removes the images this job leaves behind, and there is one per "
        "pipeline"
    )
    assert "builder prune" in steps, "the build cache behind them is never reclaimed"
    # And never the one just built: the job exists to produce it.
    assert 'grep -v "helena-ci:$CI_COMMIT_SHA"' in steps, (
        "the pruning does not exclude the image this pipeline is about to use"
    )
