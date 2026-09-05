"""Publishing is a manual, credentialed step -- and, once it exists, a second
source a deploy tries before it spends an hour compiling volume-cartographer.

Neither half runs today: no image has been pushed, and HELENA_PUBLIC_REGISTRY
only turns itself on for a host that had no config before this deploy. These
hold the shape both halves are supposed to have, so the day someone adds
DOCKERHUB_TOKEN and clicks the job, what runs is what was reviewed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = (ROOT / ".gitlab-ci.yml").read_text()
CI_PARSED = yaml.safe_load(CI)
DEPLOY = (ROOT / "containers/deploy-platform.sh").read_text()


def test_the_publish_job_is_manual_and_stays_on_its_host():
    job = CI_PARSED["publish images to docker hub"]
    assert job["needs"] == ["deploy to gpu-1"], (
        "publishing images that were never verified by a deploy is not the point")
    rule = job["rules"][0]
    assert rule["when"] == "manual", "a push to a public registry must not run on its own"
    assert rule["allow_failure"] is True, (
        "an unclicked manual job must not turn the pipeline red")
    assert job["variables"]["HELENA_DEPLOY_HOST"] == "gpu-1", (
        "the host guard reads this; unset, its own case pattern matches anything "
        "and the guard enforces nothing")
    assert "No Docker daemon" in job["script"][0], (
        "the host guard (.on_its_host) must be the first script line")


def test_the_job_refuses_gracefully_with_no_credentials():
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert '-z "$DOCKERHUB_USER"' in script and '-z "$DOCKERHUB_TOKEN"' in script, (
        "nothing guards the login; a fork with neither variable set would fail "
        "loudly instead of skipping")
    assert "exit 0" in script, (
        "missing credentials must not turn this optional job's pipeline red")


def test_every_image_the_readme_named_is_published():
    """helena-$repo is built inside publish_running/publish_built from the
    repo name passed as an argument, so the literal helena-<name> string
    never appears in the script text -- the call site is what to check."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    for call in ("publish_running helena-panel panel",
                 "publish_running helena-segment worker-cpp",
                 'publish_running "${gpu_runtime:-helena-gpu-runtime-0}" worker-gpu',
                 "publish_built containers/images/Containerfile.backup backup",
                 "publish_built containers/images/Containerfile.control-tunnel control-tunnel"):
        assert call in script, f"missing: {call}"


def test_the_gpu_runtime_container_is_found_by_its_running_name():
    """gpu-1 itself runs its one card as helena-gpu-runtime-1, not -0 -- the
    index is the device's, and a hardcoded -0 skipped a host that had
    something to publish."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert "docker ps --filter 'name=^helena-gpu-runtime-'" in script


def test_published_tags_are_by_version_not_by_commit():
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert "$version" in script and "cat VERSION" in script, (
        "a public tag has to mean the same thing on every clone, which a "
        "commit sha from this project's own history cannot")


def test_a_host_that_already_had_a_config_never_turns_this_on_by_itself():
    """gpu-1 and work-3 both had a platform.env before this existed. Staging
    exists to run the commit under test; a published tag is whatever the last
    release was, which is not guaranteed to be that commit.
    """
    block = DEPLOY[DEPLOY.index("HELENA_PUBLIC_REGISTRY=\"${HELENA_PUBLIC_REGISTRY:-}\""):]
    block = block[: block.index("panel_image=")]
    assert "platform_env_is_new" in block, (
        "the default is not conditioned on whether this host is actually new")
    assert 'HELENA_PUBLIC_REGISTRY="docker.io/limegs"' in block


def test_an_explicit_value_always_wins_over_the_default():
    idx = DEPLOY.index('HELENA_PUBLIC_REGISTRY="${HELENA_PUBLIC_REGISTRY:-}"')
    assert idx > 0
    # ${VAR:-} preserves a caller's own non-empty value and only substitutes
    # empty for a truly unset one; the freshness check that follows only fires
    # when the result is still empty.
    after = DEPLOY[idx: idx + 300]
    assert 'if [ -z "$HELENA_PUBLIC_REGISTRY" ]' in after


def test_every_built_image_tries_the_public_registry_first():
    for image, tag_var in (
        ("helena-panel", "panel_image"),
        ("helena-worker-cpp", "worker_tag"),
        ("helena-worker-gpu", "ink_tag"),
        ("helena-backup", "helena-backup:local-"),
    ):
        assert f'"$HELENA_PUBLIC_REGISTRY/{image}:' in DEPLOY, (
            f"{image} never tries HELENA_PUBLIC_REGISTRY before building")
        assert tag_var in DEPLOY


def test_a_pull_from_either_registry_skips_the_local_build():
    """Both the internal registry and HELENA_PUBLIC_REGISTRY have to be tried
    and have to fail before an hour of compiling volume-cartographer starts --
    a pull that succeeded from either must not be followed by a build anyway."""
    for flag in ("panel_pulled", "worker_pulled", "ink_pulled", "backup_pulled"):
        assert f"{flag}=false" in DEPLOY
        assert f'{flag}" = false' in DEPLOY or f'"{flag}" = false' in DEPLOY
        assert f"{flag}=true" in DEPLOY


# -- the toolchain images ----------------------------------------------------
#
# helena-villa and helena-villa-python are the expensive half and were the two
# nothing published: a stranger's first install compiled volume-cartographer
# for an hour because the only copies lived on the hosts that had already done
# so. They are not services, so they are published from the image the host
# already carries rather than from a container or a fresh build.


def test_the_toolchain_images_are_published_too():
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert 'publish_toolchain "${HELENA_REGISTRY:+${HELENA_REGISTRY%/}/}helena-villa:local" villa volume_cartographer' in script
    assert 'publish_toolchain "${HELENA_VILLA_PYTHON_IMAGE:-helena-villa-python:local}" villa-python villa_python' in script


def test_the_toolchain_is_not_recompiled_to_publish_it():
    """publish_built would rebuild villa inside this job, which is the hour
    this whole mechanism exists to remove."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    for wrong in ("publish_built containers/images/Containerfile.villa villa",
                  "publish_built containers/images/Containerfile.villa-python"):
        assert wrong not in script, f"the toolchain is being rebuilt to publish it: {wrong}"


def test_a_stale_toolchain_image_is_never_published():
    """The gate in build-worker.sh stops a stale villa reaching a worker.
    Pushing one as `latest` would hand it to every fresh install instead --
    the same failure one registry further out."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert "org.helena.villa.commit" in script, (
        "nothing reads what the image actually carries")
    assert '[ "$carries" != "$pinned" ]' in script, (
        "the label is read and never compared against the lock")
    assert "skipping helena-$repo rather than publishing a stale toolchain" in script


def test_the_toolchain_carries_its_upstream_identity_as_a_tag():
    """$version says which Helena release; it does not say which villa. An
    upstream commit means the same thing to every clone, which is exactly the
    property a tag from this project's own history lacks."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert 'short="villa-$(printf %.12s "$carries")"' in script
    assert 'for name in "$version" "$short" latest' in script


def test_the_lock_is_read_without_mounting_a_path_this_job_does_not_share():
    """This job talks to the host's daemon from inside a container, so $PWD
    here is not $PWD there -- a -v mount of the checkout would be empty or
    something else entirely. The contents are handed over as an argument, the
    way deploy-platform.sh already does it."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert "$(cat containers/images/scrollfiesta/locks/source-lock.json)" in script
    assert "-v " not in script.split("lock_commit()")[1].split("}")[0], (
        "the lock is read through a bind mount this job cannot rely on")


def test_the_published_toolchain_names_no_private_host():
    """The registry prefix is expanded from HELENA_REGISTRY, never written
    out: this file is published to the public mirror."""
    script = "\n".join(CI_PARSED["publish images to docker hub"]["script"])
    assert "${HELENA_REGISTRY:+${HELENA_REGISTRY%/}/}helena-villa:local" in script


# -- building the toolchain in the first place --------------------------------
#
# Publishing can only push what exists, and helena-villa did not: gpu-1 pulls
# it from the internal registry, which served 05dcf034 for five weeks after the
# lock moved. Nothing in this repository produced a newer one, because the
# deploy's own gate reads a registry-qualified name as one to pull rather than
# build -- correct for a deploy, and exactly wrong for the job that has to make
# the thing.

TOOLCHAIN = "build and publish the villa toolchain"


def test_the_toolchain_job_is_manual_and_on_its_host():
    job = CI_PARSED[TOOLCHAIN]
    assert job["rules"][0]["when"] == "manual", (
        "an hour of volume-cartographer must not start on its own")
    assert job["rules"][0]["allow_failure"] is True, (
        "an unclicked manual job must not turn the pipeline red")
    assert job["variables"]["HELENA_DEPLOY_HOST"] == "gpu-1"
    assert "No Docker daemon" in job["script"][0], (
        "the host guard must be the first script line")


def test_it_builds_through_the_script_that_holds_the_pins():
    """Restating commit, tree and repository in YAML is a second copy to keep
    in step with the lock, and the first one to fall out of it."""
    script = "\n".join(CI_PARSED[TOOLCHAIN]["script"])
    assert "sh containers/build-worker.sh" in script
    for restated in ("VILLA_COMMIT=", "VILLA_TREE=", "Containerfile.villa"):
        assert restated not in script, (
            f"the job restates {restated}, which build-worker.sh already pins")


def test_the_base_name_is_unprefixed_so_the_script_builds_it():
    """Prefixed, build-worker.sh reads the name as one to pull rather than
    build -- which is why no deploy has ever produced this image."""
    script = "\n".join(CI_PARSED[TOOLCHAIN]["script"])
    assert "BASE_IMAGE=helena-villa:local" in script


def test_what_is_pushed_is_tagged_by_what_it_carries():
    """Off the image's own label rather than off the lock: if those two ever
    disagree, the tag should say what is in the layers."""
    script = "\n".join(CI_PARSED[TOOLCHAIN]["script"])
    assert 'carries="$(docker image inspect "$local_image"' in script
    assert 'for name in "villa-$short" local' in script


def test_it_publishes_both_toolchain_images():
    script = "\n".join(CI_PARSED[TOOLCHAIN]["script"])
    assert "publish_internal helena-villa:local helena-villa" in script
    assert "publish_internal helena-villa-python:local helena-villa-python" in script


def test_it_is_given_room_to_finish():
    """The default job timeout is an hour and this is an hour of compiling."""
    assert CI_PARSED[TOOLCHAIN]["timeout"] == "3h"


def test_the_internal_registry_comes_from_the_variable():
    """Never written out: this file reaches the public mirror. The repository
    wide check in test_the_deployed_frontend_is_the_one_in_the_repository is
    what enforces that; this only pins where the name comes from."""
    script = "\n".join(CI_PARSED[TOOLCHAIN]["script"])
    assert "$HELENA_REGISTRY/" in script
    assert '[ -n "$HELENA_REGISTRY" ]' in script, (
        "with no registry configured this must say so, not push to a bare name")


def test_the_toolchain_push_is_bounded_and_verified_against_the_registry():
    """`docker push` retries a refused layer with no cap of its own. The first
    run of the job spent an hour re-sending an 8.4 GB layer into a registry
    answering 500; only the 3 h timeout would have ended it. And an exit code
    from push is not the bytes being there: the registry's tag list is."""
    script = "\n".join(CI_PARSED[TOOLCHAIN]["script"])
    assert 'timeout 1500 docker push "$HELENA_REGISTRY/$repo:$name"' in script
    # Through the daemon, not curl: the registry's certificate is from an
    # internal CA that a stock curl image does not carry, and the first
    # version of this check read that TLS failure as an empty tag list and
    # failed a push that had landed.
    assert 'docker pull -q "$HELENA_REGISTRY/$repo:$name"' in script, (
        "nothing asks the registry whether the tag landed")
    assert "/tags/list" not in script, "a curl without the CA reads as an empty list"
    assert "is not in the registry after the push" in script
