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
                 "publish_running helena-gpu-runtime-0 worker-gpu",
                 "publish_built containers/images/Containerfile.backup backup",
                 "publish_built containers/images/Containerfile.control-tunnel control-tunnel"):
        assert call in script, f"missing: {call}"


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


def test_the_panel_worker_cpp_and_worker_gpu_all_try_the_public_registry_first():
    for image, tag_var in (
        ("helena-panel", "panel_image"),
        ("helena-worker-cpp", "worker_tag"),
        ("helena-worker-gpu", "ink_tag"),
    ):
        assert f'"$HELENA_PUBLIC_REGISTRY/{image}:' in DEPLOY, (
            f"{image} never tries HELENA_PUBLIC_REGISTRY before building")
        assert tag_var in DEPLOY


def test_a_pull_from_either_registry_skips_the_local_build():
    """Both the internal registry and HELENA_PUBLIC_REGISTRY have to be tried
    and have to fail before an hour of compiling volume-cartographer starts --
    a pull that succeeded from either must not be followed by a build anyway."""
    for flag in ("panel_pulled", "worker_pulled", "ink_pulled"):
        assert f"{flag}=false" in DEPLOY
        assert f'{flag}" = false' in DEPLOY or f'"{flag}" = false' in DEPLOY
        assert f"{flag}=true" in DEPLOY
