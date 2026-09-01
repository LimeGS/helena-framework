"""A machine that has never run Helena must be able to install it.

The published quickstart is one curl line. It could not work on any machine
that had not already been prepared by hand, and nothing said so.

Docker creates a named volume as root:root 0755. Every service in the platform
compose runs as 1000:1000 since the fleet moved off root. So on a clean host the
panel starts, cannot create /state/tls, and exits:

    mkdir: cannot create directory '/state/tls': Permission denied

The existing hosts never showed it because their volumes were chowned by hand
during that migration -- which is what the compose comment means by "the volumes
are 1000:1000 on the hosts". A first install has nobody to have done that.

Found by running install.sh on a host with Docker and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "containers/compose/platform.compose.yaml"


@pytest.fixture(name="compose")
def _compose():
    return yaml.safe_load(PLATFORM.read_text())


def test_something_prepares_the_volumes_before_anything_uses_them(compose):
    services = compose["services"]
    assert "prepare-volumes" in services, (
        "a clean host has root-owned volumes and non-root services; something "
        "has to reconcile that before the panel starts")
    assert services["prepare-volumes"].get("user") == "root", (
        "only root can chown a root-owned volume, which is the whole job")


def test_every_volume_the_panel_writes_is_prepared(compose):
    """A volume the panel writes and nobody prepared is the same bug again,
    one directory over."""
    def mounted(service):
        return {entry.split(":")[0] for entry in
                compose["services"][service].get("volumes", [])
                if not entry.endswith(":ro")}

    panel = mounted("panel")
    prepared = mounted("prepare-volumes")
    assert panel <= prepared, f"the panel writes but nothing prepares: {panel - prepared}"


def test_the_preparation_runs_before_the_services_that_need_it(compose):
    """Ordering is the whole value: chowning after the panel has already failed
    to start is not a fix."""
    init = compose["services"]["init"].get("depends_on", {})
    assert "prepare-volumes" in init
    assert init["prepare-volumes"]["condition"] == "service_completed_successfully", (
        "started is not enough -- the chown has to have finished")
    # The panel reaches it transitively through init, which is what orders it.
    assert "init" in compose["services"]["panel"].get("depends_on", {})


def test_the_preparation_gives_up_the_privilege_it_borrowed(compose):
    """Root for one container that exits immediately is the trade. Root with a
    network and a full capability set is not."""
    prepare = compose["services"]["prepare-volumes"]
    assert prepare.get("network_mode") == "none"
    assert prepare.get("cap_drop") == ["ALL"]
    assert set(prepare.get("cap_add", [])) <= {"CHOWN", "DAC_OVERRIDE", "FOWNER"}
    assert "no-new-privileges:true" in prepare.get("security_opt", [])
    assert prepare.get("restart") == "no"


def test_a_volume_somebody_already_owns_is_left_alone(compose):
    """A bind mount to a directory with its own owner is not this container's
    to rewrite, and chown -R over a populated artifact store is slow."""
    command = " ".join(str(part) for part in
                       compose["services"]["prepare-volumes"]["command"])
    assert "stat -c %u" in command
    assert 'if [ "$$owner" = "0" ]' in command


def test_the_installer_notices_state_from_an_earlier_install():
    """Volumes outlive `compose down`. A fresh install adopts them silently,
    and if the previous one ran as a different user the panel dies inside
    uvicorn's load_cert_chain, which does not look like the cause."""
    installer = (ROOT / "install.sh").read_text()
    assert "helena-panel-state" in installer
    assert "HELENA_ADOPT_VOLUMES" in installer, (
        "refusing is right, but there has to be a way past it")
    assert "load_cert_chain" in installer, (
        "the check should name the error it prevents; that is why it exists")


def test_the_readme_states_what_the_quickstart_needs():
    readme = (ROOT / "README.md").read_text()
    for needed in ("usermod -aG docker", "DockerRootDir", "Compose v2"):
        assert needed in readme, f"the prerequisites do not mention {needed}"
    for absent in ("Node", "CUDA", "GPU"):
        assert absent in readme, (
            f"what is *not* required matters as much, and {absent} is not named")


# What this file used to assert here: that build-worker.sh *explains*
# helena-villa when it is missing -- where the source lives and how to hand it
# in -- because the alternative was `pull access denied`, which sends a newcomer
# looking for credentials that do not exist. The script now builds it instead,
# which is better than any message, so the assertion moved to
# test_the_villa_image_builds_itself.py rather than being loosened here.


def test_the_readme_is_honest_about_what_the_worker_deploy_needs():
    """It said `deploy-platform.sh nogpu` and nothing else, and that was false in
    two ways: the deploy exited 4 without env files nothing created, and then hit
    a base image that had to be compiled. Both were fixed rather than documented
    -- the deploy seeds the env files from the templates now -- so what the page
    has to keep being honest about is the compile, which no fix removes.

    The env files moved out of /etc and into the checkout, so a page still
    telling somebody to `sudo cp` into /etc/helena would be describing a step
    that is now neither required nor read by default.
    """
    readme = (ROOT / "README.md").read_text()
    assert "provision-host.sh" in readme
    assert "compiled from source" in readme, (
        "the first worker deploy compiles volume-cartographer and the page has "
        "to say so; it is the longest thing that happens")

    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    assert '.env.example' in deploy, (
        "nothing seeds the env files, so the README's promise that the deploy "
        "writes them is a promise the deploy does not keep")
    if "/etc/helena" in readme:
        assert "config/" in readme, (
            "the README names /etc/helena without naming the checkout config "
            "directory that is now the default")


def test_env_backups_are_rotated():
    """Each deploy copies /etc/helena/<stack>.env aside and kept every copy
    forever. The CI runner had 119 of them beside two real files."""
    source = (ROOT / "containers/deploy-platform.sh").read_text()
    assert 'ls -1t "$file".bak-*' in source
    assert "tail -n +11" in source, "keep ten; enough for a bad week"


def test_every_env_file_the_deploy_demands_has_an_example():
    """deploy-platform.sh exits 4 without /etc/helena/<stack>.env, and nothing
    generates them. `segment.env` had no example at all, so the only way to
    learn its contents was to read the compose or to already own a host that
    worked."""
    import re  # noqa: PLC0415

    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    named = set(re.findall(r"env_dir/([a-z-]+)\.env", deploy))
    examples = {path.name.removesuffix(".env.example")
                for path in (ROOT / "containers/compose").glob("*.env.example")}

    # postgres.env is written by the deploy from the platform's own settings,
    # not copied from an example. A template for it would have to carry a
    # placeholder password, and a placeholder that reaches a running deployment
    # is the failure this whole file is about: the shipped ones said CHANGEME
    # and REPLACE, and every worker on a clean machine tried to reach a
    # database that was not there.
    generated = {"postgres"}
    assert 'printf \'POSTGRES_PASSWORD=%s\\n\'' in deploy, (
        "postgres.env is exempt from needing an example because the deploy "
        "writes it; if it stopped writing it, it needs one")

    assert named - generated <= examples, (
        f"no example for: {sorted(named - generated - examples)}")


def test_the_segment_example_carries_the_one_variable_with_no_default():
    """Everything else in segment.compose.yaml defaults; the host id does not,
    and two machines sharing one claim each other's work."""
    example = (ROOT / "containers/compose/segment.env.example").read_text()
    assert "HELENA_SEGMENT_HOST_ID=" in example
    compose = (ROOT / "containers/compose/segment.compose.yaml").read_text()
    assert "${HELENA_SEGMENT_HOST_ID}" in compose, (
        "if this gains a default, the example should say so rather than imply "
        "it is still required")


def test_the_readme_does_not_claim_provision_host_writes_those_files():
    """It does not: it copies a control host's panel.env to a target as
    worker.env, which is for a second machine joining a fleet that already has
    a panel. Saying otherwise sends the first installer down a path that cannot
    work -- which this README did until the from-scratch test caught it."""
    readme = (ROOT / "README.md").read_text()
    assert "provision-host.sh" in readme
    provision = (ROOT / "containers/provision-host.sh").read_text()
    written = set(__import__("re").findall(r"tee -a? ?/etc/helena/([a-z]+)\.env",
                                           provision))
    assert written <= {"worker"}, (
        f"provision-host.sh now writes {written}; the README says it does not")


def test_the_gpu_profile_builds_its_base_rather_than_refusing():
    """It used to fail at the last step and name the wrong thing.

    `deploy-platform.sh gpu` failed at the ink-worker build -- after
    volume-cartographer had been compiled and the worker built on top of it --
    with "the ink-worker image failed to build". What was missing was its base:
    helena-gpu-runtime took two parent images outside this repository's build
    graph, one of which wanted a bundle assembled from binaries nobody could
    rebuild.

    Then it refused early and said what to set, which was honest and still left
    a reader stuck. All four are produced here now, villa included, so it builds
    them. The check that remains is the one no build can remove: villa must
    exist before the GPU base, because that is where the tools are compiled --
    but "must exist" is satisfied by building it, not by sending the reader to
    another profile and exiting.
    """
    deploy = (ROOT / "containers/deploy-platform.sh").read_text()

    gate = deploy.index('if [ "$profile" = gpu ]')
    worker_build = deploy.index("build-worker.sh")
    assert gate < worker_build, (
        "the GPU base is settled after the worker is built, which is the hour "
        "of compiling this check exists to spend only when it can pay off")

    # To the end of the gpu block, not a byte count. It was gate + 2600, and a
    # comment added inside the block pushed the build out of the window: the
    # test failed while the property it names was still true.
    guidance = deploy[gate:deploy.index("# Point an env file's image variable", gate)]
    assert "Containerfile.gpu-runtime" in guidance, (
        "the deploy no longer builds the surface-QC base, so a host without it "
        "is stuck again")
    assert "Containerfile.ink" in guidance, "nor its ink parent"
    # It used to refuse and name the profile that would produce villa, and the
    # message was honest: villa really was built elsewhere. That left `--gpu`,
    # which the installer offers as a choice, unable to finish on a clean
    # machine -- measured on a rented 5090: a panel, no workers, and a warning
    # to go run the other profile. So the property is no longer "it says what
    # builds villa" but "it builds villa".
    assert 'sh "$root/containers/build-worker.sh"' in guidance, (
        "the gpu profile does not build villa, so a host that cannot pull the "
        "GPU base is stuck being told to deploy nogpu first")
    assert "exit 4" in guidance, (
        "nothing stops the profile when villa still is not there afterwards")


def test_the_readme_does_not_promise_a_profile_needs_nothing_it_needs():
    """`gpu` needed an image this repository could not build, and the page
    offered the two profiles as adjacent lines in one code block -- which reads
    as a choice between more and less rather than between self-contained and
    not. Both are self-contained now, so the page may say so; if that stops
    being true it has to stop saying it.
    """
    readme = (ROOT / "README.md").read_text()
    workers = readme[readme.index("### Workers"):]
    if "Neither profile needs anything" not in workers:
        return                       # the claim was withdrawn; nothing to hold

    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    assert "Containerfile.gpu-runtime" in deploy and "Containerfile.ink" in deploy, (
        "the README says neither profile needs anything external, and the "
        "deploy does not build the GPU half")


def test_configuration_lives_in_the_checkout_and_not_in_etc():
    """Installing this needed a privileged step nothing else about it needed.

    Everything here is a container and a volume -- no systemd units, no venvs,
    no host-prepared paths -- except the configuration, which the README told
    you to `sudo cp` into /etc/helena and which the deploy then wrote back to on
    every run. In the checkout it is one directory beside the code, writable by
    whoever cloned it, and gone when they delete it.

    A host configured before this keeps working: /etc/helena is still read when
    it holds configuration and the checkout does not, because a deploy that
    silently starts reading a different, empty directory recreates a running
    stack with defaults.
    """
    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    assert 'env_dir="$root/config"' in deploy, (
        "the default configuration directory is no longer inside the checkout")
    assert "/etc/helena/platform.env" in deploy, (
        "nothing falls back to /etc/helena, so every host configured before "
        "this silently deploys with template defaults")

    ignored = (ROOT / ".gitignore").read_text()
    assert "*.env" in ignored and "!*.env.example" in ignored, (
        "config now sits in the checkout; without these the next `git add -A` "
        "commits a database password")


def test_object_storage_credentials_are_the_panels_to_keep():
    """A worker is ephemeral and cannot be where a credential lives.

    The panel holds these in the control plane, write-only, and workers adopt
    them at startup -- but an environment variable wins over the control plane,
    deliberately. So a key left in an env file makes the panel's copy inert
    while the page still reports it as set: somebody sets it there, nothing
    changes, and nothing anywhere says why. Whoever is meant to administer this
    from a web page cannot debug a variable they never see.
    """
    names = (ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py").read_text()
    assert "AWS_SECRET_ACCESS_KEY" in names, (
        "the panel no longer manages object-storage credentials")

    example = (ROOT / "containers/compose/segment.env.example").read_text()
    assert "AWS_SECRET_ACCESS_KEY=" not in example, (
        "the template invites a value that silently overrides the panel")
    assert "set on the panel" in example, (
        "and it does not say where they do belong")

    adopt = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    assert "overridden by this worker's environment" in adopt, (
        "a worker shadowing the panel's credential says nothing about it, "
        "which is the whole failure this is about")

def test_the_e2e_env_file_cannot_be_poisoned_by_a_commit_message():
    """A pipeline that fails on the prose in a commit message.

    The e2e jobs build an env file with `env | grep '^HELENA_'`. GitLab exports
    the commit message as a variable and `env` prints multi-line values raw, so
    any line of any commit message that begins with `HELENA_` was collected as
    if it were a variable. One did:

        docker: invalid env file (/tmp/e2e.env): variable 'HELENA_REGISTRY a
        build script takes a branch it never takes on the runner.' contains
        whitespaces

    -- a sentence out of the commit that was being tested, which is a failure
    nobody can predict from the code. The filter has to require the shape of an
    assignment, not just the prefix.
    """
    ci = (ROOT / ".gitlab-ci.yml").read_text()
    if "e2e.env" not in ci:
        return                       # the jobs are gone; nothing to hold

    for line in ci.splitlines():
        if "e2e.env" not in line or "grep" not in line:
            continue
        assert "=" in line.split("grep", 1)[1].split(">")[0], (
            f"this collects any line starting with the prefix, so a commit "
            f"message can name a variable: {line.strip()!r}")
