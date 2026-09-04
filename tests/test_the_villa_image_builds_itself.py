"""The worker's base image can be built from the repository alone.

`helena-villa` is volume-cartographer compiled from source. Nothing publishes
it, and the Containerfile used to refuse to fetch anything at all -- "a build
that clones is a build whose output depends on the day it ran" -- so the source
had to be handed in as a build context that only somebody who already knew
about it would have. Installing the workers was a thing a stranger could read
about and not do.

The objection is right about cloning a *branch* and wrong about cloning a
commit: what makes a fetch unreproducible is the moving reference, not the
network. So the build clones the commit the source lock pins, shallow, and then
checks the tree hash it actually received. A rewritten tag, a mirror serving
something else, a truncated fetch: all produce a different tree, and the build
stops before the toolchain is installed.

Measured on a clean host: the correct tree builds; a wrong one exits 3 saying
which tree arrived and which was expected.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINERFILE = ROOT / "containers/images/Containerfile.villa"
LOCK = ROOT / "containers/images/scrollfiesta/locks/source-lock.json"


def lock() -> dict:
    return json.loads(LOCK.read_text())["volume_cartographer"]


def _recorded_build_argv(env=None):
    """Run build-worker.sh against a docker that records its argv instead of building.

    Reading the source proves an argument is spelled somewhere; it does not
    prove it survives the shell, and the defaults here are assembled from
    nested fallbacks rather than written out.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "docker").write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "image" ]; then exit 1; fi\n'   # nothing is on this host
            # `build` and `buildx build`: the worker is built with the second,
            # and recording only the first made this watch every villa build and
            # no worker build at all -- which is exactly where the uv default is.
            'if [ "$1" = "build" ] || [ "$1$2" = "buildxbuild" ]; then\n'
            '  printf "%s\\n" "$@" >>"$RECORD"\n'
            'fi\n'
            'exit 0\n')
        (tmp / "docker").chmod(0o755)
        record = tmp / "argv"
        environ = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "RECORD": str(record)}
        # Do not inherit the host's deployment configuration. With
        # HELENA_REGISTRY set -- which the CI runner does, whatever the pipeline
        # file says -- the base name carries a registry host, the `case` treats
        # it as fetchable and the villa branch is skipped entirely: no build to
        # record, and a failure about a digest that has nothing to do with what
        # is being measured. These tests are about the script, not the machine
        # it happens to run on.
        for name in ("HELENA_REGISTRY", "BASE_IMAGE", "VILLA_IMAGE_DIGEST",
                     "VILLA_BASE_IMAGE", "UV_CONTEXT"):
            environ.pop(name, None)
        for key, value in (env or {}).items():
            environ.pop(key, None) if value is None else environ.update({key: value})
        done = subprocess.run(
            ["sh", str(ROOT / "containers/build-worker.sh"), str(ROOT)],
            capture_output=True, text=True, env=environ)
        assert record.exists(), (
            f"the build never reached docker build:\n{done.stdout}{done.stderr}")
        return record.read_text().splitlines()


def test_the_lock_pins_a_tree_and_not_only_a_commit():
    """A commit sha alone cannot be verified after the fact: it says what was
    asked for, not what arrived."""
    entry = lock()
    assert re.fullmatch(r"[0-9a-f]{40}", entry["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", entry.get("tree", "")), (
        "volume_cartographer needs a tree hash, the way scrollfiesta already has one")


def test_the_build_fetches_that_commit_and_verifies_what_it_got():
    source = CONTAINERFILE.read_text()
    assert "git fetch -q --depth 1 origin" in source, "by commit, never by branch"
    assert 'git rev-parse HEAD^{tree}' in source
    assert "exit 3" in source, "a mismatched tree has to stop the build"
    assert "${VILLA_COMMIT}" in source and "${VILLA_TREE}" in source


def test_the_stage_is_named_so_existing_callers_still_win():
    """BuildKit lets a build context replace a stage of the same name. The
    Makefile passes `--build-context villa_src=<checkout>`; if the stage were
    called anything else those callers would silently become clones of a commit
    they did not choose."""
    source = CONTAINERFILE.read_text()
    assert re.search(r"^FROM scratch AS villa_src$", source, re.M)
    assert "COPY --from=villa_src" in source
    makefile = (ROOT / "containers/images/Makefile").read_text()
    assert "--build-context villa_src=" in makefile


def test_the_fetched_tree_sits_where_a_build_context_would_put_it():
    """`--build-context villa_src=/path/to/villa` mounts that directory as the
    stage root, so a clone that leaves the repository one level down makes the
    two paths mean different things -- which is how the clone build failed on a
    file the checkout build found."""
    source = CONTAINERFILE.read_text()
    assert "FROM scratch AS villa_src\nCOPY --from=clone /src /" in source
    for copied in re.findall(r"COPY --from=villa_src (\S+)", source):
        assert not copied.startswith("/src"), (
            f"{copied} assumes the clone's own layout rather than the context's")


def test_the_base_image_default_carries_the_toolchain_upstream_asks_for():
    """install_build_deps.sh asks apt for flang-21 and libclang-rt-21-dev with
    no repository of its own. Measured: ubuntu 24.04 and 25.04 have neither and
    the build dies at "Unable to locate package libclang-rt-21-dev" after
    installing everything before it."""
    builder = (ROOT / "containers/build-worker.sh").read_text()
    assert "ubuntu:25.10" in builder
    assert "flang-21" in builder, "the reason for the version belongs beside it"
    assert "VILLA_BASE_IMAGE" in builder, "and it has to be overridable"


def test_the_worker_build_builds_its_base_rather_than_explaining_it():
    builder = (ROOT / "containers/build-worker.sh").read_text()
    assert "Containerfile.villa" in builder
    assert "source-lock.json" in builder, (
        "the commit and tree come from the lock, not from the caller")
    assert "$here" in builder, (
        "the lock and the Containerfile are the repository's, and the build "
        "context passed in is not always the repository")


# Directories whose .sh files are not ours to hold to this: vendored code, a
# virtualenv's activate scripts, node's bin shims.
NOT_OURS = {".git", ".venv", "venv", "node_modules", "__pycache__",
            "vendored", ".claude", "dist", "build"}


def _shell_scripts():
    """Every script in the tree whose shebang promises POSIX sh.

    Walked off disk rather than asked of `git ls-files`, which exits 128 inside
    a CI container: the checkout belongs to another user there and git refuses
    it as dubious ownership. This file's own CI config already documents that
    for a different variable, and the test still used git -- so it passed on a
    laptop and failed on the machine that matters.
    """
    for path in sorted(ROOT.rglob("*.sh")):
        if NOT_OURS & set(path.relative_to(ROOT).parts):
            continue
        try:
            first = path.read_text(errors="replace").splitlines()[:1]
        except OSError:
            continue
        if first and first[0].strip() in ("#!/bin/sh", "#!/usr/bin/env sh"):
            yield path


def test_what_says_it_is_posix_sh_parses_as_posix_sh():
    """`#!/bin/sh` is dash on every Debian and Ubuntu host this deploys to.

    Written and tested under bash on a Mac, a bashism parses fine and ships
    broken: this build grew an array to hold one optional argument, which bash
    accepts and dash rejects at parse time -- so the failure would not have been
    the missing argument, it would have been the whole script refusing to run,
    on the machine of whoever installed this first.
    """
    checker = shutil.which("dash") or "sh"
    for script in _shell_scripts():
        done = subprocess.run([checker, "-n", str(script)], capture_output=True, text=True)
        assert done.returncode == 0, f"{script.relative_to(ROOT)}: {done.stderr.strip()}"


def test_the_build_hands_the_pinned_commit_and_tree_to_docker():
    """What the script *says* it builds, checked by watching what it runs."""
    pinned = lock()
    argv = _recorded_build_argv()
    assert f"VILLA_COMMIT={pinned['commit']}" in argv, "the pinned commit did not reach the build"
    assert f"VILLA_TREE={pinned['tree']}" in argv, (
        "the tree did not reach the build, so the fetch would go unverified")
    assert f"VILLA_REPOSITORY={pinned['repository']}" in argv
    assert "-t" in argv and "helena-villa:local" in argv


# Bare names -- no registry, no namespace -- resolve to Docker Hub's `library/`,
# which only holds the official images. `ubuntu` is one; `uv` is not, and that
# is not visible from the name. Anything added here should be checked against
# hub.docker.com/_/<name> rather than assumed.
DOCKER_OFFICIAL = {"ubuntu", "debian", "alpine", "python", "postgres", "redis"}


def test_no_default_image_only_resolves_inside_the_fleet():
    """A default that needs the cluster registry is a default that fails outside it.

    `uv:0.11.32` looked like the others and meant Docker Hub's `library/uv`,
    which does not exist -- unlike `ubuntu:25.10` beside it, which does. Inside
    the fleet HELENA_REGISTRY is always set, so the mirror answered and the
    broken default was never reached; a stranger got `pull access denied [...]
    insufficient_scope`, which reads as a login problem and is not one: there is
    no account that would have helped. Measured on a host with no registry configured.

    Read off what the script actually runs with no registry configured, rather
    than off how the defaults are spelled: the name that broke was assembled
    from nested shell fallbacks, not written out.
    """
    # VILLA_IMAGE_DIGEST because the worker build is only reached once the base
    # has a resolved digest, and a stand-in docker cannot produce one -- the
    # same escape a local-only base legitimately uses. Without it the script
    # exits before the worker build, where the name under test actually lives,
    # and this check passed while the bug it was written for was in place.
    argv = _recorded_build_argv(env={
        "HELENA_REGISTRY": None,
        "VILLA_IMAGE_DIGEST": "helena-villa@sha256:" + "0" * 64,
    })
    assert "buildx" in argv, "the worker build was never reached"
    images = [a.split("=", 1)[1] for a in argv if a.startswith("uv_context=")]
    images += [a.split("=", 1)[1] for a in argv if a.startswith("BASE_IMAGE=")]
    assert images, "the build named no images at all, which cannot be right"

    for image in images:
        name = image.split("//")[-1]
        if name.startswith("helena-"):
            continue                      # built here, never pulled
        # Only a name with a slash can carry a registry or a namespace. Without
        # one there is nothing to inspect: `uv:0.11.32` is bare, and the colon
        # in it is the tag, not a registry port -- reading it as a port is how
        # this check first passed while the bug it was written for was in place.
        if "/" in name:
            continue                      # names its registry or its namespace
        assert name.split(":")[0] in DOCKER_OFFICIAL, (
            f"{name} is a bare name, so with no HELENA_REGISTRY set it resolves "
            f"to Docker Hub's library/{name.split(':')[0]} -- which exists only "
            f"for official images. Qualify it, or add it above if it is one.")


# -- absent and stale are the same answer ------------------------------------
#
# The gates used to ask only "is the image here". gpu-1 kept its 05dcf034 build
# through every deploy after the lock moved to 23adee04 -- the image was
# present, so nothing rebuilt it, and the workers that came out carried a
# toolchain their own lock did not pin. Nothing said so.


def _recorded_build_argv_on_a_host_that_has(label, env=None):
    """Same harness, but `docker image inspect` succeeds and reports a label.

    The fake above answers "nothing is on this host" to every inspect, so it can
    only ever exercise the absent case. A stale image is a *present* one, which
    is exactly what the old gate could not tell apart from a current one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "docker").write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then\n'
            # The label query the gate makes, and the two the digest resolution
            # makes afterwards; anything else about the image is not asked for.
            '  case "$*" in\n'
            '    *org.helena.villa.commit*) printf "%s\\n" "' + label + '" ;;\n'
            '    *.Id*) echo "sha256:feed0000" ;;\n'
            '    *RepoDigests*) echo "" ;;\n'
            '  esac\n'
            '  exit 0\n'
            'fi\n'
            'if [ "$1" = "build" ] || [ "$1$2" = "buildxbuild" ]; then\n'
            '  printf "%s\\n" "$@" >>"$RECORD"\n'
            'fi\n'
            'exit 0\n')
        (tmp / "docker").chmod(0o755)
        record = tmp / "argv"
        record.write_text("")
        environ = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "RECORD": str(record)}
        for name in ("HELENA_REGISTRY", "BASE_IMAGE", "VILLA_IMAGE_DIGEST",
                     "VILLA_BASE_IMAGE", "UV_CONTEXT"):
            environ.pop(name, None)
        for key, value in (env or {}).items():
            environ.pop(key, None) if value is None else environ.update({key: value})
        done = subprocess.run(
            ["sh", str(ROOT / "containers/build-worker.sh"), str(ROOT)],
            capture_output=True, text=True, env=environ)
        return record.read_text().splitlines(), done


def test_an_image_already_at_the_locked_commit_is_not_rebuilt():
    """The whole point of the old gate, which this must not cost: a host that
    is already current does not recompile volume-cartographer."""
    argv, _ = _recorded_build_argv_on_a_host_that_has(lock()["commit"])

    assert f"VILLA_COMMIT={lock()['commit']}" not in argv, (
        "an image already at the locked commit was rebuilt anyway")


def test_an_image_at_some_other_commit_is_rebuilt():
    """05dcf034 is the commit gpu-1 actually kept; the lock has moved since."""
    stale = "05dcf0349356bc833670d61e5eca00be58376e35"
    assert lock()["commit"] != stale, "pick a commit the lock does not pin"

    argv, done = _recorded_build_argv_on_a_host_that_has(stale)

    assert f"VILLA_COMMIT={lock()['commit']}" in argv, (
        "a stale base image was reused instead of rebuilt:\n"
        f"{done.stdout}{done.stderr}")


def test_the_stale_lane_image_is_rebuilt_too():
    """Two images read this lock, and the spiral lane is the one gpu-1 ran the
    old fitter out of."""
    stale = "05dcf0349356bc833670d61e5eca00be58376e35"
    villa_python = json.loads(LOCK.read_text())["villa_python"]
    assert villa_python["commit"] != stale

    argv, done = _recorded_build_argv_on_a_host_that_has(stale)

    # Its own `-t helena-villa-python:local`, not the worker build's
    # `LANE_IMAGE=helena-villa-python:local`: the second is there either way,
    # and matching it would pass against the gate this test exists to catch.
    tagged = [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "-t"]
    assert "helena-villa-python:local" in tagged, (
        "the stale spiral lane image was reused instead of rebuilt:\n"
        f"{done.stdout}{done.stderr}")


def test_an_image_with_no_villa_label_is_left_alone():
    """An image built before the stamp existed cannot be judged by it, and
    turning "I cannot tell" into an hour of compiling is its own failure."""
    argv, _ = _recorded_build_argv_on_a_host_that_has("")

    assert f"VILLA_COMMIT={lock()['commit']}" not in argv, (
        "an unlabelled image was rebuilt on a guess")


def test_a_registry_base_that_disagrees_with_the_lock_says_so():
    """A registry-qualified base is pulled, not built here, so the gate cannot
    fix it -- gpu-1 pulled 05dcf034 for five weeks after the lock moved. It
    still must not come out of the build quietly."""
    stale = "05dcf0349356bc833670d61e5eca00be58376e35"
    _, done = _recorded_build_argv_on_a_host_that_has(
        stale, env={"HELENA_REGISTRY": "registry.example.invalid"})

    assert "WARNING" in done.stderr and "lock pins" in done.stderr, (
        f"a stale registry base was compiled against in silence:\n{done.stdout}{done.stderr}")


# -- a published toolchain, tried before the compile --------------------------
#
# The publish job pushes helena-villa and helena-villa-python under
# villa-<upstream commit>. That only saves anybody the hour if the build tries
# the pull first -- and only stays safe if what comes back is checked rather
# than trusted for having the right name.


def _recorded_argv_with_a_public_registry(label, pull_succeeds):
    """A host with a stale image and a public registry that may or may not
    have the locked toolchain."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pulled_label = json.loads(LOCK.read_text())["volume_cartographer"]["commit"]
        (tmp / "docker").write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "pull" ]; then\n'
            f'  echo "$@" >>"$PULLED"; exit {0 if pull_succeeds else 1}\n'
            'fi\n'
            'if [ "$1" = "tag" ]; then echo "$@" >>"$PULLED"; exit 0; fi\n'
            'if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then\n'
            '  case "$*" in\n'
            # Before a pull the host has the stale one; after a successful pull
            # the retagged image is the locked one. The marker file is what
            # separates the two, the way a real retag would.
            '    *org.helena.villa.commit*)\n'
            '      if [ -f "$PULLED" ] && grep -q "^tag" "$PULLED"; then\n'
            f'        echo "{pulled_label}"\n'
            '      else\n'
            f'        echo "{label}"\n'
            '      fi ;;\n'
            '    *.Id*) echo "sha256:feed0000" ;;\n'
            '    *RepoDigests*) echo "" ;;\n'
            '  esac\n'
            '  exit 0\n'
            'fi\n'
            'if [ "$1" = "build" ] || [ "$1$2" = "buildxbuild" ]; then\n'
            '  printf "%s\\n" "$@" >>"$RECORD"\n'
            'fi\n'
            'exit 0\n')
        (tmp / "docker").chmod(0o755)
        record, pulled = tmp / "argv", tmp / "pulled"
        record.write_text("")
        environ = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}",
                   "RECORD": str(record), "PULLED": str(pulled),
                   "HELENA_PUBLIC_REGISTRY": "docker.io/limegs"}
        for name in ("HELENA_REGISTRY", "BASE_IMAGE", "VILLA_IMAGE_DIGEST",
                     "VILLA_BASE_IMAGE", "UV_CONTEXT"):
            environ.pop(name, None)
        done = subprocess.run(
            ["sh", str(ROOT / "containers/build-worker.sh"), str(ROOT)],
            capture_output=True, text=True, env=environ)
        return (record.read_text().splitlines(),
                pulled.read_text().splitlines() if pulled.exists() else [], done)


def test_a_published_toolchain_is_pulled_by_the_commit_the_lock_pins():
    """Not by release: these move on upstream's cadence, so a release tag
    would be whatever villa the last one happened to carry."""
    stale = "05dcf0349356bc833670d61e5eca00be58376e35"
    _, pulled, _ = _recorded_argv_with_a_public_registry(stale, pull_succeeds=True)

    wanted = json.loads(LOCK.read_text())["volume_cartographer"]["commit"][:12]
    assert any(f"docker.io/limegs/helena-villa:villa-{wanted}" in line for line in pulled), (
        f"the pull did not name the locked commit: {pulled}")


def test_a_successful_pull_skips_the_compile():
    """The whole point: an hour of volume-cartographer, not paid."""
    stale = "05dcf0349356bc833670d61e5eca00be58376e35"
    argv, _, done = _recorded_argv_with_a_public_registry(stale, pull_succeeds=True)

    tagged = [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "-t"]
    assert "helena-villa:local" not in tagged, (
        f"villa was compiled anyway after a successful pull:\n{done.stdout}{done.stderr}")


def test_a_registry_without_it_still_builds():
    """A published image that is not there is not a reason to give up."""
    stale = "05dcf0349356bc833670d61e5eca00be58376e35"
    argv, _, done = _recorded_argv_with_a_public_registry(stale, pull_succeeds=False)

    assert f"VILLA_COMMIT={lock()['commit']}" in argv, (
        f"a failed pull left nothing building it:\n{done.stdout}{done.stderr}")


def test_the_deploy_exports_the_variable_the_build_reads():
    """A plain assignment is not visible to build-worker.sh: a value that
    arrived in the environment stays exported and one the default just set
    does not -- so villa would compile on exactly the fresh host the default
    exists for."""
    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    assert "export HELENA_PUBLIC_REGISTRY" in deploy


# -- the environment the deploy actually runs this in -------------------------
#
# .deploy uses image: docker:27-cli, which has no python at all. read_lock
# shelled out to python3, so every read came back empty there -- and the gate
# read empty as "nothing to compare against" and reused whatever the host had.
# A fixed gate that is a no-op on the only machine it has to work on is not a
# fix. Measured: the deploy of 71510092 carried the gate and rebuilt nothing;
# helena-villa-python:local stayed at 05dcf034 through it.


def _run_without_python3(docker_run_prints):
    """build-worker.sh on a PATH with no python3, the way the deploy has it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "docker").write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "run" ]; then\n'
            f'  {docker_run_prints}\n'
            '  exit 0\n'
            'fi\n'
            'if [ "$1" = "image" ]; then exit 1; fi\n'
            'if [ "$1" = "build" ] || [ "$1$2" = "buildxbuild" ]; then\n'
            '  printf "%s\\n" "$@" >>"$RECORD"\n'
            'fi\n'
            'exit 0\n')
        (tmp / "docker").chmod(0o755)
        record = tmp / "argv"
        record.write_text("")
        # A PATH with the ordinary utilities and no python of any name --
        # /usr/bin/python3 exists on the machine these tests run on, which is
        # exactly what the deploy's image does not have. Symlinked rather than
        # filtered by PATH order, since `command -v` would find it either way.
        binz = tmp / "bin"
        binz.mkdir()
        for source in ("/bin", "/usr/bin"):
            for tool in Path(source).iterdir():
                if tool.name.startswith("python") or (binz / tool.name).exists():
                    continue
                try:
                    (binz / tool.name).symlink_to(tool)
                except OSError:
                    pass
        environ = {**os.environ, "PATH": f"{tmp}:{binz}",
                   "RECORD": str(record), "HELENA_PYTHON_IMAGE": "python:3.11-slim"}
        for name in ("HELENA_REGISTRY", "BASE_IMAGE", "VILLA_IMAGE_DIGEST",
                     "VILLA_BASE_IMAGE", "UV_CONTEXT", "HELENA_PUBLIC_REGISTRY"):
            environ.pop(name, None)
        done = subprocess.run(
            ["sh", str(ROOT / "containers/build-worker.sh"), str(ROOT)],
            capture_output=True, text=True, env=environ)
        return record.read_text().splitlines(), done


def test_without_a_host_python_the_lock_is_read_through_a_container():
    """deploy-platform.sh already had this fallback; this script did not."""
    builder = (ROOT / "containers/build-worker.sh").read_text()
    assert 'command -v python3' in builder
    assert '"${HELENA_PYTHON_IMAGE:-python:3.11-slim}" python3 -' in builder, (
        "there is no way to read the lock where the deploy actually runs this")


def test_an_unreadable_lock_stops_rather_than_reusing_whatever_is_here():
    """The failure that made the gate inert: empty reads were taken as
    'nothing to compare against' and everything on the host looked current."""
    _, done = _run_without_python3('true')   # the container prints nothing

    assert done.returncode == 2, (
        "an unreadable lock let the build carry on:\n"
        f"rc={done.returncode}\n{done.stdout}{done.stderr}")
    assert "Refusing to guess" in done.stderr


def test_the_container_fallback_lets_the_build_proceed():
    """And when the fallback does answer, the lock is not what stops it.

    This host has no villa image either, so the run still ends at the digest
    refusal further down -- a different exit 2, and not this one.
    """
    commit = lock()["commit"]
    _, done = _run_without_python3(f'echo "{commit}"')

    assert "Refusing to guess" not in done.stderr, (
        f"the lock read fine and the build refused anyway:\n{done.stdout}{done.stderr}")
