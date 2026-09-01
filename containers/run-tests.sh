#!/bin/sh
# Run the suite the way CI runs it, here.
#
#   containers/run-tests.sh [pytest args...]
#
# Three CI-only failures in one night made this worth a script rather than a
# habit. Each was a test written against a laptop and validated there:
#
#   git ls-files              exits 128 in the job container -- the checkout
#                             belongs to another user and git calls it dubious
#   no HELENA_TEST_DSN        eighty-odd tests skip: postgres_store, the real
#                             migration, the failpoints, the concurrency parity
#   no HELENA_REGISTRY        image names resolve differently, so a build script
#                             takes a branch it never takes on the runner
#
# None of the three is reachable from `pytest tests/` on a developer's machine,
# and all three stop the deploy. The difference is not the tests, it is the
# environment they are asked in -- so ask them in it.
set -eu

here="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
D="${HELENA_DOCKER:-docker}"
image="helena-ci:local"
pg="helena-local-test-postgres"
port="${HELENA_TEST_PORT:-55434}"      # not 55433: that is the CI runner's own

$D info >/dev/null 2>&1 || {
  echo "the Docker daemon is not reachable. Set HELENA_DOCKER='sudo docker' if" >&2
  echo "you are not in the docker group." >&2
  exit 1
}

echo "==> building $image"
$D build -q -f "$here/containers/images/Containerfile.ci" -t "$image" "$here" >/dev/null

if [ -z "$($D ps -q -f "name=^${pg}$")" ]; then
  echo "==> starting a throwaway postgres on 127.0.0.1:$port"
  $D rm -f "$pg" >/dev/null 2>&1 || true
  $D run -d --name "$pg" \
    -e POSTGRES_DB=helena_ci_tests -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=throwaway \
    -p "127.0.0.1:$port:5432" \
    --health-cmd "pg_isready -U postgres -d helena_ci_tests" \
    --health-interval 3s --health-retries 20 \
    postgres:16-alpine >/dev/null
fi

echo "==> waiting for it to accept connections"
i=0
until $D exec "$pg" pg_isready -U postgres -d helena_ci_tests >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -lt 40 ] || { echo "postgres never became ready" >&2; exit 1; }
  sleep 2
done

# The commit, because the QC adapter and the panel both bind the revision they
# ran under and refuse rather than guess.
#
# The plain hash, not a `-dirty` variant: the adapter validates the shape and
# refuses anything that is not a full Git SHA, so marking the value made seven
# of its tests fail on `HELENA_QC_CODE_COMMIT must be a full Git SHA`. It went
# unnoticed for a run because only untracked files were present, which `git
# diff` does not call dirty. The caveat belongs in a line of output, where it
# costs nothing and breaks nothing.
commit="$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)"
git -C "$here" diff --quiet 2>/dev/null \
  || echo "    note: tracked files are modified, so this is not exactly $(printf %.12s "$commit")"

# Copied in, not bind-mounted. Docker Desktop shares no host paths here by
# default -- `mounts denied: the path ... is not shared from the host` -- and
# waiting on a preferences dialog is not a way to run tests. It is also closer
# to what CI does, which is a fresh checkout rather than somebody's directory.
#
# Tracked files plus untracked-but-not-ignored, so a test written a minute ago
# is included and .venv, node_modules and the 206 MB of .git are not. 22 MB.
#
# COPYFILE_DISABLE because macOS tar otherwise writes an AppleDouble sidecar per
# file, and GNU tar inside the image unpacks those as real `._name` files. A
# `._x.json` matches a `*.json` glob and is binary, so a profile loader dies on
# `utf-8 codec can't decode byte 0xa3`. Checked in busybox first, which does not
# create them, and concluded there was no problem -- in the wrong image again.
#
# Except untracked .py at the top level, which are scratch here and shadow the
# standard library: an `cmd.py` beside the repo root is imported by pdb, which
# pytest imports, and the whole run dies in collection with a traceback about
# job_store. CI never sees this because CI checks out tracked files only. This
# is the same hazard as running pytest from the repo root by hand.
echo "==> packing the working tree"
tarball="$(mktemp -t helena-tests)"
trap 'rm -f "$tarball"' EXIT HUP INT TERM
(
  cd "$here"
  git ls-files --cached
  git ls-files --others --exclude-standard | grep -v '^[^/]*\.py$' || true
) | (cd "$here" && COPYFILE_DISABLE=1 tar -T - -cf "$tarball")

# --network host does not exist on Docker Desktop; the gateway alias does, and
# resolves on Linux too since 20.10.
echo "==> running the suite as CI does"
# /w/.git is an empty directory on purpose. Several modules locate the
# repository root by walking up for a `.git`, and with none they raise
# StopIteration during collection. CI has a real clone; carrying 206 MB here to
# satisfy an `.exists()` would not buy anything the marker does not.
#
# The container unpacks its own tree: `docker cp` into a stopped container can
# place a file, but nothing there can extract it, and /w does not exist in the
# image. The `sh "$@"` tail is how the caller's pytest arguments reach the
# script's own positional parameters without being re-split by a second shell.
container="$($D create \
  --add-host=host.docker.internal:host-gateway \
  -e HELENA_TEST_DSN="postgresql://postgres:throwaway@host.docker.internal:$port/helena_ci_tests" \
  -e HELENA_QC_CODE_COMMIT="$commit" \
  -e CX_DEPLOYED_REVISION="$commit" \
  -e HELENA_REGISTRY="${HELENA_REGISTRY:-registry.example.invalid/helena}" \
  "$image" \
  sh -c 'mkdir -p /w/.git && tar -xf /tmp/tree.tar -C /w 2>/dev/null && cd /w && exec python -m pytest "$@"' \
  sh "$@")"
trap 'rm -f "$tarball"; $D rm -f "$container" >/dev/null 2>&1 || true' EXIT HUP INT TERM
$D cp "$tarball" "$container:/tmp/tree.tar" >/dev/null
$D start -a "$container"
