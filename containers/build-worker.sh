#!/bin/sh
# Build the worker image, on the host that will run it.
#
#   containers/build-worker.sh [build-context-dir] [tag]
#
# Written down because it was not: the image had been built with build contexts
# and a base-image argument that lived only in somebody's shell history, so
# rebuilding it meant reconstructing the command from the Containerfile.
#
# Built where it runs, rather than built once and streamed. These hosts cannot
# reach each other on anything but 22, and `docker save | ssh docker load` moves
# five gigabytes to avoid a build that takes a few minutes on a machine with
# twelve cores.
set -eu

context="${1:-/mnt/campaignx/build}"
tag="${2:-helena-worker-cpp:local}"
# Named by its registry, not by the bare local alias it used to carry.
#
# `campaignx-villa:local` has no registry in it, so buildx resolves it as a
# Docker Hub image on any machine where it is not already in buildkit's own
# cache -- and Docker Hub answers "insufficient_scope: authorization failed",
# which reads like a credentials problem and is really a naming one. It built on
# gpu-1 only because the image had been sitting there since July.
#
# A fully qualified name cannot be mistaken for somebody else's image, and
# buildx pulls it without help.
# A registry prefix only when this host has one. Hardcoding a registry into the
# default broke exactly once: generalising it to localhost:5000 for publication
# made the deploy look for the Villa base in a registry that does not exist,
# while the image was sitting on the host under the real one. Unset means bare
# local tags, which is what an unconfigured machine can actually resolve.
# Where this script lives, so the source lock and the Containerfiles are found
# regardless of the build context passed in -- they are the repository's, not
# the context's, and the two are not always the same directory.
here="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

# The source lock, read by entry and field. Three images fetch from it now --
# the toolchain, the spiral lane and the 9 um lane -- and each pins its own
# commit: ink-detection moves on its own, so it is a separate entry rather than
# a second name for the same one.
lock="$here/containers/images/scrollfiesta/locks/source-lock.json"
read_lock() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]].get(sys.argv[3],""))' \
      "$lock" "$1" "$2" 2>/dev/null
    return
  fi
  # The deploy runs this script from docker:27-cli, which has no python at
  # all -- so every read here came back empty there, and the gates below read
  # empty as "nothing to compare against" and reused whatever was on the host.
  # That is how a fixed gate stayed a no-op on the only machine it had to work
  # on. deploy-platform.sh already had this fallback; this did not.
  #
  # Contents by argument rather than a bind mount: this runs inside a container
  # driving the host's daemon, where $PWD is not a path that daemon resolves.
  docker run --rm -i "${HELENA_PYTHON_IMAGE:-python:3.11-slim}" python3 - \
    "$(cat "$lock" 2>/dev/null)" "$1" "$2" <<'PY_LOCK' 2>/dev/null
import json, sys
try:
    print(json.loads(sys.argv[1])[sys.argv[2]].get(sys.argv[3], ""))
except Exception:
    print("")
PY_LOCK
}

# Read once, early, and stop if it cannot be. Every decision below is made from
# this file -- which image to reuse, which commit to compile, which tree to
# verify the fetch against -- and an unreadable lock silently turned all three
# into "whatever is already here". Better to say so than to build the wrong
# thing quickly.
test -f "$lock" || { echo "no source lock at $lock" >&2; exit 2; }
for _entry in volume_cartographer villa_python; do
  [ -n "$(read_lock "$_entry" commit)" ] || {
    echo "the source lock at $lock has no readable commit for $_entry." >&2
    echo "Without it this cannot tell a current image from a stale one, and" >&2
    echo "cannot tell a build which commit to fetch. Refusing to guess." >&2
    exit 2
  }
done

# Which villa commit an image on this host was actually built from.
#
# Both Containerfiles stamp org.helena.villa.commit from the VILLA_COMMIT they
# were handed, so an image can say what it carries instead of being trusted to
# be whatever the lock says today.
villa_image_commit() {
  docker image inspect "$1" \
    --format '{{index .Config.Labels "org.helena.villa.commit"}}' 2>/dev/null
}

# Absent and stale are the same answer to "can this build use it": neither is
# the toolchain the lock pins.
#
# Both gates below used to ask only whether the image existed, so a host that
# had built villa once kept that build forever. Measured on gpu-1: the lock was
# re-pinned from 05dcf034 to 23adee04 and every deploy after it -- including
# the ones whose whole point was the new toolchain -- reused the 05dcf034
# images, because they were present. Nothing said so; the worker that came out
# carried a toolchain its own lock did not pin.
#
# An image with no label at all is not judged: it predates the stamp, and
# refusing to reuse it would turn "I cannot tell" into an hour of compiling.
villa_image_at_lock() {
  _img="$1"; _entry="$2"
  docker image inspect "$_img" >/dev/null 2>&1 || return 1
  _want="$(read_lock "$_entry" commit)"
  [ -n "$_want" ] || return 0
  _have="$(villa_image_commit "$_img")"
  [ -n "$_have" ] || return 0
  [ "$_have" = "$_want" ] && return 0
  echo "  $_img carries $(printf %.12s "$_have") and the lock pins $(printf %.12s "$_want")" >&2
  return 1
}

# A published toolchain image, tried before an hour of compiling.
#
# By villa-<commit> rather than by release: that tag names the upstream commit
# the lock pins, so what comes back either is that toolchain or does not exist.
# A release tag would be whatever villa the last release happened to carry,
# which is exactly the coupling the lock exists to break -- these images move
# on upstream's cadence, not on this project's.
#
# Only when HELENA_PUBLIC_REGISTRY is set. deploy-platform.sh turns it on for a
# genuinely fresh host and leaves it off for one that already had a config,
# because staging deploys the commit under test; that decision is read here
# rather than made a second time.
#
# The pull is not trusted over the gate. A registry can serve anything under
# any name, so what makes this safe is the label check afterwards, not the tag
# it was fetched by: an image that does not carry the locked commit is not
# used, however it arrived.
villa_pull_from_public() {
  _pull_local="$1" _pull_repo="$2" _pull_entry="$3"
  [ -n "${HELENA_PUBLIC_REGISTRY:-}" ] || return 1
  _pull_pin="$(read_lock "$_pull_entry" commit)"
  [ -n "$_pull_pin" ] || return 1
  _pull_from="$HELENA_PUBLIC_REGISTRY/helena-$_pull_repo:villa-$(printf %.12s "$_pull_pin")"
  echo "  trying $_pull_from before building it"
  docker pull -q "$_pull_from" >/dev/null 2>&1 || {
    echo "  not published there; building instead"
    return 1
  }
  # Tagged under the name the rest of this script already expects, rather than
  # renaming those references -- the same reason the worker pull does it.
  docker tag "$_pull_from" "$_pull_local"
  villa_image_at_lock "$_pull_local" "$_pull_entry" || {
    echo "  what $_pull_from served is not the locked commit; building instead" >&2
    return 1
  }
  echo "  $_pull_local came from $_pull_from, no compile needed"
  return 0
}
prefix="${HELENA_REGISTRY:+${HELENA_REGISTRY%/}/}"
base="${BASE_IMAGE:-${prefix}helena-villa:local}"
# uv comes from a pinned image rather than a directory beside the source.
#
# It used to be "$context/uvctx", a 66 MB binary that was never in the
# repository and was never sent by sync-to-host.sh -- it was simply present on
# gpu-1, dropped there by hand in July. Nothing declared it, so nothing could
# restore it, and a checkout that did not happen to be on that machine could not
# build a worker at all. That is what blocked building these images in CI.
#
# Mirrored into the cluster registry when there is one, for the same reason the
# rest is: the hosts route to the VIP, and a build inside the fleet should not
# depend on the internet being up.
#
# Without a registry the name has to be the upstream one. The default here was
# `uv:0.11.32`, which with no prefix means Docker Hub's `library/uv` -- a
# repository that does not exist. Inside the fleet HELENA_REGISTRY is always
# set and the mirror answered, so the broken default was never reached; on a
# clean host it failed as `pull access denied [...] insufficient_scope`, which
# reads as a credentials problem and is not one. Measured on a host with no registry configured.
uv_context="${UV_CONTEXT:-docker-image://${prefix:-ghcr.io/astral-sh/}uv:0.11.32}"
commit="${BUILD_COMMIT:-$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)}"

test -f "$context/containers/images/Containerfile.worker-cpp" \
  || { echo "no Containerfile.worker-cpp under $context" >&2; exit 2; }
# The base is not built by anything in this repository -- it arrived on these
# hosts by hand -- so a `docker image prune` that does not know it is special
# deletes it and every worker build after that fails. It is published, so fetch
# it rather than making that a person's problem. A registry reference needs no
# restoring; only a local alias does, and one may still be passed in.
case "$base" in
  # Already carries a registry host, so it is pulled from there below rather
  # than built here. A published toolchain is still worth trying when the copy
  # on this host is not the locked commit: it is the same image, reachable
  # without the compile, and the label check is what decides either way.
  */*.*/*|*.*/*)
    villa_image_at_lock "$base" volume_cartographer \
      || villa_pull_from_public "$base" villa volume_cartographer || : ;;
  *)
    villa_image_at_lock "$base" volume_cartographer \
      || villa_pull_from_public "$base" villa volume_cartographer || {
      echo "the base image $base is not on this host at the commit the lock pins; building it"
      # Build it. Everything it needs is pinned: the repository and the commit
      # come from the source lock, and the fetch stage verifies the tree hash it
      # gets before the toolchain is installed. This used to print instructions
      # and exit, which made "install the workers" a thing a stranger could read
      # about but not do.
      test -f "$lock" || { echo "no source lock at $lock" >&2; exit 2; }
      villa_commit="$(read_lock volume_cartographer commit)"
      villa_tree="$(read_lock volume_cartographer tree)"
      villa_repo="$(read_lock volume_cartographer repository)"
      test -n "$villa_commit" -a -n "$villa_tree" || {
        echo "the source lock has no commit/tree for volume_cartographer; without both" >&2
        echo "the fetch cannot be verified and the build refuses to guess" >&2
        exit 2
      }
      echo "building $base from ${villa_repo:-the pinned default} at $(printf %.12s "$villa_commit")"
      echo "  this compiles volume-cartographer and takes a while"
      # ubuntu:25.10 because upstream's own install_build_deps.sh asks apt for
      # flang-21 and libclang-rt-21-dev with no repository of its own. Measured:
      # 24.04 and 25.04 have neither, and the build fails at "Unable to locate
      # package libclang-rt-21-dev" after installing everything before it.
      # VILLA_BASE_IMAGE overrides this for a base that carries LLVM 21 already.
      # Passed through a function so the optional repository argument can be
      # omitted rather than passed empty: --build-arg VILLA_REPOSITORY= would
      # override the Containerfile's default with nothing and fail much later,
      # deep in the build, as `git remote add origin ""`. This is /bin/sh, so
      # there are no arrays to hold an optional argument in.
      build_villa() {
        docker build "$@" \
          --build-arg BASE_IMAGE="${VILLA_BASE_IMAGE:-ubuntu:25.10}" \
          --build-arg VILLA_COMMIT="$villa_commit" \
          --build-arg VILLA_TREE="$villa_tree" \
          -f "$here/containers/images/Containerfile.villa" \
          -t "$base" "$here"
      }
      if [ -n "$villa_repo" ]; then
        build_villa --build-arg VILLA_REPOSITORY="$villa_repo"
      else
        build_villa
      fi || { echo "$base failed to build; the worker cannot be built without it" >&2; exit 2; }
    } ;;
esac

# A tag is permitted as the convenient pull name, never as provenance. Resolve
# it to the registry digest now and bake that immutable identity into the
# worker. A local-only image may provide VILLA_IMAGE_DIGEST explicitly; without
# either form the build refuses instead of recording a floating base.
villa_digest="${VILLA_IMAGE_DIGEST:-}"
if [ -z "$villa_digest" ]; then
  case "$base" in
    *@sha256:*) villa_digest="$base" ;;
    *)
      docker pull "$base" >/dev/null 2>&1 || docker image inspect "$base" >/dev/null
      villa_digest="$(docker image inspect "$base" \
        --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
      # An image compiled on this host has never been pushed, so it has no
      # RepoDigests at all and the refusal below fired on every clean install --
      # after the hour that compiled villa. Measured on a rented 5090.
      #
      # The fallback is the local image ID, which is a content address over the
      # image config: it names these exact bytes and cannot float. It is
      # deliberately not pullable, because nothing published it; what makes the
      # base auditable here is upstream in the image's own labels, the villa
      # commit and tree that this script verified before compiling.
      if [ -z "$villa_digest" ]; then
        local_id="$(docker image inspect "$base" --format '{{.Id}}' 2>/dev/null || true)"
        case "$local_id" in
          sha256:*) villa_digest="${base%%@*}@$local_id"
                    echo "  $base was built here and never pushed, so its identity is" >&2
                    echo "  the local image id rather than a registry digest" >&2 ;;
        esac
      fi
      ;;
  esac
fi
case "$villa_digest" in
  *@sha256:*) ;;
  *)
    echo "the Villa base has no resolved repository@sha256 digest and no local" >&2
    echo "image id either, which means it is not on this host; set" >&2
    echo "VILLA_IMAGE_DIGEST explicitly if you know what it should be" >&2
    exit 2
    ;;
esac
echo "Villa runtime: $villa_digest"

# The base finally in hand, checked rather than assumed.
#
# The gate above only builds a base whose name is a local alias. A
# registry-qualified one is pulled instead, and a registry can serve a stale
# image as easily as a host can keep one: the fleet's own registry served
# helena-villa:local at 05dcf034 for five weeks after the lock moved to
# 23adee04, and every worker built in that window was compiled against it
# without saying so. Nothing here can fix that -- the
# rebuild is the registry's, not this script's -- but a worker compiled
# against a toolchain its lock does not pin should not come out of this
# quietly. Said, not refused: a host mid-migration still has to be able to
# build, and the person doing the migrating is the one who decides when.
villa_image_at_lock "$base" volume_cartographer || {
  echo "WARNING: $base is not the volume_cartographer commit the lock pins." >&2
  echo "  This worker is being compiled against a toolchain this repository" >&2
  echo "  does not pin. Rebuild and republish the base to clear it." >&2
}

echo "building $tag from $base"
# SOURCE_DATE_EPOCH is left to the Containerfile's default. Zero is not usable:
# a zip written at epoch 0 fails with "ZIP does not support timestamps before
# 1980", which is a build that dies two thirds of the way through.
# The spiral lane, if this host has it. One tag either way: the lane used to be
# a second image built on top of this one, a full multi-gigabyte copy differing
# by one directory. BuildKit skips the lane stage when `worker` is the target,
# so a host without the lane image never needs to have it.
spiral_lane="${HELENA_VILLA_PYTHON_IMAGE:-helena-villa-python:local}"
# Build it if it is not here at the locked commit. It used to be skipped on any
# host that did not already have the image, because the Containerfile wanted a
# villa checkout handed in -- the same reason helena-villa was unbuildable until
# it learned to clone the commit its lock pins. It clones now, so there is
# nothing to skip; and since it clones a commit rather than a branch, a rebuild
# here is the same build twice, not a build that depends on the day it ran.
villa_image_at_lock "$spiral_lane" villa_python \
  || villa_pull_from_public "$spiral_lane" villa-python villa_python || {
  echo "  building $spiral_lane, the spiral and lasagna lane"
  docker build -q \
    --build-arg BASE_IMAGE="${VILLA_PYTHON_BASE_IMAGE:-python:3.12-slim}" \
    --build-arg VILLA_COMMIT="$(read_lock villa_python commit)" \
    --build-arg VILLA_TREE="$(read_lock villa_python tree)" \
    --build-context "uv_context=$uv_context" \
    --build-context "repo=$here" \
    -f "$here/containers/images/Containerfile.villa-python" \
    -t "$spiral_lane" "$here" >/dev/null \
    || echo "  $spiral_lane failed to build; the worker is built without the lane" >&2
}
if docker image inspect "$spiral_lane" >/dev/null 2>&1; then
  echo "  with the spiral lane from $spiral_lane"
  lane_args="--target with_lane --build-arg LANE_IMAGE=$spiral_lane"
else
  echo "  without the spiral lane: $spiral_lane is not on this host"
  lane_args="--target worker"
fi

# shellcheck disable=SC2086
docker buildx build $lane_args \
  --file "$context/containers/images/Containerfile.worker-cpp" \
  --build-context "repo=$context" \
  --build-context "uv_context=$uv_context" \
  --build-arg "BASE_IMAGE=$base" \
  --build-arg "VILLA_IMAGE_DIGEST=$villa_digest" \
  --build-arg "BUILD_COMMIT=$commit" \
  --tag "$tag" \
  --load \
  "$context"

echo "layers:"
docker inspect "$tag" --format '{{range .RootFS.Layers}}{{slice . 7 19}} {{end}}' \
  | tr ' ' '\n' | tail -3 | sed 's/^/  /'
