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
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]].get(sys.argv[3],""))' \
    "$lock" "$1" "$2" 2>/dev/null
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
  */*.*/*|*.*/*) : ;;   # already carries a registry host
  *)
    docker image inspect "$base" >/dev/null 2>&1 || {
      echo "the base image $base is not on this host; pulling it"
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
      ;;
  esac
fi
case "$villa_digest" in
  *@sha256:*) ;;
  *)
    echo "the Villa base has no resolved repository@sha256 digest; set VILLA_IMAGE_DIGEST explicitly" >&2
    exit 2
    ;;
esac
echo "Villa runtime: $villa_digest"

echo "building $tag from $base"
# SOURCE_DATE_EPOCH is left to the Containerfile's default. Zero is not usable:
# a zip written at epoch 0 fails with "ZIP does not support timestamps before
# 1980", which is a build that dies two thirds of the way through.
# The spiral lane, if this host has it. One tag either way: the lane used to be
# a second image built on top of this one, a full multi-gigabyte copy differing
# by one directory. BuildKit skips the lane stage when `worker` is the target,
# so a host without the lane image never needs to have it.
spiral_lane="${HELENA_VILLA_PYTHON_IMAGE:-helena-villa-python:local}"
# Build it if it is not here. It used to be skipped on any host that did not
# already have the image, because the Containerfile wanted a villa checkout
# handed in -- the same reason helena-villa was unbuildable until it learned to
# clone the commit its lock pins. It clones now, so there is nothing to skip.
docker image inspect "$spiral_lane" >/dev/null 2>&1 || {
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
