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
tag="${2:-helena-worker:local}"
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
# Mirrored into the cluster registry rather than pulled from ghcr, for the same
# reason the rest is: the hosts route to the VIP, and a build should not depend
# on the internet being up.
uv_context="${UV_CONTEXT:-docker-image://${prefix}uv:0.11.32}"
commit="${BUILD_COMMIT:-$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)}"

test -f "$context/containers/images/Containerfile.worker" \
  || { echo "no Containerfile.worker under $context" >&2; exit 2; }
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
      docker pull "${VILLA_IMAGE:-${prefix}helena-villa:local}" >/dev/null \
        && docker tag "${VILLA_IMAGE:-${prefix}helena-villa:local}" "$base" \
        || { echo "could not restore $base from the registry" >&2; exit 2; }
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
docker buildx build \
  --file "$context/containers/images/Containerfile.worker" \
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
