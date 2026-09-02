#!/bin/sh
# Build every image this fleet runs, tag them for the registry, and push.
#
#   containers/build-images.sh [repo-root] [registry]
#
# One build, one digest, every host pulls the same bytes. Before this, each host
# built its own: `helena-worker-cpp:local` was 2.8 GB on one machine and 5.38 GB
# on another, same tag, different image, and nothing compared them.
#
# The panel and the phase runtime are built from the repository root because
# they carry the framework; the segmentation worker uses build contexts because
# its base and its uv cache live outside the tree.
set -eu

context="${1:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
# No registry by default: the images are tagged locally and compose reads
# them from the daemon that built them. Pass one, or set HELENA_REGISTRY,
# to tag for a push.
registry="${2:-${HELENA_REGISTRY:-}}"
# Empty registry means local tags: `helena-panel:0.24.1`, which is exactly what
# the compose files default to. With one set, everything is prefixed and pushed.
if [ -n "$registry" ]; then prefix="$registry/"; else prefix=""; fi
# The version, from the one file that holds it. A commit hash is an identity
# and not a version: it says which bytes these are and nothing about whether a
# deployment may take them. Semantic version for the tag, commit for provenance.
version="${HELENA_VERSION:-$(cat "$context/VERSION" 2>/dev/null || echo 0.0.0)}"
tag="${HELENA_TAG:-$version}"
commit="${BUILD_COMMIT:-$tag}"

test -f "$context/panel/web/dist/index.html" || {
  echo "the frontend is not built; run: npm --prefix panel/web run build" >&2
  exit 2
}

# name : containerfile : base image
set -- \
  "panel:Containerfile.panel:${PANEL_BASE_IMAGE:-python:3.11-slim}" \
  "ink-worker:Containerfile.worker-gpu:${PHASE_BASE_IMAGE:-${prefix}helena-gpu-runtime:0.1.1}" \
  "backup:Containerfile.backup:${BACKUP_BASE_IMAGE:-postgres:16-alpine}" \
  "control-tunnel:Containerfile.control-tunnel:${TUNNEL_BASE_IMAGE:-alpine:3.20}"

for spec in "$@"; do
  name=$(printf '%s' "$spec" | cut -d: -f1)
  file=$(printf '%s' "$spec" | cut -d: -f2)
  base=$(printf '%s' "$spec" | cut -d: -f3-)
  # Pull rather than require: the frozen runtimes live in the registry now, so
  # a machine that has never built anything can still build these. They were on
  # exactly one host's local docker before, which for an ephemeral host means
  # one power cycle from a compile nobody has the recipe timing for.
  docker image inspect "$base" >/dev/null 2>&1 || docker pull "$base" \
    || { echo "cannot obtain the base image: $base" >&2; exit 2; }
  echo "building ${prefix}helena-$name:$tag from $base"
  docker buildx build \
    --file "$context/containers/images/$file" \
    --build-arg "BASE_IMAGE=$base" \
    --build-arg "BUILD_COMMIT=$commit" \
    --build-arg "FRAMEWORK_VERSION=$version" \
    --tag "${prefix}helena-$name:$tag" \
    --tag "${prefix}helena-$name:latest" \
    --load "$context"
done

# The worker is built with contexts rather than from the tree.
echo "building ${prefix}helena-worker-cpp:$tag"
BUILD_COMMIT="$commit" sh "$context/containers/build-worker.sh" \
  "$context" "${prefix}helena-worker-cpp:$tag"
docker tag "${prefix}helena-worker-cpp:$tag" "${prefix}helena-worker-cpp:latest"

echo
# Nothing is pushed without somewhere to push to. A clone-and-build has no
# registry and needs none: compose reads the tags out of the local daemon.
if [ -n "$registry" ]; then
  # The version tag only. `latest` stays on this daemon, where it is a
  # convenience for whoever is building; pushed, it is a moving name in a
  # registry several hosts pull from, and two hosts that pulled it a week
  # apart run different code while both report the same image.
  for name in panel ink-worker backup worker; do
    docker push "${prefix}helena-$name:$tag"
  done
else
  echo "no registry given: images are tagged locally and not pushed"
fi

# The lock: version, and the digest that version resolved to. A tag says which
# release a host is asked to run and a digest says which bytes it got, and only
# the second survives somebody rebuilding a tag. Written rather than printed,
# because a digest a person is asked to copy out of a terminal is a digest that
# eventually gets copied wrong.
lock="$context/containers/images.lock.json"
echo "{" > "$lock"
echo "  \"schema\": \"campaignx.image_lock.v1\"," >> "$lock"
echo "  \"version\": \"$version\"," >> "$lock"
echo "  \"commit\": \"$commit\"," >> "$lock"
echo "  \"built_at_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$lock"
echo "  \"images\": {" >> "$lock"
first=1
for name in panel ink-worker backup worker; do
  digest=$(docker inspect "${prefix}helena-$name:$tag" \
    --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "")
  [ -z "$digest" ] && continue
  [ $first -eq 0 ] && echo "," >> "$lock"
  printf '    "%s": "%s"' "helena-$name" "$digest" >> "$lock"
  first=0
done
echo >> "$lock"
echo "  }" >> "$lock"
echo "}" >> "$lock"
echo
echo "wrote $lock -- deploy by digest from it, not by tag"
cat "$lock"
