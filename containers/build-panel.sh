#!/bin/sh
# Build the panel image, on the host that will run it.
#
#   containers/build-panel.sh [repo-root] [tag]
#
# The context is the repository root rather than containers/images, because the
# panel image carries the contracts, profiles and frontend it serves. Build it
# where it runs, for the same reason the worker is built there: these hosts
# reach each other on port 22 and nothing else.
set -eu

context="${1:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
tag="${2:-helena-panel:local}"
base="${BASE_IMAGE:-python:3.11-slim}"
commit="${BUILD_COMMIT:-$(git -C "$context" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

test -f "$context/panel/app.py" || { echo "no panel/app.py under $context" >&2; exit 2; }
test -f "$context/panel/web/dist/index.html" || {
  echo "the frontend is not built: $context/panel/web/dist/index.html is missing" >&2
  echo "run: npm --prefix panel/web run build" >&2
  exit 2
}

echo "building $tag from $base"
docker buildx build \
  --file "$context/containers/images/Containerfile.panel" \
  --build-arg "BASE_IMAGE=$base" \
  --build-arg "BUILD_COMMIT=$commit" \
  --tag "$tag" \
  --load \
  "$context"

echo "layers:"
docker inspect "$tag" --format '{{range .RootFS.Layers}}{{slice . 7 19}} {{end}}' \
  | tr ' ' '\n' | tail -3 | sed 's/^/  /'
