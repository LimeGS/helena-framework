#!/bin/sh
# Get a Helena panel running on this machine.
#
#   curl -fsSL https://raw.githubusercontent.com/LimeGS/helena-framework/main/install.sh | sh
#
# Or, which is better, read it first:
#
#   curl -fsSLO https://raw.githubusercontent.com/LimeGS/helena-framework/main/install.sh
#   less install.sh && sh install.sh
#
# It clones the repository, builds the panel image from that checkout and starts
# the stack. Nothing is downloaded as a binary and nothing is installed outside
# the directory it clones into and Docker's own storage.
#
# ---------------------------------------------------------------------------
# Why this exists when the README is already two commands
# ---------------------------------------------------------------------------
#
# Not to save the two commands. It exists for what happens when one of them
# fails: the failures are legible here and are not legible from inside Docker.
# A full disk surfaces as apt-get exiting 100 with a message about
# /var/cache/apt, which reads as a broken base image and cost a day of looking
# in the wrong place. A compose v1 shim surfaces as a YAML parse error about a
# key that is valid. A busy port 8800 surfaces after the build, not before it.
#
# So this checks first, and says the thing that is actually wrong.
set -eu

REPO="${HELENA_REPO:-https://github.com/LimeGS/helena-framework.git}"
REF="${HELENA_REF:-main}"
DIR="${HELENA_DIR:-helena}"
PORT="${HELENA_PORT:-8800}"
# The panel image build wants room for a Node build, a Python environment and
# the layers under them. Measured on a clean host: 4.1 GB.
NEEDED_GB=6

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Before anything is downloaded or built
# --------------------------------------------------------------------------

say "checking what this needs"

command -v git >/dev/null 2>&1 || die "git is not installed."
command -v docker >/dev/null 2>&1 || die "docker is not installed. See https://docs.docker.com/get-docker/"

docker info >/dev/null 2>&1 || die "the Docker daemon is not running, or this user cannot reach it.
  On Linux: sudo systemctl start docker, and add yourself to the docker group.
  On macOS or Windows: start Docker Desktop."

# Compose v2 is a docker subcommand; v1 was a separate binary with a different
# file format. A v1 shim named `docker-compose` still on PATH is not enough.
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is not available.
  \`docker compose version\` has to work. The old \`docker-compose\` binary is v1
  and does not read these files."

# The failure this prevents is the one that does not look like itself: apt-get
# inside the build exits 100 saying it has no space in /var/cache/apt, which
# reads as a broken base image until somebody runs df.
root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
free_kb="$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "${free_kb:-}" ] && [ "$free_kb" -lt "$((NEEDED_GB * 1024 * 1024))" ]; then
  die "Docker has $((free_kb / 1024 / 1024)) GB free at $root and the build needs about ${NEEDED_GB}.
  \`docker system prune -af\` reclaims images and build cache you are not using."
fi

# Better to say so now than after a build, when compose fails to bind.
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "port $PORT is already in use. Set HELENA_PORT to another one."
fi

# The compose file names its project `helena`, so running this on a host that
# already has one does not start a second stack -- it recreates the first, with
# whatever defaults this script passes. That is not a hypothetical: testing this
# installer against a machine with a live deployment replaced its panel with a
# locally built 0.10.0 on a different port, and the real one stopped answering.
#
# A host with a deployment is a host where this script is the wrong tool.
if docker compose ls --format json 2>/dev/null | grep -q '"Name":"helena"'; then
  die "this machine already runs a Helena stack.
  This installer would recreate it rather than start a second one, because the
  compose project is named \`helena\` either way.
  To update an existing deployment use containers/deploy-platform.sh, or take the
  current one down first with:
    docker compose -p helena -f containers/compose/platform.compose.yaml down"
fi

# --------------------------------------------------------------------------
# Fetch and build
# --------------------------------------------------------------------------

if [ -d "$DIR/.git" ]; then
  say "updating the checkout in $DIR"
  git -C "$DIR" fetch --quiet origin "$REF"
  git -C "$DIR" checkout --quiet FETCH_HEAD
else
  [ -e "$DIR" ] && die "$DIR exists and is not a git checkout. Move it, or set HELENA_DIR."
  say "cloning into $DIR"
  git clone --quiet --depth 1 --branch "$REF" "$REPO" "$DIR"
fi
cd "$DIR"

say "building the panel and starting the stack (first run takes a few minutes)"
# No published image to pull, so this compiles the frontend and installs the
# Python environment. Subsequent runs reuse Docker's layer cache.
HELENA_PANEL_PORT="$PORT" docker compose \
  -f containers/compose/platform.compose.yaml up -d

# --------------------------------------------------------------------------
# Wait for it to actually answer
# --------------------------------------------------------------------------
#
# `up -d` returns when containers start, not when the panel serves. Reporting
# success at that point is how somebody opens a browser to a connection reset
# and concludes the install failed.

say "waiting for the panel"
i=0
until curl -sk -o /dev/null "https://127.0.0.1:$PORT/" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt 90 ]; then
    warn "the panel has not answered in three minutes. Its logs:"
    docker compose -f containers/compose/platform.compose.yaml logs --tail 30 panel
    die "not up. The stack is still running; nothing has been cleaned up."
  fi
  sleep 2
done

cat <<EOF

  Helena is running at https://127.0.0.1:$PORT

  The certificate is self-signed on first boot, so your browser will warn once.
  The fingerprint is in the log if you would rather check it than click through:

    docker compose -f containers/compose/platform.compose.yaml logs panel | grep -i fingerprint

  Claim the first account -- the panel offers a form, or from this shell:

    curl -sk https://127.0.0.1:$PORT/api/session/bootstrap \\
      -H 'Content-Type: application/json' \\
      -d '{"username":"you","password":"at-least-ten-characters"}'

  That endpoint answers only on loopback and closes once an account exists.

  The panel alone gives you the interface and the queue. To run phases you need
  workers -- see the README, or Documentation inside the panel.

  To stop:   docker compose -f containers/compose/platform.compose.yaml down
  With data: add -v to that, which deletes the volumes too.

EOF
