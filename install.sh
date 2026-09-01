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

# What to install. `panel` is the interface and the queue and runs no phase;
# `cpu` adds the workers for P1, P2, P3 and P8; `gpu` adds ink detection and
# surface QC on top, and wants a CUDA device.
#
# Asked rather than assumed, because the previous default left people with a
# panel that could queue work and nothing that could run it -- true, documented
# at the end, and still the wrong thing to hand somebody who typed one command.
WANT="${HELENA_INSTALL:-}"
for argument in "$@"; do
  case "$argument" in
    --panel) WANT=panel ;;
    --cpu)   WANT=cpu ;;
    --gpu)   WANT=gpu ;;
    --help|-h)
      cat <<USAGE
usage: install.sh [--panel|--cpu|--gpu]

  --panel   the panel and its queue; no phase can run
  --cpu     the above, plus workers for P1 P2 P3 P8
  --gpu     the above, plus ink detection and surface QC; needs a CUDA device

Asked interactively when not given. Set HELENA_INSTALL to the same values.
USAGE
      exit 0 ;;
    *) die "unknown argument $argument; try --help" ;;
  esac
done

# Asked from the terminal, not from stdin. The published way to run this is
#
#   curl -fsSL .../install.sh | sh
#
# where stdin *is the script*: a `read` without this consumes the rest of the
# installer and the shell runs whatever is left. /dev/tty is the terminal
# itself, and its absence is how this knows nobody is there to answer.
ask_what_to_install() {
  # Opened, not tested for readability, and opened in a subshell.
  #
  # `[ -r /dev/tty ]` asks access(2) about the device node and answers yes in a
  # process with no controlling terminal at all -- exactly the case this branch
  # exists for. So the terminal has to be opened.
  #
  # But `{ : < /dev/tty; }` opens it in *this* shell, and `:` is a POSIX special
  # built-in: a redirection error on one is fatal to a non-interactive shell.
  # Measured in dash, which is /bin/sh on every host this installs on: the whole
  # installer exits, silently, on the machines the fallback was written for. A
  # subshell takes the death instead.
  if ! (exec < /dev/tty) 2>/dev/null; then
    WANT=panel
    warn "no terminal to ask on, so installing the panel alone. It runs no
  phase. Re-run with --cpu or --gpu, or set HELENA_INSTALL, for workers."
    return
  fi
  cat <<CHOICES

  What should this install?

    1) the panel only      the interface and the queue; runs no phase
    2) panel + CPU workers segmentation, flattening, reconstruction (P1 P2 P3 P8)
    3) panel + GPU workers  the above plus ink detection and surface QC; needs a card

CHOICES
  while :; do
    printf '  choose [1/2/3, default 2]: '
    read -r reply < /dev/tty || reply=""
    case "${reply:-2}" in
      1) WANT=panel; return ;;
      2) WANT=cpu; return ;;
      3) WANT=gpu; return ;;
      *) printf '  answer 1, 2 or 3.\n' ;;
    esac
  done
}
[ -n "$WANT" ] || ask_what_to_install

# --------------------------------------------------------------------------
# Before anything is downloaded or built
# --------------------------------------------------------------------------

say "checking what this needs"

command -v git >/dev/null 2>&1 || die "git is not installed."

# Docker, possibly through something else. Membership of the `docker` group is
# root-equivalent -- it grants the daemon, and the daemon can mount the host
# filesystem -- so plenty of people deliberately do not join it and run
# `sudo docker` instead. deploy-platform.sh has always taken HELENA_DOCKER for
# exactly that; this did not, and told them their daemon was down.
D="${HELENA_DOCKER:-docker}"
command -v "${D%% *}" >/dev/null 2>&1 \
  || die "${D%% *} is not installed. See https://docs.docker.com/get-docker/"

$D info >/dev/null 2>&1 || die "the Docker daemon is not running, or this user cannot reach it.
  On Linux: sudo systemctl start docker, and add yourself to the docker group --
  or, if you would rather not (it is root-equivalent), run this as
  HELENA_DOCKER='sudo docker' sh install.sh
  On macOS or Windows: start Docker Desktop."

# Compose v2 is a docker subcommand; v1 was a separate binary with a different
# file format. A v1 shim named `docker-compose` still on PATH is not enough.
$D compose version >/dev/null 2>&1 || die "Docker Compose v2 is not available.
  \`$D compose version\` has to work. The old \`docker-compose\` binary is v1
  and does not read these files."

# The failure this prevents is the one that does not look like itself: apt-get
# inside the build exits 100 saying it has no space in /var/cache/apt, which
# reads as a broken base image until somebody runs df.
root="$($D info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
free_kb="$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "${free_kb:-}" ] && [ "$free_kb" -lt "$((NEEDED_GB * 1024 * 1024))" ]; then
  die "Docker has $((free_kb / 1024 / 1024)) GB free at $root and the build needs about ${NEEDED_GB}.
  \`$D system prune -af\` reclaims images and build cache you are not using."
fi

# Better to say so now than after a build, when compose fails to bind.
#
# ss before lsof, and this order is the whole point. Measured on Ubuntu: the
# iproute2 package that provides `ss` is Priority: important and is on
# effectively every install, while lsof is Priority: standard and is absent from
# minimal and cloud images -- which are exactly the hosts somebody installs this
# on. `command -v lsof` therefore skipped the check silently on the machines
# that needed it, and the busy port surfaced where it always did: after the
# build, as a compose bind error.
if command -v ss >/dev/null 2>&1; then
  # Column 4 is the local address; the header row cannot match the port anchor.
  listening="$(ss -ltn 2>/dev/null | awk '{print $4}' | grep -cE "[:.]${PORT}\$" || true)"
elif command -v lsof >/dev/null 2>&1; then
  listening="$(lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | wc -l || true)"
else
  listening=""
fi
# A GPU is not enough: Docker reaches it through the NVIDIA container runtime,
# and without that the workers build, start, claim work and find no device. That
# is the worst shape a missing dependency can take -- everything looks installed
# and the jobs fail one at a time, hours later. Measured on a rented 5090, where
# `docker run --gpus all` failed while `nvidia-smi` on the host was fine.
#
# A warning rather than a refusal: the machine may be getting its workers later,
# and the panel half needs none of this.
if [ "$WANT" = gpu ]; then
  # Asked of the daemon rather than by starting a container: a probe would have
  # to pull an image to prove a runtime is configured, and the answer is already
  # in `docker info`.
  if ! $D info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    warn "Docker cannot reach a GPU on this host, so the GPU workers will start
  and then find no device. Install the NVIDIA Container Toolkit, then
  \`nvidia-ctk runtime configure --runtime=docker\` and restart Docker:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  fi
fi

if [ -n "$listening" ] && [ "$listening" -gt 0 ] 2>/dev/null; then
  die "port $PORT is already in use. Set HELENA_PORT to another one."
elif [ -z "$listening" ]; then
  # Not silence: an unchecked port is a different state from a free one, and
  # the reader is about to spend a build finding out which.
  warn "neither ss nor lsof is installed, so nothing checked whether port $PORT
  is free. If it is not, compose fails to bind at the end of the build."
fi

# The compose file names its project `helena`, so running this on a host that
# already has one does not start a second stack -- it recreates the first, with
# whatever defaults this script passes. That is not a hypothetical: testing this
# installer against a machine with a live deployment replaced its panel with a
# locally built 0.10.0 on a different port, and the real one stopped answering.
#
# A host with a deployment is a host where this script is the wrong tool.
if $D compose ls --format json 2>/dev/null | grep -q '"Name":"helena"'; then
  die "this machine already runs a Helena stack.
  This installer would recreate it rather than start a second one, because the
  compose project is named \`helena\` either way.
  To update an existing deployment use containers/deploy-platform.sh, or take the
  current one down first with:
    $D compose -p helena -f containers/compose/platform.compose.yaml down"
fi

# A stack that is *not* running can still leave its state behind, and that state
# is what the next install inherits. Docker volumes outlive `compose down`; they
# are removed only by `down -v` or `volume rm`.
#
# The failure that follows is not legible from the message it produces. These
# containers run as uid 1000, an older deployment ran them as root, and the TLS
# material the panel reads back was written by that root. The panel then exits
# with `PermissionError: [Errno 13] Permission denied` from inside uvicorn's
# `load_cert_chain`, which reads as a broken image or a bad certificate rather
# than as the leftovers of something uninstalled months ago. Found by installing
# onto a host whose Helena containers had been gone since July while its volumes
# had not.
leftovers=""
for volume in helena-panel-state helena-postgres-data helena-artifacts helena-runs helena-models; do
  if $D volume inspect "$volume" >/dev/null 2>&1; then
    leftovers="$leftovers  $volume\n"
  fi
done
if [ -n "$leftovers" ]; then
  printf '\033[31m error:\033[0m this machine has Helena volumes from an earlier install:\n' >&2
  printf "$leftovers" >&2
  cat >&2 <<'WHY'

  Nothing here is running, but these hold the state a fresh install would adopt
  -- including TLS material written by whatever user that install ran as. If it
  was not this one, the panel starts and exits with

      PermissionError: [Errno 13] Permission denied

  from load_cert_chain, which does not look like the cause.

  Keep them and start from what is there:

      $D compose -f containers/compose/platform.compose.yaml up -d

  Or take a copy and start clean:

      for v in helena-panel-state helena-postgres-data helena-artifacts \
               helena-runs helena-models; do
        $D run --rm -v "$v":/from -v "$PWD":/to alpine \
          tar -C /from -czf "/to/$v.tar.gz" . 2>/dev/null
        $D volume rm "$v"
      done

  Set HELENA_ADOPT_VOLUMES=1 to install over them anyway.
WHY
  [ "${HELENA_ADOPT_VOLUMES:-0}" = "1" ] || exit 1
  warn "installing over existing volumes because HELENA_ADOPT_VOLUMES=1"
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
HELENA_PANEL_PORT="$PORT" $D compose \
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
    $D compose -f containers/compose/platform.compose.yaml logs --tail 30 panel
    die "not up. The stack is still running; nothing has been cleaned up."
  fi
  sleep 2
done

# --------------------------------------------------------------------------
# Workers, if they were asked for
# --------------------------------------------------------------------------
#
# A second script, not a second set of compose files: deploy-platform.sh holds
# what the platform is, and duplicating any of it here is how the two drift.
#
# It builds what it cannot pull. Today that means compiling volume-cartographer
# on first run -- an hour or two -- because nothing is published yet; when the
# images are, the pull path is the one that runs and this is the fallback.
install_workers() {
  profile="$1"
  say "installing the $profile workers"
  say "nothing is published yet, so this compiles: expect an hour or two"
  HELENA_DOCKER="$D" sh containers/deploy-platform.sh "$profile" || {
    warn "the workers did not come up. The panel is running and can be used;
  re-run containers/deploy-platform.sh $profile once the reason is fixed."
    return 1
  }
}

WORKER_NOTE="The panel alone gives you the interface and the queue: no phase can
  run yet. For workers: containers/deploy-platform.sh nogpu"
case "$WANT" in
  cpu) install_workers nogpu && WORKER_NOTE="Workers are running for P1, P2, P3
  and P8. Ink detection needs a card: containers/deploy-platform.sh gpu" ;;
  gpu) install_workers gpu && WORKER_NOTE="Workers are running for every phase." ;;
esac

cat <<EOF

  Helena is running at https://127.0.0.1:$PORT

  The certificate is self-signed on first boot, so your browser will warn once.
  The fingerprint is in the log if you would rather check it than click through:

    $D compose -f containers/compose/platform.compose.yaml logs panel | grep -i fingerprint

  Claim the first account -- the panel offers a form, or from this shell:

    curl -sk https://127.0.0.1:$PORT/api/session/bootstrap \\
      -H 'Content-Type: application/json' \\
      -d '{"username":"you","password":"at-least-ten-characters"}'

  That endpoint answers only on loopback and closes once an account exists.

  ${WORKER_NOTE}
  To stop:   $D compose -f containers/compose/platform.compose.yaml down
  With data: add -v to that, which deletes the volumes too.

EOF
