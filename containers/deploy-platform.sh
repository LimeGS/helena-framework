#!/bin/sh
# Bring the whole platform up on the machine this runs on.
#
#   containers/deploy-platform.sh gpu|nogpu [COMMIT]
#
# This runs ON the target, not against it. deploy-to-host.sh is the same thing
# reached over SSH from a laptop; the pipeline calls this one directly, because
# the runner is already on the host it deploys to.
#
# ---------------------------------------------------------------------------
# What "the platform" is
# ---------------------------------------------------------------------------
#
# Six compose projects, and until now only three of them were ever deployed by
# anything automatic:
#
#   helena              postgres, init, panel, backup      no GPU
#   helena-segment      segment, fleet-runner              no GPU
#   helena-spiral       the P1 spiral fitter                one card
#   helena-host-report  host-report                        GPU only via overlay
#   helena-ink-0        ink                                needs a card
#   helena-qc-N         surface-qc, one project per card   needs a card
#
# `deploy-to-host.sh HOST all` covered panel, segment and qc and exited zero, so
# ink, backup and host-report sat six commits behind through several releases
# with every signal saying the deploy had succeeded. An "all" that is not all is
# worse than no "all", because it reports success. This file is the list.
#
# ---------------------------------------------------------------------------
# The two profiles
# ---------------------------------------------------------------------------
#
# gpu     everything. gpu-1, deployed from `staging`.
# nogpu   everything that does not reserve a card. swisspost-1, from
#         `development`.
#
# A `gpus:` reservation is not conditional -- on a machine with no NVIDIA driver
# the container refuses to start -- so the difference is which files are named,
# not a flag inside them. That is also why host-report has an overlay.
#
# The two hosts are separate deployments with separate databases. Neither reads
# the other's queue, and they are expected to run different versions.
set -eu

profile="${1:?usage: deploy-platform.sh gpu|nogpu [COMMIT]}"
commit="${2:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

case "$profile" in
  gpu|nogpu) ;;
  *) echo "profile must be gpu or nogpu, not '$profile'" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"

# Named here rather than beside the build: the gpu profile needs it too, to
# build villa when nothing can be pulled.
worker_tag="helena-worker-cpp:local-$commit"
compose="$root/containers/compose"
# Configuration lives in the checkout, not in /etc.
#
# It used to default to /etc/helena, which meant installing this needed a
# privileged step nothing else needed: `sudo cp` four files into a directory the
# README told you to create. Everything else about Helena is a container and a
# volume -- no systemd units, no host-prepared paths -- and this was the one
# exception, mutated on every deploy as `set_env` writes the image tags back.
#
# In the checkout it is one directory next to the code that produced it,
# writable by whoever cloned it, gone when they delete it. `*.env` is already
# ignored by git and `!*.env.example` keeps the templates, so the secrets in it
# cannot be committed by the usual accident.
#
# Hosts installed before this keep working: /etc/helena is used when it holds
# configuration and the checkout does not, because a deploy that silently starts
# reading a different, empty directory is a deploy that recreates a running
# stack with defaults.
env_dir="${HELENA_ENV_DIR:-}"
if [ -z "$env_dir" ]; then
  env_dir="$root/config"
  if [ ! -f "$env_dir/platform.env" ] && [ -f /etc/helena/platform.env ]; then
    env_dir=/etc/helena
    echo "  using /etc/helena, where this host was configured before" >&2
    echo "  (move it to $root/config, or set HELENA_ENV_DIR, to stop seeing this)" >&2
  fi
fi
registry="${HELENA_REGISTRY:-localhost:5000/helena}"
# How many cards this host lends Helena. `surface-qc.compose.yaml` already
# treats *which* card as a per-host fact -- "it lives in the env file, not
# here" -- and the count belongs in the same place: a machine that must leave a
# card free for something else should be able to say so without editing a file
# every other host shares, and an eight-card rig should not have to either.
#
# Precedence: the environment wins, because CI passes it that way; then the
# host's own env file; then both cards, which is what this has always assumed.
devices="${HELENA_QC_DEVICES:-}"
if [ -z "$devices" ] && [ -f "$env_dir/surface-qc.env" ]; then
  devices="$(sed -n 's/^[[:space:]]*HELENA_QC_DEVICES[[:space:]]*=[[:space:]]*//p' \
    "$env_dir/surface-qc.env" | tail -1 | tr -d '"'\''' | tr ',' ' ')"
fi
# Then the cards this machine actually has. The fallback was `0 1` -- "both
# cards, which is what this has always assumed" -- and a host with one card
# fails before the container starts:
#
#   nvidia-container-cli: device error: 1: unknown device
#
# Counting them is not an assumption. nvidia-smi is on any host that can run
# this profile at all; if it cannot be asked, one card is the safer guess than
# two, because asking for a card that is not there fails and leaving one idle
# does not.
if [ -z "$devices" ]; then
  count="$(nvidia-smi --list-gpus 2>/dev/null | grep -c '^GPU ' || true)"
  case "$count" in
    ''|0) devices=0 ;;
    *) devices="$(i=0; while [ "$i" -lt "$count" ]; do printf '%s ' "$i"; i=$((i + 1)); done)" ;;
  esac
  # printf, not say(): this runs at line 115 and say() is defined at 146.
  printf '  using GPU %s, counted on this host\n' "$devices"
fi
qc_base="${HELENA_QC_BASE_IMAGE:-$registry/helena-gpu-runtime:0.1.1}"
# The full hash: CI tags the panel with $CI_COMMIT_SHA, and the short form names
# no image in the registry. Passed in rather than derived, because the pipeline's
# job container has the checkout but not necessarily git.
commit_full="${HELENA_COMMIT_FULL:-$(git -C "$root" rev-parse "$commit" 2>/dev/null || echo "$commit")}"
export HELENA_COMMIT_FULL="$commit_full"

# In a CI job this is root in a container holding the host's socket, so there is
# nothing to elevate. Run by hand on the host it needs sudo.
D="${HELENA_DOCKER:-docker}"

# Asked once, here, because every check below reads an answer from it and a
# daemon that cannot be reached returns the same "no" as a thing that is absent.
# Run without HELENA_DOCKER on a host whose user is not in the docker group,
# this reported "helena-villa is not here yet" about an image that was sitting
# on that machine -- it could not ask. install.sh has had this check since the
# same mistake was found there; this script did not.
command -v "${D%% *}" >/dev/null 2>&1 || {
  echo "${D%% *} is not installed. See https://docs.docker.com/get-docker/" >&2
  exit 1
}
$D info >/dev/null 2>&1 || {
  echo "the Docker daemon is not running, or this user cannot reach it." >&2
  echo "On Linux: sudo systemctl start docker, and add yourself to the docker" >&2
  echo "group -- or, if you would rather not (it is root-equivalent), run this" >&2
  echo "as HELENA_DOCKER='sudo docker' sh containers/deploy-platform.sh $profile" >&2
  exit 1
}

say() { printf '  %s\n' "$*"; }

# What villa was built from, for the label that replaced helena-vc3d's parent
# digest: the source rather than a digest of an image nobody published.
villa_lock() { villa_lock_entry volume_cartographer "$1"; }

# Python, wherever this runs. The deploy job runs in docker:27-cli, which has
# no python3 -- so qc_entry below died with `python3: not found` after every
# container was already recreated, and the job reported a failed deploy of a
# platform that was up. villa_lock_entry had the same fault and hid it behind
# `|| echo unknown`, which is why the villa labels read "unknown" on every CI
# deploy while the same script on a laptop filled them in.
#
# The host's python3 when there is one; otherwise the same interpreter in a
# container. Reads stdin as the program, like `python3 -`, and mounts nothing:
# the first version bind-mounted the checkout at its own path, which on a CI
# runner is a path inside the runner's container and not on the host whose
# daemon runs the image -- so the mount arrived empty and the deploy died on
# `No such file or directory: framework/registries/ink-weights-0.1.0.json`,
# one step past the fault it had just fixed. Callers hand the file's contents
# in as an argument instead; the registry and the lock are a few kilobytes.
py() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$@"
  else
    $D run --rm -i "${HELENA_PYTHON_IMAGE:-python:3.11-slim}" python3 - "$@"
  fi
}

# By entry and field: three images fetch from this lock now, each pinning its
# own commit.
villa_lock_entry() {
  py "$(cat "$root/containers/images/scrollfiesta/locks/source-lock.json" 2>/dev/null)" "$1" "$2" <<'PY_LOCK' 2>/dev/null || echo unknown
import json, sys
print(json.loads(sys.argv[1])[sys.argv[2]].get(sys.argv[3], "unknown"))
PY_LOCK
}

# Seed the configuration from the templates rather than telling somebody to.
#
# The README said `cp containers/compose/*.env.example /etc/helena/ # then edit,
# and drop .example`, which is a step that can be got wrong in three ways --
# wrong directory, forgotten rename, forgotten edit -- before anything reports a
# problem. What is copied here is inert: every variable has a default except the
# ones that cannot have one, and those still stop the deploy by name.
#
# Existing files are never touched. This fills gaps, it does not reset.
mkdir -p "$env_dir"
# Whether this run is the one that created platform.env, not whether it exists
# now -- the loop below is about to create it if it was missing. A host with a
# platform.env already has a config gpu-1 and work-3 both wrote by hand before
# any of this existed, and staging has to deploy the commit under test, not
# whichever version a registry happens to publish. That distinction is what
# HELENA_PUBLIC_REGISTRY's own default reads below.
platform_env_is_new=false
[ -f "$env_dir/platform.env" ] || platform_env_is_new=true
for template in "$compose"/*.env.example; do
  [ -f "$template" ] || continue
  wanted="$env_dir/$(basename "$template" .example)"
  [ -f "$wanted" ] && continue
  cp "$template" "$wanted"
  chmod 600 "$wanted"
  say "wrote $wanted from its template; edit it if the defaults do not fit"
done

# Published images this deploy may pull instead of building. Off by default on
# a host that already had a platform.env -- gpu-1 and work-3 both do, and
# staging exists to run the commit under test, which a published tag (cut on
# its own, slower cadence) is not guaranteed to be. A genuinely fresh host has
# no such commitment, so it defaults on there: the point of publishing is that
# a stranger's first install pulls the panel and the worker images instead of
# compiling volume-cartographer for an hour. Either way this is a caller's
# explicit HELENA_PUBLIC_REGISTRY first, since a value set on purpose -- by
# this same deploy re-run, or by someone who wants gpu-1 to try it too -- is
# not this default's business to override.
HELENA_PUBLIC_REGISTRY="${HELENA_PUBLIC_REGISTRY:-}"
if [ -z "$HELENA_PUBLIC_REGISTRY" ] && [ "$platform_env_is_new" = true ]; then
  HELENA_PUBLIC_REGISTRY="docker.io/limegs"
fi

# The compose files name their env files as ${HELENA_*_ENV:-/etc/helena/...},
# and the templates that set those variables set them to this fleet's absolute
# paths. Both were true when the env files lived in /etc; they moved into the
# checkout and nothing told compose. On a host that had been configured before,
# /etc/helena still holds them and everything works -- which is why this only
# ever broke on a clean install, as
#
#   env file /etc/helena/segment.env not found
#
# after the hour that compiles villa. Measured on a rented 5090.
#
# Point each one at the file that was just seeded, and only when it is really
# there and nobody has said otherwise: a host that keeps its env somewhere else
# has set these already, and this must not override that.
# The platform's own files are in the same list: platform.env.example used to
# carry HELENA_PANEL_ENV=/etc/helena/panel.env, so a clean install seeded a
# panel.env into config/ that the panel never read, and aws.env -- the backup
# service's credentials -- had no way to be found in config/ at all.
for pair in \
  "HELENA_SEGMENT_ENV segment.env" \
  "HELENA_HOST_REPORT_ENV segment.env" \
  "HELENA_INK_ENV ink.env" \
  "HELENA_QC_ENV surface-qc.env" \
  "HELENA_PANEL_ENV panel.env" \
  "HELENA_AWS_ENV aws.env" \
  "HELENA_INK_AWS_ENV aws.env" \
  "HELENA_QC_AWS_ENV aws.env"
do
  var="${pair%% *}" file="${pair#* }"
  eval "current=\${$var:-}"
  [ -n "$current" ] && continue
  [ -f "$env_dir/$file" ] || continue
  export "$var=$env_dir/$file"
done

# The directories the workers write into, owned by the uid they run as. These
# are host paths on purpose -- a run somebody can look at without entering a
# container -- and Docker creates a bind path root:root, so a worker running as
# 1000 got
#
#   PermissionError: '/srv/helena/runs/pherc826-p3-…'
#
# and, because it died before recording anything, the job sat leased for an hour
# and then failed as LEASE_EXHAUSTION. A permission bug that presents as a
# timeout costs a great deal more than one that presents as itself.
for setting in "HELENA_FLEET_RUNS segment.env /srv/helena/runs" \
               "HELENA_INK_RUNS ink.env /srv/helena/runs" \
               "HELENA_QC_RUN_ROOT surface-qc.env /srv/helena/runs/surface-qc-v2" \
               "HELENA_QC_RECONSTRUCTIONS surface-qc.env /srv/helena/artifacts/reconstruction-v1"
do  set -- $setting
  var="$1" file="$env_dir/$2" fallback="$3"
  [ -f "$file" ] || continue
  path="$(grep -oE "^$var=.*" "$file" 2>/dev/null | cut -d= -f2- || true)"
  path="${path:-$fallback}"
  case "$path" in
    /*) ;;                       # a host path; a volume name is not ours to make
    *) continue ;;
  esac
  mkdir -p "$path" 2>/dev/null || {
    echo "  cannot create $path for $var; the workers will fail on their first job" >&2
    continue
  }
  chown -R "${HELENA_WORKER_UID:-1000}:${HELENA_WORKER_GID:-1000}" "$path" 2>/dev/null || true
done

# The QC run root is mounted per device -- `$HELENA_QC_RUN_ROOT/gpu0` onto
# /artifacts/qc-runtime -- and Docker creates a bind mount's source as
# root:root when it does not exist. Owning the parent above was not enough:
# on a fresh install the runtime, uid 1000, claimed its first QC job and died on
#
#   PermissionError: '/artifacts/qc-runtime/d210ab66-…'
#
# which is the gpu0 directory Docker had just made. Prepared here, per device,
# before the surface-qc stacks come up.
# BEGIN qc-run-root-per-device
qc_root="$(grep -oE '^HELENA_QC_RUN_ROOT=.*' "$env_dir/surface-qc.env" 2>/dev/null | cut -d= -f2- || true)"
qc_root="${qc_root:-/srv/helena/runs/surface-qc-v2}"
case "$qc_root" in
  /*)
    for device in ${devices:-0}; do
      mkdir -p "$qc_root/gpu$device" 2>/dev/null \
        && chown -R "${HELENA_WORKER_UID:-1000}:${HELENA_WORKER_GID:-1000}" "$qc_root/gpu$device" 2>/dev/null \
        || echo "  cannot prepare $qc_root/gpu$device; surface-qc on gpu $device will fail on its first job" >&2
    done ;;
esac
# END qc-run-root-per-device

# host-report names the machine it reports on and refuses to interpolate
# without it, so a clean install brought the stack up and then failed with
#
#   required variable HELENA_HOST_ID is missing a value
#
# No template can know this, which is why it was in none of them and in every
# hand-written /etc/helena instead. The hostname is a defensible default: it is
# what the operator already calls this machine, and it is one line to change.
this_host="$(hostname 2>/dev/null || echo helena-worker)"

# Four of them, one per stack, and the templates ship two placeholders --
# `changeme-hostname` and `this-machine` -- which are not names a machine has.
# A worker registers under whatever this says, so the placeholders reach the
# panel and every host in a fleet claims to be the same one. Setting only
# HELENA_HOST_ID, as this first did, fixes host-report and leaves the segment
# worker registering as changeme-hostname, which is how the second name was
# found at all.
for pair in \
  "segment.env HELENA_HOST_ID" \
  "segment.env HELENA_SEGMENT_HOST_ID" \
  "ink.env HELENA_INK_HOST_ID" \
  "surface-qc.env HELENA_QC_HOST_ID" \
  "surface-qc.env QC_WORKER_ID"
do
  file="$env_dir/${pair%% *}" var="${pair#* }"
  [ -f "$file" ] || continue
  current="$(grep -oE "^$var=.*" "$file" 2>/dev/null | cut -d= -f2- || true)"
  case "$current" in
    ""|changeme*|this-machine|this-machine-*|REPLACE*)
      if grep -qE "^$var=" "$file"; then
        # QC_WORKER_ID is a worker name and the others are host names, so the
        # placeholder's own suffix is kept: this-machine-surface-qc becomes
        # <host>-surface-qc, not <host>.
        suffix=""
        case "$current" in this-machine-*) suffix="${current#this-machine}" ;; esac
        sed -i "s|^$var=.*|$var=$this_host$suffix|" "$file"
      else
        printf '%s=%s\n' "$var" "$this_host" >> "$file"
      fi
      say "set $var to $this_host in $(basename "$file")" ;;
  esac
done

# A key the template grew after this host's env file was written. The
# templates are the contract the compose files interpolate against, and a
# compose file refuses a required variable rather than guessing -- so gpu-1,
# configured before surface-qc.env.example carried HELENA_QC_ARTIFACTS,
# failed its deploy with `required variable HELENA_QC_ARTIFACTS is missing a
# value` while a machine installed from the same commit came up. Every
# uncommented KEY=value the template has and the env file lacks is appended
# with the template's value, and the deploy says which. Placeholders are the
# block above's business; a key that exists is never touched.
# BEGIN inherit-template-keys
for template in "$compose"/*.env.example; do
  file="$env_dir/$(basename "$template" .example)"
  [ -f "$file" ] || continue
  grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$template" | while IFS= read -r line; do
    key="${line%%=*}"
    grep -qE "^$key=" "$file" && continue
    printf '%s\n' "$line" >> "$file"
    say "inherited $key from $(basename "$template") into $(basename "$file")"
  done
done
# END inherit-template-keys

# The workers' database URL, written from what the panel's postgres actually
# runs with. The templates shipped this fleet's address and a placeholder
# password -- segment.env named another host with a CHANGEME password and
# ink.env said
# `postgresql://REPLACE` -- so on any other machine every worker started, failed
# to reach a database and kept trying:
#
#   connection to server at ..., port 55432 failed: timeout expired
#   [Errno -3] Temporary failure in name resolution
#
# Nothing published that, so the queue simply never ran. Measured on a rented
# 5090, where it is the difference between a deployment and a panel with idle
# workers beside it.
#
# The values are the compose file's own defaults, overridden by platform.env if
# it names others. The workers use host networking and postgres publishes on
# loopback, so 127.0.0.1 is the address they see.
pg_user="$(grep -oE '^POSTGRES_USER=.*' "$env_dir/platform.env" 2>/dev/null | cut -d= -f2- || true)"
pg_pass="$(grep -oE '^POSTGRES_PASSWORD=.*' "$env_dir/platform.env" 2>/dev/null | cut -d= -f2- || true)"
pg_db="$(grep -oE '^POSTGRES_DB=.*' "$env_dir/platform.env" 2>/dev/null | cut -d= -f2- || true)"
pg_port="$(grep -oE '^HELENA_POSTGRES_PORT=.*' "$env_dir/platform.env" 2>/dev/null | cut -d= -f2- || true)"
dsn="postgresql://${pg_user:-campaignx}:${pg_pass:-helena-local-only}@127.0.0.1:${pg_port:-55432}/${pg_db:-campaignx}"

# The QC stack mounts a postgres env file as a required bind, and the panel
# stack reads one too. Both templates named /srv/helena/control-plane/postgres.env
# -- this fleet's path -- so on any other machine the QC container restarted
# forever on `POSTGRES_USER missing from env file` while the bind mounted a
# directory Docker had created to satisfy it.
#
# Write it from the same values as the URL above, 0600 because it holds the
# password, and point the two variables that name it at what was written.
if [ ! -f "$env_dir/postgres.env" ]; then
  umask 077
  {
    printf 'POSTGRES_USER=%s\n' "${pg_user:-campaignx}"
    printf 'POSTGRES_DB=%s\n' "${pg_db:-campaignx}"
    printf 'POSTGRES_PASSWORD=%s\n' "${pg_pass:-helena-local-only}"
  } > "$env_dir/postgres.env"
  # 0600 and owned by the uid the workers run as. It holds the password, so it
  # stays unreadable to everyone else -- but it is bind-mounted into containers
  # that are not root, and root-owned 0600 means the QC worker restarts forever
  # on `PostgreSQL env file is not readable: /run/secrets/postgres.env` with the
  # file plainly there. The mode was right for the host and wrong for its reader.
  chmod 600 "$env_dir/postgres.env"
  chown "${HELENA_WORKER_UID:-1000}:${HELENA_WORKER_GID:-1000}" "$env_dir/postgres.env" 2>/dev/null || true
  say "wrote $env_dir/postgres.env from the platform's own settings"
fi
# Likewise HELENA_PANEL_ENV: the template named /etc/helena/panel.env, which a
# clean install does not have. The export above already wins for this deploy's
# own compose calls; the file is corrected too, so a `docker compose` run by
# hand reads the same panel.env. A value naming a file that exists is left alone.
for triple in "platform.env HELENA_POSTGRES_ENV postgres.env" \
              "surface-qc.env HELENA_QC_POSTGRES_ENV postgres.env" \
              "platform.env HELENA_PANEL_ENV panel.env"; do
  set -- $triple
  file="$env_dir/$1" var="$2" target="$env_dir/$3"
  [ -f "$file" ] || continue
  [ -f "$target" ] || continue
  current="$(grep -oE "^$var=.*" "$file" 2>/dev/null | cut -d= -f2- || true)"
  [ -n "$current" ] && [ -f "$current" ] && continue
  if grep -qE "^$var=" "$file"; then
    sed -i "s|^$var=.*|$var=$target|" "$file"
  else
    printf '%s=%s\n' "$var" "$target" >> "$file"
  fi
done

# Only when it is absent or still the shipped placeholder: an operator who has
# pointed these at a real control plane has said something this must not undo.
for pair in "segment.env FLEET_DB" "ink.env CX_DB"; do
  file="$env_dir/${pair%% *}" var="${pair#* }"
  [ -f "$file" ] || continue
  current="$(grep -oE "^$var=.*" "$file" 2>/dev/null | cut -d= -f2- || true)"
  case "$current" in
    ""|*REPLACE*|*CHANGEME*)
      if grep -qE "^$var=" "$file"; then
        sed -i "s|^$var=.*|$var=$dsn|" "$file"
      else
        printf '%s=%s\n' "$var" "$dsn" >> "$file"
      fi
      say "pointed $var in $(basename "$file") at this host's control plane" ;;
  esac
done

# The GPU half is buildable from this checkout now, so build it rather than
# refusing.
#
# `gpu` used to fail at the last step, after volume-cartographer had been
# compiled and the worker built on top of it, with "the ink-worker image failed
# to build" -- which names the wrong thing: what was missing was its base.
# helena-gpu-runtime took two parent images that were not in this repository's
# build graph, and one of them wanted a bundle tarball assembled from binaries
# nobody could rebuild. All three are produced here now.
#
# Still an hour or two on a cold host: pytorch's runtime image is 3.3 GB before
# anything of ours is installed.
if [ "$profile" = gpu ]; then
  # A known cost, not fixed here: on a host's first HELENA_PUBLIC_REGISTRY
  # deploy, this still builds qc_base (and villa under it) even when
  # helena-worker-gpu itself later pulls ready-made and never needed it. Cheap
  # to skip in principle -- check whether the worker-gpu tag exists in the
  # public registry before starting this section -- expensive to get wrong in
  # a script this size under one pass, so it stays a one-time cost per fresh
  # host: qc_base is cached locally afterward like everything else here.
  if ! $D image inspect "$qc_base" >/dev/null 2>&1 && ! $D pull -q "$qc_base" >/dev/null 2>&1; then
    say "$qc_base is not on this host and could not be pulled; building it"
    # The name build-worker.sh tags it as. `prefix` is that script's variable,
    # not this one's -- written here it is an unset parameter, and under `set -u`
    # the deploy dies on the line that was meant to be helpful.
    villa_image="${HELENA_VILLA_IMAGE:-${HELENA_REGISTRY:+${HELENA_REGISTRY%/}/}helena-villa:local}"
    # It used to refuse here and say that deploying nogpu once was the shortest
    # way to get villa. True, and useless: `--gpu` is a choice the installer
    # offers, so a clean machine that took it got a panel, no workers, and a
    # warning telling it to run the other profile. The gpu profile builds its
    # own prerequisite now.
    #
    # This is the hour the check above exists to spend only when it can pay off,
    # and it pays off here: there is nothing to pull, so compiling is the only
    # way the profile completes at all.
    $D image inspect "$villa_image" >/dev/null 2>&1 || {
      say "$qc_base needs $villa_image, which is not here yet -- building it"
      say "  this compiles volume-cartographer and takes an hour or two"
      if ! BUILD_COMMIT="$commit" sh "$root/containers/build-worker.sh" "$root" "$worker_tag" \
           > /tmp/helena-villa-build.log 2>&1; then
        echo "$villa_image failed to build; the GPU profile stops here" >&2
        tail -20 /tmp/helena-villa-build.log >&2
        exit 4
      fi
      # build-worker.sh tags villa itself, so ask again rather than assume: a
      # rename there would otherwise surface as a confusing failure three builds
      # later instead of here.
      $D image inspect "$villa_image" >/dev/null 2>&1 || {
        echo "build-worker.sh ran but $villa_image is still not here; it may" >&2
        echo "tag villa under another name than this deploy expects." >&2
        exit 4
      }
    }
    ink_image="${HELENA_INK_IMAGE_BASE:-helena-ink:local}"
    $D image inspect "$ink_image" >/dev/null 2>&1 || {
      say "building $ink_image, the frozen TimeSformer runtime"
      $D build -q --build-arg BASE_IMAGE="${HELENA_TORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}" \
        --build-arg BUILD_COMMIT="$commit" \
        -f "$root/containers/images/Containerfile.ink" \
        -t "$ink_image" "$root/containers/images" >/dev/null \
        || { echo "$ink_image failed to build; the GPU profile stops here" >&2; exit 5; }
    }
    # The base is villa's own base, read off the image rather than chosen here.
    # A bundle carries its library closure and deliberately not glibc, so the
    # tools cannot start on anything older than what compiled them. This used to
    # be a separate default -- python:3.12-slim -- which was true until villa
    # moved to a newer Ubuntu, and then the deploy built an image whose tools
    # asked for GLIBC_2.43 against a 2.41 base. Two defaults cannot be relied on
    # to agree; one value read from the other cannot disagree.
    qc_base_os="${HELENA_QC_BASE_OS_IMAGE:-$($D image inspect "$villa_image" \
      --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}' 2>/dev/null)}"
    test -n "$qc_base_os" || {
      echo "$villa_image does not label the base it was compiled on, so the" >&2
      echo "  GLIBC the bundle needs cannot be guaranteed. Set" >&2
      echo "  HELENA_QC_BASE_OS_IMAGE to that base to build anyway." >&2
      exit 5
    }
    # A tag, usually: build-worker.sh passes ubuntu:25.10 and the label records
    # what it was given. A digest is better and this accepts either -- requiring
    # one would have rejected the path every clean install takes, which is worse
    # than a mutable tag whose glibc the gate checks anyway.
    case "$qc_base_os" in
      *@sha256:*) : ;;
      *) say "  $qc_base_os is a tag, so this base is reproducible only while
  that tag means what it means today" ;;
    esac
    # Pinned to whatever that base holds rather than to a constant. The version
    # string is distribution-specific -- 20250419 on Debian, 20260601~26.04.1 on
    # Ubuntu 26.04 -- so a constant is correct for exactly one base and breaks
    # the moment villa moves, which is the failure this whole block exists for.
    ca_version="${HELENA_CA_CERTIFICATES_VERSION:-$($D run --rm --entrypoint sh \
      "$qc_base_os" -c 'apt-get update -qq >/dev/null 2>&1; apt-cache policy ca-certificates 2>/dev/null | awk "/Candidate:/{print \$2}"' 2>/dev/null)}"
    test -n "$ca_version" || {
      echo "could not read the ca-certificates version out of $qc_base_os." >&2
      echo "  Set HELENA_CA_CERTIFICATES_VERSION to pin it explicitly." >&2
      exit 5
    }
    say "building $qc_base from $ink_image and $villa_image on $qc_base_os"
    $D build -q \
      --build-arg BASE_IMAGE="$qc_base_os" \
      --build-arg CA_CERTIFICATES_VERSION="$ca_version" \
      --build-arg INK_IMAGE="$ink_image" \
      --build-arg VILLA_IMAGE="$villa_image" \
      --build-arg VILLA_COMMIT="$(villa_lock commit)" \
      --build-arg VILLA_TREE="$(villa_lock tree)" \
      --build-arg BUILD_COMMIT="$commit" \
      -f "$root/containers/images/Containerfile.gpu-runtime" \
      -t "$qc_base" "$root" >/dev/null \
      || { echo "$qc_base failed to build; the GPU profile stops here" >&2; exit 5; }
  fi
fi

# Point an env file's image variable at what was just built, and keep a copy.
# The env file names the image, so a `compose pull` cannot silently revert a
# locally built worker to whatever the registry tag means today.
set_image() {
  file="$env_dir/$1" var="$2" val="$3"
  test -f "$file" || { echo "$file is missing; this host is not configured for that stack" >&2; exit 4; }
  old="$(grep -oE "^$var=.*" "$file" | cut -d= -f2- || true)"
  # %:* and not %%:* -- the shortest match from the end. A registry image is
  # localhost:5000/helena/helena-panel:sha, and cutting at the
  # first colon would name the rollback after the registry host.
  [ -n "$old" ] && $D tag "$old" "${old%:*}:rollback-$commit" 2>/dev/null \
    && say "rollback kept as ${old%:*}:rollback-$commit" || true
  cp "$file" "$file.bak-$commit"
  # One backup per deploy, kept forever, is a directory nobody can read. The CI
  # runner had 119 `segment.env.bak-*` beside two real files -- including one
  # called `.bak-rename-work-3`, which is somebody having retried by hand in
  # among the automatic ones. Keep the last ten: enough to walk back through a
  # bad week, few enough that `ls /etc/helena` still answers a question.
  ls -1t "$file".bak-* 2>/dev/null | tail -n +11 | while read -r old_backup; do
    rm -f "$old_backup"
  done
  # Substitute if the line is there, append if it is not. It was only the
  # substitution, which is a silent no-op on a file that does not already carry
  # the variable -- and none of the templates do. On a host configured before
  # the templates existed the line was there by hand, so the deploy worked; on a
  # clean machine it wrote nothing, said nothing, and compose fell through to
  # its default and tried to pull an image that had just been built locally:
  #
  #   pull access denied for helena-worker-cpp, repository does not exist
  #
  # Measured on a rented 5090.
  if grep -qE "^$var=" "$file"; then
    sed -i "s|^$var=.*|$var=$val|" "$file"
  else
    printf '%s=%s\n' "$var" "$val" >> "$file"
  fi
  # -xF: whole line, literal. An image name has dots and slashes in it, and
  # building a regex out of it to check it was written is how the check itself
  # becomes the bug.
  grep -qxF "$var=$val" "$file" || {
    echo "$var could not be written to $file" >&2
    exit 4
  }
}

# `compose up | grep` reports grep's exit status, not compose's, so a stack
# that refused to start looked exactly like one that started -- which is how a
# host-report container that could not find its interpreter left the deploy
# green. Run it, keep the log, then decide.
up() {
  what="$1"; shift
  if ! $D compose "$@" up -d > /tmp/helena-up.log 2>&1; then
    echo "$what did not come up:" >&2
    tail -15 /tmp/helena-up.log >&2
    exit 6
  fi
  grep -E '(Started|Recreated)' /tmp/helena-up.log | sed 's/^/  /' || true
}

say "deploying $profile at $commit on $(hostname)"

# --------------------------------------------------------------------------
# helena: postgres, init, panel, backup
# --------------------------------------------------------------------------
#
# The panel is the one image nobody builds here. It is built once in the
# pipeline, named by commit and pushed, and every host runs those exact bytes.
# Building it per host is how two machines end up running different code under
# one name.
panel_image="$registry/helena-panel:$commit_full"
say "pulling $panel_image"
panel_pulled=false
if $D pull -q "$panel_image" >/dev/null 2>&1; then
  panel_pulled=true
elif [ -n "$HELENA_PUBLIC_REGISTRY" ]; then
  # No registry, or not one this host can reach -- the normal case for anybody
  # outside the deployment this was written in: HELENA_REGISTRY points at a
  # private VIP by default. HELENA_PUBLIC_REGISTRY is the second try, tagged
  # by VERSION rather than by commit: publishing is a release, not every push,
  # so the published tag is whatever the last release was, not necessarily
  # this commit. Fine for a stranger's first install; wrong for staging, which
  # is why this variable's own default never turns on for a host that already
  # had a config.
  public_panel_image="$HELENA_PUBLIC_REGISTRY/helena-panel:$(cat "$root/VERSION")"
  say "could not pull $panel_image; trying $public_panel_image"
  if $D pull -q "$public_panel_image" >/dev/null 2>&1; then
    panel_image="$public_panel_image"
    panel_pulled=true
  fi
fi
if [ "$panel_pulled" = false ]; then
  # Building from the checkout is slower and produces bytes only this host
  # has, which is exactly what a registry exists to avoid -- so it is the
  # fallback and not the default, and it says so.
  say "could not pull it; building the panel from this checkout instead"
  say "(a published image is preferable: every host then runs identical bytes)"
  panel_image="helena-panel:local-$commit"
  $D build -q -f "$root/containers/images/Containerfile.panel" \
    --build-arg BUILD_COMMIT="$commit" -t "$panel_image" "$root" >/dev/null \
    || { echo "the panel image could not be pulled or built" >&2; exit 5; }
fi

# Postgres too, which nothing here builds and nothing was refreshing.
#
# It is a pinned third-party tag, and `compose up` uses whatever copy is already
# on the host -- so once pulled, that copy stayed forever. 16-alpine moves when
# Postgres publishes a patch release, and this deployment would never have taken
# one. Pinned still means pinned: the tag is fixed in the compose file and this
# only makes sure the host has that tag as it is published today.
postgres_image="${HELENA_POSTGRES_IMAGE:-postgres:16-alpine}"
$D pull -q "$postgres_image" >/dev/null \
  || say "could not refresh $postgres_image; keeping the copy on this host"

# The backup image is small and built here; it carries this repo's own script.
$D build -q --build-arg BASE_IMAGE=postgres:16-alpine --build-arg BUILD_COMMIT="$commit" \
  -f "$root/containers/images/Containerfile.backup" -t "helena-backup:local-$commit" "$root" >/dev/null \
  || { echo "the backup image failed to build" >&2; exit 5; }
say "built helena-backup:local-$commit"

set_image platform.env HELENA_PANEL_IMAGE "$panel_image"
set_image platform.env HELENA_BACKUP_IMAGE "helena-backup:local-$commit"
# The backup service only if this host has somewhere to put a backup. Started
# without HELENA_BACKUP_S3 it does not fail, it restarts forever saying "no
# destination" -- a red container on every dashboard for a host that was never
# meant to ship its data anywhere. A development deployment is a fair example.
backup_profile=""
grep -q '^HELENA_BACKUP_S3=..*' "$env_dir/platform.env" && backup_profile="--profile backup" \
  || say "no HELENA_BACKUP_S3 on this host; the backup service is not started"
# shellcheck disable=SC2086
up "the platform stack" -p helena -f "$compose/platform.compose.yaml" \
  --env-file "$env_dir/platform.env" $backup_profile

# --------------------------------------------------------------------------
# helena-segment and helena-host-report: the CPU workers
# --------------------------------------------------------------------------
#
# Built on the host that runs them rather than pushed through the internal
# registry: these are gigabytes on a CUDA base, and the hosts reach each other
# on nothing but 22. HELENA_PUBLIC_REGISTRY is a different route -- the public
# internet, not host-to-host -- and worth trying first for the same reason it
# is for the panel: an hour of compiling volume-cartographer is the cost this
# whole mechanism exists to avoid paying on a stranger's first install.
worker_pulled=false
if [ -n "$HELENA_PUBLIC_REGISTRY" ]; then
  public_worker_image="$HELENA_PUBLIC_REGISTRY/helena-worker-cpp:$(cat "$root/VERSION")"
  say "pulling $public_worker_image"
  if $D pull -q "$public_worker_image" >/dev/null 2>&1; then
    # Tagged under the name every reference below already expects, rather than
    # renaming those references -- so a caller who hardcoded the local name
    # for anything downstream is not the thing this has to get right.
    $D tag "$public_worker_image" "$worker_tag"
    worker_pulled=true
  else
    say "could not pull it; building $worker_tag from this checkout instead"
  fi
fi
if [ "$worker_pulled" = false ]; then
  say "building $worker_tag"
  # Piping the build into grep hides its exit status behind grep's, so a build
  # that failed read as a build that succeeded and the deploy carried on to
  # start the previous image under a new commit's name. Keep the log, check
  # the status, and stop -- a deploy that half happened is worse than one that
  # refused.
  if ! BUILD_COMMIT="$commit" sh "$root/containers/build-worker.sh" "$root" "$worker_tag" \
       > /tmp/helena-worker-cpp-build.log 2>&1; then
    echo "the worker image failed to build; the deploy stops here" >&2
    tail -20 /tmp/helena-worker-cpp-build.log >&2
    exit 5
  fi
  grep -E 'naming to' /tmp/helena-worker-cpp-build.log | sed 's/^/  /' || true
fi

set_image segment.env HELENA_SEGMENT_IMAGE "$worker_tag"
up "the segment stack" -p helena-segment -f "$compose/segment.compose.yaml" \
  --env-file "$env_dir/segment.env"

# host-report reads the same env file, and on a GPU host it gets the overlay so
# it can see the cards it is reporting on.
hr_files="-f $compose/host-report.compose.yaml"
[ "$profile" = gpu ] && hr_files="$hr_files -f $compose/host-report.gpu.compose.yaml"
# shellcheck disable=SC2086
up "host-report" -p helena-host-report $hr_files --env-file "$env_dir/segment.env"

if [ "$profile" = nogpu ]; then
  say "deployed $profile at $commit -- ink and surface-qc need a card and were not started"
  exit 0
fi

# --------------------------------------------------------------------------
# The GPU half
# --------------------------------------------------------------------------
#
# Recreating these throws away whatever the cards are doing. The work is not
# lost -- a stale lease is swept back to PENDING and grown again -- but it is
# repeated. The pipeline accepts that deliberately: a deploy that waits for a
# queue to drain is a deploy that never happens on a busy machine.
ink_tag="helena-worker-gpu:local-$commit"
ink_pulled=false
if [ -n "$HELENA_PUBLIC_REGISTRY" ]; then
  public_ink_image="$HELENA_PUBLIC_REGISTRY/helena-worker-gpu:$(cat "$root/VERSION")"
  say "pulling $public_ink_image"
  if $D pull -q "$public_ink_image" >/dev/null 2>&1; then
    $D tag "$public_ink_image" "$ink_tag"
    ink_pulled=true
  else
    say "could not pull it; building $ink_tag from this checkout instead"
  fi
fi
if [ "$ink_pulled" = false ]; then
  say "building $ink_tag on $qc_base"
  # One tag, two targets. The lane used to be a second image built on top of
  # this one -- a full copy, 16.5 GB against 7.83, differing by a directory.
  # BuildKit skips the lane stage entirely when `runtime` is the target, so a
  # host without the lane image never has to fetch it.
  nine_lane="${HELENA_INK_9UM_IMAGE:-helena-ink-9um:local}"
  # Build it if it is not here. This was skipped on any host that did not
  # already have the image -- it wanted an ink-detection checkout handed in as
  # a build context, which is the same thing that made helena-villa
  # unbuildable until it learned to clone the commit its lock pins. It clones
  # now.
  #
  # Its own entry in the lock, and its own commit: ink-detection moves
  # separately from the rest of villa.
  $D image inspect "$nine_lane" >/dev/null 2>&1 || {
    say "building $nine_lane, the 9 um lane runtime"
    $D build -q \
      --build-arg BASE_IMAGE="${HELENA_INK_9UM_BASE_IMAGE:-python:3.12-slim}" \
      --build-arg VILLA_INK_COMMIT="$(villa_lock_entry villa_ink_detection commit)" \
      --build-arg VILLA_INK_TREE="$(villa_lock_entry villa_ink_detection tree)" \
      --build-context "uv_context=${UV_CONTEXT:-docker-image://ghcr.io/astral-sh/uv:0.11.32}" \
      --build-context "repo=$root" \
      -f "$root/containers/images/Containerfile.ink-9um" \
      -t "$nine_lane" "$root" >/dev/null \
      || say "$nine_lane failed to build; the 9 um slot is not deployed here"
  }
  # Through a function so the target and the lane argument can vary without
  # `set --` rewriting this script's own positional parameters. /bin/sh has no
  # arrays to hold them in.
  build_ink_worker() {
    $D build -q "$@" \
      --build-arg BASE_IMAGE="$qc_base" --build-arg BUILD_COMMIT="$commit" \
      -f "$root/containers/images/Containerfile.worker-gpu" -t "$ink_tag" "$root" >/dev/null
  }
  if $D image inspect "$nine_lane" >/dev/null 2>&1; then
    say "  with the 9 um lane from $nine_lane"
    build_ink_worker --target with_lane --build-arg LANE_IMAGE="$nine_lane"
  else
    say "  without the 9 um lane: $nine_lane is not on this host"
    build_ink_worker --target runtime
  fi || { echo "the ink-worker image failed to build; the deploy stops here" >&2; exit 5; }
fi

inflight="$($D exec helena-postgres psql -U campaignx -d campaignx -tAc \
  "select count(*) from segment_qc_jobs where state='CLAIMED'" 2>/dev/null | tr -d '[:space:]')"
[ "${inflight:-0}" != "0" ] && say "interrupting ${inflight} QC job(s); they requeue when the lease expires"

set_image ink.env HELENA_INK_IMAGE "$ink_tag"
up "the ink stack" -p helena-ink-0 -f "$compose/ink.compose.yaml" --env-file "$env_dir/ink.env"

# The 9 um slot, which is the same worker carrying a second lane's frozen
# environment.
#
# It was started by hand and then never touched again: the deploy built and
# recreated helena-ink-0 and the surface-qc slots, knew nothing about this
# container, and its drift check could not miss what it had never been told to
# expect. So it sat on the image it was born with while every other service
# moved -- and it claims P5 jobs, which is how a canonical 2 um run landed on a
# worker running hours-old code and failed on a bug that was already fixed.
#
# Skipped, loudly, on a host where the build above failed or was never
# reached: this deploy builds the 9 um lane image itself, from ink-detection's
# own frozen lock, a few dozen lines up. It used to be something only a
# separate command could produce; it is not any more.
# Two different names, and they were one for a moment: the image the frozen
# lane environment is copied from is a Docker tag, and what the worker announces
# itself as is the lane id the profile declares.
nine_runtime="${HELENA_INK_9UM_RUNTIME:-helena-ink-9um}"
if $D image inspect "$nine_lane" >/dev/null 2>&1; then
  # The worker built above already carries the lane -- one image, one tag. It
  # used to be a second image built on this one, and the argument naming its
  # base was once wrong in a way no tag could show: it said the surface-qc base,
  # which has psycopg and no checkout, so the container crash-looped for 27
  # hours on a missing ink_worker.py while `docker ps` still read "Up".
  #
  # Its own compose project, because the project name is what keeps the slots
  # from being recreations of each other. RUNTIME_IMAGE names the lane the
  # worker carries rather than the image it runs as: the lane declares
  # helena-ink-9um, and a worker calling itself helena-ink-9um-worker refuses
  # its own job. Exported rather than prefixed, for the reason the surface-qc
  # loop below gives; the shell's environment wins over --env-file, so ink.env's
  # own slot is not disturbed.
  HELENA_INK_SLOT=9um
  HELENA_INK_IMAGE="$ink_tag"
  HELENA_RUNTIME_IMAGE="$nine_runtime"
  HELENA_INK_PHASES=P5
  export HELENA_INK_SLOT HELENA_INK_IMAGE HELENA_RUNTIME_IMAGE HELENA_INK_PHASES
  up "the 9 um ink slot" -p helena-ink-9um -f "$compose/ink.compose.yaml" \
    --env-file "$env_dir/ink.env"
  unset HELENA_INK_SLOT HELENA_INK_IMAGE HELENA_RUNTIME_IMAGE HELENA_INK_PHASES
else
  say "no $nine_lane image on this host: the 9 um slot is not deployed here"
fi

# The spiral slot: the segment worker carrying the villa Python environment, so
# it can both claim a P1 job and run the fitter.
#
# Skipped, loudly, on a host where the lane image failed to build.
# build-worker.sh -- which the earlier worker build above already ran --
# builds it too, from a source checkout at the locked commit; it used to need
# a separate `make build-villa-python VILLA_SRC=` and no longer does, the
# same change that made the 9 um slot above buildable here as well.
#
# Its own compose file because the service asks for a card and the rest of the
# segment stack deliberately does not -- and not a compose profile inside that
# file, which would have brought up a second segmentation worker and a second
# fleet runner alongside it under a second project name.
spiral_lane="${HELENA_VILLA_PYTHON_IMAGE:-helena-villa-python:local}"
spiral_runtime="${HELENA_SPIRAL_RUNTIME:-helena-villa-python}"
if $D image inspect "$spiral_lane" >/dev/null 2>&1; then
  # The worker built above already carries the lane: build-worker.sh chose the
  # `with_lane` target when it found this image. One tag, not two.
  #
  # Exported rather than prefixed, for the reason the surface-qc loop below
  # gives: a VAR=x prefix is not reliably visible to what the function runs.
  HELENA_SPIRAL_IMAGE="$worker_tag"
  HELENA_SPIRAL_RUNTIME="$spiral_runtime"
  export HELENA_SPIRAL_IMAGE HELENA_SPIRAL_RUNTIME
  up "the spiral slot" -p helena-spiral \
    -f "$compose/spiral.compose.yaml" --env-file "$env_dir/segment.env"
  unset HELENA_SPIRAL_IMAGE HELENA_SPIRAL_RUNTIME
else
  say "no $spiral_lane image on this host: the spiral slot is not deployed here"
fi

# The QC checkpoint, before the stack that loads it. Without it the container
# starts, fails on `QC checkpoint does not exist: /models/model.safetensors` and
# restarts forever -- so `install.sh --gpu` left nine services up and one
# looping, which is not an installation somebody can use.
#
# Fetched here rather than through the models API because at install time there
# is no account to authenticate with: the first one is claimed after the panel
# is up. This is the installer provisioning its own volume, not a client
# reaching into a deployment.
#
# The entry comes from the weights registry, so the repository, the path and the
# digest are the ones this repository already pins, and a mismatch is deleted
# rather than installed.
qc_entry() { py "$(cat "$root/framework/registries/ink-weights-0.1.0.json")" "$1" <<'PY_ENTRY'
import json, sys
registry = json.loads(sys.argv[1])
for entry in registry["entries"]:
    if entry.get("destination", "").endswith("timesformer_GP_scroll1/model.safetensors"):
        print(entry[sys.argv[2]])
        break
PY_ENTRY
}
qc_dest="$(cd "$root" && qc_entry destination)"
qc_sha="$(cd "$root" && qc_entry sha256)"
qc_repo="$(cd "$root" && qc_entry repo)"
qc_file="$(cd "$root" && qc_entry upstream_path)"
qc_volume="${HELENA_MODELS_VOLUME:-helena-models}"

if [ -n "$qc_dest" ] && [ -n "$qc_sha" ]; then
  have="$($D run --rm -v "$qc_volume:/models" "${HELENA_BUSYBOX_IMAGE:-busybox:1.37.0}" \
    sh -c "sha256sum /models/$qc_dest 2>/dev/null | cut -d' ' -f1" 2>/dev/null || true)"
  if [ "$have" = "$qc_sha" ]; then
    say "the QC checkpoint is already installed"
  else
    say "fetching the QC checkpoint, 150 MB, once"
    # --user 0:0: curlimages/curl runs as uid 100 and the volume is not
    # writable by it, so the download fails as `client returned ERROR on write`
    # -- which reads like a network problem and is a permission one. The file
    # lands root-owned and the workers mount /models read-only, so that is all
    # the access it needs.
    if $D run --rm --user 0:0 -v "$qc_volume:/models" \
         "${HELENA_CURL_IMAGE:-curlimages/curl:8.11.1}" --silent --show-error --fail \
         --location --create-dirs --output "/models/$qc_dest" \
         "https://huggingface.co/$qc_repo/resolve/main/$qc_file" >/dev/null 2>&1
    then
      # --create-dirs makes the directory with root's umask, so it lands 0750
      # and the worker -- uid 1000 -- cannot traverse it. The file inside is
      # 0644 and unreachable, and the worker says the checkpoint does not exist
      # while it is plainly on the volume. Readable and traversable, not
      # writable: a+rX, which is what the images do for the same reason.
      $D run --rm --user 0:0 -v "$qc_volume:/models" \
        "${HELENA_BUSYBOX_IMAGE:-busybox:1.37.0}" \
        chmod -R a+rX "/models/$(dirname "$qc_dest")" >/dev/null 2>&1 || true
      got="$($D run --rm -v "$qc_volume:/models" "${HELENA_BUSYBOX_IMAGE:-busybox:1.37.0}" \
        sh -c "sha256sum /models/$qc_dest | cut -d' ' -f1" 2>/dev/null || true)"
      if [ "$got" != "$qc_sha" ]; then
        $D run --rm -v "$qc_volume:/models" "${HELENA_BUSYBOX_IMAGE:-busybox:1.37.0}" \
          rm -f "/models/$qc_dest" >/dev/null 2>&1 || true
        echo "  the QC checkpoint downloaded as $got, not $qc_sha; it has been" >&2
        echo "  deleted. Surface QC will not start until it is installed." >&2
      fi
    else
      echo "  could not fetch the QC checkpoint. Surface QC will start and fail" >&2
      echo "  on a missing model; everything else on this host is unaffected." >&2
    fi
  fi
fi

set_image surface-qc.env HELENA_QC_IMAGE "$ink_tag"
for device in $devices; do
  # -p per device, and HELENA_QC_DEVICE overridden on the command line: in
  # surface-qc.env it is `cuda`, the torch device, while the compose file wants
  # the GPU index for container_name and device_ids. Passing only --env-file
  # asks nvidia for a device called "cuda", which is refused.
  # Exported rather than prefixed: a VAR=x prefix on a function call is not
  # reliably visible to what the function runs, and this one has to reach
  # compose's interpolation or it asks nvidia for a device called "cuda".
  export HELENA_QC_DEVICE="$device"
  up "surface-qc on gpu $device" -p "helena-qc-$device" \
    -f "$compose/surface-qc.compose.yaml" --env-file "$env_dir/surface-qc.env"
done

# A list that shrinks has to take the old instance down with it. Everything
# above only ever brings stacks up, so a card dropped from the list kept
# running the instance from the previous deploy -- unmanaged, still claiming
# work, and still holding the card somebody asked to free -- while the deploy
# reported success. "Use one card" has to mean the other card ends up free,
# not merely unmentioned.
#
# Only projects no longer in the list are touched. Bringing down one that is
# still in the list would restart QC on every deploy and interrupt whatever it
# was measuring.
for running in $($D ps --filter 'name=^helena-gpu-runtime-' --format '{{.Names}}' 2>/dev/null); do
  had="${running#helena-gpu-runtime-}"
  keep=no
  for device in $devices; do
    [ "$had" = "$device" ] && keep=yes
  done
  [ "$keep" = yes ] && continue
  say "gpu $had is no longer allotted to QC; stopping helena-qc-$had"
  export HELENA_QC_DEVICE="$had"
  $D compose -p "helena-qc-$had" -f "$compose/surface-qc.compose.yaml" \
    --env-file "$env_dir/surface-qc.env" down >/dev/null 2>&1 \
    || say "could not stop helena-qc-$had; it still holds gpu $had"
done

# --------------------------------------------------------------------------
# Prove it, rather than assume it
# --------------------------------------------------------------------------
#
# Everything above asks compose to bring a stack up. None of it checks that a
# container actually ended up running the image this deploy built.
#
# That gap is not hypothetical: ink, backup and host-report sat six commits
# behind through several releases, and every deploy in between exited zero. A
# service missing from a stack, a compose that decided nothing had changed, a
# container that started and died a second later -- all of them look exactly
# like success from the outside.
#
# So the deploy ends by looking. Each expected container must exist, be running,
# and carry the image this run put in place. Anything else fails the deploy,
# which is the only way the pipeline can honestly say a host is up to date.
expect() {
  name="$1" wanted="$2"
  actual="$($D inspect "$name" --format '{{.Config.Image}}' 2>/dev/null || true)"
  state="$($D inspect "$name" --format '{{.State.Status}}' 2>/dev/null || true)"
  if [ -z "$actual" ]; then
    echo "  MISSING  $name -- expected $wanted" >&2
    drift=$((drift + 1))
    return
  fi
  if [ "$actual" != "$wanted" ]; then
    echo "  STALE    $name is on $actual, expected $wanted" >&2
    drift=$((drift + 1))
    return
  fi
  # The name matching is not enough, and this was caught by it not being
  # enough: a container can hold the right image *string* while running
  # different bytes, because a tag is a pointer and something re-pointed it.
  # It happened here -- the same commit was built twice, once per branch, and
  # the second build moved the tag under a container that kept running the
  # first. Compare what the container actually loaded against what the tag
  # means now.
  running="$($D inspect "$name" --format '{{.Image}}' 2>/dev/null || true)"
  tagged="$($D inspect "$wanted" --format '{{.Id}}' 2>/dev/null || true)"
  if [ -n "$tagged" ] && [ "$running" != "$tagged" ]; then
    echo "  STALE    $name carries $wanted but is running older bytes" >&2
    echo "           running ${running#sha256:}" >&2
    echo "           tag now ${tagged#sha256:}" >&2
    drift=$((drift + 1))
    return
  fi
  if [ "$state" != "running" ]; then
    echo "  $state  $name is on the right image but is $state" >&2
    drift=$((drift + 1))
    return
  fi
  say "ok       $name  $wanted"
}

# A moment to fall over in. A container that dies on startup is running for a
# second or two first, and asking immediately would call that a success.
sleep 5

drift=0
expect helena-postgres "$postgres_image"
expect helena-panel "$panel_image"
expect helena-segment "$worker_tag"
expect helena-fleet-runner "$worker_tag"
expect helena-host-report "$worker_tag"
# An `if`, not `[ ... ] && expect ...`. Under `set -e` an AND-list whose first
# test fails is a failed command, and the script would exit here on exactly the
# hosts that have no backup destination -- reporting a broken deploy for the one
# case this was written to allow.
if [ -n "$backup_profile" ]; then
  expect helena-backup "helena-backup:local-$commit"
fi
if [ "$profile" = gpu ]; then
  expect helena-ink-0 "$ink_tag"
  # Only when this host has the lane image; see the build above.
  $D image inspect "$nine_lane" >/dev/null 2>&1 && expect helena-ink-9um "$ink_tag"
  $D image inspect "$spiral_lane" >/dev/null 2>&1 && expect helena-spiral "$worker_tag"
  for device in $devices; do
    expect "helena-gpu-runtime-$device" "$ink_tag"
  done
fi

# --------------------------------------------------------------------------
# One artifact store, or say so
# --------------------------------------------------------------------------
#
# The panel and the workers each mount something at /artifacts, and nothing
# checked they were the same thing. On this host they were not: the panel had
# the named volume the compose file defaults to, the workers had the host paths
# their env file names. A surface uploaded through the browser landed in the
# panel's store, was registered with a path only the panel can resolve, and P2
# certified it as ARTIFACT_UNAVAILABLE -- a whole phase failing on a file that
# existed, a few hundred megabytes away.
#
# Not fatal: a deployment may legitimately split them while the panel publishes
# to S3, and refusing here would strand a host mid-upgrade. But it is said out
# loud, with the inode as the proof, because the failure it causes names neither
# store.
if [ "$profile" = gpu ]; then
  panel_store="$($D exec helena-panel sh -c 'stat -c %d:%i /artifacts' 2>/dev/null || true)"
  worker_store="$($D exec helena-ink-0 sh -c 'stat -c %d:%i /artifacts' 2>/dev/null || true)"
  if [ -n "$panel_store" ] && [ -n "$worker_store" ] \
     && [ "$panel_store" != "$worker_store" ]; then
    echo "" >&2
    echo "  NOTE     the panel and the workers hold different /artifacts" >&2
    echo "           panel   $panel_store" >&2
    echo "           workers $worker_store" >&2
    echo "           A surface uploaded in the browser is registered with a" >&2
    echo "           path the workers cannot open, and the phase that reads it" >&2
    echo "           reports ARTIFACT_UNAVAILABLE. Set HELENA_ARTIFACTS to the" >&2
    echo "           store the workers use, or publish uploads to S3." >&2
  fi
fi

if [ "$drift" != "0" ]; then
  echo "" >&2
  echo "$drift service(s) are not running what this deploy built." >&2
  echo "The deploy is not complete; do not report this host as up to date." >&2
  exit 7
fi

say "deployed $profile at $commit"
