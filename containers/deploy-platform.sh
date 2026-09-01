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
devices="${devices:-0 1}"
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

# By entry and field: three images fetch from this lock now, each pinning its
# own commit.
villa_lock_entry() {
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]].get(sys.argv[3],"unknown"))' \
    "$root/containers/images/scrollfiesta/locks/source-lock.json" "$1" "$2" 2>/dev/null || echo unknown
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
for template in "$compose"/*.env.example; do
  [ -f "$template" ] || continue
  wanted="$env_dir/$(basename "$template" .example)"
  [ -f "$wanted" ] && continue
  cp "$template" "$wanted"
  chmod 600 "$wanted"
  say "wrote $wanted from its template; edit it if the defaults do not fit"
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
  if ! $D image inspect "$qc_base" >/dev/null 2>&1 && ! $D pull -q "$qc_base" >/dev/null 2>&1; then
    say "$qc_base is not on this host and could not be pulled; building it"
    # The name build-worker.sh tags it as. `prefix` is that script's variable,
    # not this one's -- written here it is an unset parameter, and under `set -u`
    # the deploy dies on the line that was meant to be helpful.
    villa_image="${HELENA_VILLA_IMAGE:-${HELENA_REGISTRY:+${HELENA_REGISTRY%/}/}helena-villa:local}"
    $D image inspect "$villa_image" >/dev/null 2>&1 || {
      echo "$qc_base needs $villa_image, which is not here yet." >&2
      echo "containers/build-worker.sh builds it; the nogpu profile runs that" >&2
      echo "first, so deploying nogpu once is the shortest way to get it." >&2
      exit 4
    }
    ink_image="${HELENA_INK_IMAGE_BASE:-helena-ink:local}"
    $D image inspect "$ink_image" >/dev/null 2>&1 || {
      say "building $ink_image, the frozen TimeSformer runtime"
      $D build -q --build-arg BASE_IMAGE="${HELENA_TORCH_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}" \
        --build-arg BUILD_COMMIT="$commit" \
        -f "$root/containers/images/Containerfile.ink" \
        -t "$ink_image" "$root/containers/images" >/dev/null \
        || { echo "$ink_image failed to build; the GPU profile stops here" >&2; exit 5; }
    }
    # The base is not the ink image: villa's binaries ask for GLIBC_2.38 and the
    # pytorch runtime is 2.35, so that combination builds tools that cannot
    # start. The Containerfile checks and says so; this default avoids it.
    say "building $qc_base from $ink_image and $villa_image"
    $D build -q \
      --build-arg BASE_IMAGE="${HELENA_QC_BASE_OS_IMAGE:-python:3.12-slim}" \
      --build-arg CA_CERTIFICATES_VERSION="${HELENA_CA_CERTIFICATES_VERSION:-20250419}" \
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
  sed -i "s|^$var=.*|$var=$val|" "$file"
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
if ! $D pull -q "$panel_image" >/dev/null 2>&1; then
  # No registry, or not one this host can reach. That is the normal case for
  # anybody outside the deployment this was written in: HELENA_REGISTRY points
  # at a private VIP by default, and the README tells people to run this script.
  #
  # Building from the checkout is slower and produces bytes only this host has,
  # which is exactly what the registry exists to avoid -- so it is the fallback
  # and not the default, and it says so.
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
# Built on the host that runs them rather than pushed: these are gigabytes on a
# CUDA base, and the hosts reach each other on nothing but 22.
worker_tag="helena-worker-cpp:local-$commit"
say "building $worker_tag"
# Piping the build into grep hides its exit status behind grep's, so a build
# that failed read as a build that succeeded and the deploy carried on to start
# the previous image under a new commit's name. Keep the log, check the status,
# and stop -- a deploy that half happened is worse than one that refused.
if ! BUILD_COMMIT="$commit" sh "$root/containers/build-worker.sh" "$root" "$worker_tag" \
     > /tmp/helena-worker-cpp-build.log 2>&1; then
  echo "the worker image failed to build; the deploy stops here" >&2
  tail -20 /tmp/helena-worker-cpp-build.log >&2
  exit 5
fi
grep -E 'naming to' /tmp/helena-worker-cpp-build.log | sed 's/^/  /' || true

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
say "building $ink_tag on $qc_base"
# One tag, two targets. The lane used to be a second image built on top of this
# one -- a full copy, 16.5 GB against 7.83, differing by a directory. BuildKit
# skips the lane stage entirely when `runtime` is the target, so a host without
# the lane image never has to fetch it.
nine_lane="${HELENA_INK_9UM_IMAGE:-helena-ink-9um:local}"
# Build it if it is not here. This was skipped on any host that did not already
# have the image -- it wanted an ink-detection checkout handed in as a build
# context, which is the same thing that made helena-villa unbuildable until it
# learned to clone the commit its lock pins. It clones now.
#
# Its own entry in the lock, and its own commit: ink-detection moves separately
# from the rest of villa.
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
# Skipped, loudly, on a host that has no 9 um lane image: it is built from
# ink-detection's own frozen lock and is not something this script can produce.
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
# Skipped, loudly, on a host with no villa-python image. That one is built from
# a source checkout at the locked commit (`make build-villa-python VILLA_SRC=`)
# and is not something this script can produce -- the same shape as the 9 um
# slot above, and the same reason.
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
