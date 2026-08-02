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
env_dir="${HELENA_ENV_DIR:-/etc/helena}"
registry="${HELENA_REGISTRY:-localhost:5000/helena}"
devices="${HELENA_QC_DEVICES:-0 1}"
qc_base="${HELENA_QC_BASE_IMAGE:-$registry/helena-surface-qc:0.1.1}"
# The full hash: CI tags the panel with $CI_COMMIT_SHA, and the short form names
# no image in the registry. Passed in rather than derived, because the pipeline's
# job container has the checkout but not necessarily git.
commit_full="${HELENA_COMMIT_FULL:-$(git -C "$root" rev-parse "$commit" 2>/dev/null || echo "$commit")}"
export HELENA_COMMIT_FULL="$commit_full"

# In a CI job this is root in a container holding the host's socket, so there is
# nothing to elevate. Run by hand on the host it needs sudo.
D="${HELENA_DOCKER:-docker}"

say() { printf '  %s\n' "$*"; }

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
worker_tag="helena-worker:local-$commit"
say "building $worker_tag"
# Piping the build into grep hides its exit status behind grep's, so a build
# that failed read as a build that succeeded and the deploy carried on to start
# the previous image under a new commit's name. Keep the log, check the status,
# and stop -- a deploy that half happened is worse than one that refused.
if ! BUILD_COMMIT="$commit" sh "$root/containers/build-worker.sh" "$root" "$worker_tag" \
     > /tmp/helena-worker-build.log 2>&1; then
  echo "the worker image failed to build; the deploy stops here" >&2
  tail -20 /tmp/helena-worker-build.log >&2
  exit 5
fi
grep -E 'naming to' /tmp/helena-worker-build.log | sed 's/^/  /' || true

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
ink_tag="helena-ink-worker:local-$commit"
say "building $ink_tag on $qc_base"
$D build -q --build-arg BASE_IMAGE="$qc_base" --build-arg BUILD_COMMIT="$commit" \
  -f "$root/containers/images/Containerfile.ink-worker" -t "$ink_tag" "$root" >/dev/null \
  || { echo "the ink-worker image failed to build; the deploy stops here" >&2; exit 5; }

inflight="$($D exec helena-postgres psql -U campaignx -d campaignx -tAc \
  "select count(*) from segment_qc_jobs where state='CLAIMED'" 2>/dev/null | tr -d '[:space:]')"
[ "${inflight:-0}" != "0" ] && say "interrupting ${inflight} QC job(s); they requeue when the lease expires"

set_image ink.env HELENA_INK_IMAGE "$ink_tag"
up "the ink stack" -p helena-ink-0 -f "$compose/ink.compose.yaml" --env-file "$env_dir/ink.env"

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
  for device in $devices; do
    expect "helena-surface-qc-$device" "$ink_tag"
  done
fi

if [ "$drift" != "0" ]; then
  echo "" >&2
  echo "$drift service(s) are not running what this deploy built." >&2
  echo "The deploy is not complete; do not report this host as up to date." >&2
  exit 7
fi

say "deployed $profile at $commit"
