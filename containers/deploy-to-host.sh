#!/bin/sh
# Build and recreate a stack on a host that runs it.
#
#   containers/deploy-to-host.sh HOST [panel|segment|qc|all] [BUILD_COMMIT]
#
# Written down because none of it was, and every part of it bit somebody:
#
#   * A file in a multi-source rsync lands at the top of the destination, so
#     panel/app.py deployed to /ssd/vc3d/campaign-x/app.py where nothing reads it,
#     twice, while the build succeeded and the image label updated. sync-to-host.sh
#     handles that and verifies the hashes.
#
#   * The QC workers are one compose project *per GPU* -- helena-qc-0 and
#     helena-qc-1 -- so without -p the compose does not recognise the running
#     containers as its own and fails on "name already in use".
#
#   * HELENA_QC_DEVICE means two different things. In surface-qc.env it is `cuda`,
#     which is what the application wants as a torch device; in the compose file it
#     is the GPU index, used for both container_name and device_ids. Recreating
#     with only --env-file therefore asks nvidia for a device called "cuda" and is
#     refused. It has to be overridden per device on the command line.
#
#   * Containerfile.worker-gpu needs BASE_IMAGE, and there is no base-images.env on
#     the hosts, so a plain `docker build` fails with "base name should not be
#     blank".
#
# Recreating the QC stack kills whatever those workers are doing. The work is not
# lost -- a stale lease is swept back to PENDING and grown again -- but it is
# thrown away, so this says how much is in flight and makes you ask for it.
set -eu

host="${1:?usage: deploy-to-host.sh HOST [panel|segment|qc|all] [BUILD_COMMIT]}"
stack="${2:-panel}"
commit="${3:-$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)}"
root="${HELENA_REMOTE_ROOT:-/ssd/vc3d/campaign-x}"
here="$(cd "$(dirname "$0")/.." && pwd)"

# The GPUs the QC workers are pinned to, one project each.
devices="${HELENA_QC_DEVICES:-0 1}"
# The image Containerfile.worker-gpu builds on. Overridable, because the tag moves.
# Where built images are published. Both hosts route to this MetalLB VIP, so a
# deploy is a pull rather than a copy of a source tree.
registry="${HELENA_REGISTRY:-localhost:5000/helena}"
qc_base="${HELENA_QC_BASE_IMAGE:-$registry/helena-gpu-runtime:0.1.1}"
registry_host="${registry%%/*}"
# The full hash, because that is what the pipeline tags with ($CI_COMMIT_SHA).
# This script used the short one everywhere, so it looked for a tag CI never
# publishes and refused every image it had actually built. The short form stays
# for the locally built worker images and the backup filenames, which are this
# script's own naming and not a contract with CI.
commit_full="$(git -C "$here" rev-parse "$commit" 2>/dev/null || echo "$commit")"

on_host() { ssh -o BatchMode=yes "$host" "$@"; }

say() { printf '  %s\n' "$*"; }

# Keep the image that is running, under a name that says what it was, before
# anything replaces it.
keep() {
  on_host "sudo -n docker tag '$1' '$2:rollback-$commit' 2>/dev/null" \
    && say "rollback kept as $2:rollback-$commit" || true
}

sh "$here/containers/sync-to-host.sh" "$host" "$root"

case "$stack" in
  panel|all)
    # Built here, pushed once, pulled by the host. The host compiles nothing.
    #
    # It used to build on the target from an rsynced tree, and that is precisely
    # how gpu-1 came up with a panel whose Python was current and whose pages
    # were weeks old: the build input was a directory on the host that the sync
    # did not refresh. An image named by commit cannot drift -- the bytes are
    # the same everywhere or the tag is different.
    image="$registry/helena-panel:$commit_full"
    # CI built and published this tag. Nothing is built here -- not on the
    # target, and not on whoever's laptop is running the deploy either. The
    # first version of this script built locally and immediately failed on a
    # machine with no Docker daemon, which was the right lesson: a deploy is a
    # pull, and the only thing that builds is the pipeline.
    say "looking for $image in the registry"
    if ! on_host "curl -sf -o /dev/null https://$registry_host/v2/helena/helena-panel/manifests/$commit_full \
                    -H 'Accept: application/vnd.oci.image.index.v1+json' \
                    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json'"; then
      echo "deploy-to-host: $image is not published." >&2
      echo "  CI builds and pushes it. Merge this commit to staging and let the" >&2
      echo "  pipeline finish, then run this again -- or pass a tag that exists:" >&2
      echo "    containers/deploy-to-host.sh $host panel <commit>" >&2
      exit 4
    fi
    keep "$(on_host "sudo -n grep -oE '^HELENA_PANEL_IMAGE=.*' /etc/helena/platform.env | cut -d= -f2")" helena-panel
    on_host "sudo -n cp /etc/helena/platform.env /etc/helena/platform.env.bak-$commit \
             && sudo -n sed -i 's|^HELENA_PANEL_IMAGE=.*|HELENA_PANEL_IMAGE=$image|' /etc/helena/platform.env"
    on_host "sudo -n docker compose -f $root/containers/compose/platform.compose.yaml --env-file /etc/helena/platform.env pull panel" \
      | grep -iE 'pulled|error' || true
    on_host "sudo -n docker compose -f $root/containers/compose/platform.compose.yaml --env-file /etc/helena/platform.env up -d --no-build panel" \
      | grep -E 'helena-panel (Started|Recreated)' || true
    ;;
esac

case "$stack" in
  segment|all)
    tag="helena-worker-cpp:local-$commit"
    say "building $tag"
    keep "$(on_host "sudo -n grep -oE '^HELENA_SEGMENT_IMAGE=.*' /etc/helena/segment.env | cut -d= -f2")" helena-worker-cpp
    on_host "cd $root && sudo -n env BUILD_COMMIT=$commit sh containers/build-worker.sh $root $tag" \
      | grep -E 'naming to|ERROR' || true
    # The env file names the image, so it moves with the build rather than the
    # registry tag being shadowed -- a `docker compose pull` used to silently
    # revert a locally built worker.
    on_host "sudo -n cp /etc/helena/segment.env /etc/helena/segment.env.bak-$commit \
             && sudo -n sed -i 's|^HELENA_SEGMENT_IMAGE=.*|HELENA_SEGMENT_IMAGE=$tag|' /etc/helena/segment.env"
    on_host "sudo -n docker compose -f $root/containers/compose/segment.compose.yaml --env-file /etc/helena/segment.env up -d" \
      | grep -E '(Started|Recreated)' || true
    ;;
esac

case "$stack" in
  qc|all)
    tag="helena-worker-gpu:local-$commit"
    inflight="$(on_host "sudo -n docker exec helena-postgres psql -U campaignx -d campaignx -tAc \"select count(*) from segment_qc_jobs where state='CLAIMED'\"" 2>/dev/null | tr -d '[:space:]')"
    if [ "${inflight:-0}" != "0" ] && [ "${HELENA_QC_INTERRUPT:-}" != "yes" ]; then
      say "$inflight QC job(s) are running right now."
      say "Recreating throws that work away; it requeues when the lease expires."
      say "Re-run with HELENA_QC_INTERRUPT=yes to do it anyway, or wait."
      exit 3
    fi
    say "building $tag on $qc_base"
    keep "$(on_host "sudo -n grep -oE '^HELENA_QC_IMAGE=.*' /etc/helena/surface-qc.env | cut -d= -f2")" helena-worker-gpu
    on_host "cd $root && sudo -n docker build -q --build-arg BASE_IMAGE=$qc_base --build-arg BUILD_COMMIT=$commit \
             -f containers/images/Containerfile.worker-gpu -t $tag ." >/dev/null
    on_host "sudo -n cp /etc/helena/surface-qc.env /etc/helena/surface-qc.env.bak-$commit \
             && sudo -n sed -i 's|^HELENA_QC_IMAGE=.*|HELENA_QC_IMAGE=$tag|' /etc/helena/surface-qc.env"
    for device in $devices; do
      # -p per device and HELENA_QC_DEVICE overridden: see the notes at the top.
      on_host "sudo -n env HELENA_QC_DEVICE=$device docker compose -p helena-qc-$device \
               -f $root/containers/compose/surface-qc.compose.yaml \
               --env-file /etc/helena/surface-qc.env up -d" \
        | grep -E '(Started|Recreated)' | sed "s/^/  gpu $device: /" || true
    done
    ;;
esac

say "deployed $stack to $host at $commit"
