#!/bin/sh
# Turn a machine with SSH into a worker host.
#
#   containers/provision-host.sh <ssh-target> [role]
#
# Adding a host in the panel used to write a database row and nothing else, so
# a host appeared in the table, looked registered, and claimed no work -- and
# the only way to find out was to notice the queue not moving. This is what the
# row was promising.
#
# Everything the worker needs is in the image. What has to exist on the host is
# Docker, a route to the control plane, and the two units. Nothing is compiled
# here and nothing is installed outside the image: that is the whole point of
# shipping one, and this host previously had VC3D and ink models built directly
# on it, which is exactly the thing that stops being reproducible.
#
# Idempotent. Running it twice is how you upgrade a host to a new image.
set -eu

target="${1:?usage: provision-host.sh <ssh-target> [role]}"
role="${2:-worker}"
here="$(cd "$(dirname "$0")/.." && pwd)"
# Pinned, like the compose files are. `latest` on a host being provisioned is
# whatever the daemon last pulled under that name -- so two hosts provisioned a
# week apart ran different code and both reported the same image, which is the
# one failure a fleet cannot reason its way out of. VERSION is the same file
# build-images.sh tags from.
version="$(cat "$here/VERSION" 2>/dev/null || true)"
[ -n "$version" ] || {
  echo "no VERSION file beside this script; refusing to provision a host with" >&2
  echo "a floating tag. Set HELENA_WORKER_IMAGE to the exact image to run." >&2
  exit 2
}
image="${HELENA_WORKER_IMAGE:-helena-worker-cpp:$version}"
control="${HELENA_CONTROL_HOST:?set it to the host running the control plane}"

say() { printf '  %s\n' "$*"; }

# ---------------------------------------------------------------- reachability
say "checking $target"
ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$target" true \
  || { echo "cannot ssh to $target without a password" >&2; exit 2; }

# --------------------------------------------------------------------- docker
# get.docker.com rather than the distribution's docker.io: the packaged one
# lags, and the worker image is built against a current engine.
say "docker"
ssh -- "$target" 'command -v docker >/dev/null 2>&1 || {
    echo "    installing docker"
    curl -fsSL https://get.docker.com | sudo sh >/dev/null 2>&1
  }
  sudo systemctl enable --now docker >/dev/null 2>&1 || true
  docker --version'

# ----------------------------------------------------------------------- disk
# Hosts do not agree on where the big disk is, so this asks rather than assumes:
# a unit naming /ssd works on one machine and silently fills / on the next.
say "artifact directory"
artifacts=$(ssh -- "$target" '
  for candidate in /nvme /ssd /mnt/campaignx /var/lib; do
    [ -d "$candidate" ] && { echo "$candidate/campaignx/artifacts"; exit 0; }
  done
  echo /var/lib/campaignx/artifacts')
ssh -- "$target" "sudo mkdir -p '$artifacts' && sudo chmod 0755 '$artifacts'"
say "  $artifacts"

# ------------------------------------------------------------------ the image
# Streamed over the same SSH connection rather than pulled from a registry:
# these hosts cannot route to each other on anything but 22, and standing up a
# registry to move one image between two machines is a service to maintain
# forever in exchange for one file transfer.
say "image $image"
if ssh -- "$target" "sudo docker image inspect '$image' >/dev/null 2>&1"; then
  say "  already present"
else
  say "  streaming (this takes a few minutes)"
  docker save "$image" | ssh -- "$target" "sudo docker load"
fi

# ----------------------------------------------------------------------- units
say "units"
ssh -- "$target" "sudo install -d -m 0750 /etc/helena"

# The worker's environment. The token and DSN are read on the control host and
# written straight to the target: they never touch this machine's disk or a
# command line, where they would land in shell history and process listings.
ssh -- "$control" 'sudo cat /etc/helena/panel.env' \
  | sed -n 's/^CX_DB=/FLEET_DB=/p' \
  | ssh -- "$target" "sudo tee /etc/helena/worker.env >/dev/null"

ssh -- "$target" "printf '%s\n' \
  'HELENA_WORKER_IMAGE=$image' \
  'HELENA_ARTIFACT_DIR=$artifacts' \
  'QC_PROFILE_ID=\${QC_PROFILE_ID:-semantic-qc-v1@1}' \
  'WORKER_ID=\$(hostname)-$role' \
  | sudo tee -a /etc/helena/worker.env >/dev/null
  sudo chmod 0600 /etc/helena/worker.env"

# A GPU host gets --gpus all; a CPU host gets an empty variable, which the unit
# expands to no words at all. Passing an empty quoted string here is what made
# docker read the image name as an argument and refuse the whole command.
ssh -- "$target" 'if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "HELENA_WORKER_GPU_ARGS=--gpus all" | sudo tee -a /etc/helena/worker.env >/dev/null
  fi'

scp -q -- "$here/containers/systemd/helena-control-tunnel.service" \
       "$here/containers/systemd/helena-worker-cpp.service" "$target:/tmp/"
ssh -- "$target" 'sudo install -m 0644 /tmp/helena-control-tunnel.service /etc/systemd/system/
  sudo install -m 0644 /tmp/helena-worker-cpp.service /etc/systemd/system/
  rm -f /tmp/helena-*.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now helena-control-tunnel
  sudo systemctl reset-failed helena-worker-cpp 2>/dev/null || true
  sudo systemctl enable --now helena-worker-cpp'

# ----------------------------------------------------------------------- proof
# Reported rather than assumed. A provisioning script whose last line is the
# start command tells you the command ran, which is not the question.
sleep 8
say "result"
ssh -- "$target" 'printf "    tunnel: %s\n    worker: %s\n" \
    "$(systemctl is-active helena-control-tunnel)" \
    "$(systemctl is-active helena-worker-cpp)"
  printf "    container: %s\n" "$(sudo docker ps --format "{{.Names}} {{.Status}}" \
    | grep helena-worker-cpp || echo "not running")"'
