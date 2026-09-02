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
# Docker, a route to the control plane, the compose files and the env file.
# Nothing is compiled here and nothing runs outside a container: that is the
# whole point of shipping an image, and this host previously had VC3D and ink
# models built directly on it, which is exactly the thing that stops being
# reproducible.
#
# It used to say that and then install two systemd units, which is the same
# mistake in a smaller shape -- a per-host unit is a second way to deploy, with
# its own failure mode and nothing that reports it. Both are compose services.
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
tunnel_image="${HELENA_TUNNEL_IMAGE:-helena-control-tunnel:$version}"
control="${HELENA_CONTROL_HOST:?set it to the host running the control plane}"
# The key that opens the forward, on this machine, read by the target's compose
# through the copy written below. Give it its own key restricted on the control
# plane with `command="",permitopen="127.0.0.1:55432"`: it needs to open one
# port and nothing else.
control_key="${HELENA_CONTROL_KEY:-$HOME/.ssh/helena_fleet}"
[ -r "$control_key" ] || {
  echo "no readable key at $control_key; set HELENA_CONTROL_KEY to the private" >&2
  echo "key that opens the forward to $control" >&2
  exit 2
}

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

# Runs land beside the artifacts, on the same disk that was just found: a run is
# hundreds of megabytes and the default under / fills the root filesystem on a
# machine whose space is all on /nvme.
runs="$(dirname "$artifacts")/runs"
ssh -- "$target" "sudo mkdir -p '$runs' && sudo chown 1000:1000 '$runs'"
say "  $runs"

# The host's own name in the fleet's terms, asked of the host rather than
# assumed from the ssh target: `root@10.0.0.4` is not a host id, and two hosts
# sharing one claim each other's work with no receipt able to say which ran.
# The role distinguishes two workers on one machine.
host_id="$(ssh -- "$target" 'hostname -s')-$role"
say "host id $host_id"

# ------------------------------------------------------------------ the image
# Streamed over the same SSH connection rather than pulled from a registry:
# these hosts cannot route to each other on anything but 22, and standing up a
# registry to move one image between two machines is a service to maintain
# forever in exchange for one file transfer.
for wanted in "$image" "$tunnel_image"; do
  say "image $wanted"
  if ssh -- "$target" "sudo docker image inspect '$wanted' >/dev/null 2>&1"; then
    say "  already present"
  else
    say "  streaming (the worker image takes a few minutes)"
    docker save "$wanted" | ssh -- "$target" "sudo docker load"
  fi
done

# ----------------------------------------------------------------------- units
say "units"
ssh -- "$target" "sudo install -d -m 0750 /etc/helena"

# The worker's environment. The token and DSN are read on the control host and
# written straight to the target: they never touch this machine's disk or a
# command line, where they would land in shell history and process listings.
ssh -- "$control" 'sudo cat /etc/helena/panel.env' \
  | sed -n 's/^CX_DB=/FLEET_DB=/p' \
  | ssh -- "$target" "sudo tee /etc/helena/worker.env >/dev/null"

# The names segment.compose.yaml and control-tunnel.compose.yaml read. They
# were HELENA_WORKER_IMAGE / HELENA_ARTIFACT_DIR / WORKER_ID, which only the
# systemd unit ever looked at -- so this file configured a unit that is gone
# and left compose falling back to its own defaults.
#
# QC_PROFILE_ID is the profile this repository ships: the worker refuses to
# start without one, because a geometry verdict has to name the profile that
# produced it. It is written literally, not as a shell default, since nothing
# expands this file but a dotenv reader.
ssh -- "$target" "printf '%s\n' \
  'HELENA_SEGMENT_IMAGE=$image' \
  'HELENA_TUNNEL_IMAGE=$tunnel_image' \
  'HELENA_SEGMENT_ARTIFACTS=$artifacts' \
  'HELENA_SEGMENT_HOST_ID=$host_id' \
  'HELENA_FLEET_RUNS=$runs' \
  'HELENA_RUN_AS=1000:1000' \
  'HELENA_CONTROL_HOST=$control' \
  'HELENA_CONTROL_KEY=/etc/helena/helena_fleet' \
  'QC_PROFILE_ID=${QC_PROFILE_ID:-surface-qc-gp-scroll1-ct-fiber-v3@1.0.0}' \
  | sudo tee -a /etc/helena/worker.env >/dev/null
  sudo chmod 0600 /etc/helena/worker.env"

# The forward's key, placed where the tunnel's compose mounts it from. Written
# straight through, like the DSN above: it never lands on this machine's disk
# outside its own keyring and never appears on a command line.
#
# Owned by 1000, not root. The tunnel container runs as uid 1000 and a key at
# root:root 0600 is mounted fine and unreadable -- the container then restarts
# forever saying "no readable key", which reads as a missing mount rather than
# as ownership. Same shape as the artifact volumes: Docker creates a bind path
# root:root and the services run as 1000.
sudo_install_key='sudo tee /etc/helena/helena_fleet >/dev/null \
  && sudo chown 1000:1000 /etc/helena/helena_fleet \
  && sudo chmod 0600 /etc/helena/helena_fleet'
ssh -- "$target" "$sudo_install_key" < "$control_key"

# A GPU host gets --gpus all; a CPU host gets an empty variable, which the unit
# expands to no words at all. Passing an empty quoted string here is what made
# docker read the image name as an argument and refuse the whole command.
ssh -- "$target" 'if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "HELENA_WORKER_GPU_ARGS=--gpus all" | sudo tee -a /etc/helena/worker.env >/dev/null
  fi'

# ------------------------------------------------------------------- compose
# This installed two systemd units, which contradicted the paragraph at the top
# of this file: nothing runs on a Helena host outside a container. A per-host
# unit is a second way to deploy, with its own failure mode, its own logs and
# nothing that reports it to the panel -- and one of the two,
# helena-worker-cpp.service, was not even in the tree any more, so this section
# had been failing on a missing file.
#
# The compose files and the env files under /etc/helena are what a host holds
# besides Docker. They are copied rather than cloned: a checkout on a worker is
# a second copy of the source to drift.
say "compose files"
ssh -- "$target" "sudo install -d -m 0755 /etc/helena/compose"
scp -q -- "$here/containers/compose/segment.compose.yaml" \
       "$here/containers/compose/control-tunnel.compose.yaml" "$target:/tmp/"
ssh -- "$target" 'sudo install -m 0644 /tmp/segment.compose.yaml /etc/helena/compose/
  sudo install -m 0644 /tmp/control-tunnel.compose.yaml /etc/helena/compose/
  rm -f /tmp/segment.compose.yaml /tmp/control-tunnel.compose.yaml'

say "the tunnel"
ssh -- "$target" "sudo docker compose -p helena-tunnel \
    -f /etc/helena/compose/control-tunnel.compose.yaml \
    --env-file /etc/helena/worker.env up -d"

say "the worker"
ssh -- "$target" "sudo HELENA_SEGMENT_ENV=/etc/helena/worker.env docker compose \
    -p helena-segment -f /etc/helena/compose/segment.compose.yaml \
    --env-file /etc/helena/worker.env up -d"

# ----------------------------------------------------------------------- proof
# Reported rather than assumed. A provisioning script whose last line is the
# start command tells you the command ran, which is not the question.
sleep 8
say "result"
# Asked of Docker, which is the only thing running any of this now. A container
# that exited is reported with its status rather than omitted: "not deployed"
# and "started and died" are different answers and the second one is the one
# worth seeing.
ssh -- "$target" 'for name in helena-control-tunnel helena-segment; do
    printf "    %-24s %s\n" "$name" \
      "$(sudo docker inspect -f "{{.State.Status}} ({{.State.ExitCode}}), restarts {{.RestartCount}}" \
         "$name" 2>/dev/null || echo "not deployed")"
  done'
