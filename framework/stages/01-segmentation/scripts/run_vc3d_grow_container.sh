#!/bin/sh
# Execute the VC3D grow binary from the immutable Helena Framework image while
# keeping all mutable state on the worker's NVMe root. The parent directory is
# mounted, not the final surface directory: VC3D commits output through an
# atomic directory exchange which cannot replace a bind-mount point.
set -eu

image="${HELENA_VC3D_IMAGE:-localhost:5000/helena/helena-vc3d:0.3.2}"
data_root="${HELENA_WORKER_DATA_ROOT:-/srv/helena}"
expected_image_id="${HELENA_VC3D_IMAGE_ID:-}"
device="${CUDA_VISIBLE_DEVICES:-0}"
device="${device%%,*}"

case "$data_root" in
  /*) ;;
  *) echo "HELENA_WORKER_DATA_ROOT must be absolute" >&2; exit 2 ;;
esac
test -d "$data_root" || { echo "worker data root is missing: $data_root" >&2; exit 2; }
test "$#" -gt 0 || { echo "usage: run_vc3d_grow_container.sh [vc_grow arguments...]" >&2; exit 2; }

# Every local path passed by the fleet must live below the one mounted worker
# root. URLs and scalar arguments are unaffected. This prevents an apparently
# successful container run from writing into an ephemeral container layer.
for argument in "$@"; do
  case "$argument" in
    /*)
      case "$argument" in
        "$data_root"|"$data_root"/*) ;;
        *) echo "refusing an unmounted absolute path: $argument" >&2; exit 2 ;;
      esac
      ;;
  esac
done

docker_exec() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo -n docker "$@"
  fi
}

actual_image_id="$(docker_exec image inspect "$image" --format '{{.Id}}')"
if test -n "$expected_image_id" && test "$actual_image_id" != "$expected_image_id"; then
  echo "VC3D image identity mismatch" >&2
  exit 3
fi

rng_seed="${VC_GROWPATCH_RNG_SEED:-helena-default-seed}"
docker_exec run --rm \
  --name "helena-vc3d-grow-$$" \
  --gpus "device=$device" \
  --network host \
  --label campaignx.stage=01-segmentation \
  --label campaignx.runtime=vc3d-grow \
  -e "VC_GROWPATCH_RNG_SEED=$rng_seed" \
  -v "$data_root:$data_root" \
  "$image" \
  /opt/campaignx/vc3d/bin/vc_grow_seg_from_seed "$@"
