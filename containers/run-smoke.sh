#!/bin/sh
# P0 to P4 against a running platform, in a container, through the API alone.
#
#   sudo containers/run-smoke.sh USERNAME
#
# The harness is stdlib-only and already inside the panel image, so there is
# nothing to build and nothing to install on the host -- which is the point: a
# smoke test that needs a prepared machine tests the machine.
#
# The password comes from an env file the harness reads itself, never from this
# command line, so it does not land in a process list, a shell history or a
# terminal scrollback:
#
#   umask 077; printf 'HELENA_PANEL_PASSWORD=%s\n' 'the password' > /etc/helena/smoke.env
#
# Host networking, because the panel listens on loopback and the whole run is
# supposed to be reachable by anybody with the panel open in a browser.
set -eu

user="${1:?usage: run-smoke.sh USERNAME}"
panel="${HELENA_PANEL_URL:-https://127.0.0.1:8800}"
image="${HELENA_PANEL_IMAGE:-helena-panel:latest}"
credentials="${HELENA_SMOKE_ENV:-/etc/helena/smoke.env}"
scroll="${HELENA_SMOKE_SCROLL:-PHerc0826}"
volume_url="${HELENA_SMOKE_VOLUME_URL:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr}"
# As the ink worker sees it: /srv/helena/cache is mounted into that container at
# the same path, so a cache directory means one thing on both sides.
volume_cache="${HELENA_SMOKE_VOLUME_CACHE:-/srv/helena/cache/pherc826-9362}"
# Two laminae the fleet has already grown surfaces from. A positive control:
# an empty result means the pipeline, not the scroll.
seeds="${HELENA_SMOKE_SEEDS:-3161,2660,5584;3175,4679,5912}"

test -r "$credentials" || {
  echo "no credentials at $credentials -- see the header of this script" >&2
  exit 2
}

# The panel's certificate is self-signed and lives in its state directory,
# which this container does not mount. Verification is skipped for the smoke
# run specifically: it talks to 127.0.0.1 on the same host, so there is no
# network for anyone to sit in the middle of. Mount the certificate and set
# HELENA_PANEL_TLS_TRUST instead when running this from anywhere else.
exec docker run --rm --network host --env-file "$credentials" \
  -e HELENA_PANEL_TLS_INSECURE="${HELENA_PANEL_TLS_INSECURE:-1}" \
  -e HELENA_PANEL_TLS_TRUST="${HELENA_PANEL_TLS_TRUST:-}" "$image" \
  python3 scripts/harness/smoke_p0_p4.py \
    --panel "$panel" --user "$user" --scroll "$scroll" \
    --volume-url "$volume_url" --volume-cache "$volume_cache" --seeds "$seeds"
