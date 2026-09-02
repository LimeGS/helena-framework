#!/bin/sh
# One SSH forward, in the foreground, so Docker's restart policy is what keeps
# it up. The unit this replaces used Restart=always for the same reason.
set -eu

control="${HELENA_CONTROL_HOST:?set it to user@host of the control plane}"
port="${HELENA_CONTROL_PORT:-55432}"
key="${HELENA_CONTROL_KEY:-/keys/helena_fleet}"

# Readable, not merely present. This runs as uid 1000, and a key written by
# root at 0600 is mounted correctly and unreadable -- so the check says which
# of the two it is, because "no key" sends you to the mount and ownership is
# where the answer is.
if [ ! -e "$key" ]; then
  echo "no key at $key: mount one read-only, or set HELENA_CONTROL_KEY" >&2
  exit 2
fi
if [ ! -r "$key" ]; then
  echo "the key at $key is not readable by uid $(id -u): it is owned by" >&2
  echo "$(stat -c '%u:%g mode %a' "$key" 2>/dev/null || echo 'another user')." >&2
  echo "chown it to the uid this container runs as." >&2
  exit 2
fi

# ExitOnForwardFailure so a forward that cannot bind ends the process rather
# than leaving a connection that carries nothing: the container then restarts
# and says so, which is what the old unit could not do.
#
# -N because this opens a forward and runs no command. StrictHostKeyChecking
# is accept-new rather than no: the first connection is trusted, a changed key
# afterwards is refused.
exec ssh -N \
  -i "$key" \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="${HELENA_CONTROL_KNOWN_HOSTS:-/tmp/known_hosts}" \
  -L "127.0.0.1:${port}:127.0.0.1:${port}" \
  "$control"
