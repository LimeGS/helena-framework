#!/bin/sh
# Deploy the panel to a host.
#
#   panel/deploy.sh HOST
#
# The one thing this gets right that a hand-typed rsync did not: it does not
# delete the old asset chunks.
#
# Route chunks are content-hashed, so every build renames all of them. A browser
# tab that is open across a deploy still holds the previous index.html and goes
# on asking for the previous names; `rsync --delete` removes exactly those, and
# the next navigation 404s inside a lazy boundary with nothing on screen. The
# front end reloads itself when that happens now, but not deleting them means it
# does not have to.
#
# They are pruned by age instead, well past any session worth keeping.
set -eu

host="${1:?usage: deploy.sh <ssh-host> [remote-root]}"
root="${2:-/srv/helena/framework}"
here="$(cd "$(dirname "$0")/.." && pwd)"
prune_days="${PANEL_ASSET_PRUNE_DAYS:-14}"

test -f "$here/panel/web/dist/index.html" || {
  echo "no build to deploy: run 'npm --prefix panel/web run build' first" >&2
  exit 2
}

echo "assets -> $host:$root (keeping superseded chunks)"
rsync -az --rsync-path="sudo rsync" \
  "$here/panel/web/dist/" "$host:$root/panel/web/dist/"

echo "code -> $host:$root"
rsync -az --rsync-path="sudo rsync" "$here/panel/app.py" "$host:$root/panel/app.py"
rsync -az --rsync-path="sudo rsync" "$here/framework/" "$host:$root/framework/"

# index.html is the one file that must never be stale: it names the chunks.
ssh "$host" "sudo find $root/panel/web/dist/assets -type f -mtime +$prune_days -delete 2>/dev/null || true"
# The panel is not a systemd unit on this fleet, so `systemctl restart ||
# pkill` killed it and left nothing to bring it back. It stayed up only because
# earlier deploys happened to be followed by something else starting it.
ssh "$host" "sudo systemctl restart helena-panel 2>/dev/null && exit 0
  pkill -f 'uvicorn panel.app' 2>/dev/null || true
  sleep 1
  cd $root && setsid nohup panel/.venv/bin/python -m uvicorn panel.app:app \
    --host 0.0.0.0 --port 8800 > $root/panel.log 2>&1 < /dev/null & disown" || true

# Verified rather than assumed: a deploy that leaves the panel down should say
# so here, not the next time somebody opens it.
sleep 6
code=$(ssh "$host" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8800/")
echo "panel: $code"
test "$code" = "200" || { echo "the panel did not come back up" >&2; exit 1; }
ssh "$host" "grep -oE 'assets/index-[A-Za-z0-9_-]+\.js' $root/panel/web/dist/index.html | head -1"
