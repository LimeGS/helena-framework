#!/bin/sh
# Copy this checkout to a host that runs it.
#
#   containers/sync-to-host.sh gpu-1 [/ssd/vc3d/campaign-x]
#
# Written because I got it wrong twice, the same way both times. The obvious
# command is
#
#   rsync -az panel/app.py framework tests containers HOST:/ssd/vc3d/campaign-x/
#
# and it is wrong: with several sources and a directory destination, rsync places
# each source at the *top* of the destination. The directories keep their names,
# so framework/ and tests/ arrive correctly and everything looks fine -- and
# panel/app.py lands at /ssd/vc3d/campaign-x/app.py, where nothing reads it. The
# build then succeeds, the image label updates, the panel restarts, and it is
# running the previous app.py. A whole day's backend changes deployed into a file
# nobody opens, twice, because the failure has no symptom.
#
# So: every file is copied to its own explicit destination path, one rsync each.
set -eu

host="${1:?usage: sync-to-host.sh HOST [REMOTE_ROOT]}"
root="${2:-/ssd/vc3d/campaign-x}"
here="$(cd "$(dirname "$0")/.." && pwd)"

# Directories, which are safe as a group: a source without a trailing slash keeps
# its own name under the destination.
# No --delete-excluded: combined with these rules rsync 3.4.1 fails on the
# receiver with "buffer overflow: recv_rules". Excluding is all that is wanted.
rsync -az --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
  "$here/framework" "$here/tests" "$here/containers" "$host:$root/"

# Files, one explicit destination each. This is the whole point of the script.
for file in panel/app.py panel/requirements.txt; do
  rsync -az "$here/$file" "$host:$root/$file"
done

# The frontend's SOURCE, because the image builds it now.
#
# This used to send only panel/web/dist, which was right while the Containerfile
# COPYied that directory. Once the image grew a node stage that builds from
# panel/web, the host's stale source became the build input -- and the panel
# deployed to gpu-1 with a frontend from weeks earlier: no user guide, no API
# reference, no developer reference. The image label said the right commit,
# because the Python half was current. Only the pages were old.
#
# node_modules is excluded because `npm ci` recreates it from the lock file, and
# dist because the image produces it.
rsync -az --delete --exclude 'node_modules' --exclude 'dist' \
  "$here/panel/web/" "$host:$root/panel/web/"

# And prove it, rather than trusting the copy. A silent no-op deploy is exactly
# what this script exists to prevent, so it refuses to exit cleanly on one.
for file in panel/app.py panel/requirements.txt panel/web/package-lock.json \
            panel/web/src/routes/UserGuide.tsx; do
  mine="$(shasum -a 256 "$here/$file" | cut -d' ' -f1)"
  theirs="$(ssh -o BatchMode=yes "$host" "sha256sum $root/$file 2>/dev/null | cut -d' ' -f1" || true)"
  if [ "$mine" != "$theirs" ]; then
    echo "sync-to-host: $file did not arrive: local $mine, remote ${theirs:-absent}" >&2
    exit 1
  fi
  echo "  $file  $(echo "$mine" | cut -c1-16)  matches"
done

# Strays from an earlier flattened copy, which are harmless but read as current.
ssh -o BatchMode=yes "$host" "rm -f $root/app.py $root/requirements.txt"
echo "  synced to $host:$root"
