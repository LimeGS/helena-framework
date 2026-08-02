#!/bin/sh
# Move a host from the campaignx names to the helena ones.
#
#   sudo containers/migrate-to-helena.sh [--commit]
#
# Prints what it would do and changes nothing until --commit. Safe to run twice:
# every step checks whether it has already happened.
#
# What moves: /etc/campaignx to /etc/helena, the CAMPAIGNX_* variables inside
# those files to HELENA_*, and the campaignx-*.service units to helena-*. What
# stays: the old configuration directory, kept until somebody deletes it, so a
# migration that goes wrong is one `systemctl` away from being undone.
set -eu

commit=false
[ "${1:-}" = "--commit" ] && commit=true
run() {
  if $commit; then "$@"; else printf '  would: %s\n' "$*"; fi
}

test "$(id -u)" -eq 0 || { echo "run as root: it edits /etc and systemd" >&2; exit 2; }

if [ -d /etc/campaignx ] && [ ! -d /etc/helena ]; then
  echo "configuration:"
  run cp -a /etc/campaignx /etc/helena
  # The variable names inside, not just the directory. A file that moved but
  # still says CAMPAIGNX_QC_IMAGE leaves the unit expanding an empty string,
  # which docker reports as an invalid image reference and nothing else.
  if $commit; then
    for f in /etc/helena/*.env; do
      [ -f "$f" ] || continue
      sed -i 's/^CAMPAIGNX_/HELENA_/' "$f"
    done
  else
    echo "  would: rewrite CAMPAIGNX_* to HELENA_* in /etc/helena/*.env"
  fi
elif [ -d /etc/helena ]; then
  echo "configuration: /etc/helena already exists, leaving both alone"
fi

echo "units:"
for old in /etc/systemd/system/campaignx-*.service; do
  [ -e "$old" ] || continue
  new="/etc/systemd/system/$(basename "$old" | sed 's/^campaignx-/helena-/')"
  base=$(basename "$old" .service)
  # An instance unit is enabled per instance; ask systemd which ones are on
  # rather than guessing the indices.
  instances=$(systemctl list-units --all --no-legend "${base%@}@*.service" 2>/dev/null \
                | awk '{print $1}' | sed 's/^●* *//' || true)
  echo "  $(basename "$old") -> $(basename "$new")"
  case "$old" in
    *@.service)
      for unit in $instances; do run systemctl disable --now "$unit"; done ;;
    *) run systemctl disable --now "$(basename "$old")" || true ;;
  esac
  run mv "$old" "$new"
done

run systemctl daemon-reload
echo
if $commit; then
  echo "enable what should run on this host, for example:"
else
  echo "nothing was changed. Re-run with --commit, then enable:"
fi
echo "  systemctl enable --now helena-worker helena-ink"
echo "  systemctl enable --now helena-surface-qc@0 helena-surface-qc@1"
echo
echo "The panel and the control plane are not units: they come up with"
echo "  docker compose --env-file /etc/helena/platform.env \\"
echo "    -f containers/compose/platform.compose.yaml up -d"
echo
echo "/etc/campaignx is left in place. Remove it once the services have run."
