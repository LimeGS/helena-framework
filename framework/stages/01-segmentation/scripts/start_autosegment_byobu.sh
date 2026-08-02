#!/bin/sh
# Start the QC-aware Stage 01 controller in one detached byobu/tmux session.
# All runtime configuration and credentials must already be present in this
# process environment; none are copied into the command line or repository.

set -eu

script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
controller="$script_dir/run_autosegment_after_qc.sh"
session="${AUTO_SEGMENT_BYOBU_SESSION:-campaign-x-autosegment}"
log_file="${AUTO_SEGMENT_BYOBU_LOG:-${RUN_ROOT:?RUN_ROOT is required}/control/byobu-autosegment.log}"

command -v byobu-tmux >/dev/null 2>&1 || {
  printf '%s\n' "byobu-tmux is required" >&2
  exit 2
}
[ -x "$controller" ] || {
  printf '%s\n' "autosegmentation controller is not executable: $controller" >&2
  exit 2
}

if byobu-tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' "byobu session already exists: $session"
  exit 0
fi

mkdir -p "$(dirname "$log_file")"
byobu-tmux new-session -d -s "$session" \
  "exec \"$controller\" >>\"$log_file\" 2>&1"
printf 'Started detached byobu session: %s\n' "$session"
printf 'Attach with: byobu attach -t %s\n' "$session"
printf 'Log: %s\n' "$log_file"
