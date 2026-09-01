#!/bin/sh
# A lane image is a runtime, not a worker.
#
# `helena-ink-9um` carries /opt/villa and nothing else: upstream's interpreter,
# its virtualenv and its source tree. The worker that claims jobs lives in
# `helena-worker-gpu`, which carries this lane at /opt/lanes/ink-9um -- and the
# two are set side by side in one compose invocation:
#
#   HELENA_INK_IMAGE=helena-worker-gpu:local       <- what runs
#   HELENA_RUNTIME_IMAGE=helena-ink-9um            <- what it says it is
#
# Getting them the same way round produced 27 hours of crash-loop reporting
# `can't open file '.../ink_worker.py': [Errno 2] No such file or directory`,
# which is true and says nothing about which of the two variables is wrong.
set -eu

case "${*:-}" in
  *ink_worker.py*|*fleet/cli.py*)
    if [ ! -f /workspace/campaign-x/framework/stages/03-ink/fleet/ink_worker.py ]; then
      echo "This is helena-ink-9um, the lane runtime. It carries /opt/villa" >&2
      echo "and no repository, so it cannot run a fleet worker." >&2
      echo >&2
      echo "You want helena-worker-gpu, which carries this lane at" >&2
      echo "/opt/lanes/ink-9um. In the compose invocation:" >&2
      echo >&2
      echo "  HELENA_INK_IMAGE=helena-worker-gpu:local       # what runs" >&2
      echo "  HELENA_RUNTIME_IMAGE=helena-ink-9um            # what it says it is" >&2
      echo >&2
      echo "Both are needed and they are not the same image." >&2
      exit 3
    fi
    ;;
esac

exec "$@"
