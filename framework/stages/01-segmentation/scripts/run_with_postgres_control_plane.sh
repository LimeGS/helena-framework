#!/bin/sh
set -eu

# Load a PostgreSQL password without putting it in argv, receipts or Git.
# The env file must be mode 0600 and contain POSTGRES_USER, POSTGRES_DB and
# POSTGRES_PASSWORD.  Production workers should prefer a short-lived secret
# injection or workload identity over a persistent file.
: "${SEGMENT_FLEET_POSTGRES_ENV_FILE:?set SEGMENT_FLEET_POSTGRES_ENV_FILE}"
if [ ! -r "$SEGMENT_FLEET_POSTGRES_ENV_FILE" ]; then
  printf '%s\n' "PostgreSQL env file is not readable: $SEGMENT_FLEET_POSTGRES_ENV_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
. "$SEGMENT_FLEET_POSTGRES_ENV_FILE"
set +a

: "${POSTGRES_USER:?POSTGRES_USER missing from env file}"
: "${POSTGRES_DB:?POSTGRES_DB missing from env file}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD missing from env file}"

SEGMENT_FLEET_POSTGRES_HOST=${SEGMENT_FLEET_POSTGRES_HOST:-127.0.0.1}
SEGMENT_FLEET_POSTGRES_PORT=${SEGMENT_FLEET_POSTGRES_PORT:-55432}
export SEGMENT_FLEET_POSTGRES_HOST SEGMENT_FLEET_POSTGRES_PORT
export SEGMENT_FLEET_DATABASE_URL
SEGMENT_FLEET_DATABASE_URL=$(python3 -c '
import os
from urllib.parse import quote
print("postgresql://{}:{}@{}:{}/{}".format(
    quote(os.environ["POSTGRES_USER"], safe=""),
    quote(os.environ["POSTGRES_PASSWORD"], safe=""),
    os.environ["SEGMENT_FLEET_POSTGRES_HOST"],
    os.environ["SEGMENT_FLEET_POSTGRES_PORT"],
    quote(os.environ["POSTGRES_DB"], safe=""),
))
')

unset POSTGRES_PASSWORD
exec "$@"
