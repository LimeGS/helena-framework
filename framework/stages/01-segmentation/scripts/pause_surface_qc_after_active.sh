#!/bin/sh
# Stop long-lived QC containers immediately after their currently claimed job
# leaves CLAIMED. This drains in-flight scientific work without allowing the
# watch loop to claim another surface during an operator-requested pause.

set -eu

: "${QC_PAUSE_PAIRS:?space-separated container:worker_id pairs are required}"

postgres_container="${QC_POSTGRES_CONTAINER:-helena-postgres}"
postgres_user="${QC_POSTGRES_USER:-campaignx}"
postgres_database="${QC_POSTGRES_DATABASE:-campaignx}"
poll_seconds="${QC_PAUSE_POLL_SECONDS:-1}"

case "$poll_seconds" in
  ''|*[!0-9]*) echo "QC_PAUSE_POLL_SECONDS must be a positive integer" >&2; exit 2 ;;
  0) echo "QC_PAUSE_POLL_SECONDS must be positive" >&2; exit 2 ;;
esac

claimed_by_worker() {
  worker_id=$1
  case "$worker_id" in
    ''|*[!A-Za-z0-9._-]*)
      echo "worker_id contains unsupported characters: $worker_id" >&2
      return 2
      ;;
  esac
  sudo docker exec "$postgres_container" \
    psql -U "$postgres_user" -d "$postgres_database" -At \
    -c "SELECT count(*) FROM segment_qc_jobs WHERE state='CLAIMED' AND worker_id='$worker_id';"
}

pause_one() {
  pair=$1
  container=${pair%%:*}
  worker_id=${pair#*:}
  if [ "$container" = "$pair" ] || [ -z "$container" ] || [ -z "$worker_id" ]; then
    echo "invalid container:worker_id pair: $pair" >&2
    return 2
  fi
  if ! sudo docker container inspect "$container" >/dev/null 2>&1; then
    echo "QC container is absent: $container" >&2
    return 2
  fi
  echo "waiting for current QC claim to finish: container=$container worker=$worker_id"
  while :; do
    count=$(claimed_by_worker "$worker_id")
    case "$count" in
      0)
        sudo docker stop "$container" >/dev/null
        echo "QC container paused after active claim: $container"
        return 0
        ;;
      1) ;;
      *) echo "unexpected claimed-job count for $worker_id: $count" >&2; return 2 ;;
    esac
    sleep "$poll_seconds"
  done
}

pids=""
for pair in $QC_PAUSE_PAIRS; do
  pause_one "$pair" &
  pids="$pids $!"
done

status=0
for pid in $pids; do
  if ! wait "$pid"; then
    status=2
  fi
done
exit "$status"
