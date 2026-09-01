-- A worker that is not claiming looks exactly like a worker with nothing to
-- claim. Three of them stopped for eighteen hours and the fleet page showed
-- what it shows for an idle Sunday: `docker ps` said "Up 27 hours" the whole
-- time, and the only way anyone found out was noticing that a queue with
-- pending work was not draining.
--
-- ink_hosts.last_seen_at cannot answer this. It is written by the host-report
-- timer, which is a separate branch of the same loop -- so it kept reporting
-- while the claim beside it was blocked, and a host looked healthy because
-- part of it was.
--
-- This row is written by the claim itself. Its absence is the signal.
CREATE TABLE IF NOT EXISTS ink_workers (
    worker_id     TEXT PRIMARY KEY,
    host_id       TEXT NOT NULL,
    runtime       TEXT,
    last_poll_at  TIMESTAMPTZ NOT NULL,
    last_claim_at TIMESTAMPTZ,
    phases        TEXT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ink_workers_last_poll_idx
    ON ink_workers (last_poll_at DESC);
