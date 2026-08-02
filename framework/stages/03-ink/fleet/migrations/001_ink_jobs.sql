-- Ink jobs reuse the segmentation fleet's lease discipline rather than adding a
-- second scheduler: FOR UPDATE SKIP LOCKED, a hashed lease token, an attempt
-- counter and an event log. Any host running an ink worker can claim work.

CREATE TABLE IF NOT EXISTS ink_jobs (
    job_id              TEXT PRIMARY KEY,
    sample_id           TEXT NOT NULL,
    profile_id          TEXT NOT NULL,
    parameters          JSONB NOT NULL,
    state               TEXT NOT NULL DEFAULT 'pending',
    priority            INTEGER NOT NULL DEFAULT 0,
    requested_host      TEXT,
    gpu_required        BOOLEAN NOT NULL DEFAULT TRUE,
    minimum_vram_gb     INTEGER NOT NULL DEFAULT 0,
    worker_id           TEXT,
    lease_token_hash    TEXT,
    lease_expires_at    TIMESTAMPTZ,
    attempts            INTEGER NOT NULL DEFAULT 0,
    max_attempts        INTEGER NOT NULL DEFAULT 3,
    output_dir          TEXT,
    result              JSONB,
    created_by          TEXT NOT NULL DEFAULT 'panel',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ink_jobs_state CHECK (state IN
        ('pending','leased','running','succeeded','failed','cancelled'))
);

CREATE INDEX IF NOT EXISTS ink_jobs_claimable
    ON ink_jobs (priority DESC, created_at)
    WHERE state = 'pending';

CREATE INDEX IF NOT EXISTS ink_jobs_sample ON ink_jobs (sample_id);

CREATE TABLE IF NOT EXISTS ink_job_events (
    event_id    BIGSERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL REFERENCES ink_jobs(job_id) ON DELETE CASCADE,
    event_type  TEXT NOT NULL,
    payload     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ink_job_events_job ON ink_job_events (job_id, event_id);

-- A host is a place work can run. Registered here so the panel can show and
-- target them without a second source of truth in a config file somewhere.
CREATE TABLE IF NOT EXISTS ink_hosts (
    host_id         TEXT PRIMARY KEY,
    ssh_target      TEXT NOT NULL,
    roles           TEXT[] NOT NULL DEFAULT '{}',
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    notes           TEXT,
    last_seen_at    TIMESTAMPTZ,
    last_state      JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
