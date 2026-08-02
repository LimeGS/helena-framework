BEGIN;

CREATE TABLE IF NOT EXISTS segment_schema_migrations (
  version integer PRIMARY KEY,
  description text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_source_snapshots (
  source_snapshot_id text PRIMARY KEY,
  sample_id text NOT NULL,
  ct_uri text NOT NULL,
  ct_sha256 text,
  m7_uri text NOT NULL,
  m7_sha256 text,
  shape_xyz jsonb NOT NULL,
  voxel_size_um double precision NOT NULL,
  coordinate_frame text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS segment_source_snapshots_by_sample
  ON segment_source_snapshots(sample_id);

CREATE TABLE IF NOT EXISTS segment_surfaces (
  surface_id text PRIMARY KEY,
  source_snapshot_id text NOT NULL REFERENCES segment_source_snapshots(source_snapshot_id),
  sample_id text NOT NULL,
  owner text NOT NULL,
  artifact_sha256 text,
  artifact_uri text,
  bbox_xyz jsonb NOT NULL,
  sample_points jsonb,
  area_cm2 double precision,
  state text NOT NULL,
  physical_qc_state text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(source_snapshot_id, artifact_sha256)
);
-- The geometry verdict is a second axis, orthogonal to physical_qc_state: a
-- surface may be CT_SUPPORTED and GEOMETRY_REJECTED_BRIDGE at the same time.
-- Existing rows backfill to GEOMETRY_UNMEASURED, which is the truth -- nothing
-- has ever certified their geometry -- and is not GEOMETRY_CERTIFIED.
ALTER TABLE segment_surfaces ADD COLUMN IF NOT EXISTS geometry_qc_state text
  NOT NULL DEFAULT 'GEOMETRY_UNMEASURED';
CREATE INDEX IF NOT EXISTS segment_surfaces_by_sample
  ON segment_surfaces(sample_id, state);
CREATE INDEX IF NOT EXISTS segment_surfaces_by_geometry
  ON segment_surfaces(geometry_qc_state);

CREATE TABLE IF NOT EXISTS segment_tasks (
  task_id text PRIMARY KEY,
  mission_id text NOT NULL DEFAULT 'unfiled',
  source_snapshot_id text NOT NULL REFERENCES segment_source_snapshots(source_snapshot_id),
  cell_id text NOT NULL,
  grid_version text NOT NULL,
  policy_version text NOT NULL,
  bounds_xyz jsonb NOT NULL,
  center_xyz jsonb NOT NULL,
  priority double precision NOT NULL,
  parameter_envelope jsonb NOT NULL,
  catalog_snapshot_sha256 text NOT NULL,
  payload jsonb NOT NULL,
  state text NOT NULL DEFAULT 'PENDING',
  worker_id text,
  lease_token_hash text,
  lease_expires_at timestamptz,
  retry_after timestamptz,
  gpu_required boolean NOT NULL DEFAULT false,
  minimum_vram_gb double precision NOT NULL DEFAULT 0,
  seed_probe_required boolean NOT NULL DEFAULT false,
  active_attempt_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT segment_tasks_mission_identity
    UNIQUE(mission_id, source_snapshot_id, grid_version, cell_id, policy_version)
);
CREATE INDEX IF NOT EXISTS segment_tasks_ready
  ON segment_tasks(state, priority DESC, task_id);
-- Who asked for this work. The ink queue has carried created_by since it was
-- written; segmentation -- the larger consumer of GPU time -- carried nothing,
-- so the platform could not say whose tasks were running, could not show one
-- person their own runs, and could not have shared the fleet fairly even if it
-- wanted to. 'unattributed' is what the backlog gets: honest about tasks that
-- predate the column rather than crediting them to whoever asks next.
ALTER TABLE segment_tasks ADD COLUMN IF NOT EXISTS created_by text NOT NULL DEFAULT 'unattributed';
ALTER TABLE segment_tasks ADD COLUMN IF NOT EXISTS gpu_required boolean NOT NULL DEFAULT false;
ALTER TABLE segment_tasks ADD COLUMN IF NOT EXISTS minimum_vram_gb double precision NOT NULL DEFAULT 0;
ALTER TABLE segment_tasks ADD COLUMN IF NOT EXISTS seed_probe_required boolean
  NOT NULL DEFAULT false;
ALTER TABLE segment_tasks ADD COLUMN IF NOT EXISTS mission_id text
  NOT NULL DEFAULT 'unfiled';
-- The original identity treated one cell on one scroll as a global singleton.
-- That made a second mission silently reuse the first mission's completed task.
DO $$
DECLARE old_constraint text;
BEGIN
  SELECT c.conname INTO old_constraint
    FROM pg_constraint c
   WHERE c.conrelid = 'segment_tasks'::regclass
     AND c.contype = 'u'
     AND pg_get_constraintdef(c.oid) =
         'UNIQUE (source_snapshot_id, grid_version, cell_id, policy_version)';
  IF old_constraint IS NOT NULL THEN
    EXECUTE format('ALTER TABLE segment_tasks DROP CONSTRAINT %I', old_constraint);
  END IF;
END $$;
ALTER TABLE segment_tasks DROP CONSTRAINT IF EXISTS segment_tasks_mission_identity;
ALTER TABLE segment_tasks ADD CONSTRAINT segment_tasks_mission_identity
  UNIQUE(mission_id, source_snapshot_id, grid_version, cell_id, policy_version);
CREATE INDEX IF NOT EXISTS segment_tasks_by_mission
  ON segment_tasks(mission_id, created_at DESC);

CREATE TABLE IF NOT EXISTS segment_worker_capabilities (
  worker_id text PRIMARY KEY,
  capabilities jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_attempts (
  attempt_id text PRIMARY KEY,
  task_id text NOT NULL REFERENCES segment_tasks(task_id),
  attempt_number integer NOT NULL,
  worker_id text NOT NULL,
  state text NOT NULL,
  proposal jsonb,
  proposal_sha256 text,
  locked_plan jsonb,
  locked_plan_sha256 text,
  result jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(task_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS segment_artifact_sets (
  artifact_set_id text PRIMARY KEY,
  attempt_id text NOT NULL UNIQUE REFERENCES segment_attempts(attempt_id),
  manifest jsonb NOT NULL,
  manifest_sha256 text NOT NULL,
  staging_uri text NOT NULL,
  state text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_qc_jobs (
  qc_job_id text PRIMARY KEY,
  surface_id text NOT NULL REFERENCES segment_surfaces(surface_id),
  profile_id text NOT NULL,
  state text NOT NULL,
  payload jsonb NOT NULL,
  worker_id text,
  lease_token_hash text,
  lease_expires_at timestamptz,
  retry_after timestamptz,
  result jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(surface_id, profile_id)
);
ALTER TABLE segment_qc_jobs ADD COLUMN IF NOT EXISTS worker_id text;
ALTER TABLE segment_qc_jobs ADD COLUMN IF NOT EXISTS lease_token_hash text;
ALTER TABLE segment_qc_jobs ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE segment_qc_jobs ADD COLUMN IF NOT EXISTS retry_after timestamptz;
ALTER TABLE segment_qc_jobs ADD COLUMN IF NOT EXISTS result jsonb;
ALTER TABLE segment_qc_jobs ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
UPDATE segment_qc_jobs SET updated_at=created_at WHERE updated_at IS NULL;
ALTER TABLE segment_qc_jobs ALTER COLUMN updated_at SET NOT NULL;
CREATE INDEX IF NOT EXISTS segment_qc_jobs_ready
  ON segment_qc_jobs(state, retry_after, created_at, qc_job_id);

CREATE TABLE IF NOT EXISTS segment_events (
  event_id bigserial PRIMARY KEY,
  task_id text,
  attempt_id text,
  event_type text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS segment_events_by_task
  ON segment_events(task_id, event_id);

-- Noncanonical closed-loop seed evidence. A probe is never a catalogue surface
-- and never receives a downstream QC job; only its later full continuation can
-- cross that boundary through the ordinary finalizer.
CREATE TABLE IF NOT EXISTS segment_probe_runs (
  probe_run_id text PRIMARY KEY,
  task_id text NOT NULL REFERENCES segment_tasks(task_id),
  created_by_attempt_id text NOT NULL REFERENCES segment_attempts(attempt_id),
  source_snapshot_id text NOT NULL
    REFERENCES segment_source_snapshots(source_snapshot_id),
  candidate_set jsonb NOT NULL,
  candidate_set_sha256 text NOT NULL,
  policy_id text NOT NULL,
  policy jsonb NOT NULL,
  policy_sha256 text NOT NULL,
  executor_fingerprint jsonb NOT NULL,
  executor_fingerprint_sha256 text NOT NULL,
  state text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(task_id,candidate_set_sha256,policy_sha256,
         executor_fingerprint_sha256)
);
CREATE INDEX IF NOT EXISTS segment_probe_runs_by_task
  ON segment_probe_runs(task_id,state,updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS segment_probe_runs_one_per_task
  ON segment_probe_runs(task_id);

CREATE TABLE IF NOT EXISTS segment_probe_trials (
  probe_trial_id text PRIMARY KEY,
  probe_run_id text NOT NULL REFERENCES segment_probe_runs(probe_run_id),
  candidate_id text NOT NULL,
  candidate_rank integer NOT NULL,
  candidate jsonb NOT NULL,
  locked_plan jsonb,
  locked_plan_sha256 text,
  state text NOT NULL,
  result jsonb,
  worker_id text,
  lease_token_hash text,
  lease_expires_at timestamptz,
  retry_after timestamptz,
  gpu_required boolean NOT NULL DEFAULT false,
  minimum_vram_gb double precision NOT NULL DEFAULT 0,
  active_probe_attempt_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(probe_run_id,candidate_id),
  UNIQUE(probe_run_id,candidate_rank),
  UNIQUE(probe_run_id,probe_trial_id)
);
CREATE INDEX IF NOT EXISTS segment_probe_trials_ready
  ON segment_probe_trials(probe_run_id,state,retry_after,candidate_rank);

CREATE TABLE IF NOT EXISTS segment_probe_attempts (
  probe_attempt_id text PRIMARY KEY,
  probe_trial_id text NOT NULL
    REFERENCES segment_probe_trials(probe_trial_id),
  attempt_number integer NOT NULL,
  worker_id text NOT NULL,
  state text NOT NULL,
  growth_receipt jsonb,
  result jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(probe_trial_id,attempt_number)
);

CREATE TABLE IF NOT EXISTS segment_probe_artifact_sets (
  probe_artifact_set_id text PRIMARY KEY,
  probe_trial_id text NOT NULL UNIQUE
    REFERENCES segment_probe_trials(probe_trial_id),
  probe_attempt_id text NOT NULL UNIQUE
    REFERENCES segment_probe_attempts(probe_attempt_id),
  manifest jsonb NOT NULL,
  manifest_sha256 text NOT NULL,
  artifact_uri text,
  state text NOT NULL,
  retain_until timestamptz,
  deleted_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_probe_evaluations (
  evaluation_id text PRIMARY KEY,
  probe_trial_id text NOT NULL UNIQUE
    REFERENCES segment_probe_trials(probe_trial_id),
  probe_artifact_set_id text NOT NULL UNIQUE
    REFERENCES segment_probe_artifact_sets(probe_artifact_set_id),
  profile_id text NOT NULL,
  profile_sha256 text NOT NULL,
  verdict text NOT NULL,
  result jsonb NOT NULL,
  result_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_probe_decisions (
  decision_id text PRIMARY KEY,
  probe_run_id text NOT NULL UNIQUE
    REFERENCES segment_probe_runs(probe_run_id),
  policy_id text NOT NULL,
  policy_sha256 text NOT NULL,
  evidence_set_sha256 text NOT NULL,
  action text NOT NULL,
  winner_trial_id text,
  receipt jsonb NOT NULL,
  receipt_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY(probe_run_id,winner_trial_id)
    REFERENCES segment_probe_trials(probe_run_id,probe_trial_id)
);

CREATE TABLE IF NOT EXISTS segment_probe_promotions (
  promotion_id text PRIMARY KEY,
  decision_id text NOT NULL UNIQUE
    REFERENCES segment_probe_decisions(decision_id),
  winner_trial_id text NOT NULL
    REFERENCES segment_probe_trials(probe_trial_id),
  winner_probe_artifact_set_id text NOT NULL
    REFERENCES segment_probe_artifact_sets(probe_artifact_set_id),
  continuation_task_id text UNIQUE REFERENCES segment_tasks(task_id),
  continuation_attempt_id text NOT NULL
    REFERENCES segment_attempts(attempt_id),
  continuation_contract_sha256 text NOT NULL,
  continuation_locked_plan_sha256 text NOT NULL,
  canonical_artifact_set_id text
    REFERENCES segment_artifact_sets(artifact_set_id),
  surface_id text REFERENCES segment_surfaces(surface_id),
  state text NOT NULL,
  receipt jsonb NOT NULL,
  receipt_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Upgrade databases initialized while seed-probe v1 was still in shadow
-- development.  CREATE TABLE IF NOT EXISTS cannot add these bindings to an
-- existing v6/v7 table, so every column is explicit and the migration version
-- below prevents initialize() from returning before this block has run.
ALTER TABLE segment_probe_promotions
  ADD COLUMN IF NOT EXISTS continuation_attempt_id text
    REFERENCES segment_attempts(attempt_id);
ALTER TABLE segment_probe_promotions
  ADD COLUMN IF NOT EXISTS continuation_contract_sha256 text;
ALTER TABLE segment_probe_promotions
  ADD COLUMN IF NOT EXISTS continuation_locked_plan_sha256 text;
DO $seed_probe_v8$
BEGIN
  IF EXISTS (
    SELECT 1 FROM segment_probe_promotions
     WHERE continuation_attempt_id IS NULL
        OR continuation_contract_sha256 IS NULL
        OR continuation_locked_plan_sha256 IS NULL
  ) THEN
    RAISE EXCEPTION
      'seed-probe v8 cannot auto-bind legacy promotion rows; explicit operator evidence review is required';
  END IF;
END
$seed_probe_v8$;
ALTER TABLE segment_probe_promotions
  ALTER COLUMN continuation_attempt_id SET NOT NULL,
  ALTER COLUMN continuation_contract_sha256 SET NOT NULL,
  ALTER COLUMN continuation_locked_plan_sha256 SET NOT NULL;

-- P3. One flattening per surface per profile: re-running the same profile over
-- the same surface is a no-op, and a different profile is a different
-- experiment over the same ground -- the identity discipline segment_tasks
-- already uses, for the same reason.
CREATE TABLE IF NOT EXISTS surface_flattenings (
  flattening_id text PRIMARY KEY,
  surface_id text NOT NULL REFERENCES segment_surfaces(surface_id),
  profile_id text NOT NULL,
  state text NOT NULL,
  artifact_uri text,
  artifact_sha256 text,
  area_ratio double precision,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(surface_id, profile_id)
);
CREATE INDEX IF NOT EXISTS surface_flattenings_by_surface
  ON surface_flattenings(surface_id, profile_id);

INSERT INTO segment_schema_migrations(version, description)
VALUES (1, 'complete multi-host segmentation control plane')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (2, 'resource-aware GPU admission and larger-GPU retry')
ON CONFLICT(version) DO NOTHING;

-- The geometry column was added to this file above, but the version was not
-- bumped with it. initialize() returns early once the highest version it finds
-- here is recorded, so on every database created before that column was added
-- the ALTER never ran -- while the finalizer's INSERT names the column. The
-- next surface to finish would have failed on UndefinedColumn, in the one code
-- path that only runs after a segmentation succeeds.
INSERT INTO segment_schema_migrations(version, description)
VALUES (3, 'geometry certification state on surfaces')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (4, 'flattened sheets produced by P3')
ON CONFLICT(version) DO NOTHING;

-- Credentials a worker needs, held where the workers already look.
--
-- They lived in a file on one host's tmpfs: lost on reboot, absent on every
-- other machine, and placed by hand each time. A worker is ephemeral and must
-- be able to start from nothing but a database URL, so what it needs to reach
-- object storage belongs in the control plane with everything else.
--
-- The value is stored as written. This table is therefore as sensitive as the
-- credentials in it: anything that can read this database can read them, which
-- is already true of the database password every worker carries. It is not
-- encryption at rest and does not pretend to be -- a key to decrypt with would
-- have the same distribution problem this table exists to solve.
CREATE TABLE IF NOT EXISTS fleet_secrets (
  name text PRIMARY KEY,
  value text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  updated_by text NOT NULL
);
REVOKE ALL ON fleet_secrets FROM PUBLIC;

INSERT INTO segment_schema_migrations(version, description)
VALUES (5, 'fleet credentials in the control plane')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (6, 'noncanonical closed-loop seed probe ledger')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (7, 'one immutable seed probe budget per parent task')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (8, 'bind probe promotion to the exact winner continuation plan')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (9, 'enforce non-null probe promotion bindings and reject legacy rows')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (10, 'record who asked for each segmentation task')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (11, 'mission-bound segmentation task identity')
ON CONFLICT(version) DO NOTHING;

COMMIT;
