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

CREATE TABLE IF NOT EXISTS segment_campaign_budget_admissions (
  mission_id text NOT NULL,
  sample_id text NOT NULL,
  receipt_sha256 text NOT NULL,
  admission jsonb NOT NULL,
  admission_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(mission_id, sample_id, receipt_sha256)
);

CREATE TABLE IF NOT EXISTS segment_campaign_decisions (
  receipt_sha256 text PRIMARY KEY,
  mission_id text NOT NULL,
  policy_version text NOT NULL,
  evaluation_kind text NOT NULL,
  evaluation_index integer NOT NULL,
  decision text NOT NULL,
  receipt jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id, policy_version, evaluation_kind, evaluation_index)
);
CREATE INDEX IF NOT EXISTS segment_campaign_decisions_by_scope
  ON segment_campaign_decisions(
    mission_id, policy_version, evaluation_index
  );

CREATE TABLE IF NOT EXISTS segment_campaign_resume_authorizations (
  authorization_sha256 text PRIMARY KEY,
  mission_id text NOT NULL,
  sample_id text NOT NULL,
  prior_policy_version text NOT NULL,
  new_policy_version text NOT NULL,
  new_admission_sha256 text NOT NULL,
  "authorization" jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id, sample_id, new_admission_sha256)
);
CREATE UNIQUE INDEX IF NOT EXISTS segment_campaign_resume_by_predecessor
  ON segment_campaign_resume_authorizations(mission_id, prior_policy_version);

CREATE TABLE IF NOT EXISTS segment_campaign_resume_principal_attestations (
  authorization_sha256 text PRIMARY KEY,
  mission_id text NOT NULL,
  principal text NOT NULL,
  "authorization" jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

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

CREATE TABLE IF NOT EXISTS human_review_events (
  review_event_id text PRIMARY KEY,
  p7_job_id text NOT NULL,
  intent text NOT NULL,
  mission_id text NOT NULL,
  sample_id text NOT NULL,
  surface_id text NOT NULL,
  verdict_sha256 text NOT NULL,
  card_sha256 text NOT NULL,
  config_sha256 text NOT NULL,
  vetting_packet_sha256 text NOT NULL,
  author text NOT NULL,
  event jsonb NOT NULL,
  event_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL,
  UNIQUE(p7_job_id, intent)
);
CREATE INDEX IF NOT EXISTS human_reviews_by_p7
  ON human_review_events(p7_job_id, created_at, review_event_id);

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

-- First-letters discovery compute is one common, fail-closed ledger for the
-- baseline arm, alternative-source arm, and any later adaptive children.  A
-- reservation and its dispatch binding are written in the same transaction;
-- neither table is sufficient authority on its own.
CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_compute_caps (
  mission_id text PRIMARY KEY,
  cap_authority_id text NOT NULL,
  authority_sha256 text NOT NULL UNIQUE,
  cap_units bigint NOT NULL CHECK(cap_units >= 0),
  authority jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_compute_reservations (
  reservation_id text PRIMARY KEY,
  mission_id text NOT NULL
    REFERENCES segment_first_letters_discovery_compute_caps(mission_id),
  request_id text NOT NULL,
  work_kind text NOT NULL CHECK(work_kind IN
    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM','ADAPTIVE_CHILD')),
  work_authority_id text NOT NULL,
  work_authority_sha256 text NOT NULL,
  ordered_item_ids_sha256 text NOT NULL,
  item_count integer NOT NULL CHECK(item_count > 0),
  units_per_item integer NOT NULL CHECK(units_per_item = 24),
  reserved_units bigint NOT NULL CHECK(reserved_units > 0),
  reserved_before_units bigint NOT NULL CHECK(reserved_before_units >= 0),
  reserved_after_units bigint NOT NULL
    CHECK(reserved_after_units >= reserved_before_units),
  source text NOT NULL CHECK(source IN
    ('RESERVED_BEFORE_EXECUTION','IMPORTED_HISTORICAL_EXACT')),
  reservation jsonb NOT NULL,
  reservation_sha256 text NOT NULL UNIQUE,
  request_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id,request_id),
  UNIQUE(mission_id,work_kind,work_authority_sha256)
);
CREATE INDEX IF NOT EXISTS segment_first_letters_discovery_compute_by_mission
  ON segment_first_letters_discovery_compute_reservations(
    mission_id,created_at,reservation_id
  );

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_work_bindings (
  reservation_id text PRIMARY KEY
    REFERENCES segment_first_letters_discovery_compute_reservations(
      reservation_id
    ),
  mission_id text NOT NULL,
  request_id text NOT NULL,
  work_kind text NOT NULL,
  dispatch_kind text NOT NULL,
  work jsonb NOT NULL,
  work_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- This is the noncanonical discovery job/claim and exact evidence seam.  It
-- deliberately does not require a canonical segmentation task or attempt;
-- optional parent IDs are lineage only.
CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_evidence_runs (
  run_id text PRIMARY KEY,
  reservation_id text NOT NULL
    REFERENCES segment_first_letters_discovery_compute_reservations(
      reservation_id
    ),
  mission_id text NOT NULL,
  request_id text NOT NULL,
  parent_task_id text REFERENCES segment_tasks(task_id),
  parent_attempt_id text REFERENCES segment_attempts(attempt_id),
  worker_id text NOT NULL,
  cell_id text NOT NULL,
  source_snapshot_id text NOT NULL REFERENCES segment_source_snapshots(
    source_snapshot_id
  ),
  run_token_sha256 text NOT NULL UNIQUE,
  lease_expires_at timestamptz NOT NULL,
  profile_bytes bytea NOT NULL,
  profile_file_sha256 text NOT NULL,
  provider_request jsonb NOT NULL,
  run_authority jsonb NOT NULL,
  run_authority_sha256 text NOT NULL UNIQUE,
  state text NOT NULL CHECK(state IN ('CLAIMED','COMPLETED')),
  created_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  UNIQUE(reservation_id,cell_id)
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_executor_registry (
  worker_id text PRIMARY KEY,
  executor_id text NOT NULL,
  executor_sha256 text NOT NULL,
  capabilities jsonb NOT NULL,
  registration jsonb NOT NULL,
  registration_sha256 text NOT NULL UNIQUE,
  enabled boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(worker_id,executor_id)
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_executor_claims (
  claim_id text PRIMARY KEY,
  run_id text NOT NULL UNIQUE
    REFERENCES segment_first_letters_discovery_evidence_runs(run_id),
  worker_id text NOT NULL
    REFERENCES segment_first_letters_discovery_executor_registry(worker_id),
  executor_id text NOT NULL,
  executor_sha256 text NOT NULL,
  capability text NOT NULL,
  claim_attempt_number integer NOT NULL CHECK(claim_attempt_number > 0),
  execution_lease_token_sha256 text NOT NULL UNIQUE,
  lease_expires_at timestamptz NOT NULL,
  claim jsonb NOT NULL,
  claim_sha256 text NOT NULL UNIQUE,
  state text NOT NULL CHECK(state IN ('CLAIMED','COMPLETED')),
  created_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_evidence_sets (
  evidence_set_id text PRIMARY KEY,
  run_id text NOT NULL UNIQUE
    REFERENCES segment_first_letters_discovery_evidence_runs(run_id),
  evidence jsonb NOT NULL,
  evidence_set_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_evidence_files (
  evidence_set_id text NOT NULL
    REFERENCES segment_first_letters_discovery_evidence_sets(evidence_set_id),
  file_order integer NOT NULL CHECK(file_order >= 0),
  relative_path text NOT NULL,
  role text NOT NULL,
  payload bytea NOT NULL,
  byte_count bigint NOT NULL CHECK(byte_count >= 0),
  sha256 text NOT NULL,
  PRIMARY KEY(evidence_set_id,relative_path),
  UNIQUE(evidence_set_id,file_order)
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_compute_outcomes (
  mission_id text NOT NULL,
  request_id text NOT NULL,
  outcome text NOT NULL CHECK(outcome IN ('CANCELLED','FAILED','ABSTAINED')),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(mission_id,request_id,outcome)
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_compute_blocks (
  mission_id text PRIMARY KEY,
  reason text NOT NULL,
  evidence jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Promotion is a three-fact transaction: immutable authority, one fresh
-- ordinary child, and the parent task/attempt terminalized as promoted.
CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_promotions (
  promotion_id text PRIMARY KEY,
  mission_id text NOT NULL,
  request_id text NOT NULL,
  scientific_opportunity_id text NOT NULL,
  parent_task_id text NOT NULL REFERENCES segment_tasks(task_id),
  child_task_id text NOT NULL UNIQUE
    REFERENCES segment_tasks(task_id) DEFERRABLE INITIALLY DEFERRED,
  admission_sha256 text NOT NULL,
  authority jsonb NOT NULL,
  authority_sha256 text NOT NULL UNIQUE,
  request_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id,request_id),
  UNIQUE(mission_id,scientific_opportunity_id)
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_promotion_attempt_bindings (
  promotion_id text NOT NULL
    REFERENCES segment_first_letters_discovery_promotions(promotion_id),
  attempt_number integer NOT NULL CHECK(attempt_number > 0),
  attempt_id text NOT NULL UNIQUE REFERENCES segment_attempts(attempt_id),
  binding jsonb NOT NULL,
  binding_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(promotion_id,attempt_number)
);

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

INSERT INTO segment_schema_migrations(version, description)
VALUES (12, 'append-only exact-job human review events')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (13, 'immutable controlled campaign budget admissions')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (14, 'immutable controlled campaign starvation decisions')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (15, 'authenticated campaign resume principal attestations')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (16, 'first-letters discovery compute, evidence, and promotion authority')
ON CONFLICT(version) DO NOTHING;

INSERT INTO segment_schema_migrations(version, description)
VALUES (17, 'persisted first-letters discovery executor registry and claims')
ON CONFLICT(version) DO NOTHING;

-- V18 is additive over the immutable v17 claim registry.  A discovery job is
-- visibly RUNNING before any provider execution; only its exact active owner
-- may heartbeat it, and every post-RUNNING uncertainty closes permanently.
ALTER TABLE segment_first_letters_discovery_evidence_runs
  ADD COLUMN IF NOT EXISTS started_at timestamptz,
  ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz,
  ADD COLUMN IF NOT EXISTS incomplete_at timestamptz,
  ADD COLUMN IF NOT EXISTS incomplete_reason text;

ALTER TABLE segment_first_letters_discovery_executor_claims
  ADD COLUMN IF NOT EXISTS started_at timestamptz,
  ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz,
  ADD COLUMN IF NOT EXISTS incomplete_at timestamptz,
  ADD COLUMN IF NOT EXISTS incomplete_reason text;

ALTER TABLE segment_first_letters_discovery_evidence_runs
  DROP CONSTRAINT IF EXISTS
    segment_first_letters_discovery_evidence_runs_state_check;
ALTER TABLE segment_first_letters_discovery_evidence_runs
  ADD CONSTRAINT segment_first_letters_discovery_evidence_runs_state_check
  CHECK(state IN ('CLAIMED','RUNNING','COMPLETED','CONTROL_INCOMPLETE'));

ALTER TABLE segment_first_letters_discovery_executor_claims
  DROP CONSTRAINT IF EXISTS
    segment_first_letters_discovery_executor_claims_state_check;
ALTER TABLE segment_first_letters_discovery_executor_claims
  ADD CONSTRAINT segment_first_letters_discovery_executor_claims_state_check
  CHECK(state IN ('CLAIMED','RUNNING','COMPLETED','CONTROL_INCOMPLETE'));

INSERT INTO segment_schema_migrations(version, description)
VALUES (18, 'running discovery claims, heartbeats, and permanent incomplete state')
ON CONFLICT(version) DO NOTHING;

-- V19 adds the immutable, executable shadow graph. A live v16 reservation is
-- executable only when its adapter, dispatch, and complete ordered job set are
-- committed in the same transaction.
CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_history_reconciliations_v19 (
  reconciliation_seq bigint GENERATED ALWAYS AS IDENTITY,
  reconciliation_id text PRIMARY KEY,
  mission_id text NOT NULL,
  state text NOT NULL CHECK(state IN ('COMPLETE','CONTROL_INCOMPLETE')),
  watermark_sha256 text NOT NULL,
  manifest jsonb NOT NULL,
  manifest_sha256 text NOT NULL,
  fixed_units bigint NOT NULL CHECK(fixed_units >= 0),
  reason text,
  reconciliation jsonb NOT NULL,
  reconciliation_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id,manifest_sha256,state)
);
CREATE INDEX IF NOT EXISTS segment_first_letters_discovery_history_by_mission_v19
  ON segment_first_letters_discovery_history_reconciliations_v19(
    mission_id,created_at,reconciliation_seq
  );

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_historical_imports_v19 (
  import_id text PRIMARY KEY,
  reservation_id text NOT NULL
    REFERENCES segment_first_letters_discovery_compute_reservations(reservation_id),
  mission_id text NOT NULL,
  logical_execution_id text NOT NULL,
  producer_kind text NOT NULL,
  source_snapshot_sha256 text NOT NULL,
  profile_file_sha256 text NOT NULL,
  item_id text NOT NULL,
  fixed_units integer NOT NULL CHECK(fixed_units = 24),
  retained_row_ids jsonb NOT NULL,
  retained_projection_sha256 text NOT NULL,
  history_manifest_sha256 text NOT NULL,
  import_binding jsonb NOT NULL,
  import_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id,logical_execution_id)
);
DO $$
DECLARE legacy_constraint text;
BEGIN
  FOR legacy_constraint IN
    SELECT conname
      FROM pg_constraint
     WHERE conrelid =
       'segment_first_letters_discovery_historical_imports_v19'::regclass
       AND contype = 'u'
       AND pg_get_constraintdef(oid) = 'UNIQUE (reservation_id)'
  LOOP
    EXECUTE format(
      'ALTER TABLE segment_first_letters_discovery_historical_imports_v19 '
      'DROP CONSTRAINT %I', legacy_constraint
    );
  END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_native_adapters_v19 (
  reservation_id text PRIMARY KEY
    REFERENCES segment_first_letters_discovery_compute_reservations(reservation_id),
  mission_id text NOT NULL,
  request_id text NOT NULL,
  work_kind text NOT NULL CHECK(work_kind IN
    ('BASELINE_ARM','ALTERNATIVE_SOURCE_ARM')),
  producer_kind text NOT NULL CHECK(producer_kind IN
    ('BASELINE_RECONCILIATION','EXPERIMENTAL_ARM_ADMISSION')),
  native_schema text NOT NULL,
  native_authority jsonb NOT NULL,
  native_authority_sha256 text NOT NULL UNIQUE,
  generic_work_authority jsonb NOT NULL,
  generic_work_authority_sha256 text NOT NULL,
  profile_bytes bytea NOT NULL,
  adapter jsonb NOT NULL,
  adapter_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id,request_id)
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_dispatches_v19 (
  dispatch_id text PRIMARY KEY,
  reservation_id text NOT NULL UNIQUE
    REFERENCES segment_first_letters_discovery_compute_reservations(reservation_id),
  mission_id text NOT NULL,
  request_id text NOT NULL,
  work_kind text NOT NULL,
  adapter_sha256 text NOT NULL,
  profile_file_sha256 text NOT NULL,
  source_snapshot_sha256 text NOT NULL,
  ordered_item_ids_sha256 text NOT NULL,
  item_count integer NOT NULL CHECK(item_count > 0),
  dispatch jsonb NOT NULL,
  dispatch_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segment_first_letters_discovery_jobs_v19 (
  job_id text PRIMARY KEY,
  dispatch_id text NOT NULL
    REFERENCES segment_first_letters_discovery_dispatches_v19(dispatch_id),
  reservation_id text NOT NULL
    REFERENCES segment_first_letters_discovery_compute_reservations(reservation_id),
  item_order integer NOT NULL CHECK(item_order >= 0),
  item_id text NOT NULL,
  work_item_binding_sha256 text NOT NULL,
  profile_file_sha256 text NOT NULL,
  source_snapshot_sha256 text NOT NULL,
  job jsonb NOT NULL,
  job_sha256 text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(dispatch_id,item_order),
  UNIQUE(reservation_id,item_id)
);
CREATE INDEX IF NOT EXISTS segment_first_letters_discovery_jobs_ready_v19
  ON segment_first_letters_discovery_jobs_v19(
    reservation_id,item_order,job_id
  );

INSERT INTO segment_schema_migrations(version, description)
VALUES (19, 'first-letters native shadow execution bridge')
ON CONFLICT(version) DO NOTHING;

-- V20. One routing decision per surface, written in the transaction that
-- creates the surface. This is the PostgreSQL half of the SQLite
-- `surface_routing_receipts` table; the deployment runs PostgreSQL, so a gate
-- that existed only in SQLite was a gate nothing in production ever passed
-- through.
--
-- The receipt is the evidence that a surface below the effort floor was
-- classified rather than discarded, so it is immutable. SQLite refuses UPDATE
-- and DELETE with triggers; these do the same, rather than trusting every
-- future caller to leave the row alone.
--
-- `receipt` is jsonb here and `receipt_json` text in SQLite, matching how each
-- store already keeps documents. The names differ by store idiom on purpose;
-- the document they hold is byte-identical.
CREATE TABLE IF NOT EXISTS segment_surface_routing_receipts (
  surface_id text PRIMARY KEY REFERENCES segment_surfaces(surface_id),
  route text NOT NULL,
  measured_area_cm2 double precision NOT NULL,
  minimum_area_cm2 double precision NOT NULL,
  policy_version text NOT NULL,
  profile_id text NOT NULL,
  receipt_sha256 text NOT NULL,
  receipt jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION segment_surface_routing_receipt_is_evidence()
RETURNS trigger AS $segment_surface_routing_receipt_is_evidence$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'a surface routing receipt is permanent';
  END IF;
  RAISE EXCEPTION 'a surface routing receipt is immutable';
END;
$segment_surface_routing_receipt_is_evidence$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS segment_surface_routing_receipts_are_immutable
  ON segment_surface_routing_receipts;
CREATE TRIGGER segment_surface_routing_receipts_are_immutable
  BEFORE UPDATE ON segment_surface_routing_receipts
  FOR EACH ROW
  EXECUTE PROCEDURE segment_surface_routing_receipt_is_evidence();

DROP TRIGGER IF EXISTS segment_surface_routing_receipts_are_permanent
  ON segment_surface_routing_receipts;
CREATE TRIGGER segment_surface_routing_receipts_are_permanent
  BEFORE DELETE ON segment_surface_routing_receipts
  FOR EACH ROW
  EXECUTE PROCEDURE segment_surface_routing_receipt_is_evidence();

INSERT INTO segment_schema_migrations(version, description)
VALUES (20, 'immutable surface routing receipts')
ON CONFLICT(version) DO NOTHING;

-- V21. The only way out of the diagnostic path: which diagnostic surface a new
-- surface continues, resolved against a locked catalogue inside the transaction
-- that creates the successor. The uniqueness constraint is the contract -- one
-- expansion of a predecessor per policy version -- and the primary key is the
-- successor, because an authority permits making a new surface and never
-- editing an old one.
CREATE TABLE IF NOT EXISTS segment_surface_expansion_authorities (
  successor_surface_id text PRIMARY KEY REFERENCES segment_surfaces(surface_id),
  expands_surface_id text NOT NULL REFERENCES segment_surfaces(surface_id),
  predecessor_route text NOT NULL,
  predecessor_receipt_sha256 text NOT NULL,
  prior_policy_version text,
  new_policy_version text,
  authority_sha256 text NOT NULL,
  authority jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(expands_surface_id, new_policy_version)
);
CREATE INDEX IF NOT EXISTS segment_surface_expansion_authorities_by_predecessor
  ON segment_surface_expansion_authorities(expands_surface_id);

CREATE OR REPLACE FUNCTION segment_surface_expansion_authority_is_evidence()
RETURNS trigger AS $segment_surface_expansion_authority_is_evidence$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'a surface expansion authority is permanent';
  END IF;
  RAISE EXCEPTION 'a surface expansion authority is immutable';
END;
$segment_surface_expansion_authority_is_evidence$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS segment_surface_expansion_authorities_are_immutable
  ON segment_surface_expansion_authorities;
CREATE TRIGGER segment_surface_expansion_authorities_are_immutable
  BEFORE UPDATE ON segment_surface_expansion_authorities
  FOR EACH ROW
  EXECUTE PROCEDURE segment_surface_expansion_authority_is_evidence();

DROP TRIGGER IF EXISTS segment_surface_expansion_authorities_are_permanent
  ON segment_surface_expansion_authorities;
CREATE TRIGGER segment_surface_expansion_authorities_are_permanent
  BEFORE DELETE ON segment_surface_expansion_authorities
  FOR EACH ROW
  EXECUTE PROCEDURE segment_surface_expansion_authority_is_evidence();

INSERT INTO segment_schema_migrations(version, description)
VALUES (21, 'immutable surface expansion authorities')
ON CONFLICT(version) DO NOTHING;

-- V22. The preflight is work, not a query: it asks M7 through a service that
-- lives where workers live. Enqueued where the source lock is checked, executed
-- where the sources are reachable. Same lifecycle as segment_qc_jobs.
--
-- It has a version of its own because it arrived after 21 shipped. In the base
-- block it was created on fresh databases only, and the deployment -- already at
-- 21, so replaying nothing -- restarted its preflight worker against
-- `relation "segment_preflight_jobs" does not exist`.
CREATE TABLE IF NOT EXISTS segment_preflight_jobs (
  preflight_job_id text PRIMARY KEY,
  mission_id text NOT NULL,
  sample_id text NOT NULL,
  source_snapshot_id text NOT NULL,
  state text NOT NULL CHECK (state IN ('PENDING','CLAIMED','COMPLETED','FAILED')),
  request jsonb NOT NULL,
  request_sha256 text NOT NULL,
  worker_id text,
  lease_token text,
  lease_expires_at timestamptz,
  attempts integer NOT NULL DEFAULT 0,
  receipt jsonb,
  reason_code text,
  -- Beside the code: FAILED alone sends an operator to a worker's stdout, and a
  -- reason that lives only there is not evidence.
  detail text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(mission_id, sample_id, request_sha256)
);
CREATE INDEX IF NOT EXISTS segment_preflight_jobs_pending
  ON segment_preflight_jobs(state, created_at, preflight_job_id);

INSERT INTO segment_schema_migrations(version, description)
VALUES (22, 'candidate preflight queue')
ON CONFLICT(version) DO NOTHING;

-- V23. One live job per request, with any number of failed attempts behind it.
--
-- The plain UNIQUE above made a failure permanent. The control enqueues a
-- frozen request, so its digest never changes, and the run after a transient
-- source outage was handed the FAILED row back as its answer -- reported as a
-- boundary failure with nothing measured. A terminal job is never claimed
-- again, so nothing could ever have cleared it.
--
-- The failed rows stay. An attempt is a record; it just stops being the answer.
DO $$
DECLARE legacy text;
BEGIN
  SELECT conname INTO legacy
    FROM pg_constraint
   WHERE conrelid = 'segment_preflight_jobs'::regclass
     AND contype = 'u'
     AND pg_get_constraintdef(oid) =
         'UNIQUE (mission_id, sample_id, request_sha256)';
  IF legacy IS NOT NULL THEN
    EXECUTE format('ALTER TABLE segment_preflight_jobs DROP CONSTRAINT %I', legacy);
  END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS segment_preflight_jobs_one_live_per_request
  ON segment_preflight_jobs(mission_id, sample_id, request_sha256)
  WHERE state <> 'FAILED';

INSERT INTO segment_schema_migrations(version, description)
VALUES (23, 'a failed preflight does not block the next attempt')
ON CONFLICT(version) DO NOTHING;

COMMIT;

BEGIN;

-- V24. An outage the worker calls recoverable sends the preflight back to the
-- queue instead of ending it.
--
-- The worker has always classified a source failure as recoverable -- "The
-- sources did not answer, or answered unusably. May recover." -- and then had
-- only fail_preflight to call, which is terminal. One dropped connection to S3
-- ended a measurement that had been running for seventy minutes, and left a
-- FAILED row for a human to notice and re-drive.
--
-- retry_after holds the job back so the next claim does not immediately re-read
-- a source that is still down; requeues is the bound. Both are needed: without
-- the delay this spins, and without the bound a source that is genuinely gone
-- hides behind an endless retry instead of surfacing as gone.
ALTER TABLE segment_preflight_jobs
  ADD COLUMN IF NOT EXISTS retry_after timestamptz;
ALTER TABLE segment_preflight_jobs
  ADD COLUMN IF NOT EXISTS requeues integer NOT NULL DEFAULT 0;

INSERT INTO segment_schema_migrations(version, description)
VALUES (24, 'a recoverable source outage requeues the preflight instead of ending it')
ON CONFLICT(version) DO NOTHING;

-- A third axis, orthogonal to the other two: does the CT resolve a single
-- lamina under this surface?
--
-- Geometry says the mesh is a plausible sheet and states in its own non-claims
-- that it is not a claim the segmentation followed the correct lamina. The
-- physical axis says scanned material is there. Neither reads the density
-- profile along the normal, which is the question that decides whether a render
-- is worth its 29 minutes: two air/papyrus interfaces, one sheet's thickness
-- apart.
--
-- Existing rows backfill to LAMINA_UNMEASURED, which is the truth -- the gate
-- has never run on them -- and is not a pass.
ALTER TABLE segment_surfaces ADD COLUMN IF NOT EXISTS lamina_qc_state text
  NOT NULL DEFAULT 'LAMINA_UNMEASURED';
CREATE INDEX IF NOT EXISTS segment_surfaces_by_lamina
  ON segment_surfaces(lamina_qc_state);

INSERT INTO segment_schema_migrations(version, description)
VALUES (25, 'the lamina axis: whether the CT resolves one sheet under a surface')
ON CONFLICT(version) DO NOTHING;

-- A fifth judgement, and the only one that is not about this surface at all.
--
-- The other four say something about the artifact: where it came from, whether
-- the scan supports it, whether the mesh is a plausible sheet, whether the CT
-- resolves one lamina under it. This one says how far a second run of the same
-- fit landed from it -- the only error bar this geometry has, because the
-- spiral fit publishes no uncertainty and its paper reports no run-to-run
-- variability.
--
-- Separate rather than folded into geometry, because it can contradict it and
-- the contradiction is the useful part: on PHerc0826 w015 the most tangled band
-- of the patch -- 830 fold-back intersections, real self-contact -- had the
-- *best* agreement between seeds of the three bands in it. Two runs converge on
-- the same wrong surface when the failure is in the data rather than the
-- optimization, so this measures reproducibility and not correctness, and a
-- schema that let it stand in for a geometry verdict would be lying.
--
-- Existing rows backfill to SEED_UNPAIRED, which is the truth: one run, and no
-- error bar. That is a different thing from a large one -- a large one can be
-- defended with its number beside it -- so it is a state and not a null.
ALTER TABLE segment_surfaces ADD COLUMN IF NOT EXISTS seed_agreement_state text
  NOT NULL DEFAULT 'SEED_UNPAIRED';
CREATE INDEX IF NOT EXISTS segment_surfaces_by_seed_agreement
  ON segment_surfaces(seed_agreement_state);

INSERT INTO segment_schema_migrations(version, description)
VALUES (26, 'the seed-agreement axis: how far a second run of the same fit landed')
ON CONFLICT(version) DO NOTHING;

COMMIT;
