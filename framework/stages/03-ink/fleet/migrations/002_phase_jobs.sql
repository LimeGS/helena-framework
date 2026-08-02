-- Generalise the ink queue into a phase queue.
--
-- The lease discipline is unchanged -- FOR UPDATE SKIP LOCKED, hashed token,
-- attempt counting, event log. What changes is that a job now names the phase
-- it belongs to and the component that implements it, so one queue serves P1,
-- P4, P5 and P8 instead of each growing its own.

ALTER TABLE ink_jobs ADD COLUMN IF NOT EXISTS phase TEXT NOT NULL DEFAULT 'P5';
ALTER TABLE ink_jobs ADD COLUMN IF NOT EXISTS component TEXT;

-- profile_id is meaningful for P5 and empty elsewhere, so it stops being
-- required. The phase plus the component is what identifies the work now.
ALTER TABLE ink_jobs ALTER COLUMN profile_id DROP NOT NULL;

CREATE INDEX IF NOT EXISTS ink_jobs_phase ON ink_jobs (phase, state);

COMMENT ON COLUMN ink_jobs.phase IS
  'P0..P9, the phase vocabulary in framework/contracts/pipeline_phases.json';
COMMENT ON COLUMN ink_jobs.component IS
  'which implementation runs it, from phase-implementations-0.1.0.json';
