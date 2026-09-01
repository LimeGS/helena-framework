-- Somebody asked for this job to stop after it had already started.
--
-- `cancel` refused anything running, so the only way to stop a twenty-five
-- minute render that was already wrong was to kill the worker's container --
-- taking the worker down with the job and leaving the record saying it died
-- rather than that somebody stopped it.
--
-- A worker cannot be interrupted from outside, so this is a request rather than
-- an act: the worker reads it on the heartbeat it already sends, and stops.
ALTER TABLE ink_jobs ADD COLUMN IF NOT EXISTS cancel_requested TIMESTAMPTZ;
COMMENT ON COLUMN ink_jobs.cancel_requested IS
  'when a stop was asked for; the worker holding the lease acts on it and the '
  'job ends cancelled rather than failed';
