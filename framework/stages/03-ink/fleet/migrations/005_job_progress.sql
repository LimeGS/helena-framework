-- The most recent thing a running job said, so the queue is not the only
-- observable fact about it.
--
-- A worker ran its adapter with the output buffered until the process exited,
-- which made a twenty-six-minute render observable as "started" and, much
-- later, "finished" -- nothing in between, anywhere. This column is written by
-- the heartbeat that already renews the lease, so a job reports where it is on
-- a write that was happening regardless.
--
-- Deliberately one line and not a log. The log belongs on the host, where it
-- survives a control plane nobody can reach; what belongs here is the answer
-- to "where is it now", which is a single fact that replaces itself.
ALTER TABLE ink_jobs ADD COLUMN IF NOT EXISTS progress JSONB;
COMMENT ON COLUMN ink_jobs.progress IS
  'the newest line this job wrote, as {line, source, at}; replaced on each '
  'heartbeat and cleared when an attempt starts';
