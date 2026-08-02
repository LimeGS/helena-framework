-- A job belongs to a mission. Missions are directories on disk with their own
-- manifest; this column is the index, not the truth.
ALTER TABLE ink_jobs ADD COLUMN IF NOT EXISTS mission_id TEXT;
CREATE INDEX IF NOT EXISTS ink_jobs_mission ON ink_jobs (mission_id, state);
COMMENT ON COLUMN ink_jobs.mission_id IS
  'the mission directory under CX_RUNS whose MISSION.json is authoritative';
