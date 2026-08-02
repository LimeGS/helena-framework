-- One derived surface can consume several immutable parent surfaces.  The
-- mission selection table intentionally chooses one artifact per phase/sample,
-- so it cannot express stitching fan-in; this table is the scientific lineage
-- owned by the job that actually performed the N -> 1 operation.

CREATE TABLE IF NOT EXISTS surface_derivations (
    child_surface_id       TEXT NOT NULL,
    parent_surface_id      TEXT NOT NULL,
    parent_artifact_sha256 TEXT NOT NULL,
    ordinal                INTEGER NOT NULL,
    relationship           TEXT NOT NULL,
    job_id                 TEXT NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (child_surface_id, parent_surface_id),
    UNIQUE (child_surface_id, ordinal),
    FOREIGN KEY (child_surface_id) REFERENCES segment_surfaces(surface_id),
    FOREIGN KEY (parent_surface_id) REFERENCES segment_surfaces(surface_id),
    FOREIGN KEY (job_id) REFERENCES ink_jobs(job_id)
);

CREATE INDEX IF NOT EXISTS surface_derivations_parent
    ON surface_derivations(parent_surface_id, child_surface_id);
