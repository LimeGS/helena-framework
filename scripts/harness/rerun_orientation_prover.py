"""Run the orientation prover on the control's own inputs and print the real
ValueError. The route classifies it as GEOMETRY_INVALID and keeps only the
exception type, so the message exists nowhere else."""
import json, os, sys, traceback
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/scripts/harness")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
from panel import app as A
import numpy as np

MISSION, SAMPLE = "control-fl-pherc0139-dev-20260806", "PHerc0139"
JOB = "p3-e31e99be9d6d4b"

control = A.first_letters_control_policy()
orientation = (((control.get("checks") or {}).get("PIPELINE_CONTROL") or {})
               .get("orientation_parity") or {})
reference = (control.get("source_locks") or {}).get("community_surface") or {}

jobs = A.job_store()
job = jobs.job(JOB) or {}
row = next(r for r in (job.get("result") or {}).get("surfaces") or [])
surface = row["surface_id"]
print("surface:", surface)
print("source_artifact_sha256:", str(row.get("source_artifact_sha256"))[:16])

ref_xyz, _ = A.load_locked_orientation_reference(reference)
print("reference grid shape:", np.asarray(ref_xyz).shape)

try:
    v, f = A.load_hash_bound_grown_mesh(
        surface, str(row["source_artifact_sha256"]),
        expected_sample=SAMPLE, expected_binding=None)
    print("grown vertices:", np.asarray(v).shape, "faces:", np.asarray(f).shape)
    print("grown finite:", bool(np.isfinite(np.asarray(v)).all()))
except Exception:
    traceback.print_exc(); raise SystemExit(0)

from orientation_parity import prove_orientation, _grown_vertex_normals
try:
    _grown_vertex_normals(np.asarray(v, dtype=np.float64),
                          np.asarray(f, dtype=np.int64))
    print("grown normals: fine")
except Exception as e:
    print("GROWN NORMALS FAILED:", type(e).__name__, str(e)[:220])

try:
    from orientation_parity import triangulate_tifxyz_grid
    ref = triangulate_tifxyz_grid(ref_xyz)
    print("reference triangulated:", np.asarray(ref["faces"]).shape)
except Exception as e:
    print("REFERENCE TRIANGULATION FAILED:", type(e).__name__, str(e)[:220])
