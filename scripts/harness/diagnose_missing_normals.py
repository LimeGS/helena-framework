"""Which vertices have no normal, and why. Two very different cases:
an isolated vertex no triangle uses carries no orientation at all, while a
vertex whose incident faces cancel is a genuine degeneracy."""
import sys
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
from panel import app as A
import numpy as np

JOB, SAMPLE = "p3-e31e99be9d6d4b", "PHerc0139"
jobs = A.job_store()
row = next(r for r in (jobs.job(JOB) or {}).get("result", {}).get("surfaces") or [])
v, f = A.load_hash_bound_grown_mesh(row["surface_id"], str(row["source_artifact_sha256"]),
                                    expected_sample=SAMPLE, expected_binding=None)
v = np.asarray(v, dtype=np.float64); f = np.asarray(f, dtype=np.int64)

tri = v[f]
crosses = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
normals = np.zeros_like(v)
for corner in range(3):
    np.add.at(normals, f[:, corner], crosses)
lengths = np.linalg.norm(normals, axis=1)

incident = np.bincount(f.reshape(-1), minlength=len(v))
zero = lengths <= 0.0
print(f"  vertices        : {len(v)}")
print(f"  faces           : {len(f)}")
print(f"  zero-normal     : {int(zero.sum())}")
print(f"   .. of those, with NO incident face : {int((zero & (incident == 0)).sum())}")
print(f"   .. of those, WITH incident faces   : {int((zero & (incident > 0)).sum())}")
degenerate = np.linalg.norm(crosses, axis=1) <= 0.0
print(f"  zero-area faces : {int(degenerate.sum())} of {len(f)}")
