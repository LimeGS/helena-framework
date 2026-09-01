"""How far the grown surface actually is from the reference.

The proof retained 16 correspondences against a required 64. That is either a
tolerance a little too tight or two surfaces that are not in the same place, and
the distance distribution is what tells them apart."""
import sys
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/01-segmentation")
from panel import app as A
import numpy as np
from fleet.finalizer import triangulate_tifxyz_grid

JOB, SAMPLE = "p3-2e4a5d0c08b142", "PHerc0139"
control = A.first_letters_control_policy()
orientation = (((control.get("checks") or {}).get("PIPELINE_CONTROL") or {})
               .get("orientation_parity") or {})
policy = orientation["policy"]
reference = (control.get("source_locks") or {}).get("community_surface") or {}

jobs = A.job_store()
row = next(r for r in (jobs.job(JOB) or {}).get("result", {}).get("surfaces") or [])
v, f = A.load_hash_bound_grown_mesh(row["surface_id"], str(row["source_artifact_sha256"]),
                                    expected_sample=SAMPLE, expected_binding=None)
v = np.asarray(v, dtype=np.float64); f = np.asarray(f, dtype=np.int64)
used = np.zeros(len(v), bool); used[np.unique(f)] = True
v = v[used]

ref_xyz, _ = A.load_locked_orientation_reference(reference)
ref = triangulate_tifxyz_grid(ref_xyz)
rv = np.asarray(ref["vertices"], dtype=np.float64)

print(f"  grown vertices in faces : {len(v)}")
print(f"  reference vertices      : {len(rv)}")
print(f"  tolerance (policy)      : {policy['maximum_distance_ct_l0_voxels']} voxels")
print(f"  minimum correspondences : {policy['minimum_correspondences']}")

# Nearest reference VERTEX per grown vertex: an upper bound on the distance to
# the nearest reference triangle, and enough to tell a near miss from a miss.
best = np.full(len(v), np.inf)
for start in range(0, len(rv), 20000):
    chunk = rv[start:start + 20000]
    d = np.linalg.norm(v[:, None, :] - chunk[None, :, :], axis=2)
    best = np.minimum(best, d.min(axis=1))

for q in (1, 5, 25, 50, 75, 95, 99):
    print(f"    p{q:<2} distance to nearest reference vertex : {np.percentile(best, q):8.2f} voxels")
for t in (2.0, 4.0, 8.0, 16.0, 32.0):
    print(f"    within {t:>5} voxels: {int((best <= t).sum()):>7d} of {len(v)}")
