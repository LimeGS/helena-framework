"""Same coordinate frame, or not? Two surfaces far apart mean nothing until
that is settled: a frame mismatch would put them anywhere."""
import sys, json
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/01-segmentation")
from panel import app as A
import numpy as np
from fleet.finalizer import triangulate_tifxyz_grid

JOB, SAMPLE = "p3-2e4a5d0c08b142", "PHerc0139"
control = A.first_letters_control_policy()
reference = (control.get("source_locks") or {}).get("community_surface") or {}
kr = control.get("known_region") or {}

jobs = A.job_store()
row = next(r for r in (jobs.job(JOB) or {}).get("result", {}).get("surfaces") or [])
v, f = A.load_hash_bound_grown_mesh(row["surface_id"], str(row["source_artifact_sha256"]),
                                    expected_sample=SAMPLE, expected_binding=None)
v = np.asarray(v, dtype=np.float64)
ref_xyz, _ = A.load_locked_orientation_reference(reference)
rv = triangulate_tifxyz_grid(ref_xyz)["vertices"]
rv = np.asarray(rv, dtype=np.float64)

print("  grown     bbox min:", np.round(v.min(axis=0), 1), " max:", np.round(v.max(axis=0), 1))
print("  reference bbox min:", np.round(rv.min(axis=0), 1), " max:", np.round(rv.max(axis=0), 1))
print("  manifest surface_bbox_ct_l0_xyz:", json.dumps(kr.get("surface_bbox_ct_l0_xyz"))[:150])
print("  manifest anchor_ct_l0_xyz      :", [round(c,1) for c in (kr.get("anchor_ct_l0_xyz") or [])])
print("  reference coordinate frame     :", reference.get("coordinate_frame"))
# Does the grown surface sit inside the region the manifest declares?
bb = kr.get("surface_bbox_ct_l0_xyz") or {}
lo, hi = np.array(bb.get("minimum") or [0,0,0]), np.array(bb.get("maximum") or [0,0,0])
inside = ((v >= lo) & (v <= hi)).all(axis=1).sum()
print(f"  grown vertices inside the declared known region: {int(inside)} of {len(v)}")
