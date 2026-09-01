"""How many spatial-index insertions the control's reference actually needs.

The proof aborts at the cap, so the receipt says 4000001 and nothing about the
true total. Without that number there is no way to tell a bound that is short by
a little from an approach that does not scale."""
import json, sys
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/01-segmentation")
from panel import app as A
import numpy as np
from fleet.finalizer import triangulate_tifxyz_grid

control = A.first_letters_control_policy()
orientation = (((control.get("checks") or {}).get("PIPELINE_CONTROL") or {})
               .get("orientation_parity") or {})
policy = orientation.get("policy") or {}
reference = (control.get("source_locks") or {}).get("community_surface") or {}

ref_xyz, _ = A.load_locked_orientation_reference(reference)
ref = triangulate_tifxyz_grid(ref_xyz)
v = np.asarray(ref["vertices"], dtype=np.float64)
f = np.asarray(ref["faces"], dtype=np.int64)
tri = v[f]
cell = float(policy["spatial_cell_edge_voxels"])
print(f"  reference triangles : {len(f)}")
print(f"  cell edge (voxels)  : {cell}")
print(f"  configured cap      : {int(policy['maximum_spatial_index_insertions'])}")

low = np.floor(tri.min(axis=1) / cell).astype(np.int64)
high = np.floor(tri.max(axis=1) / cell).astype(np.int64)
span = (high - low + 1)
per_triangle = span[:, 0] * span[:, 1] * span[:, 2]
total = int(per_triangle.sum())
print(f"  insertions required : {total}")
print(f"  ratio to cap        : {total / float(policy['maximum_spatial_index_insertions']):.2f}x")
print(f"  per triangle: mean {per_triangle.mean():.1f}  median {np.median(per_triangle):.0f}"
      f"  max {per_triangle.max()}")
big = per_triangle > 1000
print(f"  triangles needing >1000 cells: {int(big.sum())}")
