"""What cell edge fits under the existing cap, and what it costs in candidates.

Coarser cells mean fewer insertions and more triangles per cell -- and the proof
has a separate cap on candidates per vertex. Both have to fit, so both are
measured rather than one traded blindly for the other."""
import sys
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/01-segmentation")
from panel import app as A
import numpy as np
from collections import Counter
from fleet.finalizer import triangulate_tifxyz_grid

control = A.first_letters_control_policy()
orientation = (((control.get("checks") or {}).get("PIPELINE_CONTROL") or {})
               .get("orientation_parity") or {})
policy = orientation.get("policy") or {}
reference = (control.get("source_locks") or {}).get("community_surface") or {}
cap = int(policy["maximum_spatial_index_insertions"])
cand_cap = int(policy["maximum_candidates_per_vertex"])

ref_xyz, _ = A.load_locked_orientation_reference(reference)
ref = triangulate_tifxyz_grid(ref_xyz)
v = np.asarray(ref["vertices"], dtype=np.float64)
tri = v[np.asarray(ref["faces"], dtype=np.int64)]
print(f"  caps: insertions {cap}   candidates/vertex {cand_cap}")

for cell in (4.0, 8.0, 12.0, 16.0, 24.0):
    low = np.floor(tri.min(axis=1) / cell).astype(np.int64)
    high = np.floor(tri.max(axis=1) / cell).astype(np.int64)
    span = high - low + 1
    total = int((span[:, 0] * span[:, 1] * span[:, 2]).sum())
    # Worst-case candidates: the busiest cell, since a vertex lands in one cell.
    counts = Counter()
    step = max(1, len(tri) // 20000)          # sample for the histogram
    for i in range(0, len(tri), step):
        for x in range(low[i, 0], high[i, 0] + 1):
            for y in range(low[i, 1], high[i, 1] + 1):
                for z in range(low[i, 2], high[i, 2] + 1):
                    counts[(x, y, z)] += 1
    busiest = max(counts.values()) * step if counts else 0
    print(f"  cell {cell:>5}: insertions {total:>12,}  "
          f"{'OK ' if total <= cap else 'over'}  busiest cell ~{busiest} tris "
          f"{'OK' if busiest <= cand_cap else 'OVER CANDIDATE CAP'}")
