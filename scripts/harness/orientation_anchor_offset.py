"""Where the chosen anchor sits relative to the reference surface.

The anchor was picked by CT density, to escape a degenerate screening -- not by
matching w025. If the anchor itself is far from the reference, the grow started
somewhere else and the divergence is our choice, not the grower's."""
import sys
sys.path.insert(0, "/workspace/campaign-x")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/02-flattening")
sys.path.insert(0, "/workspace/campaign-x/framework/stages/01-segmentation")
from panel import app as A
import numpy as np
from fleet.finalizer import triangulate_tifxyz_grid

control = A.first_letters_control_policy()
kr = control["known_region"]
reference = control["source_locks"]["community_surface"]

anchor = np.array(kr["anchor_ct_l0_xyz"], dtype=np.float64)
cell = kr["anchor_surface_cell_yx"]
print(f"  anchor cell {cell}  ->  xyz {np.round(anchor,1).tolist()}")

ref_xyz, _ = A.load_locked_orientation_reference(reference)
grid = np.asarray(ref_xyz, dtype=np.float64)          # (rows, cols, 3)
print(f"  reference grid shape: {grid.shape}")

valid = ~np.all(np.isclose(grid, -1.0), axis=2)
points = grid[valid]
print(f"  reference points: {len(points)} valid of {valid.size}")

d = np.linalg.norm(points - anchor, axis=1)
print(f"  anchor to nearest reference point : {d.min():.2f} voxels")
print(f"  anchor to reference, median       : {np.median(d):.2f} voxels")

# Does the anchor's own grid cell hold a valid reference point, and where is it?
row, col = int(cell[0]), int(cell[1])
if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
    at_cell = grid[row, col]
    print(f"  reference at the anchor's own cell [{row},{col}]: {np.round(at_cell,1).tolist()}")
    if not np.all(np.isclose(at_cell, -1.0)):
        print(f"    distance from the manifest anchor: "
              f"{np.linalg.norm(at_cell - anchor):.2f} voxels")
    else:
        print("    that cell has no data in the reference")
else:
    print(f"  the anchor cell {cell} is outside the reference grid {grid.shape[:2]}")
