"""Geometric ray-cast adapter: pure trimesh ray-casting along the seed
normal, no learning, no CT/zarr access.

For each seed cell, cast a ray from the seed point along
sign(direction)*normal (front (+1) hops by -normal, back (-1) by
+normal -- same convention as stb.arms/reference_src.v2_score) and take
the CLOSEST mesh intersection whose distance from the seed falls in the
open interval (RAY_LO*gap_hint_median, RAY_HI*gap_hint_median); no
qualifying intersection -> NaN (abstain), which stb.contract.score_candidate
then excludes from both the numerator and the denominator.

build_grid_mesh is the companion "reconstruct a local surface patch from
a tifxyz-like grid subset" primitive this adapter needs a mesh from in
the first place: both the real segmentation surfaces mesh_adapter is
meant to score and this module's own test oracle (stb.core.Reference's
own class +1/-1 point sets, which retain their row/col grid indices) are
exactly this shape of data -- a scattered set of grid-indexed points, not
a dense image -- so it is exposed as a public, reusable function rather
than test-only scaffolding.
"""
import numpy as np

RAY_LO, RAY_HI = 0.25, 2.5


def build_grid_mesh(rows, cols, pts):
    """Triangulate a scattered, grid-indexed point set: for every integer
    grid cell (r, c) whose 4 corners (r,c), (r+1,c), (r,c+1), (r+1,c+1)
    are ALL present in (rows, cols, pts), emit the 2 triangles
    (r,c)-(r+1,c)-(r,c+1) and (r+1,c)-(r+1,c+1)-(r,c+1); missing corners
    (rows/cols is not assumed to be a dense rectangle) simply leave a
    hole. Returns a trimesh.Trimesh (vertices = `pts` verbatim,
    process=False so trimesh does not merge/reindex/reorder them).
    """
    import trimesh

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float64)
    if not (len(rows) == len(cols) == len(pts)):
        raise ValueError("rows, cols and pts must have matching length")

    index = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    faces = []
    for (r, c), i00 in index.items():
        i10 = index.get((r + 1, c))
        i01 = index.get((r, c + 1))
        i11 = index.get((r + 1, c + 1))
        if i10 is not None and i01 is not None:
            faces.append((i00, i10, i01))
        if i10 is not None and i01 is not None and i11 is not None:
            faces.append((i10, i11, i01))
    if not faces:
        raise ValueError("no complete grid cells (all 4 corners present) to triangulate")
    return trimesh.Trimesh(vertices=pts, faces=np.asarray(faces, dtype=np.int64),
                            process=False)


def predict_topk(task, mesh, max_hits=8):
    """Return distance-sorted ray intersections without collapsing proposals.

    Output is `(points, distances)` with shapes `(n,max_hits,3)` and
    `(n,max_hits)`. Missing proposals are NaN. Relation models can rerank
    these proposals without changing the legacy one-point contract.
    """
    if max_hits <= 0:
        raise ValueError("max_hits must be positive")
    sign = -1.0 if task.direction == +1 else 1.0
    origins = np.asarray(task.seed_points, dtype=np.float64)
    directions = sign * np.asarray(task.normals, dtype=np.float64)
    n = len(origins)
    points = np.full((n, max_hits, 3), np.nan, dtype=np.float64)
    distances = np.full((n, max_hits), np.nan, dtype=np.float64)

    good = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1)
    if not good.any():
        return points, distances
    idx_good = np.where(good)[0]

    locations, index_ray, _index_tri = mesh.ray.intersects_location(
        origins[idx_good], directions[idx_good], multiple_hits=True
    )
    if len(index_ray) == 0:
        return points, distances

    lo = RAY_LO * task.gap_hint_median
    hi = RAY_HI * task.gap_hint_median
    dist = np.linalg.norm(locations - origins[idx_good][index_ray], axis=1)
    in_band = (dist > lo) & (dist < hi)

    hits = {}
    for local_i, d, loc, ok in zip(index_ray, dist, locations, in_band):
        if not ok:
            continue
        cell = idx_good[local_i]
        hits.setdefault(cell, []).append((float(d), np.asarray(loc, dtype=np.float64)))
    for cell, cell_hits in hits.items():
        for rank, (d, loc) in enumerate(sorted(cell_hits, key=lambda item: item[0])[:max_hits]):
            distances[cell, rank] = d
            points[cell, rank] = loc
    return points, distances


def predict(task, mesh):
    """Legacy nearest-hit adapter, implemented through the V4 top-K path."""
    points, _distances = predict_topk(task, mesh, max_hits=1)
    return points[:, 0]
