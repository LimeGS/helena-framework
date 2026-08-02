"""Mesh-mode metrics for score_strip.py: OBJ parsing, manifoldness /
boundary / component counts, and the strip-local cross-wrap-fusion (bridge)
detector.

Scope disclaimer, repeated wherever these numbers are reported: these are
GEOMETRIC CHECKS AGAINST ONE STRIP, not a global topology certification.
"0 fusion triangles" means no triangle of the submitted mesh straddles two
different wraps of THIS strip's reference geometry within its coverage; it
says nothing about the mesh elsewhere in the scroll.

All of this file is new for strip-v0 (no ported logic): the source
neural-tracing-audit repo scored point predictions only. The nearest-wrap
assignment used by the fusion detector reuses scoring_core.nearest_wrap,
whose provenance is documented in scoring_core.py.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scoring_core import nearest_wrap

# PROVISIONAL: coverage radius for mesh vertices, as a multiple of the
# strip's measured median inter-wrap spacing (same spirit as points mode's
# DEFAULT_COVERAGE_RADIUS_GAP_MULTIPLIER in scoring_core.py). A triangle
# with any vertex farther than this from every wrap is excluded from
# fusion analysis -- excluded, not guessed -- and reported in
# `excluded_triangles`.
MESH_COVERAGE_RADIUS_PITCH_MULTIPLIER = 3.0


# ----------------------------------------------------------------------
# OBJ I/O (stdlib only)
# ----------------------------------------------------------------------

def parse_obj(path) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Minimal OBJ reader: `v x y z` and `f ...` lines only.

    - Face vertex references may be `i`, `i/t`, `i//n` or `i/t/n`; only the
      vertex index is used.
    - Indices are 1-based; negative indices are relative to the vertices
      seen so far, per the OBJ spec.
    - Faces with more than 3 vertices are fan-triangulated (0,i,i+1) --
      note this changes triangle/edge counts relative to the polygonal
      mesh, which is the standard tradeoff for triangle-based metrics.
    - Everything else (vt, vn, usemtl, comments, ...) is ignored.
    """
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"{path}:{lineno}: malformed vertex line")
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                if len(parts) < 4:
                    raise ValueError(f"{path}:{lineno}: face with <3 vertices")
                idx = []
                for token in parts[1:]:
                    raw = token.split("/")[0]
                    i = int(raw)
                    if i > 0:
                        i -= 1
                    elif i < 0:
                        i = len(vertices) + i
                    else:
                        raise ValueError(f"{path}:{lineno}: OBJ index 0 is invalid")
                    if i < 0 or i >= len(vertices):
                        raise ValueError(
                            f"{path}:{lineno}: vertex index {token} out of range"
                        )
                    idx.append(i)
                for a in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[a], idx[a + 1]))
    return np.asarray(vertices, dtype=np.float64), faces


def write_obj(path, vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> None:
    """Minimal OBJ writer (v/f lines, 0-based faces in, 1-based out)."""
    with open(path, "w") as fh:
        for v in np.asarray(vertices, dtype=np.float64):
            fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for f in faces:
            fh.write("f " + " ".join(str(int(i) + 1) for i in f) + "\n")


def grid_to_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 valid: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Triangulate a tifxyz-style grid: two triangles per quad cell whose
    four corners are all valid. Vertices are the valid grid points."""
    rows, cols = x.shape
    index = -np.ones((rows, cols), dtype=np.int64)
    rr, cc = np.where(valid)
    index[rr, cc] = np.arange(rr.size)
    vertices = np.stack([x[rr, cc], y[rr, cc], z[rr, cc]], axis=1).astype(np.float64)
    faces: List[Tuple[int, int, int]] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if valid[r, c] and valid[r, c + 1] and valid[r + 1, c] and valid[r + 1, c + 1]:
                a = index[r, c]
                b = index[r, c + 1]
                d = index[r + 1, c]
                e = index[r + 1, c + 1]
                faces.append((a, b, e))
                faces.append((a, e, d))
    return vertices, faces


# ----------------------------------------------------------------------
# Topology counts
# ----------------------------------------------------------------------

def edge_face_counts(faces: Sequence[Sequence[int]]) -> Dict[Tuple[int, int], int]:
    """Incident-face count per undirected edge. A face listed twice counts
    twice -- that is deliberate, so duplicated faces surface as
    non-manifold edges instead of vanishing."""
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            counts[key] += 1
    return dict(counts)


def topology_counts(faces: Sequence[Sequence[int]]) -> Dict:
    counts = edge_face_counts(faces)
    non_manifold = sum(1 for c in counts.values() if c > 2)
    boundary = sum(1 for c in counts.values() if c == 1)
    return {
        "n_faces": len(faces),
        "n_edges": len(counts),
        "non_manifold_edges": non_manifold,
        "boundary_edges": boundary,
        "internal_edges": sum(1 for c in counts.values() if c == 2),
    }


def connected_components(n_vertices: int, faces: Sequence[Sequence[int]]) -> Dict:
    """Components of the face-connectivity graph (vertices linked when they
    share a face edge). Vertices not referenced by any face are not part of
    any component; they are counted separately."""
    parent = list(range(n_vertices))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    referenced = np.zeros(n_vertices, dtype=bool)
    for f in faces:
        n = len(f)
        for i in range(n):
            referenced[f[i]] = True
            union(f[i], f[(i + 1) % n])

    roots = {find(v) for v in range(n_vertices) if referenced[v]}
    return {
        "connected_components": len(roots),
        "referenced_vertices": int(referenced.sum()),
        "unreferenced_vertices": int(n_vertices - referenced.sum()),
    }


# ----------------------------------------------------------------------
# Cross-wrap fusion (bridge) detection against a strip
# ----------------------------------------------------------------------

def fusion_analysis(strip, trees, vertices: np.ndarray,
                    faces: Sequence[Sequence[int]],
                    coverage_radius: float = None) -> Dict:
    """Classify each triangle by its vertices' nearest reference wraps.

    - A vertex is assigned to the wrap whose point set it is nearest to
      (scoring_core.nearest_wrap), with the distance recorded.
    - A triangle is EXCLUDED (not scored) if any vertex is farther than
      `coverage_radius` from every wrap (no reliable reference there), or
      if any vertex's nearest reference point is edge-flagged (coverage
      boundary; near the spiral's cut line the wrap labels legitimately
      meet, so "fusion" is undefined there).
    - A scored triangle is a FUSION (bridge) triangle iff its three
      vertices' nearest wraps are not all the same wrap.

    Returns counts plus `fused_pair_histogram`: for each fusion triangle,
    every unordered pair of distinct wraps among its vertex assignments is
    tallied (a triangle spanning wraps {2,3} adds one count to "2-3").
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if coverage_radius is None:
        # Reference scale: pooled median nearest-neighbor distance between
        # consecutive wraps, measured on the strip itself (native units).
        ids = sorted(trees.keys())
        gaps = []
        for a, b in zip(ids[:-1], ids[1:]):
            pts_a = np.asarray(strip.wraps[a], dtype=np.float64)
            # subsample the query side for speed on big strips; the median
            # of NN distances is robust to this
            step = max(1, pts_a.shape[0] // 10_000)
            d, _ = trees[b].query(pts_a[::step], workers=-1)
            gaps.append(np.median(d))
        pitch_native = float(np.median(np.asarray(gaps))) if gaps else 0.0
        coverage_radius = MESH_COVERAGE_RADIUS_PITCH_MULTIPLIER * pitch_native

    nearest_wrap_id, nearest_dist = nearest_wrap(vertices, trees)

    # per-vertex: is the nearest reference point edge-flagged?
    vertex_edge = np.zeros(vertices.shape[0], dtype=bool)
    edges_by_wrap = getattr(strip, "edges", {})
    if edges_by_wrap:
        for wid in sorted(trees.keys()):
            flags = edges_by_wrap.get(wid)
            if flags is None:
                continue
            sel = nearest_wrap_id == wid
            if not sel.any():
                continue
            _, idx = trees[wid].query(vertices[sel], workers=-1)
            vertex_edge[sel] = flags[idx]

    vertex_covered = nearest_dist <= coverage_radius

    n_fusion = 0
    n_excluded = 0
    n_scored = 0
    pair_hist: Counter = Counter()
    per_wrap_vertex_counts: Counter = Counter()
    for wid, cov in zip(nearest_wrap_id, vertex_covered):
        if cov:
            per_wrap_vertex_counts[int(wid)] += 1

    for f in faces:
        ids = [int(nearest_wrap_id[v]) for v in f]
        if (not all(vertex_covered[v] for v in f)) or any(vertex_edge[v] for v in f):
            n_excluded += 1
            continue
        n_scored += 1
        distinct = sorted(set(ids))
        if len(distinct) > 1:
            n_fusion += 1
            for i in range(len(distinct)):
                for j in range(i + 1, len(distinct)):
                    pair_hist[f"{distinct[i]}-{distinct[j]}"] += 1

    return {
        "coverage_radius": float(coverage_radius),
        "scored_triangles": n_scored,
        "excluded_triangles": n_excluded,
        "cross_wrap_fusion_triangles": n_fusion,
        "fused_pair_histogram": dict(sorted(pair_hist.items())),
        "vertices_per_wrap_within_coverage": {
            str(k): int(v) for k, v in sorted(per_wrap_vertex_counts.items())
        },
        "disclaimer": (
            "geometric checks against this strip's reference wraps only; "
            "NOT a global topology certification"
        ),
    }
