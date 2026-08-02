"""Shared synthetic fixtures: an analytic Archimedean-spiral scroll.

Surface family: r(k, theta) = r0 + pitch * (k + theta / 2*pi), theta in
[0, 2*pi) per wrap k -- i.e. one continuous spiral whose consecutive turns
are exactly `pitch` apart radially at equal theta. Extruded along z to make
each wrap a 2D sheet. Everything is deterministic (no RNG unless a test
seeds one explicitly).

Two forms:
- spiral_strip(): per-wrap point sets + edge flags + a ready Strip object,
  for format/qualify/score tests.
- spiral_grid(): a tifxyz-style (rows, cols) coordinate grid where the
  column axis winds through revolutions, for make_strip tests.
"""

import numpy as np

import strip_format
from strip_format import Strip, edge_flags_from_phase

DEFAULT_META = {
    "scroll": "synthetic",
    "segment_id": "synthetic-spiral",
    "window": {"note": "analytic fixture"},
    "voxel_size_um": 1.0,
    "tier": "unknown",
    "schema_version": strip_format.SCHEMA_VERSION,
    "source_checksum": "synthetic",
}

EDGE_MARGIN_RAD = 0.01 * 2 * np.pi  # matches make_strip.EDGE_MARGIN_REVOLUTIONS


def spiral_wraps(n_wraps=4, pts_per_row=300, rows=10, r0=300.0, pitch=30.0,
                 height=40.0):
    """Per-wrap (points, phase) dicts for the analytic spiral."""
    wraps, phases = {}, {}
    for k in range(n_wraps):
        theta = np.linspace(0, 2 * np.pi, pts_per_row, endpoint=False)
        r = r0 + pitch * (k + theta / (2 * np.pi))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        zs = np.linspace(0, height, rows)
        pts = np.concatenate(
            [np.stack([x, y, np.full_like(x, z)], axis=1) for z in zs]
        )
        wraps[k] = pts.astype(np.float32)
        phases[k] = np.tile(theta, rows)
    return wraps, phases


def spiral_strip(n_wraps=4, pts_per_row=300, rows=10, r0=300.0, pitch=30.0,
                 height=40.0, with_edges=True, meta_overrides=None) -> Strip:
    wraps, phases = spiral_wraps(n_wraps, pts_per_row, rows, r0, pitch, height)
    edges = {}
    if with_edges:
        edges = {
            k: edge_flags_from_phase(phases[k], EDGE_MARGIN_RAD) for k in wraps
        }
    meta = dict(DEFAULT_META)
    if meta_overrides:
        meta.update(meta_overrides)
    return Strip(
        wraps=wraps,
        normals={},
        edges=edges,
        pitch_um={"median": float(pitch), "p10": float(pitch) * 0.97,
                  "p90": float(pitch) * 1.03},
        meta=meta,
    )


def shuffled_strip(strip: Strip, fraction=0.4, seed=42) -> Strip:
    """Corrupted copy: `fraction` of points swapped between wrap pairs
    (0<->1, 2<->3, ...) -- labels wrong, geometry intact. A qualification
    suite that cannot fail this is not measuring anything."""
    rng = np.random.default_rng(seed)
    bad = {k: v.copy() for k, v in strip.wraps.items()}
    ids = sorted(bad.keys())
    for k in ids[:-1:2]:
        n = min(bad[k].shape[0], bad[k + 1].shape[0])
        m = int(fraction * n)
        idx = rng.choice(n, size=m, replace=False)
        tmp = bad[k][idx].copy()
        bad[k][idx] = bad[k + 1][idx]
        bad[k + 1][idx] = tmp
    return Strip(wraps=bad, normals=dict(strip.normals),
                 edges=dict(strip.edges), pitch_um=dict(strip.pitch_um),
                 meta=dict(strip.meta))


def spiral_grid(rows=24, cols=5000, revolutions=2.6, r0=400.0, pitch=30.0,
                height=40.0):
    """tifxyz-style grid of the same spiral: the COLUMN axis advances
    through `revolutions`; rows extrude along z. Returns (x, y, z, valid)
    float64/bool arrays of shape (rows, cols). The scroll axis is the
    z-axis through the origin."""
    theta = np.linspace(0, revolutions * 2 * np.pi, cols)
    r = r0 + pitch * (theta / (2 * np.pi))
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    x = np.tile(x1, (rows, 1))
    y = np.tile(y1, (rows, 1))
    z = np.tile(np.linspace(0.0, height, rows)[:, None], (1, cols))
    valid = np.ones((rows, cols), dtype=bool)
    return x, y, z, valid


def wrap_surface_points(k, thetas, z_values, r0=300.0, pitch=30.0):
    """Analytic points on wrap k at given thetas/z (for building
    predictions in tests): r = r0 + pitch * (k + theta/2pi)."""
    r = r0 + pitch * (k + thetas / (2 * np.pi))
    out = []
    for z in np.atleast_1d(z_values):
        out.append(np.stack(
            [r * np.cos(thetas), r * np.sin(thetas), np.full_like(thetas, z)],
            axis=1,
        ))
    return np.concatenate(out)
