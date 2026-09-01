"""Both triangles of a quad must face the same way.

The gate split every grid quad into `(i00,i10,i01)` and `(i11,i10,i01)`. Both
walk the shared edge i10->i01 in the same direction, so the two triangles of
every quad came out with opposite normals -- exactly -1.0 on a flat quad, and
more than 90 degrees apart on 97% of the quads of a fitted spiral winding.

Nothing consumed that sign, which is why no verdict was wrong: the fold-back
detector reads only the first triangle of each quad, and the parallel test takes
an absolute value. The damage was latent and of a kind this platform has already
paid for once -- a second, silently broken source of surface orientation, beside
a render whose own default comes out inverted against the published PHerc0139
stack. `mesh["normal"]` is published; half of it was noise.

So this pins three things: the winding is consistent, it is the finalizer's own
-- which is what the module's docstring claims and did not do -- and fixing it
moved no verdict on the corpus the gate was written for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

import helena_tifxyz_geometry_gate as gate  # noqa: E402
from fleet.finalizer import triangulate_tifxyz_grid  # noqa: E402

GROWN = ROOT / "vendor/villa/volume-cartographer/core/test/data/segments"


def flat_grid(height: int = 6, width: int = 5, *, tilt: float = 0.0):
    """A patch of a plane, with real coordinates and every cell valid."""
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    points = np.stack([
        (100 + 10 * columns).astype(float),
        (200 + 10 * rows).astype(float),
        (300 + tilt * columns).astype(float),
    ], axis=-1)
    return points, np.ones((height, width), dtype=bool)


def test_the_two_triangles_of_a_quad_face_the_same_way():
    """The bug, at its smallest: one flat quad, two triangles, dot -1.0."""
    points, valid = flat_grid(2, 2)
    mesh = gate.build_mesh(points, valid)

    assert len(mesh["triangles"]) == 2
    first, second = mesh["normal"]
    assert float(first @ second) == pytest.approx(1.0)


def test_a_flat_patch_has_one_normal_everywhere():
    points, valid = flat_grid()
    normals = gate.build_mesh(points, valid)["normal"]

    # Every triangle of a plane points the same way, so the spread is zero.
    assert np.allclose(normals, normals[0]), "a plane came out folded"


def test_the_triangulation_is_the_finalizer_s_own_vertex_for_vertex():
    """What the module's docstring promises. It used to promise the split and
    quietly differ on the winding, which is the half that carries orientation."""
    points, valid = flat_grid(7, 6, tilt=1.5)
    mine = gate.build_mesh(points, valid)["triangles"]
    theirs = triangulate_tifxyz_grid(points)["faces"]

    assert sorted(map(tuple, mine.tolist())) == sorted(map(tuple, theirs.tolist()))


def test_a_hole_still_keeps_the_half_of_a_quad_that_is_valid():
    """The masks are per triangle, and the fix must not have changed that: a
    cell beside a hole contributes whichever of its two triangles is whole."""
    points, valid = flat_grid(4, 4)
    valid[0, 0] = False               # kills the first triangle of quad (0,0)

    mine = gate.build_mesh(points, valid)["triangles"]
    theirs = triangulate_tifxyz_grid(np.where(valid[..., None], points, -1.0))["faces"]

    assert sorted(map(tuple, mine.tolist())) == sorted(map(tuple, theirs.tolist()))


# Captured at import, because the regression below monkeypatches the name and a
# wrapper that looked it up again would call itself.
BUILD_MESH = gate.build_mesh


def published_winding(points, valid):
    """`build_mesh` as it shipped, so the regression compares against it."""
    mesh = BUILD_MESH(points, valid)
    height, width = valid.shape
    index = np.arange(height * width).reshape(height, width)
    i00, i10, i01, i11 = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    m00, m10, m01, m11 = valid[:-1, :-1], valid[1:, :-1], valid[:-1, 1:], valid[1:, 1:]
    triangles = np.concatenate([
        np.stack([i00, i10, i01], axis=-1)[m00 & m10 & m01],
        np.stack([i11, i10, i01], axis=-1)[m11 & m10 & m01],
    ], axis=0).astype(np.int64)
    vertices = mesh["vertices"]
    a, b, c = (vertices[triangles[:, column]] for column in range(3))
    raw = np.cross(b - a, c - a)
    length = np.linalg.norm(raw, axis=1)
    unit = np.zeros_like(raw)
    finite = length > 1e-12
    unit[finite] = raw[finite] / length[finite, None]
    mesh.update({"triangles": triangles, "corner_a": a, "corner_b": b,
                 "corner_c": c, "normal": unit, "degenerate": ~finite})
    return mesh


@pytest.mark.parametrize("segment", sorted(
    (path.name for path in GROWN.glob("*") if (path / "x.tif").is_file())
    if GROWN.is_dir() else []))
def test_the_fix_moves_no_verdict_on_the_corpus_the_gate_was_written_for(segment, monkeypatch):
    """Three grown VC3D surfaces, measured both ways. The claim in the docstring
    -- that no verdict this gate issued was affected -- is checked, not asserted."""
    pytest.importorskip("tifffile")
    points, valid = gate.load_tifxyz(GROWN / segment)

    fixed = gate.classify(gate.measure(GROWN / segment))
    monkeypatch.setattr(gate, "build_mesh", published_winding)
    published = gate.classify(gate.measure(GROWN / segment))

    assert fixed["geometry_qc_state"] == published["geometry_qc_state"]
    assert fixed["reason"] == published["reason"]
    # And these are the surfaces the gate exists to pass, so a green run here is
    # also the statement that the corpus still certifies.
    assert fixed["geometry_qc_state"] == "GEOMETRY_CERTIFIED"


def test_only_two_places_read_the_normal_field_and_neither_reads_its_sign():
    """Why the bug was latent, kept as a property rather than a comment.

    Two consumers today. The fold-back detector takes the first triangle of each
    quad, which is one consistently wound set on its own; the parallel test
    compares two normals through an absolute value. A third consumer, or either
    of these losing its guard, is a detector that has begun to depend on
    orientation -- and this is where that gets noticed rather than shipped.
    """
    import ast

    path = (ROOT / "framework/stages/04-validation/scripts"
            / "helena_tifxyz_geometry_gate.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    readers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id == "mesh"
        and isinstance(node.slice, ast.Constant) and node.slice.value == "normal"
    ]
    assert len(readers) == 3, (
        f"{len(readers)} places read mesh['normal']; there were three -- the "
        "fold-back slice and the two sides of the parallel test. A new one has "
        "to say what orientation means here before it can rely on it.")
    # The fold-back detector takes one consistently wound set.
    assert 'mesh["normal"][:first_count]' in source
    # And the only comparison between two normals discards the sign.
    assert "cosine = np.abs(" in source
