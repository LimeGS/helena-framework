import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesh_metrics import (
    connected_components,
    fusion_analysis,
    grid_to_mesh,
    parse_obj,
    topology_counts,
    write_obj,
)
from scoring_core import build_trees
from synthetic import spiral_strip

R0, PITCH = 300.0, 30.0
ROWS, COLS = 8, 60


def wrap1_grid_mesh():
    """Clean manifold triangulated grid lying on wrap 1, interior thetas."""
    thg = np.linspace(0.5, 5.8, COLS)
    zg = np.linspace(5.0, 35.0, ROWS)
    TH, ZG = np.meshgrid(thg, zg)
    rg = R0 + PITCH * (1 + TH / (2 * np.pi))
    X, Y, Z = rg * np.cos(TH), rg * np.sin(TH), ZG
    valid = np.ones((ROWS, COLS), dtype=bool)
    return grid_to_mesh(X, Y, Z, valid), thg, zg


class TopologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.verts, cls.faces), cls.thg, cls.zg = wrap1_grid_mesh()

    def test_clean_strip_mesh_counts(self):
        topo = topology_counts(self.faces)
        self.assertEqual(topo["non_manifold_edges"], 0)
        self.assertEqual(topo["n_faces"], 2 * (ROWS - 1) * (COLS - 1))
        # perimeter of an open grid patch
        self.assertEqual(topo["boundary_edges"], 2 * (ROWS - 1) + 2 * (COLS - 1))

    def test_single_component(self):
        comps = connected_components(self.verts.shape[0], self.faces)
        self.assertEqual(comps["connected_components"], 1)
        self.assertEqual(comps["unreferenced_vertices"], 0)

    def test_duplicated_face_creates_non_manifold_edges(self):
        faces = list(self.faces) + [self.faces[5]]
        topo = topology_counts(faces)
        # all 3 edges of the duplicated triangle gain an extra incidence;
        # each already had 2 (interior face), so all 3 become non-manifold
        self.assertEqual(topo["non_manifold_edges"], 3)

    def test_two_disjoint_pieces_are_two_components(self):
        verts = np.vstack([self.verts, self.verts + 1000.0])
        offset = self.verts.shape[0]
        faces = list(self.faces) + [
            tuple(v + offset for v in f) for f in self.faces
        ]
        comps = connected_components(verts.shape[0], faces)
        self.assertEqual(comps["connected_components"], 2)


class FusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strip = spiral_strip(r0=R0, pitch=PITCH)
        cls.trees = build_trees(cls.strip)
        (cls.verts, cls.faces), cls.thg, cls.zg = wrap1_grid_mesh()

    def test_clean_single_wrap_mesh_has_zero_fusion(self):
        fus = fusion_analysis(self.strip, self.trees, self.verts, self.faces)
        self.assertEqual(fus["cross_wrap_fusion_triangles"], 0)
        self.assertEqual(fus["excluded_triangles"], 0)
        self.assertEqual(fus["fused_pair_histogram"], {})

    def test_single_bridge_triangle_counts_exactly_one(self):
        # one extra vertex on wrap 2, one triangle bridging wrap 1 -> 2
        th_c = self.thg[10]
        r_c = R0 + PITCH * (2 + th_c / (2 * np.pi))
        p_c = np.array([r_c * np.cos(th_c), r_c * np.sin(th_c), self.zg[3]])
        verts = np.vstack([self.verts, p_c])
        bridge = (10, 11, verts.shape[0] - 1)
        faces = list(self.faces) + [bridge]
        fus = fusion_analysis(self.strip, self.trees, verts, faces)
        self.assertEqual(fus["cross_wrap_fusion_triangles"], 1)
        self.assertEqual(fus["fused_pair_histogram"], {"1-2": 1})

    def test_triangle_outside_coverage_is_excluded_not_fused(self):
        far = np.array([[5000.0, 5000.0, 0.0],
                        [5010.0, 5000.0, 0.0],
                        [5000.0, 5010.0, 0.0]])
        verts = np.vstack([self.verts, far])
        n = self.verts.shape[0]
        faces = list(self.faces) + [(n, n + 1, n + 2)]
        fus = fusion_analysis(self.strip, self.trees, verts, faces)
        self.assertEqual(fus["excluded_triangles"], 1)
        self.assertEqual(fus["cross_wrap_fusion_triangles"], 0)


class ObjIoTest(unittest.TestCase):
    def test_roundtrip_tiny_file(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        faces = [(0, 1, 2), (0, 2, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.obj"
            write_obj(path, verts, faces)
            v2, f2 = parse_obj(path)
        np.testing.assert_allclose(v2, verts)
        self.assertEqual(f2, faces)

    def test_parses_slash_forms_negative_indices_and_quads(self):
        content = (
            "# comment\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvn 0 0 1\n"
            "f 1/1/1 2/1/1 3/1/1\n"   # i/t/n form
            "f -4 -2 -1\n"             # negative (relative) indices
            "f 1 2 3 4\n"              # quad -> fan-triangulated
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forms.obj"
            path.write_text(content)
            verts, faces = parse_obj(path)
        self.assertEqual(verts.shape, (4, 3))
        self.assertEqual(faces, [
            (0, 1, 2),
            (0, 2, 3),
            (0, 1, 2), (0, 2, 3),  # the quad
        ])

    def test_out_of_range_index_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.obj"
            path.write_text("v 0 0 0\nf 1 2 3\n")
            with self.assertRaises(ValueError):
                parse_obj(path)


if __name__ == "__main__":
    unittest.main()
