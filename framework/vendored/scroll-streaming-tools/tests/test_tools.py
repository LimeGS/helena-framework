"""Offline unit tests for the streaming tools.

No network, no GPU, no scan data: transforms are mocked, masks are synthetic.
Run with either:
    python -m pytest tests/
    python tests/test_tools.py

infer_timesformer.py is a patched copy of the upstream script and is
deliberately not unit-tested here (its five patches are documented in its
header; the rest is upstream's responsibility).
"""
import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "adapters"))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# ---------------------------------------------------------------- helpers

def _fake_transform_urlopen(scale, trans):
    """urlopen stand-in serving a transform.json for fixed = scale*mov + trans."""
    rng = np.random.default_rng(7)
    mov = rng.uniform(100, 5000, (6, 3))
    fix = mov * scale + np.array(trans)
    doc = {
        "transformation_matrix": np.hstack([np.eye(3) * scale,
                                            np.array(trans)[:, None]]).tolist(),
        "fixed_landmarks": fix.tolist(),
        "moving_landmarks": mov.tolist(),
    }

    def fake(url, timeout=0):
        return io.BytesIO(json.dumps(doc).encode())
    return fake


# ------------------------------------------------------- render_scroll2/3

def test_parse_ppm_header_roundtrip():
    import render_scroll3 as r3
    from wedge_extract import ppm_header_str
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        f.write(ppm_header_str(7, 3).encode())
        f.write(np.zeros((3, 7, 6), np.float64).tobytes())
        path = f.name
    try:
        w, h, dim, off = r3.parse_ppm_header(path)
        assert (w, h, dim) == (7, 3, 6)
        data = np.memmap(path, dtype=np.float64, mode="r",
                         offset=off).reshape(h, w, dim)
        assert data.shape == (3, 7, 6) and float(data.sum()) == 0.0
    finally:
        os.unlink(path)


def test_load_transform_inverts_and_derives_scale():
    # fixed = 0.5*mov + t  ->  fix->mov must scale by 2, residual ~0
    import render_scroll2 as r2
    import render_scroll3 as r3
    orig = urllib.request.urlopen
    urllib.request.urlopen = _fake_transform_urlopen(0.5, [10.0, -20.0, 3.0])
    try:
        for mod in (r3, r2):
            R, t, scale = mod.load_transform()
            assert abs(scale - 2.0) < 1e-9, f"{mod.__name__}: scale {scale}"
            assert np.allclose(R, np.eye(3) * 2.0)
    finally:
        urllib.request.urlopen = orig


def test_fetch_chunk_404_means_empty():
    import render_scroll3 as r3
    orig = urllib.request.urlopen

    def fake(url, timeout=0):
        raise urllib.error.HTTPError(url, 404, "nf", None, io.BytesIO())
    urllib.request.urlopen = fake
    try:
        key, arr = r3.fetch_chunk((1, 2, 3))
        assert key == (1, 2, 3) and arr is None
    finally:
        urllib.request.urlopen = orig


# ---------------------------------------------------------- winding_tracer

def test_tracer_finds_synthetic_ring():
    import winding_tracer as wt
    shape = (2, 128, 128)
    yy, xx = np.mgrid[:128, :128]
    r = np.hypot(yy - 64, xx - 64)
    ring = ((r >= 39) & (r < 42)).astype(np.uint8)   # annulus at r ~ 40
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    mask = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    mask[:] = ring[None]
    mask.flush()
    old_shm, old_shape = wt.SHM, wt.SHAPE
    wt.SHM, wt.SHAPE = path, shape
    try:
        axis = np.full((shape[0], 2), 64.0, np.float32)
        wt._init(axis)
        z0, out = wt._trace_zband((0, 2, 8, 60))
        hits = out[out > 0]
        assert len(hits) == 16, f"expected 1 crossing x 8 rays x 2 slices, got {len(hits)}"
        # annulus [39, 42) has voxels at r = 39, 40, 41 -> true center 40.0.
        # this is exact since the half-voxel run-start bias fix.
        assert np.all(np.abs(hits - 40.0) < 0.6), f"crossing radii off: {hits}"
    finally:
        wt.SHM, wt.SHAPE = old_shm, old_shape
        os.unlink(path)


# ---------------------------------------------------------- label_transport

def test_pick_windows_density_and_separation():
    from label_transport import pick_windows
    lab = np.zeros((4096, 4096), np.uint8)
    lab[256:1024, 256:1024] = 255       # dense blob A
    lab[2560:3072, 2560:3072] = 255     # smaller blob B, far away
    picked = pick_windows(lab, size=1024, max_windows=4)
    assert 1 <= len(picked) <= 4
    # densest window first, and it must cover blob A
    d0, y0, x0 = picked[0]
    assert d0 >= picked[-1][0]
    assert y0 < 1024 and x0 < 1024
    # every pair non-overlapping
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            _, yi, xi = picked[i]
            _, yj, xj = picked[j]
            assert abs(yi - yj) >= 1024 or abs(xi - xj) >= 1024


def test_fetch_rows_batches_and_reassembles():
    import label_transport as lt
    W, nrows, off = 11, 9, 100
    payload = np.arange(nrows * W * 6, dtype=np.float64)
    blob = b"x" * off + payload.tobytes()

    def fake(req, timeout=0):
        rng = req.headers["Range"].split("=")[1]
        a, b = map(int, rng.split("-"))
        return io.BytesIO(blob[a:b + 1])

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            path = f.name
        lt.fetch_rows("segX", off, W, y0=0, nrows=nrows, out_path=path, batch=2)
        import render_scroll3 as r3
        w, h, dim, o = r3.parse_ppm_header(path)
        assert (w, h, dim) == (W, nrows, 6)
        got = np.memmap(path, dtype=np.float64, mode="r", offset=o)
        assert np.array_equal(np.asarray(got), payload), "reassembled bytes differ"
        os.unlink(path)
    finally:
        urllib.request.urlopen = orig


def test_pick_windows_empty_label():
    from label_transport import pick_windows
    assert pick_windows(np.zeros((4096, 4096), np.uint8), 1024, 4) == []


# ------------------------------------------------------------- adapters

def test_preprocess_tile_clips_at_200():
    from infer_resnet3d_1667 import preprocess_tile
    t = np.array([[0, 100, 200, 255]], np.uint8)
    out = preprocess_tile(t)
    assert np.allclose(out, [[0.0, 100 / 255, 200 / 255, 200 / 255]])
    assert out.dtype == np.float32


def test_gaussian_kernel_shape_and_peak():
    from infer_resnet3d_1667 import gaussian_kernel
    g = gaussian_kernel(64)
    assert g.shape == (64, 64) and g.dtype == np.float32
    assert np.all(g > 0)
    assert np.allclose(g, g.T)                        # symmetric
    peak = np.unravel_index(np.argmax(g), g.shape)
    assert peak in [(31, 31), (31, 32), (32, 31), (32, 32)]


def test_norm_pminmax():
    if not HAVE_TORCH:
        print("SKIP norm_pminmax (torch not installed; infer_dino3d imports it)")
        return
    from infer_dino3d import norm_pminmax
    # too few valid voxels -> all zeros
    assert float(norm_pminmax(np.zeros((16, 16, 16), np.uint8)).max()) == 0.0
    # a ramp normalizes into [0, 1] using the 1/99 percentiles of nonzero voxels
    vol = np.arange(1, 64**3 + 1, dtype=np.float64).reshape(64, 64, 64)
    vol = (vol / vol.max() * 254 + 1).astype(np.uint8)
    out = norm_pminmax(vol)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.std() > 0.2                            # not collapsed


# ---------------------------------------------------------- wedge_extract

def test_wedge_header_contract():
    # wedge PPMs must be readable by the renderers' parser: same contract
    import render_scroll3 as r3
    from wedge_extract import ppm_header_str
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        f.write(ppm_header_str(2880, 400).encode())
        f.write(np.zeros((400, 2880, 6), np.float64).tobytes())
        path = f.name
    try:
        w, h, dim, off = r3.parse_ppm_header(path)
        assert (w, h, dim) == (2880, 400, 6)
        assert os.path.getsize(path) == off + 400 * 2880 * 6 * 8
    finally:
        os.unlink(path)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"ALL_TESTS_PASSED ({len(fns)})")
