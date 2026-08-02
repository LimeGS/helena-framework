"""stb.strips (PLAN_V3.md, "Pinned regression numbers", test_strips.py):
export the 4 v2 windows (11000, 11300, 10600, 13400) + the v1 window
(12750) as strips; load them back; qualify_strip passes gates a/b for
all 5; a deliberately label-shuffled strip FAILS gate b (regression
guard proving qualify_strip's gate-b check actually depends on the
classes' spatial separation, not a no-op).

Offline: only reads configs/pherc0332.json, fixtures/band_r1145_200_xyz.npz
and fixtures/windows_v2.json (for pitch/stratum/kappa provenance on the 4
v2 windows).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from stb import gates as gates_mod
from stb import reference as reference_mod
from stb import strips as strips_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"

V1_WINDOW_START = 12750  # benchmark_core.C0 -- the original v1 seed window


@pytest.fixture(scope="module")
def windows_v2_by_start():
    with open(FIXTURES / "windows_v2.json") as f:
        data = json.load(f)
    return {w["start"]: w for w in data["windows"]}


@pytest.fixture(scope="module")
def strip_specs(windows_v2_by_start):
    """(start, pitch, meta) for the 5 windows this pack covers."""
    specs = []
    for start, w in windows_v2_by_start.items():
        specs.append((start, w.get("p2_ct"), {"stratum": w.get("stratum"), "kappa": w.get("kappa")}))
    specs.append((V1_WINDOW_START, None, {"note": "v1 anchor window (benchmark_core.C0:C1)"}))
    return sorted(specs, key=lambda s: s[0])


@pytest.fixture(scope="module")
def exported_strips(tmp_path_factory, resolved_cfg_332, band_332, strip_specs):
    xyz, valid, _row0 = band_332
    cfg = resolved_cfg_332
    out_dir = tmp_path_factory.mktemp("strips")
    paths = {}
    for start, pitch, meta in strip_specs:
        ref = reference_mod.reference_at(xyz, valid, start, cfg)
        g = gates_mod.coverage_and_gates_ab(ref, cfg)
        path = out_dir / f"strip_s{start:05d}.npz"
        strips_mod.export_strip(ref, start, cfg, g, path, pitch=pitch, meta=meta)
        paths[start] = path
    return paths


def test_export_and_qualify_all_5_windows(exported_strips):
    for start, path in exported_strips.items():
        strip = strips_mod.load_strip(path)
        assert strip["start"] == start
        result = strips_mod.qualify_strip(strip)
        assert result.get("gate_a_pass") is True, f"gate a failed for start={start}: {result}"
        assert result.get("gate_b_pass") is True, f"gate b failed for start={start}: {result}"


def test_label_shuffled_strip_fails_gate_b(exported_strips):
    # Any one window suffices to demonstrate the regression guard; use
    # the first v2 window.
    start = sorted(exported_strips)[0]
    strip = strips_mod.load_strip(exported_strips[start])

    # Sanity: the strip as exported passes gate b.
    assert strips_mod.qualify_strip(strip)["gate_b_pass"] is True

    # Label shuffle: merge classes +1 and -1 into their union so both
    # trees become identical. A genuine class-e oracle point (used by
    # gate b's wrong-side check) is then also present in the OPPOSITE
    # class's tree at distance 0, creating exact ties that
    # score_prediction's nearest_cls argmin resolves in favor of the
    # wrong class for one of the two directions -- this is not a
    # relabeling/rename (which gate b is provably invariant to, since its
    # oracle-vs-opposite-class check is self-matching regardless of
    # labels) but a genuine corruption of the classes' spatial
    # separation, which is exactly what gate b exists to catch.
    shuffled = copy.copy(strip)
    union_rows = np.concatenate([strip["rows_class_1"], strip["rows_class_-1"]])
    union_cols = np.concatenate([strip["cols_class_1"], strip["cols_class_-1"]])
    union_pts = np.concatenate([strip["pts_class_1"], strip["pts_class_-1"]], axis=0)
    shuffled["rows_class_1"] = shuffled["rows_class_-1"] = union_rows
    shuffled["cols_class_1"] = shuffled["cols_class_-1"] = union_cols
    shuffled["pts_class_1"] = shuffled["pts_class_-1"] = union_pts

    result = strips_mod.qualify_strip(shuffled)
    assert result["gate_b_pass"] is False
