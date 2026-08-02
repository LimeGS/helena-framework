"""FIX-12: the Stage 02 gates that were declared but had no producer.

Each test here pins one gate that could not previously fire:

* an OBJ with a flipped triangle must stop the adapter;
* a 73.9% ABF++ render log must stop the render gate;
* a window measured at ``valid=0.7855`` must stop the export;
* a component that crosses ``|dwinding| >= 1`` must be split in two;
* ``self_intersection_count`` with no weld-report producer must report
  ``UNMEASURED``, and ``UNMEASURED`` must not satisfy ``GEOMETRY_VALIDATED``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "framework/stages/01-segmentation/backends"
FLATTENING = ROOT / "framework/stages/02-flattening/scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKENDS))
sys.path.insert(0, str(FLATTENING))

from flattening_gates import (  # noqa: E402
    FlatteningGateError,
    evaluate_raster_gate,
    evaluate_window_validity,
    label_winding_aware_components,
    parse_rasterized_fractions,
    winding_step_report,
)
from framework.contracts.hybrid_surface_contracts import (  # noqa: E402
    HybridContractValidationError,
    validate_hybrid_contract,
)
from scrollfiesta.topology import (  # noqa: E402
    UNMEASURED,
    topology_metrics,
    weld_report_self_intersection_count,
)
from scrollfiesta.coordinate_transform import load_triangle_obj  # noqa: E402
from scrollfiesta.uv_initialization import (  # noqa: E402
    UvDistortionError,
    load_uv_mapped_obj,
    uv_distortion_metrics,
)


# --------------------------------------------------------------------------
# F-1: post-SLIM UV flips and stretch
# --------------------------------------------------------------------------

_FLAT_GRID_OBJ = """v 0 0 0
v 1 0 0
v 2 0 0
v 0 1 0
v 1 1 0
v 2 1 0
vt 0 0
vt 1 0
vt 2 0
vt 0 1
vt 1 1
vt 2 1
f 1/1 2/2 5/5
f 1/1 5/5 4/4
f 2/2 3/3 6/6
f 2/2 6/6 5/5
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_isometric_flattening_measures_no_flip_and_unit_stretch(tmp_path: Path) -> None:
    vertices, faces, uv = load_uv_mapped_obj(_write(tmp_path / "flat.obj", _FLAT_GRID_OBJ))
    metrics = uv_distortion_metrics(vertices, faces, uv)
    assert metrics["uv_flipped_triangle_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0
    assert metrics["stretch_p50"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["stretch_max"] == pytest.approx(1.0, abs=1e-9)


def test_one_flipped_triangle_is_counted(tmp_path: Path) -> None:
    # Reverse the UV winding of the last triangle only.
    flipped = _FLAT_GRID_OBJ.replace("f 2/2 6/6 5/5\n", "f 2/2 5/5 6/6\n")
    vertices, faces, uv = load_uv_mapped_obj(_write(tmp_path / "flipped.obj", flipped))
    metrics = uv_distortion_metrics(vertices, faces, uv)
    assert metrics["uv_flipped_triangle_count"] == 1


def test_anisotropic_flattening_reports_stretch_above_one(tmp_path: Path) -> None:
    # A 3:1 anisotropic UV scaling is area-preserving after normalization, so
    # only a genuine shape distortion measurement can see it.
    stretched = "\n".join(
        line
        if not line.startswith("vt ")
        else "vt " + " ".join(
            str(float(value) * (3.0 if index == 0 else 1 / 3.0))
            for index, value in enumerate(line.split()[1:])
        )
        for line in _FLAT_GRID_OBJ.splitlines()
    ) + "\n"
    vertices, faces, uv = load_uv_mapped_obj(_write(tmp_path / "aniso.obj", stretched))
    metrics = uv_distortion_metrics(vertices, faces, uv)
    assert metrics["stretch_p95"] == pytest.approx(3.0, rel=1e-9)
    assert metrics["uv_flipped_triangle_count"] == 0


FIXTURE = ROOT / "tests/fixtures/asymmetric_coordinate_fixture"


def _executable(path: Path, body: str) -> Path:
    """As in test_scrollfiesta_adapter: a shebang cannot hold a space, and this
    checkout lives under a directory that has one, so `#!{sys.executable}` sent the
    kernel looking for an interpreter named after the first word of the path."""
    script = path.with_name(path.name + ".py")
    script.write_text(body, encoding="utf-8")
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                    encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def _adapter_stubs(tmp_path: Path, *, flip_one_uv_triangle: bool):
    """Stub binaries around the shared asymmetric fixture.

    ``flatboi`` rewrites the solver input into a real UV-mapped OBJ.  With
    ``flip_one_uv_triangle`` it reverses the UV winding of exactly one face,
    which is the injected defect FIX-12 requires the adapter to reject.
    """

    native = FIXTURE / "native_zyx.obj"
    grid = _executable(
        tmp_path / "grid_weld",
        "import json, pathlib, shutil, sys\n"
        f"shutil.copyfile(pathlib.Path({str(native)!r}), pathlib.Path(sys.argv[2]))\n"
        "report = {'cubes_processed': 2, 'total_input_verts': 4, "
        "'total_unique_verts': 4, 'total_input_faces': 2, "
        "'total_unique_faces': 2, 'manifold_audit': {'unpaired': 4, "
        "'non_manifold': 0, 'same_dir_pairs': 0, 'manifold_pairs': 1, "
        "'pinch_verts': 0}}\n"
        "pathlib.Path(str(sys.argv[2]) + '.weld_report.json').write_text("
        "json.dumps(report))\n",
    )
    flatboi = _executable(
        tmp_path / "flatboi",
        "import pathlib, sys\n"
        f"flip = {flip_one_uv_triangle!r}\n"
        "source = pathlib.Path(sys.argv[1])\n"
        "destination = source.with_name(source.stem + '_flatboi.obj')\n"
        "lines = source.read_text().splitlines()\n"
        "out = []\n"
        "seen_face = 0\n"
        "for line in lines:\n"
        "    if line.startswith('f ') and flip:\n"
        "        seen_face += 1\n"
        "        if seen_face == 1:\n"
        "            a, b, c = line.split()[1:]\n"
        "            out.append('f ' + a + ' ' + c + ' ' + b)\n"
        "            continue\n"
        "    out.append(line)\n"
        "destination.write_text('\\n'.join(out) + '\\n')\n",
    )
    obj2 = _executable(
        tmp_path / "obj2tifxyz",
        "import json, pathlib, sys\n"
        "import numpy as np\n"
        "import tifffile\n"
        "source = pathlib.Path(sys.argv[1])\n"
        "destination = pathlib.Path(sys.argv[2])\n"
        "destination.mkdir()\n"
        "vertices = np.asarray([[float(v) for v in line.split()[1:4]] "
        "for line in source.read_text().splitlines() if line.startswith('v ')], "
        "dtype=np.float32)\n"
        "for axis, name in enumerate(('x.tif', 'y.tif', 'z.tif')):\n"
        "    tifffile.imwrite(destination / name, "
        "np.full((1, len(vertices)), vertices[:, axis], dtype=np.float32))\n"
        "(destination / 'meta.json').write_text(json.dumps({'format': 'tifxyz'}))\n",
    )
    return grid, flatboi, obj2


def _adapter_config(tmp_path: Path, *, flip_one_uv_triangle: bool):
    from scrollfiesta.adapter import AdapterConfig

    dump = tmp_path / "dump"
    dump.mkdir()
    grid, flatboi, obj2 = _adapter_stubs(
        tmp_path, flip_one_uv_triangle=flip_one_uv_triangle
    )
    return AdapterConfig(
        dump_dir=dump.resolve(),
        output_dir=(tmp_path / "run").resolve(),
        grid_weld_bin=grid,
        flatboi_bin=flatboi,
        obj2tifxyz_bin=obj2,
        level=0,
        voxel_size_um_xyz=(9.362, 9.362, 9.362),
        timeout_seconds=30,
        maximum_absolute_stretch_p95=3.0,
    )


def test_adapter_rejects_a_flattened_obj_with_a_flipped_triangle(
    tmp_path: Path,
) -> None:
    """FIX-12 acceptance: injecting a flipped OBJ produces an adapter error."""

    from scrollfiesta.adapter import AdapterError, run_adapter

    config = _adapter_config(tmp_path, flip_one_uv_triangle=True)
    with pytest.raises(AdapterError, match="UV flipped triangle count 1"):
        run_adapter(config)
    measured = json.loads(
        (config.output_dir / "UV_DISTORTION.json").read_text(encoding="utf-8")
    )
    assert measured["uv_flipped_triangle_count"] == 1
    failure = json.loads(
        (config.output_dir / "ADAPTER_FAILURE.json").read_text(encoding="utf-8")
    )
    assert "flipped" in failure["message"]
    assert not (config.output_dir / "tifxyz").exists()


def test_the_same_adapter_run_succeeds_without_the_injected_flip(
    tmp_path: Path,
) -> None:
    from scrollfiesta.adapter import run_adapter

    result = run_adapter(_adapter_config(tmp_path, flip_one_uv_triangle=False))
    assert result.uv_distortion["uv_flipped_triangle_count"] == 0
    assert result.uv_distortion_report.is_file()
    # The measurement must cross the stage boundary, not stay in a side file.
    for field in ("uv_flipped_triangle_count", "stretch_p50", "stretch_p95", "stretch_max"):
        assert field in result.topology


def test_the_adapter_refuses_a_stretch_gate_below_isometry(tmp_path: Path) -> None:
    from scrollfiesta.adapter import AdapterConfig, AdapterError, run_adapter

    config = _adapter_config(tmp_path, flip_one_uv_triangle=False)
    with pytest.raises(AdapterError, match="at least 1.0"):
        run_adapter(
            AdapterConfig(
                **{
                    **config.__dict__,
                    "output_dir": tmp_path / "run2",
                    "maximum_absolute_stretch_p95": 0.5,
                }
            )
        )


def test_uv_parser_refuses_an_obj_without_texture_indices(tmp_path: Path) -> None:
    path = _write(tmp_path / "no-uv.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1 2 3\n")
    with pytest.raises(UvDistortionError, match="carries no UV index"):
        load_uv_mapped_obj(path)


def test_degenerate_uv_triangle_is_counted_not_silently_dropped(tmp_path: Path) -> None:
    collapsed = _FLAT_GRID_OBJ.replace("vt 1 1\n", "vt 1 0\n")
    vertices, faces, uv = load_uv_mapped_obj(_write(tmp_path / "collapsed.obj", collapsed))
    metrics = uv_distortion_metrics(vertices, faces, uv)
    assert metrics["uv_degenerate_triangle_count"] >= 1


# --------------------------------------------------------------------------
# F-2: the ABF++ rasterized-point fraction
# --------------------------------------------------------------------------

WORST_CASE_LOG = """Loading surface
Rasterized 11647 / 15750 points (73.9492%)
Done
"""

FLEET_TYPICAL_LOG = "Rasterized 4581 / 4900 points (93.4898%)\n"

PASSING_LOG = "Rasterized 9650 / 10000 points (96.5000%)\n"


def test_parses_the_rasterized_line() -> None:
    (row,) = parse_rasterized_fractions(WORST_CASE_LOG)
    assert row["rasterized_points"] == 11647
    assert row["output_points"] == 15750
    assert row["valid_raster_fraction"] == pytest.approx(0.739492, abs=1e-6)


def test_worst_case_739_percent_render_fails_the_gate() -> None:
    with pytest.raises(FlatteningGateError, match="73.9492%"):
        evaluate_raster_gate(WORST_CASE_LOG)


def test_fleet_typical_9349_percent_render_passes_the_evidence_based_floor() -> None:
    """The fleet's own p05 must not be treated as a defect.

    93.4898% is the 5th percentile of all 166 archived render logs (median
    0.9429).  The a-priori 0.95 of FIX-12 would have failed 144 of them.  The
    floor exists to catch gross degradation, not to re-render normal work.
    """

    result = evaluate_raster_gate(FLEET_TYPICAL_LOG)
    assert result["passed"] is True
    assert result["valid_raster_fraction"] == pytest.approx(0.934898, abs=1e-6)


def test_965_percent_render_passes_and_is_receipted() -> None:
    result = evaluate_raster_gate(PASSING_LOG)
    assert result["passed"] is True
    assert result["valid_raster_fraction"] == pytest.approx(0.965)
    assert result["minimum_valid_raster_fraction"] == 0.90


def test_the_floor_isolates_the_single_genuine_outlier() -> None:
    """0.90 passes 165 of the 166 archived logs and fails only the 73.9% one.

    Pinning both sides keeps the floor honest: moving it up re-breaks the
    fleet-typical case above, moving it down stops catching real zero fill.
    """

    assert evaluate_raster_gate(PASSING_LOG)["passed"] is True
    with pytest.raises(FlatteningGateError):
        evaluate_raster_gate(WORST_CASE_LOG)


def test_a_log_without_the_line_is_unmeasured_and_fails_closed() -> None:
    with pytest.raises(FlatteningGateError, match="UNMEASURED"):
        evaluate_raster_gate("Loading surface\nDone\n")


def test_the_worst_observation_decides_when_a_log_has_several() -> None:
    with pytest.raises(FlatteningGateError, match="73.9492%"):
        evaluate_raster_gate(PASSING_LOG + WORST_CASE_LOG)


def test_a_log_whose_percentage_contradicts_its_counts_is_rejected() -> None:
    with pytest.raises(FlatteningGateError, match="disagrees"):
        evaluate_raster_gate("Rasterized 50 / 100 points (99.0000%)\n")


# --------------------------------------------------------------------------
# F-3: window validity and the -1.0 sentinel guard band
# --------------------------------------------------------------------------


def _window_with_valid_fraction(fraction: float, side: int = 40) -> tuple:
    valid = np.zeros((side, side), dtype=bool)
    window = np.ones((side, side), dtype=bool)
    total = side * side
    valid.reshape(-1)[: int(round(fraction * total))] = True
    return window, valid


def test_pherc1545_window_at_valid_07855_is_refused() -> None:
    window, valid = _window_with_valid_fraction(0.7855)
    with pytest.raises(FlatteningGateError, match=r"only 0\.785\d valid"):
        evaluate_window_validity(window, valid)


@pytest.mark.parametrize(
    "sample, fraction",
    [("PHerc1545", 0.7855), ("PHerc125", 0.7998), ("PHerc191", 0.8015), ("PHerc268", 0.8864)],
)
def test_every_measured_historical_window_is_refused(sample: str, fraction: float) -> None:
    window, valid = _window_with_valid_fraction(fraction)
    with pytest.raises(FlatteningGateError, match="below the"):
        evaluate_window_validity(window, valid)
    assert sample


def test_a_fully_valid_window_passes_and_is_intersected_with_valid() -> None:
    valid = np.ones((10, 10), dtype=bool)
    window = np.zeros((10, 10), dtype=bool)
    window[2:8, 2:8] = True
    exported, report = evaluate_window_validity(window, valid)
    assert report["passed"] is True
    assert report["valid_fraction"] == 1.0
    assert int(exported.sum()) == 36
    assert not bool((exported & ~valid).any())


def test_a_quad_touching_the_sentinel_guard_band_is_refused() -> None:
    """A 99%-valid window still fails when it hugs the sentinel."""

    valid = np.ones((20, 20), dtype=bool)
    valid[0, 0] = False
    window = np.zeros((20, 20), dtype=bool)
    window[0:10, 0:10] = True  # 99/100 valid: the fraction gate alone passes
    with pytest.raises(FlatteningGateError, match="within 1 cell"):
        evaluate_window_validity(window, valid)


def test_moving_the_window_off_the_guard_band_restores_the_export() -> None:
    valid = np.ones((20, 20), dtype=bool)
    valid[0, 0] = False
    window = np.zeros((20, 20), dtype=bool)
    window[5:9, 5:9] = True
    exported, report = evaluate_window_validity(window, valid)
    assert report["passed"] is True
    assert report["quads_within_audit_radius_of_sentinel"] == 0
    assert int(exported.sum()) == 16


def test_the_exported_mask_never_carries_an_invalid_quad() -> None:
    valid = np.ones((20, 20), dtype=bool)
    valid[19, 19] = False
    window = np.zeros((20, 20), dtype=bool)
    window[2:18, 2:18] = True
    exported, _ = evaluate_window_validity(window, valid)
    assert not bool((exported & ~valid).any())


# --------------------------------------------------------------------------
# F-4: components must not cross a sheet jump
# --------------------------------------------------------------------------


def test_a_component_crossing_a_winding_step_is_split_in_two() -> None:
    mask = np.ones((4, 6), dtype=bool)
    winding = np.zeros((4, 6), dtype=np.float64)
    winding[:, 3:] = 1.0  # a full turn of the spiral between column 2 and 3
    labels, count = label_winding_aware_components(mask, winding)
    assert count == 2
    assert len(set(labels[:, :3].reshape(-1).tolist())) == 1
    assert len(set(labels[:, 3:].reshape(-1).tolist())) == 1
    assert labels[0, 2] != labels[0, 3]


def test_the_same_mask_is_one_component_without_the_winding_cut() -> None:
    mask = np.ones((4, 6), dtype=bool)
    _, count = label_winding_aware_components(mask, None)
    assert count == 1


def test_a_sub_turn_winding_gradient_does_not_split() -> None:
    mask = np.ones((4, 6), dtype=bool)
    winding = np.tile(np.linspace(0.0, 0.9, 6), (4, 1))
    _, count = label_winding_aware_components(mask, winding)
    assert count == 1


def test_winding_labels_match_four_connectivity_on_a_flat_field() -> None:
    from scipy import ndimage

    rng = np.random.default_rng(20260724)
    mask = rng.random((30, 30)) > 0.4
    winding = np.zeros((30, 30), dtype=np.float64)
    labels, count = label_winding_aware_components(mask, winding)
    reference, expected = ndimage.label(
        mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    assert count == expected
    assert np.array_equal(labels, reference)


def test_winding_step_report_flags_and_clears_a_crossing() -> None:
    mask = np.ones((4, 6), dtype=bool)
    winding = np.zeros((4, 6), dtype=np.float64)
    winding[:, 3:] = 1.0
    assert winding_step_report(mask, winding)["status"] == "FAIL"
    assert winding_step_report(mask[:, :3], winding[:, :3])["status"] == "PASS"
    assert winding_step_report(mask, None)["status"] == "UNMEASURED"


# --------------------------------------------------------------------------
# F-6: self_intersection_count must not be a literal zero
# --------------------------------------------------------------------------


def _mesh(tmp_path: Path):
    return load_triangle_obj(
        _write(tmp_path / "tri.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    )


def test_without_a_weld_report_the_count_is_unmeasured(tmp_path: Path) -> None:
    metrics = topology_metrics(
        _mesh(tmp_path), voxel_size_um_xyz=(9.362, 9.362, 9.362)
    )
    assert metrics["self_intersection_count"] == UNMEASURED


def test_a_weld_report_without_the_field_is_still_unmeasured(tmp_path: Path) -> None:
    report = {"manifold_audit": {"non_manifold": 0, "same_dir_pairs": 0}}
    metrics = topology_metrics(
        _mesh(tmp_path),
        voxel_size_um_xyz=(9.362, 9.362, 9.362),
        weld_report=report,
    )
    assert metrics["self_intersection_count"] == UNMEASURED


def test_a_weld_report_with_a_real_count_is_carried_through(tmp_path: Path) -> None:
    for report, expected in (
        ({"self_intersection_count": 0}, 0),
        ({"self_intersection_count": 7}, 7),
        ({"manifold_audit": {"self_intersections": 3}}, 3),
    ):
        assert weld_report_self_intersection_count(report) == expected
    metrics = topology_metrics(
        _mesh(tmp_path),
        voxel_size_um_xyz=(9.362, 9.362, 9.362),
        weld_report={"self_intersection_count": 0},
    )
    assert metrics["self_intersection_count"] == 0


def test_a_negative_weld_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        weld_report_self_intersection_count({"self_intersection_count": -1})


def _surface_artifact(**metric_overrides) -> dict:
    document = json.loads(
        (ROOT / "tests/fixtures/hybrid_contracts/surface_artifact.valid.json").read_text(
            encoding="utf-8"
        )
    )
    document["validation_state"] = "GEOMETRY_VALIDATED"
    document["metrics"].update(
        {
            "uv_flipped_triangle_count": 0,
            "uv_degenerate_triangle_count": 0,
            "stretch_p50": 1.07,
            "stretch_p95": 1.25,
            "stretch_max": 1.9,
        }
    )
    document["metrics"].update(metric_overrides)
    return document


def test_unmeasured_self_intersection_cannot_be_geometry_validated() -> None:
    document = _surface_artifact(self_intersection_count=UNMEASURED)
    with pytest.raises(
        HybridContractValidationError, match="requires a measured value"
    ):
        validate_hybrid_contract(
            document, expected_contract="campaignx.surface_artifact.v2"
        )


def test_unmeasured_still_transports_on_a_provisional_artifact() -> None:
    document = _surface_artifact(self_intersection_count=UNMEASURED)
    document["validation_state"] = "PROVISIONAL"
    validate_hybrid_contract(document, expected_contract="campaignx.surface_artifact.v2")


def test_a_fully_measured_artifact_can_be_geometry_validated() -> None:
    validate_hybrid_contract(
        _surface_artifact(), expected_contract="campaignx.surface_artifact.v2"
    )


def test_geometry_validated_requires_the_distortion_measurements() -> None:
    document = _surface_artifact()
    for field in ("stretch_p95", "uv_flipped_triangle_count"):
        stripped = json.loads(json.dumps(document))
        del stripped["metrics"][field]
        with pytest.raises(
            HybridContractValidationError, match="distortion measurements"
        ):
            validate_hybrid_contract(
                stripped, expected_contract="campaignx.surface_artifact.v2"
            )


def test_geometry_validated_rejects_a_flipped_triangle() -> None:
    document = _surface_artifact(uv_flipped_triangle_count=1)
    with pytest.raises(
        HybridContractValidationError, match="zero flipped and zero"
    ):
        validate_hybrid_contract(
            document, expected_contract="campaignx.surface_artifact.v2"
        )


def test_the_schema_rejects_a_stretch_below_one() -> None:
    document = _surface_artifact(stretch_p50=0.5)
    document["validation_state"] = "PROVISIONAL"
    with pytest.raises(HybridContractValidationError, match="stretch_p50"):
        validate_hybrid_contract(
            document, expected_contract="campaignx.surface_artifact.v2"
        )


# --------------------------------------------------------------------------
# The declared profile gates now name their producers
# --------------------------------------------------------------------------


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"under_test_{name}", FLATTENING / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_render_crop_refuses_a_739_percent_log(tmp_path: Path, monkeypatch) -> None:
    """The gate runs before any pixel is written."""

    import tifffile

    crop = _load_script("crop_render_window.py")
    source = tmp_path / "tiffs"
    source.mkdir()
    for index in range(3):
        tifffile.imwrite(source / f"{index:05d}.tif", np.zeros((64, 64), np.uint16))
    log = tmp_path / "launcher.log"
    log.write_text(WORST_CASE_LOG, encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crop_render_window.py",
            "--sample-id", "PHerc826",
            "--input-dir", str(source),
            "--output-dir", str(output),
            "--pixel-um", "9.362",
            "--size-mm", "0.3",
            "--render-log", str(log),
        ],
    )
    with pytest.raises(FlatteningGateError, match="73.9492%"):
        crop.main()
    assert not output.exists()


def test_the_render_crop_accepts_a_965_percent_log_and_receipts_it(
    tmp_path: Path, monkeypatch
) -> None:
    import tifffile

    crop = _load_script("crop_render_window.py")
    source = tmp_path / "tiffs"
    source.mkdir()
    for index in range(3):
        tifffile.imwrite(source / f"{index:05d}.tif", np.zeros((64, 64), np.uint16))
    log = tmp_path / "launcher.log"
    log.write_text(PASSING_LOG, encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crop_render_window.py",
            "--sample-id", "PHerc826",
            "--input-dir", str(source),
            "--output-dir", str(output),
            "--pixel-um", "9.362",
            "--size-mm", "0.3",
            "--render-log", str(log),
        ],
    )
    assert crop.main() == 0
    receipt = json.loads(
        (output / "PHYSICAL_CROP_RECEIPT.json").read_text(encoding="utf-8")
    )
    gate = receipt["render_quality_gate"]
    assert gate["passed"] is True
    assert gate["valid_raster_fraction"] == pytest.approx(0.965)
    assert gate["minimum_valid_raster_fraction"] == 0.90
    assert len(gate["render_log_sha256"]) == 64


def test_the_render_receipt_can_reconstruct_the_physical_depth() -> None:
    """FIX-12/F-7: 65 slices x 9.362 um is 608.5 um, about 4.1 windings."""

    source = (FLATTENING / "run.py").read_text(encoding="utf-8")
    assert '"voxel_size_um": voxel_size_um,' in source
    assert '"slice_step": slice_step,' in source
    assert '"depth_extent_um": depth_extent_um,' in source
    assert "--voxel-size-um" in source
    assert "--slice-step" in source
    # The arithmetic the receipt now makes derivable.
    depth_extent_um = 65 * 1.0 * 9.362
    assert depth_extent_um == pytest.approx(608.53, abs=0.01)
    for dr_per_winding_vox in (15.53, 15.79):
        windings = depth_extent_um / (dr_per_winding_vox * 9.362)
        assert 4.0 < windings < 4.2


def test_the_new_profile_keeps_every_frozen_gate_and_adds_an_absolute_one() -> None:
    frozen = json.loads(
        (
            ROOT
            / "framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.2.json"
        ).read_text(encoding="utf-8")
    )
    updated = json.loads(
        (
            ROOT
            / "framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.3.json"
        ).read_text(encoding="utf-8")
    )
    for name, value in frozen["hard_gates"].items():
        assert updated["hard_gates"][name] == value, name
    assert updated["hard_gates"]["maximum_absolute_stretch_p95"] == 2.0
    assert updated["hard_gates"]["minimum_valid_raster_fraction"] == 0.90
    assert updated["hard_gates"]["minimum_window_valid_fraction"] == 0.95
    for gate in (
        "uv_flipped_triangles",
        "maximum_absolute_stretch_p95",
        "self_intersections",
        "minimum_valid_raster_fraction",
        "minimum_window_valid_fraction",
        "maximum_roi_winding_step",
    ):
        assert updated["gate_producers"][gate]
