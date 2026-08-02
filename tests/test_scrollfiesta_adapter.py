from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "framework/stages/01-segmentation/backends"
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/helena_run_scrollfiesta.py"
FIXTURE = ROOT / "tests/fixtures/asymmetric_coordinate_fixture"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKENDS))

from scrollfiesta.adapter import (  # noqa: E402
    AdapterConfig,
    AdapterError,
    read_tifxyz_coordinates,
    run_adapter,
)
from scrollfiesta.coordinate_transform import (  # noqa: E402
    CoordinateTransformError,
    load_triangle_obj,
    transform_native_zyx_to_canonical_xyz,
)
from scrollfiesta.component_selection import select_component_nearest_point  # noqa: E402
from scrollfiesta.orientation import (  # noqa: E402
    OrientationError,
    orient_triangle_faces,
    orient_with_conflict_quarantine,
)
from scrollfiesta.uv_initialization import (  # noqa: E402
    write_pca_uv_obj,
    write_tutte_uv_obj,
)


def _load_script():
    spec = importlib.util.spec_from_file_location("helena_run_scrollfiesta", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script()


def _write_executable(path: Path, body: str) -> Path:
    """A runnable stand-in for a native binary, on a path that may have spaces.

    The shebang used to be `#!{sys.executable}`, and a shebang cannot hold one:
    the kernel splits the line on the first space, so a checkout under a directory
    whose name has a space in it made it exec the first word and stop. The adapter
    then reported the stand-in as missing. Six tests failed on the name of the
    directory they happened to be checked out into.

    A /bin/sh wrapper instead, which can quote the interpreter.
    """
    script = path.with_name(path.name + ".py")
    script.write_text(body, encoding="utf-8")
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                    encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def _fake_binaries(tmp_path: Path, *, grid_returncode: int = 0, nan_tif: bool = False):
    native = FIXTURE / "native_zyx.obj"
    grid = _write_executable(
        tmp_path / "grid_weld",
        "import json, pathlib, shutil, sys\n"
        f"source = pathlib.Path({str(native)!r})\n"
        "destination = pathlib.Path(sys.argv[2])\n"
        "shutil.copyfile(source, destination)\n"
        "report = {'cubes_processed': 2, 'total_input_verts': 4, "
        "'total_unique_verts': 4, 'total_input_faces': 2, "
        "'total_unique_faces': 2, 'manifold_audit': {'unpaired': 4, "
        "'non_manifold': 0, 'same_dir_pairs': 0, 'manifold_pairs': 1, "
        "'pinch_verts': 0}}\n"
        "pathlib.Path(str(destination) + '.weld_report.json').write_text("
        "json.dumps(report))\n"
        f"raise SystemExit({grid_returncode})\n",
    )
    flatboi = _write_executable(
        tmp_path / "flatboi",
        "import pathlib, sys\n"
        "source = pathlib.Path(sys.argv[1])\n"
        "destination = source.with_name(source.stem + '_flatboi.obj')\n"
        "text = source.read_text()\n"
        "destination.write_text(text + 'vt 0 0\\nvt 1 0\\nvt 0 1\\nvt 1 1\\n')\n",
    )
    coordinate_value = "float('nan')" if nan_tif else "vertices[:, axis]"
    obj2 = _write_executable(
        tmp_path / "vc_obj2tifxyz_legacy",
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
        f"    data = np.full((1, len(vertices)), {coordinate_value}, dtype=np.float32)\n"
        "    tifffile.imwrite(destination / name, data)\n"
        "(destination / 'meta.json').write_text(json.dumps({'format': 'tifxyz'}))\n",
    )
    return grid, flatboi, obj2


def _config(tmp_path: Path, *, grid_returncode: int = 0, nan_tif: bool = False):
    dump = tmp_path / "dump"
    dump.mkdir()
    grid, flatboi, obj2 = _fake_binaries(
        tmp_path, grid_returncode=grid_returncode, nan_tif=nan_tif
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
        # The fixture is two triangles of a deliberately asymmetric quad and the
        # fake ``flatboi`` above performs no optimization at all, so its UV layout
        # is the raw Tutte circle initialization and measures stretch_p95 2.0223.
        # The production default stays at the profile's 2.0; relaxing it here
        # parameterizes a stub, and the flip gate below is left at zero.
        maximum_absolute_stretch_p95=3.0,
    )


def test_asymmetric_coordinate_fixture_roundtrip_is_below_gate(tmp_path: Path) -> None:
    expected = json.loads((FIXTURE / "expected_level0_xyz.json").read_text())
    destination = tmp_path / "canonical.obj"
    mesh = transform_native_zyx_to_canonical_xyz(
        FIXTURE / "native_zyx.obj", destination, level=0
    )
    assert np.max(np.abs(mesh.vertices - np.asarray(expected["vertices_xyz"]))) <= 1e-4
    assert (mesh.faces + 1).tolist() == expected["faces_one_indexed"]
    assert "winding_flip=1" in destination.read_text().splitlines()[0]


def test_transform_rejects_double_application_nan_and_degenerate(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.obj"
    transform_native_zyx_to_canonical_xyz(
        FIXTURE / "native_zyx.obj", canonical, level=0
    )
    with pytest.raises(CoordinateTransformError, match="double transform"):
        transform_native_zyx_to_canonical_xyz(canonical, tmp_path / "twice.obj", level=0)

    nan_obj = tmp_path / "nan.obj"
    nan_obj.write_text("v nan 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
    with pytest.raises(CoordinateTransformError, match="NaN or infinite"):
        load_triangle_obj(nan_obj)

    degenerate = tmp_path / "degenerate.obj"
    degenerate.write_text("v 0 0 0\nv 1 1 1\nv 2 2 2\nf 1 2 3\n")
    with pytest.raises(CoordinateTransformError, match="degenerate"):
        load_triangle_obj(degenerate)


def test_transform_can_drop_only_explicitly_requested_degenerate_triangles(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one-valid-one-degenerate.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 1\nv 2 2 2\n"
        "f 1 2 3\nf 1 4 5\n"
    )
    mesh = load_triangle_obj(source, drop_degenerate_triangles=True)
    assert mesh.source_triangle_count == 2
    assert mesh.dropped_degenerate_triangle_count == 1
    assert len(mesh.faces) == 1


def test_component_selection_uses_frozen_seed_without_fusing_meshes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-components.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "v 100 0 0\nv 101 0 0\nv 100 1 0\n"
        "f 1 2 3\nf 4 5 6\n"
    )
    mesh = load_triangle_obj(source)
    selected, report = select_component_nearest_point(mesh, (100.2, 0.2, 0.0))
    assert report["component_count"] == 2
    assert report["physical_mesh_fusion_performed"] is False
    assert len(selected.faces) == 1
    assert float(selected.vertices[:, 0].min()) == 100.0


def test_orientation_repair_flips_faces_without_changing_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-winding.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "f 1 2 3\nf 2 3 4\n"
    )
    mesh = load_triangle_obj(source)
    repaired, report = orient_triangle_faces(mesh)
    assert report["inconsistent_winding_edge_count_before"] == 1
    assert report["inconsistent_winding_edge_count_after"] == 0
    assert report["flipped_face_count"] == 1
    assert np.array_equal(repaired.vertices, mesh.vertices)
    assert np.array_equal(np.sort(repaired.faces, axis=1), np.sort(mesh.faces, axis=1))


def test_orientation_repair_fails_closed_on_non_manifold_edges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "non-manifold.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 -1 0\nv 0 0 1\n"
        "f 1 2 3\nf 2 1 4\nf 1 2 5\n"
    )
    mesh = load_triangle_obj(source)
    with pytest.raises(OrientationError, match="non-manifold"):
        orient_triangle_faces(mesh)


def test_orientation_quarantine_makes_a_mobius_fixture_orientable() -> None:
    # Six triangles form a minimal strip whose final segment is identified
    # with a twist.  Removing one implicated triangle opens that contradictory
    # micro-patch without moving any retained vertex.
    mesh = load_triangle_obj(
        FIXTURE / "mobius_six_triangles.obj",
    )
    with pytest.raises(OrientationError, match="parity conflict"):
        orient_triangle_faces(mesh)
    repaired, report = orient_with_conflict_quarantine(
        mesh, maximum_quarantined_triangle_fraction=0.2
    )
    assert report["quarantined_triangle_count"] == 1
    assert report["quarantined_triangle_fraction"] == pytest.approx(1 / 6)
    assert report["vertices_moved"] is False
    _, final_orientation = orient_triangle_faces(repaired)
    assert final_orientation["inconsistent_winding_edge_count_after"] == 0


def test_orientation_quarantine_respects_immutable_fraction_cap() -> None:
    mesh = load_triangle_obj(FIXTURE / "mobius_six_triangles.obj")
    with pytest.raises(OrientationError, match="exceed the frozen"):
        orient_with_conflict_quarantine(
            mesh, maximum_quarantined_triangle_fraction=0.01
        )


def test_pca_uv_initialization_preserves_3d_geometry(tmp_path: Path) -> None:
    mesh = load_triangle_obj(FIXTURE / "native_zyx.obj")
    output = tmp_path / "pca.obj"
    report = write_pca_uv_obj(mesh, output)
    roundtrip = load_triangle_obj(output)
    assert np.array_equal(roundtrip.faces, mesh.faces)
    assert np.max(np.abs(roundtrip.vertices - mesh.vertices)) <= 1e-10
    assert report["coordinates_3d_changed"] is False
    assert report["solver_obj_vertex_trailing_fields_stripped"] is True
    assert all(
        len(line.split()) == 4
        for line in output.read_text().splitlines()
        if line.startswith("v ")
    )
    assert sum(line.startswith("vt ") for line in output.read_text().splitlines()) == len(
        mesh.vertices
    )


def test_tutte_uv_initialization_is_flip_free_and_preserves_3d(tmp_path: Path) -> None:
    mesh = load_triangle_obj(FIXTURE / "native_zyx.obj")
    output = tmp_path / "tutte.obj"
    report = write_tutte_uv_obj(mesh, output)
    roundtrip = load_triangle_obj(output)
    assert np.array_equal(roundtrip.faces, mesh.faces)
    assert np.max(np.abs(roundtrip.vertices - mesh.vertices)) <= 1e-10
    assert report["initial_flipped_triangle_count"] == 0
    assert report["initial_degenerate_triangle_count"] == 0
    assert report["solver_obj_vertex_trailing_fields_stripped"] is True
    assert all(
        len(line.split()) == 4
        for line in output.read_text().splitlines()
        if line.startswith("v ")
    )


def test_adapter_vertical_slice_and_tifxyz_coordinate_roundtrip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_adapter(config)
    expected = np.asarray(
        json.loads((FIXTURE / "expected_level0_xyz.json").read_text())["vertices_xyz"]
    )
    coordinates = read_tifxyz_coordinates(result.tifxyz_dir).reshape(-1, 3)
    assert np.max(np.abs(coordinates - expected)) <= 1e-4
    receipt = json.loads(result.adapter_receipt.read_text())
    assert receipt["coordinate_transform"]["application_count"] == 1
    assert receipt["coordinate_transform"]["winding_flipped"] is True
    assert receipt["physical_mesh_fusion_performed"] is False
    assert result.topology["non_manifold_edge_count"] == 0

    with pytest.raises(AdapterError, match="already exists"):
        run_adapter(config)


def test_grid_weld_nonzero_is_hard_failure_even_if_obj_exists(tmp_path: Path) -> None:
    config = _config(tmp_path, grid_returncode=7)
    with pytest.raises(AdapterError, match="grid-weld failed closed with rc=7"):
        run_adapter(config)
    assert (config.output_dir / "welded_source_mesh_zyx.obj").is_file()
    assert not (config.output_dir / "canonical_mesh_xyz.obj").exists()
    failure = json.loads((config.output_dir / "ADAPTER_FAILURE.json").read_text())
    assert failure["physical_mesh_fusion_performed"] is False
    assert failure["commands"][0]["name"] == "grid-weld"
    assert failure["commands"][0]["returncode"] == 7


def test_adapter_rejects_relative_binary_and_nan_tifxyz(tmp_path: Path) -> None:
    config = _config(tmp_path)
    invalid = AdapterConfig(**{**config.__dict__, "grid_weld_bin": Path("grid_weld")})
    with pytest.raises(AdapterError, match="explicit absolute path"):
        run_adapter(invalid)
    assert not invalid.output_dir.exists()

    other = tmp_path / "nan-case"
    other.mkdir()
    nan_config = _config(other, nan_tif=True)
    with pytest.raises(AdapterError, match="NaN, or infinite"):
        run_adapter(nan_config)
    assert (nan_config.output_dir / "ADAPTER_FAILURE.json").is_file()


def test_runner_emits_and_validates_both_hybrid_contracts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = {
        "schema": "campaignx.scrollfiesta_adapter_request.v1",
        "run_id": "RUN-scrollfiesta-asymmetric-fixture-a01",
        "surface_id": "SURF-PHercTEST-asymmetric-fixture-a01",
        "sample_id": "PHercTEST",
        "backend_profile": "segmentation.scrollfiesta-m7-mesh@0.1.0",
        "campaignx_git_commit": "a" * 40,
        "scrollfiesta_repository": "https://github.com/Hob3rMallow/scrollfiesta_public",
        "scrollfiesta_git_commit": "f344c17931b9e264a17c8d760a4c478390133bd4",
        "volume_cartographer_repository": "https://github.com/ScrollPrize/villa",
        "volume_cartographer_git_commit": "b" * 40,
        "runtime": {
            "kind": "NATIVE_BUNDLE",
            "bundle_uri": "file:///fixture/scrollfiesta.tar.zst",
            "bundle_sha256": "c" * 64,
            "platform": "darwin/arm64",
        },
        "ct": {"uri": "s3://fixture/ct.zarr", "sha256_or_etag": "d" * 64},
        "surface_prediction": {
            "uri": "s3://fixture/m7.zarr",
            "sha256_or_etag": "etag:fixture-m7",
        },
        "roi_level0_zyx": [10, 20, 30, 74, 84, 94],
        "level": 0,
        "voxel_size_um_xyz": [9.362, 9.362, 9.362],
        "handedness": "RIGHT_HANDED",
        "dump_dir": str(config.dump_dir),
        "output_dir": str(config.output_dir),
        "binaries": {
            "grid_weld": str(config.grid_weld_bin),
            "flatboi": str(config.flatboi_bin),
            "obj2tifxyz": str(config.obj2tifxyz_bin),
        },
        "cpu_threads": 2,
        "flatboi_iterations": 20,
        "flatboi_energy": "symmetric_dirichlet",
        "tifxyz_step_size": 20,
        "timeout_seconds": 30,
        # See ``_config``: the two-triangle fixture plus a no-op ``flatboi`` stub
        # measures stretch_p95 2.0223 against the production gate of 2.0.
        "maximum_absolute_stretch_p95": 3.0,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    receipt_path = RUNNER.execute_request(request_path)
    receipt = json.loads(receipt_path.read_text())
    surface = json.loads((config.output_dir / "SURFACE_ARTIFACT.json").read_text())
    assert receipt["status"] == "SUCCEEDED"
    assert receipt["surface_artifact"]["surface_id"] == request["surface_id"]
    assert surface["validation_state"] == "PROVISIONAL"
    assert surface["ink_used"] is False
    assert receipt["ink_used"] is False
