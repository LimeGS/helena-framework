from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/helena_prepare_scrollfiesta_cubes_compact.py"


def load_module():
    spec = spec_from_file_location("scrollfiesta_compact_stage", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_tar(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "if '-cf' in sys.argv:\n"
        "    output=pathlib.Path(sys.argv[sys.argv.index('-cf')+1])\n"
        "    output.write_bytes(b'verified-test-archive')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_archive_compaction_preserves_only_final_tree(tmp_path: Path) -> None:
    module = load_module()
    dump = tmp_path / "dump"
    cube_id = "z00000_y00000_x00000"
    cube = dump / cube_id
    final_name = f"{cube_id}_step12_final"
    final = cube / final_name
    intermediate = cube / f"{cube_id}_step4_bpa"
    final.mkdir(parents=True)
    intermediate.mkdir()
    obj = final / f"{final_name}_all.obj"
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    (intermediate / "temporary.obj").write_text("v 1 0 0\n", encoding="utf-8")
    archives = tmp_path / "archives"
    archives.mkdir()

    archive = module.archive_and_compact_cube(
        dump_dir=dump,
        cube_id=cube_id,
        archives_dir=archives,
        tar_bin=fake_tar(tmp_path / "tar"),
    )

    assert archive.read_bytes() == b"verified-test-archive"
    assert obj.is_file()
    assert not intermediate.exists()
    assert [item.name for item in cube.iterdir()] == [final_name]


def test_file_artifact_locks_runtime_binary_bytes(tmp_path: Path) -> None:
    module = load_module()
    binary = tmp_path / "cube_mesh"
    binary.write_bytes(b"fixed-clipper2z-runtime")

    artifact = module.file_artifact(binary)

    assert artifact["bytes"] == 23
    assert artifact["sha256"] == (
        "208475f7c934d1d8c56e7120b6fda50e0d0e8a247bf1d96313d04a4781b774f7"
    )
    assert artifact["uri"] == binary.resolve().as_uri()
