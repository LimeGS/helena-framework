"""Fail-closed tests for the exact Villa runtime input packager."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "containers/images/scrollfiesta/scripts/package_villa_runtime.py"
LOCK = ROOT / "containers/images/scrollfiesta/locks/source-lock.json"
MODULE = runpy.run_path(SCRIPT)
PackagingError = MODULE["PackagingError"]
package = MODULE["package"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "villa-source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "fixture@example.invalid")
    _git(source, "config", "user.name", "Fixture")
    (source / "README").write_text("villa fixture\n")
    _git(source, "add", "README")
    _git(source, "commit", "-qm", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")

    lock = json.loads(LOCK.read_text())
    lock["volume_cartographer"]["commit"] = commit
    lock_path = tmp_path / "source-lock.json"
    lock_path.write_text(json.dumps(lock))

    build = tmp_path / "villa-build"
    (build / "bin").mkdir(parents=True)
    for name in ("flatboi", "vc_obj2tifxyz_legacy"):
        path = build / "bin" / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    toolchain_path = tmp_path / "toolchain.json"
    toolchain_path.write_text(
        json.dumps(
            {
                "schema": "campaignx.villa_toolchain_receipt.v1",
                "source_commit": commit,
                "source_tree": tree,
                "build_root": str(build),
                "toolchain": {
                    "c_compiler": "gcc 13.2.0",
                    "cxx_compiler": "g++ 13.2.0",
                    "cmake_version": "3.28.3",
                    "build_type": "Release",
                    "build_command": "cmake --build build --target flatboi vc_obj2tifxyz_legacy",
                },
            }
        )
    )

    inspector = tmp_path / "fake-ldd"
    inspector.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --version ]; then echo 'ldd fixture 1.0'; exit 0; fi\n"
        "echo 'libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x1)'\n"
        "echo '/lib64/ld-linux-x86-64.so.2 (0x2)'\n"
    )
    inspector.chmod(0o755)
    return {
        "source": source,
        "build": build,
        "flatboi": build / "bin/flatboi",
        "obj2tifxyz": build / "bin/vc_obj2tifxyz_legacy",
        "toolchain": toolchain_path,
        "lock": lock_path,
        "inspector": inspector,
    }


def _package(inputs: dict[str, Path], output: Path) -> dict:
    return package(
        source=inputs["source"],
        build_root=inputs["build"],
        flatboi=inputs["flatboi"],
        obj2tifxyz=inputs["obj2tifxyz"],
        toolchain_receipt=inputs["toolchain"],
        source_lock=inputs["lock"],
        inspector=inputs["inspector"],
        output=output,
    )


def test_packages_only_two_tools_and_generated_metadata(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output = tmp_path / "villa-runtime"
    manifest = _package(inputs, output)
    assert manifest["schema"] == "campaignx.villa_runtime_bundle.v1"
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        "VILLA_RUNTIME_MANIFEST.json",
        "VILLA_RUNTIME_PACKAGING_RECEIPT.json",
        "bin/flatboi",
        "bin/vc_obj2tifxyz_legacy",
    ]
    for name, entry in manifest["artifacts"].items():
        payload = output / entry["path"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry["sha256"]
        assert payload.stat().st_mode & 0o222 == 0
    receipt = json.loads((output / "VILLA_RUNTIME_PACKAGING_RECEIPT.json").read_text())
    assert receipt["status"] == "PACKAGED_AND_LINKAGE_VERIFIED"
    assert receipt["contains_only_required_executables"] is True
    assert receipt["dependency_inspection"]["flatboi"]["forbidden_path_violations"] == []
    receipt_path = output / manifest["packaging_receipt"]["path"]
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == manifest["packaging_receipt"]["sha256"]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output = tmp_path / "villa-runtime"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("preserve")
    try:
        _package(inputs, output)
    except PackagingError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("existing output was accepted")
    assert sentinel.read_text() == "preserve"


def test_rejects_source_or_toolchain_identity_drift(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    (inputs["source"] / "README").write_text("dirty\n")
    try:
        _package(inputs, tmp_path / "dirty-output")
    except PackagingError as exc:
        assert "dirty" in str(exc)
    else:
        raise AssertionError("dirty source was accepted")

    _git(inputs["source"], "checkout", "--", "README")
    toolchain = json.loads(inputs["toolchain"].read_text())
    toolchain["source_tree"] = "0" * 40
    inputs["toolchain"].write_text(json.dumps(toolchain))
    try:
        _package(inputs, tmp_path / "drift-output")
    except PackagingError as exc:
        assert "source_tree mismatch" in str(exc)
    else:
        raise AssertionError("drifted toolchain receipt was accepted")


def test_rejects_missing_or_build_tree_dependencies(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["inspector"].write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --version ]; then echo fixture; exit 0; fi\n"
        f"echo 'libbad.so => {inputs['build']}/lib/libbad.so (0x1)'\n"
    )
    inputs["inspector"].chmod(0o755)
    try:
        _package(inputs, tmp_path / "bad-rpath-output")
    except PackagingError as exc:
        assert "forbidden build/source paths" in str(exc)
    else:
        raise AssertionError("build-tree dynamic dependency was accepted")

    inputs = _fixture(tmp_path / "missing-case")
    inputs["inspector"].write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --version ]; then echo fixture; exit 0; fi\n"
        "echo 'libmissing.so => not found'\n"
    )
    inputs["inspector"].chmod(0o755)
    try:
        _package(inputs, tmp_path / "missing-output")
    except PackagingError as exc:
        assert "unresolved dynamic dependency" in str(exc)
    else:
        raise AssertionError("missing dynamic dependency was accepted")


def test_rejects_tool_outside_build_root_and_symlink(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    outside = tmp_path / "flatboi"
    outside.write_text("#!/bin/sh\nexit 0\n")
    outside.chmod(0o755)
    inputs["flatboi"] = outside
    try:
        _package(inputs, tmp_path / "outside-output")
    except PackagingError as exc:
        assert "not inside" in str(exc)
    else:
        raise AssertionError("out-of-build-root tool was accepted")

    inputs = _fixture(tmp_path / "symlink-case")
    real = inputs["flatboi"]
    renamed = real.with_name("flatboi-real")
    real.rename(renamed)
    real.symlink_to(renamed.name)
    try:
        _package(inputs, tmp_path / "symlink-output")
    except PackagingError as exc:
        assert "unexpected executable name" in str(exc) or "non-symlink" in str(exc)
    else:
        raise AssertionError("symlinked tool was accepted")
