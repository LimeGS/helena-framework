"""Focused I1 tests for the ScrollFiesta native/OCI runtime package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "containers/images/scrollfiesta"
SCRIPTS = RUNTIME / "scripts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_bundle(root: Path) -> None:
    executable = (
        "bin/cube_mesh",
        "bin/grid_weld",
        "bin/obj_components",
        "bin/pinhole_verdict",
        "bin/seam_audit",
        "bin/flatboi",
        "bin/vc_obj2tifxyz_legacy",
        "bin/scrollunwrap",
    )
    regular = (
        "SOURCE_LOCK.json",
        "LICENSE_INVENTORY.json",
        "SBOM.spdx.json",
        "BUILD_RECEIPT.json",
        "PYTHON_REQUIREMENTS.lock",
        "python-packages/scrollunwrap/__init__.py",
        "share/licenses/scrollfiesta/LICENSE",
        "share/licenses/scrollfiesta/THIRD_PARTY_LICENSES.md",
    )
    (root / "BUNDLE_SCHEMA").parent.mkdir(parents=True, exist_ok=True)
    (root / "BUNDLE_SCHEMA").write_text("campaignx.scrollfiesta_runtime_bundle.v1\n")
    for relative in executable:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o555)
    for relative in regular:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def test_source_and_license_locks_are_exact_and_fail_closed() -> None:
    lock = json.loads((RUNTIME / "locks/source-lock.json").read_text())
    assert lock["scrollfiesta"]["commit"] == "f344c17931b9e264a17c8d760a4c478390133bd4"
    assert lock["scrollfiesta"]["native_version"] == "0.9.0"
    assert lock["scrollfiesta"]["python_version"] == "0.1.0"
    assert lock["volume_cartographer"]["commit"] == "05dcf0349356bc833670d61e5eca00be58376e35"
    assert lock["build_policy"]["forbidden_runtime_cli"] == ["grid_pipeline"]
    assert lock["build_policy"]["distribution"] == "INTERNAL_RESEARCH_ONLY"

    licenses = json.loads((RUNTIME / "licenses/license-inventory.json").read_text())
    assert licenses["overall_distribution_status"] == "BLOCKED_PENDING_LICENSE_CLEARANCE"
    by_name = {component["name"]: component for component in licenses["components"]}
    assert by_name["Triangle"]["redistribution_status"] == "BLOCKED"
    assert by_name["andres/graph"]["redistribution_status"] == "BLOCKED"


def test_frozen_checkout_verifies_when_probe_checkout_is_available() -> None:
    checkout = Path("/private/tmp/scrollfiesta-probe-f344c179")
    if not checkout.is_dir():
        return
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "verify_source_lock.py"),
            "--source",
            str(checkout),
            "--lock",
            str(RUNTIME / "locks/source-lock.json"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "VERIFIED"


def test_villa_runtime_requires_commit_executables_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "villa"
    (root / "bin").mkdir(parents=True)
    artifacts = {}
    for name in ("flatboi", "vc_obj2tifxyz_legacy"):
        path = root / "bin" / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o555)
        artifacts[name] = {"path": f"bin/{name}", "sha256": _sha256(path)}
    (root / "VILLA_RUNTIME_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "campaignx.villa_runtime_bundle.v1",
                "source_commit": "05dcf0349356bc833670d61e5eca00be58376e35",
                "artifacts": artifacts,
            }
        )
    )
    module = runpy.run_path(SCRIPTS / "verify_villa_runtime.py")
    assert module["verify"](root)["status"] == "VERIFIED"
    (root / "bin/flatboi").chmod(0o755)
    (root / "bin/flatboi").write_text("tampered")
    try:
        module["verify"](root)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered Villa runtime was accepted")


def test_runtime_verifier_is_complete_and_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_fake_bundle(root)
    module = runpy.run_path(SCRIPTS / "verify_runtime_bundle.py")
    module["verify"](root)
    (root / "bin/grid_weld").chmod(0o755)
    (root / "bin/grid_weld").write_text("tampered")
    try:
        module["verify"](root)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered runtime bundle was accepted")


def test_spdx_sbom_is_deterministic_and_file_hashed(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    root.mkdir()
    payload = root / "bin/cube_mesh"
    payload.parent.mkdir()
    payload.write_bytes(b"locked-binary-fixture")
    module = runpy.run_path(SCRIPTS / "generate_sbom.py")
    lock = json.loads((RUNTIME / "locks/source-lock.json").read_text())
    first = module["build_sbom"](root, lock)
    second = module["build_sbom"](root, lock)
    assert first == second
    assert first["spdxVersion"] == "SPDX-2.3"
    assert first["files"] == [
        {
            "SPDXID": first["files"][0]["SPDXID"],
            "fileName": "./bin/cube_mesh",
            "checksums": [{"algorithm": "SHA256", "checksumValue": _sha256(payload)}],
        }
    ]


def test_oci_context_is_deterministic_and_contains_no_loose_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_fake_bundle(bundle)
    for ordinal in (1, 2):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "make_oci_context.py"),
                "--bundle",
                str(bundle),
                "--output",
                str(tmp_path / f"context-{ordinal}"),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.returncode == 0, completed.stderr
    first = tmp_path / "context-1/scrollfiesta-runtime.tgz"
    second = tmp_path / "context-2/scrollfiesta-runtime.tgz"
    assert _sha256(first) == _sha256(second)
    assert sorted(path.name for path in (tmp_path / "context-1").iterdir()) == [
        "OCI_CONTEXT_RECEIPT.json",
        "scrollfiesta-runtime.tgz",
    ]


def test_container_and_builder_enforce_i1_boundaries() -> None:
    containerfile = (ROOT / "containers/images/Containerfile.scrollfiesta").read_text()
    for value in (
        "COPY --from=scrollfiesta_runtime /scrollfiesta-runtime.tgz",
        "sha256sum -c SHA256SUMS",
        "INTERNAL_RESEARCH_ONLY",
        "test ! -e bin/grid_pipeline",
        "SCROLLFIESTA_CUBE_MESH=/opt/campaignx/scrollfiesta/bin/cube_mesh",
        "SCROLLFIESTA_OBJ2TIFXYZ=/opt/campaignx/scrollfiesta/bin/vc_obj2tifxyz_legacy",
    ):
        assert value in containerfile
    assert "COPY ." not in containerfile

    builder = (SCRIPTS / "build_native_bundle.sh").read_text()
    for value in (
        "git -C \"$sf_source\" archive",
        "SCROLLFIESTA_WITH_TIFF=ON",
        "SCROLLFIESTA_OPENMP=ON",
        "ctest --test-dir",
        '"$source_copy/python/tests"',
        '--ignore "$source_copy/python/tests/test_network.py"',
        "rm -f \"$stage/bin/grid_pipeline\"",
    ):
        assert value in builder
    assert "git clone" not in builder
    assert "curl " not in builder
