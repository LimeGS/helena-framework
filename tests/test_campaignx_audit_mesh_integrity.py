from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/helena_audit_mesh_integrity.py"
)


def executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def mesh(path: Path) -> Path:
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    return path


def run_audit(tmp_path: Path, *, seam_count: int, intersects: bool):
    native = mesh(tmp_path / "native.obj")
    canonical = mesh(tmp_path / "canonical.obj")
    seam = executable(
        tmp_path / "seam_audit",
        "import sys\n"
        f"n={seam_count}\n"
        "print(f'NEAR-COINCIDENT OVERLAP pairs (doubled surface / Z-fight): {n}')\n"
        "print('INTERPENETRATION pairs (non-parallel stab): 0')\n"
        "print('FOLD-BACK self-intersections (edge-adjacent, interiors overlap): 0')\n"
        "print(f'offending triangles: {n}')\n"
        "raise SystemExit(1 if n else 0)\n",
    )
    exact = executable(
        tmp_path / "exact_gate",
        "import json\n"
        f"value={intersects!r}\n"
        "print(json.dumps({'schema':'campaignx.mesh_self_intersection.v1',"
        "'vertices':3,'triangles':1,'self_intersections_present':value}))\n"
        "raise SystemExit(1 if value else 0)\n",
    )
    output = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--native-mesh",
            str(native),
            "--canonical-mesh",
            str(canonical),
            "--seam-audit",
            str(seam),
            "--self-intersection",
            str(exact),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_integrity_audit_passes_only_when_both_gates_are_clean(tmp_path: Path):
    completed, receipt = run_audit(tmp_path, seam_count=0, intersects=False)
    assert completed.returncode == 0
    assert receipt["status"] == "PASS"
    assert receipt["gate_mapping"]["self_intersections"] == 0


def test_integrity_audit_preserves_a_hard_defect(tmp_path: Path):
    completed, receipt = run_audit(tmp_path, seam_count=1, intersects=False)
    assert completed.returncode == 1
    assert receipt["status"] == "FAIL"
    assert receipt["gate_mapping"]["self_intersections"] is None
    assert receipt["seam"]["near_coincident_overlap_pairs"] == 1


def test_single_cube_without_a_seam_is_not_a_parser_failure(tmp_path: Path):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("mesh_audit", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.parse_seam(
        "seam_audit: no seam plane found (cube=128) -- nothing to audit\n"
    )
    assert metrics == {name: 0 for name in module.PATTERNS}
