#!/usr/bin/env python3
"""Run the frozen ScrollFiesta seam and exact self-intersection gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


PATTERNS = {
    "near_coincident_overlap_pairs": re.compile(
        r"NEAR-COINCIDENT OVERLAP pairs .*?:\s*(\d+)"
    ),
    "interpenetration_pairs": re.compile(
        r"INTERPENETRATION pairs .*?:\s*(\d+)"
    ),
    "fold_back_intersections": re.compile(
        r"FOLD-BACK self-intersections .*?:\s*(\d+)"
    ),
    "offending_triangles": re.compile(r"offending triangles:\s*(\d+)"),
}


SEAM_METRIC_NAMES = tuple(PATTERNS)


class AuditError(RuntimeError):
    pass


def hard_defect_count(
    seam_metrics: dict[str, int], self_intersections_present: bool
) -> int:
    """The frozen gate arithmetic: PASS requires exactly zero hard defects.

    Kept as one function so the ScrollFiesta `.obj` route and the TIFXYZ route
    in ``helena_tifxyz_geometry_gate.py`` cannot drift apart on what counts
    as a hard defect.  Extra TIFXYZ-native seam metrics are summed the same way.
    """

    return sum(int(value) for value in seam_metrics.values()) + int(
        bool(self_intersections_present)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], accepted: set[int]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode not in accepted:
        raise AuditError(
            f"command failed closed with rc={completed.returncode}: {command!r}\n"
            f"stdout={completed.stdout[-2000:]}\nstderr={completed.stderr[-2000:]}"
        )
    return completed


def parse_seam(text: str) -> dict[str, int]:
    if "no seam plane found" in text and "nothing to audit" in text:
        return {name: 0 for name in PATTERNS}
    result = {}
    for name, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise AuditError(f"seam_audit omitted required metric {name}")
        result[name] = int(match.group(1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-mesh", type=Path, required=True)
    parser.add_argument("--canonical-mesh", type=Path, required=True)
    parser.add_argument("--seam-audit", type=Path, required=True)
    parser.add_argument("--self-intersection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cube-size", type=float, default=128.0)
    args = parser.parse_args()

    inputs = [args.native_mesh, args.canonical_mesh, args.seam_audit, args.self_intersection]
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise AuditError(f"missing or empty input: {path}")
    if args.output.exists():
        raise AuditError(f"refusing to overwrite: {args.output}")

    seam_command = [
        str(args.seam_audit),
        str(args.native_mesh),
        "--cube",
        str(args.cube_size),
        "--band",
        "4",
        "--gap",
        "1.0",
        "--angle",
        "20",
        "--hinge",
    ]
    seam = run(seam_command, {0, 1})
    seam_metrics = parse_seam(seam.stdout)

    intersection_command = [str(args.self_intersection), str(args.canonical_mesh)]
    intersection = run(intersection_command, {0, 1})
    try:
        intersection_metrics = json.loads(intersection.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError("self-intersection gate did not emit JSON") from exc
    if intersection_metrics.get("schema") != "campaignx.mesh_self_intersection.v1":
        raise AuditError("self-intersection gate emitted the wrong schema")
    present = intersection_metrics.get("self_intersections_present")
    if not isinstance(present, bool) or (intersection.returncode == 1) != present:
        raise AuditError("self-intersection return code and JSON disagree")

    hard_defects = hard_defect_count(seam_metrics, present)
    receipt = {
        "schema": "campaignx.mesh_integrity_audit.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if hard_defects == 0 else "FAIL",
        "inputs": {
            "native_mesh": {"path": str(args.native_mesh.resolve()), "sha256": sha256(args.native_mesh)},
            "canonical_mesh": {"path": str(args.canonical_mesh.resolve()), "sha256": sha256(args.canonical_mesh)},
        },
        "tools": {
            "seam_audit": {"path": str(args.seam_audit.resolve()), "sha256": sha256(args.seam_audit)},
            "self_intersection": {"path": str(args.self_intersection.resolve()), "sha256": sha256(args.self_intersection)},
        },
        "commands": {"seam_audit": seam_command, "self_intersection": intersection_command},
        "seam": {**seam_metrics, "returncode": seam.returncode},
        "exact_self_intersection": {**intersection_metrics, "returncode": intersection.returncode},
        "gate_mapping": {
            "self_intersections": 0 if hard_defects == 0 else None,
            "hard_defects_observed": hard_defects,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
