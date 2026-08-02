#!/usr/bin/env python3
"""Validate every shadow export with internal and official Villa loaders."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
PHASE2 = ROOT / "phase2"
SPIRAL = ROOT / "vendor" / "villa" / "volume-cartographer" / "scripts" / "spiral"
sys.path[:0] = [str(PHASE2 / "src"), str(SPIRAL)]

from campaign_x_phase2.contracts import (  # noqa: E402
    constraints_to_point_collection,
    point_collection_to_constraints,
)
from point_collection import load_point_collection  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    records = []
    failures = []
    total_constraints = 0
    candidate_validator = Draft202012Validator(json.loads((PHASE2 / "schemas" / "candidate.schema.json").read_text()))
    constraint_validator = Draft202012Validator(json.loads((PHASE2 / "schemas" / "constraint.schema.json").read_text()))
    for path in sorted((PHASE2 / "shadow_top20").glob("*/vc_pointcollections.json")):
        document = json.loads(path.read_text())
        try:
            candidates = json.loads((path.parent / "candidate_decisions.json").read_text())
            constraints = json.loads((path.parent / "constraints.json").read_text())
            for candidate in candidates:
                candidate_validator.validate(candidate)
            for constraint in constraints:
                constraint_validator.validate(constraint)
            decoded = point_collection_to_constraints(document)
            reexported = constraints_to_point_collection(decoded, collection_name="roundtrip")
            redecoded = point_collection_to_constraints(reexported)
            if decoded != redecoded:
                raise AssertionError("internal round-trip changed a constraint")
            official = load_point_collection(str(path))
            if official is None:
                raise AssertionError("official loader returned None")
            if len(official) != len(decoded):
                raise AssertionError("official/internal collection counts differ")
            for constraint in decoded:
                for key in ("endpoint_a_xyz_l0", "endpoint_b_xyz_l0"):
                    if len(constraint[key]) != 3 or not all(math.isfinite(v) for v in constraint[key]):
                        raise AssertionError(f"invalid finite XYZ in {key}")
            status = "PASSED"
            total_constraints += len(decoded)
        except Exception as exc:  # validation report must retain every failure
            status = "FAILED"
            failures.append({"path": str(path.relative_to(ROOT)), "error": repr(exc)})
            decoded = []
        records.append(
            {
                "seed_id": path.parent.name,
                "path": str(path.relative_to(ROOT)),
                "status": status,
                "constraint_count": len(decoded),
                "point_count": 2 * len(decoded),
                "candidate_schema_count": len(candidates) if status == "PASSED" else 0,
                "constraint_schema_count": len(constraints) if status == "PASSED" else 0,
            }
        )
    smoke_path = PHASE2 / "spiral_smoke" / "OFFICIAL_LOADER_SMOKE.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None
    report = {
        "kind": "campaign_x_phase2_export_validation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "official_loader": "vendor/villa/volume-cartographer/scripts/spiral/point_collection.py",
        "file_count": len(records),
        "passed_file_count": sum(item["status"] == "PASSED" for item in records),
        "constraint_count": total_constraints,
        "point_count": 2 * total_constraints,
        "failures": failures,
        "files": records,
        "spiral_smoke_status": smoke["status"] if smoke is not None else "NOT_EXECUTED",
    }
    destination = PHASE2 / "benchmark" / "POINTCOLLECTION_VALIDATION.json"
    atomic_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures or len(records) != 20 else 0


if __name__ == "__main__":
    raise SystemExit(main())
