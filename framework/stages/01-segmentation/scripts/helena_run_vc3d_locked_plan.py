#!/usr/bin/env python3
"""Execute one immutable VC3D baseline from a fleet locked-plan artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256, utc_now, write_json_atomic  # noqa: E402
from fleet.executor import VC3DGrowExecutor  # noqa: E402


SCHEMA = "campaignx.segment_fleet_locked_plan.v1"
REQUIRED = {
    "schema",
    "task_id",
    "attempt_id",
    "sample_id",
    "backend_profile",
    "roi_level0_zyx",
    "source",
    "selected_seed",
    "parameters",
    "ink_used",
    "non_claim",
}


def load_plan(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != REQUIRED:
        keys = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"locked-plan keys mismatch: missing={sorted(REQUIRED-keys)}, "
            f"unknown={sorted(keys-REQUIRED)}"
        )
    if value["schema"] != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    if value["ink_used"] is not False:
        raise ValueError("geometry baseline must not use ink")
    roi = value["roi_level0_zyx"]
    if not isinstance(roi, list) or len(roi) != 6 or any(
        int(roi[index]) >= int(roi[index + 3]) for index in range(3)
    ):
        raise ValueError("canonical positive ROI is required")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vc3d-binary", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    args = parser.parse_args(argv)

    plan_path = args.plan.resolve(strict=True)
    output = args.output_dir
    if not output.is_absolute() or output.exists():
        print(f"ERROR: immutable absolute output-dir required: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        plan = load_plan(plan_path)
        result = VC3DGrowExecutor(
            args.vc3d_binary.resolve(strict=True),
            timeout_seconds=args.timeout_seconds,
            minimum_free_gib=args.minimum_free_gib,
        ).execute(plan, output)
        write_json_atomic(
            output / "VC3D_LOCKED_PLAN_EXECUTION.json",
            {
                "schema": "campaignx.vc3d_locked_plan_execution.v1",
                "status": "SUCCEEDED",
                "created_at": utc_now(),
                "plan": str(plan_path),
                "plan_sha256": content_sha256(plan),
                "growth_receipt": str(output / "GROWTH_RECEIPT.json"),
                "surface_dir": str(result["surface_dir"]),
                "ink_used": False,
                "non_claim": "A technical baseline is not CT-validated geometry or text evidence.",
            },
        )
    except Exception as exc:
        write_json_atomic(
            output / "VC3D_LOCKED_PLAN_FAILURE.json",
            {
                "schema": "campaignx.vc3d_locked_plan_failure.v1",
                "status": "FAILED",
                "created_at": utc_now(),
                "plan": str(plan_path),
                "error_class": type(exc).__name__,
                "message": str(exc)[:4096],
                "ink_used": False,
            },
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output / "VC3D_LOCKED_PLAN_EXECUTION.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
