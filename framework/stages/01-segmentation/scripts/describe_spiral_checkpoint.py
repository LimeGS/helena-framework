#!/usr/bin/env python3
"""What a fitted spiral checkpoint says about itself, as JSON on stdout.

Runs in the lane's own interpreter, because reading the checkpoint needs torch
and upstream's own helpers: `checkpoint_io.load_checkpoint_cpu` (which also
migrates the legacy gap parameterisation) and `gap_parameterization.
lower_bounded_dr`, the exact function the model uses to turn its learned
logit into a winding spacing. Nothing here re-derives that: a receipt that
computed dr_per_winding its own way would carry a number the fit never used.

Why this exists: winding_model_9um reads z_begin/z_end/dr_per_winding from
the fit's .ckpt, and an audit reads the same three to check a surface against
the fit that produced it. A P1 receipt that named the checkpoint's path and
nothing inside it left both to reopen a torch archive to find out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def describe(checkpoint: Path, spiral_root: Path) -> dict:
    sys.path.insert(0, str(spiral_root))
    from checkpoint_io import load_checkpoint_cpu  # noqa: PLC0415
    from gap_parameterization import lower_bounded_dr  # noqa: PLC0415

    payload = load_checkpoint_cpu(str(checkpoint))
    resolved = dict(payload.get("resolved_config") or payload.get("cfg") or {})
    state = payload["spiral_and_transform"]
    min_gap = float(resolved.get("model_gap_expander_min_gap", 1.0))
    dr = float(lower_bounded_dr(state["dr_per_winding_logit"], min_gap))
    return {
        "schema_version": payload.get("schema_version"),
        "completed_iterations": int(payload.get("completed_iterations", 0)),
        "z_begin": int(payload["z_begin"]),
        "z_end": int(payload["z_end"]),
        "spiral_outward_sense": str(payload["spiral_outward_sense"]),
        "dr_per_winding": dr,
        "model_gap_expander_min_gap": min_gap,
        "model_initial_dr_per_winding": float(
            resolved.get("model_initial_dr_per_winding", 16.0)),
        "lasagna_scale": payload.get("lasagna_scale"),
        "input_manifest": dict(payload.get("input_manifest") or {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--spiral-root", type=Path, required=True,
                        help="upstream's spiral-fitting checkout, for its own helpers")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(describe(args.checkpoint, args.spiral_root), sort_keys=True))
    except Exception as error:  # noqa: BLE001 - the caller records the reason
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
