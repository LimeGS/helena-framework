#!/usr/bin/env python3
"""Freeze an ink-blind, spatially novel VC3D seed pilot for Phase 4.

Phase 1 searched eight axial positions at fixed polar angles.  This builder
uses the complementary eight interleaved axial positions and rotated polar
angles.  It asks VC3D for candidates only in those bounded regions, rejects a
candidate near every already-grown Phase 1 surface, and selects one maximally
novel candidate per scroll.  It never reads an ink prediction or a visual QC
label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_x import McpClient, structured


DEFAULT_OUTPUT = ROOT / "phase4/geometry_first_recovery_v1/GEOMETRY_RECOVERY_V1_PLAN.json"
STRUCTURES = (("outer", 0.80, 60.0), ("middle", 0.50, 180.0), ("inner", 0.20, 300.0))
NOVELTY_CLEARANCE_VOXELS = 256.0


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def gap_to_bbox(point: list[float], bbox: list[list[float]]) -> float:
    gap2 = 0.0
    for value, low, high in zip(point, bbox[0], bbox[1], strict=True):
        gap = max(float(low) - value, 0.0, value - float(high))
        gap2 += gap * gap
    return math.sqrt(gap2)


def candidate_coordinate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize legacy and bundled MCP candidates to exact CT-L0 XYZ."""

    coordinate = candidate.get("coordinate") or candidate.get(
        "ct_l0_coordinate"
    )
    if not isinstance(coordinate, dict) and all(
        axis in candidate for axis in "xyz"
    ):
        coordinate = {axis: candidate[axis] for axis in "xyz"}
    if not isinstance(coordinate, dict) or set(coordinate) != set("xyz"):
        return None
    return {axis: coordinate[axis] for axis in "xyz"}


def interleaved_region(shape_zyx: list[int], axial_slot: int, structural: str) -> dict[str, Any]:
    """Return a bounded phase-1-complement region or its fixed boundary receipt.

    The final interleaved axial slot can be too close to the exact CT boundary
    for the immutable 64-voxel request radius.  That is not an exceptional
    reroll condition: callers must record the requested slot as terminally
    unavailable and must not query an altered region.
    """
    shape_xyz = [int(shape_zyx[2]), int(shape_zyx[1]), int(shape_zyx[0])]
    axial_axis = max(range(3), key=lambda axis: shape_xyz[axis])
    transverse = [axis for axis in range(3) if axis != axial_axis]
    radius_fraction, rotation = next(
        (fraction, angle) for name, fraction, angle in STRUCTURES if name == structural
    )
    coords = [size // 2 for size in shape_xyz]
    # Phase 1 used (slot + 0.5)/8.  These are its exact, interleaved midpoints.
    coords[axial_axis] = round((shape_xyz[axial_axis] - 1) * ((axial_slot + 1) / 8.0))
    theta = math.radians((axial_slot * 137.507764 + rotation) % 360.0)
    planar_radius = 0.42 * min(shape_xyz[axis] for axis in transverse) * radius_fraction
    coords[transverse[0]] += round(planar_radius * math.cos(theta))
    coords[transverse[1]] += round(planar_radius * math.sin(theta))
    radius = [min(64, coords[index], shape_xyz[index] - 1 - coords[index]) for index in range(3)]
    result = {
        "coordinate_space": "ct_l0_xyz",
        "axial_axis": "xyz"[axial_axis],
        "center": dict(zip("xyz", coords, strict=True)),
        "radius": dict(zip("xyz", radius, strict=True)),
        "structural_stratum": structural,
    }
    if min(radius) < 32:
        return {
            **result,
            "candidate_query_permitted": False,
            "boundary_preflight": "OUT_OF_VOLUME_BOUNDARY",
        }
    return {**result, "candidate_query_permitted": True}


def build(
    root: Path,
    samples: set[str] | None = None,
    max_queries: int | None = None,
    start_query: int = 0,
) -> dict[str, Any]:
    eligible_path = root / "phase0/eligible_volumes.json"
    inventory_path = root / "phase3/fast_ab_existing_surfaces/INVENTORY.json"
    eligible = load(eligible_path)["entries"]
    inventory = load(inventory_path)
    if inventory.get("status") != "PASSED" or len(inventory.get("surfaces", [])) != 226:
        raise RuntimeError("Phase 1 surface inventory is not the expected hash-verified 226")
    if samples:
        eligible = [entry for entry in eligible if str(entry["sample_id"]) in samples]
        if {str(entry["sample_id"]) for entry in eligible} != samples:
            raise RuntimeError("requested unknown sample")
    if start_query < 0 or start_query >= 24:
        raise RuntimeError("start_query must select one of the 24 interleaved regions")
    old_bboxes: dict[str, list[list[list[float]]]] = {}
    for row in inventory["surfaces"]:
        old_bboxes.setdefault(str(row["scroll_id"]), []).append(row["bbox_xyz"])
    url = os.environ.get("VC_MCP_URL")
    token = os.environ.get("VC_MCP_AUTH_TOKEN")
    if not url or not token:
        raise RuntimeError("VC_MCP_URL and VC_MCP_AUTH_TOKEN are required for bounded candidate discovery")
    rows: list[dict[str, Any]] = []
    grid_index = 0
    for entry in sorted(eligible, key=lambda row: str(row["sample_id"])):
        sample_id = str(entry["sample_id"])
        for axial_slot in range(8):
            for structural, _fraction, _rotation in STRUCTURES:
                if grid_index < start_query:
                    grid_index += 1
                    continue
                if max_queries is not None and len(rows) >= max_queries:
                    break
                region = interleaved_region(entry["shape_zyx"], axial_slot, structural)
                if not region["candidate_query_permitted"]:
                    rows.append({
                        "sample_id": sample_id,
                        "seed_id": f"{sample_id}-g{axial_slot + 1:02d}-{structural}",
                        "axial_stratum": axial_slot + 1,
                        "candidate_region": region,
                        "prediction_uri": entry["surface_prediction_uri"],
                        "voxel_size_um": float(entry["voxel_size_um"]),
                        "candidate_count": 0,
                        "candidate_coordinate_xyz_l0": None,
                        "foreground_voxels": None,
                        "chunks_read": 0,
                        "novelty_gap_to_prior_surface_voxels": None,
                        "state": "OUT_OF_VOLUME_BOUNDARY",
                        "boundary_preflight": region["boundary_preflight"],
                        "ink_used": False,
                    })
                    grid_index += 1
                    continue
                # The local Streamable HTTP server is intentionally kept
                # short-lived.  Reopen its session every eight regions, the
                # same maximum number of m7 chunks touched by one bounded
                # query, so a stale MCP session cannot silently truncate a
                # geometry-only coverage audit.
                if len(rows) % 8 == 0:
                    client = McpClient(url, token)
                    client.initialize()
                request = {
                    "prediction_uri": entry["surface_prediction_uri"],
                    "prediction_space": "ct_l0_xyz",
                    "region": {"center": region["center"], "radius": region["radius"]},
                    "max_candidates": 8,
                    "minimum_separation_voxels": 16,
                }
                response = client.call(
                    "vc_find_seed_candidates", request,
                    f"geometry-recovery-v1-{sample_id}-{axial_slot + 1}-{structural}",
                )
                result = structured(response)
                candidates = result.get("candidates", [])
                coordinate = None
                if candidates:
                    coordinate = candidate_coordinate(candidates[0])
                point = ([float(coordinate[axis]) for axis in "xyz"] if isinstance(coordinate, dict) else None)
                novelty = (min((gap_to_bbox(point, bbox) for bbox in old_bboxes.get(sample_id, [])), default=math.inf) if point else None)
                rows.append({
                    "sample_id": sample_id,
                    "seed_id": f"{sample_id}-g{axial_slot + 1:02d}-{structural}",
                    "axial_stratum": axial_slot + 1,
                    "candidate_region": region,
                    "prediction_uri": entry["surface_prediction_uri"],
                    "voxel_size_um": float(entry["voxel_size_um"]),
                    "candidate_count": len(candidates),
                    "candidate_coordinate_xyz_l0": coordinate,
                    "foreground_voxels": result.get("foreground_voxels"),
                    "chunks_read": result.get("chunks_read"),
                    "novelty_gap_to_prior_surface_voxels": novelty,
                    "state": ("ELIGIBLE" if point and novelty is not None and novelty >= NOVELTY_CLEARANCE_VOXELS else "EXCLUDED_NOT_SPATIALLY_NOVEL" if point else "NO_CANDIDATE"),
                    "ink_used": False,
                })
                grid_index += 1
            if max_queries is not None and len(rows) >= max_queries:
                break
        if max_queries is not None and len(rows) >= max_queries:
            break
    selected: list[dict[str, Any]] = []
    for sample_id in sorted({str(entry["sample_id"]) for entry in eligible}):
        options = [row for row in rows if row["sample_id"] == sample_id and row["state"] == "ELIGIBLE"]
        if options:
            # Greater distance from existing grown surfaces first; stable ties are fixed by slot and layer.
            chosen = sorted(options, key=lambda row: (-float(row["novelty_gap_to_prior_surface_voxels"]), int(row["axial_stratum"]), str(row["seed_id"])))[0]
            chosen = {**chosen, "state": "SELECTED_FOR_ONE_SURFACE_PILOT"}
            selected.append(chosen)
    return {
        "kind": "campaign_x_phase4_geometry_first_recovery_v1_plan",
        "schema_version": 1,
        "status": "LOCKED_READY_PILOT" if max_queries is not None else "LOCKED_READY",
        "generated_at_utc": utc_now(),
        "purpose": "Open geometrically new coverage after the 226 Phase 1 VC3D surfaces were exhausted; this is not an ink-directed search.",
        "selection_rule": {
            "interleaved_axial_grid": "(slot + 1)/8, complementing Phase 1 (slot + 0.5)/8",
            "polar_rotation_degrees": {name: rotation for name, _fraction, rotation in STRUCTURES},
            "candidate_tool": "vc_find_seed_candidates",
            "candidate_rank": "first VC3D m7 candidate only",
            "novelty_clearance_voxels": NOVELTY_CLEARANCE_VOXELS,
            "pilot_selection": "one eligible candidate per scroll by maximum gap to every Phase 1 TIFXYZ bbox",
            "ink_used": False,
            "visual_labels_used": False,
            "reroll": False,
        },
        "source_hashes": {"eligible_volumes": sha256(eligible_path), "surface_inventory": sha256(inventory_path)},
        "queried_slot_count": len(rows),
        "mcp_query_count": sum(row.get("boundary_preflight") is None for row in rows),
        "full_grid_slot_count": len(eligible) * 24,
        "query_scope": (
            "PARTIAL_PREFIX_PILOT" if max_queries is not None else "FULL_INTERLEAVED_GRID"
        ),
        "start_query": start_query,
        "eligible_slot_count": sum(row["state"] == "ELIGIBLE" for row in rows),
        "selected_pilot_count": len(selected),
        "slots": rows,
        "selected_pilot": selected,
        "next_gate": "VC3D_GROWTH_THEN_ORTHOGONAL_CT_SCREENING",
        "non_claims": [
            "m7 seed availability does not prove a single physical sheet",
            "a selected pilot is not an ink candidate or a First Letters candidate",
            "this plan does not alter the frozen Phase 1 seed plan",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--start-query", type=int, default=0)
    args = parser.parse_args()
    payload = build(
        args.root, set(args.sample) or None, args.max_queries, args.start_query
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "queried_slot_count", "eligible_slot_count", "selected_pilot_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
