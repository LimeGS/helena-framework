#!/usr/bin/env python3
"""Render an audit-only review for candidates narrowly missing the CT gate.

The frozen gate decision is never changed.  This diagnostic selects only
downranked components, ranks them by failed-check count and normalized margin,
and renders the same model-context plus orthogonal CT views used for retained
signals.  It exists to detect an over-strict gate, not to bypass it.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_CONFIGURED_ROOT = os.environ.get("HELENA_REPO_ROOT", "").strip()
_ROOT_CANDIDATES = (
    [Path(_CONFIGURED_ROOT).expanduser().resolve()] if _CONFIGURED_ROOT else []
) + list(Path(__file__).resolve().parents)
_STAGE_ROOT = next(
    candidate / "framework/stages"
    for candidate in _ROOT_CANDIDATES
    if (candidate / "framework/stages").is_dir()
)
for _scripts in _STAGE_ROOT.glob("*/scripts"):
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

from build_high_recall_retained_review import (
    load_mean_probability_floats,
    model_context_panel,
    read_json,
    resolve_screening_directory,
    resolve_under_root,
    sha256_file,
    utc_now,
    write_json,
)
from analyze_ink_stability import layer_bounds
from build_orthogonal_candidate_review import (
    labeled_panel,
    load_stack,
    map_analysis_half_size_to_source,
    map_analysis_point_to_source,
    ordered_tiff_stack_position,
    orthogonal_views,
)


def normalized_violation(decision: dict[str, Any]) -> float:
    total = 0.0
    for check in decision["checks"]:
        if check["passed"]:
            continue
        value = float(check["value"])
        threshold = float(check["threshold"])
        scale = max(abs(threshold), 1e-9)
        operator = str(check["operator"])
        if operator.startswith(">"):
            total += max(0.0, threshold - value) / scale
        elif operator.startswith("<"):
            total += max(0.0, value - threshold) / scale
        else:
            raise ValueError(f"unsupported operator: {operator}")
    return total


def select_near_misses(
    decisions: list[dict[str, Any]], *, max_failed_checks: int, limit: int
) -> list[dict[str, Any]]:
    rows = []
    for decision in decisions:
        if decision.get("retained") is not False:
            continue
        failed = list(decision.get("failed_features", []))
        if not failed or len(failed) > max_failed_checks:
            continue
        rows.append(
            {
                **decision,
                "failed_check_count": len(failed),
                "normalized_gate_violation": normalized_violation(decision),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["failed_check_count"],
            row["normalized_gate_violation"],
            str(row["candidate_id"]),
        ),
    )[:limit]


def build_review(
    *,
    root: Path,
    adapter_receipt_path: Path,
    gate_evaluation_path: Path,
    output: Path,
    half_size: int,
    max_failed_checks: int,
    limit: int,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    receipt_path = output / "CT_GATE_NEAR_MISS_AUDIT.json"
    viewer_path = output / "CT_GATE_NEAR_MISS_AUDIT.html"
    if receipt_path.exists() or viewer_path.exists():
        raise RuntimeError("refusing to overwrite CT-gate near-miss evidence")

    adapter = read_json(adapter_receipt_path.resolve())
    gate = read_json(gate_evaluation_path.resolve())
    if adapter.get("status") != "READY_FOR_FROZEN_CT_FEATURE_EXTRACTION":
        raise RuntimeError("high-recall adapter is not ready")
    if gate.get("status") != "COMPLETED":
        raise RuntimeError("CT gate evaluation is not terminal")
    decisions_path = gate_evaluation_path.parent / str(gate["artifacts"]["decisions"])
    if sha256_file(decisions_path) != gate["artifacts"]["decisions_sha256"]:
        raise RuntimeError("CT decision artifact hash mismatch")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    selected = select_near_misses(
        decisions, max_failed_checks=max_failed_checks, limit=limit
    )

    spec_path = resolve_under_root(root, str(adapter["spec"]["path"]))
    if sha256_file(spec_path) != adapter["spec"]["sha256"]:
        raise RuntimeError("adapter CT spec hash mismatch")
    spec = read_json(spec_path)
    groups = {str(row["group_id"]): row for row in spec["groups"]}
    analyses = {str(row["window_id"]): row for row in adapter["analyses"]}
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        group_id = str(row["group_id"])
        if group_id not in groups or group_id not in analyses:
            raise RuntimeError(f"near miss has no adapter group: {group_id}")
        selected_by_group.setdefault(group_id, []).append(row)

    artifacts: list[dict[str, Any]] = []
    for group_id in sorted(selected_by_group):
        group = groups[group_id]
        analysis_row = analyses[group_id]
        analysis_path = resolve_under_root(root, str(analysis_row["path"]))
        if sha256_file(analysis_path) != analysis_row["sha256"]:
            raise RuntimeError(f"adapter analysis hash mismatch: {group_id}")
        analysis = read_json(analysis_path)
        candidates = {
            str(row["candidate_id"]): row
            for row in analysis["text_like_screening"]["candidates"]
        }
        tiff_dir = resolve_under_root(root, str(group["tiff_directory"]))
        stack, tiff_files, slice_ordering = load_stack(tiff_dir)
        if len(tiff_files) != 65:
            raise RuntimeError(f"{group_id} does not contain 65 TIFF slices")
        central_position = ordered_tiff_stack_position(
            tiff_files, int(group["central_slice"])
        )
        model_shape = tuple(map(int, analysis["input"]["shape_y_x"]))
        source_shape = (int(stack.shape[1]), int(stack.shape[2]))
        source_half = map_analysis_half_size_to_source(
            analysis_half_size=half_size,
            analysis_shape_y_x=model_shape,
            source_shape_y_x=source_shape,
        )
        screening_dir = resolve_screening_directory(
            root=root, tiff_dir=tiff_dir, analysis=analysis
        )
        mean_floats = load_mean_probability_floats(screening_dir)
        mean_valid = np.isfinite(mean_floats) & (mean_floats > 0)
        mean_display_bounds = layer_bounds(mean_floats, mean_valid)
        central_image = Image.fromarray(
            np.asarray(stack[central_position], dtype=np.uint8)
        )

        group_root = output / group_id
        for decision in selected_by_group[group_id]:
            candidate_id = str(decision["candidate_id"])
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise RuntimeError(f"near miss missing from analysis: {candidate_id}")
            x0, y0, x1, y1 = map(int, candidate["bbox_xyxy"])
            analysis_y = (y0 + y1) // 2
            analysis_x = (x0 + x1) // 2
            source_y, source_x = map_analysis_point_to_source(
                analysis_y=analysis_y,
                analysis_x=analysis_x,
                analysis_shape_y_x=model_shape,
                source_shape_y_x=source_shape,
            )
            xy, xz, yz = orthogonal_views(
                stack,
                center_y=source_y,
                center_x=source_x,
                central_position=central_position,
                half_size=source_half,
                average_width=2,
            )
            panel = labeled_panel(
                xy,
                xz,
                yz,
                candidate_id=candidate_id,
                analysis_center_y=analysis_y,
                analysis_center_x=analysis_x,
                source_center_y=source_y,
                source_center_x=source_x,
                upper=200,
                depth_scale=4,
            )
            orthogonal_path = group_root / f"{candidate_id}-orthogonal-ct.png"
            orthogonal_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(orthogonal_path)
            context_path = group_root / f"{candidate_id}-model-context.png"
            model_context_panel(
                central_ct=central_image,
                mean_probability=mean_floats,
                display_bounds=mean_display_bounds,
                bbox_y0_x0_y1_x1=[y0, x0, y1, x1],
                candidate_id=candidate_id,
                output=context_path,
            )
            artifacts.append(
                {
                    "group_id": group_id,
                    "sample_id": analysis["sample_id"],
                    "candidate_id": candidate_id,
                    "slice_ordering": slice_ordering,
                    "routing_score": candidate["high_recall_routing_score"],
                    "bbox_xyxy": candidate["bbox_xyxy"],
                    "failed_features": decision["failed_features"],
                    "failed_check_count": decision["failed_check_count"],
                    "normalized_gate_violation": decision[
                        "normalized_gate_violation"
                    ],
                    "gate_checks": decision["checks"],
                    "gate_decision_unchanged": decision["decision"],
                    "orthogonal_ct": {
                        "path": str(orthogonal_path.relative_to(output)),
                        "sha256": sha256_file(orthogonal_path),
                    },
                    "model_context": {
                        "path": str(context_path.relative_to(output)),
                        "sha256": sha256_file(context_path),
                    },
                }
            )

    cards = []
    for row in sorted(
        artifacts,
        key=lambda value: (
            value["failed_check_count"],
            value["normalized_gate_violation"],
        ),
    ):
        cards.append(
            f'''<article><h2>{html.escape(row["sample_id"])} · {html.escape(row["candidate_id"])}</h2>
            <p>Downranked, unchanged. {row["failed_check_count"]} check(s) failed:
            {html.escape(", ".join(row["failed_features"]))}. Margen normalizado:
            {float(row["normalized_gate_violation"]):.4f}.</p>
            <img src="{html.escape(row["model_context"]["path"])}" alt="contexto de modelo">
            <img src="{html.escape(row["orthogonal_ct"]["path"])}" alt="CT ortogonal"></article>'''
        )
    viewer_path.parent.mkdir(parents=True, exist_ok=True)
    viewer_path.write_text(
        """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CT gate near-miss audit</title><style>
        body{margin:0;background:#08111d;color:#edf4ff;font:17px system-ui}header{position:sticky;top:0;background:#0e1a2b;padding:16px 24px;border-bottom:1px solid #345}main{padding:18px;display:grid;gap:20px}article{background:#101d2d;border:1px solid #9b7524;border-radius:12px;padding:14px}h1,h2{margin:0 0 8px}p{color:#c7d3e4}img{display:block;max-width:100%;margin:10px auto;background:#000}</style><header><h1>Helena Framework - CT audit of near misses</h1><p>All of these remain downranked. This view audits the gate; it does not replace it and it proves neither ink nor letters.</p></header><main>"""
        + "\n".join(cards)
        + "</main>",
        encoding="utf-8",
    )
    receipt = {
        "schema": "campaignx.ct_gate_near_miss_audit.v1",
        "status": "AUDIT_READY_GATE_DECISIONS_UNCHANGED",
        "generated_at_utc": utc_now(),
        "adapter_sha256": sha256_file(adapter_receipt_path),
        "gate_sha256": sha256_file(gate_evaluation_path),
        "selection": {
            "max_failed_checks": max_failed_checks,
            "limit": limit,
            "order": "failed_check_count,normalized_gate_violation,candidate_id",
        },
        "candidate_count": len(artifacts),
        "candidates": artifacts,
        "viewer": {"path": viewer_path.name, "sha256": sha256_file(viewer_path)},
        "explicit_non_claims": [
            "frozen CT gate decisions are unchanged",
            "near-miss status is not accepted ink",
            "model morphology is not a letter identity",
            "not a First Letters submission claim",
        ],
    }
    write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--gate-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--half-size", type=int, default=128)
    parser.add_argument("--max-failed-checks", type=int, default=1)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    if args.max_failed_checks < 1 or args.limit < 1:
        raise ValueError("max-failed-checks and limit must be positive")
    result = build_review(
        root=args.root,
        adapter_receipt_path=args.adapter_receipt,
        gate_evaluation_path=args.gate_evaluation,
        output=args.output,
        half_size=args.half_size,
        max_failed_checks=args.max_failed_checks,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
