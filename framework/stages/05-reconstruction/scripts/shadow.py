#!/usr/bin/env python3
"""Ingest top-20 TIFXYZ surfaces and run conservative A8 shadow inference."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
WORKSPACE = ROOT.parent
PHASE2 = ROOT / "phase2"
GEOWRAP = WORKSPACE / "geowrap-ssl-lab"
sys.path[:0] = [str(PHASE2 / "src"), str(GEOWRAP / "src")]

from campaign_x_phase2.calibration import apply_temperature  # noqa: E402
from campaign_x_phase2.constraint_graph import ConstraintGraph, Edge  # noqa: E402
from campaign_x_phase2.contracts import constraints_to_point_collection  # noqa: E402
from campaign_x_phase2.surface_candidates import (  # noqa: E402
    FEATURE_NAMES,
    Surface,
    generate_surface_candidates,
)
from geowrap.experiment import load_artifact  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ingest_surface(seed_id: str) -> tuple[Path, dict[str, Any]]:
    raw = ROOT / "phase1" / "runs" / "live-baseline-sparse-fixed-20260715" / "raw_mcp" / f"{seed_id}-job.json"
    job = json.loads(raw.read_text())
    artifact = next(item for item in job["job"]["artifacts"] if item["artifact_id"] == "surface")
    source = Path(artifact["path"])
    destination = PHASE2 / "inputs" / "surfaces" / seed_id
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json"):
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = destination / name
        shutil.copy2(src, dst)
        files.append({"name": name, "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    return destination, {
        "seed_id": seed_id,
        "source_job_id": job["job"]["job_id"],
        "source_job_record": str(raw.relative_to(ROOT)),
        "files": files,
    }


def main() -> int:
    portfolio = json.loads((PHASE2 / "TOP20_AUTOMATED_PORTFOLIO.json").read_text())
    calibration = json.loads((PHASE2 / "benchmark" / "A8_CALIBRATION.json").read_text())
    surfaces: dict[str, Surface] = {}
    ingest_records = []
    for candidate in portfolio["candidates"]:
        seed_id, scroll_id = candidate["seed_id"], candidate["scroll_id"]
        directory, ingest = ingest_surface(seed_id)
        ingest_records.append(ingest)
        surfaces[seed_id] = Surface.load(seed_id, scroll_id, directory)
    atomic_json(
        PHASE2 / "inputs" / "SURFACE_INPUT_MANIFEST.json",
        {
            "kind": "campaign_x_phase2_surface_input_manifest_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "surface_count": len(ingest_records),
            "surfaces": ingest_records,
        },
    )

    models = {}
    for seed in (13, 37, 73):
        checkpoint = PHASE2 / "runs" / "m1-cross-scroll-reproduction-20260715" / "runs" / f"A8_seed{seed}" / "checkpoint"
        models[seed] = load_artifact(checkpoint)

    global_summary = Counter()
    target_summaries = []
    for candidate in portfolio["candidates"]:
        seed_id = candidate["seed_id"]
        records, matrix = generate_surface_candidates(surfaces[seed_id], gap_hint=10.0, k=8)
        probabilities: dict[int, np.ndarray] = {}
        for seed, model in models.items():
            raw = model.predict_proba(matrix, device="cpu")
            probabilities[seed] = apply_temperature(raw, calibration["seeds"][str(seed)]["temperature"])

        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record["candidate_group"]].append(index)
        graph = ConstraintGraph()
        accepted_constraints = []
        accepted_pairs: set[tuple[str, str]] = set()
        detailed = []
        decisions = Counter()
        for group_id in sorted(groups):
            group_rows = groups[group_id]
            best_index = max(group_rows, key=lambda index: float(np.mean([probabilities[s][index, 0] for s in probabilities])))
            record = records[best_index]
            seed_classes = {str(seed): int(np.argmax(probabilities[seed][best_index])) for seed in probabilities}
            votes = Counter(seed_classes.values())
            winner, winner_count = votes.most_common(1)[0]
            mean_probability = np.mean([probabilities[seed][best_index] for seed in probabilities], axis=0)
            threshold = float(np.median([calibration["seeds"][str(seed)]["same_threshold"]["threshold"] for seed in probabilities]))
            margin = float(np.sort(mean_probability)[-1] - np.sort(mean_probability)[-2])
            if winner_count < 2:
                decision, reason = "ABSTAIN", "SEED_DISAGREEMENT"
            elif winner == 0 and float(mean_probability[0]) >= threshold and margin >= 0.10:
                decision, reason = "ACCEPT_SAME_WINDING", None
            elif winner == 1:
                decision, reason = "ABSTAIN", "UNVALIDATED_RELATIVE_SIGN"
            elif winner == 2:
                decision, reason = "REJECT_UNRELATED", None
            else:
                decision, reason = "ABSTAIN", "LOW_CONFIDENCE"
            anchor_node = f"{seed_id}:uv:{record['anchor_uv'][0]}:{record['anchor_uv'][1]}"
            target_node = f"{seed_id}:uv:{record['target_uv'][0]}:{record['target_uv'][1]}"
            canonical_pair = tuple(sorted((anchor_node, target_node)))
            if decision == "ACCEPT_SAME_WINDING" and canonical_pair in accepted_pairs:
                decision, reason = "ABSTAIN", "DUPLICATE_UNDIRECTED_PAIR"
            decisions[decision] += 1
            detail = {
                **record,
                "seed_classes": seed_classes,
                "mean_probabilities": {
                    "same_sheet": float(mean_probability[0]),
                    "adjacent_wrap": float(mean_probability[1]),
                    "unrelated": float(mean_probability[2]),
                },
                "consensus_count": winner_count,
                "class_margin": margin,
                "decision": decision,
                "abstain_reason": reason,
            }
            detailed.append(detail)
            if decision == "ACCEPT_SAME_WINDING":
                constraint_id = record["candidate_id"]
                constraint = {
                    "constraint_id": constraint_id,
                    "type": "SAME_WINDING",
                    "endpoint_a_xyz_l0": record["anchor_xyz_l0"],
                    "endpoint_b_xyz_l0": record["candidate_xyz_l0"],
                    "winding_delta": 0,
                    "confidence": float(mean_probability[0]),
                }
                inserted = graph.add(
                    Edge(
                        constraint_id,
                        anchor_node,
                        target_node,
                        0,
                        constraint["confidence"],
                    )
                )
                if inserted:
                    accepted_pairs.add(canonical_pair)
                    accepted_constraints.append(constraint)
        output_dir = PHASE2 / "shadow_top20" / seed_id
        output_dir.mkdir(parents=True, exist_ok=True)
        point_collection = constraints_to_point_collection(
            accepted_constraints, collection_name=f"campaign-x:{seed_id}:same-winding"
        )
        atomic_json(output_dir / "candidate_decisions.json", detailed)
        atomic_json(output_dir / "constraints.json", accepted_constraints)
        atomic_json(output_dir / "vc_pointcollections.json", point_collection)
        graph_summary = graph.summary()
        status = (
            "SPARSE_BUT_CONSISTENT"
            if accepted_constraints and graph_summary["conflicting_edges"] == 0
            else "INSUFFICIENT_CONSTRAINT_COVERAGE"
        )
        summary = {
            "seed_id": seed_id,
            "scroll_id": candidate["scroll_id"],
            "status": status,
            "candidate_rows": len(records),
            "candidate_groups": len(groups),
            "decisions": dict(decisions),
            "accepted_constraints": len(accepted_constraints),
            "graph": graph_summary,
            "relative_winding_exported": 0,
            "validation_limit": (
                "Surface geometry only: raw CT is not sampled, candidate recall is unmeasured, "
                "and no relative edge is exported until sign accuracy is validated against ground truth."
            ),
        }
        atomic_json(output_dir / "summary.json", summary)
        target_summaries.append(summary)
        global_summary.update(decisions)

    result = {
        "kind": "campaign_x_phase2_top20_shadow_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_status": "ENGINEERING_SHADOW_VALIDATION_BLOCKED",
        "target_count": len(target_summaries),
        "decision_counts": dict(global_summary),
        "targets": target_summaries,
        "automatic_acceptance_forbidden": True,
        "candidate_generator_limit": "TIFXYZ surface geometry only; no raw CT sampling and no measured recall@8.",
    }
    atomic_json(PHASE2 / "shadow_top20" / "SHADOW_RESULTS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
