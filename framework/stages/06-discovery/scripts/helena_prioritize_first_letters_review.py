#!/usr/bin/env python3
"""Prioritize a verified First Letters review queue without discarding rows."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUEUE_SCHEMA = "campaignx.first_letters_review_queue.v1"
PROFILE_KIND = "campaignx.discovery.ct_depth_concentration_priority.v1"
OUTPUT_SCHEMA = "campaignx.first_letters_review_priority.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# FIX-05: the priority viewer used to show review_image_paths[:3], which under
# the queue's own ordering is always model-context, orthogonal-ct and ct.png —
# three CT-derived panels and not one ink layer.  The reviewer was asked to
# judge an ink candidate without seeing the ink evidence.  At least one
# probability layer is now always carried into the gallery.
INK_LAYER_SUFFIXES = (
    "mean_probability.png",
    "robust_minimum.png",
    "persistent_overlay.png",
    "replica_agreement.png",
)
GALLERY_IMAGE_LIMIT = 4


def is_ink_layer(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in INK_LAYER_SUFFIXES)


def select_gallery_images(
    paths: list[str], *, limit: int = GALLERY_IMAGE_LIMIT
) -> list[str]:
    """Take the highest-priority panels but guarantee an ink layer is present."""

    selected = list(paths[:limit])
    if any(is_ink_layer(path) for path in selected):
        return selected
    ink = next((path for path in paths if is_ink_layer(path)), None)
    if ink is None:
        return selected
    if len(selected) >= limit:
        selected = selected[: limit - 1]
    selected.append(ink)
    return selected


def compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == "<":
        return value < threshold
    raise RuntimeError(f"unsupported operator: {operator}")


def read_decisions(queue_root: Path, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = queue_root / str(candidate["local_manifest"])
    evidence_root = manifest.parent
    decision_paths = list(evidence_root.rglob("CT_FIBER_GATE_DECISIONS.json"))
    if len(decision_paths) != 1:
        raise RuntimeError(
            f"expected one gate decision file for {candidate['candidate_id']}, "
            f"found {len(decision_paths)}"
        )
    decisions = json.loads(decision_paths[0].read_text(encoding="utf-8"))
    if not isinstance(decisions, list):
        raise RuntimeError("gate decisions must be a list")
    return decisions


def prioritize(queue_path: Path, profile_path: Path) -> dict[str, Any]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(queue, dict) or queue.get("schema") != QUEUE_SCHEMA:
        raise RuntimeError("unexpected First Letters queue schema")
    if not isinstance(profile, dict) or profile.get("kind") != PROFILE_KIND:
        raise RuntimeError("unexpected priority profile")
    if profile.get("policy", {}).get("failed_priority_is_still_reviewable") is not True:
        raise RuntimeError("priority profile violates review-preservation boundary")

    queue_root = queue_path.parent
    rows: list[dict[str, Any]] = []
    for surface in queue.get("candidates", []):
        decisions = read_decisions(queue_root, surface)
        by_id = {
            str(item.get("candidate_id")): item
            for item in decisions
            if isinstance(item, dict)
        }
        retained_ids = [str(value) for value in surface.get("retained_candidate_ids", [])]
        if len(retained_ids) != int(surface.get("retained_component_count", -1)):
            raise RuntimeError(f"retained candidate count mismatch: {surface['candidate_id']}")
        for candidate_id in retained_ids:
            decision = by_id.get(candidate_id)
            if decision is None or decision.get("retained") is not True:
                raise RuntimeError(f"retained decision missing: {candidate_id}")
            features = {
                str(check["feature"]): float(check["value"])
                for check in decision.get("checks", [])
            }
            checks = []
            for requirement in profile["requirements"]:
                feature = str(requirement["feature"])
                if feature not in features:
                    raise RuntimeError(f"priority feature missing: {feature}")
                threshold = float(requirement["threshold"])
                passed = compare(features[feature], str(requirement["operator"]), threshold)
                checks.append(
                    {
                        "feature": feature,
                        "value": features[feature],
                        "operator": requirement["operator"],
                        "threshold": threshold,
                        "passed": passed,
                    }
                )
            high_priority = all(check["passed"] for check in checks)
            image_paths = [
                path
                for path in surface.get("review_image_paths", [])
                if candidate_id in path or "comparison_layers" in path
            ]
            rows.append(
                {
                    "sample_id": surface.get("sample_id"),
                    "surface_id": surface.get("surface_id"),
                    "qc_job_id": surface.get("qc_job_id"),
                    "candidate_id": candidate_id,
                    "priority": (
                        profile["decision"]["all_requirements_pass"]
                        if high_priority
                        else profile["decision"]["any_requirement_fails"]
                    ),
                    "high_priority": high_priority,
                    "checks": checks,
                    "review_image_paths": image_paths,
                    "gallery_image_paths": select_gallery_images(image_paths),
                    "gallery_includes_ink_layer": any(
                        is_ink_layer(path)
                        for path in select_gallery_images(image_paths)
                    ),
                    "review_state": "UNREVIEWED",
                    "claim_state": "NO_FIRST_LETTERS_CLAIM",
                }
            )
    rows.sort(
        key=lambda row: (
            not row["high_priority"],
            -float(row["checks"][0]["value"]),
            row["candidate_id"],
        )
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "generated_at_utc": utc_now(),
        "source_queue": str(queue_path),
        "source_queue_sha256": sha256(queue_path),
        "source_queue_id": queue.get("queue_id"),
        "profile": str(profile_path),
        "profile_sha256": sha256(profile_path),
        "candidate_count": len(rows),
        "high_priority_count": sum(row["high_priority"] for row in rows),
        "standard_review_count": sum(not row["high_priority"] for row in rows),
        "candidates": rows,
        "acceptance_boundary": "priority only; all source rows remain reviewable and no ink, text, letter, or First Letters claim is made",
    }


def render_html(result: dict[str, Any]) -> str:
    cards = []
    queue_root = Path(result["source_queue"]).parent
    for row in result["candidates"]:
        score = row["checks"][0]
        gallery = row.get("gallery_image_paths") or select_gallery_images(
            list(row["review_image_paths"])
        )
        images = "".join(
            f'<a href="{html.escape(str(queue_root / path))}">'
            f'<img src="{html.escape(str(queue_root / path))}" loading="lazy">'
            f"</a>"
            for path in gallery
        )
        css_class = "high" if row["high_priority"] else "standard"
        cards.append(
            f'<article class="card {css_class}"><header><h2>{html.escape(str(row["sample_id"]))}</h2>'
            f'<span>{html.escape(str(row["priority"]))}</span></header>'
            f'<p><code>{html.escape(row["candidate_id"])}</code></p>'
            f'<p>Top-3 axial concentration: <strong>{score["value"]:.4f}</strong>  - priority threshold {score["operator"]} {score["threshold"]:.2f}</p>'
            f'<div class="gallery">{images}</div>'
            '<p class="warning">A review priority; it proves neither ink nor letters.</p></article>'
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Helena Framework · prioridad First Letters</title><style>
:root{{--bg:#08111e;--panel:#111d30;--line:#314765;--text:#eef4ff;--muted:#aebbd0;--high:#36d399;--standard:#f2b84b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}.lede{{color:var(--muted);max-width:90ch}}.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}.pill{{padding:8px 12px;border:1px solid var(--line);border-radius:999px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}}.card{{background:var(--panel);border:2px solid var(--standard);border-radius:14px;padding:14px}}.card.high{{border-color:var(--high)}}header{{display:flex;justify-content:space-between;gap:12px;align-items:center}}header h2{{margin:0}}header span{{font-size:.8rem;color:var(--muted)}}code{{font-size:.75rem;color:var(--muted)}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:4px;background:#000}}.gallery img{{display:block;width:100%;aspect-ratio:1.6;object-fit:contain}}.warning{{color:var(--standard);font-weight:650}}@media(max-width:650px){{main{{padding:10px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Helena Framework - review priority</h1><p class="lede">The high tier requires a depth concentration consistent with the PHerc0139 controls. The standard tier remains reviewable; it never means absence.</p><div class="summary"><span class="pill">{result["candidate_count"]} signals</span><span class="pill">{result["high_priority_count"]} high priority</span><span class="pill">{result["standard_review_count"]} standard review</span></div><section class="grid">{''.join(cards)}</section></main></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue_path = args.queue.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = prioritize(queue_path, profile_path)
    json_path = output / "FIRST_LETTERS_PRIORITY_QUEUE.json"
    html_path = output / "FIRST_LETTERS_PRIORITY_QUEUE.html"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    html_path.write_text(render_html(result))
    receipt = {
        "schema": "campaignx.first_letters_review_priority_receipt.v1",
        "status": "COMPLETED",
        "generated_at_utc": utc_now(),
        "queue_sha256": sha256(json_path),
        "viewer_sha256": sha256(html_path),
        "candidate_count": result["candidate_count"],
        "high_priority_count": result["high_priority_count"],
        "no_automatic_acceptance": True,
    }
    (output / "PRIORITY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
