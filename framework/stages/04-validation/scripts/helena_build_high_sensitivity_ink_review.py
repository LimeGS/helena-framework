#!/usr/bin/env python3
"""Package stable sub-threshold ink activations as evidence-only review bundles.

This deliberately sits after the strict stability screen.  It does not promote an
activation to ink, text, or a First Letters claim; it preserves the exact CT and
replica evidence required to investigate an activation that the strict text gate
correctly declined to queue.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


VIEW_NAMES = {
    "ct.png": "CT central (misma superficie)",
    "depth_center_025.png": "CT a profundidad 25",
    "depth_center_032.png": "CT a profundidad 32",
    "depth_center_039.png": "CT a profundidad 39",
    "mean_probability.png": "Mean probability over six replicas",
    "robust_minimum.png": "Minimum across replicas",
    "replica_agreement.png": "Replica agreement",
    "replica_disagreement.png": "Replica disagreement",
    "persistent_overlay.png": "Overlay persistente sobre CT",
}


def bounded_crop(image: Image.Image, *, center_x: int, center_y: int, size: int) -> Image.Image:
    """Return a fixed-size crop, padding outside the image with black pixels."""
    if size < 1:
        raise ValueError("crop size must be positive")
    image = image.convert("RGB")
    half = size // 2
    left, top = center_x - half, center_y - half
    right, bottom = left + size, top + size
    result = Image.new("RGB", (size, size), "black")
    source_left, source_top = max(0, left), max(0, top)
    source_right, source_bottom = min(image.width, right), min(image.height, bottom)
    if source_right > source_left and source_bottom > source_top:
        result.paste(
            image.crop((source_left, source_top, source_right, source_bottom)),
            (source_left - left, source_top - top),
        )
    return result


def nearest_hotspot(candidates: list[dict[str, Any]], center_y: int, center_x: int) -> dict[str, Any] | None:
    """Find the ranked hotspot nearest to one bounded stable component."""
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (item["center_y_x"][0] - center_y) ** 2
        + (item["center_y_x"][1] - center_x) ** 2,
    )


def build_html(*, sample_id: str, finding: dict[str, Any], images: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<figure><img src="{html.escape(filename)}" alt="{html.escape(label)}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
        for filename, label in images
    )
    candidate = finding["candidate"]
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework - high-sensitivity review - {html.escape(sample_id)}</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#09111d; color:#e7edf8 }}
  body {{ margin:0; padding:24px; max-width:1600px; margin-inline:auto }}
  header, .notice {{ background:#101c30; border:1px solid #29415f; border-radius:12px; padding:18px; margin-bottom:16px }}
  h1 {{ margin:0 0 6px; font-size:1.55rem }} h2 {{ font-size:1rem; margin:0 0 10px }}
  .notice {{ border-color:#80621d; background:#211d0e; line-height:1.5 }}
  .meta {{ color:#afc0d9; line-height:1.55 }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px }}
  figure {{ margin:0; background:#0c1728; border:1px solid #29415f; border-radius:10px; overflow:hidden }}
  img {{ display:block; width:100%; aspect-ratio:1; object-fit:contain; image-rendering:auto; background:#000 }}
  figcaption {{ padding:9px 11px; font-weight:650; font-size:.92rem }}
  code {{ background:#17263d; padding:2px 5px; border-radius:4px }}
</style>
<header><h1>High-sensitivity CT review</h1><div class="meta">{html.escape(sample_id)} · candidato <code>{html.escape(candidate['candidate_id'])}</code> · evidencia preservada</div></header>
<section class="notice"><strong>This is not a positive.</strong> The activation did not pass the text gate: there is one stable shape and one band. These images let you check whether the pattern persists physically in CT and whether a fibre, an edge or geometry explains it. They do not let you accept ink or letters.</section>
<section class="meta"><strong>Location in the analysis space:</strong> x={candidate['bbox_xyxy'][0]}-{candidate['bbox_xyxy'][2]}, y={candidate['bbox_xyxy'][1]}-{candidate['bbox_xyxy'][3]}. &nbsp; <strong>Acuerdo:</strong> {candidate['agreement_iou']:.3f}; <strong>medias de réplica:</strong> {candidate['model_a_mean']:.3f} / {candidate['model_b_mean']:.3f}.</section>
<h2>Compare the same area, layer by layer</h2><main class="grid">{cells}</main>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=384)
    args = parser.parse_args()

    analysis_path = args.analysis_json.resolve()
    analysis = json.loads(analysis_path.read_text())
    screening = analysis["text_like_screening"]
    stable = screening["candidates"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    layer_dir = analysis_path.parent / "comparison_layers"
    if not layer_dir.is_dir():
        raise FileNotFoundError(f"missing comparison layers: {layer_dir}")

    findings: list[dict[str, Any]] = []
    for candidate in stable:
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        center_x, center_y = round((x0 + x1) / 2), round((y0 + y1) / 2)
        selected = nearest_hotspot(analysis.get("ranked_review_hotspots", []), center_y, center_x)
        candidate_dir = output / candidate["candidate_id"]
        candidate_dir.mkdir(exist_ok=True)
        written: list[tuple[str, str]] = []
        for source_name, label in VIEW_NAMES.items():
            source = layer_dir / source_name
            if not source.is_file():
                continue
            destination_name = f"{Path(source_name).stem}-crop.png"
            with Image.open(source) as image:
                bounded_crop(image, center_x=center_x, center_y=center_y, size=args.crop_size).save(candidate_dir / destination_name)
            written.append((destination_name, label))
        finding = {
            "kind": "campaign_x_high_sensitivity_ink_activation_v1",
            "status": "UNVERIFIED_HIGH_SENSITIVITY_ACTIVATION",
            "decision": "CT_REVIEW_EVIDENCE_ONLY",
            "sample_id": analysis["sample_id"],
            "candidate": candidate,
            "nearest_ranked_hotspot": selected,
            "source_analysis": str(analysis_path),
            "strict_screening_outcome": screening["screening_outcome"],
            "strict_screening_policy": screening["policy"],
            "explicit_non_claims": [
                "not accepted ink", "not text", "not a First Letters claim",
                "not a geometry validation or rejection",
            ],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "views": [{"file": name, "label": label} for name, label in written],
        }
        (candidate_dir / "FINDING.json").write_text(json.dumps(finding, indent=2) + "\n")
        (candidate_dir / "REVIEW.html").write_text(build_html(sample_id=analysis["sample_id"], finding=finding, images=written))
        findings.append({"candidate_id": candidate["candidate_id"], "directory": candidate["candidate_id"], "status": finding["status"]})

    index = {
        "kind": "campaign_x_high_sensitivity_ink_review_index_v1",
        "status": "EVIDENCE_ONLY",
        "source_analysis": str(analysis_path),
        "strict_screening_outcome": screening["screening_outcome"],
        "findings": findings,
    }
    (output / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    links = "".join(
        f'<li><a href="{html.escape(item["directory"])}/REVIEW.html">'
        f'{html.escape(item["candidate_id"])}</a> — {html.escape(item["status"])}</li>'
        for item in findings
    ) or "<li>No hubo activaciones estables bajo el umbral estricto.</li>"
    (output / "INDEX.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Helena Framework · alta sensibilidad</title>"
        "<style>body{font-family:system-ui;max-width:760px;margin:3rem auto;line-height:1.5}"
        "a{color:#075ab5}</style><h1>Activaciones de alta sensibilidad</h1>"
        "<p>Not accepted ink and not text. Each link keeps its CT and replica evidence.</p>"
        f"<ul>{links}</ul>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
