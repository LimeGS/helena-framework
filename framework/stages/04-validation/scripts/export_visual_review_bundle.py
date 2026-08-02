#!/usr/bin/env python3
"""Export every visual reviewed by the Phase 4 robust screen.

The robust GPU run keeps its complete 65-slice TIFF stacks in the compute
workspace.  This exporter creates a lightweight, self-contained manual-review
bundle containing the central CT image, every rendered probability/stability
map, every hotspot crop and montage, the HTML viewer, provenance receipts,
logs, and a SHA-256 manifest.  It intentionally does not duplicate NPY arrays
or all 65 source TIFFs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_optional_gate_groups(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None or not path.is_file():
        return {}
    payload = read_json(path)
    if payload.get("status") != "COMPLETED":
        raise RuntimeError(f"CT gate evaluation is not completed: {path}")
    groups = payload.get("by_group")
    if not isinstance(groups, dict):
        raise RuntimeError(f"CT gate evaluation lacks by_group: {path}")
    return groups


def copy_file(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    output_root: Path,
    inventory: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    inventory.append(
        {
            "source": str(source.relative_to(source_root)),
            "bundle_path": str(destination.relative_to(output_root)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    )


POSITIVE_SCREENING_OUTCOME = "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"


def selected_files(window: Path, screening_name: str) -> list[Path]:
    files: list[Path] = []
    for name in (
        "analysis.stdout.log",
        "crop.stdout.log",
        "inference.stdout.log",
        f"{screening_name}.analysis.stdout.log",
        f"{screening_name}.inference.stdout.log",
    ):
        path = window / name
        if path.is_file():
            files.append(path)

    for name in ("32.tif", "PHYSICAL_CROP_RECEIPT.json"):
        path = window / "tiffs" / name
        if path.is_file():
            files.append(path)

    screening = window / screening_name
    receipt = screening / "INK_SCREENING_RECEIPT.json"
    if receipt.is_file():
        files.append(receipt)
    files.extend(sorted(screening.glob("*.png")))

    analysis = screening / "analysis"
    for path in sorted(analysis.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".html",
            ".json",
            ".png",
            ".txt",
        }:
            files.append(path)
    return files


COMPARISON_LAYER_HELP = {
    "ct": (
        "CT central",
        "Look at the original material. A useful signal should look like a thin stroke; "
        "bandas gruesas, bordes dobles y textura repetitiva suelen ser fibras.",
    ),
    "mean_probability": (
        "Probabilidad media",
        "The average of the six replicas, on the same fixed scale. White means "
        "the strongest model response, but it does not prove ink. Averaging smooths "
        "the edges: this layer locates signal and must not be read as OCR.",
    ),
    "robust_minimum": (
        "Minimum of six replicas",
        "Keeps only the weakest response at each pixel. A shape that survives "
        "visible here survived three depths and two offsets.",
    ),
    "replica_disagreement": (
        "Replica disagreement",
        "White means instability. To trust a shape we want little "
        "desacuerdo —fondo oscuro— sobre el mismo lugar.",
    ),
    "replica_agreement": (
        "Fraction of replicas > 0.5",
        "White means more replicas crossed the threshold. We are looking for shapes "
        "delgadas, acotadas y repetidas, no masas grandes.",
    ),
    "persistent_overlay": (
        "Overlay persistente sobre CT",
        "Orange/red shows the persistent response over the material. Confirm "
        "whether it falls on an independent stroke or only on a fibre, edge or void.",
    ),
}


def render_like_for_like_comparator(
    window_summaries: list[dict[str, Any]],
    control_summaries: dict[str, Any],
    screening_name: str,
    *,
    control_ct_gate_by_group: dict[str, dict[str, int]] | None = None,
    target_ct_gate_by_group: dict[str, dict[str, int]] | None = None,
) -> str:
    control_ct_gate_by_group = control_ct_gate_by_group or {}
    target_ct_gate_by_group = target_ct_gate_by_group or {}
    control_group_ids = {
        "cross-scroll-control": "PHerc0139-public-positive",
        "second-scroll-control": "PHerc172-public-positive",
    }
    controls = []
    for bundle_name in ("cross-scroll-control", "second-scroll-control"):
        item = control_summaries[bundle_name]
        passed = item["screening_outcome"] == POSITIVE_SCREENING_OUTCOME
        ct_gate = control_ct_gate_by_group.get(
            control_group_ids[bundle_name]
        )
        ct_suffix = ""
        if ct_gate is not None:
            retained = int(ct_gate.get("retained", 0))
            downranked = int(ct_gate.get("downranked", 0))
            ct_suffix = f" · CT localizado {retained}/{retained + downranked}"
        controls.append(
            {
                "base": f"{bundle_name}/comparison_layers",
                "badge": item["badge"],
                "title": item["title"],
                "status": (
                    f"SIGNAL GATE PASSED (NOT OCR): "
                    f"{item['glyph_like_candidate_count']} formas persistentes · "
                    f"{item['rows_with_at_least_four_candidates']} renglones"
                    f"{ct_suffix}"
                    if passed
                    else (
                        f"SIGNAL GATE FAILED: "
                        f"{item['glyph_like_candidate_count']} formas persistentes · "
                        f"{item['rows_with_at_least_four_candidates']} renglones"
                    )
                ),
                "status_class": "pass" if passed else "fail",
            }
        )
    data = json.dumps(
        {
            "controls": controls,
            "targets": [
                {
                    "label": item["window"],
                    "sample_id": item["sample_id"],
                    "base": (
                        f"{item['window']}/{screening_name}/analysis/"
                        "comparison_layers"
                    ),
                    "forms": item["glyph_like_candidate_count"],
                    "rows": item["rows_with_at_least_four_candidates"],
                    "passed": (
                        item["screening_outcome"]
                        == POSITIVE_SCREENING_OUTCOME
                    ),
                    "orthogonal_ct": item.get("orthogonal_ct"),
                    "ct_gate": target_ct_gate_by_group.get(item["window"]),
                }
                for item in window_summaries
            ],
            "layers": [
                {
                    "key": key,
                    "title": title,
                    "help": help_text,
                    "file": f"{key}.png",
                }
                for key, (title, help_text) in COMPARISON_LAYER_HELP.items()
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    if control_ct_gate_by_group:
        control_retained = sum(
            int(control_ct_gate_by_group.get(group_id, {}).get("retained", 0))
            for group_id in control_group_ids.values()
        )
        control_total = sum(
            sum(
                int(value)
                for key, value in control_ct_gate_by_group.get(
                    group_id, {}
                ).items()
                if key in {"retained", "downranked"}
            )
            for group_id in control_group_ids.values()
        )
        control_scope = (
            f"Both controls pass the signal gate and the CT filter keeps "
            f"{control_retained}/{control_total} componentes."
        )
    else:
        control_scope = "Both controls pass the signal gate."
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework - like-for-like comparison</title>
<style>
:root{{--bg:#070c14;--panel:#111a29;--line:#304661;--text:#eff6ff;--muted:#a9b8cc;--blue:#70b8ff;--amber:#f4c45d;--red:#ff858b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.35 system-ui,sans-serif;overflow:hidden}}
header{{height:86px;padding:10px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:14px}}
h1,h2,p{{margin:0}} h1{{font-size:clamp(21px,2.3vw,32px)}} .muted{{color:var(--muted)}} select,button,a{{background:#17243a;color:var(--text);border:1px solid #4a6586;border-radius:8px;padding:9px 11px;font:inherit}}
a{{text-decoration:none}} .controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}}
main{{height:calc(100vh - 86px);display:grid;grid-template-rows:auto 1fr;gap:8px;padding:8px}}
.lesson{{background:#172238;border-left:4px solid var(--amber);border-radius:7px;padding:8px 12px;display:grid;grid-template-columns:minmax(180px,.35fr) 1fr;gap:12px}}
.lesson strong{{color:var(--amber)}} .compare{{min-height:0;display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.scope{{color:#dfeaff}}
.side{{min-width:0;min-height:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;display:grid;grid-template-rows:auto 1fr;overflow:hidden}}
.side.control{{border-color:#277b58}} .side.target{{border-color:#9b6f2a}}
.side-head{{padding:8px 11px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px}}
.side-head h2{{font-size:17px}} .badge{{font-size:12px;font-weight:800;padding:3px 7px;border-radius:999px;background:#23334b}}
.target-meta{{display:grid;justify-items:end;gap:4px}} .target-meta a{{padding:4px 7px;font-size:13px}}
.canvas{{min-height:0;background:#000;display:grid;place-items:center;overflow:hidden}} .canvas img{{width:100%;height:100%;object-fit:contain;display:block}}
.layers{{display:flex;gap:5px;flex-wrap:wrap}} .layer.selected{{outline:3px solid var(--blue);background:#24527d}}
.result.pass{{color:#9effc8}} .result.fail{{color:#ffd2a1}}
@media(max-width:900px){{body{{overflow:auto}} header{{height:auto;align-items:flex-start;flex-direction:column}} main{{height:auto}} .lesson{{grid-template-columns:1fr}} .compare{{grid-template-columns:1fr}} .side{{height:65vh}}}}
</style></head><body>
<header><div><h1>Control y objetivo · exactamente la misma salida</h1>
<p class="muted"><strong>{control_scope}</strong> Everything uses the same GP model, six replicas and a fixed scale. Passing localises signal compatible; no significa que el mapa reconstruya letras.</p></div>
<div class="controls"><div class="layers" id="layers"></div><select id="control"></select><select id="target"></select><a href="MANUAL_VISUAL_REVIEW_INDEX.html">Auditoría completa</a></div></header>
<main>
 <section class="lesson"><strong id="layer-title"></strong><span id="layer-help" class="scope"></span></section>
 <section class="compare">
  <article class="side control"><div class="side-head"><div><span class="badge" id="control-badge"></span><h2 id="control-title"></h2></div><span class="result" id="control-status"></span></div><div class="canvas"><img id="control-image"></div></article>
  <article class="side target"><div class="side-head"><div><span class="badge" id="target-badge"></span><h2 id="target-title"></h2></div><div class="target-meta"><span id="target-metrics" class="muted"></span><a id="orthogonal-link" hidden>Ver cortes CT XY/XZ/YZ</a></div></div><div class="canvas"><img id="target-image"></div></article>
 </section>
</main>
<script>
const DATA={data};let controlIndex=0,targetIndex=0,layerIndex=0;
const $=id=>document.getElementById(id);
function render(){{
 const control=DATA.controls[controlIndex],target=DATA.targets[targetIndex],layer=DATA.layers[layerIndex];
 const ctGate=target.ct_gate;
 $("control-image").src=control.base+"/"+layer.file;
 $("target-image").src=target.base+"/"+layer.file;
 $("control-badge").textContent=control.badge;
 $("control-title").textContent=control.title;
 $("control-status").textContent=control.status;
 $("control-status").className="result "+control.status_class;
 $("target-title").textContent=target.label;
 const ctTotal=ctGate ? Number(ctGate.retained||0)+Number(ctGate.downranked||0) : 0;
 $("target-metrics").textContent=ctGate
   ? `Signal: ${{target.forms}} shapes - ${{target.rows}} rows - CT localised: ${{Number(ctGate.retained||0)}}/${{ctTotal}}`
   : `${{target.forms}} formas · ${{target.rows}} filas pobladas`;
 $("target-badge").textContent=ctGate
   ? (Number(ctGate.retained||0)>0
      ? `CT RETIENE ${{ctGate.retained}}/${{ctTotal}} · REVISAR`
      : `CT 0/${{ctTotal}}: DIFFUSE SIGNAL - LIKELY FIBRE/LAMINA`)
   : (target.passed ? "SIGNAL GATE PASSED - REVIEW CT" : "SIGNAL GATE NOT PASSED - DOES NOT PROVE ABSENCE");
 const orthogonal=$("orthogonal-link");
 orthogonal.hidden=!target.orthogonal_ct;
 if(target.orthogonal_ct) orthogonal.href=target.orthogonal_ct;
 $("layer-title").textContent=layer.title;
 $("layer-help").textContent=layer.help;
 document.querySelectorAll(".layer").forEach((button,index)=>button.classList.toggle("selected",index===layerIndex));
 $("control").value=String(controlIndex);
 $("target").value=String(targetIndex);
}}
DATA.layers.forEach((layer,index)=>{{const button=document.createElement("button");button.className="layer";button.textContent=layer.title;button.onclick=()=>{{layerIndex=index;render()}};$("layers").append(button)}});
DATA.controls.forEach((control,index)=>{{const option=document.createElement("option");option.value=String(index);option.textContent=`Control ${{index+1}} · ${{control.title}}`;$("control").append(option)}});
DATA.targets.forEach((target,index)=>{{const option=document.createElement("option");option.value=String(index);option.textContent=`${{index+1}} · ${{target.label}}`;$("target").append(option)}});
$("control").onchange=event=>{{controlIndex=Number(event.target.value);render()}};
$("target").onchange=event=>{{targetIndex=Number(event.target.value);render()}};
addEventListener("keydown",event=>{{if(event.key>="1"&&event.key<="6"){{layerIndex=Number(event.key)-1;render()}}if(event.key==="ArrowRight"){{targetIndex=(targetIndex+1)%DATA.targets.length;render()}}if(event.key==="ArrowLeft"){{targetIndex=(targetIndex+DATA.targets.length-1)%DATA.targets.length;render()}}}});
render();
</script></body></html>"""


def render_index(
    window_summaries: list[dict[str, Any]],
    control_summaries: dict[str, Any],
    screening_name: str,
) -> str:
    cards: list[str] = []
    for bundle_name in ("cross-scroll-control", "second-scroll-control"):
        item = control_summaries[bundle_name]
        passed = item["screening_outcome"] == POSITIVE_SCREENING_OUTCOME
        cards.append(
            f"""
        <article class="card positive">
          <div class="card-head">
            <div><span class="badge positive">{html.escape(item["badge"])}</span>
            <h2>{html.escape(item["title"])}</h2></div>
            <a class="button" href="{bundle_name}/INK_HOTSPOT_REVIEW.html">Abrir visor detallado</a>
          </div>
          <p><strong>Same visualisation as the targets:</strong> six replicas of the GP Scroll1 model. Result: {"PASS" if passed else "FAIL"}, {int(item["glyph_like_candidate_count"])} persistent shapes and {int(item["rows_with_at_least_four_candidates"])} populated rows.</p>
          <img loading="lazy" src="{bundle_name}/stability_montage.png" alt="{html.escape(item["title"])}">
        </article>
        """
        )
    for item in window_summaries:
        outcome = str(item["screening_outcome"])
        is_positive = outcome == POSITIVE_SCREENING_OUTCOME
        css = "candidate" if is_positive else "negative"
        badge = (
            "REVISAR CT BRUTO"
            if is_positive
            else "GATE NOT PASSED - DOES NOT PROVE ABSENCE"
        )
        window = html.escape(str(item["window"]))
        sample_id = html.escape(str(item["sample_id"]))
        forms = int(item["glyph_like_candidate_count"])
        rows = int(item["rows_with_at_least_four_candidates"])
        cards.append(
            f"""
            <article class="card {css}">
              <div class="card-head">
                <div><span class="badge {css}">{badge}</span>
                <h2>{window}</h2></div>
                <a class="button" href="{html.escape(str(item["manual_viewer"]))}">Abrir visor detallado</a>
              </div>
              <p>{sample_id}  - persistent shapes: <strong>{forms}</strong> - rows with 4+: <strong>{rows}</strong></p>
              <div class="images">
                <figure><img loading="lazy" src="{window}/{screening_name}/analysis/hotspot_contact_sheet.png" alt="Hotspots {window}"><figcaption>CT y activaciones principales</figcaption></figure>
                <figure><img loading="lazy" src="{window}/{screening_name}/analysis/stability_montage.png" alt="Estabilidad {window}"><figcaption>Persistencia entre 3 profundidades × 2 offsets</figcaption></figure>
              </div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Helena Framework · positivos y negativos robustos</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070c14; --panel:#111a29; --line:#2b3c55; --text:#eef5ff; --muted:#a7b5ca; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.42 system-ui,sans-serif; }}
    header {{ position:sticky; top:0; z-index:2; padding:18px 24px; background:#09101ceF; border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
    h1,h2,p {{ margin:0; }} h1 {{ font-size:clamp(22px,3vw,36px); }} h2 {{ font-size:18px; overflow-wrap:anywhere; }}
    header p,.card>p,figcaption {{ color:var(--muted); }} main {{ padding:18px; display:grid; gap:18px; }}
    .card {{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:14px; display:grid; gap:12px; }}
    .card.positive {{ border-color:#267d56; }} .card.candidate {{ border-color:#c88d25; }}
    .card-head {{ display:flex; gap:12px; align-items:center; justify-content:space-between; }}
    .badge {{ display:inline-block; margin-bottom:5px; padding:3px 8px; border-radius:999px; font-weight:800; font-size:12px; letter-spacing:.04em; }}
    .badge.positive {{ color:#a9ffd2; background:#143d2b; }} .badge.negative {{ color:#c9d5e7; background:#263247; }}
    .badge.candidate {{ color:#ffe1a1; background:#513712; }}
    .button {{ color:#dceaff; text-decoration:none; border:1px solid #56749d; border-radius:9px; padding:9px 12px; white-space:nowrap; }}
    .images {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }} figure {{ margin:0; min-width:0; }}
    img {{ display:block; width:100%; max-height:62vh; object-fit:contain; background:#000; border-radius:8px; }}
    figcaption {{ padding-top:5px; font-size:13px; }}
    @media(max-width:800px) {{ .images {{ grid-template-columns:1fr; }} .card-head {{ align-items:flex-start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <header><h1>Helena Framework · evidencia visual preservada</h1>
  <p>2 positive controls from different scrolls with the same GP model + {len(window_summaries)} target windows. A positive gate is only a priority for CT review: it does not yet demonstrate ink or letters.</p></header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positive-control-analysis", type=Path)
    parser.add_argument("--transfer-control-analysis", type=Path)
    parser.add_argument("--ct-fiber-control-evaluation", type=Path)
    parser.add_argument("--ct-fiber-target-evaluation", type=Path)
    parser.add_argument(
        "--screening-name",
        default="ink_screening_gp_scroll1_v1",
    )
    parser.add_argument(
        "--batch-receipt-name",
        default="ROBUST_WINDOW_BATCH_RECEIPT_GP_SCROLL1.json",
    )
    parser.add_argument(
        "--robust-root-name",
        default="robust_windows_v1",
    )
    parser.add_argument(
        "--coarse-batch-receipt-name",
        default="EXPANDED_SURFACE_BATCH_RECEIPT.json",
    )
    parser.add_argument(
        "--ranking-name",
        default="GLOBAL_COARSE_WINDOW_RANKING.json",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    batch_root = (
        args.batch_root.resolve()
        if args.batch_root
        else root / "phase4" / "expanded_candidate_surface_screen_v1"
    )
    robust_root = batch_root / args.robust_root_name
    output = (
        args.output.resolve()
        if args.output
        else batch_root / "manual_review_bundle_v1"
    )
    positive_control_analysis = (
        args.positive_control_analysis.resolve()
        if args.positive_control_analysis
        else (
            root
            / "phase4"
            / "open_data"
            / "PHerc172"
            / "in_domain_control_v1"
            / "ink_screening_gp_scroll1_v1"
            / "analysis"
        )
    )
    transfer_control_analysis = (
        args.transfer_control_analysis.resolve()
        if args.transfer_control_analysis
        else (
            root
            / "phase4"
            / "open_data"
            / "PHerc0139"
            / "end_to_end_positive_control_v1"
            / "ink_screening_gp_scroll1_v1"
            / "analysis"
        )
    )
    default_ct_gate_root = root / "phase4" / "ct_fiber_benchmark_v1"
    control_ct_gate_evaluation = (
        args.ct_fiber_control_evaluation.resolve()
        if args.ct_fiber_control_evaluation
        else default_ct_gate_root
        / "gate_evaluation_v1"
        / "CT_FIBER_GATE_EVALUATION.json"
    )
    target_ct_gate_evaluation = (
        args.ct_fiber_target_evaluation.resolve()
        if args.ct_fiber_target_evaluation
        else default_ct_gate_root
        / "target_application_v1"
        / "gate"
        / "CT_FIBER_GATE_EVALUATION.json"
    )
    control_ct_gate_by_group = read_optional_gate_groups(
        control_ct_gate_evaluation
    )
    target_ct_gate_by_group = read_optional_gate_groups(
        target_ct_gate_evaluation
    )
    if output == robust_root or robust_root in output.parents:
        raise RuntimeError("output must not be inside the robust source tree")

    robust_receipt_path = robust_root / args.batch_receipt_name
    if not robust_receipt_path.is_file():
        raise RuntimeError("robust batch receipt is missing")
    robust_receipt = read_json(robust_receipt_path)
    completed = int(robust_receipt["completed_count"])
    receipt_results = robust_receipt.get("results")
    if not isinstance(receipt_results, list):
        raise RuntimeError("robust batch receipt lacks explicit results")
    windows: list[Path] = []
    for result in receipt_results:
        window_name = (
            f"rank-{int(result['global_rank']):02d}-"
            f"{result['sample_id']}-{result['surface_id']}"
        )
        window = robust_root / window_name
        analysis_path = (
            window
            / args.screening_name
            / "analysis"
            / "INK_STABILITY_ANALYSIS.json"
        )
        if not analysis_path.is_file():
            raise RuntimeError(
                f"receipt-selected window lacks analysis: {window_name}"
            )
        windows.append(window)
    if len({window.name for window in windows}) != len(windows):
        raise RuntimeError("robust batch receipt contains duplicate windows")
    if len(windows) != completed:
        raise RuntimeError(
            f"robust receipt says {completed} completed but found {len(windows)} windows"
        )

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    inventory: list[dict[str, Any]] = []

    for name in (
        args.coarse_batch_receipt_name,
        args.ranking_name,
    ):
        source = batch_root / name
        if source.is_file():
            copy_file(
                source,
                output / name,
                source_root=batch_root,
                output_root=output,
                inventory=inventory,
            )
    copy_file(
        robust_receipt_path,
        output / "ROBUST_WINDOW_BATCH_RECEIPT.json",
        source_root=batch_root,
        output_root=output,
        inventory=inventory,
    )

    window_summaries: list[dict[str, Any]] = []
    for window in windows:
        analysis_path = (
            window
            / args.screening_name
            / "analysis"
            / "INK_STABILITY_ANALYSIS.json"
        )
        analysis = read_json(analysis_path)
        screening = analysis["text_like_screening"]
        window_summaries.append(
            {
                "window": window.name,
                "sample_id": analysis["sample_id"],
                "screening_outcome": screening["screening_outcome"],
                "glyph_like_candidate_count": screening[
                    "glyph_like_candidate_count"
                ],
                "rows_with_at_least_four_candidates": screening[
                    "rows_with_at_least_four_candidates"
                ],
                "manual_viewer": str(
                    Path(window.name)
                    / args.screening_name
                    / "analysis"
                    / "INK_HOTSPOT_REVIEW.html"
                ),
                "central_ct": str(Path(window.name) / "tiffs" / "32.tif"),
                "orthogonal_ct": (
                    str(
                        Path(window.name)
                        / args.screening_name
                        / "analysis"
                        / "orthogonal_ct_v1"
                        / "ORTHOGONAL_CT_CONTACT_SHEET.png"
                    )
                    if (
                        window
                        / args.screening_name
                        / "analysis"
                        / "orthogonal_ct_v1"
                        / "ORTHOGONAL_CT_CONTACT_SHEET.png"
                    ).is_file()
                    else None
                ),
            }
        )
        for source in selected_files(window, args.screening_name):
            relative = source.relative_to(robust_root)
            copy_file(
                source,
                output / relative,
                source_root=batch_root,
                output_root=output,
                inventory=inventory,
            )

    required_layers = {f"{name}.png" for name in COMPARISON_LAYER_HELP}
    control_sources = (
        (
            "second-scroll-control",
            positive_control_analysis,
            "CONTROL POSITIVO 2 · PHerc172",
            "PHerc172 · modelo GP Scroll1",
        ),
        (
            "cross-scroll-control",
            transfer_control_analysis,
            "CONTROL POSITIVO CRUZADO · PHerc0139",
            "PHerc0139 · modelo GP Scroll1",
        ),
    )
    controls_ready = True
    control_summaries: dict[str, Any] = {}
    for bundle_name, control_analysis, badge, title in control_sources:
        control_layers = control_analysis / "comparison_layers"
        if not control_layers.is_dir():
            controls_ready = False
            continue
        actual_layers = {path.name for path in control_layers.glob("*.png")}
        if not required_layers <= actual_layers:
            raise RuntimeError(
                f"{bundle_name} lacks required comparison layers"
            )
        for source in sorted(control_layers.glob("*.png")):
            copy_file(
                source,
                output / bundle_name / "comparison_layers" / source.name,
                source_root=root,
                output_root=output,
                inventory=inventory,
            )
        for name in (
            "INK_STABILITY_ANALYSIS.json",
            "INK_HOTSPOT_REVIEW.html",
            "stability_montage.png",
            "hotspot_contact_sheet.png",
        ):
            source = control_analysis / name
            if source.is_file():
                copy_file(
                    source,
                    output / bundle_name / name,
                    source_root=root,
                    output_root=output,
                    inventory=inventory,
                )
        analysis_receipt = control_analysis / "INK_STABILITY_ANALYSIS.json"
        if analysis_receipt.is_file():
            screening = read_json(analysis_receipt)["text_like_screening"]
            control_summaries[bundle_name] = {
                "badge": badge,
                "title": title,
                "analysis": str(analysis_receipt.relative_to(root)),
                "screening_outcome": screening["screening_outcome"],
                "glyph_like_candidate_count": screening[
                    "glyph_like_candidate_count"
                ],
                "row_band_count": screening["row_band_count"],
                "rows_with_at_least_four_candidates": screening[
                    "rows_with_at_least_four_candidates"
                ],
            }

    expected_controls = {"cross-scroll-control", "second-scroll-control"}
    if not controls_ready or set(control_summaries) != expected_controls:
        raise RuntimeError("both GP positive controls are required")
    for item in window_summaries:
        target_layers = (
            output
            / item["window"]
            / args.screening_name
            / "analysis"
            / "comparison_layers"
        )
        missing = [
            name
            for name in required_layers
            if not (target_layers / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{item['window']} lacks comparison layers: {sorted(missing)}"
            )

    manifest = {
        "kind": "campaign_x_phase4_manual_visual_review_bundle_v2",
        "status": "COMPLETE",
        "generated_at_utc": utc_now(),
        "source_batch_root": str(batch_root),
        "source_robust_root": str(robust_root),
        "screening_name": args.screening_name,
        "completed_window_count": completed,
        "controls": control_summaries,
        "window_summaries": window_summaries,
        "ct_fiber_gate": {
            "control_evaluation": (
                str(control_ct_gate_evaluation)
                if control_ct_gate_by_group
                else None
            ),
            "target_evaluation": (
                str(target_ct_gate_evaluation)
                if target_ct_gate_by_group
                else None
            ),
            "control_by_group": control_ct_gate_by_group,
            "target_by_group": target_ct_gate_by_group,
        },
        "files": sorted(inventory, key=lambda item: item["bundle_path"]),
        "retention": {
            "copied": [
                "central CT TIFF for every robust window",
                "all probability and stability PNGs",
                "all hotspot crops, overlays, montages, and HTML viewers",
                "all provenance receipts and execution logs",
            ],
            "retained_only_in_compute_workspace": [
                "remaining 64 source TIFF slices per window",
                "NPY probability arrays",
            ],
            "future_rule": (
                "any non-central source slice inspected visually must be added "
                "to this bundle before the review is closed"
            ),
        },
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not a First Letters submission claim",
        ],
    }
    (output / "MANUAL_VISUAL_REVIEW_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "MANUAL_VISUAL_REVIEW_INDEX.html").write_text(
        render_index(window_summaries, control_summaries, args.screening_name)
    )
    (output / "LIKE_FOR_LIKE_COMPARATOR.html").write_text(
        render_like_for_like_comparator(
            window_summaries,
            control_summaries,
            args.screening_name,
            control_ct_gate_by_group=control_ct_gate_by_group,
            target_ct_gate_by_group=target_ct_gate_by_group,
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "completed_window_count": completed,
                "copied_file_count": len(inventory),
                "copied_size_bytes": sum(item["size_bytes"] for item in inventory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
