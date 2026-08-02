#!/usr/bin/env python3
"""Build the label-blind CT adjudication viewer for MULTISCROLL_TRANSFER_V1.

The generated HTML never loads v3/v4 decisions.  It records a human decision
against the frozen proposal and downloads a JSON receipt; it does not mutate
the benchmark manifest or execute either gate.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def build(proposals_path: Path, output_path: Path) -> None:
    proposals = json.loads(proposals_path.read_text())
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("proposal input must be a non-empty JSON array")
    if any(row.get("expected_class") is not None for row in proposals):
        raise ValueError("viewer input must remain label-blind")
    payload = json.dumps(proposals, separators=(",", ":")).replace("</", "<\\/")
    relative_prefix = Path(
        proposals_path.parent.relative_to(output_path.parent)
        if proposals_path.parent != output_path.parent
        else "."
    )
    prefix = "" if str(relative_prefix) == "." else f"{relative_prefix.as_posix()}/"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MULTISCROLL_TRANSFER_V1 - CT adjudication</title>
<style>
:root{{--bg:#080d15;--panel:#111b2a;--line:#2b405d;--text:#eef5ff;--muted:#9fb0c8;--gold:#ffc94f;--blue:#55aaff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,sans-serif}}
header{{height:74px;padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}}
h1{{font-size:21px;margin:0}} .sub,.help{{color:var(--muted);font-size:13px}} main{{height:calc(100vh - 74px);padding:12px;display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:12px}}
.image{{min-width:0;background:#03060a;border:1px solid var(--line);border-radius:10px;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.image img{{max-width:100%;max-height:100%;object-fit:contain}} aside{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;overflow:auto}}
.badge{{display:inline-block;padding:3px 8px;border-radius:999px;background:#263954;color:#dbeaff;font-size:12px;margin-right:5px}}
.question{{font-size:20px;font-weight:700;margin:14px 0 5px}} .warning{{background:#2a2513;border-left:4px solid var(--gold);padding:10px;margin:12px 0;color:#ffe8a3}}
.answers{{display:grid;gap:8px;margin-top:12px}} button,select,input,textarea{{font:inherit;color:var(--text);background:#172338;border:1px solid #405675;border-radius:7px}}
button{{padding:11px;text-align:left;cursor:pointer}} button:hover{{border-color:var(--blue);background:#203655}} button.selected{{outline:2px solid var(--blue)}}
input,select,textarea{{width:100%;padding:8px;margin:4px 0 10px}} textarea{{min-height:70px;resize:vertical}} label{{font-size:13px;color:var(--muted)}}
.nav{{display:flex;gap:8px;margin-top:12px}} .nav button{{flex:1;text-align:center}} .meta{{font-size:13px;line-height:1.5;color:var(--muted);word-break:break-word}}
.progress{{font-weight:700}} @media(max-width:900px){{header{{height:auto}}main{{height:auto;grid-template-columns:1fr}}.image{{height:65vh}}}}
</style></head><body>
<header><div><h1>MULTISCROLL_TRANSFER_V1 - blind CT</h1><div class="sub">The prediction only chose locations. Nothing is shoestran decisiones v3/v4.</div></div>
<div><span class="progress" id="progress"></span> <button id="export">Exportar JSON</button></div></header>
<main><section class="image"><img id="review" alt="mosaico CT"></section>
<aside>
<div><span class="badge" id="scroll"></span><span class="badge" id="stratum"></span></div>
<div class="question">What does the CT directly demonstrate at this location?</div>
<div class="help">Use the locator to orient yourself only. Decide from the CT slice, the gradient, XZ/YZ and the depth montage.</div>
<div class="warning">Si el punto cae en un borde o falta cobertura, elige “extender”; no lo conviertas en negativo.</div>
<label>Revisor humano<input id="reviewer" placeholder="nombre o identificador"></label>
<label><input id="certify" type="checkbox" style="width:auto"> I reviewed the CT without seeing the v3/v4 results</label>
<div class="answers" id="answers"></div>
<label>Traceable note<textarea id="note" placeholder="What is visible and in which views; e.g. the central CT + XZ show signal attached to the lamina"></textarea></label>
<div class="meta" id="meta"></div>
<div class="nav"><button id="prev">← Anterior</button><button id="next">Siguiente →</button></div>
</aside></main>
<script>
const proposals={payload};
const prefix={json.dumps(prefix)};
const choices=[
 ["POSITIVE","Ink confirmed by CT","P"],
 ["CONFOUND:FIBER","Confusor: fibra","F"],
 ["CONFOUND:EDGE","Confusor: borde","E"],
 ["CONFOUND:CRACK","Confusor: grieta","C"],
 ["CONFOUND:DIFFUSE_SIGNAL","Confounder: diffuse signal","D"],
 ["CONFOUND:OTHER_ADJUDICATED_NON_INK","Other adjudicated non-ink","O"],
 ["TIER_C_EXTEND_OR_RESEGMENT","Cobertura insuficiente: extender","X"],
 ["INCONCLUSIVE","No puedo decidir","I"]];
let index=0; const key="MULTISCROLL_TRANSFER_V1_ADJUDICATIONS"; let saved=JSON.parse(localStorage.getItem(key)||"{{}}");
const $=id=>document.getElementById(id);
function render(){{
 const p=proposals[index], a=saved[p.proposal_id]||{{}};
 $("progress").textContent=`${{index+1}} / ${{proposals.length}} · ${{Object.keys(saved).length}} guardadas`;
 $("review").src=prefix+p.review_image; $("scroll").textContent=p.scroll_id+" · "+p.segment_id; $("stratum").textContent=p.proposal_stratum;
 $("reviewer").value=a.reviewer||localStorage.getItem(key+":reviewer")||""; $("certify").checked=!!a.certified_blind_review; $("note").value=a.note||"";
 $("answers").innerHTML=""; choices.forEach(([value,label,hotkey])=>{{let b=document.createElement("button");b.textContent=`${{hotkey}} · ${{label}}`;if(a.decision===value)b.className="selected";b.onclick=()=>save(value);$("answers").appendChild(b)}});
 $("meta").textContent=`proposal=${{p.proposal_id}} · CT xyz=${{p.ct_coordinate_xyz.map(v=>v.toFixed(1)).join(", ")}} · voxel=${{p.voxel_size_um[0]}} µm · ${{p.scanner_domain}}`;
}}
function save(decision){{
 const p=proposals[index], reviewer=$("reviewer").value.trim();
 if(!reviewer||!$("certify").checked){{alert("Name the reviewer and certify the blind review.");return}}
 localStorage.setItem(key+":reviewer",reviewer);
 const [expected,subtype]=decision.startsWith("CONFOUND:")?decision.split(":"):[decision,null];
 saved[p.proposal_id]={{proposal_id:p.proposal_id,decision,expected_class:["POSITIVE","CONFOUND"].includes(expected)?expected:null,confound_subtype:subtype,label_authority:["POSITIVE","CONFOUND"].includes(expected)?"HUMAN_CT_ADJUDICATION":null,reviewer,certified_blind_review:true,note:$("note").value.trim(),reviewed_at:new Date().toISOString()}};
 localStorage.setItem(key,JSON.stringify(saved)); if(index<proposals.length-1)index++; render();
}}
$("prev").onclick=()=>{{index=Math.max(0,index-1);render()}};$("next").onclick=()=>{{index=Math.min(proposals.length-1,index+1);render()}};
$("export").onclick=()=>{{const rows=proposals.map(p=>({{...p,adjudication:saved[p.proposal_id]||null}}));const blob=new Blob([JSON.stringify({{benchmark_id:"MULTISCROLL_TRANSFER_V1",exported_at:new Date().toISOString(),rows}},null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="MULTISCROLL_TRANSFER_V1_ADJUDICATIONS.json";a.click();URL.revokeObjectURL(a.href)}};
document.addEventListener("keydown",e=>{{if(e.target.matches("input,textarea,select"))return;const c=choices.find(x=>x[2].toLowerCase()===e.key.toLowerCase());if(c)save(c[0]);if(e.key==="ArrowLeft")$("prev").click();if(e.key==="ArrowRight")$("next").click()}});
render();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.proposals.resolve(), args.output.resolve())
    print(json.dumps({"status": "VIEWER_READY", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
