"""Build strips/pherc0332_pack/: the 5 qualified PHerc0332 strips (the 4
v2 windows -- 11000, 11300, 10600, 13400 -- plus the v1 anchor window,
12750) and PACK.md, its gates/pitch table.

Offline: reads only configs/pherc0332.json, fixtures/band_r1145_200_xyz.npz
and fixtures/windows_v2.json (for the 4 v2 windows' pitch/stratum/kappa
provenance). Re-run this script to regenerate the pack after any change
to stb.strips/gates/reference; tests/test_strips.py exercises the same
export_strip/load_strip/qualify_strip path independently (on temp-dir
strips, not this pack) and is the actual regression gate.
"""
import json
from pathlib import Path

from stb import band as stb_band
from stb import config as stb_config
from stb import gates as gates_mod
from stb import reference as reference_mod
from stb import strips as strips_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = Path(__file__).resolve().parent / "pherc0332_pack"
V1_WINDOW_START = 12750  # benchmark_core.C0 -- the original v1 seed window


def main():
    cfg = stb_config.load_config(REPO_ROOT / "configs" / "pherc0332.json")
    xyz, valid, _row0 = stb_band.load_band(cfg.band_path)
    cfg = stb_config.resolve(cfg, xyz, valid)

    with open(REPO_ROOT / "fixtures" / "windows_v2.json") as f:
        windows_v2 = {w["start"]: w for w in json.load(f)["windows"]}

    specs = [(start, w.get("p2_ct"), {"stratum": w.get("stratum"), "kappa": w.get("kappa")})
              for start, w in windows_v2.items()]
    specs.append((V1_WINDOW_START, None,
                  {"note": "v1 anchor window (benchmark_core.C0:C1); "
                           "no CT pitch/stratum/kappa recorded for it "
                           "anywhere in this repo's fixtures"}))
    specs.sort(key=lambda s: s[0])

    PACK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for start, pitch, meta in specs:
        ref = reference_mod.reference_at(xyz, valid, start, cfg)
        g = gates_mod.coverage_and_gates_ab(ref, cfg)
        path = PACK_DIR / f"strip_s{start:05d}.npz"
        # export_strip's default neighborhood_margin (300) is a generous,
        # general-purpose default (and what tests/test_strips.py uses);
        # PHerc0332's actual matched-neighborhood width per class is
        # already close to the window's own 200-column width (checked
        # directly), so a smaller margin keeps this committed pack a
        # more reasonable size while remaining fully verified: this exact
        # value is re-qualified and asserted-passing (gate a/b both, all
        # 5 windows) right here, right before PACK.md is written.
        strips_mod.export_strip(ref, start, cfg, g, path, pitch=pitch, meta=meta,
                                 neighborhood_margin=80)

        strip = strips_mod.load_strip(path)
        q = strips_mod.qualify_strip(strip)
        rows.append({
            "start": start,
            "stratum": (meta or {}).get("stratum"),
            "kappa": (meta or {}).get("kappa"),
            "coverage": g.get("coverage"),
            "gate_a_pass": g.get("gate_a_pass"),
            "gate_b_pass": g.get("gate_b_pass"),
            "requalified_gate_a_pass": q.get("gate_a_pass"),
            "requalified_gate_b_pass": q.get("gate_b_pass"),
            "pitch_p2_ct": pitch,
            "file": path.name,
            "file_bytes": path.stat().st_size,
        })
        print(f"start={start:5d} gate_a={g.get('gate_a_pass')} gate_b={g.get('gate_b_pass')} "
              f"requalified_a={q.get('gate_a_pass')} requalified_b={q.get('gate_b_pass')} "
              f"-> {path.name} ({path.stat().st_size} bytes)")

    write_pack_md(rows)
    print(f"wrote {PACK_DIR / 'PACK.md'}")


def write_pack_md(rows):
    lines = []
    lines.append("# PHerc0332 qualified strip pack")
    lines.append("")
    lines.append(
        "Five strips exported by stb.strips.export_strip from the PHerc0332 "
        "band (fixtures/band_r1145_200_xyz.npz), each independently "
        "re-qualified (stb.strips.load_strip + qualify_strip) after the "
        "export/load round-trip -- see build_pherc0332_pack.py. The 4 "
        "\"v2\" windows are the ones fixtures/windows_v2.json selected "
        "(stratified low/low/median/median over eligible kappa, gate-c "
        "pitch-agreement pass); window 12750 is the original v1 anchor "
        "window (benchmark_core.C0:C1), which predates gate c and carries "
        "no CT pitch/stratum/kappa provenance in this repo's fixtures."
    )
    lines.append("")
    lines.append(
        "| start | stratum | kappa | coverage | gate a | gate b | "
        "requalified a | requalified b | pitch (p2_ct, vox) | file |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        kappa = f"{r['kappa']:.4f}" if r["kappa"] is not None else "-"
        coverage = f"{r['coverage']:.4f}" if r["coverage"] is not None else "-"
        pitch = f"{r['pitch_p2_ct']:.1f}" if r["pitch_p2_ct"] is not None else "-"
        lines.append(
            f"| {r['start']} | {r['stratum'] or '-'} | {kappa} | {coverage} | "
            f"{r['gate_a_pass']} | {r['gate_b_pass']} | "
            f"{r['requalified_gate_a_pass']} | {r['requalified_gate_b_pass']} | "
            f"{pitch} | `{r['file']}` |"
        )
    lines.append("")
    lines.append(
        "\"gate a\"/\"gate b\" are stb.gates.coverage_and_gates_ab's own "
        "result at export time; \"requalified a\"/\"requalified b\" are "
        "stb.strips.qualify_strip's independent re-computation from ONLY "
        "the exported .npz (no band npz, no original Reference) -- both "
        "columns agreeing for all 5 rows is exactly what "
        "tests/test_strips.py::test_export_and_qualify_all_5_windows pins."
    )
    lines.append("")
    lines.append(
        "Pitch is windows_v2.json's gate-c CT-measured spacing (`p2_ct`, "
        "vox) for the 4 v2 windows; window 12750 has none recorded (see "
        "above)."
    )
    lines.append("")
    (PACK_DIR / "PACK.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
