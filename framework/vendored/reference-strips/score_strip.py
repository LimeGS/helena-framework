"""One-command scorer against a qualified reference strip.

Two modes, one standard scorecard out (JSON + markdown):

POINTS mode (tracers)
    python score_strip.py --strip strip.npz --mode points \
        --pred pred.npz --from-wrap 1 --direction front

    pred.npz holds an (M, 3) float array of predicted xyz points (key
    `points`, `xyz`, or the file's single array; a bare .npy also works).
    Headline number: wrong-hop % = fraction of included predictions that
    either land nearest to a wrap other than the target (wrong-wrap %) or
    land on the target wrap but at >= half the local gap (distance-miss %).
    Scoring rule provenance: ported from the source audit -- see
    scoring_core.py's provenance header.

MESH mode (meshers)
    python score_strip.py --strip strip.npz --mode mesh --pred mesh.obj
    python score_strip.py --strip strip.npz --mode mesh --pred tifxyz_dir/

    Metrics: non-manifold edge count (edges with >2 incident faces), open
    boundary edge count (edges with exactly 1 face), connected component
    count, and the strip-local cross-wrap fusion (bridge) count with a
    fused-pair histogram. These are GEOMETRIC CHECKS AGAINST THIS STRIP,
    not a global topology certification -- see mesh_metrics.py.

If the strip has no passing qualification report next to it
(<strip>.qualification.json with overall_pass: true, written by
qualify_strip.py), the scorer still runs but prints an UNQUALIFIED warning
and stamps it into the scorecard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mesh_metrics import (
    connected_components,
    fusion_analysis,
    grid_to_mesh,
    parse_obj,
    topology_counts,
)
from scoring_core import build_trees, score_points, summarize
from strip_format import is_qualified, load_strip


def load_pred_points(path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if "points" in data.files:
            arr = data["points"]
        elif "xyz" in data.files:
            arr = data["xyz"]
        elif len(data.files) == 1:
            arr = data[data.files[0]]
        else:
            raise SystemExit(
                f"{path}: expected key 'points' or 'xyz' (or a single "
                f"array); found {data.files}"
            )
    else:
        raise SystemExit(f"{path}: points mode expects a .npz or .npy file")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise SystemExit(f"{path}: expected shape (M, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"{path}: predictions contain NaN/Inf")
    return arr


def load_pred_mesh(path):
    path = Path(path)
    if path.is_dir():
        # tifxyz directory
        from make_strip import load_tifxyz_grids  # optional tifffile dep
        x, y, z, valid = load_tifxyz_grids(path)
        return grid_to_mesh(x, y, z, valid)
    if path.suffix.lower() == ".obj":
        return parse_obj(path)
    raise SystemExit(f"{path}: mesh mode expects a .obj file or a tifxyz directory")


def strip_summary(strip, strip_path, qualified: bool) -> dict:
    return {
        "strip_path": str(strip_path),
        "qualified": qualified,
        "schema_version": strip.meta.get("schema_version"),
        "scroll": strip.meta.get("scroll"),
        "segment_id": strip.meta.get("segment_id"),
        "tier": strip.meta.get("tier"),
        "n_wraps": strip.n_wraps,
        "wrap_indices": strip.wrap_indices,
        "pitch_um": strip.pitch_um,
        "voxel_size_um": strip.meta.get("voxel_size_um"),
    }


# ----------------------------------------------------------------------
# Points mode
# ----------------------------------------------------------------------

def run_points(strip, args, qualified: bool) -> dict:
    pred = load_pred_points(args.pred)
    trees = build_trees(strip)
    result = score_points(
        strip, trees, pred,
        from_wrap=args.from_wrap,
        direction=args.direction,
        gap_fraction=args.gap_fraction,
    )
    s = summarize(result)

    ok = result["ok"]
    dist_stats = {}
    if s["included"]:
        d = result["d_target"][ok]
        g = result["gap"][ok]
        dist_stats = {
            "median_dist_to_target_native": float(np.median(d)),
            "p90_dist_to_target_native": float(np.percentile(d, 90)),
            "median_local_gap_native": float(np.median(g)),
            "p10_local_gap_native": float(np.percentile(g, 10)),
            "p90_local_gap_native": float(np.percentile(g, 90)),
        }
        voxel_um = strip.meta.get("voxel_size_um")
        if voxel_um:
            dist_stats["median_dist_to_target_um"] = float(np.median(d) * voxel_um)
            dist_stats["median_local_gap_um"] = float(np.median(g) * voxel_um)

    return {
        "mode": "points",
        "strip": strip_summary(strip, args.strip, qualified),
        "pred_path": str(args.pred),
        "from_wrap": args.from_wrap,
        "direction": args.direction,
        "target_wrap": result["target_wrap"],
        "gap_fraction_threshold": args.gap_fraction,
        "counts": {
            "total_predictions": s["total"],
            "included": s["included"],
            "excluded": s["excluded"],
            "excluded_no_coverage": s["excluded_no_coverage"],
            "excluded_edge": s["excluded_edge"],
            "hop_correct": s["correct"],
            "wrong_hop": s["wrong_hop"],
            "wrong_wrap": s["wrong_wrap"],
            "distance_miss": s["distance_miss"],
        },
        "percentages_of_included": {
            "hop_correct_pct": s["correct_pct"],
            "wrong_hop_pct": s["wrong_hop_pct"],
            "wrong_wrap_pct": s["wrong_wrap_pct"],
            "distance_miss_pct": s["distance_miss_pct"],
        },
        "distances": dist_stats,
        "coverage_radius_native": float(result["coverage_radius"]),
    }


def points_markdown(card: dict) -> str:
    c, p = card["counts"], card["percentages_of_included"]
    lines = [
        "# Strip scorecard (points mode)",
        "",
        f"Strip: `{card['strip']['strip_path']}` "
        f"({card['strip']['scroll']} / {card['strip']['segment_id']}, "
        f"tier {card['strip']['tier']})",
        f"Qualification: "
        f"{'QUALIFIED' if card['strip']['qualified'] else '**UNQUALIFIED -- numbers below are against an unverified reference**'}",
        f"Predictions: `{card['pred_path']}` "
        f"(from wrap {card['from_wrap']}, direction {card['direction']}, "
        f"target wrap {card['target_wrap']})",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Wrong-hop %** | **{p['wrong_hop_pct']:.1f}%** |",
        f"| Hop-correct % | {p['hop_correct_pct']:.1f}% |",
        f"| Wrong-wrap % (identity) | {p['wrong_wrap_pct']:.1f}% |",
        f"| Distance-miss % (right wrap, >= half-gap) | {p['distance_miss_pct']:.1f}% |",
        f"| Included / excluded | {c['included']} / {c['excluded']} "
        f"(no-coverage {c['excluded_no_coverage']}, edge {c['excluded_edge']}) |",
    ]
    d = card["distances"]
    if d:
        lines.append(
            f"| Median dist to target | {d['median_dist_to_target_native']:.2f} native"
            + (f" = {d['median_dist_to_target_um']:.0f} um"
               if "median_dist_to_target_um" in d else "")
            + " |"
        )
        lines.append(
            f"| Median local gap | {d['median_local_gap_native']:.2f} native"
            + (f" = {d['median_local_gap_um']:.0f} um"
               if "median_local_gap_um" in d else "")
            + " |"
        )
    lines += [
        "",
        f"Threshold: {card['gap_fraction_threshold']} x local gap "
        "(half-gap headline rule, ported from the source audit).",
        "Percentages are of INCLUDED predictions; excluded points had no "
        "reliable reference coverage (radius or coverage-boundary rule) "
        "and were not guessed.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Mesh mode
# ----------------------------------------------------------------------

def run_mesh(strip, args, qualified: bool) -> dict:
    vertices, faces = load_pred_mesh(args.pred)
    trees = build_trees(strip)
    topo = topology_counts(faces)
    comps = connected_components(vertices.shape[0], faces)
    fusion = fusion_analysis(strip, trees, vertices, faces)
    return {
        "mode": "mesh",
        "strip": strip_summary(strip, args.strip, qualified),
        "pred_path": str(args.pred),
        "n_vertices": int(vertices.shape[0]),
        "topology": topo,
        "components": comps,
        "fusion": fusion,
    }


def mesh_markdown(card: dict) -> str:
    t, comps, fu = card["topology"], card["components"], card["fusion"]
    lines = [
        "# Strip scorecard (mesh mode)",
        "",
        f"Strip: `{card['strip']['strip_path']}` "
        f"({card['strip']['scroll']} / {card['strip']['segment_id']}, "
        f"tier {card['strip']['tier']})",
        f"Qualification: "
        f"{'QUALIFIED' if card['strip']['qualified'] else '**UNQUALIFIED -- numbers below are against an unverified reference**'}",
        f"Mesh: `{card['pred_path']}` "
        f"({card['n_vertices']} vertices, {t['n_faces']} triangles)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Non-manifold edges (>2 faces) | {t['non_manifold_edges']} |",
        f"| Open boundary edges (1 face) | {t['boundary_edges']} |",
        f"| Connected components | {comps['connected_components']} |",
        f"| **Cross-wrap fusion triangles** | **{fu['cross_wrap_fusion_triangles']}** |",
        f"| Scored / excluded triangles | {fu['scored_triangles']} / "
        f"{fu['excluded_triangles']} |",
        f"| Fused wrap pairs | {json.dumps(fu['fused_pair_histogram'])} |",
        "",
        f"Vertices per wrap (within coverage): "
        f"{json.dumps(fu['vertices_per_wrap_within_coverage'])}",
        "",
        "These are geometric checks against THIS strip's reference wraps "
        "(coverage radius "
        f"{fu['coverage_radius']:.2f} native units), NOT a global topology "
        "certification. Excluded triangles had a vertex outside reference "
        "coverage or at a coverage boundary.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strip", required=True, help="strip-v0 .npz file")
    parser.add_argument("--mode", required=True, choices=["points", "mesh"])
    parser.add_argument("--pred", required=True,
                        help="points: .npz/.npy of (M,3) xyz; "
                             "mesh: .obj or tifxyz directory")
    parser.add_argument("--from-wrap", type=int, default=None,
                        help="(points mode) wrap the tracer started from")
    parser.add_argument("--direction", choices=["front", "back"], default=None,
                        help="(points mode) front = from_wrap+1, back = from_wrap-1")
    parser.add_argument("--gap-fraction", type=float, default=0.5,
                        help="distance threshold as a fraction of the local "
                             "gap (default 0.5 = the half-gap headline rule)")
    parser.add_argument("--out-json", default=None,
                        help="default: <pred>.scorecard.json")
    parser.add_argument("--out-md", default=None,
                        help="default: <pred>.scorecard.md")
    return parser.parse_args()


def main():
    args = parse_args()
    strip_path = Path(args.strip)
    strip = load_strip(strip_path)
    qualified = is_qualified(strip_path)
    if not qualified:
        print(
            "score_strip: WARNING: UNQUALIFIED strip -- no passing "
            "qualification report found next to it (run qualify_strip.py "
            "first). Scores below are against an unverified reference.",
            file=sys.stderr,
        )

    if args.mode == "points":
        if args.from_wrap is None or args.direction is None:
            raise SystemExit("points mode requires --from-wrap and --direction")
        card = run_points(strip, args, qualified)
        md = points_markdown(card)
    else:
        card = run_mesh(strip, args, qualified)
        md = mesh_markdown(card)

    pred_path = Path(args.pred)
    stem = pred_path.name if pred_path.is_dir() else pred_path.stem
    out_json = Path(args.out_json) if args.out_json else pred_path.parent / f"{stem}.scorecard.json"
    out_md = Path(args.out_md) if args.out_md else pred_path.parent / f"{stem}.scorecard.md"
    out_json.write_text(json.dumps(card, indent=2) + "\n")
    out_md.write_text(md)

    print(md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
