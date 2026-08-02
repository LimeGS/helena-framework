#!/usr/bin/env python3
"""make_card.py -- combine vet_map.py / vet_pipeline.py verdict JSONs into a
single human-readable VETTING_CARD.md.

    python make_card.py [--map-verdict verdict_map.json] \\
        [--pipeline-verdict verdict_pipeline.json] --out VETTING_CARD.md

At least one of --map-verdict / --pipeline-verdict must be given. The card
is only as good as the verdicts fed into it -- this tool does not re-run
any check, it only renders what already ran.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

import card_config

FOOTER = (
    "A PASS means the supplied bytes survived calibrated automated refutation "
    "checks. It is neither confirmation of ancient text nor independent proof "
    "that a claimant ran a particular pipeline. Re-run published code against "
    "inputs whose SHA-256 matches this card before relying on it."
)


def _fmt(value, nd: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{nd}g}"
    return str(value)


def _pass_word(passed) -> str:
    if passed is None:
        return "N/A"
    return "PASS" if passed else "FAIL"


def render_map_section(verdict: dict) -> str:
    lines = ["## Claim check (vet_map.py)", ""]
    inp = verdict.get("input", {})
    lines.append(f"- Map: `{inp.get('map')}`")
    scope = inp.get("map_hash_scope")
    label = f"Map SHA-256 ({scope})" if scope else "Map SHA-256"
    lines.append(f"- {label}: `{inp.get('map_sha256', 'not recorded')}`")
    if inp.get("map_size_bytes") is not None:
        lines.append(f"- Map bytes: `{inp.get('map_size_bytes')}`")
    if inp.get("map_shape") is not None:
        lines.append(f"- Evaluated array: shape `{inp.get('map_shape')}`, dtype `{inp.get('map_dtype')}`")
    if inp.get("evaluated_crop_bbox_in_source") is not None:
        lines.append(f"- Evaluated crop in source coordinates: `{inp.get('evaluated_crop_bbox_in_source')}`")
    lines.append(f"- Bbox (x0,y0,x1,y1): `{inp.get('bbox')}`")
    lines.append(f"- px_um: `{inp.get('px_um')}`  seed: `{inp.get('seed')}`")
    evaluator = verdict.get("evaluator", {})
    if evaluator:
        lines.append(f"- Evaluator source SHA-256: `{evaluator.get('source_sha256', 'not recorded')}`")
    lines.append("")

    if verdict.get("status") == "error":
        lines.append(f"**ERROR: could not evaluate -- {verdict.get('error')}**")
        lines.append("")
        lines.append("**Overall (claim check): FAIL**")
        return "\n".join(lines)

    checks = verdict.get("checks", {})
    lines.append("| Check | Value | Threshold | Result | Notes |")
    lines.append("|---|---|---|---|---|")
    for name in ("degenerate", "letter_energy", "structure", "contrast_bimodality"):
        check = checks.get(name)
        if check is None:
            continue
        value = _fmt(check.get("value"))
        threshold = check.get("threshold")
        threshold_str = f">= {_fmt(threshold)}" if threshold is not None else "-"
        note = check.get("message") or check.get("reason") or ""
        rarity = check.get("local_rarity")
        if rarity is not None:
            note = (note + " " if note else "") + f"local rarity pct {_fmt(rarity.get('percentile'))} (context only)"
        lines.append(f"| {name} | {value} | {threshold_str} | {_pass_word(check.get('pass'))} | {note} |")

    lines.append("")
    lines.append(f"**Overall (claim check): {_pass_word(verdict.get('overall', {}).get('pass'))}**")
    return "\n".join(lines)


def render_pipeline_section(verdict: dict) -> str:
    lines = ["## Pipeline check (vet_pipeline.py)", ""]
    inp = verdict.get("input", {})
    lines.append(f"- Manifest: `{inp.get('manifest')}`")
    lines.append(f"- Manifest SHA-256: `{inp.get('manifest_sha256', 'not recorded')}`")
    lines.append(f"- Outputs: `{inp.get('outputs')}`")
    evidence = verdict.get("evidence", {})
    if evidence:
        lines.append(f"- Evidence level: **{evidence.get('level', 'not recorded')}**")
        if evidence.get("unavailable_fixture_ids"):
            lines.append(f"- Non-public fixtures (formal PASS unavailable): `{evidence['unavailable_fixture_ids']}`")
    evaluator = verdict.get("evaluator", {})
    if evaluator:
        lines.append(f"- Evaluator source SHA-256: `{evaluator.get('source_sha256', 'not recorded')}`")
    lines.append("")

    if verdict.get("status") == "error":
        lines.append(f"**ERROR: could not evaluate -- {verdict.get('error')}**")
        lines.append("")
        lines.append("**Overall (pipeline check): FAIL**")
        return "\n".join(lines)

    thresholds = verdict.get("thresholds", {})
    lines.append(
        f"- pareidolia_rate: **{_fmt(verdict.get('pareidolia_rate'))}** "
        f"(must be <= {_fmt(thresholds.get('max_pareidolia_rate'))} to pass)"
    )
    lines.append(
        f"- sensitivity: **{_fmt(verdict.get('sensitivity'))}** "
        f"(must be >= {_fmt(thresholds.get('min_sensitivity'))} to pass)"
    )
    if verdict.get("missing_ids"):
        lines.append(f"- missing claimant outputs (forces FAIL): `{verdict.get('missing_ids')}`")
    lines.append("")

    lines.append("| Fixture | Kind | Fired | Best e_frac | Output SHA-256 | Status |")
    lines.append("|---|---|---|---|---|---|")
    for fx in verdict.get("fixtures", []):
        fired = fx.get("fired")
        fired_str = "-" if fired is None else ("yes" if fired else "no")
        lines.append(
            f"| {fx.get('id')} | {fx.get('kind')} | {fired_str} | "
            f"{_fmt(fx.get('best_e_frac'))} | `{fx.get('output_sha256', '-')}` | {fx.get('status')} |"
        )

    lines.append("")
    lines.append(f"**Overall (pipeline check): {_pass_word(verdict.get('overall', {}).get('pass'))}**")
    return "\n".join(lines)


def render_card(
    map_verdict: Optional[dict], pipeline_verdict: Optional[dict], cfg=card_config
) -> str:
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [f"# Vetting Card (schema {cfg.CARD_SCHEMA_VERSION})", ""]
    lines.append(f"- Tool: vetting-card v{cfg.TOOL_VERSION}")
    lines.append(f"- Config hash: `{cfg.config_hash()}`")
    lines.append(f"- Generated: {generated}")

    source_hashes = {v.get("config_hash") for v in (map_verdict, pipeline_verdict) if v}
    if len(source_hashes) > 1:
        lines.append("")
        lines.append(
            "**Warning: the supplied verdict files were produced by different "
            "config versions (config_hash mismatch) -- re-run both checks "
            "with the same tool version before trusting this card.**"
        )
    # Both verdict paths ultimately use vet_map.py for the calibrated checks.
    # vet_pipeline.py records that dependency separately because its own file
    # hash is necessarily different from a direct vet_map.py invocation.
    map_engine_hash = (map_verdict or {}).get("evaluator", {}).get("source_sha256")
    pipeline_engine_hash = (pipeline_verdict or {}).get("evaluator", {}).get("vet_map_source_sha256")
    if map_engine_hash and pipeline_engine_hash and map_engine_hash != pipeline_engine_hash:
        lines.append("")
        lines.append(
            "**Warning: the supplied verdict files used different calibrated "
            "check-engine source files (vet_map.py SHA-256 mismatch) -- re-run "
            "both checks from one reviewed release before trusting this card.**"
        )
    lines.append("")

    overall_parts = []
    if map_verdict is not None:
        lines.append(render_map_section(map_verdict))
        lines.append("")
        overall_parts.append(bool(map_verdict.get("overall", {}).get("pass")))
    if pipeline_verdict is not None:
        lines.append(render_pipeline_section(pipeline_verdict))
        lines.append("")
        overall_parts.append(bool(pipeline_verdict.get("overall", {}).get("pass")))

    overall_pass = bool(overall_parts) and all(overall_parts)
    lines.append(f"## Overall verdict: {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_card.py",
        description="Render vet_map.py / vet_pipeline.py verdict JSONs into VETTING_CARD.md.",
    )
    parser.add_argument("--map-verdict", default=None, help="verdict JSON from vet_map.py")
    parser.add_argument("--pipeline-verdict", default=None, help="verdict JSON from vet_pipeline.py")
    parser.add_argument("--out", required=True, help="path to write the Markdown card")
    return parser


def main(argv=None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if not args.map_verdict and not args.pipeline_verdict:
        print("ERROR: at least one of --map-verdict / --pipeline-verdict is required", file=sys.stderr)
        return 2

    map_verdict = json.loads(Path(args.map_verdict).read_text()) if args.map_verdict else None
    pipeline_verdict = json.loads(Path(args.pipeline_verdict).read_text()) if args.pipeline_verdict else None

    card_text = render_card(map_verdict, pipeline_verdict)
    Path(args.out).write_text(card_text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
