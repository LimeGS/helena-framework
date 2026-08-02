#!/usr/bin/env python3
"""vet_pipeline.py -- audit output maps submitted for a claimant pipeline.

    python vet_pipeline.py --manifest fixtures/manifest.json \\
        --outputs dir_of_claimant_maps/ [--px-um 8.0] \\
        --out verdict.json

Workflow this tool assumes: the claimant runs their own pipeline on each
fixture window's source image and drops one output map per fixture id into
--outputs, named `<id>.npy` or `<id>.png`. This tool never looks at the
claimant's pipeline itself -- it only audits the maps it was handed. It
records SHA-256 digests for every submitted output, but those digests are
replay evidence rather than proof that the outputs came from a stated model.

For each fixture, this applies the complete calibrated rule from ``vet_map``
(letter_energy AND structure AND contrast_bimodality) over a simple grid of
candidate bboxes covering the submitted output map: any passing window fires.

Metrics (see card_config.py for the pass-rule rationale):
  pareidolia_rate = fraction of blank fixtures where the grid scan fired
  sensitivity     = fraction of positive fixtures where the grid scan fired
  raw result      = pareidolia_rate <= PIPELINE_MAX_PAREIDOLIA_RATE (0.0)
                    AND sensitivity >= PIPELINE_MIN_SENSITIVITY (0.5)

A formal pipeline PASS is emitted only when every fixture is independently
public and byte-pinned. The current public manifest deliberately does not
meet that bar because its blank fixtures are not publicly fetchable, so the
CLI fails closed rather than publishing a misleading pipeline qualification.

A fixture whose output file is missing from --outputs is reported as an
error entry and excluded from both rate denominators; ANY missing output
forces the overall verdict to fail, since a rate computed over incomplete
evidence should never be reported as a clean pass.

Known v0 limitation (documented in README, not silently papered over): if
a claimant's entire output for one fixture is itself degenerate (constant/
blank), the grid scan will simply report "did not fire" -- true, but this
cannot distinguish an honest pipeline from a broken one that outputs
nothing at all. This tool does not attempt that distinction; a reviewer
should sanity-check that outputs are non-degenerate before trusting a
green pipeline verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import card_config
import vet_map
from vet_map import Bbox, VetMapError


class VetPipelineError(Exception):
    """Raised for manifest/outputs problems that stop the whole run."""


# ---------------------------------------------------------------------------
# Manifest / outputs loading
# ---------------------------------------------------------------------------

def load_manifest(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise VetPipelineError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise VetPipelineError(f"manifest is not valid JSON: {path}") from exc
    fixtures = data.get("fixtures") if isinstance(data, dict) else data
    if not isinstance(fixtures, list) or not fixtures:
        raise VetPipelineError(f"manifest has no usable 'fixtures' list: {path}")
    for entry in fixtures:
        if "id" not in entry or "kind" not in entry:
            raise VetPipelineError(f"fixture entry missing required 'id'/'kind': {entry}")
        if entry["kind"] not in ("blank", "positive"):
            raise VetPipelineError(f"fixture {entry['id']!r} has unknown kind {entry['kind']!r} (expected 'blank' or 'positive')")
    return fixtures


def find_output_file(outputs_dir: Path, fixture_id: str) -> Optional[Path]:
    for suffix in (".npy", ".png"):
        candidate = outputs_dir / f"{fixture_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def unavailable_fixture_ids(fixtures: list[dict]) -> list[str]:
    """Return fixtures that prevent an independently reproducible PASS.

    Test manifests may omit release status entirely. Only explicitly
    non-public production fixtures block a formal pass; this preserves the
    small synthetic unit-test harness while making the shipped manifest fail
    closed until the blank rasters are released.
    """
    return [
        str(entry["id"])
        for entry in fixtures
        if entry.get("status") == "NOT_PUBLICLY_FETCHABLE"
    ]


# ---------------------------------------------------------------------------
# Grid scan (reuses vet_map's check machinery)
# ---------------------------------------------------------------------------

def make_grid_bboxes(shape: tuple[int, int], window_px: int, stride_px: int) -> list[Bbox]:
    h, w = shape
    window_px = max(1, min(window_px, h, w))
    stride_px = max(1, stride_px)
    max_x0, max_y0 = w - window_px, h - window_px

    xs = list(range(0, max_x0 + 1, stride_px))
    if xs[-1] != max_x0:
        xs.append(max_x0)
    ys = list(range(0, max_y0 + 1, stride_px))
    if ys[-1] != max_y0:
        ys.append(max_y0)

    return [(x, y, x + window_px, y + window_px) for y in ys for x in xs]


def scan_for_fire(map01: np.ndarray, px_um: float, cfg=card_config) -> dict:
    """Grid-scan map01 for any bbox that passes v0.2's calibrated absolute
    rule (letter_energy AND structure AND contrast_bimodality). Stops at the
    first pass ("any pass = fired") for speed. No null sampling: the v0
    percentile machinery no longer gates (see CALIBRATION.md), which also
    makes this scan orders of magnitude faster.
    """
    _, hi_scale_px = cfg.letter_scale_px_range(px_um)  # letter-scale upper bound in px, NOT a sigma
    window_px = max(cfg.MIN_BBOX_DIM_PX, round(cfg.PIPELINE_SCAN_WINDOW_LETTERSCALES * hi_scale_px))
    stride_px = max(1, round(window_px * cfg.PIPELINE_SCAN_STRIDE_FRACTION))
    grid = make_grid_bboxes(map01.shape, window_px, stride_px)

    fired = False
    best_e_frac = None
    cells_evaluated = 0
    for bbox in grid:
        energy = vet_map.check_letter_energy(map01, bbox, px_um, None, cfg)
        cells_evaluated += 1
        if best_e_frac is None or energy["value"] > best_e_frac:
            best_e_frac = energy["value"]
        if not energy["pass"]:
            continue
        structure = vet_map.check_structure(map01, bbox, px_um, None, cfg)
        contrast = vet_map.check_contrast_bimodality(map01, bbox, cfg)
        if structure["pass"] and contrast["pass"]:
            fired = True
            break

    return {
        "fired": fired,
        "best_e_frac": best_e_frac,
        "cells_evaluated": cells_evaluated,
        "cells_total": len(grid),
        "window_px": window_px,
        "stride_px": stride_px,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def grade_pipeline(
    fixtures: list[dict], outputs_dir: Path, default_px_um: float, cfg=card_config,
) -> dict:
    per_fixture = []
    n_blank_eval = n_blank_fired = 0
    n_pos_eval = n_pos_fired = 0
    missing_ids = []

    for entry in fixtures:
        fid, kind = entry["id"], entry["kind"]
        # entry.get(..., default) only falls back when the key is ABSENT; our
        # own template manifest sets "px_um": null for entries whose scale
        # isn't documented, and JSON null decodes to Python None (present,
        # falsy) -- so an explicit falsy check is required to actually fall
        # back to the CLI default instead of crashing on float(None).
        px_um_raw = entry.get("px_um")
        px_um = float(px_um_raw) if px_um_raw else default_px_um
        out_path = find_output_file(outputs_dir, fid)
        if out_path is None:
            missing_ids.append(fid)
            per_fixture.append({"id": fid, "kind": kind, "status": "missing_output", "fired": None})
            continue
        try:
            raw = vet_map.load_map(out_path)
            vet_map.validate_map_size(raw, cfg)
            map01 = vet_map.normalize01(raw) if np.max(raw) != np.min(raw) else np.zeros_like(raw, dtype=np.float64)
            scan = scan_for_fire(map01, px_um, cfg)
        except VetMapError as exc:
            per_fixture.append({"id": fid, "kind": kind, "status": "error", "error": str(exc), "fired": None})
            missing_ids.append(fid)
            continue

        fired = scan["fired"]
        per_fixture.append({
            "id": fid, "kind": kind, "status": "ok", "fired": fired,
            "output_sha256": vet_map.sha256_file(out_path),
            "output_size_bytes": out_path.stat().st_size,
            "best_e_frac": scan["best_e_frac"],
            "cells_evaluated": scan["cells_evaluated"], "cells_total": scan["cells_total"],
        })
        if kind == "blank":
            n_blank_eval += 1
            n_blank_fired += int(fired)
        else:
            n_pos_eval += 1
            n_pos_fired += int(fired)

    pareidolia_rate = (n_blank_fired / n_blank_eval) if n_blank_eval else None
    sensitivity = (n_pos_fired / n_pos_eval) if n_pos_eval else None

    complete = not missing_ids
    rates_ok = (
        pareidolia_rate is not None and sensitivity is not None
        and pareidolia_rate <= cfg.PIPELINE_MAX_PAREIDOLIA_RATE
        and sensitivity >= cfg.PIPELINE_MIN_SENSITIVITY
    )
    overall_pass = bool(complete and rates_ok)

    return {
        "pareidolia_rate": pareidolia_rate,
        "sensitivity": sensitivity,
        "thresholds": {
            "max_pareidolia_rate": cfg.PIPELINE_MAX_PAREIDOLIA_RATE,
            "min_sensitivity": cfg.PIPELINE_MIN_SENSITIVITY,
        },
        "n_blank_evaluated": n_blank_eval,
        "n_positive_evaluated": n_pos_eval,
        "missing_ids": missing_ids,
        "fixtures": per_fixture,
        "overall": {"pass": overall_pass},
    }


def build_result(manifest_path, outputs_dir, px_um: float, cfg=card_config) -> dict:
    result = {
        "schema_version": cfg.CARD_SCHEMA_VERSION,
        "tool": "vet_pipeline.py",
        "tool_version": cfg.TOOL_VERSION,
        "config_hash": cfg.config_hash(),
        "evaluator": {
            "source": Path(__file__).name,
            "source_sha256": vet_map.sha256_file(Path(__file__)),
            "vet_map_source_sha256": vet_map.evaluator_provenance()["source_sha256"],
        },
        "input": {
            "manifest": str(manifest_path), "outputs": str(outputs_dir),
            "px_um": px_um,
        },
    }
    try:
        fixtures = load_manifest(manifest_path)
        manifest = Path(manifest_path)
        result["input"]["manifest_sha256"] = vet_map.sha256_file(manifest)
        outputs_path = Path(outputs_dir)
        if not outputs_path.is_dir():
            raise VetPipelineError(f"--outputs is not a directory: {outputs_dir}")
        unavailable = unavailable_fixture_ids(fixtures)
        if unavailable:
            raise VetPipelineError(
                "formal pipeline qualification is unavailable: the manifest contains "
                f"non-public fixtures {unavailable}. Release byte-pinned public blank "
                "fixtures before using this command as a community gate."
            )
        outcome = grade_pipeline(fixtures, outputs_path, px_um, cfg=cfg)
        result["status"] = "ok"
        result["evidence"] = {"level": "self_attested_output_maps", "unavailable_fixture_ids": []}
        result.update(outcome)
    except VetPipelineError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        try:
            fixtures = load_manifest(manifest_path)
            result["evidence"] = {
                "level": "formal_pipeline_pass_unavailable",
                "unavailable_fixture_ids": unavailable_fixture_ids(fixtures),
            }
        except VetPipelineError:
            pass
        result["overall"] = {"pass": False}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vet_pipeline.py",
        description="Audit submitted output maps against a public blank/positive fixture set.",
    )
    parser.add_argument("--manifest", required=True, help="path to fixtures manifest.json")
    parser.add_argument("--outputs", required=True, help="directory of claimant output maps, named <fixture_id>.npy or .png")
    parser.add_argument(
        "--px-um", type=float, default=card_config.DEFAULT_PX_UM,
        help=f"default microns per pixel (default {card_config.DEFAULT_PX_UM}); a fixture's own 'px_um' field, if present, overrides this",
    )
    parser.add_argument("--out", required=True, help="path to write the verdict JSON")
    parser.add_argument("--card", default=None, metavar="PATH",
                        help="also render the Markdown vetting card here")
    return parser


def main(argv=None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    result = build_result(args.manifest, args.outputs, args.px_um)
    Path(args.out).write_text(json.dumps(result, indent=2))

    if getattr(args, "card", None):
        from make_card import render_card  # lazy: only when asked
        Path(args.card).write_text(render_card(None, result))
        print(f"card: wrote {args.card}")

    if result["status"] == "error":
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 2

    if result["missing_ids"]:
        print(f"WARNING: missing claimant output for fixture ids: {result['missing_ids']}", file=sys.stderr)

    verdict = "PASS" if result["overall"]["pass"] else "FAIL"
    print(
        f"{verdict}: pareidolia_rate={result['pareidolia_rate']} "
        f"sensitivity={result['sensitivity']} -- wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
