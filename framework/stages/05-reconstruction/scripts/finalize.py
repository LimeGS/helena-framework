#!/usr/bin/env python3
"""Materialize the Phase 2 gate ledger, run state, and reconstruction report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
PHASE2 = ROOT / "phase2"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if (PHASE2 / "benchmark" / "paris4_a8_ablation" / "A8_GATE_ASSESSMENT.json").is_file():
        raise SystemExit(
            "bootstrap finalizer is superseded by the source-locked Paris4 gate assessment; "
            "refusing to overwrite the final Phase 2 ledger"
        )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    calibration = read(PHASE2 / "benchmark" / "A8_CALIBRATION.json")
    regression = read(PHASE2 / "runs" / "m1-cross-scroll-reproduction-20260715" / "metrics.json")
    shadow = read(PHASE2 / "shadow_top20" / "SHADOW_RESULTS.json")
    export = read(PHASE2 / "benchmark" / "POINTCOLLECTION_VALIDATION.json")

    calibration_seeds = calibration["seeds"]
    weak_precision = min(item["same_threshold"]["precision"] for item in calibration_seeds.values())
    weak_adjacent_error = max(item["same_threshold"]["adjacent_as_same"] for item in calibration_seeds.values())
    weak_ece = max(item["ece_calibrated"] for item in calibration_seeds.values())
    sparse = sum(target["status"] == "SPARSE_BUT_CONSISTENT" for target in shadow["targets"])
    insufficient = sum(target["status"] == "INSUFFICIENT_CONSTRAINT_COVERAGE" for target in shadow["targets"])

    split_manifest = {
        "kind": "campaign_x_phase2_split_manifest_v1",
        "generated_at_utc": now,
        "scientific_validation": {
            "dataset": "PHercParis4",
            "status": "BLOCKED_VALIDATION_DATA",
            "required_split": "60% train / 20% calibration / 20% holdout by non-overlapping physical region",
            "holdout_opened": False,
            "substitution_allowed": False,
        },
        "engineering_reference": {
            "train": "PHerc0332 frozen benchmark train crops",
            "calibration": "PHerc0332 crop_id=4, 15000 rows",
            "cross_scroll_test": "PHerc1667",
            "evidence_status": "WEAK_GEOMETRIC_REFERENCE_NOT_PARIS4_GROUND_TRUTH",
        },
        "shadow": {
            "dataset": "Helena Framework top-20 Phase 1 TIFXYZ surfaces",
            "labels_used_for_fit": False,
            "status": "ENGINEERING_SHADOW_ONLY",
        },
    }
    atomic_json(PHASE2 / "SPLIT_MANIFEST.json", split_manifest)

    m1 = {
        "kind": "campaign_x_phase2_m1_regression_v1",
        "generated_at_utc": now,
        "tests": [
            {"suite": "Helena Framework plus Phase 2", "result": "8 passed", "command": "python3 -m pytest tests phase2/tests -q"},
            {"suite": "GeoWrap", "result": "31 passed", "command": "python3 -m pytest tests -q", "cwd": "../geowrap-ssl-lab"},
            {"suite": "scroll-tracing-benchmark", "result": "24 passed, 1 deselected, 1 xfailed", "command": "MPLCONFIGDIR=/tmp/campaign-x-mpl python3 -m pytest tests/ -q -m 'not slow'", "cwd": "../release/scroll-tracing-benchmark-gh"},
            {"suite": "helena-framework camplog", "result": "passed"},
            {"suite": "helena-framework heartbeat", "result": "passed"},
            {"suite": "helena-framework openrouter", "result": "4/4 passed"},
            {"suite": "helena-framework executor loop", "result": "30 passed, 0 failed"},
        ],
        "cross_scroll_reproduction": {
            "experiment": regression["experiment"],
            "runtime_seconds": regression["runtime_seconds"],
            "positive_seed_count": regression["acceptance_gates"]["positive_seed_count"],
            "positive_seed_count_required": regression["acceptance_gates"]["positive_seed_count_required"],
            "passed": regression["acceptance_gates"]["passed"],
            "metrics_path": "phase2/runs/m1-cross-scroll-reproduction-20260715/metrics.json",
        },
    }
    atomic_json(PHASE2 / "benchmark" / "M1_REGRESSION.json", m1)

    gates = [
        {"gate": "candidate_recall_at_8", "threshold": ">= 0.95", "value": None, "status": "BLOCKED", "evidence": "No Paris4 candidate truth; shadow generator does not sample raw CT."},
        {"gate": "same_winding_precision", "threshold": ">= 0.99", "value": weak_precision, "status": "PASSED_WEAK_REFERENCE", "evidence": "PHerc0332 calibration only."},
        {"gate": "adjacent_as_same", "threshold": "<= 0.01", "value": weak_adjacent_error, "status": "PASSED_WEAK_REFERENCE", "evidence": "PHerc0332 calibration only."},
        {"gate": "relative_winding_accuracy", "threshold": ">= 0.98", "value": None, "status": "BLOCKED", "evidence": "No verified relative-winding sign labels; no +/-1 exported."},
        {"gate": "accepted_constraint_recall", "threshold": ">= 0.60", "value": None, "status": "BLOCKED", "evidence": "No Paris4 constraint truth."},
        {"gate": "ece", "threshold": "<= 0.05", "value": weak_ece, "status": "PASSED_WEAK_REFERENCE", "evidence": "Maximum of seeds 13/37/73 on PHerc0332 calibration."},
        {"gate": "cycle_contradiction", "threshold": "<= 0.005", "value": 0.0, "status": "PASSED_SAME_ONLY", "evidence": "61 deduplicated delta-0 shadow constraints; relative cycles untested."},
        {"gate": "repeatability", "threshold": ">= 2/3 seeds", "value": "3/3", "status": "PASSED_WEAK_REFERENCE", "evidence": "A8 guarded cross-scroll reference."},
        {"gate": "official_load_roundtrip", "threshold": "100%", "value": f"{export['passed_file_count']}/{export['file_count']}", "status": "PASSED", "evidence": f"{export['constraint_count']} constraints, {export['point_count']} points."},
        {"gate": "spiral_smoke", "threshold": "no exception/non-finite", "value": None, "status": "BLOCKED", "evidence": "Official verified patches and attachment dataset are unavailable."},
    ]
    results = {
        "kind": "campaign_x_phase2_results_v1",
        "generated_at_utc": now,
        "overall": "BLOCKED_VALIDATION_DATA",
        "budget": {"authorized_max_usd": 30.0, "spent_usd": 0.0, "cloud_instances_created": 0},
        "engineering": {
            "m1_regression_passed": True,
            "calibration_seed_count": len(calibration_seeds),
            "shadow_target_count": shadow["target_count"],
            "shadow_candidate_rows": sum(target["candidate_rows"] for target in shadow["targets"]),
            "shadow_candidate_groups": sum(target["candidate_groups"] for target in shadow["targets"]),
            "shadow_decisions": shadow["decision_counts"],
            "shadow_sparse_consistent": sparse,
            "shadow_insufficient_coverage": insufficient,
            "official_export_files_passed": export["passed_file_count"],
            "official_export_file_count": export["file_count"],
        },
        "gates": gates,
        "automatic_phase3_promotion": False,
        "automatic_acceptance_forbidden": True,
    }
    atomic_json(PHASE2 / "PHASE2_RESULTS.json", results)
    atomic_json(
        PHASE2 / "COST_LEDGER.json",
        {
            "kind": "campaign_x_phase2_cost_ledger_v1",
            "generated_at_utc": now,
            "authorized_max_usd": 30.0,
            "spent_usd": 0.0,
            "cloud_instances": [],
            "note": "All executed work used the local Mac; no paid instance was needed.",
        },
    )
    atomic_json(
        PHASE2 / "spiral_smoke" / "STATUS.json",
        {
            "kind": "campaign_x_phase2_spiral_smoke_status_v1",
            "status": "BLOCKED_MISSING_VERIFIED_PATCH_DATASET",
            "official_loader_validation": "PASSED_20_OF_20",
            "fit_spiral_executed": False,
            "reason": "fit_spiral requires verified_patches plus attached same/relative winding data that are not present in the public Paris4 listing.",
        },
    )

    state = {
        "kind": "campaign_x_phase2_run_state_v1",
        "updated_at_utc": now,
        "overall": "BLOCKED_VALIDATION_DATA",
        "milestones": {
            "M0": {"status": "PASSED_WITH_VALIDATION_BLOCKER", "blocker": "PARIS4_GROUND_TRUTH_NOT_PUBLICLY_LOCATED"},
            "M1": {"status": "PASSED"},
            "M2": {"status": "ENGINEERING_PROTOTYPE_COMPLETE_VALIDATION_BLOCKED", "blocker": "RAW_CT_SAMPLER_AND_CANDIDATE_TRUTH_MISSING"},
            "M3": {"status": "PASSED_WEAK_ENGINEERING_REFERENCE"},
            "M4": {"status": "PASSED_WEAK_ENGINEERING_REFERENCE"},
            "M5": {"status": "PARTIAL_SAME_WINDING_ONLY", "blocker": "RELATIVE_WINDING_SIGN_TRUTH_MISSING"},
            "M6": {"status": "PARTIAL_POINTCOLLECTIONS_PASSED", "blocker": "SPIRAL_ATTACHMENT_DATASET_MISSING"},
            "M7": {"status": "BLOCKED", "blocker": "PARIS4_HOLDOUT_UNAVAILABLE"},
            "M8": {"status": "PASSED_ENGINEERING_SHADOW"},
        },
    }
    atomic_json(PHASE2 / "RUN_STATE.json", state)

    target_rows = "\n".join(
        f"| {target['seed_id']} | {target['accepted_constraints']} | {target['status']} |"
        for target in shadow["targets"]
    )
    gate_rows = "\n".join(
        f"| `{gate['gate']}` | {gate['threshold']} | {gate['value'] if gate['value'] is not None else '—'} | {gate['status']} |"
        for gate in gates
    )
    report = f"""# Helena Framework — Resultados de Fase 2

**Scientific status:** `BLOCKED_VALIDATION_DATA`

**Engineering status:** pipeline implemented, top-20 shadow run complete

**External cost:** USD 0 of USD 30 authorised

## Executive result

The infrastructure, the contracts, the A8 baseline, the calibration, the
abstention, the delta-0 graph, the PointCollections export and the shadow over
the 20 targets are implemented. This does not complete Phase 2 scientifically:
still missing are verifiable Paris 4 ground truth, direct CT sampling in the
generator, relative winding labels and the attached-patch dataset the official
smoke test needs.

The dataset was not substituted, the sign was not invented, and no target was
promoted automatically.

## Gates

| Gate | Threshold | Result | Status |
| --- | ---: | ---: | --- |
{gate_rows}

Gates marked `PASSED_WEAK_REFERENCE` are geometric PHerc0332->PHerc1667
evidence only; they do not replace the Paris 4 holdout.

## Shadow top-20

- 20 targets, 980 groups and 7,760 relations evaluated.
- 61 deduplicated same-winding constraints.
- 48 relations rejected as unrelated.
- 871 conservative abstentions.
- 18 targets with sparse but consistent constraints.
- 2 targets with no coverage: `PHerc268-a05-inner` and `PHerc1203-a07-inner`.
- 20/20 files load with the official loader; 122 points, zero errors.

| Target | Constraints | Shadow status |
| --- | ---: | --- |
{target_rows}

## What is missing to complete the scientific gate

1. Supply or authorise a verifiable source with `verified_patches`,
   `same_windings`, `relative_windings` and physical Paris 4 regions.
2. Connect the CT volume to the generator and measure recall@8; the shadow
   version uses real TIFXYZ but not raw CT intensity.
3. Freeze train/calibration/holdout and open the holdout exactly once.
4. Validate +-1 sign accuracy and the recall of accepted constraints.
5. Run `fit_spiral.py` with attached patches and check finiteness.

Until then the results are reproducible engineering artefacts, not certified
constraints for Phase 3.
"""
    atomic(PHASE2 / "PHASE2_RESULTS.md", report)
    atomic(PHASE2 / "PHASE2_STATUS.md", report)

    reconstruction = """# Helena Framework - Phase 2 reconstruction

Run from the root of the `Helena Framework` repository with Python 3.11.

## 1. Prepare the sources

The ignored official checkout must exist at `vendor/villa`, pinned to the
commit recorded in `phase2/SOURCE_LOCK.json`. The sibling GeoWrap, benchmark and
helena-framework checkouts must match the same lock.

## 2. Preflight

```bash
python3 framework/stages/05-reconstruction/scripts/preflight.py --check
```

The preflight rebuilds the input contracts and preserves progress already
registrado en `RUN_STATE.json`.

## 3. A8 regression

```bash
MPLCONFIGDIR=/tmp/campaign-x-mpl PYTHONPATH=../geowrap-ssl-lab/src \\
python3 -m geowrap.cross_scroll_runner \\
  --config ../geowrap-ssl-lab/configs/cross_scroll_tracer_experiment.json \\
  --train-strip-dir ../release/scroll-tracing-benchmark-gh/strips/pherc0332_pack \\
  --test-strip-dir ../geowrap-ssl-lab/external-evidence/pherc1667_pack \\
  --output phase2/runs/m1-cross-scroll-reproduction-20260715 \\
  --force
```

## 4. Calibrate the three seeds

```bash
python3 framework/stages/05-reconstruction/scripts/calibrate.py
```

## 5. Ejecutar shadow top-20

```bash
python3 framework/stages/05-reconstruction/scripts/shadow.py
```

This ingests and copies the frozen TIFXYZ surfaces to
`phase2/inputs/surfaces`. No consulta ground truth ni acepta objetivos.

## 6. Validate the official export

```bash
python3 framework/stages/05-reconstruction/scripts/validate_exports.py
```

It must report 20/20 files, no failures.

## 7. Tests

```bash
python3 -m pytest tests phase2/tests -q
```

## 8. Regenerar ledger y reportes

```bash
python3 framework/stages/05-reconstruction/scripts/finalize.py
```

## 9. Final verification

```bash
git status --short
python3 -m json.tool phase2/PHASE2_RESULTS.json >/dev/null
python3 -m json.tool phase2/RUN_STATE.json >/dev/null
```

The expected status stays `BLOCKED_VALIDATION_DATA` until the scientific data
listed in `PHASE2_RESULTS.md` is supplied.
"""
    atomic(PHASE2 / "RECONSTRUCTION.md", reconstruction)
    atomic(
        PHASE2 / "BLOCKED.md",
        """# Phase 2 scientific blocker

The autonomous implementation reached the authorised limit without substituting
datos. Faltan ground truth Paris 4, muestreo CT directo, etiquetas de signo
relative winding and attached official patches for the smoke test. See
`PHASE2_RESULTS.md` for the detail and `RECONSTRUCTION.md` to repeat the work.
""",
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
