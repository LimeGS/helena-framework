#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/campaign-x-phase4}"
BATCH_ROOT="$ROOT/phase4/expanded_candidate_surface_screen_v1"
EXPANSION_ROOT="$ROOT/phase4/shortlist_expansion_v1"
EXPANSION_PLAN="$EXPANSION_ROOT/SHORTLIST_EXPANSION_PLAN.json"
CHECKPOINT="$ROOT/models/timesformer_GP_scroll1/model.safetensors"
RENDERER="/workspace/villa-phase3/build-phase3-gcc13/bin/vc_render_tifxyz"
GATE="$ROOT/phase4/ct_fiber_benchmark_v1/CT_FIBER_GATE_FREEZE.json"

COARSE_SCREENING="coarse_screen_gp_scroll1_v1"
COARSE_RECEIPT="EXPANDED_SURFACE_BATCH_RECEIPT_GP_SCROLL1.json"
PER_SAMPLE_RANKING="coarse_ranking_gp_scroll1_v1"
RANKING_NAME="GLOBAL_COARSE_WINDOW_RANKING_GP_SCROLL1_ALL_SHORTLIST.json"
RANKING="$BATCH_ROOT/$RANKING_NAME"
ROBUST_ROOT_NAME="robust_windows_gp_scroll1_all_shortlist_v1"
ROBUST_RECEIPT="ROBUST_WINDOW_BATCH_RECEIPT_GP_SCROLL1_ALL_SHORTLIST.json"
ROBUST_SCREENING="ink_screening_gp_scroll1_all_shortlist_v1"
FREEZE="$BATCH_ROOT/GP_SCROLL1_ALL_SHORTLIST_RUNTIME_FREEZE.json"
LOG="$BATCH_ROOT/GP_SCROLL1_ALL_SHORTLIST_ORCHESTRATOR.stdout.log"

exec > >(tee -a "$LOG") 2>&1

wait_for_process() {
  local pattern="$1"
  local label="$2"
  for _ in $(seq 1 1440); do
    if ! pgrep -f "$pattern" >/dev/null; then
      echo "READY: $label ended"
      return 0
    fi
    sleep 30
  done
  echo "BLOCKED: timeout waiting for $label"
  return 1
}

echo "WAITING: frozen old-ranking batches and complete shortlist expansion"
wait_for_process \
  "run_remainder_21_25_after_current.sh" \
  "old-ranking ranks 21-25 pipeline"
wait_for_process \
  "run_wave1_expansion.py.*SHORTLIST_EXPANSION_PLAN.json" \
  "78-surface expansion"

python3 - "$ROOT" "$EXPANSION_PLAN" "$CHECKPOINT" "$GATE" "$RENDERER" <<'PY'
import hashlib
import json
import pathlib
import sys

root, plan_path, checkpoint, gate, renderer = map(pathlib.Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

plan = json.loads(plan_path.read_text())
if plan.get("status") != "LOCKED_READY" or plan.get("surface_count") != 78:
    raise SystemExit("shortlist expansion plan is not the frozen 78-surface plan")
expected = {
    checkpoint: "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488",
    gate: "d0ac3eb2d518ebefc544db069078c08868da902323cbf8cea2e1bfd8e4dd122b",
}
for path, frozen_hash in expected.items():
    actual = sha256(path)
    if actual != frozen_hash:
        raise SystemExit(f"frozen hash mismatch: {path}: {actual}")
if not renderer.is_file():
    raise SystemExit("renderer missing")
expanded = list((root / "phase4/targets").glob("PHerc*/candidate_surfaces/*/expanded/meta.json"))
if len(expanded) < 60:
    raise SystemExit(f"insufficient expanded shortlist coverage: {len(expanded)}/78")
print(f"READY: {len(expanded)}/78 expanded surfaces available")
PY

echo "START: GP-Scroll1 coarse screen over every available shortlist surface"
PYTHONPATH="$ROOT" python3 \
  "$ROOT/framework/stages/06-discovery/scripts/screen_expanded_candidate_surfaces.py" \
  --root "$ROOT" \
  --renderer "$RENDERER" \
  --checkpoint "$CHECKPOINT" \
  --model-family "timesformer_GP_scroll1" \
  --screening-name "$COARSE_SCREENING" \
  --batch-receipt-name "$COARSE_RECEIPT" \
  --cache-gb 12

echo "START: deterministic GP-Scroll1 global ranking"
python3 "$ROOT/framework/stages/06-discovery/scripts/rank_expanded_candidate_windows.py" \
  --root "$ROOT" \
  --batch-root "$BATCH_ROOT" \
  --batch-receipt-name "$COARSE_RECEIPT" \
  --screening-name "$COARSE_SCREENING" \
  --per-sample-ranking-name "$PER_SAMPLE_RANKING" \
  --ranking-output-name "$RANKING_NAME" \
  --minimum-valid-ratio 0.70 \
  --global-top-n 48 \
  --max-per-sample 4

SELECTED_COUNT="$(
  python3 - "$ROOT" "$RANKING" "$CHECKPOINT" "$GATE" "$FREEZE" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root, ranking_path, checkpoint, gate, output = map(pathlib.Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ranking = json.loads(ranking_path.read_text())
selected = ranking["global_priority"]
count = len(selected)
if count < 24:
    raise SystemExit(f"GP ranking produced too few robust windows: {count}")
if [row["global_rank"] for row in selected] != list(range(1, count + 1)):
    raise SystemExit("GP ranking is not contiguous")
freeze = {
    "kind": "campaign_x_phase4_gp_scroll1_all_shortlist_runtime_freeze_v1",
    "status": "FROZEN_BEFORE_ROBUST_EXECUTION",
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z"),
    "ranking": {
        "path": str(ranking_path),
        "sha256": sha256(ranking_path),
        "selected_count": count,
        "global_top_n": 48,
        "max_per_sample": 4,
    },
    "checkpoint": {
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "model_family": "timesformer_GP_scroll1",
    },
    "ct_fiber_gate": {
        "path": str(gate),
        "sha256": sha256(gate),
    },
    "robust_screen": {
        "depth_centers": [25, 32, 39],
        "tiling_offsets": [0, 8],
        "frames": 26,
        "training_pixel_um": 7.91,
        "training_slice_um": 7.91,
        "tile_size": 64,
        "stride": 16,
        "min_valid_ratio": 0.6,
        "glyph_threshold": 0.5,
        "inference_batch_size": 128,
    },
    "script_hashes": {
        name: sha256(root / "scripts" / name)
        for name in (
            "run_expanded_robust_windows.py",
            "build_ct_fiber_target_application_spec.py",
            "extract_ct_fiber_features.py",
            "apply_ct_fiber_gate.py",
            "export_visual_review_bundle.py",
        )
    },
    "policy": [
        "the GP checkpoint performs both coarse prioritization and robust screening",
        "all available expanded Phase 1 shortlist surfaces enter coarse screening",
        "the ranking and robust count are frozen before robust execution",
        "every detected component receives the unchanged frozen CT fiber gate",
        "no output automatically proves ink, letters, or a First Letters claim",
    ],
}
output.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
print(count)
PY
)"

echo "FROZEN: $SELECTED_COUNT robust windows"
PYTHONPATH="$ROOT" python3 \
  "$ROOT/framework/stages/06-discovery/scripts/run_expanded_robust_windows.py" \
  --root "$ROOT" \
  --batch-root "$BATCH_ROOT" \
  --ranking-path "$RANKING" \
  --checkpoint "$CHECKPOINT" \
  --model-family "timesformer_GP_scroll1" \
  --screening-name "$ROBUST_SCREENING" \
  --robust-root-name "$ROBUST_ROOT_NAME" \
  --batch-receipt-name "$ROBUST_RECEIPT" \
  --start-rank 1 \
  --limit "$SELECTED_COUNT" \
  --continue-after-text-like \
  --inference-batch-size 128

bash "$ROOT/framework/stages/03-ink/scripts/postprocess_robust_batch.sh" \
  "$ROOT" \
  "$ROBUST_RECEIPT" \
  1 \
  "$SELECTED_COUNT" \
  "target_application_gp_scroll1_all_shortlist_v1" \
  "manual_review_bundle_gp_scroll1_all_shortlist_v1" \
  "GP_SCROLL1_ALL_SHORTLIST_POSTPROCESS.stdout.log" \
  "$ROBUST_SCREENING" \
  "$ROBUST_ROOT_NAME" \
  "$COARSE_RECEIPT" \
  "$RANKING_NAME"

echo "COMPLETE: GP-Scroll1 shortlist pipeline"
