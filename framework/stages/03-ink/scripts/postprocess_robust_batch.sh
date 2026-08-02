#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 11 ]]; then
  echo "usage: $0 ROOT RECEIPT_NAME START_RANK LIMIT APPLICATION_NAME VIEWER_NAME LOG_NAME SCREENING_NAME [ROBUST_ROOT_NAME [COARSE_BATCH_RECEIPT_NAME [RANKING_NAME]]]" >&2
  exit 64
fi

ROOT="$1"
RECEIPT_NAME="$2"
START_RANK="$3"
LIMIT="$4"
APPLICATION_NAME="$5"
VIEWER_NAME="$6"
LOG_NAME="$7"
SCREENING_NAME="$8"
ROBUST_ROOT_NAME="${9:-robust_windows_v1}"
COARSE_BATCH_RECEIPT_NAME="${10:-EXPANDED_SURFACE_BATCH_RECEIPT.json}"
RANKING_NAME="${11:-GLOBAL_COARSE_WINDOW_RANKING.json}"

for value in "$RECEIPT_NAME" "$APPLICATION_NAME" "$VIEWER_NAME" "$LOG_NAME" "$SCREENING_NAME" "$ROBUST_ROOT_NAME" "$COARSE_BATCH_RECEIPT_NAME" "$RANKING_NAME"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "unsafe output name: $value" >&2
    exit 65
  fi
done
if [[ ! "$START_RANK" =~ ^[0-9]+$ || ! "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "start rank and limit must be positive integers" >&2
  exit 66
fi
if (( START_RANK < 1 || LIMIT < 1 )); then
  echo "start rank and limit must be positive integers" >&2
  exit 66
fi

BATCH_ROOT="$ROOT/phase4/expanded_candidate_surface_screen_v1"
ROBUST_ROOT="$BATCH_ROOT/$ROBUST_ROOT_NAME"
RECEIPT="$ROBUST_ROOT/$RECEIPT_NAME"
SPEC="$ROOT/phase4/ct_fiber_benchmark_v1/${APPLICATION_NAME^^}_SPEC.json"
APPLICATION="$ROOT/phase4/ct_fiber_benchmark_v1/$APPLICATION_NAME"
GATE="$ROOT/phase4/ct_fiber_benchmark_v1/CT_FIBER_GATE_FREEZE.json"
VIEWER="$BATCH_ROOT/$VIEWER_NAME"
LOG="$BATCH_ROOT/$LOG_NAME"

exec > >(tee -a "$LOG") 2>&1

python3 - "$RECEIPT" "$START_RANK" "$LIMIT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
start = int(sys.argv[2])
limit = int(sys.argv[3])
receipt = json.loads(path.read_text())
if receipt.get("status") not in {
    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE",
    "COMPLETED_DIAGNOSTIC_ONLY",
}:
    raise SystemExit("robust batch receipt is not complete")
if receipt.get("completed_count") != limit:
    raise SystemExit("robust batch receipt count mismatch")
if receipt.get("selected_global_ranks") != list(range(start, start + limit)):
    raise SystemExit("robust batch rank mismatch")
PY

echo "READY: robust receipt complete"
PYTHONPATH="$ROOT" python3 \
  "$ROOT/framework/stages/04-validation/scripts/build_ct_fiber_target_application_spec.py" \
  --root "$ROOT" \
  --batch-root "$BATCH_ROOT" \
  --batch-receipt "$RECEIPT" \
  --gate-freeze "$GATE" \
  --output "$SPEC" \
  --application-name "$APPLICATION_NAME" \
  --robust-root-name "$ROBUST_ROOT_NAME"

PYTHONPATH="$ROOT" python3 \
  "$ROOT/framework/stages/04-validation/scripts/extract_ct_fiber_features.py" \
  --root "$ROOT" \
  --spec "$SPEC" \
  --output "$APPLICATION/features"

python3 "$ROOT/framework/stages/04-validation/scripts/apply_ct_fiber_gate.py" \
  --features "$APPLICATION/features/CT_FIBER_FEATURES.csv" \
  --rule "$GATE" \
  --output "$APPLICATION/gate"

python3 "$ROOT/framework/stages/04-validation/scripts/export_visual_review_bundle.py" \
  --root "$ROOT" \
  --batch-root "$BATCH_ROOT" \
  --output "$VIEWER" \
  --screening-name "$SCREENING_NAME" \
  --robust-root-name "$ROBUST_ROOT_NAME" \
  --coarse-batch-receipt-name "$COARSE_BATCH_RECEIPT_NAME" \
  --ranking-name "$RANKING_NAME" \
  --batch-receipt-name "$RECEIPT_NAME" \
  --ct-fiber-control-evaluation \
    "$ROOT/phase4/ct_fiber_benchmark_v1/gate_evaluation_v1/CT_FIBER_GATE_EVALUATION.json" \
  --ct-fiber-target-evaluation \
    "$APPLICATION/gate/CT_FIBER_GATE_EVALUATION.json"

sha256sum \
  "$RECEIPT" \
  "$SPEC" \
  "$APPLICATION/features/CT_FIBER_FEATURES.csv" \
  "$APPLICATION/gate/CT_FIBER_GATE_EVALUATION.json" \
  "$VIEWER/MANUAL_VISUAL_REVIEW_MANIFEST.json"
echo "COMPLETE: ranks $START_RANK-$((START_RANK + LIMIT - 1)) CT gate and viewer"
