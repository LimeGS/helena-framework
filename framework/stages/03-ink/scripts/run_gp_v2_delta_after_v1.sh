#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

ROOT="${1:-/workspace/campaign-x-phase4}"
BATCH_ROOT="$ROOT/phase4/expanded_candidate_surface_screen_v1"
SCRIPTS="$ROOT/scripts"

script_path() {
  local name="$1"
  local candidate
  if [[ -f "$SCRIPTS/$name" ]]; then
    printf '%s\n' "$SCRIPTS/$name"
    return 0
  fi
  for candidate in \
    "$ROOT/framework/stages/01-segmentation/scripts/$name" \
    "$ROOT/framework/stages/02-flattening/scripts/$name" \
    "$ROOT/framework/stages/03-ink/scripts/$name" \
    "$ROOT/framework/stages/04-validation/scripts/$name" \
    "$ROOT/framework/stages/05-reconstruction/scripts/$name" \
    "$ROOT/framework/stages/06-discovery/scripts/$name"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}
CHECKPOINT="$ROOT/models/timesformer_GP_scroll1/model.safetensors"
GATE="$ROOT/phase4/ct_fiber_benchmark_v1/CT_FIBER_GATE_FREEZE.json"

COARSE_RECEIPT="$BATCH_ROOT/EXPANDED_SURFACE_BATCH_RECEIPT_GP_SCROLL1.json"
COARSE_SCREENING="coarse_screen_gp_scroll1_v1"
V1_RANKING="$BATCH_ROOT/GLOBAL_COARSE_WINDOW_RANKING_GP_SCROLL1_ALL_SHORTLIST.json"
V1_ROBUST_ROOT_NAME="robust_windows_gp_scroll1_all_shortlist_v1"
V1_ROBUST_RECEIPT_NAME="ROBUST_WINDOW_BATCH_RECEIPT_GP_SCROLL1_ALL_SHORTLIST.json"
V1_ROBUST_RECEIPT="$BATCH_ROOT/$V1_ROBUST_ROOT_NAME/$V1_ROBUST_RECEIPT_NAME"
V1_APPLICATION_NAME="target_application_gp_scroll1_all_shortlist_v1"
V1_CT_EVALUATION="$ROOT/phase4/ct_fiber_benchmark_v1/$V1_APPLICATION_NAME/gate/CT_FIBER_GATE_EVALUATION.json"
V1_VIEWER_NAME="manual_review_bundle_gp_scroll1_all_shortlist_v1"
V1_VIEWER_MANIFEST="$BATCH_ROOT/$V1_VIEWER_NAME/MANUAL_VISUAL_REVIEW_MANIFEST.json"
V1_LOG="$BATCH_ROOT/GP_SCROLL1_ALL_SHORTLIST_ORCHESTRATOR.stdout.log"

V2_PER_SAMPLE_RANKING="coarse_ranking_gp_scroll1_v2"
V2_RANKING_NAME="GLOBAL_COARSE_WINDOW_RANKING_GP_SCROLL1_ALL_SHORTLIST_V2.json"
V2_RANKING="$BATCH_ROOT/$V2_RANKING_NAME"
DELTA_RANKING_NAME="GLOBAL_COARSE_WINDOW_RANKING_GP_SCROLL1_ALL_SHORTLIST_V2_DELTA.json"
DELTA_RANKING="$BATCH_ROOT/$DELTA_RANKING_NAME"
DELTA_FREEZE="$BATCH_ROOT/GP_SCROLL1_ALL_SHORTLIST_V2_DELTA_FREEZE.json"
DELTA_ROBUST_ROOT_NAME="robust_windows_gp_scroll1_all_shortlist_v2_delta"
DELTA_ROBUST_RECEIPT_NAME="ROBUST_WINDOW_BATCH_RECEIPT_GP_SCROLL1_ALL_SHORTLIST_V2_DELTA.json"
DELTA_ROBUST_RECEIPT="$BATCH_ROOT/$DELTA_ROBUST_ROOT_NAME/$DELTA_ROBUST_RECEIPT_NAME"
DELTA_SCREENING="ink_screening_gp_scroll1_all_shortlist_v2_delta_v1"
DELTA_APPLICATION_NAME="target_application_gp_scroll1_all_shortlist_v2_delta_v1"
DELTA_CT_EVALUATION="$ROOT/phase4/ct_fiber_benchmark_v1/$DELTA_APPLICATION_NAME/gate/CT_FIBER_GATE_EVALUATION.json"
DELTA_APPLICATION_ROOT="$ROOT/phase4/ct_fiber_benchmark_v1/$DELTA_APPLICATION_NAME"
DELTA_APPLICATION_SPEC="$ROOT/phase4/ct_fiber_benchmark_v1/${DELTA_APPLICATION_NAME^^}_SPEC.json"
DELTA_VIEWER_NAME="manual_review_bundle_gp_scroll1_all_shortlist_v2_delta_v1"
DELTA_VIEWER_ROOT="$BATCH_ROOT/$DELTA_VIEWER_NAME"
DELTA_POSTPROCESS_LOG="GP_SCROLL1_ALL_SHORTLIST_V2_DELTA_POSTPROCESS.stdout.log"
DELTA_POSTPROCESS_LOG_PATH="$BATCH_ROOT/$DELTA_POSTPROCESS_LOG"
MERGE_MANIFEST="$BATCH_ROOT/GP_SCROLL1_ALL_SHORTLIST_V2_MERGED_EVIDENCE_MANIFEST.json"

ABSOLUTE_MIN_FREE_BYTES=$((10 * 1024 * 1024 * 1024))
# Historical robust windows occupy about 0.442 GiB each. Reserve 0.5 GiB per
# new window plus 8 GiB for CT postprocessing, viewers and filesystem margin.
# This is deliberately computed after the exact delta count is frozen.
PER_DELTA_WINDOW_BYTES=$((512 * 1024 * 1024))
POSTPROCESS_HEADROOM_BYTES=$((8 * 1024 * 1024 * 1024))
WAIT_SECONDS=30
MAX_WAITS=1440

required_scripts=(
  "validate_gp_v1_completion.py"
  "rank_coarse_ink_windows_v2.py"
  "rank_expanded_candidate_windows_v2.py"
  "prepare_gp_v2_delta.py"
  "run_expanded_robust_windows.py"
  "postprocess_robust_batch.sh"
  "merge_gp_v2_manifest.py"
)
for name in "${required_scripts[@]}"; do
  if ! script_path "$name" >/dev/null; then
    echo "BLOCKED: required script missing from flat or staged layout: $name" >&2
    exit 66
  fi
done

if (( DRY_RUN )); then
  printf '%s\n' \
    "DRY_RUN_ONLY: no files written and no jobs launched" \
    "WAIT: complete immutable v1 pipeline (87 coarse + up to 48 robust + CT + viewer)" \
    "RANK: $V2_RANKING" \
    "FREEZE: exact identity adapter -> $DELTA_RANKING and $DELTA_FREEZE" \
    "GUARD: require max(10 GiB, 8 GiB + 0.5 GiB per delta window)" \
    "ROBUST: only V2_top<=48 minus exact V1_top<=48 -> $DELTA_ROBUST_RECEIPT" \
    "POSTPROCESS: isolated CT application/viewer $DELTA_APPLICATION_NAME" \
    "MERGE: every selected v2 entry (up to 48) gets one provenance -> $MERGE_MANIFEST"
  exit 0
fi

echo "WAITING: immutable GP Scroll1 v1 pipeline"
ready=0
for _ in $(seq 1 "$MAX_WAITS"); do
  set +e
  python3 "$(script_path validate_gp_v1_completion.py)" \
    --coarse-receipt "$COARSE_RECEIPT" \
    --v1-ranking "$V1_RANKING" \
    --v1-robust-receipt "$V1_ROBUST_RECEIPT" \
    --v1-ct-evaluation "$V1_CT_EVALUATION" \
    --v1-viewer-manifest "$V1_VIEWER_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --gate-freeze "$GATE" \
    --orchestrator-log "$V1_LOG" \
    --expected-tasks 87 \
    --maximum-windows 48
  status=$?
  set -e
  if (( status == 0 )); then
    ready=1
    break
  fi
  if (( status != 75 )); then
    echo "BLOCKED: v1 completion validation failed closed" >&2
    exit "$status"
  fi
  sleep "$WAIT_SECONDS"
done
if (( ! ready )); then
  echo "BLOCKED: timed out waiting for complete v1 pipeline" >&2
  exit 75
fi

# The pipeline intentionally does not auto-resume a partial v2 namespace.
# A human must inspect a partial namespace rather than silently overwrite it.
python3 - "$BATCH_ROOT" "$V2_RANKING" "$DELTA_RANKING" "$DELTA_FREEZE" \
  "$BATCH_ROOT/$DELTA_ROBUST_ROOT_NAME" "$MERGE_MANIFEST" \
  "$DELTA_APPLICATION_ROOT" "$DELTA_APPLICATION_SPEC" "$DELTA_VIEWER_ROOT" \
  "$DELTA_POSTPROCESS_LOG_PATH" "$V2_PER_SAMPLE_RANKING" <<'PY'
import pathlib
import sys

batch = pathlib.Path(sys.argv[1])
fixed = [pathlib.Path(value) for value in sys.argv[2:11]]
per_sample_name = sys.argv[11]
present = [str(path) for path in fixed if path.exists()]
present.extend(
    str(path)
    for path in sorted(batch.glob(f"PHerc*/{per_sample_name}"))
    if path.exists()
)
if present:
    raise SystemExit(
        "BLOCKED: pre-existing v2 artifact requires explicit audit; "
        "nothing was overwritten:\n" + "\n".join(present)
    )
PY

echo "START: deterministic additive GP Scroll1 ranking v2"
python3 "$(script_path rank_expanded_candidate_windows_v2.py)" \
  --root "$ROOT" \
  --batch-root "$BATCH_ROOT" \
  --batch-receipt-name "$(basename "$COARSE_RECEIPT")" \
  --screening-name "$COARSE_SCREENING" \
  --per-sample-ranking-name "$V2_PER_SAMPLE_RANKING" \
  --ranking-output-name "$V2_RANKING_NAME" \
  --minimum-valid-ratio 0.70 \
  --global-top-n 48 \
  --max-per-sample 4 \
  --global-legacy-rescue-fraction 0.20 \
  --per-sample-legacy-rescue-fraction 0.25 \
  --v1-global-ranking "$V1_RANKING"

echo "START: exact-window delta adapter and immutable freeze"
python3 "$(script_path prepare_gp_v2_delta.py)" \
  --coarse-receipt "$COARSE_RECEIPT" \
  --v1-ranking "$V1_RANKING" \
  --v2-ranking "$V2_RANKING" \
  --v1-robust-receipt "$V1_ROBUST_RECEIPT" \
  --checkpoint "$CHECKPOINT" \
  --gate-freeze "$GATE" \
  --delta-ranking-output "$DELTA_RANKING" \
  --freeze-output "$DELTA_FREEZE" \
  --maximum-selected-count 48 \
  --script-root "$ROOT"

DELTA_COUNT="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["delta_compute_count"])' \
    "$DELTA_FREEZE"
)"
if [[ ! "$DELTA_COUNT" =~ ^[0-9]+$ ]] || (( DELTA_COUNT > 48 )); then
  echo "BLOCKED: invalid frozen delta count: $DELTA_COUNT" >&2
  exit 67
fi

if (( DELTA_COUNT > 0 )); then
  REQUIRED_FREE_BYTES=$((POSTPROCESS_HEADROOM_BYTES + DELTA_COUNT * PER_DELTA_WINDOW_BYTES))
  if (( REQUIRED_FREE_BYTES < ABSOLUTE_MIN_FREE_BYTES )); then
    REQUIRED_FREE_BYTES="$ABSOLUTE_MIN_FREE_BYTES"
  fi
  FREE_BYTES="$(
    python3 -c \
      'import os,sys; s=os.statvfs(sys.argv[1]); print(s.f_bavail*s.f_frsize)' \
      "$ROOT"
  )"
  if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]]; then
    echo "BLOCKED: could not determine free bytes" >&2
    exit 68
  fi
  if (( FREE_BYTES < REQUIRED_FREE_BYTES )); then
    echo "BLOCKED: $FREE_BYTES free bytes is below the dynamic $REQUIRED_FREE_BYTES-byte guard for delta=$DELTA_COUNT; no files deleted" >&2
    exit 75
  fi
  echo "READY: $FREE_BYTES free bytes; required=$REQUIRED_FREE_BYTES; delta=$DELTA_COUNT"

  echo "START: isolated robust screen for exact v2 set difference"
  PYTHONPATH="$SCRIPTS" python3 \
    "$(script_path run_expanded_robust_windows.py)" \
    --root "$ROOT" \
    --batch-root "$BATCH_ROOT" \
    --ranking-path "$DELTA_RANKING" \
    --checkpoint "$CHECKPOINT" \
    --model-family "timesformer_GP_scroll1" \
    --screening-name "$DELTA_SCREENING" \
    --robust-root-name "$DELTA_ROBUST_ROOT_NAME" \
    --batch-receipt-name "$DELTA_ROBUST_RECEIPT_NAME" \
    --start-rank 1 \
    --limit "$DELTA_COUNT" \
    --continue-after-text-like \
    --inference-batch-size 128

  bash "$(script_path postprocess_robust_batch.sh)" \
    "$ROOT" \
    "$DELTA_ROBUST_RECEIPT_NAME" \
    1 \
    "$DELTA_COUNT" \
    "$DELTA_APPLICATION_NAME" \
    "$DELTA_VIEWER_NAME" \
    "$DELTA_POSTPROCESS_LOG" \
    "$DELTA_SCREENING" \
    "$DELTA_ROBUST_ROOT_NAME" \
    "$(basename "$COARSE_RECEIPT")" \
    "$DELTA_RANKING_NAME"

  python3 "$(script_path merge_gp_v2_manifest.py)" \
    --freeze "$DELTA_FREEZE" \
    --v1-robust-receipt "$V1_ROBUST_RECEIPT" \
    --v1-ct-evaluation "$V1_CT_EVALUATION" \
    --delta-robust-receipt "$DELTA_ROBUST_RECEIPT" \
    --delta-ct-evaluation "$DELTA_CT_EVALUATION" \
    --output "$MERGE_MANIFEST"
else
  echo "SKIP: v2 top48 is entirely covered by exact v1 windows"
  python3 "$(script_path merge_gp_v2_manifest.py)" \
    --freeze "$DELTA_FREEZE" \
    --v1-robust-receipt "$V1_ROBUST_RECEIPT" \
    --v1-ct-evaluation "$V1_CT_EVALUATION" \
    --output "$MERGE_MANIFEST"
fi

echo "COMPLETE: additive GP Scroll1 v2 exact-delta pipeline"
