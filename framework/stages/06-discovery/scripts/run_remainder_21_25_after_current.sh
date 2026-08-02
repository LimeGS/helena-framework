#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/campaign-x-phase4}"
BATCH_ROOT="$ROOT/phase4/expanded_candidate_surface_screen_v1"
RANKING="$BATCH_ROOT/GLOBAL_COARSE_WINDOW_RANKING_REMAINDER_21_25.json"
CHECKPOINT="$ROOT/models/timesformer_GP_scroll1/model.safetensors"
GATE="$ROOT/phase4/ct_fiber_benchmark_v1/CT_FIBER_GATE_FREEZE.json"
RECEIPT_NAME="ROBUST_WINDOW_BATCH_RECEIPT_GP_SCROLL1_RANKS_21_25.json"
LOG="$BATCH_ROOT/ROBUST_RANKS_21_25.stdout.log"

exec > >(tee -a "$BATCH_ROOT/ROBUST_RANKS_21_25_ORCHESTRATOR.stdout.log") 2>&1

echo "WAITING: ranks 9-20 runner"
while pgrep -f "run_expanded_robust_windows.py.*--start-rank 9.*--limit 12" >/dev/null; do
  sleep 30
done

python3 - "$BATCH_ROOT" "$RANKING" "$CHECKPOINT" "$GATE" <<'PY'
import hashlib
import json
import pathlib
import sys

batch_root, ranking_path, checkpoint, gate = map(pathlib.Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

prior = json.loads(
    (
        batch_root
        / "robust_windows_v1"
        / "ROBUST_WINDOW_BATCH_RECEIPT_GP_SCROLL1_RANKS_09_20.json"
    ).read_text()
)
if prior.get("completed_count") != 12:
    raise SystemExit("ranks 9-20 did not complete")
if prior.get("selected_global_ranks") != list(range(9, 21)):
    raise SystemExit("ranks 9-20 receipt is not contiguous")
expected = {
    ranking_path: "c4b26589a87bd07b613debb3225a16d769defaaa6cd3b02849fe58bd351b2bce",
    checkpoint: "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488",
    gate: "d0ac3eb2d518ebefc544db069078c08868da902323cbf8cea2e1bfd8e4dd122b",
}
for path, frozen_hash in expected.items():
    actual = sha256(path)
    if actual != frozen_hash:
        raise SystemExit(f"hash mismatch: {path}: {actual}")
print("READY: frozen remainder, checkpoint, and CT gate verified")
PY

PYTHONPATH="$ROOT/scripts" python3 \
  "$ROOT/framework/stages/06-discovery/scripts/run_expanded_robust_windows.py" \
  --root "$ROOT" \
  --ranking-path "$RANKING" \
  --checkpoint "$CHECKPOINT" \
  --model-family "timesformer_GP_scroll1" \
  --screening-name "ink_screening_gp_scroll1_v1" \
  --batch-receipt-name "$RECEIPT_NAME" \
  --start-rank 21 \
  --limit 5 \
  --continue-after-text-like \
  --inference-batch-size 128 \
  2>&1 | tee "$LOG"

bash "$ROOT/framework/stages/03-ink/scripts/postprocess_robust_batch.sh" \
  "$ROOT" \
  "$RECEIPT_NAME" \
  21 \
  5 \
  "target_application_ranks_21_25_v1" \
  "manual_review_bundle_gp_scroll1_ranks21_25_v1" \
  "ROBUST_RANKS_21_25_POSTPROCESS.stdout.log" \
  "ink_screening_gp_scroll1_v1"
