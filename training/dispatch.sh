#!/usr/bin/env bash
# Multi-node SHAP dispatcher. Splits an items-file by node weights, ssh's
# each slice to a worker node, monitors aggregate progress. Results land on
# NFS-shared $OUT_DIR so no merge step is needed (each item writes its own
# npz to a shared output dir).
#
# Usage:
#   dispatch.sh <dataset> <seed>
# Env (optional):
#   PROJECT_DIR   path to this repo on the cluster (containing this dir)
#   VENV_PYTHON   python interpreter on a shared filesystem
#   OUT_DIR       results root (NFS-shared)
#   ITEMS_FILE    one item per line (drugs for ctrpv2, knockout targets for depmap)
#   TARGETS_FILE  HUGO symbols to union into the expression panel
#
# Workers are defined inline below as (name:cores) pairs. nproc per node is
# chosen to match the physical (not hyperthreaded) core count.

set -euo pipefail
DATASET=${1:?dataset required}
SEED=${2:?seed required}
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TRAINING_DIR=$PROJECT_DIR/training
VENV_PYTHON=${VENV_PYTHON:-python}
ITEMS_FILE=${ITEMS_FILE:-$TRAINING_DIR/items_${DATASET}.txt}
TARGETS_FILE=${TARGETS_FILE:-$TRAINING_DIR/targets_${DATASET}.txt}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/results}
DISPATCH_DIR=$TRAINING_DIR/_dispatch/${DATASET}_seed${SEED}
mkdir -p "$DISPATCH_DIR" "$OUT_DIR"

# (name:cores) pairs. Tune to your cluster.
WORKERS=(
  "compute02:24"
  "compute03:24"
  "compute04:20"
  "compute05:24"
  "compute08:6"
  "compute09:6"
)

TOTAL_CORES=0
for w in "${WORKERS[@]}"; do TOTAL_CORES=$(( TOTAL_CORES + ${w##*:} )); done
echo "[dispatch] $DATASET seed=$SEED  items=$(wc -l < "$ITEMS_FILE")  total cores=$TOTAL_CORES"

# Split items.txt into per-node chunks weighted by core count.
python3 - "$ITEMS_FILE" "$DISPATCH_DIR" "${WORKERS[@]}" <<'PYEOF'
import sys
from pathlib import Path
items_file, out_dir, *workers = sys.argv[1:]
items = [l.strip() for l in open(items_file) if l.strip()]
weights = [int(w.split(":")[1]) for w in workers]
total = sum(weights)
shares = [round(len(items) * w / total) for w in weights]
diff = len(items) - sum(shares)
shares[0] += diff
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
i = 0
for w, n in zip(workers, shares):
    name = w.split(":")[0]
    chunk = items[i:i + n]; i += n
    (out / f"{name}.txt").write_text("\n".join(chunk) + "\n")
    print(f"  {name}: {len(chunk)} items")
PYEOF

PIDS=()
for w in "${WORKERS[@]}"; do
  NODE=${w%%:*}; CORES=${w##*:}
  CHUNK=$DISPATCH_DIR/$NODE.txt
  LOG=$DISPATCH_DIR/$NODE.log
  if [[ ! -s "$CHUNK" ]]; then
    echo "  [$NODE] empty chunk, skip"; continue
  fi
  REMOTE_CMD="cd $TRAINING_DIR && $VENV_PYTHON gen_preds_v2.py --dataset $DATASET --seed $SEED --items-file $CHUNK --targets-file $TARGETS_FILE --union-targets --out-dir $OUT_DIR --nproc $CORES"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$NODE" "$REMOTE_CMD" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "  launched $NODE (pid $!) nproc=$CORES -> $LOG"
done

echo "[dispatch] waiting on ${#PIDS[@]} workers ..."
START=$(date +%s)
while true; do
  ALIVE=0
  for p in "${PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null && ALIVE=$(( ALIVE + 1 ))
  done
  DONE=$(ls "$OUT_DIR/${DATASET}_seed${SEED}_preds/"*.npz 2>/dev/null | wc -l | tr -d ' ')
  TOTAL=$(wc -l < "$ITEMS_FILE" | tr -d ' ')
  ELAPSED=$(( $(date +%s) - START ))
  echo "  [$(date +%H:%M:%S)] alive=$ALIVE  done=$DONE/$TOTAL  elapsed=${ELAPSED}s"
  [[ $ALIVE -eq 0 ]] && break
  sleep 30
done

echo "[dispatch] all workers exited. final tally:"
ls "$OUT_DIR/${DATASET}_seed${SEED}_preds/"*.npz 2>/dev/null | wc -l | xargs -I{} echo "  preds: {}/$(wc -l < "$ITEMS_FILE") items"
for w in "${WORKERS[@]}"; do
  NODE=${w%%:*}; LOG=$DISPATCH_DIR/$NODE.log
  [[ -s "$LOG" ]] && echo "  --- $NODE tail ---" && tail -3 "$LOG"
done
