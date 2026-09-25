#!/usr/bin/env bash
# Multi-node dispatcher for the adversarial-baseline extras (gen_preds_extra.py).
#
# Mirrors dispatch.sh: splits an items-file by node weights, ssh's each slice
# to a worker node, and monitors aggregate progress. Results land on the
# NFS-shared $OUT_DIR, so each item writes its own npz and no merge is needed.
#
# Usage:
#   dispatch_extra.sh <dataset> <seed> [extra gen_preds_extra.py flags...]
#
# Examples:
#   # seed 1, full per-cell SHAP (Figs 2/3/4b)
#   dispatch_extra.sh depmap 1
#   # seeds 2-5, per-gene importances only (Fig 4a seed-average)
#   dispatch_extra.sh depmap 2 --save-imp
#   # seeds 2-5, predictions only (Tables 1/A1/A2)
#   dispatch_extra.sh ctrpv2 2 --no-shap
#
# Env (optional): PROJECT_DIR VENV_PYTHON OUT_DIR ITEMS_FILE TARGETS_FILE
#                 ADAE_CACHE_DIR WORKERS_SPEC
#
# NOTE on ADAE_CACHE_DIR: the AD-AE encoder is phenotype-free, so items with
# identical cell sets share one. That is a large win on DepMap (200 targets ->
# 5 distinct cell sets) and worth nothing on CTRPv2 (449 drugs -> 449 distinct
# cell sets), where it would also write ~10 GB of never-reused encoders. It is
# therefore enabled for depmap only, by default.
set -euo pipefail
DATASET=${1:?dataset required}
SEED=${2:?seed required}
shift 2
PASSTHRU=("$@")

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TRAINING_DIR=$PROJECT_DIR/training
VENV_PYTHON=${VENV_PYTHON:-/storage/mi/piversen/venv-decon/bin/python}
ITEMS_FILE=${ITEMS_FILE:-$TRAINING_DIR/items_${DATASET}.txt}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/results}
DISPATCH_DIR=$TRAINING_DIR/_dispatch_extra/${DATASET}_seed${SEED}
mkdir -p "$DISPATCH_DIR" "$OUT_DIR"

if [[ "$DATASET" == "depmap" ]]; then
  ADAE_CACHE_DIR=${ADAE_CACHE_DIR:-$OUT_DIR/_adae_cache_seed${SEED}}
  CACHE_ARG="--adae-cache-dir $ADAE_CACHE_DIR"
else
  CACHE_ARG=""
fi

# (name:cores) pairs, weighted to leave headroom on nodes other users are on.
DEFAULT_WORKERS="compute05:24 compute02:20 compute01:16 compute04:16 compute03:10"
read -r -a WORKERS <<< "${WORKERS_SPEC:-$DEFAULT_WORKERS}"

TOTAL_CORES=0
for w in "${WORKERS[@]}"; do TOTAL_CORES=$(( TOTAL_CORES + ${w##*:} )); done
echo "[dispatch-extra] $DATASET seed=$SEED  items=$(wc -l < "$ITEMS_FILE")  cores=$TOTAL_CORES  flags=${PASSTHRU[*]:-none}"

python3 - "$ITEMS_FILE" "$DISPATCH_DIR" "${WORKERS[@]}" <<'PYEOF'
import sys
from pathlib import Path
items_file, out_dir, *workers = sys.argv[1:]
items = [l.strip() for l in open(items_file) if l.strip()]
weights = [int(w.split(":")[1]) for w in workers]
total = sum(weights)
shares = [round(len(items) * w / total) for w in weights]
shares[0] += len(items) - sum(shares)
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
i = 0
for w, n in zip(workers, shares):
    name = w.split(":")[0]
    (out / f"{name}.txt").write_text("\n".join(items[i:i + n]) + "\n")
    print(f"  {name}: {n} items")
    i += n
PYEOF

PIDS=()
for w in "${WORKERS[@]}"; do
  NODE=${w%%:*}; CORES=${w##*:}
  CHUNK=$DISPATCH_DIR/$NODE.txt
  LOG=$DISPATCH_DIR/$NODE.log
  [[ -s "$CHUNK" ]] || { echo "  [$NODE] empty chunk, skip"; continue; }
  REMOTE_CMD="cd $TRAINING_DIR && $VENV_PYTHON -u gen_preds_extra.py --dataset $DATASET --seed $SEED --items-file $CHUNK --out-dir $OUT_DIR --nproc $CORES $CACHE_ARG ${PASSTHRU[*]:-}"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$NODE" "$REMOTE_CMD" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "  launched $NODE (pid $!) nproc=$CORES -> $LOG"
done

# NB: `ls .../*.npz | wc -l` under `set -o pipefail` aborts the whole script
# while the output dir is still empty (ls exits 1, pipefail propagates, set -e
# kills it) -- which silently killed this monitor loop on its first iteration
# while the ssh workers carried on in the background. Count with a form that
# tolerates "no matches yet".
count_npz() {
  find "$OUT_DIR/${DATASET}_seed${SEED}_extra" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ' || echo 0
}

echo "[dispatch-extra] waiting on ${#PIDS[@]} workers ..."
START=$(date +%s)
while true; do
  ALIVE=0
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && ALIVE=$(( ALIVE + 1 )); done
  DONE=$(count_npz)
  TOTAL=$(wc -l < "$ITEMS_FILE" | tr -d ' ')
  echo "  [$(date +%H:%M:%S)] alive=$ALIVE  done=$DONE/$TOTAL  elapsed=$(( $(date +%s) - START ))s"
  [[ $ALIVE -eq 0 ]] && break
  sleep 30
done

echo "[dispatch-extra] all workers exited. final tally:"
echo "  extras: $(count_npz)/$(wc -l < "$ITEMS_FILE") items"
for w in "${WORKERS[@]}"; do
  NODE=${w%%:*}; LOG=$DISPATCH_DIR/$NODE.log
  [[ -s "$LOG" ]] && { echo "  --- $NODE tail ---"; tail -3 "$LOG"; }
done
