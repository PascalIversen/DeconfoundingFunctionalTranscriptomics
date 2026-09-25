#!/usr/bin/env bash
# Full revision run for the adversarial model-stage baselines (DANN, AD-AE).
#
# Stages, in dependency order:
#   A  depmap seed 1, full per-cell SHAP   -> Fig 2 (eta2, contamination), Fig 3 (GSEA)
#   B  ctrpv2 seed 1, full per-cell SHAP   -> Fig 2, Fig 3, Fig 4b (PAM50)
#   C  depmap seeds 2-5, per-gene imp only -> Fig 4a (seed-averaged recovery)
#   D  ctrpv2 seeds 2-5, predictions only  -> Tables 1 / A1 / A2
#   E  lambda sweeps, both datasets        -> Table A3 analogue
#   F  DANN with Ganin's own SGD + mu_p    -> faithfulness sensitivity
#
# Run under tmux; every stage appends to $LOGDIR. Resume-safe: each stage
# skips items whose npz already exists.
set -uo pipefail
P=${PROJECT_DIR:-/storage/mi/piversen/decon_rev}
LOGDIR=$P/_logs
mkdir -p "$LOGDIR"
cd "$P/training"

stage () {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name" | tee -a "$LOGDIR/driver.log"
  "$@" >> "$LOGDIR/$name.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] END   $name (rc=$?)" | tee -a "$LOGDIR/driver.log"
}

stage A_depmap_seed1  bash dispatch_extra.sh depmap 1
stage B_ctrpv2_seed1  bash dispatch_extra.sh ctrpv2 1

for s in 2 3 4 5; do
  stage C_depmap_seed$s bash dispatch_extra.sh depmap $s --save-imp
done
for s in 2 3 4 5; do
  stage D_ctrpv2_seed$s bash dispatch_extra.sh ctrpv2 $s --no-shap
done

# --- lambda sweeps (single node each; 20 items x 6 lambdas x 2 methods) ---
SWEEP_NODE=${SWEEP_NODE:-compute05}
VP=${VENV_PYTHON:-/storage/mi/piversen/venv-decon/bin/python}
for ds in depmap ctrpv2; do
  stage E_sweep_$ds ssh -o BatchMode=yes "$SWEEP_NODE" \
    "cd $P/table_irm_sweep && $VP -u run_adv_sweep.py --dataset $ds --n-items 20 --nproc 22"
done

# --- faithfulness sensitivity: Ganin's optimiser and lr schedule ---
stage F_ganin_depmap ssh -o BatchMode=yes "$SWEEP_NODE" \
  "cd $P/table_irm_sweep && $VP -u run_adv_sweep.py --dataset depmap --n-items 20 --nproc 22 --methods dann --dann-optimizer ganin --out results/adv_sweep_depmap_ganinopt.csv"

echo "=== [$(date +%H:%M:%S)] ALL STAGES COMPLETE" | tee -a "$LOGDIR/driver.log"
