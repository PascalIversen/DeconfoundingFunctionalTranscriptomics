#!/usr/bin/env bash
# Reproduce the denoising/ analyses. Reads the published results/ bundle and
# the well_predicted_subset/ item-subset definitions; writes only inside denoising/.
set -euo pipefail
cd "$(dirname "$0")"

python3 compute_stability.py          # heavy: one pass over the per-cell SHAP npz (~15 min)
python3 analyze_stability.py          # tables (needs well_predicted_subset/results/{ds}_wt_items.csv)
python3 compute_fig4_input_stability.py   # Fig-4 readout stability (~10 min)
python3 plot_denoising.py             # figures/fig_denoising.{pdf,png}
