#!/usr/bin/env bash
# Recompute every downstream result with the adversarial baselines included.
#
# Prerequisite: the extras bundles are in results/ next to the published ones:
#   results/{depmap,ctrpv2}_seed<s>_extra/   (from training/gen_preds_extra.py)
# shared/preds_io.py merges each *_extra dir with its *_preds sibling, so the
# scripts below pick up DANN and AD-AE without any further wiring.
#
# Safe to run without the extras present: every consumer skips methods whose
# arrays are absent, so this reproduces the original results unchanged.
#
#   bash scripts/recompute_with_extras.sh [python]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${1:-${PYTHON:-$HERE/.venv/bin/python}}
cd "$HERE"
echo ">> using $PY"

step () { echo; echo "=== $* "; }

step "seed-average the new methods -> depmap_seedavg_extra (Fig 4a input)"
$PY training/aggregate_seed_importances.py --root results --dataset depmap \
    --methods dann,adae --suffix seedavg_extra

step "Fig 2: per-gene attribution eta^2"
(cd fig02_ci_distribution && $PY compute_attr_eta2.py --preds-root ../results)
step "Fig 2: contamination@K"
(cd fig02_ci_distribution && $PY compute_contamination_at_k.py --preds-root ../results)
step "Fig A2: AUROC vs tau"
(cd fig02_ci_distribution && $PY compute_auroc_tau.py --preds-root ../results)

step "Fig 3: rank-difference GSEA"
(cd fig03_pathways && $PY compute_pathway_delta_gsea.py --preds-root ../results)

step "Fig 4a: self- and partner-recovery"
(cd fig04a_selfrank_partner && $PY compute_self_recovery.py \
    && $PY compute_partner_recovery.py)

step "Fig 4b: PAM50 subtype specificity"
(cd fig04b_breast_subtypes && $PY compute_subtype_specificity.py --preds-root ../results)

step "Tables 1 / A1 / A2: per-item accuracy"
(cd table_performance && $PY compute_per_item_pearson.py --preds-root ../results \
    && $PY compute_per_item_mse.py --preds-root ../results \
    && $PY make_table.py && $PY make_table_mse.py)

step "audit: every result CSV must actually carry the new methods"
# A consumer whose METHODS list was extended but whose loader still calls
# np.load runs cleanly and silently omits the new methods. Fail loudly here
# instead of shipping a four-method figure that looks finished.
$PY scripts/check_methods_present.py

step "figures"
(cd fig02_ci_distribution && $PY plot_ci_distribution.py && $PY plot_auroc_tau.py)
(cd fig03_pathways && $PY plot_pathway_delta_dotplot.py --turned)
(cd fig04a_selfrank_partner && $PY make_panelA.py)
(cd fig04b_breast_subtypes && $PY plot_breast_subtypes.py)
(cd fig04_causal_eval && $PY make_joined_causal_eval_vertical_nogap.py)

echo; echo ">> done."
