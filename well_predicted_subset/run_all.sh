#!/usr/bin/env bash
# Do the Fig-2 diagnostics and the Fig-3 GSEA hold on the subset of items with
# meaningful within-tissue predictive performance (A.6.1)?
#
#   bash well_predicted_subset/run_all.sh [python]
#
# ~5 min. Reads only the published results/ bundle and the existing figure CSVs;
# writes nothing outside well_predicted_subset/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${1:-${PYTHON:-$HERE/../.venv/bin/python}}
cd "$HERE"
echo ">> using $PY"

step () { echo; echo "=== $* "; }

step "define the subsets (A.6.1 filter + two neutrality variants)"
$PY wt_subset.py

step "Fig 2 restricted (row filter on the existing per-item CSVs)"
$PY subset_fig02.py

step "Fig 3 restricted (re-runs the pre-ranked GSEA; the delta vector is a cross-item mean)"
$PY subset_fig03.py

step "one-figure summary: full item set vs subset"
$PY plot_robustness.py --variant a61
$PY plot_robustness.py --variant marginal

step "fold-symmetric (method-neutral) within-tissue r"
$PY fold_symmetric_rwt.py

echo; echo ">> done. tables in results/, figures in figures/"
