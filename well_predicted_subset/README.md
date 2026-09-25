# `well_predicted_subset/` — do the main results hold on well-predicted items, and is the r_wt convention neutral?

Two questions, backing Appendix `suppsec:well_predicted` and the fold-symmetric r_wt
convention used in Table 1.

Nothing here writes outside `well_predicted_subset/`. Reproduce with
`bash well_predicted_subset/run_all.sh` (~5 min, reads the published `results/` bundle).

## The item subsets

`wt_subset.py` reimplements the A.6.1 filter from `results/{ds}_within_tissue_pvals.csv`
(median one-sided within-tissue-r p over the five seeds, BH-FDR q<0.05 per model) and
extends it to CTRPv2, which A.6.1 never covered:

| variant | definition |
|---|---|
| `full` | all items |
| **`a61`** | **BH(marginal) ∩ BH(residualized) — the paper's filter** |
| `marginal` | BH(marginal) only |
| `all4` | BH over marginal, residualized, IRM, within-tissue |

`a61` selects items on which both marginal and residualized models perform
significantly, which is not obviously neutral between the compared methods. `marginal`
is the neutral alternative — the shared baseline alone picks the items, so no
deconfounder is favoured — and `all4` is the strictest symmetric variant. DANN/AD-AE
have no within-tissue-r p-values (`gen_preds_extra.py` does not write them) and so
cannot enter the filter; they are still evaluated on every subset, which is fine
because the subsets are only item selections. Per-dataset item counts land in
`results/{ds}_wt_items.csv`.

## Fig 2 — attribution η² and contamination@K

Both measures are per-item, so restricting the figure is a row filter on the existing
`fig02_ci_distribution/results/*.csv`; nothing is recomputed from the `.npz`. The marker
set *L* and the chance line are defined on expression, not items, so they carry over.

Figures: `figures/fig02_{a61,marginal,all4}[_singlerow].pdf`, rendered by the paper's own
`plot_ci_distribution.py`. Tables: `results/fig02_subset_summary.csv` (per-method
subset vs `full` comparison) and `results/fig02_paired_tests.csv` (per-item paired
Wilcoxon vs marginal, per measure/dataset).

## Fig 3 — rank-difference GSEA

This one cannot be filtered post hoc: the δ vector is a *cross-item mean* importance
percentile, so the whole pre-ranked GSEA is recomputed (10 000 permutations, as in the
paper) on a symlink farm holding only the subset's `.npz`. The paper's
`compute_pathway_delta_gsea.py` and `plot_pathway_delta_dotplot.py` are called unmodified.

Figures: `figures/fig03_{a61,marginal,all4}.pdf`, `figures/fig_robustness_{a61,marginal}.pdf`
(`plot_robustness.py`, a one-figure Fig-2 + Fig-3 robustness summary: per-method
hollow/filled bars and an NES(all) vs NES(subset) scatter per dataset). Tables:
`results/fig03_concordance.csv` (full-vs-`a61` NES agreement over the shared gene-set
universe) and `results/fig03_lineage_summary.csv` (demoted/promoted counts over the
HGA lineage signatures, full vs subset).

---

## Fold-symmetric within-tissue r

How are residualized predictions compared against marginal ones in Table 1? For OVERALL
r they already are: the stored `yhat_residualized` is f(X_res) + train-fold tissue mean
(orchestrator.py -> reconstruct_raw_yhat), so both methods are correlated against the
same raw y.

**The asymmetry is in r_wt.** `compute_per_item_pearson.py` evaluates r_wt for
residualized on the deconfounded output f(X_res) = yhat − yhat_baseline, and for every
other method on the raw yhat. The justification is real: yhat_baseline is a
LEAVE-ONE-FOLD-OUT per-tissue mean, so it carries a strong negative within-tissue
correlation and, because it differs between folds inside the same tissue, centring per
tissue cannot remove it. But the consequence is that residualization's r_wt lead over
marginal exists only under that method-specific convention.

### The neutral estimator

`fold_symmetric_rwt.py` centres y and yhat within **(tissue × fold)** groups instead of
within tissue:

    r_wt_foldsym = corr(y − mean_{t,f} y,  yhat − mean_{t,f} yhat)

The leave-out baseline is constant inside such a group, so it drops out identically for
every method and no per-method convention is needed. Folds are recoverable without
retraining — `KFold(n_splits=5, shuffle=True, random_state=seed)` on the cell axis
(`orchestrator.py:199`) depends only on `(n_cells, seed)`. Runtime ~75 s over all 5 seeds.

Two built-in checks confirm the reconstruction is right (both pass on both datasets, see
`results/fold_symmetric_summary.csv`): the tissue-mean baseline's centred r_wt is exactly
zero (the centred vector is all-zero, so the correlation is undefined, not merely small),
and residualized raw vs baseline-stripped r_wt agree once fold-symmetric centring is
applied.

### What this changes

Table 1 now reports the fold-symmetric convention (Appendix, `suppsec:well_predicted`)
rather than the per-tissue one. The claim changes in strength, not in sign: "residualization
**matches** marginal within tissue while cutting tissue contamination" — i.e. deconfounding
costs no predictive accuracy — rather than "attains the highest overall *and* within-tissue
r". Overall *r* is untouched by any of this and still favours residualization.

The reason every method's r_wt rises under the neutral estimator is the whole argument:
decomposing the per-tissue centred covariance into its between-(tissue × fold) and
within-(tissue × fold) parts shows the between part is CV-harness nuisance (each cell's
prediction comes from a different one of the 5 fold models), which dilutes r for every
trained method under the per-tissue convention. Residualization's number under the
paper's original convention is already close to nuisance-free (that is what stripping
`base_yhat` accomplishes), while marginal's was still diluted — so most of
residualization's apparent lead under the old convention was marginal's own dilution,
not a residualization-specific within-tissue advantage.

Numbers: `results/fold_symmetric_summary.csv` (mean r_wt per method/dataset, both
conventions), `results/fold_symmetric_paired.csv` (paired tests vs marginal), per-item
values in `results/fold_symmetric_rwt.csv`.

## Files

| file | what it does |
|---|---|
| `wt_subset.py` | builds the three item subsets → `results/{ds}_wt_items.csv` |
| `subset_fig02.py` | filters the Fig-2 CSVs, renders Fig 2, writes the summary + paired tests |
| `subset_fig03.py` | symlink farm → re-runs the real GSEA → renders Fig 3 → concordance tables |
| `plot_robustness.py` | one-figure summary of the Fig-2/Fig-3 robustness check |
| `fold_symmetric_rwt.py` | method-neutral within-tissue r |
| `run_all.sh` | all of the above |
