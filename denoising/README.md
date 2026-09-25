# `denoising/` — is averaged attribution denoising demonstrated, not just asserted?

The models being explained are weak within tissue (see Table A1). The Discussion
defends averaging per-cell attributions into a global explanation as a denoising step,
but that claim needs direct evidence rather than an assertion. Given that the
self/partner-recovery analysis already restricts to items with significant r_wt, the
main diagnostics and GSEA should also be checked on that same well-predicted subset,
and the A.6.1 intersection filter — which selects items on which both marginal and
residualized models perform significantly — should be checked for neutrality between
the compared methods.

Three parts. The subset restriction and the filter-neutrality variants for Fig 2
(η²_attr, contamination@K) and Fig 3 (GSEA) are done in `well_predicted_subset/`; this
folder **tests the denoising claim empirically** and closes the two diagnostics
`well_predicted_subset/` does not cover (Fig A2 AUROC-τ, Fig 4b subtype specificity).

Nothing here writes outside `denoising/`. Reproduce with `bash denoising/run_all.sh`
(~20 min; the heavy step is one pass over the per-cell SHAP npz). Requires
`well_predicted_subset/results/{ds}_wt_items.csv` (run `well_predicted_subset/wt_subset.py`
first if missing).

## 1. Testing the denoising claim

The Discussion sentence under test — "averaging them can denoise the local
explanations into a more stable global explanation" — is tested directly on the shipped
per-cell |SHAP| matrices (seed 1, both datasets, all six methods + the between-tissue
control): permute cells, split into disjoint halves, average the first *n* cells of
each half, correlate the two mean-importance vectors over genes
(`compute_stability.py`, 10 repeats; grid n = 1…n_cells/2). The observed stability
curve is compared against the Spearman–Brown prophecy formula r_n = n·r₁/(1+(n−1)·r₁)
predicted from the single-cell correlation r₁ alone, to check whether cell-to-cell
variation behaves as independent noise that averaging removes at the theoretical rate.

The same per-cell-half approach is used to check robustness of the downstream readouts
actually used in the paper: the Fig-3 GSEA input (cross-item δ percentile vector) and
contamination@50 (`results/delta_split_half.csv`, `results/contamination_half.csv`),
validated against the published all-cells CSVs.

Per-item, per-method values are in `results/item_stability.csv`; dataset × method
aggregates in `results/stability_summary.csv`.

**Connection to r_wt (panels b/e).** Per-item stability is related to the item's
within-tissue r (`results/item_stability.csv`), checked for both datasets.

**Honest scope.** Split-half stability demonstrates that averaging removes *variance*;
it cannot speak to *bias* — a systematically wrong explanation would replicate across
halves too. That part of the argument is carried by the ground-truth-anchored analyses
(self-recovery, STRING/CORUM partner recovery, PAM50 specificity).

Figure: `figures/fig_denoising.{pdf,png}` — (a/d) stability vs n with Spearman–Brown
overlay, (b/e) per-item stability vs r_wt, (c/f) split-half reliability of the δ vector.

## 2. Remaining diagnostics on the well-predicted subset

`well_predicted_subset/` restricts Fig 2 and Fig 3 to the three filter variants. The
two diagnostics it does not cover are closed here, computed per item from the
all-cells mean importances (no re-read of the npz):

- **Fig A2 (AUROC-τ)** — `results/auroc_tau_subsets.csv`, per method × τ × variant
  (`full`, `a61`, `marginal`, `all4`; `full` validated against the published CSV).
- **Fig 4b (PAM50 breast-subtype specificity)** — `results/fig04b_subset.csv` (row
  filter on the published per-drug CSV, same breast-vs-non-epithelial-tissues gap
  definition as `fig04b_breast_subtypes/plot_breast_subtypes.py`).

## 3. The filter-neutrality question

Answered in `well_predicted_subset/` (see its README, "Item subsets") and inherited by
the tables here: the paper's `a61` intersection filter is compared against a
**`marginal`-only filter** — the shared baseline alone selects the items, so no
deconfounding method can be favoured — and against the strict symmetric `all4`.

## Files

| file | what it does |
|---|---|
| `compute_stability.py` | heavy pass: split-half curves + half/full mean matrices → `results/stability_curves.csv`, `results/halfmeans_{ds}.npz` |
| `analyze_stability.py` | all tables: `item_stability.csv`, `stability_summary.csv`, `delta_split_half.csv`, `contamination_half.csv`, `auroc_tau_subsets.csv`, `fig04b_subset.csv`, `eta_{ds}.csv` (cache) |
| `compute_fig4_input_stability.py` | split-half stability of the Fig-4 readouts: `fig4a_self_stability.csv`, `fig4a_partner_stability.csv`, `fig4a_{self,partner}_deltas.csv`, `fig4b_pam50_stability.csv`, `fig4b_pam50_rows.csv` |
| `plot_denoising.py` | `figures/fig_denoising.{pdf,png}` |
| `run_all.sh` | all of the above |
