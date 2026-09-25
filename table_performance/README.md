# performance_table

Predictive-accuracy table: **overall** Pearson $r$ vs **within-tissue** Pearson
$r_{\mathrm{wt}}$ for each predictor, on both datasets.

## Pipeline

1. `compute_per_item_pearson.py --preds-root <dir>`: reads the out-of-fold
   prediction arrays in `<dir>/{depmap,ctrpv2}_seed<s>_preds/*.npz` and writes
   `results/per_item_pearson.csv` (one row per dataset × seed × item × method;
   overall `pearson` and `wt_pearson`), reading the 5-seed per-cell preds from
   `../results/`.
2. `make_table.py`: aggregates `results/per_item_pearson.csv` →
   `results/performance_table.csv` (full stats) and `performance_table.tex` /
   `performance_table_sd.tex`. Overall $r$ comes from `per_item_pearson.csv`;
   within-tissue $r_{\mathrm{wt}}$ comes from the fold-symmetric convention in
   `../well_predicted_subset/results/fold_symmetric_rwt.csv` (see below), not
   from this file's `wt_pearson` column.

## Aggregation decisions

- **Replication unit = the item** (drug / knockout target); each item is its own
  trained model, so items are the independent observations.
- **Seeds averaged within item first** (5 seeds are repeats, not extra n).
- **Spread = SEM across items** (SD / √n_items): the table compares method
  *means*, so the precision of the across-item mean is the relevant uncertainty.
  The large item-to-item **SD is kept in `performance_table.csv`** and noted in
  the caption.
- All methods scored on the **same items and folds** (paired).
- `r` averaged arithmetically (not Fisher-z): descriptive summary on the
  correlation scale.

## Rows

Tissue-mean baseline (reference) + the three trained predictors. **Within-tissue
SHAP shares the marginal model** (`yhat_within ≡ yhat_marginal`, asserted in the
compute script), so it is folded into the *Marginal* row, a separate predictive
row would be a duplicate.

## Within-tissue r: the fold-symmetric convention (important)

`yhat_residualized = f(X_res) + yhat_baseline`, where `yhat_baseline` is the
leave-one-fold-out per-tissue mean, added back only to put predictions on the
raw scale (for **overall** r). That baseline term has a negative within-tissue r
(the leave-out anti-correlation, worst in small tissues), and it varies
*by fold within a tissue*, so per-tissue centring cannot remove it -- evaluating
within-tissue r on the raw reconstructed vector gives a misleadingly low value
for residualized specifically (`wt_pearson_recon` in the CSV keeps this
contaminated value for transparency; Marginal/IRM have no added-back baseline,
so for them `wt_pearson == wt_pearson_recon`).

Centring per tissue alone cannot remove this artifact, and stripping the
baseline back off before scoring residualized (but not the other methods)
would be a method-specific convention. Instead, `wt_mean`/`wt_sem`/`wt_sd`
here are the **fold-symmetric** within-tissue r
(`../well_predicted_subset/fold_symmetric_rwt.py`): centring $y$ and $\hat y$
within (tissue × fold) groups instead of within tissue removes the leave-out
baseline term identically for every method, so no per-method handling is
needed -- residualized's fold-symmetric r_wt already equals its
baseline-stripped value. See `../well_predicted_subset/README.md` ("Fold-symmetric
within-tissue r") for the full derivation and validity checks.

## Results

See `results/performance_table.csv` (full stats, both conventions) and
`performance_table.tex` / `performance_table_sd.tex` for the numbers, and
`../well_predicted_subset/README.md` for the paired significance tests behind
the fold-symmetric within-tissue comparison.
