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
   `results/performance_table.csv` (full stats) and `performance_table.tex`.

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

## The residualized within-tissue measurement (important)

`yhat_residualized = f(X_res) + yhat_baseline`, where `yhat_baseline` is the
leave-one-fold-out per-tissue mean, added back only to put predictions on the
raw scale (for **overall** r). That baseline term has within-tissue r ≈ **−0.38**
(CTRPv2), the leave-out anti-correlation, worst in small tissues, and it varies
*by fold within a tissue*, so per-tissue centring cannot remove it. Evaluating
within-tissue r on the **reconstructed** vector therefore gives a misleadingly
low **0.06** (CTRPv2) / 0.16 (DepMap), driven entirely by small tissues
(`res_raw` on tissues ≥20 cells is already ~0.16; see `_diag_residualized.py`).

Since within-tissue r is *meant* to exclude the tissue-mean term, we evaluate it
on the deconfounded model output `f(X_res) = yhat_residualized − yhat_baseline`.
This is **0.228 (DepMap) / 0.217 (CTRPv2)** and is flat across tissue sizes, the
model's real within-tissue skill. The contaminated value is kept as
`wt_pearson_recon` in the CSV for transparency. Marginal/IRM have no added-back
baseline, so for them `wt_pearson == wt_pearson_recon`.

## Key finding (read before citing)

- **Overall $r$ is inflated by tissue.** The tissue-mean baseline reaches
  $r=0.24$ (DepMap) / $0.35$ (CTRPv2) with **no** within-tissue signal
  ($r_{\mathrm{wt}}<0$, the leave-out artifact).
- **Every trained model keeps positive $r_{\mathrm{wt}}$**, far above baseline -
  genuine within-tissue signal, not lineage.
- **Residualization is the best predictor on BOTH axes** (overall + within-tissue
  r), consistent with its model being trained directly on the within-tissue
  (residual) signal. So "no cost to predictive quality" holds, and then some -
  for residualization. IRM is the weaker predictor on both. The draft's claim is
  supported once the residualized within-tissue r is measured on the model
  output rather than the baseline-contaminated reconstruction.
