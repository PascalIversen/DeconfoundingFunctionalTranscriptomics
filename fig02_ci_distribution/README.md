# fig02_ci_distribution/

Reproduces **`fig06_ci_distribution_singlerow.pdf`**.

For every (item, gene, method) tuple we compute the **tissue η² of the
per-cell SHAP attribution**, a one-way ANOVA of φ against tissue:

    η² = SS_between_tissue / SS_total

Unlike the magnitude-based contamination index, this metric is defined
identically for every method (they all emit per-cell φ), so the marginal
baseline and the three deconfounders sit on one comparable axis. Lower
η² = less tissue signal left in the attribution.

The figure is a 2×2 grid: one row per dataset (DepMap, CTRPv2), and per
row a wide histogram of the per-(item × gene) η² distribution plus a
narrow per-method median/IQR summary.

## Layout

```
fig02_ci_distribution/
├── compute_attr_eta2.py     # SHAP .npz → results/{depmap,ctrpv2}_attr_eta2.csv
├── plot_ci_distribution.py  # those CSVs → figures/fig06_ci_distribution.{pdf,png}
├── results/                 # populated by compute_attr_eta2.py
└── figures/                 # populated by plot_ci_distribution.py
```

## Reproducing the figure

### 1. Get the SHAP `.npz` files

`compute_attr_eta2.py` consumes the `{depmap,ctrpv2}_seed1_preds/*.npz`
files written by `../training/gen_preds_v2.py`
and `../training/gen_preds_v2.py`. Either re-run
training or point `--preds-root` at the existing paper outputs.

### 2. Build the per-method η² CSVs

```bash
cd fig02_ci_distribution
python compute_attr_eta2.py
# or, with a custom preds-root:
python compute_attr_eta2.py --preds-root ../training/results
```

Writes `results/depmap_attr_eta2.csv` and `results/ctrpv2_attr_eta2.csv`
(one row per item × gene × method). Takes ~1 min.

### 3. Render the figure

```bash
python plot_ci_distribution.py
```

Writes `figures/fig06_ci_distribution.{pdf,png}`. Takes a few seconds.

## Notes

- This figure does not need pathway gene-sets or any analysis beyond the
  per-cell SHAP matrices already saved by the training scripts.
