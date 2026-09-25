# fig02_ci_distribution/

Reproduces **`fig06_ci_distribution_singlerow.pdf`**.

For every (item, gene, method) tuple we compute the **tissue η² of the
per-cell SHAP attribution**, a one-way ANOVA of φ against tissue:

    η² = SS_between_tissue / SS_total

Unlike the magnitude-based contamination index, this metric is defined
identically for every method (they all emit per-cell φ), so the marginal
baseline and the three deconfounders sit on one comparable axis. Lower
η² = less tissue signal left in the attribution.

The figure also needs Contamination@K (fraction of top-K attributed genes
that are tissue markers), computed by `compute_contamination_at_k.py`. Per
dataset: one row with a wide histogram of the per-(item × gene) η²
distribution plus a narrow per-method median/IQR summary, and a
Contamination@K-vs-K panel. `plot_ci_distribution.py` renders either a 2×2
grid (both datasets, default) or, with `--row`, the single-row layout
actually used in the paper (`fig06_ci_distribution_singlerow.pdf`).

## Layout

```
fig02_ci_distribution/
├── compute_attr_eta2.py         # SHAP .npz → results/{depmap,ctrpv2}_attr_eta2.csv
├── compute_contamination_at_k.py # SHAP .npz → results/{depmap,ctrpv2}_contamination_at_k.csv (+ _meta.csv)
├── plot_ci_distribution.py      # those CSVs → figures/fig06_ci_distribution[_singlerow].{pdf,png}
├── compute_auroc_tau.py         # SHAP .npz → results/auroc_tau.csv (Appendix Fig. A2)
├── plot_auroc_tau.py            # → figures/fig_auroc_tau.{pdf,png}
├── results/                     # populated by the compute_*.py scripts
└── figures/                     # populated by the plot_*.py scripts
```

## Reproducing the figure

### 1. Get the SHAP `.npz` files

`compute_attr_eta2.py` and `compute_contamination_at_k.py` consume the
`{depmap,ctrpv2}_seed1_preds/*.npz` files written by
`../training/gen_preds_v2.py`. Either re-run training or point
`--preds-root` at the existing paper outputs.

### 2. Build the per-method η² and Contamination@K CSVs

```bash
cd fig02_ci_distribution
python compute_attr_eta2.py
python compute_contamination_at_k.py --preds-root ../training/results
```

Writes `results/{depmap,ctrpv2}_attr_eta2.csv` (one row per item × gene ×
method) and `results/{depmap,ctrpv2}_contamination_at_k.csv` (+ `_meta.csv`).
Takes a few minutes.

### 3. Render the figure

```bash
python plot_ci_distribution.py --row   # the paper's fig06_ci_distribution_singlerow.pdf
python plot_ci_distribution.py         # 2x2 grid, both datasets
```

Takes a few seconds.

## Notes

- Contamination@K needs the tissue-marker set (η²_expr > τ, see
  `compute_contamination_at_k.py`); everything else here only needs the
  per-cell SHAP matrices already saved by the training scripts.
