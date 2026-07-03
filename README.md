# Tissue deconfounding for feature attribution in functional transcriptomics

Reproduction code for the figures, tables, and other results in the paper. The pipeline
compares a confounded baseline (marginal Shapley) against three tissue-deconfounding
strategies (linear residualization at the data level, IRM at the model level, and
within-tissue SHAP at the attribution level) on DepMap CRISPR gene essentiality and
CTRPv2 drug response.

## What's here

Each paper figure/table maps to one top-level directory:

| Paper | Directory | Command | Inputs |
|-------|-----------|---------|--------|
| **Fig 1** (b–g) confounding evidence | `fig01_confounding_evidence/` | `python compute_confounding.py && python compute_classifier_per_tissue.py && python plot_confounding.py` | raw data |
| **Fig 2** eta²/contamination | `fig02_ci_distribution/` | `python plot_ci_distribution.py` (compute first: `compute_attr_eta2.py`, `compute_contamination_at_k.py`) | predictions |
| **Fig 3** rank-difference GSEA | `fig03_pathways/` | `python plot_pathway_delta_dotplot.py --turned` (compute: `compute_pathway_delta_gsea.py`) | predictions + `gene_sets/` |
| **Fig 4** biological evaluation | `fig04_causal_eval/` | `python make_joined_causal_eval_vertical_nogap.py` | `fig04a_*` + `fig04b_*` outputs |
| **Fig A1** PCA / tissue structure | `fig01_confounding_evidence/` | `plot_confounding.py` (→ `fig05b`) | raw data |
| **Fig A2** AUROC vs τ | `fig02_ci_distribution/` | `python plot_auroc_tau.py` (compute: `compute_auroc_tau.py`) | predictions |
| **Fig A3** other confounders | `fig01_confounding_evidence/` | `plot_confounding.py` (→ `fig05c`) | raw data + `DepMap/Model.csv` |
| **Fig A4** synthetic reliability | `figA4_synthetic_reliability/` | `python plot_reliability.py` (compute: `compute_npower_mixed.py`) | self-contained (synthetic) |
| **Table 1 / A1** predictive accuracy | `table_performance/` | `python make_table.py` (compute: `compute_per_item_pearson.py`) | predictions |
| **Table A2** MSE | `table_performance/` | `python make_table_mse.py` | predictions |
| **Table A3** IRM λ sweep | `table_irm_sweep/` | `python make_table_irm.py` (compute: `run_irm_sweep.py`) | retrains a subset |

`fig04a_selfrank_partner/` and `fig04b_breast_subtypes/` compute the two panels that
`fig04_causal_eval/` assembles; they are not standalone paper figures.

Shared library: `training/shared/` (data loaders, models, training, deconfounding,
SHAP estimators). Retraining entry point: `training/gen_preds_v2.py`.

## Setup

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data (Zenodo)

Large inputs and predictions are on Zenodo
([10.5281/zenodo.21171970](https://doi.org/10.5281/zenodo.21171970)); small metadata,
gene panels, and gene sets ship in the repo under `data/`, `training/`,
`fig03_pathways/gene_sets/`.

```bash
bash scripts/fetch_data.sh          # raw data -> data/ ; predictions -> results/
```

- **Deposit 1, raw data (~1.3 GB):** `DepMap/CRISPRGeneEffect.csv`,
  `DepMap/OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `CCLE/gene_expression.csv`,
  `CTRPv2/CTRPv2.csv`.
- **Deposit 2, predictions + precomputed result CSVs (~2.5–3.5 GB):** per-cell SHAP
  `.npz` (`results/ctrpv2_seed1_preds/`, `results/depmap_*_preds/`), per-seed summary
  CSVs, and the large computed CSVs (eta², GSEA, self/partner recovery, etc.).

## To reproduce

1. **Figures from provided results (minutes, CPU).** After `fetch_data.sh preds`, run
   each `plot_*.py` / `make_table*.py`. Regenerates every figure/table from the
   precomputed CSVs. No training, no raw data.
2. **Results from provided predictions (~1 h, CPU).** Run each `compute_*.py` on the
   per-cell `.npz`, then step 1. Full for both datasets: every result CSV, figure,
   and table recomputes from the provided predictions.
3. **Predictions from raw data (hours, GPU).** After `fetch_data.sh raw`, run
   `training/gen_preds_v2.py` (single node) or `training/dispatch.sh` (multi-node),
   then steps 2–1. Optional: you do not need to retrain.

## Notes

- **Predictions bundle.** Seed 1 ships full per-cell SHAP for both datasets
  (`{depmap,ctrpv2}_seed1_preds/`), all attribution figures (eta², contamination,
  GSEA, self/partner recovery) recompute from it. Seeds 2–5 ship as lean yhat-only
  npz (`{depmap,ctrpv2}_seed[2-5]_preds/`, ~170 MB total), which is all the 5-seed
  performance tables (1/A1/A2) need. `depmap_seedavg_preds/` holds the seed-averaged
  importances used by Fig 4a (regenerate with `training/aggregate_seed_importances.py`
  from the full 5-seed preds).
- **Fig 1 composite.** `fig01_confounding_evidence/` produces the data panels
  (`fig05`, `fig05b`–`fig05e`); panel **a** (the causal DAG) is drawn by hand and the
  final Fig 1 is assembled from these components in the manuscript.
