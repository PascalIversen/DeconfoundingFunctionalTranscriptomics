# fig04b_breast_subtypes/

Breast-cancer subtype-defining (**PAM50 lineage subset**) genes in the
CTRPv2 drug-response attribution ranking, and how deconfounding makes
their upvoting *breast-specific*.

For every drug and every tissue, rank the panel genes by mean `|SHAP|`
among that tissue's cell lines and score how well that ranking recovers
the PAM50 lineage subset (AUROC of PAM50-lineage membership vs.
importance). A genuinely breast-specific signature should peak in breast
(an epithelial lineage) and sit lower in non-epithelial lineages.

## Gene set choice

The reference is the **PAM50 lineage-receptor subset (28 genes)**: the full
PAM50 (Parker 2009, *J Clin Oncol* 27:1160; 50 genes, matching
`../training/pam50_genes.txt`, the panel actually used as model input
features) minus its 22-gene proliferation arm (the Hallmark G2M_CHECKPOINT /
E2F_TARGETS cell-cycle programme, which includes the canonical Nielsen 2010 /
Wallden 2015 proliferation meta-gene). In CCLE every immortalized line
proliferates uniformly, so this arm decouples from breast-subtype membership
and dilutes the AUROC signal in cell lines.

The retained 28 genes (ER/PR receptors, FOXA1/C1 lineage TFs, ERBB2/GRB7
HER2 amplicon, KRT5/14/17 basal cytokeratins, CDH3, etc.) reflect
cell-of-origin biology that survives immortalization. See
`compute_subtype_specificity.py` for the exact gene lists.

## Layout

```
fig04b_breast_subtypes/
├── compute_subtype_specificity.py  # SHAP .npz → results/subtype_tissue_specificity.csv
├── plot_breast_subtypes.py         # CSV → figures/fig_breast_subtypes.{pdf,png}
├── results/
└── figures/
```

## Reproducing

```bash
cd fig04b_breast_subtypes
python compute_subtype_specificity.py --preds-root /path/to/results
python plot_breast_subtypes.py
```

`--preds-root` must contain `ctrpv2_seed1_preds/*.npz` (per-cell SHAP
matrices produced by `../training/gen_preds_v2.py`).
