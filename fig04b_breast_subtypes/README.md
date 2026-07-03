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

The reference is the **PAM50 lineage-receptor subset (41 genes)**:
the full PAM50 (Parker 2009, *J Clin Oncol* 27:1160) minus the canonical
11-gene proliferation meta-gene (Nielsen 2010, *Clin Cancer Res*
16:5222; Wallden 2015, *BMC Med Genomics*). The excluded proliferation
genes, BIRC5, CCNB1, CDC20, CENPF, CEP55, KIF2C, MKI67, MYBL2, RRM2,
UBE2C, ANLN, were selected by PAM50 to distinguish Luminal A vs B by
proliferation index in patient tumors. In CCLE every immortalized line
proliferates uniformly, so this arm decouples from breast-subtype
membership and dilutes the AUROC signal in cell lines.

The retained 41 genes (ER/PR receptors, FOXA1/C1 lineage TFs, ERBB2/GRB7
HER2 amplicon, KRT5/14/17 basal cytokeratins, etc.) reflect cell-of-origin
biology that survives immortalization.

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
