# fig01_confounding_evidence/

The *motivating evidence* for the whole study: that **tissue is a genuine
confounder** of the pharmacogenomic models, it drives both the expression
features and the outcome. If this holds, naive feature importance leaks
tissue-identity genes, which is exactly what the four deconfounding methods
(see [../training/](../training/)) are built to fix.

```
            Tissue (or another candidate confounder)
           /                                         \
          v                                           v
   outcome (IC50 / essentiality)                gene expression
        Arm 2                                        Arm 1
```

Unlike the other `fig*/` folders, this one needs **no trained `.npz`** -
the confounding is a property of the raw data, measured straight off the
`shared/data.py` loaders so the diagnostics see the data the same way the
pipeline does (LN_IC50, `log2(x+1)` CCLE expression, tissue from the CTRPv2
response). The feature panel is **dataset-specific**: `panel_depmap` (1071
genes) for DepMap and `panel_ctrpv2` (1109 genes) for CTRPv2, both under
`data/meta/` in the annotated `gene,landmark,target,pam50,source` format.

This is a from-scratch, scanpy-free port of the diagnostics in
`Confounding/bulk_deg_analysis.py` (a TxPert-style framework, Wenkel et al.
2026); the statistical kernels were reimplemented on numpy / scipy /
scikit-learn so no new dependencies are introduced.

## The figures

- **`fig05_confounding_evidence`**, the two-arm proof (rows = DepMap / CTRPv2).
  - *Arm 1, Tissue → expression*: per-gene η²(X) histogram.
  - *Arm 2, Tissue → outcome*: per-outcome η²(y) distribution.
  - Both panels overlay a translucent **shuffled-tissue null** (η² recomputed on
    permuted tissue labels), the real distribution sits far to its right, plus
    the median η² and the η² = 0.30 contamination threshold. The
    tissue-classifier lift (Arm 1) and the per-outcome significance (Arm 2) are
    no longer drawn here; they are broken out into `fig05d` and `fig05e` below.
- **`fig05b_confounding_structure`**, supporting structure.
  - PCA of expression coloured by tissue (top-10 lineages).
  - within- vs across-tissue sample-correlation distributions + Mann-Whitney p.
- **`fig05c_confounder_comparison`**, *"tissue, or something else?"*. **Debiased
  Δη²** of **both** arms across 7 `Model.csv` confounders (`OncotreeLineage`,
  `OncotreePrimaryDisease`, `OncotreeSubtype`, `Sex`, `PrimaryOrMetastasis`,
  `GrowthPattern`, `OnboardedMedia`). DepMap reads them directly; CTRPv2 joins
  `Model.csv` onto its cell lines via `RRID` (= cellosaurus_id) and runs on the
  matched subset (~818/820; the matched count is printed). `OncotreeLineage` is
  the lineage annotation, it is relabelled "Tissue (lineage)" and bolded in
  blue so it can be compared against the alternatives.

  **Debiasing (per-item permuted floor).** Each box is `Δη² = η²(real) − η²(own
  shuffled-label floor)`, where every gene/outcome's floor is the mean η² over
  `--n-perm` (default 20) shuffles of *that item's own labels on its own cells*
 , **not** a pooled average across items, so each item is corrected against its
  own chance level (which scales with its group sizes). This removes the upward
  η² bias that high-cardinality confounders (subtype, primary disease) otherwise
  enjoy. The floor is `confounding_confounder_compare_null.csv`; the plot falls
  back to raw η² if that file is absent. The solid line marks `Δη² = 0` (the
  floor); the dotted line marks the `η² = 0.30` contamination threshold.

  **Caveat on "Tissue (lineage)" for CTRPv2.** For DepMap this row *is* the
  headline tissue, `load_depmap_crispr` literally renames `OncotreeLineage` to
  `tissue`. For CTRPv2 it is **not** the same column: fig05/fig05b use the
  response `tissue`, while fig05c uses `OncotreeLineage` from the RRID join (so
  all 7 confounders share one source). The two correspond closely but are not
  identical (different vocabularies: Colon↔Bowel, Brain↔CNS/Brain, …; also
  Blood → Lymphoid + Myeloid, and the mesenchymal Soft Tissue / Muscle lines).
- **`fig05d_classifier_per_tissue`**, the Arm-1 linear tissue classifier broken
  out **per tissue**: a beeswarm (one point = one tissue) of its accuracy,
  averaged over the same stratified 5-fold CV, for each dataset. The solid bar is
  the median tissue, the dotted bar is chance (1 / #tissues).
  Driven by a **separate** compute step (see below) so it leaves the six main
  CSVs untouched; sized to half the width of `fig05` (one histogram column).
- **`fig05e_outcome_significance`**, the Arm-2 significance, broken out
  **per outcome**: a −log10(p) strip (one point = one drug/target's Welch-ANOVA
  p) per dataset, with the solid bar at the median and a dashed bar at each
  dataset's **Bonferroni** cutoff (α/mᵢ). The box reports how many outcomes
  survive that correction. The plain α=0.05 line
  is omitted on purpose (at −log10 it sits well below the Bonferroni cutoff, too
  close to read); Bonferroni is the significance reported from now on. Reads the
  same `confounding_outcome_eta2.csv` / `confounding_summary.csv` as `fig05`;
  sized like `fig05d`.

### fig05b PCA: what "other" means

The PCA scatter colours the **10 most populated tissues** per dataset and greys
out the rest as **"other"**; the exact split is written to `results/confounding_pca.csv`.
The two datasets use different tissue label vocabularies (DepMap is `Model.csv`
`OncotreeLineage`, CTRPv2 is the response `tissue` column), so the tissue names
differ even where the biology corresponds (e.g. Lymphoid ≈ Lymph, CNS/Brain ≈
Brain, Bowel ≈ Colon).

## Layout

```
fig01_confounding_evidence/
├── confounding_utils.py     # stat kernels (numpy/scipy/sklearn), η² from shared/eval
├── compute_confounding.py   # shared loaders → results/*.csv (nine CSVs)
├── compute_classifier_per_tissue.py  # per-tissue clf accuracy → confounding_clf_per_tissue.csv
├── plot_confounding.py      # results/*.csv → figures/fig05*.{pdf,png}
├── results/                 # populated by the compute scripts
└── figures/                 # populated by plot_confounding.py
```

η² (= SS_between / SS_total) is kept in **one definition** with the training
pipeline. `confounding_utils` imports the per-gene `tissue_eta_squared` straight
from [`../training/shared/eval.py`](../training/shared/eval.py) (the same kernel
behind the pipeline's `eta_y` and contamination@K metrics); the scalar
`eta_squared` behind the per-outcome η²(y) is re-defined locally only because
`shared.eval` inlined it into `tissue_eta_squared` and dropped its standalone
copy (2026-06-25), the arithmetic is byte-for-byte identical. A vectorised
`tissue_eta_squared_cols` (asserted equal to `tissue_eta_squared` on the real
labels) powers the shuffled-tissue null. For tissue (the headline figures) no
group-size filter is applied, so the per-gene η²(X) is bit-identical to the
pipeline's `eta_sq`.

## Reproducing the figures

Everything reproduces from the bundled data in seconds, no GPU, no `.npz`,
no training.

### 1. Compute

```bash
cd fig01_confounding_evidence
python compute_confounding.py                      # per-dataset: panel_depmap / panel_ctrpv2
# reproduce the original shared-panel run:
python compute_confounding.py --gene-list paccmann_pharma
# faster smoke test: one dataset, small target panel
python compute_confounding.py --datasets depmap --top-n-targets 25
```

By default each dataset uses its own feature panel (`--gene-list-depmap
panel_depmap`, `--gene-list-ctrpv2 panel_ctrpv2`). A single `--gene-list`
overrides both for a like-for-like comparison against the training default.

Writes nine tidy CSVs to `results/`: `confounding_expr_eta2.csv` (Arm 1 per
gene), `confounding_outcome_eta2.csv` (Arm 2 per outcome), `confounding_pca.csv`,
`confounding_tissue_corr.csv`, `confounding_confounder_compare.csv` (fig05c,
long-form) with its `confounding_confounder_compare_null.csv` per-item permuted
floor (one row per item; fig05c plots their difference), the shuffled-tissue
`confounding_{expr,outcome}_eta2_null.csv` overlays, and
`confounding_summary.csv` (a citable per-dataset summary:
classifier lift, % genes with η²(X) > 0.30, % outcomes ANOVA-significant
- both uncorrected (`frac_outcomes_sig`) and **Bonferroni-corrected**
(`frac_outcomes_sig_bonf`, with `n_tests_bonf` / `bonf_alpha`), median η² per
arm, within/across correlation + MWU p). The per-outcome CSV also carries the
Bonferroni-adjusted p (`anova_p_bonf`) and `sig` / `sig_bonf` flags. Takes ~45 s
for both datasets.

For the per-tissue swarm (`fig05d`), run the companion script, it reuses the
same `build_depmap` / `build_ctrpv2` loaders and re-runs only the linear tissue
classifier, writing one extra CSV and touching nothing else:

```bash
python compute_classifier_per_tissue.py            # → results/confounding_clf_per_tissue.csv
```

Each row is one tissue's fold-averaged accuracy (`tissue, accuracy,
accuracy_std, n_samples, n_folds`) per dataset; the per-tissue breakdown is the
same `tissue_classifier_lift` CV that feeds `confounding_summary.csv`, just split
out by held-out tissue. Defaults mirror `compute_confounding.py` (panel_depmap /
panel_ctrpv2, seed 0).

### 2. Plot

```bash
python plot_confounding.py
```

Writes `figures/fig05_confounding_evidence.{pdf,png}`,
`fig05b_confounding_structure.{pdf,png}`, and
`fig05c_confounder_comparison.{pdf,png}`. Takes a few seconds.
`fig05e_outcome_significance.{pdf,png}` (Arm-2 per-outcome Welch
p-values + Bonferroni cutoff) is written from the same CSVs, no extra step.
`fig05d_classifier_per_tissue.{pdf,png}` is also written when
`results/confounding_clf_per_tissue.csv` exists (from the companion script
above); if it is missing, fig05d is skipped with a message and the others
still render.

## Notes

- **fig05c group-size filter.** High-cardinality confounders mechanically
  explain more variance, so for the confounder *comparison* only, groups with
  fewer than `--min-group` (default 5) samples are dropped. Read the comparison
  as *which kinds of grouping* carry signal, not as a calibrated ranking between
  the three Oncotree levels.
- **fig05 panel letters.** The `a`/`b`/`c`/`d` prefixes on the four η² histogram
  titles are gated behind `SHOW_PANEL_LETTERS` (top of `plot_confounding.py`),
  currently `False` (dropped for now). Set it back to `True` to restore them, no
  other change needed.
- **Shuffled-tissue null.** Both fig05 arms (and the fig05c debiasing) compare
  against a null in which the per-cell tissue label is permuted `--n-perm`
  (default 20) times and η² recomputed: one *global* cell→tissue shuffle per
  replicate, shared by both arms, for fig05, and a *per-item* label shuffle on
  each item's own cells for fig05c. fig05 overlays it as a translucent fill; the
  real η² sits far above it (the above-null Δ>0 count is printed for the caption).
- **CCLE log scale.** `load_ccle_expression` applies `log2(x+1)`; the
  diagnostics follow the loader so they match what the models train on. Whether
  the bundled matrix is already log-scaled is a pre-existing pipeline question,
  out of scope here.
- **Source.** Ported from `Confounding/bulk_deg_analysis.py`
  (`tissue_response_anova` → `outcome_anova`, `tissue_classifier_score` →
  `tissue_classifier_lift`, `tissue_tissue_corr` → `within_across_tissue_corr`,
  `pca_by_group` → `pca_by_tissue`), dropping the scanpy/anndata scaffolding.
