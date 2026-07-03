# fig03_pathways/

Produces **`figures/fig_pathway_delta_dotplot_turned.{pdf,png}`** (with `--turned`),
the cutoff-free rank-difference GSEA dot-plot.

For each (item, gene) the within-item `|SHAP|` rank is turned into a percentile, then

    delta_g = percentile_method(g) − percentile_marginal(g)

is aggregated across items (drugs / knockout-targets). Positive `delta_g` = the
deconfounding method ranks gene `g` higher than marginal (promoted); negative =
demoted. A pre-ranked GSEA (gseapy, **10000 permutations**) on the `delta` vector shows
which gene sets each method systematically promotes vs demotes, no cutoff, no top-K.

Rows are coloured by **gene-set class**:

| colour | class | what it is |
|---|---|---|
| orange | **lineage** | Human Gene Atlas tissue signatures |
| gold | **tissue / microenvironment / cancer context** | cell-extrinsic, lineage-associated programs (adhesion, ECM, junctions, immune, secreted, cancer-type pathways) |
| grey | **cell-intrinsic** | proliferation / biosynthesis / DNA / core signaling |

Headline: deconfounding **demotes lineage + tissue-context programs** and **promotes
cell-intrinsic proliferation/biosynthesis**, consistently across DepMap and CTRPv2.

## Pipeline (3 steps, self-contained)

```bash
cd fig03_pathways
python build_set_classes.py                                        # -> gene_sets/set_classes.tsv
python compute_pathway_delta_gsea.py --preds-root /path/to/results # -> results/pathway_delta_gsea.csv
python plot_pathway_delta_dotplot.py --turned                      # -> figures/fig_pathway_delta_dotplot_turned.{pdf,png}
```

`--preds-root` must contain `{depmap,ctrpv2}_seed1_preds/*.npz` (from
`../training/gen_preds_v2.py`). `compute_*` needs `set_classes.tsv`, so run
`build_set_classes.py` first.

## Gene-set classification: official, reproducible

`build_set_classes.py` derives every set's class from official category sources
(no hand-curation), written to `gene_sets/set_classes.tsv`:

| source | file | role |
|---|---|---|
| KEGG BRITE `br08901` | `gene_sets/kegg_brite.keg` | official KEGG pathway category |
| MSigDB Hallmark process categories (Liberzon 2015, Table 1, PMC4707969) | `gene_sets/hallmark_categories.tsv` | Hallmark category |
| official MSigDB Hallmark GMT (2024.1) | `gene_sets/hallmark_msigdb_official.gmt` | gene-overlap match → systematic name |
| Human Gene Atlas | `gene_sets/human_gene_atlas.gmt` | tissue (lineage) signatures |

Class rule: lineage = HGA; intrinsic = KEGG {Metabolism, Genetic Information
Processing, Cellular Processes:Cell growth&death / Transport&catabolism, Environmental
Information Processing:Signal transduction} + Hallmark {proliferation, DNA damage,
metabolic, pathway, signaling}; everything else functional = context.

Refetch the official inputs:
```bash
curl -s "https://rest.kegg.jp/get/br:br08901" -o gene_sets/kegg_brite.keg
curl -s ".../release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt" -o gene_sets/hallmark_msigdb_official.gmt
```

## Two methodological choices (both standard, both documented)

1. **KEGG "Human Diseases" except Cancer is excluded** from the GSEA universe
   (Infectious disease, Substance dependence, Neurodegenerative, Cardiovascular,
   Endocrine/metabolic, Immune disease). These are gene-overlap grab-bags
   (Alcoholism, Amoebiasis, Coronavirus disease …), not disease-specific -
   excluding such categories is standard enrichment practice. Cancer is kept
   (it is the study context). `compute_pathway_delta_gsea.py` derives the exclusion
   from `set_classes.tsv` (category-based, a priori).

2. **10000 permutations.** gseapy's gene-set-permutation FDR is Monte-Carlo noisy;
   at 1000 permutations the significant-set membership is ~40% unstable, at 10000 it
   is stable (and the disease exclusion then changes nothing but deleting the junk).

## Interpretation caveat (for the caption)

In cell lines the gold (surface / adhesion / secreted / immune) programs **likely**
reflect *lineage-determined expression* rather than an active microenvironment (cell
lines lack stroma, immune infiltrate, other cell types); deconfounding **likely**
demotes them because their expression marks tissue identity, the confounder, rather
than because of any microenvironmental function. Supported externally: tissue-specific
genes skew cell-surface/secreted (Uhlén et al. 2015, Human Protein Atlas), and lineage
of origin dominates cancer cell-line transcriptomes (Ross et al. 2000; CCLE, Barretina
et al. 2012 / Ghandi et al. 2019), whereas proliferation/biosynthesis is broadly-
expressed housekeeping (Eisenberg & Levanon 2013). Read the gold class as "lineage-
associated expression programs," not a literal microenvironment.

## Layout

```
fig03_pathways/
├── build_set_classes.py            # official sources -> gene_sets/set_classes.tsv
├── compute_pathway_delta_gsea.py   # SHAP .npz -> results/pathway_delta_gsea.csv (10k, disease-excluded)
├── plot_pathway_delta_dotplot.py   # CSV -> figures/fig_pathway_delta_dotplot_turned.{pdf,png} (--turned, 3-class)
├── pathway_utils.py                # GMT loader, .npz importance reader
├── gene_sets/                      # gmt libraries + official category sources + set_classes.tsv
├── results/pathway_delta_gsea.csv
└── figures/fig_pathway_delta_dotplot_turned.{pdf,png}
```
