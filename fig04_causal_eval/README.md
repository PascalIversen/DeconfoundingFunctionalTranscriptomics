# Fig 4: Biological evaluation of attribution specificity

Assembles the two-panel Figure 4 from the outputs of the two sibling directories:

- **panel a** (DepMap self + functional-partner recovery, ΔAUROC vs Marginal) ←
  `../fig04a_selfrank_partner/data/`
- **panel b** (CTRPv2 per-tissue PAM50 specificity) ←
  `../fig04b_breast_subtypes/results/subtype_tissue_specificity.csv`

## Run

```bash
python make_joined_causal_eval_vertical_nogap.py   # -> figures/fig_joined_causal_eval_vertical_nogap.pdf
```

Run the two sibling compute steps first if their inputs are not present (see their
READMEs). This script does no computation of its own, it only lays out and
BH-corrects the precomputed panels.
