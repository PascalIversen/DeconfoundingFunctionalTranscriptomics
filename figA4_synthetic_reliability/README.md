# figA4_synthetic_reliability/

Synthetic ground-truth reliability figure (supplementary `suppfig:synth_data`):
marginal SHAP is a lottery; residualized (FWL) and within-tissue SHAP keep the
top of the ranking free of confounders by construction.

```
figA4_synthetic_reliability/
├── plot_reliability.py        # builds fig_reliability_top{10,15,20,30,50}.pdf
├── compute_npower_mixed.py    # (slow, GPU) regenerates the results/ CSVs
├── results/
│   ├── npower_mixed.csv               # 8 base seeds, full N_GRID
│   ├── npower_mixed_extra_n1000.csv   # 24 extra seeds at n=1000 (same DGP)
│   └── npower_mixed_summary.csv       # per-run within-r, eta^2(y), contam@50
└── figures/                   # fig_reliability_top*.{pdf,png}
```

## Reproduce

Plot only (fast, no GPU), uses the shipped CSVs:

```bash
python plot_reliability.py
```

Regenerate the CSVs from scratch (GPU, hours), imports the bundled pipeline
under `../training/shared`, so this folder is self-contained:

```bash
python compute_npower_mixed.py --seed-start 1  --replicates 8    # -> npower_mixed.csv
python compute_npower_mixed.py --seed-start 9  --replicates 24   # -> npower_mixed_seed9.csv
# rename/concat the seed9 run's n=1000 rows into npower_mixed_extra_n1000.csv
```

The paper uses `fig_reliability_top10.pdf`. Data source of truth is the 32-seed
n=1000 slice (8 base + 24 extra).
```

## Mixed-class breakdown (`mixed_class.py`, Figure A5 / `suppfig:mixed_class`)

Report performance separately for the mixed class, where a gene is causal but also
tissue-amplified: that is the case where a deconfounding method could plausibly
discard true signal, and the top-10 composition alone does not resolve it. The right
test is not absolute recovery but whether mixed genes are demoted *relative to
pure-causal genes of the same true effect size*. Any differential is discarded signal.

Two DGP variants:

- **`v1`** (the published panel) — mixed genes and confounders differ in tissue
  amplitude/noise, giving a tissue-based correction a cue to tell them apart.
- **`v2`** — mixed genes and confounders are built with the *same* tissue amplitude
  and noise (η² ≈ 0.8 for both), removing that cue, for a harder test of whether
  the correction can still distinguish them.

Reporting `v1` and `v2` together shows whether the result is conditional on the DGP,
not a blanket claim. No retraining or GPU is needed — the shipped `npower_mixed*.csv`
already record `gene_class`, the true `effect`, and the SHAP `rank` per
(seed, n, method, gene).

```bash
python mixed_class.py --dgp v1
python mixed_class.py --dgp v2
```

Figures: `figures/fig_mixed_class_{v1,v2}.pdf` (a: rank shift by class; b:
mixed−causal retention@K; c: effect-matched differential by quartile). Tables:
`results/mixed_{class_ranks,vs_causal_tests,effect_matched}_{v1,v2}.csv`.
