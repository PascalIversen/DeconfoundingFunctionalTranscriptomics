# figA4_synthetic_reliability/

Synthetic ground-truth reliability figure (supplementary `suppfig:synth_data`):
marginal SHAP is a lottery; residualized (FWL) and within-tissue SHAP keep the
top of the ranking free of confounders by construction.

```
figA4_synthetic_reliability/
├── plot_reliability.py        # builds fig_reliability_top{10,15,20,30,50}.pdf
├── compute_npower_mixed.py    # (slow, GPU) regenerates the results/ CSVs
├── caption.tex                # augmented supplementary caption
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
