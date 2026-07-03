# Table A3: IRM penalty-weight (λ) sweep

Shows that no λ rescues IRM on either dataset (contamination@50, η²_attr, and
within-tissue r on a 16-item subset per dataset).

## Run

```bash
python run_irm_sweep.py        # trains the λ sweep -> results/irm_sweep_{ds}_shard*.csv  (GPU)
python make_table_irm.py       # -> irm_sweep_table.tex   (Table A3)
python plot_irm_sweep.py       # -> figures/fig_irm_sweep.pdf  (supporting, not in the paper)
```

`make_table_irm.py` and `plot_irm_sweep.py` read the shipped
`results/irm_sweep_{depmap,ctrpv2}_shard*.csv`, so the table reproduces without
rerunning the sweep. `run_irm_sweep.py` regenerates those CSVs from raw data.
