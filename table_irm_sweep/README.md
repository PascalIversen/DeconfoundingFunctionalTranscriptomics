# Model-stage penalty-weight sweeps (Table tab:adv_sweep / tab:adv_diagnostics)

Sweeps the penalty weight for all three model-stage methods (IRM, DANN,
AD-AE) on both datasets (contamination@50, η²_attr, and within-tissue r),
all on the same 20-item-per-dataset subset for a symmetric comparison.

## Run

```bash
python run_irm_sweep.py --dataset depmap --n-items 20   # trains the IRM sweep -> results/irm_sweep_{ds}.csv  (GPU)
python run_irm_sweep.py --dataset ctrpv2 --n-items 20

python run_adv_sweep.py        # trains the DANN/AD-AE sweep -> results/adv_sweep_{ds}.csv  (GPU, same 20-item subset)
python make_table_adv.py       # -> adv_sweep_table.tex, adv_diagnostics_table.tex (tab:adv_sweep, tab:adv_diagnostics)
```

`make_table_adv.py` reads the shipped `results/{adv_sweep,irm_sweep}_{depmap,ctrpv2}*.csv`,
so both tables reproduce without rerunning either sweep. `run_irm_sweep.py`
and `run_adv_sweep.py` regenerate those CSVs from raw data; `run_adv_sweep.py`
imports its lambda grid and metric functions directly from `run_irm_sweep.py`
rather than duplicating them, so the two sweeps cannot drift apart, and
selects the identical deterministic item subset.
