# training/: shared library and the prediction pipeline (Layer 3)

You do **not** need to run this to reproduce the paper: the predictions are
provided on Zenodo. This directory regenerates them from raw data.

## Shared library (`shared/`)

Imported by every figure/table script via `sys.path.insert(0, HERE.parent/"training")`:

- `data.py`: loaders (CCLE/DepMap expression, DepMap CRISPR, CTRPv2 response,
  tissue labels, gene panels). Auto-detects `../data/`. Default panels:
  `panel_depmap` (1071 genes), `panel_ctrpv2` (1109 genes).
- `models.py`, `train.py`: MLP regressor + training loop (Adam, ReduceLROnPlateau,
  early stopping).
- `deconfound.py`: FWL residualization, IRM penalty.
- `shap_utils.py`: `gradient_shap` (global background) and `within_tissue_shap`
  (per-tissue background).
- `orchestrator.py`: `run_cv`: 5-fold CV × 4 methods + baseline + between-tissue,
  with SHAP on each held-out fold.
- `eval.py`, `plot.py`, `synthetic.py`: metrics, plotting style, synthetic DGP.

## Regenerate predictions

```bash
# single node
python gen_preds_v2.py --dataset depmap --gene-list landmark_genes \
    --items-file items_depmap.txt --targets-file targets_depmap.txt \
    --union-targets --seed 1 --out-dir ../results --nproc <cores>

# multi-node (splits items_<ds>.txt across worker nodes over NFS)
bash dispatch.sh <dataset> <seed>
```

Outputs `../results/<dataset>_seed<s>_preds/<item>.npz` (per-cell `y_true`,
`tissue`, `yhat_<method>`, `shap_<method>`) and `../results/<dataset>_seed<s>.csv`.
The gene panels (`panel_depmap.csv`, `panel_ctrpv2.csv`) and item/target lists are
built by `build_item_lists.py` / `input_panel_gene_definition.py`.
