"""Per-(dataset, seed, item, method) overall and within-tissue Pearson r.

Reads the out-of-fold prediction arrays stored alongside the SHAP matrices in
    <preds-root>/<dataset>_seed<s>_preds/<item>.npz
(keys: y_true, tissue, yhat_<method>), and for every (item, method, seed)
computes two correlations between the held-out response y and the prediction
yhat:

  overall      r       = corr(y, yhat)                       across all cells
  within-tissue r_wt    = corr(y - mean_t y, g - mean_t g)        per-tissue means

The within-tissue r is the partial correlation of y and the prediction given
tissue: it strips the between-tissue (lineage) component that any per-tissue-mean
predictor would also capture, leaving only whether the model ranks cells
correctly inside a tissue. Tissues with <2 cells carry no within-tissue
deviation and are dropped (their cells contribute 0 to both centered vectors).

  g = the method's *within-tissue predictive component*.
For marginal / IRM / within-tissue this is just yhat (a single model output).
For the RESIDUALIZED method, the raw prediction is reconstructed as
    yhat_residualized = f(X_res) + yhat_baseline
where yhat_baseline is the leave-one-fold-out per-tissue mean added back only to
put predictions on the raw scale (needed for overall r). That baseline term has
a strongly *negative* within-tissue r (the leave-out anti-correlation, worst in
small tissues) and varies by fold within a tissue, so per-tissue centring cannot
remove it -- it contaminates r_wt of the reconstructed vector. The within-tissue
metric is meant to exclude exactly this tissue-mean term, so for residualized we
evaluate the deconfounded model output g = yhat_residualized - yhat_baseline =
f(X_res). We also keep the contaminated value (wt_pearson_recon) for transparency.

We score the actual predictors, not the attribution stage, so within-tissue
SHAP is omitted: it re-uses the marginal model (yhat_within_tissue ==
yhat_marginal, asserted below) and therefore has identical predictive r. We
keep the tissue-mean `baseline` as a reference: it predicts the per-tissue mean
only, so its overall r is "free" tissue signal while its within-tissue r is ~0.

Output (long, one row per dataset x seed x item x method):
    results/per_item_pearson.csv
columns: dataset, seed, item, method, n_cells, n_tissues,
         pearson, wt_pearson, wt_pearson_recon
(wt_pearson == wt_pearson_recon for every method except residualized.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Predictors that have their own out-of-fold predictions. `within_tissue` is an
# attribution-stage method on top of the marginal model and is excluded here
# (verified identical below). `between_tissue` is a diagnostic, not a method.
METHODS = ["baseline", "marginal", "residualized", "irm"]


def overall_pearson(y: np.ndarray, yhat: np.ndarray) -> float:
    if y.std() < 1e-12 or yhat.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0, 1])


def within_tissue_pearson(y: np.ndarray, yhat: np.ndarray,
                          T: np.ndarray) -> float:
    """corr(y - mean_t y, yhat - mean_t yhat), per-tissue means.

    Cells in tissues with <2 members stay at 0 in both centered vectors, so
    they contribute nothing to the correlation (matches _compute_within_tissue_r).
    """
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    y_r = np.zeros_like(y)
    yh_r = np.zeros_like(yhat)
    for t in np.unique(T):
        m = T == t
        if m.sum() < 2:
            continue
        y_r[m] = y[m] - y[m].mean()
        yh_r[m] = yhat[m] - yhat[m].mean()
    if y_r.std() < 1e-12 or yh_r.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_r, yh_r)[0, 1])


def process(preds_root: Path) -> pd.DataFrame:
    rows = []
    for ds in ("depmap", "ctrpv2"):
        label = "target" if ds == "depmap" else "drug"
        seed_dirs = sorted(preds_root.glob(f"{ds}_seed[0-9]_preds"))
        for sd in seed_dirs:
            seed = int(sd.name.split("seed")[1].split("_")[0])
            files = sorted(sd.glob("*.npz"))
            for f in files:
                d = np.load(f, allow_pickle=True)
                y = d["y_true"].astype(np.float64)
                T = d["tissue"]
                n_t = int(len(np.unique(T)))
                base = d["yhat_baseline"].astype(np.float64)
                # sanity: within-tissue SHAP shares the marginal predictor
                if "yhat_within_tissue" in d.files:
                    assert np.allclose(d["yhat_within_tissue"],
                                       d["yhat_marginal"]), f"{f}: within != marginal"
                for m in METHODS:
                    key = f"yhat_{m}"
                    if key not in d.files:
                        continue
                    yhat = d[key].astype(np.float64)
                    # within-tissue predictive component: strip the added-back
                    # tissue-mean baseline from the residualized prediction.
                    g = yhat - base if m == "residualized" else yhat
                    rows.append({
                        "dataset": ds,
                        "seed": seed,
                        "item": f.stem,
                        "method": m,
                        "n_cells": int(len(y)),
                        "n_tissues": n_t,
                        "pearson": overall_pearson(y, yhat),
                        "wt_pearson": within_tissue_pearson(y, g, T),
                        "wt_pearson_recon": within_tissue_pearson(y, yhat, T),
                    })
            print(f"  {sd.name}: {len(files)} items", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", type=Path, required=True,
                    help="dir holding {depmap,ctrpv2}_seed<s>_preds/*.npz")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results" /
                    "per_item_pearson.csv")
    a = ap.parse_args()
    df = process(a.preds_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    n_items = df.groupby("dataset")["item"].nunique().to_dict()
    n_seeds = df.groupby("dataset")["seed"].nunique().to_dict()
    print(f"wrote {a.out} ({len(df)} rows; items={n_items}, seeds={n_seeds})")


if __name__ == "__main__":
    main()
