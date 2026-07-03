"""Per-(dataset, seed, item, method) mean-squared error on the raw response scale.

Companion to compute_per_item_pearson.py. Reads the out-of-fold prediction
arrays stored alongside the SHAP matrices in
    <preds-root>/<dataset>_seed<s>_preds/<item>.npz
(keys: y_true, tissue, yhat_<method>, yhat_baseline) and for every
(item, method, seed) computes three MSEs between the held-out response y and the
prediction:

  mse            = mean((y - yhat)^2)                       raw scale, all cells
  wt_mse         = mean((y_c - g_c)^2)                       per-tissue-centred
  mse_noaddback  = mean((y - (yhat - baseline))^2)           residualized only

The residualized prediction is stored on the raw scale as
    yhat_residualized = f(X_res) + yhat_baseline
so `mse` compares every method on the same raw scale. `g` is the within-tissue
predictive component (yhat - baseline for residualized, yhat otherwise); wt_mse
centres y and g by their per-tissue means (tissues with <2 cells contribute 0).
mse_noaddback strips the added-back tissue mean and only differs from `mse` for
the residualized method (it equals `mse` for the others).

Output (long, one row per dataset x seed x item x method):
    results/per_item_mse.csv
columns: dataset, seed, item, method, mse, wt_mse, mse_noaddback
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["baseline", "marginal", "residualized", "irm"]


def overall_mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))


def within_tissue_mse(y: np.ndarray, yhat: np.ndarray, T: np.ndarray) -> float:
    """mean((y - mean_t y) - (yhat - mean_t yhat))^2, per-tissue means.

    Cells in tissues with <2 members stay at 0 in both centred vectors (matching
    within_tissue_pearson), contributing nothing to the tissue-centred error.
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
    return float(np.mean((y_r - yh_r) ** 2))


def process(preds_root: Path) -> pd.DataFrame:
    rows = []
    for ds in ("depmap", "ctrpv2"):
        seed_dirs = sorted(preds_root.glob(f"{ds}_seed[0-9]_preds"))
        for sd in seed_dirs:
            seed = int(sd.name.split("seed")[1].split("_")[0])
            files = sorted(sd.glob("*.npz"))
            for f in files:
                d = np.load(f, allow_pickle=True)
                y = d["y_true"].astype(np.float64)
                T = d["tissue"]
                base = d["yhat_baseline"].astype(np.float64)
                for m in METHODS:
                    key = f"yhat_{m}"
                    if key not in d.files:
                        continue
                    yhat = d[key].astype(np.float64)
                    g = yhat - base if m == "residualized" else yhat
                    noadd = yhat - base if m == "residualized" else yhat
                    rows.append({
                        "dataset": ds,
                        "seed": seed,
                        "item": f.stem,
                        "method": m,
                        "mse": overall_mse(y, yhat),
                        "wt_mse": within_tissue_mse(y, g, T),
                        "mse_noaddback": overall_mse(y, noadd),
                    })
            print(f"  {sd.name}: {len(files)} items", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", type=Path, required=True,
                    help="dir holding {depmap,ctrpv2}_seed<s>_preds/*.npz")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results" /
                    "per_item_mse.csv")
    a = ap.parse_args()
    df = process(a.preds_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
