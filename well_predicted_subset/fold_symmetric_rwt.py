"""Fold-symmetric within-tissue r.

How are residualized predictions compared against marginal ones in Table 1? For
OVERALL r they already are: the stored `yhat_residualized` is
f(X_res) + train-fold tissue mean (orchestrator.py -> reconstruct_raw_yhat), so both
methods are correlated against the same raw y. That part needs a sentence in 2.4, not
a new number.

The asymmetry is in r_wt. compute_per_item_pearson.py evaluates r_wt for residualized
on the deconfounded output f(X_res) = yhat - yhat_baseline, and for every other method
on the raw yhat. The justification is real: yhat_baseline is a LEAVE-ONE-FOLD-OUT
per-tissue mean, so it carries a strong negative within-tissue correlation
(r_wt ~ -0.28 / -0.38) and, because it differs between folds inside the same tissue,
centring per tissue cannot remove it. But the consequence is that residualization's r_wt
lead over marginal exists only under that method-specific convention: on the
reconstructed prediction its r_wt is 0.157 / 0.061, below marginal's 0.204 / 0.175.

This script removes the need for the convention. Centring y and yhat within
(tissue x fold) groups instead of within tissue kills the leave-out baseline term for
EVERY method identically, so no method gets special handling:

    r_wt_foldsym = corr(y - mean_{t,f} y,  yhat - mean_{t,f} yhat)

The folds are recoverable without retraining -- orchestrator.py uses
KFold(n_splits=5, shuffle=True, random_state=seed) on the cell axis, which depends only
on (n_cells, seed) -- so this runs off the shipped npz bundle.

Two built-in validity checks (printed for inspection, not asserted -- check 1's pass
condition is a NaN, which no useful assert expresses):
  1. the tissue-mean baseline must go to r_wt ~ 0 (it predicts nothing within a
     tissue x fold cell). If the fold reconstruction were wrong, it would not.
  2. for residualized, the raw and baseline-stripped predictions must now agree,
     since the term that separated them has been removed.

Outputs:
    results/fold_symmetric_rwt.csv          per (dataset, seed, item, method)
    results/fold_symmetric_summary.csv      Table-1-style aggregate, all conventions
    results/fold_symmetric_paired.csv       paired vs marginal, per convention

    python fold_symmetric_rwt.py --preds-root ../results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "training"))
from shared.preds_io import load_preds  # noqa: E402

METHODS = ["baseline", "marginal", "residualized", "irm", "dann", "adae"]
N_FOLDS = 5          # orchestrator.run_cv default, used by gen_preds_v2.py
MIN_GROUP = 2        # groups with <2 cells carry no deviation (as in the paper's metric)


def fold_labels(n: int, seed: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Which test fold each cell fell in. Mirrors orchestrator.py:199 exactly."""
    lab = np.empty(n, dtype=np.int16)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(kf.split(np.arange(n))):
        lab[te] = i
    return lab


def grouped_pearson(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray) -> tuple:
    """corr(y, yhat) after centring both within `groups`. Returns (r, n_effective).

    Cells in groups smaller than MIN_GROUP stay at 0 in both centred vectors and so
    contribute nothing -- the same convention as
    table_performance.compute_per_item_pearson.within_tissue_pearson.
    """
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    yr = np.zeros_like(y)
    hr = np.zeros_like(yhat)
    n_eff = 0
    # np.unique on a structured key is slower than factorising once; groups is
    # already an integer code from the caller.
    for g in np.unique(groups):
        m = groups == g
        k = int(m.sum())
        if k < MIN_GROUP:
            continue
        yr[m] = y[m] - y[m].mean()
        hr[m] = yhat[m] - yhat[m].mean()
        n_eff += k
    if yr.std() < 1e-12 or hr.std() < 1e-12:
        return float("nan"), n_eff
    return float(np.corrcoef(yr, hr)[0, 1]), n_eff


def process(preds_root: Path, datasets=("depmap", "ctrpv2")) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for sd in sorted(preds_root.glob(f"{ds}_seed[0-9]_preds")):
            seed = int(sd.name.split("seed")[1].split("_")[0])
            files = sorted(sd.glob("*.npz"))
            for f in files:
                d = load_preds(f)
                y = d["y_true"].astype(np.float64)
                T = d["tissue"]
                n = len(y)
                base = d["yhat_baseline"].astype(np.float64)
                fold = fold_labels(n, seed)
                # integer codes for the two groupings
                t_code = pd.factorize(T)[0]
                tf_code = t_code * N_FOLDS + fold
                for m in METHODS:
                    key = f"yhat_{m}"
                    if key not in d.files:
                        continue
                    yhat = d[key].astype(np.float64)
                    g = yhat - base if m == "residualized" else yhat
                    r_t, _ = grouped_pearson(y, g, t_code)          # paper's r_wt
                    r_t_rec, _ = grouped_pearson(y, yhat, t_code)   # raw prediction
                    r_tf, n_eff = grouped_pearson(y, yhat, tf_code)  # NEUTRAL
                    r_tf_str, _ = grouped_pearson(y, g, tf_code)     # check 2
                    rows.append({
                        "dataset": ds, "seed": seed, "item": f.stem, "method": m,
                        "n_cells": n, "n_tissues": int(len(np.unique(T))),
                        "n_eff_foldsym": n_eff,
                        "wt_tissue": r_t,
                        "wt_tissue_recon": r_t_rec,
                        "wt_foldsym": r_tf,
                        "wt_foldsym_stripped": r_tf_str,
                    })
            print(f"  {sd.name}: {len(files)} items", flush=True)
    return pd.DataFrame(rows)


COLS = ["wt_tissue", "wt_tissue_recon", "wt_foldsym", "wt_foldsym_stripped"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean per item, then mean/sem/sd across items -- as make_table.py does."""
    per_item = (df.groupby(["dataset", "item", "method"])[COLS + ["n_eff_foldsym",
                                                                 "n_cells"]]
                  .mean().reset_index())
    out = []
    for (ds, m), g in per_item.groupby(["dataset", "method"]):
        rec = {"dataset": ds, "method": m, "n_items": len(g),
               "frac_cells_used": (g.n_eff_foldsym / g.n_cells).mean()}
        for c in COLS:
            v = g[c].dropna()
            rec[f"{c}_mean"] = v.mean()
            rec[f"{c}_sem"] = v.std() / np.sqrt(max(1, len(v)))
            rec[f"{c}_sd"] = v.std()
        out.append(rec)
    return pd.DataFrame(out)


def paired_vs_marginal(df: pd.DataFrame) -> pd.DataFrame:
    """Per-item paired test of each method against marginal, per convention."""
    per_item = (df.groupby(["dataset", "item", "method"])[COLS]
                  .mean().reset_index())
    rows = []
    for ds, g in per_item.groupby("dataset"):
        for c in COLS:
            w = g.pivot(index="item", columns="method", values=c)
            if "marginal" not in w:
                continue
            for m in METHODS:
                if m == "marginal" or m not in w:
                    continue
                ok = w["marginal"].notna() & w[m].notna()
                if ok.sum() < 6:
                    continue
                d = (w[m] - w["marginal"])[ok]
                rows.append({
                    "dataset": ds, "convention": c, "method": m, "n_items": int(ok.sum()),
                    "marginal_mean": w["marginal"][ok].mean(),
                    "method_mean": w[m][ok].mean(),
                    "delta_mean": d.mean(),
                    "frac_items_better": float((d > 0).mean()),
                    "wilcoxon_p": wilcoxon(w[m][ok], w["marginal"][ok]).pvalue,
                })
    out = pd.DataFrame(rows)
    # BH across the whole method x dataset family within one convention. Without
    # this, "residualized beats marginal at p=0.028" is one of 8 paired tests and
    # does not survive; the q-value is what the claim has to rest on.
    for conv, idx in out.groupby("convention").groups.items():
        _, q, _, _ = multipletests(out.loc[idx, "wilcoxon_p"], alpha=0.05,
                                   method="fdr_bh")
        out.loc[idx, "wilcoxon_q_bh"] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", type=Path, default=REPO / "results")
    ap.add_argument("--datasets", nargs="+", default=["depmap", "ctrpv2"])
    a = ap.parse_args()

    res = HERE / "results"
    res.mkdir(parents=True, exist_ok=True)
    df = process(a.preds_root, a.datasets)
    df.to_csv(res / "fold_symmetric_rwt.csv", index=False)
    agg = aggregate(df)
    paired = paired_vs_marginal(df)
    agg.to_csv(res / "fold_symmetric_summary.csv", index=False)
    paired.to_csv(res / "fold_symmetric_paired.csv", index=False)

    pd.set_option("display.width", 220)
    order = [m for m in METHODS]
    agg["method"] = pd.Categorical(agg.method, order, ordered=True)
    print("\n=== within-tissue r under each convention (mean over items) ===")
    print(agg.sort_values(["dataset", "method"])[
        ["dataset", "method", "n_items", "frac_cells_used",
         "wt_tissue_mean", "wt_tissue_recon_mean", "wt_foldsym_mean",
         "wt_foldsym_stripped_mean"]].to_string(index=False,
                                                float_format=lambda v: f"{v:.3f}"))

    print("\n=== validity check 1: baseline must carry no fold-symmetric signal ===")
    for ds in a.datasets:
        b = agg[(agg.dataset == ds) & (agg.method == "baseline")]
        if len(b):
            v = b.wt_foldsym_mean.iloc[0]
            # The baseline IS the per-(tissue, fold) mean of y, so centring within
            # those groups zeroes it exactly: the centred vector has zero variance and
            # the correlation is undefined rather than merely small. NaN here is the
            # strongest possible pass, not a failure.
            verdict = ("exactly 0 (undefined corr: centred vector is all-zero)"
                       if np.isnan(v) else f"{v:+.3f}")
            print(f"  {ds:7s} baseline: per-tissue {b.wt_tissue_mean.iloc[0]:+.3f}"
                  f"  ->  fold-symmetric {verdict}")

    print("\n=== validity check 2: residualized raw vs stripped must now agree ===")
    for ds in a.datasets:
        r = agg[(agg.dataset == ds) & (agg.method == "residualized")]
        if len(r):
            print(f"  {ds:7s} per-tissue  raw {r.wt_tissue_recon_mean.iloc[0]:.3f} vs "
                  f"stripped {r.wt_tissue_mean.iloc[0]:.3f}"
                  f"   (gap {abs(r.wt_tissue_recon_mean.iloc[0]-r.wt_tissue_mean.iloc[0]):.3f})")
            print(f"  {ds:7s} fold-sym    raw {r.wt_foldsym_mean.iloc[0]:.3f} vs "
                  f"stripped {r.wt_foldsym_stripped_mean.iloc[0]:.3f}"
                  f"   (gap {abs(r.wt_foldsym_mean.iloc[0]-r.wt_foldsym_stripped_mean.iloc[0]):.3f})")

    print("\n=== does residualization still beat marginal? (paired over items) ===")
    p = paired[paired.method.isin(["residualized", "irm", "dann", "adae"])]
    p = p[p.convention.isin(["wt_tissue", "wt_tissue_recon", "wt_foldsym"])]
    print(p.sort_values(["dataset", "convention", "method"]).to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {res/'fold_symmetric_rwt.csv'}, {res/'fold_symmetric_summary.csv'}, "
          f"{res/'fold_symmetric_paired.csv'}")


if __name__ == "__main__":
    main()
