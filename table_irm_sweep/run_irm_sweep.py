"""IRM penalty-weight (lambda) sweep on real data.

Pre-empts the referee point "IRM just needs tuning". For a subset of items
(drugs / knockout targets) we retrain the IRM model across a grid of penalty
weights lambda and record the deconfounding metrics the paper judges methods by:

  contam@50     fraction of the top-50 attributed genes that are tissue markers
                (lower = better deconfounding)
  attr_eta2     mean over genes of the tissue eta^2 of the per-cell SHAP
                (lower = less tissue signal in the attribution)
  within_r      within-tissue Pearson r of the prediction (higher = better)
  global_r      overall Pearson r (higher = better)

Marginal, Residualized (FWL) and Within-tissue SHAP do not depend on lambda;
they are recorded once per item (method='marginal'/'residualized'/
'within_tissue', lambda=NaN) as the reference lines. IRM is recorded at every
lambda. The IRM annealing (10-epoch warm start) and full-fold penalty are kept
at the paper's settings; only lambda is swept.

Reuses the training pipeline (shared.orchestrator.run_cv) verbatim, so the
numbers are directly comparable to the main results.

Usage (one dataset per call):
    python run_irm_sweep.py --dataset depmap --n-items 20 --nproc 20
    python run_irm_sweep.py --dataset ctrpv2 --n-items 20 --nproc 20
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "training"
sys.path.insert(0, str(TRAIN))

from shared.orchestrator import run_cv                                          # noqa: E402
from shared.eval import contamination_at_k, tissue_eta_squared                  # noqa: E402
from shared.data import (load_depmap_crispr, load_ccle_expression,              # noqa: E402
                         load_ctrpv2_response, build_ctrpv2_per_drug,
                         load_drug_target_map)

LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
REF_METHODS = ["marginal", "residualized", "within_tissue"]
G = {}


# ---- metrics ---------------------------------------------------------------
def attr_eta2_mean(shap: np.ndarray, T: np.ndarray) -> float:
    """Mean over genes of the between-tissue fraction of per-cell SHAP variance."""
    valid = np.isfinite(shap).all(axis=1)
    shap, T = shap[valid], T[valid]
    grand = shap.mean(0)
    ss_tot = ((shap - grand) ** 2).sum(0)
    ss_btw = np.zeros(shap.shape[1])
    for t in np.unique(T):
        m = T == t
        ss_btw += int(m.sum()) * (shap[m].mean(0) - grand) ** 2
    eta = np.where(ss_tot > 1e-12, ss_btw / ss_tot, np.nan)
    return float(np.nanmean(eta))


def within_tissue_r(y: np.ndarray, g: np.ndarray, T: np.ndarray) -> float:
    y = y.astype(np.float64); g = g.astype(np.float64)
    yr = np.zeros_like(y); gr = np.zeros_like(g)
    for t in np.unique(T):
        m = T == t
        if m.sum() < 2:
            continue
        yr[m] = y[m] - y[m].mean(); gr[m] = g[m] - g[m].mean()
    if yr.std() < 1e-12 or gr.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(yr, gr)[0, 1])


def metrics(method, lam, res, y, T, eta_set):
    md = res[method]
    # within-tissue predictive component: strip the added-back tissue mean for FWL
    yhat = md["yhat_raw"]
    g = yhat - res["baseline"]["yhat_raw"] if method == "residualized" else yhat
    return dict(
        dataset=G["dataset"], item=G["cur_item"], method=method,
        lam=lam, n_cells=int(len(y)), n_tissues=int(len(np.unique(T))),
        contam_50=round(contamination_at_k(md["importance"], eta_set, k=50), 4),
        attr_eta2=round(attr_eta2_mean(md["shap_full"], T), 4),
        within_r=round(within_tissue_r(y, g, T), 4),
        global_r=round(md["pearson_r"], 4),
    )


# ---- per-item worker -------------------------------------------------------
def _load_item(item):
    if G["dataset"] == "depmap":
        y_full = G["crispr"][item].dropna()
        common = G["expr"].index.intersection(y_full.index)
        return (G["expr"].loc[common].values, y_full.loc[common].values,
                G["tissue"].loc[common].values)
    X, y, T = build_ctrpv2_per_drug(item, G["resp"], G["expr"])
    return X, y, T


def work(item):
    import torch; torch.set_num_threads(1)
    G["cur_item"] = item
    try:
        X, y, T = _load_item(item)
    except Exception:
        return None
    if len(y) < 100 or len(np.unique(T)) < 8:
        return None
    GC, ETA, SEED = G["gene_cols"], G["eta"], G["seed"]
    rows = []
    for i, lam in enumerate(LAMBDAS):
        res = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                     batch_size=64, shap_background=100, irm_lambda=lam,
                     seed=SEED)
        rows.append(metrics("irm", lam, res, y, T, ETA))
        if i == 0:                          # lambda-independent reference lines
            for m in REF_METHODS:
                r = metrics(m, np.nan, res, y, T, ETA)
                rows.append(r)
    return rows


# ---- data setup (mirrors gen_preds_v2) -------------------------------------
def _read_items(path: Path) -> list:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def setup(dataset, gene_list):
    extra = _read_items(TRAIN / f"targets_{dataset}.txt")
    if dataset == "depmap":
        expr, crispr, tissue = load_depmap_crispr(gene_list=gene_list,
                                                  extra_genes=extra)
        G.update(expr=expr, crispr=crispr, tissue=tissue,
                 gene_cols=list(expr.columns),
                 eta=tissue_eta_squared(expr, tissue.values))
    else:
        resp = load_ctrpv2_response(min_cells_per_drug=100, min_tissues_per_drug=8)
        expr = load_ccle_expression(gene_list=gene_list, extra_genes=extra)
        dt = load_drug_target_map()
        union = resp[resp["cellosaurus_id"].isin(expr.index)].drop_duplicates(
            "cellosaurus_id")
        Xu = expr.loc[union["cellosaurus_id"]]
        Tu = union.set_index("cellosaurus_id").loc[Xu.index, "tissue"].values
        G.update(expr=expr, resp=resp, dt=dt, gene_cols=list(expr.columns),
                 eta=tissue_eta_squared(Xu, Tu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["depmap", "ctrpv2"], required=True)
    ap.add_argument("--gene-list", default="landmark_genes")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nproc", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0,
                    help="this process handles subset[shard::nshards] "
                         "(for splitting items across GPUs)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    G["dataset"] = a.dataset
    G["seed"] = a.seed
    t0 = time.time()
    setup(a.dataset, a.gene_list)

    items = _read_items(TRAIN / f"items_{a.dataset}.txt")
    # evenly-spaced deterministic subset across the full item list
    idx = np.linspace(0, len(items) - 1, a.n_items).round().astype(int)
    subset = [items[i] for i in sorted(set(idx))]
    subset = subset[a.shard::a.nshards]          # this GPU's slice
    print(f"[{a.dataset} shard {a.shard}/{a.nshards}] {len(subset)} items x "
          f"{len(LAMBDAS)} lambdas; {len(G['gene_cols'])} genes; "
          f"nproc={a.nproc}; setup {time.time()-t0:.0f}s", flush=True)

    all_rows, done = [], 0
    ctx = mp.get_context("fork")
    with ctx.Pool(a.nproc) as pool:
        for r in pool.imap_unordered(work, subset, chunksize=1):
            done += 1
            if r:
                all_rows.extend(r)
            print(f"  {done}/{len(subset)} ({time.time()-t0:.0f}s)", flush=True)

    suffix = "" if a.nshards == 1 else f"_shard{a.shard}"
    out = a.out or HERE / "results" / f"irm_sweep_{a.dataset}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"[{a.dataset}] DONE {done} items in {time.time()-t0:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
