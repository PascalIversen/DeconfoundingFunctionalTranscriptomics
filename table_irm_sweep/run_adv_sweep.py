"""Penalty-weight (lambda) sweep for the adversarial model-stage baselines.

The companion of `run_irm_sweep.py`, and deliberately its mirror image: does
DANN or AD-AE's weak deconfounding just need more tuning, the same question
`run_irm_sweep.py` asks of IRM? Both baselines are run at their source
papers' default (lambda = 1) and across
the *same* grid, scored on the *same* four metrics, over the *same* 20-item
subset, so the three model-stage methods get identical treatment:

  contam@50     fraction of the top-50 attributed genes that are tissue
                markers (lower = better deconfounding)
  attr_eta2     mean over genes of the tissue eta^2 of the per-cell SHAP
                (lower = less tissue signal in the attribution)
  within_r      within-tissue Pearson r of the prediction (higher = better)
  global_r      overall Pearson r (higher = better)

Two lambda=0 rows do double duty as controls:
  * DANN at lambda_max=0 reduces to the marginal model: with the gradient
    reversal layer scaled to zero, the domain head receives gradients but
    contributes none to the feature extractor. Given an identical torch
    seed the two are bit-identical (asserted in the implementation tests);
    inside this sweep each method is seeded per (seed, fold, method) so
    that a run does not depend on which other methods ran alongside it, so
    the lambda=0 row is an independent draw of the same estimator rather
    than a byte-for-byte copy of the published marginal predictions. Either
    way it isolates the DANN training protocol from the adversarial effect.
  * AD-AE at lambda=0 is a plain autoencoder + downstream head, isolating
    the cost of routing the prediction through an unsupervised embedding.

The metric definitions and the lambda grid are imported from
`run_irm_sweep` rather than copied, so the two sweeps cannot drift apart.

The lambda-independent reference methods (marginal, residualized,
within-tissue) are NOT recomputed by default: they are already in the IRM
sweep's output for the identical items, seed and folds. Pass --with-refs to
regenerate them anyway.

Usage (one dataset per call):
    python run_adv_sweep.py --dataset depmap --n-items 20 --nproc 20
    python run_adv_sweep.py --dataset ctrpv2 --n-items 20 --nproc 20
    # faithfulness sensitivity: Ganin's own SGD + mu_p schedule
    python run_adv_sweep.py --dataset depmap --dann-optimizer ganin \
        --methods dann --out results/adv_sweep_depmap_ganinopt.csv
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
sys.path.insert(0, str(HERE))

from shared.orchestrator import run_cv                                          # noqa: E402
from shared.eval import contamination_at_k, tissue_eta_squared                  # noqa: E402
from shared.data import (load_depmap_crispr, load_ccle_expression,              # noqa: E402
                         load_ctrpv2_response, build_ctrpv2_per_drug,
                         load_drug_target_map)
# identical metric definitions and lambda grid as the IRM sweep
from run_irm_sweep import (LAMBDAS, attr_eta2_mean, within_tissue_r,            # noqa: E402
                           REF_METHODS)

ADV_METHODS = ["dann", "adae"]
G = {}


def metrics(method, lam, res, y, T, eta_set):
    md = res[method]
    yhat = md["yhat_raw"]
    g = yhat - res["baseline"]["yhat_raw"] if method == "residualized" else yhat
    row = dict(
        dataset=G["dataset"], item=G["cur_item"], method=method,
        lam=lam, n_cells=int(len(y)), n_tissues=int(len(np.unique(T))),
        contam_50=round(contamination_at_k(md["importance"], eta_set, k=50), 4),
        attr_eta2=round(attr_eta2_mean(md["shap_full"], T), 4),
        within_r=round(within_tissue_r(y, g, T), 4),
        global_r=round(md["pearson_r"], 4),
    )
    # fold-averaged adversary diagnostics: did the adversary actually get
    # beaten at this lambda, and did AD-AE diverge?
    for d in (md.get("diagnostics") or [])[:1]:
        keys = [k for k in d if k != "fold"]
        for k in keys:
            row[k] = round(float(np.nanmean(
                [dd[k] for dd in md["diagnostics"]])), 4)
    return row


def _load_item(item):
    if G["dataset"] == "depmap":
        y_full = G["crispr"][item].dropna()
        common = G["expr"].index.intersection(y_full.index)
        return (G["expr"].loc[common].values, y_full.loc[common].values,
                G["tissue"].loc[common].values,
                [str(c) for c in common])
    X, y, T, ids = build_ctrpv2_per_drug(item, G["resp"], G["expr"],
                                          return_ids=True)
    return X, y, T, ids


def work(item):
    import torch; torch.set_num_threads(1)
    G["cur_item"] = item
    try:
        X, y, T, ids = _load_item(item)
    except Exception:
        return None
    if len(y) < 100 or len(np.unique(T)) < 8:
        return None
    GC, ETA, SEED, CFG = G["gene_cols"], G["eta"], G["seed"], G["cfg"]
    rows = []
    for i, lam in enumerate(LAMBDAS):
        res = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                     batch_size=64, shap_background=100, seed=SEED,
                     methods=CFG["methods"], sample_ids=ids,
                     dann_kwargs={"lambda_max": lam,
                                  "optimizer": CFG["dann_optimizer"]},
                     adae_kwargs={"lambda_adv": lam,
                                  "head_hidden": CFG["adae_head_hidden"]},
                     adae_cache_dir=CFG["adae_cache_dir"])
        for m in CFG["methods"]:
            rows.append(metrics(m, lam, res, y, T, ETA))
        if i == 0 and CFG["with_refs"]:
            ref = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                         batch_size=64, shap_background=100, seed=SEED)
            for m in REF_METHODS:
                rows.append(metrics(m, np.nan, ref, y, T, ETA))
    return rows


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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--methods", default=",".join(ADV_METHODS),
                    help="comma-separated subset of dann,adae")
    ap.add_argument("--dann-optimizer", choices=["adam", "ganin"],
                    default="adam")
    ap.add_argument("--adae-head-hidden", type=int, default=0)
    ap.add_argument("--adae-cache-dir", type=Path, default=None)
    ap.add_argument("--with-refs", action="store_true",
                    help="also recompute marginal/residualized/within-tissue "
                         "(already present in the IRM sweep output)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    G["dataset"] = a.dataset
    G["seed"] = a.seed
    G["cfg"] = {
        "methods": [m.strip() for m in a.methods.split(",") if m.strip()],
        "dann_optimizer": a.dann_optimizer,
        "adae_head_hidden": a.adae_head_hidden,
        "adae_cache_dir": str(a.adae_cache_dir) if a.adae_cache_dir else None,
        "with_refs": a.with_refs,
    }
    if a.adae_cache_dir:
        a.adae_cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    setup(a.dataset, a.gene_list)

    items = _read_items(TRAIN / f"items_{a.dataset}.txt")
    # identical deterministic subset to run_irm_sweep.py
    idx = np.linspace(0, len(items) - 1, a.n_items).round().astype(int)
    subset = [items[i] for i in sorted(set(idx))]
    subset = subset[a.shard::a.nshards]
    print(f"[{a.dataset} shard {a.shard}/{a.nshards}] {len(subset)} items x "
          f"{len(LAMBDAS)} lambdas x {G['cfg']['methods']}; "
          f"{len(G['gene_cols'])} genes; nproc={a.nproc}; "
          f"dann_opt={a.dann_optimizer}; setup {time.time()-t0:.0f}s",
          flush=True)

    all_rows, done = [], 0
    ctx = mp.get_context("fork")
    with ctx.Pool(a.nproc) as pool:
        for r in pool.imap_unordered(work, subset, chunksize=1):
            done += 1
            if r:
                all_rows.extend(r)
            print(f"  {done}/{len(subset)} ({time.time()-t0:.0f}s)", flush=True)

    suffix = "" if a.nshards == 1 else f"_shard{a.shard}"
    out = a.out or HERE / "results" / f"adv_sweep_{a.dataset}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"[{a.dataset}] DONE {done} items in {time.time()-t0:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
