"""Is tissue confounding specific to Shapley values? Is the fix?

Does focusing on SHAP limit the paper's conclusions: would another attribution
method be more or less affected by tissue confounding, and would the same
corrections work on it?

This runs five attribution methods spanning three families (see
xai_methods.py) on the same models, folds and items the paper uses, under two
deconfounding conditions:

  marginal      the confounded baseline
  residualized  data-stage correction (tissue means removed from X and y)

and scores each with the paper's own diagnostics, contamination@50 and the
per-gene attribution eta^2. Two questions are then answerable directly:
does the confounding appear for every estimator, and does the correction
transfer to every estimator?

All cell lines are used. LIME costs one model evaluation per perturbation
per cell line, so the run is parallelised across items; `--n-cell-lines` can
subsample for a quick check but should be left at the default for anything
reported.

    python compute_xai_comparison.py --dataset depmap --n-items 25
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # must precede numpy/torch import

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))
sys.path.insert(0, str(HERE))

from shared.data import (load_depmap_crispr, load_ccle_expression,          # noqa: E402
                         load_ctrpv2_response, build_ctrpv2_per_drug)
from shared.deconfound import fwl_residualize                              # noqa: E402
from shared.eval import contamination_at_k, tissue_eta_squared             # noqa: E402
from shared.orchestrator import _split_train_val                           # noqa: E402
from shared.train import train_marginal, train_residualized                # noqa: E402
from xai_methods import METHODS                                            # noqa: E402

DEV = torch.device("cpu")
G = {}


def _work(item):
    """One item, in a pool worker. Globals are inherited through fork."""
    import torch as _t
    _t.set_num_threads(1)
    if G["dataset"] == "depmap":
        yf = G["crispr"][item].dropna()
        common = G["expr"].index.intersection(yf.index)
        X = G["expr"].loc[common].values
        y = yf.loc[common].values
        T = G["tissue"].loc[common].values
    else:
        X, y, T = build_ctrpv2_per_drug(item, G["resp"], G["expr"])
    if len(y) < 100 or len(np.unique(T)) < 8:
        return []
    rows = run_item(X, y, T, G["genes"], G["eta_set"], G["seed"],
                    G["n_cell_lines"], G["methods"])
    return [dict(dataset=G["dataset"], item=item, **r) for r in rows]


def attr_eta2(phi: np.ndarray, tissue: np.ndarray) -> float:
    valid = np.isfinite(phi).all(axis=1) & ~(phi == 0).all(axis=1)
    phi, tissue = phi[valid], tissue[valid]
    if len(phi) < 5:
        return np.nan
    grand = phi.mean(0)
    ss_tot = ((phi - grand) ** 2).sum(0)
    ss_btw = np.zeros(phi.shape[1])
    for t in np.unique(tissue):
        m = tissue == t
        ss_btw += int(m.sum()) * (phi[m].mean(0) - grand) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(ss_tot > 1e-12, ss_btw / ss_tot, np.nan)))


def run_item(X, y, T, genes, eta_set, seed, n_cell_lines, methods):
    """5-fold CV; attributions on every held-out cell line (or a subsample
    of each fold if `n_cell_lines` is set, for smoke tests only)."""
    rng = np.random.RandomState(seed)
    n, p = X.shape
    acc = {(cond, m): np.full((n, p), np.nan, np.float32)
           for cond in ("marginal", "residualized") for m in methods}
    picked = np.zeros(n, bool)

    for fold, (tr_full, te) in enumerate(KFold(5, shuffle=True,
                                               random_state=seed).split(X)):
        sc = StandardScaler().fit(X[tr_full])
        Xtr_f, Xte = sc.transform(X[tr_full]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
        ytr_f, yte = y[tr_full].astype(np.float32), y[te].astype(np.float32)
        Ttr_f, Tte = T[tr_full], T[te]
        tr, va = _split_train_val(len(tr_full), 0.15, seed=seed * 1000 + fold)

        if n_cell_lines is None:
            sub = np.arange(len(te))
        else:
            sub = rng.choice(len(te), min(n_cell_lines // 5, len(te)),
                             replace=False)
        picked[te[sub]] = True

        m_marg, _ = train_marginal(Xtr_f[tr], ytr_f[tr], Xtr_f[va], ytr_f[va],
                                   epochs=80, patience=12, device=DEV)
        for name in methods:
            acc[("marginal", name)][te[sub]] = METHODS[name](m_marg, Xte[sub], Xtr_f)

        Xtr_r, ytr_r, Xte_r, _, _, _ = fwl_residualize(
            Xtr_f, ytr_f, Ttr_f, Xte, Tte, yte)
        if float(np.std(ytr_r)) > 1e-6:
            m_res, _ = train_residualized(Xtr_r[tr], ytr_r[tr], Xtr_r[va],
                                          ytr_r[va], epochs=80, patience=12,
                                          device=DEV)
            for name in methods:
                acc[("residualized", name)][te[sub]] = METHODS[name](
                    m_res, Xte_r[sub], Xtr_r)

    rows = []
    for (cond, name), phi in acc.items():
        rowsel = picked & np.isfinite(phi).all(axis=1)
        if rowsel.sum() < 20:
            continue
        imp = pd.Series(np.nanmean(np.abs(phi[rowsel]), axis=0), index=genes)
        rows.append(dict(condition=cond, xai=name,
                         contam50=contamination_at_k(imp, eta_set, k=50),
                         eta2=attr_eta2(phi[rowsel], T[rowsel]),
                         n_cell_lines=int(rowsel.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["depmap", "ctrpv2"], required=True)
    ap.add_argument("--n-items", type=int, default=25)
    ap.add_argument("--n-cell-lines", type=int, default=None,
                    help="subsample for smoke tests; default uses all")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]

    if a.dataset == "depmap":
        expr, crispr, tissue = load_depmap_crispr(gene_list="panel_depmap")
        eta_set = tissue_eta_squared(expr, tissue.values)
        items = [c for c in pd.read_csv(HERE.parent / "training" / "items_depmap.txt",
                                        header=None)[0] if c in crispr.columns]
    else:
        resp = load_ctrpv2_response(min_cells_per_drug=100, min_tissues_per_drug=8)
        expr = load_ccle_expression(gene_list="panel_ctrpv2")
        u = resp[resp.cellosaurus_id.isin(expr.index)].drop_duplicates("cellosaurus_id")
        Xu = expr.loc[u.cellosaurus_id]
        eta_set = tissue_eta_squared(Xu, u.set_index("cellosaurus_id").loc[Xu.index, "tissue"].values)
        items = list(pd.read_csv(HERE.parent / "training" / "items_ctrpv2.txt", header=None)[0])

    idx = np.linspace(0, len(items) - 1, a.n_items).round().astype(int)
    items = [items[i] for i in sorted(set(idx))]
    genes = list(expr.columns)
    print(f"{a.dataset}: {len(items)} items x {len(methods)} methods x 2 conditions",
          flush=True)

    G.update(dataset=a.dataset, genes=genes, eta_set=eta_set, seed=a.seed,
             n_cell_lines=a.n_cell_lines, methods=methods, expr=expr)
    if a.dataset == "depmap":
        G.update(crispr=crispr, tissue=tissue)
    else:
        G.update(resp=resp)

    out, done = [], 0
    if a.nproc > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(a.nproc) as pool:
            for res in pool.imap_unordered(_work, items, chunksize=1):
                done += 1
                out.extend(res)
                print(f"  {done}/{len(items)}", flush=True)
    else:
        for it in items:
            out.extend(_work(it))
            done += 1
            print(f"  {done}/{len(items)} {it}", flush=True)

    df = pd.DataFrame(out)
    dest = a.out or HERE / "results" / f"xai_comparison_{a.dataset}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print(df.pivot_table(index="xai", columns="condition",
                         values=["contam50", "eta2"]).round(4).to_string())


if __name__ == "__main__":
    main()
