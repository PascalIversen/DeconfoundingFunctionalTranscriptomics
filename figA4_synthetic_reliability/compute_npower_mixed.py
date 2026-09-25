"""N-sweep on the realistic mixed-gene DGP (calibrated to DepMap/CTRPv2).

Regenerates the inputs consumed by plot_reliability.py:
  results/npower_mixed.csv          -- per (n, seed, method, gene) SHAP rank
  results/npower_mixed_summary.csv  -- per-run within-tissue r, eta^2(y), contam@50

DGP (shared.synthetic.make_synthetic_mixed, panel of 500 genes):
  200 causal (affect y, ordinary tissue structure)
  100 mixed  (causal AND tissue-confounded -- buried by marginal SHAP)
   50 confounders (pure tissue markers, no effect on y; 15 strong + 35 weak)
  150 noise
Calibrated so eta^2(y) ~ 0.21 at n=1000; within-tissue predictive r rises with n
(near-0 at 1k, ~0.24 by 80k) -- the sample-size limit. Effects are tiered
(strong/medium/weak/v.weak) so strong genes surface at small n and weak ones only
at large n.

The reliability figure uses only the n=1000 slice, over 32 seeds:
  python compute_npower_mixed.py --seed-start 1  --replicates 8   # base seeds
  python compute_npower_mixed.py --seed-start 9  --replicates 24  # extra n=1000
(the shipped results/ CSVs already contain both; re-running is only needed to
reproduce from scratch and requires a GPU).

Imports the bundled training pipeline under ../training/shared -- this
folder is self-contained.
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))   # repo shared/ package

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from shared.synthetic import make_synthetic_mixed
from shared.train import (train_marginal, train_residualized, train_irm,
                          train_dann, fit_adae_embedding, train_adae_head)
from shared.shap_utils import gradient_shap, within_tissue_shap
from shared.deconfound import fwl_residualize
from shared.orchestrator import _split_train_val

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GRID = [500, 1000, 2000, 5000, 10000, 20000, 40000, 80000]
SHAP_CELLS = 500
RESULTS = HERE / "results"
ALL_METHODS = ["marginal", "within_tissue", "residualized", "irm", "dann", "adae"]


def within_r(yp, yt, Tv):
    a, b = [], []
    for t in np.unique(Tv):
        m = Tv == t
        if m.sum() > 3:
            a.append(yp[m] - yp[m].mean()); b.append(yt[m] - yt[m].mean())
    if not a:
        return np.nan
    a, b = np.concatenate(a), np.concatenate(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 else np.nan


def rank_series(imp, gene_names):
    return pd.Series(np.asarray(imp), index=gene_names).rank(ascending=False, method="min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="comma-separated subset of " + ",".join(ALL_METHODS))
    ap.add_argument("--n-grid", default=None,
                    help="comma-separated cell counts; defaults to the full grid")
    ap.add_argument("--dgp-variant", default="v1", choices=["v1", "v2"],
                    help="v1 = originally published geometry; v2 gives mixed "
                         "genes and confounders the SAME tissue eta^2, so "
                         "tissue statistics alone cannot separate them")
    ap.add_argument("--out-suffix", default=None,
                    help="override the output filename suffix")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    n_grid = ([int(x) for x in args.n_grid.split(",")] if args.n_grid else N_GRID)
    suffix = (args.out_suffix if args.out_suffix is not None
              else ("" if args.seed_start == 1 else f"_seed{args.seed_start}"))
    out = RESULTS / f"npower_mixed{suffix}.csv"
    out_sum = RESULTS / f"npower_mixed_summary{suffix}.csv"
    rows, summ = [], []

    for seed in range(args.seed_start, args.seed_start + args.replicates):
        for n in n_grid:
            t0 = time.time()
            ds = make_synthetic_mixed(n_cells=n, seed=seed,
                                      variant=args.dgp_variant)
            cls = {g: "mixed" for g in ds.mixed_genes}
            cls.update({g: "causal" for g in ds.pure_causal_genes})
            cls.update({g: "confounder" for g in ds.lineage_genes})
            tracked = list(cls.keys())
            eff = {g: ds.effect_size.get(g, 0.0) for g in tracked}

            Xs = StandardScaler().fit_transform(ds.X).astype(np.float32)
            y = ds.y.astype(np.float32)
            loc, val = _split_train_val(n, 0.15, seed=seed)
            rng = np.random.RandomState(seed)
            sub = rng.choice(n, min(SHAP_CELLS, n), replace=False)
            ranks = {}
            extra_wr, extra_pr, extra_diag = {}, {}, {}

            # The marginal model is also the backbone that within-tissue SHAP
            # explains, so it is fitted whenever either is requested.
            need_marg = ("marginal" in methods) or ("within_tissue" in methods)
            wr = pr = np.nan

            # 1) Marginal
            if need_marg:
              m_marg, _ = train_marginal(Xs[loc], y[loc], Xs[val], y[val],
                                         epochs=70, patience=10, device=DEV)
              with torch.no_grad():
                  yp = m_marg(torch.from_numpy(Xs[val]).to(DEV)).cpu().numpy().ravel()
              wr = within_r(yp, y[val], ds.T[val])
              pr = float(np.corrcoef(yp, y[val])[0, 1])
              if "marginal" in methods:
                  ranks["marginal"] = rank_series(
                      gradient_shap(m_marg, Xs[sub], Xs[loc], ds.gene_names,
                                    device=DEV, max_background=100, seed=seed
                                    ).abs().mean(0).values, ds.gene_names)

            # 2) Within-tissue (reuse marginal model)
            if "within_tissue" in methods:
                sh_w, has = within_tissue_shap(
                    m_marg, Xs[sub], ds.T[sub], X_bg=Xs[loc], T_bg=ds.T[loc],
                    feature_names=ds.gene_names, max_background=100,
                    min_tissue_size=5, device=DEV, seed=seed)
                ranks["within_tissue"] = rank_series(
                    np.nanmean(np.abs(sh_w[has]), axis=0), ds.gene_names)

            # 3) Residualized (FWL)
            if "residualized" in methods:
              Xlr, ylr, Xsr, _, _, _ = fwl_residualize(
                  Xs[loc], y[loc], ds.T[loc], Xs[sub], ds.T[sub], y[sub])
              if float(np.std(ylr)) > 1e-6:
                  c = len(Xlr) * 85 // 100
                  m_res, _ = train_residualized(Xlr[:c], ylr[:c], Xlr[c:], ylr[c:],
                                                epochs=70, patience=10, device=DEV)
                  ranks["residualized"] = rank_series(
                      gradient_shap(m_res, Xsr, Xlr, ds.gene_names, device=DEV,
                                    max_background=100, seed=seed + 7
                                    ).abs().mean(0).values, ds.gene_names)

            # 4) IRM
            if "irm" in methods:
                m_irm, _ = train_irm(Xs[loc], y[loc], ds.T[loc], Xs[val], y[val],
                                     epochs=70, patience=10, irm_lambda=1.0, device=DEV)
                ranks["irm"] = rank_series(
                    gradient_shap(m_irm, Xs[sub], Xs[loc], ds.gene_names, device=DEV,
                                  max_background=100, seed=seed + 13
                                  ).abs().mean(0).values, ds.gene_names)

            # 5) DANN -- domain-adversarial, tissue as the domain label
            if "dann" in methods:
                m_dann, _, dg = train_dann(
                    Xs[loc], y[loc], ds.T[loc], Xs[val], y[val], ds.T[val],
                    epochs=70, patience=10, lambda_max=1.0, device=DEV)
                with torch.no_grad():
                    yp_d = m_dann(torch.from_numpy(Xs[val]).to(DEV)).cpu().numpy().ravel()
                extra_wr["dann"] = within_r(yp_d, y[val], ds.T[val])
                extra_pr["dann"] = float(np.corrcoef(yp_d, y[val])[0, 1])
                extra_diag["dann"] = dg
                ranks["dann"] = rank_series(
                    gradient_shap(m_dann, Xs[sub], Xs[loc], ds.gene_names, device=DEV,
                                  max_background=100, seed=seed + 23
                                  ).abs().mean(0).values, ds.gene_names)

            # 6) AD-AE -- adversarial deconfounding autoencoder. The encoder is
            #    phenotype-free; the head is the paper's elastic net on the
            #    frozen embedding.
            if "adae" in methods:
                adae, ag = fit_adae_embedding(
                    Xs[loc], ds.T[loc], Xs[val], ds.T[val],
                    lambda_adv=1.0, device=DEV)
                m_adae, _, hd = train_adae_head(
                    adae, Xs[loc], y[loc], Xs[val], y[val],
                    epochs=70, patience=10, device=DEV)
                with torch.no_grad():
                    yp_a = m_adae(torch.from_numpy(Xs[val]).to(DEV)).cpu().numpy().ravel()
                extra_wr["adae"] = within_r(yp_a, y[val], ds.T[val])
                extra_pr["adae"] = float(np.corrcoef(yp_a, y[val])[0, 1])
                extra_diag["adae"] = {**ag, **hd}
                ranks["adae"] = rank_series(
                    gradient_shap(m_adae, Xs[sub], Xs[loc], ds.gene_names, device=DEV,
                                  max_background=100, seed=seed + 31
                                  ).abs().mean(0).values, ds.gene_names)

            n_conf = len(ds.lineage_genes)
            for method, rk in ranks.items():
                for g in tracked:
                    rows.append(dict(seed=seed, n_cells=n, method=method, gene=g,
                                     gene_class=cls[g], effect=round(eff[g], 4),
                                     rank=int(rk[g])))
                top50 = set(rk.nsmallest(50).index)
                conf_rank = float(np.median([rk[g] for g in ds.lineage_genes]))
                # within_r / pearson_r are stored directly for the marginal
                # model (the other three either share it or predict a
                # different target); the adversarial baselines have their
                # own predictions, so theirs are recorded separately.
                m_wr = extra_wr.get(method, wr if method == "marginal" else np.nan)
                m_pr = extra_pr.get(method, pr if method == "marginal" else np.nan)
                row = dict(seed=seed, n_cells=n, method=method,
                           within_r=round(m_wr, 4) if np.isfinite(m_wr) else np.nan,
                           pearson_r=round(m_pr, 4) if np.isfinite(m_pr) else np.nan,
                           eta_y=round(ds.eta_y, 4),
                           contam50=len(top50 & set(ds.lineage_genes)) / 50,
                           conf_median_rank=conf_rank)
                for k, v in (extra_diag.get(method) or {}).items():
                    if isinstance(v, (int, float)):
                        row[f"diag_{k}"] = round(float(v), 4)
                summ.append(row)
            print(f"[{args.dgp_variant}] seed {seed} n={n} etaY={ds.eta_y:.3f} wr={wr:+.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
            pd.DataFrame(summ).to_csv(out_sum, index=False)
    print("done", out)


if __name__ == "__main__":
    main()
