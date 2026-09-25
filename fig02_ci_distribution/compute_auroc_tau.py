"""Compute AUROC(tissue-marker | gene importance) per method, per dataset, per tau.

For each item we rank genes by global importance mean_i|phi_{i,g}| and compute
the AUROC of marker-membership (expression tissue-eta^2 > tau) against that
ranking; AUROC is averaged over items. Includes the between-tissue ("tissue
only") attribution as a positive-control ceiling.

Heavy step (reads the per-item SHAP npz); run where the preds live, then plot
with plot_auroc_tau.py. Writes results/auroc_tau.csv:
    dataset, method, tau, auroc, sem, n
"""
from __future__ import annotations
import argparse
import glob
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent

# Adversarial baselines (DANN / AD-AE) live in a sibling "*_extra" directory so
# the published prediction bundle stays untouched; load_preds merges them in.
import sys as _sys
_sys.path.insert(0, str(HERE.parent / "training"))
from shared.preds_io import load_preds  # noqa: E402
METHODS = ["marginal", "residualized", "irm", "within_tissue",
           "between_tissue", "dann", "adae"]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]


def find_data_root(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / "data" / "DepMap").exists() or (c / "data" / "CTRPv2").exists():
            return c / "data"
    raise FileNotFoundError("no data/ with DepMap or CTRPv2 found")


def eta2_expr(data, genes, ds, train_dir):
    if ds == "depmap":
        model = pd.read_csv(data / "DepMap" / "Model.csv")[
            ["ModelID", "OncotreeLineage"]].dropna()
        tmap = dict(zip(model.ModelID, model.OncotreeLineage))
        f = data / "DepMap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
        logt = False
    else:
        import sys
        sys.path.insert(0, str(train_dir))
        from shared.data import load_ctrpv2_response
        resp = load_ctrpv2_response()
        tmap = dict(zip(resp.cellosaurus_id, resp.tissue))
        f = data / "CTRPv2" / "gene_expression.csv"
        if not f.exists():
            f = data / "CCLE" / "gene_expression.csv"
        logt = True
    hdr = pd.read_csv(f, nrows=0).columns
    sym = lambda c: c.split(" (")[0]
    keep = [hdr[0]] + [c for c in hdr[1:] if sym(c) in set(genes)]
    e = pd.read_csv(f, usecols=keep, index_col=0)
    e.columns = [sym(c) for c in e.columns]
    cells = [c for c in e.index if c in tmap]
    e = e.loc[cells]
    tis = np.array([tmap[c] for c in cells])
    X = np.log2(e.values + 1) if logt else e.values
    out = {}
    for j, c in enumerate(e.columns):
        v = X[:, j]; g = v.mean(); sst = ((v - g) ** 2).sum()
        out[c] = (np.nan if sst < 1e-12 else
                  sum(int((tis == t).sum()) * (v[tis == t].mean() - g) ** 2
                      for t in np.unique(tis)) / sst)
    return pd.Series(out).reindex(genes).values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--train-dir", type=Path, default=HERE.parent / "training")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data = a.data_root or find_data_root(a.preds_root)
    rows = []
    for ds in ["depmap", "ctrpv2"]:
        fs = sorted(glob.glob(f"{a.preds_root}/{ds}_seed{a.seed}_preds/*.npz"))
        if not fs:
            print(f"skip {ds}: no preds"); continue
        genes = np.array([str(g) for g in np.load(fs[0], allow_pickle=True)["gene_cols"]])
        eta = eta2_expr(data, genes, ds, a.train_dir); ok = np.isfinite(eta)
        imps = {m: [] for m in METHODS}
        for f in fs:
            d = load_preds(f)
            for m in METHODS:
                imps[m].append(np.nanmean(np.abs(d[f"shap_{m}"].astype(np.float32)), 0)
                               if f"shap_{m}" in d.files else None)
        for m in METHODS:
            for tau in TAUS:
                mk = (eta > tau).astype(int); a_ = []
                for imp in imps[m]:
                    if imp is None: continue
                    o = ok & np.isfinite(imp)
                    if mk[o].sum() >= 3 and mk[o].sum() < o.sum():
                        a_.append(roc_auc_score(mk[o], imp[o]))
                a_ = np.array(a_)
                rows.append((ds, m, tau, a_.mean(),
                             a_.std() / np.sqrt(len(a_)), len(a_)))
        print(f"{ds}: done ({len(fs)} items)")
    df = pd.DataFrame(rows, columns=["dataset", "method", "tau", "auroc", "sem", "n"])
    df.to_csv(a.out_dir / "auroc_tau.csv", index=False)
    print(f"wrote {a.out_dir / 'auroc_tau.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
