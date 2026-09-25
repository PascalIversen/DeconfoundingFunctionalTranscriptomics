"""Compute contamination@K from the per-item SHAP .npz preds.

contamination@K = fraction of a method's top-K attributed genes that are
tissue markers, where the marker set is method-independent:

    L = { gene : eta^2_expr(gene) > tau },   tau = 0.30

eta^2_expr = one-way ANOVA effect size of a gene's EXPRESSION vs tissue.
For each item we rank genes by global importance mean_i|phi_{i,g}| and take
the top-K. Chance = |L| / P.

Writes (to --out-dir):
    {dataset}_contamination_at_k.csv   long form: item, method, K, contamination
    {dataset}_contamination_meta.csv   one row: dataset, P, n_L, chance, tau

Heavy step (reads the per-item npz); writes the small CSVs that
plot_ci_distribution.py reads.
"""
from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["marginal", "residualized", "irm", "within_tissue",
           "dann", "adae"]
KS = [10, 50, 100]
TAU = 0.30
HERE = Path(__file__).resolve().parent

# Adversarial baselines (DANN / AD-AE) live in a sibling "*_extra" directory so
# the published prediction bundle stays untouched; load_preds merges them in.
import sys as _sys
_sys.path.insert(0, str(HERE.parent / "training"))
from shared.preds_io import load_preds  # noqa: E402


def find_data_root(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / "data" / "DepMap").exists() or (c / "data" / "CTRPv2").exists():
            return c / "data"
    raise FileNotFoundError("no data/ with DepMap or CTRPv2 found")


def eta2_vec(X, tissue):
    grand = X.mean(0)
    sst = ((X - grand) ** 2).sum(0)
    ssb = np.zeros(X.shape[1])
    for t in np.unique(tissue):
        m = tissue == t
        ssb += int(m.sum()) * (X[m].mean(0) - grand) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sst > 1e-12, ssb / sst, np.nan)


def depmap_expr_eta2(data, genes):
    model = pd.read_csv(data / "DepMap" / "Model.csv")[
        ["ModelID", "OncotreeLineage"]].dropna()
    tmap = dict(zip(model.ModelID, model.OncotreeLineage))
    f = data / "DepMap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    hdr = pd.read_csv(f, nrows=0).columns
    sym = lambda c: c.split(" (")[0]
    keep = [hdr[0]] + [c for c in hdr[1:] if sym(c) in set(genes)]
    expr = pd.read_csv(f, usecols=keep, index_col=0)
    expr.columns = [sym(c) for c in expr.columns]
    cells = [c for c in expr.index if c in tmap]
    expr = expr.loc[cells]
    tissue = np.array([tmap[c] for c in cells])
    return pd.Series(eta2_vec(expr.values, tissue), index=expr.columns)


def ctrpv2_expr_eta2(data, genes, train_dir):
    import sys
    sys.path.insert(0, str(train_dir))
    from shared.data import load_ctrpv2_response
    resp = load_ctrpv2_response()
    tmap = dict(zip(resp.cellosaurus_id, resp.tissue))
    src = data / "CTRPv2" / "gene_expression.csv"
    if not src.exists():
        src = data / "CCLE" / "gene_expression.csv"
    expr = pd.read_csv(src)
    expr.columns = expr.columns.str.strip()
    expr = expr.set_index("cellosaurus_id")
    cols = [g for g in genes if g in expr.columns]
    cells = [c for c in expr.index if c in tmap]
    sub = expr.loc[cells, cols]
    tissue = np.array([tmap[c] for c in cells])
    return pd.Series(eta2_vec(np.log2(sub.values + 1), tissue), index=cols)


def contamination_rows(preds_dir, L_set):
    rows = []
    for f in sorted(glob.glob(f"{preds_dir}/*.npz")):
        d = load_preds(f)
        genes = d["gene_cols"]
        item = Path(f).stem
        for m in METHODS:
            key = f"shap_{m}"
            if key not in d.files:
                continue
            imp = np.nanmean(np.abs(d[key].astype(np.float32)), axis=0)
            s = pd.Series(imp, index=genes).dropna()
            for k in KS:
                top = set(s.nlargest(k).index)
                rows.append((item, m, k, len(top & L_set) / k))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--train-dir", type=Path, default=HERE.parent / "training")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--datasets", nargs="+", default=["depmap", "ctrpv2"])
    a = p.parse_args()
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data = a.data_root or find_data_root(a.preds_root)
    print(f"data root: {data}")

    for ds in a.datasets:
        preds = a.preds_root / f"{ds}_seed{a.seed}_preds"
        if not preds.exists():
            print(f"skip {ds}: {preds} missing"); continue
        d0 = load_preds(sorted(glob.glob(f"{preds}/*.npz"))[0])
        genes = list(d0["gene_cols"])
        eta = (depmap_expr_eta2(data, genes) if ds == "depmap"
               else ctrpv2_expr_eta2(data, genes, a.train_dir))
        L = set(eta[eta > a.tau].index)
        chance = len(L) / len(genes)
        print(f"{ds}: P={len(genes)} |L|={len(L)} chance={chance:.3f} (tau={a.tau})")
        rows = contamination_rows(preds, L)
        df = pd.DataFrame(rows, columns=["item", "method", "K", "contamination"])
        df.to_csv(a.out_dir / f"{ds}_contamination_at_k.csv", index=False)
        pd.DataFrame([{"dataset": ds, "P": len(genes), "n_L": len(L),
                       "chance": chance, "tau": a.tau}]).to_csv(
            a.out_dir / f"{ds}_contamination_meta.csv", index=False)
        print(f"  wrote {ds}_contamination_at_k.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
