"""Self-rank of the target gene in its own |SHAP| ranking.

The most direct "does the method point at the right gene" check that pairs
with the functional-partner recovery panel: for each DepMap per-target model,
where does the target gene ITSELF rank among all panel genes by mean-|SHAP|?
(1 = top). A causal method should rank the target high; marginal SHAP can be
pushed down by confounded tissue markers.

We express it on the same recovery-AUROC axis as the partner panels:

    self_auroc = (N - rank) / (N - 1)

which is exactly the AUROC of ranking the single target gene above the other
N-1 panel genes by |SHAP| (1.0 = target is #1, 0.5 = median, chance).

Paired Wilcoxon marginal vs each deconfounding method (recovery_utils.METHODS) on the
the BH-intersection (marg & res both pass within-tissue-r BH-FDR).

Reads DepMap importances (../results/depmap_seedavg_preds by default).
Writes data/selfrank_newpanel.csv + data/selfrank_newpanel_summary.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recovery_utils import (OUT, SEEDAVG, METHODS, bh_intersection,  # noqa: E402
                            load_preds)

import argparse

DEFAULT_PREDS = SEEDAVG


def importance(npz, m):
    """Return mean-|SHAP| per gene. Accepts seed-averaged `imp_<m>` arrays
    or raw per-cell `shap_<m>` matrices."""
    if f"imp_{m}" in npz.files:
        return npz[f"imp_{m}"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(npz[f"shap_{m}"].astype(np.float32)), axis=0)


def has_method(npz, m):
    return (f"shap_{m}" in npz.files) or (f"imp_{m}" in npz.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(DEFAULT_PREDS),
                    help="dir of *_seed1_preds/*.npz")
    ap.add_argument("--tag", default="_newpanel",
                    help="suffix for output CSVs")
    args = ap.parse_args()
    preds = Path(args.preds)
    tag = args.tag

    inter = bh_intersection()
    print(f"BH-intersection (marg & res): {len(inter)} targets")
    print(f"preds: {preds}")

    rows = []
    for p in sorted(preds.glob("*.npz")):
        target = p.stem
        npz = load_preds(p)
        if not has_method(npz, "marginal"):
            continue
        gene_cols = np.array([str(g) for g in npz["gene_cols"]])
        if target not in set(gene_cols):
            continue                      # target not in input panel -> no self-rank
        N = len(gene_cols)
        row = {"target": target, "in_inter": target in inter, "N": N}
        for m in METHODS:
            if not has_method(npz, m):
                continue
            s = pd.Series(importance(npz, m), index=gene_cols)
            rank = int(s.rank(ascending=False, method="min").loc[target])
            row[f"rank_{m}"] = rank
            row[f"selfauroc_{m}"] = (N - rank) / (N - 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"selfrank{tag}.csv", index=False)
    print(f"wrote {OUT/f'selfrank{tag}.csv'} ({len(df)} targets)\n")

    # ---------------- paired tests + summary ----------------
    out_rows = []
    print("PAIRED WILCOXON: marginal vs method  (self-recovery AUROC = (N-rank)/(N-1))")
    print("higher = target gene ranks nearer the top of its own |SHAP|. chance=0.5")
    for setname, S in [("ALL", set(df.target)), ("INTER", inter)]:
        base = df[df.target.isin(S)]
        marg_self = base["selfauroc_marginal"].mean()
        marg_medrank = base["rank_marginal"].median()
        print(f"\n[{setname}] marginal self-AUROC={marg_self:.4f} "
              f"median rank={marg_medrank:.0f} (n={base['selfauroc_marginal'].notna().sum()})")
        for m in [x for x in METHODS if x != "marginal"]:
            if f"selfauroc_{m}" not in base:
                continue
            sub = base[[f"selfauroc_marginal", f"selfauroc_{m}",
                        "rank_marginal", f"rank_{m}"]].dropna()
            if len(sub) < 8:
                continue
            try:
                _, pval = stats.wilcoxon(sub[f"selfauroc_{m}"], sub["selfauroc_marginal"])
            except ValueError:
                pval = np.nan
            frac = (sub[f"selfauroc_{m}"] > sub["selfauroc_marginal"]).mean()
            diff = sub[f"selfauroc_{m}"].sub(sub["selfauroc_marginal"])
            out_rows.append({
                "set": setname, "method": m, "n": len(sub),
                "marg_mean": round(marg_self, 4),
                "method_mean": round(sub[f"selfauroc_{m}"].mean(), 4),
                "delta": round(diff.mean(), 4),
                "frac_better": round(frac, 3),
                "marg_medrank": int(marg_medrank),
                "method_medrank": int(sub[f"rank_{m}"].median()),
                "wilcoxon_p": pval,
                "marg_sem": round(sub["selfauroc_marginal"].sem(), 5),
                "method_sem": round(sub[f"selfauroc_{m}"].sem(), 5),
                "delta_sem": round(diff.sem(), 5)})
            print(f"    {m:13s} self-AUROC={sub[f'selfauroc_{m}'].mean():.4f} "
                  f"d={sub[f'selfauroc_{m}'].mean()-marg_self:+.4f} "
                  f"medrank={int(sub[f'rank_{m}'].median()):4d} "
                  f"frac_better={frac:.2f} n={len(sub):3d} p={pval:.2g}")

    summ = pd.DataFrame(out_rows)
    summ.to_csv(OUT / f"selfrank{tag}_summary.csv", index=False)
    print(f"\nwrote {OUT/f'selfrank{tag}_summary.csv'}")


if __name__ == "__main__":
    main()
