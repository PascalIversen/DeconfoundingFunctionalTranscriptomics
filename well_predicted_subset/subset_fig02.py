"""Fig 2 (attribution tissue-eta^2 + contamination@K) on the well-predicted subsets.

Do the main results hold on the subset of items with meaningful within-tissue
predictive performance?

Both Fig-2 measures are computed per item, so restricting the figure to a subset of
items is a row filter on the already-computed CSVs — nothing has to be recomputed
from the .npz bundles. contamination@K's chance line and marker set L are defined on
gene expression, not on items, so *_contamination_meta.csv carries over unchanged.

Writes, per subset variant from wt_subset.py:
    results/<variant>/{depmap,ctrpv2}_attr_eta2.csv          (filtered)
    results/<variant>/{depmap,ctrpv2}_contamination_at_k.csv (filtered)
    results/<variant>/{depmap,ctrpv2}_contamination_meta.csv (copied)
    figures/fig02_<variant>.{pdf,png}                        (via the real plot script)
and one comparison table across variants:
    results/fig02_subset_summary.csv
    results/fig02_paired_tests.csv

    python subset_fig02.py                 # all variants
    python subset_fig02.py --variants a61
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIG02 = REPO / "fig02_ci_distribution"
sys.path.insert(0, str(FIG02))
import plot_ci_distribution as P  # noqa: E402  (reuse the paper's renderer)

ITEM_COL = {"depmap": "target", "ctrpv2": "drug"}
DATASETS = ["depmap", "ctrpv2"]
METHODS = P.METHODS
KS = P.KS


def subset_items(dataset: str, variant: str) -> set:
    f = HERE / "results" / f"{dataset}_wt_items.csv"
    if not f.exists():
        raise SystemExit(f"{f} missing - run wt_subset.py first")
    df = pd.read_csv(f)
    col = f"in_{variant}"
    if col not in df:
        raise SystemExit(f"unknown variant '{variant}' ({f} has no {col})")
    return set(df.loc[df[col], ITEM_COL[dataset]])


def write_filtered(dataset: str, items: set, out_dir: Path) -> tuple:
    """Filter the two Fig-2 CSVs to `items`; return (eta2_full, eta2_sub, c_full, c_sub)."""
    src = FIG02 / "results"
    item_col = ITEM_COL[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)

    e = pd.read_csv(src / f"{dataset}_attr_eta2.csv")
    es = e[e[item_col].isin(items)]
    es.to_csv(out_dir / f"{dataset}_attr_eta2.csv", index=False)

    c = pd.read_csv(src / f"{dataset}_contamination_at_k.csv")
    cs = c[c["item"].isin(items)]
    cs.to_csv(out_dir / f"{dataset}_contamination_at_k.csv", index=False)

    shutil.copy(src / f"{dataset}_contamination_meta.csv",
                out_dir / f"{dataset}_contamination_meta.csv")
    return e, es, c, cs


def summarise(dataset: str, variant: str, e: pd.DataFrame, c: pd.DataFrame,
              item_col: str) -> list:
    """Per-method eta^2 and contamination@K summary rows for one (dataset, subset)."""
    rows = []
    n_items = e[item_col].nunique()
    for m in METHODS:
        em = e[e.method == m]["eta2"].dropna()
        row = {"dataset": dataset, "variant": variant, "n_items": n_items,
               "method": m,
               "eta2_mean": em.mean(), "eta2_median": em.median(),
               "eta2_q90": em.quantile(0.90)}
        cm = c[c.method == m]
        for k in KS:
            v = cm[cm.K == k]["contamination"]
            row[f"contam@{k}"] = v.mean()
            row[f"contam@{k}_sem"] = v.std() / np.sqrt(max(1, len(v)))
        rows.append(row)
    # Reductions relative to marginal, the shared baseline (the actual claim).
    base = {r["method"]: r for r in rows}["marginal"]
    for r in rows:
        r["eta2_mean_vs_marginal"] = r["eta2_mean"] / base["eta2_mean"] - 1.0
        for k in KS:
            r[f"contam@{k}_vs_marginal"] = r[f"contam@{k}"] / base[f"contam@{k}"] - 1.0
    return rows


def paired_tests(dataset: str, variant: str, e: pd.DataFrame, c: pd.DataFrame,
                 item_col: str) -> list:
    """Per-item paired Wilcoxon of each deconfounder against marginal.

    Pairing on the item removes the between-item variance that the pooled
    histogram/bar means hide, which is what makes a shrunken subset (n=144 rather
    than 200) still able to resolve the effect.
    """
    rows = []
    ei = (e.groupby([item_col, "method"])["eta2"].mean().unstack("method"))
    for m in METHODS:
        if m == "marginal" or m not in ei:
            continue
        a, b = ei["marginal"], ei[m]
        ok = a.notna() & b.notna()
        stat, p = wilcoxon(a[ok], b[ok])
        rows.append({"dataset": dataset, "variant": variant, "method": m,
                     "measure": "eta2_attr (per-item mean)", "n": int(ok.sum()),
                     "median_marginal": a[ok].median(), "median_method": b[ok].median(),
                     "median_delta": (b[ok] - a[ok]).median(),
                     "frac_lower_than_marginal": float((b[ok] < a[ok]).mean()),
                     "wilcoxon_p": p})
    for k in KS:
        ck = (c[c.K == k].pivot_table(index="item", columns="method",
                                      values="contamination"))
        for m in METHODS:
            if m == "marginal" or m not in ck:
                continue
            a, b = ck["marginal"], ck[m]
            ok = a.notna() & b.notna()
            # All-zero differences (identical rankings) make wilcoxon undefined.
            d = (b[ok] - a[ok])
            p = wilcoxon(a[ok], b[ok]).pvalue if (d != 0).any() else np.nan
            rows.append({"dataset": dataset, "variant": variant, "method": m,
                         "measure": f"contamination@{k}", "n": int(ok.sum()),
                         "median_marginal": a[ok].median(),
                         "median_method": b[ok].median(),
                         "median_delta": d.median(),
                         "frac_lower_than_marginal": float((d < 0).mean()),
                         "wilcoxon_p": p})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["a61", "marginal", "all4"])
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    summary, tests = [], []
    for variant in ["full", *a.variants]:
        out_dir = HERE / "results" / variant
        for ds in DATASETS:
            item_col = ITEM_COL[ds]
            if variant == "full":
                # Baseline row set: the published Fig-2 CSVs, unfiltered.
                e = pd.read_csv(FIG02 / "results" / f"{ds}_attr_eta2.csv")
                c = pd.read_csv(FIG02 / "results" / f"{ds}_contamination_at_k.csv")
            else:
                items = subset_items(ds, variant)
                _, e, _, c = write_filtered(ds, items, out_dir)
            summary += summarise(ds, variant, e, c, item_col)
            tests += paired_tests(ds, variant, e, c, item_col)
            print(f"{variant:9s} {ds:7s} items={e[item_col].nunique():4d}")
        if not a.no_figures and variant != "full":
            fig_dir = HERE / "figures"
            P.plot(out_dir, fig_dir, f"fig02_{variant}")
            P.plot_row(out_dir, fig_dir, f"fig02_{variant}_singlerow")

    res = HERE / "results"
    pd.DataFrame(summary).to_csv(res / "fig02_subset_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(res / "fig02_paired_tests.csv", index=False)
    print(f"wrote {res/'fig02_subset_summary.csv'}")
    print(f"wrote {res/'fig02_paired_tests.csv'}")


if __name__ == "__main__":
    main()
