"""Define the "well-predicted within tissue" item subsets.

Do the Fig-2 diagnostics and the Fig-3 GSEA still hold when restricted to items the
models actually predict within tissue, using the filter already described in A.6.1?
That filter is reimplemented here for BOTH datasets (A.6.1 only ever applied it to
DepMap, for Fig 4a):

    a61       BH-FDR(q<0.05) on the median one-sided within-tissue-r p-value over
              the five seeds, applied separately to marginal and residualized,
              intersection kept.   <- the paper's filter (recovery_utils.bh_intersection)
    marginal  BH-FDR(q<0.05) on the marginal model only.
    all4      intersection over marginal / residualized / irm / within_tissue.

`a61` selects items on which both marginal and residualized models perform
significantly, which is not obviously neutral between the compared methods.
`marginal` is the neutral alternative: the baseline every method is compared
against picks the items, so no deconfounder is favoured. `all4` is the strictest
symmetric variant. Reporting all three is what makes the answer robust.

DANN / AD-AE have no within-tissue r p-values (gen_preds_extra.py does not write
them), so they cannot enter the filter; they are still *evaluated* on every
subset, which is fine — the subsets are only item selections.

Input:  ../results/{depmap,ctrpv2}_within_tissue_pvals.csv
Output: results/{depmap,ctrpv2}_wt_items.csv   (one row per item, flags per variant)

    python wt_subset.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PVALS = REPO / "results"
OUT = HERE / "results"

ITEM_COL = {"depmap": "target", "ctrpv2": "drug"}
VARIANTS = {
    "a61": ["marginal", "residualized"],
    "marginal": ["marginal"],
    "all4": ["marginal", "residualized", "irm", "within_tissue"],
}


def bh_pass(med: pd.DataFrame, item_col: str, method: str, alpha: float = 0.05) -> set:
    """Items whose median-over-seeds one-sided p passes BH-FDR for one method."""
    sub = med[med.method == method].dropna(subset=["p_one_sided"])
    if sub.empty:
        return set()
    rej, _, _, _ = multipletests(sub["p_one_sided"], alpha=alpha, method="fdr_bh")
    return set(sub[item_col][rej])


def subsets_for(dataset: str) -> pd.DataFrame:
    item_col = ITEM_COL[dataset]
    pv = pd.read_csv(PVALS / f"{dataset}_within_tissue_pvals.csv")
    # Median over seeds, exactly as recovery_utils.bh_intersection does.
    med = (pv.groupby([item_col, "method"])[["p_one_sided", "within_tissue_r"]]
             .median().reset_index())
    passing = {m: bh_pass(med, item_col, m) for m in
               ["marginal", "residualized", "irm", "within_tissue"]}

    wide = med.pivot(index=item_col, columns="method",
                     values="within_tissue_r").add_prefix("r_wt_")
    out = wide.reset_index()
    for m, s in passing.items():
        out[f"sig_{m}"] = out[item_col].isin(s)
    for name, methods in VARIANTS.items():
        keep = set.intersection(*(passing[m] for m in methods))
        out[f"in_{name}"] = out[item_col].isin(keep)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in ["depmap", "ctrpv2"]:
        df = subsets_for(ds)
        f = OUT / f"{ds}_wt_items.csv"
        df.to_csv(f, index=False)
        n = len(df)
        parts = "  ".join(
            f"{v}={int(df[f'in_{v}'].sum())}" for v in VARIANTS)
        print(f"{ds}: {n} items total   {parts}")
        for v in VARIANTS:
            sel = df[df[f"in_{v}"]]
            print(f"    {v:9s} n={len(sel):4d}  "
                  f"median marginal r_wt {sel['r_wt_marginal'].median():.3f} "
                  f"(all items {df['r_wt_marginal'].median():.3f})")
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
