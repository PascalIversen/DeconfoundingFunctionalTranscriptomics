"""Build the predictive-performance table (overall + within-tissue Pearson r).

Input : results/per_item_pearson.csv   (compute_per_item_pearson.py)
Output: results/performance_table.csv   (full stats, every method)
        performance_table.tex            (formatted, paper-ready)

Aggregation (decided deliberately):

  * Unit of replication = the ITEM (drug / knockout target). Each item is a
    separate model trained on its own response, so items are the independent
    observations; the 5 seeds are repeats of the *same* item and are averaged
    away first (seed-mean per item) so they don't inflate the count.
  * Spread reported = SEM across items (SD / sqrt(n_items)). The table exists to
    compare method *means* -- "does deconfounding keep within-tissue r?" -- so
    the relevant uncertainty is the precision of the across-item mean, not the
    item-to-item heterogeneity (that SD is large and is reported in the CSV /
    caption instead). All methods are evaluated on the SAME items and folds, so
    the comparison is paired.
  * r is averaged arithmetically across items and seeds (not Fisher-z): the
    table is a descriptive summary, and the arithmetic mean keeps the numbers
    on the interpretable correlation scale.

Rows: tissue-mean baseline (reference) + the three trained predictors. Marginal
SHAP and within-tissue SHAP share one model, so the "Marginal" row is also the
within-tissue-SHAP row (noted in the caption); residualization and IRM are the
only methods that change the predictor and could move the numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FOLDSYM = HERE.parent / "well_predicted_subset" / "results" / "fold_symmetric_rwt.csv"

ROW_ORDER = ["baseline", "marginal", "residualized", "irm"]
ROW_LABELS = {
    "baseline":     r"Tissue-mean baseline",
    "marginal":     r"Marginal / Within-tissue SHAP",
    "residualized": r"Residualized",
    "irm":          r"IRM",
}
DS_LABELS = {"depmap": "DepMap (essentiality)",
             "ctrpv2": "CTRPv2 (drug response)"}


def _stats(v: np.ndarray) -> dict:
    n = len(v)
    return {
        "mean": float(np.mean(v)) if n else np.nan,
        "sd": float(np.std(v, ddof=1)) if n > 1 else np.nan,
        "sem": float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "median": float(np.median(v)) if n else np.nan,
        "n": n,
    }


def foldsym_wt() -> pd.DataFrame:
    """Within-tissue r, fold-symmetric convention (well_predicted_subset/
    fold_symmetric_rwt.py): centres y and yhat within (tissue x fold) groups
    instead of within tissue, which removes the leave-one-fold-out tissue-mean
    baseline term identically for every method -- no per-method reconstruction
    convention needed (residualized's `wt_foldsym` already equals its
    baseline-stripped `wt_foldsym_stripped`). This is the r_wt Table 1 and
    Table tab:performance_sd both report."""
    df = pd.read_csv(FOLDSYM)
    per_item = (df.groupby(["dataset", "item", "method"])["wt_foldsym"]
                  .mean().reset_index())
    out = []
    for (ds, m), g in per_item.groupby(["dataset", "method"]):
        rec = {"dataset": ds, "method": m}
        for k, v in _stats(g["wt_foldsym"].dropna().to_numpy()).items():
            rec[f"wt_{k}"] = v
        out.append(rec)
    return pd.DataFrame(out)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """seed-mean per item, then mean / sem / sd / median across items.

    Overall r and the (per-tissue-convention) reconstructed within-tissue r
    come from per_item_pearson.csv; within-tissue r itself comes from the
    fold-symmetric convention (see foldsym_wt()), not from this file's
    wt_pearson column.
    """
    cols = ["pearson", "wt_pearson_recon"]
    # 1) average over seeds within each (dataset, item, method)
    per_item = (df.groupby(["dataset", "item", "method"])[cols]
                  .mean().reset_index())
    out = []
    for (ds, m), g in per_item.groupby(["dataset", "method"]):
        rec = {"dataset": ds, "method": m}
        for col, tag in [("pearson", "overall"), ("wt_pearson_recon", "wtrecon")]:
            v = g[col].dropna().to_numpy()
            for k, val in _stats(v).items():
                rec[f"{tag}_{k}"] = val
        out.append(rec)
    agg = pd.DataFrame(out)
    return agg.merge(foldsym_wt(), on=["dataset", "method"], how="left")


def fmt(mean: float, sem: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sem):
        return f"{mean:.3f}"
    return f"{mean:.3f}\\,$\\pm$\\,{sem:.3f}"


def _table_lines(a: pd.DataFrame, spread: str) -> list:
    """Shared tabular: each cell is mean $\\pm$ <spread> ('sem' or 'sd')."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l cc cc}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{DepMap (essentiality)}"
        r" & \multicolumn{2}{c}{CTRPv2 (drug response)} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"    Method & Overall $r$ & Within-tissue $r_{\mathrm{wt}}$"
        r" & Overall $r$ & Within-tissue $r_{\mathrm{wt}}$ \\",
        r"    \midrule",
    ]
    for m in ROW_ORDER:
        cells = [ROW_LABELS[m]]
        for ds in ("depmap", "ctrpv2"):
            r = a.loc[(ds, m)]
            cells.append(fmt(r["overall_mean"], r[f"overall_{spread}"]))
            cells.append(fmt(r["wt_mean"], r[f"wt_{spread}"]))
        lines.append("    " + " & ".join(cells) + r" \\")
        if m == "baseline":
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    return lines


def make_tex_sd(agg: pd.DataFrame) -> str:
    """Appendix variant: same table, mean +/- SD instead of SEM."""
    a = agg.set_index(["dataset", "method"])
    lines = _table_lines(a, "sd")
    cap = (
        r"  \caption{\textbf{Predictive accuracy with across-item standard "
        r"deviation (SD).} Identical to Table~\ref{tab:performance}, but each "
        r"cell reports mean\,$\pm$\,SD across items (drugs/knockout targets) "
        r"instead of the SEM ($n=200$ targets for DepMap, $n=440$ drugs for "
        r"CTRPv2; seeds averaged within item first).}"
    )
    lines.append(cap)
    lines.append(r"  \label{tab:performance_sd}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_tex(agg: pd.DataFrame) -> str:
    a = agg.set_index(["dataset", "method"])
    lines = _table_lines(a, "sem")
    cap = (
        r"  \caption{Overall vs.\ within-tissue Pearson "
        r"correlations between the measured response and the model prediction, "
        r"as mean\,$\pm$\,SEM across items (knockout targets/drugs). Each "
        r"item's correlation is first averaged over the five seeds. "
        r"Within-tissue correlation is undefined for the tissue-mean baseline "
        r"as it has zero variance by construction. "
        r"Within-tissue SHAP explains the marginal model, so it has no "
        r"separate entry. For standard deviations, see "
        r"Table~\ref{tab:performance_sd}. MSE yields the same ranking (see "
        r"Table~\ref{tab:performance_mse}).}"
    )
    lines.append(cap)
    lines.append(r"  \label{tab:performance}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_csv(HERE / "results" / "per_item_pearson.csv")
    agg = aggregate(df)
    agg = agg.sort_values(["dataset", "method"])
    agg.to_csv(HERE / "results" / "performance_table.csv", index=False)
    (HERE / "performance_table.tex").write_text(make_tex(agg))
    (HERE / "performance_table_sd.tex").write_text(make_tex_sd(agg))

    # console preview
    show = ["overall_mean", "overall_sem", "overall_median",
            "wt_mean", "wt_sem", "wt_median", "overall_n"]
    pd.set_option("display.width", 160)
    for ds in ("depmap", "ctrpv2"):
        sub = (agg[agg.dataset == ds].set_index("method").loc[ROW_ORDER, show])
        print(f"\n=== {DS_LABELS[ds]} ===")
        print(sub.round(3).to_string())
    print("\nwrote results/performance_table.csv, performance_table.tex, "
          "performance_table_sd.tex")


if __name__ == "__main__":
    main()
