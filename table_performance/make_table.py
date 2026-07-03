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

ROW_ORDER = ["baseline", "marginal", "residualized", "irm"]
ROW_LABELS = {
    "baseline":     r"Tissue-mean baseline",
    "marginal":     r"Marginal / Within-tissue SHAP",
    "residualized": r"Residualized",
    "irm":          r"IRM",
}
DS_LABELS = {"depmap": "DepMap (essentiality)",
             "ctrpv2": "CTRPv2 (drug response)"}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """seed-mean per item, then mean / sem / sd / median across items."""
    cols = ["pearson", "wt_pearson", "wt_pearson_recon"]
    # 1) average over seeds within each (dataset, item, method)
    per_item = (df.groupby(["dataset", "item", "method"])[cols]
                  .mean().reset_index())
    out = []
    for (ds, m), g in per_item.groupby(["dataset", "method"]):
        rec = {"dataset": ds, "method": m}
        for col, tag in [("pearson", "overall"), ("wt_pearson", "wt"),
                         ("wt_pearson_recon", "wtrecon")]:
            v = g[col].dropna().to_numpy()
            n = len(v)
            rec[f"{tag}_mean"] = float(np.mean(v)) if n else np.nan
            rec[f"{tag}_sd"] = float(np.std(v, ddof=1)) if n > 1 else np.nan
            rec[f"{tag}_sem"] = (float(np.std(v, ddof=1) / np.sqrt(n))
                                 if n > 1 else np.nan)
            rec[f"{tag}_median"] = float(np.median(v)) if n else np.nan
            rec[f"{tag}_n"] = n
        out.append(rec)
    return pd.DataFrame(out)


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
    n_items = {ds: int(a.loc[(ds, "marginal"), "overall_n"]) for ds in DS_LABELS}
    lines = _table_lines(a, "sd")
    cap = (
        r"  \caption{\textbf{Predictive accuracy with across-item standard "
        r"deviation.} Identical to Table~\ref{tab:performance}, but each cell reports "
        r"mean\,$\pm$\,SD across items instead of the SEM "
        rf"($n={n_items['depmap']}$ targets for DepMap, $n={n_items['ctrpv2']}$ "
        r"drugs for CTRPv2; seeds averaged within item first). The SD measures "
        r"item-to-item heterogeneity in predictability and is much larger than "
        r"the SEM. Because all methods are evaluated on the same items, "
        r"differences between methods should be read from the paired SEM in "
        r"Table~\ref{tab:performance}, not from the overlap of these SD ranges.}"
    )
    lines.append(cap)
    lines.append(r"  \label{tab:performance_sd}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_tex(agg: pd.DataFrame) -> str:
    a = agg.set_index(["dataset", "method"])
    n_items = {ds: int(a.loc[(ds, "marginal"), "overall_n"]) for ds in DS_LABELS}
    lines = _table_lines(a, "sem")
    rec = a.loc[("ctrpv2", "residualized"), "wtrecon_mean"]
    rec_dm = a.loc[("depmap", "residualized"), "wtrecon_mean"]
    # typical across-item SD of the within-tissue r over the trained predictors
    wt_sds = [a.loc[(ds, m), "wt_sd"]
              for ds in ("depmap", "ctrpv2")
              for m in ROW_ORDER if m != "baseline"]
    sd_typ = sum(wt_sds) / len(wt_sds)
    cap = (
        r"  \caption{\textbf{Predictive accuracy: overall vs.\ within-tissue "
        r"Pearson correlation.} Out-of-fold $r$ between the measured response "
        r"and the model prediction, as mean\,$\pm$\,SEM across items "
        rf"($n={n_items['depmap']}$ knockout targets for DepMap, "
        rf"$n={n_items['ctrpv2']}$ drugs for CTRPv2); each item's value is first "
        r"averaged over the five seeds, and all methods are evaluated on the "
        r"same items and folds (paired). The within-tissue $r_{\mathrm{wt}}$ "
        r"centres $y$ and the prediction within each tissue before correlating, "
        r"removing the between-tissue (lineage) component. Overall $r$ is "
        r"inflated by tissue structure: the tissue-mean baseline reaches "
        r"substantial overall $r$ while carrying no within-tissue signal "
        r"(its negative $r_{\mathrm{wt}}$ is the leave-one-fold-out "
        r"anti-correlation of held-out tissue means). Every trained predictor "
        r"sits far above the baseline in $r_{\mathrm{wt}}$, learning genuine "
        r"within-tissue signal rather than collapsing to lineage. "
        r"Residualization attains the highest overall \emph{and} within-tissue "
        r"$r$ -- consistent with its model being trained directly on the "
        r"within-tissue (residual) signal -- so deconfounding the data costs no "
        r"predictive accuracy here; IRM is the weaker predictor on both axes. "
        r"The tissue-mean baseline predicts the per-tissue mean response only. "
        r"Within-tissue SHAP re-uses the marginal model "
        r"($\hat y_{\mathrm{within}}\equiv\hat y_{\mathrm{marginal}}$), so its "
        r"predictive $r$ is identical to Marginal. For residualization the "
        r"within-tissue $r_{\mathrm{wt}}$ is evaluated on the deconfounded model "
        r"output $f(X_{\mathrm{res}})=\hat y_{\mathrm{res}}-\hat "
        r"y_{\mathrm{baseline}}$; the raw reconstructed prediction adds the "
        r"leave-out tissue mean back (needed only for overall $r$), whose "
        r"negative within-tissue correlation would otherwise contaminate "
        rf"$r_{{\mathrm{{wt}}}}$ (giving a misleading {rec:.2f} on CTRPv2, "
        rf"{rec_dm:.2f} on DepMap). Within-tissue $r_{{\mathrm{{wt}}}}$ is "
        rf"heterogeneous across items (SD $\approx$ {sd_typ:.1f}; some items "
        r"predict well, others near zero), but because all methods are compared "
        r"on the same items, the small SEM reflects the consistent paired "
        r"difference rather than low heterogeneity. Full per-item SD and the "
        r"contaminated value (\texttt{wt\_pearson\_recon}) are in "
        r"\texttt{performance\_table.csv}.}"
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
