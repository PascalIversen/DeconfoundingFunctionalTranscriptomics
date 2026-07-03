"""Build the small overall-MSE table (companion to performance_table.tex).

Input : results/per_item_mse.csv   (from compute_per_item_mse.py: per dataset x
        seed x item x method; columns mse [raw scale], wt_mse, mse_noaddback)
Output: results/performance_table_mse.csv   (full stats, every method)
        performance_table_mse.tex            (compact, paper-ready)

Same aggregation convention as make_table.py: seed-mean per item first (the 5
seeds are repeats of one item, not extra n), then mean / SEM across items, which
are the independent observations. All methods are scored on the same items and
folds (paired). MSE is the out-of-fold squared error on the RAW response scale,
so the residualized prediction is the reconstruction f(X_res)+baseline (the
tissue-mean term is added back only to put predictions on that scale -- required
for a meaningful raw-scale error; stripping it is a within-tissue quantity and
lives in per_item_mse.csv as wt_mse).
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
    """seed-mean per item, then mean / sem / sd / median across items (raw MSE)."""
    per_item = (df.groupby(["dataset", "item", "method"])["mse"]
                  .mean().reset_index())
    out = []
    for (ds, m), g in per_item.groupby(["dataset", "method"]):
        v = g["mse"].dropna().to_numpy()
        n = len(v)
        out.append({
            "dataset": ds, "method": m,
            "mse_mean": float(np.mean(v)) if n else np.nan,
            "mse_sd": float(np.std(v, ddof=1)) if n > 1 else np.nan,
            "mse_sem": float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "mse_median": float(np.median(v)) if n else np.nan,
            "mse_n": n,
        })
    return pd.DataFrame(out)


def fmt(mean: float, sem: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sem):
        return f"{mean:.3f}"
    return f"{mean:.3f}\\,$\\pm$\\,{sem:.3f}"


def make_tex(agg: pd.DataFrame) -> str:
    a = agg.set_index(["dataset", "method"])
    n_items = {ds: int(a.loc[(ds, "marginal"), "mse_n"]) for ds in DS_LABELS}
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l c c}",
        r"    \toprule",
        r"    Method & DepMap (essentiality) & CTRPv2 (drug response) \\",
        r"    \midrule",
    ]
    for m in ROW_ORDER:
        cells = [ROW_LABELS[m]]
        for ds in ("depmap", "ctrpv2"):
            r = a.loc[(ds, m)]
            cells.append(fmt(r["mse_mean"], r["mse_sem"]))
        lines.append("    " + " & ".join(cells) + r" \\")
        if m == "baseline":
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    cap = (
        r"  \caption{\textbf{Predictive accuracy: out-of-fold mean squared "
        r"error.} Companion to Table~\ref{tab:performance}; the same models, "
        r"items and folds, scored by MSE between the measured response and the "
        r"prediction (mean\,$\pm$\,SEM across items, "
        rf"$n={n_items['depmap']}$ knockout targets for DepMap, "
        rf"$n={n_items['ctrpv2']}$ drugs for CTRPv2; seeds averaged within item "
        r"first, all methods paired). MSE is on the raw response scale, so units "
        r"are squared and dataset-specific (Chronos gene-effect$^2$ for DepMap, "
        r"$\ln$IC$_{50}^{2}$ for CTRPv2) -- compare down a column, not across "
        r"datasets. The residualized prediction is reconstructed on the raw "
        r"scale as $\hat y_{\mathrm{res}}=f(X_{\mathrm{res}})+\hat "
        r"y_{\mathrm{baseline}}$; the tissue-mean term is added back so every "
        r"method is compared on the same scale (without it the residual output "
        r"is tissue-centred and its raw-scale MSE is meaningless). Within-tissue "
        r"SHAP re-uses the marginal model, so its MSE equals Marginal. "
        r"Residualization is the only trained predictor that beats the "
        r"tissue-mean baseline on MSE; marginal and IRM sit above the baseline "
        r"-- they correlate with the response (Table~\ref{tab:performance}) but "
        r"predict the raw scale less accurately than the per-tissue mean, a "
        r"calibration effect -- and IRM is the weakest on both datasets. "
        r"Per-item SD, within-tissue MSE and the no-add-back diagnostic are in "
        r"\texttt{performance\_table\_mse.csv}.}"
    )
    lines.append(cap)
    lines.append(r"  \label{tab:performance_mse}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_csv(HERE / "results" / "per_item_mse.csv")
    agg = aggregate(df).sort_values(["dataset", "method"])
    agg.to_csv(HERE / "results" / "performance_table_mse.csv", index=False)
    (HERE / "performance_table_mse.tex").write_text(make_tex(agg))

    show = ["mse_mean", "mse_sem", "mse_sd", "mse_median", "mse_n"]
    pd.set_option("display.width", 160)
    for ds in ("depmap", "ctrpv2"):
        sub = agg[agg.dataset == ds].set_index("method").loc[ROW_ORDER, show]
        print(f"\n=== {DS_LABELS[ds]} ===")
        print(sub.round(3).to_string())
    print("\nwrote results/performance_table_mse.csv, performance_table_mse.tex")


if __name__ == "__main__":
    main()
