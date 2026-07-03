"""Render fig_breast_subtypes.{pdf,png} from subtype_tissue_specificity.csv.

Two panels:
  a — per-method breast-vs-non-epithelial specificity gap (single number).
  b — per-tissue PAM50-lineage AUROC for all four methods, sorted by
      Residualized.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
METHODS = ["marginal", "residualized", "irm", "within_tissue"]
COLORS = {"marginal": "#9aa0a6", "residualized": "#1f77b4",
          "irm": "#a51e63", "within_tissue": "#2ca02c"}
LABELS = {"marginal": "Marginal", "residualized": "Residualized",
          "irm": "IRM", "within_tissue": "Within-tissue"}
NON_EPI = ["Blood", "Lymph", "Muscle", "Bone", "Soft Tissue",
           "Nervous System", "Brain"]
MIN_DRUGS_PER_TISSUE = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=HERE / "results" / "subtype_tissue_specificity.csv")
    ap.add_argument("--out-stem", type=Path,
                    default=HERE / "figures" / "fig_breast_subtypes")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    means = (df.groupby(["tissue", "method"])
               .agg(auroc=("auroc", "mean"),
                    drugs=("auroc", "count"))
               .reset_index())
    means = means[means["drugs"] >= MIN_DRUGS_PER_TISSUE]
    pivot = means.pivot(index="tissue", columns="method", values="auroc")
    if "Breast" not in pivot.index:
        raise SystemExit("breast not in pivot — check tissue label spelling")

    non_epi = [t for t in NON_EPI if t in pivot.index]
    pivot_sorted = pivot.sort_values("residualized", ascending=False)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw=dict(width_ratios=[1, 2.5]))

    # ---- Panel a: per-method specificity gap
    for i, m in enumerate(METHODS):
        breast_auc = pivot.loc["Breast", m]
        non_epi_auc = pivot.loc[non_epi, m].mean()
        c = COLORS[m]
        axA.plot([i, i], [non_epi_auc, breast_auc], color=c, lw=3)
        axA.scatter([i], [breast_auc], color=c, s=110, zorder=3,
                    label="Breast" if i == 0 else None)
        axA.scatter([i], [non_epi_auc], color=c, s=110, facecolor="white",
                    edgecolor=c, lw=2.2, zorder=3,
                    label="Non-epithelial lineages" if i == 0 else None)
        gap = breast_auc - non_epi_auc
        axA.annotate(f"{gap:+.2f}", (i, breast_auc),
                     textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=12, color=c, fontweight="bold")
    axA.set_xticks(range(len(METHODS)))
    axA.set_xticklabels([LABELS[m] for m in METHODS], fontsize=11)
    axA.set_ylabel("PAM50 (non-proliferation) AUROC", fontsize=12)
    axA.set_title("a", loc="left", fontsize=15, fontweight="bold")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=2, fontsize=11, frameon=False,
               handletextpad=0.4, columnspacing=1.5)
    axA.spines[["top", "right"]].set_visible(False)
    axA.set_ylim(0.3, 0.78)

    # ---- Panel b: per-tissue trace
    for m in METHODS:
        axB.plot(range(len(pivot_sorted)), pivot_sorted[m],
                 color=COLORS[m], lw=2.2, marker="o", ms=7,
                 label=LABELS[m].replace("\n", " "))
    axB.set_xticks(range(len(pivot_sorted)))
    axB.set_xticklabels(pivot_sorted.index, rotation=45, ha="right",
                        fontsize=11)
    for i, t in enumerate(pivot_sorted.index):
        if t in non_epi:
            axB.axvspan(i - 0.5, i + 0.5, color="#dddddd", alpha=0.5, zorder=0)
        if t == "Breast":
            axB.axvspan(i - 0.5, i + 0.5, color="#c9e2f3", alpha=0.6, zorder=0)
            axB.get_xticklabels()[i].set_color("#1f77b4")
            axB.get_xticklabels()[i].set_fontweight("bold")
    axB.set_ylabel("PAM50 (non-proliferation) AUROC", fontsize=12)
    axB.set_title("b", loc="left", fontsize=15, fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
               ncol=4, fontsize=11, frameon=False,
               handletextpad=0.4, columnspacing=1.5)
    axB.spines[["top", "right"]].set_visible(False)
    ymin = float(pivot_sorted[METHODS].min().min())
    ymax = float(pivot_sorted[METHODS].max().max())
    axB.set_ylim(ymin - 0.04, ymax + 0.04)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.out_stem.with_suffix(".pdf")
    png = args.out_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
