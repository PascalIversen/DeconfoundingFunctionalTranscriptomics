"""Are the Fig-2 and Fig-3 results the same on the well-predicted subset?

2x3 panels, one row per dataset:
    a/d  mean attribution tissue-eta^2 per method, all items vs subset
    b/e  contamination@50 per method (mean +- sem), all items vs subset
    c/f  NES(all items) vs NES(subset) for every gene set of every deconfounder,
         with the Pearson r per method; the identity line is the "nothing changed"
         reference.

Hollow bars = all items (the published Fig 2), filled = subset. The point of the
figure is that the bar ordering and the scatter's diagonal are unchanged, i.e. the
conclusions do not depend on including the poorly-predicted items.

Reads results/fig02_subset_summary.csv, results/<variant>/pathway_delta_gsea.csv
and ../fig03_pathways/results/pathway_delta_gsea.csv.

    python plot_robustness.py                # variant a61 (the paper's A.6.1 filter)
    python plot_robustness.py --variant marginal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "fig02_ci_distribution"))
import plot_ci_distribution as P  # noqa: E402  (colours + house style)

METHODS = P.METHODS
DECONF = ["residualized", "irm", "within_tissue", "dann", "adae"]
DLAB = {"depmap": "DepMap (essentiality)", "ctrpv2": "CTRPv2 (drug response)"}
K = 50


def _paired_bars(ax, summary, dataset, col, ylabel, variant, sem_col=None):
    """Hollow (all items) vs filled (subset) bar per method."""
    x = np.arange(len(METHODS))
    for i, (v, filled) in enumerate([("full", False), (variant, True)]):
        s = summary[(summary.dataset == dataset) & (summary.variant == v)]
        s = s.set_index("method").reindex(METHODS)
        err = s[sem_col].values if sem_col else None
        ax.bar(x + (i - 0.5) * 0.4, s[col].values, 0.36,
               color=[P.METHOD_COLORS[m] for m in METHODS] if filled else "none",
               edgecolor=[P.METHOD_COLORS[m] for m in METHODS],
               linewidth=1.1, yerr=err, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([P.METHOD_LABELS[m] for m in METHODS], rotation=45,
                       ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)


def _nes_scatter(ax, full, sub, dataset):
    j = full.merge(sub, on=["dataset", "method", "library", "set"],
                   suffixes=("_f", "_s"))
    j = j[j.dataset == dataset]
    lims = [-3.0, 3.0]
    ax.plot(lims, lims, color="0.6", lw=0.7, zorder=0)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color="0.85", lw=0.5, zorder=0)
    txt = []
    for m in DECONF:
        g = j[j.method == m]
        if len(g) < 3:
            continue
        ax.scatter(g.NES_f, g.NES_s, s=3.2, alpha=0.55, linewidths=0,
                   color=P.METHOD_COLORS[m])
        txt.append(f"{P.METHOD_LABELS[m]} r={pearsonr(g.NES_f, g.NES_s)[0]:.2f}")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("NES, all items")
    ax.set_ylabel("NES, subset")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top",
            ha="left", fontsize=5.6, linespacing=1.35)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="a61")
    ap.add_argument("--stem", default=None)
    a = ap.parse_args()
    stem = a.stem or f"fig_robustness_{a.variant}"

    summary = pd.read_csv(HERE / "results" / "fig02_subset_summary.csv")
    full = pd.read_csv(REPO / "fig03_pathways" / "results" / "pathway_delta_gsea.csv")
    sub_csv = HERE / "results" / a.variant / "pathway_delta_gsea.csv"
    sub = pd.read_csv(sub_csv) if sub_csv.exists() else None
    if sub is None:
        print(f"note: {sub_csv} missing - GSEA panels left empty")

    P.apply_publication_style(font_size=7, axes_linewidth=0.8)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    letters = "abcdef"
    for r, ds in enumerate(["depmap", "ctrpv2"]):
        n_sub = int(summary[(summary.dataset == ds)
                            & (summary.variant == a.variant)]["n_items"].iloc[0])
        n_full = int(summary[(summary.dataset == ds)
                             & (summary.variant == "full")]["n_items"].iloc[0])
        _paired_bars(axes[r, 0], summary, ds, "eta2_mean",
                     r"mean tissue $\eta^2_{\mathrm{attr}}$", a.variant)
        _paired_bars(axes[r, 1], summary, ds, f"contam@{K}",
                     f"contamination@{K}", a.variant, sem_col=f"contam@{K}_sem")
        if sub is not None:
            _nes_scatter(axes[r, 2], full, sub, ds)
        # Row header folded into the first panel's letter (as in the paper's
        # Fig 2) so it cannot collide with the middle panel's letter.
        for c in range(3):
            axes[r, c].text(-0.26, 1.06, letters[r * 3 + c],
                            transform=axes[r, c].transAxes, fontsize=9,
                            fontweight="bold", va="bottom", ha="left")
        axes[r, 0].text(-0.18, 1.07, f"{DLAB[ds]}"
                        f"      all items n={n_full},  subset n={n_sub}",
                        transform=axes[r, 0].transAxes, fontsize=7.5,
                        va="bottom", ha="left", clip_on=False)

    handles = [Patch(facecolor="none", edgecolor="0.25", label="all items"),
               Patch(facecolor="0.25", edgecolor="0.25", label=f"subset ({a.variant})"),
               Line2D([0], [0], color="0.6", lw=0.7, label="identity")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=7)
    out = HERE / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out / (stem + '.pdf')}")
    print(f"wrote {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
