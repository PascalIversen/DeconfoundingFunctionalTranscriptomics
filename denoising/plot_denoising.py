"""Figure for the denoising response: does averaging per-cell attributions
denoise the global explanation, and does that depend on within-tissue r?

Panels (rows = datasets):
  a) split-half stability of the mean-|SHAP| vector vs number of cells
     averaged (solid), with the Spearman-Brown prophecy from the single-cell
     reliability r1 (dotted) — matching curves = cell-to-cell variation
     behaves as independent noise that averaging removes.
  b) per-item stability at n = n_cells/2 vs the item's within-tissue r —
     is the explanation only stable where the model predicts well?
  c) split-half reliability of the Fig-3 GSEA input (the global delta
     percentile vector) per deconfounding method.

Style follows the paper's fig02 plotting module (palette, labels, rc).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
FIG = HERE / "figures"


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P2 = _import(ROOT / "fig02_ci_distribution" / "plot_ci_distribution.py", "p2_mod")
COLORS, LABELS = P2.METHOD_COLORS, P2.METHOD_LABELS
METHODS = ["marginal", "residualized", "irm", "within_tissue", "dann", "adae"]
DECONF = ["residualized", "irm", "within_tissue", "dann", "adae"]
DS_LABEL = {"depmap": "DepMap", "ctrpv2": "CTRPv2"}


def sb(n, r1):
    return n * r1 / (1.0 + (n - 1.0) * r1)


def panel_curves(ax, curves, items, ds):
    d = curves[curves.dataset == ds]
    it = items[items.dataset == ds]
    # common grid: n where EVERY item still contributes, so the mean is not
    # distorted by item composition (CTRPv2 drugs have 132-790 cells)
    min_half = int(d.groupby("item")["n_cells"].first().min()) // 2
    med_half = int(d.groupby("item")["n_cells"].first().median()) // 2
    pow2 = [2 ** k for k in range(10)]
    for m in METHODS:
        g = d[(d.method == m) & (d.n <= min_half) & d.n.isin(pow2)]
        if g.empty:
            continue
        mc = g.groupby("n")["pearson"].mean()
        ax.plot(mc.index, mc.values, color=COLORS[m], lw=1.6,
                label=LABELS[m], zorder=3)
        # end point: each item split-half at its own n_cells//2
        end = it[it.method == m].rhalf_pearson.mean()
        ax.plot([med_half], [end], marker="o", ms=3.5, color=COLORS[m],
                zorder=4, clip_on=False)
        r1 = mc.loc[1]
        ns = np.geomspace(1, med_half, 64)
        ax.plot(ns, sb(ns, r1), color=COLORS[m], lw=0.9, ls=":", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells averaged $n$")
    ax.set_ylabel("split-half stability $r$")
    ax.set_ylim(0, 1.02)
    ax.axhline(1.0, color="0.85", lw=0.6, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def panel_scatter(ax, items, ds):
    d = items[items.dataset == ds]
    shown = ["marginal", "residualized"]
    for m in shown:
        g = d[(d.method == m)].dropna(subset=["r_wt"])
        ax.scatter(g.r_wt, g.rhalf_spearman, s=8, color=COLORS[m], alpha=0.45,
                   linewidths=0, label=LABELS[m], zorder=3)
    ax.axvline(0, color="0.85", lw=0.6, zorder=1)
    ax.set_xlabel("within-tissue $r$ of the model (per item)")
    ax.set_ylabel("split-half stability\nof its explanation")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)


def panel_delta(ax, delta, ds):
    d = delta[delta.dataset == ds].set_index("method").reindex(DECONF)
    x = np.arange(len(DECONF))
    ax.bar(x, d.pearson, width=0.62, color=[COLORS[m] for m in DECONF], zorder=3)
    for xi, v in zip(x, d.pearson):
        if np.isfinite(v):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in DECONF], rotation=35, ha="right")
    ax.set_ylabel("split-half $r$ of the\nGSEA $\\delta$ vector")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="0.85", lw=0.6, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    P2.apply_publication_style()
    curves = pd.read_csv(RES / "stability_curves.csv")
    items = pd.read_csv(RES / "item_stability.csv")
    delta = pd.read_csv(RES / "delta_split_half.csv")
    FIG.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6))
    for i, ds in enumerate(["depmap", "ctrpv2"]):
        panel_curves(axes[i, 0], curves, items, ds)
        panel_scatter(axes[i, 1], items, ds)
        panel_delta(axes[i, 2], delta, ds)
        axes[i, 0].set_title(DS_LABEL[ds], loc="left", fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h, l, frameon=False, fontsize=6, loc="lower right",
                      handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    axes[0, 0].text(0.05, 0.9, "dotted: Spearman–Brown from $r_1$",
                    transform=axes[0, 0].transAxes, fontsize=6, color="0.35")
    fig.tight_layout()
    for panel, ax in zip("abcdef", axes.flat):
        ax.text(-0.22, 1.05, panel, transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    fig.savefig(FIG / "fig_denoising.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_denoising.png", bbox_inches="tight", dpi=300)
    print(f"wrote {FIG / 'fig_denoising.pdf'}")


if __name__ == "__main__":
    main()
