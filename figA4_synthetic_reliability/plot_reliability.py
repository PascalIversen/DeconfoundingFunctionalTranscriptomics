"""Reliability figure: marginal SHAP is a lottery; residualization always wins.

Two panels per figure, one figure per focal K in K_VALUES:
  a) Top-K class composition per method at the focal K (varies per figure):
     mean +/- SEM across the seeds at n=1000.
  b) K-INDEPENDENT trajectory: confounders in top-K vs K for K in 1..500,
     four method lines with +/- SD bands across seeds. Identical across
     figures - shows that the asymmetry holds at every K, not just the
     one selected for panel a.

Inputs (this folder's results/, no re-run needed):
  results/npower_mixed.csv               -- 8 base seeds (all n on N_GRID)
  results/npower_mixed_extra_n1000.csv   -- 24 extra seeds at n=1000, same DGP
Regenerate the inputs with compute_npower_mixed.py (slow, GPU).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC_BASE = HERE / "results" / "npower_mixed.csv"
SRC_EXTRA = HERE / "results" / "npower_mixed_extra_n1000.csv"   # extra seeds, same DGP
FIG = HERE / "figures"
N_FOCUS = 1000
K_VALUES = [10, 15, 20, 30, 50]


def load_merged() -> pd.DataFrame:
    base = pd.read_csv(SRC_BASE)
    base = base[base.n_cells == N_FOCUS]
    if SRC_EXTRA.exists():
        extra = pd.read_csv(SRC_EXTRA)
        df = pd.concat([base, extra], ignore_index=True)
    else:
        df = base
    return df

METHODS = ["marginal", "residualized", "within_tissue", "irm"]
LAB = {"marginal": "Marginal", "residualized": "Residualized (FWL)",
       "within_tissue": "Within-tissue", "irm": "IRM"}
PALETTE = {"marginal": "#767676", "residualized": "#3775BA",
           "within_tissue": "#42949E", "irm": "#9A4D8E"}

CLASSES = ["causal", "mixed", "confounder", "noise"]
CLASS_COL = {"causal": "#2E7D32", "mixed": "#F39C12",
             "confounder": "#C0392B", "noise": "#D5D5D5"}
CLASS_LAB = {"causal": "Causal", "mixed": "Mixed",
             "confounder": "Confounder", "noise": "Noise"}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42, "font.size": 13,
        "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 200,
    })


def class_counts(df_msn: pd.DataFrame, k: int) -> dict:
    """Count {causal, mixed, confounder, noise} in top-k for one (method, seed)
    slice. Ranks missing from the CSV are noise genes (the DGP carries 150 inert
    genes that aren't tracked by class)."""
    by_rank = {int(r["rank"]): r["gene_class"]
               for _, r in df_msn.iterrows() if 1 <= int(r["rank"]) <= k}
    counts = {c: 0 for c in CLASSES}
    for rk in range(1, k + 1):
        counts[by_rank.get(rk, "noise")] += 1
    return counts


def panel_a(ax, df: pd.DataFrame, k: int) -> None:
    seeds = sorted(df.seed.unique())
    per_seed = {m: [class_counts(
        df[(df.method == m) & (df.seed == s)], k)
        for s in seeds] for m in METHODS}

    width = 0.18
    centers = np.arange(len(METHODS))
    offsets = np.linspace(-(width * (len(CLASSES) - 1) / 2),
                           (width * (len(CLASSES) - 1) / 2), len(CLASSES))
    n_seeds = df.seed.nunique()
    for ci, c in enumerate(CLASSES):
        means = [np.mean([d[c] for d in per_seed[m]]) for m in METHODS]
        sems = [np.std([d[c] for d in per_seed[m]], ddof=1) / np.sqrt(n_seeds)
                for m in METHODS]
        ax.bar(centers + offsets[ci], means, width * 0.92, color=CLASS_COL[c],
               edgecolor="white", lw=0.5, label=CLASS_LAB[c],
               yerr=sems, capsize=2.5,
               error_kw={"elinewidth": 0.9, "ecolor": "#444"})

    ax.set_xticks(centers)
    ax.set_xticklabels([LAB[m] for m in METHODS])
    ax.set_ylabel(f"Genes in top-{k}  (mean ± SEM, n = {n_seeds} seeds)")
    ax.set_ylim(0, k + 1)
    ax.set_title(f"a   Class composition of top-{k}", loc="left",
                 fontweight="bold", pad=6)
    ax.legend(loc="upper right", frameon=False, ncol=2)


K_TRAJECTORY = list(range(1, 501))   # full 500-gene panel


def panel_b(ax, df: pd.DataFrame, focal_k: int) -> None:
    """K-independent trajectory: mean +/- SD confounders in top-K vs K."""
    seeds = sorted(df.seed.unique())
    sub_n = df

    n_seeds = len(seeds)
    for m in METHODS:
        per_seed = np.zeros((n_seeds, len(K_TRAJECTORY)))
        for si, s in enumerate(seeds):
            ranks = sub_n[(sub_n.method == m) & (sub_n.seed == s)
                          & (sub_n.gene_class == "confounder")]["rank"].values
            for ki, k in enumerate(K_TRAJECTORY):
                per_seed[si, ki] = int((ranks <= k).sum())
        mean = per_seed.mean(0)
        sd = per_seed.std(0, ddof=1)
        ax.fill_between(K_TRAJECTORY, mean - sd, mean + sd,
                         color=PALETTE[m], alpha=0.14, lw=0)
        ax.plot(K_TRAJECTORY, mean, color=PALETTE[m], lw=1.9, label=LAB[m])

    ax.set_xlim(1, K_TRAJECTORY[-1])
    ax.set_xticks([1, 50, 100, 200, 300, 400, 500])
    ax.set_xlabel("K  (top-K SHAP ranks)")
    ax.set_ylabel(f"Confounders in top-K  (mean ± SD across {n_seeds} seeds)")
    ax.set_title("b   Confounder accumulation across K",
                 loc="left", fontweight="bold", pad=6)
    ax.legend(loc="upper left", frameon=False)


def render(df: pd.DataFrame, k: int) -> Path:
    fig = plt.figure(figsize=(13.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    panel_a(fig.add_subplot(gs[0, 0]), df, k)
    panel_b(fig.add_subplot(gs[0, 1]), df, k)
    fig.subplots_adjust(top=0.94, bottom=0.14, left=0.07, right=0.985)

    FIG.mkdir(exist_ok=True)
    out_png = FIG / f"fig_reliability_top{k}.png"
    out_pdf = FIG / f"fig_reliability_top{k}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    style()
    df = load_merged()
    print(f"loaded {df.seed.nunique()} seeds at n={N_FOCUS}")
    for k in K_VALUES:
        print("wrote", render(df, k))


if __name__ == "__main__":
    main()
