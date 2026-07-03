"""Plotting helpers and a shared rcParams preset.

The palette and rcParams use Arial, editable SVG text, and no top/right spines.
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ====================================================== palette
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "gold":           "#FFD700",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "delta_up":       "#2E9E44",
    "delta_down":     "#E53935",
}

# One color per deconfounding method, kept consistent across notebooks.
METHOD_COLORS = {
    "marginal":      PALETTE["neutral_mid"],
    "residualized":  PALETTE["blue_main"],
    "irm":           PALETTE["violet"],
    "within_tissue": PALETTE["teal"],
}
METHOD_LABELS = {
    "marginal":      "Marginal",
    "residualized":  "Residualized",
    "irm":           "IRM",
    "within_tissue": "Within-\ntissue",
}
METHOD_ORDER = ["marginal", "residualized", "irm", "within_tissue"]


# ====================================================== rcParams preset
def apply_publication_style(font_size: int = 8,
                            axes_linewidth: float = 0.8) -> None:
    """Shared rcParams: editable SVG text, no top/right spines, Arial.
    Call once at the top of a notebook."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype":      "none",
        "pdf.fonttype":      42,
        "font.size":         font_size,
        "axes.titlesize":    font_size + 1,
        "axes.labelsize":    font_size,
        "xtick.labelsize":   font_size - 1,
        "ytick.labelsize":   font_size - 1,
        "legend.fontsize":   font_size - 1,
        "axes.spines.right": False,
        "axes.spines.top":   False,
        "axes.linewidth":    axes_linewidth,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.frameon":    False,
        "figure.dpi":        130,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


# ====================================================== utilities
def _style_box(bp, colors: List[str], alpha: float = 0.85) -> None:
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(alpha)
        patch.set_edgecolor(PALETTE["neutral_black"])
        patch.set_linewidth(0.6)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(PALETTE["neutral_black"])
        w.set_linewidth(0.6)
    for m in bp["medians"]:
        m.set_color(PALETTE["neutral_black"])
        m.set_linewidth(1.2)


def method_boxplot(ax, df: pd.DataFrame, metric: str,
                    ylabel: Optional[str] = None,
                    title: Optional[str] = None,
                    log_y: bool = False) -> None:
    """Standardised one-panel method comparison boxplot."""
    data = [df.loc[df["method"] == m, metric].dropna().values
            for m in METHOD_ORDER]
    bp = ax.boxplot(
        data, labels=[METHOD_LABELS[m] for m in METHOD_ORDER],
        patch_artist=True, widths=0.6,
        flierprops=dict(marker="o", markersize=2,
                        markerfacecolor=PALETTE["neutral_light"],
                        markeredgecolor=PALETTE["neutral_dark"],
                        alpha=0.6))
    _style_box(bp, [METHOD_COLORS[m] for m in METHOD_ORDER])
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", color=PALETTE["neutral_light"],
            linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def quartile_gain_boxplot(ax, target_table: pd.DataFrame,
                           gain_col: str = "gain",
                           quartile_col: str = "eta_y_q",
                           ylabel: str = "Rank gain",
                           title: Optional[str] = None) -> None:
    """Boxplot of `gain_col` per quartile (Q1..Q4)."""
    quartiles = list(target_table[quartile_col].cat.categories)
    data = [target_table.loc[target_table[quartile_col] == q,
                              gain_col].dropna().values
            for q in quartiles]
    # Diverging color: low η² gets the strongest "win" color
    cols = [PALETTE["delta_up"], PALETTE["blue_secondary"],
            PALETTE["neutral_light"], PALETTE["neutral_mid"]]
    bp = ax.boxplot(data, labels=quartiles, patch_artist=True, widths=0.6,
                    flierprops=dict(marker="o", markersize=2,
                                    markerfacecolor=PALETTE["neutral_light"],
                                    markeredgecolor=PALETTE["neutral_dark"],
                                    alpha=0.6))
    _style_box(bp, cols[:len(quartiles)])
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.6, ls="--")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", color=PALETTE["neutral_light"],
            linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def quartile_log_ratio_strip(ax, target_table: pd.DataFrame,
                              rank_a: str = "marginal",
                              rank_b: str = "residualized",
                              quartile_col: str = "eta_y_q",
                              ylabel: Optional[str] = None,
                              title: Optional[str] = None) -> None:
    """Per-target log10(rank_a / rank_b) by quartile. Positive = b is
    better than a by 10^value times. Strip + summary mean."""
    quartiles = list(target_table[quartile_col].cat.categories)
    cols = [PALETTE["delta_up"], PALETTE["blue_secondary"],
            PALETTE["neutral_light"], PALETTE["neutral_mid"]]
    rng = np.random.RandomState(0)
    for i, q in enumerate(quartiles):
        sub = target_table[target_table[quartile_col] == q]
        ratios = np.log10(sub[rank_a].astype(float) /
                          sub[rank_b].astype(float).replace(0, np.nan))
        ratios = ratios.dropna().values
        if len(ratios) == 0:
            continue
        x = i + 1 + (rng.rand(len(ratios)) - 0.5) * 0.35
        ax.scatter(x, ratios, s=14,
                   color=cols[i % len(cols)], alpha=0.55, edgecolors="none")
        ax.plot([i + 1 - 0.25, i + 1 + 0.25],
                [ratios.mean()] * 2, color=PALETTE["neutral_black"], lw=1.5)
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.6, ls="--")
    ax.set_xticks(range(1, len(quartiles) + 1))
    ax.set_xticklabels(quartiles)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", color=PALETTE["neutral_light"],
            linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
