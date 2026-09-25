"""Render fig06_ci_distribution.{pdf,png}.

2x2 panel, one row per dataset (DepMap, CTRPv2):
    a/c  (left)  distribution of per-(item x gene) tissue-eta^2 of the per-cell
                 SHAP attribution, one histogram per method.
    b/d  (right) contamination@K: fraction of each method's top-K attributed
                 genes that are tissue markers (eta^2 of EXPRESSION > 0.30),
                 for K in {10,50,100}.

tissue-eta^2 (SS_between_tissue / SS_total of the per-cell phi) is defined
identically for every method, so marginal is the meaningful baseline; lower =
less tissue signal left in the attribution. contamination@K reads the genes
each method actually ranks at the top and asks whether they are over-enriched
for tissue markers.

Reads results/{depmap,ctrpv2}_attr_eta2.csv (from compute_attr_eta2.py) and
results/{depmap,ctrpv2}_contamination_at_k.csv (+ _meta.csv, from
compute_contamination_at_k.py).

Usage (from figure_ci_distribution/):
    python plot_ci_distribution.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent

PALETTE = {
    "red_strong":     "#B64342",
    "green_3":        "#8BCF8B",
    "blue_main":      "#0F4D92",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "neutral_mid":    "#767676",
}
METHOD_COLORS = {
    "marginal":      PALETTE["neutral_mid"],
    "residualized":  PALETTE["blue_main"],
    "irm":           PALETTE["violet"],
    "within_tissue": PALETTE["teal"],
    "dann":          PALETTE["red_strong"],
    "adae":          PALETTE["green_3"],
}
METHOD_LABELS = {
    "marginal":      "Marginal",
    "residualized":  "Residualized",
    "irm":           "IRM",
    "within_tissue": "Within-tissue",
    "dann":          "DANN",
    "adae":          "AD-AE",
}
METHODS = ["marginal", "residualized", "irm", "within_tissue",
           "dann", "adae"]
KS = [10, 50, 100]


def apply_publication_style(font_size: int = 8, axes_linewidth: float = 0.8) -> None:
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
        "axes.linewidth":    axes_linewidth,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "figure.dpi":        110,
        "savefig.dpi":       200,
    })


def _hist_panel(ax, df, item_label, title, xmax=0.4):
    """Distribution of per-(item x gene) tissue-eta^2, one step-hist per method."""
    for m in METHODS:
        sub = df[df["method"] == m]["eta2"].dropna()
        ax.hist(sub.clip(0, xmax), bins=80, alpha=0.55,
                histtype="step", linewidth=1.6,
                color=METHOD_COLORS.get(m, "k"),
                label=METHOD_LABELS.get(m, m))
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"tissue $\eta^2$ of attribution")
    ax.set_ylabel(f"# ({item_label} × gene) pairs")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right")


def _contam_panel(ax, res_dir, ds, title):
    """contamination@K grouped bars (K x method) with a chance baseline."""
    df = pd.read_csv(res_dir / f"{ds}_contamination_at_k.csv")
    chance = float(pd.read_csv(res_dir / f"{ds}_contamination_meta.csv")["chance"].iloc[0])
    x = np.arange(len(KS))
    # Derive the bar geometry from the method count. Hardcoding a 4-method
    # layout (w=0.2, centred on j-1.5) makes six methods span 1.2 units and
    # spill into the neighbouring K group.
    n = len(METHODS)
    w = 0.82 / n
    for j, m in enumerate(METHODS):
        sub = df[df["method"] == m]
        means = [sub[sub["K"] == k]["contamination"].mean() for k in KS]
        sems = [sub[sub["K"] == k]["contamination"].std()
                / np.sqrt(max(1, int((sub["K"] == k).sum()))) for k in KS]
        ax.bar(x + (j - (n - 1) / 2) * w, means, w * 0.9, yerr=sems, capsize=2,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               error_kw={"lw": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in KS])
    ax.set_ylabel(r"contamination@K")
    ax.set_title(title)


def plot(res_dir: Path, out_dir: Path, stem: str):
    apply_publication_style(font_size=10)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    e_depmap = pd.read_csv(res_dir / "depmap_attr_eta2.csv")
    e_ctrpv2 = pd.read_csv(res_dir / "ctrpv2_attr_eta2.csv")

    _hist_panel(axes[0, 0], e_depmap, "target",
                r"a  Attribution tissue-$\eta^2$ (DepMap)")
    _contam_panel(axes[0, 1], res_dir, "depmap",
                  r"b  Contamination@K (DepMap)")
    _hist_panel(axes[1, 0], e_ctrpv2, "drug",
                r"c  Attribution tissue-$\eta^2$ (CTRPv2)")
    _contam_panel(axes[1, 1], res_dir, "ctrpv2",
                  r"d  Contamination@K (CTRPv2)")

    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / (stem + '.pdf')}")
    print(f"wrote {out_dir / (stem + '.png')}")


# ---------------------------------------------------------------------------
# Single-column-optimized variant. Drawn at the *final* physical width so the
# fonts/lines are absolute-sized for the page (no post-shrink). Layout:
# column headers (measure) + rotated row labels (dataset) + one shared legend.
# ---------------------------------------------------------------------------
def _hist_panel_compact(ax, df, xmax=0.25, lw=1.7, bins=50):
    # Smooth density per method. tissue-eta^2 is bounded at 0 and the mode sits
    # ON that boundary, so a plain Gaussian KDE would leak mass below 0 and blunt
    # the peak. We boundary-correct by reflection at 0 (f(x) = k(x) + k(-x)) and
    # rescale to a "# pairs (x10^3)" count axis
    # (count-per-bin ~ N * density * binwidth), so the y-values stay comparable.
    binwidth = xmax / bins
    grid = np.linspace(0, xmax, 400)
    for m in METHODS:
        sub = df[df["method"] == m]["eta2"].dropna().to_numpy()
        if sub.size < 2:
            continue
        kde = gaussian_kde(sub)                     # bandwidth: Scott's rule
        dens = kde(grid) + kde(-grid)               # reflect at 0 -> no leakage <0
        y = dens * sub.size * binwidth * 1e-3       # density -> smoothed counts (x10^3)
        ax.plot(grid, y, color=METHOD_COLORS.get(m, "k"), linewidth=lw)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_xlabel(r"$\eta^2_{\mathrm{attr}}$")
    ax.set_ylabel(r"# pairs ($\times10^3$)")
    ax.margins(y=0.02)


def _contam_panel_compact(ax, res_dir, ds, lw=1.0):
    df = pd.read_csv(res_dir / f"{ds}_contamination_at_k.csv")
    x = np.arange(len(KS))
    n = len(METHODS)
    slot = 0.94 / n       # spacing between method bars, scaled to their number
    bw = slot * 0.87      # bar width (< slot -> a little gap between bars)
    for j, m in enumerate(METHODS):
        sub = df[df["method"] == m]
        means = [sub[sub["K"] == k]["contamination"].mean() for k in KS]
        sems = [sub[sub["K"] == k]["contamination"].std()
                / np.sqrt(max(1, int((sub["K"] == k).sum()))) for k in KS]
        ax.bar(x + (j - (n - 1) / 2) * slot, means, bw, yerr=sems, capsize=1.5,
               color=METHOD_COLORS[m], linewidth=0, error_kw={"lw": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K")
    ax.set_ylabel("contamination@K")
    ax.set_ylim(0, None)


def plot_row(res_dir: Path, out_dir: Path, stem: str):
    """Single-row figure: all four panels in one row, ~7in wide.

    Layout: [DepMap eta^2 | DepMap contam@K]  gap  [CTRPv2 eta^2 | CTRPv2
    contam@K]. constrained_layout + an *outside* top legend pack everything
    tight; the thin spacer column visually groups each dataset's pair, and the
    dataset name is folded into the left panel's letter. Fonts/lines are sized
    for the final width (no post-shrink).
    """
    apply_publication_style(font_size=8, axes_linewidth=0.9)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5 slots: hist, contam, spacer, hist, contam
    fig, axs = plt.subplots(1, 5, figsize=(9.6, 1.5), layout="constrained",
                            gridspec_kw={"width_ratios": [1.3, 1.0, 0.18, 1.3, 1.0]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05)
    axs[2].set_visible(False)          # spacer between the two dataset pairs
    axes = [axs[0], axs[1], axs[3], axs[4]]

    e_depmap = pd.read_csv(res_dir / "depmap_attr_eta2.csv")
    e_ctrpv2 = pd.read_csv(res_dir / "ctrpv2_attr_eta2.csv")

    _hist_panel_compact(axes[0], e_depmap)
    _contam_panel_compact(axes[1], res_dir, "depmap")
    _hist_panel_compact(axes[2], e_ctrpv2)
    _contam_panel_compact(axes[3], res_dir, "ctrpv2")

    # measure is conveyed by the axis labels; dataset is folded into the left
    # panel's letter and the two pairs are separated by the spacer column.
    panel_labels = ["a   DepMap", "b", "c   CTRPv2", "d"]
    for ax, lab in zip(axes, panel_labels):
        ax.text(-0.02, 1.03, lab, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="bottom", ha="left")

    # shared legend as a column to the right of the panel row
    handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=2.4,
                      label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="outside center right", ncol=1, fontsize=8,
               frameon=False, handlelength=1.0, handletextpad=0.4,
               labelspacing=0.9)

    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    print(f"wrote {out_dir / (stem + '.pdf')}")
    print(f"wrote {out_dir / (stem + '.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res-dir", type=Path, default=HERE / "results",
                   help="Directory with {depmap,ctrpv2}_attr_eta2.csv and "
                        "_contamination_at_k.csv (+ _meta.csv).")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures")
    p.add_argument("--stem", default="fig06_ci_distribution")
    p.add_argument("--row", action="store_true",
                   help="Single-row layout: all four panels in one row "
                        "(~7in wide), fonts/lines sized for the page.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.row:
        stem = args.stem if args.stem != "fig06_ci_distribution" else "fig06_ci_distribution_singlerow"
        plot_row(args.res_dir, args.out_dir, stem)
    else:
        plot(args.res_dir, args.out_dir, args.stem)
