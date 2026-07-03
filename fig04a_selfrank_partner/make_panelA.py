"""Self-contained 'panel A': self/target rank + functional-partner recovery.

Two independent ground truths (STRING 1-hop, CORUM complexes) plus a RANDOM control,
for BOTH datasets with the same layout:
  - DepMap CRISPR        (target = the knocked-out gene)
  - CTRPv2 drug response (target = the drug's known target gene)
Bars = paired Δ vs Marginal (mean of method_i − marginal_i across items in the
BH-FDR intersection). Error bars = SEM of that paired delta. Stars = paired
Wilcoxon vs Marginal, q-values from Benjamini–Hochberg across all tests in the
panel (3 self-rank + 3 GTs × 3 methods = 12 per dataset). Marginal absolute
AUROC is annotated under each group so the reader can reconstruct the absolute
value. Reads pre-computed summaries in data/, writes figures/.

Usage (from fig04a_selfrank_partner/):
    python make_panelA.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"

PALETTE = {
    "blue_secondary": "#3775BA",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}
COL = {"residualized": PALETTE["blue_secondary"],
       "within_tissue": PALETTE["teal"],
       "irm": PALETTE["violet"]}
LAB = {"residualized": "Residualized",
       "within_tissue": "Within-tissue",
       "irm": "IRM"}
METHODS3 = ["residualized", "within_tissue", "irm"]

GT_ORDER = ["string", "CORUM", "RANDOM"]
GT_LAB = {"string": "STRING (1-hop)", "CORUM": "CORUM complexes",
          "RANDOM": "Random (control)"}


def apply_publication_style(font_size: int = 9, axes_linewidth: float = 0.8) -> None:
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
        "figure.dpi":        110,
        "savefig.dpi":       200,
    })


def stars(p):
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def make(self_summ, part_summ, outstem, set_filter: str = "INTER"):
    self_summ = self_summ[self_summ.set == set_filter]
    part_summ = part_summ[part_summ.set == set_filter]
    sr_marg = self_summ["marg_mean"].iloc[0]

    gts_used = [g for g in GT_ORDER if g in set(part_summ.ground_truth)]
    keys, p_raw = [], []
    for m in METHODS3:
        r = self_summ[self_summ.method == m]
        if len(r):
            keys.append(("self", None, m)); p_raw.append(r["wilcoxon_p"].iloc[0])
    for g in gts_used:
        for m in METHODS3:
            r = part_summ[(part_summ.ground_truth == g) & (part_summ.method == m)]
            if len(r):
                keys.append(("part", g, m)); p_raw.append(r["wilcoxon_p"].iloc[0])
    p_arr = np.array(p_raw, dtype=float)
    finite = ~np.isnan(p_arr)
    q_arr = np.full_like(p_arr, np.nan)
    if finite.any():
        _, q_finite, _, _ = multipletests(p_arr[finite], method="fdr_bh")
        q_arr[finite] = q_finite
    q_by = {k: q_arr[i] for i, k in enumerate(keys)}

    fig_w = 8.6 + 0.7 * max(0, len(gts_used) - 3)
    fig = plt.figure(figsize=(fig_w, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.32, 1.0], wspace=0.30)
    axS = fig.add_subplot(gs[0, 0])
    axP = fig.add_subplot(gs[0, 1])

    # ---- self / target rank: Δ vs Marginal ----
    deltas, dses, ps = [], [], []
    for m in METHODS3:
        r = self_summ[self_summ.method == m]
        deltas.append(r["delta"].iloc[0] if len(r) else np.nan)
        dses.append(r["delta_sem"].iloc[0] if (len(r) and "delta_sem" in r.columns) else np.nan)
        ps.append(q_by.get(("self", None, m), np.nan))
    xs = np.arange(len(METHODS3))
    axS.bar(xs, deltas, 0.7, color=[COL[m] for m in METHODS3],
            yerr=dses, capsize=3.0,
            error_kw={"elinewidth": 1.1, "ecolor": PALETTE["neutral_dark"]})
    axS.axhline(0.0, color=PALETTE["neutral_dark"], lw=0.7, zorder=0)
    span = max(max(abs(v) + (0 if pd.isna(se) else se) for v, se in zip(deltas, dses)), 0.02)
    axS.set_ylim(-span * 1.4, span * 1.7)
    for xi, v, se, p in zip(xs, deltas, dses, ps):
        if pd.notna(v):
            e = 0.0 if pd.isna(se) else se
            top = max(v + e, e)
            axS.text(xi, top + 0.004, stars(p), ha="center", va="bottom",
                     fontsize=8, color=PALETTE["neutral_dark"])
    axS.set_xticks(xs)
    axS.set_xticklabels(["Resid", "Within", "IRM"], fontsize=8)
    axS.set_xlabel(f"Self / target rank\nmarg AUROC = {sr_marg:.3f}",
                   fontsize=8.2, labelpad=8)
    axS.set_ylabel("Δ Self-recovery AUROC  (vs Marginal)", fontsize=9)

    # ---- partner recovery: Δ vs Marginal, grouped by ground truth ----
    gts = gts_used
    x = np.arange(len(gts))
    w = 0.26
    marg_per_gt = {g: part_summ[part_summ.ground_truth == g]["marg_mean"].iloc[0]
                   for g in gts}

    all_lo, all_hi = 0.0, 0.0
    for j, m in enumerate(METHODS3):
        deltas, dses, ps = [], [], []
        for g in gts:
            r = part_summ[(part_summ.ground_truth == g) & (part_summ.method == m)]
            deltas.append(r["delta"].iloc[0] if len(r) else np.nan)
            dses.append(r["delta_sem"].iloc[0] if (len(r) and "delta_sem" in r.columns) else np.nan)
            ps.append(q_by.get(("part", g, m), np.nan))
        xs = x + (j - 1) * w
        axP.bar(xs, deltas, w, color=COL[m], label=LAB[m],
                yerr=dses, capsize=2.5,
                error_kw={"elinewidth": 1.0, "ecolor": PALETTE["neutral_dark"]})
        for xi, v, se, p in zip(xs, deltas, dses, ps):
            if pd.notna(v):
                e = 0.0 if pd.isna(se) else se
                top = max(v + e, e); bot = min(v - e, -e if v < 0 else 0)
                axP.text(xi, top + 0.0018, stars(p), ha="center", va="bottom",
                         fontsize=7.5, color=PALETTE["neutral_dark"])
                all_hi = max(all_hi, top)
                all_lo = min(all_lo, bot)
    axP.axhline(0.0, color=PALETTE["neutral_dark"], lw=0.7, zorder=0)
    pad = max(all_hi - all_lo, 0.02) * 0.20
    axP.set_ylim(all_lo - pad, all_hi + pad * 1.6)
    axP.set_xticks(x)
    axP.set_xticklabels(
        [f"{GT_LAB[g]}\nmarg AUROC = {marg_per_gt[g]:.3f}" for g in gts],
        fontsize=8.2)
    axP.set_ylabel("Δ Partner-recovery AUROC  (vs Marginal)", fontsize=9)
    axP.legend(fontsize=7.5, ncol=3, loc="upper center", framealpha=0.95)

    fig.subplots_adjust(top=0.94, bottom=0.18, left=0.085, right=0.985)

    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{outstem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{outstem}.png", dpi=170, bbox_inches="tight")
    print("wrote", FIG / f"{outstem}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="INTER", choices=["INTER", "ALL"],
                    help="Item subset: INTER = BH-FDR intersection (default), "
                         "ALL = no significance filter")
    args = ap.parse_args()
    suffix = "" if args.set == "INTER" else "_noBH"
    apply_publication_style(font_size=9)
    make(pd.read_csv(DATA / "selfrank_newpanel_summary.csv"),
         pd.read_csv(DATA / "panel_newpanel_depmap_summary.csv"),
         f"fig_panelA_depmap{suffix}", set_filter=args.set)
    make(pd.read_csv(DATA / "selfrank_newpanel_ctrpv2_summary.csv"),
         pd.read_csv(DATA / "panel_newpanel_ctrpv2_summary.csv"),
         f"fig_panelA_ctrpv2{suffix}", set_filter=args.set)


if __name__ == "__main__":
    main()
