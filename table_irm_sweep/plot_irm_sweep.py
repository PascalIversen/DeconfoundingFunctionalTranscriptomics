"""Appendix figure: IRM penalty-weight (lambda) sweep.

Shows that no lambda rescues IRM: across six orders of magnitude its
deconfounding metrics never reach the data-/attribution-stage references.

Reads results/irm_sweep_{dataset}[_shard*].csv (merges shards) and renders
figures/fig_irm_sweep.{pdf,png}: rows = datasets, cols = metrics
(contamination@50, attribution eta^2, within-tissue r). IRM is the swept line
(markers, +/- SEM over items); marginal / residualized / within-tissue SHAP are
horizontal reference lines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
mpl.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

PALETTE = {"marginal": "#767676", "residualized": "#0F4D92",
           "within_tissue": "#42949E", "irm": "#9A4D8E"}
REF_LABEL = {"marginal": "Marginal", "residualized": "Residualized",
             "within_tissue": "Within-tissue SHAP"}
# metric: (column, pretty, better-direction)
METRICS = [("contam_50", "Contamination@50", "lower"),
           ("attr_eta2", r"Attribution $\eta^2$ (mean)", "lower"),
           ("within_r", "Within-tissue $r$", "higher")]
DATASETS = [("depmap", "DepMap (essentiality)"),
            ("ctrpv2", "CTRPv2 (drug response)")]
LAM0_X = 0.02   # where to draw lambda=0 (ERM) on the log axis


def load(ds: str) -> pd.DataFrame:
    parts = sorted((HERE / "results").glob(f"irm_sweep_{ds}*.csv"))
    if not parts:
        raise SystemExit(f"no results for {ds}")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def agg(df, method, col, lam=None):
    sub = df[df.method == method]
    if lam is not None:
        sub = sub[np.isclose(sub.lam, lam)]
    v = sub[col].dropna()
    n = len(v)
    return v.mean(), (v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0), n


def main():
    present = [(ds, lab) for ds, lab in DATASETS
               if list((HERE / "results").glob(f"irm_sweep_{ds}*.csv"))]
    if not present:
        raise SystemExit("no sweep results found")
    fig, axes = plt.subplots(len(present), len(METRICS),
                             figsize=(11, 3.3 * len(present)), squeeze=False)
    lambdas = sorted(load(present[0][0])["lam"].dropna().unique())
    lam_pos = [LAM0_X if l == 0 else l for l in lambdas]

    for r, (ds, ds_lab) in enumerate(present):
        df = load(ds)
        n_items = df[df.method == "irm"].item.nunique()
        for c, (col, pretty, better) in enumerate(METRICS):
            ax = axes[r][c]
            # IRM swept line
            ys, es = [], []
            for l in lambdas:
                m, s, _ = agg(df, "irm", col, lam=l)
                ys.append(m); es.append(s)
            ax.errorbar(lam_pos, ys, yerr=es, color=PALETTE["irm"], marker="o",
                        ms=5, lw=1.8, capsize=2, zorder=3, label="IRM (swept)")
            # reference horizontal lines + SEM band
            for m in ("marginal", "residualized", "within_tissue"):
                mu, se, _ = agg(df, m, col)
                ax.axhline(mu, color=PALETTE[m], lw=1.4, ls="--", zorder=2)
                ax.axhspan(mu - se, mu + se, color=PALETTE[m], alpha=0.10, zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(lam_pos)
            ax.set_xticklabels(["0" if l == 0 else
                                (f"{l:g}") for l in lambdas], fontsize=8)
            if r == len(DATASETS) - 1:
                ax.set_xlabel(r"IRM penalty weight $\lambda$")
            ax.set_ylabel(pretty)
            if r == 0:
                ax.set_title(pretty, fontsize=10)
            arrow = "↓ better" if better == "lower" else "↑ better"
            ax.annotate(arrow, xy=(0.02, 0.04 if better == "lower" else 0.92),
                        xycoords="axes fraction", fontsize=7.5, color="#555")
        axes[r][0].annotate(ds_lab, xy=(-0.34, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=11, fontweight="bold")
        axes[r][0].annotate(f"$n={n_items}$ items", xy=(0.02, 0.5),
                            xycoords="axes fraction", fontsize=7.5, color="#777")

    # one shared legend
    handles = [plt.Line2D([], [], color=PALETTE["irm"], marker="o", lw=1.8,
                          label="IRM (swept $\\lambda$)")]
    for m in ("marginal", "residualized", "within_tissue"):
        handles.append(plt.Line2D([], [], color=PALETTE[m], ls="--", lw=1.4,
                                  label=REF_LABEL[m]))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    out = HERE / "figures" / "fig_irm_sweep"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=200)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
