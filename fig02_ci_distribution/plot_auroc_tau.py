"""Render fig_auroc_tau.{pdf,png}.

AUROC(tissue-marker membership | gene importance) per method vs the marker
threshold tau, with the between-tissue ("tissue-only") attribution as a
positive-control ceiling. A marker = gene with expression tissue-eta^2 > tau.
AUROC is computed per item (does the method's importance rank markers above
non-markers) and averaged over items. Reference is the tissue-only ceiling,
not 0.5: marginal/IRM sit well below it (real but modest tissue favouring),
while residualized/within-tissue fall below 0.5 (they deplete markers).

Reads results/auroc_tau.csv (from compute_auroc_tau.py). Self-contained.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PALETTE = {"blue_main": "#0F4D92", "teal": "#42949E", "violet": "#9A4D8E",
           "neutral_mid": "#767676", "red_strong": "#B64342",
           "green_3": "#8BCF8B"}
COL = {"marginal": PALETTE["neutral_mid"], "residualized": PALETTE["blue_main"],
       "irm": PALETTE["violet"], "within_tissue": PALETTE["teal"],
       "dann": PALETTE["red_strong"], "adae": PALETTE["green_3"],
       "between_tissue": "#111111"}
LAB = {"marginal": "Marginal", "residualized": "Residualized", "irm": "IRM",
       "within_tissue": "Within-tissue", "dann": "DANN", "adae": "AD-AE",
       "between_tissue": "Tissue-only (ceiling)"}
M = ["marginal", "residualized", "irm", "within_tissue", "between_tissue"]
TITLE = {"depmap": "DepMap (essentiality)", "ctrpv2": "CTRPv2 (drug response)"}


def apply_publication_style(fs=10, lw=0.8):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "font.size": fs, "axes.titlesize": fs + 1, "axes.labelsize": fs,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1, "axes.linewidth": lw,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 200,
    })


def plot(res_dir: Path, out_dir: Path, stem: str):
    apply_publication_style(fs=14)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res_dir / "auroc_tau.csv")
    taus = sorted(df["tau"].unique())
    mc_path = res_dir / "marker_counts.csv"
    mc = {}
    if mc_path.exists():
        mcdf = pd.read_csv(mc_path)
        mc = {(r.dataset, r.tau): int(r.n_markers) for r in mcdf.itertuples()}
    fig, axes = plt.subplots(1, 2, figsize=(2.9 * len(taus) + 1.5, 5.4))
    for ax, ds in zip(axes, ["depmap", "ctrpv2"]):
        x = np.arange(len(taus)); w = 0.16
        for j, m in enumerate(M):
            s = df[(df.dataset == ds) & (df.method == m)].set_index("tau")
            means = [s.loc[t, "auroc"] for t in taus]
            errs = [s.loc[t, "sem"] for t in taus]
            ax.bar(x + (j - (len(M) - 1) / 2) * w, means, w, yerr=errs, capsize=2,
                   color=COL[m], label=LAB[m], error_kw={"lw": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(
            [fr"$\tau$={t:g}" + (f"\n$n$={mc[(ds, t)]}" if (ds, t) in mc else "")
             for t in taus])
        ax.set_ylabel("AUROC(tissue-marker | importance)")
        ax.set_title(TITLE[ds]); ax.set_ylim(0, 0.80)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / (stem + '.pdf')}")
    print(f"wrote {out_dir / (stem + '.png')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--res-dir", type=Path, default=HERE / "results")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures")
    p.add_argument("--stem", default="fig_auroc_tau")
    p.add_argument("--include-adv", action="store_true",
                   help="also plot the DANN/AD-AE bars; off by default so "
                        "this reproduces the published figure.")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.include_adv:
        M[3:3] = ["dann", "adae"]  # keep within_tissue/between_tissue last
    plot(a.res_dir, a.out_dir, a.stem)
