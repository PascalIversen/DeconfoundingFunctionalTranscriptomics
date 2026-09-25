"""Mixed-class breakdown of the synthetic experiment.

Report performance separately for the mixed class, where a gene is causal but also
tissue-amplified: that is the case where a deconfounding method could plausibly
discard true signal, and the top-10 composition does not resolve it.

The worry is a cost: a mixed gene's causal signal partly lives in the between-tissue
component, so a deconfounder that removes that component could demote a gene it should
keep. The test is therefore not "are mixed genes recovered" in absolute terms (at
n=1000 nothing is: median rank ~250/500) but whether mixed genes are demoted *relative
to pure-causal genes of the same true effect size*. Any differential is the discarded
signal.

Everything here comes from the shipped
figA4_synthetic_reliability/results/npower_mixed*.csv, which record per
(seed, n_cells, method, gene): gene_class, the true `effect`, and the SHAP `rank`
(1 = most important, out of the 500-gene panel). No retraining, no GPU.

Caveat: those CSVs track only the 350 non-noise genes, so a "signal vs noise" AUROC is
not computable without re-running compute_npower_mixed.py. Every metric here is
rank-based over the full 500-gene panel, which is unaffected.

Outputs:
    results/mixed_class_ranks.csv       per method x class x n: rank, retention, dRank
    results/mixed_vs_causal_tests.csv   paired-over-seeds test of the differential
    results/mixed_effect_matched.csv    same, within effect-size quartiles
    figures/fig_mixed_class.{pdf,png}

    python mixed_class.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SYN = REPO / "figA4_synthetic_reliability" / "results"
sys.path.insert(0, str(REPO / "fig02_ci_distribution"))
import plot_ci_distribution as P  # noqa: E402  (colours + house style)

METHODS = ["marginal", "residualized", "irm", "within_tissue", "dann", "adae"]
DECONF = ["residualized", "irm", "within_tissue", "dann", "adae"]
CLASSES = ["causal", "mixed", "confounder"]
CLASS_LABEL = {"causal": "causal\n(pure)", "mixed": "mixed\n(causal+tissue)",
               "confounder": "confounder\n(pure tissue)"}
CLASS_COLOR = {"causal": "#0F4D92", "mixed": "#B64342", "confounder": "#767676"}
N_PANEL = 500          # panel size the ranks are taken over
KS = [10, 25, 50, 100]
N_FOCUS = 1000         # the slice Fig A4 uses
N_BEST = 80000         # the best-powered slice in the sweep


def load(dgp: str = "v1") -> pd.DataFrame:
    """Mirror plot_reliability.load_merged's file sets, but keep the whole n grid.

    v1 = the published panel (npower_mixed.csv, full n grid, 4 methods) plus the
         extra n=1000 seeds and the separately-written adversarial arms. The adv
         files carry byte-identical `effect` values to the base file, i.e. the same
         DGP, so they can be concatenated -- verified, not assumed.
    v2 = the rebuilt panel in which mixed genes and confounders carry the SAME
         tissue eta^2, so tissue statistics alone cannot separate them. n=1000 only,
         all six methods. This is the variant that answers the circularity
         objection, so it is the one to quote for the mixed-class question.
    """
    if dgp == "v2":
        parts = [pd.read_csv(f) for f in sorted(SYN.glob("npower_mixed_v2_s*.csv"))
                 if "summary" not in f.name]
        if not parts:
            raise SystemExit("no v2 results in " + str(SYN))
    else:
        parts = [pd.read_csv(SYN / "npower_mixed.csv"),
                 pd.read_csv(SYN / "npower_mixed_extra_n1000.csv")]
        parts += [pd.read_csv(f) for f in sorted(SYN.glob("npower_mixed_adv*.csv"))
                  if "summary" not in f.name]
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(["seed", "n_cells", "method", "gene"])
    d = d[d.method.isin(METHODS)].copy()
    # rank -> percentile so "how far up the panel" is comparable across metrics
    d["pct"] = 100.0 * (1.0 - (d["rank"] - 1) / (N_PANEL - 1))
    return d


def with_delta(d: pd.DataFrame) -> pd.DataFrame:
    """Attach each gene's marginal rank, so dRank = rank_method - rank_marginal."""
    marg = (d[d.method == "marginal"]
            .set_index(["seed", "n_cells", "gene"])["rank"].rename("rank_marginal"))
    out = d.join(marg, on=["seed", "n_cells", "gene"])
    out["d_rank"] = out["rank"] - out["rank_marginal"]
    return out


def rank_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per (n, method, class): rank centre, retention@K, and shift vs marginal."""
    rows = []
    for (n, m, c), g in d.groupby(["n_cells", "method", "gene_class"]):
        row = {"n_cells": n, "method": m, "gene_class": c,
               "n_seeds": g.seed.nunique(), "n_genes_per_seed": len(g) / g.seed.nunique(),
               "median_rank": g["rank"].median(), "mean_rank": g["rank"].mean(),
               "median_d_rank": g["d_rank"].median(), "mean_d_rank": g["d_rank"].mean()}
        for k in KS:
            row[f"retention@{k}"] = float((g["rank"] <= k).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def differential_tests(d: pd.DataFrame) -> pd.DataFrame:
    """Is the mixed class demoted relative to pure-causal? Paired over seeds.

    Per seed we take the median dRank of the mixed genes and of the pure-causal genes;
    the paired difference over seeds is the differential cost. Positive = mixed
    demoted more (or promoted less) than causal = true signal discarded.
    """
    rows = []
    per_seed = (d.groupby(["n_cells", "method", "seed", "gene_class"])
                 .agg(med_d=("d_rank", "median"), med_rank=("rank", "median"))
                 .reset_index())
    for (n, m), g in per_seed.groupby(["n_cells", "method"]):
        w = g.pivot(index="seed", columns="gene_class")
        mix, cau = w[("med_d", "mixed")], w[("med_d", "causal")]
        ok = mix.notna() & cau.notna()
        if ok.sum() < 6 or m == "marginal":
            continue
        diff = (mix - cau)[ok]
        rows.append({
            "n_cells": n, "method": m, "n_seeds": int(ok.sum()),
            "median_dRank_causal": cau[ok].median(),
            "median_dRank_mixed": mix[ok].median(),
            "differential (mixed - causal)": diff.median(),
            "frac_seeds_mixed_worse": float((diff > 0).mean()),
            "wilcoxon_p": (wilcoxon(mix[ok], cau[ok]).pvalue
                           if (diff != 0).any() else np.nan),
        })
    return pd.DataFrame(rows)


def effect_matched(d: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Same differential, within bins of the true effect size.

    A raw mixed-vs-causal comparison is confounded by effect size: the two classes are
    drawn from the same tier ladder but not identically. Binning on the true coefficient
    makes the comparison like-for-like, which is what turns "no differential" into an
    actual argument.
    """
    sig = d[d.gene_class.isin(["causal", "mixed"])].copy()
    # Bin on the pooled causal+mixed effect distribution, per (n, seed) so the bin
    # edges never mix sample sizes.
    sig["effect_bin"] = (sig.groupby(["n_cells", "seed"])["effect"]
                         .transform(lambda s: pd.qcut(s, n_bins, labels=False,
                                                      duplicates="drop")))
    rows = []
    for (n, m, b), g in sig.groupby(["n_cells", "method", "effect_bin"]):
        per_seed = (g.groupby(["seed", "gene_class"])["d_rank"].median()
                     .unstack("gene_class"))
        if "mixed" not in per_seed or "causal" not in per_seed:
            continue
        ok = per_seed["mixed"].notna() & per_seed["causal"].notna()
        if ok.sum() < 6 or m == "marginal":
            continue
        diff = (per_seed["mixed"] - per_seed["causal"])[ok]
        rows.append({
            "n_cells": n, "method": m, "effect_bin": int(b),
            # Bin edges are per-(n, seed) quantiles, so a pooled lo/hi would overlap
            # between neighbouring bins and read as an error. The median effect in
            # the bin is unambiguous.
            "effect_median": g.effect.median(),
            "n_seeds": int(ok.sum()),
            "median_dRank_causal": per_seed["causal"][ok].median(),
            "median_dRank_mixed": per_seed["mixed"][ok].median(),
            "differential (mixed - causal)": diff.median(),
            "wilcoxon_p": (wilcoxon(per_seed["mixed"][ok], per_seed["causal"][ok]).pvalue
                           if (diff != 0).any() else np.nan),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figure
def _panel_drank(ax, d, n):
    """Median dRank vs marginal per class, one bar group per deconfounder."""
    g = d[d.n_cells == n]
    x = np.arange(len(DECONF))
    w = 0.26
    for i, c in enumerate(CLASSES):
        per_seed = (g[g.gene_class == c].groupby(["method", "seed"])["d_rank"]
                    .median().unstack("method").reindex(columns=DECONF))
        med = per_seed.median()
        sem = per_seed.std() / np.sqrt(per_seed.notna().sum())
        ax.bar(x + (i - 1) * w, med.values, w * 0.9, yerr=sem.values, capsize=2,
               color=CLASS_COLOR[c], label=CLASS_LABEL[c].replace("\n", " "),
               error_kw={"lw": 0.7})
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([P.METHOD_LABELS[m] for m in DECONF], rotation=20, ha="right")
    ax.set_ylabel("median $\\Delta$rank vs marginal")
    ax.set_title(f"a   rank shift by gene class (n={n})", loc="left",
                 fontsize=8, fontweight="bold")
    # Legend placed outside the axes (above), not "upper center" inside it: the
    # confounder/mixed bars reach close to the axis top, so an inside legend
    # collides with them once this panel is squeezed into a 1x3 row.
    ax.legend(fontsize=6, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.22), ncol=3,
              columnspacing=0.9, handlelength=1.0)
    ax.text(0.99, 0.03, "↓ promoted", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6, color="0.35")


def _panel_retention(ax, d, n, k_max=100):
    """Retention@K of the mixed class MINUS that of the pure-causal class, vs K.

    The absolute retention curves for the two classes sit on top of each other, which
    is the finding but is unreadable as two overlapping lines. Plotting the difference
    with a +-SEM band shows the same thing and makes "no differential at any K"
    checkable rather than asserted.
    """
    ks = np.arange(5, k_max + 1, 5)
    g = d[d.n_cells == n]
    for m in METHODS:
        gm = g[g.method == m]
        per_seed = []
        for _, gs in gm.groupby("seed"):
            mix = gs[gs.gene_class == "mixed"]["rank"].to_numpy()
            cau = gs[gs.gene_class == "causal"]["rank"].to_numpy()
            if mix.size == 0 or cau.size == 0:
                continue
            per_seed.append([100 * ((mix <= k).mean() - (cau <= k).mean()) for k in ks])
        if not per_seed:
            continue
        a = np.array(per_seed)
        mu, se = a.mean(0), a.std(0) / np.sqrt(len(a))
        ax.plot(ks, mu, "-", lw=1.3, color=P.METHOD_COLORS[m],
                label=P.METHOD_LABELS[m])
        ax.fill_between(ks, mu - se, mu + se, color=P.METHOD_COLORS[m], alpha=0.16,
                        linewidth=0)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("retention@K:  mixed $-$ causal (pp)")
    ax.set_title(f"   no differential at any K (n={n})", loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)


def _panel_matched(ax, matched, n):
    """Effect-matched differential (mixed - causal) per effect quartile."""
    g = matched[matched.n_cells == n]
    x = np.arange(g.effect_bin.nunique())
    w = 0.26
    for i, m in enumerate(DECONF):
        s = g[g.method == m].sort_values("effect_bin")
        ax.bar(x + (i - 1) * w, s["differential (mixed - causal)"].values, w * 0.9,
               color=P.METHOD_COLORS[m], label=P.METHOD_LABELS[m])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_xlabel("true effect-size quartile (weak $\\rightarrow$ strong)")
    ax.set_ylabel("$\\Delta$rank$_{mixed}$ $-$ $\\Delta$rank$_{causal}$")
    ax.set_title(f"c   effect-matched cost to the mixed class (n={n})", loc="left",
                 fontsize=8, fontweight="bold")
    # Tall bars occupy the whole upper half of the axes in every quartile, so
    # there is no free spot inside for a legend; place it outside (above).
    ax.legend(fontsize=6, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.22), ncol=3, columnspacing=0.8,
              handlelength=1.0)
    ax.text(0.99, 0.95, "> 0 = mixed demoted more than\nmatched pure-causal genes",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="0.35")


def _panel_nsweep(ax, tests):
    """Differential vs sample size - does a cost appear once the model works?"""
    for m in DECONF:
        s = tests[tests.method == m].sort_values("n_cells")
        ax.plot(s.n_cells, s["differential (mixed - causal)"], "o-", ms=3, lw=1.2,
                color=P.METHOD_COLORS[m], label=P.METHOD_LABELS[m])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n cells")
    ax.set_ylabel("$\\Delta$rank$_{mixed}$ $-$ $\\Delta$rank$_{causal}$")
    ax.set_title("d   differential vs sample size", loc="left", fontsize=8,
                 fontweight="bold")
    ax.legend(fontsize=6, frameon=False)


def make_figure(d, tests, matched, n_focus, stem):
    # Panel d (differential vs. sample size) is dropped for the manuscript figure:
    # the v2 (hard-case) DGP was only run at n=1000, so it plots a single vertical
    # slice of points rather than an actual trend, and the paper does not make any
    # sample-size claim for this setting. `tests` is kept as a make_figure argument
    # (unused here) so callers/tests relying on the old signature do not break.
    del tests
    P.apply_publication_style(font_size=7, axes_linewidth=0.8)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6), layout="constrained")
    _panel_drank(axes[0], d, n_focus)
    _panel_retention(axes[1], d, n_focus)
    axes[1].set_title("b" + axes[1].get_title(), loc="left", fontsize=8,
                       fontweight="bold")
    _panel_matched(axes[2], matched, n_focus)
    out = HERE / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out / (stem + '.pdf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-focus", type=int, default=N_FOCUS)
    ap.add_argument("--dgp", default="v1", choices=["v1", "v2"])
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    # n=80000 has 8 seeds; scipy warns that it falls back from the normal
    # approximation. That is expected, and the exact p-value it reports is the one
    # we want.
    warnings.filterwarnings("ignore", message="Sample size too small")

    d = with_delta(load(a.dgp))
    res = HERE / "results"
    res.mkdir(parents=True, exist_ok=True)

    tab = rank_table(d)
    tests = differential_tests(d)
    matched = effect_matched(d)
    tab.to_csv(res / f"mixed_class_ranks_{a.dgp}.csv", index=False)
    tests.to_csv(res / f"mixed_vs_causal_tests_{a.dgp}.csv", index=False)
    matched.to_csv(res / f"mixed_effect_matched_{a.dgp}.csv", index=False)

    pd.set_option("display.width", 200)
    # v2 was only run at n=1000; only report n values actually present.
    present = set(d.n_cells.unique())
    for n in sorted({a.n_focus, N_BEST} & present):
        print(f"\n=== n={n}: median rank (of {N_PANEL}) and retention@50 by class ===")
        t = tab[tab.n_cells == n].pivot(index="method", columns="gene_class")
        print(t[[("median_rank", c) for c in CLASSES]
                + [("retention@50", c) for c in CLASSES]]
              .reindex(METHODS).round(3).to_string())
        print(f"\n=== n={n}: mixed-vs-causal differential (paired over seeds) ===")
        print(tests[tests.n_cells == n].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\n=== effect-matched, n={a.n_focus} ===")
    print(matched[matched.n_cells == a.n_focus].to_string(index=False,
          float_format=lambda v: f"{v:.4g}"))

    if not a.no_figure:
        make_figure(d, tests, matched, a.n_focus, f"fig_mixed_class_{a.dgp}")
    print(f"\nwrote mixed_*_{a.dgp}.csv in {res}")


if __name__ == "__main__":
    main()
