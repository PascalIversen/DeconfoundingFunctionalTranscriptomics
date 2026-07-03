"""Render the confounding-evidence figures from compute_confounding.py CSVs.

    fig05_confounding_evidence    — the two-arm proof (rows = datasets):
        col A  Tissue → expression : per-gene η²(g,expr) histogram
        col B  Tissue → Phenotype  : per-outcome η²(o) distribution
        Both panels overlay a translucent **shuffled-tissue null** (η² recomputed on
        permuted tissue labels, from confounding_*_eta2_null.csv) — the real
        distribution sits far to its right. The per-item count of real η² above
        the null (Δ>0) is printed for the caption, not drawn on the figure.
    fig05b_confounding_structure  — supporting structure (one column per dataset):
        row A  PCA coloured by tissue, one shared OncotreeLineage legend (CTRPv2
               is remapped onto OncotreeLineage via its cellosaurus↔RRID join)
        row B  within- vs across-tissue correlation + Mann-Whitney p
    fig05c_confounder_comparison  — "Tissue or something else": **debiased**
        Δη² = real − each item's own permuted floor, across 7 candidate
        confounders, both arms, per dataset (raw η² if the *_null.csv is absent).
    fig05d_classifier_per_tissue  — Arm-1 linear tissue classifier accuracy per
        tissue (beeswarm), one column per dataset.
    fig05e_outcome_significance   — Arm-2 per-outcome Welch p-values as a
        −log10(p) strip per dataset, with each dataset's Bonferroni cutoff and
        the surviving-fraction readout.

All inputs are the CSVs in results/; no trained model is required.

Usage (from figure_confounding_evidence/):
    python plot_confounding.py
    python plot_confounding.py --results-dir results --out-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent

SHORT_DATASET = {"depmap": "DepMap", "ctrpv2": "CTRPv2"}
# What one Arm-2 outcome *is* per dataset: a CRISPRko target gene vs a drug.
OUTCOME_UNIT = {"depmap": "targets", "ctrpv2": "drugs"}
ARM1_C, ARM2_C = "#3B6FB6", "#C1666B"      # expression / outcome
ETA_THRESHOLD = 0.30
# Shuffled-tissue null overlay colours: a faint translucent grey fill so the
# coloured real bars show through (drawn in _overlay_null, which explains why
# the fill is plain rather than hatched).
NULL_C, NULL_FILL = "0.30", "0.55"

# fig05b row B (within/across Pearson-r histograms): one colour, two shades —
# within saturated, across light (ColorBrewer Blues), shared by both datasets.
CORR_HIST_COLORS = {
    "depmap": ("#2166AC", "#92C5DE"),   # blue (within / across)
    "ctrpv2": ("#2166AC", "#92C5DE"),   # blue (within / across)
}

# fig05b row A (PCA): the two datasets natively use different tissue ontologies
# (DepMap = OncotreeLineage; CTRPv2 = its own CCLE lineage labels), so they can't
# share a legend as-is. We put CTRPv2 onto OncotreeLineage too (see
# _ctrpv2_to_oncotree) and colour both panels from this one shared map, ordered
# so a tissue is the same colour in a and b. 11 distinct hues (no grey — that's
# reserved for "other") cover the union of both panels' top-10 lineages.
OTHER_C = "0.8"
SHARED_TISSUE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#393b79", "#e7ba52",
]

_RRID_TO_ONCOTREE: dict | None = None


def _find_model_csv() -> Path | None:
    """Locate the bundled DepMap/Model.csv by walking up from this file."""
    for base in [HERE, *HERE.parents]:
        cand = base / "data" / "DepMap" / "Model.csv"
        if cand.exists():
            return cand
    return None


def _ctrpv2_to_oncotree(sample_ids) -> list | None:
    """Map CTRPv2 PCA samples (keyed by cellosaurus id) to DepMap's
    OncotreeLineage, so fig05b a/b can share one legend on a common ontology.

    The join is authoritative — cellosaurus id == Model.csv's RRID — rather than
    a hand-written lineage table. That matters: a naive CCLE→Oncotree string map
    would merge CTRPv2 "Blood" into a single lineage, but it actually spans
    Lymphoid + Myeloid (the RRID join keeps them apart). Unmatched ids (~0.2% of
    the cohort) and a missing Model.csv fall through to "other"/None."""
    global _RRID_TO_ONCOTREE
    if _RRID_TO_ONCOTREE is None:
        mp = _find_model_csv()
        if mp is None:
            return None
        m = pd.read_csv(mp, usecols=["RRID", "OncotreeLineage"]).dropna()
        _RRID_TO_ONCOTREE = dict(zip(m.RRID.astype(str), m.OncotreeLineage))
    return [_RRID_TO_ONCOTREE.get(str(s), "other") for s in sample_ids]

# Target canvas for fig05: one half of a landscape A4 page with 1-inch margins.
# A4 landscape = 11.69 × 8.27 in; usable after 1" margins = 9.69 × 6.27 in;
# one half (left or right) = 4.845 × 6.27 in.
A4_LANDSCAPE_IN = (11.69, 8.27)
PAGE_MARGIN_IN = 1.0
_USABLE_W = A4_LANDSCAPE_IN[0] - 2 * PAGE_MARGIN_IN
_USABLE_H = A4_LANDSCAPE_IN[1] - 2 * PAGE_MARGIN_IN
HALF_PAGE_IN = (_USABLE_W / 2, _USABLE_H)   # (4.845, 6.27)

# fig05 panel letters (a/b/c/d on the η² histogram titles); set True to show them.
SHOW_PANEL_LETTERS = False

# fig05c y-axis labels — OncotreeLineage IS the tissue, so name it as such.
CONF_LABELS = {
    "OncotreeLineage":        "Tissue (lineage)",
    "OncotreePrimaryDisease": "Primary disease",
    "OncotreeSubtype":        "Cancer subtype",
    "Sex":                    "Sex",
    "PrimaryOrMetastasis":    "Primary / metastasis",
    "GrowthPattern":          "Growth pattern",
    "OnboardedMedia":         "Culture media",
}


def apply_publication_style(font_size: int = 8,
                            axes_linewidth: float = 0.8) -> None:
    """Nature-style rcParams shared with the other paper figures."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype":      "none",
        "pdf.fonttype":      42,
        # Render $…$ mathtext in Arial too (default fontset is DejaVu Sans), so
        # subscripted symbols like η²_g,expr stay in the all-Arial set. Greek /
        # italic math glyphs come from Arial-Italic; both are the Arial family.
        "mathtext.fontset":  "custom",
        "mathtext.rm":       "Arial",
        "mathtext.it":       "Arial:italic",
        "mathtext.bf":       "Arial:bold",
        "mathtext.sf":       "Arial",
        "mathtext.cal":      "Arial:italic",
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


def _save(fig, out_dir: Path, stem: str, tight: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # tight=True crops to content (default); tight=False keeps the exact figsize
    # so the output honours a fixed page budget (e.g. one half of an A4 page).
    kw = dict(bbox_inches="tight") if tight else dict(pad_inches=0)
    fig.savefig(out_dir / f"{stem}.pdf", **kw)
    fig.savefig(out_dir / f"{stem}.png", dpi=200, **kw)
    plt.close(fig)
    print(f"wrote {out_dir / (stem + '.pdf')}")
    print(f"wrote {out_dir / (stem + '.png')}")


def _datasets(summary: pd.DataFrame) -> list:
    order = ["depmap", "ctrpv2"]
    return [d for d in order if d in set(summary["dataset"])]


# ───────────────────────────────────────────────── fig05 — two-arm evidence
def _overlay_null(ax, vals: np.ndarray, n_perm: int, *, bins: int,
                  xmax: float) -> None:
    """Overlay the shuffled-tissue null on a fig05 η² panel: a faint translucent
    fill, area-matched to one real draw (pooled counts ÷ n_perm). Same bin
    edges as the real histogram. Drawn after the real bars but before the
    median / threshold lines so those stay crisp on top."""
    edges = np.linspace(0, xmax, bins + 1)
    w = np.full(len(vals), 1.0 / n_perm)
    # Plain translucent fill (no hatch): a semi-transparent *patterned* fill is
    # flattened to opaque by Inkscape's PDF exporter, but a plain fill-opacity
    # shape survives — so the coloured bars still show through after export.
    ax.hist(vals, bins=edges, weights=w, histtype="stepfilled",
            facecolor=NULL_FILL, alpha=0.20, edgecolor=NULL_C,
            linewidth=0.0)
    ax.hist(vals, bins=edges, weights=w, histtype="step", edgecolor=NULL_C,
            linewidth=0.8)


def _print_above_null(expr, outc, expr_null, outc_null, dss) -> None:
    """Print (for the figure caption) how many genes / outcomes have a real η²
    above their own mean shuffled η² — the per-item Δ>0 count, not drawn."""
    print("fig05 above-null (real η² > mean shuffled η², per item):")
    for ds in dss:
        er = expr[expr.dataset == ds].set_index("gene")["eta2_X"]
        en = expr_null[expr_null.dataset == ds].groupby("gene")["eta2_X"].mean()
        a, b = er.align(en, join="inner")
        n1, t1 = int((a.values > b.values).sum()), len(a)
        orr = outc[outc.dataset == ds].set_index("item")["eta2_y"]
        on = outc_null[outc_null.dataset == ds].groupby("item")["eta2_y"].mean()
        c, d = orr.align(on, join="inner")
        n2, t2 = int((c.values > d.values).sum()), len(c)
        unit = OUTCOME_UNIT[ds]
        print(f"  {SHORT_DATASET[ds]:7s} Arm1 {n1}/{t1} genes "
              f"({n1/t1*100:.0f}%) | Arm2 {n2}/{t2} {unit} ({n2/t2*100:.0f}%)")


def plot_evidence(results_dir: Path, out_dir: Path, stem: str) -> None:
    """fig05 — the two-arm η² evidence: per-gene η²(X) (Arm 1) and per-outcome
    η²(y) (Arm 2) histograms, one row per dataset, each with the shuffled-tissue
    null overlay, the median line and the η²=0.30 contamination threshold."""
    summary = pd.read_csv(results_dir / "confounding_summary.csv")
    expr = pd.read_csv(results_dir / "confounding_expr_eta2.csv")
    outc = pd.read_csv(results_dir / "confounding_outcome_eta2.csv")
    # Shuffled-tissue null (optional — absent in pre-null results); when present
    # it is overlaid on both arms as a translucent fill (see _overlay_null).
    en_path = results_dir / "confounding_expr_eta2_null.csv"
    on_path = results_dir / "confounding_outcome_eta2_null.csv"
    expr_null = pd.read_csv(en_path) if en_path.exists() else None
    outc_null = pd.read_csv(on_path) if on_path.exists() else None
    dss = _datasets(summary)

    # One half of a landscape A4 page (1-inch margins), at half height, then
    # trimmed a further 20% in the vertical direction.
    figsize = (HALF_PAGE_IN[0], HALF_PAGE_IN[1] / 2 * 0.8)
    fig, axes = plt.subplots(len(dss), 2, figsize=figsize, squeeze=False)
    label_fs = 7.0   # axis labels match the tick-label size (font_size - 1)
    title_fs = 8.0   # subplot titles a touch larger than the labels/ticks
    for i, ds in enumerate(dss):
        s = summary[summary.dataset == ds].iloc[0]
        la, lb = chr(ord("a") + 2 * i), chr(ord("a") + 2 * i + 1)
        pa = f"{la}  " if SHOW_PANEL_LETTERS else ""   # panel-letter prefixes,
        pb = f"{lb}  " if SHOW_PANEL_LETTERS else ""   # toggled off for now

        # col A — Tissue → expression
        ax = axes[i][0]
        ax.set_title(f"{pa}tissue → expression ({SHORT_DATASET[ds]})",
                     fontsize=title_fs)
        ev = expr[expr.dataset == ds]["eta2_X"].values
        ax.hist(ev, bins=40, range=(0, 1), color=ARM1_C, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        if expr_null is not None:
            nv = expr_null[expr_null.dataset == ds]
            _overlay_null(ax, nv["eta2_X"].values, int(nv["perm"].nunique()),
                          bins=40, xmax=1.0)
        med = float(np.median(ev))
        ax.axvline(med, color="black", lw=1.0, ls="--")
        ax.axvline(ETA_THRESHOLD, color="0.4", lw=0.8, ls=":")
        frac_hi = float((ev > ETA_THRESHOLD).mean())
        # tissue-classifier lift + accuracy now live in a separate figure (fig05d).
        ax.text(0.97, 0.95,
                f"median $\\eta^2$={med:.2f}\n{frac_hi*100:.0f}% genes > {ETA_THRESHOLD:g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)            # y-axis meets x-axis at the origin
        # keep ~3 y-ticks: the short axes make the auto-locator drop to 2.
        ax.locator_params(axis="y", nbins=4)
        ax.set_ylabel("genes", fontsize=label_fs)
        if i == len(dss) - 1:
            # subscript notation; "tissue → expression" already lives in the title
            ax.set_xlabel(r"$\eta^2_\mathrm{g,expr}$", fontsize=label_fs)

        # col B — Tissue → outcome
        ax = axes[i][1]
        ax.set_title(f"{pb}tissue → phenotype ({SHORT_DATASET[ds]})",
                     fontsize=title_fs)
        oy = outc[outc.dataset == ds]["eta2_y"].values
        xmax = max(0.6, float(oy.max()) if len(oy) else 0.6)
        ax.hist(oy, bins=30, range=(0, xmax),
                color=ARM2_C, alpha=0.85, edgecolor="white", linewidth=0.3)
        if outc_null is not None:
            nv = outc_null[outc_null.dataset == ds]
            _overlay_null(ax, nv["eta2_y"].values, int(nv["perm"].nunique()),
                          bins=30, xmax=xmax)
        medy = float(np.median(oy)) if len(oy) else float("nan")
        ax.axvline(medy, color="black", lw=1.0, ls="--")
        unit = OUTCOME_UNIT[ds]
        # ANOVA-significance fraction now lives in a separate figure (fig05e).
        txt = (f"median $\\eta^2$={medy:.2f}\n"
               f"n={int(s.n_outcomes)} {unit}")
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
        # start both axes at the origin so the spines meet at (0, 0)
        ax.set_xlim(0, xmax)
        ax.set_ylim(bottom=0)
        # keep ~3 y-ticks: the short axes make the auto-locator drop to 2.
        ax.locator_params(axis="y", nbins=4)
        ax.set_ylabel(unit, fontsize=label_fs)
        if i == len(dss) - 1:
            # subscript notation; "tissue → outcome" already lives in the title
            ax.set_xlabel(r"$\eta^2_\mathrm{o}$", fontsize=label_fs)

    # Share the y-axis within each arm (column): both datasets on one scale (the
    # taller of the two) so their bar heights and y-ticks line up. The x-axis is
    # already common within a column (0–1 for Arm 1, 0–xmax for Arm 2).
    for j in range(2):
        top = max(axes[r][j].get_ylim()[1] for r in range(len(dss)))
        for r in range(len(dss)):
            axes[r][j].set_ylim(0, top)

    # Reserve a bottom band for the legend so it stays inside the fixed canvas
    # (the output is saved at the exact half-page size, so nothing may overflow).
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    handles = [
        Line2D([0], [0], color="black", lw=1.0, ls="--", label=r"median $\eta^2$"),
        Line2D([0], [0], color="0.4", lw=0.8, ls=":",
               label=f"$\\eta^2$ = {ETA_THRESHOLD:g} (contamination)")]
    if expr_null is not None or outc_null is not None:
        # slightly more opaque than the on-axes overlay so the swatch reads.
        handles.insert(0, Patch(facecolor=NULL_FILL, alpha=0.45,
                                edgecolor=NULL_C, linewidth=0.8,
                                label="shuffled tissue labels"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.005))
    if expr_null is not None and outc_null is not None:
        _print_above_null(expr, outc, expr_null, outc_null, dss)
    _save(fig, out_dir, stem, tight=False)


# ──────────────────────────────────── fig05d — tissue classifier, per tissue
def _beeswarm_x(y: np.ndarray, center: float, half_width: float = 0.34,
                n_bins: int = 34, lo: float = 0.0, hi: float = 1.0,
                step: float = 0.062) -> np.ndarray:
    """Bin-stacked beeswarm x-positions: tissues with near-equal accuracy fan
    out horizontally around `center` instead of overprinting. Offsets stay
    within ±`half_width`; the per-bin step shrinks if a bin is crowded."""
    y = np.asarray(y, float)
    xs = np.full(len(y), center, float)
    if len(y) == 0:
        return xs
    edges = np.linspace(lo, hi, n_bins + 1)
    b = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        idx = idx[np.argsort(y[idx])]          # tidy left→right within a bin
        k = len(idx)
        s = min(step, (2 * half_width) / max(k - 1, 1))
        xs[idx] = center + (np.arange(k) - (k - 1) / 2.0) * s
    return xs


def plot_clf_per_tissue(results_dir: Path, out_dir: Path, stem: str) -> None:
    """Per-tissue accuracy of the linear tissue classifier (Arm 1), one swarm
    per dataset. Each point is one tissue's accuracy averaged over the 5 CV
    folds; the solid bar is the median tissue, the dotted bar is chance."""
    path = results_dir / "confounding_clf_per_tissue.csv"
    if not path.exists():
        print(f"skip {stem}: {path.name} missing "
              "(run compute_classifier_per_tissue.py first)")
        return
    df = pd.read_csv(path)
    dss = [d for d in ["depmap", "ctrpv2"] if d in set(df.dataset)]

    # Half the width of the η² histogram figure (one histogram column); height is
    # the quarter-page band minus a touch, then trimmed 20% twice
    # (0.8×0.8 = 0.64; matches fig05e).
    figsize = (HALF_PAGE_IN[0] / 2, (HALF_PAGE_IN[1] / 2 - 0.45) * 0.64)
    fig, ax = plt.subplots(figsize=figsize)
    label_fs, title_fs, annot_fs = 7.0, 8.0, 6.5   # same as the η² histograms

    for c, ds in enumerate(dss):
        g = df[df.dataset == ds]
        acc = g["accuracy"].values
        ax.scatter(_beeswarm_x(acc, c), acc, s=13, color=ARM1_C, alpha=0.85,
                   edgecolor="white", linewidth=0.3, zorder=3)
        med = float(np.median(acc))
        chance = 1.0 / len(g)                  # = 1 / n_classes the clf kept
        ax.plot([c - 0.36, c + 0.36], [med, med], color="black", lw=1.3, zorder=4)
        ax.plot([c - 0.36, c + 0.36], [chance, chance], color="0.45", lw=0.9,
                ls=":", zorder=4)
        ax.text(c + 0.40, med, f"{med:.2f}", ha="left", va="center",
                fontsize=annot_fs)
        # n-tissues count in a small white box (matches the η² histogram boxes),
        # floated above the swarm so it never sits on the top points (dots reach
        # ~1.0, so the box sits at 1.15 with headroom to 1.28).
        ax.text(c, 1.15, f"n={len(g)} tissues", ha="center", va="center",
                fontsize=annot_fs, color="0.3",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

    ax.set_xlim(-0.62, len(dss) - 1 + 0.72)
    ax.set_ylim(0, 1.28)
    ax.set_xticks(range(len(dss)))
    ax.set_xticklabels([SHORT_DATASET[d] for d in dss])
    ax.set_ylabel("tissue accuracy (5-fold CV)", fontsize=label_fs)
    ax.set_title("Linear tissue classifier", fontsize=title_fs)

    # Two legend entries on one row (compact fonts/spacing keep them inside the
    # narrow canvas).
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=[
        Line2D([0], [0], color="black", lw=1.3, label="median tissue"),
        Line2D([0], [0], color="0.45", lw=0.9, ls=":", label="chance (1 / #tissues)")],
        loc="lower center", ncol=2, frameon=False, fontsize=6.5,
        bbox_to_anchor=(0.5, 0.005), handlelength=1.2, handletextpad=0.5,
        columnspacing=1.0, labelspacing=0.3)
    _save(fig, out_dir, stem, tight=False)


# ──────────────────────────────── fig05e — Tissue → outcome significance (Arm 2)
def plot_outcome_significance(results_dir: Path, out_dir: Path, stem: str) -> None:
    """Per-outcome Welch-ANOVA significance for Arm 2 (tissue → outcome), as a
    −log10(p) strip per dataset. Each point is one drug/target; the solid bar is
    the median, the dashed bar each dataset's **Bonferroni** cutoff (α/mᵢ). The
    box reports how many outcomes survive that correction — the figure's headline
    Arm-2 significance from now on. The plain α=0.05 line is intentionally
    omitted: at −log10 it sits ~2.6 below the Bonferroni cutoff, and on a
    0–`ymax` axis the two were unreadably close. Sized like fig05d so the two
    sit side by side."""
    outc = pd.read_csv(results_dir / "confounding_outcome_eta2.csv")
    summary = (pd.read_csv(results_dir / "confounding_summary.csv")
               .set_index("dataset"))
    dss = [d for d in ["depmap", "ctrpv2"] if d in set(outc.dataset)]

    # Same width as fig05d, trimmed 20% twice in height (0.8×0.8 = 0.64) so the
    # two stay matched twins. The shorter frame compresses the point cloud, so
    # the markers below are drawn a touch larger / more opaque to stay legible.
    figsize = (HALF_PAGE_IN[0] / 2, (HALF_PAGE_IN[1] / 2 - 0.45) * 0.64)
    fig, ax = plt.subplots(figsize=figsize)
    label_fs, title_fs, annot_fs = 7.0, 8.0, 6.5   # same as the η² histograms

    def _neglog10(p):
        return -np.log10(np.clip(p, 1e-300, 1.0))

    rng = np.random.default_rng(0)                 # seeded → reproducible jitter
    per = {}
    for ds in dss:
        p = outc.loc[outc.dataset == ds, "anova_p"].values
        per[ds] = _neglog10(p[np.isfinite(p)])
    ymax = float(np.ceil(max(v.max() for v in per.values()) / 10) * 10)

    for c, ds in enumerate(dss):
        nl = per[ds]
        x = c + rng.uniform(-0.28, 0.28, size=len(nl))
        ax.scatter(x, nl, s=7, color=ARM2_C, alpha=0.6, edgecolor="none",
                   zorder=3)
        med = float(np.median(nl))
        ax.plot([c - 0.40, c + 0.40], [med, med], color="black", lw=1.4, zorder=5)
        ax.text(c + 0.44, med, f"{med:.0f}", ha="left", va="center",
                fontsize=annot_fs)
        b = -np.log10(float(summary.loc[ds, "bonf_alpha"]))   # this dataset's α/m
        ax.plot([c - 0.40, c + 0.40], [b, b], color="black", lw=1.0, ls="--",
                zorder=4)
        # One count box per dataset, floated above its column (above the point
        # cloud so it never sits on a dot, cf. fig05d's "n tissues" boxes).
        n_sig = int(summary.loc[ds, "n_outcomes_sig_bonf"])
        m = int(summary.loc[ds, "n_tests_bonf"])
        pct = float(summary.loc[ds, "frac_outcomes_sig_bonf"]) * 100
        # Nudge each box a touch outward from centre: the one-line boxes are wider
        # than the 1.0-unit column spacing, so centred they would just touch.
        box_x = c + 0.06 * np.sign(c - (len(dss) - 1) / 2.0)
        ax.text(box_x, ymax * 1.07, f"sig {n_sig}/{m} ({pct:.0f}%)",
                ha="center", va="center", fontsize=annot_fs, color="0.2",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

    ax.set_xlim(-0.7, len(dss) - 1 + 0.7)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_xticks(range(len(dss)))
    ax.set_xticklabels([SHORT_DATASET[d] for d in dss])
    # Real subscript via mathtext — renders in Arial now that
    # apply_publication_style points the mathtext fontset at Arial (matches
    # fig05's η² notation). The test name ("Welch's ANOVA") lives in the title.
    ax.set_ylabel(r"$-\log_{10} p$", fontsize=label_fs)
    ax.set_title("Welch's ANOVA", fontsize=title_fs)

    # Two legend entries on one row (matched to fig05d's compact styling).
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=[
        Line2D([0], [0], color="black", lw=1.4, label="median"),
        Line2D([0], [0], color="black", lw=1.0, ls="--",
               label="Bonferroni α/m")],
        loc="lower center", ncol=2, frameon=False, fontsize=6.5,
        bbox_to_anchor=(0.5, 0.005), handlelength=1.2, handletextpad=0.5,
        columnspacing=1.0, labelspacing=0.3)
    _save(fig, out_dir, stem, tight=False)


# ───────────────────────────────────────────── fig05b — structure (PCA + corr)
def plot_structure(results_dir: Path, out_dir: Path, stem: str) -> None:
    """fig05b — supporting structure: PCA of expression coloured by tissue (top
    row) and within- vs across-tissue correlation histograms + Mann-Whitney p
    (bottom row), one column per dataset, sharing one OncotreeLineage legend."""
    summary = pd.read_csv(results_dir / "confounding_summary.csv")
    pca = pd.read_csv(results_dir / "confounding_pca.csv")
    corr = pd.read_csv(results_dir / "confounding_tissue_corr.csv")
    dss = _datasets(summary)
    n = len(dss)

    # Put CTRPv2 PCA samples on DepMap's OncotreeLineage ontology so the two PCA
    # panels share one tissue→colour map (and one legend) on a common ontology.
    pca = pca.copy()
    m_ctp = pca.dataset == "ctrpv2"
    if m_ctp.any():
        mapped = _ctrpv2_to_oncotree(pca.loc[m_ctp, "sample_id"])
        if mapped is not None:
            pca.loc[m_ctp, "tissue"] = mapped

    # Shared colour map: union of each panel's top-10 lineages, ordered by total
    # cell count so the headline tissues take the first (most distinct) hues; the
    # same tissue is then the same colour in both a and b.
    tops: set = set()
    for ds in dss:
        t = pca[(pca.dataset == ds) & (pca.tissue != "other")]["tissue"]
        tops |= set(t.value_counts().head(10).index)
    shared = (pca[pca.tissue.isin(tops)]["tissue"]
              .value_counts().index.tolist())          # freq-descending
    color_of = {t: SHARED_TISSUE_PALETTE[k % len(SHARED_TISSUE_PALETTE)]
                for k, t in enumerate(shared)}

    # Rows = panel type (PCA on top, Pearson distribution below); one column per
    # dataset. Panel letters run a, b across the top row, c, d across the bottom.
    # Full landscape-A4 page inside 1-inch margins, with the right strip reserved
    # for the shared tissue legend. Font sizes are fig05's (plot_evidence) + 1.
    title_fs, label_fs, box_fs = 9.0, 8.0, 7.5
    fig, axes = plt.subplots(2, n, figsize=(_USABLE_W, _USABLE_H), squeeze=False)
    for i, ds in enumerate(dss):
        s = summary[summary.dataset == ds].iloc[0]
        l_pca = chr(ord("a") + i)            # top row:    a, b, …
        l_corr = chr(ord("a") + n + i)       # bottom row: c, d, …

        # row 0 — PCA coloured by the shared tissue map
        ax = axes[0][i]
        # panel letter bold (Arial Bold via mathtext.bf), rest of title regular
        ax.set_title(rf"$\mathbf{{{l_pca}}}$  PCA of expression by tissue ({SHORT_DATASET[ds]})",
                     fontsize=title_fs)
        p = pca[pca.dataset == ds]
        other = p[~p.tissue.isin(color_of)]
        ax.scatter(other.PC1, other.PC2, s=5, c=OTHER_C, alpha=0.5, linewidths=0)
        for t in shared:                     # same draw order ⇒ same z-order a/b
            q = p[p.tissue == t]
            if len(q):
                ax.scatter(q.PC1, q.PC2, s=6, color=color_of[t], alpha=0.8,
                           linewidths=0)
        ax.set_xlabel(f"PC1 ({s.pc1_var_ratio*100:.0f}%)", fontsize=label_fs)
        ax.set_ylabel(f"PC2 ({s.pc2_var_ratio*100:.0f}%)", fontsize=label_fs)

        # row 1 — within vs across tissue correlation
        ax = axes[1][i]
        ax.set_title(rf"$\mathbf{{{l_corr}}}$  Within- vs across-tissue similarity ({SHORT_DATASET[ds]})",
                     fontsize=title_fs)
        c = corr[corr.dataset == ds]
        within = c[c.kind == "within"]["r"].values
        across = c[c.kind == "across"]["r"].values
        lo = float(min(within.min(), across.min())) if len(within) and len(across) else 0.0
        bins = np.linspace(lo, 1.0, 30)
        wc, ac = CORR_HIST_COLORS.get(ds, ("#4F9D69", "#B0883B"))
        # r̄ = mean of the per-pair Pearson r (matplotlib renders $\bar{r}$ in Arial)
        ax.hist(within, bins=bins, color=wc, alpha=0.7,
                label=rf"within  ($\bar{{r}}$ = {np.mean(within):.2f})", density=True,
                edgecolor="white", linewidth=0.3)
        ax.hist(across, bins=bins, color=ac, alpha=0.7,
                label=rf"across  ($\bar{{r}}$ = {np.mean(across):.2f})", density=True,
                edgecolor="white", linewidth=0.3)
        p_mwu = float(s.corr_mwu_p)
        # scipy's MWU normal-approximation tail returns absurd magnitudes
        # (10^-90+) that overstate the test's precision; report it as a bound at
        # a sensible floor rather than a spuriously exact value.
        if p_mwu < 1e-6:
            ptxt = r"$p < 10^{-6}$"
        else:
            mant, exp = f"{p_mwu:.1e}".split("e")
            ptxt = rf"$p = {mant}\times10^{{{int(exp)}}}$"
        ax.text(0.04, 0.96, f"MWU (across < within), {ptxt}",
                transform=ax.transAxes, ha="left", va="top", fontsize=box_fs,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                          lw=0.5))
        ax.set_xlabel("Pearson r", fontsize=label_fs)
        ax.set_ylabel("density", fontsize=label_fs)
        # Stack the within/across key under the MWU box in the empty upper-left
        # (the tall bars sit on the right, near r≈1), so it never overlaps them.
        ax.legend(loc="upper left", bbox_to_anchor=(0.035, 0.83), fontsize=box_fs,
                  frameon=False, handlelength=1.2, handletextpad=0.5)

    # Tick labels +1 over the global default (font_size − 1), matching the
    # bumped axis-label size for this figure.
    for ax in axes.flat:
        ax.tick_params(labelsize=label_fs)

    # Confine the panels to the left of the page, reserving the right strip for
    # the single shared tissue legend; keep the exact A4 budget (tight=False).
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=color_of[t], markeredgecolor="none",
                      label=t) for t in shared]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=4,
                          markerfacecolor=OTHER_C, markeredgecolor="none",
                          label="other"))
    # Vertically centred on the top (PCA) row, inside the reserved right strip.
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.845, 0.74),
               bbox_transform=fig.transFigure, fontsize=box_fs,
               frameon=False, handletextpad=0.2, labelspacing=0.3,
               borderaxespad=0.0, title="Tissue (OncotreeLineage)",
               title_fontsize=box_fs)
    _save(fig, out_dir, stem, tight=False)


# ─────────────────────────────────────── fig05c — confounder comparison
def _print_confounder_debias(cmp: pd.DataFrame, dss: list) -> None:
    """Caption helper: per-(dataset, arm) median Δη² and % items > 0, by
    confounder, descending — the validation that no-signal confounders collapse
    to ≈0 and the tissue family stays on top after debiasing."""
    print("fig05c debiased Δη² (real − per-item permuted floor):")
    for ds in dss:
        for arm in ("expression", "outcome"):
            a = cmp[(cmp.dataset == ds) & (cmp.arm == arm)]
            med = a.groupby("confounder")["value"].median().sort_values(
                ascending=False)
            pos = a.groupby("confounder")["value"].apply(lambda v: (v > 0).mean())
            line = ", ".join(f"{c} {med[c]:+.3f} ({pos[c]*100:.0f}%>0)"
                             for c in med.index)
            print(f"  [{ds}/{arm}] {line}")


def plot_confounder_comparison(results_dir: Path, out_dir: Path,
                               stem: str) -> None:
    """fig05c — "tissue, or something else?": debiased Δη² (real − each item's own
    permuted floor) for both arms across the candidate confounders, one boxplot
    row per confounder, per dataset. Falls back to raw η² if the null is absent."""
    cmp = pd.read_csv(results_dir / "confounding_confounder_compare.csv")
    # Per-item permuted floor → debias each item against its own chance level.
    # If the permuted-floor null is absent, fall back to raw η².
    null_path = results_dir / "confounding_confounder_compare_null.csv"
    debiased = null_path.exists()
    if debiased:
        nul = pd.read_csv(null_path)
        cmp = cmp.merge(nul, on=["dataset", "confounder", "arm", "item"],
                        how="left")
        cmp["value"] = (cmp["eta2"] - cmp["eta2_null"]).fillna(cmp["eta2"])
        pre = "Δη²"
    else:
        cmp["value"] = cmp["eta2"]
        pre = "η²"
    summary = pd.read_csv(results_dir / "confounding_summary.csv")
    dss = _datasets(summary)

    # One unified body font size for every in-axes label, tick and the legend;
    # only the panel letters (+1) and suptitle (+2) sit above it.
    FS = 8.0
    # ~0.62× the usable A4 height — seven confounders per panel don't need more,
    # and the spare vertical band was dead white space. Width still fills the page.
    fig, axes = plt.subplots(len(dss), 2,
                             figsize=(_USABLE_W, _USABLE_H * 0.62),
                             squeeze=False)
    dlt = r"\Delta" if debiased else ""        # mathtext renders in Arial (rcParams)
    # subscript "| c" = conditioned on confounder c (the per-row variable)
    arms = [("expression", ARM1_C, rf"${dlt}\eta^2_{{\mathrm{{g,expr}}\mid c}}$"),
            ("outcome",    ARM2_C, rf"${dlt}\eta^2_{{\mathrm{{o}}\mid c}}$")]

    def _ul(s):                                 # plain-Arial underline (no mathtext)
        return "".join(ch + "̲" for ch in s)

    min_whisker, max_whisker = 0.0, 0.0
    for i, ds in enumerate(dss):
        d = cmp[cmp.dataset == ds]
        # one confounder order per row: by expression-arm median, descending
        ex = d[d.arm == "expression"]
        order = (ex.groupby("confounder")["value"].median()
                 .sort_values().index.tolist())   # ascending → top = largest
        for j, (arm, color, xlab) in enumerate(arms):
            ax = axes[i][j]
            a = d[d.arm == arm]
            data = [a[a.confounder == c]["value"].values for c in order]
            data = [v if len(v) else np.array([np.nan]) for v in data]
            bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False,
                            patch_artist=True, medianprops=dict(color="black",
                                                                lw=1.0))
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=0.75, edgecolor="0.3",
                          linewidth=0.5)
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set(color="0.4", linewidth=0.5)
            for cap in bp["caps"]:               # track data extent → shared xlim
                xs = cap.get_xdata()
                max_whisker = max(max_whisker, float(np.nanmax(xs)))
                min_whisker = min(min_whisker, float(np.nanmin(xs)))
            # OncotreeLineage → just "Tissue", underlined; the reference confounder
            labels = ["Tissue" if c == "OncotreeLineage" else CONF_LABELS.get(c, c)
                      for c in order]
            ax.set_yticks(range(1, len(order) + 1))
            ax.set_yticklabels([_ul(t) if c == "OncotreeLineage" else t
                                for t, c in zip(labels, order)], fontsize=FS)
            for tick, c in zip(ax.get_yticklabels(), order):
                if c == "OncotreeLineage":
                    tick.set_fontweight("bold")
            ax.set_ylim(0.5, len(order) + 0.5)   # no vertical margin past the boxes
            if debiased:
                ax.axvline(0.0, color="0.35", lw=1.0, ls="--")  # own chance floor
            if i == len(dss) - 1:
                ax.set_xlabel(xlab, fontsize=FS)

    # Tighten the x-range to where the data actually is — at xlim=1 the right
    # ~40% of every panel sat empty. One shared range (driven by the widest
    # whisker on either side, so e.g. the CTRPv2-phenotype Subtype box's −0.06
    # whisker isn't clipped) keeps all four panels and both arms comparable.
    xhi = min(1.0, np.ceil(max_whisker * 20) / 20 + 0.02)
    xlo = min(-0.04, np.floor(min_whisker * 50) / 50) if debiased else 0.0
    for row in axes:
        for ax in row:
            ax.set_xlim(xlo, xhi)

    # Each column ranks every candidate confounder for one arm (tissue is just
    # one row), so the title names the arm, not a single confounder.
    for ax, lab in ((axes[0][0], "confounder → expression"),
                    (axes[0][1], "confounder → phenotype")):
        ax.set_title(lab, color="black", fontsize=FS)
    # Reserve thin top (suptitle), bottom (legend) and left (rotated dataset
    # labels) bands for the figure-level texts tight_layout can't see, then let
    # bbox_inches="tight" at save time crop whatever margin is left over.
    fig.tight_layout(rect=(0.028, 0.06, 0.995, 0.93))
    fig.suptitle("Comparison of other candidate confounders", y=0.95,
                 fontsize=FS + 2, fontweight="bold")
    # Panel letters a–d at each panel's top-left corner.
    for i in range(len(dss)):
        for j in range(2):
            pos = axes[i][j].get_position()
            fig.text(pos.x0 - 0.006, pos.y1 + 0.012,
                     chr(ord("a") + 2 * i + j), fontsize=FS + 1,
                     fontweight="bold", va="bottom", ha="left")
    for i, ds in enumerate(dss):
        pos = axes[i][0].get_position()
        fig.text(0.013, 0.5 * (pos.y0 + pos.y1), SHORT_DATASET[ds],
                 rotation=90, va="center", ha="center", fontweight="bold",
                 fontsize=FS, color="0.3")
    if debiased:
        fig.legend(handles=[Line2D([0], [0], color="0.35", lw=1.0, ls="--",
                                   label="Δη² = 0 (chance level)")],
                   loc="lower center", ncol=1, frameon=False, fontsize=FS,
                   bbox_to_anchor=(0.5, 0.015))
    _save(fig, out_dir, stem, tight=True)
    if debiased:
        _print_confounder_debias(cmp, dss)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    apply_publication_style(font_size=8)
    plot_evidence(args.results_dir, args.out_dir, "fig05_confounding_evidence")
    plot_clf_per_tissue(args.results_dir, args.out_dir,
                        "fig05d_classifier_per_tissue")
    plot_outcome_significance(args.results_dir, args.out_dir,
                              "fig05e_outcome_significance")
    plot_structure(args.results_dir, args.out_dir, "fig05b_confounding_structure")
    plot_confounder_comparison(args.results_dir, args.out_dir,
                               "fig05c_confounder_comparison")


if __name__ == "__main__":
    main()
