"""Dot-plot of the rank-difference GSEA, BOTH DATASETS in one figure.

Two side-by-side panels:
  (a) DepMap (essentiality)   |   (b) CTRPv2 (drug response)

Each panel: columns = the three deconfounders (Residualized / IRM / Within-tissue);
rows = gene sets reaching FDR q<0.05 in >=1 of those three methods, sorted by mean
promotion-NES. Different gene sets surface per dataset, so each panel has its own
y-axis. Dot colour = NES (red promoted, blue demoted); dot AREA is linear in
-log10(FDR q) (floored at 1e-4); filled+ring = significant (q<0.05), open ring =
n.s. A left strip + label colour mark the gene-set class (lineage / tissue-context /
cell-intrinsic) assigned by build_set_classes.py. A small stacked bar BELOW each dot plot
summarises the per-method demoted/promoted split across ALL 38 HGA tissue
signatures (sharing the dot-plot's method columns). The colourbar, size legend
and category swatches are drawn ONCE in a horizontal strip below both panels.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                            # local pathway_utils
sys.path.insert(0, str(HERE.parent / "training"))        # bundled shared/
from shared.plot import PALETTE, apply_publication_style       # noqa
from pathway_utils import load_gmt                               # noqa
apply_publication_style(font_size=8)

METHODS = ["residualized", "irm", "within_tissue"]
MLAB = {"residualized": "Residualized", "irm": "IRM",
        "within_tissue": "Within-\ntissue", "dann": "DANN", "adae": "AD-AE"}
DLAB = {"depmap": "DepMap (essentiality)", "ctrpv2": "CTRPv2 (drug response)"}
C_TISSUE, C_FUNC = "#B5651D", "#9AA0A6"
C_MICRO = "#E0A458"   # tissue / microenvironment / cancer context (cell-extrinsic)
# Row classes from build_set_classes.py (KEGG BRITE + MSigDB Hallmark process
# categories): lineage (HGA) / context (tissue-associated) / cell-intrinsic.
_CLS = {}
for _ln in (HERE / "gene_sets" / "set_classes.tsv").read_text().splitlines()[1:]:
    _p = _ln.split("\t")
    if len(_p) >= 4:
        _CLS[_p[0]] = _p[3]
def rowclass(setname, library):
    c = _CLS.get(setname)
    if c == "intrinsic": return "intrinsic"
    if c == "lineage" or library == "lineage": return "lineage"
    return "micro"   # tissue / microenvironment / cancer context
COLByClass = {"lineage": C_TISSUE, "micro": C_MICRO, "intrinsic": C_FUNC}
MIN_COND = 1
QFLOOR = 1e-4
SCALE = 34.0

GS = HERE / "gene_sets"
SRC = {}
for nm in load_gmt(GS / "kegg.gmt"):      SRC[nm] = "KEGG"
for nm in load_gmt(GS / "hallmark.gmt"):  SRC.setdefault(nm, "Hallmark")
for nm in load_gmt(GS / "human_gene_atlas.gmt"): SRC.setdefault(nm, "HGA")

df = pd.read_csv(HERE / "results" / "pathway_delta_gsea.csv")
df = df[df.method.isin(METHODS)].copy()


def sizemap(q):
    return SCALE * np.clip(-np.log10(np.clip(q, QFLOOR, 1.0)), 0, None)


def disambiguate(names):
    low = {}
    for n in names:
        low.setdefault(n.lower(), []).append(n)
    rename = {}
    for grp in low.values():
        if len(grp) > 1:
            for n in grp:
                rename[n] = f"{n} ({SRC.get(n, '?')})"
    return rename


# The very long gene-set names cannot fit two lines at this column pitch without
# colliding, so give them hand-built two-line forms (each line short).
_ABBR_EXACT = {
    "Parathyroid hormone synthesis, secretion and action":
        "Parathyroid\nhorm. synth.",
    "Transcriptional misregulation in cancer": "Transcr.\nmisreg. cancer",
    "Neuroactive ligand-receptor interaction": "Neuroactive\nligand recept.",
    "Cytokine-cytokine receptor interaction": "Cytokine\nrecept. int.",
    "Leukocyte transendothelial migration": "Leukocyte\ntranse. migr.",
    "Epithelial Mesenchymal Transition": "EMT",
    "BronchialEpithelialCells": "Bronchial\nEpith. Cells",
    "TNF-alpha Signaling via NF-kB": "TNF-α\nvia NF-κB",
    "Unfolded Protein Response": "Unfolded\nProt. Resp.",
    "Nucleotide excision repair": "Nucleotide\nexc. repair",
}
# light rule-based shortening for the rest (applied before auto-wrapping)
_ABBR_RULES = [
    (" signaling pathway", " signaling"),
    ("Th1 and Th2 cell differentiation", "Th1/Th2 cell diff."),
    ("Estrogen Response", "Estrogen Resp."),
]


def abbrev(s):
    if s in _ABBR_EXACT:
        return _ABBR_EXACT[s]
    for a, b in _ABBR_RULES:
        s = s.replace(a, b)
    return s


def wrap_label(s, width=14):
    """Wrap onto two balanced lines (split at the word boundary minimising the
    longer line). Leaves already-hand-wrapped strings (containing a newline)
    untouched."""
    if "\n" in s or len(s) <= width:
        return s
    words = s.split()
    if len(words) < 2:
        return s
    best = None
    for i in range(1, len(words)):
        l1, l2 = " ".join(words[:i]), " ".join(words[i:])
        m = max(len(l1), len(l2))
        if best is None or m < best[0]:
            best = (m, f"{l1}\n{l2}")
    return best[1]


def precompute(dataset):
    d = df[df.dataset == dataset]
    sig = d[d.fdr_q < 0.05].groupby("set").size()
    keep = sig[sig >= MIN_COND].index
    d = d[d.set.isin(keep)]
    nes = d.pivot_table(index="set", columns="method", values="NES").reindex(columns=METHODS)
    qv = d.pivot_table(index="set", columns="method", values="fdr_q").reindex(columns=METHODS)
    lib = d.groupby("set").library.first()
    order = nes.mean(axis=1).sort_values(ascending=False).index
    nes, qv = nes.loc[order], qv.loc[order]
    rows = list(order)
    rename = disambiguate(rows)
    labels = [rename.get(r, r) for r in rows]
    vmax = float(np.ceil(np.nanmax(np.abs(nes.values)) * 10) / 10)
    return dict(rows=rows, nes=nes, qv=qv, lib=lib, labels=labels, vmax=vmax)


def draw_dotplot(ax, panel, norm, cmap, show_xticks=False):
    rows, nes, qv, lib, labels = (panel["rows"], panel["nes"], panel["qv"],
                                  panel["lib"], panel["labels"])
    nrow = len(rows)
    for yi, pw in enumerate(rows):
        rc = rowclass(pw, lib[pw])
        if rc == "lineage":
            ax.axhspan(yi - 0.5, yi + 0.5, color=C_TISSUE, alpha=0.16, zorder=0)
            ax.axhline(yi, color=C_TISSUE, lw=0.5, alpha=0.5, zorder=0)
        elif rc == "micro":
            ax.axhspan(yi - 0.5, yi + 0.5, color=C_MICRO, alpha=0.13, zorder=0)
            ax.axhline(yi, color=C_MICRO, lw=0.5, alpha=0.45, zorder=0)
        else:
            ax.axhline(yi, color=PALETTE["neutral_light"], lw=0.4, alpha=0.5, zorder=0)
    means = nes.mean(axis=1).values
    cross = np.where(np.diff(np.sign(means)))[0]
    if len(cross):
        ax.axhline(cross[0] + 0.5, color=PALETTE["neutral_dark"], lw=0.7,
                   ls=(0, (4, 3)), zorder=1)
    for ci, m in enumerate(METHODS):
        for yi, pw in enumerate(rows):
            v = nes.loc[pw, m]; q = qv.loc[pw, m]
            if np.isnan(v):
                continue
            col = cmap(norm(v))
            if q < 0.05:
                ax.scatter(ci, yi, s=sizemap(q), c=[col], edgecolors="black",
                           linewidths=0.7, zorder=4)
            else:
                ax.scatter(ci, yi, s=sizemap(q), facecolors="none", edgecolors=col,
                           linewidths=0.6, alpha=0.7, zorder=2)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(-0.7, nrow - 0.3); ax.invert_yaxis()
    ax.set_xticks(np.arange(len(METHODS)))
    if show_xticks:
        ax.set_xticklabels([MLAB[m] for m in METHODS], fontsize=6.8)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(labels, fontsize=6.2, rotation=20, ha="right",
                       rotation_mode="anchor")
    for tick, pw in zip(ax.get_yticklabels(), rows):
        rc = rowclass(pw, lib[pw])
        tick.set_color(COLByClass[rc] if rc != "intrinsic" else PALETTE["neutral_black"])
        if rc != "intrinsic":
            tick.set_fontweight("bold")
    for yi, pw in enumerate(rows):
        ax.add_patch(plt.Rectangle((-0.55, yi - 0.42), 0.12, 0.84,
                     color=COLByClass[rowclass(pw, lib[pw])], zorder=4, lw=0))
    ax.tick_params(length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.text(len(METHODS) - 0.45, 0.0, "promoted", rotation=90, va="top", ha="left",
            fontsize=6.5, color=PALETTE["red_strong"], fontweight="bold")
    ax.text(len(METHODS) - 0.45, nrow - 1.0, "demoted", rotation=90, va="bottom",
            ha="left", fontsize=6.5, color=PALETTE["blue_main"], fontweight="bold")


def draw_summary(ax, dataset):
    """Per-method demoted/promoted split across ALL 53 HGA tissue signatures.
    x-axis aligns with the dot plot above (3 method columns)."""
    tis = df[(df.dataset == dataset) & (df.library == "lineage")]
    tot = tis[tis.method == "within_tissue"].shape[0]
    for i, m in enumerate(METHODS):
        tm = tis[tis.method == m]
        dem = int((tm.NES < 0).sum())
        pro = int((tm.NES > 0).sum())
        ax.bar(i, dem, color=PALETTE["blue_main"], width=0.62, zorder=2)
        ax.bar(i, pro, bottom=dem, color=PALETTE["red_strong"],
               width=0.62, zorder=2)
        ax.text(i, dem / 2, f"{dem}", va="center", ha="center",
                fontsize=6.6, color="white", fontweight="bold", zorder=3)
        if pro > 2:
            ax.text(i, dem + pro / 2, f"{pro}", va="center", ha="center",
                    fontsize=6.6, color="white", fontweight="bold", zorder=3)
        else:
            ax.text(i, dem + pro + 1.5, f"{pro}", va="bottom", ha="center",
                    fontsize=6.4, color=PALETTE["red_strong"],
                    fontweight="bold", zorder=3)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([MLAB[m] for m in METHODS], fontsize=6.8)
    ax.set_ylim(0, tot)
    ax.set_yticks([0, tot])
    ax.set_yticklabels(["0", str(tot)], fontsize=6.0)
    ax.tick_params(axis="x", length=2, width=0.5, pad=2)
    ax.tick_params(axis="y", length=2, width=0.5)
    ax.set_ylabel(f"of {tot} HGA\ntissue sig.", fontsize=6.4,
                  color=PALETTE["neutral_dark"], labelpad=4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


# ===========================================================================
# Turned (transposed) variant: methods become the 3 ROWS, gene sets run along
# the X axis. Collapses the tall portrait dotplot into a wide, short landscape
# figure. Both datasets sit side by side so the 3 method rows appear only once
# (minimal vertical height); each keeps its own gene sets.
# ===========================================================================
def draw_dotplot_turned(ax, panel, norm, cmap, show_ylabels=True):
    rows, nes, qv, lib, labels = (panel["rows"], panel["nes"], panel["qv"],
                                  panel["lib"], panel["labels"])
    ncol = len(rows)
    nmeth = len(METHODS)
    # column background shading by gene-set class
    for xi, pw in enumerate(rows):
        rc = rowclass(pw, lib[pw])
        if rc == "lineage":
            ax.axvspan(xi - 0.5, xi + 0.5, color=C_TISSUE, alpha=0.16, zorder=0)
        elif rc == "micro":
            ax.axvspan(xi - 0.5, xi + 0.5, color=C_MICRO, alpha=0.13, zorder=0)
    # vertical separator at the promoted/demoted sign crossing
    means = nes.mean(axis=1).values
    cross = np.where(np.diff(np.sign(means)))[0]
    if len(cross):
        ax.axvline(cross[0] + 0.5, color=PALETTE["neutral_dark"], lw=0.7,
                   ls=(0, (4, 3)), zorder=1)
    for yi, m in enumerate(METHODS):
        for xi, pw in enumerate(rows):
            v = nes.loc[pw, m]; q = qv.loc[pw, m]
            if np.isnan(v):
                continue
            col = cmap(norm(v))
            if q < 0.05:
                ax.scatter(xi, yi, s=sizemap(q), c=[col], edgecolors="black",
                           linewidths=0.7, zorder=4)
            else:
                ax.scatter(xi, yi, s=sizemap(q), facecolors="none", edgecolors=col,
                           linewidths=0.6, alpha=0.7, zorder=2)
    ax.set_xlim(-0.7, ncol - 0.3)
    ax.set_ylim(-0.95, nmeth - 0.4); ax.invert_yaxis()
    ax.set_yticks(range(nmeth))
    if show_ylabels:
        ax.set_yticklabels([MLAB[m] for m in METHODS], fontsize=7.5)
    else:
        ax.set_yticklabels([])
    ax.set_xticks(range(ncol))
    # vertical labels (hang straight below the axis): abbreviations keep the
    # two-line stack narrower than the column pitch, so no collisions and no
    # angle needed. NB: no rotation_mode="anchor" here -- with rotation=90 it
    # shoves the labels up into the dot rows.
    ax.set_xticklabels([wrap_label(abbrev(l)) for l in labels], rotation=90,
                       ha="center", va="top", fontsize=5.3,
                       multialignment="center", linespacing=0.85)
    ax.tick_params(axis="x", pad=2)
    for tick, pw in zip(ax.get_xticklabels(), rows):
        rc = rowclass(pw, lib[pw])
        tick.set_color(COLByClass[rc] if rc != "intrinsic" else PALETTE["neutral_black"])
        if rc != "intrinsic":
            tick.set_fontweight("bold")
    # thin class strip along the TOP of the panel
    for xi, pw in enumerate(rows):
        ax.add_patch(plt.Rectangle((xi - 0.5, -0.85), 1.0, 0.16,
                     color=COLByClass[rowclass(pw, lib[pw])], zorder=4, lw=0))
    ax.tick_params(length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    # promoted (left) / demoted (right) cues above the strip
    ax.text(-0.3, -0.92, "promoted", va="bottom", ha="left", fontsize=6.3,
            color=PALETTE["red_strong"], fontweight="bold")
    ax.text(ncol - 0.7, -0.92, "demoted", va="bottom", ha="right", fontsize=6.3,
            color=PALETTE["blue_main"], fontweight="bold")


def draw_summary_turned(ax, dataset):
    """Per-method demoted/promoted split across all HGA tissue signatures, as
    horizontal stacked bars aligned with the dot plot's 3 method rows."""
    tis = df[(df.dataset == dataset) & (df.library == "lineage")]
    tot = tis[tis.method == "within_tissue"].shape[0]
    for i, m in enumerate(METHODS):
        tm = tis[tis.method == m]
        dem = int((tm.NES < 0).sum()); pro = int((tm.NES > 0).sum())
        ax.barh(i, dem, color=PALETTE["blue_main"], height=0.6, zorder=2)
        ax.barh(i, pro, left=dem, color=PALETTE["red_strong"], height=0.6, zorder=2)
        ax.text(dem / 2, i, f"{dem}", va="center", ha="center", fontsize=6.4,
                color="white", fontweight="bold", zorder=3)
        if pro > 2:
            ax.text(dem + pro / 2, i, f"{pro}", va="center", ha="center",
                    fontsize=6.4, color="white", fontweight="bold", zorder=3)
        else:
            ax.text(dem + pro + 0.5, i, f"{pro}", va="center", ha="left",
                    fontsize=6.2, color=PALETTE["red_strong"], fontweight="bold",
                    zorder=3)
    ax.set_ylim(-0.95, len(METHODS) - 0.4); ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlim(0, tot); ax.set_xticks([0, tot])
    ax.set_xticklabels(["0", str(tot)], fontsize=6.0)
    ax.set_xlabel(f"of {tot} HGA tissue sig.", fontsize=6.2,
                  color=PALETTE["neutral_dark"], labelpad=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=2, width=0.5)


def make_turned():
    panels = {ds: precompute(ds) for ds in ["depmap", "ctrpv2"]}
    vmax = max(p["vmax"] for p in panels.values())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r
    na = len(panels["depmap"]["rows"]); nb = len(panels["ctrpv2"]["rows"])

    # --- layout in inches (STACKED: DepMap over CTRPv2) ---
    PX = 0.178                       # per gene-set column (sets the wider panel)
    PY = 0.36                        # per method row
    DOTB = PY * len(METHODS)         # dot band height
    W_dot = PX * max(na, nb)         # both panels share this width -> summaries align
    LEFT_LAB = 1.05                  # method y-labels
    SUM_W = 0.70; GAPC = 0.12        # summary bar + gap to its dot plot
    RIGHT = 0.25
    TOP = 0.10; TITLE = 0.34; XLAB = 0.62   # vertical 2-line labels (abbreviated)
    GAP_AB = 0.12                    # gap between the two stacked dataset blocks
    GAP_LEG = 0.10; LEG = 0.72; BOT = 0.10

    BLOCK = TITLE + DOTB + XLAB      # one dataset block height
    W = LEFT_LAB + W_dot + GAPC + SUM_W + RIGHT
    H = TOP + BLOCK + GAP_AB + BLOCK + GAP_LEG + LEG + BOT
    fig = plt.figure(figsize=(W, H))

    def FX(v): return v / W
    def FY(v): return v / H

    x_dot = LEFT_LAB
    x_sum = x_dot + W_dot + GAPC

    def add_block(top_in, dataset, panel):
        d_top = top_in - TITLE
        d_bot = d_top - DOTB
        ax = fig.add_axes([FX(x_dot), FY(d_bot), FX(W_dot), FY(DOTB)])
        ax_s = fig.add_axes([FX(x_sum), FY(d_bot), FX(SUM_W), FY(DOTB)])
        draw_dotplot_turned(ax, panel, norm, cmap, show_ylabels=True)
        draw_summary_turned(ax_s, dataset)
        return d_top, d_bot

    # panel a (top) and b (below)
    a_top, a_bot = add_block(H - TOP, "depmap", panels["depmap"])
    b_top, b_bot = add_block(a_bot - XLAB - GAP_AB, "ctrpv2", panels["ctrpv2"])

    # titles + (a)/(b)
    for d_top, ds, n, lab in ((a_top, "depmap", na, "a"),
                              (b_top, "ctrpv2", nb, "b")):
        fig.text(FX(x_dot + W_dot / 2), FY(d_top + 0.07), DLAB[ds],
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        fig.text(FX(0.12), FY(d_top + 0.07), lab, ha="left", va="bottom",
                 fontsize=14, fontweight="bold")

    # ===== legend block across the bottom =====
    # Two clearly separated rows (no overlap):
    #   upper row : colourbar (left)  +  FDR-q size key (right)
    #   lower row : the three gene-set category swatches, centred
    cb_y = BOT + 0.48               # colourbar bar bottom (label hangs below)
    cat_y = BOT + 0.06              # category swatch row centre

    cb_w = 2.1; cb_x = LEFT_LAB + 0.30
    cax = fig.add_axes([FX(cb_x), FY(cb_y), FX(cb_w), FY(0.13)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("promotion NES", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, pad=1)

    sl = [(0.05, "0.05"), (0.01, "0.01"), (1e-3, "0.001"), (1e-4, r"$\leq$1e-4")]
    h_size = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=PALETTE["neutral_mid"], markeredgecolor="black",
                     markeredgewidth=0.6, markersize=np.sqrt(sizemap(q)), label=f"q={lab}")
              for q, lab in sl]
    h_size += [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor=PALETTE["neutral_mid"], markeredgewidth=0.6,
                      markersize=np.sqrt(sizemap(0.2)), label="n.s. (open)")]
    fig.legend(handles=h_size, loc="center",
               bbox_to_anchor=(FX(cb_x + cb_w + 2.55), FY(cb_y + 0.05)),
               fontsize=6.8, frameon=False,
               title=r"FDR q  (area $\propto\,-\log_{10}q$)", title_fontsize=7.2,
               ncol=5, columnspacing=1.2, handletextpad=0.3)

    cat = [Line2D([0], [0], marker="s", color="w", markerfacecolor=C_TISSUE,
                  markersize=8, label="Lineage signature (HGA)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=C_MICRO,
                  markersize=8, label="Tissue / microenvironment / cancer context"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=C_FUNC,
                  markersize=8, label="Cell-intrinsic pathway")]
    fig.legend(handles=cat, loc="center", bbox_to_anchor=(0.5, FY(cat_y)),
               fontsize=6.6, frameon=False, ncol=3, handletextpad=0.4,
               columnspacing=1.4)

    stem = "fig_pathway_delta_dotplot_turned"
    fig.savefig(HERE / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figures" / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{stem}.png  (a:{na}, b:{nb} pathways)  {W:.1f}x{H:.1f}in")


def make_combined():
    panels = {ds: precompute(ds) for ds in ["depmap", "ctrpv2"]}
    vmax = max(p["vmax"] for p in panels.values())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    nrow_a = len(panels["depmap"]["rows"])
    nrow_b = len(panels["ctrpv2"]["rows"])
    H_PER = 0.18                 # dot-plot height per row, inches
    H_a = H_PER * nrow_a         # dot plot a height
    H_b = H_PER * nrow_b         # dot plot b height
    SUM_H = 0.65                 # summary bar height
    GAP_DS = 0.18                # gap dot-plot ↔ summary (a little extra white space)
    TOP_PAD = 0.55
    BOT_PAD = 0.30
    # Panel a is the tallest column (dot plot + gap + summary); panel b is
    # top-aligned and shorter, leaving an empty L-shaped region under it.
    H_total = TOP_PAD + H_a + GAP_DS + SUM_H + BOT_PAD
    W = 11.5

    fig = plt.figure(figsize=(W, H_total))

    # x positions
    LX_a, DW = 0.13, 0.27
    LX_b = 0.59

    # vertical anchors (figure fraction). Both dot plots' TOPS align.
    top_frac = 1.0 - TOP_PAD / H_total
    gap_frac = GAP_DS / H_total
    sum_h_frac = SUM_H / H_total
    a_h = H_a / H_total
    b_h = H_b / H_total

    # Panel a — extends down to the figure bottom
    ax_a_dot = fig.add_axes([LX_a, top_frac - a_h, DW, a_h])
    a_sum_top = top_frac - a_h - gap_frac
    ax_a_sum = fig.add_axes([LX_a, a_sum_top - sum_h_frac, DW, sum_h_frac])
    draw_dotplot(ax_a_dot, panels["depmap"], norm, cmap, show_xticks=False)
    draw_summary(ax_a_sum, "depmap")

    # Panel b — TOP-aligned with panel a; shorter, so empty space below
    ax_b_dot = fig.add_axes([LX_b, top_frac - b_h, DW, b_h])
    b_sum_top = top_frac - b_h - gap_frac
    ax_b_sum = fig.add_axes([LX_b, b_sum_top - sum_h_frac, DW, sum_h_frac])
    draw_dotplot(ax_b_dot, panels["ctrpv2"], norm, cmap, show_xticks=False)
    draw_summary(ax_b_sum, "ctrpv2")

    # Panel titles + (a)/(b) labels (both at the same top y)
    title_y = top_frac + 0.18 / H_total
    sub_y = top_frac + 0.04 / H_total
    fig.text(LX_a + DW / 2, title_y, DLAB["depmap"],
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(LX_a + DW / 2, sub_y,
             r"gene sets FDR q$<$0.05 in $\geq$1 of the 3 methods "
             f"(n={nrow_a})", ha="center", va="bottom", fontsize=6.6,
             color=PALETTE["neutral_dark"])
    fig.text(LX_b + DW / 2, title_y, DLAB["ctrpv2"],
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(LX_b + DW / 2, sub_y,
             r"gene sets FDR q$<$0.05 in $\geq$1 of the 3 methods "
             f"(n={nrow_b})", ha="center", va="bottom", fontsize=6.6,
             color=PALETTE["neutral_dark"])
    fig.text(0.04, title_y, "a", ha="left", va="bottom",
             fontsize=14, fontweight="bold")
    fig.text(LX_b - 0.06, title_y, "b", ha="left", va="bottom",
             fontsize=14, fontweight="bold")

    # ===== legend block, placed UNDER panel b in the empty L-shape =====
    # Top edge of panel b's empty region:
    b_sum_bottom = b_sum_top - sum_h_frac
    WHITESPACE = 0.55                       # inches of clear white space first
    leg_top = b_sum_bottom - WHITESPACE / H_total

    # Horizontal colourbar at the top of the legend block
    cb_h = 0.14 / H_total
    cax = fig.add_axes([LX_b, leg_top - cb_h, DW, cb_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.set_label("promotion NES", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, pad=1)

    # FDR q size legend just below the colourbar
    sl = [(0.05, "0.05"), (0.01, "0.01"), (1e-3, "0.001"), (1e-4, r"$\leq$1e-4")]
    h_size = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=PALETTE["neutral_mid"],
                     markeredgecolor="black", markeredgewidth=0.6,
                     markersize=np.sqrt(sizemap(q)), label=f"q={lab}")
              for q, lab in sl]
    h_size += [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor=PALETTE["neutral_mid"], markeredgewidth=0.6,
                      markersize=np.sqrt(sizemap(0.2)), label="n.s. (open)")]
    fig.legend(handles=h_size, loc="upper center",
               bbox_to_anchor=(LX_b + DW / 2, leg_top - cb_h - 0.50 / H_total),
               fontsize=6.8, frameon=False,
               title=r"FDR q  (area $\propto\,-\log_{10}q$)",
               title_fontsize=7.2, ncol=5, columnspacing=1.3, handletextpad=0.3)

    # Category swatches at the bottom of the legend block
    cat = [Line2D([0], [0], marker="s", color="w", markerfacecolor=C_TISSUE,
                  markersize=8, label="Lineage signature (HGA)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=C_MICRO,
                  markersize=8, label="Tissue / microenvironment / cancer context"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=C_FUNC,
                  markersize=8, label="Cell-intrinsic pathway")]
    fig.legend(handles=cat, loc="upper center",
               bbox_to_anchor=(LX_b + DW / 2,
                               leg_top - cb_h - 1.40 / H_total),
               fontsize=6.4, frameon=False, ncol=3,
               handletextpad=0.4, columnspacing=0.9)

    stem = "fig_pathway_delta_dotplot"
    fig.savefig(HERE / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figures" / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{stem}.png  (a:{nrow_a}, b:{nrow_b} pathways)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turned", action="store_true",
                   help="Transposed landscape layout (methods as 3 rows, gene "
                        "sets along X) optimized for minimal vertical height.")
    p.add_argument("--include-adv", action="store_true",
                   help="also plot the DANN/AD-AE columns; off by default so "
                        "this reproduces the published figure.")
    args = p.parse_args()
    if args.include_adv:
        METHODS.extend(["dann", "adae"])
    if args.turned:
        make_turned()
    else:
        make_combined()
