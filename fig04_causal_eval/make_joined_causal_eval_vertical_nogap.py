"""Figure 4: joined biological-evaluation figure (portrait, two panels).

Two panels stacked in a single column, portrait, fitting one text column:
   (a) DepMap self + functional-partner recovery (Delta AUROC vs Marginal)  [top]
   (b) CTRPv2 per-tissue PAM50 specificity (tissues down the y-axis)         [bottom, tall]
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests
from pathlib import Path

HERE = Path(__file__).resolve().parent
FF = HERE.parent
SF = FF / "fig04a_selfrank_partner" / "data"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
M3 = ["residualized", "within_tissue", "irm"]
M4 = ["marginal", "residualized", "within_tissue", "irm"]
COL = {"marginal": "#9aa0a6", "residualized": "#1f77b4",
       "within_tissue": "#2ca02c", "irm": "#a51e63"}
LAB = {"marginal": "Marginal", "residualized": "Residualized",
       "within_tissue": "Within-tissue", "irm": "IRM"}

TITLE, LABEL, TICK, LEG, ANNOT = 11, 9.5, 8.5, 8.5, 8

# ---- (a) DepMap self + partner (INTER set) ----
selfs = pd.read_csv(SF / "selfrank_newpanel_summary.csv"); selfs = selfs[selfs.set == "INTER"]
part = pd.read_csv(SF / "panel_newpanel_depmap_summary.csv"); part = part[part.set == "INTER"]
cats = [("Self", None), ("STRING", "string"), ("CORUM", "CORUM"), ("Random", "RANDOM")]
rows, keys, praw = {}, [], []
for cat, gt in cats:
    for m in M3:
        r = (selfs[selfs.method == m].iloc[0] if gt is None
             else part[(part.ground_truth == gt) & (part.method == m)].iloc[0])
        rows[(cat, m)] = (r["delta"], r["delta_sem"], r["wilcoxon_p"])
        keys.append((cat, m)); praw.append(r["wilcoxon_p"])
q = multipletests(praw, method="fdr_bh")[1]
qd = {k: q[i] for i, k in enumerate(keys)}
star = lambda x: "***" if x < 1e-3 else "**" if x < 1e-2 else "*" if x < 0.05 else ""

# ---- (b) CTRPv2 per-tissue PAM50 AUROC ----
bs = pd.read_csv(FF / "fig04b_breast_subtypes/results/subtype_tissue_specificity.csv")
NON_EPI = ["Blood", "Lymph", "Muscle", "Bone", "Soft Tissue", "Nervous System", "Brain"]
mn = (bs.groupby(["tissue", "method"]).agg(a=("auroc", "mean"), n=("auroc", "count")).reset_index())
piv = mn[mn.n >= 20].pivot(index="tissue", columns="method", values="a").sort_values("residualized", ascending=False)
non = [t for t in piv.index if t in NON_EPI]

plt.rcParams.update({
    "font.size": TICK, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEG,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})
fig = plt.figure(figsize=(5.0, 9.3))
gs = fig.add_gridspec(2, 1, height_ratios=[2.75, 3.4], hspace=0.24)
axA = fig.add_subplot(gs[0])      # top          -> recovery (horizontal bars)
axC = fig.add_subplot(gs[1])      # bottom, tall -> per-tissue

# (a) DepMap recovery, horizontal grouped bars (categories down the y-axis)
catnames = [c for c, _ in cats]; y = np.arange(len(catnames)); w = 0.26
for j, m in enumerate(M3):
    ds = [rows[(c, m)][0] for c in catnames]; es = [rows[(c, m)][1] for c in catnames]
    axA.barh(y + (j - 1) * w, ds, w, xerr=es, capsize=2, color=COL[m], label=LAB[m],
             error_kw=dict(lw=0.7))
    for yi, c in zip(y, catnames):
        d, e, _ = rows[(c, m)]; s = star(qd[(c, m)])
        if s:
            ee = e if not np.isnan(e) else 0
            end = (d + ee + 0.003) if d >= 0 else (d - ee - 0.003)
            axA.text(end, yi + (j - 1) * w, s, ha=("left" if d >= 0 else "right"),
                     va="center", fontsize=ANNOT)
axA.axvline(0, color="#444", lw=0.7)
axA.set_yticks(y); axA.set_yticklabels(catnames); axA.invert_yaxis()
axA.set_xlabel(r"$\Delta$ AUROC vs Marginal")
axA.set_title("a  Self- and partner-recovery (DepMap)", loc="left")
axA.margins(x=0.08)   # method colours are keyed by panel b's legend below

# shared marker scheme (values on x): Breast = filled circle,
# other epithelial = open circle, non-epithelial = filled square
DOT = 34
def cat_marker(ax, xval, yval, m, cat):
    if cat == "epi":
        ax.scatter(xval, yval, facecolor="white", edgecolor=COL[m], lw=1.4, s=DOT,
                   marker="o", zorder=3)
    else:
        ax.scatter(xval, yval, color=COL[m], s=DOT,
                   marker=("o" if cat == "breast" else "s"), zorder=3)

# (b) per-tissue -> tissues down the y-axis, AUROC on x
yt = np.arange(len(piv))
for i, t in enumerate(piv.index):
    if t in non: axC.axhspan(i - 0.5, i + 0.5, color="#dddddd", alpha=0.5, zorder=0)
    if t == "Breast": axC.axhspan(i - 0.5, i + 0.5, color="#c9e2f3", alpha=0.7, zorder=0)
cat_of = {t: ("breast" if t == "Breast" else "nonepi" if t in non else "epi") for t in piv.index}
for m in M4:
    axC.plot(piv[m].values, yt, color=COL[m], lw=1.7, zorder=2)
    for i, t in enumerate(piv.index):
        cat_marker(axC, piv.loc[t, m], i, m, cat_of[t])
axC.set_yticks(yt); axC.set_yticklabels(piv.index); axC.invert_yaxis()
for i, t in enumerate(piv.index):
    if t == "Breast":
        axC.get_yticklabels()[i].set_color("#1f77b4"); axC.get_yticklabels()[i].set_fontweight("bold")
axC.set_xlabel("PAM50 AUROC")
axC.set_title("b  Breast-subtype specificity (CTRPv2)", loc="left")
# data runs top-right (breast, high) -> bottom-left (blood, low), so the
# top-left and bottom-right corners are empty: put the legends there.
box = dict(frameon=True, framealpha=0.9, edgecolor="0.7", facecolor="white")
leg1 = axC.legend(handles=[Line2D([0], [0], color=COL[m], lw=2.2, label=LAB[m]) for m in M4],
                  ncol=1, loc="lower right", handletextpad=0.5, labelspacing=0.3,
                  **box)
leg1.get_frame().set_linewidth(0.6)
axC.add_artist(leg1)
leg2 = axC.legend(handles=[Line2D([0], [0], marker="o", color="0.35", lw=0, label="Breast"),
                           Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="0.35", lw=0, label="Other epithelial"),
                           Line2D([0], [0], marker="s", color="0.35", lw=0, label="Non-epithelial")],
                  ncol=1, loc="upper left", bbox_to_anchor=(0.0, 0.88),
                  handletextpad=0.3, labelspacing=0.25, **box)
leg2.get_frame().set_linewidth(0.6)
xmin, xmax = float(piv[M4].min().min()), float(piv[M4].max().max())
axC.set_xlim(xmin - 0.03, xmax + 0.05); axC.margins(y=0.01)

fig.savefig(OUT / "fig_joined_causal_eval_vertical_nogap.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig_joined_causal_eval_vertical_nogap.pdf", bbox_inches="tight", pad_inches=0.01)
print("saved", OUT / "fig_joined_causal_eval_vertical_nogap.png")
print("saved", OUT / "fig_joined_causal_eval_vertical_nogap.pdf")
