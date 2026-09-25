"""Fig 3 (rank-difference GSEA) on the well-predicted subsets.

Does the Fig-3 GSEA hold up when restricted to the A.6.1 well-predicted subset?

Unlike Fig 2, Fig 3 cannot be filtered post hoc: the delta vector fed to the
pre-ranked GSEA is a *cross-item mean* importance percentile, so dropping items
changes every gene's score and the whole enrichment has to be recomputed.

Rather than fork compute_pathway_delta_gsea.py (which would risk this analysis and
the paper drifting apart), we build a symlink farm holding only the subset's .npz
files and point the real script at it with --preds-root. shared/preds_io.py resolves
the DANN/AD-AE extras as a `*_extra` sibling of the `*_preds` dir, so the farm
mirrors both directories.

The dot plot is rendered by the paper's own plot_pathway_delta_dotplot.py, executed
with __file__ pointed at a shadow directory (its input CSV and output dir are
module-level constants derived from __file__), again to avoid a forked copy.

Outputs per variant:
    preds/<variant>/{depmap,ctrpv2}_seed1_{preds,extra}/   symlink farm
    results/<variant>/pathway_delta_gsea.csv
    figures/fig03_<variant>.{pdf,png}
and across variants:
    results/fig03_concordance.csv     full-vs-subset NES agreement per method
    results/fig03_lineage_summary.csv demoted/promoted lineage-set counts

    python subset_fig03.py                        # all variants (slow: 10k perms)
    python subset_fig03.py --variants a61
    python subset_fig03.py --skip-gsea            # re-do tables/figures only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIG03 = REPO / "fig03_pathways"
PREDS_ROOT = REPO / "results"

DATASETS = ["depmap", "ctrpv2"]
ITEM_COL = {"depmap": "target", "ctrpv2": "drug"}
DECONF = ["residualized", "irm", "within_tissue", "dann", "adae"]
SEED = 1


def subset_items(dataset: str, variant: str) -> set:
    f = HERE / "results" / f"{dataset}_wt_items.csv"
    if not f.exists():
        raise SystemExit(f"{f} missing - run wt_subset.py first")
    df = pd.read_csv(f)
    return set(df.loc[df[f"in_{variant}"], ITEM_COL[dataset]])


def build_farm(variant: str, items_by_ds: dict) -> Path:
    """Symlink only the subset's npz (and their extras) under preds/<variant>/."""
    farm = HERE / "preds" / variant
    for ds, items in items_by_ds.items():
        for suffix in ["preds", "extra"]:
            src = PREDS_ROOT / f"{ds}_seed{SEED}_{suffix}"
            dst = farm / f"{ds}_seed{SEED}_{suffix}"
            if not src.exists():
                continue
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            n = 0
            for item in sorted(items):
                f = src / f"{item}.npz"
                if f.exists():
                    (dst / f.name).symlink_to(f)
                    n += 1
            print(f"  farm {dst.relative_to(HERE)}: {n}/{len(items)} npz")
    return farm


def run_gsea(farm: Path, out_csv: Path, n_perm: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "compute_pathway_delta_gsea.py",
           "--preds-root", str(farm), "--out", str(out_csv),
           "--n-perm", str(n_perm)]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=FIG03, check=True)


def render_dotplot(gsea_csv: Path, variant: str) -> None:
    """Run the paper's dot-plot script against `gsea_csv`.

    plot_pathway_delta_dotplot.py hardcodes HERE/"results"/pathway_delta_gsea.csv
    and HERE/"figures" off `Path(__file__).resolve().parent`. We hand it a shadow
    HERE: a real directory (not a symlink - .resolve() would undo that) holding the
    subset CSV, a gene_sets symlink, and an empty figures/.
    """
    shadow = HERE / "shadow" / variant
    (shadow / "results").mkdir(parents=True, exist_ok=True)
    (shadow / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(gsea_csv, shadow / "results" / "pathway_delta_gsea.csv")
    gs = shadow / "gene_sets"
    if not gs.exists():
        gs.symlink_to(FIG03 / "gene_sets", target_is_directory=True)

    script = FIG03 / "plot_pathway_delta_dotplot.py"
    argv, path = sys.argv[:], sys.path[:]
    sys.argv = [str(script), "--turned"]
    sys.path.insert(0, str(REPO / "training"))   # shared.plot
    sys.path.insert(0, str(FIG03))               # pathway_utils
    try:
        g = {"__file__": str(shadow / script.name), "__name__": "__main__"}
        exec(compile(script.read_text(), str(script), "exec"), g)
    finally:
        sys.argv, sys.path = argv, path

    for ext in ["pdf", "png"]:
        src = shadow / "figures" / f"fig_pathway_delta_dotplot_turned.{ext}"
        if src.exists():
            dst = HERE / "figures" / f"fig03_{variant}.{ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            print(f"  wrote {dst}")


def concordance(full: pd.DataFrame, sub: pd.DataFrame, variant: str) -> list:
    """How well does the subset GSEA reproduce the full one, per (dataset, method)?"""
    rows = []
    j = full.merge(sub, on=["dataset", "method", "library", "set"],
                   suffixes=("_full", "_sub"))
    for (ds, m), g in j.groupby(["dataset", "method"]):
        ok = g.NES_full.notna() & g.NES_sub.notna()
        g = g[ok]
        if len(g) < 3:
            continue
        sig_f = g.fdr_q_full < 0.05
        sig_s = g.fdr_q_sub < 0.05
        both = sig_f & sig_s
        rows.append({
            "variant": variant, "dataset": ds, "method": m, "n_sets": len(g),
            "pearson_NES": pearsonr(g.NES_full, g.NES_sub)[0],
            "spearman_NES": spearmanr(g.NES_full, g.NES_sub)[0],
            "sign_agree": float((np.sign(g.NES_full) == np.sign(g.NES_sub)).mean()),
            "n_sig_full": int(sig_f.sum()), "n_sig_sub": int(sig_s.sum()),
            "n_sig_both": int(both.sum()),
            # Of the sets the paper calls significant, how many stay significant
            # with the same direction on the subset? This is the "does it hold"
            # number; the reverse (recall of new sets) is not the claim.
            "frac_full_sig_replicated": float(
                (both & (np.sign(g.NES_full) == np.sign(g.NES_sub))).sum()
                / max(1, sig_f.sum())),
            "sign_agree_full_sig": float(
                (np.sign(g.NES_full[sig_f]) == np.sign(g.NES_sub[sig_f])).mean()
                if sig_f.any() else np.nan),
        })
    return rows


def lineage_summary(df: pd.DataFrame, variant: str) -> list:
    """Demoted / promoted split over the HGA lineage signatures (Fig 3's stacked bar)."""
    rows = []
    lin = df[df.library == "lineage"]
    for (ds, m), g in lin.groupby(["dataset", "method"]):
        sig = g[g.fdr_q < 0.05]
        rows.append({"variant": variant, "dataset": ds, "method": m,
                     "n_lineage": len(g),
                     "n_sig": len(sig),
                     "n_sig_demoted": int((sig.NES < 0).sum()),
                     "n_sig_promoted": int((sig.NES > 0).sum()),
                     "mean_NES": g.NES.mean(),
                     "median_NES": g.NES.median()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["a61", "marginal", "all4"])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--skip-gsea", action="store_true",
                    help="reuse existing results/<variant>/pathway_delta_gsea.csv")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    full = pd.read_csv(FIG03 / "results" / "pathway_delta_gsea.csv")
    conc, lin = [], lineage_summary(full, "full")

    for variant in a.variants:
        print(f"=== {variant} ===", flush=True)
        items = {ds: subset_items(ds, variant) for ds in DATASETS}
        out_csv = HERE / "results" / variant / "pathway_delta_gsea.csv"
        if not a.skip_gsea:
            farm = build_farm(variant, items)
            run_gsea(farm, out_csv, a.n_perm)
        if not out_csv.exists():
            print(f"  skip {variant}: {out_csv} missing")
            continue
        sub = pd.read_csv(out_csv)
        conc += concordance(full, sub, variant)
        lin += lineage_summary(sub, variant)
        if not a.no_figures:
            render_dotplot(out_csv, variant)

    res = HERE / "results"
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conc).to_csv(res / "fig03_concordance.csv", index=False)
    pd.DataFrame(lin).to_csv(res / "fig03_lineage_summary.csv", index=False)
    print(f"wrote {res/'fig03_concordance.csv'}")
    print(f"wrote {res/'fig03_lineage_summary.csv'}")


if __name__ == "__main__":
    main()
