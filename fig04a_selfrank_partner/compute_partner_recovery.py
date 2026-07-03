"""DepMap functional-partner recovery (Fig 4a).

For each knockout target, rank the panel genes by each method's global
importance (mean |SHAP|) and score the AUROC of recovering the target's
functional partners -- co-essentiality (coessG/coessW), STRING neighbours,
CORUM complex co-members -- plus a size-matched RANDOM-partner negative control
(all methods ~0.5, no marginal-vs-method gap). Paired Wilcoxon vs marginal on
the ALL and BH-intersection sets.

Usage:
  python compute_partner_recovery.py            # -> data/panel_newpanel_depmap*.csv
  python compute_partner_recovery.py --preds <dir> --label <name>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recovery_utils import (  # noqa: E402
    OUT, SEEDAVG, METHODS, bh_intersection, load_gmt, curated_partners,
    load_chronos_and_tissue, tissue_residualize, string_graph,
    coess_abs_r, auroc, GS)


def importances_from_preds(npz):
    """Handle both the full pipeline (shap_<m> per-cell matrices) and the
    lean runner (imp_<m> precomputed mean-|SHAP| vectors). IRM may be absent
    in the lean format."""
    gc = np.array([str(g) for g in npz["gene_cols"]])
    imps = {}
    for m in METHODS:
        if f"shap_{m}" in npz.files:
            with np.errstate(invalid="ignore"):
                imps[m] = np.nanmean(np.abs(npz[f"shap_{m}"].astype(np.float32)), axis=0)
        elif f"imp_{m}" in npz.files:
            imps[m] = npz[f"imp_{m}"].astype(np.float64)
    if "marginal" not in imps:
        return gc, None
    return gc, imps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(SEEDAVG))
    ap.add_argument("--label", default="newpanel_depmap")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()
    preds = Path(args.preds)

    inter = bh_intersection()
    crispr, tissue = load_chronos_and_tissue()
    crispr_res = tissue_residualize(crispr, tissue)
    G = string_graph()
    corum_g2s, corum_s2g = load_gmt(GS / "CORUM.gmt")
    rng = np.random.RandomState(args.rng_seed)

    targets = sorted({p.stem for p in preds.glob("*.npz")})
    print(f"[{args.label}] {len(targets)} targets, preds={preds}")
    rows = []
    for target in targets:
        npz = np.load(preds / f"{target}.npz")
        gene_cols, imps = importances_from_preds(npz)
        if imps is None:
            continue
        panel = set(gene_cols)
        psets = {}
        for tag, mat in [("coessG", crispr), ("coessW", crispr_res)]:
            if target not in mat.columns:
                continue
            pg = [g for g in gene_cols if g in mat.columns and g != target]
            if len(pg) < 50:
                continue
            r = coess_abs_r(mat, target, pg)
            psets[f"{tag}_0.01"] = set(r.nlargest(max(int(len(r)*0.01), 5)).index)
        if target in G:
            nb = set(G.neighbors(target)) & panel
            if len(nb) >= 3:
                psets["string"] = nb
        cp = curated_partners(target, corum_g2s, corum_s2g, panel)
        if len(cp) >= 3:
            psets["CORUM"] = cp
        # random-partner control: same size as coessW_0.01 set, random members
        if "coessW_0.01" in psets:
            pool = [g for g in gene_cols if g != target]
            psets["RANDOM"] = set(rng.choice(pool,
                                  size=len(psets["coessW_0.01"]), replace=False))
        if not psets:
            continue
        for m in imps:
            row = {"target": target, "method": m}
            for tag, ps in psets.items():
                row[f"auroc_{tag}"] = auroc(imps[m], gene_cols, target, ps)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"panel_{args.label}.csv", index=False)
    val_cols = [c for c in df.columns if c.startswith("auroc_")]
    out_rows = []
    for gt in val_cols:
        piv = df.pivot_table(index="target", columns="method", values=gt)
        for setname, S in [("ALL", set(piv.index)), ("INTER", inter)]:
            base = piv.loc[[i for i in piv.index if i in S]]
            for m in ["residualized", "within_tissue", "irm"]:
                if m not in base or "marginal" not in base:
                    continue
                sub = base[["marginal", m]].dropna()
                if len(sub) < 8:
                    continue
                try:
                    _, p = stats.wilcoxon(sub[m], sub["marginal"])
                except ValueError:
                    p = np.nan
                diff = sub[m].sub(sub["marginal"])
                out_rows.append({
                    "panel": args.label, "ground_truth": gt.replace("auroc_", ""),
                    "set": setname, "method": m, "n": len(sub),
                    "marg_mean": round(sub["marginal"].mean(), 4),
                    "method_mean": round(sub[m].mean(), 4),
                    "delta": round(diff.mean(), 4),
                    "frac_better": round((sub[m] > sub["marginal"]).mean(), 3),
                    "wilcoxon_p": p,
                    "marg_sem": round(sub["marginal"].sem(), 5),
                    "method_sem": round(sub[m].sem(), 5),
                    "delta_sem": round(diff.sem(), 5)})
    summ = pd.DataFrame(out_rows)
    summ.to_csv(OUT / f"panel_{args.label}_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(summ[summ.set == "INTER"].sort_values(["ground_truth", "method"]).to_string(index=False))
    print("\n--- ALL set ---")
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(summ[summ.set == "ALL"].sort_values(["ground_truth", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()
