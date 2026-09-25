"""Split-half stability of the Fig-4 inputs (self-recovery, partner recovery,
PAM50 subtype specificity).

Fig 4a and 4b consume the same per-cell SHAP matrices as Figs 2/3, but through
different reductions:
  * self-recovery  — rank of the knocked-out gene in its item's mean-|SHAP|
    ranking (DepMap);
  * partner recovery — AUROC of coessentiality / STRING / CORUM partner sets
    against that ranking (DepMap);
  * PAM50 specificity — AUROC of the 27 lineage PAM50 genes against the
    mean-|SHAP| ranking computed WITHIN each tissue's cells (CTRPv2).

This script recomputes each metric from disjoint cell-line halves:
  A/B for the DepMap metrics come from results/halfmeans_depmap.npz (repeat 0
  of compute_stability.py); PAM50 needs within-tissue means, so it re-reads the
  CTRPv2 npz and splits each (drug, tissue) cell block with >= 2*MIN_CELLS
  cells into two halves that each meet the paper's own MIN_CELLS threshold
  (10 random splits, averaged).

Writes (to denoising/results/):
    fig4a_self_stability.csv      per method: mean self-AUROC per half, A-vs-B
                                  correlation across targets, delta vs marginal
                                  per half (ALL and INTER sets)
    fig4a_partner_stability.csv   same per (ground_truth, method)
    fig4b_pam50_stability.csv     per method: per-(drug,tissue) A-vs-B
                                  correlation and the breast-minus-other gap
                                  per half

Note: Fig 4a in the paper uses seed-averaged importances; the halves here are
seed-1 only, so 'full' columns are the seed-1 analogue, not the published
numbers.
"""
from __future__ import annotations

import glob
import importlib.util
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
sys.path.insert(0, str(ROOT / "fig04a_selfrank_partner"))
sys.path.insert(0, str(ROOT / "training"))
from recovery_utils import (METHODS, GS, auroc, bh_intersection,   # noqa: E402
                            coess_abs_r, curated_partners, load_chronos_and_tissue,
                            load_gmt, string_graph, tissue_residualize)
from shared.preds_io import load_preds  # noqa: E402


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F4B = _import(ROOT / "fig04b_breast_subtypes" / "compute_subtype_specificity.py",
              "f4b_mod")
MIN_CELLS = F4B.MIN_CELLS          # 10, the paper's own threshold
N_REPS = 10


def halfmeans():
    d = np.load(RES / "halfmeans_depmap.npz", allow_pickle=True)
    genes = np.asarray(d["genes"], dtype=str)
    items = np.asarray(d["items"], dtype=str)
    return d, genes, items


def paired_delta(piv: pd.DataFrame, m: str) -> dict:
    sub = piv[["marginal", m]].dropna()
    if len(sub) < 8:
        return {}
    try:
        _, p = stats.wilcoxon(sub[m], sub["marginal"])
    except ValueError:
        p = np.nan
    return {"n": len(sub), "marg_mean": sub["marginal"].mean(),
            "method_mean": sub[m].mean(),
            "delta": (sub[m] - sub["marginal"]).mean(), "wilcoxon_p": p}


# ------------------------------------------------------------- self-recovery
def self_stability():
    d, genes, items = halfmeans()
    inter = bh_intersection()
    gpos = {g: j for j, g in enumerate(genes)}
    N = len(genes)
    rows = []
    for i, target in enumerate(items):
        if target not in gpos:
            continue
        for m in METHODS:
            if f"halfA_{m}" not in d:
                continue
            r = {"target": target, "method": m, "in_inter": target in inter}
            for tag in ("halfA", "halfB", "full"):
                imp = d[f"{tag}_{m}"][i]
                if np.isnan(imp).all():
                    continue
                rank = int((np.nan_to_num(imp, nan=-np.inf) >=
                            imp[gpos[target]]).sum())
                r[tag] = (N - rank) / (N - 1)
            rows.append(r)
    df = pd.DataFrame(rows)
    out = []
    for m, g in df.groupby("method"):
        rec = {"method": m, "n_targets": len(g),
               "corr_AB": g[["halfA", "halfB"]].corr().iloc[0, 1],
               "mean_A": g.halfA.mean(), "mean_B": g.halfB.mean(),
               "mean_full": g.full.mean()}
        out.append(rec)
    summ = pd.DataFrame(out)
    # does the fig4a conclusion (delta vs marginal) reproduce from each half?
    deltas = []
    for setname in ("ALL", "INTER"):
        base = df if setname == "ALL" else df[df.in_inter]
        for tag in ("halfA", "halfB", "full"):
            piv = base.pivot_table(index="target", columns="method", values=tag)
            for m in [x for x in METHODS if x != "marginal" and x in piv]:
                r = paired_delta(piv, m)
                if r:
                    deltas.append({"set": setname, "half": tag, "method": m, **r})
    return summ, pd.DataFrame(deltas)


# ---------------------------------------------------------- partner recovery
def partner_stability():
    d, genes, items = halfmeans()
    inter = bh_intersection()
    crispr, tissue = load_chronos_and_tissue()
    crispr_res = tissue_residualize(crispr, tissue)
    G = string_graph()
    corum_g2s, corum_s2g = load_gmt(GS / "CORUM.gmt")
    rng = np.random.RandomState(0)
    panel = set(genes)
    rows = []
    for i, target in enumerate(sorted(items)):
        idx = int(np.where(items == target)[0][0])
        psets = {}
        for tag, mat in [("coessG", crispr), ("coessW", crispr_res)]:
            if target not in mat.columns:
                continue
            pg = [g for g in genes if g in mat.columns and g != target]
            if len(pg) < 50:
                continue
            r = coess_abs_r(mat, target, pg)
            psets[f"{tag}_0.01"] = set(r.nlargest(max(int(len(r) * 0.01), 5)).index)
        if target in G:
            nb = set(G.neighbors(target)) & panel
            if len(nb) >= 3:
                psets["string"] = nb
        cp = curated_partners(target, corum_g2s, corum_s2g, panel)
        if len(cp) >= 3:
            psets["CORUM"] = cp
        if "coessW_0.01" in psets:
            pool = [g for g in genes if g != target]
            psets["RANDOM"] = set(rng.choice(pool, size=len(psets["coessW_0.01"]),
                                             replace=False))
        if not psets:
            continue
        for m in METHODS:
            if f"halfA_{m}" not in d:
                continue
            for tag in ("halfA", "halfB", "full"):
                imp = d[f"{tag}_{m}"][idx]
                if np.isnan(imp).all():
                    continue
                row = {"target": target, "method": m, "half": tag,
                       "in_inter": target in inter}
                for gt, ps in psets.items():
                    row[f"auroc_{gt}"] = auroc(imp, genes, target, ps)
                rows.append(row)
    df = pd.DataFrame(rows)
    gts = [c.replace("auroc_", "") for c in df.columns if c.startswith("auroc_")]
    out, deltas = [], []
    for gt in gts:
        piv_ab = df.pivot_table(index=["target", "method"], columns="half",
                                values=f"auroc_{gt}").reset_index()
        for m, g in piv_ab.groupby("method"):
            out.append({"ground_truth": gt, "method": m,
                        "n_targets": g.target.nunique(),
                        "corr_AB": g[["halfA", "halfB"]].corr().iloc[0, 1],
                        "mean_A": g.halfA.mean(), "mean_B": g.halfB.mean(),
                        "mean_full": g.full.mean()})
        for setname in ("ALL", "INTER"):
            base = df if setname == "ALL" else df[df.in_inter]
            for tag in ("halfA", "halfB", "full"):
                piv = base[base.half == tag].pivot_table(
                    index="target", columns="method", values=f"auroc_{gt}")
                for m in [x for x in METHODS if x != "marginal" and x in piv]:
                    r = paired_delta(piv, m)
                    if r:
                        deltas.append({"ground_truth": gt, "set": setname,
                                       "half": tag, "method": m, **r})
    return pd.DataFrame(out), pd.DataFrame(deltas)


# ------------------------------------------------------------------- PAM50
def pam50_stability():
    lineage = F4B.PAM50_LINEAGE
    methods = F4B.METHODS
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "ctrpv2_seed1_preds" / "*.npz"))):
        d = load_preds(f)
        genes = np.array([str(g) for g in d["gene_cols"]])
        tis = d["tissue"].astype(str)
        lab = np.array([g in lineage for g in genes])
        if lab.sum() < 3 or lab.sum() == len(lab):
            continue
        shap = {m: None for m in methods}
        for t in np.unique(tis):
            mask = np.where(tis == t)[0]
            if len(mask) < 2 * MIN_CELLS:
                continue
            half = len(mask) // 2
            rng = np.random.default_rng(zlib.crc32(f"{Path(f).stem}:{t}".encode()))
            for m in methods:
                key = f"shap_{m}"
                if key not in d.files:
                    continue
                if shap[m] is None:
                    shap[m] = np.abs(d[key].astype(np.float32))
                A = shap[m][mask]
                aA, aB = [], []
                for _ in range(N_REPS):
                    perm = rng.permutation(len(mask))
                    with np.errstate(invalid="ignore"):
                        ia = np.nanmean(A[perm[:half]], axis=0)
                        ib = np.nanmean(A[perm[half:2 * half]], axis=0)
                    if np.isfinite(ia).all() and np.isfinite(ib).all():
                        aA.append(roc_auc_score(lab, ia))
                        aB.append(roc_auc_score(lab, ib))
                if aA:
                    with np.errstate(invalid="ignore"):
                        full = np.nanmean(A, axis=0)
                    rows.append((Path(f).stem, t, m, len(mask),
                                 float(np.mean(aA)), float(np.mean(aB)),
                                 float(roc_auc_score(lab, full))
                                 if np.isfinite(full).all() else np.nan))
    df = pd.DataFrame(rows, columns=["drug", "tissue", "method", "n_cells",
                                     "aurocA", "aurocB", "auroc_full"])
    df.to_csv(RES / "fig4b_pam50_rows.csv", index=False)

    # Same gap definition as fig04b_breast_subtypes/plot_breast_subtypes.py:
    # breast AUROC minus the mean of the 7 named non-epithelial tissues'
    # AUROCs (each tissue averaged over drugs first), not a row-level mean
    # over every non-breast (drug, tissue) pair.
    non_epi_tissues = ["Blood", "Lymph", "Muscle", "Bone", "Soft Tissue",
                       "Nervous System", "Brain"]
    min_drugs_per_tissue = 20

    def gap(g: pd.DataFrame, col: str) -> float:
        by_tissue = g.groupby("tissue")[col].agg(["mean", "count"])
        by_tissue = by_tissue[by_tissue["count"] >= min_drugs_per_tissue]
        non_epi = [t for t in non_epi_tissues if t in by_tissue.index]
        if "Breast" not in by_tissue.index or not non_epi:
            return float("nan")
        return by_tissue.loc["Breast", "mean"] - by_tissue.loc[non_epi, "mean"].mean()

    out = []
    for m, g in df.groupby("method"):
        out.append({"method": m, "n_rows": len(g),
                    "corr_AB": g[["aurocA", "aurocB"]].corr().iloc[0, 1],
                    "gap_A": gap(g, "aurocA"),
                    "gap_B": gap(g, "aurocB"),
                    "gap_full": gap(g, "auroc_full"),
                    "breast_full": g[g.tissue == "Breast"].auroc_full.mean()})
    return pd.DataFrame(out)


def main():
    summ, deltas = self_stability()
    summ.to_csv(RES / "fig4a_self_stability.csv", index=False)
    deltas.to_csv(RES / "fig4a_self_deltas.csv", index=False)
    print("== self-recovery: half-vs-half ==")
    print(summ.round(3).to_string(index=False))
    print("\n== self-recovery: delta vs marginal per half (INTER) ==")
    print(deltas[deltas.set == "INTER"].round(4).to_string(index=False))

    psumm, pdeltas = partner_stability()
    psumm.to_csv(RES / "fig4a_partner_stability.csv", index=False)
    pdeltas.to_csv(RES / "fig4a_partner_deltas.csv", index=False)
    print("\n== partner recovery: half-vs-half ==")
    print(psumm.round(3).to_string(index=False))

    pam = pam50_stability()
    pam.to_csv(RES / "fig4b_pam50_stability.csv", index=False)
    print("\n== PAM50 specificity: half-vs-half ==")
    print(pam.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
