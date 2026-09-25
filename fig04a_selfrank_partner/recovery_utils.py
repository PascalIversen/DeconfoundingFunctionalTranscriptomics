"""Helpers for DepMap functional-partner recovery (Fig 4a).

Ground-truth partner sets for a knockout target, all intersected with the
feature panel:
  * co-essentiality: |Pearson r| of Chronos gene-effect profiles (coessG) and
    tissue-residualized profiles (coessW), top fraction per target;
  * STRING v12 one-hop neighbours (combined score >= 700);
  * CORUM complex co-members.
Plus the BH-intersection of targets whose marginal AND residualized within-tissue
r pass BH-FDR (q<0.05), and an AUROC scorer for ranking a partner set by |SHAP|.
Used by compute_self_recovery.py and compute_partner_recovery.py.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))   # repo shared/ package
import shared.data as D  # noqa: E402

RESULTS = HERE.parent / "results"
SEEDAVG = RESULTS / "depmap_seedavg_preds"
from shared.preds_io import load_preds  # noqa: E402
PREDS = RESULTS / "depmap_seed1_preds"
STRING_DIR = HERE.parent / "data" / "partners" / "string"
GS = HERE.parent / "data" / "partners"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
METHODS = ["marginal", "residualized", "within_tissue", "irm",
           "dann", "adae"]


def bh_intersection():
    csv = RESULTS / "depmap_within_tissue_pvals.csv"
    if not csv.exists():
        print(f"[bh_intersection] {csv} missing - INTER set empty")
        return set()
    pv = pd.read_csv(csv)
    med = pv.groupby(["target", "method"])["p_one_sided"].median().reset_index()
    keep = {}
    for m in ["marginal", "residualized"]:
        sub = med[med.method == m].dropna(subset=["p_one_sided"])
        rej, _, _, _ = multipletests(sub["p_one_sided"], alpha=0.05, method="fdr_bh")
        keep[m] = set(sub.target[rej])
    return keep["marginal"] & keep["residualized"]


def load_gmt(path):
    """gene-symbol -> set of set-names, and set-name -> members."""
    gene2sets, set2genes = {}, {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            name = parts[0]
            genes = [g for g in parts[2:] if g]
            if len(genes) < 2:
                continue
            set2genes[name] = set(genes)
            for g in genes:
                gene2sets.setdefault(g, set()).add(name)
    return gene2sets, set2genes


def curated_partners(target, gene2sets, set2genes, panel):
    sets = gene2sets.get(target, set())
    partners = set()
    for s in sets:
        partners |= set2genes[s]
    partners.discard(target)
    return partners & panel


def load_chronos_and_tissue():
    crispr = pd.read_csv(D.DATA_ROOT / "DepMap" / "CRISPRGeneEffect.csv", index_col=0)
    crispr.columns = [c.split(" (")[0] for c in crispr.columns]
    model = pd.read_csv(D.DATA_ROOT / "DepMap" / "Model.csv")
    model = model[["ModelID", "OncotreeLineage"]].dropna()
    tissue = model.set_index("ModelID")["OncotreeLineage"]
    common = sorted(set(crispr.index) & set(tissue.index))
    return crispr.loc[common], tissue.loc[common]


def tissue_residualize(mat, tissue):
    df = mat.copy(); df["_t"] = tissue.loc[mat.index].values
    return mat - df.groupby("_t").transform("mean")


def string_graph():
    import networkx as nx
    id2sym = {}
    with gzip.open(STRING_DIR / "info.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            sid, sym = line.split("\t", 2)[:2]; id2sym[sid] = sym
    G = nx.Graph()
    with gzip.open(STRING_DIR / "links.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            a, b, s = line.split()
            if int(s) >= 700:
                sa, sb = id2sym.get(a), id2sym.get(b)
                if sa and sb and sa != sb:
                    G.add_edge(sa, sb)
    return G


def coess_abs_r(mat, target, panel_genes, min_overlap=20):
    """|Pearson r| of each panel gene's profile vs the target's, using
    PAIRWISE-COMPLETE observations per gene (cells where BOTH are finite).
    Vectorized; matches scipy.stats.pearsonr."""
    t = mat[target].values.astype(np.float64)
    M = mat[panel_genes].values.astype(np.float64)            # cells x genes
    mask = np.isfinite(M) & np.isfinite(t)[:, None]           # cells x genes
    n = mask.sum(0).astype(np.float64)
    Mz = np.where(mask, M, 0.0)
    tz = np.where(mask, t[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        st, sg = tz.sum(0), Mz.sum(0)
        stt = (tz * tz).sum(0)
        sgg = (Mz * Mz).sum(0)
        stg = (tz * Mz).sum(0)
        num = stg - st * sg / n
        den = np.sqrt((stt - st * st / n) * (sgg - sg * sg / n))
        r = num / den
    out = pd.Series(np.abs(r), index=panel_genes)
    return out[np.isfinite(out) & (n >= min_overlap)]


def auroc(imp, gene_cols, target, partner_set):
    keep = gene_cols != target
    labels = np.array([g in partner_set for g in gene_cols])[keep]
    scores = imp[keep]
    ok = np.isfinite(scores)
    if labels[ok].sum() < 3 or labels[ok].sum() == ok.sum():
        return np.nan
    return roc_auc_score(labels[ok], scores[ok])

