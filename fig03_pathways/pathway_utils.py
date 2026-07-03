"""Shared helpers for pathway enrichment analysis on saved SHAP matrices.

Conventions:
- A "gene set" is a list[str] of HUGO symbols.
- A "library" is dict[name -> set[str]] loaded from an Enrichr-formatted GMT.
- Top-K gene picks are returned as list[str] in descending order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_gmt(path: Path) -> Dict[str, Set[str]]:
    """Enrichr-flavoured GMT: <name>\\t\\t<gene1>\\t<gene2>\\t... Some libs use
    <name>\\t<desc>\\t<gene1>\\t..., so we tolerate both."""
    sets: Dict[str, Set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = [p for p in line.rstrip("\n").split("\t") if p]
            if len(parts) < 2:
                continue
            name = parts[0]
            # Enrichr style: parts[1] is a description if it contains spaces or
            # known prefixes; otherwise the first non-empty entry after name is
            # already a gene. We take everything except the first 1-2 columns
            # and keep entries that look like gene symbols.
            candidates = parts[1:]
            # Drop empty descriptor used by Enrichr ("" between name and genes)
            genes = [g.split(",")[0].upper() for g in candidates
                     if g and not g.startswith("http")]
            # Enrichr Hallmark format puts a blank desc; here genes is fine.
            sets[name] = set(g for g in genes if g.isalnum() or "-" in g or "_" in g)
    return sets


def top_k_genes(importance: np.ndarray, gene_cols: np.ndarray,
                k: int) -> List[str]:
    """Top-K gene symbols by |importance| (importance already absolute or signed)."""
    idx = np.argsort(-np.abs(importance))[:k]
    return [str(gene_cols[i]) for i in idx]


def hypergeometric_pvalue(top_k_set: Set[str], pathway_set: Set[str],
                          universe: Set[str]) -> Tuple[float, int]:
    """One-sided hypergeometric p-value for over-representation.

    Returns (p, overlap_size). Universe = all genes that could be ranked.
    """
    # Intersect everything with universe so the test is well-defined.
    top_in_u = top_k_set & universe
    path_in_u = pathway_set & universe
    overlap = len(top_in_u & path_in_u)
    M = len(universe)
    n = len(path_in_u)
    N = len(top_in_u)
    if M == 0 or n == 0 or N == 0:
        return float("nan"), overlap
    # P(X >= overlap) where X ~ hypergeom(M, n, N)
    p = stats.hypergeom.sf(overlap - 1, M, n, N)
    return float(p), overlap


def importance_from_npz(npz_path: Path,
                        methods: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (gene_cols, {method: mean_abs_shap_per_gene})."""
    d = np.load(npz_path)
    gene_cols = d["gene_cols"]
    out = {}
    for m in methods:
        key = f"shap_{m}"
        if key not in d.files:
            continue
        sh = d[key].astype(np.float32)
        with np.errstate(invalid="ignore"):
            out[m] = np.nanmean(np.abs(sh), axis=0)
    return gene_cols, out


def collect_per_method_importance(preds_dir: Path,
                                  methods: List[str]) -> pd.DataFrame:
    """Build a long-form DataFrame (target/drug × method × gene → importance)."""
    rows = []
    files = sorted(preds_dir.glob("*.npz"))
    for f in files:
        name = f.stem
        gene_cols, imps = importance_from_npz(f, methods)
        for m, imp in imps.items():
            for g, v in zip(gene_cols, imp):
                rows.append((name, m, str(g), float(v)))
    return pd.DataFrame(rows, columns=["item", "method", "gene", "importance"])
