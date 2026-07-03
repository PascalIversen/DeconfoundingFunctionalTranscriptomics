"""Evaluation metrics:
  - target_rank: rank of a known causal gene in the importance vector
  - contamination_at_k: fraction of top-K with high η²(X)
  - r2_on_raw_y: out-of-sample R² on raw y, using the
                 residualized model + tissue-mean baseline if applicable
  - tissue_eta_squared: η²(X) per gene
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ------------------------------------------------------------ tissue eta²
def tissue_eta_squared(X: pd.DataFrame, T: np.ndarray) -> pd.Series:
    """For each column (gene), η² = SS_between / SS_total."""
    out = {}
    for g in X.columns:
        y = X[g].values
        grand = y.mean()
        ss_total = float(np.sum((y - grand) ** 2))
        if ss_total < 1e-12:
            out[g] = 0.0
            continue
        ss_between = 0.0
        for t in np.unique(T):
            mask = T == t
            if mask.any():
                ss_between += float(mask.sum()) * (y[mask].mean() - grand) ** 2
        out[g] = ss_between / ss_total
    return pd.Series(out)


# --------------------------------------------------------- ranking metrics
def target_rank(importance: pd.Series, target_gene: str) -> int:
    """Rank of `target_gene` (1=highest |importance|) or NaN if not present."""
    if target_gene not in importance.index:
        return np.nan
    rank = importance.rank(ascending=False, method="min").astype(int)
    return int(rank.loc[target_gene])


def contamination_at_k(importance: pd.Series,
                       eta_sq: pd.Series,
                       k: int = 50,
                       eta_threshold: float = 0.30) -> float:
    """Fraction of top-K |importance| genes that have η²(X) above threshold."""
    top = set(importance.nlargest(k).index)
    high_eta = set(eta_sq[eta_sq > eta_threshold].index)
    if not top:
        return np.nan
    return len(top & high_eta) / len(top)


def hits_at_k(importance: pd.Series, target_genes: List[str], k: int) -> int:
    """How many of `target_genes` appear in top-K of importance."""
    top = set(importance.nlargest(k).index)
    return int(sum(g in top for g in target_genes))


# -------------------------------------------------------- predictive metrics
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r between y_true and y_pred. Returns NaN if either has
    zero variance (constant predictor)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.std() < 1e-12 or yp.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def reconstruct_raw_yhat(yhat_residual: np.ndarray,
                         tissue_mean_y: np.ndarray) -> np.ndarray:
    """Add the tissue-mean baseline back to a residualized prediction."""
    return yhat_residual + tissue_mean_y


