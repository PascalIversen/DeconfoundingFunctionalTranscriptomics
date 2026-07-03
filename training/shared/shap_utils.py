"""Wrapper around shap.GradientExplainer for SmallMLP models, plus a
within-tissue variant that uses training cells of the matching tissue as
the SHAP background.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import torch

logger = logging.getLogger(__name__)


def _stable_seed(base: int, key: str) -> int:
    """Deterministic 32-bit seed derived from (base, key)."""
    h = hashlib.sha1(f"{base}|{key}".encode()).hexdigest()
    return int(h[:8], 16)


def gradient_shap(model: torch.nn.Module,
                  X: np.ndarray,
                  background: np.ndarray,
                  feature_names: List[str],
                  device: Optional[torch.device] = None,
                  max_background: int = 100,
                  seed: int = 0) -> pd.DataFrame:
    """SHAP values via GradientExplainer. Background is subsampled to
    `max_background` rows."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    if len(background) > max_background:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(background), max_background, replace=False)
        background = background[idx]
    bg_t = torch.from_numpy(background.astype(np.float32)).to(device)
    expl = shap.GradientExplainer(model, bg_t)
    sh = expl.shap_values(torch.from_numpy(X.astype(np.float32)).to(device))
    if isinstance(sh, list):
        sh = sh[0]
    sh = np.asarray(sh).reshape(len(X), -1)
    return pd.DataFrame(sh, columns=feature_names)


def within_tissue_shap(model: torch.nn.Module,
                       X_explain: np.ndarray,
                       T_explain: np.ndarray,
                       *,
                       X_bg: np.ndarray,
                       T_bg: np.ndarray,
                       feature_names: List[str],
                       max_background: int = 100,
                       min_tissue_size: int = 5,
                       device: Optional[torch.device] = None,
                       seed: int = 0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """For each tissue, explain `X_explain[i]` using same-tissue cells
    from `(X_bg, T_bg)` as the background.

    Train-data is the proper background source: it's what the model saw
    during fitting, and it keeps the held-out test cells out of their own
    backgrounds.

    Returns
    -------
    shap_values : np.ndarray of shape (n_explain, n_features)
        Per-cell SHAP values. Rows for cells whose tissue has fewer than
        `min_tissue_size` background cells are filled with NaN.
    has_value : np.ndarray of bool, shape (n_explain,)
        True where SHAP was computed.
    """
    if device is None:
        device = next(model.parameters()).device
    n, p = X_explain.shape
    out = np.full((n, p), np.nan, dtype=np.float32)
    has_value = np.zeros(n, dtype=bool)

    for t in np.unique(T_explain):
        mask_exp = T_explain == t
        mask_bg = T_bg == t
        n_bg = int(mask_bg.sum())
        if n_bg < min_tissue_size:
            continue
        bg_pool = X_bg[mask_bg]
        # Sample up to max_background cells from the train-fold pool.
        if n_bg > max_background:
            rng = np.random.RandomState(_stable_seed(seed, str(t)))
            sel = rng.choice(n_bg, max_background, replace=False)
            bg = bg_pool[sel]
        else:
            bg = bg_pool
        sh = gradient_shap(model, X_explain[mask_exp], bg, feature_names,
                            device=device, max_background=max_background,
                            seed=_stable_seed(seed, str(t)))
        out[mask_exp] = sh.values
        has_value[mask_exp] = True
    return out, has_value
