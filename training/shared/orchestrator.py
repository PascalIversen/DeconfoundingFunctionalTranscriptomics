"""Cross-validated runner for the four deconfounding methods.

Per outer KFold split:
  1. Take training fold `tr_full` and held-out test fold `te`.
  2. Carve a deterministic validation slice `val` from `tr_full`; the
     residual `tr` is what each model is fit on. Early stopping uses `val`.
  3. Fit a `StandardScaler` on `tr_full` (no test info) — used to
     standardise both `tr_full` and `te`.
  4. For the residualized method, compute per-tissue means on `tr_full`,
     residualize both `tr_full` and `te`, then split into `tr` / `val`
     for training.
  5. Train marginal / IRM / residualized models on `tr`, early-stop on
     `val`, predict on `te`.
  6. SHAP attributions are computed on `te`, with:
       - marginal / residualized / IRM: background = `tr_full` cells
       - within-tissue: background = `tr_full` cells of the matching tissue
  7. R² is always reported on raw `y`. For the residualized model, the
     prediction is `f(X_te_res) + train_tissue_mean_y[T_te]`.

All four methods share the same outer folds, so paired comparisons
across (seed × target/drug) are like-for-like.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .deconfound import fwl_residualize
from .eval import pearson_r, r2_score, reconstruct_raw_yhat
from .shap_utils import gradient_shap, within_tissue_shap
from .train import train_irm, train_marginal, train_residualized

logger = logging.getLogger(__name__)

METHODS = ["marginal", "residualized", "irm", "within_tissue"]


def _predict(model, X, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (model(torch.from_numpy(X).float().to(device))
                .squeeze(-1).cpu().numpy())


def _shap_bg(X_train: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(X_train) <= n:
        return X_train
    rng = np.random.RandomState(seed)
    return X_train[rng.choice(len(X_train), n, replace=False)]


def _split_train_val(n: int, val_frac: float, seed: int):
    """Return (tr_idx, val_idx) — a deterministic shuffle of [0, n)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(int(round(val_frac * n)), 30)
    n_val = min(n_val, n // 3)   # don't let val swamp tr for tiny n
    return perm[n_val:], perm[:n_val]


def run_cv(X: np.ndarray, y: np.ndarray, T: np.ndarray,
           feature_names: List[str],
           *,
           n_folds: int = 5,
           seed: int = 42,
           val_frac: float = 0.15,
           epochs: int = 80,
           batch_size: int = 64,
           lr: float = 1e-3,
           weight_decay: float = 1e-4,
           dropout: float = 0.3,
           patience: int = 15,
           irm_lambda: float = 1.0,
           shap_background: int = 100,
           min_tissue_size_within: int = 5,
           device: Optional[torch.device] = None
           ) -> Dict:
    """See module docstring. Returns a dict with one entry per method
    plus `baseline` (tissue-mean predictor)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n, p = X.shape
    yhat_marg = np.zeros(n, dtype=np.float32)
    yhat_res = np.zeros(n, dtype=np.float32)
    yhat_irm = np.zeros(n, dtype=np.float32)
    yhat_baseline = np.zeros(n, dtype=np.float32)
    yhat_between = np.zeros(n, dtype=np.float32)
    shap_marg = np.full((n, p), np.nan, dtype=np.float32)
    shap_res = np.full((n, p), np.nan, dtype=np.float32)
    shap_irm = np.full((n, p), np.nan, dtype=np.float32)
    shap_within = np.full((n, p), np.nan, dtype=np.float32)
    shap_between = np.full((n, p), np.nan, dtype=np.float32)
    val_marg, val_res, val_irm = [], [], []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_i, (tr_full, te) in enumerate(kf.split(X)):
        # Standardize on full train fold (no test info leaks).
        sc = StandardScaler().fit(X[tr_full])
        Xtr_full_s = sc.transform(X[tr_full]).astype(np.float32)
        Xte_s = sc.transform(X[te]).astype(np.float32)
        ytr_full = y[tr_full].astype(np.float32)
        yte = y[te].astype(np.float32)
        Ttr_full = T[tr_full]
        Tte = T[te]

        # Carve val from tr_full for early stopping. Deterministic per fold.
        tr_local, val_local = _split_train_val(
            len(tr_full), val_frac, seed=seed * 1000 + fold_i)
        Xtr_s = Xtr_full_s[tr_local]
        Xval_s = Xtr_full_s[val_local]
        ytr = ytr_full[tr_local]
        yval = ytr_full[val_local]
        Ttr = Ttr_full[tr_local]

        # 1) Marginal model
        m_marg, vm = train_marginal(
            Xtr_s, ytr, Xval_s, yval,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, dropout=dropout, patience=patience,
            device=device)
        val_marg.append(vm)
        yhat_marg[te] = _predict(m_marg, Xte_s, device)
        bg = _shap_bg(Xtr_full_s, shap_background, seed=seed + fold_i)
        shap_marg[te] = gradient_shap(
            m_marg, Xte_s, bg, feature_names,
            device=device, max_background=shap_background,
            seed=seed + fold_i).values

        # Within-tissue SHAP: reuse marginal model, but build the
        # background from training cells of each tissue.
        sh_w, has_value = within_tissue_shap(
            m_marg, Xte_s, Tte,
            X_bg=Xtr_full_s, T_bg=Ttr_full,
            feature_names=feature_names,
            max_background=shap_background,
            min_tissue_size=min_tissue_size_within,
            device=device, seed=seed + fold_i)
        rows_with_value = te[has_value]
        shap_within[rows_with_value] = sh_w[has_value]

        # 2) Residualized (FWL on tr_full, then train/val split)
        (Xtr_full_res, ytr_full_res, Xte_res, yte_res,
         base_yhat, _) = fwl_residualize(
            Xtr_full_s, ytr_full, Ttr_full, Xte_s, Tte, yte)
        yhat_baseline[te] = base_yhat
        Xtr_res = Xtr_full_res[tr_local]
        Xval_res = Xtr_full_res[val_local]
        ytr_res = ytr_full_res[tr_local]
        yval_res = ytr_full_res[val_local]
        if float(np.std(ytr_res)) > 1e-6:
            m_res, vr = train_residualized(
                Xtr_res, ytr_res, Xval_res, yval_res,
                epochs=epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, dropout=dropout,
                patience=patience, device=device)
            val_res.append(vr)
            yhat_res[te] = reconstruct_raw_yhat(
                _predict(m_res, Xte_res, device), base_yhat)
            bg_res = _shap_bg(Xtr_full_res, shap_background,
                              seed=seed + fold_i + 7)
            shap_res[te] = gradient_shap(
                m_res, Xte_res, bg_res, feature_names,
                device=device, max_background=shap_background,
                seed=seed + fold_i + 7).values
        else:
            yhat_res[te] = base_yhat

        # 3) IRM
        m_irm, vi = train_irm(
            Xtr_s, ytr, Ttr, Xval_s, yval,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, dropout=dropout, patience=patience,
            irm_lambda=irm_lambda, device=device)
        val_irm.append(vi)
        yhat_irm[te] = _predict(m_irm, Xte_s, device)
        shap_irm[te] = gradient_shap(
            m_irm, Xte_s, bg, feature_names,
            device=device, max_background=shap_background,
            seed=seed + fold_i + 13).values

        # 4) Between-tissue model f_b (Ridge on tissue-mean expression →
        #    tissue-mean y). Used together with the residualized NN to
        #    decompose attribution into tissue-mediated vs within-tissue.
        unique_t = np.unique(Ttr_full)
        Xb_unique = np.stack([Xtr_full_s[Ttr_full == t].mean(axis=0)
                              for t in unique_t]).astype(np.float32)
        yb_unique = np.array([ytr_full[Ttr_full == t].mean()
                              for t in unique_t], dtype=np.float32)
        ridge_b = Ridge(alpha=1.0, fit_intercept=True).fit(Xb_unique, yb_unique)
        # Per-cell X_between on test: the train-fold mean of its tissue,
        # falling back to the global train-fold mean for unseen tissues.
        global_mean_b = Xb_unique.mean(axis=0)
        tissue_to_mean = {t: Xb_unique[i] for i, t in enumerate(unique_t)}
        Xb_te = np.stack([tissue_to_mean.get(t, global_mean_b)
                          for t in Tte]).astype(np.float32)
        yhat_between[te] = ridge_b.predict(Xb_te)
        # Analytic SHAP for Ridge with intercept: φ_g(c) = β_g (x_g(c) − x̄_g_bg).
        # Background = mean of per-tissue means in train (the unique rows).
        shap_between[te] = ((Xb_te - global_mean_b[None, :])
                            * ridge_b.coef_[None, :]).astype(np.float32)

    def imp(sh: np.ndarray) -> pd.Series:
        # nanmean: rows with no SHAP (within-tissue dropouts) don't drag
        # the importance to zero.
        with np.errstate(invalid="ignore"):
            return pd.Series(np.nanmean(np.abs(sh), axis=0),
                              index=feature_names)

    return {
        "n_cells": n, "n_features": p,
        "n_tissues": int(len(np.unique(T))),
        "y_var": float(np.var(y)),
        "y_true_full": y,
        "tissue_full": T,
        "marginal": {
            "importance": imp(shap_marg),
            "yhat_raw": yhat_marg,
            "r2_raw": r2_score(y, yhat_marg),
            "pearson_r": pearson_r(y, yhat_marg),
            "val_mse": float(np.mean(val_marg)),
            "shap_full": shap_marg,
        },
        "residualized": {
            "importance": imp(shap_res),
            "yhat_raw": yhat_res,
            "r2_raw": r2_score(y, yhat_res),
            "pearson_r": pearson_r(y, yhat_res),
            "val_mse": float(np.mean(val_res)) if val_res else float("nan"),
            "shap_full": shap_res,
        },
        "irm": {
            "importance": imp(shap_irm),
            "yhat_raw": yhat_irm,
            "r2_raw": r2_score(y, yhat_irm),
            "pearson_r": pearson_r(y, yhat_irm),
            "val_mse": float(np.mean(val_irm)),
            "shap_full": shap_irm,
        },
        "within_tissue": {
            "importance": imp(shap_within),
            "yhat_raw": yhat_marg,
            "r2_raw": r2_score(y, yhat_marg),
            "pearson_r": pearson_r(y, yhat_marg),
            "val_mse": float(np.mean(val_marg)),
            "shap_full": shap_within,
        },
        "baseline": {
            "yhat_raw": yhat_baseline,
            "r2_raw": r2_score(y, yhat_baseline),
            "pearson_r": pearson_r(y, yhat_baseline),
        },
        "between_tissue": {
            "importance": imp(shap_between),
            "yhat_raw": yhat_between,
            "r2_raw": r2_score(y, yhat_between),
            "pearson_r": pearson_r(y, yhat_between),
            "shap_full": shap_between,
        },
    }
