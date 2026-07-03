"""Tissue-residualization (FWL) and within-tissue SHAP background helpers."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def per_group_means(X: np.ndarray, T: np.ndarray
                    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return ({tissue: mean_vector}, global_mean)."""
    groups = {t: X[T == t].mean(axis=0) for t in np.unique(T)}
    return groups, X.mean(axis=0)


def residualize_with_means(X: np.ndarray, T: np.ndarray,
                           group_means: Dict[str, np.ndarray],
                           global_mean: np.ndarray) -> np.ndarray:
    """Subtract per-tissue mean from each row.

    Falls back to global mean for unseen tissues — important for proper
    out-of-sample evaluation when train and test see different tissues.
    """
    out = np.empty_like(X)
    for i, t in enumerate(T):
        out[i] = X[i] - group_means.get(t, global_mean)
    return out


def fwl_residualize(X_train: np.ndarray, y_train: np.ndarray, T_train: np.ndarray,
                    X_test: np.ndarray, T_test: np.ndarray, y_test: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
    """Frisch-Waugh-Lovell residualization with no test leakage.

    Returns: X_train_res, y_train_res, X_test_res, y_test_res,
             tissue_mean_y_test (the predictor baseline on raw y),
             tissue_mean_X_test (used in some downstream code).
    """
    X_means, X_global = per_group_means(X_train, T_train)
    y_means_dict = {t: float(y_train[T_train == t].mean())
                    for t in np.unique(T_train)}
    y_global = float(y_train.mean())

    X_train_res = residualize_with_means(X_train, T_train, X_means, X_global)
    X_test_res = residualize_with_means(X_test, T_test, X_means, X_global)
    y_train_res = np.array([y_train[i] - y_means_dict[T_train[i]]
                            for i in range(len(y_train))], dtype=np.float32)
    yhat_baseline = np.array([y_means_dict.get(T_test[i], y_global)
                              for i in range(len(T_test))], dtype=np.float32)
    y_test_res = (y_test - yhat_baseline).astype(np.float32)
    return (X_train_res.astype(np.float32),
            y_train_res,
            X_test_res.astype(np.float32),
            y_test_res,
            yhat_baseline,
            X_global.astype(np.float32))
