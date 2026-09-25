"""Cross-validated runner for the deconfounding methods.

Per outer KFold split:
  1. Take training fold `tr_full` and held-out test fold `te`.
  2. Carve a deterministic validation slice `val` from `tr_full`; the
     residual `tr` is what each model is fit on. Early stopping uses `val`.
  3. Fit a `StandardScaler` on `tr_full` (no test info) — used to
     standardise both `tr_full` and `te`.
  4. For the residualized method, compute per-tissue means on `tr_full`,
     residualize both `tr_full` and `te`, then split into `tr` / `val`
     for training.
  5. Train marginal / IRM / residualized / DANN / AD-AE models on `tr`,
     early-stop on `val`, predict on `te`.
  6. SHAP attributions are computed on `te`, with:
       - marginal / residualized / IRM / DANN / AD-AE:
                        background = `tr_full` cells
       - within-tissue: background = `tr_full` cells of the matching tissue
  7. R² is always reported on raw `y`. For the residualized model, the
     prediction is `f(X_te_res) + train_tissue_mean_y[T_te]`.

All methods share the same outer folds, so paired comparisons across
(seed × target/drug) are like-for-like. The fold structure depends only on
`(n, seed)`, so a run restricted to a subset of methods lands on exactly
the same splits as a full run — which is what lets the adversarial
baselines be added to already-published predictions without retraining
anything else.

`methods=` selects which methods to compute. It defaults to the four in
the original paper plus the between-tissue reference, so existing callers
are unaffected.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import adae_cache
from .deconfound import fwl_residualize
from .eval import pearson_r, r2_score, reconstruct_raw_yhat
from .models import ADAE
from .shap_utils import gradient_shap, within_tissue_shap
from .train import (fit_adae_embedding, train_adae_head, train_dann,
                    train_irm, train_marginal, train_residualized)

logger = logging.getLogger(__name__)

METHODS = ["marginal", "residualized", "irm", "within_tissue"]
ADVERSARIAL_METHODS = ["dann", "adae"]
ALL_METHODS = METHODS + ADVERSARIAL_METHODS
DEFAULT_METHODS = METHODS + ["between_tissue", "baseline"]


def _method_seed(seed: int, fold: int, method: str) -> int:
    """Deterministic torch seed for one (seed, fold, method) cell.

    The original four methods share one global RNG stream, so each one's
    trajectory depends on how much randomness the methods before it in the
    fold consumed. That is harmless when the whole set always runs together,
    but it would make the new baselines irreproducible: `methods=["dann"]`
    and `methods=["dann","adae"]` would give different DANN models from fold
    1 onward, and a lambda sweep would depend on the order of its grid.

    Seeding per method removes that coupling for the adversarial baselines.

    NOTE: this reseeds the *global* torch stream mid-fold, so whenever `dann`
    or `adae` appear in `methods` the original four methods diverge from fold
    1 onward. No published artefact is affected -- `gen_preds_v2.py` never
    requests the adversarial methods, and the sweep recomputes its references
    in a separate `run_cv` call -- but running all six methods in one call
    does NOT reproduce the published marginal/residualized/IRM predictions.
    Request the original four alone to reproduce those.
    """
    h = hashlib.sha1(f"{seed}|{fold}|{method}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


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
           device: Optional[torch.device] = None,
           methods: Optional[Sequence[str]] = None,
           sample_ids: Optional[Sequence[str]] = None,
           dann_kwargs: Optional[Dict] = None,
           adae_kwargs: Optional[Dict] = None,
           adae_cache_dir: Optional[str] = None,
           compute_shap: bool = True,
           ) -> Dict:
    """See module docstring. Returns a dict with one entry per method.

    Parameters specific to the adversarial baselines
    ------------------------------------------------
    methods
        Subset of `ALL_METHODS` (+ "between_tissue", "baseline") to run.
        Defaults to the paper's original set.
    sample_ids
        Per-cell identifiers, in the same order as `X`. Only used to key the
        AD-AE encoder cache; without it every item refits its own encoder.
    dann_kwargs, adae_kwargs
        Passed through to `train_dann` / `fit_adae_embedding` (e.g.
        `{"lambda_max": 10.0}`, `{"lambda_adv": 10.0}`) for the penalty
        sweep.
    adae_cache_dir
        Directory for the on-disk AD-AE encoder cache, shared across pool
        workers and cluster nodes.
    compute_shap
        When False, skip all SHAP passes and return predictions only. Used
        for the multi-seed performance tables, which need `yhat` but not
        per-cell attributions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    methods = list(DEFAULT_METHODS if methods is None else methods)
    unknown = set(methods) - set(ALL_METHODS) - {"between_tissue", "baseline"}
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    dann_kwargs = dict(dann_kwargs or {})
    adae_kwargs = dict(adae_kwargs or {})
    # Pulled out before the fold loop: it configures the phenotype head, not
    # the (cached) encoder, so it must not be consumed on the first fold.
    adae_head_hidden = int(adae_kwargs.pop("head_hidden", 0))

    want_within = "within_tissue" in methods
    want_marginal = "marginal" in methods
    # within-tissue SHAP explains the marginal model, so it needs it fitted
    need_marginal = want_marginal or want_within
    want_res = "residualized" in methods
    want_irm = "irm" in methods
    want_dann = "dann" in methods
    want_adae = "adae" in methods
    want_between = "between_tissue" in methods
    want_baseline = "baseline" in methods or want_res

    n, p = X.shape
    yhat_marg = np.zeros(n, dtype=np.float32)
    yhat_res = np.zeros(n, dtype=np.float32)
    yhat_irm = np.zeros(n, dtype=np.float32)
    yhat_dann = np.zeros(n, dtype=np.float32)
    yhat_adae = np.zeros(n, dtype=np.float32)
    yhat_baseline = np.zeros(n, dtype=np.float32)
    yhat_between = np.zeros(n, dtype=np.float32)
    shap_marg = np.full((n, p), np.nan, dtype=np.float32)
    shap_res = np.full((n, p), np.nan, dtype=np.float32)
    shap_irm = np.full((n, p), np.nan, dtype=np.float32)
    shap_within = np.full((n, p), np.nan, dtype=np.float32)
    shap_between = np.full((n, p), np.nan, dtype=np.float32)
    shap_dann = np.full((n, p), np.nan, dtype=np.float32)
    shap_adae = np.full((n, p), np.nan, dtype=np.float32)
    val_marg, val_res, val_irm, val_dann, val_adae = [], [], [], [], []
    diag_dann, diag_adae = [], []

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
        Tval = Ttr_full[val_local]

        # Shared SHAP background: a fixed subsample of the training fold.
        bg = _shap_bg(Xtr_full_s, shap_background, seed=seed + fold_i)

        # 1) Marginal model
        if need_marginal:
            m_marg, vm = train_marginal(
                Xtr_s, ytr, Xval_s, yval,
                epochs=epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, dropout=dropout, patience=patience,
                device=device)
            val_marg.append(vm)
            yhat_marg[te] = _predict(m_marg, Xte_s, device)
            if compute_shap and want_marginal:
                shap_marg[te] = gradient_shap(
                    m_marg, Xte_s, bg, feature_names,
                    device=device, max_background=shap_background,
                    seed=seed + fold_i).values

            # Within-tissue SHAP: reuse marginal model, but build the
            # background from training cells of each tissue.
            if compute_shap and want_within:
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
        if want_baseline or want_res:
            (Xtr_full_res, ytr_full_res, Xte_res, yte_res,
             base_yhat, _) = fwl_residualize(
                Xtr_full_s, ytr_full, Ttr_full, Xte_s, Tte, yte)
            yhat_baseline[te] = base_yhat
        if want_res:
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
                if compute_shap:
                    bg_res = _shap_bg(Xtr_full_res, shap_background,
                                      seed=seed + fold_i + 7)
                    shap_res[te] = gradient_shap(
                        m_res, Xte_res, bg_res, feature_names,
                        device=device, max_background=shap_background,
                        seed=seed + fold_i + 7).values
            else:
                yhat_res[te] = base_yhat

        # 3) IRM
        if want_irm:
            m_irm, vi = train_irm(
                Xtr_s, ytr, Ttr, Xval_s, yval,
                epochs=epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, dropout=dropout, patience=patience,
                irm_lambda=irm_lambda, device=device)
            val_irm.append(vi)
            yhat_irm[te] = _predict(m_irm, Xte_s, device)
            if compute_shap:
                shap_irm[te] = gradient_shap(
                    m_irm, Xte_s, bg, feature_names,
                    device=device, max_background=shap_background,
                    seed=seed + fold_i + 13).values

        # 4) DANN — domain-adversarial training, tissue as the domain label
        if want_dann:
            torch.manual_seed(_method_seed(seed, fold_i, "dann"))
            m_dann, vd, dg = train_dann(
                Xtr_s, ytr, Ttr, Xval_s, yval, Tval,
                epochs=epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, dropout=dropout, patience=patience,
                device=device, **dann_kwargs)
            val_dann.append(vd)
            dg["fold"] = fold_i
            diag_dann.append(dg)
            yhat_dann[te] = _predict(m_dann, Xte_s, device)
            if compute_shap:
                shap_dann[te] = gradient_shap(
                    m_dann, Xte_s, bg, feature_names,
                    device=device, max_background=shap_background,
                    seed=seed + fold_i + 23).values

        # 5) AD-AE — adversarial deconfounding autoencoder. The encoder is
        #    phenotype-free, so it is cached across items sharing this fold.
        if want_adae:
            torch.manual_seed(_method_seed(seed, fold_i, "adae"))
            ae_cfg = dict(lambda_adv=1.0, hidden=500, emb_dim=100,
                          ae_dropout=0.1, lr=lr, batch_size=128,
                          ae_pretrain_epochs=100, adv_pretrain_epochs=50,
                          joint_rounds=100, patience=patience)
            ae_cfg.update({k: v for k, v in adae_kwargs.items()
                           if k in ae_cfg})

            def _fit():
                return fit_adae_embedding(
                    Xtr_s, Ttr, Xval_s, Tval, device=device, **ae_cfg)

            if sample_ids is not None:
                key = adae_cache.make_key(
                    [sample_ids[i] for i in tr_full], seed=seed, fold=fold_i,
                    lambda_adv=ae_cfg["lambda_adv"], hidden=ae_cfg["hidden"],
                    emb_dim=ae_cfg["emb_dim"], ae_dropout=ae_cfg["ae_dropout"],
                    batch_size=ae_cfg["batch_size"], lr=ae_cfg["lr"],
                    ae_pretrain_epochs=ae_cfg["ae_pretrain_epochs"],
                    adv_pretrain_epochs=ae_cfg["adv_pretrain_epochs"],
                    joint_rounds=ae_cfg["joint_rounds"],
                    patience=ae_cfg["patience"], in_dim=p,
                    val_frac=val_frac, panel=feature_names)
                state, ag = adae_cache.get_or_fit(
                    key, _fit, cache_dir=adae_cache_dir, device=device)
                adae = ADAE(p, ag["n_classes"], hidden=ae_cfg["hidden"],
                            emb_dim=ae_cfg["emb_dim"],
                            dropout=ae_cfg["ae_dropout"]).to(device)
                adae.load_state_dict(state)
            else:
                adae, ag = _fit()

            m_adae, va, hd = train_adae_head(
                adae, Xtr_s, ytr, Xval_s, yval, head_hidden=adae_head_hidden,
                epochs=epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, patience=patience, device=device)
            val_adae.append(va)
            ag = dict(ag); ag.update(hd); ag["fold"] = fold_i
            diag_adae.append(ag)
            yhat_adae[te] = _predict(m_adae, Xte_s, device)
            # A degenerate (intercept-only) head makes the explained function
            # constant, so GradientExplainer returns exactly zero for every
            # cell in this fold. Leaving those rows NaN keeps them out of the
            # attribution statistics, matching the convention within-tissue
            # SHAP already uses for tissues it cannot explain.
            if compute_shap and hd.get("head_degenerate", 0.0) >= 1.0:
                logger.warning("AD-AE head degenerate on fold %d; "
                               "leaving attributions undefined", fold_i)
            elif compute_shap:
                shap_adae[te] = gradient_shap(
                    m_adae, Xte_s, bg, feature_names,
                    device=device, max_background=shap_background,
                    seed=seed + fold_i + 31).values

        # 6) Between-tissue model f_b (Ridge on tissue-mean expression →
        #    tissue-mean y). Used together with the residualized NN to
        #    decompose attribution into tissue-mediated vs within-tissue.
        if want_between:
            unique_t = np.unique(Ttr_full)
            Xb_unique = np.stack([Xtr_full_s[Ttr_full == t].mean(axis=0)
                                  for t in unique_t]).astype(np.float32)
            yb_unique = np.array([ytr_full[Ttr_full == t].mean()
                                  for t in unique_t], dtype=np.float32)
            ridge_b = Ridge(alpha=1.0, fit_intercept=True).fit(Xb_unique,
                                                                yb_unique)
            # Per-cell X_between on test: the train-fold mean of its tissue,
            # falling back to the global train-fold mean for unseen tissues.
            global_mean_b = Xb_unique.mean(axis=0)
            tissue_to_mean = {t: Xb_unique[i] for i, t in enumerate(unique_t)}
            Xb_te = np.stack([tissue_to_mean.get(t, global_mean_b)
                              for t in Tte]).astype(np.float32)
            yhat_between[te] = ridge_b.predict(Xb_te)
            # Analytic SHAP for Ridge with intercept:
            #   φ_g(c) = β_g (x_g(c) − x̄_g_bg).
            # Background = mean of per-tissue means in train (unique rows).
            shap_between[te] = ((Xb_te - global_mean_b[None, :])
                                * ridge_b.coef_[None, :]).astype(np.float32)

    def imp(sh: np.ndarray) -> pd.Series:
        # nanmean: rows with no SHAP (within-tissue dropouts) don't drag
        # the importance to zero.
        with np.errstate(invalid="ignore"):
            return pd.Series(np.nanmean(np.abs(sh), axis=0),
                              index=feature_names)

    def entry(sh, yh, val_list, extra=None):
        out = {
            "importance": imp(sh),
            "yhat_raw": yh,
            "r2_raw": r2_score(y, yh),
            "pearson_r": pearson_r(y, yh),
            "val_mse": float(np.mean(val_list)) if val_list else float("nan"),
            "shap_full": sh,
        }
        if extra:
            out.update(extra)
        return out

    res: Dict = {
        "n_cells": n, "n_features": p,
        "n_tissues": int(len(np.unique(T))),
        "y_var": float(np.var(y)),
        "y_true_full": y,
        "tissue_full": T,
        "methods_run": methods,
    }
    if want_marginal:
        res["marginal"] = entry(shap_marg, yhat_marg, val_marg)
    if want_within:
        res["within_tissue"] = entry(shap_within, yhat_marg, val_marg)
    if want_res:
        res["residualized"] = entry(shap_res, yhat_res, val_res)
    if want_irm:
        res["irm"] = entry(shap_irm, yhat_irm, val_irm)
    if want_dann:
        res["dann"] = entry(shap_dann, yhat_dann, val_dann,
                            {"diagnostics": diag_dann})
    if want_adae:
        res["adae"] = entry(shap_adae, yhat_adae, val_adae,
                            {"diagnostics": diag_adae})
    if want_baseline:
        res["baseline"] = {
            "yhat_raw": yhat_baseline,
            "r2_raw": r2_score(y, yhat_baseline),
            "pearson_r": pearson_r(y, yhat_baseline),
        }
    if want_between:
        res["between_tissue"] = entry(shap_between, yhat_between, [])
    return res
