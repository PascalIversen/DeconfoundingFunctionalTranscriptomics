"""Training loops for the marginal, residualized, and IRM models.

Early stopping uses an explicit validation set passed by the caller (the
orchestrator carves it from the training fold — see orchestrator.py). The
held-out CV test fold is never seen at training time.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .models import SmallMLP

logger = logging.getLogger(__name__)


def _train_loop(model: nn.Module,
                X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray,
                *, epochs: int, batch_size: int, lr: float,
                weight_decay: float, patience: int,
                device: torch.device,
                irm_envs: Optional[np.ndarray] = None,
                irm_lambda: float = 0.0,
                irm_min_env_size: int = 2):
    """Standard SGD with early stopping on `(X_va, y_va)`.

    If `irm_envs` is provided and `irm_lambda > 0`, every gradient step
    adds the IRM penalty computed on the *full* training fold (not on the
    mini-batch). This avoids the noisy per-batch IRM signal that occurs
    when there are many environments and small batches.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)
    mse = nn.MSELoss()

    Xtr = torch.from_numpy(X_tr).float()
    ytr = torch.from_numpy(y_tr).float()
    Xva = torch.from_numpy(X_va).float().to(device)
    yva = torch.from_numpy(y_va).float().to(device)

    use_irm = irm_envs is not None and irm_lambda > 0

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = mse(model(xb).squeeze(-1), yb)
            if use_irm:
                envs_b = irm_envs[idx.cpu().numpy()]
                loss = loss + irm_lambda * _irm_penalty_batch(
                    model, xb, yb, envs_b, device, irm_min_env_size)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v = mse(model(Xva).squeeze(-1), yva).item()
        sched.step(v)
        if v < best - 1e-4:
            best = v
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def _irm_penalty_batch(model: nn.Module,
                       x: torch.Tensor, y: torch.Tensor,
                       envs: np.ndarray, device,
                       min_env_size: int) -> torch.Tensor:
    """IRMv1 penalty on the current mini-batch (the standard form from
    Arjovsky et al. 2019):
        Σ_e ||∇_w E_e[(w·f(x) − y)²]||²   at w = 1.

    Environments with fewer than `min_env_size` samples in the batch are
    skipped (their per-env gradient estimate is too noisy).
    """
    w = torch.tensor(1.0, requires_grad=True, device=device)
    yhat = model(x).squeeze(-1) * w
    pen = torch.tensor(0.0, device=device)
    for e in np.unique(envs):
        m = envs == e
        if m.sum() < min_env_size:
            continue
        mask = torch.as_tensor(m, dtype=torch.bool, device=device)
        loss_e = ((yhat[mask] - y[mask]) ** 2).mean()
        g = torch.autograd.grad(loss_e, w, create_graph=True)[0]
        pen = pen + g ** 2
    return pen


# --------------------------------------------------------- public API
def train_marginal(X_tr, y_tr, X_va, y_va, *,
                   epochs=80, batch_size=64, lr=1e-3, weight_decay=1e-4,
                   dropout=0.3, patience=15, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallMLP(X_tr.shape[1], dropout=dropout).to(device)
    return _train_loop(model, X_tr.astype(np.float32), y_tr.astype(np.float32),
                       X_va.astype(np.float32), y_va.astype(np.float32),
                       epochs=epochs, batch_size=batch_size, lr=lr,
                       weight_decay=weight_decay, patience=patience,
                       device=device)


def train_residualized(X_tr_res, y_tr_res, X_va_res, y_va_res, **kwargs):
    """Same model and training loop as `train_marginal` — the only
    difference is the FWL-residualized inputs."""
    return train_marginal(X_tr_res, y_tr_res, X_va_res, y_va_res, **kwargs)


def train_irm(X_tr, y_tr, T_tr, X_va, y_va, *,
              epochs=80, batch_size=64, lr=1e-3, weight_decay=1e-4,
              dropout=0.3, patience=15, irm_lambda=1.0, warmstart=10,
              device=None):
    """IRM training. `warmstart` epochs of pure ERM first, then ERM + IRM
    penalty. Environments = unique values in `T_tr`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallMLP(X_tr.shape[1], dropout=dropout).to(device)
    if warmstart > 0:
        model, _ = _train_loop(
            model, X_tr.astype(np.float32), y_tr.astype(np.float32),
            X_va.astype(np.float32), y_va.astype(np.float32),
            epochs=warmstart, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=warmstart,
            device=device)
    return _train_loop(
        model, X_tr.astype(np.float32), y_tr.astype(np.float32),
        X_va.astype(np.float32), y_va.astype(np.float32),
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, patience=patience,
        device=device, irm_envs=T_tr, irm_lambda=irm_lambda)
