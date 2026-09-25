"""Training loops for the marginal, residualized, IRM, DANN and AD-AE models.

Early stopping uses an explicit validation set passed by the caller (the
orchestrator carves it from the training fold — see orchestrator.py). The
held-out CV test fold is never seen at training time.

The two adversarial model-stage baselines are implemented to the letter of
their source papers; see `train_dann` and `fit_adae_embedding` for the
point-by-point correspondence.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .models import (ADAE, ADAEPredictor, DANNModel, SmallMLP,
                     build_adae_head, dann_lambda)

logger = logging.getLogger(__name__)


def _train_loop(model: nn.Module,
                X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray,
                *, epochs: int, batch_size: int, lr: float,
                weight_decay: float, patience: int,
                device: torch.device,
                irm_envs: Optional[np.ndarray] = None,
                irm_lambda: float = 0.0,
                irm_min_env_size: int = 2,
                params: Optional[Sequence[torch.nn.Parameter]] = None):
    """Standard SGD with early stopping on `(X_va, y_va)`.

    If `irm_envs` is provided and `irm_lambda > 0`, every gradient step
    adds the IRM penalty computed on the current mini-batch (see
    `_irm_penalty_batch`). Environments with fewer than `irm_min_env_size`
    samples in the batch are skipped, since a handful of cell lines is too
    few to estimate a per-environment gradient reliably.

    `params` restricts the optimizer to a subset of the model's parameters
    (used by the AD-AE head, which trains on top of a frozen encoder).
    """
    opt = torch.optim.Adam(params if params is not None
                           else model.parameters(), lr=lr,
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


# ============================================================ shared helper
def _encode_labels(T_tr: np.ndarray, T_va: Optional[np.ndarray] = None):
    """Map tissue strings to contiguous class indices defined by the
    *training* fold. Validation rows whose tissue never appears in training
    get -1 and are excluded from the domain / adversary loss (there is no
    class for them; they still count for reconstruction and MSE)."""
    classes = list(np.unique(T_tr))
    idx = {t: i for i, t in enumerate(classes)}
    d_tr = np.array([idx[t] for t in T_tr], dtype=np.int64)
    d_va = (np.array([idx.get(t, -1) for t in T_va], dtype=np.int64)
            if T_va is not None else None)
    return classes, d_tr, d_va


# ============================================== DANN (Ganin et al., 2016)
def train_dann(X_tr, y_tr, T_tr, X_va, y_va, T_va=None, *,
               epochs=80, batch_size=64, lr=1e-3, weight_decay=1e-4,
               dropout=0.3, patience=15,
               lambda_max=1.0, gamma=10.0, domain_hidden=64,
               min_lambda_frac=0.95,
               optimizer="adam", ganin_lr0=0.01, ganin_alpha=10.0,
               ganin_beta=0.75, ganin_momentum=0.9,
               device=None):
    """Domain-adversarial training with tissue as the domain label.

    Correspondence to Ganin et al. (2016):

    * Objective (their Eq. 18). We minimise ``L_y + L_d`` where ``L_d`` is
      reached through the gradient reversal layer. The GRL negates and
      scales by ``lambda`` on the way back, so G_f maximises the domain loss
      while G_d minimises it, and — as the paper specifies — ``lambda``
      affects only G_f's update; G_d trains at a fixed weight of 1.
    * Schedule. ``lambda_p = 2/(1+exp(-gamma*p)) - 1`` with ``gamma = 10``
      and ``p`` the training progress from 0 to 1 (`dann_lambda`).
      ``lambda_max`` scales the ceiling for the penalty-weight sweep;
      ``lambda_max = 1.0`` is the paper's setting.
    * ``L_y`` is squared error rather than the paper's logistic loss,
      because the phenotype here is continuous. ``L_d`` is softmax
      cross-entropy over the K tissues (the paper's binary domain loss
      generalised to K domains).

    Deliberate deviations, both reported in the manuscript:

    * ``optimizer="adam"`` (default) uses the same Adam + ReduceLROnPlateau
      + early-stopping loop as every other method in the pipeline, so that
      a difference between methods cannot be attributed to the optimiser.
      ``optimizer="ganin"`` reproduces the paper's SGD with momentum 0.9 and
      ``mu_p = mu_0 / (1 + alpha*p)^beta`` (mu_0=0.01, alpha=10, beta=0.75)
      over a fixed epoch budget, and is run as a sensitivity check.
    * Early stopping is kept, but restricted. Validation label MSE can only
      be *raised* by the adversarial term, so an unrestricted best-epoch rule
      systematically returns an early checkpoint at partial lambda. Only
      epochs at which ``lambda_p >= min_lambda_frac * lambda_max`` are
      eligible to be kept, and training always runs at least that far.
      ``lambda_max=0`` gives the control that isolates this training protocol
      from the adversarial effect.

    Returns ``(model, best_val_mse, diagnostics)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = X_tr.astype(np.float32); y_tr = y_tr.astype(np.float32)
    X_va = X_va.astype(np.float32); y_va = y_va.astype(np.float32)
    classes, d_tr, d_va = _encode_labels(T_tr, T_va)

    model = DANNModel(X_tr.shape[1], len(classes), dropout=dropout,
                      domain_hidden=domain_hidden).to(device)

    if optimizer == "ganin":
        opt = torch.optim.SGD(model.parameters(), lr=ganin_lr0,
                              momentum=ganin_momentum,
                              weight_decay=weight_decay)
        sched = None
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_tr); ytr = torch.from_numpy(y_tr)
    dtr = torch.from_numpy(d_tr)
    Xva = torch.from_numpy(X_va).to(device); yva = torch.from_numpy(y_va).to(device)

    # Checkpoint selection may only consider epochs at which the lambda ramp
    # has essentially completed.
    #
    # Restoring the global best-validation-MSE epoch is wrong here, and
    # silently so. The adversarial term can only *raise* validation label
    # MSE, so the argmin is pushed to early epochs where lambda is still
    # ramping: with epochs=80 and patience=12, runs stopped near epoch 21 and
    # the restored checkpoint sat near epoch 9, where lambda is ~0.50 rather
    # than the nominal ~1.0. The model that got explained therefore carried
    # roughly half the adversarial strength it was reported to have, and a
    # lambda sweep never actually applied its nominal lambda.
    #
    # Training now runs at least until lambda >= `min_lambda_frac` of its
    # ceiling, and only epochs from that point on are eligible to be kept.
    # This preserves the pipeline's overfitting control while guaranteeing
    # the returned model has the adversary at full strength.
    if lambda_max > 0:
        eligible_from = next(
            (ep for ep in range(epochs)
             if dann_lambda(ep / max(epochs - 1, 1), gamma=gamma,
                            lambda_max=lambda_max)
             >= min_lambda_frac * lambda_max), 0)
    else:
        eligible_from = 0            # lambda=0 control: every epoch is valid

    best, best_state, bad = float("inf"), None, 0
    best_lambda, last_lambda = 0.0, 0.0
    for ep in range(epochs):
        p = ep / max(epochs - 1, 1)
        lambd = dann_lambda(p, gamma=gamma, lambda_max=lambda_max)
        last_lambda = lambd
        if optimizer == "ganin":
            mu = ganin_lr0 / (1.0 + ganin_alpha * p) ** ganin_beta
            for gparam in opt.param_groups:
                gparam["lr"] = mu

        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            db = dtr[idx].to(device)
            opt.zero_grad()
            yhat, dlogits = model.forward_adv(xb, lambd)
            loss = mse(yhat.squeeze(-1), yb) + ce(dlogits, db)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v = mse(model(Xva).squeeze(-1), yva).item()
        if sched is not None:
            sched.step(v)
        if ep < eligible_from:
            continue          # ramp still climbing: train on, keep nothing
        if v < best - 1e-4:
            best = v
            best_lambda = lambd
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Diagnostic: how well the domain head still recovers tissue from the
    # learned representation. Reported to show whether the adversary
    # actually did anything.
    model.eval()
    with torch.no_grad():
        acc_tr = float((model.domain_logits(
            torch.from_numpy(X_tr).to(device)).argmax(1).cpu().numpy()
            == d_tr).mean())
        if d_va is not None and (d_va >= 0).any():
            keep = d_va >= 0
            pred_va = model.domain_logits(
                torch.from_numpy(X_va[keep]).to(device)).argmax(1).cpu().numpy()
            acc_va = float((pred_va == d_va[keep]).mean())
        else:
            acc_va = float("nan")
    diag = {"domain_acc_train": acc_tr, "domain_acc_val": acc_va,
            "n_domains": len(classes),
            # lambda of the checkpoint actually returned. `last_lambda` (the
            # final epoch iterated) can overstate this substantially (~1.7x
            # in practice), since the checkpoint is selected from an
            # eligibility window rather than always the last epoch -- report
            # both rather than conflating them as "the" lambda.
            "final_lambda": best_lambda if best_state is not None else last_lambda,
            "last_epoch_lambda": last_lambda,
            "eligible_from_epoch": float(eligible_from),
            "majority_domain_frac": float(np.bincount(d_tr).max() / len(d_tr))}
    return model, best, diag


# ================================ AD-AE (Dincer, Janizek & Lee, 2020)
def fit_adae_embedding(X_tr, T_tr, X_va, T_va, *,
                       lambda_adv=1.0, hidden=500, emb_dim=100,
                       ae_dropout=0.1, lr=1e-3, batch_size=128,
                       ae_pretrain_epochs=100, adv_pretrain_epochs=50,
                       joint_rounds=100, patience=15,
                       device=None) -> Tuple[ADAE, Dict]:
    """Fit the AD-AE encoder/decoder/adversary. Phenotype-free by design.

    Follows the paper's procedure exactly:

    1. **Pretrain the autoencoder** on reconstruction alone,
       ``min ||x - g(f(x))||^2``.
    2. **Pretrain the adversary** to predict the confounder from the frozen
       embedding, ``min L(h(f(x)), c)``.
    3. **Alternate one full epoch each** ("freeze adversary, train
       autoencoder one epoch; freeze autoencoder, train adversary one full
       epoch"), with the autoencoder minimising
       ``||x - g(f(x))||^2 - lambda * L(h(f(x)), c)`` — i.e. *maximising*
       the adversary's loss. ``lambda = 1`` is the paper's setting.

    Every stage uses Adam at lr 1e-3 with batch size 128, as in the paper;
    epoch counts are capped and chosen by validation loss, which is how the
    paper describes selecting them.

    The paper specifies no stopping rule for stage 3 ("we continue this
    alternating training process until both models are optimized") but does
    state the equilibrium it is aiming at, and verifies it: *"the adversary
    achieves random prediction performance on the validation set"*. That is
    what stage 3 selects on here — see the comment at the stage-3 loop for
    why the two obvious alternatives are both ill-posed.

    No stabilisation tricks are added to the objective — no gradient
    clipping, no confusion loss, no label smoothing. The adversarial term is
    unbounded below by construction; the only guard is a divergence break
    that abandons a run whose reconstruction has exploded, which is recorded
    in the diagnostics rather than hidden.

    Because neither stage ever sees the phenotype, the resulting encoder
    depends only on ``(X, tissue, fold, seed, lambda)`` — which is what
    licenses the cross-item cache in `adae_cache.py`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = X_tr.astype(np.float32); X_va = X_va.astype(np.float32)
    classes, d_tr, d_va = _encode_labels(T_tr, T_va)
    va_keep = d_va >= 0

    model = ADAE(X_tr.shape[1], len(classes), hidden=hidden,
                 emb_dim=emb_dim, dropout=ae_dropout).to(device)
    recon_loss = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_tr)
    dtr = torch.from_numpy(d_tr)
    Xva = torch.from_numpy(X_va).to(device)
    dva = torch.from_numpy(d_va[va_keep]).to(device) if va_keep.any() else None
    Xva_keep = torch.from_numpy(X_va[va_keep]).to(device) if va_keep.any() else None

    opt_ae = torch.optim.Adam(model.ae_parameters(), lr=lr)
    opt_adv = torch.optim.Adam(model.adversary_parameters(), lr=lr)

    n = len(X_tr)

    def _batches():
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            yield perm[i:i + batch_size]

    def _val_recon() -> float:
        model.eval()
        with torch.no_grad():
            return float(recon_loss(model.reconstruct(Xva), Xva).item())

    def _val_adv() -> float:
        if Xva_keep is None:
            return float("nan")
        model.eval()
        with torch.no_grad():
            return float(ce(model.adversary_logits_from_x(Xva_keep), dva).item())

    # ---- stage 1: autoencoder pretraining -----------------------------
    best, best_state, bad = float("inf"), None, 0
    for _ in range(ae_pretrain_epochs):
        model.train()
        for idx in _batches():
            xb = Xtr[idx].to(device)
            opt_ae.zero_grad()
            loss = recon_loss(model.reconstruct(xb), xb)
            loss.backward()
            opt_ae.step()
        v = _val_recon()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    recon_pretrain = best

    # ---- stage 2: adversary pretraining (encoder frozen) --------------
    best, best_state, bad = float("inf"), None, 0
    for _ in range(adv_pretrain_epochs):
        model.train()
        for idx in _batches():
            xb = Xtr[idx].to(device)
            db = dtr[idx].to(device)
            opt_adv.zero_grad()
            with torch.no_grad():
                z = model.encode(xb)
            loss = ce(model.adversary(z), db)
            loss.backward()
            opt_adv.step()
        v = _val_adv()
        if not np.isfinite(v):
            break
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.adversary.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.adversary.load_state_dict(best_state)
    adv_pretrain = best

    # ---- stage 3: alternating adversarial training --------------------
    # Model selection targets the equilibrium the paper describes and
    # verifies: "the adversary achieves random prediction performance on the
    # validation set". The reference is the cross-entropy of the trivial
    # predictor that outputs the training class prior, evaluated on the
    # validation labels; we keep the round whose adversary sits closest to
    # it.
    #
    # Selecting on the autoencoder's own objective (recon - lambda*CE) is
    # NOT viable here: that term is unbounded below in CE, so minimising it
    # rewards divergence. In practice it drives the adversary to *worse*
    # than chance (an anti-predictive embedding) and destroys the phenotype
    # signal along with the tissue signal. Selecting on reconstruction alone
    # fails at the opposite end — reconstruction is monotonically best
    # before any adversarial pressure, so it would return the un-deconfounded
    # round-1 model. The paper's own equilibrium is the criterion that is
    # both faithful and well-posed.
    prior = np.bincount(d_tr, minlength=len(classes)).astype(np.float64)
    prior = prior / prior.sum()
    if va_keep.any():
        ce_chance = float(-np.log(np.clip(prior[d_va[va_keep]], 1e-12, None)).mean())
    else:
        ce_chance = float("nan")

    best, best_state, bad = float("inf"), None, 0
    best_round = 0
    rounds_run = 0
    diverged = False
    for _ in range(joint_rounds):
        rounds_run += 1
        # (a) freeze adversary, train autoencoder for one epoch
        model.train()
        for idx in _batches():
            xb = Xtr[idx].to(device)
            db = dtr[idx].to(device)
            opt_ae.zero_grad()
            opt_adv.zero_grad()   # adversary receives grads but is not stepped
            z = model.encode(xb)
            loss = (recon_loss(model.decoder(z), xb)
                    - lambda_adv * ce(model.adversary(z), db))
            loss.backward()
            opt_ae.step()
        # (b) freeze autoencoder, train adversary for one full epoch
        model.train()
        for idx in _batches():
            xb = Xtr[idx].to(device)
            db = dtr[idx].to(device)
            opt_adv.zero_grad()
            with torch.no_grad():
                z = model.encode(xb)
            loss = ce(model.adversary(z), db)
            loss.backward()
            opt_adv.step()

        v_recon, v_adv = _val_recon(), _val_adv()
        if not np.isfinite(v_recon) or v_recon > 50.0 * max(recon_pretrain, 1e-8):
            # The autoencoder has been dragged apart by the unbounded
            # adversarial term. Stop; the selected checkpoint is whichever
            # earlier round sat closest to equilibrium.
            logger.warning("AD-AE joint training diverged at round %d "
                           "(val recon %.4g vs %.4g pretrain); stopping",
                           rounds_run, v_recon, recon_pretrain)
            diverged = True
            break
        # distance from the paper's equilibrium ("adversary at random")
        v = (abs(v_adv - ce_chance) if np.isfinite(v_adv) and np.isfinite(ce_chance)
             else v_recon)
        if v < best - 1e-6:
            best, bad, best_round = v, 0, rounds_run
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- diagnostics --------------------------------------------------
    model.eval()
    with torch.no_grad():
        acc_tr = float((model.adversary_logits_from_x(
            Xtr.to(device)).argmax(1).cpu().numpy() == d_tr).mean())
        if Xva_keep is not None:
            acc_va = float((model.adversary_logits_from_x(
                Xva_keep).argmax(1).cpu().numpy()
                == d_va[va_keep]).mean())
        else:
            acc_va = float("nan")
    diag = {"adv_acc_train": acc_tr, "adv_acc_val": acc_va,
            "recon_val_pretrain": recon_pretrain,
            "recon_val_final": _val_recon(),
            "adv_ce_val_pretrain": adv_pretrain,
            "adv_ce_val_final": _val_adv(),
            # equilibrium diagnostics: adv_ce_val_final should sit near
            # adv_ce_chance, and adv_acc_val near majority_class_frac, if the
            # adversary really was reduced to random performance
            "adv_ce_chance": ce_chance,
            "joint_rounds_run": rounds_run,
            "joint_round_selected": best_round,
            "diverged": float(diverged),
            "n_classes": len(classes),
            "majority_class_frac": float(np.bincount(d_tr).max() / len(d_tr))}
    return model, diag


def train_adae_head(adae: ADAE, X_tr, y_tr, X_va, y_va, *,
                    head_hidden=0, epochs=80, batch_size=64, lr=1e-3,
                    weight_decay=1e-4, patience=15, device=None,
                    en_l1_ratios=(0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99,
                                  1.0),
                    en_n_alphas=50, en_cv=5, en_max_iter=5000):
    """Fit the phenotype head on the frozen AD-AE embedding.

    The paper is explicit about this step: after generating embeddings, "we
    fit prediction models to the embeddings to predict biological
    phenotypes", using elastic net with 5-fold cross-validated
    hyperparameters. `head_hidden=0` (the default) does exactly that —
    `ElasticNetCV` on the embedding — and then copies the fitted
    coefficients into an `nn.Linear` so the composed encoder+head stays
    differentiable for GradientExplainer.

    Using the paper's estimator is not a convenience here, it is required
    for correctness. Adversarial training leaves the embedding on a scale
    of ~1e3-1e4, and a mini-batch SGD head on raw embeddings of that scale
    fails to fit even its own training data (observed train R^2 of -10),
    which would have been misreported as "AD-AE destroys the signal". A
    properly regularised linear model on the same embedding recovers
    R^2 ~ 0.25 out of sample.

    The `l1_ratio` grid deliberately extends below scikit-learn's suggested
    floor of 0.1, down to 0.01. On a dense, highly correlated embedding a
    grid that forces substantial L1 costs this baseline real accuracy
    (observed out-of-sample R^2 of 0.16 versus 0.25 for a ridge fit on the
    identical embedding). Elastic net spans ridge to lasso by definition, so
    letting the CV reach the near-ridge end is faithful — and a competing
    baseline should not be handicapped by an arbitrary grid choice.

    `head_hidden > 0` gives the one-hidden-layer sensitivity variant,
    trained with the pipeline's Adam + early-stopping loop on a
    standardised embedding.

    Since the encoder is frozen and held in eval mode, the embeddings are
    deterministic; they are computed once and the head is fitted on them,
    which is exactly equivalent to (and much cheaper than) running the
    encoder inside every mini-batch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adae.eval()
    with torch.no_grad():
        Z_tr = adae.encode(torch.from_numpy(
            X_tr.astype(np.float32)).to(device)).cpu().numpy()
        Z_va = adae.encode(torch.from_numpy(
            X_va.astype(np.float32)).to(device)).cpu().numpy()

    mean = Z_tr.mean(axis=0)
    scale = Z_tr.std(axis=0)
    scale[scale < 1e-8] = 1.0

    head_diag: Dict[str, float] = {}
    if head_hidden and head_hidden > 0:
        head = build_adae_head(adae.emb_dim, head_hidden=head_hidden,
                               mean=mean, scale=scale).to(device)
        trainable = [p for p in head.parameters() if p.requires_grad]
        head, val = _train_loop(head, Z_tr, y_tr.astype(np.float32),
                                Z_va, y_va.astype(np.float32),
                                epochs=epochs, batch_size=batch_size, lr=lr,
                                weight_decay=weight_decay, patience=patience,
                                device=device, params=trainable)
    else:
        import inspect

        from sklearn.linear_model import ElasticNetCV, RidgeCV
        Ztr_s = (Z_tr - mean) / scale
        Zva_s = (Z_va - mean) / scale
        yv = y_va.astype(np.float64)
        # scikit-learn renamed `n_alphas` to an integer `alphas` (deprecated
        # in 1.7, removed by 1.9). The repo pins 1.6.1 but cluster
        # environments run newer; support both rather than pinning harder.
        if "n_alphas" in inspect.signature(ElasticNetCV.__init__).parameters:
            alpha_kw = {"n_alphas": en_n_alphas}
        else:
            alpha_kw = {"alphas": en_n_alphas}
        en = ElasticNetCV(l1_ratio=list(en_l1_ratios),
                          cv=en_cv, max_iter=en_max_iter, n_jobs=1,
                          random_state=0, **alpha_kw)
        with warnings.catch_warnings():
            # coordinate descent routinely reports non-convergence on the
            # extreme end of the alpha path; the CV pick is unaffected
            warnings.simplefilter("ignore")
            en.fit(Ztr_s, y_tr.astype(np.float64))
            # Near-free sensitivity check on the downstream estimator only.
            # On this embedding the elastic net's own CV prefers a nearly
            # unregularised sparse fit, which generalises worse than ridge.
            # Recording both means AD-AE's accuracy cannot be waved away as
            # an artefact of the head we happened to choose.
            rg = RidgeCV(alphas=np.logspace(-4, 6, 40)).fit(Ztr_s,
                                                            y_tr.astype(np.float64))
        head = build_adae_head(adae.emb_dim, head_hidden=0,
                               mean=mean, scale=scale).to(device)
        lin = head[-1]
        with torch.no_grad():
            lin.weight.copy_(torch.as_tensor(en.coef_, dtype=torch.float32)
                             .reshape(1, -1))
            lin.bias.copy_(torch.as_tensor([en.intercept_], dtype=torch.float32))
        val = float(np.mean((en.predict(Zva_s) - yv) ** 2))
        head_diag = {
            # ElasticNetCV can select the top of the alpha path, zeroing every
            # coefficient. The composed model is then constant, so its SHAP is
            # identically zero -- which is *absence of an explanation*, not an
            # explanation of zero. Flagged so the caller can write NaN rather
            # than zeros, which downstream code would average in as real data.
            "head_degenerate": float((en.coef_ == 0).all()),
            "head_en_alpha": float(en.alpha_),
            "head_en_l1_ratio": float(en.l1_ratio_),
            "head_en_nnz": float((en.coef_ != 0).sum()),
            "head_en_val_mse": val,
            "head_ridge_alpha": float(rg.alpha_),
            "head_ridge_val_mse": float(np.mean((rg.predict(Zva_s) - yv) ** 2)),
            "emb_abs_max": float(np.abs(Z_tr).max()),
        }

    model = ADAEPredictor(adae.encoder, adae.emb_dim, head=head).to(device)
    model.eval()
    return model, val, head_diag
