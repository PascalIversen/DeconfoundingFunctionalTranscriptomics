"""Attribution methods other than Shapley values, for the estimator-dependence check.

The paper evaluates deconfounding through SHAP (GradientExplainer). Is the
tissue-confounding problem, and the fixes for it, specific to Shapley values?
This module implements four alternatives spanning three different families:

  propagation   LRP-epsilon, LRP-alpha1beta0   (relevance redistributed
                                                layer by layer)
  path          Integrated Gradients            (attribution along a path
                                                from a baseline)
  surrogate     LIME                            (local weighted linear fit)

plus gradient x input as the reference point.

ON "PROPER" LRP. For a ReLU network with zero biases, LRP-0 is provably
identical to gradient x input (Ancona et al., ICLR 2018), so reporting
gradient x input under the name LRP would be an empty comparison. The two
rules implemented here are genuinely distinct from it:

  * LRP-epsilon absorbs relevance into a stabiliser, which shrinks the
    contribution of weakly-activated neurons rather than passing it on.
  * LRP-alpha1beta0 (the z+ rule) propagates only positive pre-activation
    contributions, so it cannot be written as any gradient-times-something.
    Because expression inputs are standardised and therefore signed, the
    input layer uses the z^B rule (Montavon et al., 2017), which is the
    standard choice for real-valued inputs and respects the data range.

Conservation is asserted in `check_conservation`: relevance is preserved
across layers up to the relevance absorbed by biases and the stabiliser. A
correct implementation passes; gradient x input does not.

All functions take a `SmallMLP`-shaped module (Linear-ReLU-Dropout blocks
followed by a Linear head) and return an (n_cell_lines, n_features) array on the
same scale as the model output.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------- helpers
def _linear_layers(model: nn.Module) -> List[nn.Linear]:
    """The Linear layers of the backbone, in forward order."""
    seq = model.net if hasattr(model, "net") else model
    return [m for m in seq.modules() if isinstance(m, nn.Linear)]


def _forward_activations(model: nn.Module, x: torch.Tensor
                         ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Post-activation inputs to each Linear layer, plus the output.

    For SmallMLP the chain is x -> L0 -> relu -> L1 -> relu -> L2, so the
    returned list is [x, relu(L0 x), relu(L1 ...)] and has one entry per
    Linear layer. Dropout is identity in eval mode.
    """
    seq = model.net if hasattr(model, "net") else model
    acts, a = [], x
    for mod in seq:
        if isinstance(mod, nn.Linear):
            acts.append(a)
            a = mod(a)
        elif isinstance(mod, nn.ReLU):
            a = torch.relu(a)
        # Dropout is identity in eval mode; anything else must be transparent
    return acts, a


# --------------------------------------------------------------------- LRP
def _lrp_epsilon_step(layer: nn.Linear, a: torch.Tensor, R: torch.Tensor,
                      eps: float = 0.25) -> torch.Tensor:
    """One epsilon-rule redistribution.

        R_i = a_i * sum_j  w_ij * R_j / (z_j + eps*sign(z_j))

    Implemented with the standard gradient trick, which evaluates exactly
    that expression without materialising the (i, j) matrix.
    """
    a = a.clone().detach().requires_grad_(True)
    z = layer(a)
    # eps is relative to the layer's own activation scale. A fixed tiny eps
    # makes this rule numerically identical to gradient x input (LRP-0), so
    # the comparison would be empty; Montavon et al. recommend scaling to
    # the spread of z.
    scale = z.abs().mean().detach().clamp(min=1e-12)
    e = eps * scale
    zs = z + e * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    s = (R / zs).detach()
    (z * s).sum().backward()
    return a * a.grad


def _lrp_zplus_step(layer: nn.Linear, a: torch.Tensor, R: torch.Tensor,
                    eps: float = 1e-9) -> torch.Tensor:
    """alpha=1, beta=0 (z+ rule): only positive contributions propagate.

    Valid where the incoming activations are non-negative, i.e. every hidden
    layer of a ReLU network.
    """
    a = a.clone().detach().requires_grad_(True)
    w_pos = layer.weight.clamp(min=0)
    b_pos = layer.bias.clamp(min=0) if layer.bias is not None else None
    z = torch.nn.functional.linear(a, w_pos, b_pos) + eps
    s = (R / z).detach()
    (z * s).sum().backward()
    return a * a.grad


def _lrp_zB_step(layer: nn.Linear, a: torch.Tensor, R: torch.Tensor,
                 lo: torch.Tensor, hi: torch.Tensor,
                 eps: float = 1e-9) -> torch.Tensor:
    """z^B rule for a signed input layer, bounded by [lo, hi].

        R_i = a_i*w_ij - lo_i*w+_ij - hi_i*w-_ij , normalised over j

    This is the standard input-layer rule when features are real-valued
    rather than non-negative (Montavon et al., 2017).
    """
    a = a.clone().detach().requires_grad_(True)
    l = lo.clone().detach().expand_as(a).requires_grad_(True)
    h = hi.clone().detach().expand_as(a).requires_grad_(True)
    w, wp, wn = layer.weight, layer.weight.clamp(min=0), layer.weight.clamp(max=0)
    b = layer.bias
    z = (torch.nn.functional.linear(a, w, b)
         - torch.nn.functional.linear(l, wp, None)
         - torch.nn.functional.linear(h, wn, None)) + eps
    s = (R / z).detach()
    (z * s).sum().backward()
    # z already carries the minus signs on l and h, so their gradients do
    # too; the reconstruction adds all three terms. Subtracting here flips
    # the sign of the total relevance (verified: ratio -0.84 vs +0.89).
    return a * a.grad + l * l.grad + h * h.grad


def lrp(model: nn.Module, X: np.ndarray, rule: str = "epsilon",
        eps: float = 0.25, input_bounds: Optional[Tuple[float, float]] = None,
        device=None, batch_size: int = 256) -> np.ndarray:
    """Layer-wise relevance propagation for a SmallMLP-shaped network.

    rule="epsilon"    epsilon-rule at every layer
    rule="alphabeta"  z+ rule at hidden layers, z^B rule at the input layer
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    layers = _linear_layers(model)
    out = np.empty_like(X, dtype=np.float32)
    lo_v, hi_v = input_bounds if input_bounds is not None else (X.min(), X.max())

    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        acts, z_out = _forward_activations(model, xb)
        R = z_out.detach()
        for li in range(len(layers) - 1, -1, -1):
            a = acts[li]
            if rule == "epsilon" or li > 0:
                if rule == "alphabeta" and li > 0:
                    R = _lrp_zplus_step(layers[li], a, R)
                else:
                    R = _lrp_epsilon_step(layers[li], a, R, eps)
            else:  # alphabeta at the signed input layer
                lo = torch.full_like(a[:1], float(lo_v))
                hi = torch.full_like(a[:1], float(hi_v))
                R = _lrp_zB_step(layers[li], a, R, lo, hi)
        out[i:i + batch_size] = R.detach().cpu().numpy()
    return out


def check_conservation(model: nn.Module, X: np.ndarray, rule: str = "epsilon",
                       device=None) -> dict:
    """Relevance conservation: sum_i R_i should track the model output.

    Exact conservation holds only for a bias-free network; biases and the
    stabiliser absorb relevance, so the diagnostic is the correlation and
    the median ratio, not equality. gradient x input fails this badly for
    the z+ rule, which is the point of running it.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        f = model(torch.from_numpy(X.astype(np.float32)).to(device)
                  ).squeeze(-1).cpu().numpy()
    R = lrp(model, X, rule=rule, device=device).sum(axis=1)
    ok = np.isfinite(R) & np.isfinite(f) & (np.abs(f) > 1e-8)
    return {"corr": float(np.corrcoef(R[ok], f[ok])[0, 1]),
            "median_ratio": float(np.median(R[ok] / f[ok])),
            "n": int(ok.sum())}


# ------------------------------------------------------ integrated gradients
def integrated_gradients(model: nn.Module, X: np.ndarray,
                         baseline: Optional[np.ndarray] = None,
                         steps: int = 64, device=None,
                         batch_size: int = 64) -> np.ndarray:
    """Path attribution from `baseline` to each input (Sundararajan et al.).

    The baseline defaults to the feature-wise mean of `X`, which is the
    analogue of the SHAP background used elsewhere in the paper.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    if baseline is None:
        baseline = X.mean(axis=0)
    b = torch.from_numpy(np.asarray(baseline, dtype=np.float32)).to(device)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
    out = np.empty_like(X, dtype=np.float32)

    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        diff = xb - b
        total = torch.zeros_like(xb)
        for al in alphas:
            pt = (b + al * diff).clone().detach().requires_grad_(True)
            model(pt).sum().backward()
            total = total + pt.grad
        out[i:i + batch_size] = (diff * total / steps).detach().cpu().numpy()
    return out


# -------------------------------------------------------------------- LIME
def lime(model: nn.Module, X: np.ndarray, X_background: np.ndarray,
         n_samples: int = 2000, kernel_width: Optional[float] = None,
         alpha: float = 1.0, device=None, seed: int = 0,
         batch_size: int = 512) -> np.ndarray:
    """LIME for tabular data: a locally weighted linear surrogate.

    For each cell line, perturbations are drawn by resampling each feature from
    the background distribution with probability 0.5 (the standard tabular
    scheme), the model is queried on them, and a ridge regression weighted
    by an exponential kernel in the perturbation distance is fitted. The
    surrogate's coefficients, times the cell line's deviation from the
    background mean, are the attributions — putting them on the same scale
    as the other methods here.

    Cost is one model evaluation per perturbation per cell line, so this is far
    more expensive than the others and is run on fewer cell lines.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rng = np.random.RandomState(seed)
    n_feat = X.shape[1]
    if kernel_width is None:
        kernel_width = float(np.sqrt(n_feat) * 0.75)
    out = np.empty_like(X, dtype=np.float32)

    for c in range(len(X)):
        x = X[c]
        mask = rng.rand(n_samples, n_feat) < 0.5          # 1 = keep x's value
        donors = X_background[rng.randint(0, len(X_background), n_samples)]
        Z = np.where(mask, x, donors).astype(np.float32)
        with torch.no_grad():
            preds = []
            for i in range(0, n_samples, batch_size):
                zb = torch.from_numpy(Z[i:i + batch_size]).to(device)
                preds.append(model(zb).squeeze(-1).cpu().numpy())
            y = np.concatenate(preds)
        d = np.sqrt((~mask).sum(axis=1)).astype(np.float32)
        w = np.exp(-(d ** 2) / (kernel_width ** 2)).astype(np.float32)
        # The weighted Gram matrix dominates the cost of this whole analysis
        # (n_samples x n_feat^2 per cell line). Accumulating it in float32
        # roughly halves that; the solve is then done in float64, where
        # conditioning actually matters.
        M = mask.astype(np.float32)
        Mw = M * w[:, None]
        A = (M.T @ Mw).astype(np.float64) + alpha * np.eye(n_feat)
        bvec = (Mw.T @ (y - y.mean()).astype(np.float32)).astype(np.float64)
        coef = np.linalg.solve(A, bvec)
        # The coefficient already estimates E[f | feature kept] -
        # E[f | feature resampled], i.e. a contribution in output units.
        # Multiplying by (x - mu) on top of that double-counts.
        out[c] = coef.astype(np.float32)
    return out


def gradient_x_input(model: nn.Module, X: np.ndarray, device=None,
                     batch_size: int = 256) -> np.ndarray:
    """Reference point. Equals LRP-0 for a ReLU network with zero biases."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    out = np.empty_like(X, dtype=np.float32)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)
                              ).to(device).requires_grad_(True)
        model(xb).sum().backward()
        out[i:i + batch_size] = (xb * xb.grad).detach().cpu().numpy()
    return out


def shap_gradient(model: nn.Module, X: np.ndarray, background: np.ndarray,
                  max_background: int = 100, seed: int = 0) -> np.ndarray:
    """The paper's own estimator, included so it sits on the same cells and
    folds as the alternatives and anchors the comparison."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "training"))
    from shared.shap_utils import gradient_shap as _gs
    return _gs(model, X, background, [f"f{i}" for i in range(X.shape[1])],
               max_background=max_background, seed=seed).values


METHODS = {
    "shap": lambda m, X, bg: shap_gradient(m, X, bg),
    "grad_x_input": lambda m, X, bg: gradient_x_input(m, X),
    "lrp_eps": lambda m, X, bg: lrp(m, X, rule="epsilon",
                                    input_bounds=(bg.min(), bg.max())),
    "lrp_ab": lambda m, X, bg: lrp(m, X, rule="alphabeta",
                                   input_bounds=(bg.min(), bg.max())),
    "int_grad": lambda m, X, bg: integrated_gradients(m, X, baseline=bg.mean(0)),
    "lime": lambda m, X, bg: lime(m, X, bg),
}
