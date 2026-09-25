"""Small MLPs used by all deconfounding methods.

Same backbone everywhere — the only difference is the *training loss*.
SHAP's GradientExplainer expects a 2D output, so forward returns (n, 1).

Two model-stage adversarial baselines are defined here alongside the
shared backbone:

  * `DANNModel`   — domain-adversarial training with a gradient reversal
                    layer (Ganin et al., JMLR 2016).
  * `ADAE`        — adversarial deconfounding autoencoder
                    (Dincer, Janizek & Lee, Bioinformatics 2020), plus
                    `ADAEPredictor`, the frozen-encoder + head model that
                    is what actually gets explained.

Both expose a `forward` that is the *prediction* path only, so
`shap.GradientExplainer` treats them exactly like `SmallMLP`.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """1-hidden-layer MLP with dropout. Used for marginal, residualized,
    IRM, and within-tissue SHAP (single global model)."""
    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===================================================== DANN (Ganin et al.)
class _GradientReversalFn(torch.autograd.Function):
    """The gradient reversal layer of Ganin et al. (2016), their Eqs. (16-17):

        R(x) = x            (forward: identity)
        dR/dx = -I          (backward: negate)

    scaled by the adaptation parameter lambda. Because lambda enters only
    through this backward pass, and this layer sits between the feature
    extractor G_f and the domain classifier G_d, the scaling applies *only*
    to the gradient that reaches G_f. G_d itself receives the domain loss
    gradient with coefficient 1. That reproduces the paper's rule:

        "these lambda_p were used only for updating the feature extractor
         component G_f. For updating the domain classification component,
         we used a fixed lambda = 1."
    """
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambd)


def dann_lambda(progress: float, gamma: float = 10.0,
                lambda_max: float = 1.0) -> float:
    """Ganin et al.'s adaptation-parameter schedule:

        lambda_p = 2 / (1 + exp(-gamma * p)) - 1,      gamma = 10

    with `p` the training progress moving linearly from 0 to 1. The paper
    ramps from 0 to 1; `lambda_max` rescales the ceiling so the penalty
    weight can be swept while keeping the schedule *shape* identical to the
    original (lambda_max=1.0 is the paper's setting).
    """
    p = min(max(float(progress), 0.0), 1.0)
    import math
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class DANNModel(nn.Module):
    """Domain-adversarial neural network built on the `SmallMLP` backbone.

    The backbone is split at its last hidden layer:
      * G_f (`features`)    : in_dim -> 128 -> 64, ReLU + dropout
      * G_y (`label_head`)  : 64 -> 1, trained with MSE (our task is
                              regression, so L_y is squared error rather
                              than the paper's logistic loss)
      * G_d (`domain_head`) : 64 -> 64 -> K, softmax cross-entropy over the
                              K tissues, reached through the GRL

    Parameters are constructed in the same order as `SmallMLP`
    (in->128, 128->64, 64->1) before the domain head, so that with an
    identical torch seed a DANN model and a `SmallMLP` start from
    identical weights. That makes the lambda=0 equivalence check exact.

    `forward` is the label path only, so GradientExplainer sees the same
    function shape as for every other method.
    """
    def __init__(self, in_dim: int, n_domains: int,
                 hidden: Sequence[int] = (128, 64), dropout: float = 0.3,
                 domain_hidden: int = 64):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.features = nn.Sequential(*layers)
        self.label_head = nn.Linear(d, 1)
        # Build the domain head without advancing the global RNG stream, so
        # that a DANN model and a SmallMLP built under the same seed are
        # identical *and* stay in step through training (mini-batch
        # permutations and dropout masks are drawn from the same stream).
        # That is what makes the lambda=0 run an exact reproduction of the
        # marginal model rather than merely a same-in-distribution one.
        rng_state = torch.random.get_rng_state()
        self.domain_head = nn.Sequential(
            nn.Linear(d, domain_hidden), nn.ReLU(),
            nn.Linear(domain_hidden, n_domains))
        torch.random.set_rng_state(rng_state)

    def forward(self, x):
        """Prediction path (what SHAP explains)."""
        return self.label_head(self.features(x))

    def forward_adv(self, x, lambd: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (y_hat, domain_logits) with the GRL applied at `lambd`."""
        z = self.features(x)
        return self.label_head(z), self.domain_head(grad_reverse(z, lambd))

    def domain_logits(self, x) -> torch.Tensor:
        """Domain-classifier output without gradient reversal (diagnostics)."""
        return self.domain_head(self.features(x))


# ============================================ AD-AE (Dincer, Janizek & Lee)
class ADAE(nn.Module):
    """Adversarial deconfounding autoencoder.

    Architecture follows the paper's breast-cancer configuration, which is
    the one that matches our input dimensionality (they reduce expression to
    ~1000 k-means gene centers; our panels are 1071/1109 genes):

      encoder   : in_dim -> 500 -> emb_dim     ReLU on the hidden layer,
                                               linear embedding output
      decoder   : emb_dim -> 500 -> in_dim     ReLU on the hidden layer,
                                               linear reconstruction output
      adversary : emb_dim -> emb_dim -> emb_dim -> K
                  two hidden layers, softmax over the K confounder classes
                  (the paper sizes the adversary's hidden layers to the
                  embedding dimension: 100 for emb=100, 50 for emb=50)

    Dropout (0.1 in the paper's breast setting) is applied after each hidden
    ReLU of the encoder and decoder. The adversary has no dropout.

    Objectives (paper Section 2):
        autoencoder pretrain : min_{phi,psi}  E|| x - g_psi(f_phi(x)) ||^2
        adversary  pretrain  : min_{upsilon} E[ L(h_upsilon(f_phi(x)), c) ]
        joint (autoencoder)  : min_{phi,psi}
                               E[ ||x - g(f(x))||^2 - lambda * L(h(f(x)), c) ]

    i.e. the autoencoder *maximizes* the adversary's loss. The adversary is
    never given the phenotype, so the embedding depends only on (X, tissue)
    — which is what makes the per-fold encoder cacheable across items.
    """
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 500,
                 emb_dim: int = 100, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim))
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, in_dim))
        self.adversary = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, n_classes))

    def encode(self, x):
        return self.encoder(x)

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def adversary_logits_from_x(self, x):
        return self.adversary(self.encoder(x))

    def ae_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def adversary_parameters(self):
        return list(self.adversary.parameters())


class ADAEPredictor(nn.Module):
    """Frozen AD-AE encoder + phenotype head — the model that is explained.

    The paper fits downstream predictors (elastic net) on the learned
    embedding; the differentiable equivalent that keeps SHAP flowing back to
    genes is a weight-decayed linear head. `head_hidden` optionally makes it
    a one-hidden-layer MLP for the sensitivity check.

    The encoder is frozen (requires_grad=False) so that fitting the head
    cannot undo the deconfounding — matching the paper, where the embedding
    is fixed before any phenotype model is fit. Gradients still flow through
    it to the inputs, which is what GradientExplainer needs.
    """
    def __init__(self, encoder: nn.Module, emb_dim: int,
                 head: nn.Module = None,
                 head_hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if head is not None:
            self.head = head
        elif head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        else:
            self.head = nn.Linear(emb_dim, 1)

    def train(self, mode: bool = True):
        """Keep the encoder in eval mode always.

        The paper fixes the embedding before fitting any phenotype model, so
        the encoder's dropout must not inject noise while the head is
        trained. Only the head follows the caller's train/eval mode.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        return self.head(self.encoder(x))

    def head_parameters(self):
        return list(self.head.parameters())


class FixedStandardize(nn.Module):
    """Affine rescaling of the embedding with statistics frozen from train.

    Adversarial training drives the AD-AE embedding to a very large scale
    (component magnitudes in the thousands are routine, because the
    ``-lambda*CE`` term is unbounded and rewards ever-larger logits). A
    downstream model therefore has to be scale-robust. The paper's elastic
    net is, by virtue of being properly regularised and solved in closed
    form; a plain SGD head is not, and silently underfits.

    Stored as buffers, not parameters, so no optimiser ever touches them and
    they travel with the state dict.
    """
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, z):
        return (z - self.mean) / self.scale


def build_adae_head(emb_dim: int, head_hidden: int = 0,
                    dropout: float = 0.0, mean=None, scale=None) -> nn.Module:
    """The downstream phenotype head fitted on a frozen AD-AE embedding."""
    layers = []
    if mean is not None and scale is not None:
        layers.append(FixedStandardize(mean, scale))
    if head_hidden and head_hidden > 0:
        layers += [nn.Linear(emb_dim, head_hidden), nn.ReLU(),
                   nn.Dropout(dropout), nn.Linear(head_hidden, 1)]
    else:
        layers.append(nn.Linear(emb_dim, 1))
    return nn.Sequential(*layers)
