"""Small MLPs used by all four deconfounding methods.

Same backbone everywhere — the only difference is the *training loss*.
SHAP's GradientExplainer expects a 2D output, so forward returns (n, 1).
"""
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
