"""Synthetic ground-truth generator for the causal-SHAP-under-tissue-confounding paper.

DGP (slightly nonlinear, designed to be honest):

  1. Sample `n_tissues` tissue-mean expression profiles μ_t ∈ ℝ^P
     and within-tissue noise: X_c = μ_{t(c)} + ε_c, ε_c ~ N(0, σ_w²I).
  2. Designate a disjoint set of *causal* genes (G_c) and *lineage-marker*
     genes (G_m). Lineage markers get their tissue-specific component
     amplified so their η²(X) is high; they have NO causal effect on y.
  3. Generate y per cell as
        y_c = y_tissue[t(c)]                     (between-tissue mean)
            + tanh(β · X_c[G_c])                 (nonlinear within-tissue,
                                                  one shared coefficient)
            + γ · X_c[g1] * X_c[g2]              (pairwise interaction)
            + noise
     The tanh saturates large activations (a standard mild nonlinearity)
     and the interaction term means a linear model cannot fully recover
     the causal genes' effect.

Returns gene names "g0001"..."gNNNN" so downstream code that treats
features as strings still works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SyntheticDataset:
    X: np.ndarray            # (n_cells, n_genes)
    y: np.ndarray            # (n_cells,)
    T: np.ndarray            # (n_cells,) tissue labels as strings
    gene_names: List[str]    # length n_genes
    causal_genes: List[str]  # ground-truth gene names with non-zero effect
    lineage_genes: List[str] # tissue-marker confounders (no effect on y)
    eta_y: float             # fraction of y variance explained by tissue


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = float(values.mean())
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total < 1e-12:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.any():
            ss_between += int(m.sum()) * (values[m].mean() - grand) ** 2
    return float(ss_between / ss_total)


def make_synthetic(
    n_cells: int = 800,
    n_tissues: int = 10,
    n_genes: int = 300,
    n_causal: int = 5,
    n_lineage: int = 30,
    n_strong_lineage: int = 2,
    tissue_y_sd: float = 2.5,
    causal_beta: float = 0.9,
    interaction_gamma: float = 1.2,
    within_noise: float = 0.5,
    y_noise: float = 5.0,
    lineage_amp_sd: float = 3.0,
    strong_amp_sd: float = 9.0,
    lineage_noise_sd: float = 1.0,
    strong_noise_sd: float = 0.1,
    seed: int = 0,
) -> SyntheticDataset:
    rng = np.random.RandomState(seed)

    # Tissue sizes: unequal (Dirichlet-like), at least n_tissues cells each
    sizes_raw = rng.gamma(2.0, 1.0, size=n_tissues)
    sizes = np.maximum(
        ((sizes_raw / sizes_raw.sum()) * n_cells).round().astype(int),
        max(10, n_cells // (n_tissues * 4)),
    )
    sizes = sizes * n_cells // sizes.sum()
    sizes[-1] = n_cells - sizes[:-1].sum()
    T = np.repeat([f"t{i}" for i in range(n_tissues)], sizes).astype(str)
    rng.shuffle(T)

    # Tissue baseline expression: each tissue has a distinct mean per gene
    tissue_mu = rng.normal(0.0, 1.0, size=(n_tissues, n_genes))
    tissue_idx = {t: i for i, t in enumerate(np.unique(T))}
    cell_tissue_idx = np.array([tissue_idx[t] for t in T])

    # Base expression: tissue mean + within-tissue noise
    X = tissue_mu[cell_tissue_idx] + rng.normal(
        0.0, within_noise, size=(n_cells, n_genes))

    # Choose disjoint causal and lineage-marker gene sets
    all_idx = np.arange(n_genes)
    causal_idx = rng.choice(all_idx, size=n_causal, replace=False)
    remaining = np.setdiff1d(all_idx, causal_idx)
    lineage_idx = rng.choice(remaining, size=n_lineage, replace=False)

    # Lineage genes: tissue-specific amplitude + per-cell observation noise.
    # Two tiers: a few STRONG markers with large amplitude that
    # cleanly identify the tissue — these single-handedly predict the
    # between-tissue component of y, so the *marginal* model leans on them and
    # they crowd the top of the ranking, burying the (weak, within-tissue)
    # causal genes. Residualizing tissue out of y removes the incentive to use
    # them, so causal genes can surface — the recovery mechanism. The MANY
    # WEAK markers keep contamination broad (no single one nails the tissue,
    # so the model spreads attribution across them, filling top-K).
    n_strong = min(n_strong_lineage, n_lineage)
    per_gene_sd = np.full(n_lineage, lineage_amp_sd)
    per_gene_sd[:n_strong] = strong_amp_sd
    # Strong markers are near-noiseless (each one alone identifies the tissue,
    # so the marginal model leans on them); weak markers are noisy (no single
    # one is reliable, so the model must aggregate, spreading attribution).
    per_gene_noise = np.full(n_lineage, lineage_noise_sd)
    per_gene_noise[:n_strong] = strong_noise_sd
    lineage_amp = rng.normal(0.0, 1.0, size=(n_tissues, n_lineage)) * per_gene_sd
    X[:, lineage_idx] = (lineage_amp[cell_tissue_idx]
                          + rng.normal(0.0, 1.0, size=(n_cells, n_lineage))
                            * per_gene_noise)

    # Causal genes: standard tissue+within structure, no extra amplification

    # y: tissue baseline + nonlinear within-tissue effect of causal genes.
    # The within-tissue function combines three nonlinearities so a linear
    # surrogate can't fully recover it: a saturating tanh of a linear
    # combination, a multiplicative pairwise interaction, and a quadratic
    # term on one causal gene.
    y_tissue_mean = rng.normal(0.0, tissue_y_sd, size=n_tissues)
    causal_weights = rng.normal(0.0, 1.0, size=n_causal)
    linear_combo = X[:, causal_idx] @ causal_weights * causal_beta
    nonlinear = np.tanh(linear_combo)
    interaction = (interaction_gamma
                    * X[:, causal_idx[0]] * X[:, causal_idx[1 % n_causal]])
    quadratic = (interaction_gamma * 0.5
                  * (X[:, causal_idx[2 % n_causal]] ** 2 - 1.0))
    y = (y_tissue_mean[cell_tissue_idx]
         + nonlinear
         + interaction
         + quadratic
         + rng.normal(0.0, y_noise, size=n_cells)).astype(np.float32)

    gene_names = [f"g{i:04d}" for i in range(n_genes)]
    causal_names = [gene_names[i] for i in causal_idx]
    lineage_names = [gene_names[i] for i in lineage_idx]

    return SyntheticDataset(
        X=X.astype(np.float32), y=y, T=T,
        gene_names=gene_names,
        causal_genes=causal_names,
        lineage_genes=lineage_names,
        eta_y=_eta_squared(y, T),
    )


@dataclass
class MixedDataset(SyntheticDataset):
    mixed_genes: List[str] = None       # causal AND tissue-confounded
    pure_causal_genes: List[str] = None  # causal, not tissue-confounded
    effect_size: dict = None             # gene -> within-tissue effect size r


def make_synthetic_mixed(
    n_cells: int = 1000,
    n_tissues: int = 12,
    n_genes: int = 500,
    n_causal: int = 200,        # causal, NOT tissue-confounded
    n_mixed: int = 100,         # causal AND tissue-confounded (the hard case)
    n_confounder: int = 50,     # pure tissue markers, NO effect on y
    n_strong_confounder: int = 15,
    tissue_y_sd: float = 1.0,   # between-tissue y baseline -> drives eta_y (CTRPv2-like ~0.20)
    within_noise: float = 1.0,
    y_noise: float = 2.0,       # within-tissue residual noise -> drives within-r
    effect_scale: float = 0.085,  # overall causal coefficient magnitude
    tier_fracs: Tuple = (0.05, 0.15, 0.30),   # strong / medium / weak fractions (rest v.weak)
    tier_mags: Tuple = (4.0, 1.6, 0.6, 0.12),  # magnitude per tier (strong..v.weak)
    conf_y_coupling: float = 1.0,  # how much confounders drive y's tissue baseline (spurious)
    mixed_amp_sd: float = 1.0,    # tissue amplitude of mixed genes (moderate)
    conf_amp_sd: float = 6.0,     # tissue amplitude of weak pure confounders
    strong_amp_sd: float = 9.0,   # tissue amplitude of strong pure confounders
    conf_noise_sd: float = 1.0,
    strong_noise_sd: float = 0.1,
    seed: int = 0,
) -> MixedDataset:
    """Realistic mixed-gene DGP for the N-sweep (calibrated to DepMap/CTRPv2).

    Four disjoint gene classes in a panel of `n_genes`:
      - causal      (n_causal): affect y, ordinary tissue structure
      - mixed       (n_mixed):  affect y AND tissue-amplified expression -- the
                                genes marginal SHAP buries (tissue-confounded
                                causal signal) and deconfounding should rescue
      - confounder  (n_confounder): tissue-amplified, NO effect on y (a few
                                strong + many weak), inflate eta^2(X) -> contaminate
                                attributions without inflating eta^2(y)
      - noise       (rest): inert

    Per-gene causal coefficients are lognormal (heterogeneous): a few strong
    genes are recoverable at small n, most are weak and need large n. y mixes a
    between-tissue baseline, the (full-expression) causal+mixed signal -- so
    tissue-amplified mixed genes inject confounding into y -- a mild tanh
    nonlinearity, and noise. Measured over 8 seeds, defaults give eta^2(y) ~ 0.21
    at n=1000; within-tissue predictive r rises with n from ~0.05 at n=1000 to
    ~0.25 at n=80000, reaching the real-cohort level (~0.17) near n=10-20k.
    """
    rng = np.random.RandomState(seed)

    sizes_raw = rng.gamma(2.0, 1.0, size=n_tissues)
    sizes = np.maximum(((sizes_raw / sizes_raw.sum()) * n_cells).round().astype(int),
                       max(10, n_cells // (n_tissues * 4)))
    sizes = sizes * n_cells // sizes.sum()
    sizes[-1] = n_cells - sizes[:-1].sum()
    T = np.repeat([f"t{i}" for i in range(n_tissues)], sizes).astype(str)
    rng.shuffle(T)

    tissue_mu = rng.normal(0.0, 1.0, size=(n_tissues, n_genes))
    tidx = {t: i for i, t in enumerate(np.unique(T))}
    cti = np.array([tidx[t] for t in T])
    X = tissue_mu[cti] + rng.normal(0.0, within_noise, size=(n_cells, n_genes))

    # Disjoint class assignment
    perm = rng.permutation(n_genes)
    causal_idx = perm[:n_causal]
    mixed_idx = perm[n_causal:n_causal + n_mixed]
    conf_idx = perm[n_causal + n_mixed:n_causal + n_mixed + n_confounder]

    # Tissue amplification for mixed (moderate) + pure confounders (strong/weak)
    def amplify(idx, amp_sd, noise_sd):
        amp = rng.normal(0.0, 1.0, size=(n_tissues, len(idx))) * amp_sd
        X[:, idx] = amp[cti] + rng.normal(0.0, 1.0, size=(n_cells, len(idx))) * noise_sd
        return amp

    amplify(mixed_idx, mixed_amp_sd, conf_noise_sd)
    n_strong = min(n_strong_confounder, n_confounder)
    amp_s = amplify(conf_idx[:n_strong], strong_amp_sd, strong_noise_sd)
    amp_w = amplify(conf_idx[n_strong:], conf_amp_sd, conf_noise_sd)
    conf_amp = np.concatenate([amp_s, amp_w], axis=1)   # (n_tissues, n_confounder)

    # Tiered causal coefficients over causal + mixed genes: a small strong tier
    # (recoverable at low n) + a long weak/v.weak tail (only at high n).
    eff_idx = np.concatenate([causal_idx, mixed_idx])
    n_eff = len(eff_idx)
    f_strong, f_med, f_weak = tier_fracs
    counts = [int(round(f_strong * n_eff)), int(round(f_med * n_eff)),
              int(round(f_weak * n_eff))]
    counts.append(n_eff - sum(counts))                 # v.weak = remainder
    mag = np.concatenate([np.full(c, m) for c, m in zip(counts, tier_mags)])
    mag = mag[rng.permutation(n_eff)]                  # shuffle tiers across genes
    sign = rng.choice([-1.0, 1.0], size=n_eff)
    beta = effect_scale * sign * mag

    # Tissue baseline of y is (spuriously) driven by the confounder genes' tissue
    # signature: tissue causes both confounder expression and y, so confounders
    # predict y via tissue at the POPULATION level -- persistent confounding that
    # no amount of data removes. Marginal SHAP keeps ranking them; residualizing
    # tissue out of y deletes this baseline, demoting them at every n.
    w_conf = rng.normal(0.0, 1.0, size=n_confounder)
    conf_sig = conf_amp @ w_conf
    conf_sig = (conf_sig - conf_sig.mean()) / (conf_sig.std() + 1e-8)
    indep = rng.normal(0.0, 1.0, size=n_tissues)
    base = conf_y_coupling * conf_sig + (1.0 - conf_y_coupling) * indep
    base = (base - base.mean()) / (base.std() + 1e-8)
    y_tissue = base * tissue_y_sd

    combo = X[:, eff_idx] @ beta
    signal = np.tanh(combo)                       # mild saturating nonlinearity
    y = (y_tissue[cti] + signal
         + rng.normal(0.0, y_noise, size=n_cells)).astype(np.float32)

    # Per-gene within-tissue effect size (noiseless signal corr), method-independent
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xm = np.zeros_like(Xz)
    for t in np.unique(T):
        m = T == t
        Xm[m] = Xz[m].mean(0)
    Xr = Xz - Xm                                  # within-tissue residual expression
    sig_w = signal.copy()                         # within-tissue causal signal
    for t in np.unique(T):
        m = T == t
        sig_w[m] = signal[m] - signal[m].mean()
    eff = {}
    for j, gi in enumerate(eff_idx):
        denom = (Xr[:, gi].std() * sig_w.std())
        eff[f"g{gi:04d}"] = float(abs((Xr[:, gi] * sig_w).mean() / denom)) if denom > 1e-9 else 0.0

    gene_names = [f"g{i:04d}" for i in range(n_genes)]
    causal_names = [gene_names[i] for i in causal_idx]
    mixed_names = [gene_names[i] for i in mixed_idx]
    conf_names = [gene_names[i] for i in conf_idx]

    return MixedDataset(
        X=X.astype(np.float32), y=y, T=T,
        gene_names=gene_names,
        causal_genes=causal_names + mixed_names,   # full ground-truth causal set
        lineage_genes=conf_names,
        eta_y=_eta_squared(y, T),
        mixed_genes=mixed_names,
        pure_causal_genes=causal_names,
        effect_size=eff,
    )


def make_synthetic_concentrated(n_cells: int = 1000, seed: int = 0, **overrides) -> MixedDataset:
    """Concentrated-causal sibling of make_synthetic_mixed (calibration companion).

    Few strong causal genes (5 + 2 mixed) -- the SLFN11-like regime: a single
    target has a small number of genuinely strong individual-gene predictors.
    Same DGP machinery, same confounding mechanism, same gene-class structure.

    Calibrated so a marginal MLP achieves within-tissue predictive r ~ 0.17 at
    n=1000 (matching the real DepMap target median) and rises to ~0.50 at
    n=20000 -- providing the easy/concentrated counterpart to the diffuse 300-
    causal-gene DGP. The two regimes bookend the range of real-target behaviour:
    DepMap p25 targets sit near the diffuse curve, p75 / SLFN11-like targets
    sit near this concentrated curve.
    """
    defaults = dict(
        n_causal=5, n_mixed=2, n_confounder=50, n_strong_confounder=15,
        tier_fracs=(1.0, 0.0, 0.0), tier_mags=(1.0, 1.0, 1.0, 1.0),
        effect_scale=0.25, y_noise=0.65, tissue_y_sd=0.28,
    )
    defaults.update(overrides)
    return make_synthetic_mixed(n_cells=n_cells, seed=seed, **defaults)
