"""Statistical kernels for the confounding-evidence figures.

Ported from `Confounding/bulk_deg_analysis.py` (a TxPert-style single-cell
confounding framework, Wenkel et al. 2026) but stripped of its `scanpy`/
`anndata` scaffolding so the dependency footprint matches the rest of
this repository (numpy / pandas / scipy / scikit-learn only). Every
function here operates on plain `(X, y, T)` arrays the same way the training
scripts load them via `shared/data.py`:

    X : (n_samples, n_genes) expression on the model's feature panel
    y : (n_samples,)        continuous outcome (IC50 / essentiality)
    T : (n_samples,)        grouping / candidate confounder (tissue, ...)

The two DAG arms each map to a kernel:

    Arm 1  Tissue → expression   :  tissue_classifier_lift, within_across_tissue_corr,
                                    pca_by_tissue  (+ per-gene η² via tissue_eta_squared)
    Arm 2  Tissue → outcome      :  outcome_anova

η²(X) per gene comes from `training/shared/eval.py::tissue_eta_squared`; the
scalar `eta_squared(values, groups)` behind the per-outcome η²(y) is defined
locally below; it computes the identical SS_between / SS_total as
`tissue_eta_squared`, so η²(X), η²(y) and the training pipeline's `eta_y` /
contamination metrics all remain one definition.
"""
from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu

# Per-gene η²(X) still comes from the training tree (kept in lock-step with the
# modelling pipeline). The training tree lives at ../training; insert it so
# `shared` is importable regardless of cwd.
_TRAINING = Path(__file__).resolve().parent.parent / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared.eval import tissue_eta_squared  # noqa: E402

__all__ = [
    "eta_squared",
    "tissue_eta_squared",
    "tissue_eta_squared_cols",
    "welch_anova",
    "outcome_anova",
    "tissue_classifier_lift",
    "within_across_tissue_corr",
    "pca_by_tissue",
]


# ── η² — one-way ANOVA effect size (SS_between / SS_total) ───────────────────
def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of variance in `values` explained by `groups` membership, in
    [0, 1]:  η² = SS_between / SS_total.  Returns 0.0 when `values` has
    (near-)zero total variance.

    Defined here rather than imported; the arithmetic matches
    `tissue_eta_squared`'s SS_between / SS_total, so the per-gene η²(X)
    (via `tissue_eta_squared`) and the per-outcome η²(y) (via this function in
    `outcome_anova`) stay one and the same definition.
    """
    values = np.asarray(values)
    grand = values.mean()
    ss_total = float(np.sum((values - grand) ** 2))
    if ss_total < 1e-12:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        mask = groups == g
        if mask.any():
            ss_between += float(mask.sum()) * (values[mask].mean() - grand) ** 2
    return float(ss_between / ss_total)


# ── η², vectorised over columns (for the shuffled-tissue null) ───────────────
def tissue_eta_squared_cols(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Per-column η² = SS_between / SS_total for an (N × G) matrix, vectorised
    over columns; only the group loop (≈ #tissues) runs in Python.

    Byte-for-byte the SS_between / SS_total arithmetic of `tissue_eta_squared`
    — added purely so the fig05 shuffled-tissue null can recompute η² over many
    label permutations cheaply. `compute_confounding.run_arm1` asserts this
    equals `tissue_eta_squared` on the real labels before trusting any
    permutation, so the per-gene η²(X) stays one definition.
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T)
    grand = X.mean(axis=0)
    ss_total = ((X - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(X.shape[1])
    for t in np.unique(T):
        m = T == t
        if m.any():
            ss_between += int(m.sum()) * (X[m].mean(axis=0) - grand) ** 2
    return np.where(ss_total < 1e-12, 0.0, ss_between / ss_total)


# ── Welch's one-way ANOVA — unequal-variance robust F-test ───────────────────
def welch_anova(arrays: List[np.ndarray]) -> Tuple[float, float]:
    """Welch's one-way ANOVA F-ratio and p-value, *without* assuming the groups
    share a common variance.

    Replaces the classic F-test (``scipy.stats.f_oneway``) for the Arm-2
    `outcome ~ tissue` comparison, whose equal-variance assumption fails badly:
    Levene rejects it for ~half of drugs/targets and the median across-tissue
    variance ratio is 4.8× (DepMap) / 13.1× (CTRPv2). Welch reweights each group by nᵢ/sᵢ² and
    adjusts the denominator df, staying valid under heteroscedasticity. (Non-
    normality is the one assumption it does not fix; Kruskal-Wallis is not used
    as a companion — under this same unequal-variance regime it no longer tests
    a clean location shift, only stochastic dominance.)

    Hand-rolled because no scipy version exposes Welch via ``f_oneway`` and the
    dependency pins stay frozen; only the F-distribution tail is
    borrowed (imported locally, as elsewhere in this module). Groups with < 2
    samples or (near-)zero variance are dropped — their weight 1/sᵢ² is undefined
    — and ``(nan, nan)`` is returned if fewer than two usable groups remain.
    """
    from scipy.stats import f as f_dist  # local import: matches this file's idiom

    means, variances, ns = [], [], []
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.size < 2:
            continue
        v = float(np.var(a, ddof=1))
        if v <= 1e-12:                     # zero-variance group → weight blows up
            continue
        means.append(float(a.mean()))
        variances.append(v)
        ns.append(float(a.size))

    k = len(means)
    if k < 2:
        return float("nan"), float("nan")

    means = np.asarray(means)
    w = np.asarray(ns) / np.asarray(variances)   # per-group precision weight
    ns = np.asarray(ns)
    W = float(w.sum())
    grand = float((w * means).sum() / W)         # weighted grand mean

    num = float((w * (means - grand) ** 2).sum()) / (k - 1)
    A = float((((1.0 - w / W) ** 2) / (ns - 1.0)).sum())
    F = num / (1.0 + 2.0 * (k - 2) / (k ** 2 - 1) * A)
    df2 = (k ** 2 - 1) / (3.0 * A)
    return float(F), float(f_dist.sf(F, k - 1, df2))


# ── Arm 2 — Tissue → outcome (ANOVA / Kruskal + η²) ──────────────────────────
def outcome_anova(y: np.ndarray, T: np.ndarray,
                  min_samples_per_group: int = 5) -> Dict:
    """Does a continuous outcome differ across groups? (Arm 2 of the DAG.)

    η²(y) is computed on the full `(y, T)` via the shared `eta_squared`, so it
    matches the pipeline's `eta_y`. Welch's F-test (parametric, unequal-variance
    robust — see `welch_anova`) is evaluated on groups with ≥
    `min_samples_per_group` samples so the test stays well-defined; its p-value
    is NaN when fewer than two such groups exist.
    Default 5 — the unified min-cells-per-category floor shared with the
    classifier, within/across corr and fig05c; η² itself stays unfiltered to
    match the modelling pipeline.

    Returns a dict: eta2, f_stat (Welch's F), p_anova (Welch p), n_groups
    (tested), n_samples_used (in tested groups), group_means (ascending Series).
    """
    y = np.asarray(y, dtype=np.float64)
    T = np.asarray(T).astype(str)
    eta2 = eta_squared(y, T)

    groups = {t: y[T == t] for t in np.unique(T)}
    groups = {t: v for t, v in groups.items() if len(v) >= min_samples_per_group}

    out = {
        "eta2": float(eta2),
        "f_stat": float("nan"), "p_anova": float("nan"),
        "n_groups": len(groups),
        "n_samples_used": int(sum(len(v) for v in groups.values())),
        "group_means": pd.Series({t: float(v.mean()) for t, v in groups.items()}
                                 ).sort_values(),
    }
    if len(groups) >= 2:
        arrays = list(groups.values())
        f_stat, p_anova = welch_anova(arrays)
        out.update(f_stat=float(f_stat), p_anova=float(p_anova))
    return out


# ── Arm 1 — Tissue → expression (linear tissue classifier) ───────────────────
def tissue_classifier_lift(X: np.ndarray, T: np.ndarray,
                           n_splits: int = 5, seed: int = 0,
                           max_iter: int = 200,
                           min_samples_per_tissue: int = 5) -> Dict:
    """Multinomial logistic regression: recover tissue from one sample's
    expression. If a linear model does this far above chance, tissue carries
    enough signal to confound any model that ignores it (Arm 1, strong form).

    Estimated by stratified ``n_splits``-fold cross-validation rather than a
    single holdout, so every sample is tested exactly once and the accuracy
    carries an across-fold SD (more honest than a lone split, and steadier on
    the small tissues floored at ``min_samples_per_tissue``). The
    ``StandardScaler`` is refit on each fold's *train* part only — no test
    leakage. ``accuracy`` / ``lift`` are the across-fold means; ``accuracy_std``
    / ``lift_std`` their SDs (``lift_std = accuracy_std / chance``, since chance
    is constant). If the smallest kept class has fewer than ``n_splits`` members
    the fold count is reduced to keep every fold populated.

    Operates on the model's feature panel directly (no HVG step — the panel is
    already the locked feature space). Returns accuracy, accuracy_std, chance
    (= 1/n_classes), lift, lift_std, n_classes, n_splits, n_samples, labels, a
    row-normalised confusion_matrix (DataFrame, pooled over out-of-fold
    predictions), and a per_tissue DataFrame (tissue, fold-averaged accuracy,
    accuracy_std across folds, n_samples, n_folds) — the per-tissue recall the
    swarm figure consumes, averaged over the same CV folds as `accuracy`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).astype(str)

    counts = pd.Series(T).value_counts()
    keep = set(counts[counts >= min_samples_per_tissue].index)
    mask = np.array([t in keep for t in T])
    X, T = X[mask], T[mask]
    n_classes = len(np.unique(T))
    if n_classes < 2:
        raise ValueError(
            f"Need ≥2 groups with ≥{min_samples_per_tissue} samples; "
            f"got {n_classes} after filtering.")

    # Stratified k-fold needs every class to have ≥ n_splits members; the
    # smallest kept tissue has `min_samples_per_tissue`, so cap accordingly.
    min_class = int(pd.Series(T).value_counts().min())
    n_splits = int(max(2, min(n_splits, min_class)))

    chance = 1.0 / n_classes
    labels = sorted(np.unique(T).tolist())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_acc: List[float] = []
    oof_true: List[str] = []
    oof_pred: List[str] = []
    # Per-tissue accuracy is the recall within each fold's held-out part, kept
    # per fold so each tissue carries a fold-averaged accuracy (the same across-
    # fold mean the overall `accuracy` uses, restricted to one tissue's rows).
    per_tissue_fold: Dict[str, List[float]] = {t: [] for t in labels}
    for tr, te in skf.split(X, T):
        # Refit the scaler on each fold's train part only — no test leakage.
        scaler = StandardScaler(with_mean=True).fit(X[tr])
        clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=1.0,
                                 random_state=seed)
        clf.fit(scaler.transform(X[tr]), T[tr])
        pred = clf.predict(scaler.transform(X[te]))
        fold_acc.append(float(accuracy_score(T[te], pred)))
        oof_true.extend(T[te].tolist())
        oof_pred.extend(pred.tolist())
        te_true, correct = T[te], (pred == T[te])
        for t in np.unique(te_true):
            per_tissue_fold[t].append(float(correct[te_true == t].mean()))

    acc = float(np.mean(fold_acc))
    acc_std = float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0
    cm = confusion_matrix(oof_true, oof_pred, labels=labels, normalize="true")
    per_tissue = pd.DataFrame([
        {"tissue": t,
         "accuracy": (float(np.mean(per_tissue_fold[t]))
                      if per_tissue_fold[t] else float("nan")),
         "accuracy_std": (float(np.std(per_tissue_fold[t], ddof=1))
                          if len(per_tissue_fold[t]) > 1 else 0.0),
         "n_samples": int((T == t).sum()),
         "n_folds": len(per_tissue_fold[t])}
        for t in labels])
    return {
        "accuracy": acc, "accuracy_std": acc_std,
        "chance": chance, "lift": acc / chance, "lift_std": acc_std / chance,
        "n_classes": n_classes, "n_splits": n_splits,
        "n_samples": int(len(T)), "labels": labels,
        "confusion_matrix": pd.DataFrame(cm, index=labels, columns=labels),
        "per_tissue": per_tissue,
    }


# ── Arm 1 — within- vs across-tissue sample correlations ─────────────────────
def _split_half_partitions(n: int, half: int
                           ) -> Iterator[Tuple[List[int], List[int]]]:
    """Every *distinct* unordered split of ``range(n)`` into two disjoint
    size-``half`` index groups (the ``n - 2*half`` leftover positions are
    dropped, exactly as the random path drops them for odd ``n``).

    Used to enumerate split-halves exhaustively when a tissue is too small for
    ``n_seeds`` random draws to be distinct. ``frozenset((a, b))`` collapses the
    (a, b) ≡ (b, a) symmetry — split-half Pearson r is symmetric in the halves —
    so each partition is yielded once. There are ``C(n,half)·C(n-half,half)/2``
    of them, which is why the caller only invokes this on small tissues.
    """
    seen = set()
    for a in combinations(range(n), half):
        a_set = set(a)
        rest = [i for i in range(n) if i not in a_set]
        for b in combinations(rest, half):
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            yield list(a), list(b)


def within_across_tissue_corr(X: np.ndarray, T: np.ndarray,
                              n_seeds: int = 20, seed: int = 0,
                              min_samples_per_tissue: int = 5
                              ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Within-tissue split-half correlations vs across-tissue mean-profile
    correlations (TxPert Fig 1a analogue, Arm 1).

    Default `min_samples_per_tissue=5` — the unified min-cells-per-category
    floor shared with the classifier, outcome ANOVA and fig05c.

    Within: for each tissue with ≥ `min_samples_per_tissue` samples, draw
    split-halves, mean-aggregate each half, Pearson r between halves. A tissue
    with at least `min_samples_per_tissue` samples has
    `C(n,half)·C(n-half,half)/2` distinct split-halves; when that is ≤ `n_seeds`
    we enumerate *all* of them exhaustively (deterministic, no duplicate or
    missed split), otherwise we take `n_seeds` random split-halves. Either way
    a tissue contributes at most `n_seeds` correlations, so tissues are weighted
    roughly equally in the pooled distribution.

    Across: pairwise Pearson r between per-tissue mean profiles, computed as
    `1 - pdist(means, "correlation")` (correlation distance is `1 - Pearson r`),
    which is identical to the old `np.corrcoef` double loop but vectorised.
    One-sided Mann-Whitney U tests whether across-tissue r is *less* than
    within-tissue r (cells resemble their own tissue more than others).

    Deterministic in `seed`: tissues are processed in sorted order and the
    single `default_rng(seed)` is consumed only by the random path, so the same
    `(X, T, seed)` always yields the same arrays.

    Returns (within_r, across_r, u_stat, p_value).
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).astype(str)
    tissues = sorted(pd.unique(T).tolist())
    rng = np.random.default_rng(seed)

    within = []
    for t in tissues:
        idx = np.where(T == t)[0]
        n = len(idx)
        half = n // 2
        if n < min_samples_per_tissue or half == 0:
            continue
        # Number of distinct unordered size-`half`/size-`half` split-halves.
        n_partitions = comb(n, half) * comb(n - half, half) // 2
        if n_partitions <= n_seeds:
            splits = [(idx[a], idx[b]) for a, b in _split_half_partitions(n, half)]
        else:
            splits = [(perm[:half], perm[half:2 * half])
                      for perm in (rng.permutation(idx) for _ in range(n_seeds))]
        for a_idx, b_idx in splits:
            a = X[a_idx].mean(axis=0)
            b = X[b_idx].mean(axis=0)
            within.append(float(np.corrcoef(a, b)[0, 1]))

    # Across-tissue centroids share the same `min_samples_per_tissue` floor as
    # the within-tissue split-halves above, so the within-vs-across MWU below
    # compares the *same* set of tissues. (If you ever want a broader between-
    # tissue characterisation, lower this independently — e.g. `>= 2` to keep
    # every non-singleton centroid — but then within/across are no longer a
    # matched comparison.)
    means = [X[np.where(T == t)[0]].mean(axis=0)
             for t in tissues if (T == t).sum() >= min_samples_per_tissue]
    if len(means) >= 2:
        across = (1.0 - pdist(np.vstack(means), metric="correlation")).tolist()
    else:
        across = []

    within_arr = np.asarray(within)
    across_arr = np.asarray(across)
    if len(within_arr) and len(across_arr):
        u_stat, p_val = mannwhitneyu(across_arr, within_arr, alternative="less")
    else:
        u_stat, p_val = float("nan"), float("nan")
    return within_arr, across_arr, float(u_stat), float(p_val)


# ── Arm 1 — PCA coloured by tissue ───────────────────────────────────────────
def pca_by_tissue(X: np.ndarray, T: np.ndarray, n_comps: int = 5,
                  center_by_group: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Optionally per-tissue mean-center →) center-only PCA on the raw panel.

    PCA on the gene-expression panel *as-is*: sklearn's PCA mean-centers each
    gene internally but does **not** z-score or clip. This is the bulk-RNA-seq
    convention (cf. `DESeq2::plotPCA`, which runs `prcomp(scale.=FALSE)`), in
    contrast to scanpy's single-cell `sc.pp.scale(max_value=10)` recipe. The
    feature panel is already a restricted gene set, so all genes are used
    (no HVG/top-variance subsetting).
    High-variance genes — which on this data are disproportionately the
    tissue-identity genes — therefore drive the leading PCs, which is exactly
    what this confounding diagnostic is meant to surface.

    With `center_by_group=True` each tissue's gene means are removed first,
    so any residual PC structure is not a first-moment tissue shift — a
    stretch diagnostic.

    Returns (pcs [N × n_comps], variance_ratio [n_comps], tissue_labels [N]).
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).astype(str)

    if center_by_group:
        X = X.copy()
        for t in np.unique(T):
            m = T == t
            X[m] -= X[m].mean(axis=0)

    n_comps = int(min(n_comps, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_comps, random_state=0)
    pcs = pca.fit_transform(X)
    return pcs, pca.explained_variance_ratio_, T
