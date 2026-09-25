# Is the tissue-confounding result specific to Shapley values?

Is the paper's focus on SHAP limiting: would a different attribution method be
more or less affected by tissue confounding, and would the corrections
proposed here work on it? This folder answers both halves empirically.

## What is compared

Five attribution methods spanning three families, plus the paper's own
estimator as an anchor:

| method | family | notes |
|---|---|---|
| `shap` | Shapley | `GradientExplainer`, exactly as in the paper |
| `lrp_eps` | propagation | LRP with the ε-rule |
| `lrp_ab` | propagation | LRP with α=1, β=0 (z⁺ rule); z^B rule at the signed input layer |
| `int_grad` | path | Integrated Gradients, baseline = background mean |
| `lime` | surrogate | tabular LIME, locally weighted ridge surrogate |
| `grad_x_input` | gradient | reference point |

Each is run on the **same models, folds and cells**, under two conditions —
the confounded `marginal` model and the data-stage `residualized` model — and
scored with the paper's own diagnostics (contamination@50, per-gene
attribution η²).

## On implementing LRP honestly

For a ReLU network with zero biases, **LRP-0 is provably identical to
gradient × input** (Ancona et al., ICLR 2018). Reporting gradient × input
under the name "LRP" would therefore be an empty comparison, and it is a
common way to accidentally cheat. Two things guard against it here:

* **ε is scaled to each layer's activation magnitude.** With a fixed tiny ε
  the rule degenerates toward gradient × input as ε shrinks. The default is
  relative (ε = 0.25·mean|z|), which stays meaningfully different while still
  correlated — the expected, honest behaviour.
* **The α1β0 rule is structurally not a gradient.** It propagates only
  positive pre-activation contributions and cannot be written as
  gradient-times-anything, and is not correlated with gradient × input in
  practice.

`xai_methods.check_conservation` verifies the defining property of LRP —
that relevance is conserved as it is redistributed backwards. The α1β0
implementation passes this check. The check caught a real sign error in the
z^B input-layer rule during development.

Integrated Gradients is checked against its completeness axiom
(Σφ = f(x) − f(baseline)), and LIME against a linear model whose true
attributions are known in closed form. The LIME check caught a second real
error: multiplying the surrogate coefficient by (x − μ) double-counts, since
the coefficient already measures the effect of holding a feature at its
observed value rather than resampling it.

## Running it

```bash
python compute_xai_comparison.py --dataset depmap --n-items 25
python compute_xai_comparison.py --dataset ctrpv2 --n-items 25
```

All cell lines of each item are used, so η² is estimated on the same sample
as Figure 2 and the numbers are directly comparable to it. LIME costs one
model evaluation per perturbation per cell line, which dominates the runtime,
so the run is parallelised across items with `--nproc`. `--n-cell-lines`
subsamples for smoke tests only and should not be used for reported numbers:
with fewer cell lines per tissue, between-tissue variance is estimated with
more noise and η² is biased upward.

Once both datasets have run, `python make_table_other_explainers.py` pools
`results/xai_comparison_combined.csv` over both datasets and writes
`other_explainers_table.tex`, the rows behind Table `tab:other_explainers`.
