"""Verification for the DANN / AD-AE implementations.

Runs without `shap` (uses compute_shap=False for the run_cv checks).
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn

from shared.models import (SmallMLP, DANNModel, ADAE, ADAEPredictor,
                           grad_reverse, dann_lambda)
from shared.train import (train_marginal, train_dann, fit_adae_embedding,
                          train_adae_head)
from shared import adae_cache

FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- fixtures
def make_data(n=400, p=60, k=6, seed=0):
    """Expression with strong tissue structure + a phenotype that depends
    on both a tissue-marker gene and a within-tissue causal gene."""
    rng = np.random.RandomState(seed)
    T = np.array([f"tissue{i % k}" for i in range(n)])
    tid = np.array([int(t[-1]) for t in T])
    X = rng.randn(n, p).astype(np.float32)
    # genes 0..4 are pure tissue markers
    for g in range(5):
        X[:, g] += 3.0 * (tid == (g % k))
    # gene 10 is causal within tissue
    y = (1.5 * X[:, 10] + 0.8 * X[:, 0] + 0.3 * rng.randn(n)).astype(np.float32)
    return X, y, T


X, y, T = make_data()
Xtr, ytr, Ttr = X[:300], y[:300], T[:300]
Xva, yva, Tva = X[300:], y[300:], T[300:]
dev = torch.device("cpu")

print("\n=== 1. Gradient reversal layer ===")
for lam in (0.0, 1.0, 7.5):
    x = torch.randn(4, 3, requires_grad=True)
    out = grad_reverse(x, lam)
    check(f"GRL forward is identity (lambda={lam})",
          torch.allclose(out, x))
    out.backward(torch.ones_like(out))
    expect = -lam * torch.ones_like(x)
    check(f"GRL backward = -lambda*I (lambda={lam})",
          torch.allclose(x.grad, expect), f"got {x.grad[0,0].item():.3f}")

print("\n=== 2. Ganin lambda schedule ===")
check("lambda(0) == 0", abs(dann_lambda(0.0)) < 1e-12)
check("lambda(1) ~ 0.9999", abs(dann_lambda(1.0) - 0.99990920) < 1e-6,
      f"{dann_lambda(1.0):.8f}")
check("lambda(0.3) ~ 0.905 (ramp is ~90% done at p=0.3)",
      abs(dann_lambda(0.3) - 0.90514825) < 1e-6, f"{dann_lambda(0.3):.6f}")
check("lambda_max scales the ceiling",
      abs(dann_lambda(1.0, lambda_max=10.0) - 10 * dann_lambda(1.0)) < 1e-9)

print("\n=== 3. DANN backbone matches SmallMLP at init ===")
torch.manual_seed(123); a = SmallMLP(60)
torch.manual_seed(123); b = DANNModel(60, 6)
same = (torch.allclose(a.net[0].weight, b.features[0].weight)
        and torch.allclose(a.net[3].weight, b.features[3].weight)
        and torch.allclose(a.net[6].weight, b.label_head.weight))
check("SmallMLP and DANNModel share identical initial backbone weights", same)

print("\n=== 4. DANN at lambda_max=0 reproduces the marginal model ===")
torch.manual_seed(7); np.random.seed(7)
m_marg, v_marg = train_marginal(Xtr, ytr, Xva, yva, epochs=25, patience=25,
                                device=dev)
torch.manual_seed(7); np.random.seed(7)
m_dann0, v_dann0, dg0 = train_dann(Xtr, ytr, Ttr, Xva, yva, Tva, epochs=25,
                                   patience=25, lambda_max=0.0, device=dev)
m_marg.eval(); m_dann0.eval()
with torch.no_grad():
    pm = m_marg(torch.from_numpy(Xva)).squeeze(-1).numpy()
    pd_ = m_dann0(torch.from_numpy(Xva)).squeeze(-1).numpy()
md = float(np.max(np.abs(pm - pd_)))
check("predictions identical to marginal at lambda=0", md < 1e-5,
      f"max|diff|={md:.2e}")
check("val MSE identical at lambda=0", abs(v_marg - v_dann0) < 1e-6,
      f"{v_marg:.6f} vs {v_dann0:.6f}")

print("\n=== 5. DANN adversary actually suppresses tissue ===")
torch.manual_seed(7); np.random.seed(7)
m_d1, v_d1, dg1 = train_dann(Xtr, ytr, Ttr, Xva, yva, Tva, epochs=60,
                             patience=60, lambda_max=1.0, device=dev)
torch.manual_seed(7); np.random.seed(7)
m_d10, v_d10, dg10 = train_dann(Xtr, ytr, Ttr, Xva, yva, Tva, epochs=60,
                                patience=60, lambda_max=10.0, device=dev)
print(f"    domain_acc_train: lambda=0 {dg0['domain_acc_train']:.3f} | "
      f"lambda=1 {dg1['domain_acc_train']:.3f} | "
      f"lambda=10 {dg10['domain_acc_train']:.3f}  "
      f"(majority baseline {dg1['majority_domain_frac']:.3f})")
# NB: accuracy is deliberately NOT asserted to be monotone in lambda.
# Adversarial training is unstable in the penalty weight -- that instability
# is a property to report, not a bug to assert away. What must hold is that
# any adversarial pressure collapses tissue decodability toward the majority
# baseline relative to lambda=0.
check("lambda=1 collapses domain accuracy vs lambda=0",
      dg1["domain_acc_train"] < dg0["domain_acc_train"])
check("lambda=10 collapses domain accuracy vs lambda=0",
      dg10["domain_acc_train"] < dg0["domain_acc_train"])
check("both land near the majority baseline",
      max(dg1["domain_acc_train"], dg10["domain_acc_train"])
      < dg1["majority_domain_frac"] + 0.20)

print("\n=== 6. AD-AE structure ===")
torch.manual_seed(3)
ad = ADAE(60, 6, hidden=500, emb_dim=100, dropout=0.1)
check("encoder is Linear-ReLU-Dropout-Linear (linear embedding out)",
      isinstance(ad.encoder[0], nn.Linear) and isinstance(ad.encoder[1], nn.ReLU)
      and isinstance(ad.encoder[2], nn.Dropout)
      and isinstance(ad.encoder[3], nn.Linear))
check("encoder hidden width 500, embedding 100",
      ad.encoder[0].out_features == 500 and ad.encoder[3].out_features == 100)
check("decoder mirrors encoder back to in_dim",
      ad.decoder[0].in_features == 100 and ad.decoder[3].out_features == 60)
check("adversary has 2 hidden layers sized to the embedding",
      ad.adversary[0].out_features == 100 and ad.adversary[2].out_features == 100
      and ad.adversary[4].out_features == 6)
check("ae_parameters excludes the adversary",
      not any(p is q for p in ad.ae_parameters()
              for q in ad.adversary_parameters()))

print("\n=== 7. AD-AE embedding is phenotype-free ===")
import inspect
sig = inspect.signature(fit_adae_embedding)
check("fit_adae_embedding takes no y argument",
      not any(n in sig.parameters for n in ("y", "y_tr", "y_train")),
      f"params={list(sig.parameters)[:4]}")

print("\n=== 8. AD-AE adversarial term does something ===")
torch.manual_seed(11); np.random.seed(11)
ae0, d0 = fit_adae_embedding(Xtr, Ttr, Xva, Tva, lambda_adv=0.0,
                             ae_pretrain_epochs=40, adv_pretrain_epochs=25,
                             joint_rounds=40, patience=10, device=dev)
torch.manual_seed(11); np.random.seed(11)
ae1, d1 = fit_adae_embedding(Xtr, Ttr, Xva, Tva, lambda_adv=1.0,
                             ae_pretrain_epochs=40, adv_pretrain_epochs=25,
                             joint_rounds=40, patience=10, device=dev)
torch.manual_seed(11); np.random.seed(11)
ae5, d5 = fit_adae_embedding(Xtr, Ttr, Xva, Tva, lambda_adv=5.0,
                             ae_pretrain_epochs=40, adv_pretrain_epochs=25,
                             joint_rounds=40, patience=10, device=dev)
print(f"    adv_acc_train: lambda=0 {d0['adv_acc_train']:.3f} | "
      f"lambda=1 {d1['adv_acc_train']:.3f} | lambda=5 {d5['adv_acc_train']:.3f}"
      f"  (majority {d0['majority_class_frac']:.3f})")
print(f"    recon_val:     lambda=0 {d0['recon_val_final']:.4f} | "
      f"lambda=1 {d1['recon_val_final']:.4f} | lambda=5 {d5['recon_val_final']:.4f}")
check("lambda=0 keeps tissue decodable from the embedding",
      d0["adv_acc_train"] > 0.5, f"{d0['adv_acc_train']:.3f}")
check("adversarial training lowers tissue decodability (lambda=1)",
      d1["adv_acc_train"] < d0["adv_acc_train"])
check("adversarial training lowers tissue decodability (lambda=5)",
      d5["adv_acc_train"] < d0["adv_acc_train"])
check("reconstruction degrades as lambda grows (the trade-off)",
      d5["recon_val_final"] >= d0["recon_val_final"] - 1e-6)

print("\n=== 9. AD-AE predictor / head ===")
torch.manual_seed(5); np.random.seed(5)
pred, vh, _hd = train_adae_head(ae1, Xtr, ytr, Xva, yva, epochs=40, patience=15,
                           device=dev)
check("ADAEPredictor output shape is (n,1) for GradientExplainer",
      pred(torch.from_numpy(Xva)).shape == (len(Xva), 1))
check("encoder frozen (no grad)",
      not any(p.requires_grad for p in pred.encoder.parameters()))
check("head trainable",
      all(p.requires_grad for p in pred.head_parameters()))
pred.train()
check("train() keeps the encoder in eval mode", not pred.encoder.training)
check("train() puts the head in train mode", pred.head.training)
# gradients must still reach the input, or GradientExplainer would return 0
xin = torch.from_numpy(Xva[:8]).clone().requires_grad_(True)
pred.eval(); pred(xin).sum().backward()
check("gradients flow through the frozen encoder to the inputs",
      xin.grad is not None and float(xin.grad.abs().sum()) > 0)

print("\n=== 10. AD-AE encoder cache ===")
adae_cache.clear_memory_cache()
key = adae_cache.make_key([f"c{i}" for i in range(300)], seed=1, fold=0,
                          lambda_adv=1.0, hidden=500, emb_dim=100,
                          ae_dropout=0.1, batch_size=128, lr=1e-3,
                          ae_pretrain_epochs=10, adv_pretrain_epochs=5,
                          joint_rounds=10, patience=5, in_dim=60,
                          val_frac=0.15, panel=[f'g{i}' for i in range(60)])
key2 = adae_cache.make_key([f"c{i}" for i in range(300)], seed=1, fold=1,
                           lambda_adv=1.0, hidden=500, emb_dim=100,
                           ae_dropout=0.1, batch_size=128, lr=1e-3,
                           ae_pretrain_epochs=10, adv_pretrain_epochs=5,
                           joint_rounds=10, patience=5, in_dim=60,
                           val_frac=0.15, panel=[f'g{i}' for i in range(60)])
check("different folds give different keys", key != key2)
key3 = adae_cache.make_key([f"c{i}" for i in range(299, -1, -1)], seed=1,
                           fold=0, lambda_adv=1.0, hidden=500, emb_dim=100,
                           ae_dropout=0.1, batch_size=128, lr=1e-3,
                           ae_pretrain_epochs=10, adv_pretrain_epochs=5,
                           joint_rounds=10, patience=5, in_dim=60,
                           val_frac=0.15, panel=[f'g{i}' for i in range(60)])
check("cell ORDER is part of the key (splits are positional)", key != key3)

calls = {"n": 0}
def _fit():
    calls["n"] += 1
    return fit_adae_embedding(Xtr, Ttr, Xva, Tva, lambda_adv=1.0,
                              ae_pretrain_epochs=10, adv_pretrain_epochs=5,
                              joint_rounds=10, patience=5, device=dev)

import tempfile, pathlib
tmpd = pathlib.Path(tempfile.mkdtemp())
s1, g1 = adae_cache.get_or_fit(key, _fit, cache_dir=tmpd)
s2, g2 = adae_cache.get_or_fit(key, _fit, cache_dir=tmpd)
check("memory cache prevents a second fit", calls["n"] == 1, f"fits={calls['n']}")
adae_cache.clear_memory_cache()
s3, g3 = adae_cache.get_or_fit(key, _fit, cache_dir=tmpd)
check("disk cache prevents a refit after memory is cleared",
      calls["n"] == 1, f"fits={calls['n']}")
check("disk-loaded weights identical to freshly fitted",
      all(torch.allclose(s1[k], s3[k]) for k in s1))

# determinism: same key, cold cache, different call order -> same weights
adae_cache.clear_memory_cache()
calls["n"] = 0
s4, _ = adae_cache.get_or_fit(key, _fit, cache_dir=None)
adae_cache.clear_memory_cache()
s5, _ = adae_cache.get_or_fit(key, _fit, cache_dir=None)
check("refit from the same key is bit-identical (seeded from key)",
      all(torch.allclose(s4[k], s5[k]) for k in s4), f"fits={calls['n']}")

print("\n=== 11. Folds are identical regardless of which methods run ===")
from shared.orchestrator import run_cv, _split_train_val
from sklearn.model_selection import KFold
names = [f"g{i}" for i in range(X.shape[1])]
ids = [f"cell{i}" for i in range(len(X))]


def fold_signature(nn_, seed_):
    """The exact (train, val, test) partition run_cv will use."""
    sig = []
    for fi, (tr_full, te) in enumerate(KFold(n_splits=5, shuffle=True,
                                             random_state=seed_).split(
                                                 np.zeros((nn_, 1)))):
        tr, va = _split_train_val(len(tr_full), 0.15, seed=seed_ * 1000 + fi)
        sig.append((te.tolist(), tr_full[tr].tolist(), tr_full[va].tolist()))
    return sig


s_a = fold_signature(len(X), 1)
# Burn a lot of randomness from every global stream, then recompute.
torch.manual_seed(999); np.random.seed(999)
_ = torch.randn(5000); _ = np.random.randn(5000)
s_b = fold_signature(len(X), 1)
check("CV partition is independent of the global torch/numpy RNG state",
      s_a == s_b)
check("partition covers every cell exactly once per fold",
      all(sorted(te + tr + va) == list(range(len(X))) for te, tr, va in s_a))

# This is the property the extras runner relies on: the same test cells.
part = run_cv(X, y, T, names, n_folds=3, seed=1, epochs=12, patience=12,
              methods=["marginal"], compute_shap=False, device=dev)
check("a single-method run still fills predictions for every cell",
      np.all(np.isfinite(part["marginal"]["yhat_raw"])))

print("\n=== 11b. New methods are invariant to which others run ===")
adae_cache.clear_memory_cache()
kw = dict(n_folds=3, seed=1, epochs=15, patience=15, compute_shap=False,
          device=dev, sample_ids=ids,
          adae_kwargs=dict(ae_pretrain_epochs=15, adv_pretrain_epochs=10,
                           joint_rounds=15))
both = run_cv(X, y, T, names, methods=["dann", "adae"], **kw)
adae_cache.clear_memory_cache()
solo_d = run_cv(X, y, T, names, methods=["dann"], **kw)
adae_cache.clear_memory_cache()
solo_a = run_cv(X, y, T, names, methods=["adae"], **kw)
dd = float(np.max(np.abs(both["dann"]["yhat_raw"] - solo_d["dann"]["yhat_raw"])))
da = float(np.max(np.abs(both["adae"]["yhat_raw"] - solo_a["adae"]["yhat_raw"])))
check("DANN identical whether or not AD-AE runs alongside", dd < 1e-6,
      f"max|d|={dd:.2e}")
check("AD-AE identical whether or not DANN runs alongside", da < 1e-6,
      f"max|d|={da:.2e}")

print("\n=== 12. run_cv with the adversarial methods ===")
adae_cache.clear_memory_cache()
adv = run_cv(X, y, T, names, n_folds=3, seed=1, epochs=15, patience=15,
             methods=["dann", "adae"], compute_shap=False, device=dev,
             sample_ids=ids,
             adae_kwargs=dict(ae_pretrain_epochs=15, adv_pretrain_epochs=10,
                              joint_rounds=15))
check("dann present with finite predictions",
      "dann" in adv and np.all(np.isfinite(adv["dann"]["yhat_raw"])))
check("adae present with finite predictions",
      "adae" in adv and np.all(np.isfinite(adv["adae"]["yhat_raw"])))
check("marginal NOT computed when not requested", "marginal" not in adv)
check("per-fold diagnostics recorded for both",
      len(adv["dann"]["diagnostics"]) == 3
      and len(adv["adae"]["diagnostics"]) == 3)
print(f"    dann  r={adv['dann']['pearson_r']:.3f}  "
      f"adae r={adv['adae']['pearson_r']:.3f}")
check("both recover signal (r > 0.3 on this easy synthetic task)",
      adv["dann"]["pearson_r"] > 0.3 and adv["adae"]["pearson_r"] > 0.3)

print("\n=== 13. AD-AE encoder reused across different phenotypes ===")
adae_cache.clear_memory_cache()
y2 = (y * -2.0 + 5.0).astype(np.float32)   # totally different phenotype
r1 = run_cv(X, y, T, names, n_folds=3, seed=1, epochs=15, patience=15,
            methods=["adae"], compute_shap=False, device=dev, sample_ids=ids,
            adae_kwargs=dict(ae_pretrain_epochs=15, adv_pretrain_epochs=10,
                             joint_rounds=15))
n_after_first = adae_cache.memory_cache_size()
r2 = run_cv(X, y2, T, names, n_folds=3, seed=1, epochs=15, patience=15,
            methods=["adae"], compute_shap=False, device=dev, sample_ids=ids,
            adae_kwargs=dict(ae_pretrain_epochs=15, adv_pretrain_epochs=10,
                             joint_rounds=15))
check("cache holds one encoder per fold", n_after_first == 3,
      f"entries={n_after_first}")
check("a different phenotype reuses the same encoders (no new entries)",
      adae_cache.memory_cache_size() == 3,
      f"entries={adae_cache.memory_cache_size()}")

print("\n" + "=" * 60)
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("ALL CHECKS PASSED")
