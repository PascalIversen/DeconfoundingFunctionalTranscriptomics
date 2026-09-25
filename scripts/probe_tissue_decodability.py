"""How much tissue can a *fresh* classifier recover from each method's attributions?

This replaces a circular diagnostic. Reporting a method's own adversary (or
domain classifier) as evidence that tissue was removed has two defects:

  1. The adversary is co-adapted with the encoder it was trained against. It
     can be beaten while the information is still trivially recoverable by any
     other classifier.
  2. For AD-AE the reported statistic was worse than uninformative: the
     checkpoint is *selected* by minimising |val adversary CE - chance CE|, so
     reporting that same quantity as an equilibrium check cannot fail.

The honest question for this paper is narrower and directly testable: after
deconfounding, can tissue still be decoded from the per-cell SHAP
attributions? A fresh probe, never trained jointly with anything, answers it.

Protocol, per (dataset, method, item):
  * take the per-cell attribution matrix (n_cells x n_genes),
  * 5-fold stratified CV over cells; within each training split only, fit PCA
    to `--n-pcs` components and a multinomial logistic probe,
  * score held-out cells: accuracy and balanced accuracy.

Everything the probe sees is fitted inside its own training split, so the
score is an honest out-of-sample decodability estimate. The reference is the
majority-tissue rate, which a probe gets for free.

    python scripts/probe_tissue_decodability.py --dataset depmap --n-items 40
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from shared.preds_io import load_preds  # noqa: E402

METHODS = ["marginal", "residualized", "irm", "dann", "adae", "within_tissue"]


def probe_item(phi: np.ndarray, tissue: np.ndarray, n_pcs: int, seed: int):
    """Out-of-sample tissue decodability from one attribution matrix."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    ok = np.isfinite(phi).all(axis=1)
    # A degenerate (all-zero) attribution row carries no information and is not
    # an explanation; excluding it keeps the probe from scoring padding.
    ok &= ~(phi == 0).all(axis=1)
    phi, tissue = phi[ok], tissue[ok]
    if len(phi) < 50:
        return None
    counts = pd.Series(tissue).value_counts()
    keep = np.isin(tissue, counts[counts >= 5].index)      # CV needs >=5 per class
    phi, tissue = phi[keep], tissue[keep]
    if len(phi) < 50 or len(np.unique(tissue)) < 2:
        return None

    accs, baccs = [], []
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(phi, tissue):
        sc = StandardScaler().fit(phi[tr])
        pca = PCA(n_components=min(n_pcs, len(tr) - 1, phi.shape[1]),
                  random_state=seed).fit(sc.transform(phi[tr]))
        Ztr, Zte = pca.transform(sc.transform(phi[tr])), pca.transform(sc.transform(phi[te]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                     random_state=seed).fit(Ztr, tissue[tr])
        pred = clf.predict(Zte)
        accs.append(float((pred == tissue[te]).mean()))
        baccs.append(float(balanced_accuracy_score(tissue[te], pred)))
    maj = float(pd.Series(tissue).value_counts(normalize=True).max())
    return dict(acc=float(np.mean(accs)), bacc=float(np.mean(baccs)),
                majority=maj, n_cells=len(phi),
                n_tissues=int(len(np.unique(tissue))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["depmap", "ctrpv2"], required=True)
    ap.add_argument("--preds-root", type=Path, default=ROOT / "results")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-items", type=int, default=40,
                    help="evenly spaced subset; the probe is slow per item")
    ap.add_argument("--n-pcs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    d = a.preds_root / f"{a.dataset}_seed{a.seed}_preds"
    files = sorted(d.glob("*.npz"))
    if not files:
        sys.exit(f"no predictions under {d}")
    idx = np.linspace(0, len(files) - 1, min(a.n_items, len(files))).round().astype(int)
    files = [files[i] for i in sorted(set(idx))]
    print(f"{a.dataset}: probing {len(files)} items, {a.n_pcs} PCs", flush=True)

    rows = []
    for j, f in enumerate(files, 1):
        z = load_preds(f)
        tissue = z["tissue"]
        for m in METHODS:
            key = f"shap_{m}"
            if key not in z.files:
                continue
            r = probe_item(z[key].astype(np.float32), tissue, a.n_pcs, a.seed)
            if r:
                rows.append(dict(dataset=a.dataset, item=f.stem, method=m, **r))
        if j % 10 == 0:
            print(f"  {j}/{len(files)}", flush=True)

    df = pd.DataFrame(rows)
    out = a.out or ROOT / "results" / f"tissue_probe_{a.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n=== tissue decodability from the attributions ({a.dataset}) ===")
    g = df.groupby("method")[["acc", "bacc", "majority"]].mean()
    g["acc/majority"] = g.acc / g.majority
    print(g.reindex([m for m in METHODS if m in g.index]).round(4).to_string())

    # paired vs marginal, same items
    from scipy import stats
    base = df[df.method == "marginal"].set_index("item")["acc"]
    print("\npaired vs marginal (Wilcoxon):")
    for m in METHODS:
        if m == "marginal":
            continue
        s = df[df.method == m].set_index("item")["acc"]
        common = base.index.intersection(s.index)
        if len(common) < 5:
            continue
        p = stats.wilcoxon(s.loc[common], base.loc[common]).pvalue
        print(f"  {m:14s} delta {s.loc[common].mean()-base.loc[common].mean():+.4f}  p={p:.3g}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
