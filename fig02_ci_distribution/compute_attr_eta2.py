"""Compute tissue-eta^2 of the per-cell SHAP attributions.

For each (item, gene, method) we measure the fraction of the *attribution*
variance that is explained by tissue:

    eta2 = SS_between_tissue / SS_total   (one-way ANOVA on the per-cell phi)

Unlike the magnitude-based CI (1 - mean|phi_method|/mean|phi_marginal|), this
is defined identically for every method -- they all emit per-cell phi -- so it
puts marginal, residualized, IRM and within-tissue on one comparable axis.
Lower eta2 = less tissue signal left in the attribution = better deconfounding.

Reads <preds-root>/<dataset>_seed<seed>_preds/*.npz (produced by
../training/run_{depmap,ctrpv2}.py), writes <out-dir>/<dataset>_attr_eta2.csv.

Usage (from fig02_ci_distribution/):
    python compute_attr_eta2.py
    python compute_attr_eta2.py --preds-root ../training/results
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["marginal", "residualized", "irm", "within_tissue",
           "dann", "adae"]
HERE = Path(__file__).resolve().parent

# Adversarial baselines (DANN / AD-AE) live in a sibling "*_extra" directory so
# the published prediction bundle stays untouched; load_preds merges them in.
import sys as _sys
_sys.path.insert(0, str(HERE.parent / "training"))
from shared.preds_io import load_preds  # noqa: E402


def tissue_eta2(phi: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    """One-way ANOVA eta^2 per gene of the per-cell attribution `phi`
    (n_cells, n_genes) against `tissue` (n_cells). Cells with any NaN are
    dropped (within-tissue leaves NaN for sub-threshold tissues)."""
    valid = np.isfinite(phi).all(axis=1)
    phi, tissue = phi[valid], tissue[valid]
    grand = phi.mean(0)
    ss_tot = ((phi - grand) ** 2).sum(0)
    ss_btw = np.zeros(phi.shape[1])
    for t in np.unique(tissue):
        m = tissue == t
        ss_btw += int(m.sum()) * (phi[m].mean(0) - grand) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 1e-12, ss_btw / ss_tot, np.nan)


def per_item_eta2(preds_dir: Path, item_label: str) -> pd.DataFrame:
    rows = []
    for f in sorted(preds_dir.glob("*.npz")):
        d = load_preds(f)
        tissue = d["tissue"]
        genes = d["gene_cols"]
        for m in METHODS:
            key = f"shap_{m}"
            if key not in d.files:
                continue
            e = tissue_eta2(d[key].astype(np.float32), tissue)
            for g, v in zip(genes, e):
                rows.append((f.stem, str(g), m, float(v)))
    return pd.DataFrame(rows, columns=[item_label, "gene", "method", "eta2"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds-root", type=Path,
                   default=HERE.parent / "results",
                   help="Directory containing depmap_seed<seed>_preds/ and "
                        "ctrpv2_seed<seed>_preds/. Defaults to "
                        "../results/.")
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    p.add_argument("--datasets", nargs="+", default=["depmap", "ctrpv2"])
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    for ds in args.datasets:
        label = "target" if ds == "depmap" else "drug"
        preds = args.preds_root / f"{ds}_seed{args.seed}_preds"
        if not preds.exists():
            print(f"skip {ds}: {preds} missing")
            continue
        df = per_item_eta2(preds, label)
        out = args.out_dir / f"{ds}_attr_eta2.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
