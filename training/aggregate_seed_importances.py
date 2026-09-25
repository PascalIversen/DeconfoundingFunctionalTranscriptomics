"""Seed-average the per-gene global importances into <dataset>_seedavg_preds/.

For each item, importance a_g = mean_cells |SHAP_g| is computed per seed from the
full per-cell SHAP in <root>/<dataset>_seed<s>_preds/<item>.npz, then averaged
over the available seeds. Produces the compact `imp_<method>` vectors that
fig04a (self/partner recovery) consumes -- no per-cell SHAP, one value per gene.

    python aggregate_seed_importances.py --root ../results --dataset depmap

Needs the FULL per-cell seed preds (with shap_<method>). The bundled predictions
ship seed 1 with SHAP and seeds 2-5 as yhat-only (lean), so regenerating the
5-seed average requires the full 5-seed preds (retrain, or fetch from the source).
The seed-averaged output itself is provided in results/<dataset>_seedavg_preds/.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.preds_io import load_preds, extra_dir_for  # noqa: E402

METHODS = ["marginal", "residualized", "irm", "within_tissue"]
ADVERSARIAL_METHODS = ["dann", "adae"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    ap.add_argument("--dataset", default="depmap")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma-separated methods to aggregate")
    ap.add_argument("--suffix", default="seedavg_preds",
                    help="output dir is {dataset}_{suffix}. Use "
                         "'seedavg_extra' for the adversarial baselines so "
                         "the published seedavg bundle is not overwritten; "
                         "shared.preds_io merges the two transparently.")
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    out = a.out or a.root / f"{a.dataset}_{a.suffix}"
    out.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(a.root.glob(f"{a.dataset}_seed[0-9]_preds"))
    # For the adversarial baselines the per-seed data may live only in the
    # *_extra dirs (seeds 2-5 ship per-gene importances, not per-cell SHAP).
    for d in sorted(a.root.glob(f"{a.dataset}_seed[0-9]_extra")):
        base = d.parent / d.name.replace("_extra", "_preds")
        if base not in seed_dirs:
            seed_dirs.append(base if base.exists() else d)
    seed_dirs = sorted(set(seed_dirs))
    items = sorted({os.path.basename(f)[:-4]
                    for d in seed_dirs for f in glob.glob(f"{d}/*.npz")})
    print(f"{a.dataset}: {len(seed_dirs)} seed dirs, {len(items)} items")
    for item in items:
        acc = {m: [] for m in methods}
        gene_cols = tissue = None
        for d in seed_dirs:
            p = d / f"{item}.npz"
            if not p.exists():
                # base bundle may lack this item while the extras have it
                alt = extra_dir_for(d) / f"{item}.npz"
                if not alt.exists():
                    continue
                p = alt
            z = load_preds(p)
            if gene_cols is None:
                gene_cols, tissue = z["gene_cols"], z["tissue"]
            for m in methods:
                # Per-cell SHAP if we have it; otherwise the pre-reduced
                # per-gene importances written by gen_preds_extra --save-imp.
                if f"shap_{m}" in z.files:
                    with np.errstate(invalid="ignore"):
                        acc[m].append(np.nanmean(np.abs(z[f"shap_{m}"].astype(np.float32)), axis=0))
                elif f"imp_{m}" in z.files:
                    acc[m].append(z[f"imp_{m}"].astype(np.float32))
        if gene_cols is None or not any(acc[m] for m in methods):
            continue
        payload = {"gene_cols": gene_cols, "tissue": tissue,
                   "n_seeds": np.array([max(len(acc[m]) for m in methods)],
                                       dtype=np.int32)}
        for m in methods:
            if acc[m]:
                payload[f"imp_{m}"] = np.mean(acc[m], axis=0).astype(np.float32)
                payload[f"nseed_{m}"] = np.array([len(acc[m])], dtype=np.int32)
        np.savez(out / f"{item}.npz", **payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
