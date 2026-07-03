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

METHODS = ["marginal", "residualized", "irm", "within_tissue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    ap.add_argument("--dataset", default="depmap")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.root / f"{a.dataset}_seedavg_preds"
    out.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(a.root.glob(f"{a.dataset}_seed[0-9]_preds"))
    items = sorted({os.path.basename(f)[:-4]
                    for d in seed_dirs for f in glob.glob(f"{d}/*.npz")})
    print(f"{a.dataset}: {len(seed_dirs)} seed dirs, {len(items)} items")
    for item in items:
        acc = {m: [] for m in METHODS}
        gene_cols = tissue = None
        for d in seed_dirs:
            p = d / f"{item}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            if gene_cols is None:
                gene_cols, tissue = z["gene_cols"], z["tissue"]
            for m in METHODS:
                if f"shap_{m}" in z.files:
                    with np.errstate(invalid="ignore"):
                        acc[m].append(np.nanmean(np.abs(z[f"shap_{m}"].astype(np.float32)), axis=0))
        if gene_cols is None or not acc["marginal"]:
            continue
        payload = {"gene_cols": gene_cols, "tissue": tissue,
                   "n_seeds": np.array([len(acc["marginal"])], dtype=np.int32)}
        for m in METHODS:
            if acc[m]:
                payload[f"imp_{m}"] = np.mean(acc[m], axis=0).astype(np.float32)
                payload[f"nseed_{m}"] = np.array([len(acc[m])], dtype=np.int32)
        np.savez(out / f"{item}.npz", **payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
