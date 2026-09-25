"""Split-half stability of the averaged attribution — the denoising experiment.

The Discussion claims that averaging per-cell attributions "can denoise the
local explanations into a more stable global explanation". This script tests
that claim directly from the shipped per-cell SHAP matrices.

For every (dataset, item, method) at seed 1:

  1. Permute the cells, split into two disjoint halves A and B (R repeats).
  2. For a grid of subsample sizes n, average |SHAP| over the first n cells of
     each half and correlate the two mean-importance vectors over genes
     (Pearson and Spearman). stability(n) rising with n *is* the denoising;
     whether it follows the Spearman-Brown prophecy r_n = n r1 / (1+(n-1) r1)
     tests whether cell-to-cell variation behaves as independent noise.
  3. Keep the two half-mean vectors at the maximum n (repeat 0) and the
     all-cells mean vector, so every downstream global quantity (delta
     percentile vector, contamination@K, AUROC-tau) can be recomputed per
     half without re-reading the 5.6 GB of npz.

Writes (to denoising/results/):
    stability_curves.csv        dataset, item, method, n_cells, rep, n, pearson, spearman
    halfmeans_{dataset}.npz     genes, items, n_cells, and per method the
                                item x gene matrices  full_{m}, halfA_{m}, halfB_{m}
                                (NaN rows where a method is absent for an item)

Nothing here writes outside denoising/.
"""
from __future__ import annotations

import argparse
import glob
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "training"))
from shared.preds_io import load_preds  # noqa: E402

METHODS = ["marginal", "residualized", "irm", "within_tissue", "dann", "adae",
           "between_tissue"]
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_REPS = 10


def _corrs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson and Spearman between two gene-importance vectors."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, np.nan
    a, b = a[ok], b[ok]
    pear = np.corrcoef(a, b)[0, 1]
    ra, rb = rankdata(a), rankdata(b)
    spear = np.corrcoef(ra, rb)[0, 1]
    return float(pear), float(spear)


def process_item(f: str, ds: str, rows: list, keep: dict) -> tuple[np.ndarray, int]:
    d = load_preds(f)
    item = Path(f).stem
    genes = d["gene_cols"]
    n_cells = len(d["y_true"])
    half = n_cells // 2
    grid = [n for n in N_GRID if n <= half]
    if grid[-1] != half:
        grid.append(half)
    rng = np.random.default_rng(zlib.crc32(f"{ds}:{item}".encode()))
    for m in METHODS:
        key = f"shap_{m}"
        if key not in d.files:
            continue
        A = np.abs(d[key].astype(np.float32))          # (n_cells, n_genes)
        keep.setdefault(m, {})[item] = np.nanmean(A, axis=0)
        fin = np.isfinite(A)
        Az = np.where(fin, A, 0.0)
        for rep in range(N_REPS):
            perm = rng.permutation(n_cells)
            ia, ib = perm[:half], perm[half:2 * half]
            # prefix nan-means along the permuted cell axis, all n at once
            csa, cna = Az[ia].cumsum(0), fin[ia].cumsum(0)
            csb, cnb = Az[ib].cumsum(0), fin[ib].cumsum(0)
            for n in grid:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ma = csa[n - 1] / cna[n - 1]
                    mb = csb[n - 1] / cnb[n - 1]
                pear, spear = _corrs(ma, mb)
                rows.append((ds, item, m, n_cells, rep, n, pear, spear))
                if rep == 0 and n == half:
                    keep[m][item + "\x00A"] = ma
                    keep[m][item + "\x00B"] = mb
    return np.array([str(g) for g in genes]), n_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", type=Path, default=ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--datasets", nargs="+", default=["depmap", "ctrpv2"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for ds in a.datasets:
        fs = sorted(glob.glob(f"{a.preds_root}/{ds}_seed{a.seed}_preds/*.npz"))
        if not fs:
            print(f"skip {ds}: no preds")
            continue
        rows: list = []
        keep: dict = {}
        genes, items, ncs = None, [], []
        for i, f in enumerate(fs):
            g, nc = process_item(f, ds, rows, keep)
            genes = g if genes is None else genes
            items.append(Path(f).stem)
            ncs.append(nc)
            if (i + 1) % 25 == 0:
                print(f"  {ds}: {i + 1}/{len(fs)} items", flush=True)
        all_rows += rows

        P = len(genes)
        out = {"genes": genes, "items": np.array(items),
               "n_cells": np.array(ncs)}
        for m in METHODS:
            if m not in keep:
                continue
            for tag, suff in [("full", ""), ("halfA", "\x00A"), ("halfB", "\x00B")]:
                M = np.full((len(items), P), np.nan, dtype=np.float32)
                for j, it in enumerate(items):
                    v = keep[m].get(it + suff)
                    if v is not None:
                        M[j] = v
                out[f"{tag}_{m}"] = M
        np.savez_compressed(a.out_dir / f"halfmeans_{ds}.npz", **out)
        print(f"{ds}: wrote halfmeans_{ds}.npz "
              f"({len(items)} items, methods: {sorted(keep)})")

    df = pd.DataFrame(all_rows, columns=["dataset", "item", "method", "n_cells",
                                         "rep", "n", "pearson", "spearman"])
    df.to_csv(a.out_dir / "stability_curves.csv", index=False)
    print(f"wrote stability_curves.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
