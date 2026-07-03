"""Per-tissue accuracy of the linear tissue classifier (Arm 1), for the swarm.

Re-runs *only* the multinomial logistic tissue classifier from
`compute_confounding` — via the shared `build_depmap` / `build_ctrpv2` loaders
and `confounding_utils.tissue_classifier_lift` — and writes the per-tissue,
fold-averaged accuracy table the swarm figure consumes. The headline classifier
accuracy / lift already live in `confounding_summary.csv`; this only adds the
per-tissue breakdown, so it leaves every other results CSV untouched.

Each row is one tissue's accuracy (recall) averaged over the same stratified
5-fold CV the overall classifier uses, for each dataset. Defaults mirror
`compute_confounding.py` (panel_depmap / panel_ctrpv2, seed 0) so the numbers
match what a full pipeline run would produce.

Usage (from figure_confounding_evidence/):
    python compute_classifier_per_tissue.py
    python compute_classifier_per_tissue.py --datasets depmap
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "training"))

from compute_confounding import build_depmap, build_ctrpv2  # noqa: E402
from confounding_utils import tissue_classifier_lift          # noqa: E402

BUILDERS = {"depmap": build_depmap, "ctrpv2": build_ctrpv2}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["depmap", "ctrpv2"],
                   choices=["depmap", "ctrpv2"])
    p.add_argument("--gene-list", default=None,
                   help="Feature panel for both datasets (else per-dataset).")
    p.add_argument("--gene-list-depmap", default="panel_depmap")
    p.add_argument("--gene-list-ctrpv2", default="panel_ctrpv2")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--top-n-targets", type=int, default=200)
    p.add_argument("--min-cells", type=int, default=100)
    p.add_argument("--min-tissues", type=int, default=8)
    p.add_argument("--clf-max-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    override = args.gene_list           # explicit --gene-list applies to both
    per_ds_default = {"depmap": args.gene_list_depmap,
                      "ctrpv2": args.gene_list_ctrpv2}

    rows = []
    for ds in args.datasets:
        args.gene_list = override if override else per_ds_default[ds]
        print(f"=== {ds} (gene panel: {args.gene_list}) ===", flush=True)
        X_df, tissue_arr, _outcomes, _conf_df = BUILDERS[ds](args)
        T = np.asarray(tissue_arr).astype(str)
        t0 = time.time()
        clf = tissue_classifier_lift(X_df.values, T, seed=args.seed,
                                     max_iter=args.clf_max_iter)
        pt = clf["per_tissue"].copy()
        pt.insert(0, "dataset", ds)
        rows.append(pt)
        print(f"  [{ds}] {len(pt)} tissues (>=5 cells) | overall acc="
              f"{clf['accuracy']:.2f} vs chance {clf['chance']:.2f} | per-tissue "
              f"acc median={pt['accuracy'].median():.2f} "
              f"[{pt['accuracy'].min():.2f}, {pt['accuracy'].max():.2f}] "
              f"({time.time()-t0:.0f}s)")

    out = pd.concat(rows, ignore_index=True)
    dest = args.out_dir / "confounding_clf_per_tissue.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
