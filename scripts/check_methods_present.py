"""Assert every computed result CSV actually carries the adversarial methods.

Guards against a specific, quiet failure mode: a consumer script whose
`METHODS` list was extended but whose npz loader was not switched to
`shared.preds_io.load_preds`. It then runs cleanly, writes a plausible
file, and simply omits the new methods -- which looks like "the figure is
done" rather than an error, and the gap can be easy to miss unless something
actually checks column-by-column.

    python scripts/check_methods_present.py            # exits 1 on a gap
    python scripts/check_methods_present.py --expect marginal,residualized
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (path, column holding the method name). Files absent from disk are skipped
# with a note rather than failing, so this is usable mid-pipeline.
TARGETS = [
    ("fig02_ci_distribution/results/depmap_attr_eta2.csv", "method"),
    ("fig02_ci_distribution/results/ctrpv2_attr_eta2.csv", "method"),
    ("fig02_ci_distribution/results/depmap_contamination_at_k.csv", "method"),
    ("fig02_ci_distribution/results/ctrpv2_contamination_at_k.csv", "method"),
    ("fig02_ci_distribution/results/auroc_tau.csv", "method"),
    ("fig03_pathways/results/pathway_delta_gsea.csv", "method"),
    ("fig04b_breast_subtypes/results/subtype_tissue_specificity.csv", "method"),
    ("table_performance/results/per_item_pearson.csv", "method"),
    ("table_performance/results/per_item_mse.csv", "method"),
]

# within_tissue is an attribution-stage method on top of the marginal model
# and has no predictions of its own, so the accuracy tables legitimately
# omit it; marginal/baseline are references, not deconfounding methods.
EXEMPT = {
    "table_performance/results/per_item_pearson.csv": {"within_tissue"},
    "table_performance/results/per_item_mse.csv": {"within_tissue"},
    "fig03_pathways/results/pathway_delta_gsea.csv": {"marginal"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", default="dann,adae",
                    help="comma-separated methods that must be present")
    a = ap.parse_args()
    expect = {m.strip() for m in a.expect.split(",") if m.strip()}

    missing_any = False
    for rel, col in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"  skip (absent)   {rel}")
            continue
        try:
            found = set(pd.read_csv(p, usecols=[col])[col].unique())
        except Exception as exc:
            print(f"  ERROR reading   {rel}: {exc}")
            missing_any = True
            continue
        need = expect - EXEMPT.get(rel, set())
        gap = need - found
        if gap:
            print(f"  MISSING {sorted(gap)}  {rel}   (has {sorted(found)})")
            missing_any = True
        else:
            print(f"  ok              {rel}   ({len(found)} methods)")

    if missing_any:
        print("\nFAIL: some results omit the adversarial methods. Check that the "
              "consumer uses shared.preds_io.load_preds (not np.load) AND has "
              "the methods in its METHODS list, then re-run it.")
        sys.exit(1)
    print("\nOK: every present result carries the expected methods.")


if __name__ == "__main__":
    main()
