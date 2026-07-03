"""Build the canonical item lists for the landmark∪targets rerun.

Writes:
  items_depmap.txt  — top-N most-variable knockout targets from the FULL
                      CRISPR matrix (Option B; not restricted to landmark)
  items_ctrpv2.txt  — all CTRPv2 drugs surviving min_cells/min_tissues filter
  targets_depmap.txt — the same as items_depmap, used to seed the
                       extra_genes union (so target self-expression is in panel)

Run once on the head node before dispatching.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shared.data import (load_depmap_crispr, load_ctrpv2_response,  # noqa: E402
                         load_drug_target_map, load_gene_set)


def build_depmap_items(top_n: int = 200) -> list:
    """Top-N most-variable knockouts from the FULL CRISPR matrix.

    Restricted to genes also present in the OmicsExpression matrix so
    the union'd panel will have a column for the target.
    """
    _, crispr, _ = load_depmap_crispr(gene_list="landmark_genes")
    expr_path = (HERE.parent / "data" / "DepMap"
                 / "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    expr_cols = pd.read_csv(expr_path, nrows=1).columns
    expr_genes = {c.split(" (")[0] for c in expr_cols}
    candidates = [g for g in crispr.columns if g in expr_genes]
    stds = crispr[candidates].std().dropna().sort_values(ascending=False)
    return stds.head(top_n).index.tolist()


def build_ctrpv2_items() -> list:
    resp = load_ctrpv2_response(min_cells_per_drug=100, min_tissues_per_drug=8)
    return sorted(resp["drug_name"].unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--top-n-depmap", type=int, default=200)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # DepMap targets
    depmap_items = build_depmap_items(args.top_n_depmap)
    (args.out_dir / "items_depmap.txt").write_text(
        "\n".join(depmap_items) + "\n")
    (args.out_dir / "targets_depmap.txt").write_text(
        "\n".join(depmap_items) + "\n")
    print(f"[depmap] {len(depmap_items)} targets -> "
          f"{args.out_dir / 'items_depmap.txt'}")

    # CTRPv2 drugs
    ctrpv2_items = build_ctrpv2_items()
    (args.out_dir / "items_ctrpv2.txt").write_text(
        "\n".join(ctrpv2_items) + "\n")

    # CTRPv2 union targets = all annotated targets across the kept drugs
    dt = load_drug_target_map()
    ctrpv2_targets = sorted({g for d in ctrpv2_items for g in dt.get(d, [])})
    (args.out_dir / "targets_ctrpv2.txt").write_text(
        "\n".join(ctrpv2_targets) + "\n")
    print(f"[ctrpv2] {len(ctrpv2_items)} drugs -> "
          f"{args.out_dir / 'items_ctrpv2.txt'}")
    print(f"[ctrpv2] {len(ctrpv2_targets)} distinct drug-target genes -> "
          f"{args.out_dir / 'targets_ctrpv2.txt'}")

    # Coverage diagnostics
    lm = load_gene_set("landmark_genes")
    print(f"\n[diagnostics] landmark size: {len(lm)}")
    print(f"[diagnostics] DepMap targets ∩ landmark: "
          f"{len(set(depmap_items) & lm)}/{len(depmap_items)}")
    print(f"[diagnostics] DepMap targets NOT in landmark (will be union'd in):"
          f" {len(set(depmap_items) - lm)}")
    print(f"[diagnostics] CTRPv2 union targets ∩ landmark: "
          f"{len(set(ctrpv2_targets) & lm)}/{len(ctrpv2_targets)}")
    print(f"[diagnostics] CTRPv2 union targets NOT in landmark: "
          f"{len(set(ctrpv2_targets) - lm)} "
          f"-> final CTRPv2 panel ≈ {len(lm | set(ctrpv2_targets))} genes "
          f"(of those ∩ CCLE).")


if __name__ == "__main__":
    main()
