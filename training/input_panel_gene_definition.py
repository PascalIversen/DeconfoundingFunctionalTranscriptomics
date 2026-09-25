"""Single source of truth for the model INPUT gene panel.

    panel(dataset) = L1000 landmark (867)
                     ∪ per-dataset target genes
                     ∪ PAM50 breast-subtype genes (50)

intersected with the available expression matrix at load time (done in
`shared.data.load_{depmap_crispr,ccle_expression}` via the `extra_genes`
union). This module defines the union *inputs* centrally and, when run as a
script, SAVES them so downstream code (gen_preds_v2.py / dispatch.sh / the
figure scripts) just loads the files instead of recomputing:

    items_{depmap,ctrpv2}.txt    prediction items (knockout targets / drugs)
    targets_{depmap,ctrpv2}.txt  HUGO symbols unioned into the expression panel
                                 ( = dataset targets ∪ PAM50 )
    panel_{depmap,ctrpv2}.txt    the *realised* panel (landmark ∪ targets ∪ PAM50)
                                 ∩ expression -- written only if --realize
    pam50_genes.txt              the PAM50 set actually used (provenance)

Load saved lists later with load_items() / load_targets() / load_panel().

------------------------------------------------------------------------------
PAM50 source.  Parker JS, Mullins M, Cheang MCU, et al. "Supervised Risk
Predictor of Breast Cancer Based on Subtypes." J Clin Oncol. 2009;27(8):
1160-1167.  The 50 symbols below are the gene set carried by the `pam50`
centroids in the genefu Bioconductor package (the de-facto reference
implementation).  A few genes are recorded under historical aliases in older
expression matrices; PAM50_ALIASES expands them so the union matches whichever
symbol the matrix actually uses (extras absent from expression are dropped at
intersection, so over-listing is safe).

PAM50 is defined here, in the list-builder module itself, rather than
hand-appended to targets_*.txt, so that re-running the list builder always
includes it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shared.data import (load_depmap_crispr, load_ccle_expression,        # noqa: E402
                         load_ctrpv2_response, load_drug_target_map,
                         load_gene_set)

LANDMARK = "landmark_genes"

# ---- PAM50 (Parker 2009 / genefu) ----------------------------------------
PAM50 = [
    "UBE2T", "BIRC5", "NUF2", "CDC6", "CCNB1", "TYMS", "MYBL2", "CEP55",
    "MELK", "NDC80", "RRM2", "UBE2C", "CENPF", "PTTG1", "EXO1", "ORC6",
    "ANLN", "CCNE1", "CDC20", "MKI67", "KIF2C", "ACTR3B", "MYC", "EGFR",
    "KRT5", "PHGDH", "CDH3", "MIA", "KRT17", "FOXC1", "SFRP1", "KRT14",
    "ESR1", "SLC39A6", "BAG1", "MAPT", "PGR", "CXXC5", "MLPH", "BCL2",
    "MDM2", "NAT1", "FOXA1", "BLVRA", "MMP11", "GPR160", "FGFR4", "GRB7",
    "TMEM45B", "ERBB2",
]
assert len(PAM50) == 50 and len(set(PAM50)) == 50, "PAM50 must be 50 unique genes"

# Current HUGO symbol -> historical aliases seen in expression matrices.
PAM50_ALIASES = {
    "ORC6":  ["ORC6L"],
    "NUF2":  ["CDCA1"],
    "NDC80": ["KNTC2"],
    "CEP55": ["C10orf3", "FLJ10540"],
    "KIF2C": ["KNSL6"],
    "ACTR3B": ["ARP4", "ARPM1"],
    "MIA":   ["CD-RAP", "CDRAP"],
}


def pam50_with_aliases() -> set:
    """PAM50 current symbols plus all known historical aliases."""
    out = set(PAM50)
    for aliases in PAM50_ALIASES.values():
        out |= set(aliases)
    return out


# ---- per-dataset target genes --------------------------------------------
def build_depmap_targets(top_n: int = 200) -> list:
    """Top-N most-variable knockout targets from the FULL CRISPR matrix,
    restricted to genes also present in expression (so each target's own
    expression can be a panel feature)."""
    _, crispr, _ = load_depmap_crispr(gene_list=LANDMARK)
    expr_cols = pd.read_csv(
        HERE.parent.parent / "data" / "DepMap"
        / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", nrows=1).columns
    expr_genes = {c.split(" (")[0] for c in expr_cols}
    cands = [g for g in crispr.columns if g in expr_genes]
    stds = crispr[cands].std().dropna().sort_values(ascending=False)
    return stds.head(top_n).index.tolist()


def build_ctrpv2_items() -> list:
    resp = load_ctrpv2_response(min_cells_per_drug=100, min_tissues_per_drug=8)
    return sorted(resp["drug_name"].unique())


def build_ctrpv2_targets(items: list | None = None) -> list:
    """All annotated drug-target genes across the kept CTRPv2 drugs."""
    items = items if items is not None else build_ctrpv2_items()
    dt = load_drug_target_map()
    return sorted({g for d in items for g in dt.get(d, [])})


# ---- the panel union inputs ----------------------------------------------
def panel_targets(dataset: str, depmap_targets=None, ctrpv2_targets=None) -> list:
    """Genes to union into the landmark panel for `dataset`:
    dataset targets ∪ PAM50 (+ aliases)."""
    if dataset == "depmap":
        tgts = depmap_targets if depmap_targets is not None else build_depmap_targets()
    elif dataset == "ctrpv2":
        tgts = ctrpv2_targets if ctrpv2_targets is not None else build_ctrpv2_targets()
    else:
        raise ValueError(dataset)
    return sorted(set(tgts) | pam50_with_aliases())


# ---- save / load ----------------------------------------------------------
def _write(path: Path, items) -> None:
    path.write_text("\n".join(items) + "\n")


def _read(path: Path) -> list:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_items(dataset: str, d: Path = HERE) -> list:
    return _read(d / f"items_{dataset}.txt")


def load_targets(dataset: str, d: Path = HERE) -> list:
    return _read(d / f"targets_{dataset}.txt")


def load_panel(dataset: str, d: Path = HERE) -> list:
    """Realised panel (only present if saved with --realize)."""
    return _read(d / f"panel_{dataset}.txt")


def realised_panel(dataset: str, targets: list) -> list:
    """landmark ∪ targets ∪ PAM50, intersected with the dataset's expression."""
    extra = set(targets)
    if dataset == "depmap":
        expr, _, _ = load_depmap_crispr(gene_list=LANDMARK, extra_genes=extra)
    else:
        expr = load_ccle_expression(gene_list=LANDMARK, extra_genes=extra)
    return sorted(expr.columns)


def _panel_provenance_df(panel_genes: list, dataset_targets: set) -> pd.DataFrame:
    """One row per realised-panel gene, with boolean source-of-membership cols
    (landmark / target / pam50) plus a comma-joined source string for humans."""
    lm = load_gene_set(LANDMARK)
    pam = pam50_with_aliases()
    rows = []
    for g in sorted(panel_genes):
        in_landmark = g in lm
        in_target = g in dataset_targets
        in_pam50 = g in pam
        sources = []
        if in_landmark: sources.append("landmark")
        if in_target:   sources.append("target")
        if in_pam50:    sources.append("pam50")
        rows.append({"gene": g,
                     "landmark": int(in_landmark),
                     "target":   int(in_target),
                     "pam50":    int(in_pam50),
                     "source":   ",".join(sources)})
    return pd.DataFrame(rows)


def save_lists(out_dir: Path = HERE, top_n_depmap: int = 200,
               realize: bool = False, write_csv: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "pam50_genes.txt", sorted(PAM50))

    # DepMap: items = the 200 prediction targets; targets = items ∪ PAM50.
    dep_items = build_depmap_targets(top_n_depmap)
    _write(out_dir / "items_depmap.txt", dep_items)
    dep_targets = panel_targets("depmap", depmap_targets=dep_items)
    _write(out_dir / "targets_depmap.txt", dep_targets)
    print(f"[depmap] {len(dep_items)} items, {len(dep_targets)} panel-union targets")

    # CTRPv2: items = drugs; targets = drug-target genes ∪ PAM50.
    ctr_items = build_ctrpv2_items()
    _write(out_dir / "items_ctrpv2.txt", ctr_items)
    ctr_targets = panel_targets("ctrpv2", ctrpv2_targets=build_ctrpv2_targets(ctr_items))
    _write(out_dir / "targets_ctrpv2.txt", ctr_targets)
    print(f"[ctrpv2] {len(ctr_items)} items, {len(ctr_targets)} panel-union targets")

    lm = load_gene_set(LANDMARK)
    print(f"\n[diagnostics] landmark={len(lm)}  PAM50={len(PAM50)} "
          f"(+{len(pam50_with_aliases()) - len(PAM50)} aliases)")
    for ds, tg in [("depmap", dep_targets), ("ctrpv2", ctr_targets)]:
        pam_not_lm = sorted(pam50_with_aliases() - lm)
        print(f"[diagnostics] {ds}: PAM50 not in landmark (union'd) "
              f"= {len(set(pam_not_lm) & set(tg))}")
        if realize or write_csv:
            panel = realised_panel(ds, tg)
            if realize:
                _write(out_dir / f"panel_{ds}.txt", panel)
            npam = len(set(panel) & pam50_with_aliases())
            print(f"[diagnostics] {ds}: REALISED panel = {len(panel)} genes "
                  f"(PAM50 present: {npam}/50)")
            if write_csv:
                df = _panel_provenance_df(panel, set(tg))
                csv_path = out_dir / f"panel_{ds}.csv"
                df.to_csv(csv_path, index=False)
                print(f"[diagnostics] {ds}: wrote {csv_path} "
                      f"(landmark={df['landmark'].sum()}, "
                      f"target={df['target'].sum()}, "
                      f"pam50={df['pam50'].sum()})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--top-n-depmap", type=int, default=200)
    ap.add_argument("--realize", action="store_true",
                    help="also write panel_{ds}.txt = realised panel ∩ expression "
                         "(loads the big expression matrices; slower)")
    ap.add_argument("--csv", action="store_true",
                    help="also write panel_{ds}.csv (one row per realised-panel "
                         "gene, with landmark/target/pam50 provenance columns); "
                         "implies the realise step")
    a = ap.parse_args()
    save_lists(a.out_dir, a.top_n_depmap, a.realize, a.csv)


if __name__ == "__main__":
    main()
