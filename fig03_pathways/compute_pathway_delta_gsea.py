"""Cutoff-free differential enrichment for fig_pathway_delta_dotplot.

Per-gene promotion score

    delta_g = percentile_method(g) - percentile_marginal(g)

(cross-item mean |SHAP|-rank turned into a percentile; + = the deconfounding method
ranks gene g higher than marginal), then a pre-ranked GSEA (gseapy, 10000 permutations)
of the gene-set libraries on the delta vector. Demoted (negative NES) = lineage /
tissue-context programs; promoted (positive NES) = cell-intrinsic proliferation/metabolism.

Gene-set universe = Human Gene Atlas (lineage; immortalised HGA cancer lines dropped)
+ KEGG + MSigDB Hallmark, EXCLUDING KEGG "Human Diseases" pathways other than Cancer
(Infectious disease, Substance dependence, Neurodegenerative, Cardiovascular,
Endocrine/metabolic, Immune disease) — gene-overlap grab-bags that are not
disease-specific (e.g. Alcoholism, Amoebiasis, Coronavirus disease). Standard practice
for enrichment analysis; the exclusion is category-based and applied a priori, read from
gene_sets/set_classes.tsv (run build_set_classes.py first).

10000 permutations are required: gseapy's gene-set-permutation FDR is Monte-Carlo noisy,
and 1000 permutations leaves the significant-set membership ~40% unstable.

Output: results/pathway_delta_gsea.csv  (dataset, method, library, set, NES, nom_p, fdr_q)

Usage:
    python build_set_classes.py                       # -> gene_sets/set_classes.tsv
    python compute_pathway_delta_gsea.py --preds-root /path/to/results
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pathway_utils import importance_from_npz, load_gmt          # noqa: E402
import gseapy as gp                                              # noqa: E402

GS = HERE / "gene_sets"
ALL_METHODS = ["marginal", "residualized", "irm", "within_tissue"]
DECONF = ["residualized", "irm", "within_tissue"]
N_PERM = 10000

# Human Gene Atlas mixes normal tissues with immortalised cancer lines whose markers
# overlap genuine cancer biology; drop them so "lineage" = tissue identity.
HGA_CANCER_LINES = {
    "721 B lymphoblasts", "Colorectaladenocarcinoma",
    "Leukemia chronicMyelogenousK-562", "Leukemia promyelocytic-HL-60",
    "Leukemialymphoblastic(MOLT-4)", "Lymphoma burkitts(Daudi)",
    "Lymphoma burkitts(Raji)",
}


def disease_excluded() -> set:
    """KEGG 'Human Diseases' pathways except Cancer, from set_classes.tsv."""
    sc = GS / "set_classes.tsv"
    if not sc.exists():
        raise SystemExit("run build_set_classes.py first (need gene_sets/set_classes.tsv)")
    ex = set()
    for ln in sc.read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 4 and p[1] == "KEGG" and p[2].startswith("Human Diseases") \
                and "Cancer" not in p[2]:
            ex.add(p[0])
    return ex


def aggregate_percentiles(preds_dir: Path) -> tuple:
    """Cross-item mean importance-percentile per gene per method (1 = top)."""
    files = sorted(preds_dir.glob("*.npz"))
    genes = None
    acc = {m: None for m in ALL_METHODS}
    n = 0
    for f in files:
        gene_cols, imps = importance_from_npz(f, ALL_METHODS)
        if genes is None:
            genes = np.array([str(g) for g in gene_cols])
            acc = {m: np.zeros(len(genes)) for m in ALL_METHODS}
        for m in ALL_METHODS:
            if m in imps:
                acc[m] += rankdata(np.nan_to_num(imps[m])) / len(genes)
        n += 1
    return genes, {m: acc[m] / n for m in ALL_METHODS}, n


def run_dataset(dataset: str, preds_dir: Path, exclude: set, n_perm: int) -> pd.DataFrame:
    genes, pct, n = aggregate_percentiles(preds_dir)
    hga = {k: list(v) for k, v in load_gmt(GS / "human_gene_atlas.gmt").items()
           if k not in HGA_CANCER_LINES}
    kegg = {k: list(v) for k, v in load_gmt(GS / "kegg.gmt").items()}
    hall = {k: list(v) for k, v in load_gmt(GS / "hallmark.gmt").items()}
    lineage_terms = set(hga)
    gene_sets = {k: v for k, v in {**hga, **kegg, **hall}.items() if k not in exclude}
    rows = []
    for m in DECONF:
        delta = pct[m] - pct["marginal"]
        rnk = (pd.DataFrame({"gene": genes, "score": delta})
               .sort_values("score", ascending=False))
        pre = gp.prerank(rnk=rnk, gene_sets=gene_sets, min_size=10, max_size=250,
                         permutation_num=n_perm, seed=0, outdir=None, no_plot=True,
                         threads=4, verbose=False)
        res = pre.res2d.rename(columns={"NOM p-val": "nom_p", "FDR q-val": "fdr_q"})
        for _, r in res.iterrows():
            term = r["Term"]
            rows.append((dataset, m,
                         "lineage" if term in lineage_terms else "functional",
                         term, float(r["NES"]), float(r["nom_p"]), float(r["fdr_q"])))
        print(f"  {dataset} [{m}]: {len(res)} sets", flush=True)
    print(f"  {dataset}: {n} items aggregated, universe {len(gene_sets)} sets")
    return pd.DataFrame(rows, columns=["dataset", "method", "library", "set",
                                       "NES", "nom_p", "fdr_q"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", type=Path, default=HERE.parent / "results",
                    help="root with {depmap,ctrpv2}_seed1_preds/*.npz")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "pathway_delta_gsea.csv")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()
    exclude = disease_excluded()
    print(f"excluding {len(exclude)} KEGG Human-Diseases-except-Cancer sets; n_perm={a.n_perm}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for ds in ["depmap", "ctrpv2"]:
        preds = a.preds_root / f"{ds}_seed1_preds"
        if not preds.exists():
            print(f"skip {ds}: no {preds}")
            continue
        print(f"=== {ds} ===", flush=True)
        frames.append(run_dataset(ds, preds, exclude, a.n_perm))
    pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
