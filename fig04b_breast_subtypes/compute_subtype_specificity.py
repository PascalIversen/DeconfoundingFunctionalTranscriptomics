"""Per-tissue, per-method PAM50 lineage enrichment of CTRPv2 attribution.

Reads:   <PREDS_ROOT>/ctrpv2_seed1_preds/*.npz   (per-cell SHAP matrices)
Writes:  results/subtype_tissue_specificity.csv

For every drug and every tissue with >= MIN_CELLS cell lines, rank the panel
genes by mean |SHAP| within that tissue's cells and score how well the ranking
recovers the PAM50 lineage-receptor subset (AUROC of PAM50-membership vs.
importance). A genuinely breast-specific signature should peak in breast (an
epithelial lineage) and sit lower in non-epithelial lineages.

The reference gene set is PAM50 (Parker 2009) minus its proliferation arm (the
22 cell-cycle/mitotic genes of the Hallmark G2M_CHECKPOINT and E2F_TARGETS
programme, which includes the canonical Nielsen 2010 / Wallden 2015 proliferation
meta-gene). The proliferation arm is excluded because in cultured cell lines every
line proliferates uniformly, so cell-cycle markers (BIRC5, MKI67, ...) decouple
from breast-subtype membership; the remaining 28 PAM50 genes reflect cell-of-origin
biology that survives immortalization.

Usage (from this folder):
    python compute_subtype_specificity.py
    python compute_subtype_specificity.py --preds-root /path/to/results
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# Adversarial baselines (DANN / AD-AE) live in a sibling "*_extra" directory so
# the published prediction bundle stays untouched; load_preds merges them in.
import sys as _sys
_sys.path.insert(0, str(HERE.parent / "training"))
from shared.preds_io import load_preds  # noqa: E402

# Full PAM50 (Parker 2009), canonical symbols (one per gene -- matches
# ../training/pam50_genes.txt, the panel actually used as model input features).
PAM50_ALL = set("""
ACTR3B ANLN BAG1 BCL2 BIRC5 BLVRA CCNB1 CCNE1 CDC20 CDC6 CDH3 CENPF CEP55
CXXC5 EGFR ERBB2 ESR1 EXO1 FGFR4 FOXA1 FOXC1 GPR160 GRB7 KIF2C KRT14 KRT17
KRT5 MAPT MDM2 MELK MIA MKI67 MLPH MMP11 MYBL2 MYC NAT1 NDC80 NUF2 ORC6
PGR PHGDH PTTG1 RRM2 SFRP1 SLC39A6 TMEM45B TYMS UBE2C UBE2T
""".split())

# Proliferation arm = PAM50's cell-cycle / mitotic programme (Hallmark
# G2M_CHECKPOINT and E2F_TARGETS overlap; includes the canonical Nielsen 2010
# proliferation meta-gene, Clin Cancer Res 16:5222 / Wallden 2015). Excluded
# because in cultured lines every line proliferates uniformly, so these genes
# carry no subtype signal and only dilute the breast-specificity readout.
PAM50_PROLIF = set("""
BIRC5 CCNB1 CCNE1 CDC20 CDC6 NUF2 NDC80 ANLN CENPF CEP55 EXO1
KIF2C MKI67 MELK MYBL2 MYC ORC6 PTTG1 RRM2 TYMS UBE2C UBE2T
""".split())

# PAM50 lineage-receptor subset = PAM50 minus the proliferation arm (28 genes).
PAM50_LINEAGE = PAM50_ALL - PAM50_PROLIF

METHODS = ["marginal", "residualized", "irm", "within_tissue",
           "dann", "adae"]
MIN_CELLS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", type=Path,
                    default=HERE.parent / "results",
                    help="Root containing ctrpv2_seed1_preds/*.npz")
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS,
                    help="Min cell lines per (drug, tissue) to score an AUROC")
    ap.add_argument("--out", type=Path,
                    default=HERE / "results" / "subtype_tissue_specificity.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    preds_dir = args.preds_root / "ctrpv2_seed1_preds"
    files = sorted(preds_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files under {preds_dir}")

    recs = []
    for f in files:
        d = load_preds(f)
        genes = np.array([str(g) for g in d["gene_cols"]])
        tissue = d["tissue"].astype(str)
        lab = np.array([g in PAM50_LINEAGE for g in genes])
        if lab.sum() < 3 or lab.sum() == len(lab):
            continue
        for t in np.unique(tissue):
            mask = (tissue == t)
            if mask.sum() < args.min_cells:
                continue
            for m in METHODS:
                key = f"shap_{m}"
                if key not in d.files:
                    continue
                with np.errstate(invalid="ignore"):
                    imp = np.nanmean(np.abs(d[key][mask]), axis=0)
                # A method can legitimately have no attribution for a cell
                # block -- within-tissue SHAP leaves NaN for sub-threshold
                # tissues, and AD-AE leaves NaN for folds whose head collapsed
                # to an intercept. Skip those rather than scoring padding.
                if not np.isfinite(imp).all():
                    continue
                auc = roc_auc_score(lab, imp)
                recs.append((f.stem, t, m, float(auc), int(mask.sum())))

    df = pd.DataFrame(recs, columns=[
        "drug", "tissue", "method", "auroc", "n_cells"])
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} (min_cells={args.min_cells}, {len(df)} rows; "
          f"{df['drug'].nunique()} drugs, {df['tissue'].nunique()} tissues)")
    print(f"PAM50 lineage subset: {len(PAM50_LINEAGE)} genes "
          f"(of which {int(lab.sum())} present in panel)")


if __name__ == "__main__":
    main()
