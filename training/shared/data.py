"""Data loading for the DepMap CRISPR and CTRPv2 experiments.

Auto-detects a data/ directory containing DepMap/, CCLE/, CTRPv2/ files.
Looks first in ./data/, then ../data/, then ../../data/.

Bundled small metadata:
  data/meta/panel_depmap.csv              DepMap feature panel (1071 genes)
  data/meta/panel_ctrpv2.csv              CTRPv2 feature panel (1109 genes)
  data/meta/gdsc_drug_list.csv            drug → target / pathway

Drug-target annotations are normalized via a HUGO synonym map (MEK1
→ MAP2K1, etc.) and class-expansion table (Proteasome → PSMA1..PSMB7).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
META_DIR = PKG_ROOT / "data" / "meta"


def _find_data_root() -> Path:
    for cand in [PKG_ROOT / "data", PKG_ROOT.parent / "data",
                 PKG_ROOT.parent.parent / "data"]:
        if (cand / "DepMap").exists() or (cand / "CCLE").exists() \
                or (cand / "CTRPv2").exists():
            return cand
    raise FileNotFoundError(
        f"No data/ directory with DepMap/, CCLE/, or CTRPv2/ found "
        f"under {PKG_ROOT} or parents.")


DATA_ROOT = _find_data_root()
logger.info("data root: %s", DATA_ROOT)

# META_DIR defaults to <pkg>/data/meta, but that is not always bundled next to
# the training code. Fall back to the auto-detected repo data/meta (which holds
# gdsc_drug_list.csv + ctrpv2_compound_targets.csv) so drug-target / gene-list
# lookups don't silently return empty.
if not META_DIR.exists() and (DATA_ROOT / "meta").exists():
    META_DIR = DATA_ROOT / "meta"
logger.info("meta dir: %s", META_DIR)


# ====================================================== HUGO synonyms
# Kinase / receptor / RAS-family aliases used in GDSC's "Targets" column but
# not matching HUGO. Mapping is targeted, not exhaustive.
HUGO_SYNONYMS = {
    "MEK1": ["MAP2K1"], "MEK2": ["MAP2K2"],
    "MEK1, MEK2": ["MAP2K1", "MAP2K2"],
    "ERK1": ["MAPK3"], "ERK2": ["MAPK1"],
    "ERK1, ERK2": ["MAPK1", "MAPK3"],
    "AKT": ["AKT1", "AKT2", "AKT3"],
    "AKT1, AKT2": ["AKT1", "AKT2"],
    "HER2": ["ERBB2"], "HER3": ["ERBB3"], "HER4": ["ERBB4"],
    "VEGFR": ["FLT1", "KDR", "FLT4"],
    "VEGFR1": ["FLT1"], "VEGFR2": ["KDR"], "VEGFR3": ["FLT4"],
    "VEGFR1, VEGFR2": ["FLT1", "KDR"],
    "VEGFR1, VEGFR2, VEGFR3": ["FLT1", "KDR", "FLT4"],
    "mTOR": ["MTOR"], "mTORC1": ["MTOR"], "mTORC2": ["MTOR"],
    "MTOR": ["MTOR"],
    "p38": ["MAPK14"],
    "p38alpha": ["MAPK14"], "p38beta": ["MAPK11"],
    "JNK1": ["MAPK8"], "JNK2": ["MAPK9"], "JNK3": ["MAPK10"],
    "JNK": ["MAPK8", "MAPK9", "MAPK10"],
    "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    "PI3K alpha": ["PIK3CA"], "PI3K (alpha)": ["PIK3CA"],
    "PI3K beta": ["PIK3CB"], "PI3K (beta)": ["PIK3CB"],
    "PI3K delta": ["PIK3CD"], "PI3K (delta)": ["PIK3CD"],
    "PI3K gamma": ["PIK3CG"], "PI3K (gamma)": ["PIK3CG"],
    "RAS": ["KRAS", "NRAS", "HRAS"],
    "RAF": ["BRAF", "RAF1", "ARAF"],
    "B-Raf": ["BRAF"], "C-Raf": ["RAF1"],
    "PLK": ["PLK1"],
    "Aurora A": ["AURKA"], "Aurora B": ["AURKB"],
    "Aurora A, Aurora B": ["AURKA", "AURKB"],
    "Aurora Kinases": ["AURKA", "AURKB", "AURKC"],
    "BCL-XL": ["BCL2L1"], "BCL2L": ["BCL2L1"],
    "BCL2 family": ["BCL2", "BCL2L1", "MCL1"],
    "BCR-ABL": ["ABL1"],
    "FGFRs": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "IGF1R, IR": ["IGF1R", "INSR"],
    "IGF1R, INSR": ["IGF1R", "INSR"],
    "TNKS1, TNKS2": ["TNKS", "TNKS2"],
}

# Class / family target annotations that are not single genes. Expanded
# to the HUGO members.
CLASS_EXPANSION = {
    "Proteasome": [
        "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
        "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7"],
    "HDAC": [f"HDAC{i}" for i in range(1, 12)],
    "HDAC inhibitor": [f"HDAC{i}" for i in range(1, 12)],
    "HDAC inhibitor (Class I, II)": [f"HDAC{i}" for i in range(1, 12)],
    "HDAC inhibitor Class I": ["HDAC1", "HDAC2", "HDAC3", "HDAC8"],
    "HDAC inhibitor Class II": ["HDAC4", "HDAC5", "HDAC6", "HDAC7",
                                  "HDAC9", "HDAC10"],
    "CDK": ["CDK1", "CDK2", "CDK4", "CDK6", "CDK7", "CDK9"],
    "HSP90": ["HSP90AA1", "HSP90AB1", "HSP90B1", "TRAP1"],
    "Topoisomerase": ["TOP1", "TOP2A", "TOP2B"],
    "Topoisomerase I": ["TOP1"],
    "Topoisomerase II": ["TOP2A", "TOP2B"],
    "DNMT": ["DNMT1", "DNMT3A", "DNMT3B"],
    "Tubulin": [f"TUB{x}" for x in ["A1A", "A1B", "B", "B2A", "B3", "B4A"]],
    "Ephrins": [f"EPHA{i}" for i in range(1, 9)] + [f"EPHB{i}" for i in range(1, 7)],
    "BET": ["BRD2", "BRD3", "BRD4", "BRDT"],
    "PARP": ["PARP1", "PARP2", "PARP3", "PARP4"],
    "Anthracycline": ["TOP2A", "TOP2B"],   # mechanism via topoisomerase II
    "Antimetabolite": [],                  # not a gene class
    "Antimetabolite (DNA & RNA)": [],
    "DNA alkylating agent": [],
    "DNA crosslinker": [],
}

# Drug-name normalization for fuzzy matching across sources (case +
# whitespace + hyphen + underscore differences). Used both directions.
def _norm_drug(name: str) -> str:
    if name is None:
        return ""
    return (str(name).lower()
            .replace(" ", "").replace("-", "")
            .replace("_", "").replace("/", "").replace(",", ""))


# ====================================================== gene lists
def _symbols_from_df(df: pd.DataFrame, src: str) -> set:
    """Extract HUGO symbols from a gene-list / panel CSV.

    Supports the legacy single 'Symbol' column (paccmann_pharma_genes.csv) and
    the annotated panel format (panel_depmap.csv / panel_ctrpv2.csv) whose
    symbol column is 'gene' alongside landmark/target/pam50/source flags. Falls
    back to the sole column for headerless single-column files.
    """
    for col in ("Symbol", "gene", "Gene", "symbol"):
        if col in df.columns:
            return set(df[col].dropna().astype(str))
    if df.shape[1] == 1:
        return set(df.iloc[:, 0].dropna().astype(str))
    raise ValueError(
        f"{src}: no recognised gene-symbol column "
        f"(looked for Symbol/gene; got {list(df.columns)})")


def load_gene_set(name_or_path: str = "paccmann_pharma") -> set:
    """Load a gene list as a set of HUGO symbols.

    Accepts a known list name (paccmann_pharma, panel_depmap, panel_ctrpv2,
    landmark_genes, hvg_5000, ...) or a path to a CSV. The symbol column may be
    'Symbol' (legacy) or 'gene' (annotated panels).
    """
    path = Path(name_or_path)
    if path.exists():
        return _symbols_from_df(pd.read_csv(path), str(path))
    candidates = [
        META_DIR / f"{name_or_path}_genes.csv",
        META_DIR / f"{name_or_path}.csv",
        DATA_ROOT / "meta" / "gene_lists" / f"gene_list_{name_or_path}.csv",
        DATA_ROOT / "meta" / "gene_lists" / f"{name_or_path}.csv",
    ]
    for c in candidates:
        if c.exists():
            return _symbols_from_df(pd.read_csv(c), str(c))
    raise FileNotFoundError(
        f"Couldn't find gene list '{name_or_path}'. Tried: {candidates}")


# ====================================================== drug annotations
def load_drug_pathway_map() -> Tuple[dict, dict]:
    df = pd.read_csv(META_DIR / "gdsc_drug_list.csv")
    df.columns = df.columns.str.strip()
    drug_pw = dict(zip(df["Name"].str.strip(), df["Target pathway"].str.strip()))
    drug_pw = {d: pw for d, pw in drug_pw.items()
               if pw and pw not in ("Other", "Unclassified")}
    pw_drugs = {}
    for d, pw in drug_pw.items():
        pw_drugs.setdefault(pw, []).append(d)
    return drug_pw, pw_drugs


def _normalize_targets(raw: str) -> list:
    """Expand synonyms + classes; deduplicate."""
    if not raw or raw == "nan":
        return []
    raw = raw.strip()
    # Try the whole annotation as a multi-target alias first (e.g. "MEK1, MEK2").
    if raw in HUGO_SYNONYMS:
        return list(HUGO_SYNONYMS[raw])
    if raw in CLASS_EXPANSION:
        return list(CLASS_EXPANSION[raw])
    out = []
    for t in raw.split(","):
        t = t.strip()
        if not t:
            continue
        if t in CLASS_EXPANSION:
            out.extend(CLASS_EXPANSION[t])
        elif t in HUGO_SYNONYMS:
            out.extend(HUGO_SYNONYMS[t])
        else:
            out.append(t)
    return sorted(set(out))


def load_drug_target_map() -> dict:
    """Returns {drug_name: [HUGO symbols...]}.

    Primary source: CTRPv2 PharmacoSet `gene_symbol_of_protein_target`
    annotations (~414 compounds, 324 distinct HUGO targets) — extracted
    from CTRPv2.rds and saved as data/meta/ctrpv2_compound_targets.csv.
    Fallback for drugs not in the PharmacoSet: GDSC drug list with the
    HUGO synonym / class-expansion normalization applied.

    The dict also contains normalized-name aliases so lookups by
    drug-name string from CTRPv2 response files succeed even when
    casing or whitespace/hyphen differs from the PharmacoSet's drugid.
    """
    out: dict = {}

    # 1. Primary: CTRPv2 PharmacoSet HUGO targets
    ctrpv2_path = META_DIR / "ctrpv2_compound_targets.csv"
    if ctrpv2_path.exists():
        df = pd.read_csv(ctrpv2_path)
        for _, r in df.iterrows():
            name = str(r["drug_name"]).strip()
            tgts = sorted({g.strip() for g in str(r["hugo_targets"]).split(";")
                            if g.strip()})
            if tgts:
                out[name] = tgts

    # 2. Fallback: GDSC drug list (with class expansion) for drugs not
    #    in the PharmacoSet.
    gdsc_path = META_DIR / "gdsc_drug_list.csv"
    if gdsc_path.exists():
        df = pd.read_csv(gdsc_path)
        df.columns = df.columns.str.strip()
        for _, r in df.iterrows():
            name = str(r["Name"]).strip()
            if name in out:
                continue
            tgts = _normalize_targets(str(r["Targets"]).strip())
            if tgts:
                out[name] = tgts

    # 3. Build normalized-name aliases pointing at the same target lists,
    #    so DT.get("PD-0332991") finds the same entry as "PD0332991".
    aliases = {}
    for name, tgts in out.items():
        key = _norm_drug(name)
        # If two raw names normalize to the same key with different
        # target sets, prefer the longer (more annotated) list.
        if key in aliases and len(aliases[key]) >= len(tgts):
            continue
        aliases[key] = tgts
    for k, v in aliases.items():
        out.setdefault(k, v)

    return out


# ====================================================== DepMap CRISPR
def load_depmap_crispr(gene_list: str = "panel_depmap",
                       extra_genes=None
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Returns (X_expr, Y_crispr, tissue) aligned by ModelID.

    X_expr columns are restricted to `gene_list` ∩ expression matrix,
    optionally UNION'd with `extra_genes` (e.g. knockout-target genes
    not in landmark).
    Y_crispr columns are the full CRISPR matrix (so target panels can
    be picked from any gene).
    """
    crispr_path = DATA_ROOT / "DepMap" / "CRISPRGeneEffect.csv"
    expr_path = DATA_ROOT / "DepMap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    model_path = DATA_ROOT / "DepMap" / "Model.csv"
    for p in [crispr_path, expr_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — see data/README.md for downloads.")

    logger.info("loading CRISPR…")
    crispr = pd.read_csv(crispr_path, index_col=0)
    crispr.columns = [c.split(" (")[0] for c in crispr.columns]

    logger.info("loading expression…")
    expr = pd.read_csv(expr_path, index_col=0)
    expr.columns = [c.split(" (")[0] for c in expr.columns]

    model = pd.read_csv(model_path)
    model = model[["ModelID", "OncotreeLineage"]].dropna().rename(
        columns={"OncotreeLineage": "tissue"})

    common = sorted(set(crispr.index) & set(expr.index) & set(model["ModelID"]))
    logger.info("CRISPR ∩ Expression ∩ Lineage: %d cell lines", len(common))
    crispr_m = crispr.loc[common]
    expr_m = expr.loc[common]
    tissue = model.set_index("ModelID").loc[common, "tissue"]

    gene_set = load_gene_set(gene_list)
    if extra_genes:
        gene_set = gene_set | set(extra_genes)
    keep = sorted(gene_set & set(expr_m.columns))
    expr_m = expr_m[keep]
    logger.info("expression: %d cells × %d genes (gene list: %s%s)",
                *expr_m.shape, gene_list,
                f" + {len(extra_genes)} extra" if extra_genes else "")
    return expr_m, crispr_m, tissue


def select_depmap_target_panel(crispr: pd.DataFrame, gene_list: str,
                               top_n: int = 200) -> list:
    """Top-N genes by CRISPR essentiality std, restricted to gene_list ∩
    CRISPR.

    Picks the genes where knockout-effect actually varies across cell
    lines (the genes where there's something to predict).
    """
    gene_set = load_gene_set(gene_list)
    candidates = [g for g in gene_set if g in crispr.columns]
    stds = crispr[candidates].std().dropna().sort_values(ascending=False)
    return stds.head(top_n).index.tolist()


# ====================================================== CTRPv2
def load_ctrpv2_response(min_cells_per_drug: int = 100,
                          min_tissues_per_drug: int = 8) -> pd.DataFrame:
    """Long-form drug-response DataFrame:
        cellosaurus_id, drug_name, LN_IC50, tissue, cell_line_name
    """
    cands = [
        DATA_ROOT / "CTRPv2" / "CTRPv2.csv",
        DATA_ROOT / "CTRPv2" / "ctrpv2.csv",
        DATA_ROOT / "GDSC1" / "ctrpv2.csv",
        DATA_ROOT / "CCLE" / "CCLE.csv",
    ]
    src = next((p for p in cands if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"No CTRPv2 file found. Tried: {cands}")
    logger.info("loading response from %s", src)
    df = pd.read_csv(src, low_memory=False)
    df.columns = df.columns.str.strip()

    if "LN_IC50" not in df.columns and "LN_IC50_curvecurator" in df.columns:
        df = df.rename(columns={"LN_IC50_curvecurator": "LN_IC50"})

    rename_map = {
        "DRUG_NAME": "drug_name", "Drug name": "drug_name",
        "CELL_LINE_NAME": "cell_line_name", "Cell line name": "cell_line_name",
        "TISSUE": "tissue", "Tissue": "tissue",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = ["drug_name", "LN_IC50", "tissue", "cell_line_name"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{src} missing columns {missing}")

    if "cellosaurus_id" not in df.columns:
        names_path = DATA_ROOT / "CCLE" / "cell_line_names.csv"
        if names_path.exists():
            names = pd.read_csv(names_path)
            names.columns = names.columns.str.strip()
            df = df.merge(names[["cell_line_name", "cellosaurus_id"]],
                           on="cell_line_name", how="left")

    df = df.dropna(subset=["LN_IC50", "drug_name", "tissue"])
    counts = df.groupby("drug_name").agg(n=("LN_IC50", "size"),
                                          n_t=("tissue", "nunique"))
    keep = counts[(counts["n"] >= min_cells_per_drug)
                  & (counts["n_t"] >= min_tissues_per_drug)].index
    df = df[df["drug_name"].isin(keep)]
    logger.info("drugs after coverage filter (≥%d cells, ≥%d tissues): %d",
                min_cells_per_drug, min_tissues_per_drug, len(keep))
    return df


def load_ccle_expression(gene_list: str = "panel_ctrpv2",
                          extra_genes=None) -> pd.DataFrame:
    """Expression matrix keyed by cellosaurus_id, restricted to `gene_list`.
    log2-transformed.

    extra_genes: iterable of HUGO symbols to UNION into the panel (kept if
    present in the expression matrix). Use to add drug-target genes when
    using a sparse gene_list like landmark_genes.
    """
    cands = [DATA_ROOT / "CCLE" / "gene_expression.csv",
             DATA_ROOT / "CTRPv2" / "gene_expression.csv"]
    src = next((p for p in cands if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"No expression file. Tried: {cands}")
    logger.info("loading expression from %s", src)
    expr = pd.read_csv(src)
    expr.columns = expr.columns.str.strip()
    if "cellosaurus_id" not in expr.columns:
        raise ValueError("expression file is missing cellosaurus_id")
    gene_set = load_gene_set(gene_list)
    if extra_genes:
        gene_set = gene_set | set(extra_genes)
    keep = sorted(gene_set & set(expr.columns) - {"cellosaurus_id"})
    out = expr.set_index("cellosaurus_id")[keep]
    out = out[~out.index.duplicated(keep="first")]
    out = np.log2(out + 1)
    logger.info("expression: %d cells × %d genes (gene list: %s%s)",
                *out.shape, gene_list,
                f" + {len(extra_genes)} extra" if extra_genes else "")
    return out


def build_ctrpv2_per_drug(drug: str, response: pd.DataFrame,
                          expr: pd.DataFrame
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = response[response["drug_name"] == drug].dropna(
        subset=["cellosaurus_id"])
    sub = sub[sub["cellosaurus_id"].isin(expr.index)]
    X = expr.loc[sub["cellosaurus_id"]].values.astype(np.float32)
    y = sub["LN_IC50"].values.astype(np.float32)
    T = sub["tissue"].values.astype(str)
    return X, y, T
