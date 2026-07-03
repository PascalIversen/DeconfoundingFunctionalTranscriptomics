"""Measure the two arms of the confounding DAG on the *raw* data the models
train on — no trained `.npz` required.

            Tissue (or another candidate confounder)
           /                                         \\
          v                                           v
   outcome (IC50 / essentiality)                gene expression
        Arm 2                                        Arm 1

For each dataset (DepMap CRISPR essentiality, CTRPv2 drug IC50 on CCLE
expression) this:

  Arm 1  Tissue → expression
    - per-gene η²(X)                       (shared eta_squared, no group filter
                                            → identical to the pipeline's eta_sq)
    - linear tissue classifier lift        (recover tissue from one sample)
    - within- vs across-tissue correlation (cells resemble their own tissue)
    - PCA coloured by tissue
  Arm 2  Tissue → outcome
    - per-outcome η²(y) + ANOVA over the same target/drug panel the models use

Both arms also get a **shuffled-tissue null**: `--n-perm` (default 20) replicates
in which the per-cell tissue label is permuted (one global shuffle per replicate,
shared by both arms) and η² recomputed, written to confounding_expr_eta2_null.csv
and confounding_outcome_eta2_null.csv. fig05 overlays this as the null histogram.

Plus the fig05c "Tissue *or something else*" comparison: η²(outcome) **and**
η²(expression) across 7 `Model.csv` confounders. DepMap reads them directly;
CTRPv2 joins `Model.csv` onto its cell lines via `RRID` (= cellosaurus_id) and
runs on the matched subset (matched count printed). High-cardinality confounders
trivially explain more variance, so for this comparison only we drop groups with
< `--min-group` samples (the headline tissue figures use no group filter, to
match the trained pipeline exactly). fig05c also gets a **per-item permuted
floor**: each item's own label multiset is shuffled `--n-perm` times *on its own
cells* and η² recomputed → per-item mean in confounding_confounder_compare_null.csv.
fig05c plots Δη² = real − this floor, so every gene/outcome is debiased against
its own chance level (which scales with that item's group sizes), correcting the
upward η² bias that many small groups would otherwise create.

All inputs come through `training/shared/data.py`, so the diagnostics see the
data exactly as the models do (LN_IC50, log2(x+1) CCLE expression, the
per-dataset feature panels `panel_depmap` / `panel_ctrpv2`, tissue from the
CTRPv2 response). Pass `--gene-list paccmann_pharma` to reproduce the original
shared-panel run.

Usage (from figure_confounding_evidence/):
    python compute_confounding.py
    python compute_confounding.py --datasets depmap
    python compute_confounding.py --top-n-targets 100
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

from confounding_utils import (                                       # noqa: E402
    eta_squared, outcome_anova, pca_by_tissue, tissue_classifier_lift,
    tissue_eta_squared, tissue_eta_squared_cols, within_across_tissue_corr,
)
from shared.data import (                                             # noqa: E402
    DATA_ROOT, load_ccle_expression, load_ctrpv2_response,
    load_depmap_crispr, select_depmap_target_panel,
)

CONFOUNDERS_DEFAULT = [
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
    "Sex", "PrimaryOrMetastasis", "GrowthPattern", "OnboardedMedia",
]
ALPHA = 0.05


# ──────────────────────────────────────────────────────────── helpers
def _min_group_mask(labels: np.ndarray, min_group: int) -> np.ndarray:
    """Boolean mask keeping samples whose group has ≥ `min_group` members."""
    labels = np.asarray(labels).astype(str)
    vc = pd.Series(labels).value_counts()
    keep = set(vc[vc >= min_group].index)
    return np.array([lbl in keep for lbl in labels])


def _model_df() -> pd.DataFrame:
    """DepMap Model.csv (the source of every candidate confounder)."""
    return pd.read_csv(DATA_ROOT / "DepMap" / "Model.csv")


# ──────────────────────────────────────────────────────────── dataset builders
def build_depmap(args):
    """→ (X_df[ModelID×genes], tissue_arr, outcomes[(gene, Series)], conf_df)."""
    expr, crispr, tissue = load_depmap_crispr(gene_list=args.gene_list)
    X_df = expr
    tissue_arr = tissue.reindex(X_df.index).astype(str).values

    panel = (args.targets if args.targets else
             select_depmap_target_panel(crispr, args.gene_list,
                                        top_n=args.top_n_targets))
    outcomes = []
    for g in panel:
        if g not in crispr.columns:
            continue
        s = crispr[g].dropna()
        s = s[s.index.isin(X_df.index)]
        if len(s) >= args.min_cells:
            outcomes.append((g, s))

    model = _model_df().drop_duplicates("ModelID").set_index("ModelID")
    conf_df = model.reindex(X_df.index)
    return X_df, tissue_arr, outcomes, conf_df


def build_ctrpv2(args):
    """→ (X_df[cellosaurus_id×genes], tissue_arr, outcomes[(drug, Series)], conf_df).

    Tissue (Arm 1 / 2 headline) comes from the CTRPv2 response — the labels the
    models train on. The other confounders come from Model.csv joined via RRID.
    """
    expr = load_ccle_expression(gene_list=args.gene_list)
    resp = load_ctrpv2_response(min_cells_per_drug=args.min_cells,
                                min_tissues_per_drug=args.min_tissues)

    tmap = (resp.dropna(subset=["cellosaurus_id", "tissue"])
            .drop_duplicates("cellosaurus_id")
            .set_index("cellosaurus_id")["tissue"])
    ids = expr.index.intersection(tmap.index)
    X_df = expr.loc[ids]
    tissue_arr = tmap.loc[ids].astype(str).values

    outcomes = []
    for drug, sub in resp.groupby("drug_name"):
        sub = (sub.dropna(subset=["cellosaurus_id", "LN_IC50"])
               .drop_duplicates("cellosaurus_id"))
        sub = sub[sub["cellosaurus_id"].isin(X_df.index)]
        if len(sub) >= args.min_cells:
            outcomes.append((drug, pd.Series(sub["LN_IC50"].values,
                                             index=sub["cellosaurus_id"].values)))

    # RRID join: Model.csv keyed by RRID (= cellosaurus_id), aligned to expr.
    model = _model_df().dropna(subset=["RRID"]).drop_duplicates("RRID")
    conf_df = model.set_index("RRID").reindex(X_df.index)
    n_matched = int(conf_df["OncotreeLineage"].notna().sum())
    print(f"  CTRPv2 ↔ Model.csv RRID join: {n_matched}/{len(X_df)} "
          f"cell lines matched")
    return X_df, tissue_arr, outcomes, conf_df


# ──────────────────────────────────────────────────────────── arms
def run_arm1(ds: str, X_df: pd.DataFrame, tissue_arr: np.ndarray,
             args, acc: dict, perms: list | None = None) -> dict:
    """Tissue → expression. No group-size filter (matches the pipeline)."""
    T = np.asarray(tissue_arr).astype(str)

    eta_X = tissue_eta_squared(X_df, T)             # == pipeline eta_sq for tissue
    for g, v in eta_X.items():
        acc["expr"].append({"dataset": ds, "gene": g, "eta2_X": float(v)})

    # Shuffled-tissue null (fig05 overlay): permute the per-cell tissue label and
    # recompute η²(X) per gene. The vectorised kernel is checked against the
    # reference on the *real* labels first, so the null uses fig05's definition.
    if perms:
        Xv = X_df.values
        md = float(np.max(np.abs(eta_X.values - tissue_eta_squared_cols(Xv, T))))
        assert md < 1e-9, f"vectorised η²(X) disagrees with tissue_eta_squared ({md:.2e})"
        genes = list(X_df.columns)
        for p, Tp in enumerate(perms):
            for g, v in zip(genes, tissue_eta_squared_cols(Xv, Tp)):
                acc["expr_null"].append({"dataset": ds, "perm": p, "gene": g,
                                         "eta2_X": float(v)})

    within, across, _, mwu_p = within_across_tissue_corr(
        X_df.values, T, seed=args.seed)
    for r in within:
        acc["corr"].append({"dataset": ds, "kind": "within", "r": float(r)})
    for r in across:
        acc["corr"].append({"dataset": ds, "kind": "across", "r": float(r)})

    pcs, vr, _ = pca_by_tissue(X_df.values, T, n_comps=2)
    for sid, t, pc in zip(X_df.index, T, pcs):
        acc["pca"].append({"dataset": ds, "sample_id": str(sid), "tissue": t,
                           "PC1": float(pc[0]), "PC2": float(pc[1])})

    t0 = time.time()
    try:
        clf = tissue_classifier_lift(X_df.values, T, seed=args.seed,
                                     max_iter=args.clf_max_iter)
    except ValueError as e:
        print(f"  [{ds}] classifier skipped: {e}")
        clf = {"accuracy": np.nan, "accuracy_std": np.nan, "chance": np.nan,
               "lift": np.nan, "lift_std": np.nan,
               "n_classes": len(np.unique(T))}
    print(f"  [{ds}] arm1: {X_df.shape[0]} cells × {X_df.shape[1]} genes, "
          f"{len(np.unique(T))} tissues | clf lift={clf['lift']:.1f}±{clf['lift_std']:.1f} "
          f"(acc={clf['accuracy']:.2f}±{clf['accuracy_std']:.2f} "
          f"vs chance={clf['chance']:.2f}) "
          f"| within r̃={np.median(within):.2f} across r̃={np.median(across):.2f} "
          f"({time.time()-t0:.0f}s clf)")

    return {
        "n_cells": int(X_df.shape[0]), "n_genes": int(X_df.shape[1]),
        "n_tissues": int(len(np.unique(T))),
        "clf_accuracy": clf["accuracy"], "clf_accuracy_std": clf["accuracy_std"],
        "clf_chance": clf["chance"],
        "clf_lift": clf["lift"], "clf_lift_std": clf["lift_std"],
        "clf_n_classes": clf["n_classes"],
        "median_eta_X": float(eta_X.median()),
        "frac_eta_X_gt030": float((eta_X > 0.30).mean()),
        "pc1_var_ratio": float(vr[0]), "pc2_var_ratio": float(vr[1]),
        "within_median_r": float(np.median(within)) if len(within) else np.nan,
        "across_median_r": float(np.median(across)) if len(across) else np.nan,
        "corr_mwu_p": float(mwu_p),
    }


def run_arm2(ds: str, X_df: pd.DataFrame, tissue_arr: np.ndarray,
             outcomes: list, acc: dict, perms: list | None = None) -> dict:
    """Tissue → outcome over the model's target/drug panel.

    Welch's F-test is run once per outcome, so the panel is a *family* of
    `m` simultaneous hypotheses. We report both the uncorrected count
    (p < ALPHA) and the **Bonferroni-corrected** count, which controls the
    family-wise error rate without assuming anything about the dependence
    between tests (conservative — the outcomes share tissue structure and are
    correlated). The family is one dataset's panel; the adjusted p-value is
    `min(p·m, 1)` and an outcome is Bonferroni-significant iff `p < ALPHA / m`.
    """
    tissue_map = pd.Series(np.asarray(tissue_arr).astype(str), index=X_df.index)
    rows, etas, valid = [], [], []
    for item, s in outcomes:
        common = s.index.intersection(X_df.index)
        y = s.loc[common].values
        T = tissue_map.loc[common].values
        if len(np.unique(T)) < 2:
            continue
        r = outcome_anova(y, T)
        rows.append({
            "dataset": ds, "item": item, "eta2_y": r["eta2"],
            "anova_p": r["p_anova"], "n_cells": int(len(y)),
            "n_tissues": int(len(np.unique(T))),
        })
        etas.append(r["eta2"])
        valid.append((item, common, y))      # fixed on real labels → null aligns

    n = len(rows)
    # Bonferroni family size = the outcomes that actually carry a finite Welch p.
    finite_p = np.array([np.isfinite(d["anova_p"]) for d in rows], dtype=bool)
    m = int(finite_p.sum())
    bonf_alpha = ALPHA / m if m else np.nan
    sig = sig_bonf = 0
    for d in rows:
        p = d["anova_p"]
        ok = bool(np.isfinite(p))
        d["sig"] = bool(ok and p < ALPHA)
        d["anova_p_bonf"] = float(min(p * m, 1.0)) if ok and m else float("nan")
        d["sig_bonf"] = bool(ok and m and p < bonf_alpha)
        sig += d["sig"]
        sig_bonf += d["sig_bonf"]
    acc["outcome"].extend(rows)

    # Shuffled-tissue null (fig05 overlay): one *global* permutation of the
    # cell→tissue map per replicate, sliced to each outcome's cells — so a cell
    # keeps one shuffled tissue across all outcomes, mirroring the real map.
    if perms:
        for p, Tp in enumerate(perms):
            tmap_p = pd.Series(Tp, index=X_df.index)
            for item, common, y in valid:
                en = eta_squared(y, tmap_p.loc[common].values)
                acc["outcome_null"].append({"dataset": ds, "perm": p,
                                            "item": item, "eta2_y": float(en)})

    print(f"  [{ds}] arm2: {n} outcomes | median η²(y)={np.median(etas):.3f} "
          f"| {sig}/{n} ANOVA-significant (p<{ALPHA}) "
          f"| {sig_bonf}/{n} Bonferroni (p<{bonf_alpha:.2e}=α/{m})")
    return {
        "n_outcomes": n,
        "median_eta_y": float(np.median(etas)) if n else np.nan,
        "frac_outcomes_sig": float(sig / n) if n else np.nan,
        "n_outcomes_sig": int(sig),
        # Bonferroni-corrected — the significance we report from now on.
        "frac_outcomes_sig_bonf": float(sig_bonf / n) if n else np.nan,
        "n_outcomes_sig_bonf": int(sig_bonf),
        "n_tests_bonf": m,
        "bonf_alpha": float(bonf_alpha) if m else np.nan,
    }


def run_confounder_compare(ds: str, X_df: pd.DataFrame, conf_df: pd.DataFrame,
                           outcomes: list, args, acc: dict,
                           rng=None, n_perm: int = 0) -> None:
    """η²(expression) and η²(outcome) across the candidate confounders (fig05c).

    Drops groups with < args.min_group samples so high-cardinality confounders
    aren't credited for singleton groups.

    Per-item permuted floor (fig05c debias): when `n_perm > 0`, each item's own
    label multiset is shuffled `n_perm` times *on that item's own cells* and η²
    recomputed; the per-item mean is written to `acc["cmp_null"]`. fig05c plots
    Δη² = real − this floor, so every gene/outcome is debiased against *its own*
    chance level (which scales with that item's group sizes), never a pooled
    average across items.
    """
    do_null = bool(n_perm) and rng is not None
    for conf in args.confounders:
        if conf not in conf_df.columns:
            print(f"  [{ds}] confounder '{conf}' absent — skipped")
            continue
        labels = conf_df[conf].reindex(X_df.index)
        valid = labels.notna().values
        idx = X_df.index[valid]
        lab = labels[valid].astype(str).values
        keep = _min_group_mask(lab, args.min_group)
        idx_k, lab_k = idx[keep], lab[keep]
        if len(np.unique(lab_k)) < 2:
            print(f"  [{ds}] confounder '{conf}': <2 usable groups — skipped")
            continue

        # expression arm
        eta_X = tissue_eta_squared(X_df.loc[idx_k], lab_k)
        for g, v in eta_X.items():
            acc["cmp"].append({"dataset": ds, "confounder": conf,
                               "arm": "expression", "item": g,
                               "eta2": float(v), "n": int(len(idx_k))})
        if do_null:
            Xk = X_df.loc[idx_k].values
            # the vectorised kernel must equal the scalar η² on real labels, so
            # the floor uses fig05's exact definition (same guard as run_arm1).
            md = float(np.max(np.abs(eta_X.values
                                     - tissue_eta_squared_cols(Xk, lab_k))))
            assert md < 1e-9, f"vectorised η²(X) disagrees ({md:.2e})"
            floor = np.zeros(Xk.shape[1])
            for _ in range(n_perm):
                floor += tissue_eta_squared_cols(Xk, rng.permutation(lab_k))
            floor /= n_perm
            for g, v in zip(eta_X.index, floor):
                acc["cmp_null"].append({"dataset": ds, "confounder": conf,
                                        "arm": "expression", "item": g,
                                        "eta2_null": float(v)})
        # outcome arm
        keep_set = set(idx_k)
        n_out = 0
        for item, s in outcomes:
            common = [i for i in s.index if i in keep_set]
            if len(common) < args.min_group:
                continue
            lab_o = labels.reindex(common).astype(str).values
            m = _min_group_mask(lab_o, args.min_group)
            if len(np.unique(lab_o[m])) < 2:
                continue
            yv = s.loc[common].values[m]
            r = outcome_anova(yv, lab_o[m], min_samples_per_group=args.min_group)
            acc["cmp"].append({"dataset": ds, "confounder": conf,
                               "arm": "outcome", "item": item,
                               "eta2": r["eta2"], "n": int(m.sum())})
            if do_null:
                lab_real = lab_o[m]                  # same cells, same group sizes
                floor = sum(eta_squared(yv, rng.permutation(lab_real))
                            for _ in range(n_perm)) / n_perm
                acc["cmp_null"].append({"dataset": ds, "confounder": conf,
                                        "arm": "outcome", "item": item,
                                        "eta2_null": float(floor)})
            n_out += 1
        print(f"  [{ds}] {conf:22s}: expr η̃²={eta_X.median():.3f} "
              f"(n={len(idx_k)}, {len(np.unique(lab_k))} groups) | "
              f"outcomes used={n_out}")


# ──────────────────────────────────────────────────────────── driver
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["depmap", "ctrpv2"],
                   choices=["depmap", "ctrpv2"])
    p.add_argument("--gene-list", default=None,
                   help="Feature panel for *both* datasets. If omitted, each "
                        "dataset uses its own panel (--gene-list-depmap / "
                        "--gene-list-ctrpv2).")
    p.add_argument("--gene-list-depmap", default="panel_depmap",
                   help="DepMap feature panel when --gene-list is unset.")
    p.add_argument("--gene-list-ctrpv2", default="panel_ctrpv2",
                   help="CTRPv2 feature panel when --gene-list is unset.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Explicit DepMap target panel (else top-N by std).")
    p.add_argument("--top-n-targets", type=int, default=200)
    p.add_argument("--confounders", nargs="+", default=CONFOUNDERS_DEFAULT)
    p.add_argument("--min-cells", type=int, default=100)
    p.add_argument("--min-tissues", type=int, default=8)
    p.add_argument("--min-group", type=int, default=5,
                   help="fig05c only: drop confounder groups smaller than this.")
    p.add_argument("--clf-max-iter", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=20,
                   help="shuffled-tissue null replicates for the fig05 η² "
                        "overlay (0 disables the null).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    acc = {"expr": [], "outcome": [], "pca": [], "corr": [], "cmp": [],
           "cmp_null": [], "expr_null": [], "outcome_null": []}
    summaries = []
    rng = np.random.default_rng(args.seed)   # drives the shuffled-tissue null

    builders = {"depmap": build_depmap, "ctrpv2": build_ctrpv2}
    override = args.gene_list           # explicit --gene-list applies to both
    per_ds_default = {"depmap": args.gene_list_depmap,
                      "ctrpv2": args.gene_list_ctrpv2}
    for ds in args.datasets:
        # --gene-list (if given) overrides both; else use the dataset's panel.
        args.gene_list = override if override else per_ds_default[ds]
        print(f"=== {ds} (gene panel: {args.gene_list}) ===", flush=True)
        X_df, tissue_arr, outcomes, conf_df = builders[ds](args)
        # Shared permutations for both arms: one global cell→tissue shuffle per
        # replicate, so Arm 1 and Arm 2 see the same null draws.
        T_real = np.asarray(tissue_arr).astype(str)
        perms = [rng.permutation(T_real) for _ in range(args.n_perm)]
        s1 = run_arm1(ds, X_df, tissue_arr, args, acc, perms)
        s2 = run_arm2(ds, X_df, tissue_arr, outcomes, acc, perms)
        run_confounder_compare(ds, X_df, conf_df, outcomes, args, acc,
                               rng=rng, n_perm=args.n_perm)
        summaries.append({"dataset": ds, **s1, **s2})

    out = args.out_dir
    writes = [
        ("confounding_expr_eta2.csv", acc["expr"]),
        ("confounding_outcome_eta2.csv", acc["outcome"]),
        ("confounding_expr_eta2_null.csv", acc["expr_null"]),
        ("confounding_outcome_eta2_null.csv", acc["outcome_null"]),
        ("confounding_pca.csv", acc["pca"]),
        ("confounding_tissue_corr.csv", acc["corr"]),
        ("confounding_confounder_compare.csv", acc["cmp"]),
        ("confounding_confounder_compare_null.csv", acc["cmp_null"]),
    ]
    for name, rows in writes:
        pd.DataFrame(rows).to_csv(out / name, index=False)
        print(f"wrote {out / name}  ({len(rows)} rows)")
    pd.DataFrame(summaries).to_csv(out / "confounding_summary.csv", index=False)
    print(f"wrote {out / 'confounding_summary.csv'}  ({len(summaries)} rows)")


if __name__ == "__main__":
    main()
