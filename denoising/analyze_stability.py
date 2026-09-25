"""Turn the raw stability curves + half-mean matrices into the response tables.

Reads (from compute_stability.py):
    results/stability_curves.csv
    results/halfmeans_{ds}.npz

Produces (all in denoising/results/):
    item_stability.csv      per (dataset, item, method): r1, split-half r at
                            n = n_cells//2, Spearman-Brown prediction, implied
                            full-explanation reliability, r_wt, subset flags
    stability_summary.csv   per (dataset, method) aggregates + SB fit error +
                            stability-vs-r_wt correlation + a61 vs excluded
    delta_split_half.csv    reliability of the Fig-3 GSEA input: the global
                            delta percentile vector computed from disjoint
                            cell halves, correlated per method
    contamination_half.csv  contamination@K per method computed from each half
                            (and from all cells, validated against the
                            published fig02 CSV)
    auroc_tau_subsets.csv   Fig-A2 AUROC(tau) per method on the full item set
                            and the a61 / marginal / all4 subsets (per-item
                            AUROC from the all-cells mean importance; 'full'
                            validated against the published auroc_tau.csv)
    fig04b_subset.csv       Fig-4b breast-subtype specificity summary per
                            subset variant (row filter on the published CSV)
    eta_{ds}.csv            cached expression tissue-eta^2 per gene
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
METHODS = ["marginal", "residualized", "irm", "within_tissue", "dann", "adae"]
DECONF = ["residualized", "irm", "within_tissue", "dann", "adae"]
KS = [10, 50, 100]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
VARIANTS = ["full", "a61", "marginal", "all4"]
ITEMCOL = {"depmap": "target", "ctrpv2": "drug"}


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def eta_for(ds: str, genes: np.ndarray) -> np.ndarray:
    """Expression tissue-eta^2 per gene, via the paper's own fig02 code, cached."""
    cache = RES / f"eta_{ds}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]
        return s.reindex(genes).values
    at = _import(ROOT / "fig02_ci_distribution" / "compute_auroc_tau.py", "at_mod")
    eta = at.eta2_expr(ROOT / "data", list(genes), ds, ROOT / "training")
    pd.Series(eta, index=genes, name="eta2").to_csv(cache)
    return np.asarray(eta, dtype=float)


def subset_flags(ds: str) -> pd.DataFrame:
    f = ROOT / "well_predicted_subset" / "results" / f"{ds}_wt_items.csv"
    df = pd.read_csv(f).rename(columns={ITEMCOL[ds]: "item"})
    return df[["item", "in_a61", "in_marginal", "in_all4"]]


def rwt_per_item(ds: str) -> pd.DataFrame:
    """Median-over-seeds within-tissue r per (item, method), paper convention
    (residualized = baseline-stripped wt_pearson)."""
    df = pd.read_csv(ROOT / "table_performance" / "results" / "per_item_pearson.csv")
    df = df[df.dataset == ds]
    med = (df.groupby(["item", "method"])["wt_pearson"].median()
             .rename("r_wt").reset_index())
    # within-tissue SHAP explains the marginal model (per_item_pearson has no
    # within_tissue rows because yhat_within_tissue == yhat_marginal)
    wt = med[med.method == "marginal"].assign(method="within_tissue")
    return pd.concat([med, wt], ignore_index=True)


def sb(n, r1):
    return n * r1 / (1.0 + (n - 1.0) * r1)


# ---------------------------------------------------------------- per-item
def item_tables(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_rows, sum_rows = [], []
    for ds, dcur in curves.groupby("dataset"):
        flags = subset_flags(ds)
        rwt = rwt_per_item(ds)
        per = (dcur.groupby(["item", "method", "n_cells", "n"])
               [["pearson", "spearman"]].mean().reset_index())
        half_n = {it: nc // 2 for it, nc in
                  dcur.groupby("item")["n_cells"].first().items()}
        it_df = []
        for (it, m), g in per.groupby(["item", "method"]):
            g = g.sort_values("n")
            r1p = float(g.loc[g.n == 1, "pearson"].iloc[0])
            hn = half_n[it]
            gh = g.loc[g.n == hn]
            rhp = float(gh["pearson"].iloc[0])
            rhs = float(gh["spearman"].iloc[0])
            it_df.append(dict(dataset=ds, item=it, method=m,
                              n_cells=int(g.n_cells.iloc[0]),
                              r1_pearson=r1p,
                              rhalf_pearson=rhp, rhalf_spearman=rhs,
                              sb_pred_rhalf=sb(hn, r1p) if r1p > 0 else np.nan,
                              full_reliability_sb=2 * rhp / (1 + rhp)))
        it_df = pd.DataFrame(it_df).merge(flags, on="item", how="left")
        it_df = it_df.merge(rwt, on=["item", "method"], how="left")
        item_rows.append(it_df)

        for m, g in it_df.groupby("method"):
            gper = per[per.method == m]
            # SB fit error on the dataset-mean curve, predicted from mean r1
            mean_curve = gper.groupby("n")["pearson"].mean()
            r1bar = mean_curve.loc[1]
            pred = sb(mean_curve.index.values.astype(float), r1bar)
            sb_rmse = float(np.sqrt(np.mean((mean_curve.values - pred) ** 2)))
            ok = g.dropna(subset=["r_wt"])
            rho, rho_p = (spearmanr(ok.rhalf_spearman, ok.r_wt)
                          if len(ok) > 5 else (np.nan, np.nan))
            ina = g[g.in_a61 == True]          # noqa: E712
            out = g[g.in_a61 == False]         # noqa: E712
            mw = (mannwhitneyu(ina.rhalf_spearman, out.rhalf_spearman)[1]
                  if len(ina) and len(out) else np.nan)
            sum_rows.append(dict(
                dataset=ds, method=m, n_items=len(g),
                mean_r1=g.r1_pearson.mean(),
                mean_rhalf_pearson=g.rhalf_pearson.mean(),
                mean_rhalf_spearman=g.rhalf_spearman.mean(),
                min_rhalf_spearman=g.rhalf_spearman.min(),
                sb_rmse=sb_rmse,
                mean_full_reliability_sb=g.full_reliability_sb.mean(),
                spearman_stability_vs_rwt=rho, p_stability_vs_rwt=rho_p,
                mean_rhalf_a61=ina.rhalf_spearman.mean(),
                mean_rhalf_excluded=out.rhalf_spearman.mean(),
                mw_p_a61_vs_excluded=mw))
    return pd.concat(item_rows, ignore_index=True), pd.DataFrame(sum_rows)


# ------------------------------------------------------------- global delta
def pct_vector(M: np.ndarray) -> np.ndarray:
    """Cross-item mean importance percentile (mirrors fig03 aggregate_percentiles)."""
    P = M.shape[1]
    ok = ~np.isnan(M).all(axis=1)
    ranks = np.stack([rankdata(np.nan_to_num(M[j])) / P
                      for j in np.where(ok)[0]])
    return ranks.mean(0)


def delta_table(hm: dict) -> pd.DataFrame:
    rows = []
    for ds, d in hm.items():
        pcts = {}
        for m in METHODS:
            if f"halfA_{m}" not in d:
                continue
            pcts[m] = {t: pct_vector(d[f"{t}_{m}"]) for t in ("halfA", "halfB")}
        for m in DECONF:
            if m not in pcts:
                continue
            dA = pcts[m]["halfA"] - pcts["marginal"]["halfA"]
            dB = pcts[m]["halfB"] - pcts["marginal"]["halfB"]
            pear = float(np.corrcoef(dA, dB)[0, 1])
            spear = float(spearmanr(dA, dB)[0])
            # sign agreement among genes in the top decile of |delta| (union)
            k = max(1, int(0.1 * len(dA)))
            top = np.union1d(np.argsort(-np.abs(dA))[:k],
                             np.argsort(-np.abs(dB))[:k])
            sign_agree = float((np.sign(dA[top]) == np.sign(dB[top])).mean())
            rows.append(dict(dataset=ds, method=m, pearson=pear,
                             spearman=spear, top_decile_sign_agree=sign_agree,
                             n_genes=len(dA)))
    return pd.DataFrame(rows)


# ----------------------------------------------------- contamination halves
def topk_contamination(imp: np.ndarray, genes: np.ndarray, L: set, k: int) -> float:
    s = pd.Series(imp, index=genes).dropna()
    return len(set(s.nlargest(k).index) & L) / k


def contamination_table(hm: dict) -> pd.DataFrame:
    rows = []
    for ds, d in hm.items():
        genes = d["genes"]
        eta = eta_for(ds, genes)
        L = set(genes[np.nan_to_num(eta) > 0.30])
        pub = pd.read_csv(ROOT / "fig02_ci_distribution" / "results" /
                          f"{ds}_contamination_at_k.csv")
        for m in METHODS:
            if f"full_{m}" not in d:
                continue
            for tag in ("halfA", "halfB", "full"):
                M = d[f"{tag}_{m}"]
                for k in KS:
                    vals = [topk_contamination(M[j], genes, L, k)
                            for j in range(M.shape[0])
                            if not np.isnan(M[j]).all()]
                    row = dict(dataset=ds, method=m, half=tag, K=k,
                               contamination=float(np.mean(vals)), n=len(vals))
                    if tag == "full":
                        ref = pub[(pub.method == m) & (pub.K == k)].contamination.mean()
                        row["published"] = float(ref)
                        row["abs_diff_vs_published"] = abs(row["contamination"] - ref)
                    rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------- AUROC-tau subsets
def auroc_table(hm: dict) -> pd.DataFrame:
    rows = []
    pub = pd.read_csv(ROOT / "fig02_ci_distribution" / "results" / "auroc_tau.csv")
    for ds, d in hm.items():
        genes = d["genes"]
        items = pd.Series(np.asarray(d["items"], dtype=str), name="item")
        eta = eta_for(ds, genes)
        ok_eta = np.isfinite(eta)
        flags = subset_flags(ds).set_index("item")
        members = {"full": set(items)}
        for v in ("a61", "marginal", "all4"):
            members[v] = set(flags.index[flags[f"in_{v}"] == True])  # noqa: E712
        for m in METHODS + ["between_tissue"]:
            if f"full_{m}" not in d:
                continue
            M = d[f"full_{m}"]
            per_item = {}
            for j, it in enumerate(items):
                imp = M[j]
                o = ok_eta & np.isfinite(imp)
                for tau in TAUS:
                    mk = (eta[o] > tau).astype(int)
                    if 3 <= mk.sum() < o.sum():
                        per_item.setdefault(tau, {})[it] = roc_auc_score(mk, imp[o])
            for v in VARIANTS:
                for tau in TAUS:
                    vals = np.array([a for it, a in per_item.get(tau, {}).items()
                                     if it in members[v]])
                    if not len(vals):
                        continue
                    row = dict(dataset=ds, variant=v, method=m, tau=tau,
                               auroc=vals.mean(),
                               sem=vals.std() / np.sqrt(len(vals)), n=len(vals))
                    if v == "full":
                        ref = pub[(pub.dataset == ds) & (pub.method == m)
                                  & (np.isclose(pub.tau, tau))]
                        if len(ref):
                            row["published"] = float(ref.auroc.iloc[0])
                            row["abs_diff_vs_published"] = abs(vals.mean()
                                                               - ref.auroc.iloc[0])
                    rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- fig4b subset
# Same gap definition as fig04b_breast_subtypes/plot_breast_subtypes.py:
# breast AUROC minus the mean of these 7 named non-epithelial tissues' AUROCs
# (each tissue's AUROC itself averaged over drugs first), not a row-level
# mean over every non-breast (drug, tissue) pair.
NON_EPI = ["Blood", "Lymph", "Muscle", "Bone", "Soft Tissue",
           "Nervous System", "Brain"]
MIN_DRUGS_PER_TISSUE = 20


def fig04b_table() -> pd.DataFrame:
    f = ROOT / "fig04b_breast_subtypes" / "results" / "subtype_tissue_specificity.csv"
    if not f.exists():
        print("fig04b CSV missing; skipping")
        return pd.DataFrame()
    df = pd.read_csv(f)
    flags = subset_flags("ctrpv2").set_index("item")
    rows = []
    for v in VARIANTS:
        sub = (df if v == "full" else
               df[df.drug.isin(flags.index[flags[f"in_{v}"] == True])])  # noqa: E712
        tissue_means = (sub.groupby(["tissue", "method"])
                           .agg(auroc=("auroc", "mean"), n_drugs=("auroc", "count"))
                           .reset_index())
        tissue_means = tissue_means[tissue_means.n_drugs >= MIN_DRUGS_PER_TISSUE]
        pivot = tissue_means.pivot(index="tissue", columns="method", values="auroc")
        non_epi = [t for t in NON_EPI if t in pivot.index]
        for m in pivot.columns:
            if "Breast" not in pivot.index:
                continue
            breast = pivot.loc["Breast", m]
            other = pivot.loc[non_epi, m].mean()
            rows.append(dict(variant=v, method=m,
                             n_drugs=sub[sub.method == m].drug.nunique(),
                             breast_auroc=breast, other_auroc=other,
                             breast_minus_other=breast - other))
    return pd.DataFrame(rows)


def main():
    curves = pd.read_csv(RES / "stability_curves.csv")
    hm = {}
    for ds in ("depmap", "ctrpv2"):
        f = RES / f"halfmeans_{ds}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            hm[ds] = {k: d[k] for k in d.files}
            hm[ds]["genes"] = np.asarray(hm[ds]["genes"], dtype=str)

    items, summary = item_tables(curves)
    items.to_csv(RES / "item_stability.csv", index=False)
    summary.to_csv(RES / "stability_summary.csv", index=False)
    print("== stability_summary ==")
    print(summary.round(3).to_string(index=False))

    dt = delta_table(hm)
    dt.to_csv(RES / "delta_split_half.csv", index=False)
    print("\n== delta split-half ==")
    print(dt.round(3).to_string(index=False))

    ct = contamination_table(hm)
    ct.to_csv(RES / "contamination_half.csv", index=False)
    v = ct.dropna(subset=["abs_diff_vs_published"]) if "abs_diff_vs_published" in ct else ct
    print(f"\ncontamination: max |full - published| = "
          f"{v.abs_diff_vs_published.max():.4f}" if len(v) else "no validation")

    at = auroc_table(hm)
    at.to_csv(RES / "auroc_tau_subsets.csv", index=False)
    if "abs_diff_vs_published" in at:
        print(f"auroc_tau: max |full - published| = "
              f"{at.abs_diff_vs_published.max():.4f}")

    f4 = fig04b_table()
    if len(f4):
        f4.to_csv(RES / "fig04b_subset.csv", index=False)
        print("\n== fig04b breast specificity by subset ==")
        print(f4.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
