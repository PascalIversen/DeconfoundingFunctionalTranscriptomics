"""Headline comparison of the model-stage baselines.

Pulls together, per dataset, everything needed to answer the question "does an
adversarial model-stage method deconfound the attributions?":

  * deconfounding      contamination@50, per-gene attribution eta^2
  * predictive cost    overall and within-tissue Pearson r
  * did it work at all adversary / domain-classifier accuracy against the
                       majority-tissue rate it would get for free

The last block is the one that makes a null result interpretable: if the
adversary was reduced to chance and the attributions still carry tissue,
that is a fact about the method, not about the implementation.

Also writes the headline table used in the paper (Table `tab:adv_baselines`):

    python scripts/summarize_adversarial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from shared.preds_io import collect_extra_summaries  # noqa: E402

RESULTS = ROOT / "results"
DATASETS = ("depmap", "ctrpv2")
ORDER = ["marginal", "residualized", "irm", "dann", "adae", "within_tissue"]
STAGE = {"marginal": "(none)", "residualized": "data", "irm": "model",
         "dann": "model", "adae": "model", "within_tissue": "attribution"}
TABLE_ROWS = [("marginal", "Marginal"), ("residualized", "Residualized"),
              ("irm", "IRM"), ("dann", "DANN"), ("adae", "AD-AE")]


def diagnostics(ds: str, seed: int = 1) -> pd.DataFrame:
    """Extras summary rows — the only home of the adversary diagnostics."""
    ex = collect_extra_summaries(RESULTS, ds, seed)
    if ex.empty:
        return ex
    item_col = "target" if ds == "depmap" else "drug"
    if item_col in ex.columns:
        ex = ex.rename(columns={item_col: "item"})
    return ex.drop_duplicates(["item", "method"])


# NB: the metrics below deliberately come from the *computed* result CSVs,
# not from the generation-time `results/{ds}_seed{s}.csv` summaries. Those
# summaries were appended to across several re-runs of the original
# pipeline, so each (item, method) appears 2-3 times with different values,
# and averaging them would not reproduce any published number. The computed
# CSVs are recomputed from the npz bundles by the scripts in
# `scripts/recompute_with_extras.sh` and are what the paper reports.
def contamination(ds: str, k: int = 50) -> pd.Series:
    f = ROOT / "fig02_ci_distribution" / "results" / f"{ds}_contamination_at_k.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(f)
    return d[d.K == k].groupby("method")["contamination"].mean()


def attr_eta2(ds: str) -> pd.Series:
    """Mean per-gene attribution eta^2 by method (Fig 2's y-axis)."""
    f = ROOT / "fig02_ci_distribution" / "results" / f"{ds}_attr_eta2.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(f)
    return d.groupby("method")["eta2"].mean()


def pearson(ds: str, seed=None) -> pd.DataFrame:
    f = ROOT / "table_performance" / "results" / "per_item_pearson.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d[d.dataset == ds]
    if seed is not None:
        d = d[d.seed == seed]
    return d.groupby("method")[["pearson", "wt_pearson"]].mean()


def foldsym_rwt() -> pd.DataFrame:
    """Method-neutral within-tissue r (see well_predicted_subset/), the r_wt
    convention Table 1 and this table both report."""
    f = ROOT / "well_predicted_subset" / "results" / "fold_symmetric_summary.csv"
    if not f.exists():
        return pd.DataFrame(columns=["dataset", "method", "wt_foldsym_mean"])
    return pd.read_csv(f)[["dataset", "method", "wt_foldsym_mean"]]


def headline_table() -> dict[str, pd.DataFrame]:
    """Per-dataset (method x [contam@50, attr_eta2, pearson, r_wt]).

    contam@50 / attr_eta2 are seed-1-only (the only seed with full per-cell SHAP);
    pearson r is the 5-seed average, matching Table 1. r_wt is the fold-symmetric
    convention (well_predicted_subset/fold_symmetric_rwt.py), also matching Table 1.
    """
    rwt = foldsym_rwt()
    out = {}
    for ds in DATASETS:
        eta, cont, pr = attr_eta2(ds), contamination(ds), pearson(ds, seed=None)
        rwt_ds = rwt[rwt.dataset == ds].set_index("method")["wt_foldsym_mean"]
        out[ds] = pd.DataFrame({
            "contam50": {m: cont.get(m, np.nan) for m, _ in TABLE_ROWS},
            "eta2": {m: eta.get(m, np.nan) for m, _ in TABLE_ROWS},
            "r": {m: pr["pearson"].get(m, np.nan) for m, _ in TABLE_ROWS},
            "r_wt": {m: rwt_ds.get(m, np.nan) for m, _ in TABLE_ROWS},
        })
    return out


def write_latex_table(out_path: Path) -> None:
    """Write Table `tab:adv_baselines`: DANN/AD-AE alongside the main-text methods."""
    tables = headline_table()

    def cell(ds: str, method: str, col: str) -> str:
        v = tables[ds].loc[method, col] if method in tables[ds].index else np.nan
        return f"{v:.3f}" if pd.notna(v) else "--"

    lines = []
    for method, label in TABLE_ROWS:
        cells = [cell(ds, method, col) for ds in DATASETS
                  for col in ("contam50", "eta2", "r", "r_wt")]
        lines.append("    " + label + " & " + " & ".join(cells) + r" \\")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path} (rows for Table tab:adv_baselines)")


def main():
    for ds in DATASETS:
        eta, cont, pr = attr_eta2(ds), contamination(ds), pearson(ds, seed=1)
        df = diagnostics(ds)
        methods = [m for m in ORDER
                   if m in set(eta.index) | set(cont.index) | set(pr.index)]
        if not methods:
            print(f"\n### {ds}: no computed results yet")
            continue
        out = pd.DataFrame({
            "stage": {m: STAGE.get(m, "?") for m in methods},
            "pearson_r": {m: pr["pearson"].get(m, np.nan) for m in methods},
            "wt_r": {m: pr["wt_pearson"].get(m, np.nan) for m in methods},
            "contam@50": {m: cont.get(m, np.nan) for m in methods},
            "attr_eta2": {m: eta.get(m, np.nan) for m in methods},
        }).loc[methods]
        print(f"\n### {ds}  (seed 1)")
        print(out.round(4).to_string())
        missing = [m for m in ("dann", "adae") if m not in out.index
                   or out.loc[m].drop("stage").isna().all()]
        if missing:
            print(f"  !! {missing} absent from the computed CSVs -- run "
                  f"scripts/recompute_with_extras.sh after the extras land")

        # --- was the adversary actually beaten? -------------------------
        if df.empty or "method" not in df.columns:
            print("  (no extras diagnostics for this dataset yet)")
            continue
        diag = []
        if "dann_domain_acc_val" in df.columns:
            d = df[df.method == "dann"]
            diag.append(("DANN domain classifier",
                         d.dann_domain_acc_val.mean(),
                         d.dann_majority_domain_frac.mean()))
        if "adae_adv_acc_val" in df.columns:
            a = df[df.method == "adae"]
            diag.append(("AD-AE adversary",
                         a.adae_adv_acc_val.mean(),
                         a.adae_majority_class_frac.mean()))
        if diag:
            print("\n  confounder recoverable from the representation?")
            for name, acc, chance in diag:
                verdict = "at chance" if acc <= chance * 1.5 else "still decodable"
                print(f"    {name:24s} held-out acc {acc:.3f} "
                      f"vs majority {chance:.3f}  -> {verdict}")
        a = df[df.method == "adae"]
        if "adae_diverged" in df.columns and len(a):
            print(f"    AD-AE diverged on {100*a.adae_diverged.mean():.1f}% of items; "
                  f"adversary CE {a.adae_adv_ce_val_final.mean():.2f} "
                  f"vs chance {a.adae_adv_ce_chance.mean():.2f}")
        if "adae_head_en_val_mse" in df.columns and len(a):
            print(f"    AD-AE head: elastic-net val MSE "
                  f"{a.adae_head_en_val_mse.mean():.4f} vs ridge "
                  f"{a.adae_head_ridge_val_mse.mean():.4f} "
                  f"(head choice is not what limits AD-AE)")

    # --- multi-seed accuracy table, if present --------------------------
    pi = ROOT / "table_performance" / "results" / "per_item_pearson.csv"
    if pi.exists():
        d = pd.read_csv(pi)
        if d.method.nunique() > 4:
            print("\n### 5-seed predictive accuracy (mean over items and seeds)")
            piv = (d.groupby(["dataset", "method"])["pearson"].mean()
                     .unstack(0).reindex([m for m in ORDER + ["baseline"]
                                          if m in set(d.method)]))
            print(piv.round(4).to_string())

    write_latex_table(Path(__file__).resolve().parent / "adv_baselines_table.tex")


if __name__ == "__main__":
    main()
