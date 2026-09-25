"""Appendix table for the model-stage penalty-weight sweeps.

Puts all three model-stage methods on one grid so the stage is treated
symmetrically: IRM (from `run_irm_sweep.py`) alongside DANN and AD-AE (from
`run_adv_sweep.py`), swept over the same lambdas, scored on the same
metrics, over the same items.

Writes two tables:
  adv_sweep_table.tex        the deconfounding / accuracy metrics
  adv_diagnostics_table.tex  whether the adversary was actually beaten --
                             the evidence that a null result reflects the
                             method rather than a broken implementation

    python make_table_adv.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
METRICS = ["contam_50", "attr_eta2", "within_r"]
DATASETS = ("depmap", "ctrpv2")


def load(pattern: str, ds: str) -> pd.DataFrame:
    parts = glob.glob(str(HERE / "results" / f"{pattern}_{ds}*.csv"))
    parts = [p for p in parts if "ganinopt" not in p]
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def load_irm(ds: str) -> pd.DataFrame:
    """IRM sweep rows, run on the same 20-item subset and feature panel as
    DANN and AD-AE (`run_irm_sweep.py --n-items 20`, its default), so all
    three model-stage methods are directly comparable in one table."""
    return load("irm_sweep", ds)


def load_all(ds: str) -> pd.DataFrame:
    return pd.concat([d for d in (load_irm(ds), load("adv_sweep", ds))
                      if not d.empty], ignore_index=True)


def cell(df: pd.DataFrame, method: str, lam, metric: str) -> str:
    if df.empty:
        return "--"
    m = df.method == method
    if lam is not None:
        m &= np.isclose(df.lam.astype(float), lam)
    v = df.loc[m, metric].dropna()
    return f"{v.mean():.3f}" if len(v) else "--"


def main():
    data = {ds: load_all(ds) for ds in DATASETS}
    for ds, df in data.items():
        if df.empty:
            print(f"WARNING: no sweep results for {ds}")
        else:
            print(f"{ds}: {df.method.nunique()} methods, "
                  f"{df.item.nunique()} items, methods={sorted(df.method.unique())}")

    rows = [(r"Marginal (reference)", "marginal", None)]
    for lam in LAMBDAS[1:]:
        rows.append((f"\\quad IRM, $\\lambda={lam:g}$", "irm", lam))
    for lam in LAMBDAS:
        tag = " (= marginal)" if lam == 0 else ""
        rows.append((f"\\quad DANN, $\\lambda={lam:g}${tag}", "dann", lam))
    for lam in LAMBDAS:
        tag = " (= plain AE)" if lam == 0 else ""
        rows.append((f"\\quad AD-AE, $\\lambda={lam:g}${tag}", "adae", lam))

    def line(label, method, lam):
        cells = [label] + [cell(data[ds], method, lam, c)
                           for ds in DATASETS for c in METRICS]
        return "    " + " & ".join(cells) + r" \\"

    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \begin{tabular}{l ccc ccc}", r"    \toprule",
         r"    & \multicolumn{3}{c}{DepMap (essentiality)}"
         r" & \multicolumn{3}{c}{CTRPv2 (drug response)} \\",
         r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"    & contam@50$\downarrow$ & $\eta^2_{\mathrm{attr}}\downarrow$"
         r" & $r_{\mathrm{wt}}\uparrow$"
         r" & contam@50$\downarrow$ & $\eta^2_{\mathrm{attr}}\downarrow$"
         r" & $r_{\mathrm{wt}}\uparrow$ \\",
         r"    \midrule", line(*rows[0]), r"    \midrule"]
    prev = None
    for label, method, lam in rows[1:]:
        if prev is not None and method != prev:
            L.append(r"    \addlinespace")
        L.append(line(label, method, lam))
        prev = method
    L += [r"    \bottomrule", r"  \end{tabular}"]
    L += [
        r"  \caption{\textbf{Penalty weight does not rescue any model-stage "
        r"method.} All three model-stage deconfounding methods swept over the "
        r"same penalty-weight grid, on the same item subset, scored on the "
        r"same metrics. DANN's $\lambda$ is the ceiling of Ganin et al.'s "
        r"annealing schedule $\lambda_p = 2/(1+e^{-10p})-1$; AD-AE's is the "
        r"weight on the adversarial term. Both source papers use $\lambda=1$. "
        r"The $\lambda=0$ rows are controls: DANN reduces exactly to the "
        r"marginal model, and AD-AE to a plain autoencoder plus downstream "
        r"head, so they separate each method's training protocol from its "
        r"adversarial component. contamination@50 and $\eta^2_{\mathrm{attr}}$ "
        r"lower is better; within-tissue $r_{\mathrm{wt}}$ higher is better. "
        r"Cells are means over items.}",
        r"  \label{tab:adv_sweep}", r"\end{table}"]
    (HERE / "adv_sweep_table.tex").write_text("\n".join(L) + "\n")

    # ---- diagnostics: was the adversary actually defeated? --------------
    D = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \begin{tabular}{l cc cc}", r"    \toprule",
         r"    & \multicolumn{2}{c}{DepMap} & \multicolumn{2}{c}{CTRPv2} \\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"    & confounder acc. & chance & confounder acc. & chance \\",
         r"    \midrule"]

    def diag(ds, method, lam, col, ref):
        df = data[ds]
        if df.empty or col not in df.columns:
            return "--"
        m = (df.method == method) & np.isclose(df.lam.astype(float), lam)
        v = df.loc[m, col].dropna()
        r = df.loc[m, ref].dropna() if ref in df.columns else pd.Series(dtype=float)
        return (f"{v.mean():.3f}" if len(v) else "--",
                f"{r.mean():.3f}" if len(r) else "--")

    for method, acc_col in (("dann", "domain_acc_val"), ("adae", "adv_acc_val")):
        name = "DANN" if method == "dann" else "AD-AE"
        for lam in LAMBDAS:
            cells = []
            for ds in DATASETS:
                a, c = diag(ds, method, lam, acc_col,
                            "majority_domain_frac" if method == "dann"
                            else "majority_class_frac")
                cells += [a, c]
            D.append(f"    {name}, $\\lambda={lam:g}$ & " + " & ".join(cells) + r" \\")
        D.append(r"    \addlinespace")
    D += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{\textbf{The adversaries were beaten; the attributions "
          r"were not cleaned.} Held-out accuracy of each method's own "
          r"confounder predictor (DANN's domain classifier, AD-AE's "
          r"adversary) at the end of training, against the majority-tissue "
          r"rate it would achieve by ignoring the representation. DANN "
          r"reaches chance already at $\lambda=1$; AD-AE's adversary is "
          r"reduced but stays above chance at every $\lambda$, so a weak "
          r"effect on the attribution metrics in Table~\ref{tab:adv_sweep} "
          r"is unambiguous for DANN and cannot fully rule out an "
          r"undertrained adversary for AD-AE.}",
          r"  \label{tab:adv_diagnostics}", r"\end{table}"]
    (HERE / "adv_diagnostics_table.tex").write_text("\n".join(D) + "\n")

    print("\n".join(L))
    print()
    print("\n".join(D))
    print("\nwrote adv_sweep_table.tex and adv_diagnostics_table.tex")


if __name__ == "__main__":
    main()
