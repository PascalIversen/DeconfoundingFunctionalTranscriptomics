"""Table `tab:other_explainers`: contamination@50 and attribution eta^2 for four
additional explainers, pooled over the shared item subset (25 DepMap targets, 23
CTRPv2 drugs), before and after residualization.

    python make_table_other_explainers.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROWS = [("shap", "SHAP"),
        ("int_grad", "Integrated Gradients"),
        ("lime", "LIME"),
        ("lrp_eps", r"LRP ($\epsilon$-rule)"),
        ("lrp_ab", r"LRP ($\alpha\beta$-rule)")]


def main() -> None:
    d = pd.concat([pd.read_csv(HERE / "results" / f"xai_comparison_{ds}.csv")
                   for ds in ("depmap", "ctrpv2")], ignore_index=True)
    g = d.groupby(["condition", "xai"])[["contam50", "eta2"]].mean()

    def cell(condition: str, xai: str, col: str) -> str:
        v = g.loc[(condition, xai), col] if (condition, xai) in g.index else float("nan")
        return f"{v:.3f}" if pd.notna(v) else "--"

    lines = []
    for xai, label in ROWS:
        cells = [cell(cond, xai, col) for cond in ("marginal", "residualized")
                 for col in ("contam50", "eta2")]
        lines.append("    " + label + " & " + " & ".join(cells) + r" \\")
    out = HERE / "other_explainers_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out} (rows for Table tab:other_explainers)")


if __name__ == "__main__":
    main()
