"""Dispatcher-friendly per-item preds generator.

Differences vs gen_preds_parallel.py:
  * --items-file <path>  required; one item per line (drug/target)
  * --seed N             configurable (replaces hardcoded seed=1)
  * --union-targets      union extra genes into the panel:
                           DepMap : reads targets_depmap.txt (target self-
                                    expression added to landmark)
                           CTRPv2 : reads targets_ctrpv2.txt (all drug-target
                                    genes added to landmark)
  * Output dir keyed by seed: {dataset}_seed{N}_preds/

Same npz layout as the figure scripts expect.

Designed to be sshd to many cluster nodes in parallel, each with its own
items-file slice. Skips items whose npz already exists (resume-safe).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # local shared/ package

from shared.orchestrator import run_cv                                          # noqa: E402
from shared.eval import (contamination_at_k, target_rank, hits_at_k,            # noqa: E402
                         tissue_eta_squared)
from shared.data import (load_depmap_crispr, load_ccle_expression,              # noqa: E402
                         load_ctrpv2_response, build_ctrpv2_per_drug,
                         load_drug_target_map, _norm_drug)

METHODS = ["marginal", "residualized", "irm", "within_tissue"]
G = {}


def compute_eta_y(y, T):
    grand = float(y.mean()); ss = float(((y - grand) ** 2).sum())
    if ss < 1e-12:
        return 0.0
    return float(sum(int((T == t).sum()) * (y[T == t].mean() - grand) ** 2
                     for t in np.unique(T)) / ss)


def _save_npz(path, gene_cols, y, T, res):
    np.savez_compressed(
        path, gene_cols=np.array(gene_cols), y_true=y.astype(np.float32),
        tissue=T.astype("U64"),
        yhat_marginal=res["marginal"]["yhat_raw"].astype(np.float32),
        yhat_residualized=res["residualized"]["yhat_raw"].astype(np.float32),
        yhat_irm=res["irm"]["yhat_raw"].astype(np.float32),
        yhat_within_tissue=res["within_tissue"]["yhat_raw"].astype(np.float32),
        yhat_baseline=res["baseline"]["yhat_raw"].astype(np.float32),
        yhat_between_tissue=res["between_tissue"]["yhat_raw"].astype(np.float32),
        shap_marginal=res["marginal"]["shap_full"].astype(np.float16),
        shap_residualized=res["residualized"]["shap_full"].astype(np.float16),
        shap_irm=res["irm"]["shap_full"].astype(np.float16),
        shap_within_tissue=res["within_tissue"]["shap_full"].astype(np.float16),
        shap_between_tissue=res["between_tissue"]["shap_full"].astype(np.float16))


def work_depmap(target):
    import torch; torch.set_num_threads(1)
    EXPR, CRISPR, TISSUE, ETA, GC, PRED, SEED = (
        G["expr"], G["crispr"], G["tissue"], G["eta"], G["gene_cols"],
        G["preds_dir"], G["seed"])
    if target not in CRISPR.columns:
        return None
    y_full = CRISPR[target].dropna()
    common = EXPR.index.intersection(y_full.index)
    X = EXPR.loc[common].values
    y = y_full.loc[common].values
    T = TISSUE.loc[common].values
    if len(y) < 100:
        return None
    eta_y = compute_eta_y(y, T)
    res = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                 batch_size=64, shap_background=100, irm_lambda=1.0,
                 seed=SEED)
    _save_npz(PRED / f"{target}.npz", GC, y, T, res)
    rows = []
    for m in METHODS:
        imp = res[m]["importance"]
        rows.append(dict(seed=SEED, target=target, n_cells=int(len(y)),
                         n_tissues=int(len(np.unique(T))),
                         eta_y=round(eta_y, 4), method=m,
                         self_rank=target_rank(imp, target),
                         contam_50=round(contamination_at_k(imp, ETA, k=50), 4),
                         r2_raw=round(res[m]["r2_raw"], 4),
                         pearson_r=round(res[m]["pearson_r"], 4)))
    return rows


def work_ctrpv2(drug):
    import torch; torch.set_num_threads(1)
    EXPR, RESP, DT, ETA, GC, GSET, PRED, SEED = (
        G["expr"], G["resp"], G["dt"], G["eta"], G["gene_cols"],
        G["gene_set"], G["preds_dir"], G["seed"])
    X, y, T = build_ctrpv2_per_drug(drug, RESP, EXPR)
    if len(y) < 100 or len(np.unique(T)) < 8:
        return None
    raw_targets = DT.get(drug) or DT.get(_norm_drug(drug), [])
    targets_in = [g for g in raw_targets if g in GSET]
    res = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                 batch_size=64, shap_background=100, irm_lambda=1.0,
                 seed=SEED)
    safe = drug.replace("/", "_").replace(" ", "_")
    _save_npz(PRED / f"{safe}.npz", GC, y, T, res)
    rows = []
    for m in METHODS:
        imp = res[m]["importance"]
        rk = imp.rank(ascending=False, method="min")
        sr = (int(min(int(rk.loc[g]) for g in targets_in if g in rk.index))
              if any(g in rk.index for g in targets_in) else np.nan)
        rows.append(dict(seed=SEED, drug=drug, n_cells=int(len(y)),
                         n_tissues=int(len(np.unique(T))),
                         targets_in_geneset=",".join(targets_in),
                         method=m, self_rank_best=sr,
                         hits_at_50=(hits_at_k(imp, targets_in, 50)
                                     if targets_in else np.nan),
                         contam_50=round(contamination_at_k(imp, ETA, k=50), 4),
                         r2_raw=round(res[m]["r2_raw"], 4),
                         pearson_r=round(res[m]["pearson_r"], 4)))
    return rows


def _read_items(path: Path) -> list:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["depmap", "ctrpv2"], required=True)
    ap.add_argument("--gene-list", default="landmark_genes")
    ap.add_argument("--items-file", type=Path, required=True,
                    help="one item per line (target gene for depmap, "
                         "drug name for ctrpv2)")
    ap.add_argument("--targets-file", type=Path, default=None,
                    help="optional file of HUGO symbols to union into the "
                         "expression panel (--union-targets implies this)")
    ap.add_argument("--union-targets", action="store_true",
                    help="union {dataset}-target genes into expression panel; "
                         "looks for targets_{dataset}.txt next to items-file "
                         "unless --targets-file is set")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--nproc", type=int, default=12)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = args.out_dir / f"{args.dataset}_seed{args.seed}_preds"
    preds_dir.mkdir(exist_ok=True)
    G["preds_dir"] = preds_dir
    G["seed"] = args.seed

    extra_genes = None
    if args.union_targets:
        tp = args.targets_file or args.items_file.parent / f"targets_{args.dataset}.txt"
        if not tp.exists():
            sys.exit(f"--union-targets but {tp} missing; "
                     "run build_item_lists.py first")
        extra_genes = _read_items(tp)

    t0 = time.time()
    if args.dataset == "depmap":
        expr, crispr, tissue = load_depmap_crispr(
            gene_list=args.gene_list, extra_genes=extra_genes)
        G.update(expr=expr, crispr=crispr, tissue=tissue,
                 gene_cols=list(expr.columns),
                 eta=tissue_eta_squared(expr, tissue.values))
        work = work_depmap
    else:
        resp = load_ctrpv2_response(min_cells_per_drug=100,
                                     min_tissues_per_drug=8)
        expr = load_ccle_expression(gene_list=args.gene_list,
                                     extra_genes=extra_genes)
        dt = load_drug_target_map()
        union = resp[resp["cellosaurus_id"].isin(expr.index)].drop_duplicates(
            "cellosaurus_id")
        Xu = expr.loc[union["cellosaurus_id"]]
        Tu = union.set_index("cellosaurus_id").loc[Xu.index, "tissue"].values
        G.update(expr=expr, resp=resp, dt=dt, gene_cols=list(expr.columns),
                 gene_set=set(expr.columns), eta=tissue_eta_squared(Xu, Tu))
        work = work_ctrpv2

    items = _read_items(args.items_file)
    # skip already-done
    pending = [it for it in items if not (
        preds_dir / f"{(it.replace('/', '_').replace(' ', '_'))}.npz").exists()]
    print(f"[{args.dataset} seed={args.seed}] {len(pending)}/{len(items)} "
          f"pending; gene_list={args.gene_list} "
          f"({len(G['gene_cols'])} genes); "
          f"nproc={args.nproc}; loaded in {time.time()-t0:.0f}s", flush=True)

    if not pending:
        print(f"[{args.dataset} seed={args.seed}] nothing to do", flush=True)
        return

    ctx = mp.get_context("fork")
    all_rows = []
    done = 0
    with ctx.Pool(args.nproc) as pool:
        for r in pool.imap_unordered(work, pending, chunksize=1):
            done += 1
            if r:
                all_rows.extend(r)
            if done % 10 == 0:
                print(f"  {done}/{len(pending)} ({time.time()-t0:.0f}s)",
                      flush=True)
    if all_rows:
        out_csv = args.out_dir / f"{args.dataset}_seed{args.seed}.csv"
        df = pd.DataFrame(all_rows)
        if out_csv.exists():
            df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
        df.to_csv(out_csv, index=False)
    print(f"[{args.dataset} seed={args.seed}] DONE {done} items in "
          f"{time.time()-t0:.0f}s -> {preds_dir}", flush=True)


if __name__ == "__main__":
    main()
