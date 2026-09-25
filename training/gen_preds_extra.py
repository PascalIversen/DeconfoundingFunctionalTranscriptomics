"""Per-item predictions for the *additional* model-stage baselines.

Adds the two adversarial deconfounding methods —
DANN (domain-adversarial gradient reversal) and AD-AE (adversarial
deconfounding autoencoder) — to items whose marginal / residualized / IRM /
within-tissue predictions already exist.

Why a separate runner instead of re-running `gen_preds_v2.py`:
the CV fold structure depends only on `(n_cells, seed)` and the
train/validation carve only on `(fold, seed)`, so a run restricted to the
new methods lands on *exactly* the same splits as the published run. The
comparison stays paired and like-for-like without paying to retrain the
four original methods, and the published Zenodo bundle is left untouched.

Output: `{out_dir}/{dataset}_seed{seed}_extra/{item}.npz`, holding
`gene_cols`, `y_true`, `tissue`, `yhat_<method>` and — depending on flags —
either the full per-cell `shap_<method>` (float16, for the attribution
figures) or just the per-gene mean|SHAP| `imp_<method>` (float32, which is
all the seed-averaged and multi-seed analyses need, at ~1/1000 the size).

The figure scripts pick these up transparently via `shared.preds_io`.

Usage
-----
    # seed 1, full per-cell SHAP (feeds Fig 2 / 3 / 4b)
    python gen_preds_extra.py --dataset depmap --seed 1 \
        --items-file items_depmap.txt --out-dir ../results --nproc 24

The feature panel defaults to `panel_{dataset}`, which is the exact panel the
published bundle used. Do NOT rebuild it as "landmark UNION targets": that no
longer reproduces it (DepMap gains NDC80, 1072 genes vs 1071), and since the
extras are merged with the bundle gene-for-gene, a mismatched panel silently
corrupts every downstream comparison. The run aborts on a mismatch against
`panel_reference_{dataset}.txt`.

    # seeds 2-5, importances only (feeds Fig 4a seed-average)
    python gen_preds_extra.py --dataset depmap --seed 2 --save-imp --no-shap ...

    # seeds 2-5, predictions only (feeds Tables 1 / A1 / A2)
    python gen_preds_extra.py --dataset ctrpv2 --seed 2 --no-shap ...
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(HERE))

from shared.orchestrator import run_cv                                          # noqa: E402
from shared.eval import (contamination_at_k, target_rank, hits_at_k,            # noqa: E402
                         tissue_eta_squared)
from shared.data import (load_depmap_crispr, load_ccle_expression,              # noqa: E402
                         load_ctrpv2_response, build_ctrpv2_per_drug,
                         load_drug_target_map, _norm_drug)

EXTRA_METHODS = ["dann", "adae"]
G = {}


def _save_npz(path, gene_cols, y, T, res, methods, save_shap, save_imp):
    payload = {
        "gene_cols": np.array(gene_cols),
        "y_true": y.astype(np.float32),
        "tissue": T.astype("U64"),
    }
    for m in methods:
        payload[f"yhat_{m}"] = res[m]["yhat_raw"].astype(np.float32)
        if save_shap:
            payload[f"shap_{m}"] = res[m]["shap_full"].astype(np.float16)
        if save_imp:
            payload[f"imp_{m}"] = res[m]["importance"].values.astype(np.float32)
    np.savez_compressed(path, **payload)


def _diag_row(res, m):
    """Flatten the per-fold adversary diagnostics into fold-averaged scalars.

    These are what let the manuscript state whether the adversary actually
    removed tissue information, rather than asserting it.
    """
    ds = res[m].get("diagnostics") or []
    if not ds:
        return {}
    keys = [k for k in ds[0] if k != "fold"]
    return {f"{m}_{k}": float(np.nanmean([d[k] for d in ds])) for k in keys}


def work_depmap(target):
    import torch; torch.set_num_threads(1)
    EXPR, CRISPR, TISSUE, ETA, GC, PRED, SEED, CFG = (
        G["expr"], G["crispr"], G["tissue"], G["eta"], G["gene_cols"],
        G["preds_dir"], G["seed"], G["cfg"])
    if target not in CRISPR.columns:
        return None
    y_full = CRISPR[target].dropna()
    common = EXPR.index.intersection(y_full.index)
    X = EXPR.loc[common].values
    y = y_full.loc[common].values
    T = TISSUE.loc[common].values
    if len(y) < 100:
        return None
    res = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                 batch_size=64, shap_background=100, seed=SEED,
                 methods=EXTRA_METHODS,
                 sample_ids=[str(c) for c in common],
                 dann_kwargs=CFG["dann_kwargs"],
                 adae_kwargs=CFG["adae_kwargs"],
                 adae_cache_dir=CFG["adae_cache_dir"],
                 compute_shap=CFG["compute_shap"])
    _save_npz(PRED / f"{target}.npz", GC, y, T, res, EXTRA_METHODS,
              CFG["save_shap"], CFG["save_imp"])
    rows = []
    for m in EXTRA_METHODS:
        row = dict(seed=SEED, target=target, n_cells=int(len(y)),
                   n_tissues=int(len(np.unique(T))), method=m,
                   r2_raw=round(res[m]["r2_raw"], 4),
                   pearson_r=round(res[m]["pearson_r"], 4))
        # Attribution metrics only exist if SHAP was computed; without it
        # `importance` is all-NaN and ranking it would raise.
        if CFG["compute_shap"]:
            imp = res[m]["importance"]
            row["self_rank"] = target_rank(imp, target)
            row["contam_50"] = round(contamination_at_k(imp, ETA, k=50), 4)
        row.update(_diag_row(res, m))
        rows.append(row)
    return rows


def work_ctrpv2(drug):
    import torch; torch.set_num_threads(1)
    EXPR, RESP, DT, ETA, GC, GSET, PRED, SEED, CFG = (
        G["expr"], G["resp"], G["dt"], G["eta"], G["gene_cols"],
        G["gene_set"], G["preds_dir"], G["seed"], G["cfg"])
    X, y, T, ids = build_ctrpv2_per_drug(drug, RESP, EXPR, return_ids=True)
    if len(y) < 100 or len(np.unique(T)) < 8:
        return None
    raw_targets = DT.get(drug) or DT.get(_norm_drug(drug), [])
    targets_in = [g for g in raw_targets if g in GSET]
    res = run_cv(X, y, T, GC, n_folds=5, epochs=80, patience=12,
                 batch_size=64, shap_background=100, seed=SEED,
                 methods=EXTRA_METHODS, sample_ids=ids,
                 dann_kwargs=CFG["dann_kwargs"],
                 adae_kwargs=CFG["adae_kwargs"],
                 adae_cache_dir=CFG["adae_cache_dir"],
                 compute_shap=CFG["compute_shap"])
    safe = drug.replace("/", "_").replace(" ", "_")
    _save_npz(PRED / f"{safe}.npz", GC, y, T, res, EXTRA_METHODS,
              CFG["save_shap"], CFG["save_imp"])
    rows = []
    for m in EXTRA_METHODS:
        row = dict(seed=SEED, drug=drug, n_cells=int(len(y)),
                   n_tissues=int(len(np.unique(T))),
                   targets_in_geneset=",".join(targets_in), method=m,
                   r2_raw=round(res[m]["r2_raw"], 4),
                   pearson_r=round(res[m]["pearson_r"], 4))
        if CFG["compute_shap"]:
            imp = res[m]["importance"]
            rk = imp.rank(ascending=False, method="min")
            row["self_rank_best"] = (
                int(min(int(rk.loc[g]) for g in targets_in if g in rk.index))
                if any(g in rk.index for g in targets_in) else np.nan)
            row["hits_at_50"] = (hits_at_k(imp, targets_in, 50)
                                 if targets_in else np.nan)
            row["contam_50"] = round(contamination_at_k(imp, ETA, k=50), 4)
        row.update(_diag_row(res, m))
        rows.append(row)
    return rows


def _read_items(path: Path) -> list:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["depmap", "ctrpv2"], required=True)
    ap.add_argument("--gene-list", default=None,
                    help="defaults to panel_{dataset}, the exact recorded "
                         "panel of the published bundle")
    ap.add_argument("--panel-reference", type=Path, default=None,
                    help="file of expected gene symbols, one per line; "
                         "defaults to training/panel_reference_{dataset}.txt")
    ap.add_argument("--items-file", type=Path, required=True)
    ap.add_argument("--targets-file", type=Path, default=None)
    ap.add_argument("--union-targets", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--suffix", default="extra",
                    help="output dir is {dataset}_seed{seed}_{suffix}")
    # --- method configuration -------------------------------------------
    ap.add_argument("--lambda-dann", type=float, default=1.0,
                    help="ceiling of Ganin's lambda ramp (paper default 1.0)")
    ap.add_argument("--lambda-adae", type=float, default=1.0,
                    help="AD-AE adversarial weight (paper default 1.0)")
    ap.add_argument("--dann-optimizer", choices=["adam", "ganin"],
                    default="adam",
                    help="'ganin' = SGD+momentum with the paper's mu_p "
                         "schedule (faithfulness sensitivity check)")
    ap.add_argument("--adae-head-hidden", type=int, default=0,
                    help="0 = linear head on the frozen embedding "
                         "(elastic-net analogue); >0 = one hidden layer")
    ap.add_argument("--adae-cache-dir", type=Path, default=None,
                    help="shared on-disk cache of AD-AE encoders; the "
                         "encoder is phenotype-free so items with identical "
                         "cell sets reuse it (big win on DepMap)")
    # --- output configuration -------------------------------------------
    ap.add_argument("--no-shap", action="store_true",
                    help="skip SHAP entirely (predictions only)")
    ap.add_argument("--save-imp", action="store_true",
                    help="store per-gene mean|SHAP| instead of per-cell SHAP")
    args = ap.parse_args()
    if args.gene_list is None:
        args.gene_list = f"panel_{args.dataset}"

    # --save-imp still needs SHAP *computed*; it only avoids storing the
    # per-cell matrix. --no-shap alone skips the SHAP passes entirely.
    save_imp = args.save_imp
    save_shap = not args.no_shap and not args.save_imp
    compute_shap = save_shap or save_imp

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = args.out_dir / f"{args.dataset}_seed{args.seed}_{args.suffix}"
    preds_dir.mkdir(exist_ok=True)
    if args.adae_cache_dir:
        args.adae_cache_dir.mkdir(parents=True, exist_ok=True)

    G["preds_dir"] = preds_dir
    G["seed"] = args.seed
    G["cfg"] = {
        "dann_kwargs": {"lambda_max": args.lambda_dann,
                        "optimizer": args.dann_optimizer},
        "adae_kwargs": {"lambda_adv": args.lambda_adae,
                        "head_hidden": args.adae_head_hidden},
        "adae_cache_dir": (str(args.adae_cache_dir)
                           if args.adae_cache_dir else None),
        "compute_shap": compute_shap,
        "save_shap": save_shap,
        "save_imp": save_imp,
    }

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

    # ---- fail fast on a feature-panel mismatch --------------------------
    # The extras are merged with the published bundle gene-for-gene, so an
    # off-by-one panel makes every downstream comparison silently wrong.
    # Rebuilding the panel as "landmark UNION targets" at runtime does NOT
    # reproduce it (it now picks up NDC80 on DepMap, 1072 vs 1071 genes), so
    # the extras runner pins the recorded panel and checks it here rather
    # than discovering the mismatch after hours of compute.
    ref_path = args.panel_reference or (HERE / f"panel_reference_{args.dataset}.txt")
    if ref_path.exists():
        ref = _read_items(ref_path)
        got = list(G["gene_cols"])
        if got != ref:
            only_got = sorted(set(got) - set(ref))[:10]
            only_ref = sorted(set(ref) - set(got))[:10]
            sys.exit(
                f"PANEL MISMATCH for {args.dataset}: this run has {len(got)} "
                f"genes, the published bundle has {len(ref)}.\n"
                f"  in this run only: {only_got}\n"
                f"  in bundle only:   {only_ref}\n"
                f"  order identical:  {got == sorted(got)} / {ref == sorted(ref)}\n"
                f"Use --gene-list panel_{args.dataset} (and NOT "
                f"--union-targets) to reproduce the published panel, or pass "
                f"--panel-reference '' to skip this check.")
        print(f"[panel] verified {len(got)} genes match "
              f"{ref_path.name}", flush=True)
    else:
        print(f"[panel] WARNING: no reference at {ref_path}; panel unverified",
              flush=True)

    items = _read_items(args.items_file)
    pending = [it for it in items if not (
        preds_dir / f"{(it.replace('/', '_').replace(' ', '_'))}.npz").exists()]
    print(f"[{args.dataset} seed={args.seed} EXTRA] {len(pending)}/{len(items)} "
          f"pending; methods={EXTRA_METHODS}; "
          f"lambda_dann={args.lambda_dann} lambda_adae={args.lambda_adae}; "
          f"shap={'per-cell' if save_shap else ('imp' if save_imp else 'none')}; "
          f"gene_list={args.gene_list} ({len(G['gene_cols'])} genes); "
          f"nproc={args.nproc}; loaded in {time.time()-t0:.0f}s", flush=True)

    if not pending:
        print(f"[{args.dataset} seed={args.seed} EXTRA] nothing to do",
              flush=True)
        return

    ctx = mp.get_context("fork")
    all_rows, done = [], 0
    with ctx.Pool(args.nproc) as pool:
        for r in pool.imap_unordered(work, pending, chunksize=1):
            done += 1
            if r:
                all_rows.extend(r)
            if done % 10 == 0:
                print(f"  {done}/{len(pending)} ({time.time()-t0:.0f}s)",
                      flush=True)
    if all_rows:
        # One CSV per node. The read-concat-write pattern used elsewhere is a
        # lost-update race when several dispatched nodes finish at once, and
        # this file is the *only* home of the adversary diagnostics
        # (domain/adversary accuracy, divergence flags) — they are not
        # recoverable from the npz. Per-node files make the race impossible;
        # `collect_extra_summaries()` concatenates them for analysis.
        import socket
        node = socket.gethostname().split(".")[0]
        out_csv = (args.out_dir /
                   f"{args.dataset}_seed{args.seed}_extra__{node}.csv")
        df = pd.DataFrame(all_rows)
        if out_csv.exists():
            df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
        df.to_csv(out_csv, index=False)
    print(f"[{args.dataset} seed={args.seed} EXTRA] DONE {done} items in "
          f"{time.time()-t0:.0f}s -> {preds_dir}", flush=True)


if __name__ == "__main__":
    main()
