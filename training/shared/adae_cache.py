"""Cross-item cache for AD-AE encoders.

The AD-AE embedding is fitted on expression and the confounder label only —
it never sees the phenotype. So for a fixed (cell set, CV fold, seed,
lambda, architecture) the encoder is the *same object* no matter which
target gene or drug is being modelled. Caching it is exact, not an
approximation.

This matters a lot on DepMap, where 185 of the 200 knock-out targets are
screened in exactly the same 1178 cell lines: 5 distinct cell sets x 5
folds = 25 encoders instead of 1000. CTRPv2 drugs each have their own
screened panel, so there the cache simply never hits.

Two layers:
  * an in-process dict — covers items handled by the same pool worker;
  * an optional directory on disk — shared across workers and nodes.

Determinism: the torch RNG is re-seeded from the cache key before the
encoder is built, so a cached encoder and a freshly trained one are
identical. Without this, results would depend on which worker happened to
pick up which item first.

Disk writes are atomic (temp file + os.replace). Two workers racing to
train the same key both produce the same bytes, so the loser of the race
is harmless.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Bump when the AD-AE training procedure changes in any way that would make
# previously cached encoders wrong.
CACHE_VERSION = "adae-v1"

_MEM: Dict[str, Tuple[dict, dict]] = {}


def make_key(sample_ids: Sequence[str], *, seed: int, fold: int,
             lambda_adv: float, hidden: int, emb_dim: int,
             ae_dropout: float, batch_size: int, lr: float,
             ae_pretrain_epochs: int, adv_pretrain_epochs: int,
             joint_rounds: int, patience: int, in_dim: int,
             val_frac: float, panel: Sequence[str] = ()) -> str:
    """Stable key over everything the fitted encoder depends on.

    `sample_ids` is the item's cell list **in order** — the CV split is
    positional, so ordering is part of the identity, not just membership.

    `in_dim` alone does not identify the feature panel, and `sample_ids`
    alone does not identify the training rows: two runs on different gene
    panels of equal width, or on the same cells with a different
    train/validation carve, would otherwise collide and be served a stale
    encoder. `panel` (the ordered feature names) and `val_frac` close both
    gaps -- together with the cell list they also pin the StandardScaler
    statistics, since those are a function of exactly these inputs.
    """
    panel_digest = hashlib.sha1(
        "\u0000".join(map(str, panel)).encode()).hexdigest() if panel else ""
    payload = json.dumps({
        "v": CACHE_VERSION,
        "cells": list(map(str, sample_ids)),
        "in_dim": int(in_dim),
        "panel": panel_digest,
        "val_frac": round(float(val_frac), 10),
        "seed": int(seed), "fold": int(fold),
        "lambda": float(lambda_adv), "hidden": int(hidden),
        "emb": int(emb_dim), "drop": float(ae_dropout),
        "bs": int(batch_size), "lr": float(lr),
        "ae_ep": int(ae_pretrain_epochs), "adv_ep": int(adv_pretrain_epochs),
        "joint": int(joint_rounds), "pat": int(patience),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def seed_from_key(key: str) -> int:
    """Deterministic 31-bit torch seed derived from the cache key."""
    return int(key[:8], 16) & 0x7FFFFFFF


def get_or_fit(key: str, fit_fn: Callable[[], Tuple[torch.nn.Module, dict]],
               *, cache_dir: Optional[Path] = None,
               device=None) -> Tuple[dict, dict]:
    """Return `(state_dict, diagnostics)` for the AD-AE model at `key`.

    `fit_fn` must build and train the model; it is only called on a miss.
    The torch RNG is seeded from `key` first, so the fit is reproducible.
    """
    if key in _MEM:
        return _MEM[key]

    path = (Path(cache_dir) / f"{key}.pt") if cache_dir else None
    if path is not None and path.exists():
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            _MEM[key] = (blob["state_dict"], blob["diag"])
            return _MEM[key]
        except Exception as exc:                       # truncated / racing write
            logger.warning("AD-AE cache read failed for %s (%s); refitting",
                           key[:12], exc)

    torch.manual_seed(seed_from_key(key))
    np.random.seed(seed_from_key(key) % (2 ** 32))
    model, diag = fit_fn()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _MEM[key] = (state, diag)

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            os.close(fd)
            torch.save({"state_dict": state, "diag": diag}, tmp)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("AD-AE cache write failed for %s (%s)", key[:12], exc)
    return _MEM[key]


def clear_memory_cache() -> None:
    _MEM.clear()


def memory_cache_size() -> int:
    return len(_MEM)
