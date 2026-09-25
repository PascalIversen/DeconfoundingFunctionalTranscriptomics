"""Transparent merging of the base prediction bundle with the extras bundle.

The published Zenodo deposit holds `{dataset}_seed{s}_preds/{item}.npz` with
the four original methods. The adversarial baselines
live alongside it in `{dataset}_seed{s}_extra/{item}.npz`, written by
`training/gen_preds_extra.py`.

Keeping them in separate files means the 4 GB published bundle never has to
be rewritten, and a reproduction that only fetches the deposit still works —
the extras are simply absent and the figure scripts fall back to the
original method set.

`load_preds()` returns an object that quacks like `numpy.lib.npyio.NpzFile`
(`.files`, `d[key]`, `key in d.files`), so call sites change by one line.

    d = load_preds(f)                 # instead of np.load(f, ...)
    for m in METHODS:
        if f"shap_{m}" not in d.files:
            continue
        ...
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

# Suffix of the directory holding the extra methods, given a "*_preds" dir.
EXTRA_SUFFIX = "_extra"
BASE_SUFFIX = "_preds"


class MergedPreds:
    """Read-only union of a base npz and an optional extras npz.

    Keys present in both resolve to the *base* file. The shared keys
    (`y_true`, `tissue`, `gene_cols`) are written identically by both
    runners; `verify()` checks that rather than trusting it.
    """

    def __init__(self, base_path: Path, extra_path: Optional[Path] = None):
        self.base_path = Path(base_path)
        self._base = np.load(self.base_path, allow_pickle=True)
        self.extra_path = Path(extra_path) if extra_path else None
        self._extra = (np.load(self.extra_path, allow_pickle=True)
                       if self.extra_path and self.extra_path.exists()
                       else None)

    @property
    def files(self) -> List[str]:
        out = list(self._base.files)
        if self._extra is not None:
            out += [k for k in self._extra.files if k not in out]
        return out

    @property
    def has_extra(self) -> bool:
        return self._extra is not None

    def __contains__(self, key: str) -> bool:
        return key in self.files

    def __getitem__(self, key: str):
        if key in self._base.files:
            return self._base[key]
        if self._extra is not None and key in self._extra.files:
            return self._extra[key]
        raise KeyError(f"{key} in neither {self.base_path} nor {self.extra_path}")

    def verify(self) -> None:
        """Raise if the two files disagree on the cells they describe.

        A mismatch would mean the extras were generated from a different
        item, cell set or seed, and every paired comparison downstream would
        be silently wrong.
        """
        if self._extra is None:
            return
        for key in ("y_true", "tissue", "gene_cols"):
            if key not in self._extra.files or key not in self._base.files:
                continue
            a, b = self._base[key], self._extra[key]
            if a.shape != b.shape:
                raise ValueError(
                    f"{self.base_path.name}: '{key}' shape {a.shape} in base "
                    f"vs {b.shape} in extras")
            same = (np.allclose(a, b, equal_nan=True)
                    if np.issubdtype(a.dtype, np.floating) else bool((a == b).all()))
            if not same:
                raise ValueError(
                    f"{self.base_path.name}: '{key}' differs between the base "
                    f"and extras bundle — they describe different data")


def extra_dir_for(preds_dir: Path) -> Path:
    """`.../depmap_seed1_preds` -> `.../depmap_seed1_extra`."""
    preds_dir = Path(preds_dir)
    name = preds_dir.name
    if name.endswith(BASE_SUFFIX):
        name = name[: -len(BASE_SUFFIX)] + EXTRA_SUFFIX
    else:
        name = name + EXTRA_SUFFIX
    return preds_dir.parent / name


def load_preds(npz_path: Path, verify: bool = True) -> MergedPreds:
    """Load one item, merging in its extras file when one exists."""
    npz_path = Path(npz_path)
    extra = extra_dir_for(npz_path.parent) / npz_path.name
    d = MergedPreds(npz_path, extra)
    if verify:
        d.verify()
    return d


def available_methods(preds_dir: Path, candidates: List[str],
                      kind: str = "shap") -> List[str]:
    """Subset of `candidates` actually present in `preds_dir`.

    Lets a figure script declare every method it knows about and then render
    only those the user has data for, instead of erroring out on a partial
    download.
    """
    preds_dir = Path(preds_dir)
    files = sorted(preds_dir.glob("*.npz"))
    if not files:
        return []
    d = load_preds(files[0], verify=False)
    keys = set(d.files)
    return [m for m in candidates if f"{kind}_{m}" in keys]


def collect_extra_summaries(results_root, dataset: str, seed: int):
    """Concatenate the per-node extras summary CSVs for one (dataset, seed).

    `gen_preds_extra.py` writes one CSV per worker node
    (`{dataset}_seed{seed}_extra__{node}.csv`) rather than appending to a
    shared file, because several dispatched nodes finishing at once would
    lose rows — and those rows carry the adversary diagnostics, which are
    not recoverable from the npz bundles.
    """
    import pandas as pd
    root = Path(results_root)
    parts = sorted(root.glob(f"{dataset}_seed{seed}_extra__*.csv"))
    legacy = root / f"{dataset}_seed{seed}_extra.csv"
    if legacy.exists():
        parts.append(legacy)
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
