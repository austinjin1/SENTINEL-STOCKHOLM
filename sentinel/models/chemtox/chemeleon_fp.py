# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""**Frozen** CheMeleon fingerprints: the 2048-d molecular representation, used as features.

Why this module exists (cycle 2, 2026-10-24). Our phase-1 attempt to *fine-tune* CheMeleon
end-to-end inside chemprop made things worse (CV RMSE 1.0145 vs 0.9754 for the same architecture
trained from scratch). The 2026 literature says that is the expected outcome and that the
representation should be used **frozen** instead: Ben Hicham, Rittig, Grohe & Mitsos,
"Tabular foundation models for in-context prediction of molecular properties"
(arXiv:2604.16123) report, over 58 regression tasks with 5 seeds, that a tabular model on frozen
CheMeleon fingerprints wins 86% of tasks while *fine-tuned* CheMeleon wins 41% and Chemprop 3%.
CheMeleon itself is Burns et al., "Deep Learning Foundation Models from Classical Molecular
Descriptors", arXiv:2506.15792 (weights: Zenodo record 15460715, DOI 10.5281/zenodo.15426600).

What the fingerprint is: the released ``chemeleon_mp.pt`` bond-message-passing encoder
(``d_h = 2048``, depth 6) run over the molecular graph in inference mode, followed by mean
aggregation over atoms — i.e. the encoder output with no task head, exactly what
``CheMeleonFingerprint`` produces upstream. Nothing is trained here, so the fingerprint of a
molecule is a deterministic function of its SMILES and may be computed once and cached. Because it
is a per-molecule constant, computing it for all rows (training, validation, test, prospective)
leaks nothing: no target value and no partition membership enters it.

Values are cached per canonical SMILES in ``$SENTINEL_RUNS/chemtox/pretrained/chemeleon_fp.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths

FP_DIM = 2048
SMILES_COL = "chem_rdkit_can_smiles"
CACHE_NAME = "chemeleon_fp.npz"
CITATION = ("CheMeleon foundation model, Burns et al. arXiv:2506.15792; weights Zenodo 15460715 "
            "(DOI 10.5281/zenodo.15426600); frozen-fingerprint usage after arXiv:2604.16123")


def cache_path(cache_dir: str | Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir else paths.ensure(paths.RUNS_ROOT / "chemtox" / "pretrained")
    return base / CACHE_NAME


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        keys = list(z["smiles"])
        mat = z["fp"]
    return {str(k): mat[i] for i, k in enumerate(keys)}


def _save_cache(path: Path, table: dict[str, np.ndarray]) -> None:
    """Atomic write through a process-unique temp file (several searches may run in parallel).

    The cache is a pure function of the SMILES, so a concurrent writer can only ever replace it
    with a superset; the unique name is what keeps two writers from sharing a half-written file.
    """
    import os

    keys = sorted(table)
    mat = np.vstack([table[k] for k in keys]).astype(np.float32)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, smiles=np.array(keys, dtype=object).astype(str), fp=mat)
    os.replace(tmp, path)


def compute(smiles: list[str], *, device: str = "cpu", batch_size: int = 128,
            cache_dir: str | Path | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    """Frozen CheMeleon fingerprints for a list of SMILES (cached on disk, order irrelevant)."""
    import torch
    from chemprop import data as cdata
    from chemprop import featurizers
    from chemprop import nn as cnn

    from .chemprop_model import load_chemeleon_message_passing

    path = cache_path(cache_dir)
    table = _load_cache(path)
    want = [s for s in dict.fromkeys(str(x) for x in smiles if isinstance(x, str) and x)
            if s not in table]
    if not want:
        return table
    mp = load_chemeleon_message_passing().to(device).eval()
    agg = cnn.MeanAggregation()
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    points = [cdata.MoleculeDatapoint.from_smi(s, y=np.zeros(1, dtype=float)) for s in want]
    dset = cdata.MoleculeDataset(points, featurizer=feat)
    loader = cdata.build_dataloader(dset, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch[0]
            bmg.to(device)
            h = mp(bmg)
            z = agg(h, bmg.batch)
            out.append(z.detach().cpu().numpy().astype(np.float32))
    mat = np.vstack(out)
    if mat.shape[0] != len(want):
        raise RuntimeError(f"CheMeleon fingerprints: got {mat.shape[0]} rows for {len(want)} SMILES")
    for i, s in enumerate(want):
        table[s] = mat[i]
    _save_cache(path, table)
    if verbose:
        print(f"CheMeleon: {len(want)} new fingerprints, cache now {len(table)}")
    return table


class CheMeleonBlock:
    """Feature block: frozen CheMeleon fingerprint, standardised (and optionally PCA-reduced).

    The standardiser and the PCA are fit on the **training rows only**, like every other block in
    :mod:`sentinel.models.chemtox.data`; the fingerprint itself is a per-molecule constant.
    """

    def __init__(self, n_components: int = 0, device: str = "cpu", batch_size: int = 128):
        self.n_components = int(n_components)
        self.device = device
        self.batch_size = int(batch_size)
        self.mean_ = None
        self.sd_ = None
        self.pca_ = None
        self.names_: list[str] = []
        self._table: dict[str, np.ndarray] = {}

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        s = df[SMILES_COL].astype("string").fillna("").to_list()
        missing = [x for x in dict.fromkeys(s) if x and x not in self._table]
        if missing:
            self._table.update(compute(missing, device=self.device, batch_size=self.batch_size))
        zero = np.zeros(FP_DIM, dtype=np.float32)
        return np.vstack([self._table.get(x, zero) for x in s])

    def fit(self, train: pd.DataFrame) -> CheMeleonBlock:
        V = self._matrix(train).astype(np.float64)
        self.mean_ = V.mean(axis=0)
        sd = V.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        Z = (V - self.mean_) / self.sd_
        if self.n_components and self.n_components < Z.shape[1]:
            from sklearn.decomposition import PCA

            self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(Z)
            self.names_ = [f"chemeleon_pc{i:03d}" for i in range(self.n_components)]
        else:
            self.pca_ = None
            self.names_ = [f"chemeleon{i:04d}" for i in range(Z.shape[1])]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        V = self._matrix(df).astype(np.float64)
        Z = (V - self.mean_) / self.sd_
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return np.nan_to_num(np.clip(Z, -50.0, 50.0), nan=0.0).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.names_)
