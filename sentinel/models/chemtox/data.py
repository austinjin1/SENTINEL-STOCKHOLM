# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Assembling ADORE design matrices without ever re-splitting the data.

:class:`ChemToxDataset` loads one challenge file through :mod:`sentinel.data.sources.adore` (the
loader owned by the data workstream), exposes the official partitions of one split column
(``train``/``test`` and CV folds 0-4) and builds feature matrices for a
:class:`FeatureSpec`. Every transformer is fit on the rows it is given as *training* rows only —
for CV that is the four in-fold training folds, for the final fit the whole official training
partition. No transformer ever sees validation or test rows during ``fit``.

The target is ``result_conc1_mean_mol_log`` (log10 LC50/EC50 in mol/L); ``target="mass"`` switches
to ``result_conc1_mean_log`` (log10 mg/L) for the robustness check the research plan asks for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from sentinel.data.sources import adore as A

from . import features as F

#: label used in a spec's name when the full (un-reduced) CheMeleon fingerprint is used
FP_FULL = "2048"

#: ADORE challenges whose train and test partitions share chemicals *by construction* — the
#: extrapolation they test is taxonomic, not chemical (docs/research/04 §A.2, §D.3).
SHARED_CHEMICALS_BY_DESIGN = ("a-CA2F-same", "a-FCA2FCA")


@dataclass(frozen=True)
class FeatureSpec:
    """What goes into the design matrix.

    ``fingerprint``: ``MACCS`` / ``PubChem`` / ``Morgan`` / ``ToxPrint`` / ``None``;
    ``published``: the 37-column experimental+chemical+taxonomic block (Gasser et al.);
    ``chemical``: ``four`` (MW, WS, MP, logP), ``top3`` or ``none``;
    ``mordred`` / ``mol2vec``: add those ADORE representations, standardised on training rows;
    ``rdkit_descriptors``: add the RDKit descriptor block computed from ``chem_rdkit_can_smiles``.
    """

    fingerprint: str | None = "MACCS"
    published: bool = True
    drop_first: bool = True
    include_migration: bool = False
    chemical: str = "four"
    mordred: bool = False
    mol2vec: bool = False
    rdkit_descriptors: bool = False
    rdkit_descriptors_full: bool = False
    chemeleon: bool = False
    chemeleon_components: int = 0
    bit_threshold: float = 0.1

    def chemical_columns(self) -> tuple[str, ...]:
        return {"four": F.CHEMICAL_CONTINUOUS, "top3": F.CHEMICAL_TOP3,
                "mw_logp": F.CHEMICAL_MW_LOGP, "none": ()}[self.chemical]

    def name(self) -> str:
        bits = [self.fingerprint or "nofp"]
        if self.published:
            bits.append(f"published37[{self.chemical}]" if self.drop_first else f"published[{self.chemical}]")
        if self.include_migration:
            bits.append("migration")
        if self.mordred:
            bits.append("mordred")
        if self.mol2vec:
            bits.append("mol2vec")
        if self.rdkit_descriptors:
            bits.append("rdkitdesc_full" if self.rdkit_descriptors_full else "rdkitdesc")
        if self.chemeleon:
            bits.append(f"chemeleon{self.chemeleon_components or FP_FULL}")
        return "+".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureBuilder:
    """Fit-on-train / transform-many feature assembly for one :class:`FeatureSpec`."""

    def __init__(self, spec: FeatureSpec, *, mordred_columns=(), mol2vec_columns=(),
                 chemeleon_device: str = "cpu"):
        self.spec = spec
        self.chemeleon_device = chemeleon_device
        self.parts: list = []
        self.names_: list[str] = []
        self._mordred_columns = tuple(mordred_columns)
        self._mol2vec_columns = tuple(mol2vec_columns)
        self._rdkit_cache: dict[str, np.ndarray] | None = None

    def fit(self, train: pd.DataFrame) -> FeatureBuilder:
        self.parts = []
        s = self.spec
        if s.fingerprint:
            col = F.FINGERPRINT_COLUMNS[s.fingerprint]
            self.parts.append(F.InformativeBits(column=col, threshold=s.bit_threshold).fit(train))
        if s.published:
            self.parts.append(
                F.PublishedFeatures(drop_first=s.drop_first, include_migration=s.include_migration,
                                    chemical_columns=s.chemical_columns()).fit(train)
            )
        if s.mordred:
            self.parts.append(F.StandardizedColumns(columns=self._mordred_columns).fit(train))
        if s.mol2vec:
            self.parts.append(F.StandardizedColumns(columns=self._mol2vec_columns).fit(train))
        if s.rdkit_descriptors:
            self.parts.append(_RDKitDescriptorBlock(full=s.rdkit_descriptors_full).fit(train))
        if s.chemeleon:
            from .chemeleon_fp import CheMeleonBlock

            self.parts.append(CheMeleonBlock(n_components=s.chemeleon_components,
                                             device=self.chemeleon_device).fit(train))
        self.names_ = [n for p in self.parts for n in p.names_]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.parts:
            raise RuntimeError("FeatureBuilder.transform before fit")
        return np.column_stack([p.transform(df) for p in self.parts]).astype(np.float32)

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    @property
    def n_features(self) -> int:
        return len(self.names_)

    def block_sizes(self) -> dict[str, int]:
        return {type(p).__name__ + getattr(p, "column", ""): p.n_features for p in self.parts}


class _RDKitDescriptorBlock:
    """RDKit descriptors from ``chem_rdkit_can_smiles``, standardised on training rows."""

    SMILES_COL = "chem_rdkit_can_smiles"

    def __init__(self, full: bool = False):
        self.mean_ = None
        self.sd_ = None
        self.descriptors_ = F.rdkit_descriptor_names(full)
        self.names_ = [f"desc:{n}" for n in self.descriptors_]

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        smiles = df[self.SMILES_COL].astype("string").fillna("").to_list()
        uniq = {}
        for s in set(smiles):
            uniq[s] = F.rdkit_descriptors(s, self.descriptors_)
        return np.vstack([uniq[s] for s in smiles])

    def fit(self, train: pd.DataFrame) -> _RDKitDescriptorBlock:
        V = self._matrix(train).astype(float)
        self.mean_ = np.nanmean(V, axis=0)
        sd = np.nanstd(V, axis=0)
        self.sd_ = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        self.mean_ = np.nan_to_num(self.mean_)
        return self

    #: Standardised descriptor values are clipped to this many training SDs before the float32
    #: cast, because some RDKit descriptors grow combinatorially with molecule size and overflow
    #: float32 for molecules outside ADORE (the un-normalised ``Ipc`` is excluded from the full
    #: list for the same reason), which would put ``inf`` into the design matrix.
    #:
    #: The bound is deliberately far above anything ADORE produces. Measured on the t-F2F rows:
    #: max |z| is 12.3 (train) and 7.5 (test) for the 23-descriptor list used in phase 1 — so **no
    #: registered ADORE number is affected by clipping** — and up to 148 overall, 399 inside CV
    #: fold 3, for the 216-descriptor list introduced in cycle 2. The bound is therefore 1e6: it
    #: binds only on non-finite or absurd values, never on a real molecule in either list.
    CLIP_SD = 1e6

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        V = self._matrix(df).astype(float)
        Z = np.nan_to_num((V - self.mean_) / self.sd_, nan=0.0, posinf=self.CLIP_SD,
                          neginf=-self.CLIP_SD)
        return np.clip(Z, -self.CLIP_SD, self.CLIP_SD).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.names_)


@dataclass
class ChemToxDataset:
    """One ADORE challenge under one official split column."""

    challenge: str = "t-F2F"
    split: str = A.PRIMARY_SPLIT
    target: str = "molar"
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    TARGETS: ClassVar[dict[str, str]] = {"molar": A.TARGET, "mass": A.TARGET_MASS}

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(
        cls,
        challenge: str = "t-F2F",
        split: str = A.PRIMARY_SPLIT,
        *,
        target: str = "molar",
        specs: FeatureSpec | list[FeatureSpec] | None = None,
        root: str | Path | None = None,
        extra_columns: tuple[str, ...] = (),
    ) -> ChemToxDataset:
        if isinstance(specs, FeatureSpec):
            specs = [specs]
        specs = specs or [FeatureSpec()]
        cols = set(cls.META_COLUMNS) | set(extra_columns) | {cls.TARGETS[target]}
        schema = available_columns(challenge, root=root)
        need_mordred = any(s.mordred for s in specs)
        need_mol2vec = any(s.mol2vec for s in specs)
        groups = A.feature_groups(schema)
        for s in specs:
            if s.fingerprint:
                cols.add(F.FINGERPRINT_COLUMNS[s.fingerprint])
            if s.published:
                cols |= set(
                    F.PublishedFeatures(include_migration=s.include_migration,
                                        chemical_columns=s.chemical_columns()).required_columns()
                )
            if s.rdkit_descriptors or s.chemeleon:
                cols.add(_RDKitDescriptorBlock.SMILES_COL)
        if need_mordred:
            cols |= set(groups.get("mordred", []))
        if need_mol2vec:
            cols |= set(groups.get("mol2vec", []))
        missing = sorted(c for c in cols if c not in schema)
        if missing:
            raise KeyError(f"{challenge}: missing columns {missing[:8]}{'...' if len(missing) > 8 else ''}")
        df = A.load_challenge(challenge, columns=sorted(cols), root=root)
        if split not in df.columns:
            raise KeyError(f"{challenge} has no split column {split!r}; available: {A.split_columns(df)}")
        return cls(challenge=challenge, split=split, target=target, df=df.reset_index(drop=True))

    META_COLUMNS = (A.ROW_KEY, A.CHEM_KEY, A.SPECIES_KEY, "tax_group", "result_endpoint",
                    "result_effect", F.DURATION_COL)

    # ------------------------------------------------------------------ official partitions
    @property
    def assignment(self) -> pd.Series:
        return self.df[self.split].astype("string")

    def test_mask(self) -> np.ndarray:
        return (self.assignment == "test").fillna(False).to_numpy()

    def train_mask(self) -> np.ndarray:
        a = self.assignment
        return (a.notna() & (a != "test")).fillna(False).to_numpy()

    def fold_mask(self, fold: int) -> np.ndarray:
        return (self.assignment == str(fold)).fillna(False).to_numpy()

    def folds(self) -> list[int]:
        a = self.assignment
        is_fold = a.str.fullmatch(r"\d+").fillna(False)
        return sorted(int(v) for v in a[is_fold.to_numpy()].unique())

    def cv_splits(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """(fold, train positions, validation positions) over the official folds only."""
        a = self.assignment
        is_fold = a.str.fullmatch(r"\d+").fillna(False).to_numpy()
        out = []
        for k in self.folds():
            val = is_fold & (a == str(k)).fillna(False).to_numpy()
            trn = is_fold & ~val
            out.append((k, np.flatnonzero(trn), np.flatnonzero(val)))
        return out

    # ------------------------------------------------------------------ arrays
    @property
    def y(self) -> np.ndarray:
        return pd.to_numeric(self.df[self.TARGETS[self.target]], errors="coerce").to_numpy(dtype=float)

    @property
    def chem(self) -> np.ndarray:
        return self.df[A.CHEM_KEY].astype("string").fillna("<NA>").to_numpy()

    @property
    def taxon(self) -> np.ndarray:
        return self.df[A.SPECIES_KEY].astype("string").fillna("<NA>").to_numpy()

    def rows(self, idx: np.ndarray) -> pd.DataFrame:
        return self.df.iloc[idx]

    def builder(self, spec: FeatureSpec, *, chemeleon_device: str = "cpu") -> FeatureBuilder:
        groups = A.feature_groups(self.df.columns)
        return FeatureBuilder(spec, mordred_columns=groups.get("mordred", []),
                              mol2vec_columns=groups.get("mol2vec", []),
                              chemeleon_device=chemeleon_device)

    def matrices(self, spec: FeatureSpec, train_idx: np.ndarray, apply_idx: np.ndarray):
        """Fit on ``train_idx`` rows, transform both; returns (X_train, X_apply, builder)."""
        b = self.builder(spec)
        Xtr = b.fit_transform(self.rows(train_idx))
        Xap = b.transform(self.rows(apply_idx))
        return Xtr, Xap, b

    # ------------------------------------------------------------------ integrity
    def split_report(self) -> dict:
        """Row/chemical/species counts and the train-test chemical overlap of this split."""
        rep = A.chemical_overlap(self.df, self.split)
        rep.update({
            "challenge": self.challenge,
            "target": self.TARGETS[self.target],
            "folds": self.folds(),
            "n_per_fold": {str(k): int(self.fold_mask(k).sum()) for k in self.folds()},
            "chem_overlap_train_fold": {
                str(k): len(set(self.chem[trn]) & set(self.chem[val]))
                for k, trn, val in self.cv_splits()
            },
        })
        return rep

    def assert_no_chemical_leakage(self, *, log=None) -> None:
        """Official splits must not share a chemical between train and test (ADORE §A.2).

        Two ADORE challenges share chemicals **by construction**: ``a-CA2F-same`` trains on
        crustaceans + algae and tests on fish *for the same chemicals* (that is the point of the
        challenge — taxonomic, not chemical, extrapolation), and ``a-FCA2FCA`` pools everything.
        For those the overlap is reported rather than raised; for every other challenge a non-zero
        overlap is a leakage failure and stops the run.
        """
        tr, te = self.train_mask(), self.test_mask()
        shared = set(self.chem[tr]) & set(self.chem[te])
        if not shared:
            return
        msg = f"{self.challenge}/{self.split}: {len(shared)} chemicals in both train and test"
        if self.challenge in SHARED_CHEMICALS_BY_DESIGN:
            note = (f"{msg} — expected for {self.challenge} (cross-taxa challenge with shared "
                    "chemicals); the extrapolation is taxonomic, not chemical")
            if log:
                log.warning(note)
            else:  # keep it visible even without a logger
                print(f"WARNING {note}")
            return
        raise AssertionError(msg)


def available_columns(challenge: str, *, root: str | Path | None = None) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(A.challenge_path(challenge, root)).schema_arrow.names)
