# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Featurisation for SENTINEL-ChemTox.

Two families live here.

**1. The published ADORE feature set** (Gasser et al., *Environ. Sci.: Adv.* 3:1124-1138, 2024,
DOI 10.1039/D4VA00072B, §2.4), reproduced from the paper text because the modelling code is not
retrievable (renkulab GitLab is unreachable):

* experimental (4 source columns): observation duration — ordinal with the four levels
  24/48/72/96 h, mapped to [0, 1]; media type, exposure type and concentration type — categorical,
  one-hot encoded;
* chemical (4 columns, standardised on the training rows): molecular weight, water solubility,
  melting point, log Kow (``chem_rdkit_clogp``). The paper explicitly drops the atom/bond/ring
  counts as correlated with these four;
* taxonomic: from Add-my-Pet — ecology (climate zone, ecozone, food) as **many**-hot encodings
  because a species may carry several levels (e.g. food ``D_H`` = detrivorous + herbivorous);
  life-history life span (``tax_lh_amd``) and ultimate body length (``tax_lh_licm``); pseudo-data
  energy conductance (``tax_ps_ampv``), allocation rate to soma (``tax_ps_ampkap``) and
  volume-specific somatic maintenance cost (``tax_ps_amppm``) — all standardised on the training
  rows. Phylogenetic distances are *excluded* (the paper uses them only in its GP variant).

The paper's Table 1 states 37 non-structural columns for ``t-F2F``. One-hot encoding the three
experimental categoricals with the first level dropped gives exactly 37 here
(1 + 1 + 3 + 2 + 4 + 21 + 5); keeping all levels gives 40, and adding the binary migration flag
gives 38/41. ``drop_first``/``include_migration`` expose both, and
:func:`published_feature_count` records what a configuration produced, so the discrepancy is
visible rather than hidden. The paper mentions migratory behaviour in the text but no combination
including it reaches 37.

**2. Fingerprints and descriptors.** ADORE ships MACCS (167 bits), PubChem (881), Morgan (2048)
and ToxPrint (729) bit strings plus mol2vec (300) and Mordred (719) columns.
:class:`InformativeBits` reproduces the published bit filter: a bit is kept when its standard
deviation **over the unique training chemicals** exceeds 0.1. That rule reproduces all eight
published counts exactly (occurrence split: MACCS 141, PubChem 471, Morgan 380, ToxPrint 172;
totally-random split: 143 / 469 / 385 / 174), whereas a per-row standard deviation does not — so
the filter is fixed by evidence, not by guesswork.

:func:`rdkit_features` computes MACCS / Morgan / RDKit descriptors from SMILES for molecules
outside ADORE (the spill-severity path) and is deterministic: same SMILES in, same bits out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- the published ADORE feature set

DURATION_COL = "result_obs_duration_mean"
DURATION_LEVELS = (24.0, 48.0, 72.0, 96.0)
EXPERIMENTAL_CATEGORICAL = ("test_media_type", "test_exposure_type", "result_conc1_type")
CHEMICAL_CONTINUOUS = ("chem_mw", "chem_ws", "chem_mp", "chem_rdkit_clogp")
CHEMICAL_TOP3 = ("chem_mw", "chem_ws", "chem_rdkit_clogp")  # the paper's 'top 3' ablation
#: the two chemical properties that are **computable for any molecule** (RDKit ``Descriptors.MolWt``
#: and ``Crippen.MolLogP``, both verified to reproduce ADORE's own columns). Water solubility and
#: melting point are experimental values ADORE compiled per chemical and do not exist for chemicals
#: outside ADORE, so this is the chemical block used for the prospective evaluation
#: (docs/preregistration/chemtox_prospective_plan.md §4.3).
CHEMICAL_MW_LOGP = ("chem_mw", "chem_rdkit_clogp")
TAX_MULTIHOT = ("tax_eco_climate", "tax_eco_ecozone", "tax_eco_food")
TAX_CONTINUOUS = ("tax_lh_amd", "tax_lh_licm", "tax_ps_ampv", "tax_ps_ampkap", "tax_ps_amppm")
MIGRATION_COL = "tax_eco_migrate2"
MULTIHOT_SEPARATOR = "_"

#: fingerprint bit-string columns of the ADORE challenge files
FINGERPRINT_COLUMNS = {
    "MACCS": "chem_MACCS_fp",
    "PubChem": "chem_pcp_fp",
    "Morgan": "chem_Morgan_fp",
    "ToxPrint": "chem_ToxPrint_fp",
}
PUBLISHED_OTHER_FEATURE_COUNT = 37  # Gasser et al. Table 1, t-F2F


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("<NA>")


def multihot_levels(values: pd.Series, *, separator: str = MULTIHOT_SEPARATOR) -> list[str]:
    """Sorted individual levels of a many-hot column (``'D_H'`` -> ``['D', 'H']``)."""
    levels: set[str] = set()
    for v in _as_str(values).unique():
        if v == "<NA>":
            continue
        levels.update(p for p in str(v).split(separator) if p)
    return sorted(levels)


@dataclass
class PublishedFeatures:
    """The published non-structural ADORE features; fit on training rows only.

    ``fit`` records the categorical levels, the duration scaling and the training mean/SD of every
    continuous column; ``transform`` applies them. Nothing is learned from validation or test rows.
    """

    drop_first: bool = True
    include_migration: bool = False
    chemical_columns: tuple[str, ...] = CHEMICAL_CONTINUOUS
    standardize: bool = True
    levels_: dict[str, list[str]] = field(default_factory=dict)
    multihot_: dict[str, list[str]] = field(default_factory=dict)
    mean_: dict[str, float] = field(default_factory=dict)
    sd_: dict[str, float] = field(default_factory=dict)
    names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    # ------------------------------------------------------------------ fit / transform
    def fit(self, train: pd.DataFrame) -> PublishedFeatures:
        self.levels_, self.multihot_, self.mean_, self.sd_ = {}, {}, {}, {}
        for c in EXPERIMENTAL_CATEGORICAL:
            lv = sorted(v for v in _as_str(train[c]).unique())
            self.levels_[c] = lv[1:] if (self.drop_first and len(lv) > 1) else lv
        for c in TAX_MULTIHOT:
            self.multihot_[c] = multihot_levels(train[c])
        for c in self._continuous():
            v = pd.to_numeric(train[c], errors="coerce").to_numpy(dtype=float)
            m = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
            s = float(np.nanstd(v)) if np.isfinite(v).any() else 0.0
            self.mean_[c], self.sd_[c] = m, (s if s > 0 else 1.0)
        self.names_ = self._names()
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("PublishedFeatures.transform before fit")
        blocks: list[np.ndarray] = [self._duration(df)]
        for c in EXPERIMENTAL_CATEGORICAL:
            v = _as_str(df[c]).to_numpy()
            blocks.append(np.column_stack([(v == lv).astype(np.float32) for lv in self.levels_[c]])
                          if self.levels_[c] else np.zeros((len(df), 0), dtype=np.float32))
        for c in TAX_MULTIHOT:
            v = _as_str(df[c]).to_numpy()
            parts = [np.array([lv in str(x).split(MULTIHOT_SEPARATOR) for x in v], dtype=np.float32)
                     for lv in self.multihot_[c]]
            blocks.append(np.column_stack(parts) if parts else np.zeros((len(df), 0), dtype=np.float32))
        if self.include_migration:
            v = _as_str(df[MIGRATION_COL]).to_numpy()
            blocks.append((v == "migration").astype(np.float32).reshape(-1, 1))
        for c in self._continuous():
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            v = (v - self.mean_[c]) / self.sd_[c] if self.standardize else v
            # Missing continuous values take the TRAINING mean (0 after standardisation). `t-F2F`
            # has none, but `t-C2C` and `a-CA2F-diff` lack some Add-my-Pet life-history and
            # pseudo-data values (e.g. `tax_lh_licm`), and NaN would otherwise propagate into
            # every model. The imputation is a no-op where nothing is missing.
            v = np.nan_to_num(v, nan=self.mean_[c] if not self.standardize else 0.0,
                              posinf=0.0, neginf=0.0)
            blocks.append(v.astype(np.float32).reshape(-1, 1))
        X = np.column_stack(blocks).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X.shape[1] != len(self.names_):
            raise AssertionError(f"{X.shape[1]} columns vs {len(self.names_)} names")
        return X

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    # ------------------------------------------------------------------ internals
    def _continuous(self) -> tuple[str, ...]:
        return tuple(self.chemical_columns) + TAX_CONTINUOUS

    def _duration(self, df: pd.DataFrame) -> np.ndarray:
        d = pd.to_numeric(df[DURATION_COL], errors="coerce").to_numpy(dtype=float)
        lo, hi = min(DURATION_LEVELS), max(DURATION_LEVELS)
        return ((d - lo) / (hi - lo)).astype(np.float32).reshape(-1, 1)

    def _names(self) -> list[str]:
        names = [DURATION_COL + "_scaled"]
        for c in EXPERIMENTAL_CATEGORICAL:
            names += [f"{c}={lv}" for lv in self.levels_[c]]
        for c in TAX_MULTIHOT:
            names += [f"{c}~{lv}" for lv in self.multihot_[c]]
        if self.include_migration:
            names.append(f"{MIGRATION_COL}=migration")
        names += list(self._continuous())
        return names

    @property
    def n_features(self) -> int:
        return len(self.names_)

    def required_columns(self) -> list[str]:
        cols = [DURATION_COL, *EXPERIMENTAL_CATEGORICAL, *TAX_MULTIHOT, *self._continuous()]
        if self.include_migration:
            cols.append(MIGRATION_COL)
        return cols


def published_feature_count(train: pd.DataFrame, **kw) -> int:
    """Number of non-structural columns a :class:`PublishedFeatures` configuration produces."""
    return PublishedFeatures(**kw).fit(train).n_features


# ----------------------------------------------------------------- ADORE fingerprints / descriptors


def bit_matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    """Bit-string column (e.g. ``chem_MACCS_fp``) -> float32 matrix (rows x bits)."""
    s = df[column].astype("string")
    if s.isna().any():
        raise ValueError(f"{column}: {int(s.isna().sum())} missing fingerprints")
    lens = s.str.len().unique()
    if len(lens) != 1:
        raise ValueError(f"{column}: fingerprints of different lengths {sorted(lens)}")
    arr = np.frombuffer("".join(s.to_list()).encode("ascii"), dtype=np.uint8).reshape(len(s), int(lens[0]))
    bits = arr - ord("0")
    if bits.max(initial=0) > 1:
        raise ValueError(f"{column}: not a 0/1 bit string")
    return bits.astype(np.float32)


@dataclass
class InformativeBits:
    """The published bit filter: SD over the *unique training chemicals* above ``threshold``.

    Reproduces all eight informative-bit counts of Gasser et al. Table 1 (see module docstring).
    """

    column: str = "chem_MACCS_fp"
    threshold: float = 0.1
    chem_col: str = "test_cas"
    keep_: np.ndarray | None = None
    names_: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame) -> InformativeBits:
        X = bit_matrix(train, self.column)
        cas = train[self.chem_col].astype("string").fillna("<NA>").to_numpy()
        _, first = np.unique(cas, return_index=True)
        per_chem = X[np.sort(first)]
        sd = per_chem.std(axis=0)
        self.keep_ = sd > self.threshold
        self.names_ = [f"{self.column}[{i}]" for i in np.flatnonzero(self.keep_)]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.keep_ is None:
            raise RuntimeError("InformativeBits.transform before fit")
        return bit_matrix(df, self.column)[:, self.keep_]

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    @property
    def n_features(self) -> int:
        return int(self.keep_.sum()) if self.keep_ is not None else 0


@dataclass
class StandardizedColumns:
    """Numeric columns (mol2vec / Mordred / anything continuous) standardised on training rows."""

    columns: tuple[str, ...] = ()
    drop_constant: bool = True
    keep_: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    sd_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> StandardizedColumns:
        V = train[list(self.columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mean = np.nanmean(V, axis=0)
        sd = np.nanstd(V, axis=0)
        keep = np.isfinite(mean) & np.isfinite(sd) & ((sd > 0) if self.drop_constant else True)
        self.keep_ = [c for c, k in zip(self.columns, keep) if k]
        self.mean_, self.sd_ = mean[keep], np.where(sd[keep] > 0, sd[keep], 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardizedColumns.transform before fit")
        V = df[self.keep_].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num((V - self.mean_) / self.sd_, nan=0.0).astype(np.float32)

    @property
    def names_(self) -> list[str]:
        return list(self.keep_)

    @property
    def n_features(self) -> int:
        return len(self.keep_)


# ----------------------------------------------------------------- RDKit featurisation from SMILES

#: RDKit descriptors used for molecules outside ADORE; a fixed, sorted, documented list so the
#: featuriser is reproducible across RDKit versions that add descriptors.
def rdkit_descriptor_names(full: bool = False) -> tuple[str, ...]:
    """The descriptor list used by the descriptor block.

    ``full`` returns every descriptor in RDKit's ``Descriptors.descList`` (217 in rdkit 2025.09),
    minus ``Ipc`` whose value grows combinatorially with molecule size and overflows float32 (its
    normalised twin ``AvgIpc`` is kept). The short list below is the phase-1 default and is a
    subset of it, so a configuration that used the short list is unaffected.
    """
    if not full:
        return RDKIT_DESCRIPTORS
    from rdkit.Chem import Descriptors

    return tuple(n for n, _ in Descriptors.descList if n != "Ipc")


RDKIT_DESCRIPTORS = (
    "BalabanJ", "BertzCT", "Chi0", "Chi1", "FractionCSP3", "HallKierAlpha", "HeavyAtomCount",
    "LabuteASA", "MaxPartialCharge", "MinPartialCharge", "MolLogP", "MolMR", "MolWt",
    "NHOHCount", "NOCount", "NumAliphaticRings", "NumAromaticRings", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "NumSaturatedRings", "RingCount", "TPSA",
)


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def mol_from_smiles(smiles: str):
    return _rdkit().MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None


def maccs_bits(smiles: str) -> np.ndarray:
    """167-bit MACCS key vector (bit 0 is RDKit's unused padding bit)."""
    from rdkit.Chem import MACCSkeys

    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(167, np.nan, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(list(fp.ToBitString()), dtype=np.float32)


def morgan_bits(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit.Chem import rdFingerprintGenerator

    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(n_bits, np.nan, dtype=np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(list(gen.GetFingerprint(mol).ToBitString()), dtype=np.float32)


def rdkit_descriptors(smiles: str, names: tuple[str, ...] = RDKIT_DESCRIPTORS) -> np.ndarray:
    from rdkit.Chem import Descriptors

    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(len(names), np.nan, dtype=np.float32)
    table = dict(Descriptors.descList)
    out = np.empty(len(names), dtype=np.float32)
    for i, n in enumerate(names):
        fn = table.get(n)
        try:
            out[i] = np.float32(fn(mol)) if fn is not None else np.nan
        except Exception:  # noqa: BLE001 — a descriptor can fail on exotic molecules
            out[i] = np.nan
    return out


def rdkit_features(
    smiles: list[str] | pd.Series,
    *,
    maccs: bool = True,
    morgan: bool = False,
    descriptors: bool = True,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    """Deterministic RDKit featurisation of a SMILES list; returns (matrix, feature names)."""
    smiles = list(smiles)
    blocks, names = [], []
    if maccs:
        blocks.append(np.vstack([maccs_bits(s) for s in smiles]))
        names += [f"maccs[{i}]" for i in range(167)]
    if morgan:
        blocks.append(np.vstack([morgan_bits(s, radius=radius, n_bits=n_bits) for s in smiles]))
        names += [f"morgan_r{radius}[{i}]" for i in range(n_bits)]
    if descriptors:
        blocks.append(np.vstack([rdkit_descriptors(s) for s in smiles]))
        names += [f"desc:{n}" for n in RDKIT_DESCRIPTORS]
    if not blocks:
        raise ValueError("no feature block requested")
    return np.column_stack(blocks).astype(np.float32), names
