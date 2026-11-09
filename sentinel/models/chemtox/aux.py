# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Auxiliary training data for ChemTox, with the test-chemical exclusion rule enforced in code.

Extra data may only enter if **every chemical that appears in any ``test`` partition of the
challenge being evaluated is removed first**, matched on *both* identifiers — CAS number and
InChIKey — because neither alone is reliable (ADORE carries both; ECOTOX rows can have a CAS with
a missing structure, or an InChIKey whose CAS differs by salt form). :func:`excluded_keys` builds
that key set from the union of **all** split columns' test rows, so a chemical that is held out by
*any* official split can never be seen in training through a side door.
:func:`assert_no_test_chemicals` is the assertion the test suite exercises.

Two sources are wired up:

* :func:`adore_auxiliary` — training rows of the *other* ADORE challenge files (``t-C2C``,
  ``t-A2A``, ...). They share the challenge-file schema, so the published 37-column block is
  available and these rows can be used for multitask training across taxa.
* :func:`ecotox_auxiliary` — the data workstream's processed ECOTOX table
  (``$SENTINEL_DATA/processed/ecotox/ecotox_aquatic.parquet``), filtered to acute lethal
  concentrations with a structure, a usable duration and a molar target. The ADORE core *is*
  ECOTOX, so without the exclusion this would be pure leakage; with it, what remains is extra
  chemicals only. These rows do not carry the Add-my-Pet taxonomy, so they are used to
  **pretrain the message-passing encoder** (structure + duration + endpoint), never to train the
  full model head (see :mod:`sentinel.models.chemtox.chemprop_model`).

Fold-level hygiene: an auxiliary chemical that sits in the *validation* fold of the current CV
split would leak just as badly, so :func:`drop_chemicals` is also applied per fold against the
validation rows inside the model's ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.sources import adore as A

CAS_COL = A.CHEM_KEY            # 'test_cas' in the ADORE challenge files
INCHIKEY_COL = "chem_inchikey"
ECOTOX_TABLE = "ecotox/ecotox_aquatic.parquet"

#: ECOTOX filters for "acute lethal concentration, usable as an ADORE-like label".
#: The vocabularies are the processed table's own (lower-case taxon groups, ``endpoint_norm``
#: holding LC50/EC50 while ``endpoint_group`` only says threshold/median_effect/low_effect).
ECOTOX_ENDPOINTS = ("LC50", "EC50")
ECOTOX_EFFECTS = ("MOR",)
ECOTOX_DURATIONS_H = (24.0, 48.0, 72.0, 96.0)
ECOTOX_TAXON_GROUPS = ("fish", "crustaceans", "algae")


@dataclass(frozen=True)
class ChemicalKeys:
    """Chemical identities to exclude, as two key sets (either match is a hit)."""

    cas: frozenset[str]
    inchikey: frozenset[str]
    challenge: str
    n_test_rows: int

    def __len__(self) -> int:
        return len(self.cas) + len(self.inchikey)

    def summary(self) -> dict:
        return {"challenge": self.challenge, "n_cas": len(self.cas),
                "n_inchikey": len(self.inchikey), "n_test_rows": self.n_test_rows}


def _clean(values) -> frozenset[str]:
    s = pd.Series(values, dtype="object").astype("string").str.strip().str.upper()
    s = s[s.notna() & (s != "") & (s != "<NA>") & (s != "NAN")]
    return frozenset(s.unique().tolist())


def excluded_keys(challenge: str, *, root: str | Path | None = None,
                  splits: tuple[str, ...] | None = None) -> ChemicalKeys:
    """CAS and InChIKey of every chemical in any ``test`` partition of ``challenge``.

    ``splits`` defaults to *all* split columns present in the file (the safe choice: a chemical
    held out by the scaffold split must not arrive through auxiliary data when the occurrence
    split is being evaluated).
    """
    df = A.load_challenge(challenge, columns=[CAS_COL, INCHIKEY_COL], root=root)
    cols = list(splits) if splits else A.split_columns(df)
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        mask |= (df[c].astype("string") == "test").fillna(False).to_numpy()
    test = df[mask]
    return ChemicalKeys(cas=_clean(test[CAS_COL]), inchikey=_clean(test[INCHIKEY_COL]),
                        challenge=challenge, n_test_rows=int(mask.sum()))


def _hits(df: pd.DataFrame, keys: ChemicalKeys, *, cas_col: str, inchikey_col: str | None) -> np.ndarray:
    hit = np.zeros(len(df), dtype=bool)
    if cas_col in df.columns:
        cas = df[cas_col].astype("string").str.strip().str.upper()
        hit |= cas.isin(keys.cas).fillna(False).to_numpy()
    if inchikey_col and inchikey_col in df.columns:
        ik = df[inchikey_col].astype("string").str.strip().str.upper()
        hit |= ik.isin(keys.inchikey).fillna(False).to_numpy()
    return hit


def drop_test_chemicals(df: pd.DataFrame, keys: ChemicalKeys, *, cas_col: str = CAS_COL,
                        inchikey_col: str | None = INCHIKEY_COL) -> tuple[pd.DataFrame, dict]:
    """Remove rows whose CAS **or** InChIKey is in ``keys``; returns (kept rows, counts)."""
    hit = _hits(df, keys, cas_col=cas_col, inchikey_col=inchikey_col)
    kept = df.loc[~hit].reset_index(drop=True)
    info = {"n_in": len(df), "n_dropped": int(hit.sum()), "n_kept": len(kept),
            "keys": keys.summary()}
    return kept, info


def assert_no_test_chemicals(df: pd.DataFrame, keys: ChemicalKeys, *, cas_col: str = CAS_COL,
                             inchikey_col: str | None = INCHIKEY_COL, what: str = "auxiliary data") -> None:
    """Raise if any row still carries a chemical held out by the evaluated challenge."""
    hit = _hits(df, keys, cas_col=cas_col, inchikey_col=inchikey_col)
    if hit.any():
        raise AssertionError(
            f"{what}: {int(hit.sum())} rows carry chemicals from a test partition of "
            f"{keys.challenge} (matched on CAS or InChIKey)"
        )


def drop_chemicals(df: pd.DataFrame, cas_values, inchikey_values=(), *, cas_col: str = CAS_COL,
                   inchikey_col: str | None = INCHIKEY_COL) -> pd.DataFrame:
    """Remove rows matching an explicit CAS/InChIKey list (used per CV fold for the val chemicals)."""
    keys = ChemicalKeys(cas=_clean(cas_values), inchikey=_clean(inchikey_values),
                        challenge="<fold>", n_test_rows=0)
    kept, _ = drop_test_chemicals(df, keys, cas_col=cas_col, inchikey_col=inchikey_col)
    return kept


# ----------------------------------------------------------------------------- ADORE side data


def adore_auxiliary(
    primary_challenge: str,
    *,
    extra_challenges: tuple[str, ...] = ("t-C2C", "t-A2A"),
    split: str = A.PRIMARY_SPLIT,
    columns: list[str] | None = None,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Training rows of other ADORE challenges, with the primary challenge's test chemicals removed.

    Only rows in the *training* partition of ``split`` in those files are taken (their own test
    rows are never used), and the result is asserted clean.
    """
    keys = excluded_keys(primary_challenge, root=root)
    frames, info = [], {"keys": keys.summary(), "per_challenge": {}}
    for ch in extra_challenges:
        if ch == primary_challenge:
            continue
        df = A.load_challenge(ch, columns=columns or "core", root=root)
        if split not in df.columns:
            info["per_challenge"][ch] = {"skipped": f"no split column {split}"}
            continue
        a = df[split].astype("string")
        train = df[(a.notna() & (a != "test")).to_numpy()]
        kept, counts = drop_test_chemicals(train, keys)
        assert_no_test_chemicals(kept, keys, what=f"ADORE auxiliary {ch}")
        kept = kept.assign(aux_source=ch)
        frames.append(kept)
        info["per_challenge"][ch] = counts
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    info["n_rows"] = len(out)
    return out, info


# ----------------------------------------------------------------------------- ECOTOX side data


def ecotox_path(root: str | Path | None = None) -> Path:
    return (Path(root) if root else paths.PROCESSED) / ECOTOX_TABLE


def ecotox_auxiliary(
    primary_challenge: str,
    *,
    taxon_groups: tuple[str, ...] = ECOTOX_TAXON_GROUPS,
    endpoints: tuple[str, ...] = ECOTOX_ENDPOINTS,
    effects: tuple[str, ...] = ECOTOX_EFFECTS,
    durations_h: tuple[float, ...] = ECOTOX_DURATIONS_H,
    freshwater_only: bool = True,
    drop_censored: bool = True,
    root: str | Path | None = None,
    adore_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Extra acute LC50/EC50 rows from processed ECOTOX, test chemicals removed and asserted gone.

    Returned columns: ``cas``, ``chem_inchikey``, ``smiles``, ``species``, ``taxon_group``,
    ``duration_h``, ``endpoint``, ``effect_norm``, ``y`` (log10 mol/L), ``aux_source``.
    """
    path = ecotox_path(root)
    if not path.exists():
        raise FileNotFoundError(f"processed ECOTOX table not found: {path}")
    want = ["cas", "chem_inchikey", "chem_smiles", "latin_name", "taxon_group", "duration_h",
            "endpoint_norm", "effect_norm", "log10_conc_mol_L", "censored", "water_type"]
    df = pd.read_parquet(path, columns=want)
    n0 = len(df)
    m = df["log10_conc_mol_L"].notna() & df["chem_smiles"].notna()
    m &= df["endpoint_norm"].astype("string").isin(list(endpoints)).fillna(False)
    if effects:
        m &= df["effect_norm"].astype("string").isin(list(effects)).fillna(False)
    m &= df["taxon_group"].astype("string").str.lower().isin([g.lower() for g in taxon_groups]).fillna(False)
    m &= pd.to_numeric(df["duration_h"], errors="coerce").isin(list(durations_h)).fillna(False)
    if drop_censored:
        m &= ~df["censored"].fillna(False).astype(bool)
    if freshwater_only:
        m &= df["water_type"].astype("string").str.contains("FRESH", case=False, na=False)
    sub = df.loc[m.to_numpy()].copy()
    keys = excluded_keys(primary_challenge, root=adore_root)
    kept, counts = drop_test_chemicals(sub, keys, cas_col="cas", inchikey_col="chem_inchikey")
    assert_no_test_chemicals(kept, keys, cas_col="cas", inchikey_col="chem_inchikey",
                             what="ECOTOX auxiliary")
    out = pd.DataFrame({
        "cas": kept["cas"].astype("string"),
        "chem_inchikey": kept["chem_inchikey"].astype("string"),
        "smiles": kept["chem_smiles"].astype("string"),
        "species": kept["latin_name"].astype("string").fillna("<NA>"),
        "taxon_group": kept["taxon_group"].astype("string"),
        "duration_h": pd.to_numeric(kept["duration_h"], errors="coerce"),
        "endpoint": kept["endpoint_norm"].astype("string"),
        "effect_norm": kept["effect_norm"].astype("string"),
        "y": pd.to_numeric(kept["log10_conc_mol_L"], errors="coerce"),
    }).dropna(subset=["y", "smiles"]).reset_index(drop=True)
    out["aux_source"] = "ecotox"
    info = {"n_rows_table": int(n0), "n_after_filters": len(sub), **counts,
            "n_final": len(out), "n_chemicals": int(out["cas"].nunique()),
            "filters": {"endpoints": list(endpoints), "effects": list(effects),
                        "taxon_groups": list(taxon_groups),
                        "durations_h": list(durations_h), "freshwater_only": freshwater_only,
                        "drop_censored": drop_censored}}
    return out, info
