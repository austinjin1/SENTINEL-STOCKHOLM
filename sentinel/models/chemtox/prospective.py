# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""A **new** prospective test set for ChemTox, built from ECOTOX with ADORE's curation rules.

Why this module exists
----------------------
The single pre-registered ADORE test pass is spent (``docs/preregistration/chemtox_test_plan.md``):
its numbers stay as reported and no improved model may be re-claimed on those rows
(RESEARCH_PLAN §15.5 amendment 1). An improved model therefore needs a *previously unseen* test
set. ADORE's core is ECOTOX, so the natural prospective set is **ECOTOX records that ADORE does
not contain**, curated by exactly the same rules, restricted to chemicals ADORE never saw.

ADORE's curation rules, recovered from the release itself
---------------------------------------------------------
``processed/ecotox_mortality_filtered`` (33,448 rows) and the challenge files were inspected
column by column (2026-10-24); every value set below is what ADORE actually contains:

===========================  ===========================================================
``result_effect``            ``MOR`` only
``result_endpoint``          ``LC50`` (25,976 rows of t-F2F) or ``EC50`` (138)
``result_obs_duration_mean`` exactly {24, 48, 72, 96} h
``test_media_type``          ``FW`` or ``SW``
``result_conc1_type``        ``A`` / ``F`` / ``T`` (ECOTOX also has ``NC``/``D``: absent)
``test_exposure_type``       ``S`` / ``F`` / ``R`` / ``NR`` (no filter: all four appear)
``test_location``            ``LAB`` / ``NR`` / ``FIELDA`` (no filter)
``result_conc1_mean_op``     missing, ``~`` or ``min_max_average``; never ``<`` or ``>``
taxa                         fish / crustaceans ("crusta") / algae, 203 species in total
chemicals                    2,408, of which 2,191 organic, 207 organic multicomponent,
                             10 inorganic — i.e. ADORE is a *carbon-containing* chemical space
===========================  ===========================================================

Representation reproducibility (verified here, 2026-10-24): ADORE's ``chem_MACCS_fp`` is exactly
RDKit ``MACCSkeys.GenMACCSKeys`` on ``chem_rdkit_can_smiles`` (300/300 unique chemicals checked,
bit-identical), ``chem_Morgan_fp`` is exactly the RDKit Morgan generator with radius 2 and 2048
bits (200/200), and ``chem_rdkit_clogp`` is exactly ``Crippen.MolLogP`` (max |Δ| 2e-15). So the
two fingerprints and logP used by the frozen configurations can be computed for new chemicals in
the *same* definition. ``chem_mw`` matches ``Descriptors.MolWt`` to 0.01 on average.
``chem_ws`` (water solubility) and ``chem_mp`` (melting point) are experimental values ADORE
compiled per chemical and are **not available** for chemicals outside ADORE; they are left missing
and take the training mean inside :class:`~sentinel.models.chemtox.features.PublishedFeatures`
(see the pre-registration for the ablation that quantifies what that costs on CV).
``chem_pcp_fp`` (PubChem) and ``chem_ToxPrint_fp`` cannot be computed locally, so prospective
scoring is restricted to feature specs using MACCS, Morgan or no fingerprint.

Tracks (defined before any model is scored)
-------------------------------------------
``fish_novel``   primary — fish rows, ADORE species, carbon-containing chemicals **not in ADORE**
``crust_novel``  secondary — the same for crustaceans (cross-taxa generalisation)
``algae_novel``  secondary — the same for algae (tiny; reported with its n)
``noncarbon``    secondary, out-of-domain extension — novel chemicals without carbon (metal salts
                 and other inorganics, which ADORE's chemical space essentially excludes)
``temporal``     secondary — rows from publications dated after ADORE's latest reference year
                 (2021), whatever the chemical; reported with the power it has

Exclusion logic (asserted in code, :func:`assert_disjoint_from_adore`)
----------------------------------------------------------------------
A prospective row is kept only if its chemical is absent from **every** ADORE challenge file by
CAS *and* by InChIKey (so InChIKey-level overlap is 0 by construction for the novel tracks), and
its ECOTOX ``result_id`` is absent from ADORE's filtered table. Species are restricted to ADORE's
own species so the published taxonomic block is defined; the taxonomy columns are copied from
ADORE per species (they are species-level constants, which is asserted).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.sources import adore as A

from . import features as F

ECOTOX_TABLE = "ecotox/ecotox_aquatic.parquet"
#: ADORE's latest reference publication year (computed from the release: max over its 2,432
#: matched ECOTOX references). Rows published after this year cannot be in ADORE.
ADORE_LAST_REFERENCE_YEAR = 2021
#: curation rules (see the module docstring)
ENDPOINTS = ("LC50", "EC50")
EFFECTS = ("MOR",)
DURATIONS_H = (24.0, 48.0, 72.0, 96.0)
MEDIA_TYPES = ("FW", "SW")
CONC_TYPES = ("A", "F", "T")
TAXON_GROUPS = ("fish", "crustaceans", "algae")
#: ADORE's own taxon-group spelling, by our processed ECOTOX spelling
ADORE_TAXON_GROUP = {"fish": "fish", "crustaceans": "crusta", "algae": "algae"}
CENSOR_KINDS_KEPT = ("", "approximate")
#: ADORE's chemical classes (``processed/chemicals/chem_ids.parquet``): 2,191 ``organic`` +
#: 207 ``organic_multicomponent`` + 10 ``inorganic``, and **no** ``organometallic``. A prospective
#: chemical counts as inside ADORE's domain of applicability only if it falls in the two classes
#: that make up 99.6% of ADORE.
ADORE_DOMAIN_CLASSES = ("organic", "organic_multicomponent")

#: columns the ADORE pipeline needs that are copied from ADORE per species
TAXONOMY_COLUMNS = (
    "tax_all", "tax_name", "tax_class", "tax_order", "tax_family", "tax_genus", "tax_species",
    "species_number", "tax_pdm_available", "tax_eco_climate", "tax_eco_ecozone", "tax_eco_food",
    "tax_eco_migrate5", "tax_eco_migrate2", "tax_ps_ampv", "tax_ps_ampkap", "tax_ps_amppm",
    "tax_lh_amd", "tax_lh_lbcm", "tax_lh_lpcm", "tax_lh_licm", "tax_lh_ri#/d",
)
#: the output schema (ADORE column names, so the existing pipeline can consume it unchanged)
OUTPUT_COLUMNS = (
    "prospective_track", "in_adore_domain", "parent_novel", "parent_key",
    "nn_tanimoto_adore_train", "result_id", "test_id", "reference_number",
    "publication_year",
    "test_cas", "chem_inchikey", "chem_name", "chem_rdkit_can_smiles", "chem_MACCS_fp",
    "chem_Morgan_fp", "chem_mw", "chem_rdkit_clogp", "chem_ws", "chem_mp",
    "result_effect", "result_endpoint", "result_obs_duration_mean", "test_media_type",
    "test_exposure_type", "result_conc1_type", "test_location", "test_organism_lifestage",
    "tax_gs", "tax_group", "result_conc1_mean", "result_conc1_mean_mol_log",
    "result_conc1_mean_log", "n_carbon", "has_metal", "is_multicomponent", "structure_class",
    *TAXONOMY_COLUMNS,
)


def _up(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def _keys(values) -> set[str]:
    s = _up(pd.Series(values, dtype="object"))
    s = s[s.notna() & ~s.isin(["", "<NA>", "NAN", "NONE"])]
    return set(s.unique().tolist())


# ------------------------------------------------------------------------ what ADORE contains


@dataclass(frozen=True)
class AdoreInventory:
    """Everything ADORE contains that a prospective row must not duplicate."""

    cas: frozenset[str]
    inchikey: frozenset[str]
    result_id: frozenset[str]
    species: frozenset[str]
    taxonomy: pd.DataFrame        # one row per species, TAXONOMY_COLUMNS
    n_rows: int
    parent: frozenset[str] = frozenset()   # salt/stereochemistry-insensitive structure keys

    def summary(self) -> dict:
        return {"n_cas": len(self.cas), "n_inchikey": len(self.inchikey),
                "n_result_id": len(self.result_id), "n_species": len(self.species),
                "n_rows": self.n_rows, "n_parent_keys": len(self.parent)}


def adore_inventory(*, root: str | Path | None = None,
                    challenges: tuple[str, ...] = A.CHALLENGES) -> AdoreInventory:
    """Chemicals, rows, species and per-species taxonomy of the **whole** ADORE release.

    The filtered table (all 33,448 curated rows) plus every challenge file, so a chemical that
    appears anywhere in ADORE — in any challenge, in any partition — is excluded.
    """
    frames = []
    filt = A.release_dir(root) / "processed" / "ecotox_mortality_filtered.parquet"
    cols = ["result_id", "test_cas", "chem_inchikey", "tax_gs"]
    if filt.exists():
        frames.append(pd.read_parquet(filt, columns=cols))
    tax_frames, smiles = [], set()
    for ch in challenges:
        p = A.challenge_path(ch, root)
        if not p.exists():
            continue
        frames.append(pd.read_parquet(p, columns=cols))
        tax_frames.append(pd.read_parquet(p, columns=["tax_gs", *TAXONOMY_COLUMNS]))
        s = pd.read_parquet(p, columns=["chem_rdkit_can_smiles"])["chem_rdkit_can_smiles"]
        smiles |= set(s.astype("string").dropna().unique().tolist())
    if not frames:
        raise FileNotFoundError("no ADORE release files found")
    all_rows = pd.concat(frames, ignore_index=True)
    tax = pd.concat(tax_frames, ignore_index=True).drop_duplicates()
    tax["tax_gs"] = tax["tax_gs"].astype("string").str.strip()
    # taxonomy must be a species-level constant, otherwise copying it per species is wrong
    counts = tax.groupby("tax_gs", dropna=False).size()
    bad = counts[counts > 1]
    if len(bad):
        raise AssertionError(f"ADORE taxonomy is not species-constant for {len(bad)} species: "
                             f"{list(bad.index[:5])}")
    parents = {v for v in parent_keys(sorted(smiles)).values() if v}
    return AdoreInventory(
        cas=frozenset(_keys(all_rows["test_cas"])),
        inchikey=frozenset(_keys(all_rows["chem_inchikey"])),
        result_id=frozenset(all_rows["result_id"].astype("string").str.strip().dropna().tolist()),
        species=frozenset(tax["tax_gs"].dropna().tolist()),
        taxonomy=tax.reset_index(drop=True),
        n_rows=int(len(all_rows)),
        parent=frozenset(parents),
    )


# ------------------------------------------------------------------------ curation on ECOTOX


def ecotox_path(root: str | Path | None = None) -> Path:
    return (Path(root) if root else paths.PROCESSED) / ECOTOX_TABLE


def curate_ecotox(*, root: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """ECOTOX rows that satisfy ADORE's curation rules (module docstring), with a counts report."""
    path = ecotox_path(root)
    if not path.exists():
        raise FileNotFoundError(f"processed ECOTOX table not found: {path}")
    cols = ["result_id", "test_id", "reference_number", "publication_year", "cas", "chemical_name",
            "chem_inchikey", "chem_smiles", "latin_name", "taxon_group", "endpoint_norm",
            "effect_norm", "duration_h", "media_type_norm", "exposure_type_norm",
            "conc1_type_norm", "test_location_norm", "organism_lifestage_norm", "censor_kind",
            "censored", "conc_mg_L", "log10_conc_mol_L", "log10_conc_mg_L", "conc1_mean_op"]
    df = pd.read_parquet(path, columns=cols)
    steps = {"ecotox_rows": int(len(df))}

    def keep(mask: np.ndarray, label: str) -> np.ndarray:
        steps[label] = int(mask.sum())
        return mask

    m = keep(df["effect_norm"].astype("string").isin(list(EFFECTS)).fillna(False).to_numpy(), "effect_MOR")
    m &= df["endpoint_norm"].astype("string").isin(list(ENDPOINTS)).fillna(False).to_numpy()
    steps["and_endpoint"] = int(m.sum())
    m &= pd.to_numeric(df["duration_h"], errors="coerce").isin(list(DURATIONS_H)).fillna(False).to_numpy()
    steps["and_duration"] = int(m.sum())
    m &= df["taxon_group"].astype("string").str.lower().isin(list(TAXON_GROUPS)).fillna(False).to_numpy()
    steps["and_taxon_group"] = int(m.sum())
    m &= df["media_type_norm"].astype("string").isin(list(MEDIA_TYPES)).fillna(False).to_numpy()
    steps["and_media_type"] = int(m.sum())
    m &= df["conc1_type_norm"].astype("string").isin(list(CONC_TYPES)).fillna(False).to_numpy()
    steps["and_conc_type"] = int(m.sum())
    ck = df["censor_kind"].astype("string").fillna("")
    m &= ck.isin(list(CENSOR_KINDS_KEPT)).to_numpy()
    steps["and_not_censored"] = int(m.sum())
    m &= df["log10_conc_mol_L"].notna().to_numpy()
    steps["and_molar_target"] = int(m.sum())
    m &= df["chem_smiles"].notna().to_numpy()
    steps["and_smiles"] = int(m.sum())
    out = df.loc[m].copy()
    out["species_us"] = out["latin_name"].astype("string").str.strip().str.replace(" ", "_", regex=False)
    steps["n_chemicals"] = int(_up(out["cas"]).nunique())
    steps["rules"] = {"endpoints": list(ENDPOINTS), "effects": list(EFFECTS),
                      "durations_h": list(DURATIONS_H), "media_types": list(MEDIA_TYPES),
                      "conc_types": list(CONC_TYPES), "taxon_groups": list(TAXON_GROUPS),
                      "censor_kinds_kept": list(CENSOR_KINDS_KEPT),
                      "target": "log10_conc_mol_L (log10 mol/L), reported mean only"}
    return out.reset_index(drop=True), steps


# ------------------------------------------------------------------------ chemistry


def chem_ids_path(root: str | Path | None = None) -> Path:
    return (Path(root) if root else paths.PROCESSED) / "chemicals/chem_ids.parquet"


def parent_key(smiles: str) -> str | None:
    """A salt-, charge- and stereochemistry-insensitive structural key for a molecule.

    Two chemicals share a parent key when they are the same organic structure up to counter-ion,
    protonation state and stereochemistry: aniline and aniline hydrochloride, a racemate and its
    single enantiomers, the sodium and the zinc salt of a dithiocarbamate. A test chemical that
    merely differs from a training chemical in those respects is **not** a novel structure, and the
    pre-registration reports it separately (see ``parent_novel``).

    The key is the first block (the skeleton block) of the InChIKey of the largest fragment with
    charges neutralised and stereochemistry removed; ``None`` when RDKit cannot parse the SMILES.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles) if smiles is not None else "")
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    big = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    big = Chem.RWMol(big)
    for atom in big.GetAtoms():                      # neutralise
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    try:
        mol2 = big.GetMol()
        Chem.SanitizeMol(mol2)
        Chem.RemoveStereochemistry(mol2)
        key = Chem.MolToInchiKey(mol2)
    except Exception:  # noqa: BLE001 — exotic valences; fall back to the canonical SMILES
        try:
            return Chem.MolToSmiles(big.GetMol())
        except Exception:  # noqa: BLE001
            return None
    return key.split("-")[0] if key else None


def parent_keys(smiles) -> dict[str, str | None]:
    """``parent_key`` per unique SMILES."""
    return {str(s): parent_key(s) for s in pd.unique(pd.Series(smiles).astype("string").dropna())}


def rdkit_chemistry(smiles: pd.Series) -> pd.DataFrame:
    """MACCS/Morgan bit strings, MolWt and Crippen logP per unique SMILES, in ADORE's definitions."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for s in pd.unique(smiles.astype("string").dropna()):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            rows.append({"chem_rdkit_can_smiles": s, "parse_ok": False})
            continue
        rows.append({
            "chem_rdkit_can_smiles": Chem.MolToSmiles(mol),
            "_smiles_in": s,
            "parse_ok": True,
            "chem_MACCS_fp": MACCSkeys.GenMACCSKeys(mol).ToBitString(),
            "chem_Morgan_fp": morgan.GetFingerprint(mol).ToBitString(),
            "chem_mw": float(Descriptors.MolWt(mol)),
            "chem_rdkit_clogp": float(Crippen.MolLogP(mol)),
            "n_carbon_rdkit": int(sum(a.GetSymbol() == "C" for a in mol.GetAtoms())),
            "n_fragments": len(Chem.GetMolFrags(mol)),
        })
    out = pd.DataFrame(rows)
    if "_smiles_in" not in out.columns:
        out["_smiles_in"] = out["chem_rdkit_can_smiles"]
    return out


# ------------------------------------------------------------------------ assembly


def assert_disjoint_from_adore(df: pd.DataFrame, inv: AdoreInventory, *, what: str = "prospective",
                               allow_known_chemicals: bool = False) -> dict:
    """Raise unless every row is new to ADORE; returns the overlap counts it verified."""
    cas_hit = _up(df["test_cas"]).isin(inv.cas).fillna(False).to_numpy()
    ik_hit = _up(df["chem_inchikey"]).isin(inv.inchikey).fillna(False).to_numpy()
    rid_hit = df["result_id"].astype("string").str.strip().isin(inv.result_id).fillna(False).to_numpy()
    rep = {"n_rows": int(len(df)), "cas_overlap_rows": int(cas_hit.sum()),
           "inchikey_overlap_rows": int(ik_hit.sum()), "result_id_overlap_rows": int(rid_hit.sum()),
           "n_chemicals": int(_up(df["test_cas"]).nunique()),
           "n_inchikeys": int(_up(df["chem_inchikey"]).nunique())}
    if rid_hit.any():
        raise AssertionError(f"{what}: {rep['result_id_overlap_rows']} rows are ADORE rows (result_id)")
    if not allow_known_chemicals and (cas_hit.any() or ik_hit.any()):
        raise AssertionError(
            f"{what}: chemical overlap with ADORE — {rep['cas_overlap_rows']} rows by CAS, "
            f"{rep['inchikey_overlap_rows']} rows by InChIKey (must be 0)")
    return rep


def nn_tanimoto(smiles, reference_fps) -> dict[str, float]:
    """Max Morgan/Tanimoto similarity of each unique SMILES to a reference fingerprint list."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out: dict[str, float] = {}
    for s in pd.unique(pd.Series(smiles).astype("string").dropna()):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None or not reference_fps:
            out[str(s)] = float("nan")
            continue
        fp = gen.GetFingerprint(mol)
        out[str(s)] = float(max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)))
    return out


def adore_train_fingerprints(*, challenge: str = "t-F2F", split: str = A.PRIMARY_SPLIT,
                             root: str | Path | None = None) -> list:
    """Morgan fingerprints of ADORE's **training-partition** chemicals for that challenge/split."""
    from rdkit import DataStructs

    train = pd.read_parquet(A.challenge_path(challenge, root),
                            columns=["test_cas", "chem_Morgan_fp", split])
    a = train[split].astype("string")
    train = train[(a.notna() & (a != "test")).to_numpy()].drop_duplicates("test_cas")
    return [DataStructs.CreateFromBitString(str(s)) for s in train["chem_Morgan_fp"].dropna()]


def to_adore_schema(df: pd.DataFrame, inv: AdoreInventory) -> tuple[pd.DataFrame, dict]:
    """Curated ECOTOX rows → ADORE's column names, fingerprints and per-species taxonomy.

    Used for both the prospective test frame and the ECOTOX **augmentation** rows, so the two
    are featurised by exactly the same code path as ADORE itself.
    """
    info: dict = {}
    chem = rdkit_chemistry(df["chem_smiles"])
    ok = chem[chem["parse_ok"]].copy()
    out = df.merge(ok.drop(columns=["parse_ok"]), left_on="chem_smiles", right_on="_smiles_in",
                   how="inner").drop(columns=["_smiles_in"])
    info["rows_after_rdkit_parse"] = int(len(out))

    tax = inv.taxonomy.rename(columns={"tax_gs": "species_us"})
    out = out.merge(tax, on="species_us", how="left")
    out["tax_gs"] = out["species_us"]
    out["tax_group"] = out["taxon_group"].astype("string").map(ADORE_TAXON_GROUP)
    info["rows_missing_adore_taxonomy"] = int(out["tax_eco_climate"].isna().sum())

    out = out.rename(columns={
        "cas": "test_cas", "chemical_name": "chem_name",
        "duration_h": "result_obs_duration_mean", "media_type_norm": "test_media_type",
        "exposure_type_norm": "test_exposure_type", "conc1_type_norm": "result_conc1_type",
        "test_location_norm": "test_location", "organism_lifestage_norm": "test_organism_lifestage",
        "endpoint_norm": "result_endpoint", "effect_norm": "result_effect",
        "log10_conc_mol_L": "result_conc1_mean_mol_log", "log10_conc_mg_L": "result_conc1_mean_log",
        "conc_mg_L": "result_conc1_mean",
    })
    out["in_adore_domain"] = out["structure_class"].astype("string").isin(
        list(ADORE_DOMAIN_CLASSES)).fillna(False)
    # structural novelty beyond the identifier level: a salt, a single stereoisomer or a different
    # counter-ion of an ADORE chemical has a *different* CAS and InChIKey but is not a new structure
    pk = parent_keys(out["chem_rdkit_can_smiles"])
    out["parent_key"] = out["chem_rdkit_can_smiles"].astype("string").map(pk)
    out["parent_novel"] = (~out["parent_key"].isin(inv.parent)) & out["parent_key"].notna()
    info["rows_parent_novel"] = int(out["parent_novel"].sum())
    info["rows_parent_in_adore"] = int((~out["parent_novel"]).sum())
    try:
        sims = nn_tanimoto(out["chem_rdkit_can_smiles"], adore_train_fingerprints())
        out["nn_tanimoto_adore_train"] = out["chem_rdkit_can_smiles"].astype("string").map(sims)
    except Exception as exc:  # noqa: BLE001 — a descriptor, never a filter
        info["nn_tanimoto_error"] = f"{type(exc).__name__}: {exc}"
        out["nn_tanimoto_adore_train"] = np.nan
    out["chem_ws"] = np.nan      # experimental values ADORE compiled; unavailable outside ADORE
    out["chem_mp"] = np.nan
    out["result_obs_duration_mean"] = pd.to_numeric(out["result_obs_duration_mean"], errors="coerce")
    keep = [c for c in OUTPUT_COLUMNS if c in out.columns]
    return out[keep].reset_index(drop=True), info


def build(*, root: str | Path | None = None, adore_root: str | Path | None = None
          ) -> tuple[pd.DataFrame, dict]:
    """Build the prospective test frame (all tracks) in ADORE's schema, with a provenance report."""
    inv = adore_inventory(root=adore_root)
    cur, steps = curate_ecotox(root=root)
    report: dict = {"adore": inv.summary(), "curation": steps,
                    "adore_last_reference_year": ADORE_LAST_REFERENCE_YEAR}

    # chemical identity / structure class from the data workstream's chemical table
    C = pd.read_parquet(chem_ids_path(root), columns=["cas", "inchikey", "smiles_canonical",
                                                      "n_carbon", "has_metal", "is_multicomponent",
                                                      "structure_class", "in_adore"])
    C["cas_u"] = _up(C["cas"])
    cur["cas_u"] = _up(cur["cas"])
    df = cur.merge(C.drop(columns=["cas"]).rename(columns={"inchikey": "chem_inchikey_ids"}),
                   on="cas_u", how="left")
    report["chem_ids_missing_rows"] = int(df["structure_class"].isna().sum())

    # chemical novelty and row novelty w.r.t. ADORE
    df["chem_in_adore"] = (_up(df["cas_u"]).isin(inv.cas)
                           | _up(df["chem_inchikey"]).isin(inv.inchikey)).fillna(False)
    df["row_in_adore"] = df["result_id"].astype("string").str.strip().isin(inv.result_id).fillna(False)
    df["species_in_adore"] = df["species_us"].isin(inv.species)
    year = pd.to_numeric(df["publication_year"], errors="coerce")
    df["after_adore_cutoff"] = (year > ADORE_LAST_REFERENCE_YEAR).fillna(False)
    carbon = (pd.to_numeric(df["n_carbon"], errors="coerce").fillna(0) >= 1)
    df["carbon_containing"] = carbon

    report["pool"] = {
        "rows": int(len(df)),
        "rows_in_adore": int(df["row_in_adore"].sum()),
        "rows_novel_chemical": int((~df["chem_in_adore"]).sum()),
        "rows_novel_chemical_adore_species": int((~df["chem_in_adore"] & df["species_in_adore"]).sum()),
        "rows_after_cutoff": int(df["after_adore_cutoff"].sum()),
    }

    # ---- tracks
    base = ~df["row_in_adore"]
    novel = base & ~df["chem_in_adore"] & df["species_in_adore"]
    tracks = {
        "fish_novel": novel & carbon & (df["taxon_group"].astype("string") == "fish"),
        "crust_novel": novel & carbon & (df["taxon_group"].astype("string") == "crustaceans"),
        "algae_novel": novel & carbon & (df["taxon_group"].astype("string") == "algae"),
        "noncarbon": novel & ~carbon,
        "temporal": base & df["after_adore_cutoff"] & df["species_in_adore"] & carbon,
    }
    frames = []
    for name, mask in tracks.items():
        sub = df.loc[mask.to_numpy()].copy()
        sub["prospective_track"] = name
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    # a row can belong to two tracks (a post-2021 publication about a novel chemical is both
    # fish_novel and temporal). Each (row, track) pair appears once; the cross-track count is how
    # many rows appear under more than one track, which is what a reader needs to know.
    per_row = out.groupby(out["result_id"].astype("string"))["prospective_track"].nunique()
    report["rows_in_more_than_one_track"] = int((per_row > 1).sum())
    report["duplicate_row_track_pairs"] = int(
        out.duplicated(subset=["result_id", "prospective_track"]).sum())

    out, schema_info = to_adore_schema(out, inv)
    report.update(schema_info)

    # ---- integrity assertions per track
    report["tracks"] = {}
    for name in tracks:
        sub = out[out["prospective_track"] == name]
        if sub.empty:
            report["tracks"][name] = {"n_rows": 0}
            continue
        rep = assert_disjoint_from_adore(sub, inv, what=f"prospective[{name}]",
                                         allow_known_chemicals=(name == "temporal"))
        y = pd.to_numeric(sub["result_conc1_mean_mol_log"], errors="coerce")
        rep.update({
            "n_species": int(sub["tax_gs"].nunique()),
            "n_references": int(sub["reference_number"].astype("string").nunique()),
            "taxon_groups": sub["tax_group"].astype("string").value_counts().to_dict(),
            "durations_h": sub["result_obs_duration_mean"].value_counts().to_dict(),
            "endpoints": sub["result_endpoint"].astype("string").value_counts().to_dict(),
            "y_mean": float(y.mean()), "y_sd": float(y.std()),
            "y_min": float(y.min()), "y_max": float(y.max()),
            "publication_year_median": float(pd.to_numeric(sub["publication_year"],
                                                           errors="coerce").median()),
            "structure_class": sub["structure_class"].astype("string").value_counts().to_dict(),
        })
        rep["n_rows_in_adore_domain"] = int(sub["in_adore_domain"].sum())
        rep["n_rows_parent_novel"] = int(sub["parent_novel"].sum())
        rep["n_chemicals_parent_novel"] = int(_up(sub.loc[sub["parent_novel"].to_numpy(),
                                                          "test_cas"]).nunique())
        sim = pd.to_numeric(sub["nn_tanimoto_adore_train"], errors="coerce")
        rep["nn_tanimoto_rows"] = {"median": float(sim.median()),
                                   "frac_equal_1": float((sim >= 0.999).mean()),
                                   "frac_below_0.4": float((sim < 0.4).mean())}
        dom = sub[sub["in_adore_domain"].to_numpy()]
        if len(dom):
            yd = pd.to_numeric(dom["result_conc1_mean_mol_log"], errors="coerce")
            rep["adore_domain"] = {
                "n_rows": int(len(dom)), "n_chemicals": int(_up(dom["test_cas"]).nunique()),
                "n_species": int(dom["tax_gs"].nunique()),
                "n_references": int(dom["reference_number"].astype("string").nunique()),
                "y_mean": float(yd.mean()), "y_sd": float(yd.std()),
                "replicate_sd": replicate_noise(dom),
            }
            strict = dom[dom["parent_novel"].to_numpy()]
            if len(strict):
                ys = pd.to_numeric(strict["result_conc1_mean_mol_log"], errors="coerce")
                sims = pd.to_numeric(strict["nn_tanimoto_adore_train"], errors="coerce")
                rep["adore_domain_parent_novel"] = {
                    "n_rows": int(len(strict)),
                    "n_chemicals": int(_up(strict["test_cas"]).nunique()),
                    "n_species": int(strict["tax_gs"].nunique()),
                    "n_references": int(strict["reference_number"].astype("string").nunique()),
                    "y_mean": float(ys.mean()), "y_sd": float(ys.std()),
                    "replicate_sd": replicate_noise(strict),
                    "nn_tanimoto_median": float(sims.median()),
                    "nn_tanimoto_frac_equal_1": float((sims >= 0.999).mean()),
                }
        report["tracks"][name] = rep
    report["domain_of_applicability"] = tanimoto_to_adore(out, root=adore_root)
    # the duplicate-free union of chemicals across the novel tracks
    nov = out[out["prospective_track"].isin(["fish_novel", "crust_novel", "algae_novel"])]
    report["novel_union"] = {"n_rows": int(len(nov)),
                             "n_chemicals": int(_up(nov["test_cas"]).nunique()),
                             "n_inchikeys": int(_up(nov["chem_inchikey"]).nunique())}
    return out, report


def prospective_path(root: str | Path | None = None) -> Path:
    return (Path(root) if root else paths.PROCESSED) / "chemtox_prospective/prospective.parquet"


#: tracks whose novelty is the **chemical** (so a training row may not carry their chemicals)
NOVEL_TRACKS = ("fish_novel", "crust_novel", "algae_novel", "noncarbon")
#: tracks whose novelty is the **publication date** (their chemicals may legitimately be known;
#: what must not be trained on is any row of the same publication, or any row of the test set)
TEMPORAL_TRACKS = ("temporal",)


def prospective_keys(root: str | Path | None = None) -> dict:
    """The exclusion key sets of the frozen prospective file, by what each track holds out.

    * ``cas`` / ``inchikey`` — chemicals of the **novel-chemical** tracks: a training row carrying
      one of these would destroy the chemical-generalisation claim.
    * ``result_id`` — every prospective row, whatever the track: a test row is never a training row.
    * ``reference_number`` — publications of the **temporal** tracks: the temporal track holds out
      *studies*, so no row of the same publication may be trained on (its chemicals may be).
    """
    df = pd.read_parquet(prospective_path(root))
    novel = df[df["prospective_track"].isin(list(NOVEL_TRACKS))]
    temporal = df[df["prospective_track"].isin(list(TEMPORAL_TRACKS))]
    out = {
        "cas": frozenset(_keys(novel["test_cas"])),
        "inchikey": frozenset(_keys(novel["chem_inchikey"])),
        "result_id": frozenset(df["result_id"].astype("string").str.strip().dropna().tolist()),
        "reference_number": frozenset(
            temporal["reference_number"].astype("string").str.strip().dropna().tolist()),
        "parent": frozenset(novel["parent_key"].astype("string").dropna().unique().tolist())
        if "parent_key" in novel.columns else frozenset(),
    }
    return out


def prospective_chemical_keys(root: str | Path | None = None) -> tuple[frozenset[str], frozenset[str]]:
    """(CAS, InChIKey) of the novel-chemical tracks — kept for callers that only need those."""
    k = prospective_keys(root)
    return k["cas"], k["inchikey"]


def assert_prospective_chemicals_absent(df: pd.DataFrame, *, cas_col: str = "test_cas",
                                        inchikey_col: str | None = "chem_inchikey",
                                        result_id_col: str | None = "result_id",
                                        reference_col: str | None = "reference_number",
                                        parent_col: str | None = None,
                                        what: str = "training data",
                                        root: str | Path | None = None) -> dict:
    """Raise if any row would leak the prospective test set into training.

    This is the guard that keeps the prospective set a test set: every fit — ADORE rows, ECOTOX
    augmentation, pretraining corpora, ensemble weights — must pass it before the prospective rows
    are scored. Three independent checks, matching what each track holds out (see
    :func:`prospective_keys`): novel-track chemicals by CAS **or** InChIKey, any prospective row by
    ``result_id``, and any publication of the temporal track by ``reference_number``.
    """
    keys = prospective_keys(root)
    counts = {}
    hit = np.zeros(len(df), dtype=bool)
    if cas_col in df.columns:
        h = _up(df[cas_col]).isin(keys["cas"]).fillna(False).to_numpy()
        counts["cas"] = int(h.sum())
        hit |= h
    if inchikey_col and inchikey_col in df.columns:
        h = _up(df[inchikey_col]).isin(keys["inchikey"]).fillna(False).to_numpy()
        counts["inchikey"] = int(h.sum())
        hit |= h
    if result_id_col and result_id_col in df.columns:
        h = df[result_id_col].astype("string").str.strip().isin(keys["result_id"]).fillna(False).to_numpy()
        counts["result_id"] = int(h.sum())
        hit |= h
    if parent_col and parent_col in df.columns and keys.get("parent"):
        # the salt/stereoisomer level. ADORE itself is *not* filtered this way — a prospective
        # chemical whose parent is an ADORE chemical is kept and flagged ``parent_novel = False``
        # (pre-registration §1.3) — but data we *choose* to add (ECOTOX augmentation) is filtered,
        # because there is no reason to add a near-duplicate of a test chemical.
        h = df[parent_col].astype("string").isin(keys["parent"]).fillna(False).to_numpy()
        counts["parent_key"] = int(h.sum())
        hit |= h
    if reference_col and reference_col in df.columns and keys["reference_number"]:
        h = (df[reference_col].astype("string").str.strip()
             .isin(keys["reference_number"]).fillna(False).to_numpy())
        counts["reference_number"] = int(h.sum())
        hit |= h
    if hit.any():
        raise AssertionError(f"{what}: {int(hit.sum())} rows would leak the prospective test set "
                             f"{counts} (novel-track chemicals by CAS/InChIKey, any prospective "
                             f"result_id, any temporal-track publication)")
    return {"n_rows": int(len(df)), "checked": counts,
            "n_prospective_cas": len(keys["cas"]), "n_prospective_inchikey": len(keys["inchikey"]),
            "n_prospective_rows": len(keys["result_id"]),
            "n_temporal_references": len(keys["reference_number"])}


def augmentation_rows(*, taxon_groups: tuple[str, ...] = TAXON_GROUPS,
                      require_adore_species: bool = True,
                      root: str | Path | None = None,
                      adore_root: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Extra ECOTOX training rows in ADORE's schema: not in ADORE, not a prospective chemical.

    These rows are *additional training data* for the prospective evaluation: ADORE-curated
    acute LC50/EC50 records that ADORE itself does not contain (ADORE kept only chemicals for
    which it could compile every representation), with every prospective-test chemical removed by
    CAS **and** InChIKey and the result asserted clean. ``require_adore_species`` keeps only
    species for which ADORE's Add-my-Pet block exists (otherwise the taxonomic features of those
    rows would be entirely imputed).
    """
    inv = adore_inventory(root=adore_root)
    cur, steps = curate_ecotox(root=root)
    keys = prospective_keys(root)
    C = pd.read_parquet(chem_ids_path(root), columns=["cas", "n_carbon", "has_metal",
                                                      "is_multicomponent", "structure_class"])
    C["cas_u"] = _up(C["cas"])
    cur["cas_u"] = _up(cur["cas"])
    df = cur.merge(C.drop(columns=["cas"]), on="cas_u", how="left")
    keep = ~df["result_id"].astype("string").str.strip().isin(inv.result_id).fillna(False)
    info = {"curation": steps, "n_curated": int(len(df)), "n_after_drop_adore_rows": int(keep.sum())}
    keep &= ~(_up(df["cas_u"]).isin(keys["cas"])
              | _up(df["chem_inchikey"]).isin(keys["inchikey"])).fillna(False)
    info["n_after_drop_prospective_chemicals"] = int(keep.sum())
    keep &= ~df["result_id"].astype("string").str.strip().isin(keys["result_id"]).fillna(False)
    keep &= ~df["reference_number"].astype("string").str.strip().isin(
        keys["reference_number"]).fillna(False)
    info["n_after_drop_prospective_rows_and_temporal_studies"] = int(keep.sum())
    keep &= df["taxon_group"].astype("string").isin(list(taxon_groups)).fillna(False)
    if require_adore_species:
        keep &= df["species_us"].isin(inv.species)
    info["n_after_species_and_taxa"] = int(keep.sum())
    out, schema_info = to_adore_schema(df.loc[keep.to_numpy()].copy(), inv)
    out["prospective_track"] = "augmentation"
    n_before = len(out)
    if keys.get("parent"):
        out = out[~out["parent_key"].astype("string").isin(keys["parent"]).fillna(False)]
        out = out.reset_index(drop=True)
    info["n_after_drop_prospective_parents"] = int(len(out))
    info["n_dropped_as_prospective_parents"] = int(n_before - len(out))
    assert_prospective_chemicals_absent(out, parent_col="parent_key",
                                        what="ECOTOX augmentation", root=root)
    info.update(schema_info)
    info.update({"n_rows": int(len(out)), "n_chemicals": int(_up(out["test_cas"]).nunique()),
                 "n_species": int(out["tax_gs"].nunique()),
                 "taxon_groups": out["tax_group"].astype("string").value_counts().to_dict(),
                 "require_adore_species": bool(require_adore_species)})
    return out, info


def replicate_noise(df: pd.DataFrame) -> dict:
    """Pooled within-(chemical, species, duration) SD of the target — a noise floor for this set.

    Computed from the prospective rows' own replicate structure, the same way the ADORE replicate
    floor (0.383 log units) was computed; it bounds how low any model's RMSE can go here.
    """
    y = pd.to_numeric(df["result_conc1_mean_mol_log"], errors="coerce")
    key = (_up(df["test_cas"]).fillna("<NA>") + "|" + df["tax_gs"].astype("string").fillna("<NA>")
           + "|" + df["result_obs_duration_mean"].astype("string").fillna("<NA>"))
    g = pd.DataFrame({"y": y, "k": key}).dropna().groupby("k")["y"]
    n = g.size()
    var = g.var(ddof=1)
    use = n[n >= 2].index
    if len(use) == 0:
        return {"n_groups": 0, "pooled_sd": None}
    w = (n[use] - 1)
    pooled = float(np.sqrt(float((var[use] * w).sum()) / float(w.sum())))
    return {"n_groups": int(len(use)), "n_rows_in_groups": int(n[use].sum()),
            "pooled_sd": pooled}


def tanimoto_to_adore(df: pd.DataFrame, *, root: str | Path | None = None,
                      challenge: str = "t-F2F", split: str = A.PRIMARY_SPLIT) -> dict:
    """Nearest-neighbour Morgan Tanimoto similarity of each prospective chemical to ADORE training.

    A domain-of-applicability descriptor recorded *before* freezing: it says how far the new
    chemicals sit from what the models were trained on, without touching any target.
    """
    from rdkit import DataStructs

    train = pd.read_parquet(A.challenge_path(challenge, root),
                            columns=["test_cas", "chem_Morgan_fp", split])
    a = train[split].astype("string")
    train = train[(a.notna() & (a != "test")).to_numpy()].drop_duplicates("test_cas")
    ref = [DataStructs.CreateFromBitString(str(s)) for s in train["chem_Morgan_fp"].dropna()]
    out: dict = {"challenge": challenge, "split": split, "n_adore_train_chemicals": len(ref),
                 "per_track": {}}
    for name, sub in df.groupby("prospective_track"):
        u = sub.drop_duplicates("test_cas")
        sims = []
        for s in u["chem_Morgan_fp"].dropna():
            fp = DataStructs.CreateFromBitString(str(s))
            sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, ref)))
        if not sims:
            continue
        q = np.quantile(sims, [0.1, 0.5, 0.9])
        out["per_track"][str(name)] = {
            "n_chemicals": len(sims), "nn_tanimoto_median": float(q[1]),
            "nn_tanimoto_p10": float(q[0]), "nn_tanimoto_p90": float(q[2]),
            "frac_below_0.4": float(np.mean(np.asarray(sims) < 0.4)),
        }
    return out


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return h


def load(track: str | tuple[str, ...] = "fish_novel", *, root: str | Path | None = None
         ) -> pd.DataFrame:
    """Load the frozen prospective file for one or more tracks (built by the build script)."""
    path = (Path(root) if root else paths.PROCESSED) / "chemtox_prospective/prospective.parquet"
    df = pd.read_parquet(path)
    want = (track,) if isinstance(track, str) else tuple(track)
    return df[df["prospective_track"].isin(want)].reset_index(drop=True)
